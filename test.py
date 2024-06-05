import os
import cv2
import json
import torch
import argparse
import importlib
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision.ops import nms
from pycocotools.coco import COCO
# 自定义模块
from utils.util import *
from utils.FasterRCNNAnchorUtils import *








class Test():

    def __init__(self, path, model, img_size, class_names, device, half, tta):
        self.device = device
        self.tta = tta
        self.half = half
        self.model = model
        # 半精度推理:
        if self.half: self.model.half()
        self.img_size = img_size
        self.cat_nums = len(class_names)
        self.class_names = class_names
        # 动态导入
        self.tf = dynamic_import_class(path, 'Transform')(self.img_size)
        '''每个类别都获得一个随机颜色'''
        self.image2color = dict()
        for cat in class_names:
            self.image2color[cat] = (np.random.random((1, 3)) * 0.7 + 0.3).tolist()[0]



    def predict(self, mode, path, save_vis_path=None, ckpt_path=None, T=0.3, agnostic=False, vis_heatmap=False, show_text=True, **kwargs):
        '''推理一张图
            Args:
                - path:          图片/视频路径
                - save_vis_path: 可视化图像/视频保存路径

            Returns:
                - boxes:       网络回归的box坐标    [obj_nums, 4]
                - box_scores:  网络预测的box置信度  [obj_nums]
                - box_classes: 网络预测的box类别    [obj_nums]
        '''
        # 调用模型自己的推理方法
        if mode == 'image':
            boxes, box_scores, box_classes = inferenceSingleImg(
                self.model, self.device, self.class_names, self.image2color, self.img_size, self.tf, path, save_vis_path, ckpt_path, T, agnostic, show_text, vis_heatmap, self.half, self.tta
                )
            return boxes, box_scores, box_classes
        if mode == 'video':
            inferenceVideo(self.model, self.device, self.class_names, self.image2color, self.img_size, self.tf, path, save_vis_path, ckpt_path, T, agnostic, show_text, self.half)






    def formatOneImg(self, img_path, image_id, anns_dict, T=0.3, agnostic=False, reverse_map=None):
        '''推理一张图,并返回COCO格式'annotations'下的一个字段
            Args:
                - img_path: 图片路径
                - image_id: 可视化图像保存路径
                - anns_dict: COCO格式下'annotations'字段

            Returns:
                - boxes:       网络回归的box坐标    [obj_nums, 4]
                - box_scores:  网络预测的box置信度  [obj_nums]
                - box_classes: 网络预测的box类别    [obj_nums]
        '''
        boxes, box_scores, box_classes = self.predict(mode='image', path=img_path, T=T, agnostic=agnostic)
        # 将预测结果转化为COCO格式下'annotations'下的一个字段(注意COCO的box格式是xywh)
        for box, score, cls in zip(boxes, box_scores, box_classes):
            if score>T:
                # 如果像COCO数据集一样categories id没有按顺序来，则还得映射回去
                if reverse_map!=None: cls = reverse_map[cls]
                anns_dict.append(
                    {
                        "image_id":image_id,
                        "category_id":int(cls),
                        "bbox":[
                            float(box[0]),
                            float(box[1]),
                            float(box[2] - box[0]),
                            float(box[3] - box[1])
                        ],
                        "score":float(score),
                    }
                )
        return anns_dict



    def genPredJsonAndEval(self, json_path, img_dir, log_dir, pred_json_name, T=0.01, agnostic=False, model=None, inferring=True, ckpt_path=None, reverse_map=None, fuse=False):
        '''将预测结果转为pred.json(COCO)的标注格式(但是只包含"annotations":[]里的内容), 且评估mAP
            Args:
                - json_path:      数据集json文件路径
                - img_dir:        数据集图像文件夹
                - log_dir:        预测结果json文件保存根目录
                - pred_json_name: 预测结果json文件名
                - T:              置信度(模型推理超参)
                - model:          导入的模型(当mode=='eval'时有效)
                - inferring:      是否让网络推理一遍数据集并生成json
                - printLog:       是否打印每个类别的AP表格

            Returns:
                - mAP:   所有类别平均 AP@.5:.95          
                - ap_50: 所有类别平均 AP@.5
        '''
        pred_json_path = pred_json_name
        # 是否导入权重
        if ckpt_path != None:
            print(ckpt_path)
            # self.model.load_state_dict(torch.load(ckpt_path))
            self.model = loadWeightsBySizeMatching(self.model, ckpt_path)
            # 半精度推理(貌似还有问题,推理速度和全精度差不多):
            if self.half: self.model.half()
            # yolov8:
            if fuse:
                self.model = self.model.fuse()
        self.model = model.eval()
        # faasterrcnn这里如果mode是train的话mAP会高一些，但是推理更慢(主要区别在于保留的框的数量不一样，train是600，val是300)
        '''是否在线推理, 在线推理则会在线生成一个eval的json文件(COCO anntations字段格式)'''
        if inferring:
            pred_json_path = os.path.join(log_dir, pred_json_name)
            # 为实例注释初始化COCO的API
            coco=COCO(json_path)
            # 获取验证集中所有图像对应的imgId
            img_ids = list(coco.imgs.keys())
            anns_dict = []
            '''采用一张图一张图推理的方式,未使用dataloader加载数据, 本质上还是推理一张图'''
            for img_id in tqdm(img_ids):
                # 获取图像对应的信息
                img_info = coco.loadImgs(img_id)[0]
                file_path = os.path.join(img_dir, img_info['file_name'])
                # 推理当前img_id对应的图像
                # anns_dict其实是一个字典, 随着推理不断更新里面的字段
                anns_dict = self.formatOneImg(file_path, img_info['id'], anns_dict, T, agnostic, reverse_map=reverse_map)
            # 将anns_dict保存为json文件(eval_tmp)
            with open(pred_json_path, 'w') as json_file:
                json.dump(anns_dict, json_file)

        # 计算 mAP, ap_50
        mAP, ap_50 = evalCOCOmAP(json_path, pred_json_path)
        return mAP, ap_50







    def eval(self, json_path, img_dir, log_dir, pred_json_name, T=0.01, agnostic=False, model=None, inferring=True, ckpt_path=None, reverse_map=None):
        '''将预测结果转为pred.json(COCO)的标注格式(但是只包含"annotations":[]里的内容), 且评估mAP
            Args:
                - json_path:      数据集json文件路径
                - img_dir:        数据集图像文件夹
                - pred_json_path: 预测结果json文件保存路径
                - T:              置信度(模型推理超参)
                - model:          导入的模型(当mode=='eval'时有效)
                - inferring:      是否让网络推理一遍数据集并生成json
                - printLog:       是否打印每个类别的AP表格

            Returns:
                - mAP:   所有类别平均 AP@.5:.95          
                - ap_50: 所有类别平均 AP@.5
        '''
        pass









def getArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, help='config file')
    args = parser.parse_args()
    return args



def import_module_by_path(module_path):
    """根据给定的完整路径动态导入模块(config.py)
    """
    spec = importlib.util.spec_from_file_location("module_name", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module













if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = getArgs()
    # 使用动态导入的模块
    config_path = args.config
    config_file = import_module_by_path(config_path)
    # 调用动态导入的模块的函数
    config = config_file.test_config
    tester = Test(config['mode'], config['img_size'], config['class_names'], device, ckpt_path=config['ckpt_path'], colors=config['colors'])
    if config['mode']=='test':
        tester.predictOneImg(config['img_path'], config['save_res_path'], T=config['confidence'])
    if config['mode']=='eval':
        tester.genPredJsonAndEval(config['json_path'], config['img_dir'], config['pred_json_path'], T=config['confidence'], inferring=False, printLog=True)
    
