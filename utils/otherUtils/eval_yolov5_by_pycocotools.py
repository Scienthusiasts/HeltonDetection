'''eval_yolov5_by_pycocotools.py 使用说明
    Ultralytics官方YOLOv5评估AP和mAP的方式和pycocotools里评估的方式有些差异, 由于本项目使用pycocotools的评估方式
    为了公平比较, 这份代码将使用pycocotools的评估方式评估Ultralytics官方提供的YOLOv5权重的检测性能.
'''

from typing import List
from pycocotools.coco import COCO
import json
import numpy as np
import cv2
import torch
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression, scale_boxes
from yolov5.utils.torch_utils import select_device
from yolov5.utils.augmentations import letterbox
import os
from tqdm import tqdm
from pycocotools.cocoeval import COCOeval
from tabulate import tabulate
from thop import profile
from thop import clever_format

import platform
if platform.system() == 'Windows':
    import pathlib
    pathlib.PosixPath = pathlib.WindowsPath










def evalCOCOmAP(GTAnnJsonPath, predAnnJsonPath):
    """评估模型预测结果的mAP

    Args:
        :param GTAnnJsonPath:   GT的coco格式json文件路径
        :param predAnnJsonPath: pred的coco格式json文件路径(只包括"annotations":[]里的内容)
        :param printLog:        是否打印表格

    Returns:
        None
    """
    GTCOCO = COCO(GTAnnJsonPath)
    predCOCO = GTCOCO.loadRes(predAnnJsonPath)
    cocoEval = COCOeval(GTCOCO, predCOCO, "bbox")                                            
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    '''下面是计算每个类别的mAP'''
    precisions = cocoEval.eval["precision"]
    # precision.shape=(10, 101, cls, area range, 3): ([0.5:0.95]=10, recall, cls, (all, s, m, l), max dets=[1,10,100]=取置信度top多少的框)
    header = ['categories', 'mAP', 'mAP_s', 'mAP_m', 'mAP_l', 'mAP_50', 'mAP_75']
    mAPData = []
    # 每个类别计算mAP
    AP_all = np.zeros(6)
    for idx, catId in enumerate(predCOCO.getCatIds()):
        mAPDataPerCat = []
        # max dets index -1: typically 100 per image
        nm = predCOCO.loadCats(catId)[0]
        mAPDataPerCat.append(nm["name"])
        # 计算每个类别里不同大小目标框的map
        for area in range(4):           
            precision = precisions[:, :, idx, area, -1]
            precision = precision[precision > -1]
            if precision.size: ap = np.mean(precision)
            # N/A
            else: ap = float('nan')
            mAPDataPerCat.append('%.5f' % ap)
        # 计算ap50和ap75
        for iou in [0, 5]:
            ap = precisions[iou, :, idx, 0, -1]
            ap = np.mean(ap[ap > -1])
            mAPDataPerCat.append('%.5f' % ap)
        # 记录所有类别平均AP
        mAPData.append(mAPDataPerCat)
        for i in range(1, 7):
            AP_all[i-1] += float(mAPDataPerCat[i])
    # 计算所有类别平均AP
    mAPDataAvg = []    
    mAPDataAvg.append('average')
    for i in range(6):
        mAPDataAvg.append('%.5f' % (AP_all[i]/(idx+1)))
    mAPData.append(mAPDataAvg)
    AP_table = tabulate(mAPData, headers=header, tablefmt='psql')
    print(AP_table)
    # 返回mAP, AP50
    return mAPDataAvg[1], mAPDataAvg[5]












class YOLOv5Detector:
    def __init__(self, weight: str, reverse_map:dict=None):
        self.device = select_device('')
        self.model = attempt_load(weight, device=self.device)
        self.imgsz = 640
        self.score_thres = 0.01
        self.T = 0.3
        self.reverse_map = reverse_map

    def __call__(self, frame: np.ndarray):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # img = frame
        img_processed = letterbox(img, self.imgsz, stride=32)[0]
        img_processed = torch.from_numpy(img_processed.transpose(2,0,1)).to(self.device)
        img_processed = img_processed.unsqueeze(0) / 255.

        pred = self.model(img_processed, augment=False)[0]
        pred = non_max_suppression(pred, self.score_thres, self.T, agnostic=True)[0].detach().cpu()
        pred = np.array(pred)

        bboxes, bboxes_score, bboxes_cls = [], [], []
        for xyxysc in pred:
            if len(xyxysc) == 0: continue
            xyxysc[:4] = scale_boxes(img_processed.shape[2:], xyxysc[:4], img.shape).round()
            x0, y0, x1, y1 = round(xyxysc[0]), round(xyxysc[1]), round(xyxysc[2]), round(xyxysc[3])
            score = xyxysc[4]
            cls = int(xyxysc[5])
            bboxes.append([x0, y0, x1, y1])
            bboxes_score.append(score)
            bboxes_cls.append(cls)

        return bboxes, bboxes_score, bboxes_cls
    



    def formatOneImg(self, img_path, image_id, anns_dict):
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
        image = cv2.imread(img_path)
        boxes, box_scores, box_classes = self(image)
        # 将预测结果转化为COCO格式下'annotations'下的一个字段(注意COCO的box格式是xywh)
        for box, score, cls in zip(boxes, box_scores, box_classes):
            if score>self.T:
                # 如果像COCO数据集一样categories id没有按顺序来，则还得映射回去
                if self.reverse_map!=None: cls = self.reverse_map[cls]
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
    



    def genPredJsonAndEval(self, json_path, img_dir, log_dir, pred_json_name):
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
        # faasterrcnn这里如果mode是train的话mAP会高一些，但是推理更慢(主要区别在于保留的框的数量不一样，train是600，val是300)
        '''在线推理生成一个eval的json文件(COCO anntations字段格式)'''
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
            anns_dict = self.formatOneImg(file_path, img_info['id'], anns_dict)
        # 将anns_dict保存为json文件(eval_tmp)
        if not os.path.isdir(log_dir):os.makedirs(log_dir) 
        with open(pred_json_path, 'w') as json_file:
            json.dump(anns_dict, json_file)

        '''计算 mAP, ap_50'''
        mAP, ap_50 = evalCOCOmAP(json_path, pred_json_path)

        '''使用thop分析模型的运算量和参数量'''
        input_x = torch.rand(1, 3, self.imgsz, self.imgsz).to(self.device)
        flops, params = profile(self.model, inputs=(input_x,))
        # 将结果转换为更易于阅读的格式
        flops, params = clever_format([flops, params], '%.3f')
        print(f"FLOPs↓: {flops}, 参数量↓: {params}")
        return mAP, ap_50
    



if __name__ == '__main__':
    weight = "F:/DeskTop/git/yolov5-master/yolov5s.pt"
    json_path = "E:/datasets/Universal/COCO2017/COCO/annotations/instances_val2017.json"
    img_dir = 'E:/datasets/Universal/COCO2017/COCO/val2017'
    log_dir = './log/Ultralytics_yolov5'
    pred_json_name = 'yolov5_eval.json'
    reverse_map = {0:1, 1:2, 2:3, 3:4, 4:5, 5:6, 6:7, 7:8, 8:9, 9:10, 10:11, 11:13, 12:14, 13:15, 14:16, 15:17, 16:18, 17:19, 18:20, 19:21, 20:22, 21:23, 
       22:24, 23:25, 24:27, 25:28, 26:31, 27:32, 28:33, 29:34, 30:35, 31:36, 32:37, 33:38, 34:39, 35:40, 36:41, 37:42, 38:43, 39:44, 40:46, 
       41:47, 42:48, 43:49, 44:50, 45:51, 46:52, 47:53, 48:54, 49:55, 50:56, 51:57, 52:58, 53:59, 54:60, 55:61, 56:62, 57:63, 58:64, 59:65, 
       60:67, 61:70, 62:72, 63:73, 64:74, 65:75, 66:76, 67:77, 68:78, 69:79, 70:80, 71:81, 72:82, 73:84, 74:85, 75:86, 76:87, 77:88, 78:89, 79:90}
    
    yolov5det = YOLOv5Detector(weight, reverse_map)
    yolov5det.genPredJsonAndEval(json_path, img_dir, log_dir, pred_json_name)




