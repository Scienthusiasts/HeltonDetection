import os                                    # 读取文件目录
import numpy as np
from tabulate import tabulate
import torch
import cv2
import argparse
import importlib
import json
from PIL import Image
import matplotlib.pyplot as plt
import random
from torch import nn
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from matplotlib.patches import Polygon, Rectangle



def sigmoid(x):
    y = 1.0 / (1.0 + np.exp(-x))
    return y


# 动态导入类
def dynamic_import_class(module_path, class_name='module_name', get_class=True):
    spec = importlib.util.spec_from_file_location(class_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if get_class:
        return getattr(module, class_name)
    else:
        return module





def seed_everything(seed):
    '''设置全局种子
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False






def init_weights(model, init_type, mean=0, std=0.01):
    '''权重初始化方法
    '''
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if init_type=='he':
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if init_type=='normal':
                nn.init.normal_(module.weight, mean=mean, std=std)  # 使用高斯随机初始化
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)




def CV2DrawBox(image, boxes, classes, scores, save_res_path, colors, class_names):
    '''OpenCV画框
        Args:
            :param image:         原始图像(Image格式)
            :param boxes:         网络预测的box坐标
            :param classes:       网络预测的box类别
            :param scores:        网络预测的box置信度
            :param save_res_path: 可视化结果保存路径

        Returns:
            None
    '''
    # Image转opencv
    img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    for box, cls, score in zip(boxes, classes, scores):
        x0, y0, x1, y1 = round(box[0]), round(box[1]), round(box[2]), round(box[3])
        cv2.rectangle(img, (x0, y0), (x1, y1), colors[cls], thickness=2)
        text = '{} {:.2f}'.format(class_names[cls], score)
        cv2.putText(img, text, (x0, y0), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=colors[cls], thickness=2)
    cv2.imwrite(save_res_path, img)





def PltDrawBox(image, boxes, classes, scores, save_res_path, image2color, class_names):
    '''plt画框
        Args:
            :param image:         原始图像(Image格式)
            :param boxes:         网络预测的box坐标
            :param classes:       网络预测的box类别
            :param scores:        网络预测的box置信度
            :param save_res_path: 可视化结果保存路径

        Returns:
            None
    '''
    plt.figure()
    ax = plt.gca()
    for box, cls, score in zip(boxes, classes, scores):
        x0, y0, x1, y1 = round(box[0]), round(box[1]), round(box[2]), round(box[3])
        color = image2color[class_names[cls]]
        text = '{} {:.2f}'.format(class_names[cls], score)
        # 多边形填充+矩形边界:
        ax.add_patch(Polygon(xy=[[x0, y0], [x0, y1], [x1, y1], [x1, y0]], color=color, alpha=0.3))
        ax.add_patch(Rectangle(xy=(x0, y0), width=x1-x0, height=y1-y0, fill=False, color=color, alpha=1))
        # 可视化每个bbox的类别的文本(ax.text的bbox参数用于调整文本框的样式):
        ax.text(x0, y0, text, color='white', fontsize=5, bbox=dict(facecolor=color, alpha=0.4))
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    # plt.show()
    plt.savefig(save_res_path, bbox_inches='tight', pad_inches=0.0, dpi=200)




def OpenCVDrawBox(image, boxes, classes, scores, save_vis_path, image2color, class_names, resize_size, show_text=True):
    '''plt画框
        Args:
            :param image:         原始图像(Image格式)
            :param boxes:         网络预测的box坐标
            :param classes:       网络预测的box类别
            :param scores:        网络预测的box置信度
            :param save_res_path: 可视化结果保存路径

        Returns:
            None
    '''
    H, W = image.shape[:2]
    
    max_len = max(W, H)
    w = int(W * resize_size[0] / max_len)
    h = int(H * resize_size[1] / max_len)
    boxes = mapBox2OriginalImg(boxes, w, h, [W, H], padding=False)

    image = cv2.resize(image, (w, h))
    # 框的粗细
    thickness = max(1, int(image.shape[0] * 0.003))
    for box, cls, score in zip(boxes, classes, scores):
        # if class_names[cls] not in  ['pedestrian', 'people', 'person']:continue
        x0, y0, x1, y1 = round(box[0]), round(box[1]), round(box[2]), round(box[3])
        color = np.array(image2color[class_names[cls]])
        color = tuple([int(c*255) for c in color])
        # color = (0,0,255)
        # text = 'target_1'
        # text = '{} {:.2f}'.format(text, score)
        text = '{} {:.2f}'.format(class_names[cls], score)
        # obj的框
        cv2.rectangle(image, (x0, y0), (x1, y1), color, thickness=thickness)
        # 文本的框
        if show_text:
            cv2.rectangle(image, (x0-1, y0-30), (x0+len(text)*12, y0), color, thickness=-1)
            # 文本
            cv2.putText(image, text, (x0, y0-6), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(255,255,255), thickness=2)
    # 保存
    if save_vis_path!=None:
        return image
    image = cv2.resize(image, (W, H))
    return image

        




def mapBox2OriginalImg(boxes, w, h, imgsize:list, padding=True):
    '''将box坐标(对应有黑边的图)映射回无黑边的原始图像
        Args:
            :param boxes:          网络预测的box的坐标(对应有黑边的图的尺寸)
            :param w:              原始图像的宽
            :param h:              原始图像的高
            :param imgsize:        resize后图像的大小[w, h]


        Returns:
            解码+NMS后保留的proposal
    '''
    # 黑边填充
    if padding==True:
        resize_max_len = max(imgsize[0], imgsize[1])
        if w>h:
            resize_w = resize_max_len
            resize_h = h * (resize_w / w)
            padding_len = abs(resize_max_len - resize_h) / 2
            boxes[:, [1,3]] -= padding_len
        else:
            resize_h = resize_max_len
            resize_w = w * (resize_h / h)        
            padding_len = abs(resize_max_len - resize_w) / 2
            boxes[:, [0,2]] -= padding_len
        boxes[:, [1,3]] *= (h / resize_h)
        boxes[:, [0,2]] *= (w / resize_w)
    # 按比例缩放，无黑边填充
    else:
        w_ratio = w / imgsize[0]
        h_ratio = h / imgsize[1]
        boxes[:, [1,3]] *= h_ratio
        boxes[:, [0,2]] *= w_ratio

    return boxes






def visArgsHistory(json_dir, save_dir, loss_sample_interval=50):
    '''可视化训练过程中保存的参数

    Args:
        :param json_dir: 参数的json文件路径
        :param logDir:   可视化json文件保存路径

    Returns:
        None
    '''
    if not os.path.isdir(save_dir):os.makedirs(save_dir)
    json_path = os.path.join(json_dir, 'args_history.json')
    with open(json_path) as json_file:
        args = json.load(json_file)

        for args_key in args.keys():
            arg = args[args_key]
            # loss太多了则等间隔采样
            if args_key.split('_')[-1]=='loss':
                arg = arg[::loss_sample_interval]

            # 绘制
            plt.plot(arg, linewidth=1)
            if args_key.split('_')[-1]=='loss':
                plt.xlabel(f'iter with interval {loss_sample_interval}')
            else:
                plt.xlabel('Epoch')
            plt.ylabel(args_key)
            plt.savefig(os.path.join(save_dir, f"{args_key.replace(':', '.')}.png"), dpi=200)
            plt.clf()





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






# for test only
if __name__ == '__main__':
    json_dir = 'log/fpn_p2_VOC'
    save_dir = './log/fpn_p2_VOC/plot'
    visArgsHistory(json_dir, save_dir, loss_sample_interval=50)
