import numpy as np
from tabulate import tabulate
import cv2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import torch
from thop import profile
from thop import clever_format

from datasets.preprocess import Transform











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


    




def computeParamFLOPs(device, model, img_size:list[int], ):
    '''使用thop分析模型的运算量和参数量
    '''
    input_x = torch.rand(1, 3, img_size[0], img_size[1]).to(device)
    flops, params = profile(model, inputs=(input_x,))
    # 将结果转换为更易于阅读的格式
    flops, params = clever_format([flops, params], '%.3f')
    print(f"FLOPs↓: {flops}, 参数量↓: {params}")