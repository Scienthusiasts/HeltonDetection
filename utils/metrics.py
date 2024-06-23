import numpy as np
from tabulate import tabulate
import cv2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ensemble_boxes import weighted_boxes_fusion as wbf

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






class TTA():
    '''Test Time Augmentation + WBF策略
    '''

    def __init__(self, tta_img_size=[[640,640], [832,832], [960,960]]):
        self.tta_img_size = tta_img_size
        self.tta_transform = [Transform(img_size, box_format='coco') for img_size in self.tta_img_size]        


    def infer(self,
               model, 
               image:np.array, 
               device, 
               T, 
               image2color, 
               agnostic=False, 
               vis_heatmap=False, 
               save_vis_path=None, 
               half=False, 
               tta=True
               ):
        flip_image = cv2.flip(image, 1)
        # n个尺度, 每个尺度2个增强, 一共2n张图
        tta_boxes, tta_box_scores, tta_box_classes = [], [], []
        # 每张图像分别推理:
        for i in range(len(self.tta_img_size)):
            for j in range(2):
                img = image if j==0 else flip_image
                boxes, box_scores, box_classes = model.infer(
                    img, 
                    self.tta_img_size[i], 
                    self.tta_transform[i], 
                    device, 
                    T, 
                    image2color, 
                    agnostic, 
                    False, 
                    save_vis_path, 
                    half=half, 
                    tta=tta
                    )
                if len(boxes) > 0:
                    # 归一化(wbf只接受归一化输入)
                    boxes[:, [0,2]] /= image.shape[1]
                    boxes[:, [1,3]] /= image.shape[0]
                    # 水平翻转的图像翻转回来
                    if j == 1: 
                        boxes[:, [0, 2]] = 1 - boxes[:, [2, 0]]
                tta_boxes.append(boxes)
                tta_box_scores.append(box_scores)
                tta_box_classes.append(box_classes)
        '''不同增强下的预测结果执行WBF'''
        tta_boxes, tta_box_scores, tta_box_classes = wbf(
            tta_boxes, 
            tta_box_scores, 
            tta_box_classes, 
            weights=[1]*2*len(self.tta_img_size), 
            iou_thr=0.55, 
            skip_box_thr=T
            )
        if len(tta_boxes) == 0 : return [],[],[]
        # 坐标再映射回原图大小
        tta_boxes[:, [0,2]] *= image.shape[1]
        tta_boxes[:, [1,3]] *= image.shape[0]
        return tta_boxes, tta_box_scores, tta_box_classes.astype(int)