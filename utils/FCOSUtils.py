import numpy as np
import torch
from torchvision.ops import nms
from torch.nn import functional as F
import cv2
import torch.nn as nn
import os
import math
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from loss.YOLOLoss import *






def get_grids(pred, stride):
    h, w     = pred.shape[2:4]
    shifts_x = torch.arange(0, w * stride, stride, dtype=torch.float32)
    shifts_y = torch.arange(0, h * stride, stride, dtype=torch.float32)

    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)

    shift_x = torch.reshape(shift_x, [-1])
    shift_y = torch.reshape(shift_y, [-1])
    grid    = torch.stack([shift_x, shift_y], -1) + stride // 2

    return grid



def FCOSAssigner(cls_logits, gt_boxes, classes, strides=[8, 16, 32, 64, 128], limit_ranges=[[-1,64],[64,128],[128,256],[256,512],[512,999999]], sample_radiu_ratio=1.5):
    cls_targets_all_level = []
    cnt_targets_all_level = []
    reg_targets_all_level = []

    for level in range(len(cls_logits)):
        cls_logit   = cls_logits[level]
        stride      = strides[level]
        limit_range = limit_ranges[level]
        #--------------------#
        #   计算batch_size
        #   计算种类数量
        #--------------------#
        batch_size  = cls_logit.shape[0]
        num_classes = cls_logit.shape[1]

        #-----------------------#
        #   获得网格
        #-----------------------#
        grids       = get_grids(cls_logit, stride)
        grids       = grids.type_as(cls_logit)
        x           = grids[:, 0]
        y           = grids[:, 1]
            
        #-------------------------------------#
        #   对预测结果进行reshape
        #   [batch_size, h * w, num_classes] 
        #-------------------------------------#
        cls_logit   = cls_logit.permute(0, 2, 3, 1).reshape((batch_size, -1, num_classes))
        h_mul_w     = cls_logit.shape[1]

        #----------------------------------------------------------------#
        #   左上点、右下点不可以差距很大
        #   求真实框的左上角和右下角相比于特征点的偏移情况
        #   [1, h*w, 1] - [batch_size, 1, m] --> [batch_size, h*w, m]
        #----------------------------------------------------------------#
        left_off    = x[None, :, None] - gt_boxes[...,0][:, None, :]
        top_off     = y[None, :, None] - gt_boxes[...,1][:, None, :]
        right_off   = gt_boxes[..., 2][:, None, :] - x[None, :, None]
        bottom_off  = gt_boxes[..., 3][:, None, :] - y[None, :, None]
        #----------------------------------------------------------------#
        #   [batch_size, h*w, m, 4]
        #----------------------------------------------------------------#
        ltrb_off    = torch.stack([left_off, top_off, right_off, bottom_off],dim=-1)
        
        #----------------------------------------------------------------#
        #   求每个框的面积
        #   [batch_size, h*w, m]
        #----------------------------------------------------------------#
        areas       = (ltrb_off[...,0] + ltrb_off[...,2]) * (ltrb_off[...,1] + ltrb_off[...,3])
        #----------------------------------------------------------------#
        #   [batch_size,h*w,m]
        #----------------------------------------------------------------#
        off_min     = torch.min(ltrb_off, dim=-1)[0]
        off_max     = torch.max(ltrb_off, dim=-1)[0]

        #----------------------------------------------------------------#
        #   将特征点不落在真实框内的特征点剔除。
        #   浅层特征适合小目标检测，深层特征适合大目标检测。
        #----------------------------------------------------------------#
        mask_in_gtboxes = off_min > 0
        mask_in_level = (off_max > limit_range[0]) & (off_max <= limit_range[1])

        radiu       = stride * sample_radiu_ratio
        #----------------------------------------------------------------#
        #   中心点不可以差距很大，求真实框中心相比于特征点的偏移情况
        #   计算真实框中心的x轴坐标
        #   计算真实框中心的y轴坐标
        #   [1,h*w,1] - [batch_size, 1, m] --> [batch_size,h * w, m]
        #----------------------------------------------------------------#
        gt_center_x = (gt_boxes[...,0] + gt_boxes[...,2]) / 2
        gt_center_y = (gt_boxes[...,1] + gt_boxes[...,3]) / 2
        c_left_off      = x[None, :, None] - gt_center_x[:, None, :]
        c_top_off       = y[None, :, None] - gt_center_y[:, None, :]
        c_right_off     = gt_center_x[:, None, :] - x[None, :, None]
        c_bottom_off    = gt_center_y[:, None, :] - y[None, :, None]
        #----------------------------------------------------------------#
        #   [batch_size, h*w, m, 4]
        #----------------------------------------------------------------#
        c_ltrb_off  = torch.stack([c_left_off, c_top_off, c_right_off, c_bottom_off],dim=-1)
        c_off_max   = torch.max(c_ltrb_off,dim=-1)[0]
        mask_center = c_off_max < radiu

        #----------------------------------------------------------------#
        #   为正样本的特征点
        #   [batch_size, h*w, m]
        #----------------------------------------------------------------#
        mask_pos    = mask_in_gtboxes & mask_in_level & mask_center

        #----------------------------------------------------------------#
        #   将所有不是正样本的特征点，面积设成max
        #   [batch_size, h*w, m]
        #----------------------------------------------------------------#
        areas[~mask_pos]    = 99999999
        #----------------------------------------------------------------#
        #   选取该特征点对应面积最小的框
        #   [batch_size, h*w]
        #----------------------------------------------------------------#
        areas_min_ind       = torch.min(areas, dim = -1)[1]
        #----------------------------------------------------------------#
        #   [batch_size*h*w, 4]
        #----------------------------------------------------------------#
        reg_targets = ltrb_off[torch.zeros_like(areas, dtype=torch.bool).scatter_(-1, areas_min_ind.unsqueeze(dim=-1), 1)]
        reg_targets = torch.reshape(reg_targets, (batch_size, -1, 4))
        #----------------------------------------------------------------#
        #   [batch_size, h*w, m]
        #----------------------------------------------------------------#
        _classes    = torch.broadcast_tensors(classes[:, None, :], areas.long())[0]
        cls_targets = _classes[torch.zeros_like(areas, dtype=torch.bool).scatter_(-1, areas_min_ind.unsqueeze(dim=-1), 1)]
        #----------------------------------------------------------------#
        #   [batch_size, h*w, 1]
        #----------------------------------------------------------------#
        cls_targets = torch.reshape(cls_targets,(batch_size,-1,1))

        #----------------------------------------------------------------#
        #   [batch_size, h*w]
        #----------------------------------------------------------------#
        left_right_min  = torch.min(reg_targets[..., 0], reg_targets[..., 2])
        left_right_max  = torch.max(reg_targets[..., 0], reg_targets[..., 2])
        top_bottom_min  = torch.min(reg_targets[..., 1], reg_targets[..., 3])
        top_bottom_max  = torch.max(reg_targets[..., 1], reg_targets[..., 3])
        #----------------------------------------------------------------#
        #   [batch_size, h*w, 1]
        #----------------------------------------------------------------#
        cnt_targets= ((left_right_min * top_bottom_min) / (left_right_max * top_bottom_max + 1e-10)).sqrt().unsqueeze(dim=-1)

        assert reg_targets.shape == (batch_size,h_mul_w,4)
        assert cls_targets.shape == (batch_size,h_mul_w,1)
        assert cnt_targets.shape == (batch_size,h_mul_w,1)

        #----------------------------------------------------------------#
        #   process neg grids
        #----------------------------------------------------------------#
        mask_pos_2 = mask_pos.long().sum(dim=-1) >= 1
        assert mask_pos_2.shape  == (batch_size,h_mul_w)
        cls_targets[~mask_pos_2] = -1
        cnt_targets[~mask_pos_2] = -1
        reg_targets[~mask_pos_2] = -1

        cls_targets_all_level.append(cls_targets)
        cnt_targets_all_level.append(cnt_targets)
        reg_targets_all_level.append(reg_targets)
        
    return torch.cat(cls_targets_all_level, dim=1), torch.cat(cnt_targets_all_level, dim=1), torch.cat(reg_targets_all_level, dim=1)


    










# # for test only:
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # ann_path = 'E:/datasets/Universal/COCO2017/COCO/annotations/instances_train2017.json'
    ann_path = 'E:/datasets/RemoteSensing/visdrone2019/annotations/train.json'
    datas, centers, labels = BBoxesKmeans2Anchors(ann_path, seed=22)
    print(centers, labels)

    #   绘图
    for j in range(9):
        plt.scatter(datas[labels == j][:,0], datas[labels == j][:,1], s=1)
        plt.scatter(centers[j][0], centers[j][1], marker='x', c='black')
    plt.savefig("kmeans_for_anchors.jpg", dpi=150)
    plt.show()