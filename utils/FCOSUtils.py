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







def reshape_cat_out(inputs):
    '''将不同尺度的预测结果拼在一起
    '''
    out=[]
    for pred in inputs:
        pred = pred.permute(0, 2, 3, 1)
        pred = torch.reshape(pred, [inputs[0].shape[0], -1, inputs[0].shape[1]])
        out.append(pred)
    return torch.cat(out, dim=1)



def gen_grid(cls_logits, strides):
    '''因为FCOS是anchor free, 因此需要生成网格
    '''
    # 对网格进行循环
    grids = []
    for pred, stride in zip(cls_logits, strides):
        h, w     = pred.shape[2:4]
        shifts_x = torch.arange(0, w * stride, stride, dtype=torch.float32)
        shifts_y = torch.arange(0, h * stride, stride, dtype=torch.float32)
        # indexing="ij"符合 NumPy 的默认行为
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")

        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        # 左上角点->网格的中心点
        grid = torch.stack([shift_x, shift_y], -1) + stride // 2
        # grid.shape = [w*h, 2]
        grids.append(grid)

    grids = torch.cat(grids, dim=0)
    if cls_logits[0].is_cuda: grids = grids.cuda()
    # grids.shape = [total_anchor_num, 2]
    return grids



def decode_box(cls_logits, cnt_logits, reg_preds, input_shape, strides=[8, 16, 32, 64, 128]):
    '''将预测结果解码为真实结果
    '''
    '''调整形状'''
    # [[bs, cls_num, 80, 80],...,[[bs, cls_num, 5, 5]]] -> [bs, 8525, cls_num]
    cls_preds = reshape_cat_out(cls_logits)
    # [[bs, 1, 80, 80],...,[[bs, 1, 5, 5]]] -> [bs, total_anchor_num, 1]
    cnt_preds = reshape_cat_out(cnt_logits)
    # [[bs, 4, 80, 80],...,[[bs, 4, 5, 5]]] -> [bs, total_anchor_num, 4]
    reg_preds = reshape_cat_out(reg_preds)
    # 对分类结果归一化到0,1之间
    cls_preds = torch.sigmoid(cls_preds)
    cnt_preds = torch.sigmoid(cnt_preds)
    # 生成网格(类似anchor point, 每个网格的中心点)
    grids = gen_grid(cls_logits, strides)
    # 获得得分最高对应的类别得分和类别
    cls_scores, cls_classes = torch.max(cls_preds,dim=-1)
    # 置信度是类别得分和centerness的乘积(开根号)
    # cls_scores = torch.sqrt(cls_scores * cnt_preds.squeeze(dim=-1))
    cls_scores = cls_scores * cnt_preds.squeeze(dim=-1)
    # 通过中心点和网络预测的tlbr获得box的左上角右下角点(原图的未归一化坐标)
    left_top = grids[None, :, :] - reg_preds[..., :2]
    right_bottom = grids[None, :, :] + reg_preds[..., 2:]
    # boxes.shape = [bs, total_anchor_num, 2+2=4]
    boxes = torch.cat([left_top, right_bottom], dim=-1)
    # 将预测的坐标, 类别置信度, 类别拼在一起 boxes_score_classes.shape = [bs, total_anchor_num, 4+1+1]
    boxes_score_classes = torch.cat([boxes, torch.unsqueeze(cls_scores.float(),-1), torch.unsqueeze(cls_classes.float(),-1)], dim=-1)
    return boxes_score_classes







def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4, agnostic=False):
    '''nms
    '''   
    # prediction.shape = [bs, total_anchor_num, 4+1+1=(x1, y1, x2, y2, class_conf, class_pred)]
    output = []
    # 遍历batch每张图像, 每张图像单独nms:
    for i, image_pred in enumerate(prediction):
        '''首先筛选掉置信度小于阈值的预测'''
        class_conf = image_pred[:, 4:5]
        conf_mask  = (class_conf[:, 0] >= conf_thres).squeeze()
        detections = image_pred[conf_mask]
        # 如果第一轮筛选就没有框,则继续
        if not image_pred.size(0): continue
        if agnostic:
            '''类别无关nms(eval时使用这个一般会掉点)'''
            result = NMSbyAll(detections, nms_thres).cpu().numpy()
        else:
            '''逐类别nms'''
            result = NMSbyCLS(detections, nms_thres).cpu().numpy()
        output.append(result)
    return output





def NMSbyCLS(predicts, nms_thres):
    '''逐类别nms
    '''
    cls_output = torch.tensor([])
    unique_cats = predicts[:, -1].unique()
    for cat in unique_cats:
        # 获得某一类下的所有预测结果
        detections_class = predicts[predicts[:, -1] == cat]
        # 使用官方自带的非极大抑制会速度更快一些
        final_cls_score = detections_class[:, 4] * detections_class[:, 5]
        '''接着筛选掉nms大于nms_thres的预测''' 
        keep = nms(detections_class[:, :4], final_cls_score, nms_thres)
        nms_detections = detections_class[keep]
        # 将类别nms结果记录cls_output
        cls_output = nms_detections if len(cls_output)==0 else torch.cat((cls_output, nms_detections))
    
    return cls_output
    


def NMSbyAll(predicts, nms_thres):
    '''类别无关的nms
    '''
    # 使用官方自带的非极大抑制会速度更快一些
    '''接着筛选掉nms大于nms_thres的预测''' 
    keep = nms(predicts[:, :4], predicts[:, 4], nms_thres)
    nms_detections = predicts[keep]
    
    return nms_detections







def vis_FCOS_heatmap(cls_logits, cnt_logits, ori_shape, input_shape, image, box_classes, padding=True, save_vis_path=None):
    '''可視化 YOLOv5 obj_heatmap
        # Args:
            - predicts:    多尺度特征圖
            - ori_shape:   原圖像尺寸
            - input_shape: 网络接收的尺寸
            - padding:     输入网络时是否灰边填充处理
        # Returns:

    '''
    W, H = ori_shape
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 若灰边填充, 则计算需要裁剪的部分
    if padding==True:
        resize_max_len = max(input_shape[0], input_shape[1])
        if W>H:
            resize_w = resize_max_len
            resize_h = H * (resize_w / W)
            padding_len = round(abs(resize_max_len - resize_h) / 2)
            cut_region = [padding_len, input_shape[1]-padding_len, 0, input_shape[0]]
        else:
            resize_h = resize_max_len
            resize_w = W * (resize_h / H)       
            padding_len = round(abs(resize_max_len - resize_w) / 2)
            cut_region = [0, input_shape[1], padding_len, input_shape[0]-padding_len]
    # 对三个尺度特征图分别提取 obj heatmap
    cls_num = cls_logits[0].shape[1]
    color = [np.random.random((1, 3)) * 0.7 + 0.3 for i in range(cls_num)]
    for layer, cnt_logit in enumerate(cnt_logits):
        cnt_logit = cnt_logit.cpu()
        b, c, h, w = cnt_logit.shape
        # [bs=1,1,w,h] -> [w,h]
        cnt_logit = cnt_logit[0,0,...]
        '''提取centerness-map(类别无关的obj置信度)'''
        saveVisCenternessMap(cnt_logit, image, W, H, layer, input_shape, cut_region, save_vis_path)


def saveVisCenternessMap(cnt_logit, image, W, H, layer, input_shape, cut_region, save_vis_path):
    '''提取objmap(类别无关的obj置信度)
    '''
    # 取objmap, 并执行sigmoid将value归一化到(0,1)之间
    heat_map = F.sigmoid(cnt_logit).numpy()
    # resize到网络接受的输入尺寸
    heat_map = cv2.resize(heat_map, (input_shape[0], input_shape[1]))
    # 去除padding的灰边
    heat_map = heat_map[cut_region[0]:cut_region[1], cut_region[2]:cut_region[3]]
    heat_map = (heat_map * 255).astype('uint8')
    # resize到原图尺寸
    heat_map = cv2.resize(heat_map, (W, H))
    # 灰度转伪彩色图像
    heat_map = cv2.applyColorMap(heat_map, cv2.COLORMAP_JET)
    # heatmap和原图像叠加显示
    heatmap_img = cv2.addWeighted(heat_map, 0.3, image, 0.7, 0)
    # 保存
    if save_vis_path!=None:
        save_dir, save_name = os.path.split(save_vis_path)
        save_name = f'heatmap{layer}_' + save_name
        cv2.imwrite(os.path.join(save_dir, save_name), heatmap_img)


















def get_grids(w, h, stride):
    shifts_x = torch.arange(0, w * stride, stride, dtype=torch.float32)
    shifts_y = torch.arange(0, h * stride, stride, dtype=torch.float32)

    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')

    shift_x = torch.reshape(shift_x, [-1])
    shift_y = torch.reshape(shift_y, [-1])
    grid    = torch.stack([shift_x, shift_y], -1) + stride // 2

    return grid





def FCOSAssigner(gt_boxes, classes, input_shape, strides=[8, 16, 32, 64, 128], limit_ranges=[[-1,64],[64,128],[128,256],[256,512],[512,999999]], sample_radiu_ratio=1.5):
    '''FCOS正负样本分配
        # Args:
            - gt_boxes:      GTbox  [bs, max_box_nums, 4]
            - classes:       类别gt [bs, max_box_nums]
            - input_shape:   网络输入的图像尺寸 默认[640, 640]
            - strides:    
            - limit_ranges:    
            - sample_radiu_ratio:   
        # Returns:
    '''
    cls_targets_all_level = []
    cnt_targets_all_level = []
    reg_targets_all_level = []
    # 遍历每一层特征层进行正负样本分配
    for level in range(len(strides)):
        feat_w = input_shape[0] // strides[level]
        feat_h = input_shape[1] // strides[level]
        h_mul_w     = feat_w * feat_h
        bs          = gt_boxes.shape[0]
        stride      = strides[level]
        limit_range = limit_ranges[level]

        '''获得网格'''
        # grids.shape [w*h, 2]
        grids       = get_grids(feat_w, feat_h, stride).type_as(gt_boxes)
        x, y        = grids[:, 0], grids[:, 1]
        
        '''计算两两ltrb偏移量, 以及相应指标, 用于后续筛选'''
        # 求真实框的左上角和右下角相比于特征点的偏移情况, 得到每个anchor-point相比于每个gt的偏移量(两两计算)
        # [bs, h*w, gt_nums] = [1, h*w, 1] - [bs, 1, gt_nums]
        left_off    = x[None, :, None] - gt_boxes[...,0][:, None, :]
        top_off     = y[None, :, None] - gt_boxes[...,1][:, None, :]
        right_off   = gt_boxes[..., 2][:, None, :] - x[None, :, None]
        bottom_off  = gt_boxes[..., 3][:, None, :] - y[None, :, None]
        # [bs, h*w, gt_nums, 4]
        ltrb_off    = torch.stack([left_off, top_off, right_off, bottom_off],dim=-1)
        # 求每个框的面积 [bs, h*w, gt_nums]
        areas       = (ltrb_off[...,0] + ltrb_off[...,2]) * (ltrb_off[...,1] + ltrb_off[...,3])
        # 计算偏移量中的最小/最大值[bs, h*w, gt_nums]
        off_min     = torch.min(ltrb_off, dim=-1)[0]
        off_max     = torch.max(ltrb_off, dim=-1)[0]

        '''由于上面是计算了每个anchor-point相比于每个gt的两两偏移量, 因此会有很多冗余, 下面进行过滤'''
        # 1.mask_in_gtboxes筛选那些落在真实框内的特征点
        mask_in_gtboxes = off_min > 0
        # 2.mask_in_level筛选哪些gt适合在当前特征层进行检测
        mask_in_level = (off_max > limit_range[0]) & (off_max <= limit_range[1])
        # 在radiu半径圆内的grid作为正样本
        radiu       = stride * sample_radiu_ratio
        # 计算gt中心点与grid中心点两两距离
        # [1,h*w,1] - [bs, 1, gt_nums] --> [bs,h * w, gt_nums]
        # 计算GT的中心点坐标x, y
        gt_center_x = (gt_boxes[...,0] + gt_boxes[...,2]) / 2
        gt_center_y = (gt_boxes[...,1] + gt_boxes[...,3]) / 2
        # 计算grid中心点与gt中心点四个方向的垂直距离, 取最大的作为实际距离(为啥不用欧式距离?)
        c_left_off   = x[None, :, None] - gt_center_x[:, None, :]
        c_top_off    = y[None, :, None] - gt_center_y[:, None, :]
        c_right_off  = gt_center_x[:, None, :] - x[None, :, None]
        c_bottom_off = gt_center_y[:, None, :] - y[None, :, None]
        c_ltrb_off   = torch.stack([c_left_off, c_top_off, c_right_off, c_bottom_off],dim=-1)
        c_off_max    = torch.max(c_ltrb_off,dim=-1)[0]
        # 3.正样本与GT的中心点距离小于radiu
        mask_center = c_off_max < radiu
        # 联合考虑条件1.2.3, 筛选出正样本, 得到pos_mask为bool型
        pos_mask = mask_in_gtboxes & mask_in_level & mask_center
        # 将所有不是正样本的特征点，面积设成max [bs, h*w, gt_nums](其实就是标记为负样本)
        areas[~pos_mask] = 99999999
        # 选取特征点对应面积最小的框对应的索引 [bs, h*w, gt_nums] -> [bs, h*w] (其实就是筛选出每个grid匹配的GT框)
        areas_min_idx = torch.min(areas, dim = -1)[1]
        # 通过索引生成配对mask(每个grid匹配哪个gt) [bs, h*w, max_box_nums]
        match_mask = torch.zeros_like(areas, dtype=torch.bool).scatter_(-1, areas_min_idx.unsqueeze(dim=-1), 1)

        '''为每个grid分配最佳gt, 得到reg_targets, cls_targets, cls_targets'''
        # 筛选reg_targets [bs, h*w, max_box_nums, 4] -> [bs*h*w, 4] -> [bs, h*w, 4]
        reg_targets = ltrb_off[match_mask].reshape(bs, -1, 4)
        # 筛选cls_targets 
        # 将classes[:, None, :]和areas广播为相同的形状 [bs, 1, max_box_nums] -> [bs, h*w, max_box_nums](在第二维度广播)
        _classes, _  = torch.broadcast_tensors(classes[:, None, :], areas.long())
        # 根据match_mask取出对应的正样本 [bs, h*w, max_box_nums] -> [bs, h*w] -> [bs, h*w, 1]
        cls_targets = _classes[match_mask].reshape(bs, -1, 1)
        # 根据reg_targets生成对应grid的centerness value [bs, h*w]
        left_right_min = torch.min(reg_targets[..., 0], reg_targets[..., 2])
        left_right_max = torch.max(reg_targets[..., 0], reg_targets[..., 2])
        top_bottom_min = torch.min(reg_targets[..., 1], reg_targets[..., 3])
        top_bottom_max = torch.max(reg_targets[..., 1], reg_targets[..., 3])
        # 计算centerncss [bs, h*w, 1]
        cnt_targets= ((left_right_min * top_bottom_min) / (left_right_max * top_bottom_max + 1e-10)).sqrt().unsqueeze(dim=-1)
        # 排查形状是否正确
        assert reg_targets.shape == (bs,h_mul_w,4)
        assert cls_targets.shape == (bs,h_mul_w,1)
        assert cnt_targets.shape == (bs,h_mul_w,1)

        '''正负样本筛选'''
        # 那些任意一个gt都没配对上的样本为负样本, 否则为正样本 [bs, h*w, max_box_nums] -> [bs, h*w]
        pos_mask = pos_mask.long().sum(dim=-1) >= 1
        assert pos_mask.shape == (bs, h_mul_w)
        # 负样本对应位置的gt全设为-1
        cls_targets[~pos_mask] = -1
        cnt_targets[~pos_mask] = -1
        reg_targets[~pos_mask] = -1
        # 得到当前层的正负样本分配情况
        cls_targets_all_level.append(cls_targets)
        cnt_targets_all_level.append(cnt_targets)
        reg_targets_all_level.append(reg_targets)
        
    return torch.cat(cls_targets_all_level, dim=1).reshape(-1, 1), torch.cat(cnt_targets_all_level, dim=1).reshape(-1, 1), torch.cat(reg_targets_all_level, dim=1).reshape(-1, 4)


    




def computeGIoU(preds, targets):
    '''计算GIoU(preds, targets均是原始的非归一化坐标)
    '''
    # 左上角和右下角
    lt_min = torch.min(preds[:, :2], targets[:, :2])
    rb_min = torch.min(preds[:, 2:], targets[:, 2:])
    # 重合面积计算
    wh_min = (rb_min + lt_min).clamp(min=0)
    overlap = wh_min[:, 0] * wh_min[:, 1]#[n]
    # 预测框面积和实际框面积计算
    area1 = (preds[:, 2] + preds[:, 0]) * (preds[:, 3] + preds[:, 1])
    area2 = (targets[:, 2] + targets[:, 0]) * (targets[:, 3] + targets[:, 1])
    # 计算交并比
    union = (area1 + area2 - overlap)
    iou = overlap / (union + 1e-7)
    # 计算外包围框
    lt_max = torch.max(preds[:, :2],targets[:, :2])
    rb_max = torch.max(preds[:, 2:],targets[:, 2:])
    wh_max = (rb_max + lt_max).clamp(0)
    G_area = wh_max[:, 0] * wh_max[:, 1]
    # 计算GIOU
    giou = iou - (G_area - union) / G_area.clamp(1e-10)
    return giou
    









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