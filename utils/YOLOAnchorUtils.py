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








def vis_YOLOv5_heatmap(predicts:torch.tensor, ori_shape, input_shape, image, box_classes, padding=True, save_vis_path=None):
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
    cls_num = predicts[0].shape[-1]-5
    color = [np.random.random((1, 3)) * 0.7 + 0.3 for i in range(cls_num)]
    for layer, predict in enumerate(predicts):
        predict = predict.cpu()
        b, c, h, w = predict.shape
        # predict.shape = [w, h, 3, 25], 取[0]是因为默认bs=1
        predict = np.transpose(np.reshape(predict, [b, 3, -1, h, w]), [0, 3, 4, 1, 2])[0]
        '''提取objmap(类别无关的obj置信度)'''
        saveVisHeatMap(predict, image, W, H, layer, input_shape, cut_region, save_vis_path)
        '''提取clsmap(obj置信度 * 类别相关的置信度)'''
        # saveVisClsMap(color, cls_num, predict, W, H, w, h, layer, input_shape, cut_region, save_vis_path)
        '''提取clsmap(类别相关的置信度)'''
        # saveClsHeatMap(predict, box_classes, image, W, H, layer, input_shape, cut_region, save_vis_path)







def saveVisHeatMap(predict, image, W, H, layer, input_shape, cut_region, save_vis_path):
    '''提取objmap(类别无关的obj置信度)
    '''
    # 取objmap, 并执行sigmoid将value归一化到(0,1)之间
    heat_map = F.sigmoid(predict[..., 4]).numpy()
    # 由于每个anchor都有一个objmap, 因此统一取value最大的 shape = [80, 80, 3] -> [80, 80]
    heat_map = np.max(heat_map, -1)
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



def saveClsHeatMap(predict, box_classes, image, W, H, layer, input_shape, cut_region, save_vis_path):
    '''提取clsmap(obj置信度 * 类别相关的置信度)
    '''
    heat_map = torch.max(F.sigmoid(predict), -2)[0][..., 5:]
    heat_map = heat_map[..., list(set(box_classes))]
    heat_map = torch.max(heat_map, -1)[0].cpu().numpy()
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
    heatmap_img = cv2.addWeighted(heat_map, 0.5, image, 0.5, 0)
    # 保存
    if save_vis_path!=None:
        save_dir, save_name = os.path.split(save_vis_path)
        save_name = f'heatmap{layer}_' + save_name
        cv2.imwrite(os.path.join(save_dir, save_name), heatmap_img)



def saveVisClsMap(color, cls_num, predict, W, H, w, h, layer, input_shape, cut_region, save_vis_path):
    '''提取clsmap(类别相关的置信度)
    '''
    obj_map = torch.max(F.sigmoid(predict[..., 4]).reshape(h, w, 3, 1), 2)[0]
    heat_map = predict[..., 5:]
    heat_map = torch.max(heat_map, -2)[0]
    cls_map = np.argmax(heat_map.numpy(), -1)
    # 初始化一个形状为 w, h, 3 的数组来存储 RGB 图像
    rgb_heat_map = np.zeros((w, h, 3))
    # 将分割结果映射到 RGB 值
    for label in range(cls_num):
        rgb_heat_map[cls_map == label] = color[label]
    heat_map = torch.max(F.softmax(heat_map, dim=-1), -1)[0].unsqueeze(-1)
    rgb_heat_map *= (heat_map*obj_map).cpu().numpy() 
    # rgb_heat_map *= obj_map.cpu().numpy() 
    rgb_heat_map = (rgb_heat_map*255).astype(np.uint8)
    # resize到网络接受的输入尺寸
    rgb_heat_map = cv2.resize(rgb_heat_map, (input_shape[0], input_shape[1]), interpolation=cv2.INTER_CUBIC)
    # 去除padding的灰边
    rgb_heat_map = rgb_heat_map[cut_region[0]:cut_region[1], cut_region[2]:cut_region[3]]
    # resize到原图尺寸
    rgb_heat_map = cv2.resize(rgb_heat_map, (W, H))
    # 保存
    if save_vis_path!=None:
        save_dir, save_name = os.path.split(save_vis_path)
        save_name = f'heatmap{layer}_' + save_name
        cv2.imwrite(os.path.join(save_dir, save_name), rgb_heat_map)








def BBoxesKmeans2Anchors(ann_path, seed=0):
    from sklearn.cluster import MiniBatchKMeans
    datas = []
    with open(ann_path, 'r') as json_file:
        content = json.load(json_file)
        annotations = content["annotations"]
        for box in tqdm(annotations):
            _,_,w,h = box['bbox']
            datas.append([w, h])
    datas = np.array(datas)
    cluster = MiniBatchKMeans(max_iter=1000, n_clusters=9, batch_size = 4096, random_state=seed).fit(datas)
    labels = cluster.predict(datas)
    return datas, cluster.cluster_centers_, labels
    








def bboxIoU(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """来自YOLOv8源码
    Calculate Intersection over Union (IoU) of box1(1, 4) to box2(n, 4). from yolo8 Ultralytics

    Args:
        box1 (torch.Tensor): A tensor representing a single bounding box with shape (1, 4).
        box2 (torch.Tensor): A tensor representing n bounding boxes with shape (n, 4).
        xywh (bool, optional): If True, input boxes are in (x, y, w, h) format. If False, input boxes are in
                            (x1, y1, x2, y2) format. Defaults to True.
        GIoU (bool, optional): If True, calculate Generalized IoU. Defaults to False.
        DIoU (bool, optional): If True, calculate Distance IoU. Defaults to False.
        CIoU (bool, optional): If True, calculate Complete IoU. Defaults to False.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): IoU, GIoU, DIoU, or CIoU values depending on the specified flags.
    """

    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp_(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw**2 + ch**2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi**2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU








# 平滑标签
def smooth_labels(y_true, label_smoothing, num_classes):
    return y_true * (1.0 - label_smoothing) + label_smoothing / num_classes









def get_near_points(x, y, i, j):
    '''获得正样本周围的两个相邻点,也作为正样本
    '''
    sub_x = x - i
    sub_y = y - j
    # scale_gt位于网格的第三象限
    if sub_x > 0.5 and sub_y > 0.5:
        return [[0, 0], [1, 0], [0, 1]]
    # scale_gt位于网格的第二象限
    elif sub_x < 0.5 and sub_y > 0.5:
        return [[0, 0], [-1, 0], [0, 1]]
    # scale_gt位于网格的第一象限
    elif sub_x < 0.5 and sub_y < 0.5:
        return [[0, 0], [-1, 0], [0, -1]]
    # scale_gt位于网格的第四象限
    else:
        return [[0, 0], [1, 0], [0, -1]]
















def YOLOv5BestRatioAssigner(targets, input_shape, anchors, anchors_mask, bbox_attrs, threshold=4):
    ''' - YOLOv5正负样本分配策略(根据gt与anchors的长宽比例, 是基于anchor_base的静态分配方法).
        - 该函数仅针对一张图片的分配, 因为在dataset里调用, 所以没实现batch的.
        - YOLOv5计算损失时只对正样本计算分类和回归损失, 唯一用到负样本的地方只有obj_map即这一部分.

        # Args
            - targets:         [num_gts, 5]
            - input_shape:     网络接受的输入图像的大小[640, 640]
            - anchors:         [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]]
            - anchors_mask:    [[0,1,2], [3,4,5], [6,7,8]]
            - bbox_attrs:      5 + num_cls
            - threshold:       anchor作为正样本时和GT的长宽比在 (1/threshold~threshold) 之间

        # Returns
            - output_targets:  分配结果 [[...],[...],[...]] output_targets[i].shape = (3, w, h, 5+cls_num)

    '''

    '''初始化'''
    # s代表特征图相对原图的下采样率
    s={0:8, 1:16, 2:32}
    num_layers  = len(anchors_mask)
    input_shape = np.array(input_shape, dtype='int32')
    # grid_shapes是每一层特征图的尺寸  [[20,20],[40,40],[80,80]]
    grid_shapes = [input_shape // s[l] for l in range(num_layers)] 
    # 初始化 output_targets = [[...],[...],[...]] 
    # 为每个特征层的每个锚框预留空间存储预测目标 output_targets[i].shape = (3, 20, 20, 5+cls_num), (3, 40, 40, 5+cls_num), (3, 80, 80, 5+cls_num) 5:(x0, y0, x1, y1, )
    output_targets = [np.zeros((len(anchors_mask[l]), grid_shapes[l][0], grid_shapes[l][1], bbox_attrs), dtype='float32') for l in range(num_layers)] 
    # 初始化 box_best_ratio = [[...],[...],[...]] 
    # 存储每个网格最佳锚框的匹配比例 box_best_ratio[i].shape = (3, 20, 20), (3, 40, 40), (3, 80, 80)
    best_ratio_scores = [np.zeros((len(anchors_mask[l]), grid_shapes[l][0], grid_shapes[l][1]), dtype='float32') for l in range(num_layers)] 
    # 没有GT, 则返回全0分配(全是负样本)
    if len(targets) == 0: return output_targets

    '''正负样本分配'''
    '''遍历每个特征层进行正负样本的分配(这样一个gt在不同特征层可能都会有匹配上的anchors)'''
    for l in range(num_layers):
        # 特征层的高和宽
        feat_h, feat_w = grid_shapes[l]
        # 将anchor调整到当前特征层的尺度
        lvl_anchors = np.array(anchors) / s[l] 
        scaled_targets = np.zeros_like(targets) # [num_gts, 5]
        # 对每个真实框计算其在当前特征层的位置
        scaled_targets[:, [0,2]] = targets[:, [0,2]] * feat_w
        scaled_targets[:, [1,3]] = targets[:, [1,3]] * feat_h
        scaled_targets[:, 4]     = targets[:, 4]
        # np.expand_dims(a, 1) 在a的第1维度添加一个新的维度: a = [bs, w, h] -> [bs, 1, w, h]
        # ratios_of_gt_anchors代表每一个真实框和每一个先验框的宽高的比值  [num_gts, 1, 2] [1, 9, 2] -> [num_gts, 9, 2]
        ratios_of_gt_anchors = np.expand_dims(scaled_targets[:, 2:4], 1) / np.expand_dims(lvl_anchors, 0)
        # 合并比率信息[num_true_box, 9, 4](取倒数是因为0.25~4的范围都行)
        aspect_ratios = np.concatenate([ratios_of_gt_anchors, 1 / ratios_of_gt_anchors], axis = -1)
        # 找出每个真实框与所有锚框中最佳匹配的IOU，即最大比率 [num_true_box, 9]
        max_ratios = np.max(aspect_ratios, axis = -1)
        
        '''对每个GT匹配对应的anchors'''
        for gt_id, ratio in enumerate(max_ratios):
            # 判断是否符合正样本条件
            valid_anchors = ratio < threshold
            # 强制保留最佳匹配锚框为正样本
            valid_anchors[np.argmin(ratio)] = True
            '''对每个GT一一检查当前层每个尺寸的anchor是否匹配, 是则作为正样本 k是不同尺寸的anchor, k=3'''
            for k, mask in enumerate(anchors_mask[l]):
                # 如果不符合正样本条件，跳过当前锚框组
                if not valid_anchors[mask]:
                    continue
                # 获得真实框属于哪个网格点(会取整)
                # 计算当前GT点的左上角x0对应网格的x索引
                i = int(np.floor(scaled_targets[gt_id, 0]))
                # 计算当前GT点的左上角y0对应网格的y索引
                j = int(np.floor(scaled_targets[gt_id, 1]))
                # 获取当前+附近的网格点偏移(这一步为了增加正样本数量)
                offsets = get_near_points(scaled_targets[gt_id, 0], scaled_targets[gt_id, 1], i, j)
                '''对每个正样本, 计算在特征图上的位置与gt value并记录到output_targets中'''
                for offset in offsets:
                    # 计算当前GT点的左上角x0对应网格的x索引(包括相邻网格点)
                    local_i = i + offset[0]
                    # 计算当前GT点的左上角y0对应网格的y索引(包括相邻网格点)
                    local_j = j + offset[1]
                    # 检查网格点是否超出特征图范围
                    if local_i >= feat_w or local_i < 0 or local_j >= feat_h or local_j < 0:
                        continue
                    # 如果该网格点已经有更佳匹配的锚框, 则跳过
                    if best_ratio_scores[l][k, local_j, local_i] != 0:
                        if best_ratio_scores[l][k, local_j, local_i] > ratio[mask]:
                            output_targets[l][k, local_j, local_i, :] = 0
                        else:
                            continue
                    
                    '''将正样本的信息写入output_targets'''
                    # 真实框的种类
                    c = int(scaled_targets[gt_id, 4])
                    # cx, cy, w, h
                    output_targets[l][k, local_j, local_i, :4] = scaled_targets[gt_id, :4]
                    # obj_map=1, 表示对应位置存在目标
                    output_targets[l][k, local_j, local_i, 4] = 1     
                    # 对应类别置为1, 其余为0 
                    output_targets[l][k, local_j, local_i, c + 5] = 1  
                    # 记录当前网格点的最佳匹配IOU
                    best_ratio_scores[l][k, local_j, local_i] = ratio[mask]
                    
    return output_targets










def inferDecodeBox(inputs, input_shape, num_classes, anchors, anchors_mask):
    '''YOLOv5将offset作用到anchor上进行解码得到最终预测结果(推理时用)
        # Args:
            - inputs:       多尺度特征图, inputs.shape = [1, 75, 80, 80], [1, 75, 40, 40], [1, 75, 30, 30]
            - input_shape:  网络接受的输入尺寸
            - num_classes:  类别数
            - anchors:      先验框(9个)
            - anchors_mask: 用于选取当前尺度特征图用哪些尺寸的先验框
        # Returns:
            - outputs: 存储不同尺度下的预测结果 [[...], [...], [...]] outputs[i].shape = [bs, num_anchors, 5+num_cls]
    '''    
    outputs = []
    for i, input in enumerate(inputs):
        batch_size      = input.size(0)
        input_height    = input.size(2)
        input_width     = input.size(3)
        # 输入为640x640时, stride_h = stride_w = 32 16 8
        stride_h = input_shape[0] / input_height
        stride_w = input_shape[1] / input_width
        # scaled_anchors是anchors相对于特征图的尺寸
        scaled_anchors  = [(a_w / stride_w, a_h / stride_h) for a_w, a_h in anchors]
        # 将预测结果reshape [1, 3*(5+num_cls), w, h] -> [1, 3, w, h, 5+num_cls]
        prediction = input.view(batch_size, len(anchors_mask[i]), 5 + num_classes, input_height, input_width).permute(0, 1, 3, 4, 2).contiguous()
        '''得到offset(通过sigmoid将offset的范围限制在(0,1)之间)'''
        # anchors的中心位置的调整(平移因子)(offsets)
        dx = torch.sigmoid(prediction[..., 0])  
        dy = torch.sigmoid(prediction[..., 1])
        # anchors的宽高调整参数(缩放因子)(offsets)
        dw = torch.sigmoid(prediction[..., 2]) 
        dh = torch.sigmoid(prediction[..., 3]) 
        # obj置信度, 是否有物体
        conf = torch.sigmoid(prediction[..., 4])
        # cls置信度
        pred_cls = torch.sigmoid(prediction[..., 5:])
        # 将offset作用到anchor上进行解码得到最终预测结果
        pred_boxes = YOLOv5DecodeBox(batch_size, i, dx, dy, dh, dw, anchors_mask, scaled_anchors, input_height, input_width)

        # 将输出结果转换成归一化坐标(0, 1)之间
        _scale = torch.Tensor([input_width, input_height, input_width, input_height]).type_as(dx)
        # output.shape = [bs, num_anchors, 5+num_cls]
        output = torch.cat((pred_boxes.view(batch_size, -1, 4) / _scale, conf.view(batch_size, -1, 1), pred_cls.view(batch_size, -1, num_classes)), -1)
        outputs.append(output.data)
    return outputs



def YOLOv5DecodeBox(bs, l, dx, dy, dh, dw, anchors_mask, scaled_anchors, in_h, in_w):
    '''YOLOv5将offset作用到anchor上进行解码得到最终预测结果
    '''
    # 将预测结果进行解码，判断预测结果和真实值的重合程度
    
    # 生成网格，先验框中心，网格左上角(特征图尺寸下的绝对坐标) 
    # grid_x.shape, grid_y.shape = [bs, 3, w, h]
    grid_x = torch.linspace(0, in_w - 1, in_w).repeat(in_h, 1).repeat(int(bs * len(anchors_mask[l])), 1, 1).view(dx.shape).type_as(dx)
    grid_y = torch.linspace(0, in_h - 1, in_h).repeat(in_w, 1).t().repeat(int(bs * len(anchors_mask[l])), 1, 1).view(dy.shape).type_as(dx)

    # 按照网格格式生成先验框的宽高(特征图尺寸下的绝对尺寸)
    # anchor_w.shape, anchor_h.shape = [bs, 3, w, h]
    scaled_anchors_l = np.array(scaled_anchors)[anchors_mask[l]] # 只根据anchors_mask取出对应层的anchor
    anchor_w = torch.Tensor(scaled_anchors_l).index_select(1, torch.LongTensor([0])).type_as(dx)
    anchor_h = torch.Tensor(scaled_anchors_l).index_select(1, torch.LongTensor([1])).type_as(dx)
    anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(dw.shape)
    anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(dh.shape)

    '''很重要!! 这部分是YOLOv5将对offset作用到anchor上进行解码得到最终预测结果的核心代码'''
    # NOTE:惨痛教训:这里不要写dx.data, 否则没有梯度, 导致训练时box_loss不收敛
    dx = dx * 2. - 0.5 # 将dx范围调整成(-0.5, 1.5)
    dy = dy * 2. - 0.5 # 将dy范围调整成(-0.5, 1.5)
    dw = (dw * 2) ** 2 # 将dw范围调整成(0, 4)
    dh = (dh * 2) ** 2 # 将dh范围调整成(0, 4)
    pred_boxes = torch.zeros((bs, 3, in_w, in_h, 4), device=dx.device)
    pred_boxes[..., 0] = grid_x + dx
    pred_boxes[..., 1] = grid_y + dy
    pred_boxes[..., 2] = anchor_w * dw
    pred_boxes[..., 3] = anchor_h * dh
    # pred_boxes.shape = [bs, 3, w, h, 4]
    return pred_boxes
    












def cxcywh2xyxy(box_xy, box_wh, input_shape, image_shape):
    # 把y轴放前面是因为方便预测框和图像的宽高进行相乘
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = np.array(input_shape)
    image_shape = np.array(image_shape)

    box_mins    = box_yx - (box_hw / 2.)
    box_maxes   = box_yx + (box_hw / 2.)
    boxes  = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]], axis=-1)
    boxes *= np.concatenate([input_shape, input_shape], axis=-1)
    return boxes



def non_max_suppression(prediction, input_shape, conf_thres=0.5, nms_thres=0.4, agnostic=False):
    '''推理一张图/一帧
        # Args:
            - prediction:  不同特征层的预测结果concat在一起
            - num_classes: 类别数
            - input_shape: 网络接受的输入尺寸
            - image_shape: 原图像尺寸
            - conf_thres:  nms 置信度阈值
            - nms_thres:   nms iou阈值
            - agnostic:    是否执行类无关的nms
        # Returns:
            - output: 最终预测结果, shape=[num_pred_objs, 7] 7的内容为 x1, y1, x2, y2, obj_conf, cls_score, class_id
    '''    
    #   将预测结果的cxcywh格式转换成xyxy的格式。
    #   prediction = [bs, num_anchors, num_cls+5]
    box_corner         = prediction.new(prediction.shape)
    box_corner[:,:, 0] = prediction[:,:, 0] - prediction[:,:, 2] / 2
    box_corner[:,:, 1] = prediction[:,:, 1] - prediction[:,:, 3] / 2
    box_corner[:,:, 2] = prediction[:,:, 0] + prediction[:,:, 2] / 2
    box_corner[:,:, 3] = prediction[:,:, 1] + prediction[:,:, 3] / 2
    prediction[:,:,:4] = box_corner[:,:,:4]
    # 因为预测时bs=1，所以len(output) = 1
    output = []
    # batch里每张图像逐图像进行nms
    for i, image_pred in enumerate(prediction):
        # 取出每个anchor预测置信度最大的那个类别的置信度以及类别索引
        # class_conf  [num_anchors, 1]    种类置信度
        # class_pred  [num_anchors, 1]    种类
        cls_score, cls_id = torch.max(image_pred[:, 5:], 1, keepdim=True)
        '''首先筛选掉置信度小于阈值的预测 '''
        # class_conf  [num_anchors]
        conf_keep = (image_pred[:, 4] * cls_score[:, 0] >= conf_thres).squeeze()
        image_pred = image_pred[conf_keep]
        cls_score = cls_score[conf_keep]
        cls_id = cls_id[conf_keep]
        # 如果筛选之后没有目标保留，则跳过继续
        if not image_pred.size(0): continue
        # 将坐标和预测的置信度，类别拼在一起 detections  [num_anchors, 7]
        detections = torch.cat((image_pred[:, :5], cls_score.float(), cls_id.float()), 1)
        if agnostic:
            '''类别无关nms(eval时使用这个一般会掉点)'''
            result = NMSbyAll(detections, nms_thres).cpu().numpy()
        else:
            '''逐类别nms'''
            result = NMSbyCLS(detections, nms_thres).cpu().numpy()
        if len(result) > 0:
            # 框由归一化坐标映射回原图的绝对坐标
            result[:, :4] *= np.concatenate([input_shape, input_shape], axis=-1)
        output.append(result)
    return output




def NMSbyCLS(predicts, nms_thres):
    '''逐类别nms'''
    cls_output = torch.tensor([])
    unique_cats = predicts[:, -1].cpu().unique()
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
    '''类别无关的nms'''
    # 使用官方自带的非极大抑制会速度更快一些
    final_cls_score = predicts[:, 4] * predicts[:, 5]
    '''接着筛选掉nms大于nms_thres的预测''' 
    keep = nms(predicts[:, :4], final_cls_score, nms_thres)
    nms_detections = predicts[keep]
    
    return nms_detections





    

    










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