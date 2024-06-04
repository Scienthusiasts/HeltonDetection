import numpy as np
import torch
from torchvision.ops import nms
from torch.nn import functional as F






def genAnchor(anchors_wh):
    '''生成FasterRCNN基础的先验框(默认特征图每个grid有9个anchor, 三种尺度, 每个尺度三种形状)
        Args:
            - param baseSize: 特征图一个像素代表原图几个像素(即下采样率, 600/38 ≈ 16)
            - param ratios:   anchor的长宽比
            - param scales:   anchor的尺度

        Returns:
            anchor: shape=[9, 4], anchor的四个坐标是以输入原始图像的尺寸而言
    '''
    anchors = []
    for lvl_anchors_wh in anchors_wh:
        lvl_anchors_num = len(lvl_anchors_wh)
        lvl_anchors = torch.zeros((lvl_anchors_num, 4), dtype=torch.float32)
        for i, wh in enumerate(lvl_anchors_wh):
            w, h = wh
            lvl_anchors[i, 0] = - w / 2.
            lvl_anchors[i, 1] = - h / 2.
            lvl_anchors[i, 2] = w / 2.
            lvl_anchors[i, 3] = h / 2.        
        # 格式xyxy
        anchors.append(lvl_anchors)
    return anchors



def genAnchorNP(anchors_wh):
    '''生成FasterRCNN基础的先验框(默认特征图每个grid有9个anchor, 三种尺度, 每个尺度三种形状)
        Args:
            - param baseSize: 特征图一个像素代表原图几个像素(即下采样率, 600/38 ≈ 16)
            - param ratios:   anchor的长宽比
            - param scales:   anchor的尺度

        Returns:
            anchor: shape=[9, 4], anchor的四个坐标是以输入原始图像的尺寸而言
    '''
    anchors = []
    for lvl_anchors_wh in anchors_wh:
        lvl_anchors_num = len(lvl_anchors_wh)
        lvl_anchors = np.zeros((lvl_anchors_num, 4), dtype=np.float32)
        for i, wh in enumerate(lvl_anchors_wh):
            w, h = wh
            lvl_anchors[i, 0] = - w / 2.
            lvl_anchors[i, 1] = - h / 2.
            lvl_anchors[i, 2] = w / 2.
            lvl_anchors[i, 3] = h / 2.        
        # 格式xyxy
        anchors.append(lvl_anchors)
    return anchors






def decode(cat_nums, roi_cls_loc, roi_score, roi):
    '''利用Head的预测结果对RPN proposals进行微调+解码, 获得预测框
        Args:
            - param cat_nums:    数据集类别数
            - param roi_cls_loc: Head对RPN的proposal进行回归得到的微调offset [1, 300, 21*4]
            - param roi_score:   Head对RPN的proposal进行分类得到的分类结果   [1, 300, 21]
            - param roi:         RPN预测的proposals的集合                   [1, 300, 4]
            - param nms_iou:     nms过滤的最小IoU
            - param confidence:  bbox的置信度

        Returns:
            - param boxes:       nms后的回归框坐标
            - param box_scores:  nms后的回归框得分
            - param box_classes: nms后的回归框类别
    '''
    # 对回归参数进行调整(归一化标准差)
    roi_cls_loc = roi_cls_loc * torch.Tensor([0.1, 0.1, 0.2, 0.2]).repeat(cat_nums+1)[None].to(roi_cls_loc.device)
    # roi_cls_loc.shape: [600, 21*4]->[600, 21, 4]
    roi_cls_loc = roi_cls_loc.view([-1, cat_nums+1, 4])
    # 同一区域不同类别都复制相同一份RoI[300, 4] -> [300, 21, 4](RoI也是原图尺度下的坐标)
    roi = roi.view((-1, 1, 4)).expand_as(roi_cls_loc)
    # 利用Head预测结果对RPN的proposals进行微调获得预测框(原图尺度下的坐标)
    cls_bbox = reg2Bbox(roi.reshape((-1, 4)), roi_cls_loc.reshape((-1, 4)))
    # [6300, 4] -> [300, 21, 4]
    cls_bbox = cls_bbox.view([-1, (cat_nums+1), 4])
    # 对logitsoftmax得到置信度(softmax是单分类, 这一步就决定了每个区域只能预测一个类别)
    scores = F.softmax(roi_score, dim=-1)
    return cls_bbox, scores




def NMSByCls(cat_nums, cls_bbox, scores, nms_iou, T):
    '''nms后处理(逐类别进行NMS,缺点:不同类别之间如果IOU很高, 则依然会保留)
        Args:
            - param cat_nums: 数据集类别数
            - param cls_bbox: Head回归的坐标(按类别划分)      [300, 21, 4]
            - param scores:   Head回归的框置信度(按类别划分)  [300, 21]
            - param nms_iou:  nms过滤的最小IoU
            - param T:        高于阈值的框才参与nms

        Returns:
            - param boxes:       nms后的回归框坐标
            - param box_scores:  nms后的回归框得分
            - param box_classes: nms后的回归框类别
    '''
    boxes, box_scores, box_classes = [], [], []
    # 第0类是背景类, 不考虑
    for c in range(1, cat_nums+1):
        # 取出属于该类的所有框的置信度
        box_scores_per_cat = scores[:, c]  # [300]
        # 判断是否大于置信度阈值(0.3)
        box_threshold = box_scores_per_cat > T
        if len(box_scores_per_cat[box_threshold]) > 0:
            # 得分高于confidence的框才参与nms
            boxes_to_process = cls_bbox[box_threshold, c]
            confs_to_process = box_scores_per_cat[box_threshold]
            # nms筛选(keep是保留的box的索引)
            # boxes_to_process.shape = [obj_nums, 4], confs_to_process.shape = [obj_nums]
            keep = nms(boxes_to_process, confs_to_process, nms_iou)
            # 取出nms筛选后保留的框
            keep_boxes = boxes_to_process[keep].cpu().numpy()
            keep_scores = confs_to_process[keep].cpu().numpy()
            keep_classes = (c - 1) * np.ones(len(keep)).astype(np.int32)
            # 把每个类别下的box都聚在一起
            boxes.append(keep_boxes)
            box_scores.append(keep_scores)
            box_classes.append(keep_classes)
    if len(boxes) > 0:
        return np.vstack(boxes), np.hstack(box_scores), np.hstack(box_classes)
    else:
        return np.array([]), np.array([]), np.array([])
    






def NMSAll(cls_bbox, scores, nms_iou, T):
    '''nms后处理(所有类别一起NMS,特点:不同类别之间如果IOU很高, 则只保留置信度较高的那个)
        Args:
            - param cat_nums: 数据集类别数
            - param cls_bbox: Head回归的坐标(按类别划分)    [300, 21, 4]
            - param scores: Head回归的框置信度(按类别划分)  [300, 21]
            - param nms_iou:  nms过滤的最小IoU
            - param T:        高于阈值的框才参与nms

        Returns:
            - param boxes:       nms后的回归框坐标
            - param box_scores:  nms后的回归框得分
            - param box_classes: nms后的回归框类别
    '''
    # 类别0为背景,舍弃掉这部分的回归框
    cls_bbox = cls_bbox[:, 1:, :]
    scores = scores[:, 1:]
    # 取出每个框对应类别置信度最大的那个类别 [300]
    box_labels = torch.argmax(scores, dim=1)
    # 取出每个框对应类别置信度最大的那个置信度 [300, 20] -> [300]
    box_scores = scores[torch.arange(scores.shape[0]), box_labels]
    # 只保留最高置信度类别的回归框 [300, 20, 4] -> [300, 4]
    cls_bbox = cls_bbox[torch.arange(cls_bbox.shape[0]), box_labels]
    # 判断是否大于置信度阈值(0.3)
    threshold_idx = box_scores > T
    cls_bbox = cls_bbox[threshold_idx]
    box_scores = box_scores[threshold_idx]
    box_labels = box_labels[threshold_idx]
    if len(cls_bbox) > 0:
        # nms筛选(keep是保留的box的索引)
        keep = nms(cls_bbox, box_scores, nms_iou)
        # 取出nms筛选后保留的框
        keep_boxes = cls_bbox[keep].cpu().numpy()
        keep_scores = box_scores[keep].cpu().numpy()
        keep_labels = box_labels[keep].cpu().numpy()

        return keep_boxes, keep_scores, keep_labels
    else:
        return np.array([]), np.array([]), np.array([])





    





def reg2Bbox(anchor, reg):
    '''将RPN网络预测的offset作用在anchor上生成proposal 或 将Head预测的offset作用在RPN proposals上生成最终结果
       总之, reg2Bbox()将reg(对anchor的微调结果)作用在anchor之上得到更精确的框
        Args:
            - param anchor: 所有的先验anchor 或 前一阶段的 proposals(原图尺寸)
            - param reg:    RPN的微调结果    或 Head的微调结果

        Returns:
            调整后的proposal
    '''
    # 没有anchor
    if anchor.size()[0] == 0:
        return torch.zeros((0, 4), dtype=reg.dtype)
    # anchor的格式xyxy->cxcywh
    anchorW  = torch.unsqueeze(anchor[:, 2] - anchor[:, 0], -1)
    anchorH  = torch.unsqueeze(anchor[:, 3] - anchor[:, 1], -1)
    anchorX  = torch.unsqueeze(anchor[:, 0], -1) + 0.5 * anchorW
    anchorY  = torch.unsqueeze(anchor[:, 1], -1) + 0.5 * anchorH
    # RPN预测的offset
    dx = reg[:, [0]]
    dy = reg[:, [1]]
    dw = reg[:, [2]]
    dh = reg[:, [3]]
    '''核心代码, FasterRCNN预测的结果如何解码为box'''
    '''FasterRCNN是对网格的中心点和宽高做微调'''
    # anchor+offset->proposal
    # FasterRCNN预测的dx, dy是中心点的偏移量; dw, dh是w, h的偏移量
    proposalX = dx * anchorW + anchorX
    proposalY = dy * anchorH + anchorY
    proposalW = torch.exp(dw) * anchorW
    proposalH = torch.exp(dh) * anchorH

    # proposal的格式cxcywh->xyxy
    dst_bbox = torch.zeros_like(reg)
    dst_bbox[:, [0]] = proposalX - 0.5 * proposalW
    dst_bbox[:, [1]] = proposalY - 0.5 * proposalH
    dst_bbox[:, [2]] = proposalX + 0.5 * proposalW
    dst_bbox[:, [3]] = proposalY + 0.5 * proposalH

    return dst_bbox





def applyAnchor2FeatNP(anchor_base, feat_stride, img_size):
    '''将基础先验框(9个)进行拓展到特征图所有像素上
        Args:
            - param anchor: 所有的先验anchor
            - param reg:    RPN的回归结果

        Returns:
            调整后的anchor
    '''
    anchors = []
    for lvl_anchor, lvl_stride in zip(anchor_base, feat_stride):
        shift_x             = np.arange(0, img_size[0], lvl_stride)
        shift_y             = np.arange(0, img_size[1], lvl_stride)
        shift_x, shift_y    = np.meshgrid(shift_x, shift_y)
        shift               = np.stack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel(),), axis=1)

        # 每个网格点上的9个先验框
        A       = lvl_anchor.shape[0]
        K       = shift.shape[0]
        lvl_all_anchor  = lvl_anchor.reshape((1, A, 4)) + shift.reshape((K, 1, 4))
        # 所有的先验框
        lvl_all_anchor  = lvl_all_anchor.reshape((K * A, 4))
        anchors.append(lvl_all_anchor)
    return anchors







def applyAnchor2Feat(anchor_base, feat_stride, img_size):
    '''将基础先验框(9个)进行拓展到特征图所有像素上
        Args:
            - param anchor: 所有的先验anchor
            - param reg:    RPN的回归结果

        Returns:
            调整后的anchor
    '''
    anchors = []
    for lvl_anchor, lvl_stride in zip(anchor_base, feat_stride):
        shift_x = torch.arange(0, img_size[0], lvl_stride)
        shift_y = torch.arange(0, img_size[1], lvl_stride)
        shift_x, shift_y = torch.meshgrid(shift_x, shift_y, indexing='xy')
        shift = torch.stack((shift_x.flatten(), shift_y.flatten(), shift_x.flatten(), shift_y.flatten()), dim=1)

        # 每个网格点上的9个先验框
        A       = lvl_anchor.shape[0]
        K       = shift.shape[0]
        lvl_all_anchor  = lvl_anchor.reshape((1, A, 4)) + shift.reshape((K, 1, 4))
        # 所有的先验框
        lvl_all_anchor  = lvl_all_anchor.reshape((K * A, 4))
        anchors.append(lvl_all_anchor)
    return anchors









def bbox2reg(srcBbox, dstBbox):
    '''通过调整前后的bbox参数推得网络预测回归offset的GT值(格式:xyxy)
        Args:
            - param srcBbox: 调整前的框
            - param dstBbox: GT框

        Returns:
            - param reg: offset的GT值(计算损失用)
    '''
    # 调整前的框的参数
    srcW = srcBbox[:, 2] - srcBbox[:, 0]
    srcH = srcBbox[:, 3] - srcBbox[:, 1]
    srcX = srcBbox[:, 0] + 0.5 * srcW
    srcY = srcBbox[:, 1] + 0.5 * srcH
    # 调整后的框的参数
    dstW = dstBbox[:, 2] - dstBbox[:, 0]
    dstH = dstBbox[:, 3] - dstBbox[:, 1]
    dstX = dstBbox[:, 0] + 0.5 * dstW
    dstY = dstBbox[:, 1] + 0.5 * dstH
    # eps是一个很小但>0的数,类型和height一致
    eps = torch.tensor(torch.finfo(srcH.dtype).eps, device=srcBbox.device)
    # 将可能出现的负数和零，使用eps来替换，这样在log或除法里就不会出现错误了
    srcW = torch.maximum(srcW, eps)
    srcH = torch.maximum(srcH, eps)
    # 计算出网络预测的offset(通过论文的公式)
    dx = (dstX - srcX) / srcW
    dy = (dstY - srcY) / srcH
    dw = torch.log(dstW / srcW)
    dh = torch.log(dstH / srcH)
    # 重新拼成回归参数
    reg = torch.vstack([dx, dy, dw, dh]).t()
    return reg




def bbox2regNP(srcBbox, dstBbox):
    '''通过调整前后的bbox参数推得网络预测回归offset的GT值(格式:xyxy)
        Args:
            - param srcBbox: 调整前的框
            - param dstBbox: GT框

        Returns:
            - param reg: offset的GT值(计算损失用)
    '''
    # 调整前的框的参数
    srcW = srcBbox[:, 2] - srcBbox[:, 0]
    srcH = srcBbox[:, 3] - srcBbox[:, 1]
    srcX = srcBbox[:, 0] + 0.5 * srcW
    srcY = srcBbox[:, 1] + 0.5 * srcH
    # 调整后的框的参数
    dstW = dstBbox[:, 2] - dstBbox[:, 0]
    dstH = dstBbox[:, 3] - dstBbox[:, 1]
    dstX = dstBbox[:, 0] + 0.5 * dstW
    dstY = dstBbox[:, 1] + 0.5 * dstH
    # eps是一个很小但>0的数,类型和height一致
    eps = np.finfo(srcH.dtype).eps
    # 将可能出现的负数和零，使用eps来替换，这样在log或除法里就不会出现错误了
    srcW = np.maximum(srcW, eps)
    srcH = np.maximum(srcH, eps)
    # 计算出网络预测的offset(通过论文的公式)
    dx = (dstX - srcX) / srcW
    dy = (dstY - srcY) / srcH
    dw = np.log(dstW / srcW)
    dh = np.log(dstH / srcH)
    # 重新拼成回归参数
    reg = np.vstack((dx, dy, dw, dh)).transpose()
    return reg











def calcBoxIoU(a, b):
    '''计算a和b两两的IoU
        Args:
            - param srcBbox: 调整前的框
            - param dstBbox: 利用网络预测的offset调整后的框

        Returns:
            - param IoU: a和b两两的IoU
    '''  
    # a.shape = [aNums, 4], b.shape = [bNums, 4]
    # 如果格式错误，抛出异常
    if a.shape[1] != 4 or b.shape[1] != 4:
        # 不是box的格式
        raise IndexError
    # 寻找两个box最左上角的点(a添加一个维度)
    tl = np.maximum(a[:, None, :2], b[:, :2])    # [aNums, bNums, 2]
    # 寻找两个box最右下角的点 (之所以a要再添加一个维度是为了利用广播机制使得所有的a和b都两两的计算IoU)
    br = np.minimum(a[:, None, 2:], b[:, 2:])    # [aNums, bNums, 2]
    # a框的面积(w*h) 
    aArea = np.prod(a[:, 2:] - a[:, :2], axis=1) # [aNums]
    # b框的面积(w*h)
    bArea = np.prod(b[:, 2:] - b[:, :2], axis=1) # [bNums]
    # a ∩ b 的面积(乘(tl < br)是为了保证框的面积最小为0而不是负数)
    intersection = np.prod(br - tl, axis=2) * (tl < br).all(axis=2) # [aNums, bNums]
    # a ∪ b 的面积 (之所以a要再添加一个维度是为了利用广播机制使得所有的a和b都两两的计算IoU)
    union = aArea[:, None] + bArea - intersection # [aNums, bNums]
    # IoU = a ∩ b 的面积 / a ∪ b 的面积 
    return intersection / union




def computeBoxIoU(anchors, gt_boxes):
    """Compute IoU between each pair of anchor and gt_box using vectorized operations."""
    # Calculate intersection
    inter_x1 = torch.max(anchors[:, None, 0], gt_boxes[:, 0])
    inter_y1 = torch.max(anchors[:, None, 1], gt_boxes[:, 1])
    inter_x2 = torch.min(anchors[:, None, 2], gt_boxes[:, 2])
    inter_y2 = torch.min(anchors[:, None, 3], gt_boxes[:, 3])
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    
    # Calculate union
    anchor_area = (anchors[:, 2] - anchors[:, 0]) * (anchors[:, 3] - anchors[:, 1])
    gt_box_area = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
    union_area = anchor_area[:, None] + gt_box_area - inter_area
    
    return inter_area / union_area




def computeBoxIoUNP(anchors, gt_boxes):
    """Compute IoU between each pair of anchor and gt_box using vectorized operations."""
    # Calculate intersection
    inter_x1 = np.maximum(anchors[:, None, 0], gt_boxes[:, 0])
    inter_y1 = np.maximum(anchors[:, None, 1], gt_boxes[:, 1])
    inter_x2 = np.minimum(anchors[:, None, 2], gt_boxes[:, 2])
    inter_y2 = np.minimum(anchors[:, None, 3], gt_boxes[:, 3])
    inter_area = np.clip(inter_x2 - inter_x1, 0, None) * np.clip(inter_y2 - inter_y1, 0, None)
    
    # Calculate union
    anchor_area = (anchors[:, 2] - anchors[:, 0]) * (anchors[:, 3] - anchors[:, 1])
    gt_box_area = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
    union_area = anchor_area[:, None] + gt_box_area - inter_area
    
    return inter_area / union_area











def dense2SparseProposals(roi, cls, imgSize, scale, pre_nms_num, post_nms_num, nms_iou=0.7):
    '''对RPN生成的proposal进行筛选(单张图像):去掉太小的框 + 置信度太低的框 + nms过滤的框
        Args:
            - param roi:          proposals的坐标(原图尺寸)
            - param cls:          RPN的分类结果
            - param imgSize:      原始图像尺寸
            - param scale:        当前尺度的下采样率
            - param pre_nms_num:  参与nms的proposals数量
            - param post_nms_num: nms后保留的proposals数量
            - param nms_iou:      nmsiou阈值

        Returns:
            - param roi:        保留的proposals, 这些proposals参与head进一步微调
            - param origin_roi: proposal的坐标(原图尺寸下)
    '''

    # 调整防止建议框超出图像边缘
    roi[:, [0, 2]] = torch.clamp(roi[:, [0, 2]], min = 0, max = imgSize[1])
    roi[:, [1, 3]] = torch.clamp(roi[:, [1, 3]], min = 0, max = imgSize[0])
    origin_roi = roi

    '''去掉太小的框'''
    # 只保留那些宽高均>scale(当前特征层的下采样率)的proposal
    keep = torch.where(((roi[:, 2] - roi[:, 0]) >= scale) & ((roi[:, 3] - roi[:, 1]) >= scale))[0]
    roi, cls = roi[keep, :], cls[keep]

    '''去掉置信度太低的框'''
    # 根据前景置信度大小进行排序, 且直接通过卡阈值保留置信度Top K(K=PreNMSNum)的proposal
    # 这些proposals里依然可能存在与GT IoU低的框
    order = torch.argsort(cls, descending=True)[:pre_nms_num]
    roi, cls = roi[order, :], cls[order]

    '''去掉nms过滤的框'''
    keep = nms(roi, cls, nms_iou)
    # 如果proposal筛选后的个数不足PostNMSNum,则随机重复添加直到proposal个数=PostNMSNum
    if len(keep) < post_nms_num:
        index_extra = np.random.choice(range(len(keep)), size=(post_nms_num - len(keep)), replace=True)
        keep = torch.cat([keep, keep[index_extra]])
    # 保留nms筛选后的proposal
    keep = keep[:post_nms_num]
    roi = roi[keep]
    return roi, origin_roi













# FasterRCNN RPN正负样本分配策略
def RPNMaxIoUAssignerNP(gt_boxes, anchors, total_samples=256, pos_threshold=0.7, neg_threshold=0.3, pos_ratio=0.5):
    '''正负样本分配(单张图像)(为每一个anchor都分配一个标签以及回归offset(正例1:每个正例都分配一个GT; 负例0:背景,不参与预测回归; 其他-1:不参与计算损失))
       面向第一阶段的RPN

        Args:
            - param gt_boxes:      GT框
            - param featStride:    fpn每个特征图的下采样率
            - param img_size:      frcnn网络接受输入图像尺寸
            - param anchors:       anchor框
            - param total_samples: 一幅图像最多采样几个样本参与损失的计算
            - param pos_threshold: anchor与GT的IoU大于这个值就作为正例
            - param neg_threshold: anchor与GT的IoU小于这个值就作为负例
            - param pos_ratio:     正样本与负样本的采样比例

        Returns:
            - param RPNreg:                       正例anchor对应的RPN网络回归offset的GT
            - param labels:                       anchor的标签(0, 1, -1)
            - param gt_boxes[matched_gt_indices]: 每个anchor对应GT的索引
    '''
    y_trues = []
    # 对每一个尺度分别分配样本
    for lvl_anchor in anchors:
        num_anchors = lvl_anchor.shape[0]
        # 初始化所有标签为中性(-1) [w*h*3, ]
        labels = np.full((num_anchors,), -1, dtype=np.float32)  

        '''MaxIoUAssign(核心部分)'''
        # 计算每个锚点与每个真实框之间的IoU [w*h*3, gt_nums]
        iou = computeBoxIoUNP(lvl_anchor, gt_boxes)
        # 根据IoU值分配正样本和匹配的GT索引
        # max_indices是anchor对应最匹配(IoU最大)的gt的索引
        max_indices = np.argmax(iou, axis=1) # [w*h*3, ]
        # max_iou是anchor对应最匹配(IoU最大)的gt的IoU
        max_iou = np.max(iou, axis=1)        # [w*h*3, ]
        # 和至少一个GT的IoU大于阈值的anchor被设置为前景
        labels[max_iou >= pos_threshold] = 1
        # 和任意GT的IoU均小于阈值的anchor被设置为背景
        labels[max_iou < neg_threshold] = 0

        '''确保每个GT至少有一个对应的正样本anchor'''
        # 获取每个GT的最佳(最大IoU)anchor索引
        gt_max_indices = np.argmax(iou, axis=0)
        labels[gt_max_indices] = 1
        max_indices[gt_max_indices] = np.arange(gt_boxes.shape[0])  

        '''限制正负样本数量'''
        pos_indices = np.where(labels == 1)[0]
        # 判断正样本数量是否大于阈值，如果大于则随机舍弃一些
        max_positives = int(min(total_samples * pos_ratio, len(pos_indices)))
        if len(pos_indices) > max_positives:
            disable_indices = np.random.choice(pos_indices, size=(len(pos_indices) - max_positives), replace=False)
            # 清除被舍弃的正样本
            labels[disable_indices] = -1

        neg_indices = np.where(labels == 0)[0]
        # 判断负样本数量是否大于阈值，如果大于则随机舍弃一些
        max_negatives = total_samples - len(np.where(labels == 1)[0])
        if len(neg_indices) > max_negatives:
            disable_indices = np.random.choice(neg_indices, size=(len(neg_indices) - max_negatives), replace=False)
            # 清除被舍弃的负样本
            labels[disable_indices] = -1
        
        '''整理分配结果, 作为计算损失时的label'''
        # 只要有一个anchor是正样本, 就获取所有anchor对应的RPN网络回归的offset的GT [w*h*3, 4]
        RPNreg = bbox2regNP(lvl_anchor, gt_boxes[max_indices]) if (labels > 0).any() else np.zeros_like(lvl_anchor)
        # 返回正/负例/不参与训练anchor对应的RPN网络回归值和这个anchor的标签(0, 1, -1)
        y_true = np.concatenate((RPNreg, labels[:, None], gt_boxes[max_indices]), axis=1)
        y_trues.append(y_true)
    return y_trues

















def ROIHeadMaxIoUAssignerNP(roi, bbox, label, regNormStd=(0.1, 0.1, 0.2, 0.2), maxRoINum=128, posRatio=0.5, posIoUT=0.5, negIoUTMax=0.5, negIoUTMin=0):
    '''正负样本分配(筛选proposal, 并为每一个保留的proposal分配标签和offset的GT)
       面向第二阶段

        Args:
            - param roi:        RPN微调后的proposal [600, 4]
            - param bbox:       GT box
            - param label:      roi的类别标签
            - param regNormStd: 用于回归offset归一化的标准差??
            - param maxRoINum:  一幅图像最多采样几个样本参与损失的计算
            - param posRatio:   正样本与负样本的采样比例
            - param posIoUT:    proposal与GT的IoU大于这个值就作为正例
            - param negIoUTMax: proposal与GT的IoU大于negIoUTMin小于negIoUTMax就作为负例
            - param negIoUTMin: proposal与GT的IoU大于negIoUTMin小于negIoUTMax就作为负例

        Returns:
            - param sample_roi:     最终被保留参与计算损失的正负样本
            - param sample_roi_gts: 最终被保留参与计算损失的正负样本对应的GT框
            - param gt_roi_loc:     最终被保留参与计算损失的正负样本对应的offset的GT
            - param gt_roi_label:   最终被保留参与计算损失的正负样本对应的类别标签
    '''
    pos_roi_per_image = np.round(maxRoINum * posRatio) # 64
    # 训练初期RPN可能无法得到较为优质的框,为了保证Head的训练质量,将GT框也作为Head回归的对象
    roi = np.concatenate((roi.detach().cpu().numpy(), bbox), axis=0)

    '''MaxIoUAssign(核心部分)'''
    if len(bbox) > 0:
        # 计算建议框和真实框的重合程度
        iou = calcBoxIoU(roi, bbox)
        # 获得每一个建议框最对应的真实框的索引  [num_roi, ]
        gt_assignment = iou.argmax(axis=1)
        # 获得每一个建议框最对应的真实框的iou  [num_roi, ]
        max_iou = iou.max(axis=1)
        # 真实框的标签要+1因为有背景的存在(背景是0)
        gt_roi_label = label[gt_assignment] + 1
    else:
        gt_assignment = np.zeros(len(roi), np.int32)
        max_iou = np.zeros(len(roi))
        gt_roi_label = np.zeros(len(roi))
    # 每个roi都对应一个gtbox(IoU最大的那个)
    roi_gts = bbox[gt_assignment]

    '''满足建议框和真实框重合程度大于posIoUT的作为正样本'''
    pos_index = np.where(max_iou >= posIoUT)[0]
    pos_roi_per_this_image = int(min(pos_roi_per_image, pos_index.size))
    # 将正样本的数量限制在pos_roi_per_image以内
    pos_index = np.random.choice(pos_index, size=pos_roi_per_this_image, replace=False)

    '''满足建议框和真实框重合程度小于negIoUTMax大于negIoUTMin作为负样本'''
    neg_index = np.where((max_iou < negIoUTMax) & (max_iou >= negIoUTMin))[0]
    neg_roi_per_this_image = maxRoINum - pos_roi_per_this_image
    neg_roi_per_this_image = int(min(neg_roi_per_this_image, neg_index.size))
    # 将正样本的数量和负样本的数量的总和固定成self.maxRoINum
    if neg_index.size > 0:
        neg_index = np.random.choice(neg_index, size=neg_roi_per_this_image, replace=False)
    
    # 最终被保留参与计算损失的正负样本的索引
    keep_index = np.append(pos_index, neg_index)
    # 最终被保留参与计算损失的正负样本
    sample_roi = roi[keep_index] # [maxRoINum, ]
    sample_roi_gts = roi_gts[keep_index]
    # 如果图像中没有GT:
    if len(bbox)==0:
        return sample_roi, np.zeros_like(sample_roi), gt_roi_label[keep_index]
    # 获取最终被保留参与计算损失的正负样本对应的offset的GT [maxRoINum, 4]
    gt_roi_loc = bbox2regNP(sample_roi, bbox[gt_assignment[keep_index]])
    # 归一化
    gt_roi_loc = (gt_roi_loc / np.array(regNormStd, np.float32))
    # 最终被保留参与计算损失的正负样本对应的类别标签 [maxRoINum, ]
    gt_roi_label = gt_roi_label[keep_index] 
    # 再次调整那些负样本的类别标签为0(有些可能在调整前还不是0)
    gt_roi_label[pos_roi_per_this_image:] = 0
    return sample_roi, sample_roi_gts, gt_roi_loc, gt_roi_label



















# for test only:
if __name__ == '__main__':
    anchors = genAnchor()
    print(anchors.shape)
    print(anchors)

