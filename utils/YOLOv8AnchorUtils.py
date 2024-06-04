import numpy as np
import torch
import math
from torchvision.ops import nms
import pkg_resources as pkg
import torch.nn as nn
import torch.nn.functional as F
import cv2
import os




def make_anchors(feats, strides, grid_cell_offset=0.5):
    """所谓anchor其实就是每个cell的中心点, stride就是每个anchor距离其他anchor的距离(和下采样率有关)
    """
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        _, _, h, w  = feats[i].shape
        sx          = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sy          = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
        sy, sx      = torch.meshgrid(sy, sx, indexing='ij') 
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points).transpose(0, 1), torch.cat(stride_tensor).transpose(0, 1)






def select_candidates_in_gts(xy_centers, gt_bboxes, eps=1e-9, roll_out=False):
    """select the positive anchor center in gt

    Args:
        xy_centers (Tensor): shape(h*w, 4)
        gt_bboxes (Tensor): shape(b, n_boxes, 4)
    Return:
        (Tensor): shape(b, n_boxes, h*w)
    """
    n_anchors       = xy_centers.shape[0]
    bs, n_boxes, _  = gt_bboxes.shape
    # 计算每个真实框距离每个anchors锚点的左上右下的距离，然后求min
    # 保证真实框在锚点附近，包围锚点
    if roll_out:
        bbox_deltas = torch.empty((bs, n_boxes, n_anchors), device=gt_bboxes.device)
        for b in range(bs):
            lt, rb          = gt_bboxes[b].view(-1, 1, 4).chunk(2, 2)  # left-top, right-bottom
            bbox_deltas[b]  = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]),
                                       dim=2).view(n_boxes, n_anchors, -1).amin(2).gt_(eps)
        return bbox_deltas
    else:
        # 真实框的坐上右下left-top, right-bottom 
        lt, rb      = gt_bboxes.view(-1, 1, 4).chunk(2, 2)  
        # 真实框距离每个anchors锚点的左上右下的距离
        bbox_deltas = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), dim=2).view(bs, n_boxes, n_anchors, -1)
        # return (bbox_deltas.min(3)[0] > eps).to(gt_bboxes.dtype)
        return bbox_deltas.amin(3).gt_(eps)


def select_highest_overlaps(mask_pos, overlaps, n_max_boxes):
    """if an anchor box is assigned to multiple gts,
        the one with the highest iou will be selected.

    Args:
        mask_pos (Tensor): shape(b, n_max_boxes, h*w)
        overlaps (Tensor): shape(b, n_max_boxes, h*w)
    Return:
        target_gt_idx (Tensor): shape(b, h*w)
        fg_mask (Tensor): shape(b, h*w)
        mask_pos (Tensor): shape(b, n_max_boxes, h*w)
    """
    # b, n_max_boxes, 8400 -> b, 8400
    fg_mask = mask_pos.sum(-2)
    # 如果有一个anchor被指派去预测多个真实框
    if fg_mask.max() > 1:  
        # b, n_max_boxes, 8400
        mask_multi_gts      = (fg_mask.unsqueeze(1) > 1).repeat([1, n_max_boxes, 1])  
        # 如果有一个anchor被指派去预测多个真实框，首先计算这个anchor最重合的真实框
        # 然后做一个onehot
        # b, 8400
        max_overlaps_idx    = overlaps.argmax(1)  
        # b, 8400, n_max_boxes
        is_max_overlaps     = F.one_hot(max_overlaps_idx, n_max_boxes)  
        # b, n_max_boxes, 8400
        is_max_overlaps     = is_max_overlaps.permute(0, 2, 1).to(overlaps.dtype)  
        # b, n_max_boxes, 8400
        mask_pos            = torch.where(mask_multi_gts, is_max_overlaps, mask_pos) 
        fg_mask             = mask_pos.sum(-2)
    # 找到每个anchor符合哪个gt
    target_gt_idx = mask_pos.argmax(-2)  # (b, h*w)
    return target_gt_idx, fg_mask, mask_pos





class TaskAlignedAssigner(nn.Module):

    def __init__(self, topk=13, num_classes=80, alpha=1.0, beta=6.0, eps=1e-9, roll_out_thr=0):
        super().__init__()
        self.topk           = topk
        self.num_classes    = num_classes
        self.bg_idx         = num_classes
        self.alpha          = alpha
        self.beta           = beta
        self.eps            = eps
        # roll_out_thr为64
        self.roll_out_thr   = roll_out_thr

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """This code referenced to
           https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py

        Args:
            pd_scores (Tensor)  : shape(bs, num_total_anchors, num_classes)
            pd_bboxes (Tensor)  : shape(bs, num_total_anchors, 4)
            anc_points (Tensor) : shape(num_total_anchors, 2)
            gt_labels (Tensor)  : shape(bs, n_max_boxes, 1)
            gt_bboxes (Tensor)  : shape(bs, n_max_boxes, 4)
            mask_gt (Tensor)    : shape(bs, n_max_boxes, 1)
        Returns:
            target_labels (Tensor)  : shape(bs, num_total_anchors)
            target_bboxes (Tensor)  : shape(bs, num_total_anchors, 4)
            target_scores (Tensor)  : shape(bs, num_total_anchors, num_classes)
            fg_mask (Tensor)        : shape(bs, num_total_anchors)
        """
        # 获得batch_size 
        self.bs             = pd_scores.size(0)
        # 获得真实框中的最大框数量
        self.n_max_boxes    = gt_bboxes.size(1)
        # 如果self.n_max_boxes大于self.roll_out_thr则roll_out
        self.roll_out       = self.n_max_boxes > self.roll_out_thr if self.roll_out_thr else False
    
        if self.n_max_boxes == 0:
            device = gt_bboxes.device
            return (torch.full_like(pd_scores[..., 0], self.bg_idx).to(device), torch.zeros_like(pd_bboxes).to(device),
                    torch.zeros_like(pd_scores).to(device), torch.zeros_like(pd_scores[..., 0]).to(device),
                    torch.zeros_like(pd_scores[..., 0]).to(device))

        # bs, max_num_obj, 8400
        # mask_pos      满足在真实框内、是真实框topk最重合的正样本、满足mask_gt的锚点
        # align_metric  某个先验点属于某个真实框的类的概率乘上某个先验点与真实框的重合程度
        # overlaps      所有真实框和锚点的重合程度
        mask_pos, align_metric, overlaps = self.get_pos_mask(pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt)

        # target_gt_idx     bs, 8400     每个anchor符合哪个gt
        # fg_mask           bs, 8400     每个anchor是否有符合的gt
        # mask_pos          bs, max_num_obj, 8400    one_hot后的target_gt_idx
        target_gt_idx, fg_mask, mask_pos = select_highest_overlaps(mask_pos, overlaps, self.n_max_boxes)

        # 指定目标到对应的anchor点上
        # b, 8400
        # b, 8400, 4
        # b, 8400, 80
        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)

        # 乘上mask_pos，把不满足真实框满足的锚点的都置0
        align_metric        *= mask_pos
        # 每个真实框对应的最大得分
        # b, max_num_obj
        pos_align_metrics   = align_metric.amax(axis=-1, keepdim=True) 
        # 每个真实框对应的最大重合度
        # b, max_num_obj
        pos_overlaps        = (overlaps * mask_pos).amax(axis=-1, keepdim=True)
        # 把每个真实框和先验点的得分乘上最大重合程度，再除上最大得分
        norm_align_metric   = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        # target_scores作为正则的标签
        target_scores       = target_scores * norm_align_metric

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        # pd_scores bs, num_total_anchors, num_classes
        # pd_bboxes bs, num_total_anchors, 4
        # gt_labels bs, n_max_boxes, 1
        # gt_bboxes bs, n_max_boxes, 4
        # 
        # align_metric是一个算出来的代价值，某个先验点属于某个真实框的类的概率乘上某个先验点与真实框的重合程度
        # overlaps是某个先验点与真实框的重合程度
        # align_metric, overlaps    bs, max_num_obj, 8400
        align_metric, overlaps  = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes)
        
        # 正样本锚点需要同时满足：
        # 1、在真实框内
        # 2、是真实框topk最重合的正样本
        # 3、满足mask_gt
        
        # get in_gts mask           b, max_num_obj, 8400
        # 判断先验点是否在真实框内
        mask_in_gts             = select_candidates_in_gts(anc_points, gt_bboxes, roll_out=self.roll_out)
        # get topk_metric mask      b, max_num_obj, 8400
        # 判断锚点是否在真实框的topk中
        mask_topk               = self.select_topk_candidates(align_metric * mask_in_gts, topk_mask=mask_gt.repeat([1, 1, self.topk]).bool())
        # merge all mask to a final mask, b, max_num_obj, h*w
        # 真实框存在，非padding
        mask_pos                = mask_topk * mask_in_gts * mask_gt

        return mask_pos, align_metric, overlaps

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes):
        if self.roll_out:
            align_metric    = torch.empty((self.bs, self.n_max_boxes, pd_scores.shape[1]), device=pd_scores.device)
            overlaps        = torch.empty((self.bs, self.n_max_boxes, pd_scores.shape[1]), device=pd_scores.device)
            ind_0           = torch.empty(self.n_max_boxes, dtype=torch.long)
            for b in range(self.bs):
                ind_0[:], ind_2 = b, gt_labels[b].squeeze(-1).long()
                # 获得属于这个类别的得分
                # bs, max_num_obj, 8400
                bbox_scores     = pd_scores[ind_0, :, ind_2]  
                # 计算真实框和预测框的ciou
                # bs, max_num_obj, 8400
                overlaps[b]     = bbox_iou(gt_bboxes[b].unsqueeze(1), pd_bboxes[b].unsqueeze(0), xywh=False, CIoU=True).squeeze(2).clamp(0)
                align_metric[b] = bbox_scores.pow(self.alpha) * overlaps[b].pow(self.beta)
        else:
            # 2, b, max_num_obj
            ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)       
            # b, max_num_obj  
            # [0]代表第几个图片的
            ind[0] = torch.arange(end=self.bs).view(-1, 1).repeat(1, self.n_max_boxes)  
            # [1]真是标签是什么
            ind[1] = gt_labels.long().squeeze(-1) 
            # 获得属于这个类别的得分
            # 取出某个先验点属于某个类的概率
            # b, max_num_obj, 8400
            bbox_scores = pd_scores[ind[0], :, ind[1]]  

            # 计算真实框和预测框的ciou
            # bs, max_num_obj, 8400
            overlaps        = bbox_iou(gt_bboxes.unsqueeze(2), pd_bboxes.unsqueeze(1), xywh=False, CIoU=True).squeeze(3).clamp(0)
            align_metric    = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps

    def select_topk_candidates(self, metrics, largest=True, topk_mask=None):
        """
        Args:
            metrics     : (b, max_num_obj, h*w).
            topk_mask   : (b, max_num_obj, topk) or None
        """
        # 8400
        num_anchors             = metrics.shape[-1] 
        # b, max_num_obj, topk
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=largest)
        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True) > self.eps).tile([1, 1, self.topk])
        # b, max_num_obj, topk
        topk_idxs[~topk_mask] = 0
        # b, max_num_obj, topk, 8400 -> b, max_num_obj, 8400
        # 这一步得到的is_in_topk为b, max_num_obj, 8400
        # 代表每个真实框对应的top k个先验点
        if self.roll_out:
            is_in_topk = torch.empty(metrics.shape, dtype=torch.long, device=metrics.device)
            for b in range(len(topk_idxs)):
                is_in_topk[b] = F.one_hot(topk_idxs[b], num_anchors).sum(-2)
        else:
            is_in_topk = F.one_hot(topk_idxs, num_anchors).sum(-2)
        # 判断锚点是否在真实框的topk中
        is_in_topk = torch.where(is_in_topk > 1, 0, is_in_topk)
        return is_in_topk.to(metrics.dtype)

    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        """
        Args:
            gt_labels       : (b, max_num_obj, 1)
            gt_bboxes       : (b, max_num_obj, 4)
            target_gt_idx   : (b, h*w)
            fg_mask         : (b, h*w)
        """

        # 用于读取真实框标签, (b, 1)
        batch_ind       = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
        # b, h*w    获得gt_labels，gt_bboxes在flatten后的序号
        target_gt_idx   = target_gt_idx + batch_ind * self.n_max_boxes
        # b, h*w    用于flatten后读取标签
        target_labels   = gt_labels.long().flatten()[target_gt_idx]
        # b, h*w, 4 用于flatten后读取box
        target_bboxes   = gt_bboxes.view(-1, 4)[target_gt_idx]
        
        # assigned target scores
        target_labels.clamp(0)
        # 进行one_hot映射到训练需要的形式。
        target_scores   = F.one_hot(target_labels, self.num_classes)  # (b, h*w, 80)
        fg_scores_mask  = fg_mask[:, :, None].repeat(1, 1, self.num_classes)  # (b, h*w, 80)
        target_scores   = torch.where(fg_scores_mask > 0, target_scores, 0)

        return target_labels, target_bboxes, target_scores







def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # Returns Intersection over Union (IoU) of box1(1,4) to box2(n,4)

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
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU




def bbox2dist(anchor_points, bbox, reg_max):
    """Transform bbox(xyxy) to dist(ltrb)."""
    x1y1, x2y2 = torch.split(bbox, 2, -1)
    return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp(0, reg_max - 0.01)  # dist (lt, rb)




class BboxLoss(nn.Module):
    def __init__(self, reg_max=16, use_dfl=False):
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        # 计算IOU损失
        # weight代表损失中标签应该有的置信度，0最小，1最大
        weight      = torch.masked_select(target_scores.sum(-1), fg_mask).unsqueeze(-1)
        # 计算预测框和真实框的重合程度
        iou         = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        # 然后1-重合程度，乘上应该有的置信度，求和后求平均。
        loss_iou    = ((1.0 - iou) * weight).sum() / target_scores_sum

        # 计算DFL损失
        if self.use_dfl:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl

    @staticmethod
    def _df_loss(pred_dist, target):
        # Return sum of left and right DFL losses
        # Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        # 一个点一般不会处于anchor点上，一般是xx.xx。如果要用DFL的话，不可能直接一个cross_entropy就能拟合
        # 所以把它认为是相对于xx.xx左上角锚点与右下角锚点的距离 如果距离右下角锚点距离小，wl就小，左上角损失就小
        #                                                   如果距离左上角锚点距离小，wr就小，右下角损失就小
        return (F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl +
                F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr).mean(-1, keepdim=True)






def xywh2xyxy(x):
    """
    Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner.

    Args:
        x (np.ndarray) or (torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.
    Returns:
        y (np.ndarray) or (torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y




def preprocess(targets, batch_size, scale_tensor):
    if targets.shape[0] == 0:
        out = torch.zeros(batch_size, 0, 5, device=targets.device)
    else:
        # 获得图像索引
        i           = targets[:, 0]  
        _, counts   = i.unique(return_counts=True)
        out         = torch.zeros(batch_size, counts.max(), 5, device=targets.device)
        # 对batch进行循环，然后赋值
        for j in range(batch_size):
            matches = i == j
            n = matches.sum()
            if n:
                out[j, :n] = targets[matches, 1:]
        # 缩放到原图大小。
        out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
    return out



def bbox_decode(anchor_points, pred_dist, proj, use_dfl):
    if use_dfl:
        # batch, anchors, channels
        b, a, c     = pred_dist.shape  
        # DFL的解码
        pred_dist   = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(proj.to(pred_dist.device).type(pred_dist.dtype))
        # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
    # 然后解码获得预测框
    return dist2bbox(pred_dist, anchor_points, xywh=False)




def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    # 左上右下
    lt, rb  = torch.split(distance, 2, dim)
    x1y1    = anchor_points - lt
    x2y2    = anchor_points + rb
    if xywh:
        c_xy    = (x1y1 + x2y2) / 2
        wh      = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)  # cxcywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox







def decode_box(inputs, input_shape):
    """将回归的ltrb作用到anchor点上生成框坐标.
    """
    # dbox  batch_size, 4, 8400
    # cls   batch_size, 20, 8400
    dbox, cls, x, anchors, strides = inputs
    # 获得中心宽高坐标
    dbox    = dist2bbox(dbox, anchors.unsqueeze(0), xywh=True, dim=1) * strides
    y       = torch.cat((dbox, cls.sigmoid()), 1).permute(0, 2, 1)
    # 进行归一化，到0~1之间
    y[:, :, :4] = y[:, :, :4] / torch.Tensor([input_shape[1], input_shape[0], input_shape[1], input_shape[0]]).to(y.device)
    return y













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
        #----------------------------------------------------------#
        #   对种类预测部分取max。
        #   class_conf  [num_anchors, 1]    种类置信度
        #   class_pred  [num_anchors, 1]    种类
        #----------------------------------------------------------#
        class_conf, class_pred = torch.max(image_pred[:, 4:], 1, keepdim=True)

        #----------------------------------------------------------#
        #   利用置信度进行第一轮筛选
        #----------------------------------------------------------#
        conf_mask = (class_conf[:, 0] >= conf_thres).squeeze()
        
        #----------------------------------------------------------#
        #   根据置信度进行预测结果的筛选
        #----------------------------------------------------------#
        image_pred = image_pred[conf_mask]
        class_conf = class_conf[conf_mask]
        class_pred = class_pred[conf_mask]
        if not image_pred.size(0):
            continue
        #-------------------------------------------------------------------------#
        #   detections  [num_anchors, 6]
        #   6的内容为：x1, y1, x2, y2, class_conf, class_pred
        #-------------------------------------------------------------------------#
        detections = torch.cat((image_pred[:, :4], class_conf.float(), class_pred.float()), 1)
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







def vis_YOLOv8_heatmap(predicts:torch.tensor, ori_shape, input_shape, image, box_classes, image2color, padding=True, save_vis_path=None):
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
    cls_num = predicts[0].shape[-1] - 64
    for layer, predict in enumerate(predicts):
        predict = predict[:, 64:, :, :].cpu()
        b, c, h, w = predict.shape
        # predict.shape = [w, h, cls_num], 取[0]是因为默认bs=1
        predict = np.transpose(predict, [0, 2, 3, 1])[0]
        '''提取类别置信度'''
        saveVisHeatMap(image, image2color, cls_num, predict, W, H, w, h, layer, input_shape, cut_region, save_vis_path)





def saveVisHeatMap(image, color:dict, cls_num, predict, W, H, w, h, layer, input_shape, cut_region, save_vis_path):
    '''提取clsmap(类别相关的置信度)
    '''
    color = list(color.values())
    score_map, cls_map = torch.max(F.sigmoid(predict), axis=2)
    # 初始化一个形状为 w, h, 3 的数组来存储 RGB 图像
    rgb_heat_map = np.zeros((w, h, 3))
    # 将分割结果映射到 RGB 值
    for label in range(cls_num):
        rgb_heat_map[cls_map == label] = color[label]
    rgb_heat_map *= score_map.unsqueeze(-1).numpy() 
    # rgb_heat_map *= obj_map.cpu().numpy() 
    rgb_heat_map = (rgb_heat_map*255).astype(np.uint8)
    # resize到网络接受的输入尺寸
    rgb_heat_map = cv2.resize(rgb_heat_map, (input_shape[0], input_shape[1]), interpolation=cv2.INTER_CUBIC)
    # 去除padding的灰边
    rgb_heat_map = rgb_heat_map[cut_region[0]:cut_region[1], cut_region[2]:cut_region[3]]
    # resize到原图尺寸
    rgb_heat_map = cv2.resize(rgb_heat_map, (W, H))
    # heatmap和原图像叠加显示
    heatmap_img = cv2.addWeighted(rgb_heat_map, 0.6, image, 0.4, 0)
    # 保存
    if save_vis_path!=None:
        save_dir, save_name = os.path.split(save_vis_path)
        save_name = f'heatmap{layer}_' + save_name
        cv2.imwrite(os.path.join(save_dir, save_name), heatmap_img)