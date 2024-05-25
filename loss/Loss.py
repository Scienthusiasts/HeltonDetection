import torch
import torch.nn as nn
import torch.nn.functional as F
import math








class ClassifyLoss(nn.Module):
    def __init__(self, loss_type:str, cat_num:int, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_type = loss_type
        self.cat_num = cat_num
        if self.loss_type == 'CrossEntropyLoss':
            self.loss = nn.CrossEntropyLoss(reduction='mean')
        if self.loss_type == 'FocalLoss':
            self.gamma = gamma
            self.alpha = alpha
            self.loss = nn.BCEWithLogitsLoss(reduction='mean')



    def forward(self, pred, true, mode=None):
        if self.loss_type == 'CrossEntropyLoss':
            loss = self.loss(pred, true)
        if self.loss_type == 'FocalLoss':
            nums = {'rpn':2, 'head':self.cat_num + 1}[mode]
            loss = self.focalLossForward(pred, true, nums)   
            
        return loss



    def focalLossForward(self, pred, true, num_cls):
        '''FocalLoss from: https://github.com/ultralytics/yolov5
        '''
        # 将标签转化为one-hot形式:
        true = F.one_hot(true, num_classes=num_cls).float()
        loss = self.loss(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        return loss.mean() * 100















class BBoxLoss(nn.Module):
    def __init__(self, loss_type:str):
        '''
        loss_type:SmoothL1Loss, IoULoss, GIoULoss, DIoULiss, CIoULoss
        '''
        super().__init__()
        self.loss_type = loss_type
        if self.loss_type == 'SmoothL1Loss':
            self.smooth_l1_loss = nn.SmoothL1Loss(reduction='mean', beta=1.)
    


    def forward(self, pred, true):
        if self.loss_type == 'SmoothL1Loss':
            # 这里pred, true均是未解码的回归向量
            loss = self.smooth_l1_loss(pred, true)
        if self.loss_type in ['IoULoss', 'GIoULoss', 'DIoULiss', 'CIoULoss']:
            # 这里pred, true均是已解码的,映射回原图尺寸的结果(xyxy)
            bbox_IoU = self.bboxIoU(pred, true)
            loss = (1.0 - bbox_IoU).sum() / true.shape[0]
        
        return loss



    def bboxIoU(self, box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
        """
        Calculate Intersection over Union (IoU) of box1(1, 4) to box2(n, 4).

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