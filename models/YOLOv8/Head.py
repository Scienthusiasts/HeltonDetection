import torch
import torch.nn as nn
from torch.nn import functional as F

from utils.FasterRCNNAnchorUtils import *
from utils.util import *
from loss.Loss import ClassifyLoss, BBoxLoss
from utils.YOLOv8AnchorUtils import *
from models.YOLOv8.YOLOBlocks import *




class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).
    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """本质就是对回归的值做加权求和, 只不过用卷积快一些"""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        # x = tensor([ 0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.])
        x = torch.arange(c1, dtype=torch.float)
        # 将x作为 1x1卷积核(shape=[1, 16, 1, 1])的参数:
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)






class YOLOv8Head(nn.Module):
    def __init__(self, cat_nums, base_channels, deep_mul):
        '''RPN网络
        Args:

        Returns:
            None
        '''
        super(YOLOv8Head, self).__init__()
        # 统一回归16个值，每个坐标由16个回归值共同决定
        self.reg_max = 16
        self.layer_num = 3
        self.shape = None
        self.num_classes = cat_nums
        # 类别+每个回归坐标由16个维度共同预测,一共是4个坐标, 所以是4*16
        self.num_reg = cat_nums + self.reg_max * 4  
        self.stride = torch.tensor([ 8., 16., 32.])
        roll_out_thr = 64
        self.use_dfl = self.reg_max > 1
        self.proj = torch.arange(self.reg_max, dtype=torch.float)
        '''标签分配策略'''
        self.assigner = TaskAlignedAssigner(topk=10, num_classes=self.num_classes, alpha=0.5, beta=6.0, roll_out_thr=roll_out_thr)
        '''损失函数'''
        self.bbox_loss = BboxLoss(self.reg_max - 1, use_dfl=self.use_dfl)
        self.bce = nn.BCEWithLogitsLoss(reduction='none')


        '''decoupled yolov8 head'''
        # channels
        channels = [base_channels * 4, base_channels * 8, int(base_channels * 16 * deep_mul)]
        # reg_c是回归头的中间层通道数
        reg_c = max((16, channels[0] // 4, self.reg_max * 4))
        # cls_c是分类头的中间层通道数
        cls_c = max(channels[0], self.num_classes)  
        self.cv2 = nn.ModuleList(nn.Sequential(Conv(c, reg_c, 3), Conv(reg_c, reg_c, 3), nn.Conv2d(reg_c, 4 * self.reg_max, 1)) for c in channels)
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(c, cls_c, 3), Conv(cls_c, cls_c, 3), nn.Conv2d(cls_c, self.num_classes, 1)) for c in channels)
        # DFL头, 将回归结果组合为回归值
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
        '''初始化权重(注意别把self.dfl也初始化了)'''
        init_weights(self.cv2, 'normal', 0, 0.01)
        init_weights(self.cv3, 'normal', 0, 0.01)




    def forward(self, x):
        '''前向传播
        '''
        # x = [P3, P4, P5]
        # shape = [bs, 128, 80, 80]
        shape = x[0].shape
        for i in range(self.layer_num):
            # reg_feat [bs, 16*4, w, h]
            reg_feat = self.cv2[i](x[i])
            # cls_feat [bs, cls_num, w, h]
            cls_feat = self.cv3[i](x[i])
            # 将分类和回归特征拼在一起 [4, 144, w, h]
            x[i] = torch.cat((reg_feat, cls_feat), 1)

        # 由于需要知道输入x的尺寸才能知道self.shape, self.anchors, self.strides的值, 所以这部分不在__init__里初始化
        if self.shape != shape:
            self.shape = shape
            # 所谓anchor其实就是每个cell的中心点, stride就是每个anchor距离其他anchor的距离(和下采样率有关)
            # self.anchors:[2, 8400] ; self.strides.shape[1, 8400]
            self.anchors, self.strides = make_anchors(x, self.stride, 0.5)

        # 将不同尺度的预测结果拼在一起: [bs, 16*4+cls_num, 80*80+40*40+20*20] = [bs, 144, 8400]
        reg_cat_cls_feat = torch.cat([xi.view(shape[0], self.num_reg, -1) for xi in x], 2)
        # 将回归和分类结果拆开: [bs, 144, 8400] -> [bs, 64, 8400], [bs, 80, 8400]
        box, cls = reg_cat_cls_feat.split((self.reg_max * 4, self.num_classes), 1)
        # 将回归结果组合为回归值: [bs, 64, 8400] -> [bs, 4, 8400]
        dbox = self.dfl(box)
        return dbox, cls, x, self.anchors.to(dbox.device), self.strides.to(dbox.device)
    





    def batchLoss(self, x, batch_bboxes):
        # x = (dbox, cls, x, anchors, stride)
        preds = self.forward(x)
        # 获得使用的device
        device  = preds[1].device
        # box, cls, dfl三部分的损失
        loss = torch.zeros(3, device=device)  
        # 获得特征，并进行划分
        feats   = preds[2] if isinstance(preds, tuple) else preds
        # 将不同尺度的预测结果拼在一起: [bs, 16*4+cls_num, 80*80+40*40+20*20] = [bs, 144, 8400]
        reg_cat_cls_feat = torch.cat([xi.view(feats[0].shape[0], self.num_reg, -1) for xi in feats], 2)
        # reg + cls
        pred_distri, pred_scores = reg_cat_cls_feat.split((self.reg_max * 4, self.num_classes), 1)
        # bs, num_classes + self.reg_max * 4 , 8400 =>  cls bs, num_classes, 8400; 
        # box bs, self.reg_max * 4, 8400
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        # 获得batch size与dtype
        dtype       = pred_scores.dtype
        batch_size  = pred_scores.shape[0]
        # 获得输入图片大小
        imgsz = torch.tensor(feats[0].shape[2:], device=device, dtype=dtype) * self.stride[0]  
        # 获得anchors点和步长对应的tensor
        anchor_points = preds[3].transpose(0, 1)
        stride_tensor = preds[4].transpose(0, 1)
        # 先进行初步的处理，对输入进来的gt进行padding，到最大数量，并把框的坐标进行缩放
        # bs, max_boxes_num, 5
        targets = preprocess(batch_bboxes.to(device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        # bs, max_boxes_num, 5 => bs, max_boxes_num, 1 ; bs, max_boxes_num, 4
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        # 求哪些框是有目标的，哪些是填充的
        # bs, max_boxes_num
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)
        # pboxes
        # 对预测结果进行解码，获得预测框
        # bs, 8400, 4
        pred_bboxes = bbox_decode(anchor_points, pred_distri, self.proj, self.use_dfl)  # xyxy, (b, h*w, 4)
        
        '''正负样本分配(动态)'''
        # target_bboxes     bs, 8400, 4
        # target_scores     bs, 8400, 80
        # fg_mask           bs, 8400
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt
        )
        target_bboxes /= stride_tensor
        target_scores_sum = max(target_scores.sum(), 1)
        # 计算分类的损失
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE
        # 计算bbox的损失
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                              target_scores_sum, fg_mask)
        loss[0] *= 7.5  # box gain
        loss[1] *= 0.5  # cls gain
        loss[2] *= 1.5  # dfl gain
        loss = dict(
            total_loss = loss.sum(),
            dfl_loss = loss[2],
            box_loss = loss[0],
            cls_loss = loss[1],
        )
        return loss







# for test only
if __name__ == '__main__':
    from torchsummary import summary
    # 基本配置
    phi = 's'
    depth_dict          = {'n' : 0.33, 's' : 0.33, 'm' : 0.67, 'l' : 1.00, 'x' : 1.00,}
    width_dict          = {'n' : 0.25, 's' : 0.50, 'm' : 0.75, 'l' : 1.00, 'x' : 1.25,}
    deep_width_dict     = {'n' : 1.00, 's' : 1.00, 'm' : 0.75, 'l' : 0.50, 'x' : 0.50,}
    dep_mul, wid_mul, deep_mul = depth_dict[phi], width_dict[phi], deep_width_dict[phi]
    base_channels       = int(wid_mul * 64)
    base_depth          = max(round(dep_mul * 3), 1)
    head_in_channel = {
        'n':[64,  128, 256],
        's':[128, 256, 512],
        'm':[192, 384, 576],
        'l':[256, 512, 512],
        'x':[324, 640, 640]
    }[phi]
    num_classes = 80 
    bs = 4


    head = YOLOv8Head(num_classes, base_channels, deep_mul)
    # 验证
    p3 = torch.rand((bs, head_in_channel[0], 80, 80))
    p4 = torch.rand((bs, head_in_channel[1], 40, 40))
    p5 = torch.rand((bs, head_in_channel[2], 20, 20))
    x = [p3, p4, p5]
    dbox, cls, x, anchors, strides = head(x)
    # print(dbox.shape)
    # print(cls.shape)
    # print('============================')
    # for i in x:
    #     print(i.shape)
    # print('============================')
    # print(anchors.shape)
    # print(strides.shape)


    # torch.Size([4, 4, 8400])
    # torch.Size([4, 80, 8400])
    # ============================
    # torch.Size([4, 144, 80, 80])
    # torch.Size([4, 144, 40, 40])
    # torch.Size([4, 144, 20, 20])
    # ============================
    # torch.Size([2, 8400])
    # torch.Size([1, 8400])