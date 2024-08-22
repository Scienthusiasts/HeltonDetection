import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.util import *
from utils.FCOSUtils import *





class ScaleExp(nn.Module):
    '''指数放缩可学习模块,
       通过使用指数变换，可以确保预测结果总是非负数, 同时, 学习一个放缩系数 self.scale 使得网络能够动态地调整回归值的范围.
    '''
    def __init__(self, init_value=1.0):
        super(ScaleExp,self).__init__()
        # 可学习缩放参数
        self.scale = nn.Parameter(torch.tensor([init_value], dtype=torch.float32))

    def forward(self,x):
        # 对预测的特征图的数值再进行一个指数放缩, 并且放缩的参数是可学习的
        return torch.exp(x*self.scale)
    



class Head(nn.Module):
    '''FCOS的预测头模块(共享)
    '''
    def __init__(self, num_classes, in_channel=256):
        super(Head,self).__init__()
        self.num_classes=num_classes
        cls_branch=[]
        reg_branch=[]

        # 预测头之前的特征提取部分
        for _ in range(4):
            # 分类分支特征提取(cls和centerness)
            cls_branch.append(nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1, bias=True))
            cls_branch.append(nn.GroupNorm(32, in_channel)),
            cls_branch.append(nn.ReLU(True))
            # 回归分支特征提取(reg)
            reg_branch.append(nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1, bias=True))
            reg_branch.append(nn.GroupNorm(32, in_channel)),
            reg_branch.append(nn.ReLU(True))

        # 预测头之前的共享特征提取
        self.cls_conv=nn.Sequential(*cls_branch)
        self.reg_conv=nn.Sequential(*reg_branch)
        '''分类回归头解耦'''
        # 分类头
        self.cls_head = nn.Conv2d(in_channel, num_classes, kernel_size=3, padding=1)
        # centerness头
        self.cnt_head = nn.Conv2d(in_channel, 1, kernel_size=3, padding=1)
        # 回归头
        self.reg_head = nn.Conv2d(in_channel, 4, kernel_size=3, padding=1)
        # 回归头上的可学习放缩系数
        self.scale_exp = nn.ModuleList([ScaleExp(1) for _ in range(5)])

        # 权重初始化
        init_weights(self.cls_conv, 'normal', 0, 0.01)
        init_weights(self.reg_conv, 'normal', 0, 0.01)
        init_weights(self.cls_head, 'normal', 0, 0.01)
        init_weights(self.cnt_head, 'normal', 0, 0.01)
        init_weights(self.reg_head, 'normal', 0, 0.01)
        # 对分类头的偏置专门的初始化方式(目的是, 一开始网络的分类会倾向于背景, 可以从一个合理的状态开始训练):
        prior = 0.01
        nn.init.constant_(self.cls_head.bias, -math.log((1 - prior) / prior))
    

    
    def forward(self, x):
        cls_logits = []
        cnt_logits = []
        reg_preds  = []
        # 遍历不同尺度的特征层(p3-p7),得到预测结果
        for lvl, lvl_x in enumerate(x):
            cls_conv_out=self.cls_conv(lvl_x)
            reg_conv_out=self.reg_conv(lvl_x)

            cls_logits.append(self.cls_head(cls_conv_out))
            cnt_logits.append(self.cnt_head(cls_conv_out))
            reg_preds.append(self.scale_exp[lvl](self.reg_head(reg_conv_out)))
        return cls_logits, cnt_logits, reg_preds





    def batchLoss(self, fpn_feat, batch_bboxes, batch_labels):
        # head部分前向
        cls_logits, cnt_logits, reg_preds = self.forward(fpn_feat)
        '''FCOS的正负样本分配'''
        cls_targets, cnt_targets, reg_targets = FCOSAssigner(cls_logits, batch_bboxes, batch_labels)
        # 根据centerness获得正样本
        mask_pos = (cnt_targets > -1).squeeze(dim=-1)
        cls_loss = self.compute_cls_loss(cls_logits, cls_targets, mask_pos)
        cnt_loss = self.compute_cnt_loss(cnt_logits, cnt_targets, mask_pos)
        reg_loss = self.compute_reg_loss(reg_preds, reg_targets, mask_pos)
        loss = dict(
            total_loss = cls_loss + cnt_loss + reg_loss,
            cls_loss = cls_loss,
            cnt_loss = cnt_loss,
            reg_loss = reg_loss
        )
        return loss


        
    

    def compute_cls_loss(self, preds, targets, mask, gamma=2.0, alpha=0.25):
        #--------------------#
        #   计算batch_size
        #   计算种类数量
        #--------------------#
        batch_size      = targets.shape[0]
        num_classes     = preds[0].shape[1]
        
        mask            = mask.unsqueeze(dim = -1)
        #--------------------#
        #   计算正样本数量
        #--------------------#
        num_pos         = torch.sum(mask, dim = [1, 2]).clamp_(min = 1).float()
        preds_reshape   = []
        for pred in preds:
            #--------------------#
            #   对预测结果reshape
            #--------------------#
            pred        = torch.reshape(pred.permute(0, 2, 3, 1), [batch_size, -1, num_classes])
            preds_reshape.append(pred)
        preds           = torch.cat(preds_reshape, dim = 1)
        assert preds.shape[:2]==targets.shape[:2]
        
        #--------------------#
        #   对计算损失
        #--------------------#
        loss = 0
        for batch_index in range(batch_size):
            pred_pos    = torch.sigmoid(preds[batch_index])
            target_pos  = targets[batch_index]
            #--------------------#
            #   生成one_hot标签
            #--------------------#
            target_pos  = (torch.arange(0, num_classes, device=target_pos.device)[None,:] == target_pos).float()
            
            #--------------------#
            #   计算focal_loss
            #--------------------#
            pt          = pred_pos * target_pos + (1.0 - pred_pos) * (1.0 - target_pos)
            w           = alpha * target_pos + (1.0 - alpha) * (1.0 - target_pos)
            batch_loss  = -w * torch.pow((1.0 - pt), gamma) * pt.log()
            batch_loss  = batch_loss.sum()
            loss += batch_loss
            
        return loss / torch.sum(num_pos)

    def compute_cnt_loss(self, preds, targets, mask):
        #------------------------#
        #   计算batch_size
        #   计算center长度（1）
        #------------------------#
        batch_size  = targets.shape[0]
        c           = targets.shape[-1]
        
        mask            = mask.unsqueeze(dim = -1)
        #--------------------#
        #   计算正样本数量
        #--------------------#
        num_pos         = torch.sum(mask, dim = [1, 2]).clamp_(min = 1).float()
        
        preds_reshape   = []
        for pred in preds:
            #--------------------#
            #   对预测结果reshape
            #--------------------#
            pred        = torch.reshape(pred.permute(0, 2, 3, 1), [batch_size, -1, c])
            preds_reshape.append(pred)
            
        preds           = torch.cat(preds_reshape, dim = 1)
        assert preds.shape==targets.shape
        
        #--------------------#
        #   对计算损失
        #--------------------#
        loss = 0
        for batch_index in range(batch_size):
            pred_pos    = preds[batch_index][mask[batch_index]]
            target_pos  = targets[batch_index][mask[batch_index]]
            batch_loss  = nn.functional.binary_cross_entropy_with_logits(input=pred_pos,target=target_pos,reduction='sum').view(1)
            loss += batch_loss
            
        return torch.sum(loss, dim=0) / torch.sum(num_pos)

    def giou_loss(self, preds, targets):
        #------------------------#
        #   左上角和右下角
        #------------------------#
        lt_min  = torch.min(preds[:, :2], targets[:, :2])
        rb_min  = torch.min(preds[:, 2:], targets[:, 2:])
        #------------------------#
        #   重合面积计算
        #------------------------#
        wh_min  = (rb_min + lt_min).clamp(min=0)
        overlap = wh_min[:, 0] * wh_min[:, 1]#[n]
        
        #------------------------------#
        #   预测框面积和实际框面积计算
        #------------------------------#
        area1   = (preds[:, 2] + preds[:, 0]) * (preds[:, 3] + preds[:, 1])
        area2   = (targets[:, 2] + targets[:, 0]) * (targets[:, 3] + targets[:, 1])
        
        #------------------------------#
        #   计算交并比
        #------------------------------#
        union   = (area1 + area2 - overlap)
        iou     = overlap / union

        #------------------------------#
        #   计算外包围框
        #------------------------------#
        lt_max  = torch.max(preds[:, :2],targets[:, :2])
        rb_max  = torch.max(preds[:, 2:],targets[:, 2:])
        wh_max  = (rb_max + lt_max).clamp(0)
        G_area  = wh_max[:, 0] * wh_max[:, 1]

        #------------------------------#
        #   计算GIOU
        #------------------------------#
        giou    = iou - (G_area - union) / G_area.clamp(1e-10)
        loss    = 1. - giou
        return loss.sum()
        
    def compute_reg_loss(self, preds, targets, mask):
        #------------------------#
        #   计算batch_size
        #   计算回归参数长度（4）
        #------------------------#
        batch_size  = targets.shape[0]
        c           = targets.shape[-1]
        
        num_pos     = torch.sum(mask, dim=1).clamp_(min=1).float()#[batch_size,]
        preds_reshape=[]
        for pred in preds:
            #--------------------#
            #   对预测结果reshape
            #--------------------#
            pred        = torch.reshape(pred.permute(0, 2, 3, 1), [batch_size, -1, c])
            preds_reshape.append(pred)
            
        preds           = torch.cat(preds_reshape, dim = 1)
        assert preds.shape==targets.shape
        
        loss = 0
        for batch_index in range(batch_size):
            pred_pos    = preds[batch_index][mask[batch_index]]
            target_pos  = targets[batch_index][mask[batch_index]]
            batch_loss  = self.giou_loss(pred_pos, target_pos).view(1)
            loss += batch_loss
        return torch.sum(loss, dim=0) / torch.sum(num_pos)






















# for test only
if __name__ == '__main__':
    num_cls = 15
    fpn_out_channel = 256
    bs = 4
    size = [80, 40, 20, 10, 5]
    # 模拟FPN输出:
    x = [torch.rand((bs, fpn_out_channel, lvl_size, lvl_size)) for lvl_size in size]
    head = Head(num_cls, fpn_out_channel)
    cls_logits, cnt_logits, reg_preds = head(x)

    for cls, cnt, reg in zip(cls_logits, cnt_logits, reg_preds):
        print(cls.shape, cnt.shape, reg.shape)

    # torch.Size([4, 15, 80, 80]) torch.Size([4, 1, 80, 80]) torch.Size([4, 4, 80, 80])
    # torch.Size([4, 15, 40, 40]) torch.Size([4, 1, 40, 40]) torch.Size([4, 4, 40, 40])
    # torch.Size([4, 15, 20, 20]) torch.Size([4, 1, 20, 20]) torch.Size([4, 4, 20, 20])
    # torch.Size([4, 15, 10, 10]) torch.Size([4, 1, 10, 10]) torch.Size([4, 4, 10, 10])
    # torch.Size([4, 15, 5, 5]) torch.Size([4, 1, 5, 5]) torch.Size([4, 4, 5, 5])