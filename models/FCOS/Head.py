import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.util import *
from utils.FCOSUtils import *
from loss.YOLOLoss import *





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
        '''定义网络结构'''
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
        # 分类回归头解耦
        # 分类头
        self.cls_head = nn.Conv2d(in_channel, num_classes, kernel_size=3, padding=1)
        # centerness头
        self.cnt_head = nn.Conv2d(in_channel, 1, kernel_size=3, padding=1)
        # 回归头
        self.reg_head = nn.Conv2d(in_channel, 4, kernel_size=3, padding=1)
        # 回归头上的可学习放缩系数
        self.scale_exp = nn.ModuleList([ScaleExp(1) for _ in range(5)])
        '''定义损失函数'''
        self.cntLoss = nn.BCEWithLogitsLoss()
        self.clsLoss = Loss(loss_type='FocalLoss', gamma=2.0, alpha=0.25, reduction='none')
        self.boxLoss = Loss(loss_type='GIoULoss', reduction='mean')

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
            # 这里回归出来就是原图尺寸下的偏移量
            reg_preds.append(self.scale_exp[lvl](self.reg_head(reg_conv_out)))
        return cls_logits, cnt_logits, reg_preds





    def batchLoss(self, fpn_feat, batch_bboxes, batch_labels, input_shape):
        # head部分前向
        # [[bs, cls_num, w, h],...,[...]] [[bs, 1, w, h],...,[...]] [[bs, 4, w, h],...,[...]]
        cls_logits, cnt_logits, reg_preds = self.forward(fpn_feat)
        '''FCOS的正负样本分配'''
        # 对应位置标记为-1的是负样本 [bs * total_anchor_num, 1] [bs * total_anchor_num, 1] [bs * total_anchor_num, 4]
        cls_targets, cnt_targets, reg_targets = FCOSAssigner(batch_bboxes, batch_labels, input_shape)
        # 获得正样本(bool) [bs, total_anchor_num]
        pos_mask = (cnt_targets > -1).reshape(-1)
        '''计算损失'''
        # 调整预测结果的形状:
        # [[bs, cls_num, 80, 80],...,[[bs, cls_num, 5, 5]]] -> [bs * total_anchor_num, cls_num]
        cls_preds = reshape_cat_out(cls_logits).reshape(-1, self.num_classes)
        # [[bs, 1, 80, 80],...,[[bs, 1, 5, 5]]] -> [bs * total_anchor_num, 1]
        cnt_preds = reshape_cat_out(cnt_logits).reshape(-1, 1)
        # [[bs, 4, 80, 80],...,[[bs, 4, 5, 5]]] -> [bs * total_anchor_num, 4]
        reg_preds = reshape_cat_out(reg_preds).reshape(-1, 4)
        # 计算损失:
        '''分类损失(所有样本均参与计算)'''
        # 计算batch里每张图片的正样本数量 [bs,]
        num_pos = torch.sum(pos_mask).clamp_(min=1).float()
        # 生成one_hot标签
        cls_targets  = (torch.arange(0, self.num_classes, device=cls_targets.device)[None,:] == cls_targets).float()
        cls_loss = self.clsLoss(cls_preds, cls_targets).sum() / torch.sum(num_pos)
        '''centerness损失(正样本才计算)'''
        # 计算BCE损失
        cnt_loss = self.cntLoss(cnt_preds[pos_mask], cnt_targets[pos_mask])
        '''回归损失(正样本才计算)'''
        # 计算GIoU loss
        giou = computeGIoU(reg_preds[pos_mask], reg_targets[pos_mask])
        reg_loss = (1. - giou).mean()
        '''loss以字典形式回传'''
        loss = dict(
            total_loss = cls_loss + cnt_loss + reg_loss,
            cls_loss = cls_loss,
            cnt_loss = cnt_loss,
            reg_loss = reg_loss
        )
        return loss  





















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