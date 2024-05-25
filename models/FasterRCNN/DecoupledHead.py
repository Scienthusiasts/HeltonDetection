import torch
import torch.nn as nn
from torchvision.ops import RoIPool, RoIAlign
import timm
from torchvision import models
import copy

from utils.FasterRCNNAnchorUtils import *
from utils.util import *
from models.FasterRCNN.YOLOBlocks import *
from loss.Loss import ClassifyLoss, BBoxLoss















class DecoupledHead(nn.Module):
    '''Backbone
    '''
    def __init__(self, catNums:int, roiSize:int):
        '''网络初始化

        Args:
            :param catNums:   数据集类别数
            :param modelType: 使用哪个模型(timm库里的模型)
            :param roiSize:   RoI Pooling后proposal的大小, 例:7
            :param loadckpt:  是否导入模型权重
            :param pretrain:  是否用预训练模型进行初始化(是则输入权重路径)

        Returns:
            None
        '''
        super(DecoupledHead, self).__init__()
        self.roi_size = 7
        '''损失函数'''
        self.regLoss = BBoxLoss(loss_type='SmoothL1Loss')
        # 省略掉那笑labels=-1的样本(不参与损失的计算)
        self.clsLoss = ClassifyLoss(loss_type='CrossEntropyLoss', cat_num=catNums)
        '''其他'''
        # 包括背景类别
        self.cat_nums = catNums + 1
        # spatial_scale是将输入坐标映射到box坐标的尺度因子. 默认: 1.0
        # self.roi = RoIPool(output_size=(roiSize, roiSize), spatial_scale=1.)
        self.roi = RoIAlign(output_size=(roiSize, roiSize), spatial_scale=1., sampling_ratio=0)
        # 回归分支
        self.reg_head = nn.Sequential(
            Conv(256, 256, 3, 1, 0),
            Conv(256, 256, 3, 1, 0),
            Conv(256, 256, 3, 1, 0),
        )
        self.reg = nn.Linear(256, self.cat_nums * 4)
        # 分类分支
        self.cls_head = nn.Sequential(
            Conv(256, 256, 3, 1, 0),
            Conv(256, 256, 3, 1, 0),
            Conv(256, 256, 3, 1, 0),
        )
        self.cls = nn.Linear(256, self.cat_nums)
        # 权重初始化
        init_weights(self.reg_head, 'normal', 0, 0.01)
        init_weights(self.cls_head, 'normal', 0, 0.01)
        init_weights(self.reg, 'normal', 0, 0.001)
        init_weights(self.cls, 'normal', 0, 0.01)









    def forward_head(self, bs, pooling_feat):
        '''前向传播(only head single feat)

        Args:
            :param bs:           batch size
            :param pooling_feat: 一个batch里所有图像的proposal集合

        Returns:
            :param RoIReg: 最终的proposal回归offset(未解码)
            :param RoICls: 最终的proposal分类结果
        '''
        reg_head_feat = self.reg_head(pooling_feat) # [bs*128, 256, 1, 1]
        cls_head_feat = self.cls_head(pooling_feat) # [bs*128, 256, 1, 1]
        reg_head_feat = reg_head_feat.view(reg_head_feat.size(0), -1) # [bs*128, 256]
        cls_head_feat = cls_head_feat.view(cls_head_feat.size(0), -1) # [bs*128, 256]

        # proposal的特征送去分类和回归
        # 注意，对于每个框，这里每个类别都会预测一组回归参数，而不是每个框只预测一组参数
        RoIReg = self.reg(reg_head_feat) # [bs*128, clsNum*4]
        RoICls = self.cls(cls_head_feat) # [bs*128, clsNum]
        RoIReg = RoIReg.view(bs, -1, RoIReg.size(1)) # [bs, 128, clsNum*4]
        RoICls = RoICls.view(bs, -1, RoICls.size(1)) # [bs, 128, clsNum]
        return RoIReg, RoICls










    def forward(self, lvl_x, RoIs, RoIs_idx, imgSize):
        '''前向传播(FPN)

        Args:
            :param lvl_x:      FPn输出的多尺度特征, list
            :param RoIs:       一个batch里所有图像的proposal集合
            :param RoIIndices: 一个batch里所有proposal的索引, 这样才知道哪些proposal属于哪张图像
            :param imgSize:    原始图像尺寸

        Returns:
            :param lvl_RoIReg: 最终的proposal回归offset(所有尺度拼在一起)
            :param lvl_RoICls: 最终的proposal分类结果(所有尺度拼在一起)
        '''
        bs = lvl_x[0].shape[0]
        RoIs = torch.flatten(RoIs, 0, 1) # [bs*128, 4]
        # proposals属于哪张图像的索引
        RoIs_idx = torch.flatten(RoIs_idx, 0, 1) # [bs*128]
        RoIs_with_Idx = torch.cat([RoIs_idx[:, None], RoIs], dim=1)

        '''计算每个proposal应该映射到fpn哪一层'''
        RoIs_wh = (RoIs[:, 2] - RoIs[:, 0]) * (RoIs[:, 3] - RoIs[:, 1])
        # 对应每个proposal应该映射到哪一层特征[bs*128]
        RoIs_level = self.assign_level(RoIs_wh)
        # 根据proposals大小对应提取FPN不同尺度下的RoI Feature:
        pool = torch.zeros((RoIs_idx.shape[0], lvl_x[0].shape[1], self.roi_size, self.roi_size), device=lvl_x[0].device)
        '''根据proposal分配到的层数对其映射到fpn对应特征层上并进行roi pooling'''
        for i in range(len(lvl_x)):
            # lvl_RoIs_idx取出proposal所在哪一个batch的的索引，方便之后拼回去
            lvl_RoIs_idx = torch.where(RoIs_level==i)[0]
            if lvl_RoIs_idx.shape[0] == 0: continue
            lvl_RoIs_with_Idx = RoIs_with_Idx[lvl_RoIs_idx]
            # 将坐标映射到特征图的尺度(顺序:8(p3)->16(p4)->32(p5)):
            lvl_RoIs_with_Idx[:, [1,3]] = lvl_RoIs_with_Idx[:, [1,3]] * lvl_x[i].shape[3] / imgSize[1]
            lvl_RoIs_with_Idx[:, [2,4]] = lvl_RoIs_with_Idx[:, [2,4]] * lvl_x[i].shape[2] / imgSize[0]
            # ROI Pooling  [bs*128, 256, roiSize, roiSize]
            lvl_pool = self.roi(lvl_x[i], lvl_RoIs_with_Idx)
            pool[lvl_RoIs_idx] = lvl_pool
        '''head前向传播(这里统一对所有层进行前向比在for循环里分别对每一层前向要快)'''
        # torch.Size([bs, 128, (cls_num+1)*4]) torch.Size([bs, 128, cls_num+1])
        RoIReg, RoICls = self.forward_head(bs, pool)

        return RoIReg, RoICls






    def assign_level(self, wh, min_layer_idx=0, max_layer_idx=2):
        '''RPN微调后的RoI重新根据其尺度重新分配合适的特征层'''
        # 0:8 1:16 2:32
        rois_level = torch.round(1. + torch.log2(torch.sqrt(wh + 1e-8)/224.0))
        # 限制不超出范围
        rois_level = torch.clamp(rois_level, min=min_layer_idx, max=max_layer_idx)
        return rois_level









    def batchLoss(self, img_size, bs, base_feature, batch_bboxes, batch_labels, rois, origin_rois, roi_indices):

        roi_loc_loss_all, roi_cls_loss_all  = 0, 0
        sample_rois, sample_roi_gts, sample_indexes, gt_roi_locs, gt_roi_labels = [], [], [], [], []
        # 遍历batch里每张图像:
        for bbox, label, roi, roi_indice in zip(batch_bboxes, batch_labels, rois, roi_indices):

            '''proposal正负样本分配策略(筛选proposal, 并为每一个保留的proposal都匹配一个类别标签和一个offset GT)'''
            #   sample_roi      [n_sample, ]
            #   gt_roi_loc      [n_sample, 4]
            #   gt_roi_label    [n_sample, ]
            sample_roi, sample_roi_gt, gt_roi_loc, gt_roi_label = ROIHeadMaxIoUAssignerNP(roi, bbox, label)
            sample_rois.append(torch.Tensor(sample_roi).type_as(roi))
            sample_roi_gts.append(torch.Tensor(sample_roi_gt).type_as(roi))
            sample_indexes.append(torch.ones(len(sample_roi)).type_as(roi) * roi_indice[0]) # roi_indice[0]
            gt_roi_locs.append(torch.Tensor(gt_roi_loc).type_as(roi))
            gt_roi_labels.append(torch.Tensor(gt_roi_label).type_as(roi).long())

        sample_rois = torch.stack(sample_rois, dim=0)   
        sample_roi_gts = torch.stack(sample_roi_gts, dim=0)   
        sample_indexes = torch.stack(sample_indexes, dim=0)
        '''对batch里的每张图像, 计算其Head损失'''
        # 利用FasterRCNNHead获得网络预测的回归与分类结果
        # roi_cls_locs: 最终的proposal回归offset    例:shape = [8, 128, 4*21]
        # roi_scores: 最终的proposal分类结果        例:shape = [8, 128, 21]
        roi_cls_locs, roi_scores = self.forward(base_feature, sample_rois, sample_indexes, img_size)
        # print(roi_cls_locs.shape, roi_scores.shape)
        # Head阶段计算损失
        for roi_cls_loc, sample_roi, sample_roi_gt, roi_score, gt_roi_loc, gt_roi_label in zip(roi_cls_locs, sample_rois, sample_roi_gts, roi_scores, gt_roi_locs, gt_roi_labels):
            # 根据建议框的类别GT，取出对应类别下的回归预测结果
            n_sample = roi_cls_locs.size()[1] # 128
            roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)
            roi_loc = roi_cls_loc[torch.arange(0, n_sample), gt_roi_label]
            '''head回归损失'''
            roi_loc_loss = self.regLoss(roi_loc[gt_roi_label.data > 0], gt_roi_loc[gt_roi_label.data > 0])
            '''下面这部分用来计算IoU loss'''
            # reg_std = torch.tensor([0.1, 0.1, 0.2, 0.2]).to(roi_loc.device)
            # sample_pred_bbox = reg2Bbox(sample_roi, roi_loc * reg_std)
            # pred_pos_box, pos_box_gt = sample_pred_bbox[gt_roi_label.data > 0], sample_roi_gt[gt_roi_label.data > 0]
            # roi_loc_loss = self.regLoss(pred_pos_box, pos_box_gt)
            '''head分类损失'''
            # (gt_roi_label没有-1, torch.Size([128]))(且正负样本不会刚好一半一半)
            roi_cls_loss = self.clsLoss(roi_score, gt_roi_label, mode='head')
            # 记录损失
            roi_loc_loss_all += roi_loc_loss
            roi_cls_loss_all += roi_cls_loss


        '''统一损失的格式:字典形式: {损失1, 损失2,..., 总损失}'''
        loss = dict(
            roi_reg_loss = roi_loc_loss_all / bs,
            roi_cls_loss = roi_cls_loss_all / bs,
        )
        return loss














# for test only:
if __name__ == '__main__':
    from torchsummary import summary
    from models.FasterRCNN.RPN import RPN
    BS = 8
    imgSize = [832, 832]


    rpn = RPN(inChannels=256, midChannels=256, ratios = [0.5, 1, 2], scales = [8, 16, 32], featStride = [16])
    head = DecoupledHead(catNums=20, roiSize=7)
    torch.save(head.state_dict(), "head.pt")
    print(head)
    # 验证 1
    # summary(net, input_size=[(3, 224, 224)])  
    # 验证 2
    # print(head)
    backbone_feat = [torch.rand((BS, 256, 104, 104)), torch.rand((BS, 256, 52, 52)), torch.rand((BS, 256, 26, 26))]
    RPNReg, RPNCls, RoIs, _, RoIIndices, anchor = rpn(backbone_feat, imgSize=imgSize)
    RoIReg, RoICls = head(backbone_feat[1], RoIs, RoIIndices, imgSize=imgSize)
    print(RoIReg.shape) # torch.Size([4, 600, 320])
    print(RoICls.shape) # torch.Size([4, 600, 80])