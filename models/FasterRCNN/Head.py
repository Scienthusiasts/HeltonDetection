import torch
import torch.nn as nn
from torchvision.ops import RoIPool, RoIAlign
import timm
from torchvision import models
import copy
from utils.FasterRCNNAnchorUtils import *
from utils.util import *
from loss.Loss import ClassifyLoss, BBoxLoss






class CustomBatchNorm2d(nn.Module):
    '''当bs=1时, 跳过BN
    '''
    def __init__(self, channels):
        super(CustomBatchNorm2d, self).__init__()
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        if x.size(0) == 1:
            return x
        else:
            return self.bn(x)



class ShareHead(nn.Module):
    '''Head部分回归和分类之前的共享特征提取层(目的是将ROI的7x7压缩到1x1)
    '''
    def __init__(self, channel):
        super(ShareHead, self).__init__()
        # self.squeezelayer = nn.Conv2d(channel*4, channel, 1, 1)
        self.convBlocks =  nn.Sequential(
            self._make_conv_block(channel, channel), 
            self._make_conv_block(channel, channel), 
            self._make_conv_block(channel, channel),
        )
        # 权重初始化
        # init_weights(self.squeezelayer, 'he')
        # init_weights(self.convBlocks, 'he')
        init_weights(self.convBlocks, 'normal', 0, 0.01)


    def _make_conv_block(self, in_channels, out_channels):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, 3, 1, bias=False))
        layers.append(CustomBatchNorm2d(out_channels))
        layers.append(nn.ReLU())
        in_channels = out_channels
        return nn.Sequential(*layers)


    def forward(self, x):
        # x = self.squeezelayer(x)
        x = self.convBlocks(x)
        return x    










class Head(nn.Module):
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
        super(Head, self).__init__()
        '''损失函数'''
        self.regLoss = BBoxLoss(loss_type='SmoothL1Loss')
        # 省略掉那笑labels=-1的样本(不参与损失的计算)
        self.clsLoss = ClassifyLoss(loss_type='CrossEntropyLoss', cat_num=catNums)
        '''其他'''
        self.cat_nums = catNums + 1
        # spatial_scale是将输入坐标映射到box坐标的尺度因子. 默认: 1.0
        # self.roi = RoIPool(output_size=(roiSize, roiSize), spatial_scale=1.)
        self.roi = RoIAlign(output_size=(roiSize, roiSize), spatial_scale=1., sampling_ratio=0)
        # 共享卷积层
        self.share_head = ShareHead(channel=256)
        # 回归分支
        self.reg = nn.Linear(256, self.cat_nums * 4)
        # 分类分支
        self.cls = nn.Linear(256, self.cat_nums)
        # 权重初始化
        # init_weights(self.reg, 'he')
        # init_weights(self.cls, 'he')
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
        headFeat = self.share_head(pooling_feat)       # [bs*600, 256, 1, 1]
        headFeat = headFeat.view(headFeat.size(0), -1) # [bs*600, 256]
        # proposal的特征送去分类和回归
        # 注意，对于每个框，这里每个类别都会预测一组回归参数，而不是每个框只预测一组参数
        RoIReg = self.reg(headFeat) # [bs*600, clsNum*4]
        RoICls = self.cls(headFeat) # [bs*600, clsNum]
        RoIReg = RoIReg.view(bs, -1, RoIReg.size(1)) # [bs, 600, clsNum*4]
        RoICls = RoICls.view(bs, -1, RoICls.size(1)) # [bs, 600, clsNum]
        return RoIReg, RoICls



    def forward(self, x, RoIs, RoIIndices, imgSize):
        '''前向传播(FPN-P2)

        Args:
            :param x:          backbone输出的特征, resnet50:[bs, 1024, 38, 38]
            :param RoIs:       一个batch里所有图像的proposal集合
            :param RoIIndices: 一个batch里所有proposal的索引, 这样才知道哪些proposal属于哪张图像
            :param imgSize:    原始图像尺寸

        Returns:
            :param RoIReg: 最终的proposal回归offset
            :param RoICls: 最终的proposal分类结果
        '''
        # 获取batchsize大小
        bs = x.shape[0]
        RoIs = torch.flatten(RoIs, 0, 1) # [bs*600, 4]
        RoIIndices = torch.flatten(RoIIndices, 0, 1) # [bs*600]
        # RoIs里的box坐标对应原始图像尺寸的坐标, featRoIs调整为特征图尺寸的坐标
        # 如果没有这步调整, 应该设置RoIPool里的spatial_scale=Backbone的下采样率，否则为1. (x.size / imgSize 相当于自动计算了下采样率)
        featRoIs = torch.zeros_like(RoIs)
        featRoIs[:, [0,2]] = RoIs[:, [0,2]] * x.size()[3] / imgSize[1]
        featRoIs[:, [1,3]] = RoIs[:, [1,3]] * x.size()[2] / imgSize[0] 
        # RoIsAndIdx将RoIs与其对应在一个batch里得到哪张图像上的索引拼成一块(为了满足RoIPool的输入格式)
        # RoIsAndIdx第一列是索引，之后四列是bbox坐标
        RoIsAndIdx = torch.cat([RoIIndices[:, None], featRoIs], dim=1) # [bs*600, 5]
        # 利用建议框对backbone特征层进行截取(RoIpooling之后有1024个通道，特征图大小为[roiSize, roiSize])
        pool = self.roi(x, RoIsAndIdx) # [bs*600, 256, roiSize, roiSize]
        # 利用head网络进一步对proposal的特征进行特征提取
        # 相当于把[1024, roiSize, roiSize]的特征图压缩成[2048]
        '''head前向传播'''
        RoIReg, RoICls = self.forward_head(bs, pool)
        
        return RoIReg, RoICls
    





    def forward_multi(self, lvl_x, RoIs, RoIs_idx, imgSize):
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
        RoIs_idx = torch.flatten(RoIs_idx, 0, 1) # [bs*128]
        RoIs_wh = (RoIs[:, 2] - RoIs[:, 0]) * (RoIs[:, 3] - RoIs[:, 1])
        # 对应每个proposal应该映射到那一层特征[bs*128]
        RoIs_level = self.assign_level(RoIs_wh)
        RoIs_with_Idx = torch.cat([RoIs_idx[:, None], RoIs], dim=1)
        # 根据proposals大小对应提取FPN不同尺度下的RoI Feature:
        lvl_RoIReg = torch.zeros((RoIs_idx.shape[0], self.cat_nums*4), device=lvl_x[0].device)
        lvl_RoICls = torch.zeros((RoIs_idx.shape[0], self.cat_nums), device=lvl_x[0].device)
        for i in range(4):
            lvl_RoIs_idx = torch.where(RoIs_level==i)[0]
            if lvl_RoIs_idx.shape[0] == 0: continue
            lvl_RoIs_with_Idx = RoIs_with_Idx[lvl_RoIs_idx]
            # 将坐标映射到特征图的尺度(顺序:200->100->50->25):
            lvl_RoIs_with_Idx[:, [1,3]] = lvl_RoIs_with_Idx[:, [1,3]] * lvl_x[i].shape[3] / imgSize[1]
            lvl_RoIs_with_Idx[:, [2,4]] = lvl_RoIs_with_Idx[:, [2,4]] * lvl_x[i].shape[2] / imgSize[0]
            # ROI Pooling
            lvl_pool = self.roi(lvl_x[i], lvl_RoIs_with_Idx)
            # Head预测头
            headFeat = self.share_head(lvl_pool) # [bs*128, 2048]
            headFeat = headFeat.view(headFeat.size(0), -1) 
            RoIReg = self.reg(headFeat)
            RoICls = self.cls(headFeat)
            lvl_RoIReg[lvl_RoIs_idx] = RoIReg
            lvl_RoICls[lvl_RoIs_idx] = RoICls

        lvl_RoIReg = lvl_RoIReg.view(bs, -1, lvl_RoIReg.size(1)) # [bs, 128, clsNum*4]
        lvl_RoICls = lvl_RoICls.view(bs, -1, lvl_RoICls.size(1))   # [bs, 128, clsNum]

        return lvl_RoIReg, lvl_RoICls






    def forward_multi_cat_single(self, lvl_x, RoIs, RoIs_idx, imgSize):
        '''前向传播(FPN-将不同尺度的再concat回去)

        Args:
            :param x:          backbone输出的特征, resnet50:[bs, 1024, 38, 38]
            :param RoIs:       一个batch里所有图像的proposal集合
            :param RoIIndices: 一个batch里所有proposal的索引, 这样才知道哪些proposal属于哪张图像
            :param imgSize:    原始图像尺寸

        Returns:
            :param RoIReg: 最终的proposal回归offset
            :param RoICls: 最终的proposal分类结果
        '''
        bs = lvl_x[0].shape[0]
        RoIs = torch.flatten(RoIs, 0, 1) # [bs*128, 4]
        RoIs_idx = torch.flatten(RoIs_idx, 0, 1) # [bs*128]
        # 将bs索引与RoIs拼接在一起，符合roipooling输入格式
        RoIs_with_Idx = torch.cat([RoIs_idx[:, None], RoIs], dim=1)
        # 根据proposals大小对应提取FPN不同尺度下的RoI Feature: 
        lvl_pool = torch.zeros((RoIs_idx.shape[0], lvl_x[0].shape[1]*4, 7, 7), device=lvl_x[0].device)
        # 先在每个尺度上进行roipooling, 最后将pooling的feature按通道维数拼接起来(256*4=1024)
        for i in range(4):
            # 将坐标映射到特征图的尺度:
            lvl_RoIs_with_Idx = copy.deepcopy(RoIs_with_Idx)
            lvl_RoIs_with_Idx[:, [1,3]] = lvl_RoIs_with_Idx[:, [1,3]] * lvl_x[i].shape[3] / imgSize[1]
            lvl_RoIs_with_Idx[:, [2,4]] = lvl_RoIs_with_Idx[:, [2,4]] * lvl_x[i].shape[2] / imgSize[0]
            # ROI Pooling
            lvl_pool[:, i*256:(i+1)*256, :, :] = self.roi(lvl_x[i], lvl_RoIs_with_Idx)  # [bs*128, 1024, 7, 7]

        '''head前向传播'''
        RoIReg, RoICls = self.forward_head(bs, lvl_pool)
        return RoIReg, RoICls







    def assign_level(self, wh, min_layer_idx=0, max_layer_idx=3):
        rois_level = torch.round(4. + torch.log2(torch.sqrt(wh + 1e-8)/224.0))
        # Head只用P2, P3, P4, P5的特征, 不用P6的特征
        rois_level = torch.clamp(rois_level, min=min_layer_idx+2, max=max_layer_idx+2)
        # 返回的是索引，所以-2
        return rois_level - 2 








    def batchLoss(self, img_size, bs, base_feature, batch_bboxes, batch_labels, rois, origin_rois, roi_indices):

        roi_loc_loss_all, roi_cls_loss_all  = 0, 0
        sample_rois, sample_roi_gts, sample_indexes, gt_roi_locs, gt_roi_labels = [], [], [], [], []
        for bbox, label, roi, origin_roi, roi_indice in zip(batch_bboxes, batch_labels, rois, origin_rois, roi_indices):

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
        # Head阶段计算损失
        for roi_cls_loc, sample_roi, sample_roi_gt, roi_score, gt_roi_loc, gt_roi_label in zip(roi_cls_locs, sample_rois, sample_roi_gts, roi_scores, gt_roi_locs, gt_roi_labels):
            # 根据建议框的类别GT，取出对应类别下的回归预测结果
            n_sample = roi_cls_locs.size()[1] # 128
            roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)
            roi_loc = roi_cls_loc[torch.arange(0, n_sample), gt_roi_label]
            '''回归损失(第二阶段)'''
            roi_loc_loss = self.regLoss(roi_loc[gt_roi_label.data > 0], gt_roi_loc[gt_roi_label.data > 0])
            '''下面这部分用来计算IoU loss'''
            # reg_std = torch.tensor([0.1, 0.1, 0.2, 0.2]).to(roi_loc.device)
            # sample_pred_bbox = reg2Bbox(sample_roi, roi_loc * reg_std)
            # pred_pos_box, pos_box_gt = sample_pred_bbox[gt_roi_label.data > 0], sample_roi_gt[gt_roi_label.data > 0]
            # roi_loc_loss = self.regLoss(pred_pos_box, pos_box_gt)
            '''分类损失(第二阶段)'''
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
    imgSize = [800, 800]


    rpn = RPN(inChannels=256, midChannels=256, featStride=16,)
    head = Head(catNums=80, roiSize=7)
    torch.save(head.state_dict(), "head.pt")
    print(head)
    # 验证 1
    # summary(net, input_size=[(3, 224, 224)])  
    # 验证 2
    # print(head)
    backbone_feat = torch.rand((BS, 256, 50, 50)) 
    RPNReg, RPNCls, RoIs, _, RoIIndices, anchor = rpn(backbone_feat, imgSize=imgSize)
    RoIReg, RoICls = head(backbone_feat, RoIs, RoIIndices, imgSize=imgSize)
    print(RoIReg.shape) # torch.Size([4, 600, 320])
    print(RoICls.shape) # torch.Size([4, 600, 80])