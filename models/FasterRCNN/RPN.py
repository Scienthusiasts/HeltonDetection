import torch
import torch.nn as nn
from torch.nn import functional as F

from utils.FasterRCNNAnchorUtils import *
from utils.util import *
from loss.Loss import ClassifyLoss, BBoxLoss








class RPN(nn.Module):
    def __init__(self, img_size, cat_nums, anchors, inChannels=256, midChannels=256, featStride=[4, 8, 16, 32, 64], minScale=8, mode="train"):
        '''RPN网络

        Args:
            :param inChannels:  backbone输出的特征, resnet50:[bs, 1024, 38, 38]
            :param midChannels:  
            :param ratios:     anchor的长宽比
            :param scales:     anchor的基本尺度, 实际尺度=scales*featStride
            :param featStride: 征图每个像素对应原图的几个像素, 等于下采样率
            :param minScale:   尺寸最小的特征图的下采样率 (PAFPN=8, FPN=4)
            :param mode:       train/eval(主要体现在保留的proposal数量不一样)

        Returns:
            None
        '''
        super(RPN, self).__init__()
        self.mode = mode
        '''损失函数'''
        self.regLoss = BBoxLoss(loss_type='SmoothL1Loss')
        # 省略掉那笑labels=-1的样本(不参与损失的计算)
        self.clsLoss = ClassifyLoss(loss_type='CrossEntropyLoss', cat_num=cat_nums)

        '''样本分配相关'''
        self.img_size = img_size
        self.min_scale = minScale
        # 特征图每个像素对应原图的几个像素, 等于下采样率
        self.featStride = torch.tensor(featStride)
        # 生成基础的先验框 [3(对应FPN3个尺度), 3(每个尺度三种形状), 2(w, h)]
        self.anchors = np.array(anchors)
        # 将9个先验框应用到整个特征图上, [[fpn_p3_w*fpn_p3_h*3, 4], [fpn_p4_w*fpn_p4_h*3, 4], [fpn_p5_w*fpn_p5_h*3, 4]]
        self.all_anchors = applyAnchor2Feat(genAnchor(self.anchors), self.featStride, self.img_size)
        # 一个尺度下一个grid里先验框的个数(=3)
        anchorNum = len(self.anchors[0])

        '''网络模块'''
        # 先进行一个3x3的卷积，可理解为特征整合
        self.conv = nn.Conv2d(inChannels, midChannels, 3, 1, 1)
        # 分类预测先验框内部是否包含物体(*2表示二分类,一个框只判断是前景还是背景)
        # 回归预测对先验框的坐标进行微调
        self.cls = nn.Conv2d(midChannels, anchorNum * 2, 1, 1, 0)
        self.reg = nn.Conv2d(midChannels, anchorNum * 4, 1, 1, 0)

        '''参数初始化'''
        # init_weights(self.conv, 'he')
        # init_weights(self.cls, 'he')
        # init_weights(self.reg, 'he')
        init_weights(self.conv, 'normal', 0, 0.01)
        init_weights(self.cls, 'normal', 0, 0.01)
        init_weights(self.reg, 'normal', 0, 0.001)












    def decode(self, x, RPNReg, RPNCls, imgSize, scale, layer_id):
        '''后处理(单尺度)

        Args:
            :param x:       backbone输出的特征, 
            :param RPNReg:  RPN网络输出每个proposal相对于anchor的offset
            :param RPNCls:  RPN网络输出每个proposal属于前景/背景的概率
            :param imgSize: 原始图像尺寸
            :param scale:   用于约束proposal的最小尺寸=scalex16 

        Returns:
            :param RPNReg:     RPN网络输出每个proposal相对于anchor的offset 
            :param RPNCls:     RPN网络输出每个proposal属于前景/背景的概率
            :param RoIs:       一个batch里所有图像的proposal集合
            :param RoIIndices: 一个batch里所有proposal的索引, 这样才知道哪些proposal属于哪张图像
            :param anchor:     9个初始先验框
        '''
        bs, _, h, w = x.shape
        # [bs, 3*4, w, h] -> [bs, w, h, 3*4] -> [bs, w*h*3, 4]
        RPNReg = RPNReg.permute(0, 2, 3, 1).reshape(bs, -1, 4)  # [bs, w*h*3, 4]
        RPNCls = RPNCls.permute(0, 2, 3, 1).reshape(bs, -1, 2)  # [bs, w*h*3, 2]
        # 对二分类结果进行softmax计算概率
        # RPNCls = F.softmax(RPNCls, dim=-1)
        # RPNCls[:, :, 1]的内容为包含物体的概率
        # [bs, w*h*3=12996, 2] -> [bs, w*h*3]
        RPNForeScore = F.softmax(RPNCls, dim=-1)[..., 1].reshape(bs, -1) 
        # RoIs是一个batch里所有图像的proposal集合, RoIIndices存储每个RoI对应的索引,这样才知道哪些proposal属于哪张图像
        RoIs, RoIIndices, origin_RoIs = [], [], []
        # 对一个Batch里的每张图像分别执行nms
        for i in range(bs):
            # 将RPN网络预测的offset作用在anchor上生成proposal  [12996, 4] anchor=(featW*featH*3, 4)
            roi = reg2Bbox(self.all_anchors[layer_id].to(RPNReg.device), RPNReg[i]) 
            '''dense2sparse关键一步'''
            # 筛选少量合适的proposals作为head的输入 (去掉太小的框 + 置信度太低的框 + NMS筛选)
            if self.mode == "train":
                RoI, origin_RoI = dense2SparseProposals(roi, RPNForeScore[i], imgSize, scale, 12000, 200, nms_iou=0.7)
            else:
                RoI, origin_RoI = dense2SparseProposals(roi, RPNForeScore[i], imgSize, scale, 3000, 100, nms_iou=0.7)
            
            # batch_index用于确定哪个框对应那张图像
            batch_index = i * torch.ones((len(RoI)))
            RoIs.append(RoI.unsqueeze(0))
            RoIIndices.append(batch_index.unsqueeze(0))
            origin_RoIs.append(origin_RoI.unsqueeze(0))
        # list->tensor
        RoIs = torch.cat(RoIs, dim=0).type_as(RPNReg)
        RoIIndices = torch.cat(RoIIndices, dim=0).type_as(RPNReg)
        origin_RoIs = torch.cat(origin_RoIs, dim=0).type_as(RPNReg)

        return RPNReg, RPNCls, RoIs, origin_RoIs, RoIIndices





    def forwardSinglelvl(self, x):
        '''前向传播

        Args:
            :param x:  FPN某一单尺度的特征图

        Returns:
            :param RPNReg:     RPN网络输出每个proposal相对于anchor的offset
            :param RPNCls:     RPN网络输出每个proposal属于前景/背景的概率
        '''
        # 特征图的大小featW, featH = 38
        # backbone之后先进行一个3x3的卷积，可理解为特征整合 [bs, 512,  featW, featH]
        x = F.relu(self.conv(x))
        # 回归预测对先验框进行调整 [bs, 3*4, featW, featH]
        RPNReg = self.reg(x)   
        # 分类预测先验框内部是否包含物体 [bs, 3*2, featW, featH]
        RPNCls = self.cls(x)      
        return RPNReg, RPNCls


    def forward(self, lvl_x, imgSize, tta=False):
        '''前向传播

        Args:
            :param x:       backbone输出的特征, resnet50:[bs, 1024, 38, 38]
            :param imgSize: 原始图像尺寸
            :param scale:   用于约束proposal的最小尺寸=scalex16 

        Returns:
            :param lvl_rpn_reg:     RPN网络输出每个proposal相对于anchor的offset [bs, w*h*3, 4]
            :param lvl_rpn_cls:     RPN网络输出每个proposal属于前景/背景的概率  [bs, w*h*3, 2]
            :param lvl_nms_RoIs:    一个batch里所有图像的proposal集合  [bs, 100, 4]
            :param lvl_origin_RoIs: 遍布整个特征图的初始anchor(原图尺寸)  [bs, w*h*3, 4]
            :param  lvl_RoIs_idx:   一个batch里所有proposal的索引, 这样才知道哪些proposal属于哪张图像 [bs, 100]
        ''' 
        if tta:
            self.all_anchors = applyAnchor2Feat(genAnchor(self.anchors), self.featStride, imgSize)
        # 这些变量用于保存不同尺度的前向结果
        lvl_rpn_reg, lvl_rpn_cls, lvl_nms_RoIs, lvl_origin_RoIs, lvl_RoIs_idx = [], [], [], [], []
        '''对fpn不同特征层分别做前向'''
        for layer_id, x in enumerate(lvl_x):
            # 当前layer下采样率:
            scale = self.min_scale * 2**layer_id
            # 如果当前尺度不参与计算, 则略过:
            if scale not in self.featStride:continue
            '''单尺度前向'''
            # RPNReg:[bs, 3*4, featW, featH]  RPNCls:[bs, 3*2, featW, featH]
            RPNReg, RPNCls = self.forwardSinglelvl(x)
            '''对RPN回归的参数进行解码(解码回原图尺度下的框 + NMS)'''
            # [bs, w*h*3, 4] [bs, w*h*3, 2] [bs, 100, 4] [bs, w*h*3, 4] [bs, 100] (train:200, val/test:100)
            RPNReg, RPNCls, RoIs, origin_RoI, RoIIndices = self.decode(x, RPNReg, RPNCls, imgSize, scale, layer_id)
            lvl_rpn_reg.append(RPNReg)
            lvl_rpn_cls.append(RPNCls)
            lvl_nms_RoIs.append(RoIs)
            lvl_origin_RoIs.append(origin_RoI)
            lvl_RoIs_idx.append(RoIIndices)

        return lvl_rpn_reg, lvl_rpn_cls, lvl_nms_RoIs, lvl_origin_RoIs, lvl_RoIs_idx





    def batchLoss(self, base_feature, y_trues):
        '''RPN计算损失
        
        # Args:
            :param base_feature: backbone输出的特征, 包含3个尺度
            :param y_trues:      正负样本分配结果, 这部分在dataset阶段执行

        # Returns:
            :param loss:        RPN网络输出每个proposal相对于anchor的offset [bs, w*h*3, 4]
            :param rois:        RPN网络输出每个proposal属于前景/背景的概率  [bs, w*h*3, 2]
            :param origin_rois: 一个batch里所有图像的proposal集合  [bs, 100, 4]
            :param roi_indices: 遍布整个特征图的初始anchor(原图尺寸)  [bs, w*h*3, 4]
        ''' 

        '''利用rpn网络获得调整参数、得分、建议框、先验框'''
        rpn_locs, rpn_scores, rois, origin_rois, roi_indices = self.forward(base_feature, self.img_size)
        # 将不同尺度预测结果拼在一起
        rois = torch.cat(rois, dim=1)
        roi_indices = torch.cat(roi_indices, dim=1)

        rpn_loc_loss, rpn_cls_loss = 0, 0
        # 对fpn每一层的预测结果分别输入同一个RPN, 并计算损失
        for lvl_rpn_locs, lvl_rpn_scores, y_true in zip(rpn_locs, rpn_scores, y_trues):
            '''正负样本分配(即y_true, 这部分修改为在dataset阶段执行)'''
            gt_rpn_locs = y_true[..., :4]
            gt_rpn_labels = y_true[..., 4].long()
            origin_roi_gts = y_true[..., 5:]

            '''计算RPN损失'''
            '''RPN回归损失'''
            # 只取那些正样本计算回归损失
            #                           torch.Size([num_pos, 4])             torch.Size([num_pos, 4])
            lvl_rpn_loc_loss = self.regLoss(lvl_rpn_locs[gt_rpn_labels.data > 0], gt_rpn_locs[gt_rpn_labels.data > 0])
            '''下面这部分用来计算IoU loss'''
            # origin_roi_gts = origin_roi_gts.to(gt_rpn_labels.device)
            # pred_pos_box, pos_box_gt = origin_rois[gt_rpn_labels.data > 0], origin_roi_gts[gt_rpn_labels.data > 0]
            # rpn_loc_loss = self.regLoss(pred_pos_box, pos_box_gt)
            '''RPN分类损失'''
            # 舍弃掉那些-1的样本计算分类损失 shape=256
            # (注意, 这里是根据anchor和GT的IoU来分配正负样本，而不是proposal和GT的IoU)(且正负样本不会刚好一半一半)
            #                           torch.Size([256*bs, 2])               torch.Size([256*bs])
            lvl_rpn_cls_loss = self.clsLoss(lvl_rpn_scores[gt_rpn_labels.data > -1], gt_rpn_labels[gt_rpn_labels.data > -1], mode='rpn')
            rpn_loc_loss += lvl_rpn_loc_loss
            rpn_cls_loss += lvl_rpn_cls_loss

        '''统一损失的格式:字典形式: {损失1, 损失2,..., 总损失}'''
        loss = dict(
            rpn_reg_loss = rpn_loc_loss / len(y_trues),
            rpn_cls_loss = rpn_cls_loss / len(y_trues)
        )

        return loss, rois, origin_rois, roi_indices




























# for test only
if __name__ == '__main__':
    from torchsummary import summary
    # backbone:[bs, 3, 600, 600] -> [bs, 1024, 38, 38]
    rpn = RPN()
    # 验证 1
    # summary(backbone, input_size=[(3, 600, 600)])  
    # 验证 2
    print(rpn)
    x = torch.rand((4, 1024, 38, 38))
    RPNReg, RPNCls, RoIs, RoIIndices, anchor = rpn(x, imgSize=[768,768])
    print(RPNReg.shape)    # torch.Size([4, 12996, 4])
    print(RPNCls.shape)    # torch.Size([4, 12996, 2])
    print(RoIs.shape)      # torch.Size([4, 600, 4])
    print(RoIIndices.shape)# torch.Size([4, 600])
    print(anchor.shape)    # torch.Size([1, 12996, 4])