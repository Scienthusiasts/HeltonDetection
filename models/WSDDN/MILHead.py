import torch
import torch.nn as nn
from torchvision.ops import RoIPool, RoIAlign, roi_pool
import timm
from torchvision import models
import copy
from utils.FasterRCNNAnchorUtils import *
from utils.util import *







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
    def __init__(self, inchannel, outchannel):
        super(ShareHead, self).__init__()
        self.squeezeConv = self._make_conv_block(inchannel, outchannel, 1)
        self.convBlocks =  nn.Sequential(
            self._make_conv_block(outchannel, outchannel), 
            self._make_conv_block(outchannel, outchannel), 
            self._make_conv_block(outchannel, outchannel),
        )
        # 权重初始化
        # init_weights(self.convBlocks, 'he')
        init_weights(self.squeezeConv, 'normal', 0, 0.01)
        init_weights(self.convBlocks, 'normal', 0, 0.01)


    def _make_conv_block(self, in_channels, out_channels, kernel=3):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel, 1, bias=False))
        layers.append(CustomBatchNorm2d(out_channels))
        layers.append(nn.ReLU())
        in_channels = out_channels
        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.squeezeConv(x)
        x = self.convBlocks(x)
        return x    










class MILHead(nn.Module):
    '''Backbone
    '''
    def __init__(self, backbone_name, catNums:int, roiSize:int):
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
        self.roiSize = roiSize
        super(MILHead, self).__init__()
        self.cat_nums = catNums
        # 共享卷积层
        self.in_channel = {
            'resnet50.a1_in1k':                           1024,
            'vgg16.tv_in1k':                              4096,
            }[backbone_name]
        # 共享分支
        self.share_head = ShareHead(inchannel=self.in_channel, outchannel=256)
        # MIL CLS分支
        self.cls = nn.Linear(256, self.cat_nums)
        # MIL INS分支
        self.ins = nn.Linear(256, self.cat_nums)
        # 权重初始化
        # init_weights(self.reg, 'he')
        # init_weights(self.cls, 'he')
        init_weights(self.ins, 'normal', 0, 0.01)
        init_weights(self.cls, 'normal', 0, 0.01)




    def forward(self, x, proposals):
        '''前向传播

        Args:
            :param x:          backbone输出的特征, resnet50:[bs, 1024, 38, 38]
            :param proposals: 先验框 [bs=1, numProposalPerImg, 4]

        Returns:
            :param cls_logit: 分类分支的原始预测结果
            :param ins_logit: 实例分支的原始预测结果
        '''
        # proposals默认batch为1，因此不需要batch维度
        # feature map根据 proposal提取RoI区域特征并执行RoIPooling [numProposal, C, self.RoISize, self.RoISize]
        roi_feat = roi_pool(x, [proposals.squeeze(0)], self.roiSize, 1.0 / 16)
        # RoI特征先经过FC层进一步提取特征 [numProposal, 4096, 1, 1]
        roi_feat = self.share_head(roi_feat) 
        # 每个proposal的特征都是一个4096维的向量 [numProposal, 4096]
        roi_feat = roi_feat.view(roi_feat.shape[0], -1) 
        # MIL CLS分支(对类别维度做softmax, 得到每个proposal属于每个类别的置信度) [numProposal, clsNum] 
        cls_logit = self.cls(roi_feat)
        # MIL INS分支(对proposal维度做softmax, 得到每个类别下, 每个proposal对该类别的贡献度) [numProposal, clsNum] 
        ins_logit = self.ins(roi_feat)
        return cls_logit, ins_logit

    













