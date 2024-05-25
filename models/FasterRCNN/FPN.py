import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision import models


from utils.util import *





class FPN(nn.Module):
    '''Feature Pyramid Network
    '''
    def __init__(self, C5_channel, C4_channel, C3_channel, C2_channel):
        super(FPN, self).__init__()
        self.maxpool2d = nn.MaxPool2d(1, stride=2)
        # lateral layer 是不同尺度特征图融合之前的卷积
        self.latlayer1 = nn.Conv2d(C5_channel, 256, 1, 1, 0)
        self.latlayer2 = nn.Conv2d(C4_channel, 256, 1, 1, 0)
        self.latlayer3 = nn.Conv2d(C3_channel, 256, 1, 1, 0)
        self.latlayer4 = nn.Conv2d(C2_channel, 256, 1, 1, 0)
        # 是不同尺度特征图融合之后的卷积
        self.smooth1 = nn.Conv2d(256, 256, 3, 1, 1)
        self.smooth2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.smooth3 = nn.Conv2d(256, 256, 3, 1, 1)
        self.smooth4 = nn.Conv2d(256, 256, 3, 1, 1)
        # 权重初始化
        # init_weights(self.latlayer1, 'he')
        # init_weights(self.latlayer2, 'he')
        # init_weights(self.latlayer3, 'he')
        # init_weights(self.latlayer4, 'he')
        # init_weights(self.smooth1, 'he')
        # init_weights(self.smooth2, 'he')
        # init_weights(self.smooth3, 'he')
        # init_weights(self.smooth4, 'he')
        init_weights(self.latlayer1, 'normal', 0, 0.01)
        init_weights(self.latlayer2, 'normal', 0, 0.01)
        init_weights(self.latlayer3, 'normal', 0, 0.01)
        init_weights(self.latlayer4, 'normal', 0, 0.01)
        init_weights(self.smooth1, 'normal', 0, 0.01)
        init_weights(self.smooth2, 'normal', 0, 0.01)
        init_weights(self.smooth3, 'normal', 0, 0.01)
        init_weights(self.smooth4, 'normal', 0, 0.01)


    def _upsample_add(self, x, y):
        '''将特征图x上采样到特征图y的大小并与y相加
        '''
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') + y
    

    def forward(self, x):
        c2, c3, c4, c5 = x
        # Top-down
        p5 = self.smooth1(self.latlayer1(c5))
        p4 = self.smooth2(self._upsample_add(p5, self.latlayer2(c4)))
        p3 = self.smooth3(self._upsample_add(p4, self.latlayer3(c3)))
        p2 = self.smooth4(self._upsample_add(p3, self.latlayer4(c2)))
        p6 = self.maxpool2d(p5)
        # 注意, p2只在head使用
        return [p2, p3, p4, p5, p6]











# for test only
if __name__ == '__main__':
    from models.FasterRCNN.Backbone import *
    # mobilenetv3_large_100.ra_in1k resnet50.a1_in1k  
    # backbone:[bs, 3, 600, 600] -> [bs, 1024, 38, 38]
    backbone = Backbone(modelType='resnet50.a1_in1k', pretrain=False)
    # mobilenet:[960,  112,  40,  24], resnet:[2048, 1024, 512, 256]
    fpn = FPN(2048, 1024, 512, 256)
    x = torch.rand((4, 3, 832, 832))
    x = backbone(x)
    outs = fpn(x)
    for out in outs:
        print(out.shape)

    # torch.Size([bs, 256, 208, 208])
    # torch.Size([bs, 256, 104, 104])
    # torch.Size([bs, 256, 52, 52])
    # torch.Size([bs, 256, 26, 26])
    # torch.Size([bs, 256, 13, 13])