import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision import models

from models.FasterRCNN.YOLOBlocks import *
from utils.util import *





class PAFPN(nn.Module):
    '''Feature Pyramid Network
    '''
    def __init__(self, C5_channel, C4_channel, C3_channel, T4_channel=512, T3_channel=256, P5_channel=1024, P4_channel=512):
        super(PAFPN, self).__init__()
        self.maxpool2d = nn.MaxPool2d(1, stride=2)
        # 上采样拼接后的卷积
        self.C2f_upsample_T4 = C2f(C5_channel+C4_channel, T4_channel, 3)
        self.C2f_upsample_T3 = C2f(T4_channel+C3_channel, T3_channel, 3)
        # 下采样拼接后的卷积
        self.T3_downsample_conv = Conv(T3_channel, T3_channel, 3, 2, 1)
        self.C2f_downsample_T3 = C2f(T3_channel+T4_channel, P4_channel, 3)
        self.P4_downsample_conv = Conv(P4_channel, P4_channel, 3, 2, 1)
        self.C2f_downsample_T4 = C2f(P4_channel+C5_channel, P5_channel, 3)
        # 权重初始化
        init_weights(self.C2f_upsample_T4, 'normal', 0, 0.01)
        init_weights(self.C2f_upsample_T3, 'normal', 0, 0.01)
        init_weights(self.T3_downsample_conv, 'normal', 0, 0.01)
        init_weights(self.C2f_downsample_T3, 'normal', 0, 0.01)
        init_weights(self.P4_downsample_conv, 'normal', 0, 0.01)
        init_weights(self.C2f_downsample_T4, 'normal', 0, 0.01)


    def _upsample_cat(self, x, y):
        '''将特征图x上采样到特征图y的大小并与y拼接
        '''
        _, _, H, W = y.size()
        # 按照通道维度拼接
        return torch.cat((F.interpolate(x, size=(H, W), mode='bilinear'), y), dim=1)
    

    def forward(self, x):
        # 对于输入图像大小=640x640, c3.channel=512x80x80, c4.channel=1024x40x40, c5.channel=2048x20x20 (resnet50)
        c3, c4, c5 = x
        # 上采样融合
        t5 = c5 # torch.Size([4, 1024, 20, 20])
        t4 = self.C2f_upsample_T4(self._upsample_cat(c5, c4)) # torch.Size([4, 512, 40, 40])
        t3 = self.C2f_upsample_T3(self._upsample_cat(t4, c3)) # torch.Size([4, 256, 80, 80])
        # 下采样融合
        p3 = t3 # torch.Size([4, 256, 80, 80])
        p4 = self.C2f_downsample_T3(torch.cat((self.T3_downsample_conv(t3), t4), dim=1)) # torch.Size([4, 512, 40, 40])
        p5 = self.C2f_downsample_T4(torch.cat((self.P4_downsample_conv(p4), t5), dim=1)) # torch.Size([4, 1024, 20, 20])

        return p3, p4, p5











# for test only
if __name__ == '__main__':
    from models.FasterRCNN.Backbone import *
    # mobilenetv3_large_100.ra_in1k resnet50.a1_in1k  
    # backbone:[bs, 3, 600, 600] -> [bs, 1024, 38, 38]
    backbone = Backbone(modelType='resnet50.a1_in1k', pretrain=False)
    # mobilenet:[960, 112, 40], resnet:[2048, 1024, 512]
    fpn = PAFPN(2048, 1024, 512)
    x = torch.rand((4, 3, 640, 640))
    x = backbone(x)
    outs = fpn(x)
    for out in outs:
        print(out.shape)

        # torch.Size([4, 256, 80, 80])
        # torch.Size([4, 512, 40, 40])
        # torch.Size([4, 1024, 20, 20])