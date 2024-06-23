import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision import models

from models.YOLOv8.YOLOBlocks import *
from utils.util import *





class YOLOv8PAFPN(nn.Module):
    '''Feature Pyramid Network
    '''
    def __init__(self, base_channels, base_depth, deep_mul):
        super(YOLOv8PAFPN, self).__init__()
        self.maxpool2d = nn.MaxPool2d(1, stride=2)
        # 上采样拼接后的卷积
        self.C2f_upsample_T4 = C2f(int(base_channels * 16 * deep_mul) + base_channels * 8, base_channels * 8, base_depth, shortcut=False)
        self.C2f_upsample_T3 = C2f(base_channels * 8 + base_channels * 4, base_channels * 4, base_depth, shortcut=False)
        # 下采样拼接后的卷积
        self.T3_downsample_conv = Conv(base_channels * 4, base_channels * 4, 3, 2)
        self.C2f_downsample_T3 = C2f(base_channels * 8 + base_channels * 4, base_channels * 8, base_depth, shortcut=False)
        self.P4_downsample_conv = Conv(base_channels * 8, base_channels * 8, 3, 2)
        self.C2f_downsample_T4 = C2f(int(base_channels * 16 * deep_mul) + base_channels * 8, int(base_channels * 16 * deep_mul), base_depth, shortcut=False)
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
    from models.YOLOv8.Backbone import *
    '''基本配置: n s m l x'''
    phi = 'x'
    depth_dict          = {'n' : 0.33, 's' : 0.33, 'm' : 0.67, 'l' : 1.00, 'x' : 1.00,}
    width_dict          = {'n' : 0.25, 's' : 0.50, 'm' : 0.75, 'l' : 1.00, 'x' : 1.25,}
    deep_width_dict     = {'n' : 1.00, 's' : 1.00, 'm' : 0.75, 'l' : 0.50, 'x' : 0.50,}
    dep_mul, wid_mul, deep_mul = depth_dict[phi], width_dict[phi], deep_width_dict[phi]
    base_channels       = int(wid_mul * 64)
    base_depth          = max(round(dep_mul * 3), 1)

    loadckpt = f'ckpt/yolov8_{phi}_backbone_weights.pth'
    # 骨干
    backbone = CSPDarknet(phi, base_channels, base_depth, deep_mul, loadckpt, pretrain=True, froze=True)
    # FPN
    fpn = YOLOv8PAFPN(base_channels, base_depth, deep_mul)
    x = torch.rand((4, 3, 640, 640))
    x = backbone(x)
    outs = fpn(x)
    for out in outs: print(out.shape)

    # n:[64,  128, 256]
    # s:[128, 256, 512]
    # m:[192, 384, 576]
    # l:[256, 512, 512]
    # x:[324, 640, 640]