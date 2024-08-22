import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision import models
from utils.util import *






class FPN(nn.Module):
    def __init__(self, C3_channel, C4_channel, C5_channel, out_channel=256):
        super(FPN,self).__init__()
        # backbone特征输入FPN之前的卷积
        self.prj_3 = nn.Conv2d(C3_channel, out_channel, kernel_size=1)
        self.prj_4 = nn.Conv2d(C4_channel, out_channel, kernel_size=1)
        self.prj_5 = nn.Conv2d(C5_channel, out_channel, kernel_size=1)
        # 多尺度特征融合之后的卷积
        self.conv_3 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1)
        self.conv_4 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1)
        self.conv_5 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1)
        # 下采样卷积(缩小特征图尺寸, 检测大目标)
        self.conv_out6 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, stride=2)
        self.conv_out7 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, stride=2)
        # 权重初始化
        init_weights(self.prj_3, 'normal', 0, 0.01)
        init_weights(self.prj_4, 'normal', 0, 0.01)
        init_weights(self.prj_5, 'normal', 0, 0.01)
        init_weights(self.conv_3, 'normal', 0, 0.01)
        init_weights(self.conv_4, 'normal', 0, 0.01)
        init_weights(self.conv_5, 'normal', 0, 0.01)
        init_weights(self.conv_out6, 'normal', 0, 0.01)
        init_weights(self.conv_out7, 'normal', 0, 0.01)


    def _upsample_add(self, x, y):
        '''将特征图x上采样到特征图y的大小并与y相加
        '''
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='nearest') + y
    

    def forward(self,x):
        C3, C4, C5 = x

        # 80, 80, 512 -> 80, 80, 256
        T3 = self.prj_3(C3)
        # 40, 40, 1024 -> 40, 40, 256
        T4 = self.prj_4(C4)
        # 20, 20, 2048 -> 20, 20, 256
        T5 = self.prj_5(C5)
            
        # 40, 40, 256 -> 80, 80, 256
        P3 = self.conv_3(self._upsample_add(T4, T3))
        # 20, 20, 256 -> 40, 40, 256
        P4 = self.conv_4(self._upsample_add(T5, T4))
        # 20, 20, 256
        P5 = self.conv_5(T5)

        # 10, 10, 256
        P6 = self.conv_out6(P5)
        # 5, 5, 256
        P7 = self.conv_out7(F.relu(P6))
        return [P3, P4, P5, P6, P7]








# for test only
if __name__ == '__main__':
    from models.FasterRCNN.Backbone import *
    # mobilenetv3_large_100.ra_in1k resnet50.a1_in1k  
    # backbone:[bs, 3, 600, 600] -> [bs, 1024, 38, 38]
    backbone = Backbone(modelType='resnet50.a1_in1k', pretrain=False)
    # mobilenet:[960,  112,  40,  24], resnet:[2048, 1024, 512, 256]
    fpn = FPN(512, 1024, 2048)
    x = torch.rand((4, 3, 640, 640))
    x = backbone(x)
    outs = fpn(x)
    
    for out in outs:
        print(out.shape)


    # torch.Size([bs, 256, 80, 80])
    # torch.Size([bs, 256, 40, 40])
    # torch.Size([bs, 256, 20, 20])
    # torch.Size([bs, 256, 10, 10])
    # torch.Size([bs, 256, 5,  5 ])