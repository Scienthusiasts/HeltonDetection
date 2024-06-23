import torch
import torch.nn as nn
import timm
import numpy as np
from torchvision import models

from models.YOLOv8.YOLOBlocks import *

'''
huggingface里的timm模型:
https://huggingface.co/timm?sort_models=downloads#models
'''






class Backbone(nn.Module):
    '''来自timm库的通用Backbone
    '''
    def __init__(self, modelType:str, loadckpt=False, pretrain=True, froze=True):
        '''网络初始化

        Args:
            - modelType: 使用哪个模型(timm库里的模型)
            - loadckpt:  是否导入模型权重(是则输入权重路径)
            - pretrain:  是否用预训练模型进行初始化
            - froze:     是否冻结Backbone

        Returns:
            None
        '''
        super(Backbone, self).__init__()
        # 模型接到线性层的维度

        # 加载模型(features_only=True 只加载backbone部分)
        # features_only=True 只提取特征图(5层)，不加载分类头
        # out_indices=[3] 只提取到backbone的倒数第二层特征, 第一层留给head提取
        out_indices = [2,3,4] if modelType not in ['darknetaa53.c2ns_in1k', 'cspdarknet53.ra_in1k'] else [3,4,5]
        self.backbone = timm.create_model(modelType, pretrained=pretrain, features_only=True, out_indices=out_indices)
        # 是否冻结backbone
        if froze:
            for param in self.backbone.parameters():
                param.requires_grad_(False)
        # 是否导入预训练权重
        if loadckpt: 
            self.load_state_dict(torch.load(loadckpt))
            print('backbone pretrain ckpt loaded')

    def forward(self, x):
        '''前向传播
        '''
        x = self.backbone(x)
        return x







class CSPDarknet(nn.Module):
    '''YOLOv8专属Backbone
    '''
    def __init__(self, phi, base_channels, base_depth, deep_mul, loadckpt=False, pretrain=True, froze=True):
        super().__init__()
        '''网络组件'''
        # 3, 640, 640 => 32, 640, 640 => 64, 320, 320
        self.stem = Conv(3, base_channels, 3, 2)
        # 64, 320, 320 => 128, 160, 160 => 128, 160, 160
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2),
            C2f(base_channels * 2, base_channels * 2, base_depth, True),
        )
        # 128, 160, 160 => 256, 80, 80 => 256, 80, 80
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2),
            C2f(base_channels * 4, base_channels * 4, base_depth * 2, True),
        )
        # 256, 80, 80 => 512, 40, 40 => 512, 40, 40
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2),
            C2f(base_channels * 8, base_channels * 8, base_depth * 2, True),
        )
        # 512, 40, 40 => 1024 * deep_mul, 20, 20 => 1024 * deep_mul, 20, 20
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, int(base_channels * 16 * deep_mul), 3, 2),
            C2f(int(base_channels * 16 * deep_mul), int(base_channels * 16 * deep_mul), base_depth, True),
            SPPF(int(base_channels * 16 * deep_mul), int(base_channels * 16 * deep_mul), k=5),
        )
        '''导入预训练权重'''
        if pretrain:
            backbone = "v8_cspdarknet_" + phi
            url = {
                "v8_cspdarknet_n" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_n_backbone_weights.pth',
                "v8_cspdarknet_s" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_s_backbone_weights.pth',
                'v8_cspdarknet_m' : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_m_backbone_weights.pth',
                'v8_cspdarknet_l' : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_l_backbone_weights.pth',
                'v8_cspdarknet_x' : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_x_backbone_weights.pth',
            }[backbone]
            checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", model_dir="./ckpt")
            self.load_state_dict(checkpoint, strict=False)
            print("Load weights from ", url.split('/')[-1])
            
        # 是否冻结backbone
        if froze:
            for param in self.parameters():
                param.requires_grad_(False)

        # 是否导入预训练权重
        if loadckpt!=False: 
            self.load_state_dict(torch.load(loadckpt))
            print('yolov8 backbone pretrain ckpt loaded!')


    def forward(self, x):
        p1 = self.stem(x)
        p2 = self.dark2(p1)
        p3 = self.dark3(p2)
        p4 = self.dark4(p3)
        p5 = self.dark5(p4)
        return p3, p4, p5











# for test only
if __name__ == '__main__':
    '''基本配置: n s m l x'''
    phi = 's'
    depth_dict          = {'n' : 0.33, 's' : 0.33, 'm' : 0.67, 'l' : 1.00, 'x' : 1.00,}
    width_dict          = {'n' : 0.25, 's' : 0.50, 'm' : 0.75, 'l' : 1.00, 'x' : 1.25,}
    deep_width_dict     = {'n' : 1.00, 's' : 1.00, 'm' : 0.75, 'l' : 0.50, 'x' : 0.50,}
    dep_mul, wid_mul, deep_mul = depth_dict[phi], width_dict[phi], deep_width_dict[phi]
    base_channels       = int(wid_mul * 64)
    base_depth          = max(round(dep_mul * 3), 1)

    loadckpt = False
    backbone = CSPDarknet(phi, base_channels, base_depth, deep_mul, loadckpt, pretrain=True, froze=True)
    # 验证
    x = torch.rand((4, 3, 640, 640))
    outs = backbone(x)
    for out in outs: print(out.shape)

    # n:[64,  128, 256]
    # s:[128, 256, 512]
    # m:[192, 384, 576]
    # l:[256, 512, 512]
    # x:[324, 640, 640]
