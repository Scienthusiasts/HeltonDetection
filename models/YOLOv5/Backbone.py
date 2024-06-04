import torch
import torch.nn as nn
import timm
import numpy as np
from torchvision import models

from models.YOLOv5.YOLOBlocks import *

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
    '''YOLOv5专属Backbone
    '''
    def __init__(self, phi, base_channels, base_depth, loadckpt=False, pretrain=True, froze=True):
        super().__init__()
        '''网络组件'''
        self.stem = Conv(3, base_channels, 6, 2, 2)
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2),
            C3(base_channels * 2, base_channels * 2, base_depth),
        )
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2),
            C3(base_channels * 4, base_channels * 4, base_depth * 2),
        )
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2),
            C3(base_channels * 8, base_channels * 8, base_depth * 3),
        )
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2),
            C3(base_channels * 16, base_channels * 16, base_depth),
            SPPF(base_channels * 16, base_channels * 16),
        )
        '''导入预训练权重'''
        if pretrain:
            backbone = "cspdarknet_" + phi
            url = {
                "cspdarknet_n" : 'https://github.com/bubbliiiing/yolov5-v6.1-pytorch/releases/download/v1.0/cspdarknet_n_v6.1_backbone.pth',
                "cspdarknet_s" : 'https://github.com/bubbliiiing/yolov5-v6.1-pytorch/releases/download/v1.0/cspdarknet_s_v6.1_backbone.pth',
                'cspdarknet_m' : 'https://github.com/bubbliiiing/yolov5-v6.1-pytorch/releases/download/v1.0/cspdarknet_m_v6.1_backbone.pth',
                'cspdarknet_l' : 'https://github.com/bubbliiiing/yolov5-v6.1-pytorch/releases/download/v1.0/cspdarknet_l_v6.1_backbone.pth',
                'cspdarknet_x' : 'https://github.com/bubbliiiing/yolov5-v6.1-pytorch/releases/download/v1.0/cspdarknet_x_v6.1_backbone.pth',
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
            print('yolov5 pretrain ckpt loaded!')


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
    phi = 'l'
    depth_dict          = {'n':0.33, 's':0.33, 'm':0.67, 'l':1.00, 'x':1.33}
    width_dict          = {'n':0.25, 's':0.50, 'm':0.75, 'l':1.00, 'x':1.25}
    dep_mul, wid_mul    = depth_dict[phi], width_dict[phi]
    base_channels       = int(wid_mul * 64)
    base_depth          = max(round(dep_mul * 3), 1)

    loadckpt = f'ckpt/cspdarknet_{phi}_v6.1_backbone.pth'
    backbone = CSPDarknet(phi, base_channels, base_depth, loadckpt, pretrain=True, froze=True)
    # 验证
    x = torch.rand((4, 3, 640, 640))
    outs = backbone(x)
    for out in outs: print(out.shape)

    # n:[64,  128, 256 ]
    # s:[128, 256, 512 ]
    # m:[192, 384, 768 ]
    # l:[256, 512, 1024]
    # x:[324, 640, 1280]
