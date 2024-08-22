import torch.nn as nn
from PIL import Image
from collections import Counter
import time

from utils.metrics import *
from models.FCOS.Backbone import *
from models.FCOS.FPN import *
from models.FCOS.Head import *




class Model(nn.Module):
    '''完整YOLOv5网络架构
    '''

    def __init__(self, backbone_name, img_size, num_classes, loadckpt, tta_img_size, backbone:dict, head:dict):
        super(Model, self).__init__()

        # 类别数
        self.num_classes = num_classes
        self.img_size = img_size
        # 对应backbone每一层的输出(C5, C4, C3, C2)
        model_param = {
            'resnet50.a1_in1k':              [2048, 1024, 512],
            'cspresnext50.ra_in1k':          [2048, 1024, 512],
            'mobilenetv3_large_100.ra_in1k': [960,  112,  40 ],
            'darknetaa53.c2ns_in1k':         [1024, 512,  256],
            'cspdarknet53.ra_in1k':          [1024, 512,  256],
        }[backbone_name]

        '''网络基本组件'''
        # Backbone最好使用原来的预训练权重初始化
        self.backbone = Backbone(**backbone)
        self.fpn = FPN(C3_channel=model_param[2], C4_channel=model_param[1], C5_channel=model_param[0])
        self.head = Head(**head)
        '''TTA增强'''
        self.tta = TTA(tta_img_size=tta_img_size)
        # 是否导入预训练权重
        if loadckpt: 
            # self.load_state_dict(torch.load(loadckpt))
            # 基于尺寸的匹配方式(能克服局部模块改名加载不了的问题)
            self = loadWeightsBySizeMatching(self, loadckpt)


    def forward(self, x):
        backbone_feat = self.backbone(x)
        fpn_feat = self.fpn(backbone_feat)
        cls_logits, cnt_logits, reg_preds = self.head(fpn_feat)

        return cls_logits, cnt_logits, reg_preds








    def batchLoss(self, device, img_size, batch_datas):
        '''一个batch的前向流程(不包括反向传播, 更新梯度)(核心, 要更改训练pipeline主要改这里)

        # Args:
            - `img_size`:     固定图像大小 如[832, 832]
            - `batch_imgs`:   一个batch里的图像              例:shape=[bs, 3, 600, 600]
            - `batch_bboxes`: 一个batch里的GT框              例:[(1, 4), (4, 4), (4, 4), (1, 4), (5, 4), (2, 4), (3, 4), (1, 4)]
            - `batch_labels`: 一个batch里的GT框类别          例:[(1,), (4,), (4,), (1,), (5,), (2,), (3,), (1,)]

        # # Returns:
            - losses: 所有损失组成的列表(里面必须有一个total_loss字段, 用于反向传播)
        '''
        batch_imgs, batch_bboxes, batch_labels = batch_datas[0].to(device), batch_datas[1].to(device), batch_datas[2].to(device)
        
        # 前向过程
        backbone_feat = self.backbone(batch_imgs)
        fpn_feat = self.fpn(backbone_feat)
        # 计算损失(FCOS的正负样本分配在head部分执行)
        loss = self.head.batchLoss(fpn_feat, batch_bboxes, batch_labels)

        return loss












# for test only
if __name__ == '__main__':
    backbone_name = 'resnet50.a1_in1k'
    img_size = [640, 640]
    num_classes = 15
    loadckpt = False
    tta_img_size = [[640,640], [832,832], [960,960]]
    backbone = dict(
         modelType = backbone_name, 
         loadckpt = False, 
         pretrain = True, 
         froze = True,
    )
    head = dict(
        num_classes = num_classes,
        in_channel = 256,
    )

    model = Model(backbone_name, img_size, num_classes, loadckpt, tta_img_size, backbone, head)
    
    x = torch.rand((8, 3, 640, 640))
    cls_logits, cnt_logits, reg_preds = model(x)
    for cls, cnt, reg in zip(cls_logits, cnt_logits, reg_preds):
        print(cls.shape, cnt.shape, reg.shape)

    # torch.Size([8, 15, 80, 80]) torch.Size([8, 1, 80, 80]) torch.Size([8, 4, 80, 80])
    # torch.Size([8, 15, 40, 40]) torch.Size([8, 1, 40, 40]) torch.Size([8, 4, 40, 40])
    # torch.Size([8, 15, 20, 20]) torch.Size([8, 1, 20, 20]) torch.Size([8, 4, 20, 20])
    # torch.Size([8, 15, 10, 10]) torch.Size([8, 1, 10, 10]) torch.Size([8, 4, 10, 10])
    # torch.Size([8, 15, 5, 5]) torch.Size([8, 1, 5, 5]) torch.Size([8, 4, 5, 5])
