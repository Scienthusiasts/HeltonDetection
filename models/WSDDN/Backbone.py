import torch
import torch.nn as nn
import timm
from torchvision import models



class Backbone(nn.Module):
    '''Backbone
    '''
    def __init__(self, modelType:str, loadckpt=False, pretrain=True, froze=True):
        '''网络初始化

        Args:
            :param modelType: 使用哪个模型(timm库里的模型)
            :param loadckpt:  是否导入模型权重(是则输入权重路径)
            :param pretrain:  是否用预训练模型进行初始化
            :param froze:     是否冻结Backbone

        Returns:
            None
        '''
        super(Backbone, self).__init__()
        # 模型接到线性层的维度

        # 加载模型(features_only=True 只加载backbone部分)
        # features_only=True 只提取特征图(5层)，不加载分类头
        # out_indices=[3] 只提取到backbone的倒数第二层特征, 第一层留给head提取
        out_indices = [1,2,3,4] if modelType not in ['darknetaa53.c2ns_in1k', 'cspdarknet53.ra_in1k'] else [2,3,4,5]
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




# for test only
if __name__ == '__main__':
    # mobilenetv3_large_100.ra_in1k  resnet50.a1_in1k  darknetaa53.c2ns_in1k cspdarknet53.ra_in1k cspresnext50.ra_in1k
    # backbone:[bs, 3, 600, 600] -> [bs, 1024, 38, 38]
    backbone = Backbone(modelType='resnet50.a1_in1k', pretrain=False)
    # 验证 2
    # print(backbone)
    x = torch.rand((4, 3, 800, 800))
    outs = backbone(x)
    for out in outs:
        print(out.shape)
