import torch.nn as nn
from PIL import Image
from collections import Counter
import time

from models.YOLOv8.Backbone import *
from models.YOLOv8.PAFPN import *
from models.YOLOv8.PAFPN import *
from models.YOLOv8.Head import *
from utils.YOLOv8AnchorUtils import *
from utils.YOLOv8AnchorUtils import *





class Model(nn.Module):
    '''完整YOLOv5网络架构
    '''

    def __init__(self, backbone_name, img_size, num_classes, phi, loadckpt, backbone:dict):
        super(Model, self).__init__()
        '''基本配置'''
        depth_dict          = {'n' : 0.33, 's' : 0.33, 'm' : 0.67, 'l' : 1.00, 'x' : 1.00,}
        width_dict          = {'n' : 0.25, 's' : 0.50, 'm' : 0.75, 'l' : 1.00, 'x' : 1.25,}
        deep_width_dict     = {'n' : 1.00, 's' : 1.00, 'm' : 0.75, 'l' : 0.50, 'x' : 0.50,}
        dep_mul, wid_mul, deep_mul = depth_dict[phi], width_dict[phi], deep_width_dict[phi]
        base_channels       = int(wid_mul * 64)
        base_depth          = max(round(dep_mul * 3), 1)
        # 类别数
        self.img_size = img_size
        
        '''网络基本组件'''
        self.backbone   = CSPDarknet(phi, base_channels, base_depth, deep_mul, **backbone)
        # self.backbone = Backbone(**backbone)
        self.fpn = YOLOv8PAFPN(base_channels, base_depth, deep_mul)
        self.head = YOLOv8Head(num_classes, base_channels, deep_mul)

        '''导入预训练权重'''
        if loadckpt!=False: 
            # self.load_state_dict(torch.load(loadckpt))
            # 基于尺寸的匹配方式(名称不一致也能加载)
            self = loadWeightsBySizeMatching(self, loadckpt)


    def forward(self, x):
        # backbone提取基本特征
        backbone_feat = self.backbone(x)
        # fpn进一步提取特征
        p3, p4, p5 = self.fpn(backbone_feat)
        # 分类+回归
        dbox, cls, x, anchors, stride = self.head([p3, p4, p5])
        return dbox, cls, x, anchors, stride



    def fuse(self):
        print('Fusing layers... ')
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        return self





    def batchLoss(self, device, img_size, batch_datas):

        '''数据预处理'''
        batch_imgs, batch_bboxes = batch_datas[0], batch_datas[1]
        batch_imgs = batch_imgs.to(device)
        '''特征提取'''
        # backbone提取基本特征
        backbone_feat = self.backbone(batch_imgs)
        # fpn进一步提取特征
        p3, p4, p5 = self.fpn(backbone_feat)
        '''head部分计算损失'''
        loss = self.head.batchLoss([p3, p4, p5], batch_bboxes)
        return loss









    def infer(self, image:np.array, img_size, tf, device, T, image2color=None, agnostic=False, vis_heatmap=False, save_vis_path=None, half=False):
        '''推理一张图/一帧
            # Args:
                - image:  读取的图像(nparray格式)
                - tf:     数据预处理(基于albumentation库)
                - device: cpu/cuda
                - T:      可视化的IoU阈值
            # Returns:
                - boxes:       网络回归的box坐标    [obj_nums, 4]
                - box_scores:  网络预测的box置信度  [obj_nums]
                - box_classes: 网络预测的box类别    [obj_nums]
        '''
        H, W = np.array(np.shape(image)[0:2])
        # tensor_img有padding的黑边
        # 注意permute(2,0,1) 不要写成permute(2,1,0)
        tensor_img = torch.tensor(tf.testTF(image=image)['image']).permute(2,0,1).unsqueeze(0).to(device)
        if half: tensor_img = tensor_img.half()
        with torch.no_grad():
            #   将图像输入网络当中进行预测！
            outputs_list = self.forward(tensor_img)
            outputs = decode_box(outputs_list, img_size)
            '''计算nms, 并将box坐标从归一化坐标转换为绝对坐标'''
            # torch.cat(decode_predicts, 1) : torch.Size([1, 25200, 85])
            decode_predicts = non_max_suppression(outputs, self.img_size, conf_thres=T, nms_thres=0.3, agnostic=agnostic)

            # 图像里没预测出目标的情况:
            if len(decode_predicts) == 0 : return [],[],[]
            box_classes = np.array(decode_predicts[0][:, 5], dtype = 'int32')
            box_scores = decode_predicts[0][:, 4]
            # xyxy
            boxes = decode_predicts[0][:, :4]
            '''box坐标映射(有灰边图像里的坐标->原图的坐标)'''
            # W, H 原始图像的大小
            H, W = image.shape[:2]
            max_len = max(W, H)
            # w, h 缩放后的图像的大小
            w = int(W * img_size[0] / max_len)
            h = int(H * img_size[1] / max_len)
            # 将box坐标(对应有黑边的图)映射回无黑边的原始图像
            boxes = mapBox2OriginalImg(boxes, W, H, [w, h], padding=True)
            '''是否可视化obj heatmap'''
            if vis_heatmap:vis_YOLOv8_heatmap(outputs_list[2], [W, H], img_size, image, box_classes, image2color, save_vis_path=save_vis_path)
            return boxes, box_scores, box_classes







    def inferenceSingleImg(self, device, class_names, image2color, img_size, tf, img_path, save_vis_path=None, ckpt_path=None, T=0.3, agnostic=False, show_text=True, vis_heatmap=False, half=False):
        '''推理一张图
            # Args:
                - device:        cpu/cuda
                - class_names:   每个类别的名称, list
                - image2color:   每个类别一个颜色
                - img_size:      固定图像大小 如[832, 832]
                - tf:            数据预处理(基于albumentation库)
                - img_path:      图像路径
                - save_vis_path: 可视化图像保存路径
                - ckpt_path:     模型权重路径
                - T:             可视化的IoU阈值

            # Returns:
                - boxes:       网络回归的box坐标    [obj_nums, 4]
                - box_scores:  网络预测的box置信度  [obj_nums]
                - box_classes: 网络预测的box类别    [obj_nums]
        '''
        if ckpt_path != None:
            self.load_state_dict(torch.load(ckpt_path))
            self.eval()

        image = Image.open(img_path).convert('RGB')
        # Image 转numpy
        image = np.array(image)
        '''推理一张图像'''
        # xyxy
        boxes, box_scores, box_classes = self.infer(np.array(image), img_size, tf, device, T, agnostic, vis_heatmap, save_vis_path, half=half)
        #  检测出物体才继续    
        if len(boxes) == 0: 
            print(f'no objects in image: {img_path}.')
            return boxes, box_scores, box_classes

        '''画框'''
        if save_vis_path!=None:
            # PltDrawBox(image, boxes, box_classes, box_scores, save_vis_path, image2color, class_names)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = OpenCVDrawBox(image, boxes, box_classes, box_scores, save_vis_path, image2color, class_names, resize_size=[2000, 2000], show_text=show_text)
            cv2.imwrite(save_vis_path, image)
            # 统计检测出的类别和数量
            detect_cls = dict(Counter(box_classes))
            detect_name = {}
            for key, val in detect_cls.items():
                detect_name[class_names[key]] = val
            print(f'detect result: {detect_name}')
        return boxes, box_scores, box_classes
























# for test only
if __name__ == '__main__':
    phi='s'
    num_classes = 80
    backbone_name = None
    backbone = dict(
        loadckpt = 'ckpt/yolov8_s_backbone_weights.pth',
        pretrain = False,
        froze = True
    )

    model = Model(backbone_name, num_classes, phi, loadckpt=False, backbone=backbone)
    # torch.save(model.state_dict(), f"{phi}.pt")
    # 验证 1
    # summary(model, input_size=[(3, 224, 224)])  
    # 验证 2
    x = torch.rand((8, 3, 640, 640))
    dbox, cls, x, anchors, strides = model(x)
    print(dbox.shape)
    print(cls.shape)
    print('============================')
    for i in x:
        print(i.shape)
    print('============================')
    print(anchors.shape)
    print(strides.shape)
