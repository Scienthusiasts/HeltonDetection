import torch.nn as nn
from PIL import Image
import time
from collections import Counter

from models.FasterRCNN.Backbone import *
from models.FasterRCNN.PAFPN import *
from models.FasterRCNN.FPN import *
from models.FasterRCNN.Head import *
from models.FasterRCNN.DecoupledHead import *
from models.FasterRCNN.RPN import *



class Model(nn.Module):
    '''完整FasterRCNN网络架构
    '''

    def __init__(self,  catNums, backbone_name, loadckpt, backbone:dict, rpn:dict, head:dict):
        super(Model, self).__init__()
        self.cat_nums = catNums

        # 对应backbone每一层的输出(C5, C4, C3, C2)
        model_param = {
            'resnet50.a1_in1k':              [2048, 1024, 512, 256],
            'cspresnext50.ra_in1k':          [2048, 1024, 512, 256],
            'mobilenetv3_large_100.ra_in1k': [960,  112,  40,  24 ],
            'darknetaa53.c2ns_in1k':         [1024, 512,  256, 128],
            'cspdarknet53.ra_in1k':          [1024, 512,  256, 128],
        }[backbone_name]
        # Backbone最好使用原来的预训练权重初始化
        self.backbone = Backbone(**backbone)
        # FPN提取多尺度特征 (这里原本特征图尺寸越小通道越多, 这里调整成通道数都是256)
        self.fpn = PAFPN(model_param[0], model_param[1], model_param[2], P5_channel=256, P4_channel=256)
        # self.fpn = FPN(model_param[0], model_param[1], model_param[2], model_param[3])
        # RPN网络(加了FPN, 都默认inchannel和midchannel为256)
        self.rpn = RPN(**rpn)
        # Head检测头
        self.head = DecoupledHead(**head)
        # self.head = Head(**head)

        # 是否导入预训练权重
        if loadckpt!=None: 
            # self.load_state_dict(torch.load(loadckpt))
            # print('yolov5 pretrain ckpt loaded!')

            # 加快模型训练的效率
            print('Loading weights into state dict by size matching...')
            model_dict = self.state_dict()
            pretrained_dict = torch.load(loadckpt)
            a = {}
            for (kk, vv), (k, v) in zip(pretrained_dict.items(), model_dict.items()):
                try:    
                    if np.shape(vv) ==  np.shape(v):
                        # print(f'(previous){kk} -> (current){k}')
                        a[k]=vv
                except:
                    print(f'(previous){kk} mismatch (current){k}')
            model_dict.update(a)
            self.load_state_dict(model_dict)
            print('Finished!')
            




    def forward(self, x, mode="forward"):
        # 计算输入图片的大小(W, H)
        imgSize = x.shape[2:]
        '''提取基本特征'''
        # baseFeat [[bs, 512, p3_w, p3_h] [bs, 1024, p4_w, p4_h] [bs, 2048, p5_w, p5_h]]
        baseFeat = self.backbone(x)
        # fpnFeat [[bs, 256, p3_w, p3_h] [bs, 256, p4_w, p4_h] [bs, 256, p5_w, p5_h]]
        fpnFeat = self.fpn(baseFeat)     
        if mode == 'extractor': return fpnFeat
        '''获得RPN预测结果'''
        # rpn获得建议框rois(和建议框在batch哪张图像上的索引roi_indices)
        # rois:[[bs, 100, 4], [bs, 100, 4], [bs, 100, 4]] roi_indices:[[bs, 100], [bs, 100], [bs, 100]] train:200, val/test:100
        _, _, rois, _, roi_indices = self.rpn(fpnFeat, imgSize)
        # [[bs, 100, 4], [bs, 100, 4], [bs, 100, 4]] -> [bs, 3*100, 4]
        rois = torch.cat(rois, dim=1)  
        # [[bs, 100], [bs, 100], [bs, 100]] -> [bs, 3*100]
        roi_indices = torch.cat(roi_indices, dim=1)   
        '''获得head预测结果'''
        # head获得分类结果和回归结果
        # RoIReg:[bs, 300, (cls_num+1)*4] RoICls:[bs, 300, cls_num+1]
        RoIReg, RoICls = self.head(fpnFeat, rois, roi_indices, imgSize)
        # 网络预测的offset, 网络预测的类别, RPN网络预测的proposal的四个坐标,网络预测的proposal属于哪个batch的索引(未解码)
        return RoIReg, RoICls, rois, roi_indices







    def batchLoss(self, device, img_size, batch_datas):
        '''一个batch的前向流程(不包括反向传播, 更新梯度)(核心, 要更改训练pipeline主要改这里)

        Args:
            - img_size:     固定图像大小 如[832, 832]
            - batch_imgs:   一个batch里的图像              例:shape=[bs, 3, 600, 600]
            - batch_bboxes: 一个batch里的GT框              例:[(1, 4), (4, 4), (4, 4), (1, 4), (5, 4), (2, 4), (3, 4), (1, 4)]
            - batch_labels: 一个batch里的GT框类别          例:[(1,), (4,), (4,), (1,), (5,), (2,), (3,), (1,)]

        Returns:
            - losses: 所有损失组成的列表(里面必须有一个total_loss字段, 用于反向传播)
        '''
        # 固定的三个要素: 图像, GTBox, GTBox的类别标签
        batch_imgs, batch_bboxes, batch_labels, y_trues = batch_datas[0].to(device), batch_datas[1], batch_datas[2], batch_datas[3]
        y_trues = [y_true.to(device) for y_true in y_trues]
        # 获取当前batchsize大小(不直接用self.bs是因为最后的batchsize不一定刚好=self.bs)
        bs = batch_imgs.shape[0]
        '''获取backbone特征图'''
        fpn_feature = self.forward(batch_imgs, mode = 'extractor')
        '''rpn部分计算损失'''
        # rois是筛选后参与head训练微调的正负样本proposal      shape=[8, 600, 4] 
        # origin_rois是所有proposals的回归坐标(原图尺度)      shape=[8, 52*52*9, 4] 
        # roi_indices是proposal属于batch哪张图像的索引        shape=[8, 600]
        rpn_loss, rois, origin_rois, roi_indices = self.rpn.batchLoss(fpn_feature, y_trues)
        '''head部分计算损失'''
        head_loss = self.head.batchLoss(img_size, bs, fpn_feature, batch_bboxes, batch_labels, rois, origin_rois, roi_indices)
        '''统一损失的格式:字典形式: {损失1, 损失2,..., 总损失}'''
        loss = dict()
        loss.update(rpn_loss)
        loss.update(head_loss)
        loss.update(dict(total_loss = sum(loss.values())))
        return loss





    def infer(self, image:np.array, img_size, tf, device, T, agnostic):
        '''推理一张图/一帧
            Args:
                - image:  读取的图像(nparray格式)
                - tf:     数据预处理(基于albumentation库)
                - device: cpu/cuda
                - T:      可视化的IoU阈值
            Returns:
                - boxes:       网络回归的box坐标    [obj_nums, 4]
                - box_scores:  网络预测的box置信度  [obj_nums]
                - box_classes: 网络预测的box类别    [obj_nums]
        '''
        # tensor_img有padding的黑边
        # 注意permute(2,0,1) 不要写成permute(2,1,0)
        tensor_img = torch.tensor(tf.testTF(image=image)['image']).permute(2,0,1).unsqueeze(0).to(device)
        with torch.no_grad():
            '''网络推理得到最原始的未解码未nms的结果'''
            # roi_cls_locs: Head对RPN的proposal进行回归得到的微调offset [1, 300, 21*4]
            # roi_scores:   Head对RPN的proposal进行分类得到的分类结果   [1, 300, 21]
            # rois:         RPN预测的proposals的集合                   [1, 300, 4]
            roi_cls_locs, roi_scores, rois, _ = self.forward(tensor_img)
            # print(roi_cls_locs)
            '''利用Head的预测结果对RPN proposals进行微调+解码+nms筛选, 获得预测框'''
            bboxes, bbox_scores = decode(self.cat_nums, roi_cls_locs[0], roi_scores[0], rois[0])
            '''逐类别计算nms'''
            if agnostic:
                boxes, box_scores, box_classes = NMSAll(bboxes, bbox_scores, nms_iou=0.3, T=T)
            else:
                boxes, box_scores, box_classes = NMSByCls(self.cat_nums, bboxes, bbox_scores, nms_iou=0.3, T=T)
            #  检测出物体才继续    
            if len(boxes) == 0: 
                return boxes, box_scores, box_classes
            '''box坐标映射'''
            # W, H 原始图像的大小
            H, W = image.shape[:2]
            max_len = max(W, H)
            # w, h 缩放后的图像的大小
            w = int(W * img_size[0] / max_len)
            h = int(H * img_size[1] / max_len)
            # 将box坐标(对应有黑边的图)映射回无黑边的原始图像
            boxes = mapBox2OriginalImg(boxes, W, H, [w, h], padding=True)
        return boxes, box_scores, box_classes






    def inferenceSingleImg(self, device, class_names, image2color, img_size, tf, img_path, save_vis_path=None, ckpt_path=None, T=0.3, agnostic=False, show_text=True, 
                           vis_heatmap=False, half=False):
        '''推理一张图
            Args:
                - device:        cpu/cuda
                - class_names:   每个类别的名称, list
                - image2color:   每个类别一个颜色
                - img_size:      固定图像大小 如[832, 832]
                - tf:            数据预处理(基于albumentation库)
                - img_path:      图像路径
                - save_vis_path: 可视化图像保存路径
                - ckpt_path:     模型权重路径
                - T:             可视化的IoU阈值

            Returns:
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
        boxes, box_scores, box_classes = self.infer(image, img_size, tf, device, T, agnostic)
        #  检测出物体才继续    
        if len(boxes) == 0: 
            print(f'no objects in the images: {img_path}.')
            return boxes, box_scores, box_classes
        '''画框'''
        if save_vis_path!=None:
            # PltDrawBox(image, boxes, box_classes, box_scores, save_vis_path, image2color, class_names)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = OpenCVDrawBox(image, boxes, box_classes, box_scores, save_vis_path, image2color, class_names, resize_size=[2000, 2000], show_text=show_text)
            cv2.imwrite(save_vis_path, image)
        return boxes, box_scores, box_classes









    def inferenceVideo(self, device, class_names, image2color, img_size, tf,  video_path, save_vis_path=None, ckpt_path=None, T=0.3, agnostic=False, show_text=True, 
                       vis_heatmap=False, half=False):
        '''推理一段视频
            Args:
                - device:        cpu/cuda
                - class_names:   每个类别的名称, list
                - image2color:   每个类别一个颜色
                - img_size:      固定图像大小 如[832, 832]
                - tf:            数据预处理(基于albumentation库)
                - video_pat:     视频路径
                - save_vis_path: 可视化图像保存路径
                - ckpt_path:     模型权重路径
                - T:             可视化的IoU阈值

            Returns:
                - boxes:       网络回归的box坐标    [obj_nums, 4]
                - box_scores:  网络预测的box置信度  [obj_nums]
                - box_classes: 网络预测的box类别    [obj_nums]
        '''
        if ckpt_path != None:
            self.load_state_dict(torch.load(ckpt_path))
            self.eval()

        # 创建视频捕获对象
        cap = cv2.VideoCapture(video_path)
        # 获取视频的基本信息
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # 定义视频编码器和创建 VideoWriter 对象
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 或者 'XVID' 'mp4v'
        out = cv2.VideoWriter(save_vis_path, fourcc, fps, (width, height))
        # 检查视频是否正确打开
        if not cap.isOpened():
            print("Error: Could not open video.")
            exit()

        # 逐帧读取视频
        cnt_frame =  1
        start_time = time.time()
        while True:
            ret, frame = cap.read()  # ret是一个布尔值，frame是每一帧的图像
            if not ret:
                print("Reached end of video or failed to read frame.")
                break

            '''此处为推理'''
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            boxes, box_scores, box_classes = self.infer(frame, img_size, tf, device, T, agnostic)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            #  检测出物体才继续    
            if len(boxes) != 0: 
                '''画框'''
                frame = OpenCVDrawBox(frame, boxes, box_classes, box_scores, None, image2color, class_names, resize_size=[2000, 2000], show_text=show_text)
                # 统计检测出的类别和数量
                detect_cls = dict(Counter(box_classes))
                detect_name = {}
                for key, val in detect_cls.items():
                    detect_name[class_names[key]] = val
            # 写入处理后的帧到新视频
            out.write(frame)
            print(f'process frame {frame.shape}: [{cnt_frame}/{total_frames}] | {detect_name}')
            cnt_frame += 1
        end_time = time.time()
        print(f"total_time: {end_time - start_time}(s) | fps: {cnt_frame / (end_time - start_time)}")
        # 释放视频捕获对象和视频写入对象，销毁所有OpenCV窗口
        cap.release()
        out.release()
        cv2.destroyAllWindows()













# for test only
if __name__ == '__main__':
    # 'resnet50.a1_in1k'  'mobilenetv3_large_100.ra_in1k' 
    model = Model(catNums=20, backbone_name='resnet50.a1_in1k', loadckpt='ckpt/best_COCO_fpnp2_frcnn.pt')
    # torch.save(model.state_dict(), "frcnn.pt")
    # 验证 1
    # summary(model, input_size=[(3, 224, 224)])  
    # 验证 2
    # print(model)
    x = torch.rand((8, 3, 832, 832))
    RoIReg, RoICls, RoIs, RoIIndices = model(x, mode="forward")
    print(RoIReg.shape)      # torch.Size([4, 600, 320])
    print(RoICls.shape)      # torch.Size([4, 600, 80])
    print(RoIs.shape)        # torch.Size([4, 600, 4])
    print(RoIIndices.shape)  # torch.Size([4, 600])