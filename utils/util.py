import os                                    # 读取文件目录
import numpy as np
from tabulate import tabulate
import torch
import cv2
import argparse
import importlib
import json
import time
from collections import Counter
from PIL import Image
import matplotlib.pyplot as plt
import random
from torch import nn
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from matplotlib.patches import Polygon, Rectangle
from ensemble_boxes import weighted_boxes_fusion as wbf

from datasets.preprocess import Transform


def sigmoid(x):
    y = 1.0 / (1.0 + np.exp(-x))
    return y


# 动态导入类
def dynamic_import_class(module_path, class_name='module_name', get_class=True):
    spec = importlib.util.spec_from_file_location(class_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if get_class:
        return getattr(module, class_name)
    else:
        return module





def seed_everything(seed):
    '''设置全局种子
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False






def loadWeightsBySizeMatching(model:nn.Module, ckpt_path:str):
    print('Loading weights into state dict by size matching...')
    model_dict = model.state_dict()
    pretrained_dict = torch.load(ckpt_path)
    a = {}
    for (kk, vv), (k, v) in zip(pretrained_dict.items(), model_dict.items()):
        try:    
            if np.shape(vv) ==  np.shape(v):
                # print(f'(previous){kk} -> (current){k}')
                a[k]=vv
        except:
            print(f'(previous){kk} mismatch (current){k}')
    model_dict.update(a)
    model.load_state_dict(model_dict)
    print('Finished!')

    return model







def init_weights(model, init_type, mean=0, std=0.01):
    '''权重初始化方法
    '''
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if init_type=='he':
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if init_type=='normal':
                nn.init.normal_(module.weight, mean=mean, std=std)  # 使用高斯随机初始化
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)




def CV2DrawBox(image, boxes, classes, scores, save_res_path, colors, class_names):
    '''OpenCV画框
        Args:
            :param image:         原始图像(Image格式)
            :param boxes:         网络预测的box坐标
            :param classes:       网络预测的box类别
            :param scores:        网络预测的box置信度
            :param save_res_path: 可视化结果保存路径

        Returns:
            None
    '''
    # Image转opencv
    img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    for box, cls, score in zip(boxes, classes, scores):
        x0, y0, x1, y1 = round(box[0]), round(box[1]), round(box[2]), round(box[3])
        cv2.rectangle(img, (x0, y0), (x1, y1), colors[cls], thickness=2)
        text = '{} {:.2f}'.format(class_names[cls], score)
        cv2.putText(img, text, (x0, y0), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=colors[cls], thickness=2)
    cv2.imwrite(save_res_path, img)





def PltDrawBox(image, boxes, classes, scores, save_res_path, image2color, class_names):
    '''plt画框
        Args:
            :param image:         原始图像(Image格式)
            :param boxes:         网络预测的box坐标
            :param classes:       网络预测的box类别
            :param scores:        网络预测的box置信度
            :param save_res_path: 可视化结果保存路径

        Returns:
            None
    '''
    plt.figure()
    ax = plt.gca()
    for box, cls, score in zip(boxes, classes, scores):
        x0, y0, x1, y1 = round(box[0]), round(box[1]), round(box[2]), round(box[3])
        color = image2color[class_names[cls]]
        text = '{} {:.2f}'.format(class_names[cls], score)
        # 多边形填充+矩形边界:
        ax.add_patch(Polygon(xy=[[x0, y0], [x0, y1], [x1, y1], [x1, y0]], color=color, alpha=0.3))
        ax.add_patch(Rectangle(xy=(x0, y0), width=x1-x0, height=y1-y0, fill=False, color=color, alpha=1))
        # 可视化每个bbox的类别的文本(ax.text的bbox参数用于调整文本框的样式):
        ax.text(x0, y0, text, color='white', fontsize=5, bbox=dict(facecolor=color, alpha=0.4))
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    # plt.show()
    plt.savefig(save_res_path, bbox_inches='tight', pad_inches=0.0, dpi=200)




def OpenCVDrawBox(image, boxes, classes, scores, save_vis_path, image2color, class_names, resize_size, show_text=True):
    '''plt画框
        Args:
            :param image:         原始图像(Image格式)
            :param boxes:         网络预测的box坐标
            :param classes:       网络预测的box类别
            :param scores:        网络预测的box置信度
            :param save_res_path: 可视化结果保存路径

        Returns:
            None
    '''
    H, W = image.shape[:2]
    
    max_len = max(W, H)
    w = int(W * resize_size[0] / max_len)
    h = int(H * resize_size[1] / max_len)
    boxes = mapBox2OriginalImg(boxes, w, h, [W, H], padding=False)

    image = cv2.resize(image, (w, h))
    # 框的粗细
    thickness = max(1, int(image.shape[0] * 0.003))
    for box, cls, score in zip(boxes, classes, scores):
        # if class_names[cls] not in  ['pedestrian', 'people', 'person']:continue
        x0, y0, x1, y1 = round(box[0]), round(box[1]), round(box[2]), round(box[3])
        color = np.array(image2color[class_names[cls]])
        color = tuple([int(c*255) for c in color])
        # color = (0,0,255)
        # text = 'target_1'
        # text = '{} {:.2f}'.format(text, score)
        text = '{} {:.2f}'.format(class_names[cls], score)
        # obj的框
        cv2.rectangle(image, (x0, y0), (x1, y1), color, thickness=thickness)
        # 文本的框
        if show_text:
            cv2.rectangle(image, (x0-1, y0-30), (x0+len(text)*12, y0), color, thickness=-1)
            # 文本
            cv2.putText(image, text, (x0, y0-6), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(255,255,255), thickness=2)
    # 保存
    if save_vis_path!=None:
        return image
    image = cv2.resize(image, (W, H))
    return image

        




def mapBox2OriginalImg(boxes, w, h, imgsize:list, padding=True):
    '''将box坐标(对应有黑边的图)映射回无黑边的原始图像
        Args:
            :param boxes:          网络预测的box的坐标(对应有黑边的图的尺寸)
            :param w:              原始图像的宽
            :param h:              原始图像的高
            :param imgsize:        resize后图像的大小[w, h]


        Returns:
            解码+NMS后保留的proposal
    '''
    # 黑边填充
    if padding==True:
        resize_max_len = max(imgsize[0], imgsize[1])
        if w>h:
            resize_w = resize_max_len
            resize_h = h * (resize_w / w)
            padding_len = abs(resize_max_len - resize_h) / 2
            boxes[:, [1,3]] -= padding_len
        else:
            resize_h = resize_max_len
            resize_w = w * (resize_h / h)        
            padding_len = abs(resize_max_len - resize_w) / 2
            boxes[:, [0,2]] -= padding_len
        boxes[:, [1,3]] *= (h / resize_h)
        boxes[:, [0,2]] *= (w / resize_w)
    # 按比例缩放，无黑边填充
    else:
        w_ratio = w / imgsize[0]
        h_ratio = h / imgsize[1]
        boxes[:, [1,3]] *= h_ratio
        boxes[:, [0,2]] *= w_ratio

    return boxes






def visArgsHistory(json_dir, save_dir, loss_sample_interval=50):
    '''可视化训练过程中保存的参数

    Args:
        :param json_dir: 参数的json文件路径
        :param logDir:   可视化json文件保存路径

    Returns:
        None
    '''
    if not os.path.isdir(save_dir):os.makedirs(save_dir)
    json_path = os.path.join(json_dir, 'args_history.json')
    with open(json_path) as json_file:
        args = json.load(json_file)

        for args_key in args.keys():
            arg = args[args_key]
            # loss太多了则等间隔采样
            if args_key.split('_')[-1]=='loss':
                arg = arg[::loss_sample_interval]

            # 绘制
            plt.plot(arg, linewidth=1)
            if args_key.split('_')[-1]=='loss':
                plt.xlabel(f'iter with interval {loss_sample_interval}')
            else:
                plt.xlabel('Epoch')
            plt.ylabel(args_key)
            plt.savefig(os.path.join(save_dir, f"{args_key.replace(':', '.')}.png"), dpi=200)
            plt.clf()













def inferenceSingleImg(model, device, class_names, image2color, img_size, tf, img_path, save_vis_path=None, ckpt_path=None, T=0.3, agnostic=False, show_text=True, vis_heatmap=False, half=False, tta=False):
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
        model.load_state_dict(torch.load(ckpt_path))
        model.eval()

    image = Image.open(img_path).convert('RGB')
    # Image 转numpy
    image = np.array(image)
    '''推理一张图像'''
    if tta:
        boxes, box_scores, box_classes = model.tta.infer(model, image, device, T, image2color, agnostic, vis_heatmap, save_vis_path, half=half)
    else:
        # xyxy
        boxes, box_scores, box_classes = model.infer(image, img_size, tf, device, T, image2color, agnostic, vis_heatmap, save_vis_path, half=half)
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










def inferenceVideo(model, device, class_names, image2color, img_size, tf, video_path, save_vis_path=None, ckpt_path=None, T=0.3, agnostic=False, show_text=True, half=False):
    '''推理一段视频
        # Args:
            - device:        cpu/cuda
            - class_names:   每个类别的名称, list
            - image2color:   每个类别一个颜色
            - img_size:      固定图像大小 如[832, 832]
            - tf:            数据预处理(基于albumentation库)
            - video_pat:     视频路径
            - save_vis_path: 可视化图像保存路径
            - ckpt_path:     模型权重路径
            - T:             可视化的IoU阈值

        # Returns:
            - boxes:       网络回归的box坐标    [obj_nums, 4]
            - box_scores:  网络预测的box置信度  [obj_nums]
            - box_classes: 网络预测的box类别    [obj_nums]
    '''
    if ckpt_path != None:
        model.load_state_dict(torch.load(ckpt_path))
        # model = loadWeightsBySizeMatching(model, ckpt_path)
        model.eval()
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
        infer_s = time.time()
        boxes, box_scores, box_classes = model.infer(frame, model.img_size, tf, device, T, agnostic, half=half)
        infer_e = time.time()
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
        print(f'process frame {frame.shape}: [{cnt_frame}/{total_frames}] | time(ms): {round(infer_e-infer_s, 3)} | {detect_name}')
        cnt_frame += 1
    end_time = time.time()
    print(f"total_time: {end_time - start_time}(s) | fps: {cnt_frame / (end_time - start_time)}")
    # 释放视频捕获对象和视频写入对象，销毁所有OpenCV窗口
    cap.release()
    out.release()
    # cv2.destroyAllWindows()










class TTA():
    '''Test Time Augmentation + WBF策略
    '''

    def __init__(self, tta_img_size=[[640,640], [832,832], [960,960]]):
        self.tta_img_size = tta_img_size
        self.tta_transform = [Transform(img_size, box_format='coco') for img_size in self.tta_img_size]        


    def infer(self,
               model, 
               image:np.array, 
               device, 
               T, 
               image2color, 
               agnostic=False, 
               vis_heatmap=False, 
               save_vis_path=None, 
               half=False, 
               tta=True
               ):
        flip_image = cv2.flip(image, 1)
        # n个尺度, 每个尺度2个增强, 一共2n张图
        tta_boxes, tta_box_scores, tta_box_classes = [], [], []
        # 每张图像分别推理:
        for i in range(len(self.tta_img_size)):
            for j in range(2):
                img = image if j==0 else flip_image
                boxes, box_scores, box_classes = model.infer(
                    img, 
                    self.tta_img_size[i], 
                    self.tta_transform[i], 
                    device, 
                    T, 
                    image2color, 
                    agnostic, 
                    False, 
                    save_vis_path, 
                    half=half, 
                    tta=tta
                    )
                if len(boxes) > 0:
                    # 归一化(wbf只接受归一化输入)
                    boxes[:, [0,2]] /= image.shape[1]
                    boxes[:, [1,3]] /= image.shape[0]
                    # 水平翻转的图像翻转回来
                    if j == 1: 
                        boxes[:, [0, 2]] = 1 - boxes[:, [2, 0]]
                tta_boxes.append(boxes)
                tta_box_scores.append(box_scores)
                tta_box_classes.append(box_classes)
        '''不同增强下的预测结果执行WBF'''
        tta_boxes, tta_box_scores, tta_box_classes = wbf(
            tta_boxes, 
            tta_box_scores, 
            tta_box_classes, 
            weights=[1]*2*len(self.tta_img_size), 
            iou_thr=0.55, 
            skip_box_thr=T
            )
        if len(tta_boxes) == 0 : return [],[],[]
        # 坐标再映射回原图大小
        tta_boxes[:, [0,2]] *= image.shape[1]
        tta_boxes[:, [1,3]] *= image.shape[0]
        return tta_boxes, tta_box_scores, tta_box_classes.astype(int)
    










# for test only
if __name__ == '__main__':
    json_dir = 'log/fpn_p2_VOC'
    save_dir = './log/fpn_p2_VOC/plot'
    visArgsHistory(json_dir, save_dir, loss_sample_interval=50)
