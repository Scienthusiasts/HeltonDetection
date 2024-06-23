# coding=utf-8
import numpy as np
import torch
from functools import partial
from PIL import Image
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import random
import albumentations as A
from pycocotools.coco import COCO
import os
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.util import *
from utils.YOLOAnchorUtils import *




class Transform():
    '''数据预处理/数据增强(基于albumentations库)
       https://albumentations.ai/docs/api_reference/full_reference/
    '''
    def __init__(self, imgSize, box_format='coco'):
        '''
            - imgSize:    网络接受的输入图像尺寸
            - box_format: 'yolo':norm(cxcywh), 'coco':xywh
        '''
        maxSize = max(imgSize[0], imgSize[1])
        # 训练时增强
        self.trainTF = A.Compose([
                A.BBoxSafeRandomCrop(p=0.5),
                # A.RandomSizedBBoxSafeCrop(800, 800, erosion_rate=0.0, interpolation=1, p=0.5),
                # 随机翻转
                A.HorizontalFlip(p=0.5),
                # 下面这两个只能在DOTA上用：
                # A.VerticalFlip(p=0.5),
                # A.RandomRotate90(p=0.5),
                # 参数：随机色调、饱和度、值变化
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, always_apply=False, p=0.5),
                # 随机对比度增强
                A.CLAHE(p=0.1),
                # 高斯噪声
                A.GaussNoise(var_limit=(0.05, 0.09), p=0.4),     
                # 随机转为灰度图
                A.ToGray(p=0.01),
                A.OneOf([
                    # 使用随机大小的内核将运动模糊应用于输入图像
                    A.MotionBlur(p=0.2),   
                    # 中值滤波
                    A.MedianBlur(blur_limit=3, p=0.1),    
                    # 使用随机大小的内核模糊输入图像
                    A.Blur(blur_limit=3, p=0.1),  
                ], p=0.2),
            ],
            bbox_params=A.BboxParams(format=box_format, min_area=0, min_visibility=0.0, label_fields=['category_ids']),
            )
        # 基本数据预处理
        self.normalTF = A.Compose([
                # 最长边限制为imgSize
                A.LongestMaxSize(max_size=maxSize),
                # 较短的边做padding
                A.PadIfNeeded(imgSize[0], imgSize[1], border_mode=cv2.BORDER_CONSTANT, value=[128,128,128]),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ],
            bbox_params=A.BboxParams(format=box_format, min_area=0, min_visibility=0.0, label_fields=['category_ids']),
            )
        # 测试时增强
        self.testTF = A.Compose([
                # 最长边限制为imgSize
                A.LongestMaxSize(max_size=maxSize),
                # 较短的边做padding
                A.PadIfNeeded(imgSize[0], imgSize[1], border_mode=cv2.BORDER_CONSTANT, value=[128,128,128]),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
        # 测试时增强(不padding黑边)
        self.testTFNoPad = A.Compose([
                # 最长边限制为imgSize
                A.LongestMaxSize(max_size=maxSize),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])








class BaseDataset(Dataset):

    def __init__(self, inputShape=[800, 800], trainMode=True):
        '''__init__() 为默认构造函数，传入数据集类别（训练或测试），以及数据集路径
        '''      
        self.mode = trainMode
        self.input_shape = inputShape
        self.tf = Transform(inputShape, 'coco')
        # 数据集大小
        self.datasetNum = 0


    def __len__(self):
        '''重载data.Dataset父类方法, 返回数据集大小
        '''
        return self.datasetNum
    

    def getDataByIndex(self, index):
        pass


    def augment(self, image, boxes, labels):
        '''所有数据增强+预处理操作(顺序不能乱!)
        '''   
        if (self.mode):
            # 基本的数据增强
            image, boxes, labels = self.trainAlbumAug(image, boxes, labels)
            # mosaic数据增强
            image, boxes, labels = self.yoloMosaic4(image, boxes, labels, p=0.5)
        # 数据预处理(pad成统一size, 归一化)
        image, boxes, labels = self.normalAlbumAug(image, boxes, labels)
        if (self.mode):
            # mixup数据增强
            image, boxes, labels = self.mixUp(image, boxes, labels, p=0.0)

        return image, boxes, labels

        
    def trainAlbumAug(self, image, boxes, labels):
        """基于albumentations库的训练时数据增强
        """
        # albumentation的图像维度得是[W,H,C]
        train_trans = self.tf.trainTF(image=image, bboxes=boxes, category_ids=labels)
        image, boxes, labels = train_trans['image'], train_trans['bboxes'], train_trans['category_ids']
        # 这里的box是coco格式(xywh)
        return image, boxes, labels
        


    def normalAlbumAug(self, image, boxes, labels):
        """基于albumentations库的基础数据预处理
        """
        normal_trans = self.tf.normalTF(image=image, bboxes=boxes, category_ids=labels)
        image, boxes, labels = normal_trans['image'], normal_trans['bboxes'], normal_trans['category_ids']
        # 这里的box是coco格式(xywh)
        return image, boxes, labels
    



    def mixUp(self, image1, boxes1, labels1, p=0.5):
        """mixUp数据增强: https://arxiv.org/pdf/1710.09412.pdf.
          (需要两张图像大小一致, 因此必须在基础数据预处理之后)
        """
        if (np.random.rand() < p):
            index2 = np.random.randint(self.datasetNum)
            image2, boxes2, labels2 = self.getDataByIndex(index2)
            image2, boxes2, labels2 = self.trainAlbumAug(image2, boxes2, labels2)
            image2, boxes2, labels2 = self.normalAlbumAug(image2, boxes2, labels2)
            # mixup 两张图像所占比例(分布在0.5附近)
            r = np.random.beta(32.0, 32.0)  
            mixup_image = (image1 * r + image2 * (1 - r))
            mixup_labels = labels1 + labels2
            mixup_boxes = boxes1 + boxes2
            return mixup_image, mixup_boxes, mixup_labels
        return image1, boxes1, labels1





    def yoloMosaic4(self, image1, boxes1, labels1, jitter=0.2, scale=.5, p=0.5):
        """mosaic数据增强, 将四张图像拼在一起
        """
        if (np.random.rand() < p):
            # 随机选取其他3张图像的索引
            indexs = np.random.randint(self.datasetNum, size=3)
            # 读取其余3张图像, 对图像进行数据增强
            image2, boxes2, labels2 = self.getDataByIndex(indexs[0])
            image3, boxes3, labels3 = self.getDataByIndex(indexs[1])
            image4, boxes4, labels4 = self.getDataByIndex(indexs[2])
            image2, boxes2, labels2 = self.trainAlbumAug(image2, boxes2, labels2)
            image3, boxes3, labels3 = self.trainAlbumAug(image3, boxes3, labels3)
            image4, boxes4, labels4 = self.trainAlbumAug(image4, boxes4, labels4)
            W, H = self.input_shape
            # 放置图像的中心位置
            cx = int(random.uniform(0.3, 0.7) * W)
            cy = int(random.uniform(0.3, 0.7) * H)
            images = [image1, image2, image3, image4]
            bboxes = [boxes1, boxes2, boxes3, boxes4]
            labels = [labels1, labels2, labels3, labels4]
            mosaic_img = np.ones((W, H, 3), dtype=np.uint8) * 128
            for i in range(4):
                bboxes[i] = np.array(bboxes[i])
                labels[i] = np.array(labels[i])
                w, h, _ = images[i].shape
                # 对图像进行缩放并且进行长和宽的扭曲
                scale = random.uniform(scale, 1)
                scale_w = random.uniform(1-jitter,1+jitter) * scale
                scale_h = random.uniform(1-jitter,1+jitter) * scale
                new_w, new_h = int(w * scale_w), int(h * scale_h)
                # 对图像进行缩放
                images[i] = cv2.resize(images[i], (new_h, new_w))
                # 对box进行缩放
                bboxes[i][:, [0,2]] *= scale_h
                bboxes[i][:, [1,3]] *= scale_w
                # 图像mosaic到一张图像上:
                if i==0: 
                    mosaic_img[max(cx-new_w, 0):cx, max(cy-new_h, 0):cy, :] = images[i][max(0, new_w-cx):, max(0, new_h-cy):, :]
                    # 对图像进行平移
                    bboxes[i][:,0] += (cy-new_h)
                    bboxes[i][:,1] += (cx-new_w)
                if i==1:
                    mosaic_img[cx:min(W, cx+new_w), max(cy-new_h, 0):cy, :] = images[i][:min(new_w, W-cx), max(0, new_h-cy):, :]
                    # 对图像进行平移
                    bboxes[i][:,0] += (cy-new_h)
                    bboxes[i][:,1] += cx
                if i==2: 
                    mosaic_img[max(cx-new_w, 0):cx, cy:min(H, cy+new_h), :] = images[i][max(0, new_w-cx):, :min(new_h, H-cy), :]
                    # 对图像进行平移
                    bboxes[i][:,0] += cy
                    bboxes[i][:,1] += (cx-new_w)
                if i==3: 
                    # 对图像进行平移
                    bboxes[i][:,0] += cy
                    bboxes[i][:,1] += cx
                    mosaic_img[cx:min(W, cx+new_w), cy:min(H, cy+new_h), :] = images[i][:min(new_w, W-cx), :min(new_h, H-cy), :]
                # 和边界处理 + 舍弃太小的框
                bboxes[i][:,2] += bboxes[i][:,0]
                bboxes[i][:,3] += bboxes[i][:,1]
                bboxes[i] = np.clip(bboxes[i], 0, self.input_shape[0])
                bboxes[i][:,2] -= bboxes[i][:,0]
                bboxes[i][:,3] -= bboxes[i][:,1]
                keep = np.where(np.logical_and(bboxes[i][:,2]>4, bboxes[i][:,3]>4))[0]
                bboxes[i] = bboxes[i][keep]
                labels[i] = labels[i][keep]

            labels = np.concatenate(labels, axis=0)
            bboxes = np.concatenate(bboxes, axis=0)

            if len(bboxes) != 0:
                return mosaic_img, bboxes, labels
 
        return image1, boxes1, labels1
    

    # DataLoader中collate_fn参数使用
    # 由于检测数据集每张图像上的目标数量不一
    # 因此需要自定义的如何组织一个batch里输出的内容
    @staticmethod
    def dataset_collate(batch):
        images  = []
        bboxes  = []
        for i, (img, box) in enumerate(batch):
            images.append(img)
            box[:, 0] = i
            bboxes.append(box)
                
        images  = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
        bboxes  = torch.from_numpy(np.concatenate(bboxes, 0)).type(torch.FloatTensor)
        return images, bboxes
    


    # 设置Dataloader的种子
    # DataLoader中worker_init_fn参数使
    # 为每个 worker 设置了一个基于初始种子和 worker ID 的独特的随机种子, 这样每个 worker 将产生不同的随机数序列，从而有助于数据加载过程的随机性和多样性
    @staticmethod
    def worker_init_fn(worker_id, seed):
        worker_seed = worker_id + seed
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    















class COCODataset(BaseDataset):

    def __init__(self, num_classes, annPath, imgDir, inputShape=[800, 800], anchors=None, anchors_mask=None, trainMode=True, map=None):
        '''__init__() 为默认构造函数，传入数据集类别（训练或测试），以及数据集路径

        Args:
            - annPath:     COCO annotation 文件路径
            - imgDir:      图像的根目录
            - inputShape: 网络要求输入的图像尺寸
            - trainMode:   训练集/测试集
            - trainMode:   训练集/测试集
            -  map:        categories字段映射
        Returns:
            FRCNNDataset
        '''      
        self.mode = trainMode
        self.num_classes = num_classes
        self.anchors = anchors
        self.anchors_mask = anchors_mask
        self.input_shape = inputShape
        self.tf = Transform(inputShape, 'coco')
        self.imgDir = imgDir
        self.annPath = annPath
        # 为实例注释初始化COCO的API
        self.coco=COCO(annPath)
        # 获取数据集中所有图像对应的imgId
        self.imgIds = self.coco.getImgIds()
        # 如果标签的id正好不是按顺序来的，还需进行映射
        self.map = map
        '''过滤掉那些没有框的图像,很重要!!!'''
        self.filterImgIds = self.filterImgById()
        # 数据集大小
        self.datasetNum = len(self.filterImgIds)
                

    

    def __getitem__(self, index):
        '''重载data.Dataset父类方法, 获取数据集中数据内容
           这里通过pycocotools来读取图像和标签
        '''   
        # 通过index获得图像, 图像的框, 以及框的标签
        image, boxes, labels = self.getDataByIndex(index)
        # 数据预处理与增强
        image, boxes, labels = self.augment(image, boxes, labels)
        # id映射
        if self.map != None:
            labels = [self.map[i] for i in labels]
        labels = np.array(labels)
        boxes = np.array(boxes, dtype=np.float32)
        # 再把coco格式转成YOLO格式(xywh -> norm(cxcywh)):
        boxes[:, 0] += boxes[:, 2] / 2
        boxes[:, 1] += boxes[:, 3] / 2
        boxes[:, [0, 2]] = boxes[:, [0, 2]] / image.shape[1]
        boxes[:, [1, 3]] = boxes[:, [1, 3]] / image.shape[0]
        # box坐标和box的类别拼接在一起合并为boxes变量:
        boxes = np.concatenate((labels.reshape(-1, 1), boxes), axis=1)
        # 前面一个是代表box所在batch里的图像的索引, 在collect_fn时处理
        boxes = np.concatenate((np.zeros((boxes.shape[0], 1)), boxes), axis=1)
        return image.transpose(2,0,1), boxes
    



    def getDataByIndex(self, index):
        '''通过index获得图像, 图像的框, 以及框的标签
        Args:
            - index:  数据集里数据的索引
        Returns:
            - image:   训练集/测试集
            - box:   训练集/测试集
            - label:        categories字段映射
        '''          

        # 通过imgId获取图像信息imgInfo: 例:{'id': 12465, 'license': 1, 'height': 375, 'width': 500, 'file_name': '2011_003115.jpg'}
        imgId = self.filterImgIds[index]
        imgInfo = self.coco.loadImgs(imgId)[0]
        # 载入图像 (通过imgInfo获取图像名，得到图像路径)               
        image = Image.open(os.path.join(self.imgDir, imgInfo['file_name']))
        image = np.array(image.convert('RGB'))
        # 得到图像里包含的BBox的所有id
        imgAnnIds = self.coco.getAnnIds(imgIds=imgId)   
        # 通过BBox的id找到对应的BBox信息
        anns = self.coco.loadAnns(imgAnnIds) 
        # 获取BBox的坐标和类别
        labels, boxes = [], []
        for ann in anns:
            # 过滤掉稠密聚集的标注框
            if ann['iscrowd'] == 1: continue
            # if ann['ignore'] == 1: continue
            labelName = ann['category_id']
            labels.append(labelName)
            boxes.append(ann['bbox'])
        labels = np.array(labels)
        boxes = np.array(boxes)

        return image, boxes, labels
    


    def filterImgById(self):
        '''过滤掉那些没标注的图像
        '''
        print('filtering no objects images...')
        filterImgIds = []
        for i in tqdm(range(len(self.imgIds))):
            # 获取图像信息(json文件 "images" 字段)
            imgInfo = self.coco.loadImgs(self.imgIds[i])[0]
            # 得到当前图像里包含的BBox的所有id
            annIds = self.coco.getAnnIds(imgIds=imgInfo['id'])
            # anns (json文件 "annotations" 字段)
            anns = self.coco.loadAnns(annIds)
            if len(anns)!=0:
                # 专门针对COCO数据集,这两张图片存在bbox的w或h=0的情况:
                if imgInfo['file_name'] not in ['000000200365.jpg', '000000550395.jpg', '9999985_00000_d_0000020.jpg']:
                    filterImgIds.append(self.imgIds[i])
        return filterImgIds
    









class YOLODataset(BaseDataset):

    def __init__(self, num_classes, anchors, anchors_mask, ann_dir, img_dir, inputShape=[800, 800], trainMode=True):
        '''__init__() 为默认构造函数，传入数据集类别（训练或测试），以及数据集路径

        Args:
            - annPath:     COCO annotation 文件路径
            - imgDir:      图像的根目录
            - inputShape: 网络要求输入的图像尺寸
            - trainMode:   训练集/测试集
            - trainMode:   训练集/测试集
            -  map:        categories字段映射
        Returns:
            FRCNNDataset
        '''      
        self.mode = trainMode
        self.num_classes = num_classes
        self.anchors = anchors
        self.anchors_mask = anchors_mask
        self.input_shape = inputShape
        self.tf = Transform(inputShape, 'coco')
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.ann_list = os.listdir(ann_dir)
        # 数据集大小
        self.datasetNum = len(self.ann_list)

    

    def __getitem__(self, index):
        '''重载data.Dataset父类方法, 获取数据集中数据内容
        '''   
        # 通过index获得图像, 图像的框, 以及框的标签
        image, boxes, labels = self.getDataByIndex(index)
        # 数据预处理与增强
        image, boxes, labels = self.augment(image, boxes, labels)
        labels = np.array(labels)
        boxes = np.array(boxes, dtype=np.float32)
        # 再把coco格式转成YOLO格式(xywh -> norm(cxcywh)):
        boxes[:, 0] += boxes[:, 2] / 2
        boxes[:, 1] += boxes[:, 3] / 2
        boxes[:, [0, 2]] = boxes[:, [0, 2]] / image.shape[1]
        boxes[:, [1, 3]] = boxes[:, [1, 3]] / image.shape[0]
        boxes = np.concatenate((labels.reshape(-1, 1), boxes), axis=1)
        # 前面一个是代表box所在batch里的图像的索引, 在collect_fn时处理
        boxes = np.concatenate((np.zeros((boxes.shape[0], 1)), boxes), axis=1)
        return image.transpose(2,0,1), boxes
    



    def getDataByIndex(self, index):
        '''通过index获得图像, 图像的框, 以及框的标签
        Args:
            - index:  数据集里数据的索引
        Returns:
            - image:   训练集/测试集
            - box:   训练集/测试集
            - label:        categories字段映射
        '''         
        ann_file_name = self.ann_list[index]
        ann_path = os.path.join(self.ann_dir, ann_file_name)
        img_path = os.path.join(self.img_dir, ann_file_name.split('.')[0]+'.jpg')
        # 载入图像 (通过imgInfo获取图像名，得到图像路径)               
        image = Image.open(img_path)
        W, H = image.size
        image = np.array(image.convert('RGB'))
        boxes, labels = [], []
        with open(ann_path, 'r') as ann_file:
            for line in ann_file.readlines():
                info = line[:-1].split(' ')
                cls, cx, cy, w, h = int(info[0]), float(info[1]), float(info[2]), float(info[3]), float(info[4])
                # 转成coco格式
                x0 = (cx - w / 2.) * W
                y0 = (cy - h / 2.) * H
                w *= W
                h *= H
                # print(x0, y0, w, h)
                boxes.append([x0, y0, w, h])
                labels.append(cls)
        
        boxes = boxes
        labels = labels
        return image, boxes, labels



















# 固定全局随机数种子
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




def visBatch(dataLoader:DataLoader, showText=False):
    '''可视化训练集一个batch
    Args:
        dataLoader: torch的data.DataLoader
    Retuens:
        None     
    '''
    # COCO
    catName = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    # VOC0712
    # catName = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", 
    #             "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    # visDrone2019
    # catName = ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"]
    for step, batch in enumerate(dataLoader):
        images, boxes = batch[0], batch[1]
        # 只可视化一个batch的图像：
        if step > 0: break
        # 图像均值
        mean = np.array([0.485, 0.456, 0.406]) 
        # 标准差
        std = np.array([[0.229, 0.224, 0.225]]) 
        print(boxes)
        # norm(cxcywh) -> cxcywh
        boxes[:, [2, 4]] = boxes[:, [2, 4]] * images.shape[3]
        boxes[:, [3, 5]] = boxes[:, [3, 5]] * images.shape[2]
        # cxcywh->xywh
        boxes[:, 2] -= boxes[:, 4] / 2
        boxes[:, 3] -= boxes[:, 5] / 2
        
        plt.figure(figsize = (9,9))
        for idx, img in enumerate(images):
            box = boxes[boxes[:,0]==idx]
            box = box.numpy()
            ax = plt.subplot(5,5,idx+1)
            img = img.numpy().transpose((1,2,0))
            # 由于在数据预处理时我们对数据进行了标准归一化，可视化的时候需要将其还原
            img = np.clip(img * std + mean, 0, 1)
            for instBox in box:
                cat_id, x0, y0, w, h = round(instBox[1]), round(instBox[2]), round(instBox[3]), round(instBox[4]), int(instBox[5])
                # 显示框
                ax.add_patch(plt.Rectangle((x0, y0), w, h, color='blue', fill=False, linewidth=0.6))
                # 显示类别
                if showText:
                    ax.text(x0, y0, catName[cat_id], bbox={'facecolor':'white', 'alpha':0.5})
            plt.imshow(img)
            # 在图像上方展示对应的标签
            # 取消坐标轴
            plt.axis("off")
             # 微调行间距
            plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99, wspace=0.01, hspace=0.01)
        plt.show()














def test_coco():
    # 固定随机种子
    seed = 23
    seed_everything(seed)
    # BatcchSize
    BS = 25
    # 图像尺寸
    imgSize = [1024, 1024]

    '''COCO'''
    # trainAnnPath = "E:/datasets/Universal/COCO2017/COCO/annotations/instances_train2017.json"
    # valAnnPath = "E:/datasets/Universal/COCO2017/COCO/annotations/instances_val2017.json"
    # trainImgDir =  "E:/datasets/Universal/COCO2017/COCO/train2017"
    # valImgDir = "E:/datasets/Universal/COCO2017/COCO/val2017"
    # cls_num = 80
    # map = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7, 9:8, 10:9, 11:10, 13:11, 14:12, 15:13, 16:14, 17:15, 18:16, 19:17, 20:18, 21:19, 22:20, 23:21, 
    #        24:22, 25:23, 27:24, 28:25, 31:26, 32:27, 33:28, 34:29, 35:30, 36:31, 37:32, 38:33, 39:34, 40:35, 41:36, 42:37, 43:38, 44:39, 46:40, 
    #        47:41, 48:42, 49:43, 50:44, 51:45, 52:46, 53:47, 54:48, 55:49, 56:50, 57:51, 58:52, 59:53, 60:54, 61:55, 62:56, 63:57, 64:58, 65:59, 
    #        67:60, 70:61, 72:62, 73:63, 74:64, 75:65, 76:66, 77:67, 78:68, 79:69, 80:70, 81:71, 82:72, 84:73, 85:74, 86:75, 87:76, 88:77, 89:78, 90:79}
    '''VOC0712'''
    # trainAnnPath = "E:/datasets/Universal/VOC0712/VOC2007/Annotations/coco/train.json"
    # testAnnPath = "E:/datasets/Universal/VOC0712/VOC2007/Annotations/coco/val.json"
    # trainImgDir = "E:/datasets/Universal/VOC0712/VOC2007/JPEGImages"
    # cls_num = 20
    # map = None
    '''visDrone2019'''
    # trainAnnPath = "E:/datasets/RemoteSensing/visdrone2019/annotations/train.json"
    # trainImgDir = "E:/datasets/RemoteSensing/visdrone2019/images/train/images"
    # cls_num = 10
    # map = None
    '''SODA-D'''
    # trainImgDir = "E:/datasets/Universal/SODA-D/Images/Images"
    # trainAnnPath = "E:/datasets/Universal/SODA-D/Annotations/train.json"
    # val_ann_dir = 'E:/datasets/Universal/SODA-D/Annotations/test.json'
    # val_img_dir = 'E:/datasets/Universal/SODA-D/Images/Images'
    # cls_num = 10
    # map = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7, 9:8, 10:9}
    '''DOTA-v1.0'''
    trainImgDir = "E:/datasets/RemoteSensing/DOTA-1.0_ss_1024/train/images"
    trainAnnPath = "E:/datasets/RemoteSensing/DOTA-1.0_ss_1024/coco_ann/hbox_train.json"
    cls_num = 15
    map = None





    ''' 自定义数据集读取类'''
    trainDataset = COCODataset(cls_num, trainAnnPath, trainImgDir, imgSize, trainMode=True, map=map)
    trainDataLoader = DataLoader(trainDataset, shuffle=True, batch_size=BS, num_workers=0, pin_memory=True,
                                    collate_fn=trainDataset.dataset_collate, worker_init_fn=partial(trainDataset.worker_init_fn, seed=seed))
    # validDataset = COCODataset(valAnnPath, valImgDir, imgSize, trainMode=False, map=map)
    # validDataLoader = DataLoader(validDataset, shuffle=True, batch_size=BS, num_workers=2, pin_memory=True, 
    #                               collate_fn=frcnn_dataset_collate, worker_init_fn=partial(worker_init_fn, seed=seed))



    print(f'训练集大小 : {trainDataset.__len__()}')
    visBatch(trainDataLoader)
    cnt = 0

    for step, batch in enumerate(trainDataLoader):
        images, boxes, = batch[0], batch[1]
        cnt+=1
        # torch.Size([bs, 3, 800, 800])
        print(f'images.shape : {images.shape}')   
        # 列表形式，因为每个框里的实例数量不一，所以每个列表里的box数量不一
        print(f'boxes.shape : {boxes.shape}')       
        break








def test_yolo():
    # 固定随机种子
    seed = 22
    seed_everything(seed)
    # BatcchSize
    BS = 25
    # 图像尺寸
    imgSize = [640, 640]
    anchors = [[10, 13], [16, 30], [33, 23],
                [30, 61], [62, 45], [59, 119],
                [116, 90], [156, 198], [373, 326],
                ]
    anchors_mask = [[0,1,2], [3,4,5], [6,7,8]]


    '''visDrone2019'''
    train_img_dir = "E:/datasets/RemoteSensing/visdrone2019/visdroneYOLOCrop1280_splitTrainVal/visdroneYOLOCrop12800/images/train"
    train_ann_dir = "E:/datasets/RemoteSensing/visdrone2019/visdroneYOLOCrop1280_splitTrainVal/visdroneYOLOCrop12800/labels/train"
    val_ann_dir = 'E:/datasets/RemoteSensing/visdrone2019/visdroneYOLOCrop1280_splitTrainVal/visdroneYOLOCrop12800/labels/test'
    val_img_dir = 'E:/datasets/RemoteSensing/visdrone2019/visdroneYOLOCrop1280_splitTrainVal/visdroneYOLOCrop12800/images/test'
    cls_num = 10


    ''' 自定义数据集读取类'''
    trainDataset = YOLODataset(cls_num, anchors, anchors_mask, train_ann_dir, train_img_dir, imgSize, trainMode=True)
    trainDataLoader = DataLoader(trainDataset, shuffle=True, batch_size=BS, num_workers=0, pin_memory=True,
                                    collate_fn=trainDataset.dataset_collate, worker_init_fn=partial(trainDataset.worker_init_fn, seed=seed))
    # validDataset = YOLODataset(valAnnPath, valImgDir, imgSize, trainMode=False, map=map)
    # validDataLoader = DataLoader(validDataset, shuffle=True, batch_size=BS, num_workers=2, pin_memory=True, 
    #                               collate_fn=frcnn_dataset_collate, worker_init_fn=partial(worker_init_fn, seed=seed))



    print(f'训练集大小 : {trainDataset.__len__()}')
    # visBatch(trainDataLoader)
    cnt = 0
    for step, batch in enumerate(trainDataLoader):
        images, boxes, = batch[0], batch[1]
        cnt+=1
        # torch.Size([bs, 3, 800, 800])
        print(f'images.shape : {images.shape}')   
        # 列表形式，因为每个框里的实例数量不一，所以每个列表里的box数量不一
        print(f'boxes.shape : {boxes.shape}')     
        break







# for test only:
if __name__ == "__main__":
    test_coco()