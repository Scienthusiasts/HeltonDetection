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
import json
import matplotlib.pyplot as plt
import torchvision.transforms as tf
from tqdm import tqdm





class Transform():
    '''数据预处理/数据增强(基于albumentations库)
       https://albumentations.ai/docs/api_reference/full_reference/
    '''
    def __init__(self, imgSize):
        maxSize = max(imgSize[0], imgSize[1])
        # 训练时增强
        self.trainTF = A.Compose([
                A.BBoxSafeRandomCrop(p=0.5),
                # A.RandomSizedBBoxSafeCrop(800, 800, erosion_rate=0.0, interpolation=1, p=0.5),
                # 随机翻转
                A.HorizontalFlip(p=0.5),
                # 参数：随机色调、饱和度、值变化
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, always_apply=False, p=0.5),
                # 随机对比度增强
                A.CLAHE(p=0.1),
                # 高斯噪声
                A.GaussNoise(var_limit=(0.05, 0.09), p=0.4),     
                # 随机转为灰度图
                A.ToGray(p=0.05),
                A.OneOf([
                    # 使用随机大小的内核将运动模糊应用于输入图像
                    A.MotionBlur(p=0.2),   
                    # 中值滤波
                    A.MedianBlur(blur_limit=3, p=0.1),    
                    # 使用随机大小的内核模糊输入图像
                    A.Blur(blur_limit=3, p=0.1),  
                ], p=0.2),
            ],
            bbox_params=A.BboxParams(format='coco', min_area=0, min_visibility=0.0, label_fields=['category_ids']),
            )
        # 基本数据预处理
        self.normalTF = A.Compose([
                # 最长边限制为imgSize
                A.LongestMaxSize(max_size=maxSize),
                # 较短的边做padding
                A.PadIfNeeded(imgSize[0], imgSize[1], border_mode=0, mask_value=[0,0,0]),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ],
            bbox_params=A.BboxParams(format='coco', min_area=0, min_visibility=0.0, label_fields=['category_ids']),
            )
        # 测试时增强
        self.testTF = A.Compose([
                # 最长边限制为imgSize
                A.LongestMaxSize(max_size=maxSize),
                # 较短的边做padding
                A.PadIfNeeded(imgSize[0], imgSize[1], border_mode=0, mask_value=[0,0,0]),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])











class COCODataset(Dataset):

    def __init__(self, annPath, imgDir, inputShape=[800, 800], trainMode=True, map=None, proposals_json_dir=None):
        '''__init__() 为默认构造函数，传入数据集类别（训练或测试），以及数据集路径

        Args:
            :param annPath:             COCO annotation 文件路径
            :param imgDir:              图像的根目录
            :param inputShape:          网络要求输入的图像尺寸
            :param trainMode:           训练集/测试集
            :param trainMode:           训练集/测试集
            :param  map:                categories字段映射
            :param  proposals_json_dir: 用来区分proposals是在COCOjson里面还是每一张图像单独分开
        Returns:
            FRCNNDataset
        '''      
        self.mode = trainMode
        self.tf = Transform(imgSize=inputShape)
        self.imgDir = imgDir
        self.annPath = annPath
        # 为实例注释初始化COCO的API
        self.coco=COCO(annPath)
        # 获取数据集中所有图像对应的imgId
        self.imgIds = self.coco.getImgIds()
        # 如果标签的id正好不是按顺序来的，还需进行映射
        self.map = map
        self.proposals_json_dir = proposals_json_dir
        '''过滤掉那些没有框的图像,很重要!!!'''
        self.filterImgIds = self.filterImgById()
        # 数据集大小
        self.datasetNum = len(self.filterImgIds)
                

    def __len__(self):
        '''重载data.Dataset父类方法, 返回数据集大小
        '''
        return self.datasetNum
    

    def __getitem__(self, index):
        '''重载data.Dataset父类方法, 获取数据集中数据内容
           这里通过pycocotools来读取图像和标签
        '''   
        # 通过index获得图像, 图像的框, 以及框的标签
        image, boxes, labels = self.getDataByIndex(index)
        # 数据预处理与增强
        image, boxes = self.augment(image, boxes)
        # id映射
        if self.map != None:
            labels = [self.map[i] for i in labels]
        # coco格式xywh
        boxes = [[b[0], b[1], b[2], b[3]] for b in boxes]
        return image.transpose(2,0,1), np.array(boxes), np.array(labels)
    

    def augment(self, image, boxes):
        '''所有数据增强+预处理操作(顺序不能乱!)
        '''   
        if (self.mode):
            # 基本的数据增强
            image, boxes = self.trainAlbumAug(image, boxes)
        # 数据预处理(pad成统一size, 归一化)
        image, boxes = self.normalAlbumAug(image, boxes)

        return image, boxes

        
    def trainAlbumAug(self, image, boxes):
        """基于albumentations库的训练时数据增强
        """
        # albumentation的图像维度得是[W,H,C]
        train_trans = self.tf.trainTF(image=image, bboxes=boxes, category_ids=np.zeros(len(boxes)))
        image, boxes = train_trans['image'], train_trans['bboxes']
        # 这里的box是coco格式(xywh)
        return image, boxes
        


    def normalAlbumAug(self, image, boxes):
        """基于albumentations库的基础数据预处理
        """
        normal_trans = self.tf.normalTF(image=image, bboxes=boxes, category_ids=np.zeros(len(boxes)))
        image, boxes = normal_trans['image'], normal_trans['bboxes']
        # 这里的box是coco格式(xywh)
        return image, boxes
    



    def getDataByIndex(self, index):
        '''通过index获得图像, 图像的框, 以及框的标签
        Args:
            :param index:  数据集里数据的索引
        Returns:
            :param image: 训练集/测试集
            :param box:   训练集/测试集
            :param label: categories字段映射
        '''          

        # 通过imgId获取图像信息imgInfo: 例:{'id': 12465, 'license': 1, 'height': 375, 'width': 500, 'file_name': '2011_003115.jpg'}
        imgId = self.imgIds[index]
        imgInfo = self.coco.loadImgs(imgId)[0]

        # 载入图像 (通过imgInfo获取图像名，得到图像路径)               
        image = Image.open(os.path.join(self.imgDir, imgInfo['file_name']))
        image = np.array(image.convert('RGB'))
        # 得到图像里包含的BBox的所有id
        imgAnnIds = self.coco.getAnnIds(imgIds=imgId)   
        # 获取image-level的标签
        labels = imgInfo['label']
        
        if self.proposals_json_dir == None:
            # 通过BBox的id找到对应的BBox信息
            anns = self.coco.loadAnns(imgAnnIds) 
            # 获取SS proposals
            boxes = [ann['bbox'] for ann in anns]
        else:
            with open(os.path.join(self.proposals_json_dir, imgInfo['file_name'].replace('jpg', 'json')), 'r') as f:
                boxes = json.load(f)

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
                if imgInfo['file_name'] not in ['000000200365.jpg', '000000550395.jpg']:
                    filterImgIds.append(self.imgIds[i])
        return filterImgIds






    # DataLoader中collate_fn参数使用
    # 由于检测数据集每张图像上的目标数量不一
    # 因此需要自定义的如何组织一个batch里输出的内容
    @staticmethod
    def dataset_collate(batch):
        images = []
        bboxes = []
        labels = []
        for img, box, label in batch:
            images.append(img)
            bboxes.append(box)
            labels.append(label)
        images = torch.from_numpy(np.array(images))
        return images, bboxes, labels



    # 设置Dataloader的种子
    # DataLoader中worker_init_fn参数使
    # 为每个 worker 设置了一个基于初始种子和 worker ID 的独特的随机种子, 这样每个 worker 将产生不同的随机数序列，从而有助于数据加载过程的随机性和多样性
    @staticmethod
    def worker_init_fn(worker_id, seed):
        worker_seed = worker_id + seed
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)









def visBatch(dataLoader, cat_names):
    '''可视化训练集一个batch(8x8)
    Args:
        dataLoader: torch的data.DataLoader
    Retuens:
        None     
    '''
    for step, batch in enumerate(dataLoader):
        images, boxes, labels = batch[0], batch[1], batch[2]
        # 只可视化一个batch的图像：
        if step > 0: break
        # 图像均值
        mean = np.array([0.485, 0.456, 0.406]) 
        # 标准差
        std = np.array([[0.229, 0.224, 0.225]]) 
        plt.figure(figsize = (10,10))
        for idx, imgBoxLabel in enumerate(zip(images, boxes, labels)):
            img, boxes, labels = imgBoxLabel
            ax = plt.subplot(2,2,idx+1)
            img = img.numpy().transpose((1,2,0))
            # 由于在数据预处理时我们对数据进行了标准归一化，可视化的时候需要将其还原
            img = img * std + mean
            # 绘制框和标签
            for box in boxes:
                x, y, w, h = round(box[0]),round(box[1]), round(box[2]), round(box[3])
                ax.add_patch(plt.Rectangle((x, y), w, h, color="green", fill=False, linewidth=1))
            
            label_text = ' '.join([cat_names[label] for label in labels])
            print(label_text)
            plt.imshow(img)
            # 在图像上方展示对应的标签
            # 取消坐标轴
            plt.axis("off")   
                    
             # 微调行间距            
            plt.subplots_adjust(hspace = 0.5)    
        plt.show()








# for test only:
if __name__ == "__main__":
    # 固定随机种子
    seed = 22
    # batchsize大小
    BS = 4
    # 图像大小
    imgSize = [800, 800]
    '''VOC2007'''
    # mode='trainval'  
    # AnnPath = f"E:/datasets/Universal/VOC2007/WSDDN_COCO_format/VOC07_{mode}.json"
    # imgDir = f"E:/datasets/Universal/VOC2007/VOC{mode}_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages"
    # proposal_json_dir = None
    # cat_names = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", 
    #             "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    '''cats_vs_dogs''' 
    AnnPath = f"E:/datasets/Classification/cats_dogs/WSDDN_format/labels/train_without_proposals.json"
    imgDir = f"E:/datasets/Classification/cats_dogs/WSDDN_format/trainval"    
    proposal_json_dir = 'E:/datasets/Classification/cats_dogs/WSDDN_format/proposals'
    cat_names = ["cat", "dog"]


    # 定义数据集读取
    Dataset = COCODataset(AnnPath, imgDir, imgSize, trainMode=True, proposals_json_dir=proposal_json_dir)
    DataLoader = DataLoader(Dataset, shuffle=True, batch_size = BS, num_workers=2, pin_memory=True, drop_last=False, collate_fn=Dataset.dataset_collate, worker_init_fn=partial(Dataset.worker_init_fn, seed=seed))
    print(f'数据集大小 : {Dataset.__len__()}')
    # 可视化数据集
    visBatch(DataLoader, cat_names)
    # 模拟读取
    for step, batch in enumerate(DataLoader):
        images, boxes, labels = batch[0], batch[1], batch[2]
        # torch.Size([bs, 3, 800, 800])
        print(f'images.shape : {images.shape}')   
        # 列表形式，因为每个框里的实例数量不一，所以每个列表里的box数量不一
        print(f'len(boxes) : {len(boxes)}')     
        # 列表形式，因为每个框里的实例数量不一，所以每个列表里的label数量不一  
        print(f'len(labels) : {len(labels)}')    
        print(labels)
        break