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
from utils.FasterRCNNAnchorUtils import *
from datasets.preprocess import Transform






class COCODataset(Dataset):

    def __init__(self, annPath, imgDir, anchors, featStride, inputShape=[800, 800], trainMode=True, map=None):
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
        self.anchors = anchors
        self.featStride = featStride
        self.tf = Transform(imgSize=inputShape)
        self.imgDir = imgDir
        self.input_shape = inputShape
        self.annPath = annPath
        # 为实例注释初始化COCO的API
        self.coco=COCO(annPath)
        # 获取数据集中所有图像对应的imgId
        self.imgIds = self.coco.getImgIds()
        # 如果标签的id正好不是按顺序来的，还需进行映射
        self.map = map
        # 从wh转为xyxy(以(0,0)为中心)
        anchors = genAnchorNP(anchors)
        # 将9个先验框应用到整个特征图上, 当输入图片为600,600,3的时候，shape为(38x38x9=12996, 4)
        self.anchors = applyAnchor2FeatNP(anchors, featStride, self.input_shape) # [featW*featH*9, 4]
        # self.anchors = self.anchors[0]
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
        image, boxes, labels = self.augment(image, boxes, labels)
        # id映射
        if self.map != None:
            labels = [self.map[i] for i in labels]
        # 再把coco格式转成VOC格式(x0, y0, x1, y1):
        boxes = np.array([[b[0], b[1], b[2]+b[0], b[3]+b[1]] for b in boxes])
        # y_true.shape = (w0*h0*3, 9) (w1*h1*3, 9) (w2*h3*3, 9) (从浅层到深层)
        # 9 = 4(xywh回归值的gt) + 1(类别) + 4(gt的原图坐标)
        y_trues = RPNMaxIoUAssignerNP(boxes, self.anchors)
        return image.transpose(2,0,1), boxes, np.array(labels), y_trues
    

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



    def yoloMosaic4(self, image1, boxes1, labels1, jitter=0.2, scale=.6, p=0.5):
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
                # 
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
                keep = np.where(np.logical_and(bboxes[i][:,2]>8, bboxes[i][:,3]>8))[0]
                bboxes[i] = bboxes[i][keep]
                labels[i] = labels[i][keep]

            labels = np.concatenate(labels, axis=0)
            bboxes = np.concatenate(bboxes, axis=0)

            if len(bboxes) != 0:
                return mosaic_img, bboxes, labels
 
        return image1, boxes1, labels1
    
    


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
            if ann['ignore'] == 1: continue
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




    # DataLoader中collate_fn参数使用
    # 由于检测数据集每张图像上的目标数量不一
    # 因此需要自定义的如何组织一个batch里输出的内容
    @staticmethod
    def dataset_collate(batch):
        images = []
        bboxes = []
        labels = []
        y_trues = [[] for _ in batch[0][3]]
        for img, box, label, y_true in batch:
            images.append(img)
            bboxes.append(box)
            labels.append(label)
            for i, lvl_y_true in enumerate(y_true):
                y_trues[i].append(lvl_y_true)

        images = torch.from_numpy(np.array(images))
        # y_trues = [[...], [...], [...]], y_trues[i].shape = torch.Size([bs, w*h*3, 9])
        y_trues = [torch.from_numpy(np.array(ann, np.float32)).type(torch.FloatTensor) for ann in y_trues]
        return images, bboxes, labels, y_trues



    # 设置Dataloader的种子
    # DataLoader中worker_init_fn参数使
    # 为每个 worker 设置了一个基于初始种子和 worker ID 的独特的随机种子, 这样每个 worker 将产生不同的随机数序列，从而有助于数据加载过程的随机性和多样性
    @staticmethod
    def worker_init_fn(worker_id, seed):
        worker_seed = worker_id + seed
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)


# 固定全局随机数种子
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




def visBatch(dataLoader:DataLoader):
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
    for step, batch in enumerate(dataLoader):
        images, boxes, labels = batch[0], batch[1], batch[2]
        # 只可视化一个batch的图像：
        if step > 0: break
        # 图像均值
        mean = np.array([0.485, 0.456, 0.406]) 
        # 标准差
        std = np.array([[0.229, 0.224, 0.225]]) 
        plt.figure(figsize = (8,8))
        for idx, imgBoxLabel in enumerate(zip(images, boxes, labels)):
            img, box, label = imgBoxLabel
            ax = plt.subplot(4,4,idx+1)
            img = img.numpy().transpose((1,2,0))
            # 由于在数据预处理时我们对数据进行了标准归一化，可视化的时候需要将其还原
            img = np.clip(img * std + mean, 0, 1)
            for instBox, instLabel in zip(box, label):
                x0, y0, x1, y1 = round(instBox[0]),round(instBox[1]), round(instBox[2]), round(instBox[3])
                # 显示框
                ax.add_patch(plt.Rectangle((x0, y0), x1-x0, y1-y0, color='blue', fill=False, linewidth=1))
                # 显示类别
                ax.text(x0, y0, catName[instLabel], bbox={'facecolor':'white', 'alpha':0.5})
            plt.imshow(img)
            # 在图像上方展示对应的标签
            # 取消坐标轴
            plt.axis("off")
             # 微调行间距
            plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.05, hspace=0.05)
        plt.show()













# for test only:
if __name__ == "__main__":
    # 固定随机种子
    seed = 22
    seed_everything(seed)
    # BatcchSize
    BS = 16
    # 图像尺寸
    imgSize = [800, 800]
    anchors = [[[90.5097, 181.0193], [128, 128], [181.0193, 90.5097]], [[181.0193, 362.0387], [256, 256], [362.0387, 181.0193]], [[362.0387, 724.0773], [512, 512], [724.0773, 362.0387]]]
    s = [8, 16, 32]
    '''COCO'''
    trainAnnPath = "E:/datasets/Universal/COCO2017/COCO/annotations/instances_train2017.json"
    valAnnPath = "E:/datasets/Universal/COCO2017/COCO/annotations/instances_val2017.json"
    trainImgDir =  "E:/datasets/Universal/COCO2017/COCO/train2017"
    valImgDir = "E:/datasets/Universal/COCO2017/COCO/val2017"
    cls_num = 80
    map = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7, 9:8, 10:9, 11:10, 13:11, 14:12, 15:13, 16:14, 17:15, 18:16, 19:17, 20:18, 21:19, 22:20, 23:21, 
           24:22, 25:23, 27:24, 28:25, 31:26, 32:27, 33:28, 34:29, 35:30, 36:31, 37:32, 38:33, 39:34, 40:35, 41:36, 42:37, 43:38, 44:39, 46:40, 
           47:41, 48:42, 49:43, 50:44, 51:45, 52:46, 53:47, 54:48, 55:49, 56:50, 57:51, 58:52, 59:53, 60:54, 61:55, 62:56, 63:57, 64:58, 65:59, 
           67:60, 70:61, 72:62, 73:63, 74:64, 75:65, 76:66, 77:67, 78:68, 79:69, 80:70, 81:71, 82:72, 84:73, 85:74, 86:75, 87:76, 88:77, 89:78, 90:79}
    '''VOC0712'''
    # trainAnnPath = "E:/datasets/Universal/VOC0712/VOC2007/Annotations/coco/train.json"
    # testAnnPath = "E:/datasets/Universal/VOC0712/VOC2007/Annotations/coco/val.json"
    # imgDir = "E:/datasets/Universal/VOC0712/VOC2007/JPEGImages"
    # cls_num = 20
    # 自定义数据集读取类
    trainDataset = COCODataset(trainAnnPath, trainImgDir, anchors, s, imgSize, trainMode=True, map=map)
    trainDataLoader = DataLoader(trainDataset, shuffle=True, batch_size=BS, num_workers=0, pin_memory=True,
                                    collate_fn=trainDataset.dataset_collate, worker_init_fn=partial(trainDataset.worker_init_fn, seed=seed))
    # validDataset = COCODataset(valAnnPath, valImgDir, imgSize, trainMode=False, map=map)
    # validDataLoader = DataLoader(validDataset, shuffle=True, batch_size=BS, num_workers=2, pin_memory=True, 
    #                               collate_fn=frcnn_dataset_collate, worker_init_fn=partial(worker_init_fn, seed=seed))


    print(f'训练集大小 : {trainDataset.__len__()}')
    # visBatch(trainDataLoader)
    # cnt = 0
    for step, batch in enumerate(trainDataLoader):
        images, boxes, labels = batch[0], batch[1], batch[2]
        # torch.Size([bs, 3, 800, 800])
        print(f'images.shape : {images.shape}')   
        # 列表形式，因为每个框里的实例数量不一，所以每个列表里的box数量不一
        print(f'len(boxes) : {len(boxes)}')     
        # 列表形式，因为每个框里的实例数量不一，所以每个列表里的label数量不一  
        print(f'len(labels) : {len(labels)}')     
        break