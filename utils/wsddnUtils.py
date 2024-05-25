import cv2
import numpy as np
import json
import os
import xml.etree.ElementTree as ET
from tqdm import tqdm
from pycocotools.coco import COCO
from PIL import Image






# DEF selectiveSearch : SS方法生成proposals
def selectiveSearch(image:np.array, mode='fast'):
    """SS方法生成proposals.

    Args:
        :param image: 图像
        :param mode:  快速模式或是质量模式

    Retuens:
        :param proposals: SS提取的proposals
        :param image:     图像
    """

    # 创建一个Selective Search分割对象
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    # 设置要使用的图像
    ss.setBaseImage(image)
    if mode=='fast':
        ss.switchToSelectiveSearchFast()
    elif mode =='quality':
        ss.switchToSelectiveSearchQuality()
    # 运行Selective Search算法(格式xywh)
    proposals = ss.process()
    return proposals, image








'''批量图像生成ss先验框(基于已经有了COCO标注的数据集)'''
def genCOCOBatchImgSSBox(jsonPath, imgDir, saveJsonPath):
    annsDict = {"images":[], "annotations":[], "categories":[]}
    boxId = 1
    # 为实例注释初始化COCO的API
    coco=COCO(jsonPath)
    # 获取数据集中所有图像对应的imgId
    imgIds = list(coco.imgs.keys())
    for imgId in tqdm(imgIds):
        # 图像信息
        imgInfo = coco.loadImgs(imgId)
        # 得到图像里包含的BBox的所有id
        imgAnnIds = coco.getAnnIds(imgIds=imgId)   
        imgInfo = coco.loadImgs(imgId)[0]
        # 通过BBox的id找到对应的BBox信息
        anns = coco.loadAnns(imgAnnIds) 
        # 通过box的信息找到图像里所有的类别:
        catPerImg = list(set([ann['category_id'] for ann in anns]))
        '''添加images字段'''
        annsDict['images'].append(
            {
                "id": imgId,
                "height": imgInfo['height'],
                "width": imgInfo['width'],
                "file_name": imgInfo['file_name'],
                "label": catPerImg
            }
        )
        # 载入图像 (通过imgInfo获取图像名，得到图像路径)               
        image = Image.open(os.path.join(imgDir, imgInfo['file_name']))
        image = np.array(image.convert('RGB'))
        # ss方法生成先验框(格式xywh)
        proposals, _ = selectiveSearch(image)
        '''添加annotations字段'''
        for SSBox in proposals:
            x, y, w, h = int(SSBox[0]), int(SSBox[1]), int(SSBox[2]), int(SSBox[3])
            annsDict['annotations'].append(
                {
                    "id": boxId,
                    "image_id": imgId,
                    "category_id": 0,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                }
            )
            boxId += 1
    '''添加categories字段'''
    annsDict['categories'].append(
    {
        "id": 0,
        "name": 'SS',
    })           
    with open(saveJsonPath, 'w') as jsonFile:
        json.dump(annsDict, jsonFile)





'''批量图像生成ss先验框(基于分类数据集)'''
def genCOCOBatchImgSSBoxWithoutAnn(img_dir, save_json_path):
    annsDict = {"images":[], "annotations":[], "categories":[]}
    boxId = 1
    imgId = 1
    for cat_dir in os.listdir(img_dir):
        print(f'processing {cat_dir}...')
        for cat_img in tqdm(os.listdir(os.path.join(img_dir, cat_dir))):
            # 图像标签由图像所在类别文件夹命名
            label = int(cat_dir)
            # 载入图像 (通过imgInfo获取图像名，得到图像路径)               
            image = Image.open(os.path.join(img_dir, cat_dir, cat_img))
            image = np.array(image.convert('RGB'))
            # 获取图像尺寸
            H, W = image.shape[:2]
            annsDict['images'].append(
                {
                    "id": imgId,
                    "height": H,
                    "width": W,
                    "file_name": cat_img,
                    "label": [label]
                }
            )
            # ss方法生成先验框(格式xywh)
            proposals, _ = selectiveSearch(image)
            if(len(proposals)<2):
                print(f'less than 2 proposals in {cat_img}, ignore.')
                continue
            '''添加annotations字段'''
            for SSBox in proposals:
                x, y, w, h = int(SSBox[0]), int(SSBox[1]), int(SSBox[2]), int(SSBox[3])
                annsDict['annotations'].append(
                    {
                        "id": boxId,
                        "image_id": imgId,
                        # box没有标签，默认全为0
                        "category_id": 0,
                        "bbox": [x, y, w, h],
                        "area": w * h,
                    }
                )
                boxId += 1

            # 填充, 无意义:
            # annsDict['annotations'].append(
            #     {
            #         "id": boxId,
            #         "image_id": imgId,
            #         # box没有标签，默认全为0
            #         "category_id": 0,
            #         "bbox": [0,0,1,1],
            #         "area": 0,
            #     }
            # )
            # boxId += 1
            # imgId += 1
    '''添加categories字段(box没有标签, 默认全为0)'''
    annsDict['categories'].append(
    {
        "id": 0,
        "name": 'SS',
    })           
    with open(save_json_path, 'w') as jsonFile:
        json.dump(annsDict, jsonFile)





'''COCO标注导出json格式的ss proposals(每张图片一个json)'''
def splitCOCO2SSBox(jsonPath, saveJsonDir):
    if not os.path.isdir(saveJsonDir):os.makedirs(saveJsonDir)
    # 为实例注释初始化COCO的API
    coco=COCO(jsonPath)
    # 获取数据集中所有图像对应的imgId
    imgIds = list(coco.imgs.keys())
    for imgId in tqdm(imgIds):
        img_name = coco.loadImgs(imgId)[0]['file_name'][:-4]
        # 得到图像里包含的BBox的所有id
        imgAnnIds = coco.getAnnIds(imgIds=imgId)   
        # 通过BBox的id找到对应的BBox信息
        anns = coco.loadAnns(imgAnnIds) 
        proposals = []
        # 抽取出proposals字段
        for ann in anns:
            proposals.append(ann['bbox'])
        with open(os.path.join(saveJsonDir, img_name+'.json'), 'w') as jsonFile:
            json.dump(proposals, jsonFile)













if __name__ == '__main__':



    '''批量图像生成ss 先验框(有COCO标注)'''
    # mode = 'test'
    # jsonPath = f"E:/datasets/Universal/VOC2007/VOC{mode}_06-Nov-2007/VOCdevkit/VOC2007/COCOAnnotations/{mode}.json"
    # imgDir = f"E:/datasets/Universal/VOC2007/VOC{mode}_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages"
    # saveJsonPath = f"./VOC07_{mode}.json"
    # genCOCOBatchImgSSBox(jsonPath, imgDir, saveJsonPath)

    '''批量图像生成ss 先验框(分类数据集, 无标注)'''
    mode = 'train'
    img_dir = f"E:/datasets/Classification/cats_dogs/classification_{mode}"
    save_json_path = f"E:/datasets/Classification/cats_dogs/labels/{mode}_without_proposals.json"
    genCOCOBatchImgSSBoxWithoutAnn(img_dir, save_json_path)

    '''COCO标注导出json格式的ss proposals(每张图片一个json)'''
    # json_path = "E:/datasets/Classification/cats_dogs/labels/train.json"
    # save_json_dir = f"E:/datasets/Classification/cats_dogs/proposals"
    # splitCOCO2SSBox(json_path, save_json_dir)

    
