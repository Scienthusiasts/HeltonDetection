import numpy as np
import torch
from torchvision.ops import nms
from torch.nn import functional as F
import cv2
import torch.nn as nn
import os
import math
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image













def DOTAtxt2LongSideFormatYOLOtxt(ann_dir:str, tgt_dir:str, cat_names2id:dict, img_size:int):
    '''将DOTA数据集的标注文件转化为YOLO格式的标注文件(cx, cy, longside, shortside, θ), 基于长边表示法
        Args:
            - ann_dir:      txt标注文件根目录
            - tgt_dir:      转换后的txt标注文件保存的根目录
            - cat_names2id: 将字符类别转为数字id
            - img_size:     图像尺寸

        Returns:
            anchor: shape=[9, 4], anchor的四个坐标是以输入原始图像的尺寸而言
    '''
    if not os.path.isdir(tgt_dir):os.makedirs(tgt_dir)
    for ann_file_name in tqdm(os.listdir(ann_dir)):
        ann_file_path = os.path.join(ann_dir, ann_file_name)
        tgt_ann_file_path = os.path.join(tgt_dir, ann_file_name)
        
        boxes_per_img = []
        # 逐行读取box信息并转换为yolo格式
        with open(ann_file_path, 'r') as ann_file:
            for line in ann_file.readlines():
                info = line[:-1].split(' ')
                x0, y0, x1, y1 = float(info[0]), float(info[1]), float(info[2]), float(info[3])
                x2, y2, x3, y3 = float(info[4]), float(info[5]), float(info[6]), float(info[7])
                poly = np.array([[x0, y0], [x1, y1], [x2, y2], [x3, y3]], dtype=np.float32)
                # [(c_x, c_y), (w, h), theta] θ是矩形的长边相对于x轴的逆时针旋转角度, 范围为[0, 90), 旧版本是[-90, 0)
                [(c_x, c_y), (w, h), theta] = cv2.minAreaRect(poly)  
                # 将rect转为长边表示法, [c_x, c_y, w, h, θ] -> [c_x, c_y, longside, shortside, θ] θ∈[-180, 0)
                # 长边表示法中θ为最长边到x轴逆时针旋转的夹角，逆时针方向角度为负
                rect = np.array(cvminAreaRect2longsideformat(c_x, c_y, w, h, -theta))
                rect[:4] /= img_size
                cls = cat_names2id[info[8]]
                boxes_per_img.append(f'{cls} {rect[0]} {rect[1]} {rect[2]} {rect[3]} {rect[4]}\n')

        # 逐行写入yolo格式box信息
        with open(tgt_ann_file_path, 'w') as tgtg_ann_file:   
            for line in boxes_per_img:
                tgtg_ann_file.write(line)
        






def DOTAtxt2HBoxYOLOtxt(ann_dir:str, tgt_dir:str, cat_names2id:dict, img_size:int):
    '''将DOTA数据集的标注文件转化为YOLO格式的标注文件(cx, cy, longside, shortside, θ), 基于长边表示法
        Args:
            - ann_dir:      txt标注文件根目录
            - tgt_dir:      转换后的txt标注文件保存的根目录
            - cat_names2id: 将字符类别转为数字id
            - img_size:     图像尺寸

        Returns:
            anchor: shape=[9, 4], anchor的四个坐标是以输入原始图像的尺寸而言
    '''
    if not os.path.isdir(tgt_dir):os.makedirs(tgt_dir)
    for ann_file_name in tqdm(os.listdir(ann_dir)):
        ann_file_path = os.path.join(ann_dir, ann_file_name)
        tgt_ann_file_path = os.path.join(tgt_dir, ann_file_name)
        
        boxes_per_img = []
        # 逐行读取box信息并转换为yolo格式
        with open(ann_file_path, 'r') as ann_file:
            for line in ann_file.readlines():
                info = line[:-1].split(' ')
                cls = cat_names2id[info[8]]
                # 原始的标注是四边形四个角点
                x0, y0, x1, y1 = float(info[0]), float(info[1]), float(info[2]), float(info[3])
                x2, y2, x3, y3 = float(info[4]), float(info[5]), float(info[6]), float(info[7])
                # 求四边形的最小外接水平框(不超出图像边界)
                tl_x = max(min(x0, x1, x2, x3), 0)
                tl_y = max(min(y0, y1, y2, y3), 0)
                br_x = min(max(x0, x1, x2, x3), img_size)
                br_y = min(max(y0, y1, y2, y3), img_size)
                # 转化为yolo格式
                w = (br_x - tl_x) / img_size
                h = (br_y - tl_y) / img_size
                cx = ((tl_x + br_x) / 2) / img_size
                cy = ((tl_y + br_y) / 2) / img_size
                if (w*h>0):
                    boxes_per_img.append(f'{cls} {cx} {cy} {w} {h}\n')

        # 逐行写入yolo格式box信息
        with open(tgt_ann_file_path, 'w') as tgtg_ann_file:   
            for line in boxes_per_img:
                tgtg_ann_file.write(line)








def DOTAtxt2COCOjson(txtRoot, imgRoot, cat_names2id:dict, jsonPath, mode='DOTA', imgFormat='png'):
    """DOTA格式txt转COCO格式json

    Args:
        txtRoot:    the root directory of visdrone txt file.
        imgRoot:    The directory of image file.
        categories: label name ['airplane', 'car', 'person', ...]
        jsonPath:   the path where json file to be saved.
        imgFormat:  image file format.

    Returns:
        None
    """
    annsDict = {"images":[], "annotations":[], "categories":[]}
    boxId = 0
    for imgId, txtFile in tqdm(enumerate(os.listdir(txtRoot))):
        imgName = txtFile.replace('.txt', f'.{imgFormat}')
        imgFile = Image.open(os.path.join(imgRoot, imgName))
        W, H = imgFile.size
        annsDict['images'].append(
            {
                "id": imgId,
                "license": 1,
                "height": H,
                "width": W,
                "file_name": imgName
            }
        )
        with open(os.path.join(txtRoot, txtFile), 'r') as txt:
            # DOTA的四个角点标注方法(旋转框)
            if mode == 'DOTA':
                for line in txt.readlines():
                    line = line[:-1].split(' ')[:9]
                    x0, y0, x1, y1, x2, y2, x3, y3 = [float(i) for i in line[:8]]
                    cat_name = line[8]
                    annsDict['annotations'].append(
                        {
                            "id": boxId,
                            "image_id": imgId,
                            "category_id": cat_names2id[cat_name],
                            "bbox": [
                                x0, y0, x1, y1, x2, y2, x3, y3
                            ],
                            "area": 0, # 旋转框无法直接计算外接水平框的面积
                            "iscrowd": 0,
                            "ignore": 0
                        }
                    )
                    boxId += 1
            # YOLO的标注方法(水平框)
            if mode == 'YOLO':
                for line in txt.readlines():
                    line = line[:-1].split(' ')
                    cx, cy, w, h = [float(i) for i in line[1:]]
                    x = cx - w / 2
                    y = cy - h / 2
                    cat_id = int(line[0])
                    annsDict['annotations'].append(
                        {
                            "id": boxId,
                            "image_id": imgId,
                            "category_id": cat_id,
                            "bbox": [
                                round(x*W), round(y*H), round(w*W), round(h*H)
                            ],
                            "area": w*h, 
                            "iscrowd": 0,
                            "ignore": 0
                        }
                    )
                    boxId += 1
    for i, cat in enumerate(cat_names2id.keys()):
        annsDict['categories'].append(
        {
            "id": i,
            "name": cat,
        }
    )

    with open(jsonPath, 'w') as jsonFile:
        json.dump(annsDict, jsonFile)




















def cvminAreaRect2longsideformat(x_c, y_c, width, height, theta):
    '''
    trans minAreaRect(x_c, y_c, width, height, θ) to longside format(x_c, y_c, longside, shortside, θ)
    两者区别为:
            当opencv表示法中width为最长边时（包括正方形的情况），则两种表示方法一致
            当opencv表示法中width不为最长边 ，则最长边表示法的角度要在opencv的Θ基础上-90度         
    @param x_c: center_x
    @param y_c: center_y
    @param width: x轴逆时针旋转碰到的第一条边
    @param height: 与width不同的边
    @param theta: x轴逆时针旋转与width的夹角，由于原点位于图像的左上角，逆时针旋转角度为负 [-90, 0)
    @return: 
            x_c: center_x
            y_c: center_y
            longside: 最长边
            shortside: 最短边
            theta_longside: 最长边和x轴逆时针旋转的夹角，逆时针方向角度为负 [-180, 0)
    '''
    '''
    意外情况:(此时要将它们恢复符合规则的opencv形式：wh交换，Θ置为-90)
    竖直box：box_width < box_height  θ=0
    水平box：box_width > box_height  θ=0
    '''
    if theta == 0:
        theta = -90
        buffer_width = width
        width = height
        height = buffer_width

    if theta > 0:
        if theta != 90:  # Θ=90说明wh中有为0的元素，即gt信息不完整，无需提示异常，直接删除
            print('θ计算出现异常，当前数据为：%.16f, %.16f, %.16f, %.16f, %.1f;超出opencv表示法的范围：[-90,0)' % (x_c, y_c, width, height, theta))
        return False

    if theta < -90:
        print('θ计算出现异常，当前数据为：%.16f, %.16f, %.16f, %.16f, %.1f;超出opencv表示法的范围：[-90,0)' % (x_c, y_c, width, height, theta))
        return False

    if width != max(width, height):  # 若width不是最长边
        longside = height
        shortside = width
        theta_longside = theta - 90
    else:  # 若width是最长边(包括正方形的情况)
        longside = width
        shortside = height
        theta_longside = theta

    if longside < shortside:
        print('旋转框转换表示形式后出现问题：最长边小于短边;[%.16f, %.16f, %.16f, %.16f, %.1f]' % (x_c, y_c, longside, shortside, theta_longside))
        return False
    if (theta_longside < -180 or theta_longside >= 0):
        print('旋转框转换表示形式时出现问题:θ超出长边表示法的范围：[-180,0);[%.16f, %.16f, %.16f, %.16f, %.1f]' % (x_c, y_c, longside, shortside, theta_longside))
        return False

    return x_c, y_c, longside, shortside, theta_longside







def longsideformat2cvminAreaRect(x_c, y_c, longside, shortside, theta_longside):
    '''
    trans longside format(x_c, y_c, longside, shortside, θ) to minAreaRect(x_c, y_c, width, height, θ)
    两者区别为:
            当opencv表示法中width为最长边时（包括正方形的情况），则两种表示方法一致
            当opencv表示法中width不为最长边 ，则最长边表示法的角度要在opencv的Θ基础上-90度         
    @param x_c: center_x
    @param y_c: center_y
    @param longside: 最长边
    @param shortside: 最短边
    @param theta_longside: 最长边和x轴逆时针旋转的夹角，逆时针方向角度为负 [-180, 0)
    @return: ((x_c, y_c),(width, height),Θ)
            x_c: center_x
            y_c: center_y
            width: x轴逆时针旋转碰到的第一条边最长边
            height: 与width不同的边
            theta: x轴逆时针旋转与width的夹角，由于原点位于图像的左上角，逆时针旋转角度为负 [-90, 0)
    '''
    if (theta_longside >= -180 and theta_longside < -90):  # width is not the longest side
        width = shortside
        height = longside
        theta = theta_longside + 90
    else:
        width = longside
        height =shortside
        theta = theta_longside

    if theta < -90 or theta >= 0:
        print('当前θ=%.1f，超出opencv的θ定义范围[-90, 0)' % theta)

    return ((x_c, y_c), (width, height), theta)





# for test only:
if __name__ == '__main__':
    img_size = 1024
    mode = 'val'
    ann_dir = f'E:/datasets/RemoteSensing/DOTA-1.0_ss_1024/{mode}/yolo_hbox_annfiles'
    img_dir = f'E:/datasets/RemoteSensing/DOTA-1.0_ss_1024/{mode}/images'
    tgt_dir = f'E:/datasets/RemoteSensing/DOTA-1.0_ss_1024/{mode}/yolo_hbox_annfiles'
    json_path = f'E:/datasets/RemoteSensing/DOTA-1.0_ss_1024/coco_ann/hbox_{mode}.json'
    cat_names2id = {
        'plane':0, 'baseball-diamond':1, 'bridge':2, 'ground-track-field':3,
        'small-vehicle':4, 'large-vehicle':5, 'ship':6, 'tennis-court':7,
        'basketball-court':8, 'storage-tank':9, 'soccer-ball-field':10, 
        'roundabout':11, 'harbor':12, 'swimming-pool':13, 'helicopter':14
    }
    # DOTAtxt2LongSideFormatYOLOtxt(ann_dir, tgt_dir, cat_names2id, img_size)
    # DOTAtxt2HBoxYOLOtxt(ann_dir, tgt_dir, cat_names2id, img_size)
    DOTAtxt2COCOjson(ann_dir, img_dir, cat_names2id, json_path, mode='YOLO', imgFormat='png')
