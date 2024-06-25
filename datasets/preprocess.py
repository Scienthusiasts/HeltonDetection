import albumentations as A
import cv2








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
        # 随机DropBlock
        self.CoarseDropout = A.Compose([
                A.CoarseDropout(max_holes=60, max_height=15, max_width=15, min_holes=30, min_height=5, min_width=5, fill_value=128, p=0.5),
        ])
        # 训练时增强
        self.trainTF = A.Compose([
                A.BBoxSafeRandomCrop(p=0.5),
                # A.RandomSizedBBoxSafeCrop(800, 800, erosion_rate=0.0, interpolation=1, p=0.5),
                # 随机翻转
                A.HorizontalFlip(p=0.5),
                # NOTE:下面这两个只能在DOTA上用:
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
