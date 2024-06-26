# HeltonDetection
从0开始，搭建一个深度学习目标检测框架！(基于Pytorch，不定期更新) 

**24/5/25 更新 :**

发布`dev`分支 

- FasterRCNN-FPN(FPN, PAFPN(YOLOv8), Decoupled-Head, 可更换Backbone, 或自定义neck, head和其中的组件)
- YOLOv5(PAFPN(YOLOv5), 可更换Backbone, 或自定义neck, head和其中的组件)
- COCO格式数据集读取, YOLO格式数据集读取(eval未实现), DOTA格式旋转框数据集读取(debug中), 丰富的数据增强方法
- 支持config文件配置模型、训练、评估、测试超参，命令行一键训练/评估/测试
- 支持warmup+余弦学习率衰减策略
- 支持logger， tensorboard等日志记录方式
- 支持图像/视频推理

**24/6/5 更新 :**

- 添加了TTA策略, 并基于WBF(Weighted Boxes Fusion)进行Bboxes融合

**24/6/23 更新 :**

- 为YOLOv5添加了onnx export和onnx-runtime推理
- eval时添加了FlOPs和模型参数量Params指标
- 添加了`utils\otherUtils\eval_yolov5_by_pycocotools.py`用于对Ultralytics官方提供的YOLOv5模型进行评估

**24/6/26 更新 :**

- **支持pytorch DDP多GPU分布式训练！** 并调整了相关代码逻辑，支持单卡/多卡训练

## Demo

![1](https://github.com/Scienthusiasts/HeltonDetection/blob/dev/demo/1.jpg)

![2](https://github.com/Scienthusiasts/HeltonDetection/blob/dev/demo/2.jpg)

![3](https://github.com/Scienthusiasts/HeltonDetection/blob/dev/demo/3.jpg)

## Environments

```
整理与完善中...
```



## Train/Eval/Test

```
整理与完善中...
```



## Experiment

基于`pycocotools`提供的接口进行评估；默认使用warmup+cos学习率衰减策略

### FasterRCNN

默认使用ResNet50+RoIAlign；PAFPN基于YOLOv8的PAFPN结构，输出通道统一为256(备注P2代表Head部分只使用金字塔P2的特征)

- `VOC0712`

image-size=[832, 832]

|                    Model                    | Mosaic |     lr      | epoch |  bs  |  AP50(%)   | mAP50-95(%) |
| :-----------------------------------------: | :----: | :---------: | :---: | :--: | :--------: | :---------: |
|         FasterRCNN-FPNP2-RoIPooling         | **×**  | 2e-4(adamw) |  36   |  16  |   77.501   |   47.539    |
|              FasterRCNN-FPNP2               | **×**  | 2e-4(adamw) |  36   |  16  |   78.383   |   49.662    |
|             FasterRCNN-PAFPNP2              | **×**  | 2e-4(adamw) |  36   |  12  |   78.887   |   54.085    |
|      FasterRCNN-PAFPNP2-DecoupledHead       | **×**  | 2e-4(adamw) |  36   |  16  |   79.668   |   55.152    |
|      FasterRCNN-PAFPNP2-DecoupledHead       | p=0.5  | 2e-4(adamw) |  36   |  16  |   81.835   |   58.116    |
|       FasterRCNN-PAFPN-DecoupledHead        | p=0.5  | 2e-4(adamw) |  36   |  16  |   81.784   |   58.527    |
| FasterRCNN-PAFPN-DecoupledHead-COCOPretrain | p=0.5  | 2e-4(adamw) |  36   |  16  | **85.204** | **63.817**  |

- `COCO2017`

image-size=[832, 832]

|              Model               | Mosaic |     lr      | epoch |  bs  |  AP50(%)   | mAP50-95(%) |
| :------------------------------: | :----: | :---------: | :---: | :--: | :--------: | :---------: |
| FasterRCNN-PAFPNP2-DecoupledHead | **×**  | 2e-4(adamw) |  36   |  16  |   58.064   |   39.377    |
|  FasterRCNN-PAFPN-DecoupledHead  | p=0.5  | 2e-4(adamw) |  36   |  16  | **62.182** | **42.513**  |

- `VisDrone2019`

image-size=[1280, 1280]

|             Model              | Mosaic |     lr      | epoch |  bs  |  AP50(%)   | mAP50-95(%) |
| :----------------------------: | :----: | :---------: | :---: | :--: | :--------: | :---------: |
| FasterRCNN-PAFPN-DecoupledHead | p=0.5  | 1e-4(adamw) |  36   |  8   | **37.175** | **21.164**  |

### YOLOv5

- `VOC0712`

image-size=[640, 640]

|            Model            | Mosaic |     lr      | epoch |  bs  |  AP50(%)   | mAP50-95(%) |
| :-------------------------: | :----: | :---------: | :---: | :--: | :--------: | :---------: |
|           YOLOv5s           | **×**  | 1e-3(adamw) |  48   |  16  |   69.324   |   44.595    |
|           YOLOv5s           | p=0.5  | 1e-3(adamw) |  48   |  16  |   71.852   |   46.374    |
|   YOLOv5s-focalloss(root)   | p=0.5  | 1e-3(adamw) |  48   |  16  |   72.709   |   46.741    |
| YOLOv5s-focalloss(root_cls) | p=0.5  | 1e-3(adamw) |  48   |  16  |   73.095   |   46.017    |
|           YOLOv5s           | p=1.0  | 1e-3(adamw) |  48   |  16  |   63.649   |   35.859    |
|   YOLOv5l-timm_cspdarknet   | p=0.5  | 1e-3(adamw) |  48   |  16  |   73.305   |   49.557    |
|           YOLOv5l           | p=0.5  | 1e-3(adamw) |  48   |  16  | **74.341** | **50.417**  |

- `COCO2017`

image-size=[640, 640]

|           Model           | Mosaic |     lr      | epoch |  bs  |  AP50(%)   | mAP50-95(%) |
| :-----------------------: | :----: | :---------: | :---: | :--: | :--------: | :---------: |
|          YOLOv5s          | **×**  | 1e-3(adamw) |  48   |  16  |   47.401   |   29.663    |
|          YOLOv5s          | p=0.5  | 1e-3(adamw) |  48   |  16  |   48.678   |   30.148    |
| YOLOv5s (**Ultralytics**) |   -    |      -      |   -   |  -   |   45.120   |   30.928    |
|          YOLOv5l          | p=0.5  | 1e-3(adamw) |  48   |  16  | **57.808** |   39.717    |
| YOLOv5l (**Ultralytics**) |   -    |      -      |   -   |  -   |   56.170   | **42.015**  |

- `VisDrone2019`

image-size=[1280, 1280]

|          Model          | Mosaic |     lr      | epoch |  bs  |  AP50(%)   | mAP50-95(%) |
| :---------------------: | :----: | :---------: | :---: | :--: | :--------: | :---------: |
|         YOLOv5s         | p=0.5  | 1e-3(adamw) |  48   |  16  |   32.210   |   17.681    |
| YOLOv5s-focalloss(root) | p=0.5  | 1e-3(adamw) |  48   |  16  |   33.050   |   18.131    |
| YOLOv5l-focalloss(root) | p=0.5  | 1e-3(adamw) |  48   |  16  | **39.029** | **22.589**  |

- `DOTAv1.0-h`

image-size=[1024, 1024]

|          Model          | Mosaic |     lr      | epoch |  bs  |  AP50(%)   | mAP50-95(%) |
| :---------------------: | :----: | :---------: | :---: | :--: | :--------: | :---------: |
|         YOLOv5s         | p=0.5  | 1e-3(adamw) |  48   |  16  |   64.349   |   39.500    |
| YOLOv5s-focalloss(root) | p=0.5  | 1e-3(adamw) |  48   |  16  | **65.174** | **39.257**  |

## reference

[bubbliiiing/faster-rcnn-pytorch: 这是一个faster-rcnn的pytorch实现的库，可以利用voc数据集格式的数据进行训练。 (github.com)](https://github.com/bubbliiiing/faster-rcnn-pytorch)

[bubbliiiing/yolov5-v6.1-pytorch: 这是一个yolov5-v6.1-pytorch的源码，可以用于训练自己的模型。 (github.com)](https://github.com/bubbliiiing/yolov5-v6.1-pytorch)

[ultralytics/yolov5: YOLOv5 🚀 in PyTorch > ONNX > CoreML > TFLite (github.com)](https://github.com/ultralytics/yolov5)

[open-mmlab/mmdetection: OpenMMLab Detection Toolbox and Benchmark (github.com)](https://github.com/open-mmlab/mmdetection)

