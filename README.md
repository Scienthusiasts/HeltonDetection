# HeltonDetection
从0开始，搭建一个深度学习目标检测框架！(基于Pytorch)


**24/5/25 更新 :**

发布`dev1.0`分支 

- FasterRCNN-FPN(FPN, PAFPN(YOLOv8), Decoupled-Head, 可更换Backbone, 或自定义neck, head和其中的组件)
- YOLOv5(PAFPN(YOLOv5), 可更换Backbone, 或自定义neck, head和其中的组件)
- COCO格式数据集读取, YOLO格式数据集读取(eval未实现), DOTA格式数据集实现(debug中), 丰富的数据增强方法

**24/6/5 更新 :**

- 添加了TTA策略, 并基于WBF(Weighted Boxes Fusion)进行融合

**demo**

![1](https://github.com/Scienthusiasts/HeltonDetection/blob/dev/demo/1.jpg)

![2](https://github.com/Scienthusiasts/HeltonDetection/blob/dev/demo/2.jpg)


**reference**

[bubbliiiing/faster-rcnn-pytorch: 这是一个faster-rcnn的pytorch实现的库，可以利用voc数据集格式的数据进行训练。 (github.com)](https://github.com/bubbliiiing/faster-rcnn-pytorch)

[bubbliiiing/yolov5-v6.1-pytorch: 这是一个yolov5-v6.1-pytorch的源码，可以用于训练自己的模型。 (github.com)](https://github.com/bubbliiiing/yolov5-v6.1-pytorch)

[ultralytics/yolov5: YOLOv5 🚀 in PyTorch > ONNX > CoreML > TFLite (github.com)](https://github.com/ultralytics/yolov5)

[open-mmlab/mmdetection: OpenMMLab Detection Toolbox and Benchmark (github.com)](https://github.com/open-mmlab/mmdetection)

