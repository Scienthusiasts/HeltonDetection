# train eval test
MODE = 'train'
# mobilenetv3_large_100.ra_in1k  resnet50.a1_in1k  darknetaa53.c2ns_in1k cspdarknet53.ra_in1k cspresnext50.ra_in1k
FROZEBACKBONE = True
# log_yolov5_VOC_mosaic_0.5_focalloss_obj_root_cls  log_yolov5_VOC_mosaic_0.5_focalloss_root_obj_root_cls_balance_4_1_0.4 
# ./log/yolo/log_yolov5{PHI}_COCO_mosaic_0.5/best_mAP.pt /log/yolo/log_yolov5s_visDrone_mosaic_0.5_root_focalloss/best_mAP.pt
PHI = 's'
# 'last.pt' 
TESTCKPT = 'last.pt'
BACKBONE = f'ckpt/yolov8_{PHI}_backbone_weights.pth'
LOADCKPT = 'last.pt'
RESUME = False
# [640, 640] [832, 832] [1024, 1024] [1280, 1280] [2048, 2048]
IMGSIZE = [640, 640]
TTA = [[640,640], [832,832], [960,960]]
TTAOPEN = False

'''VOC'''
# CATNUMS = 20
# train_json_path = 'E:/datasets/Universal/VOC0712/VOC2007/Annotations/coco/train.json'
# val_json_path =   'E:/datasets/Universal/VOC0712/VOC2007/Annotations/coco/test.json'
# train_img_dir =   'E:/datasets/Universal/VOC0712/VOC2007/JPEGImages'
# val_img_dir   =   'E:/datasets/Universal/VOC0712/VOC2007/JPEGImages'
# cat_names = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", 
#                 "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
# cat_map = None
# reverse_map = None

'''COCO'''
CATNUMS = 80
train_json_path = 'E:/datasets/Universal/COCO2017/COCO/annotations/instances_train2017.json'
val_json_path =   'E:/datasets/Universal/COCO2017/COCO/annotations/instances_val2017.json'
train_img_dir =   'E:/datasets/Universal/COCO2017/COCO/train2017'
val_img_dir =     'E:/datasets/Universal/COCO2017/COCO/val2017'
cat_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
           'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
           'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
           'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
           'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
           'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
cat_map = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7, 9:8, 10:9, 11:10, 13:11, 14:12, 15:13, 16:14, 17:15, 18:16, 19:17, 20:18, 21:19, 22:20, 23:21, 
       24:22, 25:23, 27:24, 28:25, 31:26, 32:27, 33:28, 34:29, 35:30, 36:31, 37:32, 38:33, 39:34, 40:35, 41:36, 42:37, 43:38, 44:39, 46:40, 
       47:41, 48:42, 49:43, 50:44, 51:45, 52:46, 53:47, 54:48, 55:49, 56:50, 57:51, 58:52, 59:53, 60:54, 61:55, 62:56, 63:57, 64:58, 65:59, 
       67:60, 70:61, 72:62, 73:63, 74:64, 75:65, 76:66, 77:67, 78:68, 79:69, 80:70, 81:71, 82:72, 84:73, 85:74, 86:75, 87:76, 88:77, 89:78, 90:79}
reverse_map = {0:1, 1:2, 2:3, 3:4, 4:5, 5:6, 6:7, 7:8, 8:9, 9:10, 10:11, 11:13, 12:14, 13:15, 14:16, 15:17, 16:18, 17:19, 18:20, 19:21, 20:22, 21:23, 
       22:24, 23:25, 24:27, 25:28, 26:31, 27:32, 28:33, 29:34, 30:35, 31:36, 32:37, 33:38, 34:39, 35:40, 36:41, 37:42, 38:43, 39:44, 40:46, 
       41:47, 42:48, 43:49, 44:50, 45:51, 46:52, 47:53, 48:54, 49:55, 50:56, 51:57, 52:58, 53:59, 54:60, 55:61, 56:62, 57:63, 58:64, 59:65, 
       60:67, 61:70, 62:72, 63:73, 64:74, 65:75, 66:76, 67:77, 68:78, 69:79, 70:80, 71:81, 72:82, 73:84, 74:85, 75:86, 76:87, 77:88, 78:89, 79:90}

'''visDrone2019'''
# CATNUMS = 10
# train_json_path = 'E:/datasets/RemoteSensing/visdrone2019/annotations/train.json'
# val_json_path =   'E:/datasets/RemoteSensing/visdrone2019/annotations/test.json'
# train_img_dir =   'E:/datasets/RemoteSensing/visdrone2019/images/train/images'
# val_img_dir   =   'E:/datasets/RemoteSensing/visdrone2019/images/test/images'
# cat_names = ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"]
# cat_map = None
# reverse_map = None

'''DOTA'''
# CATNUMS = 15
# train_json_path = 'E:/datasets/RemoteSensing/DOTA-1.0_ss_1024/coco_ann/hbox_train.json'
# val_json_path =   'E:/datasets/RemoteSensing/DOTA-1.0_ss_1024/coco_ann/hbox_val.json'
# train_img_dir =   'E:/datasets/RemoteSensing/DOTA-1.0_ss_1024/train/images'
# val_img_dir   =   'E:/datasets/RemoteSensing/DOTA-1.0_ss_1024/val/images'
# cat_names = ['PL', 'BD', 'BR', 'GTF', 'SV', 'LV', 'SH', 'TC', 'BC', 'ST', 'SBF', 'RA', 'HB', 'SP', 'HC']
# cat_map = None
# reverse_map = None






runner = dict(
    seed = 22,
    mode = MODE,
    resume = RESUME,
    img_size = IMGSIZE,
    epoch = 12*4,
    log_dir = './log/yolov8',
    log_interval = 1,
    eval_interval = 1,
    reverse_map = reverse_map,
    class_names = cat_names, 

    dataset = dict(
        bs = 8*2,
        num_workers = 0,
        # 自定义的Dataset:
        my_dataset = dict(
            path = 'datasets/YOLOv8Dataset.py',
            train_dataset = dict(
                num_classes = CATNUMS,
                annPath = train_json_path, 
                imgDir = train_img_dir,
                map = cat_map,
                inputShape = IMGSIZE, 
                trainMode=True, 
            ),
            val_dataset = dict(
                num_classes = CATNUMS,
                annPath = val_json_path, 
                imgDir = val_img_dir,
                map = cat_map,
                inputShape = IMGSIZE, 
                trainMode=False,                 
            ),
        ),
    ),

    model = dict(
        path = 'models/YOLOv8/YOLOv8.py',
        num_classes = CATNUMS, 
        phi = PHI, 
        img_size = IMGSIZE,
        loadckpt = LOADCKPT,           
        backbone_name = BACKBONE,
        tta_img_size = TTA,
        backbone = dict(
            loadckpt=BACKBONE, 
            pretrain=False, 
            froze=FROZEBACKBONE,
        ),
        # backbone = dict(
        #     modelType = 'cspdarknet53.ra_in1k',
        #     loadckpt = './ckpt/cspdarknet53.ra_in1k.pt',
        #     pretrain = False,
        #     froze = FROZEBACKBONE,            
        # ),
    ),
    test = dict(
        # 是否半精度推理
        half = False,
        tta = TTAOPEN,
    ),
    optimizer = dict(
        optim_type = 'adamw',
        lr = 1e-3,
        lr_min_ratio = 0.1,
        warmup_lr_init_ratio = 0.01,
    ),
)

eval = dict(
    fuse = False,
    inferring = True,
    pred_json_name = 'eval_tmp.json',
    ckpt_path = TESTCKPT,
    T = 0.01,        
)

test = dict(
    # image video
    mode = 'image',
    # ./samples/imgs/3.jpg   
    # "E:/datasets/RemoteSensing/visdrone2019/images/test/images/1.jpg"
    # sal/COCO2017/unlabeled2017/000000001234.jpg" 2382 2000 5611 1356 1800 1808 2548 
    # E:/datasets/RemoteSensing/visdrone2019/images/test/images/0000087_00009_d_0000001.jpg
    # '''DOTA'''
    # "E:/datasets/RemoteSensing/DOTA-1.0_ss_1024/val/images/P0019__1024__4608___0.png" P0086__1024__0___0.png 
    # P0168__1024__1024___512.png P0262__1024__512___0.png P0476__1024__122___205.png P0660__1024__136___0.png
    # P0833__1024__617___0.png
    path = "./samples/imgs/cars.jpg",
    save_vis_path = './samples/imgs/res1.jpg',
    # video
    # path = "./samples/videos/cars.mp4",
    # save_vis_path = './samples/videos/res1.mp4',
    ckpt_path = TESTCKPT,
    T = 0.20,
    agnostic = False,
    show_text = True,
    vis_heatmap = True,
)