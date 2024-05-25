# train eval test
MODE = 'test'
# mobilenetv3_large_100.ra_in1k  resnet50.a1_in1k  darknetaa53.c2ns_in1k cspdarknet53.ra_in1k cspresnext50.ra_in1k
BACKBONE = 'resnet50.a1_in1k'
FROZEBACKBONE = True
CATNUMS = 20
TESTCKPT = "ckpt/best_VOC_wsddn.pt"
IMGSIZE = [832, 832]

# VOC
# train_json_path = 'E:/datasets/Universal/VOC2007/WSDDN_COCO_format/VOC07_trainval.json'
# val_json_path =   'E:/datasets/Universal/VOC2007/WSDDN_COCO_format/VOC07_test.json'
# train_img_dir =   'E:/datasets/Universal/VOC2007/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages'
# val_img_dir   =   'E:/datasets/Universal/VOC2007/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages'
# cat_names = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", 
#                 "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
# cat_map = None
# reverse_map = None



# cat&dog
train_json_path = 'E:/datasets/Universal/VOC2007/WSDDN_COCO_format/VOC07_trainval.json'
val_json_path =   'E:/datasets/Universal/VOC2007/WSDDN_COCO_format/VOC07_test.json'
train_img_dir =   'E:/datasets/Universal/VOC2007/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages'
val_img_dir   =   'E:/datasets/Universal/VOC2007/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages'
cat_names = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", 
                "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
cat_map = None
reverse_map = None





runner = dict(
    seed = 22,
    mode = MODE,
    resume = 'ckpt/best_VOC_wsddn.pt',
    img_size = IMGSIZE,
    epoch = 52,
    log_dir = './log',
    log_interval = 1,
    eval_interval = 1,
    reverse_map = None,
    class_names = cat_names, 

    dataset = dict(
        bs = 8,
        num_workers = 2,
        # 自定义的Dataset:
        my_dataset = dict(
            path = 'datasets/WSDDNDataset.py',
            train_dataset = dict(
                annPath = train_json_path, 
                imgDir = train_img_dir,
                map = cat_map,
                inputShape = IMGSIZE, 
                trainMode=True, 
            ),
            val_dataset = dict(
                annPath = val_json_path, 
                imgDir = val_img_dir,
                map = cat_map,
                inputShape = IMGSIZE, 
                trainMode=False,                 
            ),
        ),
    ),

    model = dict(
        path = 'models/WSDDN/WSDDN.py',
        backbone_name = BACKBONE,
        loadckpt = 'ckpt/backbone_resnet50.a1_in1k.pt',
        backbone = dict(
            modelType = BACKBONE,
            loadckpt = None,
            pretrain = False,
            froze = FROZEBACKBONE,            
        ),
        mil_head = dict(
            backbone_name = BACKBONE,
            catNums = CATNUMS, 
            roiSize = 7,
        ),
    ),

    optimizer = dict(
        optim_type = 'adamw',
        lr = 1e-4,
        lr_min_ratio = 0.1,
        warmup_lr_init_ratio = 0.01,
    ),
)

eval = dict(
    inferring = True,
    pred_json_name = 'eval_tmp.json',
    ckpt_path = TESTCKPT,
    T = 0.01,        
)

test = dict(
    # ./images/4.jpg   
    # E:/datasets/Universal/VOC2007/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages/000537.jpg
    # "E:/datasets/Universal/COCO2017/unlabeled2017/000000001234.jpg" 2382 2000 5611 1356 1800 1808 2548 
    # "E:/datasets/Universal/COCO2017/test2017/000000007841.jpg" 8359 24526 35892
    img_path = "E:/datasets/Universal/VOC2007/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages/002407.jpg",
    save_vis_path = './images/res.jpg',
    ckpt_path = TESTCKPT,
    T = 0.3,
)