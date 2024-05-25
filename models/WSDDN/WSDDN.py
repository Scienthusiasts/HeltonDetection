import os
import timm
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import roi_pool

from datasets.WSDDNDataset import *
from models.WSDDN.Backbone import *
from models.WSDDN.MILHead import *
from utils.wsddnUtils import *

class Model(nn.Module):
    """WSDDN网络"""
    def __init__(self, loadckpt, backbone:dict, mil_head:dict, **kwargs):
        super(Model, self).__init__()
        # 允许调换不同的backbone
        self.backbone = Backbone(**backbone)
        self.MILHead = MILHead(**mil_head)

        # 是否导入预训练权重
        if loadckpt!=None: 
            self.load_state_dict(torch.load(loadckpt))
            print('wsddn pretrain ckpt loaded!')


    def forward(self, img, proposals):
        """前向
            Args:
                :param img:       图像   [BS=1, C=3, W, H]
                :param proposals: 先验框 [numProposalPerImg, 4]

            Retuens:
                :param cls_logit: 分类分支的原始预测结果
                :param ins_logit: 实例分支的原始预测结果
        """

        # proposals默认batch为1 torch.Size([1, 1022, 4])
        proposals = torch.tensor(np.array(proposals), device=img.device, dtype=torch.float32)
        # 图像先经过backbone得到feature map [BS, C, featW, featH]
        backbone_feat = self.backbone(img)[2]  
        cls_logit, ins_logit = self.MILHead(backbone_feat, proposals)
        return cls_logit, ins_logit


    def calcBatchLoss(self, img_size, batch_imgs, batch_bboxes, batch_labels):
        """一个batch的前向流程(不包括反向传播, 更新梯度)(核心, 要更改训练pipeline主要改这里)
            Args:
                :param img:       图像   [BS=1, C=3, W, H]
                :param proposals: 先验框 [numProposalPerImg, 4]

            Retuens:

        """
        losses = {}
        total_loss = 0
        # 由于bs只能为1, 因此遍历batch中的每张图, 一个batch后再反向(伪batch)
        for img, bboxes, labels in zip(batch_imgs, batch_bboxes, batch_labels):
            img = img.unsqueeze(0)
            # 前向
            cls_logit, ins_logit = self.forward(img, bboxes)
            combined_score = F.softmax(cls_logit, dim=1) * F.softmax(ins_logit, dim=0)
            combined_score = combined_score.sum(dim=0)
            combined_score = torch.clamp(combined_score, 0., 1.)
            # 将标签转化为onehot形式，但是是多分类(存在多个1)
            onehot_label = torch.zeros(combined_score.shape[0])
            onehot_label[labels] = 1
            onehot_label = onehot_label.to(batch_imgs.device)
            # 计算分类损失(不包含sigmoid)
            total_loss += F.binary_cross_entropy(combined_score, onehot_label, reduction='sum')

        losses.update({ 'total_loss':total_loss / batch_imgs.shape[0]}) 
        return losses
    

    def inference(self, device, class_names, image2color, img_size, tf, img_path, save_vis_path=None, ckpt_path=None, T=0.3):
        '''推理一张图
            Args:
                :param img_path:      图片路径
                :param save_vis_path: 可视化图像保存路径

            Returns:
                :param boxes:       网络回归的box坐标    [obj_nums, 4]
                :param box_scores:  网络预测的box置信度  [obj_nums]
                :param box_classes: 网络预测的box类别    [obj_nums]
        '''
        if ckpt_path != None:
            self.load_state_dict(torch.load(ckpt_path))
            self.eval()

        image = Image.open(img_path).convert('RGB')
        proposals, _ = selectiveSearch(np.array(image))
        normal_trans = tf.normalTF(image=np.array(image), bboxes=proposals, category_ids=np.zeros(len(proposals)))
        # tensor_img有padding的黑边
        # 注意permute(2,0,1) 不要写成permute(2,1,0)
        tensor_img = torch.tensor(normal_trans['image']).permute(2,0,1).unsqueeze(0).to(device)
        proposals = np.array(normal_trans['bboxes'])
        print(proposals.shape)
        W, H = image.size
        with torch.no_grad():
            cls_logit, ins_logit = self.forward(tensor_img, proposals)
            combined_score = F.softmax(cls_logit, dim=1) * F.softmax(ins_logit, dim=0)
            # 每个proposal对应的置信度最大的那个类别:
            ins_cat_id = torch.argmax(cls_logit, dim=1).cpu().numpy()
            # 分类结果:
            multi_classify = torch.clamp(combined_score.sum(dim=0), 0., 1.)
            print(multi_classify)
            img = np.array(image)
            for cat_id in range(multi_classify.shape[0]):
                # if cat_id!=6:continue
                if multi_classify[cat_id]<0.1:continue
                print(cat_id)
                proposals_per_cat = proposals[np.where(ins_cat_id==cat_id)]
                proposals_per_cat = mapBox2OriginalImg(proposals_per_cat, W, H, max(img_size))
                for box in proposals_per_cat:
                    x, y, w, h = round(box[0]), round(box[1]), round(box[2]), round(box[3])
                    # print(x, y, w, h)
                    cv2.rectangle(img, (x, y), (x+w, y+h), color=(0,255,0), thickness=1)
                cv2.imshow('ss', img)
                cv2.waitKey(0)



            return 
            #  检测出物体才继续    
            if len(boxes) == 0: 
                print(f'no objects in the images: {img_path}.')
                return boxes, box_scores, box_classes
            # 将box坐标(对应有黑边的图)映射回无黑边的原始图像
            boxes = mapBox2OriginalImg(boxes, w, h, max(img_size))
            '''画框'''
            if save_vis_path!=None:
                PltDrawBox(image, boxes, box_classes, box_scores, save_vis_path, image2color, class_names)

            return boxes, box_scores, box_classes











if __name__ == '__main__':
    device = 'cuda'
    # 固定随机种子
    seed = 22
    # batchsize大小
    BS = 1
    # 图像大小
    imgSize = [832, 832]
    # x训练集还是测试集 test/trainval
    mode='trainval'  
    AnnPath = f"E:/datasets/Universal/VOC2007/WSDDN_COCO_format/VOC07_{mode}.json"
    imgDir = f"E:/datasets/Universal/VOC2007/VOC{mode}_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages"
    # 定义数据集读取
    Dataset = COCODataset(AnnPath, imgDir, imgSize, trainMode=True)
    DataLoader = DataLoader(Dataset, shuffle=True, batch_size = BS, num_workers=2, pin_memory=True, collate_fn=COCODataset.dataset_collate, worker_init_fn=partial(COCODataset.worker_init_fn, seed=seed))
    print(f'数据集大小 : {Dataset.__len__()}')
    backbone_dict = {'modelType':'resnet50.a1_in1k', 'loadckpt':False, 'pretrain':False, 'froze':True}
    milhead_dict = {'backbone_name':'resnet50.a1_in1k', 'catNums':20, 'roiSize':7}
    net = Model(backbone_dict, milhead_dict).to(device)
    # 模拟读取
    for step, batch in enumerate(DataLoader):
        images, boxes, labels = batch[0], batch[1], batch[2]
        images = images.to(device)
        boxes = boxes[0]
        cls_logit, ins_logit = net(images, boxes)
        print(cls_logit.shape, ins_logit.shape)
        break




