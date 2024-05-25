# coding=utf-8
import os
import json
import torch
import logging
import datetime
import numpy as np
from torch import nn
from tqdm import tqdm
from PIL import Image
import torch.optim as optim
from functools import partial
from pycocotools.coco import COCO
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from timm.scheduler import CosineLRScheduler


# 自定义模块
from test import Test
from utils.util import *
from utils.FasterRCNNAnchorUtils import *
from torch.utils.tensorboard import SummaryWriter






class ArgsHistory():
    '''记录train或val过程中的一些变量(比如 loss, lr等) 以及tensorboard
    '''
    def __init__(self, json_save_dir):
        # tensorboard 对象
        self.tb_writer = SummaryWriter(log_dir=json_save_dir)
        self.json_save_dir = json_save_dir
        self.args_history_dict = {}

    def record(self, key, value):
        '''记录args
        Args:
            - key:   要记录的当前变量的名字
            - value: 要记录的当前变量的数值
            
        Returns:
            None
        '''
        # 可能存在json格式不支持的类型, 因此统一转成float类型
        value = float(value)
        # 如果日志中还没有这个变量，则新建
        if key not in self.args_history_dict.keys():
            self.args_history_dict[key] = []
        # 更新history dict
        self.args_history_dict[key].append(value)
        # 更新tensorboard
        self.tb_writer.add_scalar(key, value, len(self.args_history_dict[key]))

    def saveRecord(self):
        '''以json格式保存args
        '''
        if not os.path.isdir(self.json_save_dir):os.makedirs(self.json_save_dir) 
        json_save_path = os.path.join(self.json_save_dir, 'args_history.json')
        # 保存
        with open(json_save_path, 'w') as json_file:
            json.dump(self.args_history_dict, json_file)

    def loadRecord(self, json_load_dir):
        '''导入上一次训练时的args(一般用于resume)
        '''
        json_path = os.path.join(json_load_dir, 'args_history.json')
        with open(json_path, "r", encoding="utf-8") as json_file:
            self.args_history_dict = json.load(json_file)







class Runner():
    '''训练/验证/推理时的流程'''
    def __init__(self, seed:int, mode:str, class_names:list, img_size:list, epoch:int, resume:str, log_dir:str, log_interval:int, eval_interval:int, reverse_map:dict, 
                 dataset:dict, test:dict, model:dict, optimizer:dict):
        '''Runner初始化
        Args:
            - mode:            当前模式是训练/验证/推理
            - backbone_name:   骨干网络名称
            - froze_backbone:  训练时是否冻结Backbone
            - img_size:        统一图像尺寸的大小
            - class_names:     数据集类别名称
            - train_json_path: 训练集json文件路径
            - val_json_path:   验证集json文件路径
            - train_img_dir:   训练集图像路径
            - val_img_dir:     验证集图像路径
            - eopch:           训练批次
            - bs:              训练batch size
            - lr:              学习率
            - log_dir:         日志文件保存目录
            - log_interval:    训练或验证时隔多少bs打印一次日志
            - optim_type:      优化器类型
            - load_ckpt:
            - resume:          是否恢复断点训练
            - seed:            固定随机种子
            - map:             数据集类别id映射字典(COCO)(json文件里的id->按顺序的id)
            - reverse_map:     数据集类别id逆映射字典(COCO)(按顺序的id->json文件里的id)

        Returns:
            None
        '''
        # 设置全局种子
        self.seed = seed
        self.mode = mode
        self.resume = resume
        seed_everything(self.seed)
        self.class_names = class_names
        self.cat_nums = len(class_names)
        self.img_size = img_size
        self.epoch = epoch
        self.log_dir = log_dir
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.reverse_map = reverse_map
        '''GPU/CPU'''
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        '''日志模块'''
        if mode in ['train', 'eval']:
            self.logger, self.log_save_path = self.myLogger()
            '''训练/验证时参数记录模块'''
            json_save_dir, _ = os.path.split(self.log_save_path)
            self.argsHistory = ArgsHistory(json_save_dir)
        '''导入数据集'''
        self.loadDatasets(**dataset)
        '''导入网络'''
        # 根据模型名称动态导入模块
        self.model = dynamic_import_class(model.pop('path'), 'Model')(**model).to(self.device)
        cudnn.benchmark = True
            

        '''定义优化器(自适应学习率的带动量梯度下降方法)'''
        if mode == 'train':
            self.optimizer, self.scheduler = self.optimSheduler(**optimizer)
        '''当恢复断点训练'''
        self.start_epoch = 0
        if self.resume != None and self.mode=='train':
            self.trainResume()
        '''导入评估模块'''
        self.test = Test(dataset['my_dataset']['path'], self.model, self.img_size, self.class_names, self.device, **test)
        '''打印训练参数'''
        self.printRunnerArgs(model['backbone_name'])
        # torch.save(self.model.state_dict(), "./tmp.pt")



    def printRunnerArgs(self, backbone_name):
        if self.mode in ['train', 'eval']:
            self.logger.info(f'CPU/CUDA:   {self.device}')
            self.logger.info(f'骨干网络: {backbone_name}')
            self.logger.info(f'全局种子: {self.seed}')
            self.logger.info(f'图像大小:   {self.img_size}')
            if self.mode == 'train':
                self.logger.info('训练集大小:   %d' % self.train_data.__len__())
            if self.mode in ['train', 'eval']:
                self.logger.info('验证集大小:   %d' % self.val_data.__len__())
                self.logger.info('数据集类别数: %d' % self.cat_nums)
            if self.mode == 'train':
                self.logger.info(f'优化器: {self.optimizer}')

            self.logger.info('='*150)        



    def trainResume(self):
        checkpoint = torch.load(self.resume)
        self.start_epoch = checkpoint['epoch'] + 1 # +1是因为从当前epoch的下一个epoch开始训练
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optim_state_dict'])
        # self.scheduler.load_state_dict(checkpoint['sched_state_dict'])
        self.logger.info(f'resume:{self.resume}')
        self.logger.info(f'start_epoch:{self.start_epoch}')
        # 导入上一次中断训练时的args
        json_dir, _ = os.path.split(self.resume)
        self.argsHistory.loadRecord(json_dir)



    def loadDatasets(self, bs, num_workers, my_dataset:dict):
        '''导入数据集
            Args:
                - seed:当前模式是训练/验证/推理
                - val_json_path 
                - val_img_dir
                - train_json_path
                - train_img_dir
                - map

            Returns:
            None
        '''
        # 动态导入机制和多进程存在冲突, 因此不能动态导入
        # COCODataset = dynamic_import_class(my_dataset['path'], 'COCODataset')
        
        dataset_type = my_dataset['path'].split('/')[-1][:-3]
        # 退而求其次, 用判断条件动态的导入
        if dataset_type == 'FasterRCNNDataset':
            from datasets.FasterRCNNDataset import COCODataset
        if dataset_type == 'YOLODataset':
            from datasets.YOLODataset import COCODataset
        if dataset_type == 'WSDDNDataset':
            from datasets.WSDDNDataset import COCODataset

        if self.mode != 'test':
            # 导入验证集
            self.val_json_path = my_dataset['val_dataset']['annPath']
            self.val_img_dir = my_dataset['val_dataset']['imgDir']
            # 固定每个方法里都有一个COCODataset
            self.val_data = COCODataset(**my_dataset['val_dataset'])
            self.val_data_loader = DataLoader(self.val_data, shuffle=False, batch_size=bs, num_workers=num_workers, pin_memory=True, 
                                        collate_fn=COCODataset.dataset_collate, worker_init_fn=partial(COCODataset.worker_init_fn, seed=self.seed))   
                 
        if self.mode == 'train':
            # 导入训练集
            self.train_data = COCODataset(**my_dataset['train_dataset'])
            self.train_data_loader = DataLoader(self.train_data, shuffle=True, batch_size=bs, num_workers=num_workers, pin_memory=True,
                                            collate_fn=COCODataset.dataset_collate, worker_init_fn=partial(COCODataset.worker_init_fn, seed=self.seed))




    def optimSheduler(self, optim_type:str, lr:float, lr_min_ratio:float, warmup_lr_init_ratio:float):
        '''定义优化器和学习率衰减策略
        '''
        optimizer = {
            # adam会导致weight_decay错误，使用adam时建议设置为 0
            'adamw' : optim.AdamW(self.model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0),
            'adam' : optim.Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0),
            'sgd'  : optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=0)
        }[optim_type]
        # 使用warmup+余弦退火学习率
        scheduler = CosineLRScheduler(
            optimizer=optimizer,
            # 总迭代数
            t_initial=self.epoch*len(self.train_data_loader),  
            # 余弦退火最低的学习率        
            lr_min=lr*lr_min_ratio,               
            # 学习率预热阶段的epoch数量                        
            warmup_t=2*len(self.train_data_loader), 
            # 学习率预热阶段的lr起始值
            warmup_lr_init=lr*warmup_lr_init_ratio,                               
        )

        return optimizer, scheduler




    def myLogger(self):
        '''生成日志对象
        '''
        logger = logging.getLogger('runer')
        logger.setLevel(level=logging.DEBUG)
        # 日志格式
        formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')

        if self.mode == 'train':
            # 写入文件的日志
            self.log_dir = os.path.join(self.log_dir, f"{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_train")
            # 日志文件保存路径
            log_save_path = os.path.join(self.log_dir, f"{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_train.log")
        if self.mode == 'eval':
            # 写入文件的日志
            self.log_dir = os.path.join(self.log_dir, f"{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_val")
             # 日志文件保存路径
            log_save_path = os.path.join(self.log_dir, f"{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_val.log")
        if not os.path.isdir(self.log_dir):os.makedirs(self.log_dir)
        file_handler = logging.FileHandler(log_save_path, encoding="utf-8", mode="a")
        file_handler.setLevel(level=logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        # 终端输出的日志
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        return logger, log_save_path



    def recoardArgs(self, mode, losses=None, mAP=None, ap_50=None):
        '''训练/验证过程中记录变量(每个iter都会记录, 不间断)
            Args:
                - mode: 模式(train, val, epoch)
                - loss: 损失(字典)
                - mAP:  
                - ap_50:  

            Returns:
                None
        '''        
        if mode == 'train':
            current_lr = self.optimizer.param_groups[0]['lr']
            self.argsHistory.record('lr', current_lr)
            # 记录所有损失:
            for loss_name, loss_value in losses.items():
                self.argsHistory.record(loss_name, loss_value.item())

        # 一个epoch结束后val评估结果的平均值
        if mode == 'epoch':             
            self.argsHistory.record('val_mAP@.5:.95', mAP)
            self.argsHistory.record('val_mAP@.5', ap_50)



    def printLog(self, mode, step, epoch, batch_num=None, losses=None):
        '''训练/验证过程中打印日志
            Args:
                - mode:       模式(train, val, epoch)
                - step:       当前迭代到第几个batch
                - epoch:      当前迭代到第几个epoch
                - batch_num:  当前batch的大小
                - losses:     当前batch的loss(字典)

            Returns:
                None
        '''        
        if mode != 'epoch':
            # 每间隔self.log_interval个iter才打印一次
            if step % self.log_interval == 0 and mode == 'train':
                # 右对齐, 打印更美观
                batch_idx = '{:>{}}'.format(step, len(f"{batch_num}"))
                log = ("Epoch(train)  [%d][%s/%d]  lr: %8f  ") % (epoch+1, batch_idx, batch_num, self.optimizer.param_groups[0]['lr'])
                for loss_name, loss_value in losses.items():
                    loss_log = (loss_name+": %.5f  " % (loss_value.item()))
                    log += loss_log
                self.logger.info(log)

        elif mode == 'epoch':
                mAP_list = self.argsHistory.args_history_dict['val_mAP@.5:.95']
                AP50_list = self.argsHistory.args_history_dict['val_mAP@.5']
                # 找到mAP最大对应的epoch
                best_epoch = AP50_list.index(max(AP50_list)) 
                self.logger.info('=' * 150)
                log = ("Epoch  [%d]  val_mAP@.5:.95: %.5f  val_mAP@.5: %.5f best_epoch %d" % (epoch+1, mAP_list[-1], AP50_list[-1], best_epoch+1))
                self.logger.info(log)
                self.logger.info('=' * 150)



    def saveCkpt(self, epoch):
        '''保存权重和训练断点
            Args:
                - epoch:        当前epoch
                - max_acc:      当前最佳模型在验证集上的准确率
                - mean_val_acc: 当前epoch准确率
                - best_epoch:   当前最佳模型对应的训练epoch

            Returns:
                None
        '''  
        # checkpoint_dict能够恢复断点训练
        checkpoint_dict = {
            'epoch': epoch, 
            'model_state_dict': self.model.state_dict(), 
            'optim_state_dict': self.optimizer.state_dict(),
            'sched_state_dict': self.scheduler.state_dict()
            }
        torch.save(checkpoint_dict, os.path.join(self.log_dir, f"epoch_{epoch}.pt"))
        torch.save(self.model.state_dict(), os.path.join(self.log_dir, "last.pt"))
        # 如果本次Epoch的val AP50最大，则保存参数(网络权重)
        AP50_list = self.argsHistory.args_history_dict['val_mAP@.5']
        if epoch == AP50_list.index(max(AP50_list)):
            torch.save(self.model.state_dict(), os.path.join(self.log_dir, 'best_AP50.pt'))
            self.logger.info('best checkpoint(AP50) has saved !')
        # 如果本次Epoch的val mAP最大，则保存参数(网络权重)
        mAP_list = self.argsHistory.args_history_dict['val_mAP@.5:.95']
        if epoch == mAP_list.index(max(mAP_list)):
            torch.save(self.model.state_dict(), os.path.join(self.log_dir, 'best_mAP.pt'))
            self.logger.info('best checkpoint(mAP) has saved !')






    def fitBatch(self, step, train_batch_num, epoch, batch_datas):
        '''一个batch的训练流程(前向+反向)

        Args:
            - imgs:   一个batch里的图像              例:shape=[bs, 3, 600, 600]
            - bboxes: 一个batch里的GT框              例:[(1, 4), (4, 4), (4, 4), (1, 4), (5, 4), (2, 4), (3, 4), (1, 4)]
            - labels: 一个batch里的GT框类别          例:[(1,), (4,), (4,), (1,), (5,), (2,), (3,), (1,)]

        Returns:
            - losses:      所有损失组成的列表
            - total_loss:  所有损失之和
        '''
        # 一个batch的前向传播+计算损失
        losses = self.model.batchLoss(self.device, self.img_size, batch_datas)
        # 将上一次迭代计算的梯度清零
        self.optimizer.zero_grad()
        # 反向传播
        losses['total_loss'].backward()
        # 更新权重
        self.optimizer.step()
        # 更新学习率
        self.scheduler.step(epoch * train_batch_num + step) 

        return losses



    def fitEpoch(self, epoch):
        '''一个epoch的训练
        '''
        self.model.train()
        train_batch_num = len(self.train_data_loader)
        for step, batch_datas in enumerate(self.train_data_loader):
            '''一个batch的训练, 并得到损失'''
            losses = self.fitBatch(step, train_batch_num, epoch, batch_datas)
            '''打印日志'''
            self.printLog('train', step, epoch, train_batch_num, losses)
            '''记录变量(loss, lr等, 每个iter都记录)'''
            self.recoardArgs('train', losses)




    def valEpoch(self, T, agnostic=False, vis_heatmap=False, save_vis_path=None, half=False):
        '''一个epoch的评估(基于验证集)
        '''
        self.model.eval()
        val_batch_num = len(self.val_data_loader)
        for step, batch_datas in tqdm(enumerate(self.val_data_loader)):
            '''一个batch的评估, 并得到batch的评估指标'''
            self.model.batchVal(self.device, self.img_size, batch_datas, T, agnostic=False, vis_heatmap=False, save_vis_path=None, half=False)






    def trainer(self):
        '''所有epoch的训练流程(训练+验证)
        '''
        for epoch in range(self.start_epoch, self.epoch):
            '''一个epoch的训练'''
            self.fitEpoch(epoch)
            '''以json格式保存args'''
            self.argsHistory.saveRecord()
            '''一个epoch的验证'''
            self.evaler(epoch)
            if epoch % self.eval_interval == 0 and (epoch!=0 or self.eval_interval==1):
                '''打印日志(一个epoch结束)'''
                self.printLog('epoch', 0, epoch)
                '''保存网络权重(一个epoch结束)'''
                self.saveCkpt(epoch)



    def evaler(self, epoch, inferring=True, pred_json_name='eval_tmp.json', ckpt_path=None, T=0.01):
        '''一个epoch的验证(验证集)
        '''
        if (epoch % self.eval_interval == 0 and (epoch!=0 or self.eval_interval==1)) or self.mode=='eval':
            '''在验证集上评估并计算AP'''
            # self.valEpoch(T, agnostic=False, vis_heatmap=False, save_vis_path=None, half=False)
            # 采用一张图一张图遍历的方式,并生成评估结果json文件
            mAP, ap_50 = self.test.genPredJsonAndEval(self.val_json_path, self.val_img_dir, self.log_dir, pred_json_name, T=T, model=self.model, inferring=inferring, ckpt_path=ckpt_path, reverse_map=self.reverse_map)
            '''记录变量'''
            self.recoardArgs('epoch', mAP=mAP, ap_50=ap_50)
            if self.mode == 'eval':            
                self.printLog('epoch', 0, 0)
        else:
            if epoch < self.eval_interval -1:
                ap_50, mAP = 0, 0
            else:
                ap_50 = self.argsHistory.args_history_dict['val_mAP@.5'][-1]
                mAP = self.argsHistory.args_history_dict['val_mAP@.5:.95'][-1]
            self.recoardArgs('epoch', mAP=mAP, ap_50=ap_50)






    def tester(self, mode, path, save_vis_path, ckpt_path, T, agnostic, vis_heatmap, show_text):
        '''推理一张图像/一段视频
        '''
        self.test.predict(mode, path, save_vis_path=save_vis_path, ckpt_path=ckpt_path, T=T, agnostic=agnostic, vis_heatmap=vis_heatmap, show_text=show_text)






def getArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, help='config file')
    args = parser.parse_args()
    return args










if __name__ == '__main__':
    args = getArgs()
    config_path = args.config
    # 使用动态导入的模块
    config_file = dynamic_import_class(config_path, get_class=False)
    # 调用参数
    runner_config = config_file.runner
    eval_config = config_file.eval
    test_config = config_file.test

    runner = Runner(**runner_config)
    # 训练模式
    if runner_config['mode'] == 'train':
        runner.trainer()
    # 验证模式
    elif runner_config['mode'] == 'eval':
        runner.evaler(epoch=0, **eval_config)
    # 测试模式
    elif runner_config['mode'] == 'test':
        runner.tester(**test_config)
    else:
        print("mode not valid. it must be 'train', 'eval' or 'test'.")