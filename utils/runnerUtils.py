# coding=utf-8
import os
import json
import torch
import logging
from logging import Logger
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






def getArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, help='config file')
    args = parser.parse_args()
    return args





def loadDatasets(mode:str, seed:int, bs:int, num_workers:int, my_dataset:dict):
    '''导入数据集
        Args:
            - mode:        当前模式(train/eval)
            - seed:        固定种子点
            - bs:          batch size
            - num_workers: nw
            - my_dataset:  dataset相关参数(在config文件中定义)

        Returns:
            - train_data:
            - train_data_loader:
            - val_json_path:
            - val_img_dir:
            - val_data:
            - val_data_loader: 
    ''' 
    # 动态导入机制和多进程存在冲突, 因此不能动态导入
    # COCODataset = dynamic_import_class(my_dataset['path'], 'COCODataset')
    
    dataset_type = my_dataset['path'].split('/')[-1][:-3]
    # 退而求其次, 用判断条件动态的导入
    if dataset_type == 'FasterRCNNDataset':
        from datasets.FasterRCNNDataset import COCODataset
    if dataset_type == 'YOLOv5Dataset':
        from datasets.YOLOv5Dataset import COCODataset
    if dataset_type == 'YOLOv8Dataset':
        from datasets.YOLOv8Dataset import COCODataset
    if dataset_type == 'WSDDNDataset':
        from datasets.WSDDNDataset import COCODataset

    # 导入验证集
    val_json_path = my_dataset['val_dataset']['annPath']
    val_img_dir = my_dataset['val_dataset']['imgDir']
    # 固定每个方法里都有一个COCODataset
    val_data = COCODataset(**my_dataset['val_dataset'])
    val_data_loader = DataLoader(val_data, shuffle=False, batch_size=bs, num_workers=num_workers, pin_memory=True, 
                                collate_fn=COCODataset.dataset_collate, worker_init_fn=partial(COCODataset.worker_init_fn, seed=seed))   
    if mode == 'eval':
        return None, None, val_json_path, val_img_dir, val_data, val_data_loader
    if mode == 'train':
        # 导入训练集
        train_data = COCODataset(**my_dataset['train_dataset'])
        train_data_loader = DataLoader(train_data, shuffle=True, batch_size=bs, num_workers=num_workers, pin_memory=True,
                                        collate_fn=COCODataset.dataset_collate, worker_init_fn=partial(COCODataset.worker_init_fn, seed=seed))
        return train_data, train_data_loader, val_json_path, val_img_dir, val_data, val_data_loader







def optimSheduler(
        model:nn.Module,
        total_epoch:int,
        optim_type:str, 
        train_data_loader,
        lr:float, 
        lr_min_ratio:float, 
        warmup_lr_init_ratio:float):
    '''定义优化器和学习率衰减策略
        Args:
            - model:                网络模型
            - total_epoch:          总epoch数
            - optim_type:           优化器类型
            - train_data_loader:    数据集dataloader
            - lr:                   最大学习率
            - lr_min_ratio:         余弦衰减到最大学习率的比例
            - warmup_lr_init_ratio: 初始学习率为最大学习率的比例

        Returns:
            None
    '''
    optimizer = {
        # adam会导致weight_decay错误，使用adam时建议设置为 0
        'adamw' : optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0),
        'adam' : optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0),
        'sgd'  : optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=0)
    }[optim_type]
    # 使用warmup+余弦退火学习率
    scheduler = CosineLRScheduler(
        optimizer=optimizer,
        # 总迭代数
        t_initial=total_epoch*len(train_data_loader),  
        # 余弦退火最低的学习率        
        lr_min=lr*lr_min_ratio,               
        # 学习率预热阶段的epoch数量                        
        warmup_t=2*len(train_data_loader), 
        # 学习率预热阶段的lr起始值
        warmup_lr_init=lr*warmup_lr_init_ratio,                               
    )

    return optimizer, scheduler








def saveCkpt(
        epoch:int, 
        model:nn.Module, 
        optimizer:optim, 
        scheduler:CosineLRScheduler, 
        log_dir:str, 
        argsHistory:ArgsHistory, 
        logger:Logger):
    '''保存权重和训练断点
        Args:
            - epoch:       当前epoch
            - model:       网络模型
            - optimizer:   优化器
            - scheduler:   学习率衰减
            - log_dir:     日志文件保存目录
            - argsHistory: 日志文件记录实例
            - logger:      日志输出实例

        Returns:
            None
    '''  
    # checkpoint_dict能够恢复断点训练
    checkpoint_dict = {
        'epoch': epoch, 
        'model_state_dict': model.state_dict(), 
        'optim_state_dict': optimizer.state_dict(),
        'sched_state_dict': scheduler.state_dict()
        }
    torch.save(checkpoint_dict, os.path.join(log_dir, f"epoch_{epoch}.pt"))
    torch.save(model.state_dict(), os.path.join(log_dir, "last.pt"))
    # 如果本次Epoch的val AP50最大，则保存参数(网络权重)
    AP50_list = argsHistory.args_history_dict['val_mAP@.5']
    if epoch == AP50_list.index(max(AP50_list)):
        torch.save(model.state_dict(), os.path.join(log_dir, 'best_AP50.pt'))
        logger.info('best checkpoint(AP50) has saved !')
    # 如果本次Epoch的val mAP最大，则保存参数(网络权重)
    mAP_list = argsHistory.args_history_dict['val_mAP@.5:.95']
    if epoch == mAP_list.index(max(mAP_list)):
        torch.save(model.state_dict(), os.path.join(log_dir, 'best_mAP.pt'))
        logger.info('best checkpoint(mAP) has saved !')









def myLogger(mode:str, log_dir:str):
    '''生成日志对象
        Args:
            - mode:       网络模型
            - log_dir:     日志文件保存目录

        Returns: 
            - logger:        日志记录实例
            - log_dir:     日志文件保存目录(添加了时间)
            - log_save_path: 日志保存路径
    '''
    logger = logging.getLogger('runer')
    logger.setLevel(level=logging.DEBUG)
    # 日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')

    if mode == 'train':
        # 写入文件的日志
        log_dir = os.path.join(log_dir, f"{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_train")
        # 日志文件保存路径
        log_save_path = os.path.join(log_dir, f"{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_train.log")
    if mode == 'eval':
        # 写入文件的日志
        log_dir = os.path.join(log_dir, f"{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_val")
            # 日志文件保存路径
        log_save_path = os.path.join(log_dir, f"{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_val.log")
    if not os.path.isdir(log_dir):os.makedirs(log_dir)
    file_handler = logging.FileHandler(log_save_path, encoding="utf-8", mode="a")
    file_handler.setLevel(level=logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    # 终端输出的日志
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger, log_dir, log_save_path








def printLog(
        mode:str, 
        logger:Logger,
        step:int, 
        epoch:int, 
        log_interval:int=None,
        optimizer:optim=None,
        argsHistory:ArgsHistory=None,
        batch_num:int=None, 
        losses:dict=None):
    '''训练/验证过程中打印日志
        # Args: 
            - mode:         模式(train, val, epoch)
            - log_interval: 日志打印间隔(几个iter打印一次)
            - logger:       日志实例
            - optimizer:    优化器实例
            - argsHistory:  日志记录实例
            - step:         当前迭代到第几个batch
            - epoch:        当前迭代到第几个epoch
            - batch_num:    当前batch的大小
            - losses:       当前batch的loss(字典)

        Returns:
            None
    '''         
    if mode != 'epoch':
        # 每间隔log_interval个iter才打印一次
        if step % log_interval == 0 and mode == 'train':
            # 右对齐, 打印更美观
            batch_idx = '{:>{}}'.format(step, len(f"{batch_num}"))
            log = ("Epoch(train)  [%d][%s/%d]  lr: %8f  ") % (epoch+1, batch_idx, batch_num, optimizer.param_groups[0]['lr'])
            for loss_name, loss_value in losses.items():
                loss_log = (loss_name+": %.5f  " % (loss_value.item()))
                log += loss_log
            logger.info(log)

    elif mode == 'epoch':
            mAP_list = argsHistory.args_history_dict['val_mAP@.5:.95']
            AP50_list = argsHistory.args_history_dict['val_mAP@.5']
            # 找到mAP最大对应的epoch
            best_epoch = AP50_list.index(max(AP50_list)) 
            logger.info('=' * 150)
            log = ("Epoch  [%d]  val_mAP@.5:.95: %.5f  val_mAP@.5: %.5f best_epoch %d" % (epoch+1, mAP_list[-1], AP50_list[-1], best_epoch+1))
            logger.info(log)
            logger.info('=' * 150)






    

def printRunnerArgs(
        backbone_name:str, 
        mode:str, 
        device:str, 
        seed:int, 
        bs:int, 
        img_size:list[int,int], 
        train_data_len:int, 
        val_data_len:int, 
        cat_nums:int, 
        optimizer:optim,
        logger:Logger):
    '''训练前打印基本信息
    '''
    if mode in ['train', 'eval']:
        logger.info(f'CPU/CUDA:   {device}')
        logger.info(f'骨干网络: {backbone_name}')
        logger.info(f'全局种子: {seed}')
        logger.info(f'图像大小:   {img_size}')
        logger.info('验证集大小:   %d' % val_data_len)
        logger.info('数据集类别数: %d' % cat_nums)
        if mode == 'train':
            logger.info(f'Batch Size: {bs}')
            logger.info('训练集大小:   %d' % train_data_len)
            logger.info(f'优化器: {optimizer}')

        logger.info('='*150)   








def recoardArgs(
        mode:str, 
        argsHistory:ArgsHistory,
        optimizer:optim=None,
        loss:dict=None, 
        mAP=None, 
        ap_50=None):
    '''训练/验证过程中记录变量(每个iter都会记录, 不间断)
        Args:
            - mode:        训练模式(train, epoch)
            - optimizer:   优化器实例
            - argsHistory: 日志记录实例
            - loss:        损失
            - acc:         准确率
            - mAP:         所有类别的平均ap
            - mF1Score:    所有类别的平均F1Score

        Returns:
            None
    '''       
    if mode == 'train':
        current_lr = optimizer.param_groups[0]['lr']
        argsHistory.record('lr', current_lr)
        # 记录所有损失:
        for loss_name, loss_value in loss.items():
            argsHistory.record(loss_name, loss_value.item())

    # 一个epoch结束后val评估结果的平均值
    if mode == 'epoch':             
        argsHistory.record('val_mAP@.5:.95', mAP)
        argsHistory.record('val_mAP@.5', ap_50)







def trainResume(
        resume, 
        model:nn.Module, 
        optimizer:optim, 
        logger:logging.Logger, 
        argsHistory:ArgsHistory):
    '''保存权重和训练断点
        Args:
            - resume:      是否恢复断点训练
            - model:       网络模型
            - optimizer:   优化器
            - logger:      日志输出实例
            - argsHistory: 日志文件记录实例

        Returns:
            None
    '''  
    checkpoint = torch.load(resume)
    start_epoch = checkpoint['epoch'] + 1 # +1是因为从当前epoch的下一个epoch开始训练
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optim_state_dict'])
    # self.scheduler.load_state_dict(checkpoint['sched_state_dict'])
    logger.info(f'resume:{resume}')
    logger.info(f'start_epoch:{start_epoch}')
    # 导入上一次中断训练时的args
    json_dir, _ = os.path.split(resume)
    argsHistory.loadRecord(json_dir)