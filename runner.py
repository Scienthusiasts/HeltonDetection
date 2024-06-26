# coding=utf-8
import os
import shutil
import torch
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
# 多卡并行训练:
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
# 自定义模块
from test import Test
from utils.util import *
from utils.runnerUtils import *
from utils.metrics import computeParamFLOPs
from utils.exportUtils import torchExportOnnx







class Runner():
    '''训练/验证/推理时的流程'''
    def __init__(self, 
                 seed:int, 
                 mode:str, 
                 class_names:list, 
                 img_size:list, 
                 epoch:int, 
                 resume:str, 
                 log_dir:str, 
                 log_interval:int, 
                 eval_interval:int, 
                 reverse_map:dict, 
                 dataset:dict, 
                 test:dict, 
                 model:dict, 
                 optimizer:dict):
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
        # NOTE:多卡:
        if self.mode=='train_ddp':
            dist.init_process_group('nccl')
            self.local_rank = dist.get_rank()

        '''日志模块'''
        if mode in ['train', 'train_ddp', 'eval']:
            self.logger, self.log_dir, self.log_save_path = myLogger(self.mode, self.log_dir)
            '''训练/验证时参数记录模块'''
            json_save_dir, _ = os.path.split(self.log_save_path)
            self.argsHistory = ArgsHistory(json_save_dir)
        '''导入数据集'''
        if self.mode in ['train', 'train_ddp', 'eval']:
            self.train_data, \
            self.train_data_loader, \
            self.val_json_path, \
            self.val_img_dir, \
            self.val_data, \
            self.val_data_loader = loadDatasets(mode=self.mode, seed=self.seed, **dataset)
        '''导入网络'''
        # 根据模型名称动态导入模块
        self.model = dynamic_import_class(model.pop('path'), 'Model')(**model).to(self.device)
        cudnn.benchmark = True

        '''是否恢复断点训练'''
        self.start_epoch = 0
        if self.resume and self.mode in ['train', 'train_ddp']:
            trainResume(self.resume, self.model, self.optimizer, self.logger, self.argsHistory)

        # NOTE:多卡:
        if self.mode=='train_ddp':
            self.model = nn.parallel.DistributedDataParallel(self.model.cuda(self.local_rank), device_ids=[self.local_rank])

        '''定义优化器(自适应学习率的带动量梯度下降方法)'''
        if mode in ['train', 'train_ddp']:
            self.optimizer, self.scheduler = optimSheduler(**optimizer, 
                                                           model=self.model, 
                                                           total_epoch=self.epoch, 
                                                           train_data_loader=self.train_data_loader)

        '''导入评估模块'''
        if self.mode =='train_ddp':
            # NOTE:多卡:
            self.test = Test(dataset['my_dataset']['path'], self.model.module, self.img_size, self.class_names, self.device, **test)
        elif self.mode =='train':
            self.test = Test(dataset['my_dataset']['path'], self.model, self.img_size, self.class_names, self.device, **test)
        '''打印训练参数'''
        if self.mode in ['train', 'train_ddp', 'eval']:
            val_data_len = self.val_data.__len__()
            train_data_len = self.train_data.__len__() if self.mode in ['train', 'train_ddp'] else 0
            printRunnerArgs(
                backbone_name=model['backbone_name'], 
                mode=self.mode, 
                logger=self.logger, 
                device=self.device, 
                seed=self.seed, 
                bs=dataset['bs'], 
                img_size=self.img_size, 
                train_data_len=train_data_len, 
                val_data_len=val_data_len, 
                cat_nums=self.cat_nums, 
                optimizer=optimizer)
        # torch.save(self.model.state_dict(), "./tmp.pt")









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
        if self.mode=='train_ddp':
            losses = self.model.module.batchLoss(self.local_rank, self.img_size, batch_datas)
        else:
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
        # NOTE:多卡
        # 通过维持各个进程之间的相同随机数种子使不同进程能获得同样的shuffle效果
        if self.mode=='train_ddp':
            self.train_data_loader.sampler.set_epoch(epoch)
        for step, batch_datas in enumerate(self.train_data_loader):
            '''一个batch的训练, 并得到损失'''
            losses = self.fitBatch(step, train_batch_num, epoch, batch_datas)
            '''打印日志'''
            printLog(
                mode='train', 
                log_interval=self.log_interval, 
                logger=self.logger, 
                optimizer=self.optimizer, 
                step=step, 
                epoch=epoch, 
                batch_num=train_batch_num, 
                losses=losses)
            # NOTE:多卡
            if self.mode=='train' or (self.mode=='train_ddp' and dist.get_rank() == 0):
                '''记录变量(loss, lr等, 每个iter都记录)'''
                recoardArgs(mode='train', optimizer=self.optimizer, argsHistory=self.argsHistory, loss=losses)




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
            # 同步所有进程确保训练完全完成(类似阻塞)
            if self.mode=='train_ddp':dist.barrier()
            # NOTE:多卡
            if self.mode=='train' or (self.mode=='train_ddp' and dist.get_rank() == 0):
                '''以json格式保存args'''
                self.argsHistory.saveRecord()
                '''一个epoch的验证'''
                self.evaler(epoch, self.model.module) if self.mode=='train_ddp' else self.evaler(epoch, self.model)
                if epoch % self.eval_interval == 0 and (epoch!=0 or self.eval_interval==1):
                    '''打印日志(一个epoch结束)'''
                    printLog(mode='epoch', logger=self.logger, argsHistory=self.argsHistory, step=0, epoch=epoch)
                    '''保存网络权重(一个epoch结束)'''
                    if self.mode=='train':
                        saveCkpt(epoch, self.model, self.optimizer, self.scheduler, self.log_dir, self.argsHistory, self.logger)
                    elif self.mode=='train_ddp':
                        saveCkpt(epoch, self.model.module, self.optimizer, self.scheduler, self.log_dir, self.argsHistory, self.logger)

            # 可以考虑在此处再次同步，确保主进程的评估和日志记录不会被后续的训练步骤干扰(类似阻塞)
            if self.mode=='train_ddp':dist.barrier()




    def evaler(self, epoch, model, inferring=True, pred_json_name='eval_tmp.json', ckpt_path=None, T=0.01, fuse=False):
        '''一个epoch的验证(验证集)
        '''
        if (epoch % self.eval_interval == 0 and (epoch!=0 or self.eval_interval==1)) or self.mode=='eval':
            '''在验证集上评估并计算AP'''
            # self.valEpoch(T, agnostic=False, vis_heatmap=False, save_vis_path=None, half=False)
            # 采用一张图一张图遍历的方式,并生成评估结果json文件
            mAP, ap_50 = self.test.genPredJsonAndEval(self.val_json_path, self.val_img_dir, self.log_dir, pred_json_name, T=T, model=model, inferring=inferring, ckpt_path=ckpt_path, reverse_map=self.reverse_map, fuse=fuse)
            '''最后一个epoch计算模型参数量, FLOPs'''
            if epoch == self.epoch:
                computeParamFLOPs(self.device, model, self.img_size)
            '''记录变量'''
            recoardArgs(mode='epoch', argsHistory=self.argsHistory, mAP=mAP, ap_50=ap_50)
            if self.mode == 'eval':            
                printLog(mode='epoch', logger=self.logger, argsHistory=self.argsHistory, step=0, epoch=0)
        else:
            if epoch < self.eval_interval -1:
                ap_50, mAP = 0, 0
            else:
                ap_50 = self.argsHistory.args_history_dict['val_mAP@.5'][-1]
                mAP = self.argsHistory.args_history_dict['val_mAP@.5:.95'][-1]
            '''记录变量'''
            recoardArgs(mode='epoch', argsHistory=self.argsHistory, mAP=mAP, ap_50=ap_50)





    def tester(self, mode, img_path, save_vis_path, ckpt_path, T, agnostic, vis_heatmap, show_text, onnx_path):
        '''推理一张图像/一段视频
        '''
        self.test.predict(mode, img_path = img_path, onnx_path=onnx_path, save_vis_path=save_vis_path, ckpt_path=ckpt_path, T=T, agnostic=agnostic, vis_heatmap=vis_heatmap, show_text=show_text)






    def exporter(self, export_dir, export_name, export_param, ckpt_path):
        '''导出为onnx格式
        '''
        torchExportOnnx(self.model, self.device, self.img_size, export_dir, export_name, export_param, ckpt_path)





if __name__ == '__main__':
    args = getArgs()
    config_path = args.config
    # 使用动态导入的模块
    config_file = dynamic_import_class(config_path, get_class=False)
    # 调用参数
    runner_config = config_file.runner
    eval_config = config_file.eval
    test_config = config_file.test
    export_config = config_file.export

    runner = Runner(**runner_config)
    # 训练模式
    if runner_config['mode'] in ['train', 'train_ddp']:
        # 拷贝一份当前训练对应的config文件(方便之后查看细节)
        shutil.copy(config_path, os.path.join(runner.log_dir, 'config.py'))
        runner.trainer()
    # 验证模式
    elif runner_config['mode'] == 'eval':
        runner.evaler(epoch=0, **eval_config)
    # 测试模式
    elif runner_config['mode'] == 'test':
        runner.tester(**test_config)
    elif runner_config['mode'] == 'export':
        runner.exporter(**export_config)
    else:
        print("mode not valid. it must be 'train', 'eval' or 'test'.")