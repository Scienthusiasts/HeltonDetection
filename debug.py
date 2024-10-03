from opt.scheduler import ConditionalStepLRScheduler
from nbdev.showdoc import show_doc
from matplotlib import pyplot as plt
from timm import create_model 
from timm.optim import create_optimizer
from types import SimpleNamespace

model = create_model('resnet34')

args = SimpleNamespace()
args.weight_decay = 0
args.lr = 1e-4
args.opt = 'adam' 
args.momentum = 0.9


def get_lr_per_epoch(optimizer, num_iter):
    lr_per_epoch = []
    for iter in range(num_iter):
        lr_per_epoch.append(optimizer.param_groups[0]['lr'])
        scheduler.step(iter) 
    return lr_per_epoch


optimizer = create_optimizer(args, model)
num_epoch = 36
total_batch_num = 5000
decay_t_list=[total_batch_num * num_epoch * 6/9, total_batch_num * num_epoch * 8/9]
decay_rate = 0.1
warmup_lr_init = 1e-5
scheduler = ConditionalStepLRScheduler(optimizer, 
                                      warmup_t=2*total_batch_num, 
                                      warmup_lr_init=warmup_lr_init, 
                                      decay_rate=decay_rate, 
                                      decay_t_list=decay_t_list
                                      )
lr_per_epoch = get_lr_per_epoch(optimizer, total_batch_num * num_epoch)
plt.plot([i for i in range(total_batch_num * num_epoch)], lr_per_epoch, label="With warmup")
plt.show()