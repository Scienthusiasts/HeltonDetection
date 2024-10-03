""" Step Scheduler
modify from path-to-conda-env/timm/scheduler/step_lr.py
"""
import math
import torch
from typing import List
from timm.scheduler.scheduler import Scheduler


class ConditionalStepLRScheduler(Scheduler):
    """
    """

    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            warmup_t=0,
            warmup_lr_init=0,
            # DIY:
            decay_rate: float = 0.1,
            decay_t_list: List = [120000, 160000],
            max_lr: float = 0.0025,

            t_in_epochs=True,
            noise_range_t=None,
            noise_pct=0.67,
            noise_std=1.0,
            noise_seed=42,
            initialize=True,
    ) -> None:
        super().__init__(
            optimizer,
            param_group_field="lr",
            t_in_epochs=t_in_epochs,
            noise_range_t=noise_range_t,
            noise_pct=noise_pct,
            noise_std=noise_std,
            noise_seed=noise_seed,
            initialize=initialize,
        )
        self.max_lr = max_lr
        self.lrs = 0
        self.decay_t_list = decay_t_list
        self.decay_rate = decay_rate
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        if self.warmup_t:
            self.warmup_steps = (self.max_lr - warmup_lr_init) / self.warmup_t
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = 1

    def _get_lr(self, t: int) -> List[float]:
        # warmup时学习率
        if t < self.warmup_t:
            self.lrs = self.warmup_lr_init + t * self.warmup_steps 
        # warmup后学习率调整
        else:
            step = 0
            for i in range(len(self.decay_t_list)):
                if t >= self.decay_t_list[i]: 
                    step += 1
            self.lrs = self.max_lr * self.decay_rate ** step
                
        return self.lrs
