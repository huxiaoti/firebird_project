# -*- coding: utf-8 -*-

import numpy as np
from torch.optim.lr_scheduler import _LRScheduler

class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, lr_mul, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.lr_mul = lr_mul # 2.0
        self.d_model = d_model # 512
        self.n_warmup_steps = n_warmup_steps # 5, default=4000
        self.n_steps = 0

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients with the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        d_model = self.d_model #512
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        return (d_model ** -0.5) * min(n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5))
        # (512**-0.5)*min(1**-0.5, 1*4000**-1.5)=0.0441*min(1,3.95e-6)=1.7e-7
        # (512**-0.5)*min(100**-0.5, 100*4000**-1.5)=1.7e-5

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_steps += 1
        lr = self.lr_mul * self._get_lr_scale() # 2*1.7e-7

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

class PolyLr(_LRScheduler):
    def __init__(self, optimizer, gamma=0.9, max_iteration=1000, minimum_lr=1.0e-07, warmup_iteration=100, last_epoch=-1):
        self.gamma = gamma
        self.max_iteration = max_iteration
        self.minimum_lr = minimum_lr
        self.warmup_iteration = warmup_iteration

        super(PolyLr, self).__init__(optimizer, last_epoch)

    def poly_lr(self, base_lr, step):
        return (base_lr - self.minimum_lr) * ((1 - (step / self.max_iteration)) ** self.gamma) + self.minimum_lr

    def warmup_lr(self, base_lr, alpha):
        return base_lr * (1 / 10.0 * (1 - alpha) + alpha)

    def get_lr(self):
        if self.last_epoch < self.warmup_iteration:
            alpha = self.last_epoch / self.warmup_iteration
            lrs = [min(self.warmup_lr(base_lr, alpha), self.poly_lr(base_lr, self.last_epoch)) for base_lr in self.base_lrs]
        else:
            lrs = [self.poly_lr(base_lr, self.last_epoch) for base_lr in self.base_lrs]

        return lrs