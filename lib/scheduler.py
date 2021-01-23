import math
from bisect import bisect_right
from functools import partial
import torch.optim as optim
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

class CosineAnnealingLR_withwarmup(_LRScheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr and
    :math:`T_{cur}` is the number of epochs since the last restart in SGDR:

    .. math::

        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 +
        \cos(\frac{T_{cur}}{T_{max}}\pi))

    When last_epoch=-1, sets initial lr as lr.

    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_. Note that this only
    implements the cosine annealing part of SGDR, and not the restarts.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        eta_min (float): Minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self, optimizer, total_steps, warmup_lr=0.0, warmup_steps=0, hold_base_rate_steps=0, eta_min=0, last_epoch=-1):
        self.total_steps = total_steps
        self.eta_min = eta_min
        self.warmup_lr = warmup_lr
        self.warmup_steps = warmup_steps
        self.hold_base_rate_steps = hold_base_rate_steps
        super(CosineAnnealingLR_withwarmup, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        # return [self.eta_min + (base_lr - self.eta_min) *
        #         (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
        #         for base_lr in self.base_lrs]
        learning_rate = [self.eta_min + 0.5 *
                               base_lr *
                              (1 + math.cos(2 * math.pi * (self.last_epoch - self.warmup_steps - self.hold_base_rate_steps)
                               / (self.total_steps - self.warmup_steps - self.hold_base_rate_steps)))
                for base_lr in self.base_lrs]

        if self.warmup_steps > 0:
            if self.last_epoch < self.warmup_steps:
                slope = [(base_lr - self.warmup_lr)/self.warmup_steps for base_lr in self.base_lrs]
                learning_rate = [s * self.last_epoch + self.warmup_lr for s in slope]

        return learning_rate

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# if __name__ == '__main__':
#     net = mobilenetv1(pretrained=False)
#     parameter_dict = dict(net.named_parameters())
#     params = []
#     for name, param in parameter_dict.items():
#         params += [{'params': [param], 'lr': 0.1, 'weight_decay': 0.0004}]
#
#     optimizer = optim.SGD(params, lr=0.1, momentum=0.9, weight_decay=0.0004)
#     scheduler = CosineAnnealingLR_withwarmup(optimizer, total_steps=1000, warmup_lr=0.02, warmup_steps=100, hold_base_rate_steps=0, eta_min=0)
#
#     lr_to_follow = []
#     for i in range(1000):
#         scheduler.step()
#
#         lr = scheduler.get_lr()
#         lr_to_follow.append(lr[0])
#
#     plt.plot(lr_to_follow)
#     plt.show()

