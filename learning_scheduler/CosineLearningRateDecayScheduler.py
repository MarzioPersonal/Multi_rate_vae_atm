# import math
# from typing import Optional
# import torch
# from torch.optim.lr_scheduler import LRScheduler
# # taken from https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup
#
# class CosineAnnealingWarmupRestarts(LRScheduler):
#     """
#         optimizer (Optimizer): Wrapped optimizer.
#         first_cycle_steps (int): First cycle step size.
#         cycle_mult(float): Cycle steps magnification. Default: -1.
#         max_lr(float): First cycle's max learning rate. Default: 0.1.
#         min_lr(float): Min learning rate. Default: 0.001.
#         warmup_steps(int): Linear warmup step size. Default: 0.
#         gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
#         last_epoch (int): The index of last epoch. Default: -1.
#     """
#
#     def __init__(self,
#                  optimizer: torch.optim.Optimizer,
#                  first_cycle_steps: int,
#                  cycle_mult: float = 1.,
#                  max_lr: float = 0.1,
#                  min_lr: float = 0.001,
#                  warmup_steps: int = 0,
#                  gamma: float = 1.,
#                  last_epoch: int = -1,
#                  verbose: bool = False
#                  ):
#         assert warmup_steps < first_cycle_steps
#
#         self.first_cycle_steps = first_cycle_steps  # first cycle step size
#         self.cycle_mult = cycle_mult  # cycle steps magnification
#         self.base_max_lr = max_lr  # first max learning rate
#         self.max_lr = max_lr  # max learning rate in the current cycle
#         self.min_lr = min_lr  # min learning rate
#         self.warmup_steps = warmup_steps  # warmup step size
#         self.gamma = gamma  # decrease rate of max learning rate by cycle
#
#         self.cur_cycle_steps = first_cycle_steps  # first cycle step size
#         self.cycle = 0  # cycle count
#         self.step_in_cycle = last_epoch  # step size of the current cycle
#
#         super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch, verbose)
#
#         # set learning rate min_lr
#         self.init_lr()
#
#         self._last_lr = None
#
#     def init_lr(self):
#         self.base_lrs = []
#         for param_group in self.optimizer.param_groups:
#             param_group['lr'] = self.min_lr
#             self.base_lrs.append(self.min_lr)
#
#     def get_lr(self) -> list[float]:
#         if self.step_in_cycle == -1:
#             return self.base_lrs
#         elif self.step_in_cycle < self.warmup_steps:
#             return [(self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr for base_lr in
#                     self.base_lrs]
#         else:
#             return [base_lr + (self.max_lr - base_lr)
#                     * (1 + math.cos(math.pi * (self.step_in_cycle - self.warmup_steps)
#                                     / (self.cur_cycle_steps - self.warmup_steps))) / 2
#                     for base_lr in self.base_lrs]
#
#     def step(self, epoch: Optional[int] = None):
#         if epoch is None:
#             epoch = self.last_epoch + 1
#             self.step_in_cycle = self.step_in_cycle + 1
#             if self.step_in_cycle >= self.cur_cycle_steps:
#                 self.cycle += 1
#                 self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
#                 self.cur_cycle_steps = int(
#                     (self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
#         else:
#             if epoch >= self.first_cycle_steps:
#                 if self.cycle_mult == 1.:
#                     self.step_in_cycle = epoch % self.first_cycle_steps
#                     self.cycle = epoch // self.first_cycle_steps
#                 else:
#                     n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
#                     self.cycle = n
#                     self.step_in_cycle = epoch - int(
#                         self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
#                     self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** n
#             else:
#                 self.cur_cycle_steps = self.first_cycle_steps
#                 self.step_in_cycle = epoch
#
#         self.max_lr = self.base_max_lr * (self.gamma ** self.cycle)
#         self.last_epoch = math.floor(epoch)
#
#         class _enable_get_lr_call:
#
#             def __init__(self, o):
#                 self.o = o
#
#             def __enter__(self):
#                 self.o._get_lr_called_within_step = True
#                 return self
#
#             def __exit__(self, type, value, traceback):
#                 self.o._get_lr_called_within_step = False
#                 return self
#
#         with _enable_get_lr_call(self):
#             for i, (param_group, lr) in enumerate(zip(self.optimizer.param_groups, self.get_lr())):
#                 param_group['lr'] = lr
#                 self.print_lr(self.verbose, i, lr, epoch)
#
#         self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
