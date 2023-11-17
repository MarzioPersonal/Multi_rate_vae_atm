import math
import torch


class WarmupCosineDecayScheduler(torch.optim.lr_scheduler.LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.cosine_decay_epochs = total_epochs - warmup_epochs
        super(WarmupCosineDecayScheduler, self).__init__(optimizer, last_epoch)



    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            return [base_lr * (self.last_epoch / self.warmup_epochs) for base_lr in self.base_lrs]
        else:
            # Cosine decay
            return [base_lr * (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / self.cosine_decay_epochs)) / 2
                    for base_lr in self.base_lrs]



