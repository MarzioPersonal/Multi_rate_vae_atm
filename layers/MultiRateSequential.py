import torch
import torch.nn as nn



class MultiRateSequential(nn.Sequential):
    def forward(self, input, beta=None):
        for i, module in enumerate(self):
            if i != 0:
                input = module(input)
            else:
                input = module(input, beta)
        return input

class ModifiedSequential(nn.Sequential):
    def forward(self, input, beta=None):
        for i, module in enumerate(self):
            input = module(input)
        return input
