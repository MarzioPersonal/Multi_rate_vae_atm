from typing import Optional, Union, List, Iterator, overload

import torch.nn as nn
from torch import Tensor
from torch.nn import Module
from torch.nn.common_types import _size_2_t
from torch.nn.modules.container import T


class MultiRateSequential(nn.Sequential):
    def forward(self, input, beta):
        for i, module in enumerate(self):
            if i != 0:
                input = module(input)
            else:
                input = module(input, beta)
        return input



class MultiRateLinearSequential(nn.Sequential):

    def forward(self, input, beta):
        for module in self:
            if isinstance(module, nn.Flatten):
                input = module(input)
            else:
                input = module(input, beta)
        return input
