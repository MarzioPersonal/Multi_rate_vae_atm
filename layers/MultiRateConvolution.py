from typing import Optional, Union, List, Iterator, overload

import torch.nn as nn
from torch import Tensor
from torch.nn import Module
from torch.nn.common_types import _size_2_t
from torch.nn.modules.container import T

from layers.MultiRateVaeLayer import EncoderMRVAELayer, DecoderMRVAELayer


class MultiRateEncoderConv2d(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_2_t, stride: _size_2_t = 1,
                 padding: Union[str, _size_2_t] = 0, dilation: _size_2_t = 1, groups: int = 1, bias: bool = True,
                 padding_mode: str = 'zeros', device=None, dtype=None) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode,
                         device, dtype)

        self.multi_rate_layer = EncoderMRVAELayer(out_channels)

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        return super()._conv_forward(input, weight, bias)

    def forward(self, input: Tensor, beta: Tensor) -> Tensor:
        x = super().forward(input)
        return self.multi_rate_layer.forward(x, beta)


class MultiRateDecoderConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_2_t, stride: _size_2_t = 1,
                 padding: _size_2_t = 0, output_padding: _size_2_t = 0, groups: int = 1, bias: bool = True,
                 dilation: _size_2_t = 1, padding_mode: str = 'zeros', device=None, dtype=None) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias,
                         dilation, padding_mode, device, dtype)

        self.multi_rate_layer = DecoderMRVAELayer(out_channels)

    def forward(self, input: Tensor, beta: Tensor, output_size: Optional[List[int]] = None) -> Tensor:
        x = super().forward(input, output_size)
        return self.multi_rate_layer(x, beta)


