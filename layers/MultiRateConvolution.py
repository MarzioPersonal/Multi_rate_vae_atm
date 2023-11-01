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


class MultiRateSequential(nn.Sequential):
    @overload
    def __init__(self, *args: Module) -> None:
        ...

    @overload
    def __init__(self, arg: 'OrderedDict[str, Module]') -> None:
        ...

    def __init__(self, *args):
        super().__init__(*args)

    def _get_item_by_idx(self, iterator, idx) -> T:
        return super()._get_item_by_idx(iterator, idx)

    def __getitem__(self, idx: Union[slice, int]) -> Union['Sequential', T]:
        return super().__getitem__(idx)

    def __setitem__(self, idx: int, module: Module) -> None:
        super().__setitem__(idx, module)

    def __delitem__(self, idx: Union[slice, int]) -> None:
        super().__delitem__(idx)

    def __len__(self) -> int:
        return super().__len__()

    def __add__(self, other) -> 'Sequential':
        return super().__add__(other)

    def pop(self, key: Union[int, slice]) -> Module:
        return super().pop(key)

    def __iadd__(self, other) -> 'Sequential':
        return super().__iadd__(other)

    def __mul__(self, other: int) -> 'Sequential':
        return super().__mul__(other)

    def __rmul__(self, other: int) -> 'Sequential':
        return super().__rmul__(other)

    def __imul__(self, other: int) -> 'Sequential':
        return super().__imul__(other)

    def __dir__(self):
        return super().__dir__()

    def __iter__(self) -> Iterator[Module]:
        return super().__iter__()

    def forward(self, input, beta):
        for i, module in enumerate(self):
            if i != 0:
                input = module(input)
            else:
                input = module(input, beta)
        return input

    def append(self, module: Module) -> 'Sequential':
        return super().append(module)

    def insert(self, index: int, module: Module) -> 'Sequential':
        return super().insert(index, module)

    def extend(self, sequential) -> 'Sequential':
        return super().extend(sequential)