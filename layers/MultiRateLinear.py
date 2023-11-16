# linear with multi rate additional layer


import torch.nn as nn
from torch import Tensor
from layers.MultiRateVaeLayer import EncoderMRVAELayer, DecoderMRVAELayer


class MultiRateLinearEncoder(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self.multi_rate_layer = EncoderMRVAELayer(out_features)


    def reset_parameters(self) -> None:
        super().reset_parameters()

    def forward(self, input: Tensor, betas) -> Tensor:
        x = super().forward(input)
        return self.multi_rate_layer.forward(x, betas)

    def extra_repr(self) -> str:
        return super().extra_repr()



class MultiRateLinearDecoder(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self.multi_rate_layer = DecoderMRVAELayer(out_features)

    def reset_parameters(self) -> None:
        super().reset_parameters()

    def forward(self, input: Tensor, betas) -> Tensor:
        x = super().forward(input)
        return self.multi_rate_layer.forward(x, betas)

    def extra_repr(self) -> str:
        return super().extra_repr()
