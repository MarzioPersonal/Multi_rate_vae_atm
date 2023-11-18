import torch

import torch.nn as nn


class EncoderActivation(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x):
        # for the encoder sigma is = 1 / (1 + e^(-x)) (the sigmoid)
        return nn.functional.sigmoid(x)


class DecoderActivation(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x):
        # for the decoder sigma is  = ReLU(1-exp(x))^(1/2)
        return torch.sqrt(nn.functional.relu(1. - torch.exp(x)))
