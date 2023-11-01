from torch import nn
import torch

from layers.activation_mrvae import DecoderActivation, EncoderActivation


class MRVAELayer(nn.Module):
    def __init__(self, features: int, activation_fnc: nn.Module) -> None:
        super().__init__()
        self.features = features
        self.hyper_block_scale = nn.Linear(1, self.features, bias=True)
        self.activation_fnc = activation_fnc

    def forward(self, inputs: torch.Tensor, betas: torch.Tensor) -> torch.Tensor:
        scale = self.hyper_block_scale(betas)
        scale = self.activation_fnc(scale)
        if len(inputs.shape) == 4:
            # Unsqueeze for convolutional layers.
            scale = scale.unsqueeze(-1).unsqueeze(-1)
        return scale * inputs


class EncoderMRVAELayer(MRVAELayer):
    def __init__(self, features: int) -> None:
        super().__init__(features, EncoderActivation())

    def forward(self, inputs: torch.Tensor, betas: torch.Tensor) -> torch.Tensor:
        return super().forward(inputs, betas)


class DecoderMRVAELayer(MRVAELayer):
    def __init__(self, features: int) -> None:
        super().__init__(features, DecoderActivation())

    def forward(self, inputs: torch.Tensor, betas: torch.Tensor) -> torch.Tensor:
        return super().forward(inputs, betas)
