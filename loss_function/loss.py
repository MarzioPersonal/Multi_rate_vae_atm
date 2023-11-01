import torch
import torch.nn as nn

from loss_function.functional.loss import beta_vae_loss


class VariateBetaVaeLoss(nn.Module):
    def __init__(self):
        super(VariateBetaVaeLoss, self).__init__()

    def forward(self, x_reconstructed, x, mu, logvar, beta):
        return beta_vae_loss(x_reconstructed, x, mu, logvar, beta)

class FixedBetaVaeLoss(nn.Module):
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def forward(self, x_reconstructed, x, mu, logvar):
        return beta_vae_loss(x_reconstructed, x, mu, logvar, self.beta)


class VaeLoss(FixedBetaVaeLoss):
    def __init__(self):
        super().__init__(beta=1.)

    def forward(self, x_reconstructed, x, mu, logvar):
        return beta_vae_loss(x_reconstructed, x, mu, logvar, self.beta)

