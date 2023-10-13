import torch.nn as nn
from functional.loss import vae_loss, beta_vae_loss


class VaeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_reconstructed, x, mu, logvar):
        return vae_loss(x_reconstructed, x, mu, logvar)


class BetaVaeLoss(nn.Module):
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def forward(self, x_reconstructed, x, mu, logvar):
        return beta_vae_loss(x_reconstructed, x, mu, logvar, self.beta)
