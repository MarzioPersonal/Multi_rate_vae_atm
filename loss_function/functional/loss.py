import torch
import torch.nn.functional as F
from torch import Tensor


def vae_loss(x_reconstructed, x, mu, logvar):
    rec_loss: Tensor = reconstruction_loss(x_reconstructed, x)
    KDL: Tensor = kdl_loss(mu, logvar)
    loss: Tensor = rec_loss + KDL

    return rec_loss, KDL, loss


def beta_vae_loss(x_reconstructed, x, mu, logvar, beta):
    rec_loss: Tensor = reconstruction_loss(x_reconstructed, x)
    KDL: Tensor = kdl_loss(mu, logvar)
    loss: Tensor = rec_loss + beta * KDL

    return rec_loss, KDL, loss


def reconstruction_loss(x_reconstructed, x):
    rec_loss: Tensor = (0.5 * F.mse_loss(x_reconstructed.reshape(x.shape[0], -1), x.reshape(x.shape[0], -1),
                                         reduction="none", ).sum(dim=-1))
    return rec_loss


def kdl_loss(mu, logvar):
    KDL: Tensor = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    return KDL
