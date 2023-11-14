import torch
import torch.nn.functional as F
from torch import Tensor



def beta_vae_loss(x_reconstructed, x, mu, logvar, beta):
    rec_loss: Tensor = gaussian_reconstruction_loss(x_reconstructed, x)
    KDL: Tensor = kdl_loss(mu, logvar)
    loss: Tensor = rec_loss + beta * KDL

    return rec_loss, KDL, loss
#
#
def reconstruction_loss(x_reconstructed : torch.Tensor, x: torch.Tensor):
    rec_loss = F.mse_loss(x_reconstructed, x)
    return rec_loss


def gaussian_reconstruction_loss(x_reconstructed: torch.Tensor, x: torch.Tensor):
    return F.binary_cross_entropy(x_reconstructed, x)

def kdl_loss(mu, logvar):
    KDL: Tensor = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return KDL
