import torch
import torch.nn as nn
import torch.nn.functional as F


# def gaussian_vae_loss(x_pred: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor,
#                       beta: float | torch.Tensor = 1.):
#     reconstruction_loss_ = F.mse_loss(x_pred, x, reduction='sum')
#     kdl_loss_ = torch.mean(-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1), dim=0)

#     return reconstruction_loss_ + beta * kdl_loss_, (reconstruction_loss_.detach().cpu().item(), kdl_loss_.detach().cpu().item())

class GaussianVAELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_pred: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor, beta: float | torch.Tensor = 1.):
        reconstruction_loss_ = F.binary_cross_entropy(x_pred, x, reduction='mean')
        kdl_loss_ = torch.mean(-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1), dim=0)

        return reconstruction_loss_ + beta * kdl_loss_, (reconstruction_loss_.detach().cpu().item(), kdl_loss_.detach().cpu().item())


def vae_loss(x_pred: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor,
             beta: float | torch.Tensor = 1.):
    reconstruction_loss_ = F.mse_loss(x_pred, x)
    kdl_loss_ = torch.sum(-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1), dim=0)

    return reconstruction_loss_ + beta * kdl_loss_, (reconstruction_loss_.detach().cpu().item(), kdl_loss_.detach().cpu().item())
