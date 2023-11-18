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
        kdl_loss_ = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
        kdl_loss_no_beta = torch.mean(kdl_loss_, dim=0)
        kdl_loss_ = torch.mean(beta * kdl_loss_, dim=0)

        return reconstruction_loss_ + kdl_loss_, (kdl_loss_no_beta.detach().cpu().item(), reconstruction_loss_.detach().cpu().item())


class NonGaussianVAELoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x_pred: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor,
                 beta: float | torch.Tensor = 1.):
        reconstruction_loss_ = F.mse_loss(x_pred, x)
        kdl_loss_ = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
        kdl_loss_no_beta = torch.mean(kdl_loss_, dim=0)
        kdl_loss_ = torch.mean(beta * kdl_loss_, dim=0)
        return reconstruction_loss_ + beta * kdl_loss_, (kdl_loss_no_beta.detach().cpu().item(), reconstruction_loss_.detach().cpu().item())
