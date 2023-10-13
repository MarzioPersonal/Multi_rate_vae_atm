import torch
import torch.nn as nn
from loss_function.loss import VaeLoss


class CnnVae(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 128, 4, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, 4, 2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 1024, 4, 2),
        )

        self.mu = nn.Linear(1024, 100)
        self.logvar = nn.Linear(1024, 100)

        self.decoder = nn.Sequential(
            nn.Linear(100, 16384),
            nn.ConvTranspose2d(16384, 512, 3, 2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 3, 2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 1, 3, 2),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        mu = self.mu(x)
        logvar = self.logvar(x)
        std = torch.exp(0.5 * logvar)
        z, eps = self.reparameterize(mu, std)
        x_reconstructed = self.decoder(z)
        reconstruction_loss, KDL, loss = VaeLoss()(x_reconstructed, x, mu, logvar)

        return x_reconstructed, x, z, mu, logvar, reconstruction_loss, KDL, loss

    def reparameterize(self, mu, std):
        eps = torch.randn_like(std)
        return mu + eps * std, eps
