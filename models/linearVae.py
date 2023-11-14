# linear vae for the experiment of section 5.1.
# dataset: MNIST
#
# beta-vae: we trained 10 separate two-layer linear β-VAEs, each with different β values.
#
# mr-vae: We trained MR-VAEs on the MNIST dataset (Deng, 2012) by sampling the KL weighting term β from 0.01 to 10.
import torch.nn as nn
import torch

from distributions.beta_distribution import BetaUniform
from layers.MultiRateLinear import MultiRateLinearEncoder, MultiRateLinearDecoder
from layers.MultiRateSequential import MultiRateLinearSequential
from loss_function.loss import FixedBetaVaeLoss, VariateBetaVaeLoss


class AbstractLinearVae(nn.Module):
    def __init__(self, image_shape=(1, 28, 28), latent_dim=32, use_multi_rate=False):
        super().__init__()
        if use_multi_rate:
            encoder_fn = MultiRateLinearEncoder
            decoder_fn = MultiRateLinearDecoder
            sequential_fn = MultiRateLinearSequential
        else:
            encoder_fn = nn.Linear
            decoder_fn = nn.Linear
            sequential_fn = nn.Sequential

        self.image_shape = image_shape

        self.encoder = sequential_fn(
            nn.Flatten(start_dim=1),
            encoder_fn(28*28, 392),
            encoder_fn(392, 196),
        )
        self.out = nn.Sigmoid()

        self.mu = nn.Linear(196, latent_dim)
        self.logvar = nn.Linear(196, latent_dim)

        self.decoder = sequential_fn(
            decoder_fn(latent_dim, 392),
            decoder_fn(392, 784)
        )

    def reshape_back(self, x):
        return x.reshape(x.shape[0], *self.image_shape)

    def reparameterize(self, mu, std):
        eps = torch.randn_like(std)
        return mu + eps * std, eps


class LinearVae(AbstractLinearVae):

    def __init__(self, image_shape=(1, 28, 28), latent_dim=32, beta=1.):
        super().__init__(image_shape, latent_dim, False)
        self.beta = beta
        self.loss_fn = FixedBetaVaeLoss(self.beta)

    def forward(self, x):
        out = self.encoder.forward(x)
        mu = self.mu(out)
        logvar = self.logvar(out)
        std = torch.exp(0.5 * logvar)
        z, eps = self.reparameterize(mu, std)

        x_reconstructed = self.out(self.decoder.forward(z))
        x_reconstructed = self.reshape_back(x_reconstructed)
        reconstruction_loss, KDL, loss = self.loss_fn(x_reconstructed, x, mu, logvar)
        return x_reconstructed, x, z, mu, logvar, reconstruction_loss, KDL, loss


class MultiRateLinearVae(AbstractLinearVae):

    def __init__(self, image_shape=(1, 28, 28), latent_dim=32):
        super().__init__(image_shape, latent_dim, True)
        self.loss_fn = VariateBetaVaeLoss()
        self.loss_fn_infer = FixedBetaVaeLoss(beta=1.)

        self.betas = BetaUniform()

    def forward(self, x):
        betas = self.betas.sample(sample_shape=torch.Size((x.shape[0],1)))
        out = self.encoder.forward(x, betas)
        mu = self.mu(out)
        logvar = self.logvar(out)
        std = torch.exp(0.5 * logvar)
        z, eps = self.reparameterize(mu, std)

        x_reconstructed = self.out(self.decoder.forward(z, betas))
        x_reconstructed = self.reshape_back(x_reconstructed)
        reconstruction_loss, KDL, loss = self.loss_fn(x_reconstructed, x, mu, logvar, torch.mean(torch.exp(betas)))
        return x_reconstructed, x, z, mu, logvar, reconstruction_loss, KDL, loss

    def infer_training(self, x):
        beta = torch.log(torch.ones(size=(1, 1)))
        out = self.encoder.forward(x, beta)
        mu = self.mu(out)
        logvar = self.logvar(out)
        std = torch.exp(0.5 * logvar)
        z, eps = self.reparameterize(mu, std)
        x_reconstructed = self.decoder.forward(z, beta)
        x_reconstructed = self.reshape_back(x_reconstructed)
        reconstruction_loss, KDL, loss = self.loss_fn_infer(x_reconstructed, x, mu, logvar)
        return  loss
