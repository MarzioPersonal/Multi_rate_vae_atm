# linear vae for the experiment of section 5.1.
# dataset: MNIST
#
# beta-vae: we trained 10 separate two-layer linear β-VAEs, each with different β values.
#
# mr-vae: We trained MR-VAEs on the MNIST dataset (Deng, 2012) by sampling the KL weighting term β from 0.01 to 10.
import torch.nn as nn
import torch

from layers.MultiRateLinear import MultiRateLinearDecoder, MultiRateLinearEncoder
from layers.MultiRateSequential import MultiRateSequential, ModifiedSequential


class LinearVae(nn.Module):
    def __init__(self, image_shape=(1, 28, 28), latent_dim=32, use_multi_rate=False):
        super().__init__()
        if use_multi_rate:
            encoder_fn = MultiRateLinearEncoder
            decoder_fn = MultiRateLinearDecoder
            sequential_fn = MultiRateSequential
        else:
            encoder_fn = nn.Linear
            decoder_fn = nn.Linear
            sequential_fn = ModifiedSequential
        self.image_shape = image_shape
        self.encoder = nn.ModuleList()
        self.encoder_in = nn.Flatten(start_dim=1)
        self.encoder.append(sequential_fn(
            encoder_fn(28 * 28, 392),
            nn.ReLU(),
        ))
        self.encoder.append(sequential_fn(
            encoder_fn(392, 196),
            nn.ReLU()
        ))

        self.mu = nn.Linear(196, latent_dim)
        self.logvar = nn.Linear(196, latent_dim)
        self.decoder = nn.ModuleList()
        self.decoder.append(sequential_fn(
            decoder_fn(latent_dim, 392),
            nn.ReLU()
        ))
        self.decoder.append(sequential_fn(
            decoder_fn(392, 784),
            nn.Sigmoid()
        ))

    def encode(self, state: torch.Tensor, beta: torch.Tensor | None):
        state = self.encoder_in(state)
        for m in self.encoder:
            state = m(state, beta)
        return state

    def decode(self, state: torch.Tensor, beta: torch.Tensor | None):
        for m in self.decoder:
            state = m(state, beta)
        state = self.reshape_back(state)
        return state

    def forward(self, x: torch.Tensor, beta: torch.Tensor | None):
        encoded_x = self.encode(x, beta)
        mu = self.mu(encoded_x)
        logvar = self.logvar(encoded_x)
        std = torch.exp(0.5 * logvar)
        z, eps = self.reparameterize(mu, std)
        x_pred = self.decode(z, beta)
        return x_pred, (mu, logvar)

    def reshape_back(self, x):
        return x.reshape(x.shape[0], *self.image_shape)

    def reparameterize(self, mu, std):
        eps = torch.randn_like(std)
        return mu + eps * std, eps
