import torch
import torch.nn as nn

from loss_function.loss import VaeLoss


def test_shape(input_p, in_channels, latent_dim, is_cifar, is_celeba):
    with torch.no_grad():
        model = CnnVae(in_channels, latent_dim, is_cifar, is_celeba)
        x_rec = model.forward(input_p)[0]
    assert input_p.shape == x_rec.shape, f'found {input_p.shape} and {x_rec.shape}'


class CnnVae(nn.Module):
    def __init__(self, in_channels: int, latent_dimension: int, is_cifar: bool = False, is_celeba: bool = False, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        if is_cifar and is_celeba:
            raise AttributeError(f'is_cifar and is_celeba cannot be both true')
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 128, 4, 2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, 4, 2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 1024, 4, 2, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )
        if is_cifar:
            self.mu = nn.Linear(4096, latent_dimension)
            self.logvar = nn.Linear(4096, latent_dimension)
            kernel_size = 4
            out_first_linear = 65536
        elif is_celeba:
            self.mu = nn.Linear(16384, latent_dimension)
            self.logvar = nn.Linear(16384, latent_dimension)
            kernel_size = 5
            out_first_linear = 65536
        else:
            self.mu = nn.Linear(1024, latent_dimension)
            self.logvar = nn.Linear(1024, latent_dimension)
            kernel_size = 3
            out_first_linear = 16384

        self.decoder_emb = nn.Linear(latent_dimension, out_first_linear)

        if is_cifar:
            self.shapes = (1024, 8, 8)
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(1024, 512, kernel_size, 2, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.ConvTranspose2d(512, 256, kernel_size, 2, padding=1, output_padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.ConvTranspose2d(256, 3, kernel_size, 1, padding=2),
                nn.Sigmoid()
            )
        elif is_celeba:
            self.shapes = (1024, 8, 8)
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(1024, 512, kernel_size, 2, padding=2),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.ConvTranspose2d(512, 256, kernel_size, 2, padding=1, output_padding=0),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.ConvTranspose2d(256, 128, kernel_size, 2, padding=2, output_padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.ConvTranspose2d(128, in_channels, kernel_size, 1, padding=1),
                nn.Sigmoid()
            )
        else:
            self.shapes = (1024, 4, 4)
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(1024, 512, kernel_size, 2, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.ConvTranspose2d(512, 256, kernel_size, 2, padding=1, output_padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.ConvTranspose2d(256, 1, kernel_size, 2, padding=1, output_padding=1),
                nn.Sigmoid()
            )

        self.loss_fn = VaeLoss()

    def encode(self, x):
        batch_size = x.shape[0]
        x = self.encoder(x)
        return x.reshape(batch_size, -1)

    def decode(self, x):
        batch_size = x.shape[0]
        x = self.decoder_emb(x)
        x = x.reshape(batch_size, *self.shapes)
        return self.decoder(x)

    def forward(self, x):
        out = self.encode(x)
        mu = self.mu(out)
        logvar = self.logvar(out)
        std = torch.exp(0.5 * logvar)
        z, eps = self.reparameterize(mu, std)
        x_reconstructed = self.decode(z)
        reconstruction_loss, KDL, loss = self.loss_fn(x_reconstructed, x, mu, logvar)

        return x_reconstructed, x, z, mu, logvar, reconstruction_loss, KDL, loss

    def reparameterize(self, mu, std):
        eps = torch.randn_like(std)
        return mu + eps * std, eps


def test_shapes():
    input_mnist = torch.randn(size=(32, 1, 28, 28))
    input_cifar = torch.randn(size=(32, 3, 32, 32))
    input_celeb = torch.randn(size=(32, 3, 64, 64))
    latent_dim = 32
    # MINST
    print('MNIST')
    test_shape(input_mnist, 1, latent_dim, False, False)
    # CIFAR
    print('CIFAR')
    test_shape(input_cifar, 3, latent_dim, True, False)
    # CELEBA
    print('CELEBA')
    test_shape(input_celeb, 3, latent_dim, False, True)

if __name__ == '__main__':
    test_shapes()