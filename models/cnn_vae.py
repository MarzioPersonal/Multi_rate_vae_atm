import torch
import torch.nn as nn

from loss_function.loss import FixedBetaVaeLoss, VariateBetaVaeLoss
from layers.MultiRateConvolution import MultiRateDecoderConvTranspose2d, MultiRateEncoderConv2d
from layers.MultiRateSequential import MultiRateSequential
from distributions.beta_distribution import BetaUniform


def test_shape(input_p, in_channels, latent_dim, is_cifar, is_celeba):
    with torch.no_grad():
        model = MultiRateCnnVae(in_channels, latent_dim, is_cifar, is_celeba)
        x_rec = model.forward(input_p)[0]
    assert input_p.shape == x_rec.shape, f'found {input_p.shape} and {x_rec.shape}'


class AbstractCnnVae(nn.Module):
    def __init__(self,
                 in_channels: int,
                 latent_dimension: int,
                 is_cifar: bool = False,
                 is_celeba: bool = False,
                 use_multi_rate=False
                 ):
        super(AbstractCnnVae, self).__init__()
        if is_cifar and is_celeba:
            raise AttributeError(f'is_cifar and is_celeba cannot be both true')
        self.use_multi_rate = use_multi_rate
        if use_multi_rate:
            encoder_fn = MultiRateEncoderConv2d
            decoder_fn = MultiRateDecoderConvTranspose2d
            sequential_fn = MultiRateSequential
        else:
            encoder_fn = nn.Conv2d
            decoder_fn = nn.ConvTranspose2d
            sequential_fn = nn.Sequential

        def encode_block(in_channels, out_channels, kernel_size=4, stride=2, padding=1, last=False):
            if last:
                return sequential_fn(
                    encoder_fn(in_channels, out_channels, kernel_size, stride, padding=padding),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.Flatten(start_dim=1)
                )
            return sequential_fn(
                encoder_fn(in_channels, out_channels, kernel_size, stride, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )

        def decode_block(in_channels, out_channels, kernel_size, stride, padding=0, output_padding=0, last=False):
            if last:
                return sequential_fn(
                    decoder_fn(in_channels, out_channels, kernel_size, stride, padding=padding,
                               output_padding=output_padding),
                    nn.Sigmoid()
                )
            return sequential_fn(
                decoder_fn(in_channels, out_channels, kernel_size, stride, padding=padding,
                           output_padding=output_padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )

        self.encoder_list = nn.ModuleList()
        self.encoder_list.append(encode_block(in_channels=in_channels, out_channels=128))
        self.encoder_list.append(encode_block(in_channels=128, out_channels=256))
        self.encoder_list.append(encode_block(in_channels=256, out_channels=512))
        self.encoder_list.append(encode_block(in_channels=512, out_channels=1024, last=True))

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
        self.decoder_list = nn.ModuleList()
        if is_cifar:
            self.shapes = (1024, 8, 8)

            self.decoder_list.append(
                decode_block(in_channels=1024, out_channels=512, kernel_size=kernel_size, stride=2, padding=1,
                             output_padding=0))
            self.decoder_list.append(
                decode_block(in_channels=512, out_channels=256, kernel_size=kernel_size, stride=2, padding=1,
                             output_padding=1))
            self.decoder_list.append(
                decode_block(in_channels=256, out_channels=in_channels, kernel_size=kernel_size, stride=1, padding=2,
                             output_padding=0, last=True))

        elif is_celeba:
            self.shapes = (1024, 8, 8)

            self.decoder_list.append(
                decode_block(in_channels=1024, out_channels=512, kernel_size=kernel_size, stride=2, padding=2,
                             output_padding=0))
            self.decoder_list.append(
                decode_block(in_channels=512, out_channels=256, kernel_size=kernel_size, stride=2, padding=1,
                             output_padding=0))
            self.decoder_list.append(
                decode_block(in_channels=256, out_channels=128, kernel_size=kernel_size, stride=2, padding=2,
                             output_padding=1))
            self.decoder_list.append(
                decode_block(in_channels=128, out_channels=in_channels, kernel_size=kernel_size, stride=1, padding=1,
                             output_padding=0, last=True))


        else:
            self.shapes = (1024, 4, 4)

            self.decoder_list.append(
                decode_block(in_channels=1024, out_channels=512, kernel_size=kernel_size, stride=2, padding=1,
                             output_padding=0))
            self.decoder_list.append(
                decode_block(in_channels=512, out_channels=256, kernel_size=kernel_size, stride=2, padding=1,
                             output_padding=1))
            self.decoder_list.append(
                decode_block(in_channels=256, out_channels=in_channels, kernel_size=kernel_size, stride=2, padding=1,
                             output_padding=1, last=True))

    def decode_prep(self, x):
        batch_size = x.shape[0]
        x = self.decoder_emb(x)
        x = x.reshape(batch_size, *self.shapes)
        return x

    def reparameterize(self, mu, std):
        eps = torch.randn_like(std)
        return mu + eps * std, eps


class BetaCnnVae(AbstractCnnVae):

    def __init__(self, in_channels: int, latent_dimension: int, is_cifar: bool = False, is_celeba: bool = False,
                 beta=1.):
        super().__init__(in_channels, latent_dimension, is_cifar, is_celeba, False)
        self.beta = beta
        self.loss_fn = FixedBetaVaeLoss(beta)

    def encode(self, input):
        for module in self.encoder_list:
            input = module(input)
        return input

    def decode(self, input):
        input = self.decode_prep(input)
        for module in self.decoder_list:
            input = module(input)
        return input

    def forward(self, x):
        out = self.encode(x)
        mu = self.mu(out)
        logvar = self.logvar(out)
        std = torch.exp(0.5 * logvar)
        z, eps = self.reparameterize(mu, std)

        x_reconstructed = self.decode(z)
        reconstruction_loss, KDL, loss = self.loss_fn(x_reconstructed, x, mu, logvar)
        return x_reconstructed, x, z, mu, logvar, reconstruction_loss, KDL, loss


class MultiRateCnnVae(AbstractCnnVae):

    def __init__(self, in_channels: int, latent_dimension: int, is_cifar: bool = False, is_celeba: bool = False):
        super().__init__(in_channels, latent_dimension, is_cifar, is_celeba, True)

        self.beta_uniform = BetaUniform()
        self.loss_fn = VariateBetaVaeLoss()

    def encode(self, input, beta):
        for module in self.encoder_list:
            input = module(input, beta)
        return input

    def decode(self, input, beta):
        input = self.decode_prep(input)
        for module in self.decoder_list:
            input = module(input, beta)
        return input

    def decode_prep(self, x):
        return super().decode_prep(x)

    def reparameterize(self, mu, std):
        return super().reparameterize(mu, std)

    def forward(self, x):
        log_beta = self.beta_uniform.sample(sample_shape=torch.Size((x.shape[0], 1)))
        out = self.encode(x, log_beta)
        mu = self.mu(out)
        logvar = self.logvar(out)
        std = torch.exp(0.5 * logvar)
        z, eps = self.reparameterize(mu, std)

        x_reconstructed = self.decode(z, log_beta)
        reconstruction_loss, KDL, loss = self.loss_fn.forward(x_reconstructed, x, mu, logvar, torch.exp(log_beta))
        return x_reconstructed, x, z, mu, logvar, reconstruction_loss, KDL, loss, log_beta


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
