import torch
import torch.nn as nn

from layers.MultiRateConvolution import MultiRateConv2d, MultiRateConvTranspose2d
from layers.MultiRateSequential import MultiRateSequential, ModifiedSequential


# def test_shape(input_p, in_channels, latent_dim, is_cifar, is_celeba):
#     with torch.no_grad():
#         model = CnnVae(in_channels, latent_dim, is_cifar, is_celeba)
#         x_rec = model.forward(input_p, None)[0]
#     assert input_p.shape == x_rec.shape, f'found {input_p.shape} and {x_rec.shape}'


class CnnVae(nn.Module):
    def __init__(self,
                 in_channels: int,
                 latent_dimension: int,
                 is_cifar: bool = False,
                 is_celeba: bool = False,
                 use_multi_rate=False
                 ):
        super(CnnVae, self).__init__()
        if is_cifar and is_celeba:
            raise AttributeError(f'is_cifar and is_celeba cannot be both true')
        self.use_multi_rate = use_multi_rate
        if use_multi_rate:
            encoder_fn = MultiRateConv2d
            decoder_fn = MultiRateConvTranspose2d
            sequential_fn = MultiRateSequential
        else:
            encoder_fn = nn.Conv2d
            decoder_fn = nn.ConvTranspose2d
            sequential_fn = ModifiedSequential

        def encode_block(in_channels_, out_channels, kernel_size=4, stride=2, padding=1, last=False):
            if last:
                return sequential_fn(
                    encoder_fn(in_channels_, out_channels, kernel_size, stride, padding=padding),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.Flatten(start_dim=1)
                )
            return sequential_fn(
                encoder_fn(in_channels_, out_channels, kernel_size, stride, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )

        def decode_block(in_channels_, out_channels, kernel_size, stride, padding=0, output_padding=0, last=False):
            if last:
                return sequential_fn(
                    decoder_fn(in_channels_, out_channels, kernel_size, stride, padding=padding,
                               output_padding=output_padding),
                    nn.Sigmoid()
                )
            return sequential_fn(
                decoder_fn(in_channels_, out_channels, kernel_size, stride, padding=padding,
                           output_padding=output_padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )

        self.encoder_list = nn.ModuleList()
        self.encoder_list.append(encode_block(in_channels_=in_channels, out_channels=128))
        self.encoder_list.append(encode_block(in_channels_=128, out_channels=256))
        self.encoder_list.append(encode_block(in_channels_=256, out_channels=512))
        self.encoder_list.append(encode_block(in_channels_=512, out_channels=1024, last=True))

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

        self.decoder_pre = nn.Linear(latent_dimension, out_first_linear)
        self.decoder_list = nn.ModuleList()
        if is_cifar:
            self.shapes = (1024, 8, 8)

            self.decoder_list.append(
                decode_block(in_channels_=1024, out_channels=512, kernel_size=kernel_size, stride=2, padding=1,
                             output_padding=0))
            self.decoder_list.append(
                decode_block(in_channels_=512, out_channels=256, kernel_size=kernel_size, stride=2, padding=1,
                             output_padding=1))
            self.decoder_list.append(
                decode_block(in_channels_=256, out_channels=in_channels, kernel_size=kernel_size, stride=1, padding=2,
                             output_padding=0, last=True))

        elif is_celeba:
            self.shapes = (1024, 8, 8)

            self.decoder_list.append(
                decode_block(in_channels_=1024, out_channels=512, kernel_size=kernel_size, stride=2, padding=2,
                             output_padding=0))
            self.decoder_list.append(
                decode_block(in_channels_=512, out_channels=256, kernel_size=kernel_size, stride=2, padding=1,
                             output_padding=0))
            self.decoder_list.append(
                decode_block(in_channels_=256, out_channels=128, kernel_size=kernel_size, stride=2, padding=2,
                             output_padding=1))
            self.decoder_list.append(
                decode_block(in_channels_=128, out_channels=in_channels, kernel_size=kernel_size, stride=1, padding=1,
                             output_padding=0, last=True))
        else:
            self.shapes = (1024, 4, 4)

            self.decoder_list.append(
                decode_block(in_channels_=1024, out_channels=512, kernel_size=kernel_size, stride=2, padding=1,
                             output_padding=0))
            self.decoder_list.append(
                decode_block(in_channels_=512, out_channels=256, kernel_size=kernel_size, stride=2, padding=1,
                             output_padding=1))
            self.decoder_list.append(
                decode_block(in_channels_=256, out_channels=in_channels, kernel_size=kernel_size, stride=2, padding=1,
                             output_padding=1, last=True))

    def encode(self, state: torch.Tensor, beta: torch.Tensor | None):
        for m in self.encoder_list:
            state = m(state, beta)
        return state

    def decode(self, state: torch.Tensor, beta: torch.Tensor | None):
        state = self.decoder_pre(state)
        state = self.reshape_back(state)
        for m in self.decoder_list:
            state = m(state, beta)
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
        return x.reshape(x.shape[0], *self.shapes)

    def reparameterize(self, mu, std):
        eps = torch.randn_like(std)
        return mu + eps * std, eps



# def test_shapes():
#     input_mnist = torch.randn(size=(32, 1, 28, 28))
#     input_cifar = torch.randn(size=(32, 3, 32, 32))
#     input_celeb = torch.randn(size=(32, 3, 64, 64))
#     latent_dim = 32
#     # MINST
#     print('MNIST')
#     test_shape(input_mnist, 1, latent_dim, False, False)
#     # CIFAR
#     print('CIFAR')
#     test_shape(input_cifar, 3, latent_dim, True, False)
#     # CELEBA
#     print('CELEBA')
#     test_shape(input_celeb, 3, latent_dim, False, True)
#
#
# if __name__ == '__main__':
#     test_shapes()
