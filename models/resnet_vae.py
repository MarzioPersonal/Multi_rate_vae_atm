import torch
import torch.nn as nn

from layers.MultiRateConvolution import MultiRateConv2d, MultiRateConvTranspose2d
from layers.MultiRateSequential import MultiRateSequential, ModifiedSequential


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, multi_rate=None, encoding=None):
        nn.Module.__init__(self)

        self.conv_block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, in_channels, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x: torch.tensor) -> torch.Tensor:
        return x + self.conv_block(x)


class ResNetVae(nn.Module):
    def __init__(self, latent_dimension: int, is_cifar: bool = False, is_celeba: bool = False,
                 use_multi_rate=False, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.use_multi_rate = use_multi_rate
        self.encoder_list = nn.ModuleList()
        self.decoder_list = nn.ModuleList()

        self.mu = None
        self.logvar = None
        self.first_decode = None

        if use_multi_rate:
            sequential_fn = MultiRateSequential
            conv2d_fn = MultiRateConv2d
            conv2d_transpose_fn = MultiRateConvTranspose2d
        else:
            sequential_fn = ModifiedSequential
            conv2d_fn = nn.Conv2d
            conv2d_transpose_fn = nn.ConvTranspose2d

        def define_encoder(in_channels: list[int], kernel_sizes, strides, paddings, dim_hidden):
            for i in range(len(in_channels) - 1):
                input_t = in_channels[i]
                output_t = in_channels[i + 1]
                self.encoder_list.append(sequential_fn(
                    conv2d_fn(input_t, output_t, kernel_size=kernel_sizes[i], stride=strides[i], padding=paddings[i])
                ))
            self.encoder_list.append(ModifiedSequential(
                ResBlock(128, 32),
                ResBlock(128, 32),
                nn.Flatten(start_dim=1)
            ))

            # latent functions
            self.mu = nn.Linear(dim_hidden, latent_dimension)
            self.logvar = nn.Linear(dim_hidden, latent_dimension)
            self.decoder_pre = nn.Linear(latent_dimension, dim_hidden)

        if is_cifar:
            # CIFAR
            self.image_shape = (128, 8, 8)
            dim_hidden = 8192
            in_channels = [3, 64, 128, 128]
            kernel_sizes = [4, 4, 3]
            strides = [2, 2, 1]
            paddings = [1, 1, 1]
            define_encoder(in_channels, kernel_sizes, strides, paddings, dim_hidden)
            self.decoder_list.append(
                ModifiedSequential(
                    ResBlock(128, 32),
                    ResBlock(128, 32)
                )
            )
            self.decoder_list.append(sequential_fn(
                conv2d_transpose_fn(128, 64, kernel_size=4, stride=2, padding=1)
            ))
            self.decoder_list.append(sequential_fn(
                conv2d_transpose_fn(64, 3, kernel_size=4, stride=2, padding=1),
                nn.Sigmoid()
            ))

        elif is_celeba:
            # CELEBA
            self.image_shape = (128, 4, 4)
            dim_hidden = 2048
            in_channels = [3, 64, 128, 128, 128]
            kernel_sizes = [4, 4, 3, 3]
            strides = [2, 2, 2, 2]
            paddings = [1, 1, 1, 1]
            define_encoder(in_channels, kernel_sizes, strides, paddings, dim_hidden)
            self.decoder_list.append(sequential_fn(
                conv2d_transpose_fn(128, 128, kernel_size=3, stride=2, padding=1)
            ))
            self.decoder_list.append(
                ModifiedSequential(
                    ResBlock(128, 32),
                    ResBlock(128, 32)
                )
            )
            self.decoder_list.append(sequential_fn(
                conv2d_transpose_fn(128, 128, kernel_size=5, stride=2, padding=1),
                nn.Sigmoid()
            ))
            self.decoder_list.append(sequential_fn(
                conv2d_transpose_fn(128, 64, kernel_size=5, stride=2, padding=1, output_padding=1),
                nn.Sigmoid(),
            ))
            self.decoder_list.append(sequential_fn(
                conv2d_transpose_fn(64, 3, kernel_size=4, stride=2, padding=1),
                nn.Sigmoid()
            ))
        else:
            self.image_shape = (128, 4, 4)
            dim_hidden = 2048
            in_channels = [1, 64, 128, 128]
            kernel_sizes = [4, 4, 3]
            strides = [2, 2, 2]
            paddings = [1, 1, 1]
            define_encoder(in_channels, kernel_sizes, strides, paddings, dim_hidden)
            self.decoder_list.append(sequential_fn(
                conv2d_transpose_fn(128, 128, kernel_size=3, stride=2, padding=1)
            ))
            self.decoder_list.append(
                ModifiedSequential(
                    ResBlock(128, 32),
                    ResBlock(128, 32),
                    nn.ReLU()
                )
            )
            self.decoder_list.append(sequential_fn(
                conv2d_transpose_fn(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU()
            ))
            self.decoder_list.append(sequential_fn(
                conv2d_transpose_fn(64, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.Sigmoid()
            ))

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
        return x.reshape(x.shape[0], *self.image_shape)

    def reparameterize(self, mu, std):
        eps = torch.randn_like(std)
        return mu + eps * std, eps

if __name__ == '__main__':
    input_mnist = torch.randn(size=(32, 1, 28, 28))
    input_cifar = torch.randn(size=(32, 3, 32, 32))
    input_celeb = torch.randn(size=(32, 3, 64, 64))
    latent_dim = 32
    # MINST
    print('MNIST')
    model = ResNetVae(latent_dimension=32)
    input_mnist_p, *_ = model.forward(input_mnist, beta=None)
    assert input_mnist_p.shape == input_mnist.shape
    # CIFAR
    print('CIFAR')
    model = ResNetVae(latent_dimension=32, is_cifar=True)
    input_cifar_p, *_ = model.forward(input_cifar, beta=None)
    assert input_cifar_p.shape == input_cifar.shape
    # CELEBA
    print('CELEBA')
    model = ResNetVae(latent_dimension=32, is_celeba=True)
    input_celeb_p, *_ = model.forward(input_celeb, beta=None)
    assert input_celeb_p.shape == input_celeb.shape
