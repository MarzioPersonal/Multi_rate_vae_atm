import torch.nn
import torchvision
from torchvision.datasets import MNIST, Omniglot, CIFAR10, SVHN, CelebA
from torchvision import transforms, datasets
from torch.utils.data import DataLoader


def get_dataloaders(train_dataset,
                    test_dataset,
                    seed,
                    batch_size_train: int = 32,
                    batch_size_test: int = 1,
                    train_size: float = None,
                    val_size: float = None,
                    shuffle: bool = True
                    ):
    generator = torch.Generator().manual_seed(seed) if seed is not None else torch.Generator()
    if val_size is not None:
        if train_size is None:
            train_size = 1. - val_size
        else:
            assert train_size + val_size == 1, f'train_size and val_size must sum up to 1, found {train_size} and {val_size}'
        dataset_train, dataset_val = torch.utils.data.random_split(train_dataset, lengths=[train_size, val_size],
                                                                   generator=generator)
        train_dataloader = DataLoader(dataset_train, shuffle=shuffle, batch_size=batch_size_train)
        val_dataloader = DataLoader(dataset_val, shuffle=False, batch_size=batch_size_train)
    else:
        train_dataloader = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size_train)
        val_dataloader = None
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size_test)
    return train_dataloader, val_dataloader, test_dataloader


def get_mnist_binary_static_loaders(
        batch_size_train: int = 32,
        batch_size_test: int = 1,
        train_size: float = None,
        val_size: float = None,
        seed=None,
        shuffle: bool = True
) -> tuple[DataLoader, DataLoader | None, DataLoader]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        lambda x: x >= 0.1307,
        lambda x: x.float()
    ])

    data_train_val = MNIST(root='./data', train=True, download=True, transform=transform)
    validation_size = 10000
    validation_start_index = len(data_train_val) - validation_size
    validation_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
    validation_dataset.data = data_train_val.data[validation_start_index:]
    validation_dataset.targets = data_train_val.targets[validation_start_index:]
    data_train_val.data = data_train_val.data[:validation_start_index]
    data_train_val.targets = data_train_val.targets[:validation_start_index]
    data_test = MNIST(root='./data', train=False, download=True, transform=transform)
    train_dataloader, _, test_dataloader = get_dataloaders(data_train_val, data_test,
                           seed=seed,
                           batch_size_train=batch_size_train,
                           batch_size_test=batch_size_test,
                           train_size=train_size,
                           val_size=None,
                           shuffle=shuffle
                           )
    val_dataloader = DataLoader(validation_dataset, shuffle=False, batch_size=batch_size_train)
    return train_dataloader, val_dataloader, test_dataloader


def get_omniglot_loaders(
        batch_size_train: int = 32,
        batch_size_test: int = 1,
        train_size: float = None,
        val_size: float = None,
        seed=None,
        shuffle: bool = True
):
    data_train_val = Omniglot(root='./', background=True, download=True)
    data_test = Omniglot(root='./', background=False, download=True)
    return get_dataloaders(data_train_val, data_test,
                           seed=seed,
                           batch_size_train=batch_size_train,
                           batch_size_test=batch_size_test,
                           train_size=train_size,
                           val_size=val_size,
                           shuffle=shuffle
                           )


def get_cifar_10_loaders(
        batch_size_train: int = 32,
        batch_size_test: int = 1,
        train_size: float = None,
        val_size: float = None,
        seed=None,
        shuffle: bool = True
):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    data_train_val: datasets.cifar.CIFAR10 = CIFAR10(root='./data', train=True, download=True, transform=transform)
    validation_size = 10000
    validation_start_index = len(data_train_val) - validation_size
    validation_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    validation_dataset.data = data_train_val.data[validation_start_index:]
    validation_dataset.targets = data_train_val.targets[validation_start_index:]
    data_train_val.data = data_train_val.data[:validation_start_index]
    data_train_val.targets = data_train_val.targets[:validation_start_index]
    data_test = CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_dataloader, _, test_dataloader = get_dataloaders(data_train_val, data_test,
                                                           seed=seed,
                                                           batch_size_train=batch_size_train,
                                                           batch_size_test=batch_size_test,
                                                           train_size=train_size,
                                                           val_size=None,
                                                           shuffle=shuffle
                                                           )
    val_dataloader = DataLoader(validation_dataset, shuffle=False, batch_size=batch_size_train)
    return train_dataloader, val_dataloader, test_dataloader





def get_svhn_loaders(
        batch_size_train: int = 32,
        batch_size_test: int = 1,
        train_size: float = None,
        val_size: float = None,
        seed=None,
        shuffle: bool = True
):
    data_train_val = SVHN(root='./data', split='train', download=True)
    data_val = SVHN(root='./data', split='extra', download=True)
    data_test = SVHN(root='./data', split='test', download=True)
    train_dataloader, _, test_dataloader = get_dataloaders(data_train_val, data_test,
                                                           seed=seed,
                                                           batch_size_train=batch_size_train,
                                                           batch_size_test=batch_size_test,
                                                           train_size=train_size,
                                                           val_size=None,
                                                           shuffle=shuffle
                                                           )
    val_dataloader = DataLoader(data_val, batch_size=batch_size_train, shuffle=False)
    return train_dataloader, val_dataloader, test_dataloader


def get_celabA_loaders(
        batch_size_train: int = 32,
        batch_size_test: int = 1,
        train_size: float = None,
        val_size: float = None,
        seed=None,
        shuffle: bool = True
):
    data_train = CelebA(root='./data', split='train', download=True)
    data_val = CelebA(root='./data', split='valid', download=True)
    data_test = CelebA(root='./data', split='test', download=True)
    train_dataloader, _, test_dataloader = get_dataloaders(data_train, data_test,
                                                           seed=seed,
                                                           batch_size_train=batch_size_train,
                                                           batch_size_test=batch_size_test,
                                                           train_size=train_size,
                                                           val_size=None,
                                                           shuffle=shuffle
                                                           )
    val_dataloader = DataLoader(data_val, batch_size=batch_size_train, shuffle=False)
    return train_dataloader, val_dataloader, test_dataloader


def get_yahoo_loaders(
        batch_size_train: int = 32,
        batch_size_test: int = 1,
        train_size: float = None,
        val_size: float = None,
        seed=None,
        shuffle: bool = True
):
    raise NotImplementedError('Yahoo dataset not found yet')

