import numpy as np
import pandas as pd
import torch.nn
import torch
import torch.optim as optim
from models.resnet_vae import ResNetVae
from models.cnn_vae import CnnVae
from loss_function.loss import GaussianVAELoss, NonGaussianVAELoss
from torch.distributions import Uniform
from learning_scheduler.WarmupCosineLearningRateScheduler import WarmupCosineDecayScheduler

DEVICE = torch.device('cuda' if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else 'cpu')
# DEVICE = "cpu"
import os

from tqdm.notebook import tqdm


class ResNetTrainer:

    def __init__(self, loaders: tuple, a: float, b: float, beta=1.,
                 lr=1e-3, latent_dimension=32, warmup_phase=10, epochs=200):

        model = ResNetVae(latent_dimension=latent_dimension,
                          use_multi_rate=True)

        self.model = model.to(DEVICE)
        self.train_loader, self.val_loader, self.test_loader = loaders
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        # images are binary
        self.loss_fn = NonGaussianVAELoss().to(DEVICE)

        self.name = f'mrvae_{lr}_{1.}'
        self.beta_distribution = Uniform(low=np.log(a), high=np.log(b))

        self.beta = beta
        self.lr = lr
        self.epochs = epochs
        self.len_training = len(self.train_loader)
        self.scheduler = WarmupCosineDecayScheduler(
            self.optimizer,
            warmup_epochs=warmup_phase,
            total_epochs=epochs
        )

        self.max_val_counter = 30
        if b == 0.1:
            self.max_val_counter = 5
        elif b == 1:
            self.max_val_counter = 15

        self.best_loss = np.inf
        self.val_counter = 0

    def sample_beta(self, batch_size=0):
        with torch.no_grad():
            beta = self.beta_distribution.sample(sample_shape=torch.Size((batch_size, 1)))
            beta = beta.to(DEVICE)
            # normalize
        return beta

    def train(self):
        for ep in tqdm(range(self.epochs)):
            self.model.train()
            # sample mini-batches
            for inputs, _ in self.train_loader:
                inputs = inputs.to(DEVICE)
                # sample beta
                log_betas = self.sample_beta(batch_size=inputs.shape[0])
                self.optimizer.zero_grad(set_to_none=True)
                x_pred, (mu, logvar) = self.model.forward(inputs, log_betas)
                loss, *_ = self.loss_fn(x_pred, inputs, mu, logvar, torch.exp(log_betas).squeeze(-1))
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()

            # early stopping
            val_loss = self.best_on_validation()
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.val_counter = 0
            else:
                self.val_counter += 1
                if self.val_counter >= self.max_val_counter:
                    print('Early stopping at epoch:', ep + 1)
                    break
        return self.best_loss

            # print(f'Ep {ep + 1}; loss:{ep_loss / len(self.train_loader)}')

    def best_on_validation(self):
        self.model.eval()
        with torch.no_grad():
            train_loss = 0
            for inputs, _ in self.val_loader:
                inputs = inputs.to(DEVICE)
                beta = torch.ones(size=(inputs.shape[0], 1), device=DEVICE, dtype=torch.float32)
                beta_loss = beta.squeeze(-1)
                beta = torch.log(beta)
                x_pred, (mu, logvar) = self.model.forward(inputs, beta)
                loss, *_ = self.loss_fn(x_pred, inputs, mu, logvar, beta_loss)
                train_loss += loss.item()
        train_loss = train_loss / len(self.val_loader)
        return train_loss

    def rate_distortion_curve_value(self, beta_in: float, beta_loss: float):
        rates = 0
        distortions = 0
        losses = 0
        self.model.eval()
        with torch.no_grad():
            for inputs, _ in self.test_loader:
                inputs = inputs.to(DEVICE)
                beta_in_el = torch.full(size=(inputs.shape[0], 1), fill_value=beta_in, device=DEVICE,
                                        dtype=torch.float32)
                beta_loss_el = torch.exp(beta_in_el).squeeze(-1)
                x_pred, (mu, logvar) = self.model.forward(inputs, beta_in_el)
                loss, (rate, distortion) = self.loss_fn(x_pred, inputs, mu, logvar, beta_loss_el)
                rate = rate.detach().cpu().item()
                distortion = distortion.detach().cpu().item()
                rates += rate
                distortions += distortion
                losses += loss.item()
        losses = losses / len(self.test_loader)
        rates = rates / len(self.test_loader)
        distortions = distortions / len(self.test_loader)
        return losses, (rates, distortions)


class ExperimentThree:
    def __init__(self):
        self.dfs_beta_fixed = []
        self.dfs_alpha_fixed = []
        self.a_set = [0.001, 0.01, 0.1, 1]
        self.b_set = [10, 1, 0.1]
        self.lrs = [0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003, 0.00001]
        self.seeds = [1, 10, 100]

    def fixed_b_vary_a(self, loaders):
        b = 10.
        for a in self.a_set:
            for lr in self.lrs:
                mean_tr = []
                for seed in self.seeds:
                    torch.manual_seed(seed)
                    trainer = ResNetTrainer(loaders, a, b, lr=lr, latent_dimension=64)
                    best_loss = trainer.train()
                    mean_tr.append(best_loss)
                dictionary = {
                    'a': a,
                    'b': b,
                    'lr': lr,
                    'mean_loss': np.mean(mean_tr)
                }
                self.dfs_beta_fixed.append(pd.DataFrame(dictionary, index=[0]))

        return pd.concat(self.dfs_beta_fixed)

    def fixed_a_vary_b(self, loaders):
        a = 0.01
        for b in self.b_set:
            for lr in self.lrs:
                mean_tr = []
                for seed in self.seeds:
                    torch.manual_seed(seed)
                    trainer = ResNetTrainer(loaders, a, b, lr=lr, latent_dimension=64)
                    best_loss = trainer.train()
                    mean_tr.append(best_loss)
                dictionary = {
                    'a': a,
                    'b': b,
                    'lr': lr,
                    'mean_loss': np.mean(mean_tr)
                }
                self.dfs_alpha_fixed.append(pd.DataFrame(dictionary, index=[0]))
        return pd.concat(self.dfs_alpha_fixed)

    def conduct_experiment(self, loaders, path='experiment_3'):
        print('Using device:', DEVICE)
        if len(path.split('/')) > 1:
            tmp_path = path.split('/')[0]
            if not os.path.exists(tmp_path):
                os.mkdir(tmp_path)
        if not os.path.exists(path):
            os.mkdir(path)
        df_fixed_beta = self.fixed_b_vary_a(loaders)
        df_fixed_alpha = self.fixed_a_vary_b(loaders)
        df_fixed_beta.to_csv(f'{path}/fixed_beta.csv')
        df_fixed_alpha.to_csv(f'{path}/fixed_alpha.csv')



