import numpy as np
import pandas as pd
import torch.nn
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from models.linearVae import LinearVae
from loss_function.loss import GaussianVAELoss
from distributions.beta_distribution import BetaUniform
from learning_scheduler.WarmupCosineLearningRateScheduler import WarmupCosineDecayScheduler

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import os

from tqdm.notebook import tqdm


class LinearTrainer:
    def __init__(self, loaders: tuple, use_multi_rate=False, beta=1., lr=1e-3, warmup_phase=10, epochs=200):
        self.model = LinearVae(use_multi_rate=use_multi_rate).to(DEVICE)
        # self.model = torch.compile(self.model)
        self.train_loader, self.val_loader, self.test_loader = loaders
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = GaussianVAELoss().to(DEVICE)
        self.use_multi_rate = use_multi_rate
        if use_multi_rate:
            self.name = f'mrvae_{lr}_{1.}'
            self.beta_distribution = BetaUniform()
        else:
            self.name = f'beta_vae_{lr}_{beta}'
        self.beta = beta
        self.lr = lr
        self.epochs = epochs
        self.len_training = len(self.train_loader)
        self.scheduler = WarmupCosineDecayScheduler(
            self.optimizer,
            warmup_epochs=warmup_phase,
            total_epochs=epochs
        )

        self.best_loss = np.inf
        self.val_counter = 0

    def sample_beta(self, batch_size=0):
        if self.use_multi_rate:
            with torch.no_grad():
                beta = self.beta_distribution.sample(sample_shape=torch.Size((batch_size, 1)))
                beta = beta.to(DEVICE, non_blocking=True)
                # normalize
            return beta
        return self.beta

    def train(self):
        for ep in tqdm(range(self.epochs)):
            # ep_loss = 0
            self.model.train()
            # sample mini-batches
            for inputs, _ in self.train_loader:
                inputs = inputs.to(DEVICE, non_blocking=True)
                # sample beta
                log_betas = self.sample_beta(batch_size=inputs.shape[0])
                self.optimizer.zero_grad(set_to_none=True)
                x_pred, (mu, logvar) = self.model.forward(inputs, log_betas)
                if self.use_multi_rate:
                    loss, *_ = self.loss_fn(x_pred, inputs, mu, logvar, torch.exp(log_betas).squeeze(-1))
                else:
                    loss, *_ = self.loss_fn(x_pred, inputs, mu, logvar, self.beta)
                loss.backward()
                self.optimizer.step()

                # ep_loss += loss.item()
            self.scheduler.step()
            val_loss = self.best_on_validation()
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.val_counter = 0
            else:
                self.val_counter += 1
                if self.val_counter >= 40:
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
                if self.use_multi_rate:
                    beta = torch.ones(size=(inputs.shape[0], 1), device=DEVICE, dtype=torch.float32)
                    beta_loss = beta.squeeze(-1)
                    beta = torch.log(beta)
                else:
                    beta = self.beta
                    beta_loss = beta
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
                if self.use_multi_rate:
                    beta_in_el = torch.full(size=(inputs.shape[0], 1), fill_value=beta_in, device=DEVICE, dtype=torch.float32)
                    beta_loss_el = torch.exp(beta_in_el).squeeze(-1)
                else:
                    beta_in_el = beta_in
                    beta_loss_el = beta_loss
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

    # def save_model(self, path: str):
    #     if not path.endswith('model'):
    #         path = path + '/model/'
    #         if not os.path.exists(path):
    #             os.mkdir(path)
    #     if self.use_multi_rate:
    #         postfix = 'mae_vae'
    #     else:
    #         postfix = 'beta_vae'
    #     filename = path + f'{self.beta}_{self.lr}_' + postfix + '.pt'
    #     torch.save(self.model.state_dict(), filename)


class GridSearcher:
    def __init__(self, loaders):
        self.lrs = [0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003, 0.00001]
        self.betas = np.linspace(np.log(0.01), np.log(10), num=10)
        self.mrvae_models = []
        self.beta_models = []
        self.seeds = [1, 10, 100]

        self.dfs_beta: list[pd.DataFrame] = []
        self.dfs_mr_vae: list[pd.DataFrame] = []
        self.loaders = loaders

    def b_vae_(self):
        use_multi_rate = False
        print('Beta-VAE')
        for b in self.betas:
            b = np.exp(b)
            for lr in self.lrs:
                print(f'conducting experiment 1 with beta: {b} and learning rate: {lr}')
                mean_tr = []
                for s in self.seeds:
                    print(f'using seed: {s}')
                    torch.manual_seed(s)
                    np.random.seed(s)
                    tr = LinearTrainer(loaders=self.loaders, use_multi_rate=False, beta=b, lr=lr)

                    train_loss = tr.train()
                    mean_tr.append(train_loss)
                dictionary = {
                    'use_multi_rate': [use_multi_rate],
                    'mean_loss': [np.mean(mean_tr)],
                    'beta': [b],
                    'lr': [lr]
                }
                self.dfs_beta.append(pd.DataFrame(dictionary))

        return pd.concat(self.dfs_beta)

    def mr_vae_(self):
        print('MR-VAE')
        use_multi_rate = True
        for lr in self.lrs:
            print(f'conducting experiment 1 and learning rate: {lr}')
            mean_tr = []
            for s in self.seeds:
                torch.manual_seed(s)
                np.random.seed(s)
                tr = LinearTrainer(loaders=self.loaders, use_multi_rate=use_multi_rate, lr=lr)

                train_loss = tr.train()
                mean_tr.append(train_loss)
            dictionary = {
                'use_multi_rate': [use_multi_rate],
                'mean_loss': [np.mean(mean_tr)],
                'beta': [1.],
                'lr': [lr]
            }
            self.dfs_mr_vae.append(pd.DataFrame(dictionary))

        return pd.concat(self.dfs_mr_vae)

    def conduct_experiment(self, path='experiment_1', do_only_mrvae=False):
        if not os.path.exists(path):
            os.mkdir(path)
        if not do_only_mrvae:
            df_beta_vae = self.b_vae_()
            # df_mr_vae = self.mr_vae_()
            df_beta_vae.to_csv(f'{path}/beta_vae.csv')
            # df_mr_vae.to_csv(f'{path}/mr_vae.csv')
        else:
            df_mr_vae = self.mr_vae_()
            df_mr_vae.to_csv(f'{path}/mr_vae.csv')