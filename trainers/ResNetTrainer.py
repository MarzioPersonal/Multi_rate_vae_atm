from torch import optim
import torch

from distributions.beta_distribution import BetaUniform
from learning_scheduler.WarmupCosineLearningRateScheduler import WarmupCosineDecayScheduler
from models.resnet_vae import ResNetVae
from loss_function.loss import NonGaussianVAELoss

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from tqdm import tqdm


class ResNetTrainer:
    def __init__(self, loaders, latent_dim, is_cifar, is_celab, lr, beta, use_multi_rate=False, epochs=200, warmup_phase=10, beta_uniform=BetaUniform):
        self.model = ResNetVae(latent_dim, is_cifar, is_celab, use_multi_rate)
        self.train_loader, self.val_loader, self.test_loader = loaders
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = NonGaussianVAELoss().to(DEVICE)
        self.use_multi_rate = use_multi_rate
        if use_multi_rate:
            self.name = f'mrvae_{lr}_{1.}'
            self.beta_distribution = beta_uniform
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

    def sample_beta(self, batch_size=0):
        if self.use_multi_rate:
            with torch.no_grad():
                beta = self.beta_distribution.sample(sample_shape=torch.Size((batch_size, 1)))
                beta = beta.to(DEVICE)
                # normalize
            return beta
        return self.beta

    def train(self):
        for ep in tqdm(range(self.epochs)):
            ep_loss = 0
            self.model.train()
            # sample mini-batches
            for inputs, _ in self.train_loader:
                inputs = inputs.to(DEVICE)
                # sample beta
                log_betas = self.sample_beta(batch_size=inputs.shape[0])
                self.optimizer.zero_grad()
                x_pred, (mu, logvar) = self.model.forward(inputs, log_betas)
                loss, *_ = self.loss_fn(x_pred, inputs, mu, logvar, self.beta)
                loss.backward()
                self.optimizer.step()

                ep_loss += loss.item()
            self.scheduler.step()
            # print(f'Ep {ep + 1}; loss:{ep_loss / len(self.train_loader)}')

    def best_on_validation(self):
        self.model.eval()
        with torch.no_grad():
            train_loss = 0
            for inputs, _ in self.val_loader:
                inputs = inputs.to(DEVICE)
                if self.use_multi_rate:
                    beta = torch.ones(size=(inputs.shape[0], 1), device=DEVICE)
                    beta = torch.log(beta)
                    beta_loss = 1.
                else:
                    beta = self.beta
                    beta_loss = beta
                x_pred, (mu, logvar) = self.model.forward(inputs, beta)
                loss, *_ = self.loss_fn(x_pred, inputs, mu, logvar, beta_loss)
                train_loss += loss.item()
        train_loss = train_loss / len(self.val_loader)
        return train_loss

    def rate_distortion_curve_value(self, beta_in: float, beta_loss: float):
        rec_losses = 0
        kdl_losses = 0
        losses = 0
        with torch.no_grad():
            for inputs, _ in self.test_loader:
                inputs = inputs.to(DEVICE)
                if self.use_multi_rate:
                    beta_in = torch.tensor([[beta_in]], dtype=torch.float32, device=DEVICE)
                x_pred, (mu, logvar) = self.model.forward(inputs, beta_in)
                loss, (rec_loss, kdl_loss) = self.loss_fn(x_pred, inputs, mu, logvar, beta_loss)
                rec_losses += rec_loss
                kdl_losses += kdl_loss
                losses += loss.item()
        losses = losses / len(self.test_loader)
        kdl_losses = kdl_losses / len(self.test_loader)
        rec_losses = rec_losses / len(self.test_loader)
        return losses, (rec_losses, kdl_losses)



class ExperimentThree:
    def __init__(self):
        a_set = [0.001, 0.01, 0.1, 1]
        b_set = [10, 1, 0.1]
        lr = 1e-3

    def fixed_b_vary_a(self):
        trainer = ResNetTrainer()