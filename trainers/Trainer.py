import torch.nn
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from models.linearVae import LinearVae
from loss_function.loss import gaussian_vae_loss
from distributions.beta_distribution import BetaUniform


class LinearTrainer:
    def __init__(self, loaders: tuple, use_multi_rate=False, beta=1.):
        self.model = LinearVae(use_multi_rate=use_multi_rate)
        self.train_loader, self.val_loader, self.test_loader = loaders
        self.optimizer = optim.Adam(self.model.parameters())
        self.loss_fn = gaussian_vae_loss
        self.use_multi_rate = use_multi_rate
        if use_multi_rate:
            self.beta_distribution = BetaUniform()
        self.beta = beta

    def sample_beta(self, batch_size=0):
        if self.use_multi_rate:
            with torch.no_grad():
                beta = self.beta_distribution.sample(sample_shape=torch.Size((batch_size, 1)))
                # normalize
            return beta
        return self.beta

    def train(self, epochs, warmup=0):
        for ep in range(epochs):
            ep_loss = 0
            self.model.train()
            # sample mini-batches
            for inputs, _ in self.train_loader:
                # sample beta
                log_betas = self.sample_beta(batch_size=inputs.shape[0])
                self.optimizer.zero_grad()
                x_pred, (mu, logvar) = self.model.forward(inputs, log_betas)
                loss, (rec_loss, kdl_loss) = self.loss_fn(x_pred, inputs, mu, logvar, self.beta)
                loss.backward()
                self.optimizer.step()
                ep_loss += loss.item()
            print(f'Ep {ep + 1}; loss:{ep_loss / len(self.train_loader)}')
            self.validate()

    def validate(self):
        if self.val_loader is None:
            return
        self.model.eval()
        val_loss = 0
        val_loss_rec = 0
        val_loss_kdl = 0
        with torch.no_grad():
            for inputs, _ in self.val_loader:
                beta = torch.ones(size=(inputs.shape[0], 1))
                x_pred, (mu, logvar) = self.model.forward(inputs, beta)
                loss, (rec_loss, kdl_loss) = self.loss_fn(x_pred, inputs, mu, logvar, 1.)
                val_loss += loss.item()
                val_loss_rec += rec_loss.item()
                val_loss_kdl += kdl_loss.item()

        print(f'Val loss: {val_loss / len(self.val_loader)}')
        print(f'REC: {val_loss_rec/ len(self.val_loader)} KDL: {val_loss_kdl/ len(self.val_loader)}')
        print('')
    #
    # def test(self):
    #     self.model.eval()
    #     test_loss = 0
    #     with torch.no_grad():
    #         for inputs, _ in self.test_loader:
    #             *_, loss = self.model.forward(inputs)
    #             test_loss += loss.item()
    #     print(f'Test loss: {test_loss / len(self.test_loader)}')
