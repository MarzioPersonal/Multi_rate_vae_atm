import torch.nn
import torch
import torch.optim as optim


class Trainer:
    def __init__(self, model: torch.nn.Module, loaders: tuple, lr=1e-3):
        self.model = model
        self.train_loader, self.val_loader, self.test_loader = loaders
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self, epochs, warmup=0):
        for ep in range(epochs):
            ep_loss = 0
            self.model.train()
            for inputs, _ in self.train_loader:
                self.optimizer.zero_grad()
                *_, loss = self.model.forward(inputs)
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
        with torch.no_grad():
            for inputs, _ in self.val_loader:
                *_, loss = self.model.forward(inputs)
                val_loss += loss.item()

        print(f'Val loss: {val_loss / len(self.val_loader)}')

    def test(self):
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for inputs, _ in self.test_loader:
                *_, loss = self.model.forward(inputs)
                test_loss += loss.item()
        print(f'Test loss: {test_loss / len(self.test_loader)}')
