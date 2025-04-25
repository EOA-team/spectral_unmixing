import torch
import torch.nn as nn

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class SimpleNN(nn.Module):
    def __init__(self, input_dim, num_layers=2, hidden_dim=64, batch_size=32, epochs=100, lr=0.001, optimizer='adam'):
        super().__init__()

        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.optimizer_name = optimizer.lower()

        # Define layers
        dims = [input_dim] + [hidden_dim] * num_layers + [1]
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.model = nn.Sequential(*layers)

        # Device and optimization setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = self._get_optimizer()

    def _get_optimizer(self):
        if self.optimizer_name == 'adam':
            return torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer_name == 'sgd':
            return torch.optim.SGD(self.parameters(), lr=self.lr)
        elif self.optimizer_name == 'rmsprop':
            return torch.optim.RMSprop(self.parameters(), lr=self.lr)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")

    def forward(self, x):
        return self.model(x)

    def fit(self, X_train, y_train):
        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        train_ds = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        self.train()
        for epoch in range(self.epochs):
            print(f"Epoch {epoch}")
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                self.optimizer.zero_grad()
                preds = self(xb)
                loss = self.criterion(preds, yb)
                loss.backward()
                self.optimizer.step()

    def predict(self, X_test):
        self.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X_test, dtype=torch.float32).to(self.device)
            y_pred = self(X_tensor).cpu().squeeze().numpy()
        return y_pred
