import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class SimpleNN(nn.Module):
    def __init__(self, input_dim, num_layers=2, hidden_dim=64, batch_size=32, epochs=100, lr=0.001, optimizer='Adam',
                momentum=None, scheduler=None, scheduler_params=None):
        super().__init__()

        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.optimizer_name = optimizer
        self.momentum = momentum
        self.scheduler_name = scheduler
        self.scheduler_params = scheduler_params or {}

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
        self.scheduler = self._get_scheduler()

    def _get_optimizer(self):
        if self.optimizer_name == 'Adam':
            return torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer_name == 'SGD':
            return torch.optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum)
        elif self.optimizer_name == 'AdamW':
            return torch.optim.AdamW(self.parameters(), lr=self.lr)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")

    def _get_scheduler(self):
        if self.scheduler_name is None:
            return None
        elif self.scheduler_name == 'ReduceLROnPlateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, **self.scheduler_params)
        elif self.scheduler_name == 'CosineAnnealingLR':
            return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, **{'T_max':50})
        else:
            raise ValueError(f"Unsupported scheduler: {self.scheduler_name}")


    def forward(self, x):
        return self.model(x)

    def fit(self, X_train, y_train, X_test, y_test):
        # Create a DataLoader for batch training
        train_ds = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        # Create a DataLoader for batch testing
        test_dataset = TensorDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False) 

        self.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                self.optimizer.zero_grad()
                preds = self(xb)
                loss = self.criterion(preds, yb)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            # Check test set loss
            self.model.eval()  # Set model to evaluation mode
            test_loss = 0.0

            with torch.no_grad():  # No need to track gradients during validation
                for batch_X, batch_y in test_loader:
                    outputs = self.model(batch_X)
                    loss = self.criterion(outputs, batch_y)
                    test_loss += loss.item() * batch_X.size(0)  # Accumulate batch loss
          
            test_loss /= len(test_loader.dataset)  # Average loss over all validation batches

            if self.scheduler_name == 'ReduceLROnPlateau':
                self.scheduler.step(test_loss)
            elif self.scheduler is not None:
                self.scheduler.step()


    def predict(self, X_test):
        self.eval()
        with torch.no_grad():
            X_tensor = X_test.clone().detach().to(self.device).float()
            y_pred = self(X_tensor).cpu().squeeze().numpy()
        return y_pred
