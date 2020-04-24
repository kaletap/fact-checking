import torch
import torch.nn as nn
import numpy as np
from scipy.sparse.csr import csr_matrix
import matplotlib.pyplot as plt; plt.style.use("fivethirtyeight")


class WeightedLogisticRegression:
    def __init__(self, loss_weights=None, max_iter=300, weight_decay=0.0005, lr=0.1, tol=1e-4, plot_loss=False):
        self.loss_weights = loss_weights or (1, 1, 1, 1, 1)
        self.max_iter = max_iter
        self.weight_decay = weight_decay
        self.lr = lr
        self.tol = tol
        self.plot_loss = plot_loss
        self.w = None
        self.losses = []

    @staticmethod
    def tensor(x):
        if type(x) == csr_matrix:
            x = x.toarray()
        if type(x) != torch.Tensor:
            x = torch.tensor(x)
        return x.float()

    def fit(self, x, y_multi):
        self.losses = []
        x = self.tensor(x)
        y = torch.tensor([1 if y == 0 else 0 for y in y_multi]).float()
        n_features = x.shape[1]
        self.w = torch.randn(size=(n_features,)).requires_grad_()
        weight_dict = {
            0: 1,
            1: self.loss_weights[0],
            2: self.loss_weights[1],
            3: self.loss_weights[2],
            4: self.loss_weights[3],
            5: self.loss_weights[4]
        }
        weight = torch.tensor([weight_dict[label] for label in y_multi])
        criterion = nn.BCEWithLogitsLoss(weight=weight)
        optimizer = torch.optim.Adam((self.w,), lr=self.lr, weight_decay=self.weight_decay)
        previous_loss = -100
        for i in range(self.max_iter):
            optimizer.zero_grad()
            logits = x @ self.w
            loss = criterion(logits, y)
            if abs(loss.item() - previous_loss) < self.tol:
                break
            loss.backward()
            self.losses.append(loss.item())
            optimizer.step()
            previous_loss = loss.item()

        if self.plot_loss:
            plt.plot(self.losses)
            plt.title("Loss")
            plt.xlabel("iteration")
            plt.show()

    def predict(self, x):
        x = self.tensor(x)
        logits = x @ self.w
        probas = torch.sigmoid(logits)
        y_pred = np.array([1 if p > 0.5 else 0 for p in probas])
        return y_pred

    def predict_proba(self, x):
        x = self.tensor(x)
        logits = x @ self.w
        probas = torch.sigmoid(logits).detach()
        return torch.stack([1 - probas, probas]).transpose(0, 1).numpy()
