import torch
import torch.nn as nn
import numpy as np
from scipy.sparse.csr import csr_matrix
import matplotlib.pyplot as plt; plt.style.use("fivethirtyeight")

RANDOM_SEED = 42


class WeightedLogisticRegression:
    def __init__(self, loss_weights=None, max_iter=300, l1=0., l2=0.0005, lr=0.1, tol=1e-4, plot_loss=False, verbose=False):
        np.random.seed(RANDOM_SEED)
        torch.manual_seed(RANDOM_SEED)
        self.loss_weights = loss_weights or (1, 1, 1, 1, 1)
        self.max_iter = max_iter
        self.weight_decay = l2
        self.l1 = l1
        self.lr = lr
        self.tol = tol
        self.plot_loss = plot_loss
        self.verbose = verbose
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
        self.w = torch.zeros(size=(n_features,)).requires_grad_()
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
            loss = criterion(logits, y) + self.l1 * torch.mean(torch.abs(self.w))
            if abs(loss.item() - previous_loss) < self.tol:
                if self.verbose:
                    print("Converged after {} iterations.".format(i))
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


class WeightedNeuralNetwork:
    def __init__(self, loss_weights=None, hidden_size=128, max_iter=300, l1=0., l2=0.0005, lr=0.1, tol=1e-4,
                 plot_loss=False, verbose=False):
        np.random.seed(RANDOM_SEED)
        torch.manual_seed(RANDOM_SEED)
        self.loss_weights = loss_weights or (1, 1, 1, 1, 1)
        self.hidden_size = hidden_size
        self.max_iter = max_iter
        self.weight_decay = l2
        self.l1 = l1
        self.lr = lr
        self.tol = tol
        self.plot_loss = plot_loss
        self.verbose = verbose
        self.w1 = None
        self.b1 = None
        self.w2 = None
        self.b2 = None
        self.losses = []

    @staticmethod
    def tensor(x):
        if type(x) == csr_matrix:
            x = x.toarray()
        if type(x) != torch.Tensor:
            x = torch.tensor(x)
        return torch.t(x.float())

    def forward(self, x):
        h = self.w1 @ x + self.b1
        logits = (self.w2 @ h + self.b2).view(-1)
        return logits

    def fit(self, x, y_multi):
        self.losses = []
        x = self.tensor(x)
        y = torch.tensor([1 if y == 0 else 0 for y in y_multi]).float()
        n_features = x.shape[0]
        sd = 0.001
        self.w1 = (sd * torch.randn(size=(self.hidden_size, n_features))).requires_grad_()
        self.b1 = (sd * torch.randn(size=(self.hidden_size, 1))).requires_grad_()
        self.w2 = (sd * torch.randn(size=(1, self.hidden_size))).requires_grad_()
        self.b2 = (sd * torch.randn(size=(1, self.hidden_size))).requires_grad_()
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
        optimizer = torch.optim.Adam((self.w1, self.w2, self.b1, self.b2), lr=self.lr, weight_decay=self.weight_decay)
        previous_loss = -100
        for i in range(self.max_iter):
            optimizer.zero_grad()
            logits = self.forward(x)
            loss = criterion(logits, y) + self.l1 * (torch.mean(torch.abs(self.w1)) + torch.mean(torch.abs(self.w2)))
            if abs(loss.item() - previous_loss) < self.tol:
                if self.verbose:
                    print("Converged after {} iterations.".format(i))
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
        logits = self.forward(x)
        probas = torch.sigmoid(logits)
        y_pred = np.array([1 if p > 0.5 else 0 for p in probas])
        return y_pred

    def predict_proba(self, x):
        x = self.tensor(x)
        logits = self.forward(x)
        probas = torch.sigmoid(logits).detach()
        return torch.stack([1 - probas, probas]).transpose(0, 1).numpy()
