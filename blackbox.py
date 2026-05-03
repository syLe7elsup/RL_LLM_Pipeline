"""Blackbox classifier M: small MLP that maps a derived-feature trajectory
vector (32-dim, see ``features.py``) to a binary high/low-outcome label.

Conservative sizing to avoid overfitting (n ~ 10000, F = 32):
    32 -> 64 -> 64 -> 2     (~7k params, data/param ~ 0.7)
    dropout 0.1 between hidden layers
    weight_decay 1e-3
    early stopping on val loss

We expose the second hidden layer's post-activation as the representation
that the SAE will be trained on. (h2 in ``representation``.)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def auto_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class OutcomeMLP(nn.Module):
    def __init__(self, in_dim: int = 32, hidden_dim: int = 64, n_classes: int = 2, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, n_classes)
        self.drop = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim

    def representation(self, x: torch.Tensor) -> torch.Tensor:
        h1 = F.relu(self.fc1(x))
        h1 = self.drop(h1)
        h2 = F.relu(self.fc2(h1))
        return h2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h2 = self.representation(x)
        h2 = self.drop(h2)
        return self.head(h2)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return F.softmax(self.forward(x), dim=-1)


@dataclass
class TrainResult:
    train_acc: float
    val_acc: float
    train_losses: list = field(default_factory=list)
    val_losses: list = field(default_factory=list)
    best_epoch: int = 0
    overfit_warning: bool = False  # True if (train_acc - val_acc) > 0.05


def train_blackbox(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    hidden_dim: int = 64,
    epochs: int = 300,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    dropout: float = 0.1,
    batch_size: int = 256,
    patience: int = 10,
    device: torch.device | None = None,
    seed: int = 0,
    standardize: bool = True,
) -> tuple[OutcomeMLP, TrainResult, dict]:
    """Train the outcome MLP with early stopping. Returns (best_model, result, scaler).

    ``scaler`` is the dict ``{"mean": ..., "std": ...}`` used to standardize
    inputs; you'll need it later when feeding new trajectories to the model.
    """
    device = device or auto_device()
    torch.manual_seed(seed)

    if standardize:
        mean = X_train.mean(axis=0, keepdims=True)
        std = X_train.std(axis=0, keepdims=True) + 1e-8
    else:
        mean = np.zeros((1, X_train.shape[1]))
        std = np.ones((1, X_train.shape[1]))
    scaler = {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}

    X_tr = torch.from_numpy(((X_train - mean) / std).astype(np.float32)).to(device)
    y_tr = torch.from_numpy(y_train.astype(np.int64)).to(device)
    X_va = torch.from_numpy(((X_val - mean) / std).astype(np.float32)).to(device)
    y_va = torch.from_numpy(y_val.astype(np.int64)).to(device)

    model = OutcomeMLP(in_dim=X_train.shape[1], hidden_dim=hidden_dim, dropout=dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_no_improve = 0
    train_losses, val_losses = [], []

    n = X_tr.shape[0]
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            logits = model(X_tr[idx])
            loss = F.cross_entropy(logits, y_tr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * idx.numel()
        train_losses.append(epoch_loss / n)

        model.eval()
        with torch.no_grad():
            val_logits = model(X_va)
            val_loss = F.cross_entropy(val_logits, y_va).item()
        val_losses.append(val_loss)

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        train_acc = (model(X_tr).argmax(-1) == y_tr).float().mean().item()
        val_acc = (model(X_va).argmax(-1) == y_va).float().mean().item()
    overfit = (train_acc - val_acc) > 0.05

    return (
        model,
        TrainResult(
            train_acc=train_acc,
            val_acc=val_acc,
            train_losses=train_losses,
            val_losses=val_losses,
            best_epoch=best_epoch,
            overfit_warning=overfit,
        ),
        scaler,
    )


@torch.no_grad()
def collect_representations(
    model: OutcomeMLP,
    X: np.ndarray,
    scaler: dict,
    device: torch.device | None = None,
    batch_size: int = 1024,
) -> np.ndarray:
    """Run the (already-trained) MLP and return the second-hidden-layer activations."""
    device = device or auto_device()
    model.eval()
    X_norm = ((X - scaler["mean"]) / scaler["std"]).astype(np.float32)
    X_t = torch.from_numpy(X_norm).to(device)
    out = []
    for i in range(0, X_t.shape[0], batch_size):
        out.append(model.representation(X_t[i : i + batch_size]).cpu())
    return torch.cat(out, dim=0).numpy()
