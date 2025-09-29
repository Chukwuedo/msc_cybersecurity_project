from __future__ import annotations
from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


class NpDataset(Dataset):
    """Thin NumPy-backed dataset for tensors."""

    def __init__(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        assert isinstance(X, np.ndarray), "X must be a NumPy array"
        self.X = torch.from_numpy(X).float()
        self.y = None if y is None else torch.from_numpy(y).long()

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        if self.y is None:
            return self.X[idx]
        return self.X[idx], self.y[idx]


class TeacherNet(pl.LightningModule):
    """
    4-layer MLP teacher. Final layer outputs logits (no softmax).
    Supports post-hoc **temperature scaling** for calibration.
    """

    def __init__(
        self,
        in_dim: int,
        n_classes: int,
        hidden: Tuple[int, int, int] = (256, 128, 64),
        dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        h1, h2, h3 = hidden
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, h3),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h3, n_classes),
        )
        self.loss = nn.CrossEntropyLoss()
        # Temperature parameter for calibration (set to 1.0 by default)
        self.register_buffer("temperature", torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # logits

    def training_step(self, batch, _):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

    @torch.no_grad()
    def softmax_temperature(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling to logits and return calibrated probabilities."""
        T = self.temperature.clamp_min(1e-6)
        z = logits / T
        return torch.softmax(z, dim=1)


def train_teacher(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    *,
    max_epochs: int = 20,
    batch_size: int = 2048,
    num_workers: int = 4,
    hidden: Tuple[int, int, int] = (256, 128, 64),
    dropout: float = 0.1,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 42,
) -> TeacherNet:
    pl.seed_everything(seed, workers=True)
    n_classes = int(np.unique(y_train).size)
    model = TeacherNet(
        X_train.shape[1],
        n_classes,
        hidden=hidden,
        dropout=dropout,
        lr=lr,
        weight_decay=weight_decay,
    )

    train_loader = DataLoader(
        NpDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = None
    if X_val is not None and y_val is not None:
        val_loader = DataLoader(
            NpDataset(X_val, y_val),
            batch_size=batch_size * 2,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    callbacks = []
    if val_loader is not None:
        callbacks = [
            pl.callbacks.EarlyStopping(monitor="val_loss", patience=3, mode="min"),
            pl.callbacks.ModelCheckpoint(
                monitor="val_loss", mode="min", save_last=True
            ),
        ]

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices="auto",
        callbacks=callbacks,
        log_every_n_steps=50,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    return model


def predict_logits(
    model: TeacherNet, X: np.ndarray, batch_size: int = 8192, num_workers: int = 4
) -> np.ndarray:
    """Return raw logits (no softmax). Shape: [n_samples, n_classes]."""
    loader = DataLoader(
        NpDataset(X),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    model.eval()
    preds = []
    device = next(model.parameters()).device
    with torch.no_grad():
        for xb in loader:
            xb = xb.to(device)
            logits = model(xb)
            preds.append(logits.cpu().numpy())
    return np.concatenate(preds, axis=0)


def calibrate_temperature(
    model: TeacherNet,
    X_val: np.ndarray,
    y_val: np.ndarray,
    max_iters: int = 1000,
    lr: float = 0.01,
) -> float:
    """Fit temperature by minimising NLL on a held-out set (val/test)."""
    device = next(model.parameters()).device
    model.eval()
    T = torch.ones(1, device=device, requires_grad=True)
    optimiser = torch.optim.LBFGS([T], lr=lr, max_iter=max_iters)
    nll = nn.CrossEntropyLoss()

    X = torch.from_numpy(X_val).float().to(device)
    y = torch.from_numpy(y_val).long().to(device)

    def closure():
        optimiser.zero_grad()
        with torch.no_grad():
            logits = model(X)
        loss = nll(logits / T.clamp_min(1e-6), y)
        loss.backward()
        return loss

    optimiser.step(closure)
    with torch.no_grad():
        model.temperature.copy_(T.detach().cpu())
    return float(T.detach().cpu().item())


def logits_to_probs(logits: np.ndarray) -> np.ndarray:
    """Stable softmax (NumPy)."""
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def logits_to_calibrated_probs(model: TeacherNet, logits: np.ndarray) -> np.ndarray:
    """Apply learned temperature to logits to yield calibrated probabilities."""
    T = max(1e-6, float(model.temperature.item()))
    z = logits / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)
