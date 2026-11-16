"""
Training engine, early stopping, and evaluation routines for PyTorch forecasting models.
"""

import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class EarlyStopping:
    """
    Early stops training when validation loss stops improving after a given patience.
    Saves and reloads the best model checkpoint weights.
    """
    def __init__(self, patience: int = 5, min_delta: float = 1e-4, checkpoint_path: Optional[str] = None):
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_path = checkpoint_path
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False
        self.best_epoch = 0

    def step(self, val_loss: float, model: nn.Module, epoch: int) -> bool:
        """
        Returns True if new best model is found, False otherwise.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch
            if self.checkpoint_path:
                os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
                torch.save(model.state_dict(), self.checkpoint_path)
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False

    def load_best_weights(self, model: nn.Module) -> None:
        """Reload the checkpoint weights corresponding to the lowest validation loss."""
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            model.load_state_dict(torch.load(self.checkpoint_path, weights_only=True))


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    clip_grad_norm: float = 1.0
) -> float:
    """Train the model for one full pass over the training DataLoader."""
    model.train()
    running_loss = 0.0
    total_samples = 0

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        predictions = model(batch_x)
        loss = criterion(predictions, batch_y)
        loss.backward()

        # Gradient clipping prevents exploding gradients common in unrolled RNNs
        if clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)

        optimizer.step()

        running_loss += loss.item() * batch_x.size(0)
        total_samples += batch_x.size(0)

    return running_loss / total_samples


def evaluate_loss(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """Evaluate loss on a dataset (typically validation) without computing gradients."""
    model.eval()
    running_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)

            running_loss += loss.item() * batch_x.size(0)
            total_samples += batch_x.size(0)

    return running_loss / total_samples


def predict_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> np.ndarray:
    """Generate multi-horizon forecast predictions for an entire DataLoader."""
    model.eval()
    preds_list = []

    with torch.no_grad():
        for batch_x, _ in dataloader:
            batch_x = batch_x.to(device)
            preds = model(batch_x)
            preds_list.append(preds.cpu().numpy())

    return np.concatenate(preds_list, axis=0)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    checkpoint_name: str,
    device: torch.device
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Complete training pipeline with validation monitoring and early stopping.
    """
    epochs = config["training"]["epochs"]
    lr = config["training"]["learning_rate"]
    weight_decay = config["training"]["weight_decay"]
    patience = config["training"]["patience"]
    clip_grad = config["training"]["clip_grad_norm"]

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    checkpoint_path = os.path.join(config["results_dir"], "checkpoints", f"{checkpoint_name}.pt")
    early_stopping = EarlyStopping(patience=patience, checkpoint_path=checkpoint_path)

    history = {"train_loss": [], "val_loss": []}

    print(f"Starting training on {device} for {epochs} max epochs (patience={patience})...")
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, clip_grad)
        val_loss = evaluate_loss(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        improved = early_stopping.step(val_loss, model, epoch)
        marker = "*" if improved else " "
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f} {marker}")

        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch}. Restoring best epoch {early_stopping.best_epoch}.")
            break

    # Restore best checkpoint
    early_stopping.load_best_weights(model)
    print(f"Best validation MSE: {early_stopping.best_loss:.6f} from epoch {early_stopping.best_epoch}")

    return model, history
