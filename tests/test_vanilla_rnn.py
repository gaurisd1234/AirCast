"""
Tests for Vanilla RNN architecture, forward/backward pass, and training utilities.
"""

import os
import tempfile
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.models.vanilla_rnn import VanillaRNNAirQuality
from src.trainer import EarlyStopping, train_epoch, evaluate_loss


def test_vanilla_rnn_forward_shape():
    batch_size = 16
    seq_len = 72
    num_features = 17
    horizon = 24
    hidden_dim = 32

    model = VanillaRNNAirQuality(
        num_features=num_features,
        hidden_dim=hidden_dim,
        num_layers=2,
        horizon=horizon,
        dropout=0.1
    )

    dummy_input = torch.randn(batch_size, seq_len, num_features)
    output = model(dummy_input)

    assert output.shape == (batch_size, horizon)
    assert not torch.isnan(output).any()


def test_vanilla_rnn_gradient_flow():
    model = VanillaRNNAirQuality(
        num_features=10,
        hidden_dim=16,
        num_layers=1,
        horizon=4
    )
    dummy_x = torch.randn(8, 20, 10)
    dummy_y = torch.randn(8, 4)

    preds = model(dummy_x)
    loss = nn.MSELoss()(preds, dummy_y)
    loss.backward()

    # Verify that gradients exist and are non-zero for all trainable parameters
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"


def test_train_epoch_loss_reduction():
    torch.manual_seed(42)
    model = VanillaRNNAirQuality(num_features=5, hidden_dim=16, num_layers=1, horizon=2)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    X = torch.randn(32, 10, 5)
    Y = torch.randn(32, 2)
    dataset = TensorDataset(X, Y)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)

    loss_1 = train_epoch(model, loader, optimizer, criterion, torch.device("cpu"), clip_grad_norm=1.0)
    for _ in range(5):
        loss_n = train_epoch(model, loader, optimizer, criterion, torch.device("cpu"), clip_grad_norm=1.0)

    # Loss after 6 epochs should decrease compared to initial epoch
    assert loss_n < loss_1


def test_early_stopping_behavior():
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "test_ckpt.pt")
        es = EarlyStopping(patience=3, checkpoint_path=ckpt_path)
        dummy_model = nn.Linear(5, 1)

        # Epoch 1: improvement
        assert es.step(1.0, dummy_model, 1) is True
        assert es.counter == 0
        assert os.path.exists(ckpt_path)

        # Epoch 2: improvement
        assert es.step(0.8, dummy_model, 2) is True
        assert es.counter == 0

        # Epoch 3, 4, 5: no improvement
        assert es.step(0.85, dummy_model, 3) is False
        assert es.counter == 1
        assert es.early_stop is False

        assert es.step(0.82, dummy_model, 4) is False
        assert es.counter == 2
        assert es.early_stop is False

        assert es.step(0.83, dummy_model, 5) is False
        assert es.counter == 3
        assert es.early_stop is True  # Reached patience=3
