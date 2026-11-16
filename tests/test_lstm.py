"""
Tests for LSTM architecture, gate dimensions, gradient flow, and forward pass.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.models.lstm import LSTMAirQuality
from src.models.vanilla_rnn import VanillaRNNAirQuality
from src.trainer import train_epoch


def test_lstm_forward_shape():
    batch_size = 16
    seq_len = 72
    num_features = 17
    horizon = 24
    hidden_dim = 32

    model = LSTMAirQuality(
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


def test_lstm_has_four_times_recurrent_weights_than_rnn():
    num_features = 10
    hidden_dim = 20

    rnn = VanillaRNNAirQuality(num_features=num_features, hidden_dim=hidden_dim, num_layers=1, horizon=4)
    lstm = LSTMAirQuality(num_features=num_features, hidden_dim=hidden_dim, num_layers=1, horizon=4)

    # In PyTorch, nn.RNN has weight_ih of shape (hidden_dim, input_size)
    # nn.LSTM has weight_ih of shape (4 * hidden_dim, input_size) for the 4 gates (i, f, g, o)
    rnn_weight_ih = rnn.rnn.weight_ih_l0
    lstm_weight_ih = lstm.lstm.weight_ih_l0

    assert lstm_weight_ih.shape[0] == 4 * rnn_weight_ih.shape[0]


def test_lstm_gradient_flow():
    model = LSTMAirQuality(
        num_features=10,
        hidden_dim=16,
        num_layers=2,
        horizon=4
    )
    dummy_x = torch.randn(8, 20, 10)
    dummy_y = torch.randn(8, 4)

    preds = model(dummy_x)
    loss = nn.MSELoss()(preds, dummy_y)
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"


def test_lstm_loss_reduction():
    torch.manual_seed(42)
    model = LSTMAirQuality(num_features=5, hidden_dim=16, num_layers=1, horizon=2)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    X = torch.randn(32, 10, 5)
    Y = torch.randn(32, 2)
    dataset = TensorDataset(X, Y)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)

    loss_1 = train_epoch(model, loader, optimizer, criterion, torch.device("cpu"), clip_grad_norm=1.0)
    for _ in range(5):
        loss_n = train_epoch(model, loader, optimizer, criterion, torch.device("cpu"), clip_grad_norm=1.0)

    assert loss_n < loss_1
