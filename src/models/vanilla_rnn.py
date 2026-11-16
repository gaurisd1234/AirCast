"""
Vanilla (Elman) Recurrent Neural Network for multi-horizon air quality forecasting.
"""

import torch
import torch.nn as nn


class VanillaRNNAirQuality(nn.Module):
    """
    Multi-horizon Vanilla RNN forecaster.
    
    Architecture:
    Input:  Tensor of shape (batch_size, lookback_seq_len, num_features)
    RNN:    Multi-layer Elman RNN with tanh non-linearity
            h_t = tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)
    Head:   Linear projection mapping final hidden state h_L to (batch_size, horizon)
    """
    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        horizon: int = 24,
        dropout: float = 0.2
    ):
        super().__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.horizon = horizon

        # nn.RNN applies tanh recurrent updates.
        # batch_first=True expects tensors of shape (batch, seq, feature) instead of (seq, batch, feature).
        # Note: PyTorch only applies dropout between recurrent layers if num_layers > 1.
        self.rnn = nn.RNN(
            input_size=num_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity="tanh",
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.dropout = nn.Dropout(dropout)
        
        # Final linear projection from hidden dimension to all future forecast horizons
        self.fc = nn.Linear(hidden_dim, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        x: (batch_size, seq_len, num_features)
        Returns: (batch_size, horizon)
        """
        # out shape: (batch_size, seq_len, hidden_dim)
        # h_n shape: (num_layers, batch_size, hidden_dim)
        out, _ = self.rnn(x)

        # Extract the hidden state from the final timestep of the sequence (t = seq_len - 1).
        # This vector summarizes all sequential information gathered across the lookback window.
        last_hidden = out[:, -1, :]  # Shape: (batch_size, hidden_dim)

        # Apply dropout regularization before the output projection
        out_drop = self.dropout(last_hidden)

        # Project directly to multi-horizon predictions [y_{t+1}, y_{t+2}, ..., y_{t+H}]
        predictions = self.fc(out_drop)  # Shape: (batch_size, horizon)

        return predictions
