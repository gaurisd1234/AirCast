"""
Long Short-Term Memory (LSTM) network for multi-horizon air quality forecasting.
"""

import torch
import torch.nn as nn


class LSTMAirQuality(nn.Module):
    """
    Multi-horizon LSTM forecaster.
    
    Architecture:
    Input:  Tensor of shape (batch_size, lookback_seq_len, num_features)
    LSTM:   Multi-layer Long Short-Term Memory network
            Gated cell state updates mitigate vanishing gradients:
            f_t = sigmoid(W_f [h_{t-1}, x_t] + b_f)  (Forget Gate)
            i_t = sigmoid(W_i [h_{t-1}, x_t] + b_i)  (Input Gate)
            C~_t = tanh(W_c [h_{t-1}, x_t] + b_c)    (Candidate Cell)
            C_t = f_t * C_{t-1} + i_t * C~_t         (Additive Cell Update)
            o_t = sigmoid(W_o [h_{t-1}, x_t] + b_o)  (Output Gate)
            h_t = o_t * tanh(C_t)                    (Hidden State)
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

        # nn.LSTM encapsulates the 4 internal gate transformations per layer.
        # batch_first=True expects (batch, seq, feature) tensors.
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.dropout = nn.Dropout(dropout)
        
        # Linear projection head mapping final LSTM hidden state to all future forecast horizons
        self.fc = nn.Linear(hidden_dim, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        x: (batch_size, seq_len, num_features)
        Returns: (batch_size, horizon)
        """
        # out shape: (batch_size, seq_len, hidden_dim)
        # h_n shape: (num_layers, batch_size, hidden_dim)
        # c_n shape: (num_layers, batch_size, hidden_dim)
        out, (h_n, c_n) = self.lstm(x)

        # Extract the hidden state corresponding to the final historical timestep (t = seq_len - 1).
        # This vector encodes both short-term dynamics and long-term memory retained through the cell carousel.
        last_hidden = out[:, -1, :]  # Shape: (batch_size, hidden_dim)

        # Apply dropout regularization to avoid co-adaptation of features
        out_drop = self.dropout(last_hidden)

        # Project directly to multi-horizon predictions [y_{t+1}, ..., y_{t+H}]
        predictions = self.fc(out_drop)  # Shape: (batch_size, horizon)

        return predictions
