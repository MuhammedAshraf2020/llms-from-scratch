import math

import torch
import torch.nn as nn
from torch.nn import functional as F


class PositionwiseFeedForward(nn.Module):
    """Position-wise Feed-Forward Network"""

    def __init__(
        self,
        embed_dim: int,
        ff_dim: int,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(p=dropout)

        self.activation = getattr(F, activation)

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters using Xavier uniform initialization"""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.constant_(self.fc2.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class PositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding"""

    def __init__(self, embed_dim: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, embed_dim)
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)
