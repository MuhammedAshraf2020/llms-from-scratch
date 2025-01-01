from typing import Optional

import torch
import torch.nn as nn
from layers import PositionwiseFeedForward

from attention import MultiHeadAttention


class TransformerEncoderLayer(nn.Module):
    """Transformer Encoder Layer"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
        activation: str = "relu",
        norm_first: bool = False,
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ff = PositionwiseFeedForward(embed_dim, ff_dim, dropout, activation)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.norm_first = norm_first

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            src: (batch_size, seq_len, embed_dim)
            src_mask: Optional attention mask (seq_len, seq_len)
            src_key_padding_mask: Optional mask for padded tokens (batch_size, seq_len)
        """
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))
        return x

    def _sa_block(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        x, _ = self.self_attn(x, x, x, key_padding_mask, attn_mask)
        return self.dropout1(x)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout2(self.ff(x))
