from typing import Optional

import torch
import torch.nn as nn
from layers import PositionwiseFeedForward

from attention import MultiHeadAttention


class TransformerDecoderLayer(nn.Module):
    """Transformer Decoder Layer"""

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
        self.cross_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ff = PositionwiseFeedForward(embed_dim, ff_dim, dropout, activation)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.norm_first = norm_first

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            tgt: (batch_size, seq_len, embed_dim)
            memory: Encoder output (batch_size, src_seq_len, embed_dim)
            tgt_mask: Optional attention mask (seq_len, seq_len)
            memory_mask: Optional cross-attention mask (seq_len, src_seq_len)
            tgt_key_padding_mask: Optional mask for padded tokens (batch_size, seq_len)
            memory_key_padding_mask: Optional mask for padded source tokens (batch_size, src_seq_len)
        """
        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
            x = x + self._ca_block(
                self.norm2(x), memory, memory_mask, memory_key_padding_mask
            )
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask))
            x = self.norm2(
                x + self._ca_block(x, memory, memory_mask, memory_key_padding_mask)
            )
            x = self.norm3(x + self._ff_block(x))
        return x

    def _sa_block(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        x, _ = self.self_attn(x, x, x, key_padding_mask, attn_mask)
        return self.dropout1(x)

    def _ca_block(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        x, _ = self.cross_attn(x, memory, memory, key_padding_mask, attn_mask)
        return self.dropout2(x)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout3(self.ff(x))
