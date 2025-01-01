import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention mechanism"""

    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: (batch_size, num_heads, seq_len_q, head_dim)
            key: (batch_size, num_heads, seq_len_k, head_dim)
            value: (batch_size, num_heads, seq_len_v, head_dim)
            mask: Optional attention mask (seq_len_q, seq_len_k)
            key_padding_mask: Optional mask for padded keys (batch_size, seq_len_k)
        """
        d_k = query.size(-1)

        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        # Apply key padding mask if provided
        if key_padding_mask is not None:
            # Expand key_padding_mask to match attention scores dimensions
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(
                2
            )  # (batch_size, 1, 1, seq_len_k)
            scores = scores.masked_fill(key_padding_mask, float("-inf"))

        # Apply attention mask if provided
        if mask is not None:
            # Convert float mask to boolean if necessary
            if mask.dtype != torch.bool:
                mask = mask < 0.5
            # Expand mask to match attention scores dimensions
            mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len_q, seq_len_k)
            scores = scores.masked_fill(mask, float("-inf"))

        # Compute attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Compute output
        output = torch.matmul(attention_weights, value)

        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module"""

    def __init__(
        self, embed_dim: int, num_heads: int, dropout: float = 0.1, bias: bool = True
    ):
        super().__init__()
        assert (
            embed_dim % num_heads == 0
        ), f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = float(self.head_dim) ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(p=dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters using Xavier uniform initialization"""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0.0)
            nn.init.constant_(self.k_proj.bias, 0.0)
            nn.init.constant_(self.v_proj.bias, 0.0)
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            query: (batch_size, seq_len_q, embed_dim)
            key: (batch_size, seq_len_k, embed_dim)
            value: (batch_size, seq_len_v, embed_dim)
            key_padding_mask: Optional mask for padded keys (batch_size, seq_len_k)
            attn_mask: Optional attention mask (seq_len_q, seq_len_k)
            need_weights: If True, returns attention weights
        """
        batch_size, seq_len_q, _ = query.size()
        seq_len_k = key.size(1)
        seq_len_v = value.size(1)

        # Project and reshape
        q = self.q_proj(query) * self.scaling
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Reshape to (batch_size, num_heads, seq_len, head_dim)
        q = (
            q.contiguous()
            .view(batch_size, seq_len_q, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            k.contiguous()
            .view(batch_size, seq_len_k, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            v.contiguous()
            .view(batch_size, seq_len_v, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Ensure masks are boolean
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.bool()
        if attn_mask is not None:
            if attn_mask.dtype == torch.float32:
                attn_mask = attn_mask.bool()

        # Compute attention
        attn_output, attn_weights = self.attention(q, k, v, attn_mask, key_padding_mask)

        # Reshape and project output
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len_q, self.embed_dim)
        )
        output = self.out_proj(attn_output)

        return (output, attn_weights) if need_weights else (output, None)
