import math
from typing import Optional

import torch
import torch.nn as nn
from decoder import TransformerDecoderLayer
from encoder import TransformerEncoderLayer
from layers import PositionalEncoding


class Transformer(nn.Module):
    """
    A complete Transformer model for sequence-to-sequence tasks.
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        norm_first: bool = False,
        pad_idx: int = 0,
        max_seq_length: int = 5000,
    ):
        super().__init__()

        self.pad_idx = pad_idx
        self.embed_dim = embed_dim

        # Source and target embeddings
        self.src_embedding = nn.Embedding(
            src_vocab_size, embed_dim, padding_idx=pad_idx
        )
        self.tgt_embedding = nn.Embedding(
            tgt_vocab_size, embed_dim, padding_idx=pad_idx
        )

        # Scale embeddings by sqrt(embed_dim)
        self.embed_scale = math.sqrt(embed_dim)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(embed_dim, max_seq_length, dropout)

        # Encoder and Decoder layers
        self.encoder_layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    embed_dim, num_heads, ff_dim, dropout, activation, norm_first
                )
                for _ in range(num_encoder_layers)
            ]
        )

        self.decoder_layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    embed_dim, num_heads, ff_dim, dropout, activation, norm_first
                )
                for _ in range(num_decoder_layers)
            ]
        )

        # Final layer normalization
        self.encoder_norm = nn.LayerNorm(embed_dim)
        self.decoder_norm = nn.LayerNorm(embed_dim)

        # Output projection
        self.output_projection = nn.Linear(embed_dim, tgt_vocab_size)

        # Initialize parameters
        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters using Xavier uniform initialization"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def create_padding_mask(self, src: torch.Tensor) -> torch.Tensor:
        """Create mask for padding tokens"""
        return (src == self.pad_idx).to(src.device)

    def create_causal_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        """Create causal mask for decoder self-attention"""
        sz = tgt.size(1)
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask.to(tgt.device)

    def encode(
        self, src: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode source sequence.

        Args:
            src: Source tokens (batch_size, src_len)
            src_key_padding_mask: Mask for padding tokens (batch_size, src_len)
        """
        # Embedding and positional encoding
        x = self.src_embedding(src) * self.embed_scale
        x = self.pos_encoding(x)

        # Create padding mask if not provided
        if src_key_padding_mask is None:
            src_key_padding_mask = self.create_padding_mask(src)

        # Apply encoder layers
        for layer in self.encoder_layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)

        return self.encoder_norm(x)

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Decode target sequence.

        Args:
            tgt: Target tokens (batch_size, tgt_len)
            memory: Encoder output (batch_size, src_len, embed_dim)
            tgt_mask: Causal mask for decoder self-attention
            memory_key_padding_mask: Mask for padding in encoder output
            tgt_key_padding_mask: Mask for padding in target tokens
        """
        # Embedding and positional encoding
        x = self.tgt_embedding(tgt) * self.embed_scale
        x = self.pos_encoding(x)

        # Create masks if not provided
        if tgt_mask is None:
            tgt_mask = self.create_causal_mask(tgt)
        if tgt_key_padding_mask is None:
            tgt_key_padding_mask = self.create_padding_mask(tgt)

        # Apply decoder layers
        for layer in self.decoder_layers:
            x = layer(
                x,
                memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
            )

        x = self.decoder_norm(x)
        x = self.output_projection(x)

        return x

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the transformer.

        Args:
            src: Source tokens (batch_size, src_len)
            tgt: Target tokens (batch_size, tgt_len)
            src_key_padding_mask: Mask for padding in source tokens
            tgt_key_padding_mask: Mask for padding in target tokens
            memory_key_padding_mask: Mask for padding in encoder output
            tgt_mask: Causal mask for decoder self-attention
        """
        memory = self.encode(src, src_key_padding_mask)

        if memory_key_padding_mask is None:
            memory_key_padding_mask = src_key_padding_mask

        output = self.decode(
            tgt, memory, tgt_mask, memory_key_padding_mask, tgt_key_padding_mask
        )

        return output


def generate_square_subsequent_mask(sz: int, device: torch.device) -> torch.Tensor:
    """Generate causal mask for decoder"""
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask.to(device)


# Example usage:
def create_transformer(
    src_vocab_size: int = 10000,
    tgt_vocab_size: int = 10000,
    embed_dim: int = 512,
    num_heads: int = 8,
    num_encoder_layers: int = 6,
    num_decoder_layers: int = 6,
    ff_dim: int = 2048,
    dropout: float = 0.1,
    activation: str = "relu",
    norm_first: bool = False,
    pad_idx: int = 0,
    max_seq_length: int = 5000,
) -> Transformer:
    """Create a Transformer model with default parameters"""
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        ff_dim=ff_dim,
        dropout=dropout,
        activation=activation,
        norm_first=norm_first,
        pad_idx=pad_idx,
        max_seq_length=max_seq_length,
    )

    # Initialize parameters
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model


# Example of creating and using the model:
if __name__ == "__main__":
    # Create model
    model = create_transformer()

    # Generate dummy data
    batch_size = 32
    src_len = 20
    tgt_len = 15
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    src = torch.randint(1, 10000, (batch_size, src_len)).to(device)
    tgt = torch.randint(1, 10000, (batch_size, tgt_len)).to(device)

    # Create masks
    src_key_padding_mask = src == 0
    tgt_key_padding_mask = tgt == 0
    tgt_mask = generate_square_subsequent_mask(tgt_len, device)

    # Forward pass
    model = model.to(device)
    output = model(
        src,
        tgt,
        src_key_padding_mask=src_key_padding_mask,
        tgt_key_padding_mask=tgt_key_padding_mask,
        tgt_mask=tgt_mask,
    )

    print(f"Output shape: {output.shape}")
    print(f"Device: {output.device}")
