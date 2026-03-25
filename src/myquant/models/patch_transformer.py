from __future__ import annotations

import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Classic sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()

        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class PatchTransformerClassifier(nn.Module):
    """
    Compact patch-based transformer classifier for rolling market-feature sequences.

    Input shape:
        [batch, lookback, feature_dim]
    """

    def __init__(
        self,
        feature_dim: int,
        sequence_length: int,
        num_classes: int,
        patch_length: int = 5,
        patch_stride: int = 5,
        d_model: int = 128,
        n_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
        ff_multiplier: int = 4,
    ) -> None:
        super().__init__()
        if sequence_length < patch_length:
            raise ValueError("sequence_length must be at least patch_length")

        self.patch_embed = nn.Conv1d(
            in_channels=feature_dim,
            out_channels=d_model,
            kernel_size=patch_length,
            stride=patch_stride,
        )
        num_patches = ((sequence_length - patch_length) // patch_stride) + 1
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.positional_encoding = PositionalEncoding(d_model=d_model, max_len=num_patches + 1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_multiplier,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        tokens = self.patch_embed(x).transpose(1, 2)
        cls = self.cls_token.expand(tokens.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = self.positional_encoding(tokens)
        encoded = self.encoder(tokens)
        pooled = self.norm(encoded[:, 0])
        return self.head(pooled)
