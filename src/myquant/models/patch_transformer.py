from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class PatchTransformerOutput:
    logits: torch.Tensor
    probability_distribution: torch.Tensor
    regime_state: torch.Tensor
    task_logits: dict[str, torch.Tensor]
    task_probability_distributions: dict[str, torch.Tensor]
    primary_task_name: str

    @property
    def probabilities(self) -> torch.Tensor:
        return self.probability_distribution

    def probability_distribution_for(self, task_name: str) -> torch.Tensor:
        return self.task_probability_distributions[task_name]

    def logits_for(self, task_name: str) -> torch.Tensor:
        return self.task_logits[task_name]


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
        num_classes: int | None = None,
        patch_length: int = 5,
        patch_stride: int = 5,
        d_model: int = 128,
        n_heads: int = 4,
        num_layers: int = 3,
        regime_dim: int = 32,
        dropout: float = 0.1,
        ff_multiplier: int = 4,
        *,
        task_output_dims: dict[str, int] | None = None,
        primary_task_name: str = "primary",
    ) -> None:
        super().__init__()
        if sequence_length < patch_length:
            raise ValueError("sequence_length must be at least patch_length")
        if regime_dim < 1:
            raise ValueError("regime_dim must be at least 1")

        if task_output_dims is None:
            if num_classes is None:
                raise ValueError("num_classes must be provided when task_output_dims is omitted")
            task_output_dims = {primary_task_name: num_classes}
        if primary_task_name not in task_output_dims:
            raise ValueError("primary_task_name must exist in task_output_dims")
        if any(output_dim < 2 for output_dim in task_output_dims.values()):
            raise ValueError("each task output dimension must be at least 2")

        self.primary_task_name = primary_task_name

        self.patch_embed = nn.Conv1d(
            in_channels=feature_dim,
            out_channels=d_model,
            kernel_size=patch_length,
            stride=patch_stride,
        )
        num_patches = ((sequence_length - patch_length) // patch_stride) + 1
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.regime_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.positional_encoding = PositionalEncoding(d_model=d_model, max_len=num_patches + 2)
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
        self.regime_projection = nn.Sequential(
            nn.Linear(d_model, regime_dim),
            nn.Tanh(),
        )
        self.heads = nn.ModuleDict(
            {
                task_name: nn.Sequential(
                    nn.Linear(d_model + regime_dim, d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model, output_dim),
                )
                for task_name, output_dim in task_output_dims.items()
            }
        )

        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        nn.init.normal_(self.regime_token, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> PatchTransformerOutput:
        x = x.transpose(1, 2)
        tokens = self.patch_embed(x).transpose(1, 2)
        cls = self.cls_token.expand(tokens.size(0), -1, -1)
        regime = self.regime_token.expand(tokens.size(0), -1, -1)
        tokens = torch.cat([cls, regime, tokens], dim=1)
        tokens = self.positional_encoding(tokens)
        encoded = self.encoder(tokens)
        cls_state = self.norm(encoded[:, 0])
        regime_context = self.norm(encoded[:, 1])
        regime_state = self.regime_projection(regime_context)
        shared_state = torch.cat([cls_state, regime_state], dim=1)
        task_logits = {
            task_name: head(shared_state)
            for task_name, head in self.heads.items()
        }
        task_probability_distributions = {
            task_name: torch.softmax(task_logits[task_name], dim=-1)
            for task_name in task_logits
        }
        logits = task_logits[self.primary_task_name]
        probability_distribution = task_probability_distributions[self.primary_task_name]
        return PatchTransformerOutput(
            logits=logits,
            probability_distribution=probability_distribution,
            regime_state=regime_state,
            task_logits=task_logits,
            task_probability_distributions=task_probability_distributions,
            primary_task_name=self.primary_task_name,
        )

    @torch.no_grad()
    def predict_distribution(self, x: torch.Tensor) -> torch.Tensor:
        return self(x).probability_distribution
