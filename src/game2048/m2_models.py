"""M2 neural-network baselines for 2048.

The models accept only torch.uint8 exponent boards with shape (B, 16).
Exponent values 21..255 are mapped to the NN overflow token 21; the
environment representation itself is never modified.
"""
from __future__ import annotations

import torch
from torch import nn

CELL_COUNT = 16
TILE_VOCAB_SIZE = 22
OVERFLOW_TOKEN = 21


def _validate_boards(boards: torch.Tensor) -> None:
    if not isinstance(boards, torch.Tensor):
        raise TypeError(f"boards must be torch.Tensor, got {type(boards).__name__}")
    if boards.ndim != 2 or boards.shape[1] != CELL_COUNT or boards.shape[0] < 1:
        raise ValueError(f"boards must have shape (B, 16) with B >= 1, got {tuple(boards.shape)}")
    if boards.dtype is not torch.uint8:
        raise TypeError(f"boards dtype must be torch.uint8, got {boards.dtype}")


def tokenize_boards(boards: torch.Tensor) -> torch.Tensor:
    """Validate exponent boards and return int64 embedding indices in 0..21."""
    _validate_boards(boards)
    return boards.clamp_max(OVERFLOW_TOKEN).to(dtype=torch.long)


class Transformer2048(nn.Module):
    """~5M parameter Transformer baseline with independent Q / V / A heads."""
    def __init__(
        self,
        hidden: int = 256,
        blocks: int = 6,
        attention_heads: int = 8,
        ffn: int = 1024,
    ) -> None:
        super().__init__()
        self.tile_embedding = nn.Embedding(TILE_VOCAB_SIZE, hidden)
        self.position_embedding = nn.Parameter(torch.empty(1, CELL_COUNT, hidden))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=attention_heads,
            dim_feedforward=ffn,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.backbone = nn.TransformerEncoder(layer, num_layers=blocks)
        self.final_norm = nn.LayerNorm(hidden)
        self.q_head = nn.Linear(hidden, 4)
        self.value_head = nn.Linear(hidden, 1)
        self.afterstate_head = nn.Linear(hidden, 1)
        nn.init.normal_(self.tile_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

    def encode(self, boards: torch.Tensor) -> torch.Tensor:
        tokens = tokenize_boards(boards)
        x = self.tile_embedding(tokens) + self.position_embedding
        x = self.backbone(x)
        return self.final_norm(x.mean(dim=1))

    def forward(self, boards: torch.Tensor) -> torch.Tensor:
        """Final-inference path: return Q only, shape (B, 4)."""
        features = self.encode(boards)
        return self.q_head(features)

    def forward_state(self, boards: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encode(boards)
        return self.q_head(features), self.value_head(features).squeeze(-1)

    def forward_afterstate(self, afterstates: torch.Tensor) -> torch.Tensor:
        features = self.encode(afterstates)
        return self.afterstate_head(features).squeeze(-1)


class _ResidualBlock(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.fc1 = nn.Linear(hidden, hidden)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(hidden, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return residual + x


class ResidualMLP2048(nn.Module):
    """Parameter-matched residual MLP baseline with independent Q / V / A heads."""

    def __init__(
        self,
        embedding_dim: int = 32,
        hidden: int = 640,
        blocks: int = 6,
    ) -> None:
        super().__init__()
        self.tile_embedding = nn.Embedding(TILE_VOCAB_SIZE, embedding_dim)
        self.input_projection = nn.Linear(CELL_COUNT * embedding_dim, hidden)
        self.input_activation = nn.GELU()
        self.backbone = nn.ModuleList([_ResidualBlock(hidden) for _ in range(blocks)])
        self.final_norm = nn.LayerNorm(hidden)
        self.q_head = nn.Linear(hidden, 4)
        self.value_head = nn.Linear(hidden, 1)
        self.afterstate_head = nn.Linear(hidden, 1)
        nn.init.normal_(self.tile_embedding.weight, mean=0.0, std=0.02)

    def encode(self, boards: torch.Tensor) -> torch.Tensor:
        tokens = tokenize_boards(boards)
        x = self.tile_embedding(tokens).flatten(1)
        x = self.input_activation(self.input_projection(x))
        for block in self.backbone:
            x = block(x)
        return self.final_norm(x)

    def forward(self, boards: torch.Tensor) -> torch.Tensor:
        """Final-inference path: return Q only, shape (B, 4)."""
        features = self.encode(boards)
        return self.q_head(features)
    def forward_state(self, boards: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encode(boards)
        return self.q_head(features), self.value_head(features).squeeze(-1)

    def forward_afterstate(self, afterstates: torch.Tensor) -> torch.Tensor:
        features = self.encode(afterstates)
        return self.afterstate_head(features).squeeze(-1)


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


__all__ = [
    "CELL_COUNT",
    "OVERFLOW_TOKEN",
    "TILE_VOCAB_SIZE",
    "Transformer2048",
    "ResidualMLP2048",
    "tokenize_boards",
    "trainable_parameter_count",
]
