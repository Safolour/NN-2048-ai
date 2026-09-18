"""Batched Torch D4 helpers for M2.

Permutation tables are derived from the frozen M0 symmetry oracle at import
time; the training hot path uses Torch indexing only.
"""
from __future__ import annotations

import numpy as np
import torch

from .symmetry import TRANSFORM_COUNT, transform_action, transform_board


def _build_board_permutations() -> torch.Tensor:
    canonical = np.arange(16, dtype=np.uint8)
    rows = [transform_board(canonical, tid).astype(np.int64) for tid in range(TRANSFORM_COUNT)]
    return torch.tensor(np.stack(rows), dtype=torch.long)


def _build_action_permutations() -> torch.Tensor:
    rows = [
        [int(transform_action(action, tid)) for action in range(4)]
        for tid in range(TRANSFORM_COUNT)
    ]
    return torch.tensor(rows, dtype=torch.long)


BOARD_PERMUTATIONS = _build_board_permutations()
ACTION_PERMUTATIONS = _build_action_permutations()


def _validate_transform_ids(transform_ids: torch.Tensor, batch: int, device: torch.device) -> None:
    if not isinstance(transform_ids, torch.Tensor):
        raise TypeError("transform_ids must be torch.Tensor")
    if transform_ids.shape != (batch,):
        raise ValueError(f"transform_ids must have shape ({batch},), got {tuple(transform_ids.shape)}")
    if transform_ids.device != device:
        raise ValueError("transform_ids must be on the same device as the data")
    if transform_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("transform_ids must have int32 or int64 dtype")
    if bool(((transform_ids < 0) | (transform_ids >= TRANSFORM_COUNT)).any().item()):
        raise ValueError("transform_ids values must lie in 0..7")
def transform_board_batch(boards: torch.Tensor, transform_ids: torch.Tensor) -> torch.Tensor:
    if not isinstance(boards, torch.Tensor):
        raise TypeError("boards must be torch.Tensor")
    if boards.ndim != 2 or boards.shape[1] != 16 or boards.shape[0] < 1:
        raise ValueError(f"boards must have shape (B, 16), got {tuple(boards.shape)}")
    _validate_transform_ids(transform_ids, boards.shape[0], boards.device)
    table = BOARD_PERMUTATIONS.to(device=boards.device)
    indices = table[transform_ids.to(torch.long)]
    return torch.gather(boards, 1, indices)


def transform_action_batch(actions: torch.Tensor, transform_ids: torch.Tensor) -> torch.Tensor:
    if not isinstance(actions, torch.Tensor):
        raise TypeError("actions must be torch.Tensor")
    if actions.ndim != 1 or actions.shape[0] < 1:
        raise ValueError(f"actions must have shape (B,), got {tuple(actions.shape)}")
    if actions.dtype not in (torch.int32, torch.int64):
        raise TypeError("actions must have int32 or int64 dtype")
    _validate_transform_ids(transform_ids, actions.shape[0], actions.device)
    if bool(((actions < 0) | (actions > 3)).any().item()):
        raise ValueError("actions values must lie in 0..3")
    table = ACTION_PERMUTATIONS.to(device=actions.device)
    return table[transform_ids.to(torch.long), actions.to(torch.long)].to(dtype=actions.dtype)


def transform_q_values(q_values: torch.Tensor, transform_ids: torch.Tensor) -> torch.Tensor:
    """Map Q(s, a) into transformed-board action coordinates."""
    if not isinstance(q_values, torch.Tensor):
        raise TypeError("q_values must be torch.Tensor")
    if q_values.ndim != 2 or q_values.shape[1] != 4 or q_values.shape[0] < 1:
        raise ValueError(f"q_values must have shape (B, 4), got {tuple(q_values.shape)}")
    _validate_transform_ids(transform_ids, q_values.shape[0], q_values.device)
    mapping = ACTION_PERMUTATIONS.to(device=q_values.device)[transform_ids.to(torch.long)]
    out = torch.empty_like(q_values)
    return out.scatter(1, mapping, q_values)


def inverse_transform_q_values(q_values: torch.Tensor, transform_ids: torch.Tensor) -> torch.Tensor:
    """Map transformed-board Q values back to original action coordinates."""
    if not isinstance(q_values, torch.Tensor):
        raise TypeError("q_values must be torch.Tensor")
    if q_values.ndim != 2 or q_values.shape[1] != 4 or q_values.shape[0] < 1:
        raise ValueError(f"q_values must have shape (B, 4), got {tuple(q_values.shape)}")
    _validate_transform_ids(transform_ids, q_values.shape[0], q_values.device)
    mapping = ACTION_PERMUTATIONS.to(device=q_values.device)[transform_ids.to(torch.long)]
    return torch.gather(q_values, 1, mapping)


__all__ = [
    "ACTION_PERMUTATIONS",
    "BOARD_PERMUTATIONS",
    "transform_action_batch",
    "transform_board_batch",
    "transform_q_values",
    "inverse_transform_q_values",
]
