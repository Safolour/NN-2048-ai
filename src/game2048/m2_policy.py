"""M2 legal masking and tie-safe greedy action selection."""
from __future__ import annotations

import torch

TIE_ATOL = 1e-6


def _validate(q_values: torch.Tensor, legal_mask: torch.Tensor) -> None:
    if not isinstance(q_values, torch.Tensor) or not isinstance(legal_mask, torch.Tensor):
        raise TypeError("q_values and legal_mask must be torch.Tensor")
    if q_values.ndim != 2 or q_values.shape[1] != 4 or q_values.shape[0] < 1:
        raise ValueError(f"q_values must have shape (B, 4), got {tuple(q_values.shape)}")
    if legal_mask.shape != q_values.shape:
        raise ValueError(
            f"legal_mask must have shape {tuple(q_values.shape)}, got {tuple(legal_mask.shape)}"
        )
    if legal_mask.dtype is not torch.bool:
        raise TypeError(f"legal_mask dtype must be bool, got {legal_mask.dtype}")
    if legal_mask.device != q_values.device:
        raise ValueError("legal_mask and q_values must be on the same device")
    if bool((~legal_mask.any(dim=1)).any().item()):
        raise ValueError("each row must contain at least one legal action")


def apply_legal_mask(q_values: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    _validate(q_values, legal_mask)
    return q_values.masked_fill(~legal_mask, float("-inf"))


def select_greedy_actions(
    q_values: torch.Tensor,
    legal_mask: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
    tie_atol: float = TIE_ATOL,
) -> torch.Tensor:
    """Select a legal greedy action with uniform random tie-breaking."""
    if tie_atol < 0:
        raise ValueError("tie_atol must be non-negative")
    masked = apply_legal_mask(q_values, legal_mask)
    best = masked.max(dim=1, keepdim=True).values
    tied = legal_mask & ((masked - best).abs() <= tie_atol)

    # Random scores turn argmax into a uniform choice among tied candidates.
    random_scores = torch.rand(
        q_values.shape,
        device=q_values.device,
        dtype=torch.float32,
        generator=generator,
    )
    random_scores = random_scores.masked_fill(~tied, -1.0)
    return random_scores.argmax(dim=1)


__all__ = ["TIE_ATOL", "apply_legal_mask", "select_greedy_actions"]
