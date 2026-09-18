"""M2 rollout environment using the scalar C++ movement/legal backend."""
from __future__ import annotations

import copy

import numpy as np

from .fast_env import (
    BatchStepResult,
    Fast2048BatchEnv,
    _as_actions,
    _checked_add_nonnegative_int64,
)
from .m2_fast_backend import legal_mask_batch, move_selected_batch


class M2RolloutBatchEnv(Fast2048BatchEnv):
    """FastEnv-compatible M2 rollout path with a cached current legal mask.

    Frozen M1 remains the NumPy oracle. This subclass changes only the M2 rollout
    implementation: selected movement and all-actions legality are delegated to
    the scalar C++ backend; spawn RNG remains NumPy PCG64.
    """

    def __init__(self, num_envs: int, seed: int | None = None) -> None:
        super().__init__(num_envs, seed=seed)
        self._current_legal = np.zeros((self._num_envs, 4), dtype=bool)

    @property
    def current_legal(self) -> np.ndarray:
        """Read-only live view of the cached legal mask for the current boards."""
        view = self._current_legal.view()
        view.flags.writeable = False
        return view

    def reset(self, seed: int | None = None) -> np.ndarray:
        boards = super().reset(seed=seed)
        self._current_legal[:] = legal_mask_batch(self._boards)
        return boards

    def reset_where(self, mask: np.ndarray) -> None:
        selection = np.asarray(mask)
        if selection.shape != (self._num_envs,):
            raise ValueError(
                f"mask must have shape ({self._num_envs},), got {selection.shape}"
            )
        if selection.dtype != np.bool_:
            raise ValueError(f"mask dtype must be bool, got {selection.dtype}")
        rows = np.flatnonzero(selection)
        if rows.size == 0:
            return
        super().reset_where(selection)
        reset_boards = np.ascontiguousarray(self._boards[rows])
        self._current_legal[rows] = legal_mask_batch(reset_boards)

    def step(self, actions: np.ndarray) -> BatchStepResult:
        action_array = _as_actions(actions, self._num_envs)
        move = move_selected_batch(self._boards, action_array)
        moved = move.moved

        rewards = np.where(moved, move.rewards, np.int64(0)).astype(
            np.int64, copy=False
        )
        new_scores = _checked_add_nonnegative_int64(
            self._scores,
            rewards,
            context="score accumulation",
        )

        spawn_indices = np.full(self._num_envs, -1, dtype=np.int64)
        spawn_exponents = np.zeros(self._num_envs, dtype=np.uint8)
        rows = np.flatnonzero(moved)

        # Do not mutate official board/score until the candidate next state has
        # passed next-legal calculation. RNG is the only mutable component used
        # while constructing the candidate, so snapshot/restore it on failure.
        candidate = move.afterstates.copy()
        rng_state = None
        if rows.size:
            rng_state = copy.deepcopy(self._rng.bit_generator.state)
            try:
                draws = self._rng.random((rows.size, 1, 2))
                scratch = move.afterstates[rows].copy()
                indices, exponents = self._place_tiles(
                    scratch, draws[:, 0], rows
                )
                candidate[rows] = scratch
                spawn_indices[rows] = indices
                spawn_exponents[rows] = exponents
                next_legal = legal_mask_batch(candidate)
            except Exception:
                self._rng.bit_generator.state = rng_state
                raise
        else:
            next_legal = legal_mask_batch(candidate)

        terminated = ~next_legal.any(axis=1)

        # Commit only after every range/legality operation has succeeded.
        self._boards[:] = candidate
        self._scores[:] = new_scores
        self._current_legal[:] = next_legal

        return BatchStepResult(
            states=self._boards.copy(),
            afterstates=move.afterstates,
            rewards=rewards,
            legal=moved.copy(),
            terminated=terminated.copy(),
            spawn_indices=spawn_indices,
            spawn_exponents=spawn_exponents,
        )


__all__ = ["M2RolloutBatchEnv"]
