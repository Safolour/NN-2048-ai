"""M3 Teacher sample schema and game-level split helpers."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .m3_search import LEAF_EVALUATOR, SEARCH_VALUE_RAW_LEAF, ExpectimaxTeacher
from .reference_env import ACTIONS, move_without_spawn

SPLIT_SEED = 20260919

@dataclass(frozen=True)
class TeacherSample:
    state: np.ndarray
    teacher_value: np.ndarray
    reward: np.ndarray
    afterstate: np.ndarray
    legal_mask: np.ndarray
    game_id: int
    step_index: int
    current_score: int
    max_tile_exp: int
    teacher_version: str
    checkpoint_sha256: str
    decision_depth: int
    data_source: str
    value_semantics: str = SEARCH_VALUE_RAW_LEAF
    search_mode: str = "expectimax_formal_state_leaf"
    leaf_evaluator: str = LEAF_EVALUATOR

def build_teacher_sample(
    search: ExpectimaxTeacher,
    state: np.ndarray,
    *,
    game_id: int,
    step_index: int,
    current_score: int,
    teacher_version: str,
    checkpoint_sha256: str,
    data_source: str,
) -> TeacherSample:
    board = np.asarray(state)
    if board.shape != (16,) or board.dtype != np.uint8:
        raise ValueError("state must be uint8 with shape (16,)")
    values = search.action_values(board)
    rewards = np.zeros(4, dtype=np.int64)
    afterstates = np.zeros((4, 16), dtype=np.uint8)
    legal = np.zeros(4, dtype=np.bool_)
    for action in ACTIONS:
        moved = move_without_spawn(board, action)
        if moved.moved:
            index = int(action)
            legal[index] = True
            rewards[index] = int(moved.reward)
            afterstates[index] = moved.afterstate
    values = values.astype(np.float64, copy=False)
    values[~legal] = np.nan
    return TeacherSample(
        state=board.copy(),
        teacher_value=values.copy(),
        reward=rewards,
        afterstate=afterstates,
        legal_mask=legal,
        game_id=int(game_id),
        step_index=int(step_index),
        current_score=int(current_score),
        max_tile_exp=int(board.max(initial=0)),
        teacher_version=str(teacher_version),
        checkpoint_sha256=str(checkpoint_sha256),
        decision_depth=search.decision_depth,
        data_source=str(data_source),
    )

def split_game_ids(game_ids, *, seed: int = SPLIT_SEED) -> dict[str, set[int]]:
    unique = np.array(sorted({int(x) for x in game_ids}), dtype=np.int64)
    if unique.size == 0:
        raise ValueError("at least one game_id is required")
    rng = np.random.default_rng(seed)
    shuffled = unique[rng.permutation(unique.size)]
    train_end = int(np.floor(unique.size * 0.8))
    val_end = train_end + int(np.floor(unique.size * 0.1))
    return {
        "train": set(int(x) for x in shuffled[:train_end]),
        "validation": set(int(x) for x in shuffled[train_end:val_end]),
        "test": set(int(x) for x in shuffled[val_end:]),
    }

def split_labels(game_ids, *, seed: int = SPLIT_SEED) -> dict[int, str]:
    groups = split_game_ids(game_ids, seed=seed)
    labels: dict[int, str] = {}
    for name, ids in groups.items():
        for game_id in ids:
            if game_id in labels:
                raise RuntimeError("game_id leaked across splits")
            labels[game_id] = name
    return labels

__all__ = ["SPLIT_SEED", "TeacherSample", "build_teacher_sample", "split_game_ids", "split_labels"]
