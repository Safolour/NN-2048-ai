"""M3 decision-depth Expectimax Teacher search."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from typing import Protocol

import numpy as np

from .reference_env import ACTIONS, enumerate_spawns, move_without_spawn

SEARCH_VALUE_RAW_LEAF = "SEARCH_VALUE_RAW_LEAF"
LEAF_EVALUATOR = "tuple_afterstate_greedy_1ply_adapter"

class FormalStateEvaluator(Protocol):
    def formal_state_leaf(self, state: np.ndarray) -> float: ...

@dataclass
class SearchStats:
    root_calls: int = 0
    player_nodes: int = 0
    chance_nodes: int = 0
    leaf_calls: int = 0
    move_calls: int = 0
    chance_outcomes: int = 0
    cache_lookups: int = 0
    cache_hits: int = 0
    hash_seconds: float = 0.0
    move_generation_seconds: float = 0.0
    chance_expansion_seconds: float = 0.0
    tuple_evaluator_seconds: float = 0.0
    total_seconds: float = 0.0

    def to_dict(self) -> dict:
        result = asdict(self)
        result["cache_hit_rate"] = (
            self.cache_hits / self.cache_lookups if self.cache_lookups else 0.0
        )
        tracked = (
            self.hash_seconds + self.move_generation_seconds
            + self.chance_expansion_seconds + self.tuple_evaluator_seconds
        )
        result["recursive_python_orchestration_seconds"] = max(0.0, self.total_seconds - tracked)
        return result

class ExpectimaxTeacher:
    """Formal-state Expectimax; chance nodes do not consume decision depth."""

    def __init__(
        self,
        evaluator: FormalStateEvaluator,
        *,
        decision_depth: int = 3,
        use_cache: bool = True,
    ) -> None:
        if decision_depth < 1:
            raise ValueError("decision_depth must be >= 1")
        self.evaluator = evaluator
        self.decision_depth = int(decision_depth)
        self.use_cache = bool(use_cache)
        self.stats = SearchStats()
        self._cache: dict[tuple[str, int, bytes], float] = {}

    def reset(self) -> None:
        self.stats = SearchStats()
        self._cache.clear()

    def _key(self, node_type: str, remaining: int, board: np.ndarray) -> tuple[str, int, bytes]:
        start = time.perf_counter()
        key = (node_type, int(remaining), np.asarray(board, dtype=np.uint8).tobytes())
        self.stats.hash_seconds += time.perf_counter() - start
        return key

    def _cached(self, key: tuple[str, int, bytes]) -> float | None:
        if not self.use_cache:
            return None
        self.stats.cache_lookups += 1
        value = self._cache.get(key)
        if value is not None:
            self.stats.cache_hits += 1
        return value

    def _store(self, key: tuple[str, int, bytes], value: float) -> float:
        if self.use_cache:
            self._cache[key] = value
        return value

    def _leaf(self, state: np.ndarray) -> float:
        self.stats.leaf_calls += 1
        start = time.perf_counter()
        value = float(self.evaluator.formal_state_leaf(state))
        self.stats.tuple_evaluator_seconds += time.perf_counter() - start
        return value

    def _player(self, state: np.ndarray, remaining: int) -> float:
        self.stats.player_nodes += 1
        key = self._key("player", remaining, state)
        hit = self._cached(key)
        if hit is not None:
            return hit

        best = -np.inf
        legal = False
        for action in ACTIONS:
            start = time.perf_counter()
            moved = move_without_spawn(state, action)
            self.stats.move_generation_seconds += time.perf_counter() - start
            self.stats.move_calls += 1
            if not moved.moved:
                continue
            legal = True
            value = float(moved.reward) + self._chance(moved.afterstate, remaining - 1)
            if value > best:
                best = value
        if not legal:
            best = 0.0
        return self._store(key, float(best))

    def _chance(self, afterstate: np.ndarray, remaining: int) -> float:
        self.stats.chance_nodes += 1
        key = self._key("chance", remaining, afterstate)
        hit = self._cached(key)
        if hit is not None:
            return hit

        start = time.perf_counter()
        outcomes = enumerate_spawns(afterstate)
        self.stats.chance_expansion_seconds += time.perf_counter() - start
        self.stats.chance_outcomes += len(outcomes)
        if not outcomes:
            value = self._leaf(afterstate) if remaining == 0 else self._player(afterstate, remaining)
            return self._store(key, value)

        total = 0.0
        if remaining == 0 and hasattr(self.evaluator, "formal_state_leaf_batch"):
            states = np.stack([outcome.state for outcome in outcomes])
            self.stats.leaf_calls += len(outcomes)
            leaf_start = time.perf_counter()
            continuations = np.asarray(
                self.evaluator.formal_state_leaf_batch(states), dtype=np.float64
            )
            self.stats.tuple_evaluator_seconds += time.perf_counter() - leaf_start
            probabilities = np.fromiter(
                (float(outcome.probability) for outcome in outcomes),
                dtype=np.float64, count=len(outcomes),
            )
            total = float(probabilities @ continuations)
        else:
            for outcome in outcomes:
                continuation = (
                    self._leaf(outcome.state)
                    if remaining == 0
                    else self._player(outcome.state, remaining)
                )
                total += float(outcome.probability) * continuation
        return self._store(key, total)

    def action_values(self, state: np.ndarray) -> np.ndarray:
        board = np.asarray(state)
        if board.shape != (16,) or board.dtype != np.uint8:
            raise ValueError("state must be uint8 with shape (16,)")
        before = board.tobytes()
        started = time.perf_counter()
        self.stats.root_calls += 1
        values = np.full(4, np.nan, dtype=np.float64)
        for action in ACTIONS:
            move_start = time.perf_counter()
            moved = move_without_spawn(board, action)
            self.stats.move_generation_seconds += time.perf_counter() - move_start
            self.stats.move_calls += 1
            if moved.moved:
                values[int(action)] = float(moved.reward) + self._chance(
                    moved.afterstate, self.decision_depth - 1
                )
        self.stats.total_seconds += time.perf_counter() - started
        if board.tobytes() != before:
            raise RuntimeError("Teacher search mutated its input board")
        return values

    def best_action(self, state: np.ndarray) -> int:
        values = self.action_values(state)
        if np.isnan(values).all():
            raise ValueError("cannot choose an action on a terminal board")
        return int(np.nanargmax(values))

__all__ = [
    "ExpectimaxTeacher", "LEAF_EVALUATOR", "SEARCH_VALUE_RAW_LEAF", "SearchStats",
]
