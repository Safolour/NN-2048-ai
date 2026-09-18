from __future__ import annotations

import numpy as np

from game2048.m3_search import ExpectimaxTeacher, LEAF_EVALUATOR
from game2048.reference_env import ACTIONS, enumerate_spawns, move_without_spawn

class SumLeaf:
    def formal_state_leaf(self, state: np.ndarray) -> float:
        return float(np.asarray(state, dtype=np.float64).sum())

def _manual_depth1(state: np.ndarray) -> np.ndarray:
    result = np.full(4, np.nan, dtype=np.float64)
    evaluator = SumLeaf()
    for action in ACTIONS:
        moved = move_without_spawn(state, action)
        if not moved.moved:
            continue
        expectation = sum(
            outcome.probability * evaluator.formal_state_leaf(outcome.state)
            for outcome in enumerate_spawns(moved.afterstate)
        )
        result[int(action)] = moved.reward + expectation
    return result

def test_depth1_is_player_then_chance_then_formal_state_leaf():
    state = np.array([1, 1, 0, 0] + [0] * 12, dtype=np.uint8)
    search = ExpectimaxTeacher(SumLeaf(), decision_depth=1, use_cache=False)
    actual = search.action_values(state)
    expected = _manual_depth1(state)
    assert np.allclose(actual, expected, equal_nan=True)
    assert search.stats.leaf_calls > 0
    assert LEAF_EVALUATOR == "tuple_afterstate_greedy_1ply_adapter"

def test_terminal_future_value_is_zero():
    terminal = np.array([
        1,2,1,2,
        2,1,2,1,
        1,2,1,2,
        2,1,2,1,
    ], dtype=np.uint8)
    search = ExpectimaxTeacher(SumLeaf(), decision_depth=2)
    values = search.action_values(terminal)
    assert np.isnan(values).all()

def test_illegal_action_never_receives_a_value():
    state = np.array([1, 0, 0, 0] + [0] * 12, dtype=np.uint8)
    values = ExpectimaxTeacher(SumLeaf(), decision_depth=1).action_values(state)
    for action in ACTIONS:
        moved = move_without_spawn(state, action)
        assert bool(np.isnan(values[int(action)])) == (not moved.moved)

def test_cache_on_off_correctness_differential():
    state = np.array([
        1,2,3,4,
        2,3,4,5,
        3,4,5,6,
        4,5,6,0,
    ], dtype=np.uint8)
    uncached = ExpectimaxTeacher(SumLeaf(), decision_depth=2, use_cache=False).action_values(state)
    cached_search = ExpectimaxTeacher(SumLeaf(), decision_depth=2, use_cache=True)
    cached = cached_search.action_values(state)
    assert np.allclose(cached, uncached, rtol=0, atol=1e-12, equal_nan=True)
    assert cached_search.stats.cache_lookups > 0

def test_high_tile_and_no_input_mutation():
    state = np.array([16,16,0,0] + [0] * 12, dtype=np.uint8)
    before = state.copy()
    values = ExpectimaxTeacher(SumLeaf(), decision_depth=1).action_values(state)
    assert np.isfinite(values[~np.isnan(values)]).all()
    assert np.array_equal(state, before)

def test_chance_probabilities_sum_to_one_and_are_90_10_uniform():
    afterstate = np.zeros(16, dtype=np.uint8)
    afterstate[0] = 1
    outcomes = enumerate_spawns(afterstate)
    assert abs(sum(o.probability for o in outcomes) - 1.0) < 1e-12
    tile2 = [o for o in outcomes if o.spawn_exponent == 1]
    tile4 = [o for o in outcomes if o.spawn_exponent == 2]
    assert len(tile2) == len(tile4) == 15
    assert all(abs(o.probability - 0.9 / 15) < 1e-12 for o in tile2)
    assert all(abs(o.probability - 0.1 / 15) < 1e-12 for o in tile4)


class BatchSumLeaf(SumLeaf):
    def formal_state_leaf_batch(self, states: np.ndarray) -> np.ndarray:
        return np.asarray(states, dtype=np.float64).sum(axis=1)

def test_leaf_batching_is_semantics_preserving():
    state = np.array([
        1,2,3,4,
        2,3,4,5,
        3,4,5,6,
        4,5,0,0,
    ], dtype=np.uint8)
    scalar = ExpectimaxTeacher(SumLeaf(), decision_depth=2, use_cache=True).action_values(state)
    batched = ExpectimaxTeacher(BatchSumLeaf(), decision_depth=2, use_cache=True).action_values(state)
    assert np.allclose(batched, scalar, rtol=0, atol=1e-12, equal_nan=True)
