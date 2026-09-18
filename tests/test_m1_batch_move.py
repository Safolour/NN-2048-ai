"""M1: correctness of the batched movement kernel.

Every case compares ``game2048.fast_env.move_batch`` against the frozen M0
``move_without_spawn``.
"""

from __future__ import annotations

import numpy as np
import pytest

from game2048 import Action
from game2048.fast_env import move_batch

from _m1_helpers import batch, board, compare_move, random_boards, SEED

#: Regression vectors from the frozen M0 specification (section 4.1), as rows.
M0_REGRESSION_ROWS = [
    ((1, 1, 1, 1), (2, 2, 0, 0), 8),
    ((1, 1, 2, 0), (2, 2, 0, 0), 4),
    ((2, 2, 2, 0), (3, 2, 0, 0), 8),
    ((1, 1, 1, 0), (2, 1, 0, 0), 4),
    ((1, 0, 1, 1), (2, 1, 0, 0), 4),
    ((1, 1, 2, 2), (2, 3, 0, 0), 12),
    ((3, 3, 3, 3), (4, 4, 0, 0), 32),
]

DIRECTIONS = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]


def _place_row(values, action):
    """Put ``values`` on the row/column the action merges first."""
    if action == Action.DOWN:
        values = tuple(reversed(values))
    if action == Action.UP:
        values = tuple(reversed(values))
    target = np.zeros(16, dtype=np.uint8)
    if action in (Action.LEFT, Action.RIGHT):
        target[0:4] = values
    else:
        target[0:16:4] = values
    return target


@pytest.mark.parametrize("action", DIRECTIONS)
def test_m0_regression_vectors_in_every_direction(action):
    """The six M0 regression vectors must survive the fast path, all directions."""
    for values, expected_row, expected_reward in M0_REGRESSION_ROWS:
        entry = _place_row(values, action)
        result = move_batch(batch([entry]), np.array([int(action)], dtype=np.uint8))
        # The fast result must always agree with M0 ...
        compare_move(result, None, entry, int(action), 0)
        # ... and on the LEFT row it must produce the documented row and reward.
        if action == Action.LEFT:
            assert result.afterstates[0][0:4].tolist() == list(expected_row)
            assert int(result.rewards[0]) == expected_reward


def test_one_one_two_zero_must_not_collapse_to_three():
    """``[1,1,2,0] -> [2,2,0,0]``; ``[3,0,0,0]`` would be the classic bug."""
    entry = board((1, 1, 2, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0))
    result = move_batch(batch([entry]), np.array([int(Action.LEFT)], dtype=np.uint8))
    assert result.afterstates[0][0:4].tolist() == [2, 2, 0, 0]
    assert int(result.rewards[0]) == 4


def test_every_tile_merges_at_most_once():
    """``[1,1,1,1] -> [2,2,0,0]`` with reward 8, never ``[3,1,0,0]``."""
    entry = board((1, 1, 1, 1), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0))
    result = move_batch(batch([entry]), np.array([int(Action.LEFT)], dtype=np.uint8))
    assert result.afterstates[0][0:4].tolist() == [2, 2, 0, 0]
    assert int(result.rewards[0]) == 8


def test_mixed_action_batch_is_grouped_correctly():
    """A batch mixing all four actions must not leak results across groups."""
    boards = []
    actions = []
    for action in DIRECTIONS:
        for values, _row, _reward in M0_REGRESSION_ROWS:
            boards.append(_place_row(values, action))
            actions.append(int(action))
    entries = batch(boards)
    action_array = np.array(actions, dtype=np.uint8)
    result = move_batch(entries, action_array)
    for index in range(entries.shape[0]):
        compare_move(result, None, entries[index], actions[index], index)


def test_movement_differential_on_random_boards():
    """10,000 random boards x 4 actions against M0: zero mismatches allowed.

    seed = 20260918, sample count = 10,000, exponent range = 0..17.
    """
    entries = random_boards(10_000, 17, SEED)
    for action in DIRECTIONS:
        actions = np.full(entries.shape[0], int(action), dtype=np.uint8)
        result = move_batch(entries, actions)
        for index in range(entries.shape[0]):
            compare_move(result, None, entries[index], int(action), index)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_float_actions_are_rejected(dtype):
    entries = batch([board((1, 1, 1, 1), (0,) * 4, (0,) * 4, (0,) * 4)])
    with pytest.raises(ValueError):
        move_batch(entries, np.array([1.9], dtype=dtype))


def test_bool_actions_are_rejected():
    """``True`` must not be silently coerced to action 1."""
    entries = batch([board((1, 1, 1, 1), (0,) * 4, (0,) * 4, (0,) * 4)])
    with pytest.raises(ValueError):
        move_batch(entries, np.array([True], dtype=bool))


def test_string_actions_are_rejected():
    entries = batch([board((1, 1, 1, 1), (0,) * 4, (0,) * 4, (0,) * 4)])
    with pytest.raises(ValueError):
        move_batch(entries, np.array(["1"]))


def test_out_of_range_actions_are_rejected():
    entries = batch([board((1, 1, 1, 1), (0,) * 4, (0,) * 4, (0,) * 4)])
    for value in (-1, 4, 99):
        with pytest.raises(ValueError):
            move_batch(entries, np.array([value], dtype=np.int64))


def test_wrong_action_shape_is_rejected():
    entries = batch(
        [
            board((1, 1, 1, 1), (0,) * 4, (0,) * 4, (0,) * 4),
            board((2, 2, 2, 0), (0,) * 4, (0,) * 4, (0,) * 4),
        ]
    )
    with pytest.raises(ValueError):
        move_batch(entries, np.array([[0, 1]], dtype=np.int64))
    with pytest.raises(ValueError):
        move_batch(entries, np.array([0], dtype=np.int64))


def test_wrong_board_shape_is_rejected():
    with pytest.raises(ValueError):
        move_batch(np.zeros((2, 15), dtype=np.uint8), np.zeros(2, dtype=np.int64))
    with pytest.raises(ValueError):
        move_batch(np.zeros((2, 4, 4), dtype=np.uint8), np.zeros(2, dtype=np.int64))
    with pytest.raises(ValueError):
        move_batch(np.zeros(16, dtype=np.uint8), np.zeros(1, dtype=np.int64))


def test_float_boards_are_rejected():
    with pytest.raises(ValueError):
        move_batch(np.zeros((2, 16), dtype=np.float32), np.zeros(2, dtype=np.int64))


def test_out_of_range_exponents_are_rejected():
    with pytest.raises(ValueError):
        move_batch(np.full((1, 16), 300, dtype=np.int64), np.zeros(1, dtype=np.int64))
    with pytest.raises(ValueError):
        move_batch(np.full((1, 16), -1, dtype=np.int64), np.zeros(1, dtype=np.int64))


def test_move_batch_does_not_modify_its_inputs():
    entries = random_boards(64, 17, SEED)
    action_array = np.full(entries.shape[0], int(Action.LEFT), dtype=np.uint8)
    entries_before = entries.copy()
    actions_before = action_array.copy()
    result = move_batch(entries, action_array)
    assert np.array_equal(entries, entries_before)
    assert np.array_equal(action_array, actions_before)
    # The afterstate must be a fresh array, never a view of the input.
    assert not np.shares_memory(result.afterstates, entries)
    result.afterstates[0, 0] = 123
    assert np.array_equal(entries, entries_before)
