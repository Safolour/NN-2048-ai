"""M1: differential tests of the fast environment against the frozen M0 oracle.

Conventions for every test in this file
---------------------------------------
* seed = :data:`_m1_helpers.SEED` (20260918) unless stated otherwise;
* the sample count and exponent range are stated in each docstring;
* board / reward / moved / legal / terminal comparisons require **exact**
  equality -- no tolerance is ever widened;
* a failure reports the sample index, the board, the action and both results.
"""

from __future__ import annotations

import numpy as np
import pytest

from game2048 import Action, inverse_transform_id, transform_action, transform_board
from game2048.fast_env import (
    is_terminal_batch,
    legal_mask_batch,
    move_batch,
)

from _m1_helpers import (
    SEED,
    batch,
    board,
    board_str,
    compare_move,
    random_boards,
    reference_legal_mask,
    reference_terminal,
)

DIRECTIONS = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]

#: M0 regression vectors reused through the fast path.
REGRESSION_VECTORS = [
    ((1, 1, 1, 1), (0,) * 4, (0,) * 4, (0,) * 4),
    ((1, 1, 2, 0), (0,) * 4, (0,) * 4, (0,) * 4),
    ((2, 2, 2, 0), (0,) * 4, (0,) * 4, (0,) * 4),
    ((1, 1, 1, 0), (0,) * 4, (0,) * 4, (0,) * 4),
    ((1, 0, 1, 1), (0,) * 4, (0,) * 4, (0,) * 4),
    ((1, 1, 2, 2), (0,) * 4, (0,) * 4, (0,) * 4),
    ((3, 3, 3, 3), (0,) * 4, (0,) * 4, (0,) * 4),
    ((1, 0, 0, 1), (0,) * 4, (0,) * 4, (0,) * 4),
    ((1, 1, 2, 2), (0,) * 4, (1, 1, 2, 2), (0,) * 4),
]

#: The same vectors rotated so that every direction has a merge available.
TRANSPOSED_VECTORS = [tuple(zip(*vector)) for vector in REGRESSION_VECTORS]


@pytest.mark.parametrize("action", DIRECTIONS)
def test_m0_regression_vectors_through_fast_env(action):
    """Every M0 regression vector must reproduce exactly, in all four directions."""
    entries = batch([board(*vector) for vector in REGRESSION_VECTORS])
    actions = np.full(entries.shape[0], int(action), dtype=np.uint8)
    result = move_batch(entries, actions)
    for index in range(entries.shape[0]):
        compare_move(result, None, entries[index], int(action), index)


@pytest.mark.parametrize("action", DIRECTIONS)
def test_m0_regression_vectors_transposed(action):
    """Column-oriented versions of the same vectors (UP/DOWN get real work)."""
    entries = batch([board(*vector) for vector in TRANSPOSED_VECTORS])
    actions = np.full(entries.shape[0], int(action), dtype=np.uint8)
    result = move_batch(entries, actions)
    for index in range(entries.shape[0]):
        compare_move(result, None, entries[index], int(action), index)


def test_ordinary_differential_10k_boards_all_actions():
    """10,000 boards, exponents 0..17, all four actions: 0 mismatches required.

    seed = 20260918, sample count = 10,000, exponent range = 0..17,
    comparisons = 40,000 board-action pairs.
    """
    entries = random_boards(10_000, 17, SEED)
    for action in DIRECTIONS:
        actions = np.full(entries.shape[0], int(action), dtype=np.uint8)
        result = move_batch(entries, actions)
        for index in range(entries.shape[0]):
            compare_move(result, None, entries[index], int(action), index)


def test_high_tile_differential_2k_boards_all_actions():
    """2,000 boards, exponents 0..22, all four actions: 0 mismatches required.

    seed = 20260918, sample count = 2,000, exponent range = 0..22.
    """
    entries = random_boards(2_000, 22, SEED)
    for action in DIRECTIONS:
        actions = np.full(entries.shape[0], int(action), dtype=np.uint8)
        result = move_batch(entries, actions)
        for index in range(entries.shape[0]):
            compare_move(result, None, entries[index], int(action), index)


def test_legal_mask_differential_10k_boards():
    """``legal_mask_batch`` must equal M0's ``legal_mask`` on 10,000 boards."""
    entries = random_boards(10_000, 17, SEED)
    fast_mask = legal_mask_batch(entries)
    for index in range(entries.shape[0]):
        expected = reference_legal_mask(entries[index])
        assert np.array_equal(fast_mask[index], expected), (
            f"sample={index} board={board_str(entries[index])}\n"
            f"  fast  legal = {fast_mask[index].tolist()}\n"
            f"  ref   legal = {expected.tolist()}"
        )


def test_terminal_differential_10k_boards():
    """``is_terminal_batch`` must equal M0's ``is_terminal`` on 10,000 boards."""
    entries = random_boards(10_000, 17, SEED)
    fast_terminal = is_terminal_batch(entries)
    for index in range(entries.shape[0]):
        expected = reference_terminal(entries[index])
        assert bool(fast_terminal[index]) == expected, (
            f"sample={index} board={board_str(entries[index])}\n"
            f"  fast terminal = {bool(fast_terminal[index])}\n"
            f"  ref  terminal = {expected}"
        )


def test_legal_mask_matches_moved_by_definition():
    """``legal[a] <=> move(board, a).moved`` for every action, on random boards."""
    entries = random_boards(2_000, 17, SEED)
    mask = legal_mask_batch(entries)
    for action in DIRECTIONS:
        actions = np.full(entries.shape[0], int(action), dtype=np.uint8)
        moved = move_batch(entries, actions).moved
        assert np.array_equal(mask[:, int(action)], moved)


def test_terminal_is_no_legal_action_and_not_full_board():
    """A full board with an adjacent equal pair is **not** terminal."""
    full_but_mergeable = board(
        (1, 2, 1, 2), (2, 1, 2, 1), (1, 2, 1, 2), (2, 1, 2, 2)
    )
    assert not bool(is_terminal_batch(batch([full_but_mergeable]))[0])
    assert not reference_terminal(full_but_mergeable)

    checkerboard = board((1, 2, 1, 2), (2, 1, 2, 1), (1, 2, 1, 2), (2, 1, 2, 1))
    assert bool(is_terminal_batch(batch([checkerboard]))[0])
    assert reference_terminal(checkerboard)


def test_empty_board_is_terminal_by_definition():
    """The all-empty board has no legal action, so M0 defines it as terminal."""
    empty = np.zeros(16, dtype=np.uint8)
    assert bool(is_terminal_batch(batch([empty]))[0])
    assert reference_terminal(empty)


@pytest.mark.parametrize("transform", range(8))
def test_d4_equivariance_random_boards(transform):
    """``T(FastMove(s,a)) == FastMove(T(s), T(a))`` for 2,000 x 8 x 4 cases.

    seed = 20260918, sample count = 2,000, exponent range = 0..22.
    """
    entries = random_boards(2_000, 22, SEED)
    transformed = batch([transform_board(entry, transform) for entry in entries])
    for action in DIRECTIONS:
        source_actions = np.full(entries.shape[0], int(action), dtype=np.uint8)
        result = move_batch(entries, source_actions)

        mapped_actions = int(transform_action(action, transform))
        target_actions = np.full(entries.shape[0], mapped_actions, dtype=np.uint8)
        transformed_result = move_batch(transformed, target_actions)

        for index in range(entries.shape[0]):
            mapped = transform_board(result.afterstates[index], transform)
            assert np.array_equal(mapped, transformed_result.afterstates[index]), (
                f"transform={transform} action={action.name} sample={index}\n"
                f"  board = {board_str(entries[index])}\n"
                f"  T(fast)   = {board_str(mapped)}\n"
                f"  fast(T)   = {board_str(transformed_result.afterstates[index])}"
            )
            assert int(result.rewards[index]) == int(transformed_result.rewards[index])
            assert bool(result.moved[index]) == bool(transformed_result.moved[index])


def test_d4_inverse_round_trip_on_fast_afterstates():
    """``inverse(T)(T(s)) == s`` and terminal is invariant under D4."""
    entries = random_boards(500, 22, SEED)
    for transform in range(8):
        transformed = batch([transform_board(entry, transform) for entry in entries])
        restored = batch(
            [
                transform_board(entry, inverse_transform_id(transform))
                for entry in transformed
            ]
        )
        assert np.array_equal(restored, entries)
        assert np.array_equal(
            is_terminal_batch(transformed), is_terminal_batch(entries)
        )


def test_d4_legal_mask_equivariance():
    """The legal mask permutes exactly as ``transform_action`` says it does."""
    entries = random_boards(1_000, 22, SEED)
    source_mask = legal_mask_batch(entries)
    for transform in range(8):
        transformed = batch([transform_board(entry, transform) for entry in entries])
        target_mask = legal_mask_batch(transformed)
        for action in DIRECTIONS:
            mapped = int(transform_action(action, transform))
            assert np.array_equal(
                source_mask[:, int(action)], target_mask[:, mapped]
            )
