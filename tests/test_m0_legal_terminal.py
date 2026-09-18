"""M0 tests for ``legal_mask`` and ``is_terminal``.

Both are defined *only* through ``move_without_spawn``: an action is legal iff
the move changes the board, and a state is terminal iff no action is legal.
"""

import numpy as np
import pytest

from game2048 import (
    ACTIONS,
    ACTION_COUNT,
    Action,
    is_terminal,
    legal_mask,
    move_without_spawn,
)

EMPTY_ROW = (0, 0, 0, 0)


def B(*rows):
    """Build a ``(16,)`` ``uint8`` board from four rows of exponents."""
    return np.array(rows, dtype=np.uint8).reshape(16)


# A full board with no two adjacent equal tiles: completely stuck.
FULL_NO_MERGE = B(
    (1, 2, 3, 4),
    (5, 6, 7, 8),
    (9, 10, 11, 12),
    (13, 14, 15, 16),
)

# Same shape, but 15/15 sit side by side in the last row (columns 0-2 still
# hold distinct values, and every column is gapless, so only LEFT/RIGHT work).
FULL_WITH_HORIZONTAL_PAIR = B(
    (1, 2, 3, 4),
    (5, 6, 7, 8),
    (9, 10, 11, 12),
    (13, 14, 15, 15),
)

# Full board whose only mergeable pair is vertical: (2, 2) and (3, 2).
FULL_WITH_VERTICAL_PAIR = B(
    (1, 2, 3, 4),
    (5, 6, 7, 8),
    (9, 10, 11, 12),
    (13, 14, 11, 16),
)

# Two empty rows on top, two full distinct rows below.  Every column is already
# packed downwards and merge-free, and every non-empty row is gapless, so the
# only possible action is UP.
ONLY_UP = B(
    (0, 0, 0, 0),
    (0, 0, 0, 0),
    (1, 2, 3, 4),
    (5, 6, 7, 8),
)

# Transpose of ONLY_UP: only LEFT is possible.
ONLY_LEFT = B(
    (0, 0, 1, 2),
    (0, 0, 3, 4),
    (0, 0, 5, 6),
    (0, 0, 7, 8),
)


# --------------------------------------------------------------------------- #
# legal_mask
# --------------------------------------------------------------------------- #


def test_legal_mask_shape_and_dtype():
    mask = legal_mask(FULL_WITH_HORIZONTAL_PAIR)
    assert mask.shape == (4,)
    assert mask.dtype == bool


def test_legal_mask_order_is_up_down_left_right():
    # Exactly one action is legal, and it is UP -> slot 0.
    assert legal_mask(ONLY_UP).tolist() == [True, False, False, False]
    # Exactly one action is legal, and it is LEFT -> slot 2.
    assert legal_mask(ONLY_LEFT).tolist() == [False, False, True, False]


def test_legal_mask_matches_move_without_spawn_on_random_boards():
    rng = np.random.Generator(np.random.PCG64(20240518))
    for _ in range(500):
        board = rng.integers(0, 12, size=16).astype(np.uint8)
        mask = legal_mask(board)
        expected = [move_without_spawn(board, action).moved for action in ACTIONS]
        assert list(mask) == expected
        assert mask.dtype == bool


def test_legal_mask_all_false_on_full_board_without_pairs():
    assert legal_mask(FULL_NO_MERGE).tolist() == [False] * ACTION_COUNT


def test_legal_mask_only_horizontal_on_a_horizontal_pair():
    mask = legal_mask(FULL_WITH_HORIZONTAL_PAIR)
    assert not mask[Action.UP]
    assert not mask[Action.DOWN]
    assert mask[Action.LEFT]
    assert mask[Action.RIGHT]


def test_legal_mask_only_vertical_on_a_vertical_pair():
    mask = legal_mask(FULL_WITH_VERTICAL_PAIR)
    assert mask[Action.UP]
    assert mask[Action.DOWN]
    assert not mask[Action.LEFT]
    assert not mask[Action.RIGHT]


def test_legal_mask_for_two_tiles_side_by_side():
    # Row 0 = 2,2.  The column is already flush upwards, so UP is blocked; the
    # pair can merge left or right, and both tiles can fall down.
    side = B((1, 1, 0, 0), EMPTY_ROW, EMPTY_ROW, EMPTY_ROW)
    assert legal_mask(side).tolist() == [False, True, True, True]


def test_legal_mask_for_two_tiles_stacked():
    # Column 0 = 2,2.  Both rows are already flush left, so LEFT is blocked.
    stacked = B((1, 0, 0, 0), (1, 0, 0, 0), EMPTY_ROW, EMPTY_ROW)
    assert legal_mask(stacked).tolist() == [True, True, False, True]


def test_full_board_is_legal_when_a_merge_exists():
    """A full board is NOT automatically terminal."""
    assert bool(legal_mask(FULL_WITH_HORIZONTAL_PAIR).any())
    assert is_terminal(FULL_WITH_HORIZONTAL_PAIR) is False


def test_legal_mask_does_not_modify_input():
    board = FULL_WITH_HORIZONTAL_PAIR.copy()
    before = board.copy()
    legal_mask(board)
    assert np.array_equal(before, board)


# --------------------------------------------------------------------------- #
# is_terminal
# --------------------------------------------------------------------------- #


def test_terminal_on_full_board_without_equal_neighbours():
    assert is_terminal(FULL_NO_MERGE) is True


def test_not_terminal_on_full_board_with_a_horizontal_pair():
    assert is_terminal(FULL_WITH_HORIZONTAL_PAIR) is False


def test_not_terminal_on_full_board_with_a_vertical_pair():
    assert is_terminal(FULL_WITH_VERTICAL_PAIR) is False


def test_terminal_is_exactly_no_legal_action():
    rng = np.random.Generator(np.random.PCG64(777))
    for _ in range(500):
        board = rng.integers(0, 6, size=16).astype(np.uint8)
        assert is_terminal(board) == (not legal_mask(board).any())


def test_single_tile_is_never_terminal():
    """One tile can always slide somewhere on a 4x4 board."""
    for index in range(16):
        board = np.zeros(16, dtype=np.uint8)
        board[index] = 1
        assert is_terminal(board) is False
        assert bool(legal_mask(board).any())


def test_two_tiles_are_never_terminal():
    for first in range(16):
        for second in range(first + 1, 16):
            board = np.zeros(16, dtype=np.uint8)
            board[first] = 1
            board[second] = 3
            assert is_terminal(board) is False


def test_terminal_when_the_board_is_jammed():
    board = B(
        (1, 2, 1, 2),
        (2, 1, 2, 1),
        (1, 2, 1, 2),
        (2, 1, 2, 1),
    )
    assert is_terminal(board) is True
    assert legal_mask(board).tolist() == [False] * ACTION_COUNT


def test_terminal_on_empty_board():
    """Degenerate but forced by the frozen definition.

    The all-empty board admits no legal action, hence it is terminal.  This
    state is unreachable in real play: ``reset`` always spawns two tiles and an
    action can never reduce the tile count below one.
    """
    board = B(EMPTY_ROW, EMPTY_ROW, EMPTY_ROW, EMPTY_ROW)
    assert is_terminal(board) is True
    assert legal_mask(board).tolist() == [False] * ACTION_COUNT


def test_terminal_does_not_modify_input():
    board = FULL_NO_MERGE.copy()
    before = board.copy()
    is_terminal(board)
    assert np.array_equal(before, board)


@pytest.mark.parametrize("action", list(Action))
def test_terminal_implies_every_action_is_illegal(action):
    assert is_terminal(FULL_NO_MERGE) is True
    assert move_without_spawn(FULL_NO_MERGE, action).moved is False
