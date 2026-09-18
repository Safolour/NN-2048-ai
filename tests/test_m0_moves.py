"""M0 regression vectors for the pure movement core ``move_without_spawn``.

Every expected value in this file is a hand-computed 2048 regression vector.
Boards are written as four rows of **exponents** (0 = empty, 1 = 2, 2 = 4,
3 = 8, ...), which is exactly the frozen internal representation.
"""

import numpy as np
import pytest

from game2048 import Action, move_without_spawn


def B(*rows):
    """Build a ``(16,)`` ``uint8`` board from four rows of exponents."""
    return np.array(rows, dtype=np.uint8).reshape(16)


EMPTY_ROW = (0, 0, 0, 0)


def afterstate_of(board, action):
    return move_without_spawn(board, action).afterstate


def reward_of(board, action):
    return move_without_spawn(board, action).reward


# --------------------------------------------------------------------------- #
# LEFT
# --------------------------------------------------------------------------- #


def test_left_four_twos_gives_two_fours():
    """[2,2,2,2] LEFT -> [4,4,0,0], reward 8 (two separate merges)."""
    board = B((1, 1, 1, 1), EMPTY_ROW, EMPTY_ROW, EMPTY_ROW)
    result = move_without_spawn(board, Action.LEFT)

    assert result.afterstate.tolist() == B((2, 2, 0, 0), EMPTY_ROW, EMPTY_ROW, EMPTY_ROW).tolist()
    assert result.reward == 8
    assert result.moved is True


def test_left_two_two_four_does_not_chain():
    """[2,2,4,0] LEFT -> [4,4,0,0], reward 4; the new 4 must NOT merge to 8."""
    board = B((1, 1, 2, 0), EMPTY_ROW, EMPTY_ROW, EMPTY_ROW)
    result = move_without_spawn(board, Action.LEFT)

    assert result.afterstate.tolist() == B((2, 2, 0, 0), EMPTY_ROW, EMPTY_ROW, EMPTY_ROW).tolist()
    assert result.reward == 4
    # The forbidden, chain-merging answer:
    assert result.afterstate.tolist() != B((3, 0, 0, 0), EMPTY_ROW, EMPTY_ROW, EMPTY_ROW).tolist()


def test_left_four_four_four():
    """[4,4,4,0] LEFT -> [8,4,0,0], reward 8."""
    board = B((2, 2, 2, 0), EMPTY_ROW, EMPTY_ROW, EMPTY_ROW)
    result = move_without_spawn(board, Action.LEFT)

    assert result.afterstate.tolist() == B((3, 2, 0, 0), EMPTY_ROW, EMPTY_ROW, EMPTY_ROW).tolist()
    assert result.reward == 8


def test_left_three_twos():
    """[2,2,2,0] LEFT -> [4,2,0,0], reward 4."""
    board = B((1, 1, 1, 0), EMPTY_ROW, EMPTY_ROW, EMPTY_ROW)
    result = move_without_spawn(board, Action.LEFT)

    assert result.afterstate.tolist() == B((2, 1, 0, 0), EMPTY_ROW, EMPTY_ROW, EMPTY_ROW).tolist()
    assert result.reward == 4


def test_left_gap_then_pair():
    """[2,0,2,2] LEFT -> [4,2,0,0], reward 4."""
    board = B((1, 0, 1, 1), EMPTY_ROW, EMPTY_ROW, EMPTY_ROW)
    result = move_without_spawn(board, Action.LEFT)

    assert result.afterstate.tolist() == B((2, 1, 0, 0), EMPTY_ROW, EMPTY_ROW, EMPTY_ROW).tolist()
    assert result.reward == 4


def test_left_two_independent_pairs():
    """[2,2,4,4] LEFT -> [4,8,0,0], reward 4 + 8 = 12."""
    board = B((1, 1, 2, 2), EMPTY_ROW, EMPTY_ROW, EMPTY_ROW)
    result = move_without_spawn(board, Action.LEFT)

    assert result.afterstate.tolist() == B((2, 3, 0, 0), EMPTY_ROW, EMPTY_ROW, EMPTY_ROW).tolist()
    assert result.reward == 12


def test_left_pure_slide_without_merge():
    """[0,0,0,2] LEFT -> [2,0,0,0], reward 0, but moved is True."""
    board = B((0, 0, 0, 1), EMPTY_ROW, EMPTY_ROW, EMPTY_ROW)
    result = move_without_spawn(board, Action.LEFT)

    assert result.afterstate.tolist() == B((1, 0, 0, 0), EMPTY_ROW, EMPTY_ROW, EMPTY_ROW).tolist()
    assert result.reward == 0
    assert result.moved is True


def test_left_illegal_when_already_packed():
    """[4,2,0,0] LEFT is illegal: nothing changes, reward 0."""
    board = B((2, 1, 0, 0), EMPTY_ROW, EMPTY_ROW, EMPTY_ROW)
    result = move_without_spawn(board, Action.LEFT)

    assert result.afterstate.tolist() == board.tolist()
    assert result.reward == 0
    assert result.moved is False


# --------------------------------------------------------------------------- #
# RIGHT
# --------------------------------------------------------------------------- #


def test_right_four_eights():
    """[8,8,8,8] RIGHT -> [0,0,16,16], reward 32."""
    board = B((3, 3, 3, 3), EMPTY_ROW, EMPTY_ROW, EMPTY_ROW)
    result = move_without_spawn(board, Action.RIGHT)

    assert result.afterstate.tolist() == B((0, 0, 4, 4), EMPTY_ROW, EMPTY_ROW, EMPTY_ROW).tolist()
    assert result.reward == 32


def test_right_three_twos_at_the_end():
    """[0,2,2,2] RIGHT -> [0,0,2,4], reward 4 (rightmost pair merges)."""
    board = B((0, 1, 1, 1), EMPTY_ROW, EMPTY_ROW, EMPTY_ROW)
    result = move_without_spawn(board, Action.RIGHT)

    assert result.afterstate.tolist() == B((0, 0, 1, 2), EMPTY_ROW, EMPTY_ROW, EMPTY_ROW).tolist()
    assert result.reward == 4


def test_right_gap_between_pair():
    """[2,2,0,2] RIGHT -> [0,0,2,4], reward 4."""
    board = B((1, 1, 0, 1), EMPTY_ROW, EMPTY_ROW, EMPTY_ROW)
    result = move_without_spawn(board, Action.RIGHT)

    assert result.afterstate.tolist() == B((0, 0, 1, 2), EMPTY_ROW, EMPTY_ROW, EMPTY_ROW).tolist()
    assert result.reward == 4


def test_right_pure_slide_without_merge():
    """[2,0,0,0] RIGHT -> [0,0,0,2], reward 0, moved True."""
    board = B((1, 0, 0, 0), EMPTY_ROW, EMPTY_ROW, EMPTY_ROW)
    result = move_without_spawn(board, Action.RIGHT)

    assert result.afterstate.tolist() == B((0, 0, 0, 1), EMPTY_ROW, EMPTY_ROW, EMPTY_ROW).tolist()
    assert result.reward == 0
    assert result.moved is True


# --------------------------------------------------------------------------- #
# UP
# --------------------------------------------------------------------------- #


def test_up_single_merge():
    board = B((1, 0, 0, 0), (1, 0, 0, 0), EMPTY_ROW, EMPTY_ROW)
    result = move_without_spawn(board, Action.UP)

    assert result.afterstate.tolist() == B((2, 0, 0, 0), EMPTY_ROW, EMPTY_ROW, EMPTY_ROW).tolist()
    assert result.reward == 4
    assert result.moved is True


def test_up_four_columns_of_pairs():
    board = B((1, 1, 1, 1), (1, 1, 1, 1), EMPTY_ROW, EMPTY_ROW)
    result = move_without_spawn(board, Action.UP)

    # Each column holds two mergeable 2s, so each column collapses to one 4
    # in the top row.
    expected = B((2, 2, 2, 2), EMPTY_ROW, EMPTY_ROW, EMPTY_ROW)
    assert result.afterstate.tolist() == expected.tolist()
    # Four columns, one merge each: 4 * (2 + 2 -> 4) = 4 * 4 = 16.
    assert result.reward == 16


def test_up_mixed_columns():
    board = B((3, 0, 1, 0), (3, 0, 1, 0), (1, 0, 0, 0), EMPTY_ROW)
    result = move_without_spawn(board, Action.UP)

    expected = B((4, 0, 2, 0), (1, 0, 0, 0), EMPTY_ROW, EMPTY_ROW)
    assert result.afterstate.tolist() == expected.tolist()
    # Column 0: 8 + 8 -> 16 (reward 16).  Column 2: 2 + 2 -> 4 (reward 4).
    assert result.reward == 20


def test_up_illegal_when_already_packed():
    board = B((1, 2, 0, 0), (3, 0, 0, 0), (4, 0, 0, 0), EMPTY_ROW)
    result = move_without_spawn(board, Action.UP)

    assert result.afterstate.tolist() == board.tolist()
    assert result.moved is False
    assert result.reward == 0


# --------------------------------------------------------------------------- #
# DOWN
# --------------------------------------------------------------------------- #


def test_down_single_merge():
    board = B((1, 0, 0, 0), (1, 0, 0, 0), EMPTY_ROW, EMPTY_ROW)
    result = move_without_spawn(board, Action.DOWN)

    expected = B(EMPTY_ROW, EMPTY_ROW, EMPTY_ROW, (2, 0, 0, 0))
    assert result.afterstate.tolist() == expected.tolist()
    assert result.reward == 4


def test_down_four_columns_of_pairs():
    board = B(EMPTY_ROW, EMPTY_ROW, (1, 1, 1, 1), (1, 1, 1, 1))
    result = move_without_spawn(board, Action.DOWN)

    # Each column collapses to a single 4 at the bottom.
    expected = B(EMPTY_ROW, EMPTY_ROW, EMPTY_ROW, (2, 2, 2, 2))
    assert result.afterstate.tolist() == expected.tolist()
    assert result.reward == 16


def test_down_mixed_columns():
    board = B(EMPTY_ROW, (1, 0, 0, 0), (3, 0, 1, 0), (3, 0, 1, 0))
    result = move_without_spawn(board, Action.DOWN)

    # Column 0 (top to bottom): 0, 2, 8, 8  ->  8+8 = 16 at the bottom, 2 above.
    # Column 2 (top to bottom): 0, 0, 2, 2  ->  2+2 = 4 at the bottom.
    expected = B(EMPTY_ROW, EMPTY_ROW, (1, 0, 0, 0), (4, 0, 2, 0))
    assert result.afterstate.tolist() == expected.tolist()
    assert result.reward == 20


def test_down_illegal_when_already_packed():
    board = B(EMPTY_ROW, (0, 0, 0, 1), (0, 0, 0, 2), (0, 0, 0, 3))
    result = move_without_spawn(board, Action.DOWN)

    assert result.afterstate.tolist() == board.tolist()
    assert result.moved is False
    assert result.reward == 0


# --------------------------------------------------------------------------- #
# Multi-row boards and reward summation
# --------------------------------------------------------------------------- #


def test_left_full_board_multi_row_reward_sum():
    board = B((1, 1, 2, 0), (2, 2, 2, 0), EMPTY_ROW, (3, 3, 3, 3))
    result = move_without_spawn(board, Action.LEFT)

    expected = B((2, 2, 0, 0), (3, 2, 0, 0), EMPTY_ROW, (4, 4, 0, 0))
    assert result.afterstate.tolist() == expected.tolist()
    # row 0: 4, row 1: 8, row 3: 16 + 16 = 32  ->  44
    assert result.reward == 44


def test_rows_do_not_interact_under_left_right():
    """Vertical neighbours must never merge under a horizontal move."""
    board = B((1, 0, 0, 0), (1, 0, 0, 0), EMPTY_ROW, EMPTY_ROW)
    result = move_without_spawn(board, Action.LEFT)

    # Both rows are already flush left, so the move is illegal.
    assert result.moved is False
    assert result.afterstate.tolist() == board.tolist()


def test_columns_do_not_interact_under_up_down():
    board = B((1, 1, 0, 0), EMPTY_ROW, EMPTY_ROW, EMPTY_ROW)
    result = move_without_spawn(board, Action.UP)

    assert result.moved is False
    assert result.afterstate.tolist() == board.tolist()


# --------------------------------------------------------------------------- #
# Purity / determinism / API details
# --------------------------------------------------------------------------- #


def test_move_does_not_modify_input():
    board = B((1, 1, 2, 0), (0, 2, 0, 2), (3, 3, 3, 3), (0, 0, 0, 1))
    before = board.copy()

    move_without_spawn(board, Action.LEFT)
    move_without_spawn(board, Action.RIGHT)
    move_without_spawn(board, Action.UP)
    move_without_spawn(board, Action.DOWN)

    assert np.array_equal(before, board)


def test_move_is_deterministic():
    board = B((1, 2, 1, 2), (2, 1, 2, 1), (1, 1, 2, 2), (0, 3, 0, 3))
    first = move_without_spawn(board, Action.DOWN)
    for _ in range(20):
        again = move_without_spawn(board, Action.DOWN)
        assert again.reward == first.reward
        assert again.moved == first.moved
        assert np.array_equal(again.afterstate, first.afterstate)


def test_afterstate_is_uint8_and_owned():
    board = B((1, 1, 0, 0), EMPTY_ROW, EMPTY_ROW, EMPTY_ROW)
    result = move_without_spawn(board, Action.LEFT)

    assert result.afterstate.shape == (16,)
    assert result.afterstate.dtype == np.uint8
    # Writing into the result must not touch the caller's board.
    result.afterstate[0] = 9
    assert board[0] == 1


def test_accepts_plain_int_actions():
    board = B((1, 1, 0, 0), EMPTY_ROW, EMPTY_ROW, EMPTY_ROW)
    assert move_without_spawn(board, 2) == move_without_spawn(board, Action.LEFT)


def test_action_numbering_is_frozen():
    assert Action.UP == 0
    assert Action.DOWN == 1
    assert Action.LEFT == 2
    assert Action.RIGHT == 3
    assert [int(a) for a in Action] == [0, 1, 2, 3]


def test_rejects_out_of_range_action():
    board = B((1, 1, 0, 0), EMPTY_ROW, EMPTY_ROW, EMPTY_ROW)
    for bad in (4, -1, 99):
        with pytest.raises(ValueError):
            move_without_spawn(board, bad)


def test_rejects_wrong_board_shape():
    with pytest.raises(ValueError):
        move_without_spawn(np.zeros(4, dtype=np.uint8), Action.LEFT)
    with pytest.raises(ValueError):
        move_without_spawn(np.zeros((4, 4), dtype=np.uint8), Action.LEFT)


def test_rejects_non_integer_board():
    with pytest.raises(ValueError):
        move_without_spawn(np.zeros(16, dtype=np.float32), Action.LEFT)


def test_all_empty_board_is_immobile():
    board = B(EMPTY_ROW, EMPTY_ROW, EMPTY_ROW, EMPTY_ROW)
    for action in Action:
        result = move_without_spawn(board, action)
        assert result.moved is False
        assert result.reward == 0
        assert result.afterstate.tolist() == board.tolist()
