"""M0 regression tests for high exponents.

These tests exist specifically to stop anyone from clamping the environment's
internal exponent to the network's overflow bucket (which is a later, network
level concern).  ``exp 20 + exp 20 -> exp 21`` and ``exp 21 + exp 21 -> exp 22``
must work, and the merge reward must never be squeezed into ``uint8``.
"""

import numpy as np
import pytest

from game2048 import (
    Action,
    MAX_EXPONENT,
    Reference2048Env,
    is_terminal,
    legal_mask,
    move_without_spawn,
)

EMPTY_ROW = (0, 0, 0, 0)


def B(*rows):
    """Build a ``(16,)`` ``uint8`` board from four rows of exponents."""
    return np.array(rows, dtype=np.uint8).reshape(16)


E16 = 16  # 65536
E17 = 17  # 131072
E20 = 20  # 1048576
E21 = 21  # 2097152
E22 = 22  # 4194304


# --------------------------------------------------------------------------- #
# The two exponents required by the M0 acceptance list
# --------------------------------------------------------------------------- #


def test_65536_plus_65536_gives_131072():
    board = B((E16, E16, 0, 0), EMPTY_ROW, EMPTY_ROW, EMPTY_ROW)
    result = move_without_spawn(board, Action.LEFT)

    assert result.afterstate[0] == E17
    assert result.afterstate.tolist() == B((E17, 0, 0, 0), EMPTY_ROW, EMPTY_ROW, EMPTY_ROW).tolist()
    assert result.reward == 131072
    assert result.moved is True


def test_exp20_plus_exp20_gives_exp21():
    board = B((E20, E20, 0, 0), EMPTY_ROW, EMPTY_ROW, EMPTY_ROW)
    result = move_without_spawn(board, Action.LEFT)

    assert result.afterstate[0] == E21
    assert result.reward == 2 ** 21


def test_exp21_plus_exp21_gives_exp22():
    board = B((E21, E21, 0, 0), EMPTY_ROW, EMPTY_ROW, EMPTY_ROW)
    result = move_without_spawn(board, Action.LEFT)

    assert result.afterstate[0] == E22
    assert result.reward == 2 ** 22


def test_high_tiles_are_never_clamped_to_the_overflow_bucket():
    """exp 20/21 must survive verbatim; nothing may be rewritten to 21."""
    board = B((E20, 0, E21, 0), EMPTY_ROW, EMPTY_ROW, EMPTY_ROW)
    result = move_without_spawn(board, Action.RIGHT)

    assert result.afterstate.tolist() == B((0, 0, E20, E21), EMPTY_ROW, EMPTY_ROW, EMPTY_ROW).tolist()
    assert result.afterstate[2] == E20
    assert result.afterstate[3] == E21


def test_exponents_are_stored_verbatim():
    """A tile keeps its exact exponent while sliding."""
    for exponent in (1, 15, 16, 17, 20, 21, 22, 30, 100, 254, 255):
        board = np.zeros(16, dtype=np.uint8)
        board[4] = exponent  # row 1, column 0
        result = move_without_spawn(board, Action.RIGHT)

        assert result.afterstate[7] == exponent
        assert result.moved is True
        assert result.reward == 0


# --------------------------------------------------------------------------- #
# High tiles in every direction
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "action,expected_index",
    [
        (Action.LEFT, 0),
        (Action.RIGHT, 3),
        (Action.UP, 0),
        (Action.DOWN, 12),
    ],
)
def test_high_tile_merge_in_every_direction(action, expected_index):
    board = np.zeros(16, dtype=np.uint8)
    if action == Action.LEFT:
        board[0] = E16
        board[1] = E16
    elif action == Action.RIGHT:
        board[2] = E16
        board[3] = E16
    elif action == Action.UP:
        board[0] = E16
        board[4] = E16
    else:
        board[8] = E16
        board[12] = E16

    result = move_without_spawn(board, action)
    assert result.afterstate[expected_index] == E17
    assert result.reward == 131072
    assert result.moved is True


def test_four_high_tiles_merge_into_two():
    board = B((E20, E20, E20, E20), EMPTY_ROW, EMPTY_ROW, EMPTY_ROW)
    result = move_without_spawn(board, Action.LEFT)

    assert result.afterstate.tolist() == B((E21, E21, 0, 0), EMPTY_ROW, EMPTY_ROW, EMPTY_ROW).tolist()
    assert result.reward == 2 ** 21 + 2 ** 21


def test_mixed_high_and_low_reward_sum():
    board = B((1, 1, E20, E20), EMPTY_ROW, EMPTY_ROW, EMPTY_ROW)
    result = move_without_spawn(board, Action.LEFT)

    assert result.afterstate.tolist() == B((2, E21, 0, 0), EMPTY_ROW, EMPTY_ROW, EMPTY_ROW).tolist()
    assert result.reward == 4 + 2 ** 21


# --------------------------------------------------------------------------- #
# Reward type and score accumulation must not use uint8
# --------------------------------------------------------------------------- #


def test_reward_is_a_python_int_beyond_uint8():
    board = B((E20, E20, 0, 0), EMPTY_ROW, EMPTY_ROW, EMPTY_ROW)
    reward = move_without_spawn(board, Action.LEFT).reward

    assert isinstance(reward, int)
    assert not isinstance(reward, np.uint8)
    assert reward == 2097152
    assert reward > 255
    assert reward % 256 != reward  # would fail if truncated to uint8


def test_environment_score_accumulates_high_rewards():
    """White-box setup: a high-tile board is installed directly so that the
    environment must accumulate a reward far beyond the uint8 range."""
    env = Reference2048Env(seed=0)
    board = B((E20, E20, E20, E20), EMPTY_ROW, EMPTY_ROW, EMPTY_ROW)
    env._board = board.copy()
    env._score = 0

    result = env.step(Action.LEFT)
    assert result.reward == 2 ** 21 + 2 ** 21
    assert env.score == 2 ** 21 + 2 ** 21
    assert isinstance(env.score, int)
    assert env.score > 255


# --------------------------------------------------------------------------- #
# Overflow instead of silent uint8 wraparound
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("action", list(Action))
def test_merging_at_max_exponent_raises_overflow_error(action):
    board = np.zeros(16, dtype=np.uint8)
    if action == Action.LEFT:
        board[0] = MAX_EXPONENT
        board[1] = MAX_EXPONENT
    elif action == Action.RIGHT:
        board[2] = MAX_EXPONENT
        board[3] = MAX_EXPONENT
    elif action == Action.UP:
        board[0] = MAX_EXPONENT
        board[4] = MAX_EXPONENT
    else:
        board[8] = MAX_EXPONENT
        board[12] = MAX_EXPONENT

    with pytest.raises(OverflowError):
        move_without_spawn(board, action)


def test_exponent_254_still_merges_to_255():
    board = np.zeros(16, dtype=np.uint8)
    board[0] = 254
    board[1] = 254
    result = move_without_spawn(board, Action.LEFT)

    assert result.afterstate[0] == 255
    assert result.reward == 2 ** 255


def test_max_exponent_tile_may_still_slide():
    """A 2**255 tile that cannot merge must slide normally."""
    board = np.zeros(16, dtype=np.uint8)
    board[3] = MAX_EXPONENT
    result = move_without_spawn(board, Action.LEFT)

    assert result.afterstate[0] == MAX_EXPONENT
    assert result.reward == 0
    assert result.moved is True


def test_no_silent_uint8_wraparound_on_terminal_check():
    """A board holding 2**255 tiles must not silently wrap when probed."""
    board = np.array([MAX_EXPONENT] * 16, dtype=np.uint8)
    with pytest.raises(OverflowError):
        legal_mask(board)
    with pytest.raises(OverflowError):
        is_terminal(board)
