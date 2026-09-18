"""M1: high-tile support, the ``int64`` reward boundary and the overflow contract.

M0 stores the raw exponent (never clamping to a network overflow bucket) and uses
an arbitrary-precision Python ``int`` for the reward.  M1 accumulates the reward
in ``numpy.int64``, so:
``e = 61`` pays ``2**62`` (representable) and ``e = 62`` needs ``2**63``
(not representable).  ``MAX_SAFE_MERGE_EXPONENT = 62`` therefore means
"an exponent of 62 or above may not be merged".

That is the **single documented divergence** between M1 and M0, and it is always
an explicit ``OverflowError`` -- never a silently wrapped number.

This file pins the *single-merge* side of that boundary.  The **aggregate** side
(several individually representable merges summing past ``INT64_MAX`` within one
row or one board, plus ``step`` atomicity) lives in
``tests/test_m1_reward_overflow.py``; see ``reports/m1/M1_REPORT.md`` §S.
"""

from __future__ import annotations

import numpy as np
import pytest

from game2048 import Action, MAX_EXPONENT, move_without_spawn
from game2048.fast_env import (
    MAX_SAFE_MERGE_EXPONENT,
    Fast2048BatchEnv,
    apply_spawn_batch,
    is_terminal_batch,
    legal_mask_batch,
    move_batch,
    spawn_random_batch,
)

from _m1_helpers import SEED, batch, board, board_str, compare_move, random_boards

DIRECTIONS = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]

E16 = 16
E17 = 17
E20 = 20
E21 = 21
E22 = 22


def _merge_board(action, exponent):
    """Two equal ``exponent`` tiles placed so ``action`` merges them."""
    entry = np.zeros(16, dtype=np.uint8)
    if action == Action.LEFT:
        entry[0] = entry[1] = exponent
    elif action == Action.RIGHT:
        entry[2] = entry[3] = exponent
    elif action == Action.UP:
        entry[0] = entry[4] = exponent
    else:
        entry[8] = entry[12] = exponent
    return entry


# --------------------------------------------------------------------------- #
# The high tiles the project must actually support
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("action", DIRECTIONS)
def test_exp16_plus_exp16_in_every_direction(action):
    entry = _merge_board(action, E16)
    result = move_batch(batch([entry]), np.array([int(action)], dtype=np.uint8))
    reference = move_without_spawn(entry, action)
    assert np.array_equal(result.afterstates[0], reference.afterstate)
    assert int(result.rewards[0]) == 131072
    assert int(result.rewards[0]) == 2 ** 17
    assert bool(result.moved[0])


@pytest.mark.parametrize("action", DIRECTIONS)
def test_exp20_plus_exp20_in_every_direction(action):
    entry = _merge_board(action, E20)
    result = move_batch(batch([entry]), np.array([int(action)], dtype=np.uint8))
    assert np.array_equal(
        result.afterstates[0], move_without_spawn(entry, action).afterstate
    )
    assert int(result.rewards[0]) == 2 ** 21
    assert int(result.rewards[0]) == 2097152


@pytest.mark.parametrize("action", DIRECTIONS)
def test_exp21_plus_exp21_in_every_direction(action):
    entry = _merge_board(action, E21)
    result = move_batch(batch([entry]), np.array([int(action)], dtype=np.uint8))
    assert np.array_equal(
        result.afterstates[0], move_without_spawn(entry, action).afterstate
    )
    assert int(result.rewards[0]) == 2 ** 22
    assert int(result.rewards[0]) == 4194304


@pytest.mark.parametrize("action", DIRECTIONS)
def test_exp22_plus_exp22_in_every_direction(action):
    entry = _merge_board(action, E22)
    result = move_batch(batch([entry]), np.array([int(action)], dtype=np.uint8))
    assert np.array_equal(
        result.afterstates[0], move_without_spawn(entry, action).afterstate
    )
    assert int(result.rewards[0]) == 2 ** 23


def test_high_tiles_are_never_clamped():
    """exp 20/21/22 survive verbatim; nothing is rewritten to an overflow bucket."""
    for exponent in (E20, E21, E22, 30, 40, 52, 100, 254, MAX_EXPONENT):
        entry = np.zeros(16, dtype=np.uint8)
        entry[4] = exponent
        result = move_batch(
            batch([entry]), np.array([int(Action.RIGHT)], dtype=np.uint8)
        )
        assert int(result.afterstates[0][7]) == exponent, (
            f"exponent {exponent} was rewritten to "
            f"{int(result.afterstates[0][7])}"
        )
        assert int(result.rewards[0]) == 0
        assert bool(result.moved[0])


def test_mixed_high_and_low_rewards_sum():
    entry = board((1, 1, E20, E20), (0,) * 4, (0,) * 4, (0,) * 4)
    result = move_batch(batch([entry]), np.array([int(Action.LEFT)], dtype=np.uint8))
    assert np.array_equal(
        result.afterstates[0], move_without_spawn(entry, Action.LEFT).afterstate
    )
    assert int(result.rewards[0]) == 4 + 2 ** 21


def test_four_high_tiles_merge_into_two():
    entry = board((E20, E20, E20, E20), (0,) * 4, (0,) * 4, (0,) * 4)
    result = move_batch(batch([entry]), np.array([int(Action.LEFT)], dtype=np.uint8))
    assert result.afterstates[0][0:4].tolist() == [E21, E21, 0, 0]
    assert int(result.rewards[0]) == 2 ** 21 + 2 ** 21


def test_reward_is_int64_never_uint8():
    entry = _merge_board(Action.LEFT, E20)
    result = move_batch(batch([entry]), np.array([int(Action.LEFT)], dtype=np.uint8))
    assert result.rewards.dtype == np.int64
    assert int(result.rewards[0]) > 255
    assert int(result.rewards[0]) % 256 != int(result.rewards[0])


def test_high_tile_differential_against_m0():
    """2,000 boards, exponents 0..22, all four actions, afterstate + reward.

    seed = 20260918, sample count = 2,000, exponent range = 0..22.
    """
    entries = random_boards(2_000, 22, SEED, empty_probability=0.45)
    for action in DIRECTIONS:
        actions = np.full(entries.shape[0], int(action), dtype=np.uint8)
        result = move_batch(entries, actions)
        for index in range(entries.shape[0]):
            compare_move(result, None, entries[index], int(action), index)


def test_legal_and_terminal_differential_at_high_exponents():
    """Legal mask and terminal at exponents 0..22 against M0.

    seed = 20260918, sample count = 2,000, exponent range = 0..22.
    """
    entries = random_boards(2_000, 22, SEED, empty_probability=0.45)
    from game2048 import is_terminal, legal_mask

    fast_mask = legal_mask_batch(entries)
    fast_terminal = is_terminal_batch(entries)
    for index in range(entries.shape[0]):
        assert np.array_equal(fast_mask[index], legal_mask(entries[index])), (
            f"sample={index} board={board_str(entries[index])}"
        )
        assert bool(fast_terminal[index]) == is_terminal(entries[index])


# --------------------------------------------------------------------------- #
# The int64 reward boundary and the explicit-overflow contract
# --------------------------------------------------------------------------- #


def test_merges_below_the_boundary_are_exact():
    """Every representable merge is exact, up to exponent 61."""
    for exponent in (16, 20, 21, 22, 30, 40, 50, 52, 60, 61):
        entry = _merge_board(Action.LEFT, exponent)
        result = move_batch(
            batch([entry]), np.array([int(Action.LEFT)], dtype=np.uint8)
        )
        assert int(result.afterstates[0][0]) == exponent + 1
        assert int(result.rewards[0]) == 2 ** (exponent + 1)


@pytest.mark.parametrize("exponent", [62, 63, 100, 200, 254, 255])
def test_merges_above_the_boundary_raise_overflow(exponent):
    """No silent ``int64`` wraparound: the merge raises ``OverflowError``."""
    entry = _merge_board(Action.LEFT, exponent)
    with pytest.raises(OverflowError):
        move_batch(batch([entry]), np.array([int(Action.LEFT)], dtype=np.uint8))


def test_boundary_constant_is_documented():
    assert MAX_SAFE_MERGE_EXPONENT == 62
    # The largest reward M1 can emit is 2 ** 63 - 1 worth of merges at exponent 61.
    assert 2 ** (MAX_SAFE_MERGE_EXPONENT + 1) > np.iinfo(np.int64).max
    assert 2 ** MAX_SAFE_MERGE_EXPONENT <= np.iinfo(np.int64).max


def test_overflow_is_raised_for_every_direction():
    for action in DIRECTIONS:
        entry = _merge_board(action, 254)
        with pytest.raises(OverflowError):
            move_batch(batch([entry]), np.array([int(action)], dtype=np.uint8))


def test_m0_still_supports_the_boards_m1_rejects():
    """The divergence is one-sided: M0 returns a Python int where M1 raises."""
    entry = _merge_board(Action.LEFT, 254)
    reference = move_without_spawn(entry, Action.LEFT)
    assert reference.afterstate[0] == 255
    assert reference.reward == 2 ** 255
    assert isinstance(reference.reward, int)
    with pytest.raises(OverflowError):
        move_batch(batch([entry]), np.array([int(Action.LEFT)], dtype=np.uint8))


def test_non_merging_high_tiles_never_overflow():
    """A high tile that only slides must not raise: no reward is involved."""
    for exponent in (62, 100, 255):
        entry = np.zeros(16, dtype=np.uint8)
        entry[0] = exponent
        result = move_batch(
            batch([entry]), np.array([int(Action.RIGHT)], dtype=np.uint8)
        )
        assert int(result.afterstates[0][3]) == exponent
        assert int(result.rewards[0]) == 0


def test_environment_score_accumulates_high_rewards():
    """A high-tile board drives ``score`` far beyond the ``uint8`` range."""
    entry = board((E20, E20, E20, E20), (0,) * 4, (0,) * 4, (0,) * 4)
    env = Fast2048BatchEnv(1, seed=SEED)
    env._boards[0] = entry
    result = env.step(np.array([int(Action.LEFT)], dtype=np.uint8))
    assert int(result.rewards[0]) == 2 ** 21 + 2 ** 21
    assert int(env.scores[0]) == 2 ** 21 + 2 ** 21
    assert env.scores.dtype == np.int64


def test_high_tile_boards_survive_a_rollout():
    """Roll a high-tile game forward; every step must stay within the boundary."""
    entry = board(
        (E20, E20, 1, 1), (E21, E21, 2, 2), (E16, E16, 3, 0), (0, 0, 0, 4)
    )
    env = Fast2048BatchEnv(1, seed=SEED)
    env._boards[0] = entry
    rng = np.random.default_rng(SEED)
    for _ in range(40):
        action = int(rng.integers(0, 4))
        result = env.step(np.array([action], dtype=np.uint8))
        reference = move_without_spawn(entry, Action(action))
        assert np.array_equal(result.afterstates[0], reference.afterstate)
        assert int(result.rewards[0]) == reference.reward
        entry = env.copy_boards()[0]
        if bool(result.terminated[0]):
            env.reset_where(np.array([True]))
            entry = env.copy_boards()[0]


def test_high_exponents_spawn_and_enumerate_correctly():
    """Spawn enumeration must not clamp or corrupt high exponents."""
    from game2048 import enumerate_spawns
    from game2048.fast_env import enumerate_spawns_batch

    entries = batch(
        [
            board((0, E20, 0, E21), (0, 0, 0, 0), (0, 0, 0, 0), (E22, 0, 0, 0)),
        ]
    )
    enumeration = enumerate_spawns_batch(entries)
    expected = enumerate_spawns(entries[0])
    assert enumeration.offsets[1] == len(expected)
    for position, outcome in enumerate(expected):
        assert np.array_equal(enumeration.states[position], outcome.state)

    result = spawn_random_batch(entries, np.random.default_rng(SEED))
    assert result.states.shape == (1, 16)
    cell = int(result.spawn_indices[0])
    assert int(result.states[0, cell]) in (1, 2)
    for index, value in enumerate(entries[0]):
        if index != cell:
            assert int(result.states[0, index]) == int(value)


def test_apply_spawn_batch_preserves_high_exponents():
    entry = batch(
        [
            board((0, E20, 0, E21), (0, 0, 0, 0), (0, 0, 0, 0), (E22, 0, 0, 0)),
        ]
    )
    states = apply_spawn_batch(
        entry, np.array([0], dtype=np.int64), np.array([1], dtype=np.uint8)
    )
    assert int(states[0, 1]) == E20
    assert int(states[0, 3]) == E21
    assert int(states[0, 12]) == E22
    assert int(states[0, 0]) == 1
