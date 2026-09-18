"""M1: ``Fast2048BatchEnv`` -- batch step / reset / reset_where / illegal actions.

The environment-level differential tests never rely on the fast and reference RNG
streams lining up.  Instead they exercise the *deterministic* transition

    state -> action -> reference afterstate -> predetermined spawn outcome

and compare it with ``move_batch`` + ``apply_spawn_batch``.  The random sampler is
checked separately, by distribution.
"""

from __future__ import annotations

import numpy as np
import pytest

from game2048 import (
    Action,
    Reference2048Env,
    enumerate_spawns,
    is_terminal,
    legal_mask,
    move_without_spawn,
    spawn_random,
)
from game2048.fast_env import (
    Fast2048BatchEnv,
    apply_spawn_batch,
    is_terminal_batch,
    move_batch,
)

from _m1_helpers import SEED, batch, board, board_str, random_boards

DIRECTIONS = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]


def _two_boards():
    """Two small boards where LEFT is illegal in row 0 of the first."""
    first = board((1, 0, 0, 0), (2, 2, 2, 0), (0, 0, 0, 0), (4, 0, 0, 0))
    second = board((2, 4, 2, 4), (0, 0, 0, 0), (1, 1, 3, 0), (0, 5, 0, 5))
    return batch([first, second])


# --------------------------------------------------------------------------- #
# Construction and buffers
# --------------------------------------------------------------------------- #


def test_construction_does_not_create_reference_environments():
    env = Fast2048BatchEnv(64, seed=SEED)
    assert env.num_envs == 64
    assert env.boards.shape == (64, 16)
    assert env.boards.dtype == np.uint8
    assert env.boards.flags.c_contiguous
    assert env.scores.shape == (64,)
    assert env.scores.dtype == np.int64


def test_construction_validates_num_envs():
    for bad in (0, -1):
        with pytest.raises(ValueError):
            Fast2048BatchEnv(bad)
    with pytest.raises(TypeError):
        Fast2048BatchEnv(True)
    with pytest.raises(TypeError):
        Fast2048BatchEnv(4.0)


def test_boards_and_scores_are_live_but_write_protected():
    env = Fast2048BatchEnv(8, seed=SEED)
    env.reset()
    with pytest.raises(ValueError):
        env.boards[0, 0] = 5
    with pytest.raises(ValueError):
        env.scores[0] = 5


def test_copy_boards_is_independent():
    env = Fast2048BatchEnv(8, seed=SEED)
    env.reset()
    copy = env.copy_boards()
    assert copy.flags.writeable
    copy[0, 0] = 99
    assert env.boards[0, 0] != 99
    assert not np.shares_memory(copy, env.boards)


# --------------------------------------------------------------------------- #
# reset / reset_where
# --------------------------------------------------------------------------- #


def test_reset_is_reproducible_for_a_fixed_seed():
    first = Fast2048BatchEnv(32, seed=1).reset(seed=SEED)
    second = Fast2048BatchEnv(32, seed=999).reset(seed=SEED)
    assert np.array_equal(first, second)


def test_reset_gives_exactly_two_spawned_tiles():
    """Every game starts with exactly two non-empty tiles, both exponent 1 or 2."""
    boards = Fast2048BatchEnv(256, seed=SEED).reset()
    non_empty = (boards != 0).sum(axis=1)
    assert np.all(non_empty == 2), f"tile counts {np.unique(non_empty).tolist()}"
    assert np.all(boards <= 2), f"exponents {np.unique(boards).tolist()}"


def test_reset_zeroes_the_scores():
    """``reset`` clears a score that a real ``step`` actually accumulated.

    The score must not be faked by writing ``env._scores`` directly: the point is
    to prove that the normal step path produced a non-zero score first.  So every
    game is given the same mergeable board

        [1, 1, 0, 0] / [0, 0, 0, 0] / [0, 0, 0, 0] / [0, 0, 0, 0]

    Because the public ``env.boards`` is a read-only live view (assigning through
    it raises), the test writes this fixture straight into the environment's
    internal ``_boards`` buffer in order to build a deterministic internal state.
    ``LEFT`` then merges the two exponent-1 tiles into one exponent-2 tile and pays
    exactly ``2 ** 2 == 4``.  A random initial board could not be used here: two
    freshly spawned tiles often do not merge at all, so the pre-reset score would
    be unreliable.
    """
    count = 16
    mergeable = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint8)

    env = Fast2048BatchEnv(count, seed=SEED)
    env.reset(seed=SEED)
    assert np.all(env.scores == 0), "reset must start every game at score 0"

    # ``env.boards`` is a read-only live view (assignment through it raises), so the
    # deterministic fixture is written into the internal ``_boards`` buffer directly.
    assert env._boards.shape == (count, 16)
    env._boards[:] = mergeable

    result = env.step(np.full(count, int(Action.LEFT), dtype=np.uint8))

    assert np.all(result.legal), "LEFT is legal on this board for every game"
    assert np.all(result.rewards == 4), f"rewards {np.unique(result.rewards).tolist()}"
    assert np.all(env.scores == 4), f"scores {np.unique(env.scores).tolist()}"

    env.reset(seed=SEED)
    assert np.all(env.scores == 0), f"scores {np.unique(env.scores).tolist()}"


def test_reset_where_only_touches_the_selected_environments():
    env = Fast2048BatchEnv(32, seed=SEED)
    env.reset(seed=SEED)
    env.step(np.full(32, int(Action.LEFT), dtype=np.uint8))
    env.step(np.full(32, int(Action.DOWN), dtype=np.uint8))
    before_boards = env.copy_boards()
    before_scores = env.copy_scores()

    mask = np.zeros(32, dtype=bool)
    mask[[3, 7, 11]] = True
    env.reset_where(mask)

    untouched = ~mask
    assert np.array_equal(env.boards[untouched], before_boards[untouched])
    assert np.array_equal(env.scores[untouched], before_scores[untouched])
    for index in np.flatnonzero(mask):
        assert int((env.boards[index] != 0).sum()) == 2, index
        assert int(env.scores[index]) == 0


def test_reset_where_with_an_empty_mask_is_a_no_op():
    env = Fast2048BatchEnv(8, seed=SEED)
    env.reset(seed=SEED)
    before = env.copy_boards()
    env.reset_where(np.zeros(8, dtype=bool))
    assert np.array_equal(env.boards, before)


def test_reset_where_validates_its_mask():
    env = Fast2048BatchEnv(8, seed=SEED)
    with pytest.raises(ValueError):
        env.reset_where(np.zeros(7, dtype=bool))
    with pytest.raises(ValueError):
        env.reset_where(np.zeros(8, dtype=np.uint8))
    with pytest.raises(ValueError):
        env.reset_where(np.zeros((8, 1), dtype=bool))


# --------------------------------------------------------------------------- #
# step: deterministic differential against M0
# --------------------------------------------------------------------------- #


def test_step_is_a_deterministic_reference_transition():
    """``step`` == M0 afterstate + a predetermined spawn, for random boards."""
    entries = random_boards(256, 17, SEED, empty_probability=0.4)
    env = Fast2048BatchEnv(entries.shape[0], seed=SEED)
    env._boards[:] = entries
    actions = np.arange(entries.shape[0], dtype=np.uint8) % 4

    result = env.step(actions)
    for index in range(entries.shape[0]):
        entry = entries[index]
        action = int(actions[index])
        expected_move = move_without_spawn(entry, Action(action))
        assert bool(result.legal[index]) == expected_move.moved, (
            f"sample={index} board={board_str(entry)} action={Action(action).name}"
        )
        assert np.array_equal(result.afterstates[index], expected_move.afterstate)
        assert int(result.rewards[index]) == expected_move.reward
        if not expected_move.moved:
            assert np.array_equal(result.states[index], entry)
            assert int(result.spawn_indices[index]) == -1
            assert int(result.spawn_exponents[index]) == 0
            continue
        # Rebuild the official state from the reported spawn.
        rebuilt = apply_spawn_batch(
            expected_move.afterstate[None, :],
            np.array([result.spawn_indices[index]], dtype=np.int64),
            np.array([result.spawn_exponents[index]], dtype=np.uint8),
        )[0]
        assert np.array_equal(result.states[index], rebuilt)
        assert bool(result.terminated[index]) == is_terminal(result.states[index])


def test_step_scores_only_accumulate_legal_rewards():
    entries = batch(
        [
            board((1, 0, 0, 0), (1, 0, 0, 0), (1, 0, 0, 0), (1, 0, 0, 0)),
            board((0, 0, 0, 1), (1, 1, 0, 0), (2, 0, 0, 0), (0, 0, 0, 0)),
        ]
    )
    env = Fast2048BatchEnv(2, seed=SEED)
    env._boards[:] = entries
    # UP merges two pairs of 1s in the first column: reward 4 + 4 = 8.
    result = env.step(np.array([int(Action.UP), int(Action.LEFT)], dtype=np.uint8))
    assert int(result.rewards[0]) == 8
    assert int(env.scores[0]) == int(result.rewards[0])
    # Board 1: LEFT merges [1,1,0,0] -> [2,0,0,0], reward 4.
    assert bool(result.legal[1])
    assert int(result.rewards[1]) == 4
    assert int(env.scores[1]) == 4


def test_illegal_action_semantics():
    """Illegal action: board unchanged, reward 0, score unchanged, no spawn."""
    entries = batch(
        [
            board((1, 2, 4, 8), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)),
            board((0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)),
        ]
    )
    env = Fast2048BatchEnv(2, seed=SEED)
    env._boards[:] = entries
    env._scores[:] = [123, 456]

    result = env.step(np.array([int(Action.LEFT), int(Action.LEFT)], dtype=np.uint8))
    for index in (0, 1):
        assert not bool(result.legal[index])
        assert np.array_equal(result.states[index], entries[index])
        assert np.array_equal(result.afterstates[index], entries[index])
        assert np.array_equal(env.boards[index], entries[index])
        assert int(result.rewards[index]) == 0
        assert int(result.spawn_indices[index]) == -1
        assert int(result.spawn_exponents[index]) == 0
    assert env.scores.tolist() == [123, 456]


def test_illegal_action_does_not_consume_any_randomness():
    """A step where every requested action is illegal must not touch the RNG.

    This is the batch form of M0's rule: an illegal action consumes no spawn
    randomness, so the generator state before and after the step is identical.
    """
    entry = board((1, 2, 4, 8), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0))
    env = Fast2048BatchEnv(2, seed=SEED)
    env._boards[0] = entry
    env._boards[1] = entry
    before = env.rng.bit_generator.state

    result = env.step(np.full(2, int(Action.LEFT), dtype=np.uint8))
    assert not result.legal.any()
    assert np.array_equal(env.rng.bit_generator.state["state"]["state"], before["state"]["state"])
    assert np.array_equal(env.boards[0], entry)
    assert np.array_equal(env.boards[1], entry)
    assert result.spawn_indices.tolist() == [-1, -1]
    assert result.spawn_exponents.tolist() == [0, 0]

    # The very next legal step must therefore see the untouched stream.
    twin = Fast2048BatchEnv(2, seed=SEED)
    twin._boards[0] = entry
    twin._boards[1] = entry
    after = env.step(np.full(2, int(Action.RIGHT), dtype=np.uint8))
    expected = twin.step(np.full(2, int(Action.RIGHT), dtype=np.uint8))
    assert np.array_equal(after.states, expected.states)
    assert np.array_equal(after.spawn_indices, expected.spawn_indices)
    assert np.array_equal(after.spawn_exponents, expected.spawn_exponents)


def test_partial_illegal_step_keeps_streams_in_step():
    """Two identical envs fed the same action sequence agree on every spawn."""
    entries = batch(
        [
            board((1, 2, 4, 8), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)),
            board((0, 0, 0, 1), (1, 1, 0, 0), (2, 0, 0, 0), (0, 0, 0, 0)),
        ]
    )
    left = np.full(2, int(Action.LEFT), dtype=np.uint8)
    right = np.full(2, int(Action.RIGHT), dtype=np.uint8)

    env_a = Fast2048BatchEnv(2, seed=SEED)
    env_a._boards[:] = entries
    env_b = Fast2048BatchEnv(2, seed=SEED)
    env_b._boards[:] = entries

    for _ in range(8):
        first_a = env_a.step(left)
        first_b = env_b.step(left)
        assert np.array_equal(first_a.states, first_b.states)
        assert np.array_equal(first_a.spawn_indices, first_b.spawn_indices)
        # Board 0's LEFT is illegal, so it reports no spawn either way.
        assert int(first_a.spawn_indices[0]) == -1
        assert int(first_a.spawn_exponents[0]) == 0
        assert np.array_equal(env_a.boards[0], env_b.boards[0])

        second_a = env_a.step(right)
        second_b = env_b.step(right)
        assert np.array_equal(second_a.states, second_b.states)


def test_step_does_not_auto_reset_terminal_games():
    """A terminal game stays terminal until the caller resets it explicitly."""
    terminal_board = board((1, 2, 1, 2), (2, 1, 2, 1), (1, 2, 1, 2), (2, 1, 2, 1))
    env = Fast2048BatchEnv(1, seed=SEED)
    env._boards[0] = terminal_board
    assert bool(is_terminal_batch(env.boards)[0])

    result = env.step(np.array([int(Action.LEFT)], dtype=np.uint8))
    assert bool(result.terminated[0])
    assert np.array_equal(env.boards[0], terminal_board)
    assert int(result.spawn_indices[0]) == -1

    env.reset_where(np.array([True]))
    assert not bool(is_terminal_batch(env.boards)[0])
    assert np.array_equal(env.boards[0], env.copy_boards()[0])


def test_step_validates_actions():
    env = Fast2048BatchEnv(4, seed=SEED)
    env.reset(seed=SEED)
    with pytest.raises(ValueError):
        env.step(np.array([0, 1, 2], dtype=np.int64))
    with pytest.raises(ValueError):
        env.step(np.array([0.0, 1.0, 2.0, 3.0]))
    with pytest.raises(ValueError):
        env.step(np.array([True, False, True, False]))
    with pytest.raises(ValueError):
        env.step(np.array([0, 1, 2, 4], dtype=np.int64))


def test_step_returns_independent_buffers():
    env = Fast2048BatchEnv(4, seed=SEED)
    env.reset(seed=SEED)
    result = env.step(np.full(4, int(Action.LEFT), dtype=np.uint8))
    states_before = env.copy_boards()
    result.states[0, 0] = 200
    result.afterstates[0, 0] = 200
    result.rewards[0] = 12345
    result.legal[0] = not result.legal[0]
    result.terminated[0] = not result.terminated[0]
    result.spawn_indices[0] = 99
    result.spawn_exponents[0] = 99
    assert np.array_equal(env.boards, states_before)
    env.step(np.full(4, int(Action.LEFT), dtype=np.uint8))


# --------------------------------------------------------------------------- #
# Trajectory-level checks
# --------------------------------------------------------------------------- #


def test_full_trajectories_match_reference_with_predetermined_spawns():
    """Drive both envs through the same action and spawn script.

    The reference environment is advanced by *installing* the spawn outcome that
    the fast environment reported, so the two never have to share an RNG stream.
    """
    rng = np.random.default_rng(SEED)
    for _ in range(5):
        entry = rng.integers(0, 18, size=16).astype(np.uint8)
        entry[rng.random(16) < 0.4] = 0
        env = Fast2048BatchEnv(1, seed=SEED + int(entry[0]))
        env._boards[0] = entry
        reference = Reference2048Env(seed=0)
        reference._board = entry.copy()
        reference._score = 0

        for _step in range(40):
            action = int(rng.integers(0, 4))
            result = env.step(np.array([action], dtype=np.uint8))
            expected_move = move_without_spawn(reference._board, Action(action))
            assert np.array_equal(result.afterstates[0], expected_move.afterstate)
            assert int(result.rewards[0]) == expected_move.reward

            # ``terminated`` always describes the *new* official state.  When the
            # action was legal the new state is ``result.states[0]``; when it was
            # illegal the board is unchanged, so the reference board is still the
            # right thing to compare against.
            if expected_move.moved:
                spawned = apply_spawn_batch(
                    expected_move.afterstate[None, :],
                    np.array([result.spawn_indices[0]], dtype=np.int64),
                    np.array([result.spawn_exponents[0]], dtype=np.uint8),
                )[0]
                assert np.array_equal(result.states[0], spawned)
                reference._board = spawned
                reference._score += int(expected_move.reward)
                assert int(env.scores[0]) == reference.score
            else:
                assert int(result.spawn_indices[0]) == -1
                assert int(result.spawn_exponents[0]) == 0
                assert np.array_equal(result.states[0], reference._board)

            assert bool(result.terminated[0]) == is_terminal(reference._board), (
                f"board={board_str(reference._board)} "
                f"fast={bool(result.terminated[0])}"
            )
            if bool(result.terminated[0]):
                env.reset_where(np.array([True]))
                reference._board = env.copy_boards()[0]
                reference._score = 0


def test_legal_mask_and_terminal_agree_with_reference_over_a_rollout():
    """Roll a batch forward and compare masks against M0 at every state."""
    env = Fast2048BatchEnv(32, seed=SEED)
    env.reset(seed=SEED)
    rng = np.random.default_rng(SEED)
    for _ in range(30):
        boards = env.copy_boards()
        legal = np.zeros((32, 4), dtype=bool)
        terminal = np.zeros(32, dtype=bool)
        for index in range(32):
            legal[index] = legal_mask(boards[index])
            terminal[index] = is_terminal(boards[index])

        from game2048.fast_env import legal_mask_batch

        assert np.array_equal(legal_mask_batch(boards), legal)
        assert np.array_equal(is_terminal_batch(boards), terminal)

        actions = rng.integers(0, 4, size=32).astype(np.uint8)
        env.step(actions)
        env.reset_where(terminal)


def test_spawn_random_batch_matches_m0_distribution_support():
    """``spawn_random_batch`` only ever produces M0-legal spawn outcomes."""
    entries = random_boards(512, 17, SEED, empty_probability=0.4)
    from game2048.fast_env import spawn_random_batch

    result = spawn_random_batch(entries, np.random.default_rng(SEED))
    for index in range(entries.shape[0]):
        allowed = {
            (outcome.spawn_index, outcome.spawn_exponent)
            for outcome in enumerate_spawns(entries[index])
        }
        assert (int(result.spawn_indices[index]), int(result.spawn_exponents[index])) in allowed, (
            f"sample={index} board={board_str(entries[index])} produced "
            f"({int(result.spawn_indices[index])}, "
            f"{int(result.spawn_exponents[index])})"
        )


def test_move_batch_used_by_env_matches_reference_on_env_boards():
    """The environment's own board buffer drives ``move_batch`` correctly."""
    env = Fast2048BatchEnv(64, seed=SEED)
    env.reset(seed=SEED)
    rng = np.random.default_rng(SEED)
    for _ in range(20):
        boards = env.copy_boards()
        actions = rng.integers(0, 4, size=64).astype(np.uint8)
        result = move_batch(boards, actions)
        for index in range(64):
            expected = move_without_spawn(boards[index], Action(int(actions[index])))
            assert np.array_equal(result.afterstates[index], expected.afterstate)
            assert int(result.rewards[index]) == expected.reward
            assert bool(result.moved[index]) == expected.moved
        env.step(actions)


def test_reference_env_single_spawn_matches_fast_single_spawn_support():
    """``spawn_random`` and ``spawn_random_batch`` agree on support and weights."""
    entry = board((0, 0, 3, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0))
    reference_rng = np.random.default_rng(SEED)
    counts = {1: 0, 2: 0}
    for _ in range(2_000):
        counts[spawn_random(entry, reference_rng).spawn_exponent] += 1
    assert abs(counts[1] / 2_000 - 0.9) < 0.03


def test_env_boards_are_always_contiguous_uint8():
    env = Fast2048BatchEnv(128, seed=SEED)
    env.reset(seed=SEED)
    assert env._boards.dtype == np.uint8
    assert env._boards.flags.c_contiguous
    assert np.array_equal(env.boards, env._boards)
    assert env._scores.dtype == np.int64
