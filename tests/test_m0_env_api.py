"""M0 tests for the public ``Reference2048Env`` API.

Covers ``reset``, ``step``, ``board``, ``score``, ``legal_mask`` and
``is_terminal``, the illegal-action contract (including the "an illegal action
must not consume spawn randomness" requirement) and full deterministic
playouts.
"""

import numpy as np
import pytest

from game2048 import (
    ACTIONS,
    SPAWN_EXPONENT_TILE_2,
    SPAWN_EXPONENT_TILE_4,
    Action,
    Reference2048Env,
    is_terminal,
    legal_mask,
    move_without_spawn,
)

MAX_PLAYOUT_STEPS = 3000


def board_value_sum(board):
    """Sum of all tile values (``0 -> 0``, ``e > 0 -> 2 ** e``)."""
    return int(sum(0 if int(e) == 0 else 2 ** int(e) for e in np.asarray(board).tolist()))


def non_empty_count(board):
    return int(np.count_nonzero(np.asarray(board)))


def find_seed_with_illegal_and_legal_action():
    """Find a starting board that has both an illegal and a legal action."""
    for seed in range(2000):
        board = Reference2048Env(seed=seed).reset()
        mask = legal_mask(board)
        illegal = [action for action in ACTIONS if not mask[action]]
        legal = [action for action in ACTIONS if mask[action]]
        if illegal and legal:
            return seed, board, illegal[0], legal[0]
    raise AssertionError("no seed produced a board with an illegal action")


# --------------------------------------------------------------------------- #
# reset
# --------------------------------------------------------------------------- #


def test_reset_starts_with_exactly_two_tiles():
    for seed in range(200):
        env = Reference2048Env(seed=seed)
        board = env.reset()

        assert board.shape == (16,)
        assert board.dtype == np.uint8
        assert non_empty_count(board) == 2
        assert env.score == 0
        assert set(np.unique(board).tolist()) - {0} <= {
            SPAWN_EXPONENT_TILE_2,
            SPAWN_EXPONENT_TILE_4,
        }


def test_reset_places_the_two_tiles_in_different_cells():
    for seed in range(200):
        board = Reference2048Env(seed=seed).reset()
        assert non_empty_count(board) == 2


def test_reset_is_reproducible_for_a_given_seed():
    first = Reference2048Env(seed=42).reset()
    second = Reference2048Env(seed=42).reset()
    assert first.tolist() == second.tolist()


def test_reset_with_seed_matches_constructor_seed():
    assert Reference2048Env(seed=7).reset().tolist() == Reference2048Env().reset(seed=7).tolist()


def test_reset_reseeds_on_demand():
    env = Reference2048Env(seed=1)
    env.reset()
    env.step(Action.LEFT)
    assert env.reset(seed=1).tolist() == Reference2048Env(seed=1).reset().tolist()


def test_reset_returns_a_copy():
    env = Reference2048Env(seed=3)
    board = env.reset()
    board[:] = 0
    assert non_empty_count(env.board) == 2


def test_board_is_empty_before_reset():
    env = Reference2048Env(seed=5)
    assert env.board.tolist() == [0] * 16
    assert env.score == 0


# --------------------------------------------------------------------------- #
# step: legal actions
# --------------------------------------------------------------------------- #


def test_step_returns_structured_result():
    env = Reference2048Env(seed=11)
    env.reset()
    before = env.board
    mask = env.legal_mask()
    action = next(a for a in ACTIONS if mask[a])

    result = env.step(action)

    assert result.legal is True
    assert result.spawn_index is not None
    assert result.spawn_exponent in (SPAWN_EXPONENT_TILE_2, SPAWN_EXPONENT_TILE_4)
    assert result.reward >= 0
    assert result.state.tolist() == env.board.tolist()
    assert result.terminated == env.is_terminal()

    expected_afterstate = move_without_spawn(before, action).afterstate
    assert result.afterstate.tolist() == expected_afterstate.tolist()
    assert result.reward == move_without_spawn(before, action).reward


def test_legal_step_spawns_exactly_one_tile():
    env = Reference2048Env(seed=13)
    env.reset()
    for _ in range(50):
        if env.is_terminal():
            break
        action = next(a for a in ACTIONS if env.legal_mask()[a])
        result = env.step(action)

        assert non_empty_count(result.state) == non_empty_count(result.afterstate) + 1
        delta = board_value_sum(result.state) - board_value_sum(result.afterstate)
        assert delta in (2, 4)


def test_step_afterstate_matches_the_pure_core():
    env = Reference2048Env(seed=17)
    env.reset()
    for _ in range(30):
        if env.is_terminal():
            break
        action = next(a for a in ACTIONS if env.legal_mask()[a])
        before = env.board
        result = env.step(action)
        pure = move_without_spawn(before, action)
        assert result.afterstate.tolist() == pure.afterstate.tolist()
        assert result.reward == pure.reward


def test_score_only_counts_merge_rewards():
    env = Reference2048Env(seed=19)
    env.reset()
    total = 0
    while not env.is_terminal():
        action = next(a for a in ACTIONS if env.legal_mask()[a])
        result = env.step(action)
        total += result.reward
        assert env.score == total


# --------------------------------------------------------------------------- #
# step: illegal actions
# --------------------------------------------------------------------------- #


def test_illegal_action_leaves_everything_unchanged():
    seed, board, illegal, _ = find_seed_with_illegal_and_legal_action()
    env = Reference2048Env(seed=seed)
    env.reset()
    assert env.board.tolist() == board.tolist()

    score_before = env.score
    result = env.step(illegal)

    assert result.legal is False
    assert result.reward == 0
    assert result.state.tolist() == board.tolist()
    assert result.afterstate.tolist() == board.tolist()
    assert env.board.tolist() == board.tolist()
    assert env.score == score_before


def test_illegal_action_does_not_spawn():
    seed, board, illegal, _ = find_seed_with_illegal_and_legal_action()
    env = Reference2048Env(seed=seed)
    env.reset()
    count_before = non_empty_count(board)

    for _ in range(5):
        result = env.step(illegal)
        assert result.spawn_index is None
        assert result.spawn_exponent is None
        assert non_empty_count(env.board) == count_before
        assert board_value_sum(env.board) == board_value_sum(board)


def test_illegal_action_does_not_consume_spawn_rng():
    """Illegal-then-legal must equal legal-only, from the same RNG state."""
    seed, _, illegal, legal = find_seed_with_illegal_and_legal_action()

    env_with_illegal = Reference2048Env(seed=seed)
    env_with_illegal.reset()
    rng_state_before = env_with_illegal.rng.bit_generator.state
    env_with_illegal.step(illegal)
    assert env_with_illegal.rng.bit_generator.state == rng_state_before

    env_with_illegal.step(legal)

    env_direct = Reference2048Env(seed=seed)
    env_direct.reset()
    env_direct.step(legal)

    assert env_with_illegal.board.tolist() == env_direct.board.tolist()
    assert env_with_illegal.score == env_direct.score
    assert env_with_illegal.rng.bit_generator.state == env_direct.rng.bit_generator.state


def test_repeated_illegal_actions_are_completely_inert():
    seed, board, illegal, legal = find_seed_with_illegal_and_legal_action()

    env_many = Reference2048Env(seed=seed)
    env_many.reset()
    for _ in range(25):
        env_many.step(illegal)
    env_many.step(legal)

    env_direct = Reference2048Env(seed=seed)
    env_direct.reset()
    env_direct.step(legal)

    assert env_many.board.tolist() == env_direct.board.tolist()
    assert env_many.rng.bit_generator.state == env_direct.rng.bit_generator.state


def test_illegal_action_on_terminal_board_is_inert():
    env = Reference2048Env(seed=23)
    env.reset()
    while not env.is_terminal():
        action = next(a for a in ACTIONS if env.legal_mask()[a])
        env.step(action)

    board_before = env.board
    score_before = env.score
    rng_before = env.rng.bit_generator.state

    for action in ACTIONS:
        result = env.step(action)
        assert result.legal is False
        assert result.reward == 0
        assert result.spawn_index is None
        assert result.terminated is True

    assert env.board.tolist() == board_before.tolist()
    assert env.score == score_before
    assert env.rng.bit_generator.state == rng_before


def test_step_rejects_invalid_actions():
    env = Reference2048Env(seed=29)
    env.reset()
    for bad in (4, -1, 16, "LEFT", None):
        with pytest.raises(ValueError):
            env.step(bad)


# --------------------------------------------------------------------------- #
# board / legal_mask / is_terminal accessors
# --------------------------------------------------------------------------- #


def test_board_property_returns_an_independent_copy():
    env = Reference2048Env(seed=31)
    env.reset()
    snapshot = env.board
    snapshot[:] = 0
    assert non_empty_count(env.board) == 2


def test_step_result_arrays_are_independent():
    env = Reference2048Env(seed=37)
    env.reset()
    action = next(a for a in ACTIONS if env.legal_mask()[a])
    result = env.step(action)

    expected_state = env.board.tolist()
    result.state[:] = 0
    result.afterstate[:] = 0
    assert env.board.tolist() == expected_state


def test_legal_mask_and_is_terminal_match_the_free_functions():
    env = Reference2048Env(seed=41)
    env.reset()
    while not env.is_terminal():
        assert env.legal_mask().tolist() == legal_mask(env.board).tolist()
        assert env.is_terminal() == is_terminal(env.board)
        action = next(a for a in ACTIONS if env.legal_mask()[a])
        env.step(action)
    assert env.legal_mask().tolist() == [False] * 4


# --------------------------------------------------------------------------- #
# Deterministic playouts
# --------------------------------------------------------------------------- #


def play_random_game(seed, max_steps=MAX_PLAYOUT_STEPS):
    """Play uniformly random legal actions and check every step invariant."""
    env = Reference2048Env(seed=seed)
    env.reset()
    chooser = np.random.Generator(np.random.PCG64(seed + 1000003))

    trajectory = []
    total_reward = 0
    previous_value_sum = board_value_sum(env.board)

    while not env.is_terminal() and len(trajectory) < max_steps:
        mask = env.legal_mask()
        legal_actions = [action for action in ACTIONS if mask[action]]
        action = legal_actions[int(chooser.integers(0, len(legal_actions)))]

        result = env.step(action)

        # Every chosen action is legal, so the step must be a real step.
        assert result.legal is True
        assert result.spawn_index is not None
        assert result.terminated == env.is_terminal()
        assert result.state.tolist() == env.board.tolist()

        # Afterstate -> state adds exactly one tile worth 2 or 4.
        assert non_empty_count(result.state) == non_empty_count(result.afterstate) + 1
        delta = board_value_sum(result.state) - board_value_sum(result.afterstate)
        assert delta in (2, 4)

        # Moving conserves the tile value, spawning only ever adds 2 or 4.
        assert board_value_sum(result.afterstate) == previous_value_sum
        assert board_value_sum(result.state) == previous_value_sum + delta
        previous_value_sum = board_value_sum(result.state)

        total_reward += result.reward
        assert env.score == total_reward

        trajectory.append((int(action), result.reward, result.state.tolist()))

    return env, trajectory


def test_random_playout_reaches_terminal_and_respects_invariants():
    env, trajectory = play_random_game(seed=2024)

    assert len(trajectory) > 0
    assert env.is_terminal() is True
    assert env.legal_mask().tolist() == [False] * 4
    assert env.score > 0


def test_random_playout_is_reproducible():
    _, first = play_random_game(seed=99)
    _, second = play_random_game(seed=99)
    assert first == second


def test_different_seeds_give_different_games():
    _, first = play_random_game(seed=1)
    _, second = play_random_game(seed=2)
    assert first != second


@pytest.mark.parametrize("seed", [0, 3, 8, 21])
def test_playout_invariants_for_several_seeds(seed):
    env, trajectory = play_random_game(seed=seed)
    assert env.is_terminal() is True
    assert len(trajectory) > 10


def test_env_can_be_constructed_from_an_explicit_generator():
    rng = np.random.Generator(np.random.PCG64(20240519))
    env = Reference2048Env(rng=rng)
    env.reset()
    assert non_empty_count(env.board) == 2

    twin = Reference2048Env(rng=np.random.Generator(np.random.PCG64(20240519)))
    assert twin.reset().tolist() == env.board.tolist()


def test_env_rejects_seed_and_rng_together():
    with pytest.raises(ValueError):
        Reference2048Env(seed=1, rng=np.random.Generator(np.random.PCG64(1)))
