from __future__ import annotations

import copy

import numpy as np
import pytest

from game2048.fast_env import Fast2048BatchEnv, legal_mask_batch as numpy_legal
from game2048.m2_fast_backend import legal_mask_batch as cpp_legal
from game2048.m2_rollout_env import M2RolloutBatchEnv


SEED = 20260919


def rng_state(env):
    return copy.deepcopy(env.rng.bit_generator.state)


def assert_rng_equal(a, b):
    assert a == b


def choose_actions(legal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    priority = rng.random(legal.shape)
    priority[~legal] = -1.0
    return priority.argmax(axis=1).astype(np.uint8)


def assert_step_equal(a, b):
    for field in (
        "states",
        "afterstates",
        "rewards",
        "legal",
        "terminated",
        "spawn_indices",
        "spawn_exponents",
    ):
        assert np.array_equal(getattr(a, field), getattr(b, field)), field


def test_seeded_rollout_matches_frozen_m1_spawn_rng_and_reset_where():
    n = 512
    frozen = Fast2048BatchEnv(n, seed=SEED)
    rollout = M2RolloutBatchEnv(n, seed=SEED)
    assert np.array_equal(frozen.reset(seed=SEED), rollout.reset(seed=SEED))
    assert np.array_equal(rollout.current_legal, numpy_legal(frozen.boards))
    action_rng = np.random.default_rng(SEED + 1)

    for _ in range(200):
        legal = numpy_legal(frozen.boards)
        actions = choose_actions(legal, action_rng)
        a = frozen.step(actions)
        b = rollout.step(actions)
        assert_step_equal(a, b)
        assert np.array_equal(frozen.boards, rollout.boards)
        assert np.array_equal(frozen.scores, rollout.scores)
        assert np.array_equal(rollout.current_legal, numpy_legal(frozen.boards))
        assert_rng_equal(rng_state(frozen), rng_state(rollout))

        if a.terminated.any():
            frozen.reset_where(a.terminated)
            rollout.reset_where(a.terminated)
            assert np.array_equal(frozen.boards, rollout.boards)
            assert np.array_equal(frozen.scores, rollout.scores)
            assert np.array_equal(rollout.current_legal, numpy_legal(frozen.boards))
            assert_rng_equal(rng_state(frozen), rng_state(rollout))


def test_illegal_action_is_noop_and_consumes_no_rng():
    env = M2RolloutBatchEnv(1, seed=SEED + 10)
    env.reset(seed=SEED + 10)
    env._boards[0] = np.array(
        [1, 2, 0, 0] + [0] * 12, dtype=np.uint8
    )
    env._scores[0] = 1234
    env._current_legal[:] = cpp_legal(env._boards)

    board_before = env.copy_boards()
    score_before = env.copy_scores()
    rng_before = rng_state(env)
    result = env.step(np.array([2], dtype=np.uint8))  # LEFT: already compressed

    assert result.legal.tolist() == [False]
    assert result.rewards.tolist() == [0]
    assert result.spawn_indices.tolist() == [-1]
    assert result.spawn_exponents.tolist() == [0]
    assert np.array_equal(env.boards, board_before)
    assert np.array_equal(env.scores, score_before)
    assert_rng_equal(rng_state(env), rng_before)


def test_reset_where_updates_only_selected_legal_cache_rows():
    env = M2RolloutBatchEnv(64, seed=SEED + 20)
    env.reset(seed=SEED + 20)
    boards_before = env.copy_boards()
    legal_before = env.current_legal.copy()
    mask = np.zeros(64, dtype=bool)
    mask[::7] = True

    env.reset_where(mask)

    assert np.array_equal(env.boards[~mask], boards_before[~mask])
    assert np.array_equal(env.current_legal[~mask], legal_before[~mask])
    assert np.array_equal(env.current_legal[mask], cpp_legal(np.ascontiguousarray(env.boards[mask])))


@pytest.mark.parametrize("kind", ["reward", "score", "tile_selected", "tile_next_legal"])
def test_overflow_atomicity(kind):
    env = M2RolloutBatchEnv(1, seed=SEED + 30)
    env.reset(seed=SEED + 30)

    if kind == "reward":
        env._boards[0] = np.array([61, 61, 61, 61] + [0] * 12, dtype=np.uint8)
        env._scores[0] = 0
        action = 2
    elif kind == "score":
        env._boards[0] = np.array([61, 61, 0, 0] + [0] * 12, dtype=np.uint8)
        env._scores[0] = np.iinfo(np.int64).max - (1 << 62) + 1
        action = 2
    elif kind == "tile_selected":
        env._boards[0] = np.array([255, 255, 0, 0] + [0] * 12, dtype=np.uint8)
        env._scores[0] = 0
        action = 2
    else:
        env._boards[0] = np.array([255, 255, 0, 0] + [0] * 12, dtype=np.uint8)
        env._scores[0] = 0
        action = 0  # UP does not merge; next-legal sees the overflowing LEFT move.

    board_before = env.copy_boards()
    score_before = env.copy_scores()
    rng_before = rng_state(env)

    with pytest.raises(OverflowError):
        env.step(np.array([action], dtype=np.uint8))

    assert np.array_equal(env.boards, board_before)
    assert np.array_equal(env.scores, score_before)
    assert_rng_equal(rng_state(env), rng_before)


def test_live_views_remain_live_across_step_and_reset():
    env = M2RolloutBatchEnv(64, seed=SEED + 40)
    env.reset(seed=SEED + 40)
    boards_view = env.boards
    scores_view = env.scores
    legal_view = env.current_legal

    actions = np.argmax(env.current_legal, axis=1).astype(np.uint8)
    result = env.step(actions)
    assert np.array_equal(boards_view, env.boards)
    assert np.array_equal(scores_view, env.scores)
    assert np.array_equal(legal_view, env.current_legal)
    assert np.array_equal(result.terminated, ~env.current_legal.any(axis=1))

    mask = np.zeros(64, dtype=bool)
    mask[:8] = True
    env.reset_where(mask)
    assert np.array_equal(boards_view, env.boards)
    assert np.array_equal(scores_view, env.scores)
    assert np.array_equal(legal_view, env.current_legal)
