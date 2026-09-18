"""Fine-grained M2 CPU hotspot benchmark; read-only with respect to frozen M1."""
from __future__ import annotations

import cProfile
import io
import json
import pstats
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from game2048.fast_env import (  # noqa: E402
    BatchStepResult,
    Fast2048BatchEnv,
    _checked_add_nonnegative_int64,
    _move_batch,
    _move_groups,
    is_terminal_batch,
    legal_mask_batch,
)

SIZES = (4096, 8192, 16384)
SEED = 204802
MIN_WARMUP = 20
MIN_ITERS = 100
MIN_SECONDS = 2.0
OUT = ROOT / "m2_cpu_hotspots.json"
def choose_legal(legal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    priorities = rng.random(legal.shape)
    priorities[~legal] = -1.0
    return priorities.argmax(axis=1).astype(np.uint8)


def prepare_env(n: int, seed: int) -> tuple[Fast2048BatchEnv, np.random.Generator]:
    env = Fast2048BatchEnv(n, seed=seed)
    env.reset()
    policy_rng = np.random.default_rng(seed + 1)
    for _ in range(48):
        legal = legal_mask_batch(env.boards)
        dead = ~legal.any(axis=1)
        if dead.any():
            env.reset_where(dead)
            legal = legal_mask_batch(env.boards)
        actions = choose_legal(legal, policy_rng)
        result = env.step(actions)
        if result.terminated.any():
            env.reset_where(result.terminated)
    return env, policy_rng


def bench(fn, boards_per_call: int) -> dict:
    for _ in range(MIN_WARMUP):
        fn()
    start = time.perf_counter()
    iterations = 0
    while iterations < MIN_ITERS or time.perf_counter() - start < MIN_SECONDS:
        fn()
        iterations += 1
    elapsed = time.perf_counter() - start
    us = elapsed * 1e6 / iterations
    return {
        "iterations": iterations,
        "elapsed_s": elapsed,
        "us_per_iteration": us,
        "boards_per_s": boards_per_call * iterations / elapsed,
        "allocation_count": None,
        "allocation_note": "NumPy native allocations are not reliably counted by Python tracemalloc.",
    }


def bench_step(env: Fast2048BatchEnv, rng: np.random.Generator) -> dict:
    for _ in range(MIN_WARMUP):
        legal = legal_mask_batch(env.boards)
        dead = ~legal.any(axis=1)
        if dead.any():
            env.reset_where(dead)
            legal = legal_mask_batch(env.boards)
        result = env.step(choose_legal(legal, rng))
        if result.terminated.any():
            env.reset_where(result.terminated)

    measured = 0.0
    iterations = 0
    n = env.num_envs
    while iterations < MIN_ITERS or measured < MIN_SECONDS:
        legal = legal_mask_batch(env.boards)
        dead = ~legal.any(axis=1)
        if dead.any():
            env.reset_where(dead)
            legal = legal_mask_batch(env.boards)
        actions = choose_legal(legal, rng)
        start = time.perf_counter()
        result = env.step(actions)
        measured += time.perf_counter() - start
        if result.terminated.any():
            env.reset_where(result.terminated)
        iterations += 1
    us = measured * 1e6 / iterations
    return {
        "iterations": iterations,
        "elapsed_s": measured,
        "us_per_iteration": us,
        "boards_per_s": n * iterations / measured,
        "allocation_count": None,
        "allocation_note": "Step allocation is represented separately by output_copy_allocation.",
    }


def profile_text(n: int = 4096, loops: int = 30) -> str:
    env, rng = prepare_env(n, SEED + 700)
    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(loops):
        legal = legal_mask_batch(env.boards)
        dead = ~legal.any(axis=1)
        if dead.any():
            env.reset_where(dead)
            legal = legal_mask_batch(env.boards)
        result = env.step(choose_legal(legal, rng))
        if result.terminated.any():
            env.reset_where(result.terminated)
    profiler.disable()
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumulative").print_stats(35)
    return stream.getvalue()


def benchmark_size(n: int) -> tuple[dict, dict]:
    env, rng = prepare_env(n, SEED + n)
    boards = env.copy_boards()
    scores = env.copy_scores()
    legal = legal_mask_batch(boards)
    dead = ~legal.any(axis=1)
    if dead.any():
        env.reset_where(dead)
        boards = env.copy_boards()
        scores = env.copy_scores()
        legal = legal_mask_batch(boards)
    actions = choose_legal(legal, rng)

    move_reward = _move_batch(boards, actions)
    moved = move_reward.moved
    rows = np.flatnonzero(moved)
    rewards = np.where(moved, move_reward.rewards, np.int64(0)).astype(np.int64, copy=False)
    score_sink = np.empty_like(scores)

    placement_env = Fast2048BatchEnv(n, seed=SEED + n + 77)
    placement_env.reset()
    scratch_base = move_reward.afterstates[rows].copy()
    placement_rows = rows.copy()
    placement_draws = np.random.default_rng(SEED + n + 88).random((rows.size, 2))
    placement_scratch = np.empty_like(scratch_base)

    post_env = Fast2048BatchEnv(n, seed=SEED + n + 99)
    post_env._boards[:] = boards
    post_env._scores[:] = scores
    post_result = post_env.step(actions)
    post_states = post_result.states.copy()

    reset_env = Fast2048BatchEnv(n, seed=SEED + n + 111)
    reset_env.reset()
    reset_mask = np.zeros(n, dtype=bool)
    reset_mask[::97] = True

    def selected_move_only():
        return _move_groups(boards, actions, compute_rewards=False)

    def selected_move_reward():
        return _move_batch(boards, actions)

    def score_check_update():
        new_scores = _checked_add_nonnegative_int64(
            scores, rewards, context="m2 hotspot benchmark score accumulation"
        )
        score_sink[:] = new_scores
        return score_sink

    def spawn_prep():
        active_rows = np.flatnonzero(moved)
        return active_rows, move_reward.afterstates[active_rows].copy()

    draw_rng = np.random.default_rng(SEED + n + 122)

    def random_draw():
        return draw_rng.random((rows.size, 1, 2))

    def spawn_place():
        placement_scratch[:] = scratch_base
        return placement_env._place_tiles(
            placement_scratch, placement_draws, placement_rows
        )

    spawn_indices = post_result.spawn_indices.copy()
    spawn_exponents = post_result.spawn_exponents.copy()
    terminated = post_result.terminated.copy()

    def output_copy_allocation():
        return BatchStepResult(
            states=post_states.copy(),
            afterstates=move_reward.afterstates.copy(),
            rewards=rewards.copy(),
            legal=moved.copy(),
            terminated=terminated.copy(),
            spawn_indices=spawn_indices.copy(),
            spawn_exponents=spawn_exponents.copy(),
        )

    metrics: dict[str, dict] = {}
    metrics["current_legal"] = bench(lambda: legal_mask_batch(boards), n)
    metrics["selected_move_only"] = bench(selected_move_only, n)
    reward_mode = bench(selected_move_reward, n)
    metrics["selected_move_with_reward"] = reward_mode

    move_only_us = metrics["selected_move_only"]["us_per_iteration"]
    reward_mode_us = reward_mode["us_per_iteration"]
    reward_overhead_us = max(0.0, reward_mode_us - move_only_us)
    metrics["reward_calculation_aggregation"] = {
        "method": "reward-mode minus movement-only on the same _move_groups kernel",
        "us_per_iteration": reward_overhead_us,
        "boards_per_s": (n * 1e6 / reward_overhead_us) if reward_overhead_us > 0 else None,
        "allocation_count": None,
        "allocation_note": "Derived differential; not a standalone allocation count.",
    }
    metrics["score_overflow_check_update"] = bench(score_check_update, n)
    metrics["spawn_preparation"] = bench(spawn_prep, n)
    metrics["random_draw_generation"] = bench(random_draw, rows.size)
    metrics["spawn_placement"] = bench(spawn_place, rows.size)
    metrics["post_spawn_terminal"] = bench(lambda: is_terminal_batch(post_states), n)
    metrics["output_copy_allocation"] = bench(output_copy_allocation, n)
    metrics["reset_where"] = bench(lambda: reset_env.reset_where(reset_mask), int(reset_mask.sum()))

    step_env, step_rng = prepare_env(n, SEED + n + 133)
    metrics["total_env_step"] = bench_step(step_env, step_rng)

    cpu_decision_us = (
        metrics["current_legal"]["us_per_iteration"]
        + metrics["total_env_step"]["us_per_iteration"]
        + metrics["reset_where"]["us_per_iteration"]
    )
    for value in metrics.values():
        us = value.get("us_per_iteration")
        if isinstance(us, (int, float)):
            value["percent_of_cpu_decision"] = us / cpu_decision_us * 100.0

    terminal_from_legal = ~legal_mask_batch(post_states).any(axis=1)
    duplicate_ok = bool(np.array_equal(post_result.terminated, terminal_from_legal))
    if not duplicate_ok:
        raise AssertionError("post-step terminal does not equal ~next_legal.any(axis=1)")

    duplicate = {
        "terminal_equals_not_next_legal_any": duplicate_ok,
        "current_state_legal_directional_moves": 4,
        "selected_action_directional_moves": 1,
        "post_spawn_terminal_directional_moves": 4,
        "next_iteration_legal_directional_moves": 4,
        "steady_state_directional_move_evaluations": 13,
        "avoidable_duplicate_directional_move_evaluations": 4,
        "avoidable_directional_move_share_percent": 4.0 / 13.0 * 100.0,
        "reusable_contract": [
            "next_legal supplies the current terminal verdict",
            "the same next_legal is the next policy illegal-action mask",
        ],
    }
    return metrics, duplicate


def main() -> int:
    payload = {
        "methodology": {
            "sizes": list(SIZES),
            "warmup_min_iterations": MIN_WARMUP,
            "measurement_min_iterations": MIN_ITERS,
            "measurement_min_seconds": MIN_SECONDS,
            "reward_component_note": (
                "Reward cost is isolated as reward-producing movement minus identical "
                "movement-only execution; frozen M1 source is not modified."
            ),
            "complete_decision_denominator": (
                "current legal + env.step + reset_where CPU time; GPU work is intentionally excluded"
            ),
        },
        "sizes": {},
        "profiler": {},
    }
    duplicate_reference = None
    for n in SIZES:
        print(f"benchmarking {n} boards...", flush=True)
        metrics, duplicate = benchmark_size(n)
        payload["sizes"][str(n)] = metrics
        if duplicate_reference is None:
            duplicate_reference = duplicate
    payload["duplicate_movement"] = duplicate_reference
    payload["profiler"]["4096_cprofile"] = profile_text()

    ranking_source = payload["sizes"]["4096"]
    ranked = sorted(
        (
            (name, data.get("us_per_iteration", 0.0))
            for name, data in ranking_source.items()
            if name not in {"total_env_step", "selected_move_with_reward"}
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    payload["microbench_top3_4096"] = [
        {"name": name, "us_per_iteration": value} for name, value in ranked[:3]
    ]

    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(OUT),
        "top3_4096": payload["microbench_top3_4096"],
        "duplicate_movement": payload["duplicate_movement"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
