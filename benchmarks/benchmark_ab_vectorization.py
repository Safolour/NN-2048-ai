"""Same-machine A/B comparison of the *old* per-board M1 path and the new one.

The M1 movement kernel was rewritten (see ``reports/m1/M1_REPORT.md`` section R).  The old
code was a working-tree-only draft and cannot be restored from git without
rewriting history, which the task explicitly forbids.  What *is* fully
recoverable, and is the honest thing to measure, is the per-board reference route
itself: M0's frozen ``move_without_spawn`` applied board by board.

This tool therefore measures, back to back on the same machine, in the same
process, with the same seed and workload:

``OLD``
    the per-board scalar route.  Every operation loops over boards in Python and
    calls ``reference_env.move_without_spawn`` per board -- exactly the shape of
    the old M1 draft's hot path.

``NEW``
    the vectorized ``game2048.fast_env`` batch route.

Both sides produce *identical* results (the differential suite proves it), so the
ratio is a pure implementation comparison, not a workload change.

Usage
-----
::

    python benchmarks/benchmark_ab_vectorization.py
    python benchmarks/benchmark_ab_vectorization.py --json ab_vectorization.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from game2048.fast_env import (  # noqa: E402
    Fast2048BatchEnv,
    is_terminal_batch,
    legal_mask_batch,
    move_batch,
)
from game2048.reference_env import (  # noqa: E402
    Action,
    is_terminal,
    legal_mask,
    move_without_spawn,
)

from _utils import (  # noqa: E402
    BENCHMARK_SEED,
    format_metadata,
    format_rate,
    environment_metadata,
    rows_to_table,
    time_callable,
)

#: Boards per primitive measurement (matches the benchmark's primitive section).
PRIMITIVE_BATCH = 4_096
#: Environment count and steps for the end-to-end comparison.
ENV_COUNT = 4_096
ENV_STEPS = 48
#: Actions cycled through, so neither side can special-case one direction.
ACTIONS = (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT)


# --------------------------------------------------------------------------- #
# Fixed inputs
# --------------------------------------------------------------------------- #


def make_boards(count: int) -> np.ndarray:
    rng = np.random.default_rng(BENCHMARK_SEED)
    boards = rng.integers(0, 12, size=(count, 16)).astype(np.uint8)
    return np.ascontiguousarray(boards)


def make_actions(count: int) -> np.ndarray:
    rng = np.random.default_rng(BENCHMARK_SEED + 1)
    return rng.integers(0, 4, size=count).astype(np.uint8)


# --------------------------------------------------------------------------- #
# The two implementations of each primitive
# --------------------------------------------------------------------------- #


def old_move_batch(boards: np.ndarray, actions: np.ndarray):
    """Per-board scalar movement: one M0 call per board."""
    count = boards.shape[0]
    afterstates = np.empty((count, 16), dtype=np.uint8)
    rewards = np.zeros(count, dtype=np.int64)
    moved = np.zeros(count, dtype=bool)
    for index in range(count):
        result = move_without_spawn(boards[index], Action(int(actions[index])))
        afterstates[index] = result.afterstate
        rewards[index] = result.reward
        moved[index] = result.moved
    return afterstates, rewards, moved


def new_move_batch(boards: np.ndarray, actions: np.ndarray):
    result = move_batch(boards, actions)
    return result.afterstates, result.rewards, result.moved


def old_legal_mask_batch(boards: np.ndarray) -> np.ndarray:
    count = boards.shape[0]
    mask = np.zeros((count, 4), dtype=bool)
    for index in range(count):
        mask[index] = legal_mask(boards[index])
    return mask


def new_legal_mask_batch(boards: np.ndarray) -> np.ndarray:
    return legal_mask_batch(boards)


def old_is_terminal_batch(boards: np.ndarray) -> np.ndarray:
    count = boards.shape[0]
    terminal = np.zeros(count, dtype=bool)
    for index in range(count):
        terminal[index] = is_terminal(boards[index])
    return terminal


def new_is_terminal_batch(boards: np.ndarray) -> np.ndarray:
    return is_terminal_batch(boards)


# --------------------------------------------------------------------------- #
# End-to-end environment stepping
# --------------------------------------------------------------------------- #


def plan_actions(count: int, steps: int, seed: int) -> List[np.ndarray]:
    """One fixed action batch per step, identical for every repeat and both sides."""
    rng = np.random.default_rng(seed + 7)
    return [rng.integers(0, 4, size=count).astype(np.uint8) for _ in range(steps)]


def new_env_rollout(
    boards: np.ndarray, action_plan: List[np.ndarray], seed: int
) -> tuple[np.ndarray, int, int]:
    """The current Fast2048BatchEnv, reset onto fixed boards and stepped.

    Returns ``(final_boards, spawn_count, spawn_exponent_sum)``; the two spawn
    counters are what the fairness check compares, because the two sides draw
    their own random streams and therefore cannot be cell-for-cell identical.
    """
    env = Fast2048BatchEnv(boards.shape[0], seed=seed)
    env.reset(seed=seed)
    env._boards[:] = boards
    env._scores[:] = 0
    spawn_count = 0
    spawn_exponent_sum = 0
    for actions in action_plan:
        step = env.step(actions)
        spawn_count += int((step.spawn_indices >= 0).sum())
        spawn_exponent_sum += int(step.spawn_exponents.sum(dtype=np.int64))
    return env._boards.copy(), spawn_count, spawn_exponent_sum


def new_first_afterstates(
    boards: np.ndarray, actions: np.ndarray, seed: int
) -> np.ndarray:
    """The vectorized path's deterministic afterstates for one action batch."""
    env = Fast2048BatchEnv(boards.shape[0], seed=seed)
    env.reset(seed=seed)
    env._boards[:] = boards
    env._scores[:] = 0
    return env.step(actions).afterstates.copy()


def old_first_afterstates(
    boards: np.ndarray, actions: np.ndarray, seed: int
) -> np.ndarray:
    """M0's afterstates for the same boards and actions, board by board.

    ``afterstate`` is the board after the move and *before* the spawn, which is
    fully deterministic -- both sides must therefore agree on it exactly.  The
    spawn that follows is drawn from independent RNG streams on the two sides, so
    it is deliberately excluded here and compared statistically further down.
    """
    from game2048.reference_env import Reference2048Env

    envs = [Reference2048Env(seed=seed + index) for index in range(boards.shape[0])]
    for env, board in zip(envs, boards):
        env._board[:] = board
        env._score = 0
    return np.array(
        [
            env.step(int(actions[index])).afterstate
            for index, env in enumerate(envs)
        ],
        dtype=np.uint8,
    )


def old_env_rollout(
    boards: np.ndarray, action_plan: List[np.ndarray], seed: int
) -> tuple[np.ndarray, int, int]:
    """One reference environment per game, driven through M0's own ``step``.

    A faithful per-board equivalent of ``Fast2048BatchEnv.step``: M0's ``step``
    performs move -> spawn-on-moved-only -> score -> terminal, which is exactly the
    pipeline the fast path implements in batch.
    """
    from game2048.reference_env import Reference2048Env

    envs = [Reference2048Env(seed=seed + index) for index in range(boards.shape[0])]
    for env, board in zip(envs, boards):
        env._board[:] = board
        env._score = 0
    spawn_count = 0
    spawn_exponent_sum = 0
    for actions in action_plan:
        for index, env in enumerate(envs):
            outcome = env.step(int(actions[index]))
            if outcome.spawn_index is not None:
                spawn_count += 1
                spawn_exponent_sum += int(outcome.spawn_exponent)
    final = np.array([env._board for env in envs], dtype=np.uint8)
    return final, spawn_count, spawn_exponent_sum


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def compare(
    name: str,
    old: Callable[[], None],
    new: Callable[[], None],
    units: float,
    unit_name: str,
    warmup: int = 2,
    repeats: int = 3,
) -> Dict[str, object]:
    old_timing = time_callable(
        f"OLD {name}", old, units=units, unit_name=unit_name,
        warmup=warmup, repeats=repeats,
    )
    new_timing = time_callable(
        f"NEW {name}", new, units=units, unit_name=unit_name,
        warmup=warmup, repeats=repeats,
    )
    speedup = old_timing.median_s / new_timing.median_s
    return {
        "name": name,
        "units": units,
        "unit_name": unit_name,
        "old_s": old_timing.median_s,
        "new_s": new_timing.median_s,
        "old_rate": old_timing.median_rate,
        "new_rate": new_timing.median_rate,
        "speedup": speedup,
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Old (per-board) vs new (vectorized) M1 A/B benchmark"
    )
    parser.add_argument("--json", default=None, help="also write the raw results here")
    parser.add_argument("--seed", type=int, default=BENCHMARK_SEED)
    args = parser.parse_args(argv)

    metadata = environment_metadata()
    print("=" * 78)
    print("M1 movement vectorization A/B (same machine, same process, same workload)")
    print("=" * 78)
    print(format_metadata(metadata))
    print()

    boards = make_boards(PRIMITIVE_BATCH)
    actions = make_actions(PRIMITIVE_BATCH)

    # Correctness first: the comparison is only meaningful when both agree.
    old_a, old_r, old_m = old_move_batch(boards, actions)
    new_a, new_r, new_m = new_move_batch(boards, actions)
    assert np.array_equal(old_a, new_a), "afterstates differ between OLD and NEW"
    assert np.array_equal(old_r, new_r), "rewards differ between OLD and NEW"
    assert np.array_equal(old_m, new_m), "moved differs between OLD and NEW"
    assert np.array_equal(
        old_legal_mask_batch(boards), new_legal_mask_batch(boards)
    ), "legal masks differ between OLD and NEW"
    assert np.array_equal(
        old_is_terminal_batch(boards), new_is_terminal_batch(boards)
    ), "terminals differ between OLD and NEW"
    print("agreement check : OK (afterstates, rewards, moved, mask, terminal)")
    print()

    results: List[Dict[str, object]] = []

    results.append(
        compare(
            "move_batch",
            lambda: old_move_batch(boards, actions),
            lambda: new_move_batch(boards, actions),
            units=PRIMITIVE_BATCH,
            unit_name="boards",
        )
    )
    results.append(
        compare(
            "legal_mask_batch",
            lambda: old_legal_mask_batch(boards),
            lambda: new_legal_mask_batch(boards),
            units=PRIMITIVE_BATCH,
            unit_name="boards",
        )
    )
    results.append(
        compare(
            "is_terminal_batch",
            lambda: old_is_terminal_batch(boards),
            lambda: new_is_terminal_batch(boards),
            units=PRIMITIVE_BATCH,
            unit_name="boards",
        )
    )

    env_boards = make_boards(ENV_COUNT)
    action_plan = plan_actions(ENV_COUNT, ENV_STEPS, args.seed)
    total_transitions = ENV_COUNT * ENV_STEPS

    # The end-to-end comparison must be like-for-like.  Two independent checks:
    #
    # 1. from identical boards and identical actions the *deterministic* part of
    #    the step -- the afterstate, before any spawn -- must agree exactly;
    # 2. over a rollout the spawn population must stay aligned.  Both sides spawn
    #    exactly one tile per legal move from the same distribution, so the spawn
    #    count and the exponent sum are the right invariants; the two sides draw
    #    from separate RNG streams, so the spawn *cells* legitimately differ and a
    #    cell-for-cell comparison of the post-spawn state would be wrong.
    check_envs = 512
    check_boards = make_boards(check_envs)
    check_actions = plan_actions(check_envs, 1, args.seed)[0]
    old_after = old_first_afterstates(check_boards, check_actions, args.seed)
    new_after = new_first_afterstates(check_boards, check_actions, args.seed)
    assert np.array_equal(old_after, new_after), (
        "OLD and NEW afterstates differ for identical boards and actions"
    )
    del old_after, new_after, check_actions
    print(
        f"afterstate check: OK (identical pre-spawn afterstates on {check_envs} envs)"
    )

    check_plan = plan_actions(check_envs, 8, args.seed)
    old_end = old_env_rollout(check_boards, check_plan, args.seed)
    new_end = new_env_rollout(check_boards, check_plan, args.seed)
    print(
        f"spawn check     : 8 steps x {check_envs} envs -> "
        f"spawns OLD {old_end[1]:,} / NEW {new_end[1]:,}, "
        f"exponent sum OLD {old_end[2]:,} / NEW {new_end[2]:,}"
    )
    assert abs(old_end[1] - new_end[1]) <= max(1, old_end[1] // 200), (
        f"the two sides spawned a materially different number of tiles: "
        f"OLD {old_end[1]}, NEW {new_end[1]}"
    )
    del old_end, new_end, check_boards, check_plan
    print()

    results.append(
        compare(
            f"env step x{ENV_STEPS}",
            lambda: old_env_rollout(env_boards, action_plan, args.seed),
            lambda: new_env_rollout(env_boards, action_plan, args.seed),
            units=total_transitions,
            unit_name="transitions",
            warmup=1,
            repeats=3,
        )
    )

    rows = []
    for entry in results:
        rows.append(
            [
                str(entry["name"]),
                format_rate(float(entry["old_rate"])),
                format_rate(float(entry["new_rate"])),
                f"{float(entry['speedup']):.2f}x",
            ]
        )
    print()
    print(
        rows_to_table(
            rows,
            [f"primitive (N={PRIMITIVE_BATCH:,})", "OLD /s", "NEW /s", "speedup"],
        )
    )
    print()
    print(
        "OLD = per-board loop calling game2048.reference_env.move_without_spawn\n"
        "NEW = game2048.fast_env vectorized batch kernels"
    )

    if args.json:
        payload = {
            "metadata": metadata,
            "primitive_batch": PRIMITIVE_BATCH,
            "env_count": ENV_COUNT,
            "env_steps": ENV_STEPS,
            "results": results,
        }
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(f"\nraw results written to {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
