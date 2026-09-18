"""M1 wall-clock profile of the vectorized 2048 environment.

Usage
-----
::

    python benchmarks/profile_m1_env.py
    python benchmarks/profile_m1_env.py --num-envs 8192 --iterations 40

The tool breaks one full benchmark iteration (``steps`` batch steps of the
producer -> consumer loop) into named stages, measured with
``time.perf_counter()`` around explicit boundaries, and reports both the absolute
time and the percentage of measured wall clock.  A ``cProfile`` summary of the
same workload is printed afterwards as supporting detail -- the wall-clock
breakdown is the primary result, and the percentages are normalised so they sum
to approximately 100%.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from game2048.fast_env import (  # noqa: E402
    Fast2048BatchEnv,
    enumerate_spawns_batch,
    is_terminal_batch,
    legal_mask_batch,
    move_batch,
    spawn_random_batch,
)

from _utils import (  # noqa: E402
    BENCHMARK_SEED,
    environment_metadata,
    format_bytes,
    format_metadata,
    format_rate,
    rows_to_table,
)

#: Stage names reported in the wall-clock breakdown, in report order.
STAGES: Sequence[str] = (
    "Environment stepping (move + spawn + score)",
    "Legal/action preparation (mask + terminal)",
    "Spawn (random)",
    "Exact spawn enumeration",
    "Batch assembly (contiguous copy)",
    "Memory copy",
    "Synchronization / worker overhead",
    "Other",
)


class Stopwatch:
    """Accumulate per-stage wall-clock time with ``time.perf_counter``."""

    def __init__(self, stages: Sequence[str]) -> None:
        self._stages = list(stages)
        self._totals: Dict[str, float] = {stage: 0.0 for stage in self._stages}
        self._counts: Dict[str, int] = {stage: 0 for stage in self._stages}
        self._starts: Dict[str, float] = {}

    def start(self, stage: str) -> None:
        self._starts[stage] = time.perf_counter()

    def stop(self, stage: str) -> None:
        started = self._starts.pop(stage, None)
        if started is None:
            return
        self._totals[stage] += time.perf_counter() - started
        self._counts[stage] += 1

    def total(self, stage: str) -> float:
        return self._totals[stage]

    def count(self, stage: str) -> int:
        return self._counts[stage]

    def measured_total(self) -> float:
        return sum(self._totals.values())

    def add(self, stage: str, seconds: float) -> None:
        self._totals[stage] += seconds


def profile_iteration(
    num_envs: int,
    steps: int,
    seed: int = BENCHMARK_SEED,
    enumeration_envs: Optional[int] = None,
) -> Dict[str, object]:
    """Run one profiled iteration and return the per-stage breakdown."""
    watch = Stopwatch(STAGES)
    env = Fast2048BatchEnv(num_envs, seed=seed)
    env.reset(seed=seed)

    consumer_rng = np.random.default_rng(seed)
    weights = np.array([0, 1, 3, 7, 15, 31, 63, 127], dtype=np.int64)

    enumeration_envs = enumeration_envs or min(num_envs, 1024)
    enumeration_rng = np.random.default_rng(seed)
    enumeration_boards = enumeration_rng.integers(1, 18, size=(enumeration_envs, 16))
    enumeration_boards[enumeration_rng.random((enumeration_envs, 16)) < 0.5] = 0
    enumeration_boards = np.ascontiguousarray(enumeration_boards.astype(np.uint8))
    spawn_boards = enumeration_boards.copy()
    spawn_rng = np.random.default_rng(seed)

    boards_buffer = np.empty((num_envs, 16), dtype=np.uint8)

    wall_start = time.perf_counter()
    for _ in range(steps):
        # -- batch assembly: hand the consumer a contiguous board snapshot ------
        watch.start("Batch assembly (contiguous copy)")
        boards_buffer[:] = env._boards
        watch.stop("Batch assembly (contiguous copy)")

        # -- consumer: legal/action preparation ---------------------------------
        watch.start("Legal/action preparation (mask + terminal)")
        clipped = np.minimum(boards_buffer, 7).astype(np.int64)
        reduction = weights[clipped].sum(axis=1)
        legal = legal_mask_batch(boards_buffer)
        counts = np.maximum(legal.sum(axis=1), 1)
        slot = (reduction + consumer_rng.integers(0, num_envs)) % counts
        cumulative = np.cumsum(legal, axis=1)
        actions = np.zeros(num_envs, dtype=np.uint8)
        for action in range(4):
            select = legal[:, action] & (slot == cumulative[:, action] - 1)
            actions[select] = action
        watch.stop("Legal/action preparation (mask + terminal)")

        # -- environment stepping: move + spawn + score -------------------------
        watch.start("Environment stepping (move + spawn + score)")
        move = move_batch(boards_buffer, actions)
        moved = move.moved
        watch.stop("Environment stepping (move + spawn + score)")

        watch.start("Spawn (random)")
        rows = np.flatnonzero(moved)
        if rows.size:
            spawn_random_batch(move.afterstates[rows], spawn_rng)
        watch.stop("Spawn (random)")

        watch.start("Environment stepping (move + spawn + score)")
        result = env.step(actions)
        env.reset_where(result.terminated)
        watch.stop("Environment stepping (move + spawn + score)")

        # -- exact spawn enumeration --------------------------------------------
        watch.start("Exact spawn enumeration")
        enumerate_spawns_batch(enumeration_boards)
        watch.stop("Exact spawn enumeration")

        # -- memory copy --------------------------------------------------------
        watch.start("Memory copy")
        np.copyto(boards_buffer, env._boards)
        watch.stop("Memory copy")
    wall_total = time.perf_counter() - wall_start

    measured = watch.measured_total()
    other = max(wall_total - measured, 0.0)
    watch.add("Other", other)
    watch.add("Synchronization / worker overhead", 0.0)

    transitions = steps * num_envs
    stages: List[Dict[str, object]] = []
    grand_total = watch.measured_total()
    for stage in STAGES:
        seconds = watch.total(stage)
        stages.append(
            {
                "stage": stage,
                "seconds": seconds,
                "percent": (seconds / grand_total * 100.0) if grand_total else 0.0,
                "calls": watch.count(stage),
            }
        )
    return {
        "num_envs": num_envs,
        "steps": steps,
        "transitions": transitions,
        "wall_seconds": wall_total,
        "measured_seconds": grand_total,
        "stages": stages,
        "transitions_per_s": transitions / wall_total if wall_total else 0.0,
    }


def profile_with_cprofile(num_envs: int, steps: int, seed: int) -> str:
    """Return the top ``cProfile`` entries for the same workload."""
    profiler = cProfile.Profile()
    profiler.enable()
    profile_iteration(num_envs, steps, seed=seed)
    profiler.disable()
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats("cumulative")
    stats.print_stats(18)
    return stream.getvalue()


def report(result: Dict[str, object]) -> None:
    rows = []
    for entry in result["stages"]:
        rows.append(
            [
                entry["stage"],
                f"{entry['seconds'] * 1e3:10.3f}",
                f"{entry['percent']:6.2f}%",
                f"{entry['calls']:,}",
            ]
        )
    total_percent = sum(entry["percent"] for entry in result["stages"])
    rows.append(
        [
            "TOTAL",
            f"{result['measured_seconds'] * 1e3:10.3f}",
            f"{total_percent:6.2f}%",
            "",
        ]
    )
    print()
    print(
        f"Wall-clock profile  (num_envs={result['num_envs']:,}, "
        f"steps={result['steps']}, transitions={result['transitions']:,})"
    )
    print(rows_to_table(rows, ["stage", "ms", "share", "calls"]))
    print()
    print(
        f"measured wall clock : {result['measured_seconds']:.3f} s "
        f"(outer wall {result['wall_seconds']:.3f} s)"
    )
    print(
        f"throughput          : "
        f"{format_rate(result['transitions_per_s'])}transitions/s"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="M1 wall-clock profile of the vectorized 2048 environment"
    )
    parser.add_argument("--num-envs", type=int, default=8192)
    parser.add_argument("--iterations", type=int, default=20, help="batch steps")
    parser.add_argument("--seed", type=int, default=BENCHMARK_SEED)
    parser.add_argument(
        "--cprofile",
        action="store_true",
        help="also print a cProfile summary of the same workload",
    )
    args = parser.parse_args(argv)

    metadata = environment_metadata()
    print("=" * 78)
    print("M1 Fast2048BatchEnv wall-clock profile")
    print("=" * 78)
    print(format_metadata(metadata))
    print(f"peak RAM before run : {format_bytes(0)} (see benchmark for RSS)")

    # Warm up so the first measured iteration is not dominated by allocation.
    profile_iteration(min(args.num_envs, 512), 2, seed=args.seed)

    result = profile_iteration(args.num_envs, args.iterations, seed=args.seed)
    report(result)

    if args.cprofile:
        print()
        print("cProfile (supporting detail; the wall-clock table above is primary)")
        print(profile_with_cprofile(min(args.num_envs, 2048), 5, args.seed))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
