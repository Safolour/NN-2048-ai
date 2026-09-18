"""M1 throughput benchmark for the vectorized 2048 environment.

Usage
-----
Full M1 sweep (no arguments)::

    python benchmarks/benchmark_m1_env.py

Single configuration::

    python benchmarks/benchmark_m1_env.py --num-envs 4096 --steps 40
    python benchmarks/benchmark_m1_env.py --workers 4

What it measures
----------------
* single-worker environment sweep over 256 / 1024 / 4096 / 8192 (and 16384 when
  the machine can hold it);
* worker scaling with 1 / 2 / 4 independent worker processes, each owning its own
  ``Fast2048BatchEnv``;
* primitive throughput: movement, legal mask, random spawn and exact spawn
  enumeration;
* a producer -> consumer harness that pushes whole ``(N, 16)`` board batches
  through a synthetic NumPy consumer and feeds the actions back.

Notes
-----
* Action policy is **uniform random legal action**, driven by a dedicated
  benchmark RNG that is kept separate from the environment's spawn RNG.
* Terminal games are explicitly ``reset_where``-ed so the workload keeps
  producing real moves, merges, spawns and resets.
* Throughput is recorded, never asserted: correctness lives in pytest.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import subprocess
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
    Timing,
    environment_metadata,
    format_bytes,
    format_metadata,
    format_rate,
    format_timing,
    rows_to_table,
    time_callable,
)

#: Environment sizes required by the M1 specification.
SWEEP_SIZES: Sequence[int] = (256, 1024, 4096, 8192, 16384)
#: Sizes that must complete for M1 to pass.
REQUIRED_SIZES: Sequence[int] = (256, 1024, 4096, 8192)
#: Worker counts to test (capped by the logical CPU count at run time).
WORKER_COUNTS: Sequence[int] = (1, 2, 4)

#: Target transitions per measured sweep point.  Throughput is a rate, so the
#: *number* of transitions does not have to grow with the environment count --
#: bounding it keeps the whole sweep reproducible in a few minutes while still
#: giving each size several full batch steps.
TARGET_TRANSITIONS = 200_000
#: Lower bound on steps per measurement, so small sizes still measure a batch.
MIN_STEPS = 8
#: Upper bound, so a huge environment count does not run for hours.
MAX_STEPS = 64
#: Default steps when ``--steps`` is not given.
DEFAULT_STEPS_PER_ENV = 32


def default_steps(num_envs: int, requested: Optional[int] = None) -> int:
    """Steps to use for a given environment count."""
    if requested is not None:
        return requested
    return max(MIN_STEPS, min(MAX_STEPS, TARGET_TRANSITIONS // max(num_envs, 1)))


# --------------------------------------------------------------------------- #
# Action policy: uniform random legal action
# --------------------------------------------------------------------------- #


class LegalRandomPolicy:
    """Uniform random choice among the legal actions of each environment.

    The benchmark RNG is deliberately separate from the environment's spawn RNG,
    so the workload is reproducible without coupling the two streams.
    """

    def __init__(self, num_envs: int, seed: int = BENCHMARK_SEED) -> None:
        self._rng = np.random.default_rng(seed)
        self._num_envs = num_envs
        self._fallback = np.zeros(num_envs, dtype=np.uint8)

    def __call__(self, boards: np.ndarray) -> np.ndarray:
        legal = legal_mask_batch(boards)
        any_legal = legal.any(axis=1)
        # Uniform draw scaled by the legal count, then walk the four actions so
        # the choice is uniform over the legal set.
        draws = self._rng.random(self._num_envs) * np.maximum(
            legal.sum(axis=1), 1
        )
        actions = np.zeros(self._num_envs, dtype=np.uint8)
        cursor = np.zeros(self._num_envs, dtype=np.float64)
        chosen = np.zeros(self._num_envs, dtype=bool)
        for action in range(4):
            take = legal[:, action] & ~chosen
            cursor = np.where(take, cursor, cursor)
            select = take & (cursor >= draws)
            # ``cursor`` counts how many legal actions were seen so far.
            select = take & (draws < np.cumsum(legal, axis=1)[:, action])
            actions[select] = action
            chosen |= select
        actions[~any_legal] = 0
        return actions


# --------------------------------------------------------------------------- #
# Single-worker environment sweep
# --------------------------------------------------------------------------- #


def run_env_sweep(
    num_envs: int,
    steps: int,
    seed: int = BENCHMARK_SEED,
    warmup: int = 1,
    repeats: int = 2,
) -> Dict[str, object]:
    """Step one environment ``steps`` times and report throughput."""
    env = Fast2048BatchEnv(num_envs, seed=seed)
    env.reset(seed=seed)
    policy = LegalRandomPolicy(num_envs, seed=seed)

    # One iteration == one full batch step over all environments.
    total_transitions = steps * num_envs

    def body() -> None:
        for _ in range(steps):
            actions = policy(env._boards)
            result = env.step(actions)
            env.reset_where(result.terminated)

    try:
        timing = time_callable(
            f"env {num_envs}",
            body,
            units=total_transitions,
            unit_name="transitions",
            warmup=warmup,
            repeats=repeats,
        )
    except MemoryError:
        return {
            "num_envs": num_envs,
            "status": "SKIPPED_RESOURCE_LIMIT",
            "reason": "MemoryError while allocating the board batch",
        }

    return {
        "num_envs": num_envs,
        "status": "OK",
        "iterations": steps,
        "total_transitions": total_transitions,
        "wall_time_s": timing.wall_s,
        "median_step_s": timing.median_s,
        "transitions_per_s": timing.median_rate,
        "boards_per_s": timing.median_rate,
        "min_transitions_per_s": timing.min_rate,
        "max_transitions_per_s": timing.max_rate,
        "cpu_utilization": timing.cpu_utilization,
        "peak_rss_bytes": timing.peak_rss_bytes,
    }


# --------------------------------------------------------------------------- #
# Primitive throughput
# --------------------------------------------------------------------------- #


def run_primitive_benchmarks(
    num_envs: int = 4096,
    seed: int = BENCHMARK_SEED,
    warmup: int = 2,
    repeats: int = 3,
) -> List[Dict[str, object]]:
    """Raw throughput of the movement / legal / spawn primitives."""
    rng = np.random.default_rng(seed)
    boards = rng.integers(0, 18, size=(num_envs, 16)).astype(np.uint8)
    boards[rng.random((num_envs, 16)) < 0.45] = 0
    actions = rng.integers(0, 4, size=num_envs).astype(np.uint8)

    results: List[Dict[str, object]] = []

    def add(label: str, timing: Timing, extra: Optional[Dict[str, object]] = None):
        entry = {
            "primitive": label,
            "units": timing.units,
            "unit_name": timing.unit_name,
            "median_s": timing.median_s,
            "median_rate": timing.median_rate,
            "min_rate": timing.min_rate,
            "max_rate": timing.max_rate,
            "cpu_utilization": timing.cpu_utilization,
            "peak_rss_bytes": timing.peak_rss_bytes,
        }
        if extra:
            entry.update(extra)
        results.append(entry)

    add(
        "move_batch",
        time_callable(
            "move",
            lambda: move_batch(boards, actions),
            units=num_envs,
            unit_name="boards",
            warmup=warmup,
            repeats=repeats,
        ),
    )
    add(
        "legal_mask_batch",
        time_callable(
            "legal",
            lambda: legal_mask_batch(boards),
            units=num_envs,
            unit_name="boards",
            warmup=warmup,
            repeats=repeats,
        ),
    )
    add(
        "is_terminal_batch",
        time_callable(
            "terminal",
            lambda: is_terminal_batch(boards),
            units=num_envs,
            unit_name="boards",
            warmup=warmup,
            repeats=repeats,
        ),
    )
    spawn_rng = np.random.default_rng(seed)
    add(
        "spawn_random_batch",
        time_callable(
            "random spawn",
            lambda: spawn_random_batch(boards, spawn_rng),
            units=num_envs,
            unit_name="boards",
            warmup=warmup,
            repeats=repeats,
        ),
    )

    # Exact spawn enumeration, swept over representative empty-cell counts, since
    # the output size is 2 * n per parent and dominates the cost.
    for empty_cells in (2, 4, 8, 12):
        sparse = np.ones((num_envs, 16), dtype=np.uint8)
        for index in range(num_envs):
            cells = rng.choice(16, size=empty_cells, replace=False)
            sparse[index, cells] = 0
        enumeration = enumerate_spawns_batch(sparse)
        outcomes = int(enumeration.states.shape[0])
        timing = time_callable(
            f"exact spawn n={empty_cells}",
            lambda sparse=sparse: enumerate_spawns_batch(sparse),
            units=num_envs,
            unit_name="parents",
            warmup=warmup,
            repeats=repeats,
        )
        add(
            f"enumerate_spawns_batch n={empty_cells}",
            timing,
            extra={
                "empty_cells": empty_cells,
                "outcomes": outcomes,
                "outcomes_per_s": outcomes / timing.median_s
                if timing.median_s > 0
                else 0.0,
            },
        )
        del sparse, enumeration

    return results


# --------------------------------------------------------------------------- #
# Producer -> consumer harness
# --------------------------------------------------------------------------- #


class SyntheticConsumer:
    """A deterministic, allocation-light NumPy consumer of whole board batches.

    It reads the full contiguous ``(N, 16)`` batch, computes a deterministic
    pseudo-score per board from the tile distribution, and picks an action from
    the legal mask.  It exists to exercise the *batch data path* -- it makes no
    attempt to play well, and it never loops over boards.
    """

    def __init__(self, seed: int = BENCHMARK_SEED) -> None:
        self._weights = np.array(
            [0, 1, 3, 7, 15, 31, 63, 127], dtype=np.int64
        )
        self._rng = np.random.default_rng(seed)

    def __call__(self, boards: np.ndarray) -> np.ndarray:
        # Batch reduction: one hot-weight sum per board, fully vectorized.
        clipped = np.minimum(boards, 7).astype(np.int64)
        reduction = (self._weights[clipped]).sum(axis=1)

        # Legal mask, computed in one shot for the whole batch.
        legal = legal_mask_batch(boards)
        counts = np.maximum(legal.sum(axis=1), 1)

        # Deterministic pseudo-score chooses the legal slot, with a small random
        # tie-break from the consumer's own RNG (never the environment's).
        slot = (reduction + self._rng.integers(0, 4, size=boards.shape[0])) % counts
        cumulative = np.cumsum(legal, axis=1)
        actions = np.zeros(boards.shape[0], dtype=np.uint8)
        for action in range(4):
            select = legal[:, action] & (slot == cumulative[:, action] - 1)
            actions[select] = action
        return actions


def run_producer_consumer(
    num_envs: int = 4096,
    steps: int = DEFAULT_STEPS_PER_ENV,
    seed: int = BENCHMARK_SEED,
    warmup: int = 1,
    repeats: int = 2,
) -> Dict[str, object]:
    """FastEnv -> contiguous batch -> consumer -> actions -> FastEnv."""
    env = Fast2048BatchEnv(num_envs, seed=seed)
    env.reset(seed=seed)
    consumer = SyntheticConsumer(seed=seed)

    assembly_seconds = 0.0
    consumer_seconds = 0.0
    stepper_seconds = 0.0
    total_transitions = steps * num_envs

    def body() -> None:
        nonlocal assembly_seconds, consumer_seconds, stepper_seconds
        assembly_seconds = 0.0
        consumer_seconds = 0.0
        stepper_seconds = 0.0
        for _ in range(steps):
            start = time.perf_counter()
            boards = env.copy_boards()
            assembly_seconds += time.perf_counter() - start

            start = time.perf_counter()
            actions = consumer(boards)
            consumer_seconds += time.perf_counter() - start

            start = time.perf_counter()
            result = env.step(actions)
            env.reset_where(result.terminated)
            stepper_seconds += time.perf_counter() - start

    timing = time_callable(
        f"producer/consumer {num_envs}",
        body,
        units=total_transitions,
        unit_name="transitions",
        warmup=warmup,
        repeats=repeats,
    )
    measured = assembly_seconds + consumer_seconds + stepper_seconds
    return {
        "num_envs": num_envs,
        "steps": steps,
        "total_transitions": total_transitions,
        "median_s": timing.median_s,
        "transitions_per_s": timing.median_rate,
        "boards_per_s": timing.median_rate,
        "min_transitions_per_s": timing.min_rate,
        "max_transitions_per_s": timing.max_rate,
        "batch_assembly_s": assembly_seconds,
        "consumer_s": consumer_seconds,
        "environment_step_s": stepper_seconds,
        "batch_assembly_fraction": assembly_seconds / measured if measured else 0.0,
        "consumer_fraction": consumer_seconds / measured if measured else 0.0,
        "environment_step_fraction": stepper_seconds / measured if measured else 0.0,
        "cpu_utilization": timing.cpu_utilization,
        "peak_rss_bytes": timing.peak_rss_bytes,
    }


# --------------------------------------------------------------------------- #
# Worker scaling
# --------------------------------------------------------------------------- #


def run_worker_scaling(
    workers: Sequence[int],
    envs_per_worker: int = 4096,
    steps: int = DEFAULT_STEPS_PER_ENV,
    seed: int = BENCHMARK_SEED,
    workdir: Optional[Path] = None,
) -> List[Dict[str, object]]:
    """Run ``k`` independent worker processes, each owning its own batch env.

    Workers are launched with ``subprocess`` and report through a JSON file
    instead of ``multiprocessing``: on Windows the ``spawn`` start method needs an
    anonymous bootstrap pipe, which some sandboxes forbid, while ``subprocess``
    with ``stdio='ignore'`` needs none.  Each worker is a full, independent
    ``Fast2048BatchEnv``; the ``k == 1`` case is the single-process baseline the
    scaling efficiency is measured against.
    """
    directory = Path(workdir) if workdir is not None else Path.cwd()
    script = Path(__file__).resolve()
    results: List[Dict[str, object]] = []

    for worker_count in workers:
        paths = [
            directory / f".m1_worker_{worker_count}_{index}.json"
            for index in range(worker_count)
        ]
        for path in paths:
            path.unlink(missing_ok=True)

        commands = [
            [
                sys.executable,
                str(script),
                "--worker-child",
                str(index),
                "--num-envs",
                str(envs_per_worker),
                "--steps",
                str(steps),
                "--seed",
                str(seed),
                "--worker-result",
                str(paths[index]),
            ]
            for index in range(worker_count)
        ]
        started = time.perf_counter()
        children = [
            subprocess.Popen(
                command,
                cwd=str(directory),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            for command in commands
        ]
        for child in children:
            child.wait(timeout=3600)
        wall = time.perf_counter() - started

        payloads: List[Dict[str, object]] = []
        problems: List[str] = []
        for index, path in enumerate(paths):
            if not path.exists():
                problems.append(
                    f"worker {index} produced no result file "
                    f"(exit code {children[index].returncode})"
                )
                continue
            try:
                payloads.append(json.loads(path.read_text(encoding="utf-8")))
            except Exception as error:  # pragma: no cover - corrupt file
                problems.append(f"worker {index} result unreadable: {error!r}")
            finally:
                path.unlink(missing_ok=True)

        problems.extend(
            str(item.get("error")) for item in payloads if not item.get("ok")
        )
        if problems or len(payloads) != worker_count:
            results.append(
                {
                    "workers": worker_count,
                    "status": "FAILED",
                    "reason": "; ".join(problems) or "incomplete worker results",
                }
            )
            continue

        slowest = max(float(item["median_step_s"]) for item in payloads)
        total_transitions = worker_count * envs_per_worker * steps
        results.append(
            {
                "workers": worker_count,
                "status": "OK",
                "envs_per_worker": envs_per_worker,
                "steps": steps,
                "total_transitions": total_transitions,
                # The slowest worker bounds one parallel round's wall clock.
                "parallel_step_s": slowest,
                "wall_s": wall,
                "transitions_per_s": total_transitions / wall if wall > 0 else 0.0,
            }
        )
    return results


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def print_sweep(results: Sequence[Dict[str, object]]) -> None:
    rows = []
    for entry in results:
        if entry["status"] != "OK":
            rows.append(
                [
                    str(entry["num_envs"]),
                    entry["status"],
                    "-",
                    "-",
                    "-",
                    "-",
                    "-",
                ]
            )
            continue
        rows.append(
            [
                f"{entry['num_envs']:,}",
                f"{entry['iterations']}",
                f"{entry['total_transitions']:,}",
                f"{entry['wall_time_s']:.3f}",
                f"{format_rate(entry['transitions_per_s'])}/s",
                f"{entry['cpu_utilization']:.2f}x",
                format_bytes(entry["peak_rss_bytes"]),
            ]
        )
    print()
    print("Single-worker environment sweep (uniform random legal actions)")
    print(
        rows_to_table(
            rows,
            [
                "num_envs",
                "iterations",
                "transitions",
                "wall_s",
                "throughput",
                "CPU",
                "peak RAM",
            ],
        )
    )


def print_primitives(results: Sequence[Dict[str, object]]) -> None:
    rows = []
    for entry in results:
        extra = ""
        if "empty_cells" in entry:
            extra = f"{entry['outcomes_per_s'] / 1e6:,.2f} M outcomes/s"
        rows.append(
            [
                entry["primitive"],
                f"{entry['median_s'] * 1e6:,.1f}",
                f"{format_rate(entry['median_rate'])}/s",
                f"{entry['cpu_utilization']:.2f}x",
                extra,
            ]
        )
    print()
    print("Primitive throughput")
    print(
        rows_to_table(
            rows,
            ["primitive", "median us", "rate", "CPU", "note"],
        )
    )


def print_worker_scaling(results: Sequence[Dict[str, object]]) -> None:
    rows = []
    baseline = None
    for entry in results:
        if entry["status"] != "OK":
            rows.append([str(entry["workers"]), entry["status"], "-", "-", "-"])
            continue
        if entry["workers"] == 1:
            baseline = entry["transitions_per_s"]
        efficiency = (
            entry["transitions_per_s"] / (entry["workers"] * baseline)
            if baseline
            else float("nan")
        )
        rows.append(
            [
                str(entry["workers"]),
                f"{entry['envs_per_worker']:,}",
                f"{format_rate(entry['transitions_per_s'])}/s",
                f"{efficiency:.3f}" if baseline else "-",
                f"{entry['wall_s']:.3f}",
            ]
        )
    print()
    print("Worker scaling (independent processes, own Fast2048BatchEnv each)")
    print(
        rows_to_table(
            rows,
            ["workers", "envs/worker", "throughput", "efficiency", "wall_s"],
        )
    )


def print_producer_consumer(result: Dict[str, object]) -> None:
    print()
    print("Producer -> consumer throughput")
    print(
        rows_to_table(
            [
                ["num_envs", f"{result['num_envs']:,}"],
                ["steps", f"{result['steps']}"],
                ["total transitions", f"{result['total_transitions']:,}"],
                ["throughput", f"{format_rate(result['transitions_per_s'])}/s"],
                ["CPU", f"{result['cpu_utilization']:.2f}x"],
                ["peak RAM", format_bytes(result["peak_rss_bytes"])],
                ["batch assembly", f"{result['batch_assembly_fraction'] * 100:.1f}%"],
                ["consumer", f"{result['consumer_fraction'] * 100:.1f}%"],
                ["environment step", f"{result['environment_step_fraction'] * 100:.1f}%"],
            ],
            ["metric", "value"],
        )
    )


def _run_worker_child(args) -> int:
    """One worker process: measure a single batch environment and report to file."""
    payload: Dict[str, object] = {"worker": args.worker_child}
    try:
        result = run_env_sweep(
            args.num_envs,
            args.steps,
            seed=args.seed + args.worker_child,
            warmup=1,
            repeats=2,
        )
        payload["ok"] = True
        payload["status"] = result["status"]
        payload["median_step_s"] = result["median_step_s"]
        payload["transitions_per_s"] = result["transitions_per_s"]
        payload["num_envs"] = result["num_envs"]
    except Exception as error:  # pragma: no cover - child process
        payload["ok"] = False
        payload["error"] = repr(error)
    if args.worker_result is not None:
        Path(args.worker_result).write_text(
            json.dumps(payload, default=str), encoding="utf-8"
        )
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="M1 throughput benchmark for the vectorized 2048 environment"
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=None,
        help="run a single environment size instead of the full sweep",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help=(
            "environment steps per benchmark iteration; by default each size gets "
            "a bounded workload (see TARGET_TRANSITIONS)"
        ),
    )
    parser.add_argument("--seed", type=int, default=BENCHMARK_SEED)
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="run only the worker scaling test with this worker count",
    )
    parser.add_argument(
        "--envs-per-worker",
        type=int,
        default=4096,
        help="boards per worker process in the scaling test",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="also write the raw results to this JSON file",
    )
    parser.add_argument(
        "--worker-child",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--worker-result",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)

    if args.worker_child is not None:
        return _run_worker_child(args)

    metadata = environment_metadata()
    print("=" * 78)
    print("M1 Fast2048BatchEnv benchmark")
    print("=" * 78)
    print(format_metadata(metadata))

    payload: Dict[str, object] = {"metadata": metadata}

    if args.workers is not None:
        workers = [args.workers]
        steps = default_steps(args.envs_per_worker, args.steps)
        scaling = run_worker_scaling(
            workers,
            envs_per_worker=args.envs_per_worker,
            steps=steps,
            seed=args.seed,
        )
        print_worker_scaling(scaling)
        payload["worker_scaling"] = scaling
    elif args.num_envs is not None:
        steps = default_steps(args.num_envs, args.steps)
        single = run_env_sweep(args.num_envs, steps, seed=args.seed)
        print_sweep([single])
        payload["env_sweep"] = [single]
    else:
        sweep: List[Dict[str, object]] = []
        for size in SWEEP_SIZES:
            steps = default_steps(size, args.steps)
            print(f"  sweeping {size:,} envs x {steps} steps ...", flush=True)
            entry = run_env_sweep(size, steps, seed=args.seed)
            entry["steps"] = steps
            sweep.append(entry)
            if entry["status"] != "OK":
                print(
                    f"  {size:,} envs -> {entry['status']}: {entry.get('reason', '')}",
                    flush=True,
                )
        print_sweep(sweep)
        payload["env_sweep"] = sweep

        primitives = run_primitive_benchmarks(seed=args.seed)
        print_primitives(primitives)
        payload["primitives"] = primitives

        consumer_steps = default_steps(4096, args.steps)
        consumer = run_producer_consumer(steps=consumer_steps, seed=args.seed)
        print_producer_consumer(consumer)
        payload["producer_consumer"] = consumer

        logical = int(metadata["logical_cores"])
        worker_counts = [count for count in WORKER_COUNTS if count <= logical]
        if not worker_counts:
            worker_counts = [1]
        scaling = run_worker_scaling(
            worker_counts,
            envs_per_worker=args.envs_per_worker,
            steps=default_steps(args.envs_per_worker, args.steps),
            seed=args.seed,
        )
        print_worker_scaling(scaling)
        payload["worker_scaling"] = scaling

    if args.json is not None:
        args.json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(f"\nraw results written to {args.json}")

    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
