from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))

from _m2_utils import NvidiaSmiSampler, write_json
from _utils import ResourceSampler
from benchmark_m2_pipeline_tuning import (
    ThreadedRunner,
    build_model,
    gpu_policy,
    make_action_generator,
)
from game2048.fast_env import (
    Fast2048BatchEnv,
    is_terminal_batch as numpy_terminal,
    legal_mask_batch as numpy_legal,
    move_batch as numpy_move,
)
from game2048.m2_fast_backend import (
    backend_info,
    legal_mask_batch as cpp_legal,
    move_selected_batch as cpp_move,
)
from game2048.m2_rollout_env import M2RolloutBatchEnv

SEED = 20260919
SIZES = (1024, 4096, 8192, 16384)
OUT = ROOT / "m2_fast_backend_benchmark.json"
CLOSED_LOOP = ROOT / "m2_closed_loop_benchmark.json"
DEVICE = torch.device("cuda")


def bench_call(fn, units: int) -> dict:
    for _ in range(20):
        fn()
    start = time.perf_counter()
    iterations = 0
    while iterations < 100 or time.perf_counter() - start < 2.0:
        fn()
        iterations += 1
    elapsed = time.perf_counter() - start
    return {
        "iterations": iterations,
        "elapsed_s": elapsed,
        "us_per_iteration": elapsed * 1e6 / iterations,
        "units_per_s": units * iterations / elapsed,
    }


def prepare_boards(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    env = Fast2048BatchEnv(n, seed=seed)
    env.reset(seed=seed)
    rng = np.random.default_rng(seed + 1)
    for _ in range(48):
        legal = numpy_legal(env.boards)
        dead = ~legal.any(axis=1)
        if dead.any():
            env.reset_where(dead)
            legal = numpy_legal(env.boards)
        priority = rng.random(legal.shape)
        priority[~legal] = -1.0
        actions = priority.argmax(axis=1).astype(np.uint8)
        result = env.step(actions)
        if result.terminated.any():
            env.reset_where(result.terminated)
    boards = env.copy_boards()
    legal = numpy_legal(boards)
    actions = np.argmax(legal, axis=1).astype(np.uint8)
    return boards, actions


def bench_step(env_cls, n: int, seed: int) -> dict:
    env = env_cls(n, seed=seed)
    env.reset(seed=seed)
    rng = np.random.default_rng(seed + 2)

    def one(timed: bool):
        if isinstance(env, M2RolloutBatchEnv):
            legal = env.current_legal
        else:
            legal = numpy_legal(env.boards)
        dead = ~legal.any(axis=1)
        if dead.any():
            env.reset_where(dead)
            legal = env.current_legal if isinstance(env, M2RolloutBatchEnv) else numpy_legal(env.boards)
        priority = rng.random(legal.shape)
        priority[~legal] = -1.0
        actions = priority.argmax(axis=1).astype(np.uint8)
        if timed:
            start = time.perf_counter()
            result = env.step(actions)
            elapsed = time.perf_counter() - start
        else:
            result = env.step(actions)
            elapsed = 0.0
        if result.terminated.any():
            env.reset_where(result.terminated)
        return elapsed

    for _ in range(20):
        one(False)
    measured = 0.0
    iterations = 0
    while iterations < 100 or measured < 2.0:
        measured += one(True)
        iterations += 1
    return {
        "iterations": iterations,
        "elapsed_s": measured,
        "us_per_iteration": measured * 1e6 / iterations,
        "transitions_per_s": n * iterations / measured,
    }


def bench_fused_decision(env_cls, n: int, seed: int) -> dict:
    env = env_cls(n, seed=seed)
    env.reset(seed=seed)

    def body():
        legal = env.current_legal if isinstance(env, M2RolloutBatchEnv) else numpy_legal(env.boards)
        dead = ~legal.any(axis=1)
        if dead.any():
            env.reset_where(dead)
            legal = env.current_legal if isinstance(env, M2RolloutBatchEnv) else numpy_legal(env.boards)
        actions = np.argmax(legal, axis=1).astype(np.uint8)
        result = env.step(actions)
        if result.terminated.any():
            env.reset_where(result.terminated)

    return bench_call(body, n)


class CppRolloutRunner(ThreadedRunner):
    def __init__(self, total: int, workers: int) -> None:
        super().__init__(total, workers)
        self.envs = [
            M2RolloutBatchEnv(self.size, seed=SEED + i * 100003)
            for i in range(workers)
        ]
        for i, env in enumerate(self.envs):
            env.reset(seed=SEED + i * 100003)

    def legal_one(self, idx: int) -> None:
        sl = self.slices[idx]
        np.copyto(self.legal_np[sl], self.envs[idx].current_legal)


def one_iteration(runner, model, generator):
    runner.boards()
    runner.legal()
    gpu_policy(runner, model, generator)
    runner.step()
    runner.reset()


def closed_loop(runner_cls, env_count: int, workers: int = 2) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    runner = runner_cls(env_count, workers)
    model = build_model()
    generator = make_action_generator()
    try:
        for _ in range(20):
            one_iteration(runner, model, generator)
        torch.cuda.synchronize()
        iterations = 0
        with ResourceSampler() as cpu_sampler, NvidiaSmiSampler() as gpu_sampler:
            start = time.perf_counter()
            while iterations < 100 or time.perf_counter() - start < 2.0:
                one_iteration(runner, model, generator)
                iterations += 1
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
        return {
            "workers": workers,
            "env_count": env_count,
            "iterations": iterations,
            "elapsed_s": elapsed,
            "decisions_per_s": env_count * iterations / elapsed,
            "cpu_utilization_cores": cpu_sampler.cpu_utilization,
            "gpu": gpu_sampler.summary(),
            "peak_vram_allocated": torch.cuda.max_memory_allocated(),
            "peak_vram_reserved": torch.cuda.max_memory_reserved(),
        }
    finally:
        runner.close()


def profile_cpp(env_count: int, workers: int = 2, iterations: int = 30) -> dict:
    runner = CppRolloutRunner(env_count, workers)
    model = build_model()
    generator = make_action_generator()
    totals = {
        "environment_step": 0.0,
        "legal_cache_copy": 0.0,
        "cpu_batch_preparation": 0.0,
        "h2d": 0.0,
        "nn_inference": 0.0,
        "action_selection": 0.0,
        "d2h": 0.0,
        "reset_where": 0.0,
    }
    transferred = 0
    try:
        for _ in range(20):
            one_iteration(runner, model, generator)
        for _ in range(iterations):
            t = time.perf_counter()
            runner.boards()
            totals["cpu_batch_preparation"] += time.perf_counter() - t

            t = time.perf_counter()
            runner.legal()
            totals["legal_cache_copy"] += time.perf_counter() - t

            torch.cuda.synchronize()
            t = time.perf_counter()
            runner.board_gpu.copy_(runner.board_cpu, non_blocking=True)
            runner.legal_gpu.copy_(runner.legal_cpu, non_blocking=True)
            torch.cuda.synchronize()
            totals["h2d"] += time.perf_counter() - t
            transferred += runner.board_cpu.numel() + runner.legal_cpu.numel()

            torch.cuda.synchronize()
            t = time.perf_counter()
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                q = model(runner.board_gpu)
            torch.cuda.synchronize()
            totals["nn_inference"] += time.perf_counter() - t

            torch.cuda.synchronize()
            t = time.perf_counter()
            from game2048.m2_policy import select_greedy_actions
            actions = select_greedy_actions(q.float(), runner.legal_gpu, generator=generator)
            torch.cuda.synchronize()
            totals["action_selection"] += time.perf_counter() - t

            torch.cuda.synchronize()
            t = time.perf_counter()
            runner.action_cpu.copy_(actions.to(torch.uint8), non_blocking=True)
            torch.cuda.synchronize()
            totals["d2h"] += time.perf_counter() - t

            t = time.perf_counter()
            runner.step()
            totals["environment_step"] += time.perf_counter() - t

            t = time.perf_counter()
            runner.reset()
            totals["reset_where"] += time.perf_counter() - t
    finally:
        runner.close()

    total = sum(totals.values())
    percent = {k: v / total * 100.0 for k, v in totals.items()}
    feed = (
        totals["environment_step"]
        + totals["legal_cache_copy"]
        + totals["cpu_batch_preparation"]
        + totals["h2d"]
    ) / total * 100.0
    stall = (
        totals["cpu_batch_preparation"] + totals["h2d"] + totals["d2h"]
    ) / total * 100.0
    return {
        "workers": workers,
        "env_count": env_count,
        "iterations": iterations,
        "stage_seconds": totals,
        "stage_percent": percent,
        "cpu_h2d_feed_percent": feed,
        "pipeline_stall_percent": stall,
        "h2d_effective_bytes_per_s": transferred / totals["h2d"],
    }


def main() -> int:
    payload = {
        "backend": backend_info(),
        "primitive": {},
        "closed_loop": [],
    }

    for n in SIZES:
        print(f"primitive {n}", flush=True)
        boards, actions = prepare_boards(n, SEED + n)
        primitive = {
            "numpy_legal": bench_call(lambda: numpy_legal(boards), n),
            "cpp_legal": bench_call(lambda: cpp_legal(boards), n),
            "numpy_selected_move": bench_call(lambda: numpy_move(boards, actions), n),
            "cpp_selected_move": bench_call(lambda: cpp_move(boards, actions), n),
            "numpy_terminal": bench_call(lambda: numpy_terminal(boards), n),
            "cpp_terminal": bench_call(lambda: ~cpp_legal(boards).any(axis=1), n),
            "numpy_step": bench_step(Fast2048BatchEnv, n, SEED + n + 10),
            "cpp_rollout_step": bench_step(M2RolloutBatchEnv, n, SEED + n + 10),
            "numpy_fused_decision": bench_fused_decision(Fast2048BatchEnv, n, SEED + n + 20),
            "cpp_fused_next_legal": bench_fused_decision(M2RolloutBatchEnv, n, SEED + n + 20),
        }
        payload["primitive"][str(n)] = primitive

    for n in SIZES:
        print(f"closed-loop {n}", flush=True)
        baseline = closed_loop(ThreadedRunner, n)
        cpp = closed_loop(CppRolloutRunner, n)
        payload["closed_loop"].append(
            {
                "env_count": n,
                "numpy_baseline": baseline,
                "cpp_backend": cpp,
                "improvement_percent": (cpp["decisions_per_s"] / baseline["decisions_per_s"] - 1.0) * 100.0,
            }
        )

    best = max(
        (entry["cpp_backend"] for entry in payload["closed_loop"]),
        key=lambda item: item["decisions_per_s"],
    )
    profile = profile_cpp(best["env_count"])
    old = json.loads(CLOSED_LOOP.read_text(encoding="utf-8"))
    model_only = old["models"]["ResidualMLP2048"]["config"]["model_only_states_per_s"]
    gpu_mean = best["gpu"].get("gpu_util_mean")
    ratio = best["decisions_per_s"] / model_only
    gate_a = bool(gpu_mean is not None and gpu_mean < 60.0 and profile["cpu_h2d_feed_percent"] >= 40.0)
    gate_b = bool(profile["pipeline_stall_percent"] >= 20.0)
    gate_c = bool(ratio < 0.70 and profile["cpu_h2d_feed_percent"] >= 40.0)
    historical = old["models"]["ResidualMLP2048"]["safe_pipeline_optimization"]["selected_after"]["decisions_per_s"]

    payload["best_cpp_closed_loop"] = best
    payload["profile_best_cpp"] = profile
    payload["historical_tuned_baseline_decisions_per_s"] = historical
    payload["improvement_vs_historical_tuned_percent"] = (
        best["decisions_per_s"] / historical - 1.0
    ) * 100.0
    payload["model_only_states_per_s"] = model_only
    payload["closed_loop_model_only_ratio"] = ratio
    payload["starvation_tests"] = {"A": gate_a, "B": gate_b, "C": gate_c}
    payload["starvation_pass"] = not (gate_a or gate_b or gate_c)

    write_json(OUT, payload)
    print(json.dumps({
        "best_cpp_closed_loop": best,
        "improvement_vs_historical_tuned_percent": payload["improvement_vs_historical_tuned_percent"],
        "profile": profile,
        "starvation_tests": payload["starvation_tests"],
        "starvation_pass": payload["starvation_pass"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
