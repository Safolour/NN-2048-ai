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
import game2048.fast_env as fast_env
from benchmark_m2_pipeline_tuning import (
    ThreadedRunner,
    build_model,
    make_action_generator,
    gpu_policy,
)

SEED = 20260918
DEVICE = torch.device("cuda")
CLOSED_LOOP = ROOT / "reports/m2/m2_closed_loop_benchmark.json"
OUT = ROOT / "m2_python_final_ab.json"


class CachedNextLegalRunner(ThreadedRunner):
    """P4 prototype: reuse post-step legal mask on the next policy iteration."""

    def __init__(self, total: int, workers: int) -> None:
        super().__init__(total, workers)
        self.cached_legals: list[np.ndarray | None] = [None] * workers
        self._board_owner = {id(env._boards): i for i, env in enumerate(self.envs)}
        self._orig_terminal = fast_env.is_terminal_batch

        def terminal_and_cache(boards: np.ndarray) -> np.ndarray:
            legal = fast_env.legal_mask_batch(boards)
            owner = self._board_owner.get(id(boards))
            if owner is not None:
                self.cached_legals[owner] = legal
            return ~legal.any(axis=1)

        self._terminal_and_cache = terminal_and_cache
        fast_env.is_terminal_batch = terminal_and_cache

    def close(self) -> None:
        fast_env.is_terminal_batch = self._orig_terminal
        super().close()

    def legal_one(self, idx: int) -> None:
        sl = self.slices[idx]
        cached = self.cached_legals[idx]
        if cached is None:
            np.copyto(self.legal_np[sl], fast_env.legal_mask_batch(self.board_np[sl]))
        else:
            np.copyto(self.legal_np[sl], cached)

    def reset_one(self, idx: int) -> None:
        terminated = self.terminated[idx]
        if terminated is not None and terminated.any():
            self.envs[idx].reset_where(terminated)
            self.cached_legals[idx] = None


def one_iteration(runner, model, generator):
    runner.boards()
    runner.legal()
    gpu_policy(runner, model, generator)
    runner.step()
    runner.reset()


def throughput(runner_cls, *, workers: int = 2, env_count: int = 4096) -> dict:
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


def profile_cached(*, workers: int = 2, env_count: int = 4096, iterations: int = 30) -> dict:
    runner = CachedNextLegalRunner(env_count, workers)
    model = build_model()
    generator = make_action_generator()
    totals = {
        "environment_step": 0.0,
        "legal_mask_or_cache": 0.0,
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
            totals["legal_mask_or_cache"] += time.perf_counter() - t

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
            actions = fast_select(q.float(), runner.legal_gpu, generator)
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
        + totals["legal_mask_or_cache"]
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


def fast_select(q, legal, generator):
    from game2048.m2_policy import select_greedy_actions
    return select_greedy_actions(q, legal, generator=generator)


def correctness_probe(env_count: int = 256, steps: int = 200) -> dict:
    """Compare prototype trajectory to frozen Fast2048BatchEnv under identical actions."""
    baseline = fast_env.Fast2048BatchEnv(env_count, seed=SEED + 9001)
    cached = CachedNextLegalRunner(env_count, 1)
    # Replace the runner-created env with one seeded identically to baseline.
    cached.close()
    cached = CachedNextLegalRunner(env_count, 1)
    cached.envs[0] = fast_env.Fast2048BatchEnv(env_count, seed=SEED + 9001)
    cached.envs[0].reset(seed=SEED + 9001)
    cached._board_owner = {id(cached.envs[0]._boards): 0}
    baseline.reset(seed=SEED + 9001)
    rng = np.random.default_rng(SEED + 9002)
    try:
        for step in range(steps):
            legal = fast_env.legal_mask_batch(baseline.boards)
            dead = ~legal.any(axis=1)
            if dead.any():
                baseline.reset_where(dead)
                cached.envs[0].reset_where(dead)
                cached.cached_legals[0] = None
                legal = fast_env.legal_mask_batch(baseline.boards)
            priority = rng.random(legal.shape)
            priority[~legal] = -1.0
            actions = priority.argmax(axis=1).astype(np.uint8)
            a = baseline.step(actions)
            b = cached.envs[0].step(actions)
            fields = ("states", "afterstates", "rewards", "legal", "terminated", "spawn_indices", "spawn_exponents")
            for field in fields:
                if not np.array_equal(getattr(a, field), getattr(b, field)):
                    raise AssertionError(f"cached-next-legal prototype diverged at step {step}, field {field}")
            if not np.array_equal(b.terminated, ~cached.cached_legals[0].any(axis=1)):
                raise AssertionError("cached legal does not reproduce terminal verdict")
        return {"env_count": env_count, "steps": steps, "pass": True}
    finally:
        cached.close()


def main() -> int:
    data = json.loads(CLOSED_LOOP.read_text(encoding="utf-8"))
    mlp = data["models"]["ResidualMLP2048"]
    model_only = mlp["config"]["model_only_states_per_s"]
    historical = mlp["safe_pipeline_optimization"]["selected_after"]["decisions_per_s"]

    correctness = correctness_probe()
    baseline_same_run = throughput(ThreadedRunner)
    cached = throughput(CachedNextLegalRunner)
    prof = profile_cached()

    improvement_vs_historical = (cached["decisions_per_s"] / historical - 1.0) * 100.0
    improvement_same_run = (cached["decisions_per_s"] / baseline_same_run["decisions_per_s"] - 1.0) * 100.0
    gpu_mean = cached["gpu"].get("gpu_util_mean")
    ratio = cached["decisions_per_s"] / model_only
    gate_a = bool(gpu_mean is not None and gpu_mean < 60.0 and prof["cpu_h2d_feed_percent"] >= 40.0)
    gate_b = bool(prof["pipeline_stall_percent"] >= 20.0)
    gate_c = bool(ratio < 0.70 and prof["cpu_h2d_feed_percent"] >= 40.0)

    payload = {
        "correctness_probe": correctness,
        "historical_tuned_baseline_decisions_per_s": historical,
        "baseline_same_run": baseline_same_run,
        "cached_next_legal": cached,
        "cached_profile": prof,
        "improvement_percent_vs_historical_tuned": improvement_vs_historical,
        "improvement_percent_same_run": improvement_same_run,
        "model_only_states_per_s": model_only,
        "closed_loop_model_only_ratio": ratio,
        "starvation_tests": {"A": gate_a, "B": gate_b, "C": gate_c},
        "starvation_pass": not (gate_a or gate_b or gate_c),
        "buffer_reuse_decision": {
            "tested_as_new_probe": False,
            "reason": "P1 measured BatchStepResult/output-copy allocation at only 0.028% of the 4096 CPU decision; no large allocation hotspot exists to justify a separate buffer-reuse rewrite.",
        },
        "process_probe_decision": {
            "tested": False,
            "reason": "P1 cProfile is dominated by NumPy movement primitives; there is no profiler evidence that GIL contention is the bottleneck. Existing 2-thread gain and 4/8-thread regressions are retained as prior evidence.",
        },
    }
    write_json(OUT, payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
