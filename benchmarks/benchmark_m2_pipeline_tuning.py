from __future__ import annotations
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))

from _m2_utils import NvidiaSmiSampler, write_json
from _utils import ResourceSampler
from game2048.fast_env import Fast2048BatchEnv, legal_mask_batch
from game2048.m2_data import prepare_board_batch_for_transfer
from game2048.m2_models import ResidualMLP2048
from game2048.m2_policy import select_greedy_actions

SEED = 20260918
DEVICE = torch.device("cuda")
CLOSED_LOOP = ROOT / "m2_closed_loop_benchmark.json"

class ThreadedRunner:
    def __init__(self, total: int, workers: int) -> None:
        if total % workers:
            raise ValueError("total must be divisible by workers")
        self.total = total
        self.workers = workers
        self.size = total // workers
        self.envs = [
            Fast2048BatchEnv(self.size, seed=SEED + i * 100003)
            for i in range(workers)
        ]
        for i, env in enumerate(self.envs):
            env.reset(seed=SEED + i * 100003)
        self.slices = [
            slice(i * self.size, (i + 1) * self.size)
            for i in range(workers)
        ]
        self.board_cpu = torch.empty(
            (total, 16), dtype=torch.uint8, pin_memory=True
        )
        self.legal_cpu = torch.empty(
            (total, 4), dtype=torch.bool, pin_memory=True
        )
        self.action_cpu = torch.empty(
            (total,), dtype=torch.uint8, pin_memory=True
        )
        self.board_np = self.board_cpu.numpy()
        self.legal_np = self.legal_cpu.numpy()
        self.action_np = self.action_cpu.numpy()
        self.board_gpu = torch.empty(
            (total, 16), dtype=torch.uint8, device=DEVICE
        )
        self.legal_gpu = torch.empty(
            (total, 4), dtype=torch.bool, device=DEVICE
        )
        self.terminated = [None] * workers
        self.pool = ThreadPoolExecutor(max_workers=workers)

    def close(self) -> None:
        self.pool.shutdown(wait=True)

    def board_one(self, idx: int) -> None:
        sl = self.slices[idx]
        boards = prepare_board_batch_for_transfer(self.envs[idx].boards)
        np.copyto(self.board_np[sl], boards)

    def legal_one(self, idx: int) -> None:
        sl = self.slices[idx]
        np.copyto(self.legal_np[sl], legal_mask_batch(self.board_np[sl]))

    def step_one(self, idx: int) -> None:
        sl = self.slices[idx]
        result = self.envs[idx].step(self.action_np[sl])
        self.terminated[idx] = result.terminated

    def reset_one(self, idx: int) -> None:
        terminated = self.terminated[idx]
        if terminated is not None and terminated.any():
            self.envs[idx].reset_where(terminated)

    def boards(self) -> None:
        list(self.pool.map(self.board_one, range(self.workers)))

    def legal(self) -> None:
        list(self.pool.map(self.legal_one, range(self.workers)))

    def step(self) -> None:
        list(self.pool.map(self.step_one, range(self.workers)))

    def reset(self) -> None:
        list(self.pool.map(self.reset_one, range(self.workers)))

def build_model():
    torch.manual_seed(SEED)
    return ResidualMLP2048().to(DEVICE).eval()

def make_action_generator():
    return torch.Generator(device=DEVICE).manual_seed(SEED)

def gpu_policy(runner, model, generator):
    runner.board_gpu.copy_(runner.board_cpu, non_blocking=True)
    runner.legal_gpu.copy_(runner.legal_cpu, non_blocking=True)
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16
    ):
        q = model(runner.board_gpu)
    actions = select_greedy_actions(
        q.float(), runner.legal_gpu, generator=generator
    )
    runner.action_cpu.copy_(actions.to(torch.uint8), non_blocking=True)
    torch.cuda.synchronize()

def one_iteration(runner, model, generator):
    runner.boards()
    runner.legal()
    gpu_policy(runner, model, generator)
    runner.step()
    runner.reset()

def throughput(workers: int, env_count: int = 4096) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    runner = ThreadedRunner(env_count, workers)
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

def profile(workers: int, env_count: int = 4096, iterations: int = 30) -> dict:
    torch.cuda.empty_cache()
    runner = ThreadedRunner(env_count, workers)
    model = build_model()
    generator = make_action_generator()
    totals = {
        "environment_step": 0.0,
        "legal_mask": 0.0,
        "cpu_batch_preparation": 0.0,
        "h2d": 0.0,
        "nn_inference": 0.0,
        "action_selection": 0.0,
        "d2h": 0.0,
        "reset_where": 0.0,
    }
    transferred = 0
    try:
        for _ in range(iterations):
            t = time.perf_counter()
            runner.boards()
            totals["cpu_batch_preparation"] += time.perf_counter() - t

            t = time.perf_counter()
            runner.legal()
            totals["legal_mask"] += time.perf_counter() - t

            torch.cuda.synchronize()
            t = time.perf_counter()
            runner.board_gpu.copy_(runner.board_cpu, non_blocking=True)
            runner.legal_gpu.copy_(runner.legal_cpu, non_blocking=True)
            torch.cuda.synchronize()
            totals["h2d"] += time.perf_counter() - t
            transferred += runner.board_cpu.numel() + runner.legal_cpu.numel()

            torch.cuda.synchronize()
            t = time.perf_counter()
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16
            ):
                q = model(runner.board_gpu)
            torch.cuda.synchronize()
            totals["nn_inference"] += time.perf_counter() - t

            torch.cuda.synchronize()
            t = time.perf_counter()
            actions = select_greedy_actions(
                q.float(), runner.legal_gpu, generator=generator
            )
            torch.cuda.synchronize()
            totals["action_selection"] += time.perf_counter() - t

            torch.cuda.synchronize()
            t = time.perf_counter()
            runner.action_cpu.copy_(
                actions.to(torch.uint8), non_blocking=True
            )
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
    percent["other"] = 0.0
    feed = (
        totals["environment_step"]
        + totals["legal_mask"]
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
        "h2d_total_bytes": transferred,
    }

def main() -> int:
    data = json.loads(CLOSED_LOOP.read_text(encoding="utf-8"))
    mlp = data["models"]["ResidualMLP2048"]
    baseline = mlp["best_closed_loop"]
    model_only = mlp["config"]["model_only_states_per_s"]
    probes = [throughput(w) for w in (2, 4, 8)]
    best = max(probes, key=lambda p: p["decisions_per_s"])
    prof = profile(best["workers"], best["env_count"])
    gpu_mean = best["gpu"].get("gpu_util_mean")
    ratio = best["decisions_per_s"] / model_only
    gate_a = (
        gpu_mean is not None
        and gpu_mean < 60.0
        and prof["cpu_h2d_feed_percent"] >= 40.0
    )
    gate_b = prof["pipeline_stall_percent"] >= 20.0
    gate_c = ratio < 0.70 and prof["cpu_h2d_feed_percent"] >= 40.0
    starvation = bool(gate_a or gate_b or gate_c)
    tuning = {
        "reason": "baseline MLP triggered starvation gates A and C",
        "allowed_changes": [
            "worker-sharded Fast2048BatchEnv instances",
            "reused pinned CPU staging buffers",
            "reused GPU buffers",
            "non_blocking H2D/D2H copies",
        ],
        "baseline_decisions_per_s": baseline["decisions_per_s"],
        "probes": probes,
        "selected_after": best,
        "throughput_improvement_percent": (
            best["decisions_per_s"] / baseline["decisions_per_s"] - 1.0
        ) * 100.0,
        "profile_after": prof,
        "model_only_states_per_s": model_only,
        "model_only_to_closed_loop_ratio_after": ratio,
        "starvation_tests_after": {"A": gate_a, "B": gate_b, "C": gate_c},
        "gpu_starvation_gate_after": "FAIL" if starvation else "PASS",
    }
    mlp["safe_pipeline_optimization"] = tuning
    mlp["pass_after_optimization"] = not starvation
    data["pass"] = all(
        m.get("pass", False)
        if "pass_after_optimization" not in m
        else m["pass_after_optimization"]
        for m in data["models"].values()
    )
    write_json(CLOSED_LOOP, data)
    print(json.dumps(tuning, indent=2))
    return 1 if starvation else 0


if __name__ == "__main__":
    raise SystemExit(main())
