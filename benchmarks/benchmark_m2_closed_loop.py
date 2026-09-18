from __future__ import annotations

import contextlib
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
from game2048.fast_env import Fast2048BatchEnv, legal_mask_batch
from game2048.m2_data import prepare_board_batch_for_transfer
from game2048.m2_models import ResidualMLP2048, Transformer2048
from game2048.m2_policy import select_greedy_actions

GPU_RESULTS = ROOT / "reports/m2/m2_gpu_benchmark.json"
OUTPUT = ROOT / "reports/m2/m2_closed_loop_benchmark.json"
DEVICE = torch.device("cuda")
SEED = 20260918
MODELS = {"Transformer2048": Transformer2048, "ResidualMLP2048": ResidualMLP2048}


def autocast_for(precision: str):
    if precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def build_model(model_cls, execution: str):
    torch.manual_seed(SEED)
    model = model_cls().to(DEVICE).eval()
    return torch.compile(model) if execution == "compile" else model


def one_iteration(env, model, precision, generator):
    boards_np = prepare_board_batch_for_transfer(env.boards)
    legal_np = legal_mask_batch(boards_np)
    boards = torch.from_numpy(boards_np).to(DEVICE, non_blocking=False)
    legal = torch.from_numpy(legal_np).to(DEVICE, non_blocking=False)
    with torch.inference_mode(), autocast_for(precision):
        q = model(boards)
    actions = select_greedy_actions(q.float(), legal, generator=generator)
    actions_np = actions.to("cpu").numpy().astype(np.uint8, copy=False)
    step = env.step(actions_np)
    if step.terminated.any():
        env.reset_where(step.terminated)


def throughput_point(model_cls, config: dict, env_count: int) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    env = Fast2048BatchEnv(env_count, seed=SEED)
    env.reset(seed=SEED)
    model = build_model(model_cls, config["execution"])
    generator = torch.Generator(device=DEVICE).manual_seed(SEED)
    for _ in range(20):
        one_iteration(env, model, config["precision"], generator)
    torch.cuda.synchronize()

    iterations = 0
    with ResourceSampler() as cpu_sampler, NvidiaSmiSampler() as gpu_sampler:
        start = time.perf_counter()
        while iterations < 100 or time.perf_counter() - start < 2.0:
            one_iteration(env, model, config["precision"], generator)
            iterations += 1
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    return {
        "env_count": env_count,
        "iterations": iterations,
        "elapsed_s": elapsed,
        "decisions_per_s": env_count * iterations / elapsed,
        "environment_transitions_per_s": env_count * iterations / elapsed,
        "nn_states_per_s": env_count * iterations / elapsed,
        "cpu_utilization_cores": cpu_sampler.cpu_utilization,
        "peak_rss_bytes": cpu_sampler.peak_rss,
        "peak_vram_allocated": torch.cuda.max_memory_allocated(),
        "peak_vram_reserved": torch.cuda.max_memory_reserved(),
        "gpu": gpu_sampler.summary(),
    }
def profile_point(model_cls, config: dict, env_count: int, iterations: int = 30) -> dict:
    torch.cuda.empty_cache()
    env = Fast2048BatchEnv(env_count, seed=SEED + 1)
    env.reset(seed=SEED + 1)
    model = build_model(model_cls, config["execution"])
    generator = torch.Generator(device=DEVICE).manual_seed(SEED + 1)
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
    transferred_bytes = 0
    for _ in range(iterations):
        t = time.perf_counter()
        boards_np = prepare_board_batch_for_transfer(env.boards)
        totals["cpu_batch_preparation"] += time.perf_counter() - t

        t = time.perf_counter()
        legal_np = legal_mask_batch(boards_np)
        totals["legal_mask"] += time.perf_counter() - t

        torch.cuda.synchronize()
        t = time.perf_counter()
        boards = torch.from_numpy(boards_np).to(DEVICE)
        legal = torch.from_numpy(legal_np).to(DEVICE)
        torch.cuda.synchronize()
        totals["h2d"] += time.perf_counter() - t
        transferred_bytes += boards_np.nbytes + legal_np.nbytes

        torch.cuda.synchronize()
        t = time.perf_counter()
        with torch.inference_mode(), autocast_for(config["precision"]):
            q = model(boards)
        torch.cuda.synchronize()
        totals["nn_inference"] += time.perf_counter() - t

        torch.cuda.synchronize()
        t = time.perf_counter()
        actions = select_greedy_actions(q.float(), legal, generator=generator)
        torch.cuda.synchronize()
        totals["action_selection"] += time.perf_counter() - t

        torch.cuda.synchronize()
        t = time.perf_counter()
        actions_np = actions.cpu().numpy().astype(np.uint8, copy=False)
        torch.cuda.synchronize()
        totals["d2h"] += time.perf_counter() - t

        t = time.perf_counter()
        step = env.step(actions_np)
        totals["environment_step"] += time.perf_counter() - t

        t = time.perf_counter()
        if step.terminated.any():
            env.reset_where(step.terminated)
        totals["reset_where"] += time.perf_counter() - t

    total = sum(totals.values())
    other = max(0.0, total - sum(totals.values()))
    percentages = {key: value / total * 100.0 for key, value in totals.items()}
    percentages["other"] = other / total * 100.0 if total else 0.0
    h2d_bandwidth = transferred_bytes / totals["h2d"] if totals["h2d"] > 0 else 0.0
    pipeline_stall = totals["cpu_batch_preparation"] + totals["h2d"] + totals["d2h"]
    return {
        "env_count": env_count,
        "iterations": iterations,
        "stage_seconds": totals,
        "stage_percent": percentages,
        "cpu_h2d_feed_percent": (
            totals["environment_step"]
            + totals["legal_mask"]
            + totals["cpu_batch_preparation"]
            + totals["h2d"]
        ) / total * 100.0,
        "pipeline_stall_percent": pipeline_stall / total * 100.0,
        "h2d_effective_bytes_per_s": h2d_bandwidth,
        "h2d_total_bytes": transferred_bytes,
    }
def main() -> int:
    if not GPU_RESULTS.exists():
        write_json(OUTPUT, {"pass": False, "reason": "reports/m2/m2_gpu_benchmark.json missing"})
        return 2
    gpu_results = json.loads(GPU_RESULTS.read_text(encoding="utf-8"))
    payload = {"models": {}, "pass": True}
    total_vram = torch.cuda.get_device_properties(0).total_memory

    for name, model_cls in MODELS.items():
        selected = gpu_results["models"][name].get("selected_inference")
        if not selected:
            payload["models"][name] = {"pass": False, "reason": "no selected inference config"}
            payload["pass"] = False
            continue
        config = {
            "precision": selected["precision"],
            "execution": selected["execution"],
            "model_only_states_per_s": selected["states_per_s"],
        }
        points = []
        for env_count in [1024, 4096, 8192]:
            try:
                point = throughput_point(model_cls, config, env_count)
                points.append(point)
                print(name, env_count, point)
            except torch.cuda.OutOfMemoryError as exc:
                points.append({"env_count": env_count, "status": "OOM", "error": str(exc)})
                torch.cuda.empty_cache()

        valid = [p for p in points if "decisions_per_s" in p]
        if not valid:
            payload["models"][name] = {"config": config, "throughput": points, "pass": False}
            payload["pass"] = False
            continue
        best = max(valid, key=lambda p: p["decisions_per_s"])

        # Optional 16384 when 8192 is healthy and resources permit.
        p8192 = next((p for p in valid if p["env_count"] == 8192), None)
        if p8192 and p8192["peak_vram_reserved"] < 0.70 * total_vram:
            try:
                p16384 = throughput_point(model_cls, config, 16384)
                points.append(p16384)
                if p16384["decisions_per_s"] > best["decisions_per_s"]:
                    best = p16384
            except torch.cuda.OutOfMemoryError as exc:
                points.append({"env_count": 16384, "status": "OOM", "error": str(exc)})
                torch.cuda.empty_cache()

        profile = profile_point(model_cls, config, best["env_count"])
        gpu_mean = best.get("gpu", {}).get("gpu_util_mean")
        ratio = best["nn_states_per_s"] / config["model_only_states_per_s"]
        gate_a = (
            gpu_mean is not None
            and gpu_mean < 60.0
            and profile["cpu_h2d_feed_percent"] >= 40.0
        )
        gate_b = profile["pipeline_stall_percent"] >= 20.0
        gate_c = ratio < 0.70 and profile["cpu_h2d_feed_percent"] >= 40.0
        starvation = bool(gate_a or gate_b or gate_c)
        model_payload = {
            "config": config,
            "throughput": points,
            "best_closed_loop": best,
            "profile": profile,
            "model_only_to_closed_loop_ratio": ratio,
            "starvation_tests": {"A": gate_a, "B": gate_b, "C": gate_c},
            "gpu_starvation_gate": "FAIL" if starvation else "PASS",
            "pass": not starvation,
        }
        payload["models"][name] = model_payload
        if starvation:
            payload["pass"] = False

    write_json(OUTPUT, payload)
    print(payload)
    return 0 if payload["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
