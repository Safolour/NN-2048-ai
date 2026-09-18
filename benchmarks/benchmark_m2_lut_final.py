from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))

from _m2_utils import write_json
from benchmark_m2_fast_backend import (
    CppRolloutRunner,
    bench_call,
    bench_fused_decision,
    bench_step,
    closed_loop,
    prepare_boards,
    profile_cpp,
)
from game2048.m2_fast_backend import (
    backend_info,
    legal_mask_batch as lut_legal,
    move_selected_batch as lut_move,
)
from game2048.m2_rollout_env import M2RolloutBatchEnv

SEED = 20260919
SIZES = (1024, 4096, 8192, 16384)
SCALAR = ROOT / "m2_fast_backend_scalar_benchmark.json"
OLD_CLOSED_LOOP = ROOT / "m2_closed_loop_benchmark.json"
OUT = ROOT / "m2_fast_backend_benchmark.json"


def scalar_closed_loop_by_size(scalar: dict) -> dict[int, dict]:
    return {
        int(item["env_count"]): item["cpp_backend"]
        for item in scalar["closed_loop"]
    }


def main() -> int:
    scalar = json.loads(SCALAR.read_text(encoding="utf-8"))
    old = json.loads(OLD_CLOSED_LOOP.read_text(encoding="utf-8"))
    info = backend_info()
    if info["kind"] != "scalar+row-lut" or not info["lut_used"] or info["simd_used"]:
        raise RuntimeError(f"unexpected final backend metadata: {info}")

    payload = {
        "backend": info,
        "baseline_source": str(SCALAR),
        "methodology": {
            "scalar_rerun": False,
            "same_seed_and_workload": True,
            "sizes": list(SIZES),
            "warmup_min_iterations": 20,
            "measurement_min_iterations": 100,
            "measurement_min_seconds": 2.0,
            "primary_metric": "ResidualMLP closed-loop decisions/s",
        },
        "primitive": {},
        "closed_loop": [],
    }

    for n in SIZES:
        print(f"LUT primitive {n}", flush=True)
        boards, actions = prepare_boards(n, SEED + n)
        baseline = scalar["primitive"][str(n)]
        lut = {
            "legal": bench_call(lambda: lut_legal(boards), n),
            "selected_move": bench_call(lambda: lut_move(boards, actions), n),
            "terminal": bench_call(lambda: ~lut_legal(boards).any(axis=1), n),
            "rollout_step": bench_step(M2RolloutBatchEnv, n, SEED + n + 10),
            "fused_next_legal": bench_fused_decision(
                M2RolloutBatchEnv, n, SEED + n + 20
            ),
        }
        payload["primitive"][str(n)] = {
            "numpy_legal": baseline["numpy_legal"],
            "scalar_cpp_legal": baseline["cpp_legal"],
            "lut_cpp_legal": lut["legal"],
            "numpy_selected_move": baseline["numpy_selected_move"],
            "scalar_cpp_selected_move": baseline["cpp_selected_move"],
            "lut_cpp_selected_move": lut["selected_move"],
            "numpy_terminal": baseline["numpy_terminal"],
            "scalar_cpp_terminal": baseline["cpp_terminal"],
            "lut_cpp_terminal": lut["terminal"],
            "numpy_step": baseline["numpy_step"],
            "scalar_cpp_rollout_step": baseline["cpp_rollout_step"],
            "lut_cpp_rollout_step": lut["rollout_step"],
            "numpy_fused_decision": baseline["numpy_fused_decision"],
            "scalar_cpp_fused_next_legal": baseline["cpp_fused_next_legal"],
            "lut_cpp_fused_next_legal": lut["fused_next_legal"],
            "scalar_to_lut_percent": {
                "legal": (
                    lut["legal"]["units_per_s"]
                    / baseline["cpp_legal"]["units_per_s"]
                    - 1.0
                ) * 100.0,
                "selected_move": (
                    lut["selected_move"]["units_per_s"]
                    / baseline["cpp_selected_move"]["units_per_s"]
                    - 1.0
                ) * 100.0,
                "fused_next_legal": (
                    lut["fused_next_legal"]["units_per_s"]
                    / baseline["cpp_fused_next_legal"]["units_per_s"]
                    - 1.0
                ) * 100.0,
            },
        }

    scalar_cl = scalar_closed_loop_by_size(scalar)
    for n in SIZES:
        print(f"LUT closed-loop {n}", flush=True)
        lut_point = closed_loop(CppRolloutRunner, n)
        scalar_point = scalar_cl[n]
        payload["closed_loop"].append(
            {
                "env_count": n,
                "scalar_cpp": scalar_point,
                "lut_cpp": lut_point,
                "scalar_to_lut_improvement_percent": (
                    lut_point["decisions_per_s"]
                    / scalar_point["decisions_per_s"]
                    - 1.0
                ) * 100.0,
            }
        )

    best_lut = max(
        (item["lut_cpp"] for item in payload["closed_loop"]),
        key=lambda x: x["decisions_per_s"],
    )
    best_scalar = scalar["best_cpp_closed_loop"]
    profile = profile_cpp(best_lut["env_count"])

    model_only = old["models"]["ResidualMLP2048"]["config"]["model_only_states_per_s"]
    gpu_mean = best_lut["gpu"].get("gpu_util_mean")
    ratio = best_lut["decisions_per_s"] / model_only
    feed = profile["cpu_h2d_feed_percent"]

    gate_a = bool(gpu_mean is not None and gpu_mean < 60.0 and feed >= 40.0)
    gate_b = bool(profile["pipeline_stall_percent"] >= 20.0)
    gate_c = bool(ratio < 0.70 and feed >= 40.0)

    payload["best_scalar_closed_loop"] = best_scalar
    payload["best_lut_closed_loop"] = best_lut
    payload["best_scalar_to_lut_improvement_percent"] = (
        best_lut["decisions_per_s"] / best_scalar["decisions_per_s"] - 1.0
    ) * 100.0
    payload["profile_best_lut"] = profile
    payload["model_only_states_per_s"] = model_only
    payload["closed_loop_model_only_ratio"] = ratio
    payload["gpu_mean_percent"] = gpu_mean
    payload["cpu_h2d_feed_percent"] = feed
    payload["pipeline_stall_percent"] = profile["pipeline_stall_percent"]
    payload["starvation_tests"] = {"A": gate_a, "B": gate_b, "C": gate_c}
    payload["starvation_pass"] = not (gate_a or gate_b or gate_c)

    write_json(OUT, payload)
    print(json.dumps({
        "backend": info,
        "best_scalar_closed_loop": best_scalar,
        "best_lut_closed_loop": best_lut,
        "best_scalar_to_lut_improvement_percent": payload["best_scalar_to_lut_improvement_percent"],
        "gpu_mean_percent": gpu_mean,
        "cpu_h2d_feed_percent": feed,
        "closed_loop_model_only_ratio": ratio,
        "pipeline_stall_percent": profile["pipeline_stall_percent"],
        "starvation_tests": payload["starvation_tests"],
        "starvation_pass": payload["starvation_pass"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
