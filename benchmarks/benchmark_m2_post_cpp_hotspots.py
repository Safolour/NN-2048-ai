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
sys.path.insert(0, str(ROOT / "src"))

from game2048.m2_rollout_env import M2RolloutBatchEnv

SOURCE = ROOT / "m2_fast_backend_benchmark.json"
OUT = ROOT / "m2_post_cpp_hotspots.json"
N = 16384
SEED = 20260919 + 40000


def profile_rollout(steps: int = 100) -> tuple[str, float]:
    env = M2RolloutBatchEnv(N, seed=SEED)
    env.reset(seed=SEED)
    profiler = cProfile.Profile()
    profiler.enable()
    start = time.perf_counter()
    for _ in range(steps):
        legal = env.current_legal
        dead = ~legal.any(axis=1)
        if dead.any():
            env.reset_where(dead)
            legal = env.current_legal
        actions = np.argmax(legal, axis=1).astype(np.uint8)
        result = env.step(actions)
        if result.terminated.any():
            env.reset_where(result.terminated)
    elapsed = time.perf_counter() - start
    profiler.disable()
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumulative").print_stats(40)
    return stream.getvalue(), elapsed


def main() -> int:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    p = source["primitive"][str(N)]
    move_us = p["cpp_selected_move"]["us_per_iteration"]
    legal_us = p["cpp_legal"]["us_per_iteration"]
    step_us = p["cpp_rollout_step"]["us_per_iteration"]
    movement_legal_us = move_us + legal_us
    profiler_text, elapsed = profile_rollout()

    payload = {
        "env_count": N,
        "wall_clock_cross_check": {
            "cpp_selected_move_us": move_us,
            "cpp_next_legal_us": legal_us,
            "cpp_rollout_step_us": step_us,
            "selected_move_plus_next_legal_us": movement_legal_us,
            "movement_legal_share_of_step_percent": movement_legal_us / step_us * 100.0,
            "note": "P8 primitive wall-clock uses the same scalar backend on realistic 16384-board batches.",
        },
        "cprofile_100_rollout_steps": {
            "elapsed_s": elapsed,
            "text": profiler_text,
        },
        "decision": {
            "movement_legal_still_primary_inside_environment_step": movement_legal_us / step_us >= 0.5,
            "lut_eligible": movement_legal_us / step_us >= 0.5,
            "simd_or_lut_rule": "scalar correctness passed; movement/legal remains dominant, so one scalar->LUT/SIMD optimization is evidence-eligible",
        },
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "wall_clock_cross_check": payload["wall_clock_cross_check"],
        "decision": payload["decision"],
        "cprofile_head": profiler_text.splitlines()[:18],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
