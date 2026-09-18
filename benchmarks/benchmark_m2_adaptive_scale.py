from __future__ import annotations

import argparse
import ctypes
import json
import sys
import threading
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))

from _m2_utils import NvidiaSmiSampler, write_json
from _utils import ResourceSampler
from benchmark_m2_fast_backend import (
    CppRolloutRunner,
    build_model,
    make_action_generator,
    one_iteration,
    profile_cpp,
)

BENCHMARK = ROOT / "reports/m2/m2_fast_backend_benchmark.json"
CLOSED_LOOP = ROOT / "reports/m2/m2_closed_loop_benchmark.json"
DEVICE = torch.device("cuda")


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_uint32),
        ("dwMemoryLoad", ctypes.c_uint32),
        ("ullTotalPhys", ctypes.c_uint64),
        ("ullAvailPhys", ctypes.c_uint64),
        ("ullTotalPageFile", ctypes.c_uint64),
        ("ullAvailPageFile", ctypes.c_uint64),
        ("ullTotalVirtual", ctypes.c_uint64),
        ("ullAvailVirtual", ctypes.c_uint64),
        ("ullAvailExtendedVirtual", ctypes.c_uint64),
    ]


def memory_snapshot() -> dict:
    status = _MemoryStatusEx()
    status.dwLength = ctypes.sizeof(_MemoryStatusEx)
    ok = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    if not ok:
        return {}
    return {
        "memory_load_percent": int(status.dwMemoryLoad),
        "total_phys_bytes": int(status.ullTotalPhys),
        "available_phys_bytes": int(status.ullAvailPhys),
        "total_pagefile_bytes": int(status.ullTotalPageFile),
        "available_pagefile_bytes": int(status.ullAvailPageFile),
    }


class SystemMemorySampler:
    def __init__(self, interval: float = 0.05) -> None:
        self.interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples: list[dict] = []

    def __enter__(self):
        self.samples.append(memory_snapshot())
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.samples.append(memory_snapshot())

    def _loop(self):
        while not self._stop.wait(self.interval):
            self.samples.append(memory_snapshot())

    def summary(self) -> dict:
        samples = [s for s in self.samples if s]
        if not samples:
            return {"available": False}
        return {
            "available": True,
            "sample_count": len(samples),
            "total_phys_bytes": samples[0]["total_phys_bytes"],
            "max_memory_load_percent": max(s["memory_load_percent"] for s in samples),
            "min_available_phys_bytes": min(s["available_phys_bytes"] for s in samples),
            "min_available_pagefile_bytes": min(s["available_pagefile_bytes"] for s in samples),
        }


def closed_loop_supplemental(env_count: int, workers: int = 2) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    runner = CppRolloutRunner(env_count, workers)
    model = build_model()
    generator = make_action_generator()
    try:
        for _ in range(20):
            one_iteration(runner, model, generator)
        torch.cuda.synchronize()

        iterations = 0
        with (
            ResourceSampler() as cpu_sampler,
            NvidiaSmiSampler() as gpu_sampler,
            SystemMemorySampler() as memory_sampler,
        ):
            start = time.perf_counter()
            while iterations < 100 or time.perf_counter() - start < 2.0:
                one_iteration(runner, model, generator)
                iterations += 1
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

        return {
            "status": "PASS",
            "workers": workers,
            "env_count": env_count,
            "iterations": iterations,
            "elapsed_s": elapsed,
            "decisions_per_s": env_count * iterations / elapsed,
            "cpu_utilization_cores": cpu_sampler.cpu_utilization,
            "peak_rss_bytes": cpu_sampler.peak_rss,
            "system_memory": memory_sampler.summary(),
            "gpu": gpu_sampler.summary(),
            "peak_vram_allocated": torch.cuda.max_memory_allocated(),
            "peak_vram_reserved": torch.cuda.max_memory_reserved(),
            "total_vram_bytes": torch.cuda.get_device_properties(0).total_memory,
        }
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        return {
            "status": "OOM",
            "workers": workers,
            "env_count": env_count,
            "error": str(exc),
        }
    finally:
        runner.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-count", type=int, required=True)
    args = parser.parse_args()

    payload = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    closed = json.loads(CLOSED_LOOP.read_text(encoding="utf-8"))
    model_only = float(
        closed["models"]["ResidualMLP2048"]["config"]["model_only_states_per_s"]
    )

    point = closed_loop_supplemental(args.env_count)
    result = {"closed_loop": point}

    if point["status"] == "PASS":
        profile = profile_cpp(args.env_count)
        gpu_mean = point["gpu"].get("gpu_util_mean")
        feed = profile["cpu_h2d_feed_percent"]
        ratio = point["decisions_per_s"] / model_only
        gate_a = bool(
            gpu_mean is not None
            and gpu_mean < 60.0
            and feed >= 40.0
        )
        gate_b = bool(profile["pipeline_stall_percent"] >= 20.0)
        gate_c = bool(ratio < 0.70 and feed >= 40.0)
        result.update(
            {
                "profile": profile,
                "model_only_states_per_s": model_only,
                "closed_loop_model_only_ratio": ratio,
                "cpu_h2d_feed_percent": feed,
                "pipeline_stall_percent": profile["pipeline_stall_percent"],
                "starvation_tests": {
                    "A": gate_a,
                    "B": gate_b,
                    "C": gate_c,
                },
                "starvation_pass": not (gate_a or gate_b or gate_c),
            }
        )

    ext = payload.setdefault(
        "adaptive_scale_extension",
        {
            "trigger": (
                "16384 LUT closed-loop improved >=5% over 8192, so §27.3 "
                "requires 32768 validation"
            ),
            "probes": {},
        },
    )
    ext["probes"][str(args.env_count)] = result
    write_json(BENCHMARK, payload)
    print(json.dumps(result, indent=2))
    return 0 if point["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
