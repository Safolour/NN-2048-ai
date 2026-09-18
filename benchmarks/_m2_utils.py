"""Small shared helpers for M2 GPU benchmarks."""
from __future__ import annotations

import json
import subprocess
import threading
import time
from pathlib import Path


class NvidiaSmiSampler:
    def __init__(self, interval: float = 0.1) -> None:
        self.interval = interval
        self.samples: list[tuple[float, float]] = []
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=True,
                )
                first = result.stdout.strip().splitlines()[0]
                util, mem = [float(part.strip()) for part in first.split(",")[:2]]
                self.samples.append((util, mem))
            except Exception as exc:
                self.error = repr(exc)
                return
            self._stop.wait(self.interval)


    def summary(self) -> dict:
        if not self.samples:
            return {
                "available": False,
                "reason": self.error or "no samples",
                "sample_count": 0,
            }
        utils = [sample[0] for sample in self.samples]
        mems = [sample[1] for sample in self.samples]
        return {
            "available": True,
            "sample_count": len(utils),
            "gpu_util_mean": sum(utils) / len(utils),
            "gpu_util_min": min(utils),
            "gpu_util_max": max(utils),
            "memory_used_mib_max": max(mems),
        }


def write_json(path: str | Path, payload: object) -> None:
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def synchronize(torch_module) -> None:
    if torch_module.cuda.is_available():
        torch_module.cuda.synchronize()


def timed_cuda_loop(torch_module, body, *, minimum_seconds: float = 2.0, minimum_iters: int = 100):
    for _ in range(20):
        body()
    synchronize(torch_module)
    start = time.perf_counter()
    iterations = 0
    while iterations < minimum_iters or time.perf_counter() - start < minimum_seconds:
        body()
        iterations += 1
    synchronize(torch_module)
    elapsed = time.perf_counter() - start
    return iterations, elapsed
