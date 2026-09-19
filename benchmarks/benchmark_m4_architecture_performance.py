
"""M4 model-only and closed-loop performance benchmark (M4_WO_CLOSURE_V2)."""
from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from game2048.m2_data import prepare_board_batch_for_transfer
from game2048.m2_models import ResidualMLP2048, Transformer2048
from game2048.m2_policy import select_greedy_actions
from game2048.m2_rollout_env import M2RolloutBatchEnv
from game2048.m4_compare import (
    ARCHITECTURES,
    DATASET_SHA256,
    RESIDUAL_MLP,
    TRAINING_SEEDS,
    TRANSFORMER,
    WORK_ORDER_VERSION,
    atomic_write_json,
    sha256_file,
)

ARTIFACT = ROOT / "artifacts" / "m4"
PROGRESS = ARTIFACT / "progress"
SESSION = PROGRESS / "session.json"
PRIMARY = PROGRESS / "primary.json"
PERFORMANCE = PROGRESS / "performance.json"

MODEL_CLASSES = {
    TRANSFORMER: Transformer2048,
    RESIDUAL_MLP: ResidualMLP2048,
}
MODEL_BATCHES = (1, 256, 1024, 2048, 4096, 8192)
CLOSED_ENVS = (2048, 8192)
REPEATS = 5


class GpuSampler:
    def __init__(self, available: bool) -> None:
        self.available = bool(available)
        self.util: list[float] = []
        self.memory: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample_once(self) -> None:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if result.returncode != 0:
                return
            line = result.stdout.strip().splitlines()[0]
            util, used, _total = [
                float(piece.strip()) for piece in line.split(",")
            ]
            self.util.append(util)
            self.memory.append(used)
        except Exception:
            return

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample_once()
            self._stop.wait(1.0)

    def start(self) -> None:
        if not self.available:
            return
        self._thread = threading.Thread(
            target=self._run, daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=6)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _autocast(precision: str):
    if precision == "BF16 autocast":
        return torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        )
    if precision == "FP32":
        return contextlib.nullcontext()
    raise RuntimeError("M4_BLOCKED_RESUME_METADATA_MISMATCH")


def _load_model(
    architecture: str,
    checkpoint_path: Path,
    expected_sha: str,
    expected_precision: str,
    device: torch.device,
) -> torch.nn.Module:
    if not checkpoint_path.exists():
        raise RuntimeError("M4_BLOCKED_RESUME_ARTIFACT_CORRUPT")
    if sha256_file(checkpoint_path) != expected_sha:
        raise RuntimeError("M4_BLOCKED_PERFORMANCE_CHECKPOINT_MISMATCH")
    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    expected = {
        "architecture": architecture,
        "seed": 20260919,
        "precision": expected_precision,
        "dataset_sha256": DATASET_SHA256,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(
                "M4_BLOCKED_PERFORMANCE_CHECKPOINT_MISMATCH"
            )
    model = MODEL_CLASSES[architecture]()
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    return model


def _run_info(primary: dict, architecture: str) -> dict:
    return primary["primary_training"][architecture]["20260919"]


def _checkpoint_manifest(primary: dict) -> dict:
    output = {}
    for architecture in ARCHITECTURES:
        run = _run_info(primary, architecture)
        output[architecture] = {
            "path": run["final_checkpoint_path"],
            "sha256": run["final_checkpoint_sha256"],
        }
    return output


def _new_model_point() -> dict:
    return {
        TRANSFORMER: {
            "repeat_states_per_s": [],
            "repeat_latency_s": [],
            "_repeat_peak_vram_bytes": [],
        },
        RESIDUAL_MLP: {
            "repeat_states_per_s": [],
            "repeat_latency_s": [],
            "_repeat_peak_vram_bytes": [],
        },
    }


def _new_closed_point() -> dict:
    return {
        TRANSFORMER: {
            "repeat_decisions_per_s": [],
            "repeat_wall_s": [],
            "_repeat_peak_vram_bytes": [],
            "_repeat_cpu_percent": [],
            "_gpu_util_samples": [],
            "_gpu_memory_samples": [],
        },
        RESIDUAL_MLP: {
            "repeat_decisions_per_s": [],
            "repeat_wall_s": [],
            "_repeat_peak_vram_bytes": [],
            "_repeat_cpu_percent": [],
            "_gpu_util_samples": [],
            "_gpu_memory_samples": [],
        },
    }


def _load_or_create_progress(
    primary: dict,
    session: dict,
) -> dict:
    precision = primary["precision"]["selected_precision"]
    checkpoints = _checkpoint_manifest(primary)
    if PERFORMANCE.exists():
        progress = _read_json(PERFORMANCE)
        expected = {
            "schema_version": 1,
            "work_order_version": WORK_ORDER_VERSION,
            "dataset_sha256": DATASET_SHA256,
            "precision": precision,
            "checkpoints": checkpoints,
        }
        for key, value in expected.items():
            if progress.get(key) != value:
                raise RuntimeError(
                    "M4_BLOCKED_RESUME_METADATA_MISMATCH"
                )
        return progress

    progress = {
        "schema_version": 1,
        "work_order_version": WORK_ORDER_VERSION,
        "dataset_sha256": DATASET_SHA256,
        "precision": precision,
        "checkpoints": checkpoints,
        "nvidia_smi_available": bool(
            session.get("nvidia_smi_available", False)
        ),
        "status": "running",
        "model_only": {},
        "closed_loop": {},
        "training": {},
    }
    atomic_write_json(PERFORMANCE, progress)
    return progress


def _validate_positive(value: float) -> None:
    if not np.isfinite(value) or value <= 0:
        raise RuntimeError("M4_BLOCKED_PERFORMANCE_INVALID")


def _synthetic_input(
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    rng = np.random.Generator(
        np.random.PCG64(20261300 + int(batch_size))
    )
    boards = rng.integers(
        0,
        22,
        size=(batch_size, 16),
        dtype=np.uint8,
    )
    return torch.from_numpy(boards).to(
        device=device, non_blocking=False
    )


def _model_only_repeat(
    architecture: str,
    checkpoint: dict,
    batch_size: int,
    precision: str,
    device: torch.device,
) -> dict:
    model = _load_model(
        architecture,
        ROOT / checkpoint["path"],
        checkpoint["sha256"],
        precision,
        device,
    )
    boards = _synthetic_input(batch_size, device)
    torch.cuda.reset_peak_memory_stats()

    with torch.inference_mode():
        for _ in range(50):
            with _autocast(precision):
                model(boards)
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(200):
            with _autocast(precision):
                model(boards)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started

    states_per_s = batch_size * 200 / elapsed
    latency = elapsed / 200
    _validate_positive(states_per_s)
    _validate_positive(latency)
    row = {
        "states_per_s": float(states_per_s),
        "latency_s": float(latency),
        "peak_vram_bytes": int(
            torch.cuda.max_memory_allocated()
        ),
    }
    del boards, model
    gc.collect()
    torch.cuda.empty_cache()
    return row


def _commit_model_pair(
    progress: dict,
    batch_size: int,
    repeat_index: int,
    pair: dict,
) -> None:
    point = progress["model_only"].setdefault(
        str(batch_size), _new_model_point()
    )
    for architecture in ARCHITECTURES:
        row = pair[architecture]
        arch_point = point[architecture]
        if len(arch_point["repeat_states_per_s"]) != repeat_index:
            raise RuntimeError(
                "M4_BLOCKED_RESUME_ARTIFACT_CORRUPT"
            )
        arch_point["repeat_states_per_s"].append(
            row["states_per_s"]
        )
        arch_point["repeat_latency_s"].append(
            row["latency_s"]
        )
        arch_point["_repeat_peak_vram_bytes"].append(
            row["peak_vram_bytes"]
        )
    atomic_write_json(PERFORMANCE, progress)


def _finalize_model_point(
    progress: dict,
    batch_size: int,
) -> None:
    point = progress["model_only"][str(batch_size)]
    for architecture in ARCHITECTURES:
        row = point[architecture]
        if (
            len(row["repeat_states_per_s"]) != REPEATS
            or len(row["repeat_latency_s"]) != REPEATS
            or len(row["_repeat_peak_vram_bytes"]) != REPEATS
        ):
            raise RuntimeError(
                "M4_BLOCKED_RESUME_ARTIFACT_CORRUPT"
            )
        speeds = np.asarray(
            row["repeat_states_per_s"],
            dtype=np.float64,
        )
        latencies = np.asarray(
            row["repeat_latency_s"],
            dtype=np.float64,
        )
        if not bool(np.isfinite(speeds).all()) or bool(
            (speeds <= 0).any()
        ):
            raise RuntimeError(
                "M4_BLOCKED_PERFORMANCE_INVALID"
            )
        if not bool(np.isfinite(latencies).all()) or bool(
            (latencies <= 0).any()
        ):
            raise RuntimeError(
                "M4_BLOCKED_PERFORMANCE_INVALID"
            )
        row["median_states_per_s"] = float(
            np.median(speeds)
        )
        row["median_latency_s"] = float(
            np.median(latencies)
        )
        row["peak_vram_bytes"] = int(
            max(row["_repeat_peak_vram_bytes"])
        )
    atomic_write_json(PERFORMANCE, progress)


def _run_model_only(
    progress: dict,
    device: torch.device,
) -> None:
    precision = progress["precision"]
    for batch_size in MODEL_BATCHES:
        point = progress["model_only"].setdefault(
            str(batch_size), _new_model_point()
        )
        lengths = [
            len(point[architecture]["repeat_states_per_s"])
            for architecture in ARCHITECTURES
        ]
        if lengths[0] != lengths[1]:
            raise RuntimeError(
                "M4_BLOCKED_RESUME_ARTIFACT_CORRUPT"
            )
        start_repeat = lengths[0]
        if start_repeat > REPEATS:
            raise RuntimeError(
                "M4_BLOCKED_RESUME_ARTIFACT_CORRUPT"
            )

        for repeat_index in range(start_repeat, REPEATS):
            order = (
                (TRANSFORMER, RESIDUAL_MLP)
                if repeat_index in (0, 2, 4)
                else (RESIDUAL_MLP, TRANSFORMER)
            )
            pair = {}
            for architecture in order:
                try:
                    row = _model_only_repeat(
                        architecture,
                        progress["checkpoints"][architecture],
                        batch_size,
                        precision,
                        device,
                    )
                except torch.cuda.OutOfMemoryError as exc:
                    raise RuntimeError(
                        "M4_BLOCKED_PERFORMANCE_OOM"
                    ) from exc
                pair[architecture] = row
                print(
                    f"performance model-only batch={batch_size} "
                    f"repeat={repeat_index} architecture={architecture} "
                    f"states/s={row['states_per_s']:.1f} "
                    f"latency={row['latency_s']*1000:.4f}ms",
                    flush=True,
                )
            _commit_model_pair(
                progress,
                batch_size,
                repeat_index,
                pair,
            )
        _finalize_model_point(progress, batch_size)


def _closed_step(
    env: M2RolloutBatchEnv,
    model: torch.nn.Module,
    generator: torch.Generator,
    precision: str,
    device: torch.device,
) -> None:
    boards_np = prepare_board_batch_for_transfer(
        env.boards
    )
    legal_np = np.ascontiguousarray(
        env.current_legal
    )
    boards = torch.from_numpy(boards_np).to(
        device=device, non_blocking=False
    )
    legal = torch.from_numpy(legal_np).to(
        device=device, non_blocking=False
    )
    with torch.inference_mode():
        with _autocast(precision):
            q_values = model(boards)
        actions = select_greedy_actions(
            q_values.float(),
            legal,
            generator=generator,
            tie_atol=1e-6,
        )
    actions_np = (
        actions.to("cpu")
        .numpy()
        .astype(np.uint8, copy=False)
    )
    step = env.step(actions_np)
    if bool(step.terminated.any()):
        env.reset_where(step.terminated)


def _closed_loop_repeat(
    architecture: str,
    checkpoint: dict,
    env_count: int,
    repeat_index: int,
    precision: str,
    device: torch.device,
    nvidia_available: bool,
) -> dict:
    model = _load_model(
        architecture,
        ROOT / checkpoint["path"],
        checkpoint["sha256"],
        precision,
        device,
    )
    seed = 20261200 + repeat_index
    env = M2RolloutBatchEnv(
        env_count, seed=seed
    )
    env.reset(seed=seed)
    generator = torch.Generator(
        device="cuda"
    ).manual_seed(
        20261400 + env_count * 10 + repeat_index
    )

    for _ in range(20):
        _closed_step(
            env,
            model,
            generator,
            precision,
            device,
        )

    torch.cuda.reset_peak_memory_stats()
    sampler = GpuSampler(nvidia_available)
    sampler.start()
    cpu0 = time.process_time()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(500):
        _closed_step(
            env,
            model,
            generator,
            precision,
            device,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    cpu_seconds = time.process_time() - cpu0
    sampler.stop()

    decisions_per_s = env_count * 500 / elapsed
    normalized_cpu = (
        100.0
        * cpu_seconds
        / elapsed
        / (os.cpu_count() or 1)
    )
    _validate_positive(decisions_per_s)
    _validate_positive(elapsed)
    row = {
        "decisions_per_s": float(decisions_per_s),
        "wall_s": float(elapsed),
        "peak_vram_bytes": int(
            torch.cuda.max_memory_allocated()
        ),
        "normalized_cpu_percent": float(
            normalized_cpu
        ),
        "gpu_util_samples": list(sampler.util),
        "gpu_memory_samples": list(sampler.memory),
        "seed": seed,
    }
    del env, model
    gc.collect()
    torch.cuda.empty_cache()
    return row


def _commit_closed_pair(
    progress: dict,
    env_count: int,
    repeat_index: int,
    pair: dict,
) -> None:
    point = progress["closed_loop"].setdefault(
        str(env_count), _new_closed_point()
    )
    for architecture in ARCHITECTURES:
        row = pair[architecture]
        arch_point = point[architecture]
        if len(arch_point["repeat_decisions_per_s"]) != repeat_index:
            raise RuntimeError(
                "M4_BLOCKED_RESUME_ARTIFACT_CORRUPT"
            )
        arch_point["repeat_decisions_per_s"].append(
            row["decisions_per_s"]
        )
        arch_point["repeat_wall_s"].append(
            row["wall_s"]
        )
        arch_point["_repeat_peak_vram_bytes"].append(
            row["peak_vram_bytes"]
        )
        arch_point["_repeat_cpu_percent"].append(
            row["normalized_cpu_percent"]
        )
        arch_point["_gpu_util_samples"].extend(
            row["gpu_util_samples"]
        )
        arch_point["_gpu_memory_samples"].extend(
            row["gpu_memory_samples"]
        )
    atomic_write_json(PERFORMANCE, progress)


def _gpu_point_summary(
    nvidia_available: bool,
    util_samples: list[float],
    memory_samples: list[float],
) -> dict | str:
    if not nvidia_available:
        return "N/A_NVIDIA_SMI_UNAVAILABLE"
    if not util_samples:
        return "N/A_NO_SAMPLES"
    util = np.asarray(util_samples, dtype=np.float64)
    memory = np.asarray(
        memory_samples, dtype=np.float64
    )
    return {
        "samples": int(util.size),
        "mean_utilization_pct": float(util.mean()),
        "p95_utilization_pct": float(
            np.quantile(util, 0.95)
        ),
        "peak_memory_used_mib": float(memory.max()),
    }


def _finalize_closed_point(
    progress: dict,
    env_count: int,
) -> None:
    point = progress["closed_loop"][str(env_count)]
    model_point = progress["model_only"][
        str(env_count)
    ]
    for architecture in ARCHITECTURES:
        row = point[architecture]
        if len(row["repeat_decisions_per_s"]) != REPEATS:
            raise RuntimeError(
                "M4_BLOCKED_RESUME_ARTIFACT_CORRUPT"
            )
        speeds = np.asarray(
            row["repeat_decisions_per_s"],
            dtype=np.float64,
        )
        walls = np.asarray(
            row["repeat_wall_s"],
            dtype=np.float64,
        )
        cpus = np.asarray(
            row["_repeat_cpu_percent"],
            dtype=np.float64,
        )
        if not bool(np.isfinite(speeds).all()) or bool(
            (speeds <= 0).any()
        ):
            raise RuntimeError(
                "M4_BLOCKED_PERFORMANCE_INVALID"
            )
        if not bool(np.isfinite(walls).all()) or bool(
            (walls <= 0).any()
        ):
            raise RuntimeError(
                "M4_BLOCKED_PERFORMANCE_INVALID"
            )
        median_speed = float(np.median(speeds))
        row["median_decisions_per_s"] = median_speed
        row["peak_vram_bytes"] = int(
            max(row["_repeat_peak_vram_bytes"])
        )
        row["gpu_utilization"] = _gpu_point_summary(
            bool(progress["nvidia_smi_available"]),
            row["_gpu_util_samples"],
            row["_gpu_memory_samples"],
        )
        row["cpu_utilization"] = {
            "repeat_normalized_cpu_percent": [
                float(value) for value in cpus
            ],
            "median_normalized_cpu_percent": float(
                np.median(cpus)
            ),
        }
        model_speed = float(
            model_point[architecture][
                "median_states_per_s"
            ]
        )
        ratio = model_speed / median_speed
        _validate_positive(ratio)
        row["pipeline_ratio"] = float(ratio)
    atomic_write_json(PERFORMANCE, progress)


def _run_closed_loop(
    progress: dict,
    device: torch.device,
) -> None:
    precision = progress["precision"]
    nvidia_available = bool(
        progress["nvidia_smi_available"]
    )
    for env_count in CLOSED_ENVS:
        point = progress["closed_loop"].setdefault(
            str(env_count), _new_closed_point()
        )
        lengths = [
            len(
                point[architecture][
                    "repeat_decisions_per_s"
                ]
            )
            for architecture in ARCHITECTURES
        ]
        if lengths[0] != lengths[1]:
            raise RuntimeError(
                "M4_BLOCKED_RESUME_ARTIFACT_CORRUPT"
            )
        start_repeat = lengths[0]
        if start_repeat > REPEATS:
            raise RuntimeError(
                "M4_BLOCKED_RESUME_ARTIFACT_CORRUPT"
            )

        for repeat_index in range(start_repeat, REPEATS):
            order = (
                (TRANSFORMER, RESIDUAL_MLP)
                if repeat_index in (0, 2, 4)
                else (RESIDUAL_MLP, TRANSFORMER)
            )
            pair = {}
            for architecture in order:
                try:
                    row = _closed_loop_repeat(
                        architecture,
                        progress["checkpoints"][architecture],
                        env_count,
                        repeat_index,
                        precision,
                        device,
                        nvidia_available,
                    )
                except torch.cuda.OutOfMemoryError as exc:
                    raise RuntimeError(
                        "M4_BLOCKED_PERFORMANCE_OOM"
                    ) from exc
                pair[architecture] = row
                print(
                    f"performance closed-loop envs={env_count} "
                    f"repeat={repeat_index} architecture={architecture} "
                    f"decisions/s={row['decisions_per_s']:.1f} "
                    f"wall={row['wall_s']:.3f}s",
                    flush=True,
                )
            _commit_closed_pair(
                progress,
                env_count,
                repeat_index,
                pair,
            )
        _finalize_closed_point(progress, env_count)


def _training_performance(primary: dict) -> dict:
    output = {}
    for architecture in ARCHITECTURES:
        samples = []
        step_medians = []
        step_p95s = []
        peaks = []
        runs = {}
        for seed in TRAINING_SEEDS:
            row = primary["primary_training"][
                architecture
            ][str(seed)]
            run = {
                "training_only_samples_per_s": float(
                    row["training_only_samples_per_s"]
                ),
                "step_time_median_ms": float(
                    row["step_time_median_ms"]
                ),
                "step_time_p95_ms": float(
                    row["step_time_p95_ms"]
                ),
                "peak_vram_bytes": int(
                    row["peak_vram_bytes"]
                ),
            }
            runs[str(seed)] = run
            samples.append(
                run["training_only_samples_per_s"]
            )
            step_medians.append(
                run["step_time_median_ms"]
            )
            step_p95s.append(
                run["step_time_p95_ms"]
            )
            peaks.append(run["peak_vram_bytes"])
        output[architecture] = {
            "runs": runs,
            "training_only_samples_per_s_median": float(
                np.median(samples)
            ),
            "step_time_median_ms": float(
                np.median(step_medians)
            ),
            "step_time_p95_ms": float(
                np.median(step_p95s)
            ),
            "peak_vram_bytes": int(max(peaks)),
        }
    return output


def _strip_internal_fields(progress: dict) -> None:
    for point in progress["model_only"].values():
        for architecture in ARCHITECTURES:
            for key in list(point[architecture]):
                if key.startswith("_"):
                    del point[architecture][key]
    for point in progress["closed_loop"].values():
        for architecture in ARCHITECTURES:
            for key in list(point[architecture]):
                if key.startswith("_"):
                    del point[architecture][key]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.resume:
        raise RuntimeError("M4 performance requires --resume")

    if not torch.cuda.is_available():
        raise RuntimeError("M4_BLOCKED_RUNTIME_ERROR")
    if not SESSION.exists() or not PRIMARY.exists():
        raise RuntimeError(
            "M4_BLOCKED_RESUME_ARTIFACT_CORRUPT"
        )
    session = _read_json(SESSION)
    if session.get("state") != "PERFORMANCE_RUNNING":
        raise RuntimeError(
            "M4_BLOCKED_RESUME_METADATA_MISMATCH"
        )
    primary = _read_json(PRIMARY)
    if (
        primary.get("schema_version") != 1
        or primary.get("work_order_version")
        != WORK_ORDER_VERSION
    ):
        raise RuntimeError(
            "M4_BLOCKED_RESUME_METADATA_MISMATCH"
        )

    progress = _load_or_create_progress(
        primary, session
    )
    if progress.get("status") == "complete":
        print("M4 performance already complete", flush=True)
        return

    device = torch.device("cuda")
    _run_model_only(progress, device)
    _run_closed_loop(progress, device)
    progress["training"] = _training_performance(
        primary
    )
    _strip_internal_fields(progress)
    progress["status"] = "complete"
    atomic_write_json(PERFORMANCE, progress)
    print("M4 performance complete", flush=True)


if __name__ == "__main__":
    main()
