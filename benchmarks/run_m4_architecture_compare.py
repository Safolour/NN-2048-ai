
"""M4 equal-protocol architecture comparison runner (M4_WO_CLOSURE_V2)."""
from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from game2048.m2_models import (
    ResidualMLP2048,
    Transformer2048,
)
from game2048.m2_symmetry import (
    inverse_transform_q_values,
    transform_action_batch,
    transform_board_batch,
)
from game2048.m4_compare import (
    ARCHITECTURES,
    BASE_HEAD,
    BATCH_SIZE,
    CHECKPOINT_SHA256,
    DATASET_SHA256,
    EPOCHS,
    EVALUATION_GAMES,
    EVALUATION_SEED_START,
    GAME_SHARD_SIZE,
    PRIMARY_RUN_ORDER,
    PROMPT_SHA256,
    RESIDUAL_MLP,
    SECONDARY_PLAN_ID,
    TIE_TOLERANCE,
    TRAINING_SEEDS,
    TRANSFORMER,
    WORK_ORDER_VERSION,
    append_game_shard,
    atomic_write_json,
    build_secondary_epoch_plan,
    build_training_plan,
    evaluate_game_shard,
    load_game_results,
    load_training_plan,
    paired_bootstrap,
    save_game_results,
    save_training_plan,
    score_summary,
    select_architecture,
    sha256_file,
    strength_conclusion,
    teacher_best_action,
    test_indices,
    training_indices,
    validate_primary_game_progress,
    validation_indices,
)

DATASET = ROOT / "artifacts" / "m3" / "m3_teacher_validation_8192.npz"
M3_DATASET_META = ROOT / "reports" / "m3" / "m3_teacher_dataset.json"
ARTIFACT = ROOT / "artifacts" / "m4"
PROGRESS = ARTIFACT / "progress"
CHECKPOINTS = ARTIFACT / "checkpoints"
PLANS = ARTIFACT / "plans"
GAMES = ARTIFACT / "games"
SECONDARY = ARTIFACT / "secondary"
REPORTS = ROOT / "reports" / "m4"
SESSION = PROGRESS / "session.json"
PRIMARY_MANIFEST = PROGRESS / "primary.json"
PRIMARY_WORK = PROGRESS / "primary_work.json"
PERFORMANCE_MANIFEST = PROGRESS / "performance.json"
FINALIZE_MANIFEST = PROGRESS / "finalize.json"

MODEL_CLASSES = {
    TRANSFORMER: Transformer2048,
    RESIDUAL_MLP: ResidualMLP2048,
}
PARAMETER_COUNTS = {
    TRANSFORMER: 4_750_342,
    RESIDUAL_MLP: 5_264_710,
}
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
BOOTSTRAP_SEEDS = {
    20260919: 20261101,
    20260920: 20261102,
    20260921: 20261103,
}
AGGREGATE_BOOTSTRAP_SEED = 20261104
SECONDARY_BOOTSTRAP_SEED = 20261105


class SplitCorpus:
    def __init__(
        self,
        canonical_indices: np.ndarray,
        states: np.ndarray,
        teacher_values: np.ndarray,
        legal_mask: np.ndarray,
    ) -> None:
        self.canonical_indices = np.asarray(
            canonical_indices, dtype=np.int64
        )
        self.states = torch.from_numpy(
            np.ascontiguousarray(states, dtype=np.uint8)
        )
        self.teacher_values = torch.from_numpy(
            np.ascontiguousarray(teacher_values, dtype=np.float64)
        )
        self.legal_mask = torch.from_numpy(
            np.ascontiguousarray(legal_mask, dtype=np.bool_)
        )
        self.target = torch.from_numpy(
            teacher_best_action(teacher_values)
        )
        self.row_to_local = {
            int(row): local
            for local, row in enumerate(self.canonical_indices.tolist())
        }


class GpuUtilSampler:
    def __init__(self, available: bool) -> None:
        self.available = bool(available)
        self.util: list[float] = []
        self.memory_mib: list[float] = []
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
            self.memory_mib.append(used)
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

    def samples(self) -> tuple[list[float], list[float]]:
        return list(self.util), list(self.memory_mib)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _session() -> dict:
    payload = _read_json(SESSION)
    expected = {
        "schema_version": 1,
        "work_order_version": WORK_ORDER_VERSION,
        "base_head": BASE_HEAD,
        "prompt_sha256": PROMPT_SHA256,
        "dataset_sha256": DATASET_SHA256,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError("M4_BLOCKED_RESUME_METADATA_MISMATCH")
    return payload


def _load_dataset_without_test_consumption() -> dict:
    if not DATASET.exists():
        raise RuntimeError("M4_BLOCKED_M3_DATASET_MISSING")
    if sha256_file(DATASET) != DATASET_SHA256:
        raise RuntimeError("M4_BLOCKED_M3_DATASET_SHA_MISMATCH")
    with np.load(DATASET, allow_pickle=False) as data:
        arrays = {
            key: np.ascontiguousarray(data[key])
            for key in data.files
        }
    if arrays["state"].shape != (8192, 16):
        raise RuntimeError("M4_BLOCKED_M3_DATASET_METADATA")
    if np.unique(arrays["game_id"]).size != 64:
        raise RuntimeError("M4_BLOCKED_M3_DATASET_METADATA")
    if np.unique(arrays["decision_depth"]).tolist() != [3]:
        raise RuntimeError("M4_BLOCKED_M3_DATASET_METADATA")
    if np.unique(arrays["checkpoint_sha256"]).tolist() != [
        CHECKPOINT_SHA256
    ]:
        raise RuntimeError("M4_BLOCKED_M3_DATASET_METADATA")
    if np.unique(arrays["value_semantics"]).tolist() != [
        "SEARCH_VALUE_RAW_LEAF"
    ]:
        raise RuntimeError("M4_BLOCKED_M3_DATASET_METADATA")
    metadata = _read_json(M3_DATASET_META)
    if metadata.get("canonical_unaugmented") is not True:
        raise RuntimeError("M4_BLOCKED_M3_DATASET_METADATA")
    return arrays


def _make_corpus(
    arrays: dict,
    indices: np.ndarray,
) -> SplitCorpus:
    return SplitCorpus(
        indices,
        arrays["state"][indices],
        arrays["teacher_value"][indices],
        arrays["legal_mask"][indices],
    )


def _set_determinism(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True, warn_only=True)


def _optimizer(model: torch.nn.Module) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1e-8,
        amsgrad=False,
        foreach=False,
        fused=False,
        capturable=False,
    )


def _autocast(precision: str):
    if precision == "BF16 autocast":
        return torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        )
    if precision == "FP32":
        return contextlib.nullcontext()
    raise ValueError(f"unknown precision: {precision}")


def _pairwise_counts(
    logits: torch.Tensor,
    teacher_values: torch.Tensor,
    legal: torch.Tensor,
) -> tuple[int, int]:
    correct = 0
    total = 0
    for action_a in range(4):
        for action_b in range(action_a + 1, 4):
            pair = (
                legal[:, action_a]
                & legal[:, action_b]
                & (
                    teacher_values[:, action_a]
                    != teacher_values[:, action_b]
                )
            )
            if not bool(pair.any()):
                continue
            teacher_sign = (
                teacher_values[pair, action_a]
                > teacher_values[pair, action_b]
            )
            student_sign = (
                logits[pair, action_a] > logits[pair, action_b]
            )
            correct += int((teacher_sign == student_sign).sum().item())
            total += int(pair.sum().item())
    return correct, total


@torch.no_grad()
def _teacher_metrics(
    model: torch.nn.Module,
    corpus: SplitCorpus,
    device: torch.device,
) -> dict:
    model.eval()
    total_loss = 0.0
    total = 0
    raw_correct = 0
    legal_correct = 0
    rank_correct = 0
    rank_total = 0
    for start in range(0, len(corpus.states), 1024):
        end = min(start + 1024, len(corpus.states))
        boards = corpus.states[start:end].to(
            device=device, non_blocking=False
        )
        target = corpus.target[start:end].to(
            device=device, non_blocking=False
        )
        legal = corpus.legal_mask[start:end].to(
            device=device, non_blocking=False
        )
        teacher = corpus.teacher_values[start:end].to(
            device=device, non_blocking=False
        )
        logits = model(boards).float()
        total_loss += float(
            F.cross_entropy(
                logits, target, reduction="sum"
            ).item()
        )
        batch = end - start
        total += batch
        raw_correct += int(
            (logits.argmax(dim=1) == target).sum().item()
        )
        masked = logits.masked_fill(~legal, -torch.inf)
        legal_correct += int(
            (masked.argmax(dim=1) == target).sum().item()
        )
        correct, pairs = _pairwise_counts(
            logits, teacher, legal
        )
        rank_correct += correct
        rank_total += pairs
    return {
        "samples": total,
        "ce": total_loss / total,
        "raw_best_action_accuracy": raw_correct / total,
        "legal_best_action_accuracy": legal_correct / total,
        "legal_pairwise_ranking_accuracy": (
            rank_correct / rank_total
        ),
        "ranking_pairs": rank_total,
    }


@torch.no_grad()
def _d4_test_metrics(
    model: torch.nn.Module,
    corpus: SplitCorpus,
    device: torch.device,
) -> dict:
    model.eval()
    per_transform = {}
    overall_same = 0
    overall_total = 0
    absolute_sum = 0.0
    absolute_count = 0

    for transform_id in range(1, 8):
        transform_same = 0
        transform_total = 0
        for start in range(0, len(corpus.states), 1024):
            end = min(start + 1024, len(corpus.states))
            boards = corpus.states[start:end].to(
                device=device, non_blocking=False
            )
            legal = corpus.legal_mask[start:end].to(
                device=device, non_blocking=False
            )
            canonical = model(boards).float()
            centered = canonical - canonical.mean(
                dim=1, keepdim=True
            )
            canonical_best = canonical.masked_fill(
                ~legal, -torch.inf
            ).argmax(dim=1)

            tids = torch.full(
                (end - start,),
                transform_id,
                dtype=torch.long,
                device=device,
            )
            transformed_boards = transform_board_batch(
                boards, tids
            )
            transformed_q = model(transformed_boards).float()
            restored = inverse_transform_q_values(
                transformed_q, tids
            )
            restored_centered = restored - restored.mean(
                dim=1, keepdim=True
            )
            restored_best = restored.masked_fill(
                ~legal, -torch.inf
            ).argmax(dim=1)
            same = int(
                (restored_best == canonical_best).sum().item()
            )
            batch = end - start
            transform_same += same
            transform_total += batch
            overall_same += same
            overall_total += batch
            absolute_sum += float(
                torch.abs(
                    restored_centered - centered
                ).sum().item()
            )
            absolute_count += int(restored.numel())

        per_transform[str(transform_id)] = (
            transform_same / transform_total
        )

    return {
        "per_transform_legal_argmax_consistency": per_transform,
        "overall_legal_argmax_consistency": (
            overall_same / overall_total
        ),
        "mean_centered_logit_mae": (
            absolute_sum / absolute_count
        ),
        "comparisons": overall_total,
    }


def _prepare_primary_plans(
    train_ids: np.ndarray,
) -> dict[str, dict]:
    PLANS.mkdir(parents=True, exist_ok=True)
    output = {}
    for seed in TRAINING_SEEDS:
        path = PLANS / f"plan_{seed}.npz"
        expected = build_training_plan(train_ids, seed)
        if path.exists():
            observed = load_training_plan(path)
            if (
                observed.sha256 != expected.sha256
                or not np.array_equal(
                    observed.sample_indices,
                    expected.sample_indices,
                )
                or not np.array_equal(
                    observed.transform_ids,
                    expected.transform_ids,
                )
            ):
                raise RuntimeError(
                    "M4_BLOCKED_RESUME_ARTIFACT_CORRUPT"
                )
            plan = observed
        else:
            save_training_plan(path, expected)
            plan = expected
        if plan.sample_indices.shape != (EPOCHS, 6528):
            raise RuntimeError(
                "M4_BLOCKED_RESUME_ARTIFACT_CORRUPT"
            )
        output[str(seed)] = {
            "path": str(path.relative_to(ROOT)),
            "plan_sha256": plan.sha256,
        }
        print(
            f"plan seed={seed} sha256={plan.sha256}",
            flush=True,
        )
    return output


def _batch_from_primary_plan(
    train_corpus: SplitCorpus,
    plan,
    epoch: int,
    start: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    canonical_rows = plan.sample_indices[
        epoch, start : start + BATCH_SIZE
    ]
    local_rows = np.fromiter(
        (
            train_corpus.row_to_local[int(row)]
            for row in canonical_rows
        ),
        dtype=np.int64,
        count=len(canonical_rows),
    )
    tids_np = plan.transform_ids[
        epoch, start : start + BATCH_SIZE
    ].astype(np.int64, copy=False)

    boards = train_corpus.states[local_rows].to(
        device=device, non_blocking=False
    )
    target = train_corpus.target[local_rows].to(
        device=device, non_blocking=False
    )
    tids = torch.from_numpy(
        np.ascontiguousarray(tids_np)
    ).to(device=device, non_blocking=False)
    return (
        transform_board_batch(boards, tids),
        transform_action_batch(target, tids),
    )


def _batch_from_secondary_plan(
    train_corpus: SplitCorpus,
    plan,
    start: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    canonical_rows = plan.sample_indices[
        0, start : start + BATCH_SIZE
    ]
    local_rows = np.fromiter(
        (
            train_corpus.row_to_local[int(row)]
            for row in canonical_rows
        ),
        dtype=np.int64,
        count=len(canonical_rows),
    )
    tids_np = plan.transform_ids[
        0, start : start + BATCH_SIZE
    ].astype(np.int64, copy=False)
    boards = train_corpus.states[local_rows].to(
        device=device, non_blocking=False
    )
    target = train_corpus.target[local_rows].to(
        device=device, non_blocking=False
    )
    tids = torch.from_numpy(
        np.ascontiguousarray(tids_np)
    ).to(device=device, non_blocking=False)
    return (
        transform_board_batch(boards, tids),
        transform_action_batch(target, tids),
    )


def _all_gradients_finite(model: torch.nn.Module) -> bool:
    for parameter in model.parameters():
        if parameter.grad is not None:
            if not bool(
                torch.isfinite(parameter.grad).all().item()
            ):
                return False
    return True


def _all_parameters_finite(model: torch.nn.Module) -> bool:
    for parameter in model.parameters():
        if not bool(torch.isfinite(parameter).all().item()):
            return False
    return True


def _smoke_architecture(
    architecture: str,
    train_corpus: SplitCorpus,
    plan,
    precision: str,
    steps_required: int,
    device: torch.device,
) -> dict:
    _set_determinism(20260919)
    model = MODEL_CLASSES[architecture]().to(device)
    optimizer = _optimizer(model)
    completed = 0
    try:
        model.train()
        for epoch in range(EPOCHS):
            for start in range(0, 6528, BATCH_SIZE):
                boards, target = _batch_from_primary_plan(
                    train_corpus,
                    plan,
                    epoch,
                    start,
                    device,
                )
                optimizer.zero_grad(set_to_none=True)
                with _autocast(precision):
                    logits = model(boards)
                    loss = F.cross_entropy(
                        logits, target, reduction="mean"
                    )
                if not bool(torch.isfinite(loss).item()):
                    return {
                        "status": "FAIL_NONFINITE",
                        "completed_steps": completed,
                    }
                loss.backward()
                if not _all_gradients_finite(model):
                    return {
                        "status": "FAIL_NONFINITE",
                        "completed_steps": completed,
                    }
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), GRAD_CLIP
                )
                optimizer.step()
                if not _all_parameters_finite(model):
                    return {
                        "status": "FAIL_NONFINITE",
                        "completed_steps": completed,
                    }
                completed += 1
                if completed == steps_required:
                    return {
                        "status": "PASS",
                        "completed_steps": completed,
                    }
        raise AssertionError("smoke plan exhausted")
    except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
        return {
            "status": "FAIL_EXCEPTION",
            "completed_steps": completed,
            "error": repr(exc),
        }
    finally:
        del optimizer, model
        gc.collect()
        torch.cuda.empty_cache()


def _precision_decision(
    train_corpus: SplitCorpus,
    plans: dict[str, dict],
    device: torch.device,
) -> dict:
    path = PROGRESS / "precision.json"
    if path.exists():
        payload = _read_json(path)
        if (
            payload.get("work_order_version")
            != WORK_ORDER_VERSION
            or payload.get("dataset_sha256")
            != DATASET_SHA256
            or payload.get("plan_sha256")
            != plans["20260919"]["plan_sha256"]
        ):
            raise RuntimeError(
                "M4_BLOCKED_RESUME_METADATA_MISMATCH"
            )
        return payload

    plan = load_training_plan(
        ROOT / plans["20260919"]["path"]
    )
    bf16_supported = bool(torch.cuda.is_bf16_supported())
    bf16 = {
        TRANSFORMER: {
            "status": "NOT_SUPPORTED",
            "completed_steps": 0,
        },
        RESIDUAL_MLP: {
            "status": "NOT_SUPPORTED",
            "completed_steps": 0,
        },
    }
    fp32 = {
        TRANSFORMER: {
            "status": "NOT_RUN",
            "completed_steps": 0,
        },
        RESIDUAL_MLP: {
            "status": "NOT_RUN",
            "completed_steps": 0,
        },
    }

    if bf16_supported:
        for architecture in ARCHITECTURES:
            bf16[architecture] = _smoke_architecture(
                architecture,
                train_corpus,
                plan,
                "BF16 autocast",
                20,
                device,
            )

    bf16_pass = (
        bf16_supported
        and all(
            row["status"] == "PASS"
            and int(row["completed_steps"]) == 20
            for row in bf16.values()
        )
    )
    fallback = not bf16_pass

    if bf16_pass:
        selected = "BF16 autocast"
    else:
        for architecture in ARCHITECTURES:
            fp32[architecture] = _smoke_architecture(
                architecture,
                train_corpus,
                plan,
                "FP32",
                5,
                device,
            )
        if not all(
            row["status"] == "PASS"
            and int(row["completed_steps"]) == 5
            for row in fp32.values()
        ):
            raise RuntimeError("M4_BLOCKED_NUMERIC_SMOKE")
        selected = "FP32"

    payload = {
        "schema_version": 1,
        "work_order_version": WORK_ORDER_VERSION,
        "dataset_sha256": DATASET_SHA256,
        "plan_sha256": plans["20260919"]["plan_sha256"],
        "bf16_supported": bf16_supported,
        "bf16": bf16,
        "fallback_to_fp32": fallback,
        "fp32": fp32,
        "selected_precision": selected,
    }
    atomic_write_json(path, payload)
    print(
        f"precision selected={selected} "
        f"bf16_supported={bf16_supported}",
        flush=True,
    )
    return payload


def _snapshot_parameters(
    model: torch.nn.Module,
) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
    }


def _atomic_torch_save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _verify_run_metadata(
    payload: dict,
    architecture: str,
    seed: int,
    precision: str,
    plan_sha: str,
) -> None:
    expected = {
        "architecture": architecture,
        "seed": int(seed),
        "precision": precision,
        "dataset_sha256": DATASET_SHA256,
        "plan_sha256": plan_sha,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(
                "M4_BLOCKED_RESUME_METADATA_MISMATCH"
            )


def _parameter_update_checks(
    model: torch.nn.Module,
    initial: dict[str, torch.Tensor],
) -> dict:
    current = {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
    }
    q_changed = any(
        not torch.equal(current[name], initial[name])
        for name in current
        if name.startswith("q_head.")
    )
    backbone_changed = any(
        not torch.equal(current[name], initial[name])
        for name in current
        if not (
            name.startswith("q_head.")
            or name.startswith("value_head.")
            or name.startswith("afterstate_head.")
        )
    )
    value_heads_unchanged = all(
        torch.equal(current[name], initial[name])
        for name in current
        if (
            name.startswith("value_head.")
            or name.startswith("afterstate_head.")
        )
    )
    return {
        "q_head_updated": bool(q_changed),
        "backbone_updated": bool(backbone_changed),
        "value_heads_unchanged": bool(
            value_heads_unchanged
        ),
    }


def _gpu_util_summary(
    available: bool,
    util_samples: list[float],
    memory_samples: list[float],
) -> dict | str:
    if not available:
        return "N/A_NVIDIA_SMI_UNAVAILABLE"
    if not util_samples:
        return "N/A_NO_SAMPLES"
    util = np.asarray(util_samples, dtype=np.float64)
    memory = np.asarray(memory_samples, dtype=np.float64)
    return {
        "samples": int(util.size),
        "mean_utilization_pct": float(util.mean()),
        "p95_utilization_pct": float(
            np.quantile(util, 0.95)
        ),
        "peak_memory_used_mib": float(memory.max()),
    }


def _run_primary_training(
    architecture: str,
    seed: int,
    train_corpus: SplitCorpus,
    validation_corpus: SplitCorpus,
    precision: str,
    plan_info: dict,
    device: torch.device,
    nvidia_available: bool,
) -> dict:
    progress_path = (
        PROGRESS / f"train_{architecture}_{seed}.json"
    )
    latest_path = (
        CHECKPOINTS / f"{architecture}_{seed}_latest.pt"
    )
    final_path = (
        CHECKPOINTS / f"{architecture}_{seed}_final.pt"
    )
    plan = load_training_plan(ROOT / plan_info["path"])
    plan_sha = plan.sha256

    if progress_path.exists():
        progress = _read_json(progress_path)
        _verify_run_metadata(
            progress,
            architecture,
            seed,
            precision,
            plan_sha,
        )
        if progress.get("completed") is True:
            if not final_path.exists():
                raise RuntimeError(
                    "M4_BLOCKED_RESUME_ARTIFACT_CORRUPT"
                )
            observed = sha256_file(final_path)
            if (
                observed
                != progress.get("final_checkpoint_sha256")
            ):
                raise RuntimeError(
                    "M4_BLOCKED_RESUME_ARTIFACT_CORRUPT"
                )
            return progress["summary"]
        if not latest_path.exists():
            raise RuntimeError(
                "M4_BLOCKED_RESUME_ARTIFACT_CORRUPT"
            )
    else:
        progress = {
            "schema_version": 1,
            "work_order_version": WORK_ORDER_VERSION,
            "architecture": architecture,
            "seed": int(seed),
            "precision": precision,
            "dataset_sha256": DATASET_SHA256,
            "plan_sha256": plan_sha,
            "next_epoch": 0,
            "completed": False,
            "history": [],
            "step_times_ms": [],
            "training_compute_wall": 0.0,
            "peak_vram_bytes": 0,
            "gpu_util_samples": [],
            "gpu_memory_samples_mib": [],
        }

    _set_determinism(seed)
    model = MODEL_CLASSES[architecture]().to(device)
    optimizer = _optimizer(model)

    if int(progress["next_epoch"]) > 0:
        checkpoint = torch.load(
            latest_path,
            map_location="cpu",
            weights_only=False,
        )
        _verify_run_metadata(
            checkpoint,
            architecture,
            seed,
            precision,
            plan_sha,
        )
        if int(checkpoint.get("next_epoch", -1)) != int(
            progress["next_epoch"]
        ):
            raise RuntimeError(
                "M4_BLOCKED_RESUME_ARTIFACT_CORRUPT"
            )
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(
            checkpoint["optimizer_state"]
        )
        initial_parameters = checkpoint[
            "initial_parameters"
        ]
    else:
        initial_parameters = _snapshot_parameters(model)
        progress["initial_train_metrics"] = (
            _teacher_metrics(
                model, train_corpus, device
            )
        )
        progress["initial_validation_metrics"] = (
            _teacher_metrics(
                model, validation_corpus, device
            )
        )
        atomic_write_json(progress_path, progress)

    torch.cuda.reset_peak_memory_stats()

    for epoch in range(
        int(progress["next_epoch"]), EPOCHS
    ):
        model.train()
        weighted_loss = 0.0
        seen = 0
        step_events: list[
            tuple[torch.cuda.Event, torch.cuda.Event]
        ] = []
        epoch_sampler = GpuUtilSampler(nvidia_available)

        torch.cuda.synchronize()
        epoch_sampler.start()
        segment_started = time.perf_counter()

        for start in range(0, 6528, BATCH_SIZE):
            boards, target = _batch_from_primary_plan(
                train_corpus,
                plan,
                epoch,
                start,
                device,
            )
            optimizer.zero_grad(set_to_none=True)

            event_start = torch.cuda.Event(
                enable_timing=True
            )
            event_end = torch.cuda.Event(
                enable_timing=True
            )
            event_start.record()
            with _autocast(precision):
                logits = model(boards)
                loss = F.cross_entropy(
                    logits, target, reduction="mean"
                )
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError(
                    "M4_BLOCKED_PRIMARY_NUMERIC"
                )
            loss.backward()
            if not _all_gradients_finite(model):
                raise RuntimeError(
                    "M4_BLOCKED_PRIMARY_NUMERIC"
                )
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), GRAD_CLIP
            )
            optimizer.step()
            event_end.record()
            if not _all_parameters_finite(model):
                raise RuntimeError(
                    "M4_BLOCKED_PRIMARY_NUMERIC"
                )
            step_events.append(
                (event_start, event_end)
            )
            batch_size = int(target.shape[0])
            weighted_loss += (
                float(loss.detach().float().item())
                * batch_size
            )
            seen += batch_size

        torch.cuda.synchronize()
        segment_elapsed = (
            time.perf_counter() - segment_started
        )
        epoch_sampler.stop()
        util_samples, memory_samples = (
            epoch_sampler.samples()
        )
        step_times = [
            float(start_event.elapsed_time(end_event))
            for start_event, end_event in step_events
        ]

        progress["training_compute_wall"] = (
            float(progress["training_compute_wall"])
            + segment_elapsed
        )
        progress["step_times_ms"].extend(step_times)
        progress["peak_vram_bytes"] = max(
            int(progress["peak_vram_bytes"]),
            int(torch.cuda.max_memory_allocated()),
        )
        progress["gpu_util_samples"].extend(
            util_samples
        )
        progress["gpu_memory_samples_mib"].extend(
            memory_samples
        )

        validation_metrics = _teacher_metrics(
            model, validation_corpus, device
        )
        row = {
            "epoch": epoch + 1,
            "train_ce_augmented": (
                weighted_loss / seen
            ),
            "validation_ce": validation_metrics[
                "ce"
            ],
            "validation_legal_best_action_accuracy": (
                validation_metrics[
                    "legal_best_action_accuracy"
                ]
            ),
            "validation_legal_pairwise_ranking_accuracy": (
                validation_metrics[
                    "legal_pairwise_ranking_accuracy"
                ]
            ),
            "training_compute_wall_seconds": (
                segment_elapsed
            ),
            "optimizer_steps": 7,
            "samples": seen,
            "samples_per_second": (
                seen / segment_elapsed
            ),
        }
        progress["history"].append(row)
        progress["next_epoch"] = epoch + 1

        latest_payload = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "initial_parameters": initial_parameters,
            "architecture": architecture,
            "seed": int(seed),
            "precision": precision,
            "dataset_sha256": DATASET_SHA256,
            "plan_sha256": plan_sha,
            "next_epoch": epoch + 1,
        }
        _atomic_torch_save(
            latest_path, latest_payload
        )
        atomic_write_json(progress_path, progress)

        print(
            f"train architecture={architecture} "
            f"seed={seed} epoch={epoch+1}/{EPOCHS} "
            f"train_ce={row['train_ce_augmented']:.6f} "
            f"val_ce={row['validation_ce']:.6f} "
            f"val_acc={row['validation_legal_best_action_accuracy']:.4f} "
            f"elapsed={progress['training_compute_wall']:.3f}s "
            f"samples/s={row['samples_per_second']:.1f}",
            flush=True,
        )

    checks = _parameter_update_checks(
        model, initial_parameters
    )
    if not (
        checks["q_head_updated"]
        and checks["backbone_updated"]
    ):
        raise RuntimeError(
            "M4_BLOCKED_NO_PARAMETER_UPDATE"
        )
    if not checks["value_heads_unchanged"]:
        raise RuntimeError(
            "M4_BLOCKED_VALUE_HEAD_UPDATED"
        )

    final_train = _teacher_metrics(
        model, train_corpus, device
    )
    final_validation = _teacher_metrics(
        model, validation_corpus, device
    )

    final_payload = {
        "model_state": model.state_dict(),
        "architecture": architecture,
        "seed": int(seed),
        "precision": precision,
        "dataset_sha256": DATASET_SHA256,
        "plan_sha256": plan_sha,
    }
    _atomic_torch_save(final_path, final_payload)
    final_sha = sha256_file(final_path)

    step_times_array = np.asarray(
        progress["step_times_ms"], dtype=np.float64
    )
    if step_times_array.size != 210:
        raise RuntimeError(
            "M4_BLOCKED_RESUME_ARTIFACT_CORRUPT"
        )
    total_wall = float(
        progress["training_compute_wall"]
    )
    util_summary = _gpu_util_summary(
        nvidia_available,
        list(progress["gpu_util_samples"]),
        list(progress["gpu_memory_samples_mib"]),
    )

    summary = {
        "architecture": architecture,
        "seed": int(seed),
        "precision": precision,
        "plan_sha256": plan_sha,
        "final_checkpoint_path": str(
            final_path.relative_to(ROOT)
        ),
        "final_checkpoint_sha256": final_sha,
        "total_training_compute_wall": total_wall,
        "training_only_samples_per_s": (
            6528 * EPOCHS / total_wall
        ),
        "step_time_mean_ms": float(
            step_times_array.mean()
        ),
        "step_time_median_ms": float(
            np.median(step_times_array)
        ),
        "step_time_p95_ms": float(
            np.quantile(step_times_array, 0.95)
        ),
        "peak_vram_bytes": int(
            progress["peak_vram_bytes"]
        ),
        "gpu_utilization": util_summary,
        "initial_train_metrics": progress[
            "initial_train_metrics"
        ],
        "initial_validation_metrics": progress[
            "initial_validation_metrics"
        ],
        "final_train_metrics": final_train,
        "final_validation_metrics": final_validation,
        "d4_consistency": "DEFERRED_TO_§11",
        "parameter_update_checks": checks,
    }

    progress.update(
        {
            "completed": True,
            "final_checkpoint_path": summary[
                "final_checkpoint_path"
            ],
            "final_checkpoint_sha256": final_sha,
            "total_training_compute_wall": total_wall,
            "step_time_mean_ms": summary[
                "step_time_mean_ms"
            ],
            "step_time_median_ms": summary[
                "step_time_median_ms"
            ],
            "step_time_p95_ms": summary[
                "step_time_p95_ms"
            ],
            "peak_vram_bytes": summary[
                "peak_vram_bytes"
            ],
            "final_train_metrics": final_train,
            "final_validation_metrics": (
                final_validation
            ),
            "summary": summary,
        }
    )
    atomic_write_json(progress_path, progress)

    del optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def _load_final_model(
    run: dict,
    device: torch.device,
) -> torch.nn.Module:
    path = ROOT / run["final_checkpoint_path"]
    if not path.exists():
        raise RuntimeError(
            "M4_BLOCKED_RESUME_ARTIFACT_CORRUPT"
        )
    if sha256_file(path) != run[
        "final_checkpoint_sha256"
    ]:
        raise RuntimeError(
            "M4_BLOCKED_RESUME_ARTIFACT_CORRUPT"
        )
    payload = torch.load(
        path, map_location="cpu", weights_only=False
    )
    expected = {
        "architecture": run["architecture"],
        "seed": int(run["seed"]),
        "precision": run["precision"],
        "dataset_sha256": DATASET_SHA256,
        "plan_sha256": run["plan_sha256"],
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(
                "M4_BLOCKED_RESUME_METADATA_MISMATCH"
            )
    model = MODEL_CLASSES[
        run["architecture"]
    ]()
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    return model


def _nested_runs(
    flat_runs: dict[str, dict],
) -> dict:
    output = {
        TRANSFORMER: {},
        RESIDUAL_MLP: {},
    }
    for architecture in ARCHITECTURES:
        for seed in TRAINING_SEEDS:
            key = f"{architecture}_{seed}"
            output[architecture][str(seed)] = (
                flat_runs[key]
            )
    return output


def _teacher_and_d4_after_all_runs(
    arrays: dict,
    flat_runs: dict[str, dict],
    train_corpus: SplitCorpus,
    validation_corpus: SplitCorpus,
    device: torch.device,
    work: dict,
) -> tuple[dict, dict]:
    if (
        "teacher_metrics" in work
        and "d4" in work
    ):
        return work["teacher_metrics"], work["d4"]

    test_ids = test_indices(arrays["split"])
    if test_ids.size != 896:
        raise RuntimeError(
            "M4_BLOCKED_M3_DATASET_METADATA"
        )
    test_corpus = _make_corpus(arrays, test_ids)

    teacher = {
        TRANSFORMER: {},
        RESIDUAL_MLP: {},
    }
    d4 = {
        TRANSFORMER: {},
        RESIDUAL_MLP: {},
    }
    for architecture, seed in PRIMARY_RUN_ORDER:
        key = f"{architecture}_{seed}"
        model = _load_final_model(
            flat_runs[key], device
        )
        metrics = {
            "train": _teacher_metrics(
                model, train_corpus, device
            ),
            "validation": _teacher_metrics(
                model, validation_corpus, device
            ),
            "test": _teacher_metrics(
                model, test_corpus, device
            ),
        }
        d4_metrics = _d4_test_metrics(
            model, test_corpus, device
        )
        teacher[architecture][str(seed)] = metrics
        d4[architecture][str(seed)] = d4_metrics
        del model
        gc.collect()
        torch.cuda.empty_cache()
        print(
            f"teacher+d4 complete architecture={architecture} "
            f"seed={seed} test_acc="
            f"{metrics['test']['legal_best_action_accuracy']:.4f}",
            flush=True,
        )

    work["teacher_metrics"] = teacher
    work["d4"] = d4
    atomic_write_json(PRIMARY_WORK, work)
    return teacher, d4


def _game_correctness_gate(
    flat_runs: dict[str, dict],
    device: torch.device,
    work: dict,
) -> dict:
    if "game_correctness_gate" in work:
        return work["game_correctness_gate"]

    seeds = np.arange(
        20261001, 20261033, dtype=np.int64
    )
    result = {}
    for architecture in ARCHITECTURES:
        key = f"{architecture}_20260919"
        model = _load_final_model(
            flat_runs[key], device
        )
        first = evaluate_game_shard(
            model,
            seeds,
            device,
            verify_reference=True,
        )
        second = evaluate_game_shard(
            model,
            seeds,
            device,
            verify_reference=False,
        )
        for field in (
            "game_seed",
            "final_score",
            "max_tile_exp",
            "moves",
        ):
            if not np.array_equal(
                first[field], second[field]
            ):
                raise RuntimeError(
                    "M4_BLOCKED_GAME_EVALUATOR_CORRECTNESS"
                )
        result[architecture] = {
            "seeds": [
                int(value) for value in seeds
            ],
            "repeat_deterministic": True,
            "fast_vs_reference": True,
        }
        del model
        gc.collect()
        torch.cuda.empty_cache()
        print(
            f"game correctness gate PASS "
            f"architecture={architecture}",
            flush=True,
        )

    work["game_correctness_gate"] = result
    atomic_write_json(PRIMARY_WORK, work)
    return result


def _evaluate_primary_games(
    architecture: str,
    seed: int,
    run: dict,
    device: torch.device,
) -> dict:
    path = GAMES / f"{architecture}_{seed}.npz"
    existing = (
        load_game_results(path)
        if path.exists()
        else None
    )
    completed_shards = validate_primary_game_progress(
        existing
    )
    model = _load_final_model(run, device)
    started = time.perf_counter()

    for shard_id in range(
        completed_shards,
        EVALUATION_GAMES // GAME_SHARD_SIZE,
    ):
        seed_start = (
            EVALUATION_SEED_START
            + shard_id * GAME_SHARD_SIZE
        )
        shard_seeds = np.arange(
            seed_start,
            seed_start + GAME_SHARD_SIZE,
            dtype=np.int64,
        )
        shard_started = time.perf_counter()
        shard = evaluate_game_shard(
            model, shard_seeds, device
        )
        existing = append_game_shard(
            existing, shard
        )
        save_game_results(path, existing)
        completed_games = int(
            existing["game_seed"].size
        )
        shard_elapsed = (
            time.perf_counter() - shard_started
        )
        overall_elapsed = (
            time.perf_counter() - started
        )
        print(
            f"games architecture={architecture} seed={seed} "
            f"completed_shards={shard_id+1}/20 "
            f"completed_games={completed_games}/2000 "
            f"elapsed={overall_elapsed:.2f}s "
            f"games/s={GAME_SHARD_SIZE/shard_elapsed:.3f} "
            f"running_mean="
            f"{float(np.mean(existing['final_score'])):.1f}",
            flush=True,
        )

    del model
    gc.collect()
    torch.cuda.empty_cache()

    if existing is None:
        raise RuntimeError(
            "M4_BLOCKED_RESUME_ARTIFACT_CORRUPT"
        )
    if validate_primary_game_progress(existing) != 20:
        raise RuntimeError(
            "M4_BLOCKED_RESUME_ARTIFACT_CORRUPT"
        )
    return {
        "raw_npz_path": str(path.relative_to(ROOT)),
        "raw_npz_sha256": sha256_file(path),
        "complete_games": 2000,
        "complete_shards": 20,
        "summary": score_summary(
            existing["final_score"],
            existing["max_tile_exp"],
            existing["moves"],
        ),
    }


def _all_primary_game_evaluations(
    flat_runs: dict[str, dict],
    device: torch.device,
    work: dict,
) -> dict:
    output = work.setdefault(
        "game_evaluation",
        {TRANSFORMER: {}, RESIDUAL_MLP: {}},
    )
    for architecture, seed in PRIMARY_RUN_ORDER:
        seed_key = str(seed)
        if seed_key in output.get(
            architecture, {}
        ):
            row = output[architecture][seed_key]
            path = ROOT / row["raw_npz_path"]
            if (
                not path.exists()
                or sha256_file(path)
                != row["raw_npz_sha256"]
            ):
                raise RuntimeError(
                    "M4_BLOCKED_RESUME_ARTIFACT_CORRUPT"
                )
            results = load_game_results(path)
            if (
                validate_primary_game_progress(
                    results
                )
                != 20
            ):
                raise RuntimeError(
                    "M4_BLOCKED_RESUME_ARTIFACT_CORRUPT"
                )
            continue

        key = f"{architecture}_{seed}"
        row = _evaluate_primary_games(
            architecture,
            seed,
            flat_runs[key],
            device,
        )
        output.setdefault(architecture, {})[
            seed_key
        ] = row
        atomic_write_json(PRIMARY_WORK, work)
    return output


def _paired_statistics(
    game_evaluation: dict,
) -> dict:
    per_seed = []
    transformer_scores = []
    mlp_scores = []

    for seed in TRAINING_SEEDS:
        transformer_path = (
            ROOT
            / game_evaluation[TRANSFORMER][
                str(seed)
            ]["raw_npz_path"]
        )
        mlp_path = (
            ROOT
            / game_evaluation[RESIDUAL_MLP][
                str(seed)
            ]["raw_npz_path"]
        )
        transformer = load_game_results(
            transformer_path
        )
        mlp = load_game_results(mlp_path)
        if not np.array_equal(
            transformer["game_seed"],
            mlp["game_seed"],
        ):
            raise RuntimeError(
                "M4_BLOCKED_RESUME_ARTIFACT_CORRUPT"
            )
        row = paired_bootstrap(
            transformer["final_score"],
            mlp["final_score"],
            seed=BOOTSTRAP_SEEDS[seed],
            resamples=10_000,
        )
        row["training_seed"] = int(seed)
        per_seed.append(row)
        transformer_scores.append(
            transformer["final_score"].astype(
                np.float64
            )
        )
        mlp_scores.append(
            mlp["final_score"].astype(np.float64)
        )

    aggregate_transformer = np.mean(
        np.stack(transformer_scores), axis=0
    )
    aggregate_mlp = np.mean(
        np.stack(mlp_scores), axis=0
    )
    aggregate = paired_bootstrap(
        aggregate_transformer,
        aggregate_mlp,
        seed=AGGREGATE_BOOTSTRAP_SEED,
        resamples=10_000,
    )
    strength = strength_conclusion(
        per_seed, aggregate
    )
    return {
        "per_training_seed": per_seed,
        "aggregate": aggregate,
        "strength_conclusion": strength,
    }


def _primary_phase() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("M4_BLOCKED_RUNTIME_ERROR")
    session = _session()
    if session["state"] != "PRIMARY_RUNNING":
        raise RuntimeError(
            "M4_BLOCKED_RESUME_METADATA_MISMATCH"
        )
    if PRIMARY_MANIFEST.exists():
        manifest = _read_json(PRIMARY_MANIFEST)
        if (
            manifest.get("schema_version") == 1
            and manifest.get("work_order_version")
            == WORK_ORDER_VERSION
        ):
            print(
                "primary manifest already complete",
                flush=True,
            )
            return
        raise RuntimeError(
            "M4_BLOCKED_RESUME_ARTIFACT_CORRUPT"
        )

    ARTIFACT.mkdir(parents=True, exist_ok=True)
    PROGRESS.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(
        parents=True, exist_ok=True
    )
    PLANS.mkdir(parents=True, exist_ok=True)
    GAMES.mkdir(parents=True, exist_ok=True)

    arrays = _load_dataset_without_test_consumption()
    train_ids = training_indices(arrays["split"])
    validation_ids = validation_indices(
        arrays["split"]
    )
    if train_ids.size != 6528:
        raise RuntimeError(
            "M4_BLOCKED_M3_DATASET_METADATA"
        )
    if validation_ids.size != 768:
        raise RuntimeError(
            "M4_BLOCKED_M3_DATASET_METADATA"
        )

    train_corpus = _make_corpus(
        arrays, train_ids
    )
    validation_corpus = _make_corpus(
        arrays, validation_ids
    )
    device = torch.device("cuda")
    plans = _prepare_primary_plans(train_ids)
    precision = _precision_decision(
        train_corpus, plans, device
    )
    nvidia_available = bool(
        session.get("nvidia_smi_available", False)
    )

    flat_runs = {}
    for architecture, seed in PRIMARY_RUN_ORDER:
        key = f"{architecture}_{seed}"
        run = _run_primary_training(
            architecture,
            seed,
            train_corpus,
            validation_corpus,
            precision["selected_precision"],
            plans[str(seed)],
            device,
            nvidia_available,
        )
        flat_runs[key] = run

    if PRIMARY_WORK.exists():
        work = _read_json(PRIMARY_WORK)
        if (
            work.get("work_order_version")
            != WORK_ORDER_VERSION
            or work.get("dataset_sha256")
            != DATASET_SHA256
        ):
            raise RuntimeError(
                "M4_BLOCKED_RESUME_METADATA_MISMATCH"
            )
    else:
        work = {
            "schema_version": 1,
            "work_order_version": WORK_ORDER_VERSION,
            "dataset_sha256": DATASET_SHA256,
        }

    teacher_metrics, d4 = (
        _teacher_and_d4_after_all_runs(
            arrays,
            flat_runs,
            train_corpus,
            validation_corpus,
            device,
            work,
        )
    )
    _game_correctness_gate(
        flat_runs, device, work
    )
    game_evaluation = (
        _all_primary_game_evaluations(
            flat_runs, device, work
        )
    )
    if "paired_statistics" in work:
        paired_statistics = work[
            "paired_statistics"
        ]
    else:
        paired_statistics = _paired_statistics(
            game_evaluation
        )
        work["paired_statistics"] = (
            paired_statistics
        )
        atomic_write_json(PRIMARY_WORK, work)

    manifest = {
        "schema_version": 1,
        "work_order_version": WORK_ORDER_VERSION,
        "dataset": {
            "path": str(DATASET.relative_to(ROOT)),
            "sha256": DATASET_SHA256,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "states": 8192,
            "games": 64,
            "split_counts": {
                "train": {
                    "games": 51,
                    "states": 6528,
                },
                "validation": {
                    "games": 6,
                    "states": 768,
                },
                "test": {
                    "games": 7,
                    "states": 896,
                },
            },
        },
        "precision": precision,
        "plans": plans,
        "primary_training": _nested_runs(
            flat_runs
        ),
        "teacher_metrics": teacher_metrics,
        "d4": d4,
        "game_evaluation": game_evaluation,
        "paired_statistics": paired_statistics,
    }
    atomic_write_json(PRIMARY_MANIFEST, manifest)
    if PRIMARY_WORK.exists():
        PRIMARY_WORK.unlink()
    print(
        "primary complete strength="
        + paired_statistics["strength_conclusion"],
        flush=True,
    )


def _secondary_progress_paths(
    architecture: str,
) -> tuple[Path, Path, Path]:
    return (
        SECONDARY
        / f"{architecture}_progress.json",
        SECONDARY / f"{architecture}_latest.pt",
        SECONDARY / f"{architecture}_final.pt",
    )


def _run_secondary_training(
    architecture: str,
    train_corpus: SplitCorpus,
    validation_corpus: SplitCorpus,
    precision: str,
    common_budget: float,
    device: torch.device,
) -> tuple[torch.nn.Module, dict]:
    progress_path, latest_path, final_path = (
        _secondary_progress_paths(architecture)
    )
    SECONDARY.mkdir(
        parents=True, exist_ok=True
    )

    if progress_path.exists():
        progress = _read_json(progress_path)
        expected = {
            "architecture": architecture,
            "training_seed": 20260919,
            "precision": precision,
            "dataset_sha256": DATASET_SHA256,
            "secondary_plan_id": SECONDARY_PLAN_ID,
        }
        for key, value in expected.items():
            if progress.get(key) != value:
                raise RuntimeError(
                    "M4_BLOCKED_RESUME_METADATA_MISMATCH"
                )
        if progress.get("completed") is True:
            if not final_path.exists():
                raise RuntimeError(
                    "M4_BLOCKED_RESUME_ARTIFACT_CORRUPT"
                )
            if (
                sha256_file(final_path)
                != progress[
                    "final_checkpoint_sha256"
                ]
            ):
                raise RuntimeError(
                    "M4_BLOCKED_RESUME_ARTIFACT_CORRUPT"
                )
            model = MODEL_CLASSES[
                architecture
            ]().to(device)
            payload = torch.load(
                final_path,
                map_location="cpu",
                weights_only=False,
            )
            model.load_state_dict(
                payload["model_state"]
            )
            return model, progress["summary"]
        if not latest_path.exists():
            raise RuntimeError(
                "M4_BLOCKED_RESUME_ARTIFACT_CORRUPT"
            )
    else:
        progress = {
            "schema_version": 1,
            "architecture": architecture,
            "training_seed": 20260919,
            "precision": precision,
            "dataset_sha256": DATASET_SHA256,
            "secondary_plan_id": SECONDARY_PLAN_ID,
            "next_epoch": 0,
            "completed_steps": 0,
            "training_only_wall": 0.0,
            "completed": False,
        }

    _set_determinism(20260919)
    model = MODEL_CLASSES[architecture]().to(device)
    optimizer = _optimizer(model)

    if int(progress["next_epoch"]) > 0:
        checkpoint = torch.load(
            latest_path,
            map_location="cpu",
            weights_only=False,
        )
        expected = {
            "architecture": architecture,
            "training_seed": 20260919,
            "precision": precision,
            "dataset_sha256": DATASET_SHA256,
            "secondary_plan_id": SECONDARY_PLAN_ID,
        }
        for key, value in expected.items():
            if checkpoint.get(key) != value:
                raise RuntimeError(
                    "M4_BLOCKED_RESUME_METADATA_MISMATCH"
                )
        if int(checkpoint["next_epoch"]) != int(
            progress["next_epoch"]
        ):
            raise RuntimeError(
                "M4_BLOCKED_RESUME_ARTIFACT_CORRUPT"
            )
        model.load_state_dict(
            checkpoint["model_state"]
        )
        optimizer.load_state_dict(
            checkpoint["optimizer_state"]
        )

    train_ids = train_corpus.canonical_indices
    while (
        float(progress["training_only_wall"])
        < common_budget
    ):
        epoch = int(progress["next_epoch"])
        plan = build_secondary_epoch_plan(
            train_ids, epoch
        )
        model.train()
        torch.cuda.synchronize()
        started = time.perf_counter()
        for start in range(0, 6528, BATCH_SIZE):
            boards, target = _batch_from_secondary_plan(
                train_corpus,
                plan,
                start,
                device,
            )
            optimizer.zero_grad(set_to_none=True)
            with _autocast(precision):
                logits = model(boards)
                loss = F.cross_entropy(
                    logits, target, reduction="mean"
                )
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError(
                    "M4_BLOCKED_PRIMARY_NUMERIC"
                )
            loss.backward()
            if not _all_gradients_finite(model):
                raise RuntimeError(
                    "M4_BLOCKED_PRIMARY_NUMERIC"
                )
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), GRAD_CLIP
            )
            optimizer.step()
            if not _all_parameters_finite(model):
                raise RuntimeError(
                    "M4_BLOCKED_PRIMARY_NUMERIC"
                )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started

        progress["training_only_wall"] = (
            float(progress["training_only_wall"])
            + elapsed
        )
        progress["completed_steps"] = (
            int(progress["completed_steps"]) + 7
        )
        progress["next_epoch"] = epoch + 1

        latest = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "architecture": architecture,
            "training_seed": 20260919,
            "precision": precision,
            "dataset_sha256": DATASET_SHA256,
            "secondary_plan_id": SECONDARY_PLAN_ID,
            "next_epoch": epoch + 1,
            "completed_steps": progress[
                "completed_steps"
            ],
            "training_only_wall": progress[
                "training_only_wall"
            ],
        }
        _atomic_torch_save(
            latest_path, latest
        )
        atomic_write_json(
            progress_path, progress
        )
        print(
            f"secondary architecture={architecture} "
            f"epoch={epoch+1} "
            f"steps={progress['completed_steps']} "
            f"wall={progress['training_only_wall']:.3f}/"
            f"{common_budget:.3f}s",
            flush=True,
        )

    validation = _teacher_metrics(
        model, validation_corpus, device
    )
    final_payload = {
        "model_state": model.state_dict(),
        "architecture": architecture,
        "training_seed": 20260919,
        "precision": precision,
        "dataset_sha256": DATASET_SHA256,
        "secondary_plan_id": SECONDARY_PLAN_ID,
    }
    _atomic_torch_save(final_path, final_payload)
    final_sha = sha256_file(final_path)
    summary = {
        "architecture": architecture,
        "training_seed": 20260919,
        "precision": precision,
        "secondary_plan_id": SECONDARY_PLAN_ID,
        "budget_seconds": float(common_budget),
        "actual_training_only_wall": float(
            progress["training_only_wall"]
        ),
        "completed_steps": int(
            progress["completed_steps"]
        ),
        "completed_epochs": int(
            progress["next_epoch"]
        ),
        "validation_teacher_metrics": validation,
        "final_checkpoint_path": str(
            final_path.relative_to(ROOT)
        ),
        "final_checkpoint_sha256": final_sha,
    }
    progress.update(
        {
            "completed": True,
            "final_checkpoint_sha256": final_sha,
            "final_checkpoint_path": summary[
                "final_checkpoint_path"
            ],
            "summary": summary,
        }
    )
    atomic_write_json(progress_path, progress)
    del optimizer
    return model, summary


def _secondary_games(
    model: torch.nn.Module,
    architecture: str,
    device: torch.device,
) -> dict:
    games_dir = SECONDARY / "games"
    games_dir.mkdir(
        parents=True, exist_ok=True
    )
    path = games_dir / f"{architecture}.npz"
    existing = (
        load_game_results(path)
        if path.exists()
        else None
    )
    completed_shards = validate_primary_game_progress(
        existing
    )
    started = time.perf_counter()
    for shard_id in range(completed_shards, 20):
        seed_start = (
            EVALUATION_SEED_START
            + shard_id * GAME_SHARD_SIZE
        )
        seeds = np.arange(
            seed_start,
            seed_start + GAME_SHARD_SIZE,
            dtype=np.int64,
        )
        shard_started = time.perf_counter()
        shard = evaluate_game_shard(
            model, seeds, device
        )
        existing = append_game_shard(
            existing, shard
        )
        save_game_results(path, existing)
        elapsed = time.perf_counter() - shard_started
        print(
            f"secondary games architecture={architecture} "
            f"completed_shards={shard_id+1}/20 "
            f"elapsed={time.perf_counter()-started:.2f}s "
            f"games/s={GAME_SHARD_SIZE/elapsed:.3f} "
            f"running_mean={float(np.mean(existing['final_score'])):.1f}",
            flush=True,
        )
    if existing is None:
        raise RuntimeError(
            "M4_BLOCKED_RESUME_ARTIFACT_CORRUPT"
        )
    return {
        "raw_npz_path": str(path.relative_to(ROOT)),
        "raw_npz_sha256": sha256_file(path),
        "summary": score_summary(
            existing["final_score"],
            existing["max_tile_exp"],
            existing["moves"],
        ),
    }


def _equal_wall_secondary(
    primary: dict,
    wall_ratio: float,
    device: torch.device,
) -> dict:
    if wall_ratio <= 1.20:
        return {
            "status": "NOT_TRIGGERED",
            "wall_ratio": float(wall_ratio),
        }

    arrays = _load_dataset_without_test_consumption()
    train_ids = training_indices(arrays["split"])
    validation_ids = validation_indices(
        arrays["split"]
    )
    train_corpus = _make_corpus(
        arrays, train_ids
    )
    validation_corpus = _make_corpus(
        arrays, validation_ids
    )
    precision = primary["precision"][
        "selected_precision"
    ]
    t_walls = [
        float(
            primary["primary_training"][
                TRANSFORMER
            ][str(seed)][
                "total_training_compute_wall"
            ]
        )
        for seed in TRAINING_SEEDS
    ]
    m_walls = [
        float(
            primary["primary_training"][
                RESIDUAL_MLP
            ][str(seed)][
                "total_training_compute_wall"
            ]
        )
        for seed in TRAINING_SEEDS
    ]
    common_budget = max(
        float(np.median(t_walls)),
        float(np.median(m_walls)),
    )

    rows = {}
    models = {}
    for architecture in ARCHITECTURES:
        model, summary = _run_secondary_training(
            architecture,
            train_corpus,
            validation_corpus,
            precision,
            common_budget,
            device,
        )
        models[architecture] = model
        rows[architecture] = summary

    for architecture in ARCHITECTURES:
        games = _secondary_games(
            models[architecture],
            architecture,
            device,
        )
        rows[architecture][
            "game_evaluation"
        ] = games
        del models[architecture]
        gc.collect()
        torch.cuda.empty_cache()

    t_results = load_game_results(
        ROOT
        / rows[TRANSFORMER][
            "game_evaluation"
        ]["raw_npz_path"]
    )
    m_results = load_game_results(
        ROOT
        / rows[RESIDUAL_MLP][
            "game_evaluation"
        ]["raw_npz_path"]
    )
    paired = paired_bootstrap(
        t_results["final_score"],
        m_results["final_score"],
        seed=SECONDARY_BOOTSTRAP_SEED,
        resamples=10_000,
    )
    return {
        "status": "TRIGGERED_COMPLETE",
        "secondary_plan_id": SECONDARY_PLAN_ID,
        "common_budget_seconds": float(
            common_budget
        ),
        TRANSFORMER: rows[TRANSFORMER],
        RESIDUAL_MLP: rows[RESIDUAL_MLP],
        "paired_delta": paired,
    }


def _finalize_phase() -> None:
    session = _session()
    if session["state"] != "FINALIZE_RUNNING":
        raise RuntimeError(
            "M4_BLOCKED_RESUME_METADATA_MISMATCH"
        )
    if not PRIMARY_MANIFEST.exists():
        raise RuntimeError(
            "M4_BLOCKED_RESUME_ARTIFACT_CORRUPT"
        )
    if not PERFORMANCE_MANIFEST.exists():
        raise RuntimeError(
            "M4_BLOCKED_RESUME_ARTIFACT_CORRUPT"
        )
    if FINALIZE_MANIFEST.exists():
        existing = _read_json(FINALIZE_MANIFEST)
        if existing.get("schema_version") == 1:
            print(
                "finalize manifest already complete",
                flush=True,
            )
            return
        raise RuntimeError(
            "M4_BLOCKED_RESUME_ARTIFACT_CORRUPT"
        )

    primary = _read_json(PRIMARY_MANIFEST)
    performance = _read_json(
        PERFORMANCE_MANIFEST
    )
    if (
        primary.get("work_order_version")
        != WORK_ORDER_VERSION
        or performance.get("work_order_version")
        != WORK_ORDER_VERSION
    ):
        raise RuntimeError(
            "M4_BLOCKED_RESUME_METADATA_MISMATCH"
        )

    t_walls = [
        float(
            primary["primary_training"][
                TRANSFORMER
            ][str(seed)][
                "total_training_compute_wall"
            ]
        )
        for seed in TRAINING_SEEDS
    ]
    m_walls = [
        float(
            primary["primary_training"][
                RESIDUAL_MLP
            ][str(seed)][
                "total_training_compute_wall"
            ]
        )
        for seed in TRAINING_SEEDS
    ]
    t_median = float(np.median(t_walls))
    m_median = float(np.median(m_walls))
    wall_ratio = max(t_median, m_median) / min(
        t_median, m_median
    )

    device = torch.device("cuda")
    equal_wall = _equal_wall_secondary(
        primary, wall_ratio, device
    )

    strength = primary["paired_statistics"][
        "strength_conclusion"
    ]
    transformer_8192 = float(
        performance["closed_loop"]["8192"][
            TRANSFORMER
        ]["median_decisions_per_s"]
    )
    mlp_8192 = float(
        performance["closed_loop"]["8192"][
            RESIDUAL_MLP
        ]["median_decisions_per_s"]
    )
    selection = select_architecture(
        strength,
        transformer_8192,
        mlp_8192,
    )

    payload = {
        "schema_version": 1,
        "wall_ratio": float(wall_ratio),
        "equal_wall_clock_secondary": equal_wall,
        "strength_conclusion": strength,
        "speed_ratio": selection["speed_ratio"],
        "selected_architecture": selection[
            "selected_architecture"
        ],
        "selection_rule": selection[
            "selection_rule"
        ],
    }
    atomic_write_json(
        FINALIZE_MANIFEST, payload
    )
    print(
        f"finalize selected={payload['selected_architecture']} "
        f"strength={strength} "
        f"rule={payload['selection_rule']} "
        f"speed_ratio={payload['speed_ratio']:.6f}",
        flush=True,
    )


def _pytest_counts(path: Path) -> dict:
    root = ET.parse(path).getroot()
    if root.tag == "testsuite":
        suites = [root]
    else:
        suites = list(root.findall("testsuite"))

    def _sum(name: str) -> int:
        if name in root.attrib:
            return int(root.attrib[name])
        return sum(
            int(suite.attrib.get(name, 0))
            for suite in suites
        )

    tests = _sum("tests")
    failures = _sum("failures")
    errors = _sum("errors")
    skipped = _sum("skipped")
    return {
        "tests": tests,
        "passed": tests
        - failures
        - errors
        - skipped,
        "failures": failures,
        "errors": errors,
        "skipped": skipped,
        "xfailed": 0,
    }


def _report_markdown(summary: dict) -> str:
    game_lines = []
    for architecture in ARCHITECTURES:
        for seed in TRAINING_SEEDS:
            row = summary["game_evaluation"][
                architecture
            ][str(seed)]["summary"]
            game_lines.append(
                f"- {architecture}/{seed}: "
                f"mean={row['mean_score']:.3f}, "
                f"median={row['median_score']:.3f}, "
                f"p10={row['p10_score']:.3f}, "
                f"p90={row['p90_score']:.3f}"
            )

    regression = summary["regression"]
    selection = summary["selection"]
    lines = [
        "# M4 Transformer vs ResidualMLP Candidate Report",
        "",
        "## 1. RESULT",
        "",
        "**M4_CANDIDATE_EVIDENCE_COMPLETE**",
        "",
        "This is candidate evidence, not M4 audited/frozen status.",
        "",
        "## 2. STARTING STATE",
        "",
        f"- base HEAD: `{BASE_HEAD}`",
        f"- work order: `{WORK_ORDER_VERSION}`",
        f"- prompt SHA-256: `{PROMPT_SHA256}`",
        "",
        "## 3. FROZEN VERIFICATION",
        "",
        "M0/M1/M2/M3 authoritative tags were verified at FRESH P0. Frozen M2 source diff gate was zero.",
        "",
        "## 4. DATASET VERIFICATION",
        "",
        f"- dataset SHA-256: `{DATASET_SHA256}`",
        "- 8192 states / 64 games",
        "- train 51 games / 6528 states",
        "- validation 6 games / 768 states",
        "- test 7 games / 896 states",
        "- value semantics: `SEARCH_VALUE_RAW_LEAF`",
        "",
        "## 5. MODEL DEFINITIONS / PARAM COUNTS",
        "",
        f"- Transformer2048: {PARAMETER_COUNTS[TRANSFORMER]:,}",
        f"- ResidualMLP2048: {PARAMETER_COUNTS[RESIDUAL_MLP]:,}",
        "",
        "## 6. FAIRNESS CONTRACT",
        "",
        "Same frozen dataset, deterministic D4 plans, three paired training seeds, 30 epochs / 210 optimizer steps per run, AdamW 3e-4, batch 1024, no architecture-specific tuning.",
        "",
        "## 7. PRECISION DECISION",
        "",
        f"Selected primary precision: `{summary['precision']['selected_precision']}`.",
        "",
        "## 8. PRIMARY TRAINING RESULTS",
        "",
        "All six epoch-30 final checkpoints completed. Full per-run training metrics, checkpoint SHA values and timing are in `m4_architecture_compare.json`.",
        "",
        "## 9. VALIDATION / TEST TEACHER METRICS",
        "",
        "Canonical FP32 train/validation/test Teacher metrics were measured for all six final models only after all primary runs completed.",
        "",
        "## 10. GREEDY GAME SCORE RESULTS",
        "",
        *game_lines,
        "",
        "## 11. PAIRED STATISTICS",
        "",
        f"Strength conclusion: **{summary['paired_statistics']['strength_conclusion']}**",
        f"Aggregate paired result: `{summary['paired_statistics']['aggregate']}`",
        "",
        "## 12. D4 DIAGNOSTIC",
        "",
        "D4 legal-argmax consistency by transform id, overall consistency and centered-logit MAE were measured on the complete 896-state canonical test split.",
        "",
        "## 13. INFERENCE PERFORMANCE",
        "",
        "Required eager model-only batches 1/256/1024/2048/4096/8192 completed for both architectures.",
        "",
        "## 14. TRAINING PERFORMANCE",
        "",
        "Training-only samples/s, CUDA-event step timings and peak VRAM were recorded for all six primary runs.",
        "",
        "## 15. CLOSED-LOOP PERFORMANCE",
        "",
        "Required 2048-env and 8192-env closed-loop points completed with five paired-order repeats per architecture.",
        "",
        "## 16. EQUAL-WALL-CLOCK SECONDARY",
        "",
        f"Status: `{summary['equal_wall_clock_secondary']['status']}`",
        "",
        "## 17. ARCHITECTURE SELECTION",
        "",
        f"- selected: **{selection['selected_architecture']}**",
        f"- strength: **{selection['strength_conclusion']}**",
        f"- selection_rule: **{selection['selection_rule']}**",
        f"- speed_ratio: {selection['speed_ratio']:.6f}",
        "",
        "## 18. PYTEST",
        "",
        f"{regression['passed']} passed / {regression['failures']} failed / {regression['errors']} errors / {regression['skipped']} skipped / {regression['xfailed']} xfailed.",
        "",
        "## 19. CI",
        "",
        "- candidate_sha=PENDING_BY_DESIGN",
        "- candidate_ci=PENDING_BY_DESIGN",
        "- closeout_commit_sha=SELF_NOT_EMBEDDABLE_BY_DESIGN",
        "- closeout_ci=PENDING_BY_DESIGN",
        "",
        "## 20. FILES CHANGED",
        "",
        "Only the six M4 tracked paths allowed by the work order.",
        "",
        "## 21. ARTIFACTS / CLEANUP",
        "",
        "Primary final checkpoints, plans, raw game NPZ files, progress/session manifests, performance evidence and pytest.xml are retained. Runtime latest/temp/lock files are removed before P10.",
        "",
        "## 22. FINAL GIT STATE",
        "",
        "Candidate commit is intentionally pending at candidate report generation time.",
        "",
        "## 23. REMAINING BLOCKERS",
        "",
        "Local candidate evidence is complete. Candidate CI and mandatory report-only closeout CI remain before `M4 CANDIDATE COMPLETE`.",
        "",
    ]
    return "\\n".join(lines)


def _report_phase() -> None:
    session = _session()
    if session["state"] != "P10_DONE":
        raise RuntimeError(
            "M4_BLOCKED_RESUME_METADATA_MISMATCH"
        )
    pytest_xml = ARTIFACT / "pytest.xml"
    for path in (
        PRIMARY_MANIFEST,
        PERFORMANCE_MANIFEST,
        FINALIZE_MANIFEST,
        pytest_xml,
    ):
        if not path.exists():
            raise RuntimeError(
                "M4_BLOCKED_RESUME_ARTIFACT_CORRUPT"
            )

    primary = _read_json(PRIMARY_MANIFEST)
    performance = _read_json(
        PERFORMANCE_MANIFEST
    )
    finalize = _read_json(
        FINALIZE_MANIFEST
    )
    regression = _pytest_counts(pytest_xml)
    if regression != {
        "tests": 533,
        "passed": 533,
        "failures": 0,
        "errors": 0,
        "skipped": 0,
        "xfailed": 0,
    }:
        raise RuntimeError(
            "M4_BLOCKED_RUNTIME_ERROR"
        )

    summary = {
        "schema_version": 1,
        "result": "M4_CANDIDATE_EVIDENCE_COMPLETE",
        "dataset": primary["dataset"],
        "models": {
            TRANSFORMER: {
                "parameter_count": PARAMETER_COUNTS[
                    TRANSFORMER
                ]
            },
            RESIDUAL_MLP: {
                "parameter_count": PARAMETER_COUNTS[
                    RESIDUAL_MLP
                ]
            },
        },
        "fairness": {
            "work_order_version": WORK_ORDER_VERSION,
            "prompt_sha256": session[
                "prompt_sha256"
            ],
            "base_head": session["base_head"],
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "optimizer": "torch.optim.AdamW",
            "lr": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "grad_clip": GRAD_CLIP,
            "training_seeds": list(
                TRAINING_SEEDS
            ),
            "evaluation_seed_start": (
                EVALUATION_SEED_START
            ),
            "evaluation_games": EVALUATION_GAMES,
            "game_shard_size": GAME_SHARD_SIZE,
            "tie_tolerance": TIE_TOLERANCE,
        },
        "precision": primary["precision"],
        "plans": primary["plans"],
        "primary_training": primary[
            "primary_training"
        ],
        "teacher_metrics": primary[
            "teacher_metrics"
        ],
        "game_evaluation": primary[
            "game_evaluation"
        ],
        "paired_statistics": primary[
            "paired_statistics"
        ],
        "d4": primary["d4"],
        "performance": {
            "model_only": performance[
                "model_only"
            ],
            "closed_loop": performance[
                "closed_loop"
            ],
            "training": performance["training"],
        },
        "equal_wall_clock_secondary": finalize[
            "equal_wall_clock_secondary"
        ],
        "selection": {
            "strength_conclusion": finalize[
                "strength_conclusion"
            ],
            "selected_architecture": finalize[
                "selected_architecture"
            ],
            "selection_rule": finalize[
                "selection_rule"
            ],
            "speed_ratio": finalize[
                "speed_ratio"
            ],
        },
        "regression": regression,
    }

    expected_top = {
        "schema_version",
        "result",
        "dataset",
        "models",
        "fairness",
        "precision",
        "plans",
        "primary_training",
        "teacher_metrics",
        "game_evaluation",
        "paired_statistics",
        "d4",
        "performance",
        "equal_wall_clock_secondary",
        "selection",
        "regression",
    }
    if set(summary) != expected_top:
        raise RuntimeError(
            "M4_BLOCKED_RUNTIME_ERROR"
        )

    REPORTS.mkdir(
        parents=True, exist_ok=True
    )
    json_path = (
        REPORTS / "m4_architecture_compare.json"
    )
    report_path = REPORTS / "M4_REPORT.md"
    json_path.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    report_path.write_text(
        _report_markdown(summary),
        encoding="utf-8",
    )
    print(
        f"report written {json_path} {report_path}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        required=True,
        choices=("primary", "finalize", "report"),
    )
    parser.add_argument(
        "--resume", action="store_true"
    )
    args = parser.parse_args()

    if not args.resume:
        raise RuntimeError(
            "M4 phases require --resume under V2"
        )

    if args.phase == "primary":
        _primary_phase()
    elif args.phase == "finalize":
        _finalize_phase()
    else:
        _report_phase()


if __name__ == "__main__":
    main()
