"""M5 Teacher policy-pretraining orchestrator."""
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

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from game2048.m2_models import ResidualMLP2048, trainable_parameter_count
from game2048.m2_symmetry import transform_action_batch, transform_board_batch
from game2048.m5_pretrain import (
    ARCHITECTURE, BASE_HEAD, BATCH_SIZE, BOOTSTRAP_SEEDS, DEV_GAMES,
    DEV_SEED_START, EPOCHS, FINAL_GAMES, FINAL_SEED_START,
    GAME_SHARD_SIZE, M4_PRIMARY_SHA256, PARAMETER_COUNT,
    PROMPT_SHA256, SCALE_ROWS, TEACHER_SHA256, TEACHER_VERSION,
    TRAINING_SEEDS, WORK_ORDER_VERSION, aggregate_seed_scores,
    atomic_save_npz, atomic_torch_save, atomic_write_json,
    autocast_context, d4_metrics, evaluate_game_shard,
    high_tile_metrics, load_label_split, load_test_split,
    make_optimizer, median_validation, paired_bootstrap,
    parameter_update_checks, plan_id, plan_sha256,
    random_legal_baseline, scale_gate_131k, score_summary,
    select_with_524k, set_determinism, sha256_file,
    snapshot_parameters, stateless_epoch_plan, teacher_best_action,
    teacher_metrics, validate_manifest_completed_shards,
)

ART = ROOT / "artifacts" / "m5"
PROGRESS = ART / "progress"
CHECKPOINTS = ART / "checkpoints"
GAMES = ART / "games"
FINAL = ART / "final"
DATA = ART / "datasets"
REPORTS = ROOT / "reports" / "m5"
SESSION = PROGRESS / "session.json"
LABEL_MANIFEST = DATA / "label_manifest.json"
PRECISION = PROGRESS / "precision.json"
DEVELOPMENT = PROGRESS / "development.json"
SCALE_GATE = PROGRESS / "scale_gate.json"
SELECTION = PROGRESS / "scale_selection.json"
TEACHER_TEST = PROGRESS / "teacher_test.json"
FINAL_GAMEPLAY = PROGRESS / "final_gameplay.json"
SEARCH_PROFILE = ART / "search_profile.json"
M4_PRIMARY = ROOT / "artifacts" / "m4" / "progress" / "primary.json"
M4_GAME_PATHS = [
    ROOT / "artifacts" / "m4" / "games" / f"ResidualMLP2048_{seed}.npz"
    for seed in (20260919, 20260920, 20260921)
]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _session() -> dict:
    row = _read_json(SESSION)
    expected = {
        "schema_version": 1,
        "work_order_version": WORK_ORDER_VERSION,
        "base_head": BASE_HEAD,
        "prompt_sha256": PROMPT_SHA256,
        "teacher_sha256": TEACHER_SHA256,
        "m4_primary_sha256": M4_PRIMARY_SHA256,
    }
    for key, value in expected.items():
        if row.get(key) != value:
            raise RuntimeError("M5_BLOCKED_RESUME_METADATA_MISMATCH")
    return row


def _label_manifest() -> dict:
    row = _read_json(LABEL_MANIFEST)
    validate_manifest_completed_shards(row, ROOT)
    return row


def _scale_train_split(scale: str) -> dict[str, np.ndarray]:
    shards = {"32k": 16, "131k": 64, "524k": 256}[scale]
    return load_label_split(ROOT, _label_manifest(), "train", max_shards=shards)
def _validation_split() -> dict[str, np.ndarray]:
    return load_label_split(
        ROOT,
        _label_manifest(),
        "validation",
    )


def _batch(
    split: dict[str, np.ndarray],
    order: np.ndarray,
    tids: np.ndarray,
    start: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = order[start:start + BATCH_SIZE]
    boards = torch.from_numpy(
        np.ascontiguousarray(split["state"][rows])
    ).to(device)
    targets = torch.from_numpy(
        teacher_best_action(
            split["teacher_value"][rows],
            split["legal_mask"][rows],
        )
    ).to(device)
    transform_ids = torch.from_numpy(
        np.ascontiguousarray(tids[start:start + len(rows)], dtype=np.int64)
    ).to(device)
    return (
        transform_board_batch(boards, transform_ids),
        transform_action_batch(targets, transform_ids),
    )


def _finite_gradients(model: torch.nn.Module) -> bool:
    return all(
        parameter.grad is None
        or bool(torch.isfinite(parameter.grad).all().item())
        for parameter in model.parameters()
    )
def _finite_parameters(model: torch.nn.Module) -> bool:
    return all(
        bool(torch.isfinite(parameter).all().item())
        for parameter in model.parameters()
    )


def _smoke(
    precision: str,
    steps: int,
    train_split: dict[str, np.ndarray],
    device: torch.device,
) -> dict:
    set_determinism(TRAINING_SEEDS[0])
    model = ResidualMLP2048().to(device)
    optimizer = make_optimizer(model)
    completed = 0
    try:
        order, tids = stateless_epoch_plan(
            "32k",
            TRAINING_SEEDS[0],
            0,
        )
        model.train()
        for start in range(0, steps * BATCH_SIZE, BATCH_SIZE):
            boards, target = _batch(
                train_split,
                order,
                tids,
                start,
                device,
            )
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(precision):
                logits = model(boards)
                loss = F.cross_entropy(
                    logits,
                    target,
                    reduction="mean",
                )
            if not bool(torch.isfinite(loss).item()):
                return {"status": "FAIL_NONFINITE", "completed_steps": completed}
            loss.backward()
            if not _finite_gradients(model):
                return {
                    "status": "FAIL_NONFINITE",
                    "completed_steps": completed,
                }
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )
            optimizer.step()
            if not _finite_parameters(model):
                return {
                    "status": "FAIL_NONFINITE",
                    "completed_steps": completed,
                }
            completed += 1
        return {
            "status": "PASS",
            "completed_steps": completed,
        }
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


def _precision_phase(device: torch.device) -> dict:
    session = _session()
    if session["state"] not in {"DATA_32K_DONE", "TRAIN_32K_RUNNING"}:
        raise RuntimeError("M5_BLOCKED_RESUME_GIT_STATE")
    if PRECISION.exists():
        return _read_json(PRECISION)
    train_split = _scale_train_split("32k")
    bf16_supported = bool(torch.cuda.is_bf16_supported())
    bf16 = {
        "status": "NOT_SUPPORTED",
        "completed_steps": 0,
    }
    if bf16_supported:
        bf16 = _smoke(
            "BF16 autocast",
            20,
            train_split,
            device,
        )
    if (
        bf16_supported
        and bf16["status"] == "PASS"
        and int(bf16["completed_steps"]) == 20
    ):
        selected = "BF16 autocast"
        fp32 = {
            "status": "NOT_RUN",
            "completed_steps": 0,
        }
    else:
        fp32 = _smoke(
            "FP32",
            5,
            train_split,
            device,
        )
        if not (
            fp32["status"] == "PASS"
            and int(fp32["completed_steps"]) == 5
        ):
            raise RuntimeError("M5_BLOCKED_NUMERIC_SMOKE")
        selected = "FP32"
    payload = {
        "schema_version": 1,
        "work_order_version": WORK_ORDER_VERSION,
        "bf16_supported": bf16_supported,
        "bf16": bf16,
        "fp32": fp32,
        "selected_precision": selected,
    }
    atomic_write_json(PRECISION, payload)
    print(
        f"precision selected={selected} "
        f"bf16_supported={bf16_supported}",
        flush=True,
    )
    return payload


def _progress_path(scale: str, seed: int) -> Path:
    return PROGRESS / f"train_{scale}_{seed}.json"


def _latest_path(scale: str, seed: int) -> Path:
    return CHECKPOINTS / scale / f"{seed}_latest.pt"


def _final_path(scale: str, seed: int) -> Path:
    return CHECKPOINTS / scale / f"{seed}_final.pt"


def _dataset_manifest_sha() -> str:
    return sha256_file(LABEL_MANIFEST)


def _verify_checkpoint_meta(
    payload: dict,
    scale: str,
    seed: int,
    precision: str,
) -> None:
    expected = {
        "architecture": ARCHITECTURE,
        "scale": scale,
        "seed": int(seed),
        "precision": precision,
        "teacher_sha256": TEACHER_SHA256,
        "plan_id": plan_id(scale, seed),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError("M5_BLOCKED_RESUME_METADATA_MISMATCH")
def _train_one(
    scale: str,
    seed: int,
    train_split: dict[str, np.ndarray],
    validation: dict[str, np.ndarray],
    precision: str,
    device: torch.device,
) -> dict:
    progress_path = _progress_path(scale, seed)
    latest_path = _latest_path(scale, seed)
    final_path = _final_path(scale, seed)
    plan_name = plan_id(scale, seed)
    plan_hash = plan_sha256(scale, seed)

    if progress_path.exists():
        progress = _read_json(progress_path)
        for key, value in {
            "scale": scale,
            "seed": int(seed),
            "precision": precision,
            "plan_id": plan_name,
            "plan_sha256": plan_hash,
        }.items():
            if progress.get(key) != value:
                raise RuntimeError("M5_BLOCKED_RESUME_METADATA_MISMATCH")
        if progress.get("completed") is True:
            if (
                not final_path.exists()
                or sha256_file(final_path)
                != progress["final_checkpoint_sha256"]
            ):
                raise RuntimeError("M5_BLOCKED_RESUME_ARTIFACT_CORRUPT")
            return progress["summary"]
        if int(progress["next_epoch"]) > 0 and not latest_path.exists():
            raise RuntimeError("M5_BLOCKED_RESUME_ARTIFACT_CORRUPT")
    else:
        progress = {
            "schema_version": 1,
            "work_order_version": WORK_ORDER_VERSION,
            "architecture": ARCHITECTURE,
            "scale": scale,
            "seed": int(seed),
            "precision": precision,
            "teacher_sha256": TEACHER_SHA256,
            "dataset_manifest_sha256": _dataset_manifest_sha(),
            "plan_id": plan_name,
            "plan_sha256": plan_hash,
            "next_epoch": 0,
            "completed": False,
            "history": [],
            "training_compute_wall": 0.0,
            "peak_vram_bytes": 0,
        }

    set_determinism(seed)
    model = ResidualMLP2048().to(device)
    if trainable_parameter_count(model) != PARAMETER_COUNT:
        raise RuntimeError("M5_BLOCKED_MODEL_DRIFT")
    optimizer = make_optimizer(model)

    if int(progress["next_epoch"]) > 0:
        checkpoint = torch.load(
            latest_path,
            map_location="cpu",
            weights_only=False,
        )
        _verify_checkpoint_meta(
            checkpoint,
            scale,
            seed,
            precision,
        )
        if int(checkpoint.get("next_epoch", -1)) != int(progress["next_epoch"]):
            raise RuntimeError("M5_BLOCKED_RESUME_ARTIFACT_CORRUPT")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        initial_parameters = checkpoint["initial_parameters"]
    else:
        initial_parameters = snapshot_parameters(model)
        progress["initial_validation_metrics"] = teacher_metrics(
            model,
            validation["state"],
            validation["teacher_value"],
            validation["legal_mask"],
            device,
        )
        atomic_write_json(progress_path, progress)
    torch.cuda.reset_peak_memory_stats()
    rows = SCALE_ROWS[scale]
    steps_per_epoch = rows // BATCH_SIZE

    for epoch in range(int(progress["next_epoch"]), EPOCHS):
        order, tids = stateless_epoch_plan(scale, seed, epoch)
        model.train()
        weighted_loss = 0.0
        seen = 0
        torch.cuda.synchronize()
        started = time.perf_counter()
        for start in range(0, rows, BATCH_SIZE):
            boards, target = _batch(
                train_split,
                order,
                tids,
                start,
                device,
            )
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(precision):
                logits = model(boards)
                loss = F.cross_entropy(
                    logits,
                    target,
                    reduction="mean",
                )
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError("M5_BLOCKED_NUMERIC")
            loss.backward()
            if not _finite_gradients(model):
                raise RuntimeError("M5_BLOCKED_NUMERIC")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if not _finite_parameters(model):
                raise RuntimeError("M5_BLOCKED_NUMERIC")
            batch = int(target.shape[0])
            weighted_loss += float(loss.detach().float().item()) * batch
            seen += batch
        torch.cuda.synchronize()
        epoch_wall = time.perf_counter() - started
        progress["training_compute_wall"] = (
            float(progress["training_compute_wall"]) + epoch_wall
        )
        progress["peak_vram_bytes"] = max(
            int(progress["peak_vram_bytes"]),
            int(torch.cuda.max_memory_allocated()),
        )
        validation_metrics = teacher_metrics(
            model,
            validation["state"],
            validation["teacher_value"],
            validation["legal_mask"],
            device,
        )
        row = {
            "epoch": epoch + 1,
            "train_ce_augmented": float(weighted_loss / seen),
            "validation_ce": validation_metrics["ce"],
            "validation_legal_best_action_accuracy": (
                validation_metrics["legal_best_action_accuracy"]
            ),
            "validation_legal_pairwise_ranking_accuracy": (
                validation_metrics["legal_pairwise_ranking_accuracy"]
            ),
            "training_compute_wall_seconds": float(epoch_wall),
            "optimizer_steps": int(steps_per_epoch),
            "samples": int(seen),
            "samples_per_second": float(seen / epoch_wall),
            "peak_vram_bytes": int(progress["peak_vram_bytes"]),
        }
        progress["history"].append(row)
        progress["next_epoch"] = epoch + 1
        latest_payload = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "initial_parameters": initial_parameters,
            "architecture": ARCHITECTURE,
            "scale": scale,
            "seed": int(seed),
            "precision": precision,
            "teacher_sha256": TEACHER_SHA256,
            "dataset_manifest_sha256": progress["dataset_manifest_sha256"],
            "plan_id": plan_name,
            "plan_sha256": plan_hash,
            "next_epoch": epoch + 1,
        }
        atomic_torch_save(latest_path, latest_payload)
        atomic_write_json(progress_path, progress)
        print(
            f"train architecture={ARCHITECTURE} scale={scale} "
            f"seed={seed} epoch={epoch + 1}/{EPOCHS} "
            f"train_ce={row['train_ce_augmented']:.6f} "
            f"val_ce={row['validation_ce']:.6f} "
            f"val_acc={row['validation_legal_best_action_accuracy']:.4f} "
            f"elapsed={progress['training_compute_wall']:.1f}s "
            f"samples/s={row['samples_per_second']:.1f} "
            f"peak_vram={row['peak_vram_bytes']}",
            flush=True,
        )

    checks = parameter_update_checks(model, initial_parameters)
    if not (checks["q_head_updated"] and checks["backbone_updated"]):
        raise RuntimeError("M5_BLOCKED_NO_PARAMETER_UPDATE")
    if not checks["value_heads_unchanged"]:
        raise RuntimeError("M5_BLOCKED_VALUE_HEAD_UPDATED")

    final_validation = teacher_metrics(
        model,
        validation["state"],
        validation["teacher_value"],
        validation["legal_mask"],
        device,
    )
    final_d4 = d4_metrics(
        model,
        validation["state"],
        validation["legal_mask"],
        device,
    )
    strata = high_tile_metrics(model, validation, device)
    final_payload = {
        "architecture": ARCHITECTURE,
        "scale": scale,
        "seed": int(seed),
        "precision": precision,
        "teacher_sha256": TEACHER_SHA256,
        "dataset_manifest_sha256": progress["dataset_manifest_sha256"],
        "plan_id": plan_name,
        "plan_sha256": plan_hash,
        "epoch": EPOCHS,
        "model_state": model.state_dict(),
    }
    atomic_torch_save(final_path, final_payload)
    final_sha = sha256_file(final_path)
    total_wall = float(progress["training_compute_wall"])
    summary = {
        "architecture": ARCHITECTURE,
        "scale": scale,
        "seed": int(seed),
        "precision": precision,
        "plan_id": plan_name,
        "plan_sha256": plan_hash,
        "final_checkpoint_path": str(final_path.relative_to(ROOT)).replace("\\", "/"),
        "final_checkpoint_sha256": final_sha,
        "epochs": EPOCHS,
        "optimizer_steps": int(steps_per_epoch * EPOCHS),
        "training_compute_wall_seconds": total_wall,
        "training_only_samples_per_s": float(rows * EPOCHS / total_wall),
        "peak_vram_bytes": int(progress["peak_vram_bytes"]),
        "parameter_update_checks": checks,
        "validation_metrics": final_validation,
        "d4": final_d4,
        "high_tile_strata": strata,
    }
    progress.update({
        "completed": True,
        "final_checkpoint_sha256": final_sha,
        "summary": summary,
    })
    atomic_write_json(progress_path, progress)
    if latest_path.exists():
        latest_path.unlink()
    del optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def _load_final_model(
    scale: str,
    seed: int,
    device: torch.device,
) -> tuple[torch.nn.Module, dict]:
    progress = _read_json(_progress_path(scale, seed))
    if progress.get("completed") is not True:
        raise RuntimeError("M5_BLOCKED_RESUME_ARTIFACT_CORRUPT")
    summary = progress["summary"]
    path = ROOT / summary["final_checkpoint_path"]
    if (
        not path.exists()
        or sha256_file(path) != summary["final_checkpoint_sha256"]
    ):
        raise RuntimeError("M5_BLOCKED_RESUME_ARTIFACT_CORRUPT")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    _verify_checkpoint_meta(
        payload,
        scale,
        seed,
        progress["precision"],
    )
    if int(payload.get("epoch", -1)) != EPOCHS:
        raise RuntimeError("M5_BLOCKED_RESUME_ARTIFACT_CORRUPT")
    model = ResidualMLP2048()
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    return model, summary
def _load_game_file(path: Path) -> dict[str, np.ndarray] | None:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        required = {"game_seed", "final_score", "max_tile_exp", "moves"}
        if set(data.files) != required:
            raise RuntimeError("M5_BLOCKED_RESUME_ARTIFACT_CORRUPT")
        result = {
            "game_seed": np.ascontiguousarray(data["game_seed"], dtype=np.int64),
            "final_score": np.ascontiguousarray(data["final_score"], dtype=np.int64),
            "max_tile_exp": np.ascontiguousarray(data["max_tile_exp"], dtype=np.uint8),
            "moves": np.ascontiguousarray(data["moves"], dtype=np.int64),
        }
    count = len(result["game_seed"])
    if any(value.shape != (count,) for value in result.values()):
        raise RuntimeError("M5_BLOCKED_RESUME_ARTIFACT_CORRUPT")
    return result


def _append_game(
    existing: dict[str, np.ndarray] | None,
    shard: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    if existing is None:
        return {key: np.ascontiguousarray(value) for key, value in shard.items()}
    return {
        key: np.ascontiguousarray(
            np.concatenate([existing[key], shard[key]])
        )
        for key in existing
    }


def _save_game(path: Path, result: dict[str, np.ndarray]) -> None:
    atomic_save_npz(
        path,
        game_seed=result["game_seed"],
        final_score=result["final_score"],
        max_tile_exp=result["max_tile_exp"],
        moves=result["moves"],
    )
def _evaluate_dev(
    scale: str,
    seed: int,
    model: torch.nn.Module,
    device: torch.device,
) -> dict:
    path = GAMES / scale / f"{seed}.npz"
    result = _load_game_file(path)
    completed = 0 if result is None else len(result["game_seed"])
    if completed % GAME_SHARD_SIZE != 0 or completed > DEV_GAMES:
        raise RuntimeError("M5_BLOCKED_RESUME_ARTIFACT_CORRUPT")
    expected_prefix = np.arange(
        DEV_SEED_START,
        DEV_SEED_START + completed,
        dtype=np.int64,
    )
    if completed and not np.array_equal(result["game_seed"], expected_prefix):
        raise RuntimeError("M5_BLOCKED_RESUME_ARTIFACT_CORRUPT")
    started = time.perf_counter()
    while completed < DEV_GAMES:
        take = min(GAME_SHARD_SIZE, DEV_GAMES - completed)
        seeds = np.arange(
            DEV_SEED_START + completed,
            DEV_SEED_START + completed + take,
            dtype=np.int64,
        )
        shard = evaluate_game_shard(model, seeds, device)
        result = _append_game(result, shard)
        completed = len(result["game_seed"])
        _save_game(path, result)
        elapsed = time.perf_counter() - started
        print(
            f"dev scale={scale} seed={seed} "
            f"games={completed}/{DEV_GAMES} "
            f"elapsed={elapsed:.1f}s "
            f"rate={completed / elapsed:.1f} games/s",
            flush=True,
        )
    summary = score_summary(
        result["final_score"],
        result["max_tile_exp"],
        result["moves"],
    )
    return {
        "scale": scale,
        "seed": int(seed),
        "raw_path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "raw_sha256": sha256_file(path),
        "summary": summary,
    }


def _development() -> dict:
    if DEVELOPMENT.exists():
        return _read_json(DEVELOPMENT)
    return {
        "schema_version": 1,
        "work_order_version": WORK_ORDER_VERSION,
        "m4_baseline": None,
        "scales": {},
    }


def _save_development(row: dict) -> None:
    atomic_write_json(DEVELOPMENT, row)


def _m4_aggregate() -> tuple[np.ndarray, dict]:
    primary = _read_json(M4_PRIMARY)
    if sha256_file(M4_PRIMARY) != M4_PRIMARY_SHA256:
        raise RuntimeError("M5_BLOCKED_M4_BASELINE_EVIDENCE")
    rows = primary["game_evaluation"]["ResidualMLP2048"]
    scores = []
    seeds_ref = None
    summaries = {}
    for seed in ("20260919", "20260920", "20260921"):
        info = rows[seed]
        path = ROOT / info["raw_npz_path"]
        if sha256_file(path) != info["raw_npz_sha256"]:
            raise RuntimeError("M5_BLOCKED_M4_BASELINE_EVIDENCE")
        game = _load_game_file(path)
        expected = np.arange(DEV_SEED_START, DEV_SEED_START + DEV_GAMES)
        if game is None or not np.array_equal(game["game_seed"], expected):
            raise RuntimeError("M5_BLOCKED_M4_BASELINE_EVIDENCE")
        scores.append(game["final_score"].astype(np.float64))
        summaries[seed] = info["summary"]
        if seeds_ref is None:
            seeds_ref = game["game_seed"].copy()
        elif not np.array_equal(seeds_ref, game["game_seed"]):
            raise RuntimeError("M5_BLOCKED_M4_BASELINE_EVIDENCE")
    aggregate = np.stack(scores).mean(axis=0)
    return aggregate, {
        "game_seeds": [int(seeds_ref[0]), int(seeds_ref[-1])],
        "games": DEV_GAMES,
        "mean_score": float(aggregate.mean()),
        "source_seed_summaries": summaries,
        "primary_sha256": M4_PRIMARY_SHA256,
    }


def _scale_score_vector(scale: str, development: dict) -> np.ndarray:
    vectors = []
    for seed in TRAINING_SEEDS:
        row = development["scales"][scale]["runs"][str(seed)]
        path = ROOT / row["raw_path"]
        if sha256_file(path) != row["raw_sha256"]:
            raise RuntimeError("M5_BLOCKED_RESUME_ARTIFACT_CORRUPT")
        game = _load_game_file(path)
        expected = np.arange(DEV_SEED_START, DEV_SEED_START + DEV_GAMES)
        if game is None or not np.array_equal(game["game_seed"], expected):
            raise RuntimeError("M5_BLOCKED_RESUME_ARTIFACT_CORRUPT")
        vectors.append(game["final_score"])
    return aggregate_seed_scores(vectors)


def _run_scale(
    scale: str,
    device: torch.device,
) -> None:
    session = _session()
    expected_state = {
        "32k": {"DATA_32K_DONE", "TRAIN_32K_RUNNING"},
        "131k": {"DATA_131K_DONE", "TRAIN_131K_RUNNING"},
        "524k": {"DATA_524K_DONE", "TRAIN_524K_RUNNING"},
    }[scale]
    if session["state"] not in expected_state:
        raise RuntimeError("M5_BLOCKED_RESUME_GIT_STATE")
    precision = _read_json(PRECISION)["selected_precision"]
    train_split = _scale_train_split(scale)
    validation = _validation_split()
    if len(train_split["state"]) != SCALE_ROWS[scale]:
        raise RuntimeError("M5_BLOCKED_DATASET_ARTIFACT_CORRUPT")
    running_state = f"TRAIN_{scale.upper()}_RUNNING"
    done_state = f"TRAIN_{scale.upper()}_DONE"
    session["state"] = running_state
    atomic_write_json(SESSION, session)

    development = _development()
    development["scales"].setdefault(scale, {"runs": {}})
    for seed in TRAINING_SEEDS:
        summary = _train_one(
            scale,
            seed,
            train_split,
            validation,
            precision,
            device,
        )
        model, _loaded = _load_final_model(
            scale,
            seed,
            device,
        )
        game = _evaluate_dev(
            scale,
            seed,
            model,
            device,
        )
        development["scales"][scale]["runs"][str(seed)] = {
            **game,
            "checkpoint_path": summary["final_checkpoint_path"],
            "checkpoint_sha256": summary["final_checkpoint_sha256"],
            "validation_metrics": summary["validation_metrics"],
            "d4": summary["d4"],
            "high_tile_strata": summary["high_tile_strata"],
        }
        _save_development(development)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    score_vector = _scale_score_vector(scale, development)
    validation_rows = [
        development["scales"][scale]["runs"][str(seed)]["validation_metrics"]
        for seed in TRAINING_SEEDS
    ]
    development["scales"][scale]["aggregate"] = {
        "mean_score": float(score_vector.mean()),
        "median_validation": median_validation(validation_rows),
    }
    if development.get("m4_baseline") is None:
        _m4_scores, m4_summary = _m4_aggregate()
        development["m4_baseline"] = m4_summary
    _save_development(development)
    session["state"] = done_state
    atomic_write_json(SESSION, session)
    print(
        f"training scale={scale} complete "
        f"aggregate_mean={score_vector.mean():.3f}",
        flush=True,
    )


def _scale_gate_phase() -> dict:
    session = _session()
    if session["state"] not in {"TRAIN_131K_DONE", "SCALE_GATE_DONE"}:
        raise RuntimeError("M5_BLOCKED_RESUME_GIT_STATE")
    if SCALE_GATE.exists():
        payload = _read_json(SCALE_GATE)
        if session["state"] != "SCALE_GATE_DONE":
            session["state"] = "SCALE_GATE_DONE"
            atomic_write_json(SESSION, session)
        return payload
    development = _development()
    scores32 = _scale_score_vector("32k", development)
    scores131 = _scale_score_vector("131k", development)
    med32 = development["scales"]["32k"]["aggregate"]["median_validation"]
    med131 = development["scales"]["131k"]["aggregate"]["median_validation"]
    gate = scale_gate_131k(scores32, scores131, med32, med131)
    m4_scores, _ = _m4_aggregate()
    gate["m4_vs_32k"] = paired_bootstrap(
        scores32,
        m4_scores,
        seed=BOOTSTRAP_SEEDS["m4_vs_32k"],
    )
    gate["status_524k"] = (
        "TRIGGERED"
        if gate["trigger_524k"]
        else "NOT_TRIGGERED"
    )
    atomic_write_json(SCALE_GATE, gate)
    session["state"] = "SCALE_GATE_DONE"
    atomic_write_json(SESSION, session)
    print(json.dumps(gate, indent=2), flush=True)
    return gate


def _median_high_tile(
    scale: str,
    development: dict,
) -> dict:
    output = {}
    for threshold in (">=11", ">=12", ">=13"):
        rows = [
            development["scales"][scale]["runs"][str(seed)][
                "high_tile_strata"
            ][threshold]
            for seed in TRAINING_SEEDS
        ]
        samples = int(rows[0]["samples"])
        if any(int(row["samples"]) != samples for row in rows):
            raise RuntimeError("M5_BLOCKED_RESUME_ARTIFACT_CORRUPT")
        if samples == 0:
            output[threshold] = {"samples": 0}
            continue
        output[threshold] = {
            "samples": samples,
            "random_legal_baseline": float(
                np.median([row["random_legal_baseline"] for row in rows])
            ),
            "median_legal_best_action_accuracy": float(
                np.median([row["legal_best_action_accuracy"] for row in rows])
            ),
            "median_pairwise_ranking_accuracy": float(
                np.median([row["legal_pairwise_ranking_accuracy"] for row in rows])
            ),
        }
    return output
def _finalize_phase() -> dict:
    session = _session()
    gate = _read_json(SCALE_GATE)
    required_state = (
        "TRAIN_524K_DONE"
        if gate["trigger_524k"]
        else "SCALE_GATE_DONE"
    )
    if session["state"] not in {required_state, "FINAL_SCALE_SELECTED"}:
        raise RuntimeError("M5_BLOCKED_RESUME_GIT_STATE")
    if SELECTION.exists():
        payload = _read_json(SELECTION)
        if session["state"] != "FINAL_SCALE_SELECTED":
            session["state"] = "FINAL_SCALE_SELECTED"
            atomic_write_json(SESSION, session)
        return payload

    development = _development()
    previous = gate["pre_524_winner"]
    selected = previous
    comparison_524 = None
    if gate["trigger_524k"]:
        previous_scores = _scale_score_vector(previous, development)
        scores524 = _scale_score_vector("524k", development)
        previous_median = development["scales"][previous]["aggregate"][
            "median_validation"
        ]
        median524 = development["scales"]["524k"]["aggregate"][
            "median_validation"
        ]
        comparison_524 = select_with_524k(
            previous,
            previous_scores,
            scores524,
            previous_median,
            median524,
        )
        selected = comparison_524["selected_scale"]

    selected_scores = _scale_score_vector(selected, development)
    m4_scores, m4_summary = _m4_aggregate()
    final_vs_m4 = paired_bootstrap(
        selected_scores,
        m4_scores,
        seed=BOOTSTRAP_SEEDS["final_vs_m4"],
    )
    if float(final_vs_m4["ci95"][0]) <= 0.0:
        raise RuntimeError("M5_BLOCKED_PRETRAIN_GAIN_NOT_CONFIRMED")
    median_val = development["scales"][selected]["aggregate"][
        "median_validation"
    ]
    random_baseline = float(
        development["scales"][selected]["runs"][str(TRAINING_SEEDS[0])][
            "validation_metrics"
        ]["random_legal_baseline"]
    )
    if not np.isfinite(float(median_val["ce"])):
        raise RuntimeError("M5_BLOCKED_TEACHER_LEARNING")
    if (
        float(median_val["legal_best_action_accuracy"])
        < random_baseline + 0.10
        or float(median_val["legal_pairwise_ranking_accuracy"]) < 0.65
    ):
        raise RuntimeError("M5_BLOCKED_TEACHER_LEARNING")

    high_tile = _median_high_tile(selected, development)
    for row in high_tile.values():
        if int(row["samples"]) >= 128:
            if (
                float(row["median_legal_best_action_accuracy"])
                < float(row["random_legal_baseline"]) + 0.05
            ):
                raise RuntimeError("M5_BLOCKED_HIGH_TILE_TEACHER_FIT")

    for seed in TRAINING_SEEDS:
        d4 = development["scales"][selected]["runs"][str(seed)]["d4"]
        if not (
            np.isfinite(float(d4["overall_legal_argmax_consistency"]))
            and np.isfinite(float(d4["mean_centered_logit_mae"]))
        ):
            raise RuntimeError("M5_BLOCKED_TEACHER_LEARNING")

    seed_means = {
        int(seed): float(
            development["scales"][selected]["runs"][str(seed)]["summary"][
                "mean_score"
            ]
        )
        for seed in TRAINING_SEEDS
    }
    champion_seed = min(
        seed_means,
        key=lambda seed: (-seed_means[seed], seed),
    )
    champion_run = development["scales"][selected]["runs"][
        str(champion_seed)
    ]
    selection = {
        "schema_version": 1,
        "work_order_version": WORK_ORDER_VERSION,
        "selected_scale": selected,
        "pre_524_winner": previous,
        "trigger_524k": bool(gate["trigger_524k"]),
        "comparison_524k": comparison_524,
        "selected_scale_vs_m4": final_vs_m4,
        "m4_baseline": m4_summary,
        "median_validation": median_val,
        "random_legal_baseline": random_baseline,
        "high_tile_gate": high_tile,
        "champion_seed": int(champion_seed),
        "champion_seed_dev_means": {
            str(key): value for key, value in seed_means.items()
        },
        "champion_checkpoint_path": champion_run["checkpoint_path"],
        "champion_checkpoint_sha256": champion_run["checkpoint_sha256"],
    }
    atomic_write_json(SELECTION, selection)
    session["state"] = "FINAL_SCALE_SELECTED"
    atomic_write_json(SESSION, session)
    print(json.dumps(selection, indent=2), flush=True)
    return selection


def _teacher_test_phase(
    model: torch.nn.Module,
    device: torch.device,
) -> dict:
    if TEACHER_TEST.exists():
        return _read_json(TEACHER_TEST)
    manifest = _label_manifest()
    test_split = load_test_split(ROOT, manifest)
    if len(test_split["state"]) != 8192:
        raise RuntimeError("M5_BLOCKED_DATASET_ARTIFACT_CORRUPT")
    overall = teacher_metrics(
        model,
        test_split["state"],
        test_split["teacher_value"],
        test_split["legal_mask"],
        device,
    )
    strata = high_tile_metrics(
        model,
        test_split,
        device,
    )
    d4 = d4_metrics(
        model,
        test_split["state"],
        test_split["legal_mask"],
        device,
    )
    payload = {
        "schema_version": 1,
        "work_order_version": WORK_ORDER_VERSION,
        "samples": 8192,
        "overall": overall,
        "high_tile_strata": strata,
        "d4": d4,
    }
    atomic_write_json(TEACHER_TEST, payload)
    return payload


def _final_gameplay_phase(
    model: torch.nn.Module,
    device: torch.device,
) -> dict:
    raw_path = FINAL / "final_games_10000.npz"
    result = _load_game_file(raw_path)
    completed = 0 if result is None else len(result["game_seed"])
    if completed % GAME_SHARD_SIZE != 0 or completed > FINAL_GAMES:
        raise RuntimeError("M5_BLOCKED_RESUME_ARTIFACT_CORRUPT")
    expected_prefix = np.arange(
        FINAL_SEED_START,
        FINAL_SEED_START + completed,
        dtype=np.int64,
    )
    if completed and not np.array_equal(result["game_seed"], expected_prefix):
        raise RuntimeError("M5_BLOCKED_RESUME_ARTIFACT_CORRUPT")

    started = time.perf_counter()
    while completed < FINAL_GAMES:
        take = min(GAME_SHARD_SIZE, FINAL_GAMES - completed)
        seeds = np.arange(
            FINAL_SEED_START + completed,
            FINAL_SEED_START + completed + take,
            dtype=np.int64,
        )
        shard = evaluate_game_shard(
            model,
            seeds,
            device,
        )
        result = _append_game(result, shard)
        completed = len(result["game_seed"])
        _save_game(raw_path, result)
        elapsed = time.perf_counter() - started
        print(
            f"final-games {completed}/{FINAL_GAMES} "
            f"elapsed={elapsed:.1f}s "
            f"rate={completed / elapsed:.1f} games/s",
            flush=True,
        )
    summary = score_summary(
        result["final_score"],
        result["max_tile_exp"],
        result["moves"],
    )
    payload = {
        "schema_version": 1,
        "work_order_version": WORK_ORDER_VERSION,
        "raw_path": str(raw_path.relative_to(ROOT)).replace("\\", "/"),
        "raw_sha256": sha256_file(raw_path),
        "summary": summary,
    }
    atomic_write_json(
        FINAL_GAMEPLAY,
        payload,
    )
    return payload


def _test_phase(device: torch.device) -> None:
    session = _session()
    if session["state"] not in {"TEST_RUNNING", "FINAL_TEST_DONE"}:
        raise RuntimeError("M5_BLOCKED_TEST_ISOLATION")
    selection = _read_json(SELECTION)
    model, _summary = _load_final_model(
        selection["selected_scale"],
        int(selection["champion_seed"]),
        device,
    )
    _teacher_test_phase(
        model,
        device,
    )
    _final_gameplay_phase(
        model,
        device,
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()
    session["state"] = "FINAL_TEST_DONE"
    atomic_write_json(SESSION, session)


def _training_evidence(development: dict) -> dict:
    output = {}
    for scale in development["scales"]:
        output[scale] = {}
        for seed in TRAINING_SEEDS:
            path = _progress_path(scale, seed)
            if not path.exists():
                continue
            progress = _read_json(path)
            if progress.get("completed") is not True:
                raise RuntimeError("M5_BLOCKED_RESUME_ARTIFACT_CORRUPT")
            output[scale][str(seed)] = progress["summary"]
    return output


def _validation_evidence(development: dict) -> dict:
    output = {}
    for scale, scale_row in development["scales"].items():
        output[scale] = {}
        for seed, run in scale_row["runs"].items():
            output[scale][seed] = {
                "metrics": run["validation_metrics"],
                "d4": run["d4"],
                "high_tile_strata": run["high_tile_strata"],
            }
    return output


def _dataset_evidence() -> dict:
    label = _label_manifest()
    sources = {}
    for split in ("train", "validation", "test"):
        path = DATA / f"source_{split}_manifest.json"
        if not path.exists():
            raise RuntimeError("M5_BLOCKED_RESUME_ARTIFACT_CORRUPT")
        sources[split] = {
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256_file(path),
            "manifest": _read_json(path),
        }
    return {
        "canonical_unaugmented": True,
        "shard_states": 2048,
        "games_per_shard": 16,
        "source_manifests": sources,
        "label_manifest_path": str(LABEL_MANIFEST.relative_to(ROOT)).replace("\\", "/"),
        "label_manifest_sha256": sha256_file(LABEL_MANIFEST),
        "label_manifest": label,
    }


def _performance_evidence(
    training: dict,
    dataset: dict,
) -> dict:
    label_wall = 0.0
    for split in dataset["label_manifest"]["splits"].values():
        for shard in split["shards"]:
            label_wall += float(shard.get("label_wall_seconds", 0.0))
    training_rows = []
    for scale in training.values():
        training_rows.extend(scale.values())
    return {
        "label_median_roots_per_second": _read_json(SEARCH_PROFILE)[
            "median_roots_per_second"
        ],
        "label_projection_seconds": _read_json(SEARCH_PROFILE)[
            "projections_seconds"
        ],
        "actual_label_wall_seconds": label_wall,
        "training_runs": [
            {
                "scale": row["scale"],
                "seed": row["seed"],
                "samples_per_second": row["training_only_samples_per_s"],
                "wall_seconds": row["training_compute_wall_seconds"],
                "peak_vram_bytes": row["peak_vram_bytes"],
            }
            for row in training_rows
        ],
    }
def _pytest_counts() -> dict:
    import xml.etree.ElementTree as ET

    path = ART / "pytest.xml"
    if not path.exists():
        raise RuntimeError("M5_BLOCKED_RESUME_ARTIFACT_CORRUPT")
    root = ET.parse(path).getroot()
    if root.tag == "testsuites":
        suites = list(root.findall("testsuite"))
        if not suites:
            raise RuntimeError("M5_BLOCKED_RUNTIME_ERROR")
        tests = sum(int(suite.attrib.get("tests", 0)) for suite in suites)
        failures = sum(int(suite.attrib.get("failures", 0)) for suite in suites)
        errors = sum(int(suite.attrib.get("errors", 0)) for suite in suites)
        skipped = sum(int(suite.attrib.get("skipped", 0)) for suite in suites)
    else:
        tests = int(root.attrib["tests"])
        failures = int(root.attrib.get("failures", 0))
        errors = int(root.attrib.get("errors", 0))
        skipped = int(root.attrib.get("skipped", 0))
    if (tests, failures, errors, skipped) != (543, 0, 0, 0):
        raise RuntimeError("M5_BLOCKED_RUNTIME_ERROR")
    return {
        "tests": tests,
        "passed": tests - failures - errors - skipped,
        "failures": failures,
        "errors": errors,
        "skipped": skipped,
        "junit_path": "artifacts/m5/pytest.xml",
        "junit_sha256": sha256_file(path),
    }


def _provenance() -> dict:
    tags = {
        "m0-reference-pass": "3f2def1d95f56eff776e671143188947bf64485b",
        "m1-fastenv-audited-pass": "e5486017a90eeec4fb9814de7880b3c413dc06dd",
        "m2-network-audited-pass": "6a9da5b1bf7c72207acd6e89bc667d439d4823a8",
        "m3-teacher-audited-pass": "7c95f9c5f543065b220fb5f9e9114521732cdc0b",
        "m4-architecture-audited-pass": "300d51a818fa55394ec56de7507bcade111a06d1",
    }
    return {
        "base_head": _session()["base_head"],
        "prompt_sha256": PROMPT_SHA256,
        "peeled_tags": tags,
    }
def _report_phase() -> None:
    session = _session()
    if session["state"] not in {"P10_DONE", "REPORT_WRITTEN"}:
        raise RuntimeError("M5_BLOCKED_RESUME_GIT_STATE")
    development = _development()
    dataset = _dataset_evidence()
    training = _training_evidence(development)
    validation = _validation_evidence(development)
    regression = _pytest_counts()
    selection = _read_json(SELECTION)
    summary = {
        "schema_version": 1,
        "result": "M5_CANDIDATE_EVIDENCE_COMPLETE",
        "provenance": _provenance(),
        "teacher": {
            "path": "teacher_checkpoints/m3/ordinary_td_comparator_ep4800000_7192719323a0.bin",
            "sha256": TEACHER_SHA256,
            "teacher_version": TEACHER_VERSION,
            "semantics": {
                "tuple": "RAW_TUPLE_HEURISTIC",
                "formal_leaf": "FORMAL_STATE_TUPLE_HEURISTIC",
                "root_action": "SEARCH_VALUE_RAW_LEAF",
            },
            "promotion_audit": "reports/m5/M5_TEACHER_PROMOTION_AUDIT.md",
            "promotion_result": "NO PROMOTION",
        },
        "search_profile": _read_json(SEARCH_PROFILE),
        "dataset": dataset,
        "precision": _read_json(PRECISION),
        "training": training,
        "validation": validation,
        "development_gameplay": development,
        "scale_gate": _read_json(SCALE_GATE),
        "selection": selection,
        "teacher_test": _read_json(TEACHER_TEST),
        "final_gameplay": _read_json(FINAL_GAMEPLAY),
        "performance": _performance_evidence(training, dataset),
        "regression": regression,
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    machine_path = REPORTS / "m5_teacher_pretrain.json"
    atomic_write_json(machine_path, summary)
    final_summary = summary["final_gameplay"]["summary"]
    paired = selection["selected_scale_vs_m4"]
    median_val = selection["median_validation"]
    lines = [
        "# M5 Teacher Pretraining Candidate Report",
        "",
        "## 1. RESULT",
        "",
        "**M5_CANDIDATE_EVIDENCE_COMPLETE**",
        "",
        "This is candidate evidence, not M5 audited/frozen status.",
        "",
        "## 2. STARTING STATE",
        "",
        f"- base HEAD: {summary['provenance']['base_head']}",
        f"- work order: {WORK_ORDER_VERSION}",
        f"- prompt SHA-256: {PROMPT_SHA256}",
        "",
        "## 3. FROZEN PROVENANCE",
        "",
        "- M0/M1/M2/M3/M4 authoritative tags verified.",
        "- Frozen M2/M3/M4 implementation diff gates remained zero.",
        "",
        "## 4. TEACHER",
        "",
        "- promotion audit result: **NO PROMOTION**",
        f"- retained Teacher SHA-256: {TEACHER_SHA256}",
        "- Search root values retain SEARCH_VALUE_RAW_LEAF semantics.",
        "- M5 used policy CE only; no absolute Q/V/A regression.",
        "",
        "## 5. SEARCH PROFILE",
        "",
        f"- median roots/s: {summary['search_profile']['median_roots_per_second']:.6f}",
        f"- orchestration share: {summary['search_profile']['orchestration_share']:.6f}",
        "- three complete 256-root repeats passed exact correctness gates.",
        "",
        "## 6. DATASET",
        "",
        "- 32k and 131k rungs completed.",
        f"- 524k trigger: {summary['scale_gate']['status_524k']}",
        "- canonical source boards remained unaugmented.",
        "- D4 augmentation was applied only in training batches.",
        "- validation contains 64 complete source games / 8192 states.",
        "- test labels were generated only after champion selection.",
        "",
        "## 7. PRECISION",
        "",
        f"- selected precision: {summary['precision']['selected_precision']}",
        "- Teacher metrics, D4, and gameplay used FP32 inference.",
        "",
        "## 8. TRAINING",
        "",
        "- architecture: ResidualMLP2048",
        "- parameter count: 5,264,710",
        "- epochs per run: 30",
        "- batch size: 1024",
        "- optimizer: AdamW, lr 3e-4, weight decay 1e-4",
        "- value and afterstate heads remained bit-identical to initialization.",
        "",
        "## 9. SCALE SELECTION",
        "",
        f"- selected scale: **{selection['selected_scale']}**",
        f"- champion seed: **{selection['champion_seed']}**",
        f"- champion checkpoint SHA-256: {selection['champion_checkpoint_sha256']}",
        f"- selected vs M4 paired mean delta: {paired['mean_delta']:.6f}",
        f"- selected vs M4 CI95: [{paired['ci95'][0]:.6f}, {paired['ci95'][1]:.6f}]",
        "",
        "## 10. VALIDATION",
        "",
        f"- median CE: {median_val['ce']:.6f}",
        f"- median legal best-action accuracy: {median_val['legal_best_action_accuracy']:.6f}",
        f"- median pairwise ranking accuracy: {median_val['legal_pairwise_ranking_accuracy']:.6f}",
        f"- random-legal baseline: {selection['random_legal_baseline']:.6f}",
        "",
        "## 11. TEACHER TEST",
        "",
        "- champion Teacher test was consumed once after champion selection.",
        f"- test samples: {summary['teacher_test']['samples']}",
        "",
        "## 12. FINAL GAMEPLAY",
        "",
        f"- games: {final_summary['games']}",
        f"- mean score: {final_summary['mean_score']:.6f}",
        f"- median score: {final_summary['median_score']:.6f}",
        f"- p10 score: {final_summary['p10_score']:.6f}",
        f"- p90 score: {final_summary['p90_score']:.6f}",
        f"- mean moves: {final_summary['mean_moves']:.6f}",
        "",
        f"- reach 2048: {final_summary['reach']['2048']:.6f}",
        f"- reach 4096: {final_summary['reach']['4096']:.6f}",
        f"- reach 8192: {final_summary['reach']['8192']:.6f}",
        f"- reach 16384: {final_summary['reach']['16384']:.6f}",
        f"- reach 32768: {final_summary['reach']['32768']:.6f}",
        f"- reach 65536: {final_summary['reach']['65536']:.6f}",
        "",
        "## 13. PERFORMANCE",
        "",
        f"- actual Teacher label wall: {summary['performance']['actual_label_wall_seconds']:.3f}s",
        "",
        "## 14. REGRESSION",
        "",
        f"- pytest passed: {regression['passed']}",
        "- pytest failures/errors/skipped: 0/0/0",
        "",
        "## 15. GIT / CI",
        "",
        "- candidate_sha=PENDING_BY_DESIGN",
        "- candidate_ci=PENDING_BY_DESIGN",
        "- closeout_commit_sha=SELF_NOT_EMBEDDABLE_BY_DESIGN",
        "- closeout_ci=PENDING_BY_DESIGN",
        "",
        "## 16. AUDIT STATUS",
        "",
        "M5 remains candidate evidence only.",
        "Independent M5 audit is still required before any audited tag.",
        "M6 has not been entered.",
        "",
    ]
    text = "\n".join(lines) + "\n"
    report_path = REPORTS / "M5_REPORT.md"
    report_path.write_text(text, encoding="utf-8")
    physical_lines = report_path.read_text(encoding="utf-8").splitlines()
    if len(physical_lines) < 50:
        raise RuntimeError("M5_BLOCKED_REPORT_RENDERING")
    json.loads(machine_path.read_text(encoding="utf-8"))
    session["state"] = "REPORT_WRITTEN"
    atomic_write_json(SESSION, session)
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        required=True,
        choices=(
            "precision",
            "train-32k",
            "train-131k",
            "scale-gate",
            "train-524k",
            "finalize",
            "test",
            "report",
        ),
    )
    parser.add_argument("--resume", action="store_true", required=True)
    args = parser.parse_args()

    if args.phase == "report":
        _report_phase()
        return

    if not torch.cuda.is_available():
        raise RuntimeError("M5_BLOCKED_RUNTIME_ENVIRONMENT")
    device = torch.device("cuda")
    if trainable_parameter_count(ResidualMLP2048()) != PARAMETER_COUNT:
        raise RuntimeError("M5_BLOCKED_MODEL_DRIFT")

    if args.phase == "precision":
        _precision_phase(device)
    elif args.phase == "train-32k":
        _run_scale("32k", device)
    elif args.phase == "train-131k":
        _run_scale("131k", device)
    elif args.phase == "scale-gate":
        _scale_gate_phase()
    elif args.phase == "train-524k":
        _run_scale("524k", device)
    elif args.phase == "finalize":
        _finalize_phase()
    elif args.phase == "test":
        _test_phase(device)


if __name__ == "__main__":
    main()
