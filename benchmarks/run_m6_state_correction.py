"""M6 Student State Correction training/evaluation orchestrator."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from game2048.m2_models import ResidualMLP2048, trainable_parameter_count
from game2048.m4_compare import evaluate_game_shard
from game2048.m5_pretrain import (
    load_label_split as load_m5_label_split,
    validate_manifest_completed_shards as validate_m5_manifest,
)
from game2048.m6_state_correction import (
    ANCHOR_ROWS, ARCHITECTURE, BATCH_SIZE, CORRECTION_ROWS,
    DEV_BOOTSTRAP_SEEDS, DEV_GAMES, DEV_SEED_START, EPOCHS,
    FINAL_BOOTSTRAP_SEED, FINAL_GAMES, FINAL_SEED_START, GAME_SHARD_SIZE,
    M5_ANCHOR_MANIFEST_SHA256, M5_CHECKPOINT_SHA256,
    M5_PRETRAIN_TEACHER_SHA256, M5_TAG_SHA,
    PARAMETER_COUNT, PROMPT_SHA256, TEACHER_FILENAME, TEACHER_SHA256,
    TRAINING_SEEDS,
    VALIDATION_BOOTSTRAP_SEED, VALIDATION_GAMES,
    VALIDATION_GAME_SEED_START, WORK_ORDER_VERSION, atomic_save_npz,
    atomic_torch_save, atomic_write_json, correction_anchor_batch,
    d4_metrics, epoch_plan, fresh_finetune, high_tile_metrics,
    load_m5_model, paired_bootstrap, parameter_update_checks, plan_id,
    plan_sha256, promotion_result, score_summary, set_determinism,
    sha256_file, snapshot_parameters, teacher_best_action, teacher_metrics,
    training_label_manifest_sha256, validate_completed_shard,
)

ART = ROOT / "artifacts" / "m6"
PROGRESS = ART / "progress"
CHECKPOINTS = ART / "checkpoints"
GAMES = ART / "games"
FINAL = ART / "final"
DATA = ART / "datasets"
REPORTS = ROOT / "reports" / "m6"
SESSION = PROGRESS / "session.json"
LABEL_MANIFEST = DATA / "label_manifest.json"
M5_LABEL_MANIFEST = ROOT / "artifacts" / "m5" / "datasets" / "label_manifest.json"
M5_CHECKPOINT = ROOT / "artifacts" / "m5" / "checkpoints" / "524k" / "20262103_final.pt"
TEACHER_PATH = ROOT / "teacher_checkpoints" / "m6" / TEACHER_FILENAME
SEARCH_PROFILE = ART / "search_profile.json"
ANCHOR_VERIFY = ART / "anchor_verification.json"
SMOKE = PROGRESS / "smoke.json"
TEACHER_VALIDATION = ART / "teacher_validation.json"
DEVELOPMENT = ART / "development.json"
SELECTION = ART / "candidate_selection.json"
GAMEPLAY_VALIDATION = ART / "gameplay_validation.json"
TEACHER_TEST = ART / "teacher_test.json"
FINAL_GAMEPLAY = ART / "final_gameplay.json"
PROMOTION = ART / "promotion.json"
PRECISION = "BF16 autocast"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _session() -> dict:
    row = _read_json(SESSION)
    expected = {
        "schema_version": 1,
        "work_order_version": WORK_ORDER_VERSION,
        "base_head": "6a9b7614e1f7e971e246ae9820f3d052e07875ed",
        "prompt_sha256": PROMPT_SHA256,
        "m5_tag_sha": M5_TAG_SHA,
        "m5_student_checkpoint_sha256": M5_CHECKPOINT_SHA256,
        "teacher_sha256": TEACHER_SHA256,
    }
    for key, value in expected.items():
        if row.get(key) != value:
            raise RuntimeError("M6_BLOCKED_RESUME_METADATA_MISMATCH")
    return row


def _m6_manifest() -> dict:
    if not LABEL_MANIFEST.exists():
        raise RuntimeError("M6_BLOCKED_DATASET_ARTIFACT_CORRUPT")
    row = _read_json(LABEL_MANIFEST)
    for split in row.get("splits", {}).values():
        for shard in split.get("shards", []):
            if shard.get("status") == "complete":
                validate_completed_shard(ROOT / shard["path"], shard["sha256"])
    return row


def _load_m6_split(split: str) -> dict[str, np.ndarray]:
    manifest = _m6_manifest()
    rows = manifest["splits"][split]["shards"]
    arrays = {
        "state": [], "teacher_value": [], "legal_mask": [],
        "max_tile_exp": [], "disagreement": [],
    }
    for row in sorted(rows, key=lambda item: int(item["shard_index"])):
        if row.get("status") != "complete":
            raise RuntimeError("M6_BLOCKED_DATASET_ARTIFACT_CORRUPT")
        validate_completed_shard(ROOT / row["path"], row["sha256"])
        with np.load(ROOT / row["path"], allow_pickle=False) as data:
            for key in arrays:
                arrays[key].append(np.ascontiguousarray(data[key]))
    if not rows:
        raise RuntimeError("M6_BLOCKED_DATASET_ARTIFACT_CORRUPT")
    return {key: np.concatenate(parts, axis=0) for key, parts in arrays.items()}


def _load_anchor() -> dict[str, np.ndarray]:
    if sha256_file(M5_LABEL_MANIFEST) != M5_ANCHOR_MANIFEST_SHA256:
        raise RuntimeError("M6_BLOCKED_M5_ANCHOR_EVIDENCE")
    manifest = _read_json(M5_LABEL_MANIFEST)
    validate_m5_manifest(manifest, ROOT)
    train = manifest.get("splits", {}).get("train", {}).get("shards", [])
    if (
        len(train) != 256
        or sum(int(row.get("rows", 0)) for row in train) != ANCHOR_ROWS
        or manifest.get("teacher_sha256", "").upper()
        != M5_PRETRAIN_TEACHER_SHA256
        or manifest.get("value_semantics") != "SEARCH_VALUE_RAW_LEAF"
    ):
        raise RuntimeError("M6_BLOCKED_M5_ANCHOR_EVIDENCE")
    return load_m5_label_split(ROOT, manifest, "train")


def _anchor_phase() -> dict:
    session = _session()
    if session["state"] != "DATA_READY":
        raise RuntimeError("M6_BLOCKED_RESUME_GIT_STATE")
    if ANCHOR_VERIFY.exists():
        return _read_json(ANCHOR_VERIFY)
    anchor = _load_anchor()
    payload = {
        "schema_version": 1,
        "work_order_version": WORK_ORDER_VERSION,
        "manifest_path": "artifacts/m5/datasets/label_manifest.json",
        "manifest_sha256": M5_ANCHOR_MANIFEST_SHA256,
        "train_shards": 256,
        "train_rows": int(len(anchor["state"])),
        "teacher_sha256": M5_PRETRAIN_TEACHER_SHA256,
        "value_semantics": "SEARCH_VALUE_RAW_LEAF",
        "all_train_shards_sha_valid": True,
        "validation_or_test_used_for_training": False,
    }
    if payload["train_rows"] != ANCHOR_ROWS:
        raise RuntimeError("M6_BLOCKED_M5_ANCHOR_EVIDENCE")
    atomic_write_json(ANCHOR_VERIFY, payload)
    print(json.dumps(payload, indent=2), flush=True)
    return payload


def _finite_gradients(model: torch.nn.Module) -> bool:
    return all(
        p.grad is None or bool(torch.isfinite(p.grad).all().item())
        for p in model.parameters()
    )


def _finite_parameters(model: torch.nn.Module) -> bool:
    return all(bool(torch.isfinite(p).all().item()) for p in model.parameters())


def _targets(split: dict[str, np.ndarray]) -> np.ndarray:
    return teacher_best_action(
        split["teacher_value"], split["legal_mask"]
    ).astype(np.int64)


def _smoke_phase(device: torch.device) -> dict:
    session = _session()
    if session["state"] != "DATA_READY":
        raise RuntimeError("M6_BLOCKED_RESUME_GIT_STATE")
    if SMOKE.exists():
        payload = _read_json(SMOKE)
        if payload.get("status") != "PASS" or payload.get("completed_steps") != 5:
            raise RuntimeError("M6_BLOCKED_NUMERIC_SMOKE")
        return payload
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("M6_BLOCKED_NUMERIC_SMOKE")
    correction = _load_m6_split("train")
    anchor = _load_anchor()
    correction_target = _targets(correction)
    anchor_target = _targets(anchor)
    model, optimizer, _ = fresh_finetune(
        TRAINING_SEEDS[0], M5_CHECKPOINT, device
    )
    plan = epoch_plan(TRAINING_SEEDS[0], 0)
    completed = 0
    losses = []
    try:
        model.train()
        for step in range(5):
            boards, target = correction_anchor_batch(
                correction["state"], correction_target,
                anchor["state"], anchor_target, plan, step, device
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(boards)
                loss = F.cross_entropy(logits, target, reduction="mean")
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError("M6_BLOCKED_NUMERIC_SMOKE")
            loss.backward()
            if not _finite_gradients(model):
                raise RuntimeError("M6_BLOCKED_NUMERIC_SMOKE")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if not _finite_parameters(model):
                raise RuntimeError("M6_BLOCKED_NUMERIC_SMOKE")
            completed += 1
            losses.append(float(loss.detach().float().item()))
            print(
                f"smoke step={completed}/5 loss={losses[-1]:.6f}",
                flush=True,
            )
        payload = {
            "schema_version": 1,
            "work_order_version": WORK_ORDER_VERSION,
            "precision": PRECISION,
            "bf16_supported": True,
            "completed_steps": completed,
            "losses": losses,
            "status": "PASS",
            "formal_training_state_reused": False,
        }
        atomic_write_json(SMOKE, payload)
        return payload
    finally:
        del optimizer, model
        gc.collect()
        torch.cuda.empty_cache()


def _progress_path(seed: int) -> Path:
    return PROGRESS / f"train_{int(seed)}.json"


def _latest_path(seed: int) -> Path:
    return CHECKPOINTS / f"{int(seed)}_latest.pt"


def _final_path(seed: int) -> Path:
    return CHECKPOINTS / f"{int(seed)}_final.pt"


def _verify_training_checkpoint(payload: dict, seed: int, manifest_sha: str) -> None:
    expected = {
        "architecture": ARCHITECTURE,
        "seed": int(seed),
        "work_order_version": WORK_ORDER_VERSION,
        "prompt_sha256": PROMPT_SHA256,
        "m5_baseline_checkpoint_sha256": M5_CHECKPOINT_SHA256,
        "teacher_sha256": TEACHER_SHA256,
        "correction_manifest_sha256": manifest_sha,
        "m5_anchor_manifest_sha256": M5_ANCHOR_MANIFEST_SHA256,
        "plan_id": plan_id(seed),
        "precision": PRECISION,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError("M6_BLOCKED_RESUME_METADATA_MISMATCH")


def _train_one(
    seed: int,
    correction: dict[str, np.ndarray],
    anchor: dict[str, np.ndarray],
    validation: dict[str, np.ndarray],
    device: torch.device,
) -> dict:
    progress_path = _progress_path(seed)
    latest_path = _latest_path(seed)
    final_path = _final_path(seed)
    manifest_sha = training_label_manifest_sha256(LABEL_MANIFEST, ROOT)
    correction_target = _targets(correction)
    anchor_target = _targets(anchor)
    plan_name = plan_id(seed)
    plan_hash = plan_sha256(seed)

    if progress_path.exists():
        progress = _read_json(progress_path)
        expected = {
            "seed": int(seed),
            "plan_id": plan_name,
            "plan_sha256": plan_hash,
            "correction_manifest_sha256": manifest_sha,
            "m5_anchor_manifest_sha256": M5_ANCHOR_MANIFEST_SHA256,
        }
        for key, value in expected.items():
            if progress.get(key) != value:
                raise RuntimeError("M6_BLOCKED_RESUME_METADATA_MISMATCH")
        if progress.get("completed") is True:
            if (
                not final_path.exists()
                or sha256_file(final_path) != progress["final_checkpoint_sha256"]
            ):
                raise RuntimeError("M6_BLOCKED_RESUME_ARTIFACT_CORRUPT")
            return progress["summary"]
        if int(progress.get("next_epoch", 0)) > 0 and not latest_path.exists():
            raise RuntimeError("M6_BLOCKED_RESUME_ARTIFACT_CORRUPT")
    else:
        progress = {
            "schema_version": 1,
            "work_order_version": WORK_ORDER_VERSION,
            "architecture": ARCHITECTURE,
            "seed": int(seed),
            "precision": PRECISION,
            "m5_baseline_checkpoint_sha256": M5_CHECKPOINT_SHA256,
            "teacher_sha256": TEACHER_SHA256,
            "correction_manifest_sha256": manifest_sha,
            "m5_anchor_manifest_sha256": M5_ANCHOR_MANIFEST_SHA256,
            "plan_id": plan_name,
            "plan_sha256": plan_hash,
            "next_epoch": 0,
            "completed": False,
            "optimizer_steps": 0,
            "history": [],
            "training_compute_wall_seconds": 0.0,
            "peak_vram_bytes": 0,
        }

    model, optimizer, baseline_payload = fresh_finetune(
        seed, M5_CHECKPOINT, device
    )
    initial_parameters = {
        name: baseline_payload["model_state"][name].detach().cpu().clone()
        for name, _ in model.named_parameters()
    }
    if int(progress["next_epoch"]) > 0:
        checkpoint = torch.load(
            latest_path, map_location="cpu", weights_only=False
        )
        _verify_training_checkpoint(checkpoint, seed, manifest_sha)
        if int(checkpoint.get("next_epoch", -1)) != int(progress["next_epoch"]):
            raise RuntimeError("M6_BLOCKED_RESUME_ARTIFACT_CORRUPT")
        if int(checkpoint.get("optimizer_steps", -1)) != int(progress["optimizer_steps"]):
            raise RuntimeError("M6_BLOCKED_RESUME_ARTIFACT_CORRUPT")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    if len(correction["state"]) != CORRECTION_ROWS or len(anchor["state"]) != ANCHOR_ROWS:
        raise RuntimeError("M6_BLOCKED_DATASET_ARTIFACT_CORRUPT")
    steps_per_epoch = CORRECTION_ROWS // 512
    torch.cuda.reset_peak_memory_stats()

    for epoch in range(int(progress["next_epoch"]), EPOCHS):
        plan = epoch_plan(seed, epoch)
        if np.unique(plan["anchor_subset"]).size != CORRECTION_ROWS:
            raise RuntimeError("M6_BLOCKED_TRAINING_PLAN")
        model.train()
        total_loss_sum = 0.0
        correction_loss_sum = 0.0
        anchor_loss_sum = 0.0
        seen = 0
        torch.cuda.synchronize()
        started = time.perf_counter()
        for step in range(steps_per_epoch):
            boards, target = correction_anchor_batch(
                correction["state"], correction_target,
                anchor["state"], anchor_target,
                plan, step, device,
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(boards)
                loss = F.cross_entropy(logits, target, reduction="mean")
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError("M6_BLOCKED_NUMERIC")
            with torch.no_grad():
                corr_loss = F.cross_entropy(
                    logits[:512].float(), target[:512], reduction="mean"
                )
                anchor_loss = F.cross_entropy(
                    logits[512:].float(), target[512:], reduction="mean"
                )
            loss.backward()
            if not _finite_gradients(model):
                raise RuntimeError("M6_BLOCKED_NUMERIC")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if not _finite_parameters(model):
                raise RuntimeError("M6_BLOCKED_NUMERIC")
            total_loss_sum += float(loss.detach().float().item()) * BATCH_SIZE
            correction_loss_sum += float(corr_loss.item()) * 512
            anchor_loss_sum += float(anchor_loss.item()) * 512
            seen += BATCH_SIZE
        torch.cuda.synchronize()
        epoch_wall = time.perf_counter() - started
        progress["training_compute_wall_seconds"] += float(epoch_wall)
        progress["peak_vram_bytes"] = max(
            int(progress["peak_vram_bytes"]),
            int(torch.cuda.max_memory_allocated()),
        )
        val = teacher_metrics(
            model, validation["state"], validation["teacher_value"],
            validation["legal_mask"], device
        )
        progress["optimizer_steps"] = int(progress["optimizer_steps"]) + steps_per_epoch
        completed_epoch = epoch + 1
        mean_epoch = float(progress["training_compute_wall_seconds"]) / completed_epoch
        eta = mean_epoch * (EPOCHS - completed_epoch)
        row = {
            "epoch": completed_epoch,
            "train_total_ce": float(total_loss_sum / seen),
            "correction_ce": float(correction_loss_sum / (CORRECTION_ROWS)),
            "anchor_ce": float(anchor_loss_sum / (CORRECTION_ROWS)),
            "teacher_validation_ce": float(val["ce"]),
            "teacher_validation_legal_accuracy": float(
                val["legal_best_action_accuracy"]
            ),
            "training_wall_seconds": float(epoch_wall),
            "samples_per_second": float(seen / epoch_wall),
            "peak_vram_bytes": int(progress["peak_vram_bytes"]),
            "estimated_remaining_training_wall_seconds": float(eta),
            "optimizer_steps_total": int(progress["optimizer_steps"]),
        }
        progress["history"].append(row)
        progress["next_epoch"] = completed_epoch
        latest_payload = {
            "architecture": ARCHITECTURE,
            "seed": int(seed),
            "work_order_version": WORK_ORDER_VERSION,
            "prompt_sha256": PROMPT_SHA256,
            "m5_baseline_checkpoint_sha256": M5_CHECKPOINT_SHA256,
            "teacher_sha256": TEACHER_SHA256,
            "correction_manifest_sha256": manifest_sha,
            "m5_anchor_manifest_sha256": M5_ANCHOR_MANIFEST_SHA256,
            "plan_id": plan_name,
            "plan_sha256": plan_hash,
            "next_epoch": completed_epoch,
            "optimizer_steps": int(progress["optimizer_steps"]),
            "precision": PRECISION,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        }
        atomic_torch_save(latest_path, latest_payload)
        atomic_write_json(progress_path, progress)
        print(
            f"train seed={seed} epoch={completed_epoch}/{EPOCHS} "
            f"total_ce={row['train_total_ce']:.6f} "
            f"corr_ce={row['correction_ce']:.6f} anchor_ce={row['anchor_ce']:.6f} "
            f"val_ce={row['teacher_validation_ce']:.6f} "
            f"val_acc={row['teacher_validation_legal_accuracy']:.6f} "
            f"elapsed={progress['training_compute_wall_seconds']:.1f}s "
            f"samples/s={row['samples_per_second']:.1f} "
            f"peak_vram={row['peak_vram_bytes']} eta={eta:.1f}s",
            flush=True,
        )
    checks = parameter_update_checks(model, initial_parameters)
    if not (checks["q_head_updated"] and checks["backbone_updated"]):
        raise RuntimeError("M6_BLOCKED_NO_PARAMETER_UPDATE")
    if not checks["value_heads_unchanged"]:
        raise RuntimeError("M6_BLOCKED_VALUE_HEAD_UPDATED")

    final_validation = teacher_metrics(
        model, validation["state"], validation["teacher_value"],
        validation["legal_mask"], device,
    )
    final_d4 = d4_metrics(
        model, validation["state"], validation["legal_mask"], device,
    )
    strata = high_tile_metrics(model, validation, device)
    finite = (
        np.isfinite(float(final_validation["ce"]))
        and np.isfinite(float(final_validation["legal_best_action_accuracy"]))
        and np.isfinite(float(final_validation["legal_pairwise_ranking_accuracy"]))
        and np.isfinite(float(final_d4["overall_legal_argmax_consistency"]))
        and np.isfinite(float(final_d4["mean_centered_logit_mae"]))
    )
    if not finite:
        raise RuntimeError("M6_BLOCKED_CORRECTION_LEARNING_SANITY")
    if (
        float(final_validation["legal_best_action_accuracy"])
        < float(final_validation["random_legal_baseline"]) + 0.10
        or float(final_validation["legal_pairwise_ranking_accuracy"]) < 0.65
    ):
        raise RuntimeError("M6_BLOCKED_CORRECTION_LEARNING_SANITY")
    if int(progress["optimizer_steps"]) != 7680:
        raise RuntimeError("M6_BLOCKED_RESUME_ARTIFACT_CORRUPT")
    final_payload = {
        "architecture": ARCHITECTURE,
        "seed": int(seed),
        "work_order_version": WORK_ORDER_VERSION,
        "prompt_sha256": PROMPT_SHA256,
        "m5_baseline_checkpoint_sha256": M5_CHECKPOINT_SHA256,
        "teacher_sha256": TEACHER_SHA256,
        "correction_manifest_sha256": manifest_sha,
        "m5_anchor_manifest_sha256": M5_ANCHOR_MANIFEST_SHA256,
        "plan_id": plan_name,
        "plan_sha256": plan_hash,
        "epoch": EPOCHS,
        "optimizer_steps": int(progress["optimizer_steps"]),
        "precision": PRECISION,
        "model_state": model.state_dict(),
    }
    atomic_torch_save(final_path, final_payload)
    final_sha = sha256_file(final_path)
    total_wall = float(progress["training_compute_wall_seconds"])
    summary = {
        "architecture": ARCHITECTURE,
        "seed": int(seed),
        "precision": PRECISION,
        "plan_id": plan_name,
        "plan_sha256": plan_hash,
        "final_checkpoint_path": str(final_path.relative_to(ROOT)).replace("\\", "/"),
        "final_checkpoint_sha256": final_sha,
        "epochs": EPOCHS,
        "optimizer_steps": int(progress["optimizer_steps"]),
        "training_compute_wall_seconds": total_wall,
        "training_only_samples_per_s": float(
            (CORRECTION_ROWS * 2 * EPOCHS) / total_wall
        ),
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


def _load_final_model(seed: int, device: torch.device) -> tuple[torch.nn.Module, dict]:
    progress = _read_json(_progress_path(seed))
    if progress.get("completed") is not True:
        raise RuntimeError("M6_BLOCKED_RESUME_ARTIFACT_CORRUPT")
    summary = progress["summary"]
    path = ROOT / summary["final_checkpoint_path"]
    if not path.exists() or sha256_file(path) != summary["final_checkpoint_sha256"]:
        raise RuntimeError("M6_BLOCKED_RESUME_ARTIFACT_CORRUPT")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    _verify_training_checkpoint(
        payload, seed, training_label_manifest_sha256(LABEL_MANIFEST, ROOT)
    )
    if int(payload.get("epoch", -1)) != EPOCHS:
        raise RuntimeError("M6_BLOCKED_RESUME_ARTIFACT_CORRUPT")
    if int(payload.get("optimizer_steps", -1)) != 7680:
        raise RuntimeError("M6_BLOCKED_RESUME_ARTIFACT_CORRUPT")
    model = ResidualMLP2048()
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    return model, summary


def _disagreement_accuracy(
    model: torch.nn.Module,
    split: dict[str, np.ndarray],
    device: torch.device,
) -> dict:
    mask = np.asarray(split["disagreement"], dtype=np.bool_)
    count = int(mask.sum())

    if count == 0:
        return {
            "samples": 0,
            "legal_best_action_accuracy": float("nan"),
        }
    target = _targets(split)
    correct = 0
    model.eval()
    with torch.inference_mode():
        ids = np.flatnonzero(mask)
        for start in range(0, len(ids), BATCH_SIZE):
            rows = ids[start:start + BATCH_SIZE]
            boards = torch.from_numpy(
                np.ascontiguousarray(split["state"][rows])
            ).to(device)
            logits = model(boards).float().cpu().numpy()
            legal = split["legal_mask"][rows]
            predicted = np.argmax(
                np.where(legal, logits, -np.inf), axis=1
            )
            correct += int(np.sum(predicted == target[rows]))
    return {
        "samples": count,
        "legal_best_action_accuracy": float(correct / count),
    }


def _teacher_eval(
    model: torch.nn.Module,
    split: dict[str, np.ndarray],
    device: torch.device,
) -> dict:
    overall = teacher_metrics(
        model,
        split["state"],
        split["teacher_value"],
        split["legal_mask"],
        device,
    )
    return {
        "overall": overall,
        "baseline_disagreement": _disagreement_accuracy(
            model, split, device
        ),
        "d4": d4_metrics(
            model, split["state"], split["legal_mask"], device
        ),
        "high_tile_strata": high_tile_metrics(model, split, device),
    }


def _teacher_eval_sane(row: dict) -> bool:
    overall = row["overall"]
    d4 = row["d4"]
    finite = np.asarray([
        overall["ce"],
        overall["legal_best_action_accuracy"],
        overall["legal_pairwise_ranking_accuracy"],
        d4["overall_legal_argmax_consistency"],
        d4["mean_centered_logit_mae"],
    ], dtype=np.float64)
    return bool(
        np.isfinite(finite).all()
        and float(overall["legal_best_action_accuracy"])
        >= float(overall["random_legal_baseline"]) + 0.10
        and float(overall["legal_pairwise_ranking_accuracy"]) >= 0.65
    )


def _train_phase(device: torch.device) -> dict:
    session = _session()
    if session["state"] == "TRAIN_DONE":
        if not TEACHER_VALIDATION.exists():
            raise RuntimeError("M6_BLOCKED_RESUME_ARTIFACT_CORRUPT")
        return _read_json(TEACHER_VALIDATION)
    if session["state"] not in {"DATA_READY", "TRAIN_RUNNING"}:
        raise RuntimeError("M6_BLOCKED_RESUME_GIT_STATE")
    if not ANCHOR_VERIFY.exists():
        _anchor_phase()
    smoke = _smoke_phase(device)
    if smoke.get("status") != "PASS":
        raise RuntimeError("M6_BLOCKED_NUMERIC_SMOKE")

    session["state"] = "TRAIN_RUNNING"
    atomic_write_json(SESSION, session)
    correction = _load_m6_split("train")
    validation = _load_m6_split("validation")
    anchor = _load_anchor()
    if len(correction["state"]) != CORRECTION_ROWS:
        raise RuntimeError("M6_BLOCKED_DATASET_ARTIFACT_CORRUPT")
    if len(validation["state"]) != 8192:
        raise RuntimeError("M6_BLOCKED_DATASET_ARTIFACT_CORRUPT")

    baseline_model, _ = load_m5_model(M5_CHECKPOINT, device)
    baseline_metrics = _teacher_eval(
        baseline_model, validation, device
    )
    del baseline_model
    gc.collect()
    torch.cuda.empty_cache()

    training: dict[str, dict] = {}
    run_metrics: dict[str, dict] = {}
    sanity: dict[str, bool] = {}
    for seed in TRAINING_SEEDS:
        training[str(seed)] = _train_one(
            seed, correction, anchor, validation, device
        )
        model, _ = _load_final_model(seed, device)
        metrics = _teacher_eval(model, validation, device)
        run_metrics[str(seed)] = metrics
        sanity[str(seed)] = _teacher_eval_sane(metrics)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    if not all(sanity.values()):
        raise RuntimeError("M6_BLOCKED_CORRECTION_LEARNING_SANITY")
    payload = {
        "schema_version": 1,
        "work_order_version": WORK_ORDER_VERSION,
        "samples": int(len(validation["state"])),
        "frozen_student_teacher_disagreement_rate": float(
            np.mean(validation["disagreement"])
        ),
        "m5_baseline": baseline_metrics,
        "m6_runs": run_metrics,
        "learning_sanity": sanity,
        "training": training,
    }
    atomic_write_json(TEACHER_VALIDATION, payload)
    session["state"] = "TRAIN_DONE"
    atomic_write_json(SESSION, session)
    print(json.dumps({
        "phase": "train",
        "state": "TRAIN_DONE",
        "learning_sanity": sanity,
    }, indent=2), flush=True)
    return payload


def _load_game_file(path: Path) -> dict[str, np.ndarray] | None:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        required = {"game_seed", "final_score", "max_tile_exp", "moves"}
        if set(data.files) != required:
            raise RuntimeError("M6_BLOCKED_RESUME_ARTIFACT_CORRUPT")
        result = {
            "game_seed": np.ascontiguousarray(
                data["game_seed"], dtype=np.int64
            ),
            "final_score": np.ascontiguousarray(
                data["final_score"], dtype=np.int64
            ),
            "max_tile_exp": np.ascontiguousarray(
                data["max_tile_exp"], dtype=np.uint8
            ),
            "moves": np.ascontiguousarray(
                data["moves"], dtype=np.int64
            ),
        }
    lengths = {value.shape for value in result.values()}
    if len(lengths) != 1:
        raise RuntimeError("M6_BLOCKED_RESUME_ARTIFACT_CORRUPT")
    if any(value.ndim != 1 for value in result.values()):
        raise RuntimeError("M6_BLOCKED_RESUME_ARTIFACT_CORRUPT")
    seeds = result["game_seed"]
    if not np.array_equal(seeds, np.sort(seeds)):
        raise RuntimeError("M6_BLOCKED_RESUME_ARTIFACT_CORRUPT")
    if np.unique(seeds).size != seeds.size:
        raise RuntimeError("M6_BLOCKED_RESUME_ARTIFACT_CORRUPT")
    return result


def _append_game(
    existing: dict[str, np.ndarray] | None,
    shard: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    if existing is None:
        merged = {
            key: np.ascontiguousarray(value)
            for key, value in shard.items()
        }
    else:
        merged = {
            key: np.concatenate((existing[key], shard[key]))
            for key in ("game_seed", "final_score", "max_tile_exp", "moves")
        }
    order = np.argsort(merged["game_seed"])
    merged = {
        key: np.ascontiguousarray(value[order])
        for key, value in merged.items()
    }
    if np.unique(merged["game_seed"]).size != len(merged["game_seed"]):
        raise RuntimeError("M6_BLOCKED_RESUME_ARTIFACT_CORRUPT")
    return merged


def _save_game(path: Path, result: dict[str, np.ndarray]) -> None:
    atomic_save_npz(
        path,
        game_seed=np.asarray(result["game_seed"], dtype=np.int64),
        final_score=np.asarray(result["final_score"], dtype=np.int64),
        max_tile_exp=np.asarray(result["max_tile_exp"], dtype=np.uint8),
        moves=np.asarray(result["moves"], dtype=np.int64),
    )


def _run_game_set(
    model: torch.nn.Module,
    path: Path,
    seed_start: int,
    games: int,
    device: torch.device,
    label: str,
) -> tuple[dict[str, np.ndarray], dict]:
    result = _load_game_file(path)
    completed = 0 if result is None else len(result["game_seed"])
    if completed % GAME_SHARD_SIZE != 0 or completed > int(games):
        raise RuntimeError("M6_BLOCKED_RESUME_ARTIFACT_CORRUPT")
    expected = np.arange(
        int(seed_start), int(seed_start) + completed, dtype=np.int64
    )
    if completed and not np.array_equal(result["game_seed"], expected):
        raise RuntimeError("M6_BLOCKED_RESUME_ARTIFACT_CORRUPT")

    started = time.perf_counter()
    while completed < int(games):
        take = min(GAME_SHARD_SIZE, int(games) - completed)
        seeds = np.arange(
            int(seed_start) + completed,
            int(seed_start) + completed + take,
            dtype=np.int64,
        )
        shard = evaluate_game_shard(model, seeds, device)
        result = _append_game(result, shard)
        completed = len(result["game_seed"])
        _save_game(path, result)
        elapsed = time.perf_counter() - started
        rate = completed / max(elapsed, 1e-12)
        rolling = float(
            np.mean(result["final_score"][max(0, completed - 1000):])
        )
        print(
            f"gameplay phase={label} completed={completed}/{games} "
            f"elapsed={elapsed:.1f}s games/s={rate:.2f} "
            f"rolling_mean={rolling:.1f} eta={(games-completed)/rate:.1f}s",
            flush=True,
        )
    payload = {
        "raw_path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "raw_sha256": sha256_file(path),
        "summary": score_summary(
            result["final_score"],
            result["max_tile_exp"],
            result["moves"],
        ),
    }
    return result, payload


def _win_loss_tie(
    candidate: np.ndarray,
    baseline: np.ndarray,
) -> dict:
    cand = np.asarray(candidate, dtype=np.int64)
    base = np.asarray(baseline, dtype=np.int64)
    if cand.shape != base.shape or cand.ndim != 1:
        raise ValueError("paired scores must be equal 1-D arrays")
    return {
        "wins": int(np.sum(cand > base)),
        "losses": int(np.sum(cand < base)),
        "ties": int(np.sum(cand == base)),
    }


def _paired_game_stats(
    candidate: dict[str, np.ndarray],
    baseline: dict[str, np.ndarray],
    bootstrap_seed: int,
) -> dict:
    if not np.array_equal(
        candidate["game_seed"], baseline["game_seed"]
    ):
        raise RuntimeError("M6_BLOCKED_GAMEPLAY_SEED_MISMATCH")
    paired = paired_bootstrap(
        candidate["final_score"],
        baseline["final_score"],
        seed=int(bootstrap_seed),
    )
    return {
        **paired,
        **_win_loss_tie(
            candidate["final_score"], baseline["final_score"]
        ),
    }


def _development_phase(device: torch.device) -> dict:
    session = _session()
    if session["state"] == "DEV_DONE":
        if not DEVELOPMENT.exists():
            raise RuntimeError("M6_BLOCKED_RESUME_ARTIFACT_CORRUPT")
        return _read_json(DEVELOPMENT)
    if session["state"] not in {"TRAIN_DONE", "DEV_RUNNING"}:
        raise RuntimeError("M6_BLOCKED_RESUME_GIT_STATE")
    session["state"] = "DEV_RUNNING"
    atomic_write_json(SESSION, session)

    baseline_model, _ = load_m5_model(M5_CHECKPOINT, device)
    baseline_raw, baseline = _run_game_set(
        baseline_model,
        GAMES / "development" / "m5_baseline.npz",
        DEV_SEED_START,
        DEV_GAMES,
        device,
        "development/m5_baseline",
    )
    del baseline_model
    gc.collect()
    torch.cuda.empty_cache()

    runs = {}
    for seed in TRAINING_SEEDS:
        model, summary = _load_final_model(seed, device)
        raw, evidence = _run_game_set(
            model,
            GAMES / "development" / f"{seed}.npz",
            DEV_SEED_START,
            DEV_GAMES,
            device,
            f"development/{seed}",
        )
        evidence["checkpoint_path"] = summary["final_checkpoint_path"]
        evidence["checkpoint_sha256"] = summary["final_checkpoint_sha256"]
        evidence["paired_vs_m5"] = _paired_game_stats(
            raw, baseline_raw, DEV_BOOTSTRAP_SEEDS[int(seed)]
        )
        runs[str(seed)] = evidence
        del model
        gc.collect()
        torch.cuda.empty_cache()

    payload = {
        "schema_version": 1,
        "work_order_version": WORK_ORDER_VERSION,
        "seed_start": DEV_SEED_START,
        "games": DEV_GAMES,
        "policy_contract": {
            "pure_nn": True,
            "single_forward": True,
            "fp32": True,
            "search": False,
            "tuple": False,
            "d4_ensemble": False,
            "exploration": False,
            "tie_tolerance": 1e-7,
        },
        "m5_baseline": baseline,
        "m6_runs": runs,
    }
    atomic_write_json(DEVELOPMENT, payload)
    session["state"] = "DEV_DONE"
    atomic_write_json(SESSION, session)
    return payload


def _select_phase() -> dict:
    session = _session()
    if session["state"] == "CANDIDATE_LOCKED":
        if not SELECTION.exists():
            raise RuntimeError("M6_BLOCKED_RESUME_ARTIFACT_CORRUPT")
        return _read_json(SELECTION)
    if session["state"] != "DEV_DONE":
        raise RuntimeError("M6_BLOCKED_RESUME_GIT_STATE")
    development = _read_json(DEVELOPMENT)
    means = {
        int(seed): float(
            development["m6_runs"][str(seed)]["summary"]["mean_score"]
        )
        for seed in TRAINING_SEEDS
    }
    candidate_seed = min(
        means,
        key=lambda seed: (-means[seed], seed),
    )
    selected = development["m6_runs"][str(candidate_seed)]
    paired = selected["paired_vs_m5"]
    payload = {
        "schema_version": 1,
        "work_order_version": WORK_ORDER_VERSION,
        "selection_rule": (
            "maximum development mean score; exact tie -> smaller seed"
        ),
        "candidate_seed": int(candidate_seed),
        "candidate_checkpoint_path": selected["checkpoint_path"],
        "candidate_checkpoint_sha256": selected["checkpoint_sha256"],
        "development_mean_scores": {
            str(seed): means[int(seed)] for seed in TRAINING_SEEDS
        },
        "candidate_paired_vs_m5": paired,
        "dev_positive": bool(float(paired["mean_delta"]) > 0.0),
        "locked_before_validation_and_test": True,
    }
    atomic_write_json(SELECTION, payload)
    session["state"] = "CANDIDATE_LOCKED"
    atomic_write_json(SESSION, session)
    print(json.dumps(payload, indent=2), flush=True)
    return payload


def _gameplay_validation_phase(device: torch.device) -> dict:
    session = _session()
    if session["state"] == "GAMEPLAY_VALIDATION_DONE":
        if not GAMEPLAY_VALIDATION.exists():
            raise RuntimeError("M6_BLOCKED_RESUME_ARTIFACT_CORRUPT")
        return _read_json(GAMEPLAY_VALIDATION)
    if session["state"] not in {
        "CANDIDATE_LOCKED", "GAMEPLAY_VALIDATION_RUNNING"
    }:
        raise RuntimeError("M6_BLOCKED_RESUME_GIT_STATE")
    session["state"] = "GAMEPLAY_VALIDATION_RUNNING"
    atomic_write_json(SESSION, session)
    selection = _read_json(SELECTION)
    candidate_seed = int(selection["candidate_seed"])

    baseline_model, _ = load_m5_model(M5_CHECKPOINT, device)
    baseline_raw, baseline = _run_game_set(
        baseline_model,
        GAMES / "validation" / "m5_baseline.npz",
        VALIDATION_GAME_SEED_START,
        VALIDATION_GAMES,
        device,
        "validation/m5_baseline",
    )
    del baseline_model
    gc.collect()
    torch.cuda.empty_cache()

    model, summary = _load_final_model(candidate_seed, device)
    candidate_raw, candidate = _run_game_set(
        model,
        GAMES / "validation" / "m6_candidate.npz",
        VALIDATION_GAME_SEED_START,
        VALIDATION_GAMES,
        device,
        "validation/m6_candidate",
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()

    paired = _paired_game_stats(
        candidate_raw, baseline_raw, VALIDATION_BOOTSTRAP_SEED
    )
    validation_gain_pass = bool(
        float(paired["mean_delta"]) > 0.0
        and float(paired["ci95"][0]) > 0.0
    )
    candidate["checkpoint_path"] = summary["final_checkpoint_path"]
    candidate["checkpoint_sha256"] = summary["final_checkpoint_sha256"]
    payload = {
        "schema_version": 1,
        "work_order_version": WORK_ORDER_VERSION,
        "seed_start": VALIDATION_GAME_SEED_START,
        "games": VALIDATION_GAMES,
        "m5_baseline": baseline,
        "m6_candidate": candidate,
        "paired": paired,
        "validation_gain_pass": validation_gain_pass,
        "retraining_or_reselection_after_result": False,
    }
    atomic_write_json(GAMEPLAY_VALIDATION, payload)
    session["state"] = "GAMEPLAY_VALIDATION_DONE"
    atomic_write_json(SESSION, session)
    print(json.dumps({
        "phase": "gameplay-validation",
        "paired": paired,
        "validation_gain_pass": validation_gain_pass,
    }, indent=2), flush=True)
    return payload


def _teacher_test_phase(device: torch.device) -> dict:
    session = _session()
    if session["state"] == "TEACHER_TEST_DONE":
        if not TEACHER_TEST.exists():
            raise RuntimeError("M6_BLOCKED_RESUME_ARTIFACT_CORRUPT")
        return _read_json(TEACHER_TEST)
    if session["state"] != "TEST_DATA_RUNNING":
        raise RuntimeError("M6_BLOCKED_TEST_ISOLATION")
    test = _load_m6_split("test")
    if len(test["state"]) != 8192:
        raise RuntimeError("M6_BLOCKED_DATASET_ARTIFACT_CORRUPT")
    selection = _read_json(SELECTION)
    candidate_seed = int(selection["candidate_seed"])

    baseline_model, _ = load_m5_model(M5_CHECKPOINT, device)
    baseline = _teacher_eval(baseline_model, test, device)
    del baseline_model
    gc.collect()
    torch.cuda.empty_cache()

    model, summary = _load_final_model(candidate_seed, device)
    candidate = _teacher_eval(model, test, device)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    payload = {
        "schema_version": 1,
        "work_order_version": WORK_ORDER_VERSION,
        "samples": 8192,
        "candidate_seed": candidate_seed,
        "candidate_checkpoint_path": summary["final_checkpoint_path"],
        "candidate_checkpoint_sha256": summary["final_checkpoint_sha256"],
        "m5_baseline": baseline,
        "m6_candidate": candidate,
        "one_time_test_consumed": True,
        "affected_selection": False,
    }
    atomic_write_json(TEACHER_TEST, payload)
    session["state"] = "TEACHER_TEST_DONE"
    atomic_write_json(SESSION, session)
    return payload


def _final_gameplay_phase(device: torch.device) -> dict:
    session = _session()
    if session["state"] == "FINAL_GAMEPLAY_DONE":
        if not FINAL_GAMEPLAY.exists():
            raise RuntimeError("M6_BLOCKED_RESUME_ARTIFACT_CORRUPT")
        return _read_json(FINAL_GAMEPLAY)
    if session["state"] not in {
        "TEACHER_TEST_DONE", "FINAL_GAMEPLAY_RUNNING"
    }:
        raise RuntimeError("M6_BLOCKED_RESUME_GIT_STATE")
    session["state"] = "FINAL_GAMEPLAY_RUNNING"
    atomic_write_json(SESSION, session)
    selection = _read_json(SELECTION)
    candidate_seed = int(selection["candidate_seed"])

    baseline_model, _ = load_m5_model(M5_CHECKPOINT, device)
    baseline_raw, baseline = _run_game_set(
        baseline_model,
        FINAL / "m5_baseline_10000.npz",
        FINAL_SEED_START,
        FINAL_GAMES,
        device,
        "final/m5_baseline",
    )
    del baseline_model
    gc.collect()
    torch.cuda.empty_cache()

    model, summary = _load_final_model(candidate_seed, device)
    candidate_raw, candidate = _run_game_set(
        model,
        FINAL / "m6_candidate_10000.npz",
        FINAL_SEED_START,
        FINAL_GAMES,
        device,
        "final/m6_candidate",
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()

    paired = _paired_game_stats(
        candidate_raw, baseline_raw, FINAL_BOOTSTRAP_SEED
    )
    final_gain_pass = bool(
        float(paired["mean_delta"]) > 0.0
        and float(paired["ci95"][0]) > 0.0
    )
    candidate["checkpoint_path"] = summary["final_checkpoint_path"]
    candidate["checkpoint_sha256"] = summary["final_checkpoint_sha256"]
    payload = {
        "schema_version": 1,
        "work_order_version": WORK_ORDER_VERSION,
        "seed_start": FINAL_SEED_START,
        "games": FINAL_GAMES,
        "m5_baseline": baseline,
        "m6_candidate": candidate,
        "paired": paired,
        "final_gain_pass": final_gain_pass,
        "retraining_or_reselection_after_result": False,
    }
    atomic_write_json(FINAL_GAMEPLAY, payload)
    session["state"] = "FINAL_GAMEPLAY_DONE"
    atomic_write_json(SESSION, session)
    print(json.dumps({
        "phase": "final-gameplay",
        "paired": paired,
        "final_gain_pass": final_gain_pass,
    }, indent=2), flush=True)
    return payload


def _promote_phase() -> dict:
    session = _session()
    if session["state"] == "PROMOTION_DECIDED":
        if not PROMOTION.exists():
            raise RuntimeError("M6_BLOCKED_RESUME_ARTIFACT_CORRUPT")
        return _read_json(PROMOTION)
    if session["state"] != "FINAL_GAMEPLAY_DONE":
        raise RuntimeError("M6_BLOCKED_RESUME_GIT_STATE")
    selection = _read_json(SELECTION)
    gameplay_validation = _read_json(GAMEPLAY_VALIDATION)
    final_gameplay = _read_json(FINAL_GAMEPLAY)
    teacher_validation = _read_json(TEACHER_VALIDATION)

    dev_positive = bool(selection["dev_positive"])
    validation_gain_pass = bool(
        gameplay_validation["validation_gain_pass"]
    )
    final_gain_pass = bool(final_gameplay["final_gain_pass"])
    learning_sanity_pass = all(
        bool(value)
        for value in teacher_validation["learning_sanity"].values()
    )
    pre_p10_gates_pass = bool(
        SEARCH_PROFILE.exists()
        and ANCHOR_VERIFY.exists()
        and LABEL_MANIFEST.exists()
        and TEACHER_TEST.exists()
        and SELECTION.exists()
    )
    result = promotion_result(
        dev_positive=dev_positive,
        validation_gain_pass=validation_gain_pass,
        final_gain_pass=final_gain_pass,
        learning_sanity_pass=learning_sanity_pass,
        all_gates_pass=pre_p10_gates_pass,
    )
    payload = {
        "schema_version": 1,
        "work_order_version": WORK_ORDER_VERSION,
        "dev_positive": dev_positive,
        "validation_gain_pass": validation_gain_pass,
        "final_gain_pass": final_gain_pass,
        "correction_learning_sanity_pass": learning_sanity_pass,
        "pre_p10_provenance_data_test_isolation_gates_pass": (
            pre_p10_gates_pass
        ),
        "promotion_result": result,
        "note": (
            "P10 regression/candidate gates remain mandatory; "
            "a P10 failure blocks candidate completion."
        ),
    }
    atomic_write_json(PROMOTION, payload)
    session["state"] = "PROMOTION_DECIDED"
    atomic_write_json(SESSION, session)
    print(json.dumps(payload, indent=2), flush=True)
    return payload


def _pytest_counts() -> dict:
    path = ART / "pytest.xml"
    if not path.exists():
        raise RuntimeError("M6_BLOCKED_RESUME_ARTIFACT_CORRUPT")
    root = ET.parse(path).getroot()
    suites = list(root.findall("testsuite")) if root.tag == "testsuites" else [root]
    if not suites:
        raise RuntimeError("M6_BLOCKED_RUNTIME_ERROR")
    tests = sum(int(suite.attrib.get("tests", 0)) for suite in suites)
    failures = sum(int(suite.attrib.get("failures", 0)) for suite in suites)
    errors = sum(int(suite.attrib.get("errors", 0)) for suite in suites)
    skipped = sum(int(suite.attrib.get("skipped", 0)) for suite in suites)
    if (tests, failures, errors, skipped) != (553, 0, 0, 0):
        raise RuntimeError("M6_BLOCKED_RUNTIME_ERROR")
    return {
        "tests": tests,
        "passed": tests - failures - errors - skipped,
        "failures": failures,
        "errors": errors,
        "skipped": skipped,
        "junit_path": "artifacts/m6/pytest.xml",
        "junit_sha256": sha256_file(path),
    }


def _source_manifest_evidence(split: str) -> dict:
    path = DATA / f"source_{split}_manifest.json"
    if not path.exists():
        raise RuntimeError("M6_BLOCKED_RESUME_ARTIFACT_CORRUPT")
    manifest = _read_json(path)
    source_path = ROOT / manifest["source_path"]
    if sha256_file(source_path) != manifest["source_sha256"]:
        raise RuntimeError("M6_BLOCKED_DATASET_ARTIFACT_CORRUPT")
    return {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "sha256": sha256_file(path),
        "complete_games": int(manifest["complete_games"]),
        "states": int(manifest["states"]),
        "retry_count": int(sum(
            int(game.get("retry_index", 0))
            for game in manifest["games"]
        )),
        "policy_contract": manifest["policy_contract"],
        "canonical_unaugmented": bool(manifest["canonical_unaugmented"]),
    }


def _correction_dataset_evidence() -> dict:
    manifest = _m6_manifest()
    splits = {}
    all_disagreement = []
    for split_name, split in manifest["splits"].items():
        rows = sorted(
            split["shards"], key=lambda row: int(row["shard_index"])
        )
        total_rows = 0
        rates = []
        for row in rows:
            validate_completed_shard(ROOT / row["path"], row["sha256"])
            total_rows += int(row["rows"])
            rates.append(float(row.get("disagreement_rate", 0.0)))
        splits[split_name] = {
            "shards": len(rows),
            "rows": total_rows,
            "all_sha_valid": True,
            "mean_shard_disagreement_rate": (
                float(np.mean(rates)) if rates else None
            ),
        }
        all_disagreement.extend(rates)
    return {
        "manifest_path": str(LABEL_MANIFEST.relative_to(ROOT)).replace("\\", "/"),
        "manifest_sha256": sha256_file(LABEL_MANIFEST),
        "teacher_sha256": manifest["teacher_checkpoint_sha256"],
        "student_checkpoint_sha256": manifest["student_checkpoint_sha256"],
        "decision_depth": int(manifest["decision_depth"]),
        "value_semantics": manifest["value_semantics"],
        "splits": splits,
        "overall_mean_shard_disagreement_rate": (
            float(np.mean(all_disagreement)) if all_disagreement else None
        ),
    }


def _training_evidence() -> dict:
    runs = {}
    for seed in TRAINING_SEEDS:
        path = _progress_path(seed)
        if not path.exists():
            raise RuntimeError("M6_BLOCKED_RESUME_ARTIFACT_CORRUPT")
        progress = _read_json(path)
        if progress.get("completed") is not True:
            raise RuntimeError("M6_BLOCKED_RESUME_ARTIFACT_CORRUPT")
        summary = progress["summary"]
        if int(summary["optimizer_steps"]) != 7680:
            raise RuntimeError("M6_BLOCKED_RESUME_ARTIFACT_CORRUPT")
        final_path = ROOT / summary["final_checkpoint_path"]
        if sha256_file(final_path) != summary["final_checkpoint_sha256"]:
            raise RuntimeError("M6_BLOCKED_RESUME_ARTIFACT_CORRUPT")
        runs[str(seed)] = summary
    return {
        "architecture": ARCHITECTURE,
        "exact_start_checkpoint_sha256": M5_CHECKPOINT_SHA256,
        "batch_size": 1024,
        "correction_rows_per_batch": 512,
        "anchor_rows_per_batch": 512,
        "epochs": EPOCHS,
        "optimizer_steps_per_seed": 7680,
        "precision": PRECISION,
        "runs": runs,
    }


def _performance_evidence(training: dict, dataset: dict) -> dict:
    label_manifest = _m6_manifest()
    label_wall = sum(
        float(shard.get("label_wall_seconds", 0.0))
        for split in label_manifest["splits"].values()
        for shard in split["shards"]
    )
    return {
        "search_median_roots_per_second": float(
            _read_json(SEARCH_PROFILE)["median_roots_per_second"]
        ),
        "search_projections_seconds": _read_json(
            SEARCH_PROFILE
        )["projections_seconds"],
        "actual_teacher_label_wall_seconds": float(label_wall),
        "training_runs": {
            seed: {
                "wall_seconds": float(row["training_compute_wall_seconds"]),
                "samples_per_second": float(
                    row["training_only_samples_per_s"]
                ),
                "peak_vram_bytes": int(row["peak_vram_bytes"]),
            }
            for seed, row in training["runs"].items()
        },
        "gameplay_raw_sets": {
            "development": 4,
            "validation": 2,
            "final": 2,
        },
        "dataset_rows": dataset["splits"],
    }


def _provenance() -> dict:
    path = ART / "provenance.json"
    if not path.exists():
        raise RuntimeError("M6_BLOCKED_RESUME_ARTIFACT_CORRUPT")
    return _read_json(path)


def _student_rollout_evidence() -> dict:
    return {
        "train": _source_manifest_evidence("train"),
        "validation": _source_manifest_evidence("validation"),
        "test": _source_manifest_evidence("test"),
        "sampling": "np.rint(np.linspace(0,moves-1,128)).astype(int64)",
        "states_per_accepted_game": 128,
        "retry_seed_stride": 1_000_000_000,
        "max_retry_index": 32,
    }


def _report_phase() -> dict:
    session = _session()
    if session["state"] == "REPORT_WRITTEN":
        machine = REPORTS / "m6_state_correction.json"
        if not machine.exists():
            raise RuntimeError("M6_BLOCKED_RESUME_ARTIFACT_CORRUPT")
        return _read_json(machine)
    if session["state"] not in {"P10_RUNNING", "P10_DONE"}:
        raise RuntimeError("M6_BLOCKED_RESUME_GIT_STATE")

    regression = _pytest_counts()
    correction = _correction_dataset_evidence()
    if correction["splits"]["train"]["rows"] != 131_072:
        raise RuntimeError("M6_BLOCKED_DATASET_ARTIFACT_CORRUPT")
    if correction["splits"]["validation"]["rows"] != 8_192:
        raise RuntimeError("M6_BLOCKED_DATASET_ARTIFACT_CORRUPT")
    if correction["splits"]["test"]["rows"] != 8_192:
        raise RuntimeError("M6_BLOCKED_DATASET_ARTIFACT_CORRUPT")
    training = _training_evidence()
    promotion = _read_json(PROMOTION)
    result_suffix = promotion["promotion_result"]
    result = f"M6_CANDIDATE_EVIDENCE_COMPLETE_{result_suffix}"

    summary = {
        "schema_version": 1,
        "result": result,
        "provenance": _provenance(),
        "student_baseline": {
            "checkpoint_path": (
                "artifacts/m5/checkpoints/524k/20262103_final.pt"
            ),
            "checkpoint_sha256": M5_CHECKPOINT_SHA256,
            "scale": "524k",
            "seed": 20262103,
            "architecture": ARCHITECTURE,
        },
        "teacher": {
            "path": f"teacher_checkpoints/m6/{TEACHER_FILENAME}",
            "sha256": TEACHER_SHA256,
            "semantics": {
                "tuple": "RAW_TUPLE_HEURISTIC",
                "formal_leaf": "FORMAL_STATE_TUPLE_HEURISTIC",
                "root_action": "SEARCH_VALUE_RAW_LEAF",
            },
            "search_depth": 3,
        },
        "search_profile": _read_json(SEARCH_PROFILE),
        "student_rollout": _student_rollout_evidence(),
        "correction_dataset": correction,
        "anchor": _read_json(ANCHOR_VERIFY),
        "training": training,
        "teacher_validation": _read_json(TEACHER_VALIDATION),
        "development_gameplay": _read_json(DEVELOPMENT),
        "selection": _read_json(SELECTION),
        "gameplay_validation": _read_json(GAMEPLAY_VALIDATION),
        "teacher_test": _read_json(TEACHER_TEST),
        "final_gameplay": _read_json(FINAL_GAMEPLAY),
        "promotion": promotion,
        "performance": _performance_evidence(training, correction),
        "regression": regression,
    }
    expected_keys = {
        "schema_version", "result", "provenance", "student_baseline",
        "teacher", "search_profile", "student_rollout",
        "correction_dataset", "anchor", "training",
        "teacher_validation", "development_gameplay", "selection",
        "gameplay_validation", "teacher_test", "final_gameplay",
        "promotion", "performance", "regression",
    }
    if set(summary) != expected_keys:
        raise RuntimeError("M6_BLOCKED_REPORT_RENDERING")

    REPORTS.mkdir(parents=True, exist_ok=True)
    machine = REPORTS / "m6_state_correction.json"
    atomic_write_json(machine, summary)
    session["state"] = "P10_DONE"
    atomic_write_json(SESSION, session)


    selection = summary["selection"]
    gameplay_validation = summary["gameplay_validation"]
    final_gameplay = summary["final_gameplay"]
    final_base = final_gameplay["m5_baseline"]["summary"]
    final_cand = final_gameplay["m6_candidate"]["summary"]
    lines = [
        "# M6 Student State Correction Candidate Report",
        "",
        "## 1. RESULT",
        "",
        f"**{result}**",
        "",
        "Candidate evidence only; independent M6 audit is still required.",
        "",
        "## 2. STARTING STATE",
        "",
        f"- planning base: {summary['provenance']['base_head']}",
        f"- work order: {WORK_ORDER_VERSION}",
        f"- prompt SHA-256: {PROMPT_SHA256}",
        "",
        "## 3. FROZEN PROVENANCE",
        "",
        "- M0-M5 authoritative frozen tags were verified unchanged.",
        f"- M5 Student SHA-256: {M5_CHECKPOINT_SHA256}",
        f"- Teacher SHA-256: {TEACHER_SHA256}",
        "",
        "## 4. SEARCH PROFILE",
        "",
        f"- median roots/s: {summary['search_profile']['median_roots_per_second']:.6f}",
        f"- orchestration share: {summary['search_profile']['orchestration_share']:.6f}",
        "- all three 256-root repeats passed exact correctness.",
        "",
        "## 5. STUDENT ROLLOUT",
        "",
        "- source policy: frozen M5 pure-NN single-forward FP32.",
        "- spawn RNG and tie RNG are independent and seed-stable.",
        "- 128 canonical states were sampled per accepted game.",
        "",
        "## 6. CORRECTION DATASET",
        "",
        f"- train states: {correction['splits']['train']['rows']}",
        f"- validation states: {correction['splits']['validation']['rows']}",
        f"- test states: {correction['splits']['test']['rows']}",
        "- Teacher relabel depth: 3, SEARCH_VALUE_RAW_LEAF.",
        "",
        "## 7. ANCHOR",
        "",
        f"- M5 anchor manifest SHA-256: {M5_ANCHOR_MANIFEST_SHA256}",
        "- anchor rows: 524,288 M5 train rows only.",
        "",
        "## 8. TRAINING",
        "",
        "- architecture: ResidualMLP2048",
        "- three seeds: 20263101 / 20263102 / 20263103",
        "- exact mixture: 512 correction + 512 anchor per batch.",
        "- 30 epochs / 7,680 optimizer steps per seed.",
        "- value/afterstate heads remained bit-identical.",
        "",
        "## 9. CANDIDATE SELECTION",
        "",
        f"- locked candidate seed: {selection['candidate_seed']}",
        f"- candidate checkpoint SHA-256: {selection['candidate_checkpoint_sha256']}",
        f"- dev_positive: {selection['dev_positive']}",
        "",
        "## 10. GAMEPLAY VALIDATION",
        "",
        f"- paired mean delta: {gameplay_validation['paired']['mean_delta']:.6f}",
        f"- CI95: {gameplay_validation['paired']['ci95']}",
        f"- validation_gain_pass: {gameplay_validation['validation_gain_pass']}",
        "",
        "## 11. TEACHER TEST",
        "",
        "- one-time Teacher-test split consumed only after candidate lock.",
        "- Teacher test did not alter selection or training.",
        "",
        "## 12. FINAL 10,000-GAME TEST",
        "",
        f"- paired mean delta: {final_gameplay['paired']['mean_delta']:.6f}",
        f"- CI95: {final_gameplay['paired']['ci95']}",
        f"- final_gain_pass: {final_gameplay['final_gain_pass']}",
        "",
        f"- M5 mean / median: {final_base['mean_score']:.6f} / {final_base['median_score']:.6f}",
        f"- M6 mean / median: {final_cand['mean_score']:.6f} / {final_cand['median_score']:.6f}",
        "",
        "## 13. PROMOTION",
        "",
        f"- result: **{promotion['promotion_result']}**",
        "- NO_PROMOTION is a valid completed research outcome.",
        "",
        "## 14. REGRESSION",
        "",
        f"- pytest: {regression['passed']} passed",
        "- failures / errors / skipped: 0 / 0 / 0",
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
        "M6 is not audited or frozen by this execution.",
        "No M6 audited tag was created.",
        "M7 was not entered.",
        "",
    ]
    report = REPORTS / "M6_REPORT.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if len(report.read_text(encoding="utf-8").splitlines()) < 50:
        raise RuntimeError("M6_BLOCKED_REPORT_RENDERING")
    json.loads(machine.read_text(encoding="utf-8"))
    session["state"] = "REPORT_WRITTEN"
    atomic_write_json(SESSION, session)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        required=True,
        choices=(
            "anchor-check",
            "smoke",
            "train",
            "development",
            "select",
            "gameplay-validation",
            "teacher-test",
            "final-gameplay",
            "promote",
            "report",
        ),
    )
    parser.add_argument("--resume", action="store_true", required=True)
    args = parser.parse_args()

    if args.phase == "report":
        _report_phase()
        return

    if not torch.cuda.is_available():
        raise RuntimeError("M6_BLOCKED_RUNTIME_ENVIRONMENT")
    if trainable_parameter_count(ResidualMLP2048()) != PARAMETER_COUNT:
        raise RuntimeError("M6_BLOCKED_MODEL_DRIFT")
    device = torch.device("cuda")

    if args.phase == "anchor-check":
        _anchor_phase()
    elif args.phase == "smoke":
        _smoke_phase(device)
    elif args.phase == "train":
        _train_phase(device)
    elif args.phase == "development":
        _development_phase(device)
    elif args.phase == "select":
        _select_phase()
    elif args.phase == "gameplay-validation":
        _gameplay_validation_phase(device)
    elif args.phase == "teacher-test":
        _teacher_test_phase(device)
    elif args.phase == "final-gameplay":
        _final_gameplay_phase(device)
    elif args.phase == "promote":
        _promote_phase()


if __name__ == "__main__":
    main()
