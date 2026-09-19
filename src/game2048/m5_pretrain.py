"""M5 deterministic helpers for Teacher policy pretraining."""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .m2_models import ResidualMLP2048
from .m2_symmetry import inverse_transform_q_values, transform_action_batch, transform_board_batch
from .m4_compare import evaluate_game_shard, score_summary

WORK_ORDER_VERSION = "M5_WO_CLOSURE_V2"
BASE_HEAD = "0058fa117a9ccc6d999597796fa50387348bf987"
PROMPT_SHA256 = "C31E9B74A24EE65BFBD3C134679991B8573D4733C74593041B2E167B21A89ACF"
TEACHER_SHA256 = "7192719323A073BA2B6B19B62CB7D46EF4AA90ECC8C4AE6BAF27AD0C51566A84"
M4_PRIMARY_SHA256 = "2196DCADEBE1661E8BBF5AADA6B7C1F9D1F199C6944FF70F0CD1C5C9B5B1BBD5"
TEACHER_VERSION = "ordinary_td_comparator_ep4800000_7192719323a0"
ARCHITECTURE = "ResidualMLP2048"
PARAMETER_COUNT = 5_264_710
TRAINING_SEEDS = (20262101, 20262102, 20262103)
SCALES = ("32k", "131k", "524k")
SCALE_ROWS = {"32k": 32_768, "131k": 131_072, "524k": 524_288}
SCALE_GAMES = {"32k": 256, "131k": 1024, "524k": 4096}
SCALE_OFFSETS = {"32k": 0, "131k": 100_000, "524k": 200_000}
STATES_PER_GAME = 128
SOURCE_RETRY_SEED_STRIDE = 1_000_000_000
SOURCE_RETRY_MAX_INDEX = 32
SHARD_STATES = 2048
GAMES_PER_SHARD = 16
EPOCHS = 30
BATCH_SIZE = 1024
TRAIN_SEED_BASE = 20265001
VALIDATION_SEED_BASE = 20275001
TEST_SEED_BASE = 20276001
DEV_SEED_START = 20261001
DEV_GAMES = 2000
FINAL_SEED_START = 20277001
FINAL_GAMES = 10_000
GAME_SHARD_SIZE = 100
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEEDS = {
    "m4_vs_32k": 20265201,
    "32k_vs_131k": 20265202,
    "pre524_vs_524k": 20265203,
    "final_vs_m4": 20265204,
}
REQUIRED_SHARD_KEYS = {
    "state", "teacher_value", "reward", "afterstate", "legal_mask",
    "game_id", "game_seed", "step_index", "current_score", "max_tile_exp",
    "teacher_version", "checkpoint_sha256", "decision_depth",
    "value_semantics", "data_source",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def atomic_write_json(path: str | Path, payload: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, target)


def atomic_save_npz(path: str | Path, **arrays: np.ndarray) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, target)


def atomic_torch_save(path: str | Path, payload: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, target)


def plan_id(scale: str, seed: int) -> str:
    if scale not in SCALES or int(seed) not in TRAINING_SEEDS:
        raise ValueError("unsupported M5 scale/seed")
    return f"M5_PLAN_V1:{scale}:{int(seed)}:PCG64_STATELESS_EPOCH"


def plan_sha256(scale: str, seed: int) -> str:
    return hashlib.sha256(plan_id(scale, seed).encode("ascii")).hexdigest().upper()


def stateless_epoch_plan(
    scale: str, seed: int, epoch: int, rows: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    if scale not in SCALES:
        raise ValueError("unknown scale")
    if not 0 <= int(epoch) < EPOCHS:
        raise ValueError("epoch must be 0..29")
    count = SCALE_ROWS[scale] if rows is None else int(rows)
    if count <= 0:
        raise ValueError("rows must be positive")
    rng = np.random.Generator(
        np.random.PCG64(int(seed) + SCALE_OFFSETS[scale] + int(epoch))
    )
    order = rng.permutation(np.arange(count, dtype=np.int64))
    tids = rng.integers(0, 8, size=count, dtype=np.uint8)
    return np.ascontiguousarray(order), np.ascontiguousarray(tids)


def sample_positions(moves: int) -> np.ndarray:
    moves = int(moves)
    if moves < STATES_PER_GAME:
        raise RuntimeError("M5_BLOCKED_SOURCE_GAME_TOO_SHORT")
    positions = np.rint(
        np.linspace(0, moves - 1, STATES_PER_GAME)
    ).astype(np.int64)
    if np.unique(positions).size != STATES_PER_GAME:
        raise RuntimeError("M5_BLOCKED_SOURCE_SAMPLE_DUPLICATE")
    return positions


def split_game_seed(split: str, local_index: int) -> tuple[int, int]:
    i = int(local_index)
    if split == "train":
        if not 0 <= i < 4096:
            raise ValueError("train local index out of range")
        return i, TRAIN_SEED_BASE + i
    if split == "validation":
        if not 0 <= i < 64:
            raise ValueError("validation local index out of range")
        return 100_000 + i, VALIDATION_SEED_BASE + i
    if split == "test":
        if not 0 <= i < 64:
            raise ValueError("test local index out of range")
        return 200_000 + i, TEST_SEED_BASE + i
    raise ValueError("unknown split")


def source_attempt_seed(
    split: str,
    local_index: int,
    retry_index: int,
) -> tuple[int, int, int]:
    retry = int(retry_index)
    if not 0 <= retry <= SOURCE_RETRY_MAX_INDEX:
        raise ValueError("source retry index out of range")
    game_id, canonical_seed = split_game_seed(split, local_index)
    effective_seed = (
        int(canonical_seed)
        + retry * SOURCE_RETRY_SEED_STRIDE
    )
    return int(game_id), int(canonical_seed), int(effective_seed)


def scale_shard_ranges(scale: str) -> list[tuple[int, int]]:
    rows = SCALE_ROWS[scale]
    return [(start, start + SHARD_STATES) for start in range(0, rows, SHARD_STATES)]


def shard_game_ids(split: str, shard_index: int) -> np.ndarray:
    start_local = int(shard_index) * GAMES_PER_SHARD
    if split == "train":
        max_games = 4096
        base = 0
    elif split == "validation":
        max_games = 64
        base = 100_000
    elif split == "test":
        max_games = 64
        base = 200_000
    else:
        raise ValueError("unknown split")
    if start_local < 0 or start_local + GAMES_PER_SHARD > max_games:
        raise ValueError("shard index out of range")
    return np.arange(
        base + start_local, base + start_local + GAMES_PER_SHARD, dtype=np.int64
    )


def teacher_best_action(
    teacher_value: np.ndarray, legal_mask: np.ndarray
) -> np.ndarray:
    values = np.asarray(teacher_value, dtype=np.float64)
    legal = np.asarray(legal_mask, dtype=np.bool_)
    if values.ndim != 2 or values.shape[1] != 4 or legal.shape != values.shape:
        raise ValueError("teacher_value/legal_mask must have shape (N,4)")
    if bool((~legal.any(axis=1)).any()):
        raise ValueError("terminal rows cannot be Teacher targets")
    masked = np.where(legal, values, np.nan)
    if bool((~np.isfinite(masked[legal])).any()):
        raise ValueError("legal Teacher values must be finite")
    return np.nanargmax(masked, axis=1).astype(np.int64)


def random_legal_baseline(legal_mask: np.ndarray) -> float:
    legal = np.asarray(legal_mask, dtype=np.bool_)
    if legal.ndim != 2 or legal.shape[1] != 4 or bool((~legal.any(axis=1)).any()):
        raise ValueError("legal_mask must be nonterminal (N,4)")
    return float(np.mean(1.0 / legal.sum(axis=1).astype(np.float64)))


def set_determinism(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True, warn_only=True)


def make_optimizer(model: torch.nn.Module) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=1e-4,
        betas=(0.9, 0.999), eps=1e-8, amsgrad=False,
        foreach=False, fused=False, capturable=False,
    )


def autocast_context(precision: str):
    if precision == "BF16 autocast":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if precision == "FP32":
        return contextlib.nullcontext()
    raise ValueError("unsupported precision")


def snapshot_parameters(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
    }


def parameter_update_checks(
    model: torch.nn.Module, initial: dict[str, torch.Tensor]
) -> dict:
    current = {name: p.detach().cpu() for name, p in model.named_parameters()}
    q_changed = any(
        not torch.equal(current[n], initial[n])
        for n in current if n.startswith("q_head.")
    )
    backbone_changed = any(
        not torch.equal(current[n], initial[n])
        for n in current
        if not (
            n.startswith("q_head.")
            or n.startswith("value_head.")
            or n.startswith("afterstate_head.")
        )
    )
    value_unchanged = all(
        torch.equal(current[n], initial[n])
        for n in current
        if n.startswith("value_head.") or n.startswith("afterstate_head.")
    )
    return {
        "q_head_updated": bool(q_changed),
        "backbone_updated": bool(backbone_changed),
        "value_heads_unchanged": bool(value_unchanged),
    }


def policy_only_step(
    model: torch.nn.Module,
    boards: torch.Tensor,
    target: torch.Tensor,
    optimizer: torch.optim.Optimizer,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    logits = model(boards)
    loss = F.cross_entropy(logits, target, reduction="mean")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(loss.detach().item())


def paired_bootstrap(
    first_scores: np.ndarray,
    second_scores: np.ndarray,
    *,
    seed: int,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict:
    first = np.asarray(first_scores, dtype=np.float64)
    second = np.asarray(second_scores, dtype=np.float64)
    if first.ndim != 1 or first.shape != second.shape or first.size == 0:
        raise ValueError("paired score vectors must be equal nonempty 1-D arrays")
    delta = first - second
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    means = np.empty(int(resamples), dtype=np.float64)
    done = 0
    while done < int(resamples):
        take = min(128, int(resamples) - done)
        indices = rng.integers(
            0, delta.size, size=(take, delta.size), dtype=np.int64
        )
        means[done:done + take] = delta[indices].mean(axis=1)
        done += take
    lower, upper = np.quantile(means, [0.025, 0.975])
    return {
        "mean_delta": float(delta.mean()),
        "ci95": [float(lower), float(upper)],
        "pairs": int(delta.size),
        "bootstrap_resamples": int(resamples),
        "bootstrap_seed": int(seed),
    }


def teacher_positive_signal(metrics_old: dict, metrics_new: dict) -> bool:
    return bool(
        float(metrics_new["ce"]) <= 0.99 * float(metrics_old["ce"])
        or float(metrics_new["legal_best_action_accuracy"])
        >= float(metrics_old["legal_best_action_accuracy"]) + 0.005
        or float(metrics_new["legal_pairwise_ranking_accuracy"])
        >= float(metrics_old["legal_pairwise_ranking_accuracy"]) + 0.005
    )


def scale_gate_131k(
    scores_32k: np.ndarray,
    scores_131k: np.ndarray,
    median_32k: dict,
    median_131k: dict,
) -> dict:
    paired = paired_bootstrap(
        scores_131k, scores_32k, seed=BOOTSTRAP_SEEDS["32k_vs_131k"]
    )
    lower, upper = paired["ci95"]
    score_positive = paired["mean_delta"] > 0.0 and upper > 0.0
    teacher_positive = teacher_positive_signal(median_32k, median_131k)
    if lower > 0.0:
        winner = "131k"
    elif upper < 0.0:
        winner = "32k"
    else:
        winner = "131k" if teacher_positive else "32k"
    return {
        "paired_131k_minus_32k": paired,
        "score_positive_signal": bool(score_positive),
        "teacher_positive_signal": bool(teacher_positive),
        "trigger_524k": bool(score_positive or teacher_positive),
        "pre_524_winner": winner,
    }


def select_with_524k(
    previous_scale: str,
    previous_scores: np.ndarray,
    scores_524k: np.ndarray,
    previous_median: dict,
    median_524k: dict,
) -> dict:
    paired = paired_bootstrap(
        scores_524k, previous_scores, seed=BOOTSTRAP_SEEDS["pre524_vs_524k"]
    )
    lower, upper = paired["ci95"]
    teacher_improved = bool(
        float(median_524k["ce"]) <= 0.99 * float(previous_median["ce"])
        or float(median_524k["legal_best_action_accuracy"])
        >= float(previous_median["legal_best_action_accuracy"]) + 0.005
        or float(median_524k["legal_pairwise_ranking_accuracy"])
        >= float(previous_median["legal_pairwise_ranking_accuracy"]) + 0.005
    )
    if lower > 0.0:
        selected = "524k"
    elif upper < 0.0:
        selected = previous_scale
    else:
        selected = "524k" if teacher_improved else previous_scale
    return {
        "paired_524k_minus_previous": paired,
        "teacher_improved": teacher_improved,
        "selected_scale": selected,
    }


def aggregate_seed_scores(score_vectors: Sequence[np.ndarray]) -> np.ndarray:
    if len(score_vectors) != 3:
        raise ValueError("exactly three training-seed score vectors required")
    matrix = np.stack([np.asarray(x, dtype=np.float64) for x in score_vectors])
    if matrix.shape[1] != DEV_GAMES:
        raise ValueError("development vectors must contain 2000 games")
    return matrix.mean(axis=0)


def median_validation(metrics: Sequence[dict]) -> dict:
    if len(metrics) != 3:
        raise ValueError("exactly three validation metrics required")
    keys = ("ce", "legal_best_action_accuracy", "legal_pairwise_ranking_accuracy")
    return {key: float(np.median([float(row[key]) for row in metrics])) for key in keys}


def validate_completed_shard(path: str | Path, expected_sha256: str) -> dict:
    target = Path(path)
    if not target.exists() or sha256_file(target) != str(expected_sha256).upper():
        raise RuntimeError("M5_BLOCKED_DATASET_ARTIFACT_CORRUPT")
    try:
        with np.load(target, allow_pickle=False) as data:
            if set(data.files) != REQUIRED_SHARD_KEYS:
                raise RuntimeError("M5_BLOCKED_DATASET_ARTIFACT_CORRUPT")
            shapes = {
                "state": (SHARD_STATES, 16),
                "teacher_value": (SHARD_STATES, 4),
                "reward": (SHARD_STATES, 4),
                "afterstate": (SHARD_STATES, 4, 16),
                "legal_mask": (SHARD_STATES, 4),
                "game_id": (SHARD_STATES,),
                "game_seed": (SHARD_STATES,),
                "step_index": (SHARD_STATES,),
                "current_score": (SHARD_STATES,),
                "max_tile_exp": (SHARD_STATES,),
                "teacher_version": (SHARD_STATES,),
                "checkpoint_sha256": (SHARD_STATES,),
                "decision_depth": (SHARD_STATES,),
                "value_semantics": (SHARD_STATES,),
                "data_source": (SHARD_STATES,),
            }
            for key, shape in shapes.items():
                if data[key].shape != shape:
                    raise RuntimeError("M5_BLOCKED_DATASET_ARTIFACT_CORRUPT")
            if data["state"].dtype != np.uint8:
                raise RuntimeError("M5_BLOCKED_DATASET_ARTIFACT_CORRUPT")
            if data["teacher_value"].dtype != np.float64:
                raise RuntimeError("M5_BLOCKED_DATASET_ARTIFACT_CORRUPT")
            if data["reward"].dtype != np.int32:
                raise RuntimeError("M5_BLOCKED_DATASET_ARTIFACT_CORRUPT")
            if data["afterstate"].dtype != np.uint8:
                raise RuntimeError("M5_BLOCKED_DATASET_ARTIFACT_CORRUPT")
            if data["legal_mask"].dtype != np.bool_:
                raise RuntimeError("M5_BLOCKED_DATASET_ARTIFACT_CORRUPT")
            game_ids = np.asarray(data["game_id"], dtype=np.int64)
            unique, counts = np.unique(game_ids, return_counts=True)
            if unique.size != GAMES_PER_SHARD or not bool((counts == STATES_PER_GAME).all()):
                raise RuntimeError("M5_BLOCKED_DATASET_ARTIFACT_CORRUPT")
            if not bool(np.asarray(data["legal_mask"]).any(axis=1).all()):
                raise RuntimeError("M5_BLOCKED_DATASET_ARTIFACT_CORRUPT")
            illegal = ~np.asarray(data["legal_mask"])
            if not bool(np.isnan(np.asarray(data["teacher_value"])[illegal]).all()):
                raise RuntimeError("M5_BLOCKED_DATASET_ARTIFACT_CORRUPT")
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError("M5_BLOCKED_DATASET_ARTIFACT_CORRUPT") from exc
    return {"path": str(target), "sha256": str(expected_sha256).upper()}


def validate_manifest_completed_shards(manifest: dict, root: str | Path) -> None:
    base = Path(root)
    for split in manifest.get("splits", {}).values():
        for row in split.get("shards", []):
            if row.get("status") == "complete":
                validate_completed_shard(base / row["path"], row["sha256"])


def load_label_split(
    root: str | Path, manifest: dict, split: str, max_shards: int | None = None
) -> dict[str, np.ndarray]:
    if split == "test":
        raise RuntimeError("M5_BLOCKED_TEST_ISOLATION")
    return _load_split_impl(root, manifest, split, max_shards=max_shards)


def load_test_split(
    root: str | Path, manifest: dict
) -> dict[str, np.ndarray]:
    return _load_split_impl(root, manifest, "test", max_shards=None)


def _load_split_impl(
    root: str | Path, manifest: dict, split: str, max_shards: int | None
) -> dict[str, np.ndarray]:
    rows = manifest["splits"][split]["shards"]
    if max_shards is not None:
        rows = rows[:int(max_shards)]
    arrays: dict[str, list[np.ndarray]] = {
        "state": [], "teacher_value": [], "legal_mask": [],
        "game_id": [], "max_tile_exp": [],
    }
    for row in rows:
        if row.get("status") != "complete":
            raise RuntimeError("M5_BLOCKED_DATASET_ARTIFACT_CORRUPT")
        path = Path(root) / row["path"]
        validate_completed_shard(path, row["sha256"])
        with np.load(path, allow_pickle=False) as data:
            for key in arrays:
                arrays[key].append(np.ascontiguousarray(data[key]))
    if not rows:
        raise RuntimeError("M5_BLOCKED_DATASET_ARTIFACT_CORRUPT")
    return {key: np.concatenate(parts, axis=0) for key, parts in arrays.items()}


def teacher_metrics(
    model: torch.nn.Module,
    state: np.ndarray,
    teacher_value: np.ndarray,
    legal_mask: np.ndarray,
    device: torch.device,
) -> dict:
    boards_cpu = torch.from_numpy(np.ascontiguousarray(state, dtype=np.uint8))
    values_cpu = torch.from_numpy(np.ascontiguousarray(teacher_value, dtype=np.float64))
    legal_cpu = torch.from_numpy(np.ascontiguousarray(legal_mask, dtype=np.bool_))
    target_cpu = torch.from_numpy(teacher_best_action(teacher_value, legal_mask))
    model.eval()
    total_loss = raw_correct = legal_correct = rank_correct = rank_total = total = 0
    with torch.inference_mode():
        for start in range(0, len(boards_cpu), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(boards_cpu))
            boards = boards_cpu[start:end].to(device)
            values = values_cpu[start:end].to(device)
            legal = legal_cpu[start:end].to(device)
            target = target_cpu[start:end].to(device)
            logits = model(boards).float()
            total_loss += float(F.cross_entropy(logits, target, reduction="sum").item())
            total += end - start
            raw_correct += int((logits.argmax(1) == target).sum().item())
            legal_correct += int(
                (logits.masked_fill(~legal, -torch.inf).argmax(1) == target).sum().item()
            )
            for a in range(4):
                for b in range(a + 1, 4):
                    pair = legal[:, a] & legal[:, b] & (values[:, a] != values[:, b])
                    if bool(pair.any()):
                        t = values[pair, a] > values[pair, b]
                        s = logits[pair, a] > logits[pair, b]
                        rank_correct += int((t == s).sum().item())
                        rank_total += int(pair.sum().item())
    return {
        "samples": int(total),
        "ce": float(total_loss / total),
        "raw_best_action_accuracy": float(raw_correct / total),
        "legal_best_action_accuracy": float(legal_correct / total),
        "legal_pairwise_ranking_accuracy": float(rank_correct / rank_total),
        "ranking_pairs": int(rank_total),
        "random_legal_baseline": random_legal_baseline(legal_mask),
    }


def d4_metrics(
    model: torch.nn.Module,
    state: np.ndarray,
    legal_mask: np.ndarray,
    device: torch.device,
) -> dict:
    boards_cpu = torch.from_numpy(np.ascontiguousarray(state, dtype=np.uint8))
    legal_cpu = torch.from_numpy(np.ascontiguousarray(legal_mask, dtype=np.bool_))
    per_transform: dict[str, float] = {}
    overall_same = overall_total = absolute_count = 0
    absolute_sum = 0.0
    model.eval()
    with torch.inference_mode():
        for tid in range(1, 8):
            same = count = 0
            for start in range(0, len(boards_cpu), BATCH_SIZE):
                end = min(start + BATCH_SIZE, len(boards_cpu))
                boards = boards_cpu[start:end].to(device)
                legal = legal_cpu[start:end].to(device)
                canonical = model(boards).float()
                canonical_best = canonical.masked_fill(~legal, -torch.inf).argmax(1)
                tids = torch.full((end - start,), tid, dtype=torch.long, device=device)
                transformed = transform_board_batch(boards, tids)
                restored = inverse_transform_q_values(model(transformed).float(), tids)
                restored_best = restored.masked_fill(~legal, -torch.inf).argmax(1)
                batch_same = int((restored_best == canonical_best).sum().item())
                same += batch_same
                count += end - start
                overall_same += batch_same
                overall_total += end - start
                c0 = canonical - canonical.mean(1, keepdim=True)
                c1 = restored - restored.mean(1, keepdim=True)
                absolute_sum += float(torch.abs(c1 - c0).sum().item())
                absolute_count += int(restored.numel())
            per_transform[str(tid)] = float(same / count)
    return {
        "per_transform_legal_argmax_consistency": per_transform,
        "overall_legal_argmax_consistency": float(overall_same / overall_total),
        "mean_centered_logit_mae": float(absolute_sum / absolute_count),
        "comparisons": int(overall_total),
    }

def high_tile_metrics(
    model: torch.nn.Module,
    split: dict[str, np.ndarray],
    device: torch.device,
) -> dict:
    output = {}
    for threshold in (11, 12, 13):
        mask = np.asarray(split["max_tile_exp"]) >= threshold
        samples = int(mask.sum())
        if samples == 0:
            output[f">={threshold}"] = {"samples": 0}
            continue
        metrics = teacher_metrics(
            model,
            split["state"][mask],
            split["teacher_value"][mask],
            split["legal_mask"][mask],
            device,
        )
        output[f">={threshold}"] = {
            "samples": samples,
            "random_legal_baseline": metrics["random_legal_baseline"],
            "legal_best_action_accuracy": metrics["legal_best_action_accuracy"],
            "legal_pairwise_ranking_accuracy": metrics["legal_pairwise_ranking_accuracy"],
        }
    return output

__all__ = [
    "ARCHITECTURE", "BASE_HEAD", "BATCH_SIZE", "BOOTSTRAP_SEEDS",
    "DEV_GAMES", "DEV_SEED_START", "EPOCHS", "FINAL_GAMES",
    "FINAL_SEED_START", "GAME_SHARD_SIZE", "GAMES_PER_SHARD",
    "M4_PRIMARY_SHA256", "PARAMETER_COUNT", "PROMPT_SHA256",
    "REQUIRED_SHARD_KEYS", "SCALES", "SCALE_GAMES", "SCALE_OFFSETS",
    "SCALE_ROWS", "SHARD_STATES", "STATES_PER_GAME", "SOURCE_RETRY_MAX_INDEX",
    "SOURCE_RETRY_SEED_STRIDE", "TEACHER_SHA256",
    "TEACHER_VERSION", "TEST_SEED_BASE", "TRAINING_SEEDS",
    "TRAIN_SEED_BASE", "VALIDATION_SEED_BASE", "WORK_ORDER_VERSION",
    "aggregate_seed_scores", "atomic_save_npz", "atomic_torch_save",
    "atomic_write_json", "autocast_context", "d4_metrics",
    "evaluate_game_shard", "high_tile_metrics", "load_label_split",
    "load_test_split", "make_optimizer", "median_validation",
    "paired_bootstrap", "parameter_update_checks", "plan_id",
    "plan_sha256", "policy_only_step", "random_legal_baseline",
    "sample_positions", "scale_gate_131k", "scale_shard_ranges",
    "score_summary", "select_with_524k", "set_determinism",
    "sha256_file", "shard_game_ids", "snapshot_parameters",
    "source_attempt_seed", "split_game_seed", "stateless_epoch_plan", "teacher_best_action",
    "teacher_metrics", "teacher_positive_signal",
    "validate_completed_shard", "validate_manifest_completed_shards",
]
