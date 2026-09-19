"""M4 deterministic helpers for the Transformer2048 vs ResidualMLP2048 comparison."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Sequence
import json
import os

import numpy as np
import torch

from .m2_fast_backend import legal_mask_batch, move_selected_batch
from .reference_env import move_without_spawn, spawn_random

WORK_ORDER_VERSION = "M4_WO_CLOSURE_V2"
DATASET_SHA256 = "F0E5D806A3E17A27284998AAD52A683245A3C74EBDD15B23EFC13448B9D96582"
CHECKPOINT_SHA256 = "7192719323a073ba2b6b19b62cb7d46ef4aa90ecc8c4ae6baf27ad0c51566a84"
PROMPT_SHA256 = "3EB37DFD85999F1D2344A7922F74A4E566582968AAD11ECF402BFE10B602605B"
BASE_HEAD = "cdc250477cce21fbe388871f50c013990a749521"
TRANSFORMER = "Transformer2048"
RESIDUAL_MLP = "ResidualMLP2048"
ARCHITECTURES = (TRANSFORMER, RESIDUAL_MLP)
TRAINING_SEEDS = (20260919, 20260920, 20260921)
PRIMARY_RUN_ORDER = (
    (TRANSFORMER, 20260919),
    (RESIDUAL_MLP, 20260919),
    (RESIDUAL_MLP, 20260920),
    (TRANSFORMER, 20260920),
    (TRANSFORMER, 20260921),
    (RESIDUAL_MLP, 20260921),
)
EVALUATION_SEED_START = 20261001
EVALUATION_GAMES = 2000
GAME_SHARD_SIZE = 100
TIE_TOLERANCE = 1e-7
EPOCHS = 30
BATCH_SIZE = 1024
TRAIN_STATES = 6528
SECONDARY_PLAN_ID = "FBDCB22E4A9A16359073740890F3FBC1F5C91031656F385487AEB3F507B48642"


@dataclass(frozen=True)
class TrainingPlan:
    sample_indices: np.ndarray
    transform_ids: np.ndarray
    sha256: str


def sha256_file(path: str | Path) -> str:
    digest = sha256()
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
        np.savez(handle, **arrays)
    os.replace(temporary, target)


def plan_sha256(sample_indices: np.ndarray, transform_ids: np.ndarray) -> str:
    sample = np.ascontiguousarray(sample_indices)
    transforms = np.ascontiguousarray(transform_ids)
    sample_sha = sha256(sample.tobytes()).hexdigest()
    transform_sha = sha256(transforms.tobytes()).hexdigest()
    combined = f"{sample_sha}:{transform_sha}".encode("ascii")
    return sha256(combined).hexdigest().upper()


def build_training_plan(
    train_indices: Sequence[int] | np.ndarray,
    seed: int,
    *,
    epochs: int = EPOCHS,
) -> TrainingPlan:
    indices = np.asarray(train_indices, dtype=np.int64)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError("train_indices must be a non-empty 1-D array")
    indices = np.sort(indices)
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    orders = np.empty((epochs, indices.size), dtype=np.int64)
    transforms = np.empty((epochs, indices.size), dtype=np.uint8)
    for epoch in range(epochs):
        orders[epoch] = rng.permutation(indices)
        transforms[epoch] = rng.integers(
            0, 8, size=indices.size, dtype=np.uint8
        )
    return TrainingPlan(orders, transforms, plan_sha256(orders, transforms))


def save_training_plan(path: str | Path, plan: TrainingPlan) -> None:
    atomic_save_npz(
        path,
        sample_indices=np.ascontiguousarray(plan.sample_indices, dtype=np.int64),
        transform_ids=np.ascontiguousarray(plan.transform_ids, dtype=np.uint8),
    )


def load_training_plan(path: str | Path) -> TrainingPlan:
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {"sample_indices", "transform_ids"}:
            raise RuntimeError(
                "M4 plan keys must be exactly sample_indices and transform_ids"
            )
        sample = np.ascontiguousarray(data["sample_indices"], dtype=np.int64)
        transforms = np.ascontiguousarray(data["transform_ids"], dtype=np.uint8)
    if sample.shape != transforms.shape:
        raise RuntimeError("M4 plan arrays have different shapes")
    return TrainingPlan(sample, transforms, plan_sha256(sample, transforms))


def build_secondary_epoch_plan(
    train_indices: Sequence[int] | np.ndarray,
    epoch: int,
) -> TrainingPlan:
    indices = np.sort(np.asarray(train_indices, dtype=np.int64))
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError("train_indices must be a non-empty 1-D array")
    rng = np.random.Generator(np.random.PCG64(20262000 + int(epoch)))
    order = rng.permutation(indices)[None, :]
    transforms = rng.integers(
        0, 8, size=indices.size, dtype=np.uint8
    )[None, :]
    return TrainingPlan(order, transforms, plan_sha256(order, transforms))


def teacher_best_action(teacher_value: np.ndarray) -> np.ndarray:
    values = np.asarray(teacher_value)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("teacher_value must have shape (N,4)")
    return np.nanargmax(values, axis=1).astype(np.int64)


def training_indices(split: np.ndarray) -> np.ndarray:
    labels = np.asarray(split)
    return np.flatnonzero(labels == "train").astype(np.int64)


def validation_indices(split: np.ndarray) -> np.ndarray:
    labels = np.asarray(split)
    return np.flatnonzero(labels == "validation").astype(np.int64)


def test_indices(split: np.ndarray) -> np.ndarray:
    labels = np.asarray(split)
    return np.flatnonzero(labels == "test").astype(np.int64)


def make_game_rngs(
    game_seed: int,
) -> tuple[np.random.Generator, np.random.Generator]:
    seed = int(game_seed)
    spawn_rng = np.random.Generator(np.random.PCG64(seed))
    tie_rng = np.random.Generator(np.random.PCG64(seed ^ 0x9E3779B9))
    return spawn_rng, tie_rng


def initialize_game(
    game_seed: int,
) -> tuple[np.ndarray, np.random.Generator, np.random.Generator]:
    spawn_rng, tie_rng = make_game_rngs(game_seed)
    board = np.zeros(16, dtype=np.uint8)
    board = spawn_random(board, spawn_rng).state
    board = spawn_random(board, spawn_rng).state
    return board, spawn_rng, tie_rng


def select_game_actions(
    logits: np.ndarray,
    legal: np.ndarray,
    tie_rngs: Sequence[np.random.Generator],
    *,
    tolerance: float = TIE_TOLERANCE,
) -> np.ndarray:
    q = np.asarray(logits)
    mask = np.asarray(legal, dtype=np.bool_)
    if q.ndim != 2 or q.shape[1] != 4 or mask.shape != q.shape:
        raise ValueError("logits and legal must both have shape (N,4)")
    if len(tie_rngs) != q.shape[0]:
        raise ValueError("one tie RNG is required for every board")
    if bool((~mask.any(axis=1)).any()):
        raise ValueError("terminal boards cannot be sent to action selection")
    actions = np.empty(q.shape[0], dtype=np.uint8)
    for row in range(q.shape[0]):
        legal_ids = np.flatnonzero(mask[row])
        legal_values = q[row, legal_ids]
        if not bool(np.isfinite(legal_values).all()):
            raise FloatingPointError("M4_BLOCKED_EVAL_NONFINITE")
        qmax = float(np.max(legal_values))
        candidates = np.flatnonzero(
            mask[row] & ((qmax - q[row]) <= float(tolerance))
        )
        if candidates.size == 1:
            actions[row] = np.uint8(candidates[0])
        else:
            candidate_index = int(tie_rngs[row].integers(0, candidates.size))
            actions[row] = np.uint8(candidates[candidate_index])
    return actions


def evaluate_game_shard(
    model: torch.nn.Module,
    game_seeds: Sequence[int],
    device: torch.device,
    *,
    verify_reference: bool = False,
) -> dict[str, np.ndarray]:
    seeds = np.asarray(game_seeds, dtype=np.int64)
    if seeds.ndim != 1 or seeds.size == 0:
        raise ValueError("game_seeds must be a non-empty 1-D sequence")
    if not np.array_equal(seeds, np.sort(seeds)):
        raise ValueError("game_seeds must be sorted ascending")
    if np.unique(seeds).size != seeds.size:
        raise ValueError("game_seeds must be unique")

    count = int(seeds.size)
    boards = np.zeros((count, 16), dtype=np.uint8)
    spawn_rngs: list[np.random.Generator] = []
    tie_rngs: list[np.random.Generator] = []
    for index, game_seed in enumerate(seeds):
        board, spawn_rng, tie_rng = initialize_game(int(game_seed))
        boards[index] = board
        spawn_rngs.append(spawn_rng)
        tie_rngs.append(tie_rng)

    scores = np.zeros(count, dtype=np.int64)
    moves = np.zeros(count, dtype=np.int64)
    active = np.ones(count, dtype=np.bool_)
    model.eval()

    with torch.inference_mode():
        while bool(active.any()):
            active_ids = np.flatnonzero(active)
            current = np.ascontiguousarray(boards[active_ids])
            legal = legal_mask_batch(current)
            terminal = ~legal.any(axis=1)
            if bool(terminal.any()):
                active[active_ids[terminal]] = False
                keep = ~terminal
                active_ids = active_ids[keep]
                current = np.ascontiguousarray(current[keep])
                legal = np.ascontiguousarray(legal[keep])
                if active_ids.size == 0:
                    continue

            boards_gpu = torch.from_numpy(current).to(
                device=device, non_blocking=False
            )
            logits = model(boards_gpu).float().cpu().numpy()
            row_rngs = [tie_rngs[int(index)] for index in active_ids]
            actions = select_game_actions(logits, legal, row_rngs)

            selected_legal = legal[
                np.arange(actions.size, dtype=np.int64), actions.astype(np.int64)
            ]
            if not bool(selected_legal.all()):
                raise RuntimeError("M4_BLOCKED_GAME_EVALUATOR_CORRECTNESS")

            fast = move_selected_batch(current, actions)
            if not bool(fast.moved.all()):
                raise RuntimeError("M4_BLOCKED_GAME_EVALUATOR_CORRECTNESS")

            if verify_reference:
                for row, action in enumerate(actions):
                    reference = move_without_spawn(current[row], int(action))
                    if not (
                        np.array_equal(
                            fast.afterstates[row], reference.afterstate
                        )
                        and int(fast.rewards[row]) == int(reference.reward)
                        and bool(fast.moved[row]) == bool(reference.moved)
                    ):
                        raise RuntimeError(
                            "M4_BLOCKED_GAME_EVALUATOR_CORRECTNESS"
                        )

            scores[active_ids] += fast.rewards
            moves[active_ids] += 1

            for row, game_index in enumerate(active_ids):
                boards[int(game_index)] = spawn_random(
                    fast.afterstates[row], spawn_rngs[int(game_index)]
                ).state

    return {
        "game_seed": seeds.copy(),
        "final_score": scores,
        "max_tile_exp": boards.max(axis=1).astype(np.uint8),
        "moves": moves,
    }


def load_game_results(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        required = {"game_seed", "final_score", "max_tile_exp", "moves"}
        if set(data.files) != required:
            raise RuntimeError("M4_BLOCKED_RESUME_ARTIFACT_CORRUPT")
        result = {
            "game_seed": np.ascontiguousarray(data["game_seed"], dtype=np.int64),
            "final_score": np.ascontiguousarray(
                data["final_score"], dtype=np.int64
            ),
            "max_tile_exp": np.ascontiguousarray(
                data["max_tile_exp"], dtype=np.uint8
            ),
            "moves": np.ascontiguousarray(data["moves"], dtype=np.int64),
        }
    lengths = {value.shape for value in result.values()}
    if len(lengths) != 1 or next(iter(lengths))[0] != result["game_seed"].size:
        raise RuntimeError("M4_BLOCKED_RESUME_ARTIFACT_CORRUPT")
    if any(value.ndim != 1 for value in result.values()):
        raise RuntimeError("M4_BLOCKED_RESUME_ARTIFACT_CORRUPT")
    if not np.array_equal(result["game_seed"], np.sort(result["game_seed"])):
        raise RuntimeError("M4_BLOCKED_RESUME_ARTIFACT_CORRUPT")
    if np.unique(result["game_seed"]).size != result["game_seed"].size:
        raise RuntimeError("M4_BLOCKED_RESUME_ARTIFACT_CORRUPT")
    return result


def validate_primary_game_progress(
    results: dict[str, np.ndarray] | None,
) -> int:
    if results is None:
        return 0
    completed = int(results["game_seed"].size)
    if completed % GAME_SHARD_SIZE != 0 or completed > EVALUATION_GAMES:
        raise RuntimeError("M4_BLOCKED_RESUME_ARTIFACT_CORRUPT")
    expected = np.arange(
        EVALUATION_SEED_START,
        EVALUATION_SEED_START + completed,
        dtype=np.int64,
    )
    if not np.array_equal(results["game_seed"], expected):
        raise RuntimeError("M4_BLOCKED_RESUME_ARTIFACT_CORRUPT")
    return completed // GAME_SHARD_SIZE


def append_game_shard(
    existing: dict[str, np.ndarray] | None,
    shard: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    if existing is None:
        merged = {key: np.asarray(value) for key, value in shard.items()}
    else:
        merged = {
            key: np.concatenate(
                [np.asarray(existing[key]), np.asarray(shard[key])]
            )
            for key in ("game_seed", "final_score", "max_tile_exp", "moves")
        }
    order = np.argsort(merged["game_seed"])
    merged = {
        key: np.ascontiguousarray(value[order]) for key, value in merged.items()
    }
    if np.unique(merged["game_seed"]).size != merged["game_seed"].size:
        raise RuntimeError("duplicate game seed while merging shard")
    return merged


def save_game_results(
    path: str | Path,
    results: dict[str, np.ndarray],
) -> None:
    atomic_save_npz(
        path,
        game_seed=np.asarray(results["game_seed"], dtype=np.int64),
        final_score=np.asarray(results["final_score"], dtype=np.int64),
        max_tile_exp=np.asarray(results["max_tile_exp"], dtype=np.uint8),
        moves=np.asarray(results["moves"], dtype=np.int64),
    )


def score_summary(
    final_score: np.ndarray,
    max_tile_exp: np.ndarray,
    moves: np.ndarray,
) -> dict:
    scores = np.asarray(final_score, dtype=np.float64)
    exponents = np.asarray(max_tile_exp, dtype=np.int64)
    move_count = np.asarray(moves, dtype=np.float64)
    if scores.ndim != 1 or exponents.ndim != 1 or move_count.ndim != 1:
        raise ValueError("game result vectors must be 1-D")
    if not (scores.size == exponents.size == move_count.size) or scores.size == 0:
        raise ValueError("game result vectors must have identical non-zero length")
    p10, p90 = np.quantile(scores, [0.1, 0.9])
    unique, counts = np.unique(exponents, return_counts=True)
    return {
        "games": int(scores.size),
        "mean_score": float(scores.mean()),
        "median_score": float(np.median(scores)),
        "p10_score": float(p10),
        "p90_score": float(p90),
        "mean_moves": float(move_count.mean()),
        "reach": {
            "2048": float(np.mean(exponents >= 11)),
            "4096": float(np.mean(exponents >= 12)),
            "8192": float(np.mean(exponents >= 13)),
            "16384": float(np.mean(exponents >= 14)),
            "32768": float(np.mean(exponents >= 15)),
            "65536": float(np.mean(exponents >= 16)),
        },
        "max_tile_distribution": {
            str(int(exp)): int(count)
            for exp, count in zip(unique.tolist(), counts.tolist())
        },
    }


def paired_bootstrap(
    transformer_scores: np.ndarray,
    mlp_scores: np.ndarray,
    *,
    seed: int,
    resamples: int = 10_000,
) -> dict:
    transformer = np.asarray(transformer_scores, dtype=np.float64)
    mlp = np.asarray(mlp_scores, dtype=np.float64)
    if transformer.ndim != 1 or transformer.shape != mlp.shape:
        raise ValueError("paired score vectors must be equal 1-D arrays")
    if transformer.size == 0:
        raise ValueError("paired score vectors cannot be empty")
    delta = transformer - mlp
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    means = np.empty(int(resamples), dtype=np.float64)
    done = 0
    while done < resamples:
        take = min(128, resamples - done)
        indices = rng.integers(
            0, delta.size, size=(take, delta.size), dtype=np.int64
        )
        means[done : done + take] = delta[indices].mean(axis=1)
        done += take
    lower, upper = np.quantile(means, [0.025, 0.975])
    return {
        "mean_delta": float(delta.mean()),
        "ci95": [float(lower), float(upper)],
        "pairs": int(delta.size),
        "bootstrap_resamples": int(resamples),
        "bootstrap_seed": int(seed),
    }


def strength_conclusion(
    per_seed_rows: Sequence[dict],
    aggregate: dict,
) -> str:
    means = [float(row["mean_delta"]) for row in per_seed_rows]
    lower, upper = [float(value) for value in aggregate["ci95"]]
    if lower > 0.0 and sum(value > 0.0 for value in means) >= 2:
        return "TRANSFORMER_STRENGTH_SIGNIFICANT"
    if upper < 0.0 and sum(value < 0.0 for value in means) >= 2:
        return "MLP_STRENGTH_SIGNIFICANT"
    return "ARCHITECTURE_STRENGTH_NOT_STATISTICALLY_RESOLVED"


def select_architecture(
    strength: str,
    transformer_8192: float,
    mlp_8192: float,
) -> dict:
    if strength == "TRANSFORMER_STRENGTH_SIGNIFICANT":
        return {
            "selected_architecture": "M4_SELECT_TRANSFORMER",
            "selection_rule": "SCORE_SIGNIFICANCE",
            "speed_ratio": float(
                max(transformer_8192, mlp_8192)
                / min(transformer_8192, mlp_8192)
            ),
        }
    if strength == "MLP_STRENGTH_SIGNIFICANT":
        return {
            "selected_architecture": "M4_SELECT_RESIDUAL_MLP",
            "selection_rule": "SCORE_SIGNIFICANCE",
            "speed_ratio": float(
                max(transformer_8192, mlp_8192)
                / min(transformer_8192, mlp_8192)
            ),
        }

    values = np.asarray([transformer_8192, mlp_8192], dtype=np.float64)
    if not bool(np.isfinite(values).all()) or bool((values <= 0).any()):
        raise ValueError("8192 closed-loop throughputs must be finite and positive")
    speed_ratio = float(values.max() / values.min())
    if speed_ratio >= 1.10:
        selected = (
            "M4_SELECT_TRANSFORMER"
            if transformer_8192 > mlp_8192
            else "M4_SELECT_RESIDUAL_MLP"
        )
        rule = "8192_SPEED"
    else:
        selected = "M4_SELECT_TRANSFORMER"
        rule = "PARAMETER_COUNT"
    return {
        "selected_architecture": selected,
        "selection_rule": rule,
        "speed_ratio": speed_ratio,
    }


__all__ = [
    "ARCHITECTURES",
    "BASE_HEAD",
    "BATCH_SIZE",
    "CHECKPOINT_SHA256",
    "DATASET_SHA256",
    "EPOCHS",
    "EVALUATION_GAMES",
    "EVALUATION_SEED_START",
    "GAME_SHARD_SIZE",
    "PRIMARY_RUN_ORDER",
    "PROMPT_SHA256",
    "RESIDUAL_MLP",
    "SECONDARY_PLAN_ID",
    "TIE_TOLERANCE",
    "TRAINING_SEEDS",
    "TRAIN_STATES",
    "TRANSFORMER",
    "TrainingPlan",
    "WORK_ORDER_VERSION",
    "append_game_shard",
    "atomic_save_npz",
    "atomic_write_json",
    "build_secondary_epoch_plan",
    "build_training_plan",
    "evaluate_game_shard",
    "initialize_game",
    "load_game_results",
    "load_training_plan",
    "make_game_rngs",
    "paired_bootstrap",
    "plan_sha256",
    "save_game_results",
    "save_training_plan",
    "score_summary",
    "select_architecture",
    "select_game_actions",
    "sha256_file",
    "strength_conclusion",
    "teacher_best_action",
    "test_indices",
    "training_indices",
    "validate_primary_game_progress",
    "validation_indices",
]
