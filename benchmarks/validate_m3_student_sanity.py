"""M3 policy-only Student learning sanity on the canonical Teacher dataset.

Search values remain SEARCH_VALUE_RAW_LEAF, so this script deliberately uses no
absolute Q/V/A regression. The Q head is treated as four policy logits for
teacher-best-action cross entropy; V/afterstate heads receive no fabricated loss.
"""
from __future__ import annotations

import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from game2048.m2_models import ResidualMLP2048
from game2048.m2_symmetry import (
    inverse_transform_q_values,
    transform_action_batch,
    transform_board_batch,
)

DATASET = ROOT / "artifacts" / "m3" / "m3_teacher_validation_8192.npz"
OUT = ROOT / "m3_student_sanity.json"
SEEDS = (20260919, 20260920)
EPOCHS = 30
BATCH_SIZE = 1024
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def _pairwise_ranking_accuracy(
    logits: torch.Tensor,
    teacher_values: torch.Tensor,
    legal: torch.Tensor,
) -> tuple[int, int]:
    correct = 0
    total = 0
    for a in range(4):
        for b in range(a + 1, 4):
            mask = legal[:, a] & legal[:, b] & (teacher_values[:, a] != teacher_values[:, b])
            if not bool(mask.any()):
                continue
            teacher_sign = teacher_values[mask, a] > teacher_values[mask, b]
            student_sign = logits[mask, a] > logits[mask, b]
            correct += int((teacher_sign == student_sign).sum().item())
            total += int(mask.sum().item())
    return correct, total


@torch.no_grad()
def _evaluate(model, boards, target, legal, teacher_values, device):
    model.eval()
    total_loss = 0.0
    total = 0
    raw_correct = 0
    legal_correct = 0
    rank_correct = 0
    rank_total = 0
    for start in range(0, len(boards), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(boards))
        b = boards[start:end].to(device)
        y = target[start:end].to(device)
        m = legal[start:end].to(device)
        tv = teacher_values[start:end].to(device)
        logits = model(b)
        loss = F.cross_entropy(logits, y, reduction="sum")
        total_loss += float(loss.item())
        total += len(b)
        raw_correct += int((logits.argmax(dim=1) == y).sum().item())
        masked = logits.masked_fill(~m, -torch.inf)
        legal_correct += int((masked.argmax(dim=1) == y).sum().item())
        c, n = _pairwise_ranking_accuracy(logits, tv, m)
        rank_correct += c
        rank_total += n
    return {
        "loss": total_loss / total,
        "raw_best_action_accuracy": raw_correct / total,
        "legal_best_action_accuracy": legal_correct / total,
        "legal_pairwise_ranking_accuracy": rank_correct / rank_total if rank_total else None,
        "samples": total,
        "ranking_pairs": rank_total,
    }


@torch.no_grad()
def _d4_consistency(model, boards, legal, device):
    model.eval()
    total = 0
    argmax_same = 0
    centered_abs_sum = 0.0
    centered_count = 0
    for start in range(0, len(boards), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(boards))
        b = boards[start:end].to(device)
        m = legal[start:end].to(device)
        canonical = model(b)
        canonical_centered = canonical - canonical.mean(dim=1, keepdim=True)
        canonical_best = canonical.masked_fill(~m, -torch.inf).argmax(dim=1)
        for tid in range(1, 8):
            tids = torch.full((len(b),), tid, dtype=torch.long, device=device)
            transformed = transform_board_batch(b, tids)
            q_t = model(transformed)
            restored = inverse_transform_q_values(q_t, tids)
            restored_centered = restored - restored.mean(dim=1, keepdim=True)
            best = restored.masked_fill(~m, -torch.inf).argmax(dim=1)
            argmax_same += int((best == canonical_best).sum().item())
            total += len(b)
            centered_abs_sum += float(torch.abs(restored_centered - canonical_centered).sum().item())
            centered_count += int(restored.numel())
    return {
        "legal_argmax_consistency": argmax_same / total,
        "mean_centered_logit_mae": centered_abs_sum / centered_count,
        "comparisons": total,
    }


def _random_legal_baseline(legal: torch.Tensor) -> float:
    counts = legal.sum(dim=1).to(torch.float64)
    return float((1.0 / counts).mean().item())


def _run(seed, train_boards, train_target, train_legal, train_values,
         val_boards, val_target, val_legal, val_values, device):
    _set_seed(seed)
    model = ResidualMLP2048().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    initial_train = _evaluate(
        model, train_boards, train_target, train_legal, train_values, device
    )
    initial_val = _evaluate(
        model, val_boards, val_target, val_legal, val_values, device
    )

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed + 991)
    history = []
    started = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = torch.randperm(len(train_boards), generator=gen)
        running_loss = 0.0
        seen = 0
        for start in range(0, len(order), BATCH_SIZE):
            ids = order[start:start + BATCH_SIZE]
            b = train_boards[ids].to(device)
            y = train_target[ids].to(device)
            tids = torch.randint(0, 8, (len(ids),), generator=gen, dtype=torch.long)
            tids = tids.to(device)
            b = transform_board_batch(b, tids)
            y = transform_action_batch(y, tids)

            optimizer.zero_grad(set_to_none=True)
            logits = model(b)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * len(ids)
            seen += len(ids)

        val = _evaluate(model, val_boards, val_target, val_legal, val_values, device)
        row = {
            "epoch": epoch,
            "augmented_train_ce": running_loss / seen,
            "validation_loss": val["loss"],
            "validation_legal_best_action_accuracy": val["legal_best_action_accuracy"],
            "validation_legal_pairwise_ranking_accuracy": val["legal_pairwise_ranking_accuracy"],
        }
        history.append(row)
        if epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS:
            print(
                f"student seed={seed} epoch={epoch}/{EPOCHS} "
                f"train_ce={row['augmented_train_ce']:.4f} "
                f"val_ce={row['validation_loss']:.4f} "
                f"val_acc={row['validation_legal_best_action_accuracy']:.3f}",
                flush=True,
            )

    final_train = _evaluate(
        model, train_boards, train_target, train_legal, train_values, device
    )
    final_val = _evaluate(
        model, val_boards, val_target, val_legal, val_values, device
    )
    d4 = _d4_consistency(model, val_boards, val_legal, device)
    return model, {
        "seed": seed,
        "initial_train": initial_train,
        "initial_validation": initial_val,
        "final_train": final_train,
        "final_validation": final_val,
        "d4_consistency_validation": d4,
        "history": history,
        "wall_seconds": time.perf_counter() - started,
    }


def main():
    data = np.load(DATASET)
    boards_np = np.ascontiguousarray(data["state"])
    values_np = np.asarray(data["teacher_value"], dtype=np.float32)
    legal_np = np.asarray(data["legal_mask"], dtype=np.bool_)
    split_np = np.asarray(data["split"])
    game_ids = np.asarray(data["game_id"], dtype=np.int64)

    target_np = np.nanargmax(values_np, axis=1).astype(np.int64)
    boards = torch.from_numpy(boards_np)
    values = torch.from_numpy(values_np)
    legal = torch.from_numpy(legal_np)
    target = torch.from_numpy(target_np)

    masks = {name: split_np == name for name in ("train", "validation", "test")}
    split_games = {name: set(game_ids[masks[name]].tolist()) for name in masks}
    if (
        split_games["train"] & split_games["validation"]
        or split_games["train"] & split_games["test"]
        or split_games["validation"] & split_games["test"]
    ):
        raise RuntimeError("game_id leakage across canonical splits")

    tensors = {}
    for name, mask in masks.items():
        ids = np.flatnonzero(mask)
        tensors[name] = (
            boards[ids],
            target[ids],
            legal[ids],
            values[ids],
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"student sanity device={device}", flush=True)
    runs = []
    models = []
    for seed in SEEDS:
        model, result = _run(
            seed,
            *tensors["train"],
            *tensors["validation"],
            device,
        )
        models.append(model)
        runs.append(result)

    # Test split is touched only after all fixed training runs are complete.
    test_results = []
    for model, seed in zip(models, SEEDS):
        test_metric = _evaluate(model, *tensors["test"], device)
        d4_test = _d4_consistency(model, tensors["test"][0], tensors["test"][2], device)
        test_results.append({
            "seed": seed,
            "metrics": test_metric,
            "d4_consistency": d4_test,
        })

    train_random = _random_legal_baseline(tensors["train"][2])
    val_random = _random_legal_baseline(tensors["validation"][2])
    test_random = _random_legal_baseline(tensors["test"][2])

    per_run_direction = []
    for run in runs:
        train_improved = (
            run["final_train"]["loss"] < 0.8 * run["initial_train"]["loss"]
            and run["final_train"]["legal_best_action_accuracy"]
            > run["initial_train"]["legal_best_action_accuracy"] + 0.10
        )
        val_not_diverged = (
            np.isfinite(run["final_validation"]["loss"])
            and run["final_validation"]["loss"]
            <= 1.25 * run["initial_validation"]["loss"]
        )
        val_above_random = (
            run["final_validation"]["legal_best_action_accuracy"]
            >= val_random + 0.10
        )
        per_run_direction.append({
            "seed": run["seed"],
            "train_metric_clearly_improved": bool(train_improved),
            "validation_not_diverged": bool(val_not_diverged),
            "validation_accuracy_above_random_baseline": bool(val_above_random),
            "pass": bool(train_improved and val_not_diverged and val_above_random),
        })

    report = {
        "result": (
            "M3_STUDENT_SANITY_PASS"
            if all(x["pass"] for x in per_run_direction)
            else "M3_STUDENT_SANITY_FAIL"
        ),
        "supervision": {
            "kind": "policy_only",
            "loss": "teacher_best_action_cross_entropy",
            "absolute_q_regression": False,
            "absolute_v_regression": False,
            "absolute_afterstate_regression": False,
            "reason": "depth-3 Teacher values remain SEARCH_VALUE_RAW_LEAF; no fabricated absolute targets",
        },
        "model": "ResidualMLP2048",
        "architecture_modified": False,
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "protocol": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "optimizer": "AdamW",
            "train_d4_augmentation": "random transform 0..7 after game-level split",
            "test_used_for_tuning": False,
            "seeds": list(SEEDS),
        },
        "dataset": {
            "states": int(len(boards)),
            "games": int(len(np.unique(game_ids))),
            "split_states": {name: int(masks[name].sum()) for name in masks},
            "split_games": {name: len(split_games[name]) for name in masks},
            "game_overlap": 0,
            "canonical_artifact_unaugmented": True,
        },
        "random_legal_action_baseline": {
            "train": train_random,
            "validation": val_random,
            "test": test_random,
        },
        "runs": runs,
        "test_results_after_fixed_training": test_results,
        "direction_checks": per_run_direction,
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "result": report["result"],
        "random_baseline": report["random_legal_action_baseline"],
        "direction_checks": per_run_direction,
        "final_validation": [
            {
                "seed": r["seed"],
                **r["final_validation"],
                "d4": r["d4_consistency_validation"],
            }
            for r in runs
        ],
        "test": test_results,
    }, indent=2))


if __name__ == "__main__":
    main()
