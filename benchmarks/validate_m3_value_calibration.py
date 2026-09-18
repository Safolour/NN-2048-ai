"""M3 §18 greedy_1ply future-score calibration probe.

This is NOT repeated depth-3 Search rollout. Continuation policy is the frozen
checkpoint's native greedy_1ply policy; depth-3 values are diagnostic only.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from game2048.m2_fast_backend import move_selected_batch
from game2048.m3_teacher_data import split_labels
from game2048.m3_tuple_teacher import (
    EXPECTED_SHA256,
    FORMAL_STATE_TUPLE_HEURISTIC,
    RAW_TUPLE_HEURISTIC,
    TupleTeacher,
)
from game2048.m3_search import SEARCH_VALUE_RAW_LEAF

STATE_COUNT = 128
ROLLOUTS_PER_STATE = 16
BASE_SEED = 202609190000
ACTIONS = np.arange(4, dtype=np.uint8)


def _rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1)
        i = j
    return ranks


def _corr(x: np.ndarray, y: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return {"pearson": None, "spearman": None}
    return {
        "pearson": float(np.corrcoef(x, y)[0, 1]),
        "spearman": float(np.corrcoef(_rankdata(x), _rankdata(y))[0, 1]),
    }


def _metrics(x: np.ndarray, y: np.ndarray, slope: float, intercept: float) -> dict:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    pred = slope * x + intercept
    err = pred - y
    sse = float(np.square(err).sum())
    centered = y - y.mean()
    sst = float(np.square(centered).sum())
    return {
        "count": int(len(y)),
        **_corr(x, y),
        "r2": None if sst == 0 else float(1.0 - sse / sst),
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt(np.square(err).mean())),
        "target_mean": float(y.mean()),
        "target_std": float(y.std()),
        "prediction_mean": float(pred.mean()),
        "prediction_std": float(pred.std()),
    }


def _select_states(corpus) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    states = np.ascontiguousarray(corpus["states"])
    game_ids = np.asarray(corpus["game_id"], dtype=np.int64)
    selected = []
    for gid in sorted(np.unique(game_ids).tolist()):
        idx = np.flatnonzero(game_ids == gid)
        if len(idx) < 8:
            raise RuntimeError(f"game_id {gid} has fewer than 8 semantic states")
        # Semantic corpus has 16 evenly-spaced states/game. Pick every other slot.
        selected.extend(idx[np.arange(0, 16, 2)].tolist())
    selected = np.asarray(selected, dtype=np.int64)
    if len(selected) != STATE_COUNT:
        raise RuntimeError(f"expected {STATE_COUNT} calibration states, got {len(selected)}")
    return states[selected], game_ids[selected], selected


def _greedy_descriptors(teacher: TupleTeacher, states: np.ndarray):
    n = len(states)
    repeated = np.repeat(np.ascontiguousarray(states), 4, axis=0)
    actions = np.tile(ACTIONS, n)
    moved = move_selected_batch(repeated, actions)
    legal = np.asarray(moved.moved, dtype=bool).reshape(n, 4)
    flat = np.full(n * 4, np.nan, dtype=np.float64)
    legal_flat = legal.reshape(-1)
    if legal_flat.any():
        after = np.ascontiguousarray(moved.afterstates[legal_flat])
        future = teacher.afterstate_values(after).astype(np.float64)
        flat[legal_flat] = moved.rewards[legal_flat].astype(np.float64) + future
    values = flat.reshape(n, 4)
    if np.any(~legal.any(axis=1)):
        raise RuntimeError("calibration corpus unexpectedly contains terminal states")
    best_action = np.nanargmax(values, axis=1).astype(np.int64)
    row = np.arange(n)
    chosen_flat = row * 4 + best_action
    chosen_after = np.ascontiguousarray(moved.afterstates[chosen_flat])
    raw_after_value = teacher.afterstate_values(chosen_after).astype(np.float64)
    immediate_reward = moved.rewards[chosen_flat].astype(np.int64)
    formal_leaf = values[row, best_action]
    return best_action, raw_after_value, immediate_reward, formal_leaf


def _spawn_in_place(afterstate: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    board = np.array(afterstate, dtype=np.uint8, copy=True)
    empty = np.flatnonzero(board == 0)
    if len(empty) == 0:
        raise RuntimeError("legal move produced no empty cell for spawn")
    pos = int(rng.integers(0, len(empty)))
    exponent = 1 if float(rng.random()) < 0.9 else 2
    board[int(empty[pos])] = np.uint8(exponent)
    return board


def _batched_rollouts(teacher: TupleTeacher, states: np.ndarray):
    total = len(states) * ROLLOUTS_PER_STATE
    boards = np.repeat(np.ascontiguousarray(states), ROLLOUTS_PER_STATE, axis=0)
    scores = np.zeros(total, dtype=np.int64)
    decisions = np.zeros(total, dtype=np.int32)
    active = np.ones(total, dtype=bool)
    seeds = np.empty(total, dtype=np.int64)
    rngs = []
    for state_i in range(len(states)):
        for rollout_i in range(ROLLOUTS_PER_STATE):
            flat = state_i * ROLLOUTS_PER_STATE + rollout_i
            seed = BASE_SEED + state_i * 100 + rollout_i
            seeds[flat] = seed
            rngs.append(np.random.default_rng(int(seed)))

    progress_path = ROOT / "artifacts" / "m3" / "m3_calibration_progress.json"
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    sweep = 0
    total_policy_decisions = 0

    while active.any():
        sweep += 1
        active_idx = np.flatnonzero(active)
        current = np.ascontiguousarray(boards[active_idx])
        n = len(current)
        repeated = np.repeat(current, 4, axis=0)
        action_ids = np.tile(ACTIONS, n)
        moved = move_selected_batch(repeated, action_ids)
        legal = np.asarray(moved.moved, dtype=bool).reshape(n, 4)
        has_legal = legal.any(axis=1)

        terminal_global = active_idx[~has_legal]
        active[terminal_global] = False
        if not has_legal.any():
            continue

        live_local = np.flatnonzero(has_legal)
        live_global = active_idx[live_local]
        legal_flat = np.asarray(moved.moved, dtype=bool)
        flat_values = np.full(n * 4, np.nan, dtype=np.float64)
        eval_after = np.ascontiguousarray(moved.afterstates[legal_flat])
        flat_values[legal_flat] = (
            moved.rewards[legal_flat].astype(np.float64)
            + teacher.afterstate_values(eval_after).astype(np.float64)
        )
        values = flat_values.reshape(n, 4)
        best = np.nanargmax(values[live_local], axis=1).astype(np.int64)
        chosen_flat = live_local * 4 + best
        chosen_after = moved.afterstates[chosen_flat]
        chosen_reward = moved.rewards[chosen_flat].astype(np.int64)

        scores[live_global] += chosen_reward
        decisions[live_global] += 1
        total_policy_decisions += len(live_global)
        for k, global_i in enumerate(live_global.tolist()):
            boards[global_i] = _spawn_in_place(chosen_after[k], rngs[global_i])

        if sweep % 100 == 0 or not active.any():
            elapsed = time.perf_counter() - started
            payload = {
                "kind": "m3_greedy_1ply_calibration_rollout",
                "sweep": sweep,
                "active_rollouts": int(active.sum()),
                "completed_rollouts": int(total - active.sum()),
                "total_rollouts": total,
                "total_policy_decisions": int(total_policy_decisions),
                "elapsed_seconds": elapsed,
                "policy_decisions_per_second": total_policy_decisions / elapsed,
            }
            progress_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(
                f"calibration sweep={sweep} active={payload['active_rollouts']}/{total} "
                f"completed={payload['completed_rollouts']} decisions={total_policy_decisions} "
                f"rate={payload['policy_decisions_per_second']:.1f}/s",
                flush=True,
            )

    elapsed = time.perf_counter() - started
    return scores, decisions, seeds, elapsed


def main():
    corpus = np.load(ROOT / "artifacts" / "m3" / "semantic_profile_states.npz")
    states, game_ids, corpus_indices = _select_states(corpus)
    split_by_game = split_labels(game_ids)
    split = np.asarray([split_by_game[int(g)] for g in game_ids])

    teacher = TupleTeacher(backend="cpp", cpp_prefetch=True)
    best_action, raw_after, immediate_reward, formal_leaf = _greedy_descriptors(teacher, states)

    search_values_path = ROOT / "artifacts" / "m3" / "search_ab_p6_cpp_values.npz"
    search_all = np.load(search_values_path)["values"]
    search_values = np.asarray(search_all[corpus_indices], dtype=np.float64)
    search_best = np.nanmax(search_values, axis=1)

    rollout_scores, rollout_decisions, seeds, rollout_wall = _batched_rollouts(teacher, states)
    returns = rollout_scores.reshape(STATE_COUNT, ROLLOUTS_PER_STATE)
    mean_return = returns.mean(axis=1)
    std_return = returns.std(axis=1, ddof=1)

    output_npz = ROOT / "artifacts" / "m3" / "m3_calibration_rollouts.npz"
    np.savez_compressed(
        output_npz,
        states=states,
        game_id=game_ids,
        corpus_index=corpus_indices,
        split=split,
        best_action=best_action,
        raw_afterstate_value=raw_after,
        immediate_reward=immediate_reward,
        formal_state_leaf=formal_leaf,
        search_best_action_value=search_best,
        rollout_future_score=returns.astype(np.int64),
        rollout_seeds=seeds.reshape(STATE_COUNT, ROLLOUTS_PER_STATE),
        rollout_decisions=rollout_decisions.reshape(STATE_COUNT, ROLLOUTS_PER_STATE),
    )

    report = {
        "result": "CALIBRATION_PROBE_COMPLETE",
        "checkpoint_sha256": EXPECTED_SHA256,
        "continuation_policy": "greedy_1ply",
        "continuation_definition": "argmax immediate_reward + V_tuple(afterstate), deterministic lowest-index tie-break",
        "spawn_semantics": "uniform empty cell; 0.9 tile-2 / 0.1 tile-4",
        "states": STATE_COUNT,
        "source_games": int(len(np.unique(game_ids))),
        "rollouts_per_state": ROLLOUTS_PER_STATE,
        "total_rollouts": int(STATE_COUNT * ROLLOUTS_PER_STATE),
        "seed_base": BASE_SEED,
        "selection": "8 of 16 evenly-spaced semantic states per each of 16 game_ids (slots 0,2,...,14)",
        "fit_split": "game-level 80/10/10 via seed 20260919",
        "split_counts": {name: int(np.count_nonzero(split == name)) for name in ("train", "validation", "test")},
        "rollout_wall_seconds": rollout_wall,
        "rollout_policy_decisions": int(rollout_decisions.sum()),
        "rollout_policy_decisions_per_second": float(rollout_decisions.sum() / rollout_wall),
        "future_score_summary": {
            "mean_of_state_means": float(mean_return.mean()),
            "std_of_state_means": float(mean_return.std()),
            "min_state_mean": float(mean_return.min()),
            "max_state_mean": float(mean_return.max()),
            "mean_within_state_std": float(std_return.mean()),
        },
        "correlations_all_128": {
            "raw_selected_afterstate_v_vs_return": _corr(raw_after, mean_return),
            "formal_state_leaf_vs_return": _corr(formal_leaf, mean_return),
            "depth3_search_best_vs_return_diagnostic_only": _corr(search_best, mean_return),
        },
        "depth3_search_semantics": SEARCH_VALUE_RAW_LEAF,
        "raw_tuple_semantics_before_review": RAW_TUPLE_HEURISTIC,
        "formal_leaf_semantics_before_review": FORMAL_STATE_TUPLE_HEURISTIC,
    }

    train = split == "train"
    for name, x in (
        ("raw_selected_afterstate_v", raw_after),
        ("formal_state_leaf", formal_leaf),
    ):
        slope, intercept = np.polyfit(x[train], mean_return[train], 1)
        mapping = {
            "slope": float(slope),
            "intercept": float(intercept),
            "metrics": {},
            "identity_diagnostic": {
                "mae": float(np.abs(x - mean_return).mean()),
                "rmse": float(np.sqrt(np.square(x - mean_return).mean())),
            },
        }
        for split_name in ("train", "validation", "test"):
            mask = split == split_name
            mapping["metrics"][split_name] = _metrics(
                x[mask], mean_return[mask], float(slope), float(intercept)
            )
        report.setdefault("affine_mapping", {})[name] = mapping

    (ROOT / "m3_teacher_calibration.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    progress = ROOT / "artifacts" / "m3" / "m3_calibration_progress.json"
    if progress.exists():
        progress.unlink()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
