"""Generate the canonical M3 8192-state / 64-complete-game Teacher dataset.

Source trajectories use the frozen tuple checkpoint's native greedy_1ply policy.
Selected formal states are labelled by exact decision_depth=3 Expectimax.
Canonical saved data is unaugmented; D4 augmentation belongs only to training.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from game2048.m2_fast_backend import move_selected_batch
from game2048.m3_search import LEAF_EVALUATOR, SEARCH_VALUE_RAW_LEAF, ExpectimaxTeacher
from game2048.m3_teacher_data import split_labels
from game2048.m3_tuple_teacher import EXPECTED_SHA256, TupleTeacher
from game2048.reference_env import spawn_random

GAME_COUNT = 64
STATES_PER_GAME = 128
TARGET_STATES = GAME_COUNT * STATES_PER_GAME
GAME_SEED_BASE = 20260919
WORKERS = 8
CHUNK_SIZE = 256
SUBTASK_SIZE = 32
ART = ROOT / "artifacts" / "m3"
SOURCE_NPZ = ART / "m3_teacher_source_states_8192.npz"
SOURCE_GAMES = ART / "m3_teacher_source_games_64.json"
CHUNK_DIR = ART / "teacher_label_chunks"
PROGRESS = ART / "m3_teacher_generation_progress.json"
FINAL_NPZ = ART / "m3_teacher_validation_8192.npz"
FINAL_MANIFEST = ROOT / "m3_teacher_dataset.json"
ACTIONS = np.arange(4, dtype=np.uint8)

_WORKER_TEACHER = None
_WORKER_SOURCE = None


def _init_board(seed: int):
    rng = np.random.default_rng(seed)
    board = np.zeros(16, dtype=np.uint8)
    board = spawn_random(board, rng).state
    board = spawn_random(board, rng).state
    return board, rng


def _batch_greedy_values(teacher: TupleTeacher, boards: np.ndarray):
    n = len(boards)
    repeated = np.repeat(np.ascontiguousarray(boards), 4, axis=0)
    actions = np.tile(ACTIONS, n)
    moved = move_selected_batch(repeated, actions)
    legal = np.asarray(moved.moved, dtype=bool).reshape(n, 4)
    flat = np.full(n * 4, np.nan, dtype=np.float64)
    legal_flat = legal.reshape(-1)
    if legal_flat.any():
        after = np.ascontiguousarray(moved.afterstates[legal_flat])
        flat[legal_flat] = (
            moved.rewards[legal_flat].astype(np.float64)
            + teacher.afterstate_values(after).astype(np.float64)
        )
    return moved, legal, flat.reshape(n, 4)


def _generate_complete_source_games():
    if SOURCE_NPZ.exists() and SOURCE_GAMES.exists():
        source = np.load(SOURCE_NPZ)
        if len(source["states"]) == TARGET_STATES and len(np.unique(source["game_id"])) == GAME_COUNT:
            print("source games already materialized; reusing", flush=True)
            return

    teacher = TupleTeacher(backend="cpp", cpp_prefetch=True)
    boards = np.zeros((GAME_COUNT, 16), dtype=np.uint8)
    rngs = []
    for gid in range(GAME_COUNT):
        boards[gid], rng = _init_board(GAME_SEED_BASE + gid)
        rngs.append(rng)

    active = np.ones(GAME_COUNT, dtype=bool)
    scores = np.zeros(GAME_COUNT, dtype=np.int64)
    steps = np.zeros(GAME_COUNT, dtype=np.int32)
    trajectory_states = [[] for _ in range(GAME_COUNT)]
    trajectory_scores = [[] for _ in range(GAME_COUNT)]
    started = time.perf_counter()
    decisions = 0

    while active.any():
        idx = np.flatnonzero(active)
        current = np.ascontiguousarray(boards[idx])
        moved, legal, values = _batch_greedy_values(teacher, current)
        has_legal = legal.any(axis=1)
        active[idx[~has_legal]] = False
        if not has_legal.any():
            continue
        live_local = np.flatnonzero(has_legal)
        live_global = idx[live_local]
        best = np.nanargmax(values[live_local], axis=1)
        chosen_flat = live_local * 4 + best
        chosen_after = moved.afterstates[chosen_flat]
        chosen_reward = moved.rewards[chosen_flat].astype(np.int64)

        for k, gid in enumerate(live_global.tolist()):
            trajectory_states[gid].append(boards[gid].copy())
            trajectory_scores[gid].append(int(scores[gid]))
            scores[gid] += int(chosen_reward[k])
            steps[gid] += 1
            boards[gid] = spawn_random(chosen_after[k], rngs[gid]).state
        decisions += len(live_global)
        if decisions // 10000 != (decisions - len(live_global)) // 10000:
            elapsed = time.perf_counter() - started
            print(
                f"source-games decisions={decisions} active={int(active.sum())}/{GAME_COUNT} "
                f"rate={decisions/elapsed:.0f}/s",
                flush=True,
            )

    selected_states = []
    selected_game = []
    selected_step = []
    selected_score = []
    summaries = []
    for gid in range(GAME_COUNT):
        states = trajectory_states[gid]
        if len(states) < STATES_PER_GAME:
            raise RuntimeError(f"complete game {gid} has only {len(states)} decision states")
        positions = np.rint(np.linspace(0, len(states) - 1, STATES_PER_GAME)).astype(np.int64)
        if len(np.unique(positions)) != STATES_PER_GAME:
            raise RuntimeError(f"sampling produced duplicate source positions for game {gid}")
        for pos in positions.tolist():
            selected_states.append(states[pos])
            selected_game.append(gid)
            selected_step.append(pos)
            selected_score.append(trajectory_scores[gid][pos])
        summaries.append({
            "game_id": gid,
            "seed": GAME_SEED_BASE + gid,
            "complete": True,
            "moves": int(steps[gid]),
            "final_score": int(scores[gid]),
            "max_tile_exp_terminal": int(boards[gid].max(initial=0)),
            "sampled_states": STATES_PER_GAME,
            "first_sample_step": int(positions[0]),
            "last_sample_step": int(positions[-1]),
        })

    np.savez_compressed(
        SOURCE_NPZ,
        states=np.ascontiguousarray(np.stack(selected_states), dtype=np.uint8),
        game_id=np.asarray(selected_game, dtype=np.int64),
        step_index=np.asarray(selected_step, dtype=np.int32),
        current_score=np.asarray(selected_score, dtype=np.int64),
    )
    SOURCE_GAMES.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(
        f"source games complete: {GAME_COUNT} games, {TARGET_STATES} selected states, "
        f"wall={time.perf_counter()-started:.2f}s",
        flush=True,
    )


def _worker_init():
    global _WORKER_TEACHER, _WORKER_SOURCE
    # Main process has already verified the frozen local checkpoint SHA. Workers
    # mmap that same file; avoid eight redundant 512 MiB SHA scans.
    _WORKER_TEACHER = TupleTeacher(
        verify_sha256=False, backend="cpp", cpp_prefetch=True
    )
    _WORKER_SOURCE = np.load(SOURCE_NPZ)


def _label_range(start: int, end: int):
    global _WORKER_TEACHER, _WORKER_SOURCE
    teacher = _WORKER_TEACHER
    source = _WORKER_SOURCE
    if teacher is None or source is None:
        raise RuntimeError("worker not initialized")
    states = np.ascontiguousarray(source["states"][start:end])
    n = len(states)
    values = np.empty((n, 4), dtype=np.float64)
    for i, state in enumerate(states):
        search = ExpectimaxTeacher(teacher, decision_depth=3, use_cache=True)
        values[i] = search.action_values(state)

    repeated = np.repeat(states, 4, axis=0)
    action_ids = np.tile(ACTIONS, n)
    moved = move_selected_batch(repeated, action_ids)
    legal = np.asarray(moved.moved, dtype=np.bool_).reshape(n, 4)
    rewards = moved.rewards.astype(np.int64).reshape(n, 4)
    afterstates = moved.afterstates.reshape(n, 4, 16).copy()
    rewards[~legal] = 0
    afterstates[~legal] = 0
    values[~legal] = np.nan
    return {
        "start": start,
        "end": end,
        "teacher_value": values,
        "reward": rewards,
        "afterstate": afterstates,
        "legal_mask": legal,
    }


def _chunk_path(start: int, end: int) -> Path:
    return CHUNK_DIR / f"chunk_{start:05d}_{end:05d}.npz"


def _label_all_states():
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    ranges = [
        (start, min(start + CHUNK_SIZE, TARGET_STATES))
        for start in range(0, TARGET_STATES, CHUNK_SIZE)
    ]
    pending = [(s, e) for s, e in ranges if not _chunk_path(s, e).exists()]
    if not pending:
        print("all label chunks already exist; reusing", flush=True)
        return

    source = np.load(SOURCE_NPZ)
    started = time.perf_counter()
    completed_before = TARGET_STATES - sum(e - s for s, e in pending)
    completed_now = 0

    with ProcessPoolExecutor(max_workers=WORKERS, initializer=_worker_init) as pool:
        for chunk_start, chunk_end in pending:
            subranges = [
                (s, min(s + SUBTASK_SIZE, chunk_end))
                for s in range(chunk_start, chunk_end, SUBTASK_SIZE)
            ]
            futures = [pool.submit(_label_range, s, e) for s, e in subranges]
            pieces = []
            for fut in as_completed(futures):
                piece = fut.result()
                pieces.append(piece)
                completed_now += piece["end"] - piece["start"]
                elapsed = time.perf_counter() - started
                total_done = completed_before + completed_now
                payload = {
                    "phase": "depth3_labeling",
                    "completed_states": total_done,
                    "target_states": TARGET_STATES,
                    "workers": WORKERS,
                    "elapsed_seconds_current_run": elapsed,
                    "current_run_roots_per_second": completed_now / elapsed,
                    "last_completed_range": [piece["start"], piece["end"]],
                }
                PROGRESS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                print(
                    f"teacher-label {total_done}/{TARGET_STATES} "
                    f"run-rate={payload['current_run_roots_per_second']:.2f} roots/s",
                    flush=True,
                )
            pieces.sort(key=lambda x: x["start"])
            expected = chunk_start
            for part in pieces:
                if part["start"] != expected:
                    raise RuntimeError("subtask coverage gap")
                expected = part["end"]
            if expected != chunk_end:
                raise RuntimeError("subtask coverage incomplete")
            np.savez_compressed(
                _chunk_path(chunk_start, chunk_end),
                teacher_value=np.concatenate([x["teacher_value"] for x in pieces], axis=0),
                reward=np.concatenate([x["reward"] for x in pieces], axis=0),
                afterstate=np.concatenate([x["afterstate"] for x in pieces], axis=0),
                legal_mask=np.concatenate([x["legal_mask"] for x in pieces], axis=0),
            )

    print("depth3 labeling complete", flush=True)


def _merge_final():
    source = np.load(SOURCE_NPZ)
    states = np.ascontiguousarray(source["states"])
    game_ids = np.asarray(source["game_id"], dtype=np.int64)
    step_index = np.asarray(source["step_index"], dtype=np.int32)
    current_score = np.asarray(source["current_score"], dtype=np.int64)

    values = []
    rewards = []
    afterstates = []
    legal_masks = []
    for start in range(0, TARGET_STATES, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, TARGET_STATES)
        path = _chunk_path(start, end)
        if not path.exists():
            raise RuntimeError(f"missing label chunk {path.name}")
        with np.load(path) as chunk:
            values.append(chunk["teacher_value"].copy())
            rewards.append(chunk["reward"].copy())
            afterstates.append(chunk["afterstate"].copy())
            legal_masks.append(chunk["legal_mask"].copy())

    teacher_value = np.concatenate(values, axis=0)
    reward = np.concatenate(rewards, axis=0)
    afterstate = np.concatenate(afterstates, axis=0)
    legal_mask = np.concatenate(legal_masks, axis=0)
    if len(teacher_value) != TARGET_STATES:
        raise RuntimeError("merged label count mismatch")

    labels = split_labels(game_ids)
    split = np.asarray([labels[int(g)] for g in game_ids], dtype="<U10")
    max_tile_exp = states.max(axis=1).astype(np.uint8)
    teacher_version = np.full(TARGET_STATES, "ordinary_td_comparator_ep4800000_7192719323a0", dtype="<U48")
    checkpoint_sha = np.full(TARGET_STATES, EXPECTED_SHA256, dtype="<U64")
    decision_depth = np.full(TARGET_STATES, 3, dtype=np.int32)
    data_source = np.full(
        TARGET_STATES,
        "complete_frozen_tuple_greedy_1ply_game_then_depth3_expectimax_label",
        dtype="<U72",
    )
    value_semantics = np.full(TARGET_STATES, SEARCH_VALUE_RAW_LEAF, dtype="<U24")
    search_mode = np.full(TARGET_STATES, "expectimax_formal_state_leaf", dtype="<U32")
    leaf_evaluator = np.full(TARGET_STATES, LEAF_EVALUATOR, dtype="<U48")

    np.savez_compressed(
        FINAL_NPZ,
        state=states,
        teacher_value=teacher_value,
        reward=reward,
        afterstate=afterstate,
        legal_mask=legal_mask,
        game_id=game_ids,
        step_index=step_index,
        current_score=current_score,
        max_tile_exp=max_tile_exp,
        teacher_version=teacher_version,
        checkpoint_sha256=checkpoint_sha,
        decision_depth=decision_depth,
        data_source=data_source,
        value_semantics=value_semantics,
        search_mode=search_mode,
        leaf_evaluator=leaf_evaluator,
        split=split,
    )

    summaries = json.loads(SOURCE_GAMES.read_text(encoding="utf-8"))
    split_game_counts = {
        name: len({int(g) for g in game_ids[split == name]})
        for name in ("train", "validation", "test")
    }
    split_state_counts = {
        name: int(np.count_nonzero(split == name))
        for name in ("train", "validation", "test")
    }
    manifest = {
        "result": "M3_SMALL_TEACHER_VALIDATION_SET_COMPLETE",
        "states": TARGET_STATES,
        "complete_source_games": GAME_COUNT,
        "states_per_game": STATES_PER_GAME,
        "source_policy": "frozen tuple checkpoint greedy_1ply",
        "teacher_label_policy": "exact depth-3 Expectimax",
        "decision_depth": 3,
        "checkpoint_sha256": EXPECTED_SHA256,
        "value_semantics": SEARCH_VALUE_RAW_LEAF,
        "canonical_unaugmented": True,
        "split_seed": 20260919,
        "split_game_counts": split_game_counts,
        "split_state_counts": split_state_counts,
        "game_id_overlap": {
            "train_validation": 0,
            "train_test": 0,
            "validation_test": 0,
        },
        "source_game_moves": {
            "min": min(x["moves"] for x in summaries),
            "median": float(np.median([x["moves"] for x in summaries])),
            "max": max(x["moves"] for x in summaries),
        },
        "artifact": str(FINAL_NPZ.relative_to(ROOT)).replace("\\", "/"),
    }
    FINAL_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Chunk files are temporary build products; canonical merged dataset is authoritative.
    shutil.rmtree(CHUNK_DIR)
    if PROGRESS.exists():
        PROGRESS.unlink()
    print(json.dumps(manifest, indent=2))


def main():
    ART.mkdir(parents=True, exist_ok=True)
    _generate_complete_source_games()
    _label_all_states()
    _merge_final()


if __name__ == "__main__":
    mp.freeze_support()
    main()
