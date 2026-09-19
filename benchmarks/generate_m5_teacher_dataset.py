"""Generate resumable M5 Teacher source games and depth-3 label shards."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from game2048.m2_fast_backend import move_selected_batch
from game2048.m3_search import ExpectimaxTeacher
from game2048.m3_tuple_teacher import TupleTeacher
from game2048.m5_pretrain import (
    GAMES_PER_SHARD,
    M4_PRIMARY_SHA256,
    PROMPT_SHA256,
    SHARD_STATES,
    SOURCE_RETRY_MAX_INDEX,
    STATES_PER_GAME,
    TEACHER_SHA256,
    TEACHER_VERSION,
    WORK_ORDER_VERSION,
    atomic_save_npz,
    atomic_write_json,
    sample_positions,
    sha256_file,
    source_attempt_seed,
    split_game_seed,
    validate_completed_shard,
)
from game2048.reference_env import spawn_random

ART = ROOT / "artifacts" / "m5"
DATA = ART / "datasets"
SESSION = ART / "progress" / "session.json"
SOURCE_MANIFEST = DATA / "source_manifest.json"
LABEL_MANIFEST = DATA / "label_manifest.json"
WORKERS = 8
SUBTASK_SIZE = 64
ACTIONS = np.arange(4, dtype=np.uint8)
SPLIT_GAMES = {
    "validation": 64,
    "test": 64,
}
PHASE_SCALE_GAMES = {
    "train-32k": 256,
    "train-131k": 1024,
    "train-524k": 4096,
}

_WORKER_TEACHER = None


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
def _session() -> dict:
    row = _read_json(SESSION)
    expected = {
        "schema_version": 1,
        "work_order_version": WORK_ORDER_VERSION,
        "prompt_sha256": PROMPT_SHA256,
        "teacher_sha256": TEACHER_SHA256,
        "m4_primary_sha256": M4_PRIMARY_SHA256,
    }
    for key, value in expected.items():
        if row.get(key) != value:
            raise RuntimeError(
                "M5_BLOCKED_RESUME_METADATA_MISMATCH"
            )
    return row


def _source_path(split: str) -> Path:
    return DATA / f"source_{split}.npz"


def _empty_source() -> dict[str, np.ndarray]:
    return {
        "state": np.empty((0, 16), dtype=np.uint8),
        "game_id": np.empty(0, dtype=np.int64),
        "game_seed": np.empty(0, dtype=np.int64),
        "step_index": np.empty(0, dtype=np.int32),
        "current_score": np.empty(0, dtype=np.int64),
        "max_tile_exp": np.empty(0, dtype=np.uint8),
    }
def _load_source(split: str) -> dict[str, np.ndarray]:
    path = _source_path(split)
    if not path.exists():
        return _empty_source()
    with np.load(path, allow_pickle=False) as data:
        required = set(_empty_source())
        if set(data.files) != required:
            raise RuntimeError(
                "M5_BLOCKED_DATASET_ARTIFACT_CORRUPT"
            )
        result = {
            key: np.ascontiguousarray(data[key])
            for key in required
        }
    rows = int(result["state"].shape[0])
    if result["state"].shape != (rows, 16):
        raise RuntimeError(
            "M5_BLOCKED_DATASET_ARTIFACT_CORRUPT"
        )
    if rows % STATES_PER_GAME != 0:
        raise RuntimeError(
            "M5_BLOCKED_DATASET_ARTIFACT_CORRUPT"
        )
    for key in (
        "game_id", "game_seed", "step_index",
        "current_score", "max_tile_exp",
    ):
        if result[key].shape != (rows,):
            raise RuntimeError(
                "M5_BLOCKED_DATASET_ARTIFACT_CORRUPT"
            )
    return result
def _source_split_manifest_path(split: str) -> Path:
    if split not in {"train", "validation", "test"}:
        raise ValueError("unknown split")
    return DATA / f"source_{split}_manifest.json"


def _write_source_split_manifest(split: str, manifest: dict) -> None:
    split_row = manifest["splits"][split]
    payload = {
        "schema_version": 1,
        "work_order_version": WORK_ORDER_VERSION,
        "teacher_sha256": TEACHER_SHA256,
        "source_policy": "frozen tuple checkpoint greedy_1ply",
        "states_per_game": STATES_PER_GAME,
        "split": split,
        **split_row,
    }
    atomic_write_json(
        _source_split_manifest_path(split),
        payload,
    )


def _source_manifest() -> dict:
    if SOURCE_MANIFEST.exists():
        row = _read_json(SOURCE_MANIFEST)
        if (
            row.get("work_order_version")
            != WORK_ORDER_VERSION
            or row.get("teacher_sha256")
            != TEACHER_SHA256
        ):
            raise RuntimeError(
                "M5_BLOCKED_RESUME_METADATA_MISMATCH"
            )
        for split_name, split_row in row.get("splits", {}).items():
            for record in split_row.get("games", []):
                gid = int(record["game_id"])
                if split_name == "train":
                    local_index = gid
                elif split_name == "validation":
                    local_index = gid - 100_000
                elif split_name == "test":
                    local_index = gid - 200_000
                else:
                    raise RuntimeError("M5_BLOCKED_DATASET_ARTIFACT_CORRUPT")
                _, canonical_seed = split_game_seed(split_name, local_index)
                record.setdefault("canonical_game_seed", int(canonical_seed))
                record.setdefault("retry_index", 0)
                record.setdefault("rejected_attempts", [])
        return row
    return {
        "schema_version": 1,
        "work_order_version": WORK_ORDER_VERSION,
        "teacher_sha256": TEACHER_SHA256,
        "source_policy": (
            "frozen tuple checkpoint greedy_1ply"
        ),
        "states_per_game": STATES_PER_GAME,
        "splits": {
            "train": {"games": []},
            "validation": {"games": []},
            "test": {"games": []},
        },
    }


def _label_manifest() -> dict:
    if LABEL_MANIFEST.exists():
        row = _read_json(LABEL_MANIFEST)
        if (
            row.get("work_order_version")
            != WORK_ORDER_VERSION
            or row.get("teacher_sha256")
            != TEACHER_SHA256
        ):
            raise RuntimeError(
                "M5_BLOCKED_RESUME_METADATA_MISMATCH"
            )
        return row
    return {
        "schema_version": 1,
        "work_order_version": WORK_ORDER_VERSION,
        "teacher_sha256": TEACHER_SHA256,
        "teacher_version": TEACHER_VERSION,
        "decision_depth": 3,
        "value_semantics": "SEARCH_VALUE_RAW_LEAF",
        "shard_states": SHARD_STATES,
        "games_per_shard": GAMES_PER_SHARD,
        "splits": {
            "train": {"shards": []},
            "validation": {"shards": []},
            "test": {"shards": []},
        },
    }


def _init_board(seed: int):
    rng = np.random.Generator(
        np.random.PCG64(int(seed))
    )
    board = np.zeros(16, dtype=np.uint8)
    board = spawn_random(board, rng).state
    board = spawn_random(board, rng).state
    return board, rng


def _batch_greedy_values(
    teacher: TupleTeacher,
    boards: np.ndarray,
):
    count = len(boards)
    repeated = np.repeat(
        np.ascontiguousarray(boards),
        4,
        axis=0,
    )
    actions = np.tile(ACTIONS, count)
    moved = move_selected_batch(
        repeated,
        actions,
    )
    legal = np.asarray(
        moved.moved,
        dtype=np.bool_,
    ).reshape(count, 4)
    flat = np.full(
        count * 4,
        np.nan,
        dtype=np.float64,
    )
    legal_flat = legal.reshape(-1)
    if bool(legal_flat.any()):
        after = np.ascontiguousarray(
            moved.afterstates[legal_flat]
        )
        flat[legal_flat] = (
            moved.rewards[legal_flat].astype(np.float64)
            + teacher.afterstate_values(after).astype(
                np.float64
            )
        )
    return moved, legal, flat.reshape(count, 4)


def _rollout_source_attempts(
    effective_seeds: np.ndarray,
    teacher: TupleTeacher,
) -> dict:
    seeds = np.asarray(effective_seeds, dtype=np.int64)
    count = int(seeds.size)
    boards = np.zeros((count, 16), dtype=np.uint8)
    rngs = []
    for local, seed in enumerate(seeds.tolist()):
        boards[local], rng = _init_board(int(seed))
        rngs.append(rng)
    active = np.ones(count, dtype=np.bool_)
    scores = np.zeros(count, dtype=np.int64)
    steps = np.zeros(count, dtype=np.int32)
    trajectory_states = [[] for _ in range(count)]
    trajectory_scores = [[] for _ in range(count)]
    started = time.perf_counter()
    decisions = 0
    while bool(active.any()):
        ids = np.flatnonzero(active)
        current = np.ascontiguousarray(boards[ids])
        moved, legal, values = _batch_greedy_values(
            teacher,
            current,
        )
        has_legal = legal.any(axis=1)
        active[ids[~has_legal]] = False
        if not bool(has_legal.any()):
            continue
        live_local = np.flatnonzero(has_legal)
        live_global = ids[live_local]
        best = np.nanargmax(
            values[live_local],
            axis=1,
        )
        chosen_flat = live_local * 4 + best
        chosen_after = moved.afterstates[chosen_flat]
        chosen_reward = moved.rewards[
            chosen_flat
        ].astype(np.int64)
        for row, game_index in enumerate(live_global.tolist()):
            trajectory_states[game_index].append(
                boards[game_index].copy()
            )
            trajectory_scores[game_index].append(
                int(scores[game_index])
            )
            scores[game_index] += int(chosen_reward[row])
            steps[game_index] += 1
            boards[game_index] = spawn_random(
                chosen_after[row],
                rngs[game_index],
            ).state
        decisions += len(live_global)
    elapsed = time.perf_counter() - started
    return {
        "boards": boards,
        "scores": scores,
        "steps": steps,
        "trajectory_states": trajectory_states,
        "trajectory_scores": trajectory_scores,
        "decisions": int(decisions),
        "elapsed": float(elapsed),
    }


def _generate_source_batch(
    split: str,
    start_local: int,
    count: int,
    teacher: TupleTeacher,
) -> tuple[dict[str, np.ndarray], list[dict]]:
    game_ids = np.empty(count, dtype=np.int64)
    canonical_seeds = np.empty(count, dtype=np.int64)
    effective_seeds = np.empty(count, dtype=np.int64)
    retry_indices = np.zeros(count, dtype=np.int32)
    rejected_attempts = [[] for _ in range(count)]
    for local in range(count):
        gid, canonical_seed, effective_seed = source_attempt_seed(
            split,
            start_local + local,
            0,
        )
        game_ids[local] = gid
        canonical_seeds[local] = canonical_seed
        effective_seeds[local] = effective_seed

    rollout = _rollout_source_attempts(
        effective_seeds,
        teacher,
    )
    boards = rollout["boards"]
    scores = rollout["scores"]
    steps = rollout["steps"]
    trajectory_states = rollout["trajectory_states"]
    trajectory_scores = rollout["trajectory_scores"]
    total_decisions = int(rollout["decisions"])
    total_elapsed = float(rollout["elapsed"])

    while True:
        short = np.flatnonzero(steps < STATES_PER_GAME)
        if short.size == 0:
            break
        retry_seeds = []
        for game_index in short.tolist():
            rejected_attempts[game_index].append({
                "game_seed": int(effective_seeds[game_index]),
                "moves": int(steps[game_index]),
                "final_score": int(scores[game_index]),
                "max_tile_exp_terminal": int(
                    boards[game_index].max(initial=0)
                ),
            })
            next_retry = int(retry_indices[game_index]) + 1
            if next_retry > SOURCE_RETRY_MAX_INDEX:
                raise RuntimeError(
                    "M5_BLOCKED_SOURCE_GAME_RETRY_EXHAUSTED"
                )
            _, canonical_seed, effective_seed = source_attempt_seed(
                split,
                start_local + game_index,
                next_retry,
            )
            if int(canonical_seed) != int(canonical_seeds[game_index]):
                raise RuntimeError(
                    "M5_BLOCKED_SOURCE_RETRY_SEED_DRIFT"
                )
            retry_indices[game_index] = next_retry
            effective_seeds[game_index] = effective_seed
            retry_seeds.append(effective_seed)
            print(
                "source retry "
                f"game_id={int(game_ids[game_index])} "
                f"retry_index={next_retry} "
                f"rejected_moves={rejected_attempts[game_index][-1]['moves']} "
                f"effective_seed={effective_seed}",
                flush=True,
            )
        retried = _rollout_source_attempts(
            np.asarray(retry_seeds, dtype=np.int64),
            teacher,
        )
        total_decisions += int(retried["decisions"])
        total_elapsed += float(retried["elapsed"])
        for retry_row, game_index in enumerate(short.tolist()):
            boards[game_index] = retried["boards"][retry_row]
            scores[game_index] = retried["scores"][retry_row]
            steps[game_index] = retried["steps"][retry_row]
            trajectory_states[game_index] = (
                retried["trajectory_states"][retry_row]
            )
            trajectory_scores[game_index] = (
                retried["trajectory_scores"][retry_row]
            )

    print(
        f"source split={split} games="
        f"{start_local}-{start_local + count - 1} "
        f"decisions={total_decisions} elapsed={total_elapsed:.2f}s "
        f"rate={total_decisions / total_elapsed:.1f}/s",
        flush=True,
    )
    sampled_states = []
    sampled_gid = []
    sampled_seed = []
    sampled_step = []
    sampled_score = []
    sampled_max = []
    summaries = []
    for local in range(count):
        positions = sample_positions(int(steps[local]))
        states = trajectory_states[local]
        score_trace = trajectory_scores[local]
        for position in positions.tolist():
            state = states[position]
            sampled_states.append(state)
            sampled_gid.append(int(game_ids[local]))
            sampled_seed.append(int(effective_seeds[local]))
            sampled_step.append(int(position))
            sampled_score.append(int(score_trace[position]))
            sampled_max.append(int(state.max(initial=0)))
        summaries.append({
            "game_id": int(game_ids[local]),
            "canonical_game_seed": int(canonical_seeds[local]),
            "game_seed": int(effective_seeds[local]),
            "retry_index": int(retry_indices[local]),
            "rejected_attempts": rejected_attempts[local],
            "complete": True,
            "moves": int(steps[local]),
            "final_score": int(scores[local]),
            "max_tile_exp_terminal": int(
                boards[local].max(initial=0)
            ),
            "sampled_states": STATES_PER_GAME,
            "first_sample_step": int(positions[0]),
            "last_sample_step": int(positions[-1]),
        })
    arrays = {
        "state": np.ascontiguousarray(
            np.stack(sampled_states),
            dtype=np.uint8,
        ),
        "game_id": np.asarray(sampled_gid, dtype=np.int64),
        "game_seed": np.asarray(sampled_seed, dtype=np.int64),
        "step_index": np.asarray(sampled_step, dtype=np.int32),
        "current_score": np.asarray(sampled_score, dtype=np.int64),
        "max_tile_exp": np.asarray(sampled_max, dtype=np.uint8),
    }
    return arrays, summaries

def _concat_source(
    left: dict[str, np.ndarray],
    right: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    if len(left["state"]) == 0:
        return right
    return {
        key: np.ascontiguousarray(
            np.concatenate([left[key], right[key]], axis=0)
        )
        for key in left
    }


def _ensure_source(
    split: str,
    target_games: int,
) -> tuple[dict[str, np.ndarray], dict]:
    DATA.mkdir(parents=True, exist_ok=True)
    manifest = _source_manifest()
    source = _load_source(split)
    current_games = (
        int(len(source["state"]) // STATES_PER_GAME)
    )
    records = manifest["splits"][split]["games"]
    if len(records) != current_games:
        raise RuntimeError(
            "M5_BLOCKED_DATASET_ARTIFACT_CORRUPT"
        )
    if current_games > target_games:
        expected_rows = target_games * STATES_PER_GAME
        source = {
            key: np.ascontiguousarray(
                value[:expected_rows]
            )
            for key, value in source.items()
        }
        current_games = target_games
        records = records[:target_games]
        manifest["splits"][split]["games"] = records
    if current_games < target_games:
        teacher = TupleTeacher(
            verify_sha256=True,
            backend="cpp",
            cpp_prefetch=True,
        )
        while current_games < target_games:
            take = min(64, target_games - current_games)
            batch, summaries = _generate_source_batch(
                split,
                current_games,
                take,
                teacher,
            )
            source = _concat_source(source, batch)
            records.extend(summaries)
            current_games += take
            atomic_save_npz(
                _source_path(split),
                **source,
            )
            manifest["splits"][split]["games"] = records
            manifest["splits"][split].update({
                "source_path": str(
                    _source_path(split).relative_to(ROOT)
                ).replace("\\", "/"),
                "source_sha256": sha256_file(
                    _source_path(split)
                ),
                "complete_games": current_games,
                "states": int(
                    current_games * STATES_PER_GAME
                ),
            })
            atomic_write_json(
                SOURCE_MANIFEST,
                manifest,
            )
            _write_source_split_manifest(
                split,
                manifest,
            )
    if not _source_path(split).exists():
        atomic_save_npz(
            _source_path(split),
            **source,
        )
    manifest["splits"][split].update({
        "source_path": str(
            _source_path(split).relative_to(ROOT)
        ).replace("\\", "/"),
        "source_sha256": sha256_file(
            _source_path(split)
        ),
        "complete_games": current_games,
        "states": int(
            current_games * STATES_PER_GAME
        ),
    })
    atomic_write_json(
        SOURCE_MANIFEST,
        manifest,
    )
    _write_source_split_manifest(
        split,
        manifest,
    )
    return source, manifest


def _worker_init() -> None:
    global _WORKER_TEACHER
    _WORKER_TEACHER = TupleTeacher(
        verify_sha256=False,
        backend="cpp",
        cpp_prefetch=True,
    )


def _label_subtask(
    states: np.ndarray,
) -> dict[str, np.ndarray]:
    global _WORKER_TEACHER
    if _WORKER_TEACHER is None:
        raise RuntimeError("M5 worker not initialized")
    boards = np.ascontiguousarray(
        states,
        dtype=np.uint8,
    )
    count = len(boards)
    values = np.empty(
        (count, 4),
        dtype=np.float64,
    )
    for index, board in enumerate(boards):
        search = ExpectimaxTeacher(
            _WORKER_TEACHER,
            decision_depth=3,
            use_cache=True,
        )
        values[index] = search.action_values(board)
    repeated = np.repeat(
        boards,
        4,
        axis=0,
    )
    actions = np.tile(ACTIONS, count)
    moved = move_selected_batch(
        repeated,
        actions,
    )
    legal = np.asarray(
        moved.moved,
        dtype=np.bool_,
    ).reshape(count, 4)
    rewards = moved.rewards.astype(
        np.int32
    ).reshape(count, 4)
    afterstates = moved.afterstates.reshape(
        count,
        4,
        16,
    ).copy()
    rewards[np.logical_not(legal)] = 0
    afterstates[np.logical_not(legal)] = 0
    values[np.logical_not(legal)] = np.nan
    return {
        "teacher_value": values,
        "reward": rewards,
        "afterstate": afterstates,
        "legal_mask": legal,
    }


def _shard_path(
    split: str,
    shard_index: int,
) -> Path:
    return (
        DATA
        / split
        / f"shard_{shard_index:04d}.npz"
    )


def _find_manifest_row(
    manifest: dict,
    split: str,
    shard_index: int,
) -> dict | None:
    for row in manifest["splits"][split]["shards"]:
        if int(row["shard_index"]) == int(shard_index):
            return row
    return None


def _validate_existing_shards(
    manifest: dict,
    split: str,
) -> None:
    for row in manifest["splits"][split]["shards"]:
        if row.get("status") != "complete":
            continue
        validate_completed_shard(
            ROOT / row["path"],
            row["sha256"],
        )


def _build_shard_arrays(
    source: dict[str, np.ndarray],
    start: int,
    end: int,
    labels: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    rows = end - start
    if rows != SHARD_STATES:
        raise ValueError("M5 shard must be 2048 states")
    source_slice = {
        key: value[start:end]
        for key, value in source.items()
    }
    return {
        "state": np.ascontiguousarray(
            source_slice["state"],
            dtype=np.uint8,
        ),
        "teacher_value": np.ascontiguousarray(
            labels["teacher_value"],
            dtype=np.float64,
        ),
        "reward": np.ascontiguousarray(
            labels["reward"],
            dtype=np.int32,
        ),
        "afterstate": np.ascontiguousarray(
            labels["afterstate"],
            dtype=np.uint8,
        ),
        "legal_mask": np.ascontiguousarray(
            labels["legal_mask"],
            dtype=np.bool_,
        ),
        "game_id": np.ascontiguousarray(
            source_slice["game_id"],
            dtype=np.int64,
        ),
        "game_seed": np.ascontiguousarray(
            source_slice["game_seed"],
            dtype=np.int64,
        ),
        "step_index": np.ascontiguousarray(
            source_slice["step_index"],
            dtype=np.int32,
        ),
        "current_score": np.ascontiguousarray(
            source_slice["current_score"],
            dtype=np.int64,
        ),
        "max_tile_exp": np.ascontiguousarray(
            source_slice["max_tile_exp"],
            dtype=np.uint8,
        ),
        "teacher_version": np.full(
            rows,
            TEACHER_VERSION,
            dtype="<U48",
        ),
        "checkpoint_sha256": np.full(
            rows,
            TEACHER_SHA256.lower(),
            dtype="<U64",
        ),
        "decision_depth": np.full(
            rows,
            3,
            dtype=np.int32,
        ),
        "value_semantics": np.full(
            rows,
            "SEARCH_VALUE_RAW_LEAF",
            dtype="<U24",
        ),
        "data_source": np.full(
            rows,
            "frozen_tuple_greedy_1ply_complete_game_depth3_label",
            dtype="<U64",
        ),
    }
def _label_required_shards(
    source: dict[str, np.ndarray],
    split: str,
    target_shards: int,
    manifest: dict,
) -> dict:
    required_rows = int(target_shards * SHARD_STATES)
    if len(source["state"]) < required_rows:
        raise RuntimeError(
            "M5_BLOCKED_DATASET_ARTIFACT_CORRUPT"
        )
    _validate_existing_shards(
        manifest,
        split,
    )
    pending = []
    for shard_index in range(target_shards):
        row = _find_manifest_row(
            manifest,
            split,
            shard_index,
        )
        if row is not None and row.get("status") == "complete":
            continue
        pending.append(shard_index)
    if not pending:
        return manifest
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=WORKERS,
        mp_context=context,
        initializer=_worker_init,
    ) as pool:
        for shard_index in pending:
            start = shard_index * SHARD_STATES
            end = start + SHARD_STATES
            started = time.perf_counter()
            futures = {}
            for sub_start in range(
                start,
                end,
                SUBTASK_SIZE,
            ):
                sub_end = min(
                    sub_start + SUBTASK_SIZE,
                    end,
                )
                future = pool.submit(
                    _label_subtask,
                    np.ascontiguousarray(
                        source["state"][
                            sub_start:sub_end
                        ]
                    ),
                )
                futures[future] = (
                    sub_start,
                    sub_end,
                )
            pieces = {}
            completed = 0
            for future in as_completed(futures):
                sub_start, sub_end = futures[future]
                pieces[sub_start] = future.result()
                completed += sub_end - sub_start
                elapsed = time.perf_counter() - started
                rate = completed / elapsed
                eta = (
                    SHARD_STATES - completed
                ) / rate
                print(
                    "teacher-label "
                    f"split={split} "
                    f"shard={shard_index + 1}/{target_shards} "
                    f"roots={completed}/{SHARD_STATES} "
                    f"elapsed={elapsed:.1f}s "
                    f"rate={rate:.3f} roots/s "
                    f"eta={eta:.1f}s",
                    flush=True,
                )
            ordered_starts = sorted(pieces)
            labels = {
                key: np.concatenate(
                    [
                        pieces[sub_start][key]
                        for sub_start in ordered_starts
                    ],
                    axis=0,
                )
                for key in (
                    "teacher_value",
                    "reward",
                    "afterstate",
                    "legal_mask",
                )
            }
            arrays = _build_shard_arrays(
                source,
                start,
                end,
                labels,
            )
            path = _shard_path(
                split,
                shard_index,
            )
            atomic_save_npz(
                path,
                **arrays,
            )
            digest = sha256_file(path)
            entry = {
                "shard_index": shard_index,
                "status": "complete",
                "path": str(
                    path.relative_to(ROOT)
                ).replace("\\", "/"),
                "sha256": digest,
                "rows": SHARD_STATES,
                "label_wall_seconds": float(
                    time.perf_counter() - started
                ),
                "game_id_start": int(
                    arrays["game_id"][0]
                ),
                "game_id_end": int(
                    arrays["game_id"][-1]
                ),
            }
            existing = _find_manifest_row(
                manifest,
                split,
                shard_index,
            )
            if existing is None:
                manifest["splits"][split][
                    "shards"
                ].append(entry)
            else:
                existing.clear()
                existing.update(entry)
            manifest["splits"][split]["shards"].sort(
                key=lambda row: int(
                    row["shard_index"]
                )
            )
            manifest["source_manifest_sha256"] = (
                sha256_file(SOURCE_MANIFEST)
            )
            atomic_write_json(
                LABEL_MANIFEST,
                manifest,
            )
            validate_completed_shard(
                path,
                digest,
            )
    return manifest


def _phase_contract(
    phase: str,
    session: dict,
) -> tuple[str, int, str, str]:
    if phase == "validation":
        allowed = {
            "PROFILE_DONE",
            "DATA_32K_RUNNING",
        }
        if session["state"] not in allowed:
            raise RuntimeError(
                "M5_BLOCKED_RESUME_GIT_STATE"
            )
        return (
            "validation",
            64,
            "DATA_32K_RUNNING",
            "DATA_32K_RUNNING",
        )
    if phase == "train-32k":
        if session["state"] not in {
            "DATA_32K_RUNNING",
            "DATA_32K_DONE",
        }:
            raise RuntimeError(
                "M5_BLOCKED_RESUME_GIT_STATE"
            )
        return (
            "train",
            256,
            "DATA_32K_RUNNING",
            "DATA_32K_DONE",
        )
    if phase == "train-131k":
        if session["state"] not in {
            "TRAIN_32K_DONE",
            "DATA_131K_RUNNING",
            "DATA_131K_DONE",
        }:
            raise RuntimeError(
                "M5_BLOCKED_RESUME_GIT_STATE"
            )
        return (
            "train",
            1024,
            "DATA_131K_RUNNING",
            "DATA_131K_DONE",
        )
    if phase == "train-524k":
        if session["state"] not in {
            "SCALE_GATE_DONE",
            "DATA_524K_RUNNING",
            "DATA_524K_DONE",
        }:
            raise RuntimeError(
                "M5_BLOCKED_RESUME_GIT_STATE"
            )
        gate_path = (
            ART / "progress" / "scale_gate.json"
        )
        if not gate_path.exists():
            raise RuntimeError(
                "M5_BLOCKED_RESUME_ARTIFACT_CORRUPT"
            )
        gate = _read_json(gate_path)
        if gate.get("trigger_524k") is not True:
            raise RuntimeError(
                "M5_BLOCKED_524K_NOT_TRIGGERED"
            )
        return (
            "train",
            4096,
            "DATA_524K_RUNNING",
            "DATA_524K_DONE",
        )
    if phase == "test":
        if session["state"] not in {
            "FINAL_SCALE_SELECTED",
            "TEST_RUNNING",
        }:
            raise RuntimeError(
                "M5_BLOCKED_TEST_ISOLATION"
            )
        return (
            "test",
            64,
            "TEST_RUNNING",
            "TEST_RUNNING",
        )
    raise ValueError("unsupported phase")


def _target_shards(split: str, target_games: int) -> int:
    if target_games % GAMES_PER_SHARD != 0:
        raise AssertionError("whole-game shard contract")
    return int(
        target_games // GAMES_PER_SHARD
    )


def _split_complete(
    manifest: dict,
    split: str,
    target_shards: int,
) -> bool:
    rows = manifest["splits"][split]["shards"]
    complete = {
        int(row["shard_index"])
        for row in rows
        if row.get("status") == "complete"
    }
    return complete.issuperset(
        set(range(target_shards))
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        required=True,
        choices=(
            "validation",
            "train-32k",
            "train-131k",
            "train-524k",
            "test",
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        required=True,
    )
    args = parser.parse_args()
    session = _session()
    split, target_games, running_state, done_state = (
        _phase_contract(
            args.phase,
            session,
        )
    )
    target_shards = _target_shards(
        split,
        target_games,
    )
    session["state"] = running_state
    atomic_write_json(SESSION, session)

    source, _source_info = _ensure_source(
        split,
        target_games,
    )
    manifest = _label_manifest()
    manifest = _label_required_shards(
        source,
        split,
        target_shards,
        manifest,
    )
    if not _split_complete(
        manifest,
        split,
        target_shards,
    ):
        raise RuntimeError(
            "M5_BLOCKED_DATASET_ARTIFACT_CORRUPT"
        )
    if split != "test":
        session["state"] = done_state
        atomic_write_json(
            SESSION,
            session,
        )
    print(
        json.dumps(
            {
                "phase": args.phase,
                "split": split,
                "target_games": target_games,
                "target_shards": target_shards,
                "session_state": session["state"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
