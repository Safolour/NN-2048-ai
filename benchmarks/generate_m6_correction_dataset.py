"""Generate resumable M6 Student-rollout correction data and Teacher labels."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from game2048.m2_fast_backend import move_selected_batch
from game2048.m3_search import ExpectimaxTeacher
from game2048.m6_state_correction import (
    GAMES_PER_SHARD, M5_CHECKPOINT_SHA256, M5_TAG_SHA, PROMPT_SHA256,
    SHARD_STATES, SOURCE_RETRY_MAX_INDEX, STATES_PER_GAME, TEACHER_FILENAME,
    TEACHER_SHA256, TEACHER_VERSION, TEST_SOURCE_GAMES, TRAIN_SOURCE_GAMES,
    VALIDATION_SOURCE_GAMES, WORK_ORDER_VERSION, assert_test_allowed,
    atomic_save_npz, atomic_write_json, load_m5_model, load_m6_teacher,
    rollout_student_games, sample_positions, sha256_file, source_attempt_seed,
    teacher_best_action, validate_completed_shard,
)

ART = ROOT / "artifacts" / "m6"
DATA = ART / "datasets"
PROGRESS = ART / "progress"
SESSION = PROGRESS / "session.json"
LABEL_MANIFEST = DATA / "label_manifest.json"
M5_CHECKPOINT = ROOT / "artifacts" / "m5" / "checkpoints" / "524k" / "20262103_final.pt"
TEACHER_PATH = ROOT / "teacher_checkpoints" / "m6" / TEACHER_FILENAME
WORKERS = 8
SUBTASK_SIZE = 64
ACTIONS = np.arange(4, dtype=np.uint8)
SPLIT_GAMES = {
    "train": TRAIN_SOURCE_GAMES,
    "validation": VALIDATION_SOURCE_GAMES,
    "test": TEST_SOURCE_GAMES,
}
_WORKER_TEACHER = None


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


def _source_path(split: str) -> Path:
    return DATA / f"source_{split}.npz"


def _source_manifest_path(split: str) -> Path:
    return DATA / f"source_{split}_manifest.json"


def _empty_source() -> dict[str, np.ndarray]:
    return {
        "state": np.empty((0, 16), dtype=np.uint8),
        "student_logits": np.empty((0, 4), dtype=np.float32),
        "student_action": np.empty(0, dtype=np.uint8),
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
        if set(data.files) != set(_empty_source()):
            raise RuntimeError("M6_BLOCKED_DATASET_ARTIFACT_CORRUPT")
        output = {key: np.ascontiguousarray(data[key]) for key in data.files}
    rows = len(output["state"])
    shapes = {
        "state": (rows, 16), "student_logits": (rows, 4),
        "student_action": (rows,), "game_id": (rows,), "game_seed": (rows,),
        "step_index": (rows,), "current_score": (rows,), "max_tile_exp": (rows,),
    }
    if any(output[key].shape != shape for key, shape in shapes.items()):
        raise RuntimeError("M6_BLOCKED_DATASET_ARTIFACT_CORRUPT")
    if rows % STATES_PER_GAME:
        raise RuntimeError("M6_BLOCKED_DATASET_ARTIFACT_CORRUPT")
    return output


def _source_manifest(split: str) -> dict:
    path = _source_manifest_path(split)
    if path.exists():
        row = _read_json(path)
        if (
            row.get("work_order_version") != WORK_ORDER_VERSION
            or row.get("split") != split
            or row.get("student_checkpoint_sha256") != M5_CHECKPOINT_SHA256
        ):
            raise RuntimeError("M6_BLOCKED_RESUME_METADATA_MISMATCH")
        return row
    return {
        "schema_version": 1,
        "work_order_version": WORK_ORDER_VERSION,
        "split": split,
        "student_checkpoint_sha256": M5_CHECKPOINT_SHA256,
        "policy_contract": {
            "architecture": "ResidualMLP2048",
            "pure_nn": True,
            "fp32_inference": True,
            "teacher_search_tuple": False,
            "d4_ensemble": False,
            "exploration": False,
            "tie_tolerance": 1e-7,
            "spawn_rng": "PCG64(game_seed)",
            "tie_rng": "PCG64(game_seed XOR 0x9E3779B9)",
        },
        "canonical_unaugmented": True,
        "states_per_game": STATES_PER_GAME,
        "games": [],
        "complete_games": 0,
        "states": 0,
    }


def _concat(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if len(left["state"]) == 0:
        return {key: np.ascontiguousarray(value) for key, value in right.items()}
    return {
        key: np.ascontiguousarray(np.concatenate((left[key], right[key]), axis=0))
        for key in left
    }


def _accepted_game_arrays(
    game_id: int,
    canonical_seed: int,
    accepted_seed: int,
    retry_index: int,
    rejected: list[dict],
    trajectory: dict,
) -> tuple[dict[str, np.ndarray], dict]:
    positions = sample_positions(int(trajectory["moves"]))
    arrays = {
        "state": np.ascontiguousarray(trajectory["state"][positions], dtype=np.uint8),
        "student_logits": np.ascontiguousarray(
            trajectory["student_logits"][positions], dtype=np.float32
        ),
        "student_action": np.ascontiguousarray(
            trajectory["student_action"][positions], dtype=np.uint8
        ),
        "game_id": np.full(STATES_PER_GAME, int(game_id), dtype=np.int64),
        "game_seed": np.full(STATES_PER_GAME, int(accepted_seed), dtype=np.int64),
        "step_index": np.ascontiguousarray(positions, dtype=np.int32),
        "current_score": np.ascontiguousarray(
            trajectory["current_score"][positions], dtype=np.int64
        ),
        "max_tile_exp": np.ascontiguousarray(
            trajectory["max_tile_exp"][positions], dtype=np.uint8
        ),
    }
    summary = {
        "game_id": int(game_id),
        "canonical_game_seed": int(canonical_seed),
        "game_seed": int(accepted_seed),
        "retry_index": int(retry_index),
        "rejected_attempts": rejected,
        "complete": True,
        "moves": int(trajectory["moves"]),
        "final_score": int(trajectory["final_score"]),
        "max_tile_exp_terminal": int(trajectory["max_tile_exp_terminal"]),
        "sampled_states": STATES_PER_GAME,
        "sampled_step_indices": positions.tolist(),
    }
    return arrays, summary


def _generate_source_batch(
    split: str,
    start_local: int,
    count: int,
    model: torch.nn.Module,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], list[dict]]:
    accepted_arrays = []
    summaries = []
    pending = list(range(count))
    retry_indices = np.zeros(count, dtype=np.int32)
    rejected: list[list[dict]] = [[] for _ in range(count)]
    canonical = np.empty(count, dtype=np.int64)
    game_ids = np.empty(count, dtype=np.int64)
    accepted: list[dict | None] = [None] * count
    accepted_seed = np.empty(count, dtype=np.int64)
    started = time.perf_counter()

    for local in range(count):
        gid, can, _ = source_attempt_seed(split, start_local + local, 0)
        game_ids[local] = gid
        canonical[local] = can

    while pending:
        seeds = np.asarray([
            source_attempt_seed(
                split, start_local + local, int(retry_indices[local])
            )[2]
            for local in pending
        ], dtype=np.int64)
        results = rollout_student_games(model, seeds, device)
        next_pending = []
        for row, local in enumerate(pending):
            result = results[row]
            if int(result["moves"]) >= STATES_PER_GAME:
                accepted[local] = result
                accepted_seed[local] = int(seeds[row])
                continue
            rejected[local].append({
                "game_seed": int(seeds[row]),
                "moves": int(result["moves"]),
                "final_score": int(result["final_score"]),
                "max_tile_exp": int(result["max_tile_exp_terminal"]),
            })
            next_retry = int(retry_indices[local]) + 1
            if next_retry > SOURCE_RETRY_MAX_INDEX:
                raise RuntimeError("M6_BLOCKED_SOURCE_GAME_RETRY_EXHAUSTED")
            retry_indices[local] = next_retry
            next_pending.append(local)
            print(
                f"student-source retry split={split} game_id={int(game_ids[local])} "
                f"retry={next_retry} rejected_moves={int(result['moves'])}",
                flush=True,
            )
        pending = next_pending

    for local in range(count):
        result = accepted[local]
        if result is None:
            raise RuntimeError("M6_BLOCKED_RUNTIME_ERROR")
        arrays, summary = _accepted_game_arrays(
            int(game_ids[local]), int(canonical[local]),
            int(accepted_seed[local]), int(retry_indices[local]),
            rejected[local], result,
        )
        accepted_arrays.append(arrays)
        summaries.append(summary)

    batch = {
        key: np.concatenate([row[key] for row in accepted_arrays], axis=0)
        for key in accepted_arrays[0]
    }
    elapsed = time.perf_counter() - started
    mean_score = float(np.mean([row["final_score"] for row in summaries]))
    print(
        f"student-source split={split} games={start_local + count}/{SPLIT_GAMES[split]} "
        f"batch={count} elapsed={elapsed:.2f}s rate={count / elapsed:.2f} games/s "
        f"rolling_batch_mean={mean_score:.1f}",
        flush=True,
    )
    return batch, summaries


def _ensure_source(split: str, target_games: int) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    manifest = _source_manifest(split)
    source = _load_source(split)
    current = len(source["state"]) // STATES_PER_GAME
    if int(manifest.get("complete_games", 0)) != current:
        raise RuntimeError("M6_BLOCKED_DATASET_ARTIFACT_CORRUPT")
    if len(manifest.get("games", [])) != current:
        raise RuntimeError("M6_BLOCKED_DATASET_ARTIFACT_CORRUPT")
    if current > target_games:
        raise RuntimeError("M6_BLOCKED_DATASET_ARTIFACT_CORRUPT")
    if current and sha256_file(_source_path(split)) != manifest.get("source_sha256"):
        raise RuntimeError("M6_BLOCKED_DATASET_ARTIFACT_CORRUPT")

    device = torch.device("cuda")
    model, _ = load_m5_model(M5_CHECKPOINT, device)
    overall_started = time.perf_counter()
    while current < target_games:
        take = min(64, target_games - current)
        batch, summaries = _generate_source_batch(
            split, current, take, model, device
        )
        source = _concat(source, batch)
        manifest["games"].extend(summaries)
        current += take
        atomic_save_npz(_source_path(split), **source)
        manifest.update({
            "source_path": str(_source_path(split).relative_to(ROOT)).replace("\\", "/"),
            "source_sha256": sha256_file(_source_path(split)),
            "complete_games": int(current),
            "states": int(current * STATES_PER_GAME),
        })
        atomic_write_json(_source_manifest_path(split), manifest)
        elapsed = time.perf_counter() - overall_started
        rate = current / elapsed
        print(
            f"student-source phase={split} completed={current}/{target_games} "
            f"elapsed={elapsed:.1f}s throughput={rate:.2f} games/s "
            f"eta={(target_games-current)/rate:.1f}s",
            flush=True,
        )
    del model
    torch.cuda.empty_cache()


def _label_manifest() -> dict:
    if LABEL_MANIFEST.exists():
        row = _read_json(LABEL_MANIFEST)
        if (
            row.get("work_order_version") != WORK_ORDER_VERSION
            or row.get("teacher_checkpoint_sha256") != TEACHER_SHA256
            or row.get("student_checkpoint_sha256") != M5_CHECKPOINT_SHA256
        ):
            raise RuntimeError("M6_BLOCKED_RESUME_METADATA_MISMATCH")
        return row
    return {
        "schema_version": 1,
        "work_order_version": WORK_ORDER_VERSION,
        "student_checkpoint_sha256": M5_CHECKPOINT_SHA256,
        "teacher_checkpoint_sha256": TEACHER_SHA256,
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


def _worker_init() -> None:
    global _WORKER_TEACHER
    _WORKER_TEACHER = load_m6_teacher(
        TEACHER_PATH, verify_sha256=False, backend="cpp", cpp_prefetch=True
    )


def _label_subtask(states: np.ndarray) -> dict[str, np.ndarray]:
    global _WORKER_TEACHER
    if _WORKER_TEACHER is None:
        raise RuntimeError("M6 worker not initialized")
    boards = np.ascontiguousarray(states, dtype=np.uint8)
    values = np.empty((len(boards), 4), dtype=np.float64)
    for index, board in enumerate(boards):
        search = ExpectimaxTeacher(
            _WORKER_TEACHER, decision_depth=3, use_cache=True
        )
        values[index] = search.action_values(board)
    repeated = np.repeat(boards, 4, axis=0)
    actions = np.tile(ACTIONS, len(boards))
    moved = move_selected_batch(repeated, actions)
    legal = np.asarray(moved.moved, dtype=np.bool_).reshape(len(boards), 4)
    rewards = moved.rewards.astype(np.int32).reshape(len(boards), 4)
    afterstates = moved.afterstates.reshape(len(boards), 4, 16).copy()
    rewards[~legal] = 0
    afterstates[~legal] = 0
    values[~legal] = np.nan
    return {
        "teacher_value": values,
        "reward": rewards,
        "afterstate": afterstates,
        "legal_mask": legal,
    }


def _shard_path(split: str, shard_index: int) -> Path:
    return DATA / split / f"shard_{shard_index:04d}.npz"


def _find_row(manifest: dict, split: str, shard_index: int) -> dict | None:
    for row in manifest["splits"][split]["shards"]:
        if int(row["shard_index"]) == int(shard_index):
            return row
    return None


def _validate_existing(manifest: dict, split: str) -> None:
    for row in manifest["splits"][split]["shards"]:
        if row.get("status") == "complete":
            validate_completed_shard(ROOT / row["path"], row["sha256"])


def _build_shard(
    source: dict[str, np.ndarray],
    start: int,
    end: int,
    labels: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    rows = end - start
    legal = np.ascontiguousarray(labels["legal_mask"], dtype=np.bool_)
    values = np.ascontiguousarray(labels["teacher_value"], dtype=np.float64)
    best = teacher_best_action(values, legal).astype(np.uint8)
    student_action = np.ascontiguousarray(
        source["student_action"][start:end], dtype=np.uint8
    )
    return {
        "state": np.ascontiguousarray(source["state"][start:end], dtype=np.uint8),
        "teacher_value": values,
        "reward": np.ascontiguousarray(labels["reward"], dtype=np.int32),
        "afterstate": np.ascontiguousarray(labels["afterstate"], dtype=np.uint8),
        "legal_mask": legal,
        "student_logits": np.ascontiguousarray(
            source["student_logits"][start:end], dtype=np.float32
        ),
        "student_action": student_action,
        "teacher_best_action": best,
        "disagreement": np.ascontiguousarray(best != student_action, dtype=np.bool_),
        "game_id": np.ascontiguousarray(source["game_id"][start:end], dtype=np.int64),
        "game_seed": np.ascontiguousarray(source["game_seed"][start:end], dtype=np.int64),
        "step_index": np.ascontiguousarray(source["step_index"][start:end], dtype=np.int32),
        "current_score": np.ascontiguousarray(
            source["current_score"][start:end], dtype=np.int64
        ),
        "max_tile_exp": np.ascontiguousarray(
            source["max_tile_exp"][start:end], dtype=np.uint8
        ),
        "student_checkpoint_sha256": np.full(
            rows, M5_CHECKPOINT_SHA256, dtype="<U64"
        ),
        "teacher_checkpoint_sha256": np.full(
            rows, TEACHER_SHA256, dtype="<U64"
        ),
        "decision_depth": np.full(rows, 3, dtype=np.int32),
        "value_semantics": np.full(
            rows, "SEARCH_VALUE_RAW_LEAF", dtype="<U24"
        ),
        "data_source": np.full(
            rows,
            "m5_student_pure_nn_rollout_depth3_teacher_relabel",
            dtype="<U64",
        ),
    }


def _label_split(split: str, target_games: int) -> None:
    source = _load_source(split)
    if len(source["state"]) != target_games * STATES_PER_GAME:
        raise RuntimeError("M6_BLOCKED_DATASET_ARTIFACT_CORRUPT")
    source_manifest = _source_manifest(split)
    if sha256_file(_source_path(split)) != source_manifest.get("source_sha256"):
        raise RuntimeError("M6_BLOCKED_DATASET_ARTIFACT_CORRUPT")
    target_shards = target_games // GAMES_PER_SHARD
    manifest = _label_manifest()
    _validate_existing(manifest, split)
    pending = []
    for index in range(target_shards):
        row = _find_row(manifest, split, index)
        if row is None or row.get("status") != "complete":
            pending.append(index)
    if not pending:
        return

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
            for sub_start in range(start, end, SUBTASK_SIZE):
                sub_end = min(sub_start + SUBTASK_SIZE, end)
                future = pool.submit(
                    _label_subtask,
                    np.ascontiguousarray(source["state"][sub_start:sub_end]),
                )
                futures[future] = (sub_start, sub_end)
            pieces = {}
            completed = 0
            for future in as_completed(futures):
                sub_start, sub_end = futures[future]
                pieces[sub_start] = future.result()
                completed += sub_end - sub_start
                elapsed = time.perf_counter() - started
                rate = completed / elapsed
                progress = {
                    "schema_version": 1,
                    "split": split,
                    "shard_index": shard_index,
                    "shard_number": shard_index + 1,
                    "shards_total": target_shards,
                    "completed_roots": completed,
                    "total_roots": SHARD_STATES,
                    "elapsed_seconds": float(elapsed),
                    "roots_per_second": float(rate),
                    "eta_seconds": float((SHARD_STATES - completed) / rate),
                    "status": "running",
                }
                atomic_write_json(PROGRESS / "label_progress.json", progress)
                print(
                    f"teacher-label split={split} shard={shard_index + 1}/{target_shards} "
                    f"roots={completed}/{SHARD_STATES} elapsed={elapsed:.1f}s "
                    f"rate={rate:.3f} roots/s eta={(SHARD_STATES-completed)/rate:.1f}s",
                    flush=True,
                )

            starts = sorted(pieces)
            labels = {
                key: np.concatenate([pieces[pos][key] for pos in starts], axis=0)
                for key in ("teacher_value", "reward", "afterstate", "legal_mask")
            }
            arrays = _build_shard(source, start, end, labels)
            path = _shard_path(split, shard_index)
            atomic_save_npz(path, **arrays)
            digest = sha256_file(path)
            entry = {
                "shard_index": shard_index,
                "status": "complete",
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": digest,
                "rows": SHARD_STATES,
                "label_wall_seconds": float(time.perf_counter() - started),
                "game_id_start": int(arrays["game_id"][0]),
                "game_id_end": int(arrays["game_id"][-1]),
                "disagreement_rate": float(np.mean(arrays["disagreement"])),
            }
            existing = _find_row(manifest, split, shard_index)
            if existing is None:
                manifest["splits"][split]["shards"].append(entry)
            else:
                existing.clear()
                existing.update(entry)
            manifest["splits"][split]["shards"].sort(
                key=lambda row: int(row["shard_index"])
            )
            manifest["splits"][split]["source_manifest_path"] = str(
                _source_manifest_path(split).relative_to(ROOT)
            ).replace("\\", "/")
            manifest["splits"][split]["source_manifest_sha256"] = sha256_file(
                _source_manifest_path(split)
            )
            atomic_write_json(LABEL_MANIFEST, manifest)
            validate_completed_shard(path, digest)
            atomic_write_json(PROGRESS / "label_progress.json", {
                **entry,
                "split": split,
                "status": "complete",
                "completed_roots": SHARD_STATES,
                "total_roots": SHARD_STATES,
            })


def _source_phase(split: str, session: dict) -> None:
    target = SPLIT_GAMES[split]
    if split == "train":
        if session["state"] not in {
            "PROFILE_DONE", "SOURCE_TRAIN_RUNNING", "SOURCE_TRAIN_DONE"
        }:
            raise RuntimeError("M6_BLOCKED_RESUME_GIT_STATE")
        if session["state"] == "SOURCE_TRAIN_DONE":
            return
        session["state"] = "SOURCE_TRAIN_RUNNING"
        atomic_write_json(SESSION, session)
        _ensure_source(split, target)
        session["state"] = "SOURCE_TRAIN_DONE"
    elif split == "validation":
        if session["state"] not in {
            "SOURCE_TRAIN_DONE", "SOURCE_VALIDATION_RUNNING",
            "SOURCE_VALIDATION_DONE",
        }:
            raise RuntimeError("M6_BLOCKED_RESUME_GIT_STATE")
        if session["state"] == "SOURCE_VALIDATION_DONE":
            return
        session["state"] = "SOURCE_VALIDATION_RUNNING"
        atomic_write_json(SESSION, session)
        _ensure_source(split, target)
        session["state"] = "SOURCE_VALIDATION_DONE"
    else:
        raise ValueError("source phase supports train/validation")
    atomic_write_json(SESSION, session)


def _label_phase(split: str, session: dict) -> None:
    if split == "train":
        if session["state"] not in {
            "SOURCE_VALIDATION_DONE", "LABEL_TRAIN_RUNNING", "LABEL_TRAIN_DONE"
        }:
            raise RuntimeError("M6_BLOCKED_RESUME_GIT_STATE")
        if session["state"] == "LABEL_TRAIN_DONE":
            return
        session["state"] = "LABEL_TRAIN_RUNNING"
        atomic_write_json(SESSION, session)
        _label_split("train", TRAIN_SOURCE_GAMES)
        session["state"] = "LABEL_TRAIN_DONE"
        atomic_write_json(SESSION, session)
        return
    if split == "validation":
        if session["state"] not in {
            "LABEL_TRAIN_DONE", "LABEL_VALIDATION_RUNNING",
            "LABEL_VALIDATION_DONE",
        }:
            raise RuntimeError("M6_BLOCKED_RESUME_GIT_STATE")
        session["state"] = "LABEL_VALIDATION_RUNNING"
        atomic_write_json(SESSION, session)
        _label_split("validation", VALIDATION_SOURCE_GAMES)
        session["state"] = "LABEL_VALIDATION_DONE"
        atomic_write_json(SESSION, session)
        session["state"] = "DATA_READY"
        atomic_write_json(SESSION, session)
        return
    raise ValueError("label phase supports train/validation")


def _test_phase(session: dict) -> None:
    assert_test_allowed(session["state"])
    if session["state"] not in {
        "GAMEPLAY_VALIDATION_DONE", "TEST_DATA_RUNNING"
    }:
        raise RuntimeError("M6_BLOCKED_RESUME_GIT_STATE")
    session["state"] = "TEST_DATA_RUNNING"
    atomic_write_json(SESSION, session)
    _ensure_source("test", TEST_SOURCE_GAMES)
    _label_split("test", TEST_SOURCE_GAMES)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", required=True, choices=("train", "validation", "test")
    )
    parser.add_argument("--resume", action="store_true", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("M6_BLOCKED_RUNTIME_ENVIRONMENT")
    if sha256_file(TEACHER_PATH).upper() != TEACHER_SHA256:
        raise RuntimeError("M6_BLOCKED_TEACHER_CHECKPOINT")
    session = _session()
    if args.phase == "test":
        _test_phase(session)
    elif args.phase == "train":
        if session["state"] in {
            "PROFILE_DONE", "SOURCE_TRAIN_RUNNING", "SOURCE_TRAIN_DONE"
        }:
            _source_phase("train", session)
        else:
            _label_phase("train", session)
    elif args.phase == "validation":
        if session["state"] in {
            "SOURCE_TRAIN_DONE", "SOURCE_VALIDATION_RUNNING",
            "SOURCE_VALIDATION_DONE",
        }:
            _source_phase("validation", session)
        else:
            _label_phase("validation", session)
    print(json.dumps(_session(), indent=2), flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
