"""M6 deterministic helpers for Student State Correction."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .m2_fast_backend import legal_mask_batch, move_selected_batch
from .m2_models import ResidualMLP2048, trainable_parameter_count
from .m2_symmetry import transform_action_batch, transform_board_batch
from .m3_tuple_teacher import (
    FEATURE_COUNT_PER_PATTERN,
    HEADER_BYTES,
    PATTERN_COUNT,
    SYMMETRY_COUNT,
    WEIGHT_COUNT_PER_STAGE,
    TupleTeacher,
    parse_checkpoint_header,
)
from .m4_compare import (
    TIE_TOLERANCE,
    initialize_game,
    select_game_actions,
    score_summary,
)
from .m5_pretrain import (
    atomic_save_npz,
    atomic_torch_save,
    atomic_write_json,
    d4_metrics,
    high_tile_metrics,
    make_optimizer,
    paired_bootstrap,
    parameter_update_checks,
    random_legal_baseline,
    set_determinism,
    sha256_file,
    snapshot_parameters,
    teacher_best_action,
    teacher_metrics,
)

WORK_ORDER_VERSION = "M6_WO_STATE_CORRECTION_V2"
BASE_HEAD = "6a9b7614e1f7e971e246ae9820f3d052e07875ed"
PROMPT_SHA256 = "70BD2D2C67DC56C37290BF05A9254AB1F7FFF40155C68F7514904A1265CFCBA3"
M5_TAG_SHA = "36fe93cb211b08916d40df3a7e80bdea0b04a658"
M5_CHECKPOINT_SHA256 = "4C1313E9085E3A0C1FEC4C6AAA1200A0A5A4377F3EAE0176659F80E9EEAE5784"
M5_ANCHOR_MANIFEST_SHA256 = "AE6BD51FCF0815A3EB9B919CC4D5980E41EC614CC01ECE174B58F20E3AFE1551"
M5_PRETRAIN_TEACHER_SHA256 = "7192719323A073BA2B6B19B62CB7D46EF4AA90ECC8C4AE6BAF27AD0C51566A84"
TEACHER_SHA256 = "908BA8B8D01A4AFF76D32BE65220D61B6ED702696AA83E10B41594EB5A25AACC"
TEACHER_VERSION = "formal_afterstate_td0_tc_ep10000000_908ba8b8d01a"
TEACHER_FILENAME = "formal_afterstate_td0_tc_ep10000000_908ba8b8d01a.bin"
TEACHER_FILE_SIZE = 3_221_225_728
ARCHITECTURE = "ResidualMLP2048"
PARAMETER_COUNT = 5_264_710
TRAINING_SEEDS = (20263101, 20263102, 20263103)
EPOCHS = 30
BATCH_SIZE = 1024
HALF_BATCH = 512
CORRECTION_ROWS = 131_072
ANCHOR_ROWS = 524_288
STATES_PER_GAME = 128
SHARD_STATES = 2048
GAMES_PER_SHARD = 16
SOURCE_RETRY_SEED_STRIDE = 1_000_000_000
SOURCE_RETRY_MAX_INDEX = 32
TRAIN_SOURCE_GAMES = 1024
VALIDATION_SOURCE_GAMES = 64
TEST_SOURCE_GAMES = 64
TRAIN_SOURCE_SEED_BASE = 20288001
VALIDATION_SOURCE_SEED_BASE = 20290001
TEST_SOURCE_SEED_BASE = 20290101
DEV_SEED_START = 20291001
DEV_GAMES = 2000
VALIDATION_GAME_SEED_START = 20294001
VALIDATION_GAMES = 5000
FINAL_SEED_START = 20300001
FINAL_GAMES = 10_000
GAME_SHARD_SIZE = 100
BOOTSTRAP_RESAMPLES = 10_000
DEV_BOOTSTRAP_SEEDS = {
    20263101: 20266201,
    20263102: 20266202,
    20263103: 20266203,
}
VALIDATION_BOOTSTRAP_SEED = 20266210
FINAL_BOOTSTRAP_SEED = 20266220

M6_TEACHER_EXPECTED_METADATA = {
    "format_version": 2,
    "stage_count": 2,
    "stage_thresholds": (0, 16384),
    "stage_policy_version": 1,
    "rule": 1,
    "phase": 2,
    "alpha_normalizer": 128,
    "global_episodes_completed": 10_000_000,
    "otd_episodes_completed": 9_000_000,
    "tc_episodes_completed": 1_000_000,
    "otd_episode_budget": 9_000_000,
    "tc_episode_budget": 1_000_000,
    "training_seed": 1,
    "has_coherence_stats": True,
}


def load_m6_teacher(
    checkpoint_path: str | Path,
    *,
    verify_sha256: bool = True,
    backend: str = "cpp",
    cpp_prefetch: bool = True,
) -> TupleTeacher:
    """Load the promoted M6 TC checkpoint while ignoring training-only tail stats."""
    path = Path(checkpoint_path).resolve()
    if path.stat().st_size != TEACHER_FILE_SIZE:
        raise RuntimeError("M6_BLOCKED_TEACHER_CHECKPOINT")
    with path.open("rb") as handle:
        metadata = parse_checkpoint_header(handle.read(HEADER_BYTES))
    for key, expected in M6_TEACHER_EXPECTED_METADATA.items():
        if getattr(metadata, key) != expected:
            raise RuntimeError("M6_BLOCKED_TEACHER_CHECKPOINT")
    weight_end = HEADER_BYTES + metadata.stage_count * WEIGHT_COUNT_PER_STAGE * 4
    if weight_end >= path.stat().st_size:
        raise RuntimeError("M6_BLOCKED_TEACHER_CHECKPOINT")
    digest = sha256_file(path) if verify_sha256 else None
    if verify_sha256 and digest.upper() != TEACHER_SHA256:
        raise RuntimeError("M6_BLOCKED_TEACHER_CHECKPOINT")

    teacher = TupleTeacher.__new__(TupleTeacher)
    teacher.path = path
    teacher.backend = str(backend).lower()
    if teacher.backend not in {"python", "cpp"}:
        raise ValueError("backend must be 'python' or 'cpp'")
    teacher.cpp_prefetch = bool(cpp_prefetch)
    teacher.metadata = metadata
    teacher.sha256 = digest
    teacher._weights = np.memmap(
        path,
        dtype="<f4",
        mode="r",
        offset=HEADER_BYTES,
        shape=(metadata.stage_count, WEIGHT_COUNT_PER_STAGE),
    )
    teacher._feature_bases = np.repeat(
        np.arange(PATTERN_COUNT, dtype=np.uint32) * FEATURE_COUNT_PER_PATTERN,
        SYMMETRY_COUNT,
    )
    teacher._stage_thresholds_array = np.ascontiguousarray(
        np.asarray(metadata.stage_thresholds, dtype=np.uint64)
    )
    return teacher

REQUIRED_SHARD_KEYS = {
    "state", "teacher_value", "reward", "afterstate", "legal_mask",
    "student_logits", "student_action", "teacher_best_action", "disagreement",
    "game_id", "game_seed", "step_index", "current_score", "max_tile_exp",
    "student_checkpoint_sha256", "teacher_checkpoint_sha256",
    "decision_depth", "value_semantics", "data_source",
}


def plan_id(seed: int) -> str:
    if int(seed) not in TRAINING_SEEDS:
        raise ValueError("unsupported M6 training seed")
    return f"M6_PLAN_V1:{int(seed)}:CORR50_ANCHOR50:PCG64_STATELESS_EPOCH"


def plan_sha256(seed: int) -> str:
    return hashlib.sha256(plan_id(seed).encode("ascii")).hexdigest().upper()


def sample_positions(moves: int) -> np.ndarray:
    if int(moves) < STATES_PER_GAME:
        raise RuntimeError("M6_BLOCKED_SOURCE_GAME_TOO_SHORT")
    positions = np.rint(
        np.linspace(0, int(moves) - 1, STATES_PER_GAME)
    ).astype(np.int64)
    if np.unique(positions).size != STATES_PER_GAME:
        raise RuntimeError("M6_BLOCKED_SOURCE_SAMPLE_DUPLICATE")
    return positions


def split_game_seed(split: str, local_index: int) -> tuple[int, int]:
    i = int(local_index)
    if split == "train":
        if not 0 <= i < TRAIN_SOURCE_GAMES:
            raise ValueError("train local index out of range")
        return i, TRAIN_SOURCE_SEED_BASE + i
    if split == "validation":
        if not 0 <= i < VALIDATION_SOURCE_GAMES:
            raise ValueError("validation local index out of range")
        return 100_000 + i, VALIDATION_SOURCE_SEED_BASE + i
    if split == "test":
        if not 0 <= i < TEST_SOURCE_GAMES:
            raise ValueError("test local index out of range")
        return 200_000 + i, TEST_SOURCE_SEED_BASE + i
    raise ValueError("unknown split")


def source_attempt_seed(
    split: str, local_index: int, retry_index: int
) -> tuple[int, int, int]:
    retry = int(retry_index)
    if not 0 <= retry <= SOURCE_RETRY_MAX_INDEX:
        raise ValueError("source retry index out of range")
    game_id, canonical = split_game_seed(split, local_index)
    return (
        game_id,
        canonical,
        canonical + retry * SOURCE_RETRY_SEED_STRIDE,
    )


def epoch_plan(
    seed: int,
    epoch: int,
    *,
    correction_rows: int = CORRECTION_ROWS,
    anchor_rows: int = ANCHOR_ROWS,
) -> dict[str, np.ndarray]:
    if int(seed) not in TRAINING_SEEDS:
        raise ValueError("unsupported M6 training seed")
    if not 0 <= int(epoch) < EPOCHS:
        raise ValueError("epoch must be 0..29")
    if correction_rows <= 0 or anchor_rows < correction_rows:
        raise ValueError("invalid row counts")
    crng = np.random.Generator(np.random.PCG64(int(seed) + 300_000 + int(epoch)))
    correction_order = crng.permutation(
        np.arange(correction_rows, dtype=np.int64)
    )
    correction_tids = crng.integers(
        0, 8, size=correction_rows, dtype=np.uint8
    )
    arng = np.random.Generator(np.random.PCG64(int(seed) + 400_000 + int(epoch)))
    anchor_permutation = arng.permutation(
        np.arange(anchor_rows, dtype=np.int64)
    )
    anchor_subset = np.ascontiguousarray(
        anchor_permutation[:correction_rows], dtype=np.int64
    )
    anchor_tids = arng.integers(
        0, 8, size=correction_rows, dtype=np.uint8
    )
    return {
        "correction_order": np.ascontiguousarray(correction_order),
        "correction_tids": np.ascontiguousarray(correction_tids),
        "anchor_subset": anchor_subset,
        "anchor_tids": np.ascontiguousarray(anchor_tids),
    }


def load_m5_model(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[ResidualMLP2048, dict]:
    path = Path(checkpoint_path)
    if sha256_file(path) != M5_CHECKPOINT_SHA256:
        raise RuntimeError("M6_BLOCKED_M5_BASELINE_CHECKPOINT")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    expected = {
        "architecture": ARCHITECTURE,
        "scale": "524k",
        "seed": 20262103,
        "epoch": 30,
        "teacher_sha256": M5_PRETRAIN_TEACHER_SHA256,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError("M6_BLOCKED_M5_BASELINE_CHECKPOINT")
    set_determinism(0)
    model = ResidualMLP2048()
    if trainable_parameter_count(model) != PARAMETER_COUNT:
        raise RuntimeError("M6_BLOCKED_MODEL_DRIFT")
    model.load_state_dict(payload["model_state"])
    model.to(device)
    return model, payload


def fresh_model_from_state(
    model_state: dict[str, torch.Tensor],
    seed: int,
    device: torch.device,
) -> tuple[ResidualMLP2048, torch.optim.Optimizer]:
    set_determinism(int(seed))
    model = ResidualMLP2048()
    if trainable_parameter_count(model) != PARAMETER_COUNT:
        raise RuntimeError("M6_BLOCKED_MODEL_DRIFT")
    model.load_state_dict(model_state)
    model.to(device)
    optimizer = make_optimizer(model)
    return model, optimizer


def fresh_finetune(
    seed: int,
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[ResidualMLP2048, torch.optim.Optimizer, dict]:
    path = Path(checkpoint_path)
    if sha256_file(path) != M5_CHECKPOINT_SHA256:
        raise RuntimeError("M6_BLOCKED_M5_BASELINE_CHECKPOINT")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model, optimizer = fresh_model_from_state(
        payload["model_state"], int(seed), device
    )
    return model, optimizer, payload


def correction_anchor_batch(
    correction_state: np.ndarray,
    correction_target: np.ndarray,
    anchor_state: np.ndarray,
    anchor_target: np.ndarray,
    plan: dict[str, np.ndarray],
    step: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    start = int(step) * HALF_BATCH
    end = start + HALF_BATCH
    ci = plan["correction_order"][start:end]
    ai = plan["anchor_subset"][start:end]
    if ci.size != HALF_BATCH or ai.size != HALF_BATCH:
        raise ValueError("M6 batch must have 512 correction and 512 anchor rows")
    cb = torch.from_numpy(np.ascontiguousarray(correction_state[ci])).to(device)
    ct = torch.from_numpy(np.ascontiguousarray(correction_target[ci], dtype=np.int64)).to(device)
    ab = torch.from_numpy(np.ascontiguousarray(anchor_state[ai])).to(device)
    at = torch.from_numpy(np.ascontiguousarray(anchor_target[ai], dtype=np.int64)).to(device)
    ctid = torch.from_numpy(
        np.ascontiguousarray(plan["correction_tids"][start:end], dtype=np.int64)
    ).to(device)
    atid = torch.from_numpy(
        np.ascontiguousarray(plan["anchor_tids"][start:end], dtype=np.int64)
    ).to(device)
    cb = transform_board_batch(cb, ctid)
    ct = transform_action_batch(ct, ctid)
    ab = transform_board_batch(ab, atid)
    at = transform_action_batch(at, atid)
    return torch.cat((cb, ab), dim=0), torch.cat((ct, at), dim=0)


def rollout_student_games(
    model: torch.nn.Module,
    game_seeds: Sequence[int],
    device: torch.device,
) -> list[dict]:
    seeds = np.asarray(game_seeds, dtype=np.int64)
    if seeds.ndim != 1 or seeds.size == 0:
        raise ValueError("game seeds must be nonempty 1-D")
    if np.unique(seeds).size != seeds.size:
        raise ValueError("game seeds must be unique")
    count = int(seeds.size)
    boards = np.zeros((count, 16), dtype=np.uint8)
    spawn_rngs = []
    tie_rngs = []
    for i, seed in enumerate(seeds.tolist()):
        board, spawn_rng, tie_rng = initialize_game(int(seed))
        boards[i] = board
        spawn_rngs.append(spawn_rng)
        tie_rngs.append(tie_rng)
    trajectories = [
        {"state": [], "student_logits": [], "student_action": [],
         "current_score": [], "max_tile_exp": []}
        for _ in range(count)
    ]
    scores = np.zeros(count, dtype=np.int64)
    moves = np.zeros(count, dtype=np.int32)
    active = np.ones(count, dtype=np.bool_)
    model.eval()
    with torch.inference_mode():
        while bool(active.any()):
            ids = np.flatnonzero(active)
            current = np.ascontiguousarray(boards[ids])
            legal = legal_mask_batch(current)
            terminal = ~legal.any(axis=1)
            if bool(terminal.any()):
                active[ids[terminal]] = False
                keep = ~terminal
                ids = ids[keep]
                current = np.ascontiguousarray(current[keep])
                legal = np.ascontiguousarray(legal[keep])
                if ids.size == 0:
                    continue
            logits = (
                model(torch.from_numpy(current).to(device)).float().cpu().numpy()
            )
            actions = select_game_actions(
                logits, legal, [tie_rngs[int(i)] for i in ids],
                tolerance=TIE_TOLERANCE,
            )
            moved = move_selected_batch(current, actions)
            if not bool(moved.moved.all()):
                raise RuntimeError("M6_BLOCKED_RUNTIME_ERROR")
            for row, game_index in enumerate(ids.tolist()):
                tr = trajectories[game_index]
                tr["state"].append(current[row].copy())
                tr["student_logits"].append(
                    np.asarray(logits[row], dtype=np.float32).copy()
                )
                tr["student_action"].append(int(actions[row]))
                tr["current_score"].append(int(scores[game_index]))
                tr["max_tile_exp"].append(int(current[row].max(initial=0)))
                scores[game_index] += int(moved.rewards[row])
                moves[game_index] += 1
                boards[game_index] = __import__(
                    "game2048.reference_env", fromlist=["spawn_random"]
                ).spawn_random(
                    moved.afterstates[row], spawn_rngs[game_index]
                ).state
    output = []
    for i in range(count):
        tr = trajectories[i]
        output.append({
            "game_seed": int(seeds[i]),
            "moves": int(moves[i]),
            "final_score": int(scores[i]),
            "max_tile_exp_terminal": int(boards[i].max(initial=0)),
            "state": np.asarray(tr["state"], dtype=np.uint8),
            "student_logits": np.asarray(tr["student_logits"], dtype=np.float32),
            "student_action": np.asarray(tr["student_action"], dtype=np.uint8),
            "current_score": np.asarray(tr["current_score"], dtype=np.int64),
            "max_tile_exp": np.asarray(tr["max_tile_exp"], dtype=np.uint8),
        })
    return output


def select_candidate(run_means: dict[int, float]) -> int:
    if set(map(int, run_means)) != set(TRAINING_SEEDS):
        raise ValueError("candidate selection requires exactly the three M6 seeds")
    return min(
        (int(seed) for seed in run_means),
        key=lambda seed: (-float(run_means[seed]), seed),
    )


def promotion_result(
    *,
    dev_positive: bool,
    validation_gain_pass: bool,
    final_gain_pass: bool,
    learning_sanity_pass: bool,
    all_gates_pass: bool,
) -> str:
    if all((
        dev_positive,
        validation_gain_pass,
        final_gain_pass,
        learning_sanity_pass,
        all_gates_pass,
    )):
        return "PROMOTION_ELIGIBLE"
    return "NO_PROMOTION"


def assert_test_allowed(session_state: str) -> None:
    allowed = {
        "CANDIDATE_LOCKED", "GAMEPLAY_VALIDATION_RUNNING",
        "GAMEPLAY_VALIDATION_DONE", "TEST_DATA_RUNNING",
        "TEACHER_TEST_DONE", "FINAL_GAMEPLAY_RUNNING",
        "FINAL_GAMEPLAY_DONE", "PROMOTION_DECIDED",
        "P10_RUNNING", "P10_DONE", "REPORT_WRITTEN",
        "CANDIDATE_PUSHED", "CLOSEOUT_REPORT_READY",
        "CLOSEOUT_PUSHED", "COMPLETE",
    }
    if str(session_state) not in allowed:
        raise RuntimeError("M6_BLOCKED_TEST_ISOLATION")


def training_label_manifest_sha256(
    manifest_path: str | Path,
    root_dir: str | Path,
) -> str:
    """Return the immutable pre-test M6 correction-manifest identity.

    The formal M6 label manifest is append-only across the test boundary: train
    and validation are frozen before training, while the test split is populated
    only after candidate lock.  Training checkpoints therefore bind to the
    writer-identical manifest view with an empty test split.
    """
    target = Path(manifest_path)
    root = Path(root_dir)
    try:
        raw = target.read_bytes()
        manifest = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise RuntimeError("M6_BLOCKED_RESUME_METADATA_MISMATCH") from exc

    expected = {
        "schema_version": 1,
        "work_order_version": WORK_ORDER_VERSION,
        "student_checkpoint_sha256": M5_CHECKPOINT_SHA256,
        "teacher_checkpoint_sha256": TEACHER_SHA256,
        "teacher_version": TEACHER_VERSION,
        "decision_depth": 3,
        "value_semantics": "SEARCH_VALUE_RAW_LEAF",
        "shard_states": SHARD_STATES,
        "games_per_shard": GAMES_PER_SHARD,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError("M6_BLOCKED_RESUME_METADATA_MISMATCH")

    splits = manifest.get("splits")
    if not isinstance(splits, dict) or set(splits) != {"train", "validation", "test"}:
        raise RuntimeError("M6_BLOCKED_RESUME_METADATA_MISMATCH")

    for split in ("train", "validation"):
        section = splits.get(split)
        if not isinstance(section, dict):
            raise RuntimeError("M6_BLOCKED_RESUME_METADATA_MISMATCH")
        source_path = section.get("source_manifest_path")
        source_sha = section.get("source_manifest_sha256")
        shards = section.get("shards")
        if not isinstance(source_path, str) or not isinstance(source_sha, str):
            raise RuntimeError("M6_BLOCKED_RESUME_METADATA_MISMATCH")
        source = root / source_path
        if not source.exists() or sha256_file(source) != source_sha.upper():
            raise RuntimeError("M6_BLOCKED_DATASET_ARTIFACT_CORRUPT")
        if not isinstance(shards, list) or not shards:
            raise RuntimeError("M6_BLOCKED_RESUME_METADATA_MISMATCH")
        indices = [int(row.get("shard_index", -1)) for row in shards]
        if indices != list(range(len(shards))):
            raise RuntimeError("M6_BLOCKED_RESUME_METADATA_MISMATCH")
        for row in shards:
            if row.get("status") != "complete" or int(row.get("rows", -1)) != SHARD_STATES:
                raise RuntimeError("M6_BLOCKED_RESUME_METADATA_MISMATCH")
            rel_path = row.get("path")
            digest = row.get("sha256")
            if not isinstance(rel_path, str) or not isinstance(digest, str):
                raise RuntimeError("M6_BLOCKED_RESUME_METADATA_MISMATCH")
            shard_path = root / rel_path
            if not shard_path.exists() or sha256_file(shard_path) != digest.upper():
                raise RuntimeError("M6_BLOCKED_DATASET_ARTIFACT_CORRUPT")

    training_view = dict(manifest)
    training_view["splits"] = {
        "train": splits["train"],
        "validation": splits["validation"],
        "test": {"shards": []},
    }
    rendered = json.dumps(
        training_view, indent=2, sort_keys=True, allow_nan=False
    )
    newline = "\r\n" if b"\r\n" in raw else "\n"
    if newline != "\n":
        rendered = rendered.replace("\n", newline)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest().upper()


def validate_completed_shard(path: str | Path, expected_sha256: str) -> dict:
    target = Path(path)
    if not target.exists() or sha256_file(target) != str(expected_sha256).upper():
        raise RuntimeError("M6_BLOCKED_DATASET_ARTIFACT_CORRUPT")
    try:
        with np.load(target, allow_pickle=False) as data:
            if set(data.files) != REQUIRED_SHARD_KEYS:
                raise RuntimeError("M6_BLOCKED_DATASET_ARTIFACT_CORRUPT")
            n = SHARD_STATES
            shapes = {
                "state": (n, 16), "teacher_value": (n, 4),
                "reward": (n, 4), "afterstate": (n, 4, 16),
                "legal_mask": (n, 4), "student_logits": (n, 4),
                "student_action": (n,), "teacher_best_action": (n,),
                "disagreement": (n,), "game_id": (n,), "game_seed": (n,),
                "step_index": (n,), "current_score": (n,),
                "max_tile_exp": (n,), "student_checkpoint_sha256": (n,),
                "teacher_checkpoint_sha256": (n,), "decision_depth": (n,),
                "value_semantics": (n,), "data_source": (n,),
            }
            for key, shape in shapes.items():
                if data[key].shape != shape:
                    raise RuntimeError("M6_BLOCKED_DATASET_ARTIFACT_CORRUPT")
            if data["state"].dtype != np.uint8:
                raise RuntimeError("M6_BLOCKED_DATASET_ARTIFACT_CORRUPT")
            if data["teacher_value"].dtype != np.float64:
                raise RuntimeError("M6_BLOCKED_DATASET_ARTIFACT_CORRUPT")
            if data["student_logits"].dtype != np.float32:
                raise RuntimeError("M6_BLOCKED_DATASET_ARTIFACT_CORRUPT")
            legal = np.asarray(data["legal_mask"], dtype=np.bool_)
            values = np.asarray(data["teacher_value"], dtype=np.float64)
            if not bool(legal.any(axis=1).all()):
                raise RuntimeError("M6_BLOCKED_DATASET_ARTIFACT_CORRUPT")
            if not bool(np.isnan(values[~legal]).all()):
                raise RuntimeError("M6_BLOCKED_DATASET_ARTIFACT_CORRUPT")
            best = teacher_best_action(values, legal).astype(np.uint8)
            if not np.array_equal(best, data["teacher_best_action"]):
                raise RuntimeError("M6_BLOCKED_DATASET_ARTIFACT_CORRUPT")
            disagreement = best != np.asarray(data["student_action"], dtype=np.uint8)
            if not np.array_equal(disagreement, data["disagreement"]):
                raise RuntimeError("M6_BLOCKED_DATASET_ARTIFACT_CORRUPT")
            gids, counts = np.unique(data["game_id"], return_counts=True)
            if gids.size != GAMES_PER_SHARD or not bool((counts == STATES_PER_GAME).all()):
                raise RuntimeError("M6_BLOCKED_DATASET_ARTIFACT_CORRUPT")
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError("M6_BLOCKED_DATASET_ARTIFACT_CORRUPT") from exc
    return {"path": str(target), "sha256": str(expected_sha256).upper()}


__all__ = [
    "ANCHOR_ROWS", "ARCHITECTURE", "BASE_HEAD", "BATCH_SIZE",
    "BOOTSTRAP_RESAMPLES", "CORRECTION_ROWS", "DEV_BOOTSTRAP_SEEDS",
    "DEV_GAMES", "DEV_SEED_START", "EPOCHS", "FINAL_BOOTSTRAP_SEED",
    "FINAL_GAMES", "FINAL_SEED_START", "GAME_SHARD_SIZE",
    "GAMES_PER_SHARD", "HALF_BATCH", "M5_ANCHOR_MANIFEST_SHA256",
    "M5_CHECKPOINT_SHA256", "M5_TAG_SHA", "PARAMETER_COUNT",
    "PROMPT_SHA256", "REQUIRED_SHARD_KEYS", "SHARD_STATES",
    "SOURCE_RETRY_MAX_INDEX", "SOURCE_RETRY_SEED_STRIDE",
    "STATES_PER_GAME", "TEACHER_SHA256", "TEACHER_VERSION",
    "TEST_SOURCE_GAMES", "TRAINING_SEEDS", "TRAIN_SOURCE_GAMES",
    "VALIDATION_BOOTSTRAP_SEED", "VALIDATION_GAMES",
    "VALIDATION_GAME_SEED_START", "VALIDATION_SOURCE_GAMES",
    "WORK_ORDER_VERSION", "assert_test_allowed", "atomic_save_npz",
    "atomic_torch_save", "atomic_write_json", "correction_anchor_batch",
    "d4_metrics", "epoch_plan", "fresh_finetune", "high_tile_metrics",
    "load_m5_model", "make_optimizer", "paired_bootstrap",
    "parameter_update_checks", "plan_id", "plan_sha256",
    "promotion_result", "random_legal_baseline", "rollout_student_games",
    "sample_positions", "score_summary", "select_candidate",
    "set_determinism", "sha256_file", "snapshot_parameters",
    "source_attempt_seed", "split_game_seed", "teacher_best_action",
    "teacher_metrics", "training_label_manifest_sha256",
    "validate_completed_shard",
]
