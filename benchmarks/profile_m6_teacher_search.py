"""M6 required three-repeat frozen Teacher Search re-profile."""
from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from game2048.m3_search import ExpectimaxTeacher, SearchStats
from game2048.m5_pretrain import validate_completed_shard as validate_m5_shard
from game2048.m6_state_correction import (
    M5_ANCHOR_MANIFEST_SHA256,
    M5_CHECKPOINT_SHA256,
    M5_PRETRAIN_TEACHER_SHA256,
    M5_TAG_SHA,
    PROMPT_SHA256,
    TEACHER_FILENAME,
    TEACHER_SHA256,
    WORK_ORDER_VERSION,
    atomic_write_json,
    load_m6_teacher,
    sha256_file,
)

ART = ROOT / "artifacts" / "m6"
SESSION = ART / "progress" / "session.json"
OUTPUT = ART / "search_profile.json"
STATES_PATH = ROOT / "artifacts" / "m3" / "semantic_profile_states.npz"
REFERENCE_PATH = ROOT / "artifacts" / "m3" / "search_ab_p6_cpp_values.npz"
M3_REPORT = ROOT / "reports" / "m3" / "m3_search_performance_unblock.json"
M5_CHECKPOINT = ROOT / "artifacts" / "m5" / "checkpoints" / "524k" / "20262103_final.pt"
TEACHER_PATH = ROOT / "teacher_checkpoints" / "m6" / TEACHER_FILENAME
M5_LABEL_MANIFEST = ROOT / "artifacts" / "m5" / "datasets" / "label_manifest.json"
STATES_SHA = "9AFD3A503819C8BD777C7F48D3CCDCA2B1C2F02A7D28DE7DF8997735EE68E512"
REFERENCE_SHA = "8E09F0D7774BCEC55C496A29550F2362FB83DEE142643F69472F12AE14963C51"
MIN_ROOTS_PER_SECOND = 8.803491691009711
REPEATS = 3
ROOTS = 256


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


def _verify_anchor() -> dict:
    if sha256_file(M5_CHECKPOINT) != M5_CHECKPOINT_SHA256:
        raise RuntimeError("M6_BLOCKED_M5_BASELINE_CHECKPOINT")
    if sha256_file(TEACHER_PATH) != TEACHER_SHA256:
        raise RuntimeError("M6_BLOCKED_TEACHER_CHECKPOINT")
    if sha256_file(M5_LABEL_MANIFEST) != M5_ANCHOR_MANIFEST_SHA256:
        raise RuntimeError("M6_BLOCKED_M5_ANCHOR_EVIDENCE")
    manifest = _read_json(M5_LABEL_MANIFEST)
    train = manifest.get("splits", {}).get("train", {}).get("shards", [])
    if (
        len(train) != 256
        or sum(int(row.get("rows", 0)) for row in train) != 524_288
        or manifest.get("teacher_sha256", "").upper() != M5_PRETRAIN_TEACHER_SHA256
        or manifest.get("value_semantics") != "SEARCH_VALUE_RAW_LEAF"
    ):
        raise RuntimeError("M6_BLOCKED_M5_ANCHOR_EVIDENCE")
    for row in train:
        if row.get("status") != "complete":
            raise RuntimeError("M6_BLOCKED_M5_ANCHOR_EVIDENCE")
        try:
            validate_m5_shard(ROOT / row["path"], row["sha256"])
        except RuntimeError as exc:
            raise RuntimeError("M6_BLOCKED_M5_ANCHOR_EVIDENCE") from exc
    return {
        "manifest_sha256": M5_ANCHOR_MANIFEST_SHA256,
        "train_shards": 256,
        "train_rows": 524_288,
        "m5_anchor_teacher_sha256": M5_PRETRAIN_TEACHER_SHA256,
        "m6_correction_teacher_sha256": TEACHER_SHA256,
        "value_semantics": "SEARCH_VALUE_RAW_LEAF",
        "all_train_shards_sha_valid": True,
    }


def _add_stats(total: SearchStats, item: SearchStats) -> None:
    for field in fields(SearchStats):
        setattr(
            total,
            field.name,
            getattr(total, field.name) + getattr(item, field.name),
        )


def _count_signature(data: dict) -> dict:
    keys = (
        "root_calls", "player_nodes", "chance_nodes", "leaf_calls",
        "move_calls", "chance_outcomes", "cache_lookups", "cache_hits",
    )
    return {key: int(data[key]) for key in keys}


def _run_repeat(
    repeat_index: int,
    teacher: TupleTeacher,
    states: np.ndarray,
) -> tuple[dict, np.ndarray]:
    aggregate = SearchStats()
    latencies: list[float] = []
    values: list[np.ndarray] = []
    wall_start = time.perf_counter()
    for index, state in enumerate(states):
        search = ExpectimaxTeacher(
            teacher, decision_depth=3, use_cache=True
        )
        started = time.perf_counter()
        root_values = search.action_values(state)
        latencies.append(time.perf_counter() - started)
        values.append(root_values)
        _add_stats(aggregate, search.stats)
        completed = index + 1
        if completed % 64 == 0:
            elapsed = time.perf_counter() - wall_start
            print(
                f"profile repeat={repeat_index + 1}/{REPEATS} "
                f"roots={completed}/{ROOTS} elapsed={elapsed:.2f}s "
                f"rate={completed / elapsed:.6f} roots/s",
                flush=True,
            )
    wall = time.perf_counter() - wall_start
    data = aggregate.to_dict()
    latency = np.asarray(latencies, dtype=np.float64)
    data.update({
        "repeat": repeat_index + 1,
        "roots": int(len(states)),
        "wall_seconds": float(wall),
        "roots_per_second": float(len(states) / wall),
        "per_root_p50_seconds": float(np.quantile(latency, 0.50)),
        "per_root_p95_seconds": float(np.quantile(latency, 0.95)),
        "per_root_max_seconds": float(latency.max()),
    })
    return data, np.stack(values)


def _correctness(
    candidate_values: np.ndarray,
    legacy_reference_values: np.ndarray,
    repeat: dict,
    expected_counts: dict,
) -> dict:
    # Teacher promotion changes values/best actions by design. The frozen M3
    # reference remains authoritative only for board legality and traversal shape.
    legal_equal = bool(np.array_equal(
        np.isfinite(candidate_values), np.isfinite(legacy_reference_values)
    ))
    counts = _count_signature(repeat)
    count_equal = counts == expected_counts
    return {
        "legacy_legal_mask_equal": legal_equal,
        "node_cache_counts_equal": count_equal,
        "observed_counts": counts,
        "expected_counts": expected_counts,
        "pass": bool(legal_equal and count_equal),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", required=True)
    parser.parse_args()
    session = _session()
    if session["state"] == "PROFILE_DONE":
        if not OUTPUT.exists():
            raise RuntimeError("M6_BLOCKED_RESUME_ARTIFACT_CORRUPT")
        print(json.dumps(_read_json(OUTPUT), indent=2), flush=True)
        return
    if session["state"] not in {"P1_READY", "PROFILE_RUNNING"}:
        raise RuntimeError("M6_BLOCKED_RESUME_GIT_STATE")

    anchor = _verify_anchor()
    if sha256_file(STATES_PATH) != STATES_SHA:
        raise RuntimeError("M6_BLOCKED_SEARCH_CORRECTNESS")
    if sha256_file(REFERENCE_PATH) != REFERENCE_SHA:
        raise RuntimeError("M6_BLOCKED_SEARCH_CORRECTNESS")
    with np.load(STATES_PATH, allow_pickle=False) as data:
        states = np.ascontiguousarray(data["states"], dtype=np.uint8)
    with np.load(REFERENCE_PATH, allow_pickle=False) as data:
        reference = np.ascontiguousarray(data["values"], dtype=np.float64)
    if states.shape != (ROOTS, 16) or reference.shape != (ROOTS, 4):
        raise RuntimeError("M6_BLOCKED_SEARCH_CORRECTNESS")
    m3 = _read_json(M3_REPORT)
    expected_counts = {
        key: int(value)
        for key, value in m3["p6_m2_cpp_move"]["node_counts"].items()
    }

    session["state"] = "PROFILE_RUNNING"
    atomic_write_json(SESSION, session)
    teacher = load_m6_teacher(
        TEACHER_PATH, verify_sha256=True, backend="cpp", cpp_prefetch=True
    )
    python_teacher = load_m6_teacher(
        TEACHER_PATH, verify_sha256=False, backend="python", cpp_prefetch=False
    )
    cpp_leaf = teacher.formal_state_leaf_batch(states)
    python_leaf = python_teacher.formal_state_leaf_batch(states)
    backend_equivalence = bool(np.array_equal(cpp_leaf, python_leaf))
    if not backend_equivalence:
        raise RuntimeError("M6_BLOCKED_SEARCH_CORRECTNESS")

    repeats = []
    correctness = []
    search_values = []
    for repeat_index in range(REPEATS):
        row, values = _run_repeat(repeat_index, teacher, states)
        check = _correctness(values, reference, row, expected_counts)
        if not check["pass"]:
            raise RuntimeError("M6_BLOCKED_SEARCH_CORRECTNESS")
        repeats.append(row)
        correctness.append(check)
        search_values.append(values)

    repeat_values_identical = all(
        np.array_equal(candidate, search_values[0], equal_nan=True)
        for candidate in search_values[1:]
    )
    if not repeat_values_identical:
        raise RuntimeError("M6_BLOCKED_SEARCH_CORRECTNESS")
    rates = np.asarray(
        [row["roots_per_second"] for row in repeats], dtype=np.float64
    )
    median_rate = float(np.median(rates))
    if median_rate < MIN_ROOTS_PER_SECOND:
        raise RuntimeError("M6_BLOCKED_SEARCH_PERFORMANCE_REGRESSION")
    orchestration = sum(
        float(row["recursive_python_orchestration_seconds"]) for row in repeats
    )
    total = sum(float(row["total_seconds"]) for row in repeats)
    orchestration_share = float(orchestration / total)
    if orchestration_share > 0.50:
        raise RuntimeError("M6_BLOCKED_SEARCH_PERFORMANCE_UNBLOCK_REQUIRED")

    projections = {
        "train_131072": float(131_072 / median_rate),
        "validation_8192": float(8_192 / median_rate),
        "test_8192": float(8_192 / median_rate),
    }
    projections["total_147456"] = float(sum(projections.values()))
    output = {
        "schema_version": 1,
        "work_order_version": WORK_ORDER_VERSION,
        "states_path": str(STATES_PATH.relative_to(ROOT)).replace("\\", "/"),
        "states_sha256": STATES_SHA,
        "reference_path": str(REFERENCE_PATH.relative_to(ROOT)).replace("\\", "/"),
        "reference_sha256": REFERENCE_SHA,
        "profile_repeat_count": REPEATS,
        "roots_per_repeat": ROOTS,
        "repeats": repeats,
        "correctness": correctness,
        "cpp_python_leaf_bit_identical": backend_equivalence,
        "search_repeat_values_bit_identical": repeat_values_identical,
        "median_roots_per_second": median_rate,
        "minimum_roots_per_second": MIN_ROOTS_PER_SECOND,
        "orchestration_share": orchestration_share,
        "projections_seconds": projections,
        "m5_anchor_reverification": anchor,
    }
    atomic_write_json(OUTPUT, output)
    session["state"] = "PROFILE_DONE"
    atomic_write_json(SESSION, session)
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
