"""M5 required three-repeat Teacher Search re-profile."""
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
from game2048.m3_tuple_teacher import TupleTeacher
from game2048.m5_pretrain import (
    M4_PRIMARY_SHA256,
    PROMPT_SHA256,
    TEACHER_SHA256,
    WORK_ORDER_VERSION,
    atomic_write_json,
    sha256_file,
)

ART = ROOT / "artifacts" / "m5"
SESSION = ART / "progress" / "session.json"
OUTPUT = ART / "search_profile.json"
STATES_PATH = ROOT / "artifacts" / "m3" / "semantic_profile_states.npz"
REFERENCE_PATH = ROOT / "artifacts" / "m3" / "search_ab_p6_cpp_values.npz"
M3_REPORT = ROOT / "reports" / "m3" / "m3_search_performance_unblock.json"
STATES_SHA = (
    "9AFD3A503819C8BD777C7F48D3CCDCA2B1C2F02A7D28DE7DF8997735EE68E512"
)
REFERENCE_SHA = (
    "8E09F0D7774BCEC55C496A29550F2362FB83DEE142643F69472F12AE14963C51"
)
MIN_ROOTS_PER_SECOND = 5.688376766
REPEATS = 3
ROOTS = 256


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


def _add_stats(
    total: SearchStats,
    item: SearchStats,
) -> None:
    for field in fields(SearchStats):
        setattr(
            total,
            field.name,
            getattr(total, field.name)
            + getattr(item, field.name),
        )


def _count_signature(data: dict) -> dict:
    keys = (
        "root_calls",
        "player_nodes",
        "chance_nodes",
        "leaf_calls",
        "move_calls",
        "chance_outcomes",
        "cache_lookups",
        "cache_hits",
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
            teacher,
            decision_depth=3,
            use_cache=True,
        )
        started = time.perf_counter()
        root_values = search.action_values(state)
        latency = time.perf_counter() - started
        values.append(root_values)
        latencies.append(latency)
        _add_stats(aggregate, search.stats)
        completed = index + 1
        if completed % 64 == 0:
            elapsed = time.perf_counter() - wall_start
            print(
                f"profile repeat={repeat_index + 1}/{REPEATS} "
                f"roots={completed}/{ROOTS} "
                f"elapsed={elapsed:.2f}s "
                f"rate={completed / elapsed:.6f} roots/s",
                flush=True,
            )
    wall = time.perf_counter() - wall_start
    data = aggregate.to_dict()
    latency_array = np.asarray(
        latencies,
        dtype=np.float64,
    )
    data.update({
        "repeat": repeat_index + 1,
        "roots": int(len(states)),
        "wall_seconds": float(wall),
        "roots_per_second": float(len(states) / wall),
        "per_root_p50_seconds": float(
            np.quantile(latency_array, 0.50)
        ),
        "per_root_p95_seconds": float(
            np.quantile(latency_array, 0.95)
        ),
        "per_root_max_seconds": float(
            latency_array.max()
        ),
    })
    return data, np.stack(values)


def _correctness(
    candidate_values: np.ndarray,
    reference_values: np.ndarray,
    repeat: dict,
    expected_counts: dict,
) -> dict:
    legal_equal = bool(
        np.array_equal(
            np.isfinite(candidate_values),
            np.isfinite(reference_values),
        )
    )
    value_equal = bool(
        np.array_equal(
            candidate_values,
            reference_values,
            equal_nan=True,
        )
    )
    best_equal = bool(
        np.array_equal(
            np.nanargmax(candidate_values, axis=1),
            np.nanargmax(reference_values, axis=1),
        )
    )
    counts = _count_signature(repeat)
    count_equal = counts == expected_counts
    passed = (
        legal_equal
        and value_equal
        and best_equal
        and count_equal
    )
    return {
        "legal_mask_equal": legal_equal,
        "action_values_bit_identical": value_equal,
        "best_action_equal": best_equal,
        "node_cache_counts_equal": count_equal,
        "observed_counts": counts,
        "expected_counts": expected_counts,
        "pass": bool(passed),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        action="store_true",
        required=True,
    )
    parser.parse_args()
    session = _session()
    if session["state"] == "PROFILE_DONE":
        if not OUTPUT.exists():
            raise RuntimeError(
                "M5_BLOCKED_RESUME_ARTIFACT_CORRUPT"
            )
        print(json.dumps(_read_json(OUTPUT), indent=2))
        return
    if session["state"] not in {
        "P1_READY",
        "PROFILE_RUNNING",
    }:
        raise RuntimeError("M5_BLOCKED_RESUME_GIT_STATE")
    if sha256_file(STATES_PATH) != STATES_SHA:
        raise RuntimeError(
            "M5_BLOCKED_SEARCH_CORRECTNESS"
        )
    if sha256_file(REFERENCE_PATH) != REFERENCE_SHA:
        raise RuntimeError(
            "M5_BLOCKED_SEARCH_CORRECTNESS"
        )
    with np.load(
        STATES_PATH,
        allow_pickle=False,
    ) as source:
        states = np.ascontiguousarray(
            source["states"],
            dtype=np.uint8,
        )
    with np.load(
        REFERENCE_PATH,
        allow_pickle=False,
    ) as source:
        reference = np.ascontiguousarray(
            source["values"],
            dtype=np.float64,
        )
    if (
        states.shape != (ROOTS, 16)
        or reference.shape != (ROOTS, 4)
    ):
        raise RuntimeError(
            "M5_BLOCKED_SEARCH_CORRECTNESS"
        )
    m3 = _read_json(M3_REPORT)
    expected_counts = {
        key: int(value)
        for key, value in m3[
            "p6_m2_cpp_move"
        ]["node_counts"].items()
    }
    session["state"] = "PROFILE_RUNNING"
    atomic_write_json(SESSION, session)
    teacher = TupleTeacher(
        verify_sha256=True,
        backend="cpp",
        cpp_prefetch=True,
    )
    repeats = []
    correctness = []
    for repeat_index in range(REPEATS):
        row, values = _run_repeat(
            repeat_index,
            teacher,
            states,
        )
        check = _correctness(
            values,
            reference,
            row,
            expected_counts,
        )
        if not check["pass"]:
            raise RuntimeError(
                "M5_BLOCKED_SEARCH_CORRECTNESS"
            )
        repeats.append(row)
        correctness.append(check)
    throughputs = np.asarray(
        [row["roots_per_second"] for row in repeats],
        dtype=np.float64,
    )
    median_rate = float(np.median(throughputs))
    if median_rate < MIN_ROOTS_PER_SECOND:
        raise RuntimeError(
            "M5_BLOCKED_SEARCH_PERFORMANCE_REGRESSION"
        )
    orchestration = sum(
        float(row["recursive_python_orchestration_seconds"])
        for row in repeats
    )
    total = sum(
        float(row["total_seconds"])
        for row in repeats
    )
    orchestration_share = float(
        orchestration / total
    )
    if orchestration_share > 0.50:
        raise RuntimeError(
            "M5_BLOCKED_SEARCH_PERFORMANCE_UNBLOCK_REQUIRED"
        )
    output = {
        "schema_version": 1,
        "work_order_version": WORK_ORDER_VERSION,
        "states_path": str(
            STATES_PATH.relative_to(ROOT)
        ),
        "states_sha256": STATES_SHA,
        "reference_path": str(
            REFERENCE_PATH.relative_to(ROOT)
        ),
        "reference_sha256": REFERENCE_SHA,
        "profile_repeat_count": REPEATS,
        "roots_per_repeat": ROOTS,
        "repeats": repeats,
        "correctness": correctness,
        "median_roots_per_second": median_rate,
        "minimum_roots_per_second": (
            MIN_ROOTS_PER_SECOND
        ),
        "orchestration_share": orchestration_share,
        "projections_seconds": {
            "32768": float(32768 / median_rate),
            "131072": float(131072 / median_rate),
            "524288": float(524288 / median_rate),
        },
    }
    atomic_write_json(OUTPUT, output)
    session["state"] = "PROFILE_DONE"
    atomic_write_json(SESSION, session)
    print(
        json.dumps(output, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
