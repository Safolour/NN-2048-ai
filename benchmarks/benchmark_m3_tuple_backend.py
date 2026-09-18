"""M3 P3/P4 real-checkpoint differential and primitive A/B."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from game2048.m3_tuple_backend import feature_indices as cpp_feature_indices
from game2048.m3_tuple_teacher import TupleTeacher, feature_indices_batch
from game2048.symmetry import transform_board

UPSTREAM = Path(r"D:\CodexTasks\2048-ai")
UPSTREAM_EXE = UPSTREAM / "build" / "2048_ai.exe"
BATCHES = [1, 16, 32, 128, 512, 2048, 8192]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rss_bytes():
    try:
        import psutil
        return int(psutil.Process().memory_info().rss)
    except Exception:
        return None


def timed(fn):
    start = time.perf_counter()
    fn()
    first = time.perf_counter() - start
    for _ in range(20):
        fn()
    count = 0
    started = time.perf_counter()
    elapsed = 0.0
    while count < 100 or elapsed < 2.0:
        fn()
        count += 1
        elapsed = time.perf_counter() - started
    return {
        "first_call_seconds": first,
        "iterations": count,
        "seconds": elapsed,
        "seconds_per_call": elapsed / count,
    }


def realistic_boards(count, seed):
    rng = np.random.default_rng(seed)
    return np.ascontiguousarray(rng.integers(0, 19, size=(count, 16), dtype=np.uint8))


def differential_boards():
    rng = np.random.default_rng(20260919)
    random = rng.integers(0, 19, size=(10000, 16), dtype=np.uint8)
    edges = [np.zeros(16, dtype=np.uint8)]
    single = np.zeros(16, dtype=np.uint8)
    single[5] = 1
    edges.append(single)
    edges.append(np.arange(16, dtype=np.uint8))
    for exp in [16, 17, 20, 31, 63, 64, 127, 255]:
        board = np.arange(16, dtype=np.uint8) % 6
        board[0] = exp
        edges.append(board)
    base = np.array([1,2,3,4,4,3,2,1,0,1,2,3,0,0,1,1], dtype=np.uint8)
    for transform_id in range(8):
        edges.append(transform_board(base, transform_id))
    return np.ascontiguousarray(np.vstack([random, edges]))


def to_tile_values(board):
    return ",".join("0" if int(e) == 0 else str(1 << int(e)) for e in board)


def upstream_values(board):
    proc = subprocess.run(
        [str(UPSTREAM_EXE), "infer", "--agent", "m6",
         "--model", str(ROOT / "teacher_checkpoints" / "m3" /
                        "ordinary_td_comparator_ep4800000_7192719323a0.bin"),
         "--board", to_tile_values(board)],
        cwd=UPSTREAM, check=True, capture_output=True, text=True,
    )
    payload = json.loads(proc.stdout.strip())
    values = np.full(4, np.nan, dtype=np.float64)
    for item in payload["legal_actions"]:
        values[int(item["action_id"])] = float(item["value"])
    return values


def compare_arrays(left, right):
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    diff = np.abs(a[mask] - b[mask])
    return {
        "count": int(diff.size),
        "max_abs_error": float(diff.max(initial=0.0)),
        "mean_abs_error": float(diff.mean()) if diff.size else 0.0,
        "exact": bool(np.array_equal(left, right, equal_nan=True)),
    }


def main():
    py_teacher = TupleTeacher(backend="python")
    cpp_no = TupleTeacher(backend="cpp", cpp_prefetch=False)
    cpp_pre = TupleTeacher(backend="cpp", cpp_prefetch=True)

    boards = differential_boards()
    py_features = feature_indices_batch(boards)
    cpp_features = cpp_feature_indices(boards)
    py_values = py_teacher.afterstate_values(boards)
    cpp_no_values = cpp_no.afterstate_values(boards)
    cpp_pre_values = cpp_pre.afterstate_values(boards)

    differential = {
        "boards": int(len(boards)),
        "feature_indices_exact": bool(np.array_equal(py_features, cpp_features)),
        "python_vs_cpp_no_prefetch": compare_arrays(py_values, cpp_no_values),
        "python_vs_cpp_prefetch": compare_arrays(py_values, cpp_pre_values),
        "cpp_no_vs_prefetch": compare_arrays(cpp_no_values, cpp_pre_values),
    }

    corpus = np.load(ROOT / "artifacts" / "m3" / "semantic_profile_states.npz")
    roots = np.ascontiguousarray(corpus["states"])
    leaf_py = py_teacher.formal_state_leaf_batch(roots)
    leaf_cpp = cpp_pre.formal_state_leaf_batch(roots)
    differential["formal_leaf_256"] = compare_arrays(leaf_py, leaf_cpp)

    consistency_source = realistic_boards(2048, 20260920)
    whole = cpp_pre.afterstate_values(consistency_source)
    consistency = {}
    for chunk in [1, 16, 128, 2048]:
        pieces = [
            cpp_pre.afterstate_values(consistency_source[i:i + chunk])
            for i in range(0, len(consistency_source), chunk)
        ]
        joined = np.concatenate(pieces)
        consistency[str(chunk)] = compare_arrays(whole, joined)


    anchor_rows = []
    anchor_max = 0.0
    anchor_disagreements = 0
    for i, board in enumerate(roots[:256]):
        local = cpp_pre.one_ply_action_values(board)
        upstream = upstream_values(board)
        finite = np.isfinite(local) & np.isfinite(upstream)
        diff = np.abs(local[finite] - upstream[finite])
        anchor_max = max(anchor_max, float(diff.max(initial=0.0)))
        if np.isfinite(local).any() and int(np.nanargmax(local)) != int(np.nanargmax(upstream)):
            anchor_disagreements += 1
        anchor_rows.append({
            "index": i,
            "max_abs_error": float(diff.max(initial=0.0)),
            "local_best": None if not np.isfinite(local).any() else int(np.nanargmax(local)),
            "upstream_best": None if not np.isfinite(upstream).any() else int(np.nanargmax(upstream)),
        })

    primitive = {}
    rng = np.random.default_rng(20260921)
    for n in BATCHES:
        batch = np.ascontiguousarray(rng.integers(0, 19, size=(n, 16), dtype=np.uint8))
        row = {}
        for name, teacher in [
            ("python", py_teacher),
            ("cpp_no_prefetch", cpp_no),
            ("cpp_prefetch", cpp_pre),
        ]:
            stat = timed(lambda teacher=teacher, batch=batch: teacher.afterstate_values(batch))
            stat["boards_per_second"] = n / stat["seconds_per_call"]
            stat["us_per_board"] = stat["seconds_per_call"] * 1e6 / n
            stat["rss_bytes"] = rss_bytes()
            row[name] = stat
        row["speedup_cpp_no_vs_python"] = (
            row["cpp_no_prefetch"]["boards_per_second"] / row["python"]["boards_per_second"]
        )
        row["speedup_cpp_prefetch_vs_python"] = (
            row["cpp_prefetch"]["boards_per_second"] / row["python"]["boards_per_second"]
        )
        row["prefetch_gain"] = (
            row["cpp_prefetch"]["boards_per_second"] / row["cpp_no_prefetch"]["boards_per_second"]
        )
        primitive[str(n)] = row
        print("batch", n, json.dumps(row), flush=True)


    empty = cpp_pre.afterstate_values(np.empty((0, 16), dtype=np.uint8))
    one = realistic_boards(1, 99)
    repeated_a = cpp_pre.afterstate_values(one)
    repeated_b = cpp_pre.afterstate_values(one)
    owner_a = TupleTeacher(backend="cpp").afterstate_values(one)
    owner_b = TupleTeacher(backend="cpp").afterstate_values(one)
    safety = {
        "n1_ok": bool(repeated_a.shape == (1,)),
        "empty_ok": bool(empty.shape == (0,)),
        "readonly_weights": bool(not py_teacher._weights.flags.writeable),
        "repeated_calls_exact": bool(np.array_equal(repeated_a, repeated_b)),
        "independent_memmap_owner_exact": bool(np.array_equal(owner_a, owner_b)),
    }

    report = {
        "upstream_reference": {
            "m6_cpp_sha256": sha256(UPSTREAM / "src" / "ntuple" / "m6.cpp"),
            "m6_hpp_sha256": sha256(UPSTREAM / "src" / "ntuple" / "m6.hpp"),
        },
        "differential": differential,
        "batch_consistency": consistency,
        "upstream_anchor": {
            "boards": 256,
            "max_abs_error": anchor_max,
            "action_disagreement": anchor_disagreements,
            "cases": anchor_rows,
        },
        "memory_safety": safety,
        "primitive": primitive,
    }
    out = ROOT / "reports/m3/m3_tuple_backend_benchmark.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "upstream_anchor"}, indent=2))


if __name__ == "__main__":
    main()
