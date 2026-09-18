"""M3 P1 evaluator decomposition, batch microbench, and exact leaf diagnostics."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from game2048.fast_env import move_batch
from game2048.m3_search import ExpectimaxTeacher
from game2048.m3_tuple_teacher import (
    FEATURE_COUNT_PER_PATTERN, PATTERN_COUNT, SYMMETRY_COUNT,
    TupleTeacher, feature_indices_batch,
)

BATCH_AFTERSTATE = [1, 16, 32, 128, 512, 2048, 8192]
BATCH_FORMAL = [1, 8, 16, 32, 128, 512, 2048]


def rss_bytes():
    try:
        import psutil
        return int(psutil.Process().memory_info().rss)
    except Exception:
        return None


def timed(fn, *, warmup=20):
    for _ in range(warmup):
        fn()
    count = 0
    started = time.perf_counter()
    elapsed = 0.0
    while count < 100 or elapsed < 2.0:
        fn()
        count += 1
        elapsed = time.perf_counter() - started
    return {"iterations": count, "seconds": elapsed, "seconds_per_call": elapsed / count}


def make_boards(n, seed):
    rng = np.random.default_rng(seed)
    return np.ascontiguousarray(rng.integers(0, 18, size=(n, 16), dtype=np.uint8))


def microbench(teacher):
    result = {"afterstate_values": {}, "formal_state_leaf_batch": {}}
    for n in BATCH_AFTERSTATE:
        boards = make_boards(n, 1000 + n)
        stat = timed(lambda: teacher.afterstate_values(boards))
        stat["boards_per_second"] = n / stat["seconds_per_call"]
        stat["us_per_board"] = stat["seconds_per_call"] * 1e6 / n
        stat["rss_bytes"] = rss_bytes()
        result["afterstate_values"][str(n)] = stat
    for n in BATCH_FORMAL:
        boards = make_boards(n, 2000 + n)
        stat = timed(lambda: teacher.formal_state_leaf_batch(boards))
        stat["boards_per_second"] = n / stat["seconds_per_call"]
        stat["us_per_board"] = stat["seconds_per_call"] * 1e6 / n
        stat["rss_bytes"] = rss_bytes()
        result["formal_state_leaf_batch"][str(n)] = stat
    return result


def decompose(teacher, n=2048, iterations=100):
    boards = make_boards(n, 3001)
    bases = np.repeat(
        np.arange(PATTERN_COUNT, dtype=np.uint32) * FEATURE_COUNT_PER_PATTERN,
        SYMMETRY_COUNT,
    )
    totals = Counter()
    for _ in range(iterations):
        t = time.perf_counter()
        repeated = np.repeat(boards, 4, axis=0)
        actions = np.tile(np.arange(4, dtype=np.uint8), len(boards))
        moved = move_batch(repeated, actions)
        totals["move_batch"] += time.perf_counter() - t

        t = time.perf_counter()
        legal = np.asarray(moved.moved, dtype=bool)
        legal_after = np.ascontiguousarray(moved.afterstates[legal])
        legal_rewards = moved.rewards[legal].astype(np.float64)
        totals["legal_compaction"] += time.perf_counter() - t

        t = time.perf_counter()
        max_exp = legal_after.max(axis=1, initial=0).astype(np.uint16)
        stages = np.zeros(len(legal_after), dtype=np.intp)
        for i, threshold in enumerate(teacher.metadata.stage_thresholds):
            if i == 0:
                continue
            threshold_exp = int(threshold).bit_length() - 1
            stages[max_exp >= threshold_exp] = i
        totals["stage_selection"] += time.perf_counter() - t

        t = time.perf_counter()
        features = feature_indices_batch(legal_after)
        absolute = bases.reshape(1, 64) + features
        totals["feature_packing"] += time.perf_counter() - t

        t = time.perf_counter()
        gathered = teacher._weights[stages[:, None], absolute]
        totals["weight_lookup"] += time.perf_counter() - t

        t = time.perf_counter()
        future = gathered.astype(np.float64).sum(axis=1).astype(np.float32)
        totals["weight_accumulation"] += time.perf_counter() - t

        t = time.perf_counter()
        flat = np.full(len(repeated), np.nan, dtype=np.float64)
        flat[legal] = legal_rewards + future.astype(np.float64)
        matrix = flat.reshape(len(boards), 4)
        totals["reward_continuation"] += time.perf_counter() - t

        t = time.perf_counter()
        has_legal = np.isfinite(matrix).any(axis=1)
        out = np.zeros(len(boards), dtype=np.float64)
        out[has_legal] = np.nanmax(matrix[has_legal], axis=1)
        totals["max_legal"] += time.perf_counter() - t
        if not np.isfinite(out).all():
            raise RuntimeError("decomposition produced non-finite output")

    known = sum(totals.values())
    total_started = time.perf_counter()
    for _ in range(iterations):
        teacher.formal_state_leaf_batch(boards)
    formal_total = time.perf_counter() - total_started
    result = {k: float(v) for k, v in totals.items()}
    result["formal_total_reference"] = float(formal_total)
    result["python_call_allocation_residual"] = max(0.0, float(formal_total - known))
    result["boards"] = n
    result["iterations"] = iterations
    return result


class RecordingEvaluator:
    def __init__(self, teacher):
        self.teacher = teacher
        self.batch_sizes = []
        self.total_leaf = 0
        self.global_unique = set()
        self.root_seen = None
        self.within_root_duplicates = 0

    def begin_root(self):
        self.root_seen = set()

    def formal_state_leaf(self, state):
        return self.formal_state_leaf_batch(np.asarray(state).reshape(1, 16))[0]

    def formal_state_leaf_batch(self, states):
        arr = np.ascontiguousarray(states, dtype=np.uint8)
        self.batch_sizes.append(len(arr))
        self.total_leaf += len(arr)
        for row in arr:
            key = row.tobytes()
            self.global_unique.add(key)
            if self.root_seen is not None:
                if key in self.root_seen:
                    self.within_root_duplicates += 1
                else:
                    self.root_seen.add(key)
        return self.teacher.formal_state_leaf_batch(arr)


def bucket(size):
    if size == 1: return "1"
    if size <= 4: return "2-4"
    if size <= 8: return "5-8"
    if size <= 16: return "9-16"
    if size <= 32: return "17-32"
    if size <= 64: return "33-64"
    return ">64"


def leaf_diagnostics(teacher):
    corpus = np.load(ROOT / "artifacts" / "m3" / "semantic_profile_states.npz")
    states = np.ascontiguousarray(corpus["states"])
    rec = RecordingEvaluator(teacher)
    per_root_unique = []
    started = time.perf_counter()
    for i, state in enumerate(states):
        rec.begin_root()
        search = ExpectimaxTeacher(rec, decision_depth=3, use_cache=True)
        search.action_values(state)
        per_root_unique.append(len(rec.root_seen or ()))
        if (i + 1) % 16 == 0:
            print(f"leafdiag {i+1}/{len(states)}", flush=True)
    elapsed = time.perf_counter() - started
    sizes = np.asarray(rec.batch_sizes, dtype=np.int64)
    hist = Counter(bucket(int(x)) for x in sizes)
    total = rec.total_leaf
    unique_global = len(rec.global_unique)
    return {
        "roots": int(len(states)),
        "wall_seconds": elapsed,
        "batch_calls": int(len(sizes)),
        "batch_histogram": dict(hist),
        "batch_mean": float(sizes.mean()),
        "batch_median": float(np.median(sizes)),
        "batch_p90": float(np.percentile(sizes, 90)),
        "batch_p99": float(np.percentile(sizes, 99)),
        "batch_max": int(sizes.max()),
        "total_leaf_states": int(total),
        "unique_leaf_states": int(unique_global),
        "within_root_duplicate_count": int(rec.within_root_duplicates),
        "within_root_duplicate_ratio": float(rec.within_root_duplicates / total),
        "cross_root_or_global_duplicate_count": int(total - unique_global),
        "cross_root_or_global_duplicate_ratio": float((total - unique_global) / total),
        "per_root_unique_mean": float(np.mean(per_root_unique)),
    }


def main():
    teacher = TupleTeacher()
    report = {
        "microbench": microbench(teacher),
        "decomposition": decompose(teacher),
        "leaf_diagnostics": leaf_diagnostics(teacher),
    }
    out = ROOT / "artifacts" / "m3" / "m3_tuple_evaluator_profile.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
