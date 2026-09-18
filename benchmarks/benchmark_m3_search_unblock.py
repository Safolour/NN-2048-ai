"""M3 P5 exact 256-root Search A/B for Python vs C++ tuple evaluators."""
from __future__ import annotations

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


def add_stats(total, item):
    for field in fields(SearchStats):
        setattr(total, field.name, getattr(total, field.name) + getattr(item, field.name))


def run_variant(name, teacher, states):
    aggregate = SearchStats()
    latencies = []
    values = []
    wall_start = time.perf_counter()
    progress_json = ROOT / "artifacts" / "m3" / f"search_ab_{name}_progress.json"
    progress_npz = ROOT / "artifacts" / "m3" / f"search_ab_{name}_values.npz"
    for i, state in enumerate(states):
        search = ExpectimaxTeacher(teacher, decision_depth=3, use_cache=True)
        started = time.perf_counter()
        value = search.action_values(state)
        latencies.append(time.perf_counter() - started)
        values.append(value)
        add_stats(aggregate, search.stats)
        if (i + 1) % 16 == 0:
            partial = aggregate.to_dict()
            partial.update({
                "variant": name,
                "completed_roots": i + 1,
                "total_roots": int(len(states)),
                "elapsed_seconds": time.perf_counter() - wall_start,
                "mean_seconds_per_root": float(np.mean(latencies)),
            })
            progress_json.write_text(json.dumps(partial, indent=2), encoding="utf-8")
            np.savez_compressed(progress_npz, values=np.stack(values))
            print(
                f"{name} {i+1}/{len(states)} mean={np.mean(latencies):.4f}s "
                f"rate={(i+1)/(time.perf_counter()-wall_start):.3f} roots/s",
                flush=True,
            )
    wall = time.perf_counter() - wall_start
    data = aggregate.to_dict()
    data.update({
        "variant": name,
        "states": int(len(states)),
        "wall_seconds": wall,
        "root_decisions_per_second": len(states) / wall,
        "nodes_per_second": (
            aggregate.player_nodes + aggregate.chance_nodes + aggregate.leaf_calls
        ) / wall,
        "per_root_mean_seconds": float(np.mean(latencies)),
        "per_root_median_seconds": float(np.median(latencies)),
        "per_root_p95_seconds": float(np.percentile(latencies, 95)),
        "per_root_max_seconds": float(np.max(latencies)),
    })
    return data, np.stack(values)


def compare(reference, candidate):
    legal_ref = np.isfinite(reference)
    legal_cand = np.isfinite(candidate)
    legal_equal = bool(np.array_equal(legal_ref, legal_cand))
    mask = legal_ref & legal_cand
    diff = np.abs(reference[mask] - candidate[mask])
    ref_best = np.nanargmax(reference, axis=1)
    cand_best = np.nanargmax(candidate, axis=1)
    return {
        "legal_mask_equal": legal_equal,
        "action_value_count": int(diff.size),
        "max_abs_error": float(diff.max(initial=0.0)),
        "mean_abs_error": float(diff.mean()) if diff.size else 0.0,
        "bit_identical": bool(np.array_equal(reference, candidate, equal_nan=True)),
        "best_action_disagreement": int(np.count_nonzero(ref_best != cand_best)),
    }


def count_signature(data):
    return {
        key: data[key] for key in [
            "root_calls", "player_nodes", "chance_nodes", "leaf_calls", "move_calls",
            "chance_outcomes", "cache_lookups", "cache_hits",
        ]
    }


def main():
    corpus = np.load(ROOT / "artifacts" / "m3" / "semantic_profile_states.npz")
    states = np.ascontiguousarray(corpus["states"])

    py_data, py_values = run_variant("python", TupleTeacher(backend="python"), states)
    cpp0_data, cpp0_values = run_variant(
        "cpp_no_prefetch", TupleTeacher(backend="cpp", cpp_prefetch=False), states
    )
    cpp1_data, cpp1_values = run_variant(
        "cpp_prefetch", TupleTeacher(backend="cpp", cpp_prefetch=True), states
    )


    report = {
        "historical_baseline": {
            "root_decisions_per_second": 1.1271717682426399,
            "wall_seconds": 227.11711490000016,
            "formal_leaf_evaluator_share": 0.866,
        },
        "variants": {
            "python": py_data,
            "cpp_no_prefetch": cpp0_data,
            "cpp_prefetch": cpp1_data,
        },
        "differential": {
            "cpp_no_prefetch": compare(py_values, cpp0_values),
            "cpp_prefetch": compare(py_values, cpp1_values),
            "node_counts_python": count_signature(py_data),
            "node_counts_cpp_no_prefetch": count_signature(cpp0_data),
            "node_counts_cpp_prefetch": count_signature(cpp1_data),
        },
    }
    faster = (
        "cpp_prefetch"
        if cpp1_data["root_decisions_per_second"] > cpp0_data["root_decisions_per_second"]
        else "cpp_no_prefetch"
    )
    report["selected_cpp_variant"] = faster
    selected = report["variants"][faster]
    report["final_root_decisions_per_second"] = selected["root_decisions_per_second"]
    report["speedup_vs_historical"] = (
        selected["root_decisions_per_second"]
        / report["historical_baseline"]["root_decisions_per_second"]
    )
    report["projected_8192_seconds"] = 8192 / selected["root_decisions_per_second"]
    out = ROOT / "m3_search_performance_unblock.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
