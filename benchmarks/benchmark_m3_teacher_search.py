"""Re-profile the fixed M3 depth-3 search on the exact saved 256-state corpus."""
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
from game2048.m3_tuple_teacher import TupleTeacher, feature_indices
from game2048.reference_env import legal_mask

def add_stats(total, item):
    for field in fields(SearchStats):
        setattr(total, field.name, getattr(total, field.name) + getattr(item, field.name))

def main():
    corpus = np.load(ROOT / "artifacts" / "m3" / "semantic_profile_states.npz")
    states = np.ascontiguousarray(corpus["states"])
    game_ids = corpus["game_id"]
    teacher = TupleTeacher()
    warmup = ExpectimaxTeacher(teacher, decision_depth=3, use_cache=True)
    warmup.action_values(states[-1])
    aggregate = SearchStats()
    per_root = []
    wall_start = time.perf_counter()
    for index, state in enumerate(states):
        search = ExpectimaxTeacher(teacher, decision_depth=3, use_cache=True)
        started = time.perf_counter()
        search.action_values(state)
        elapsed = time.perf_counter() - started
        add_stats(aggregate, search.stats)
        per_root.append(elapsed)
        if (index + 1) % 16 == 0:
            print(f"optimized {index+1}/256 mean={np.mean(per_root):.3f}s p95={np.percentile(per_root,95):.3f}s", flush=True)
    wall = time.perf_counter() - wall_start
    data = aggregate.to_dict()
    data.update({
        "variant": "low-risk-leaf-batching",
        "decision_depth": 3,
        "states": int(len(states)),
        "games": int(len(set(int(x) for x in game_ids))),
        "corpus": "exact same saved corpus as pre-batch profile",
        "wall_seconds": wall,
        "root_decisions_per_second": len(states) / wall,
        "nodes_per_second": (aggregate.player_nodes + aggregate.chance_nodes + aggregate.leaf_calls) / wall,
        "per_root_mean_seconds": float(np.mean(per_root)),
        "per_root_median_seconds": float(np.median(per_root)),
        "per_root_p95_seconds": float(np.percentile(per_root,95)),
        "per_root_max_seconds": float(np.max(per_root)),
    })
    pre = json.loads((ROOT / "artifacts" / "m3" / "m3_teacher_profile_pre_batch.json").read_text())
    data["pre_batch_root_decisions_per_second"] = pre["root_decisions_per_second"]
    data["speedup_vs_pre_batch"] = data["root_decisions_per_second"] / pre["root_decisions_per_second"]
    (ROOT / "m3_teacher_profile.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(json.dumps(data, indent=2))

if __name__ == "__main__":
    main()
