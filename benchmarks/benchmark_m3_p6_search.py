"""M3 P6 exact Search A/B after reusing frozen M2 C++ movement in formal leaves."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from benchmarks.benchmark_m3_search_unblock import run_variant, compare, count_signature
from game2048.m3_tuple_teacher import TupleTeacher


def main():
    corpus=np.load(ROOT/"artifacts"/"m3"/"semantic_profile_states.npz")
    states=np.ascontiguousarray(corpus["states"])
    ref=np.load(ROOT/"artifacts"/"m3"/"search_ab_python_values.npz")["values"]
    report_path=ROOT/"reports/m3/m3_search_performance_unblock.json"
    report=json.loads(report_path.read_text(encoding="utf-8"))

    data, values=run_variant(
        "p6_cpp_prefetch_m2_move",
        TupleTeacher(backend="cpp", cpp_prefetch=True),
        states,
    )
    diff=compare(ref, values)
    python_counts=report["differential"]["node_counts_python"]
    node_counts=count_signature(data)
    node_counts_equal=(python_counts==node_counts)
    p6={
        "variant":data,
        "differential":diff,
        "node_counts":node_counts,
        "node_counts_equal":node_counts_equal,
        "speedup_vs_historical":data["root_decisions_per_second"]/report["historical_baseline"]["root_decisions_per_second"],
        "speedup_vs_p5_cpp_prefetch":data["root_decisions_per_second"]/report["variants"]["cpp_prefetch"]["root_decisions_per_second"],
        "projected_8192_seconds":8192/data["root_decisions_per_second"],
    }
    report["p6_m2_cpp_move"]=p6
    report_path.write_text(json.dumps(report,indent=2),encoding="utf-8")
    np.savez_compressed(ROOT/"artifacts"/"m3"/"search_ab_p6_cpp_values.npz",values=values)
    print(json.dumps(p6,indent=2))


if __name__=="__main__":
    main()
