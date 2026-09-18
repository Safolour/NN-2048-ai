"""M3 P6 reprofile after the C++ tuple evaluator."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from game2048.fast_env import move_batch as numpy_move_batch
from game2048.m2_fast_backend import move_selected_batch as cpp_move_batch
from game2048.m3_tuple_teacher import TupleTeacher

SIZES = [2, 4, 8, 9, 14, 16, 28, 32, 128]


def timed(fn):
    for _ in range(20):
        fn()
    count = 0
    start = time.perf_counter()
    elapsed = 0.0
    while count < 100 or elapsed < 2.0:
        fn()
        count += 1
        elapsed = time.perf_counter() - start
    return elapsed / count, count


def formal_leaf_with(teacher, boards, mover):
    repeated = np.repeat(np.ascontiguousarray(boards), 4, axis=0)
    actions = np.tile(np.arange(4, dtype=np.uint8), len(boards))
    moved = mover(repeated, actions)
    flat = np.full(len(repeated), np.nan, dtype=np.float64)
    legal = np.asarray(moved.moved, dtype=bool)
    if legal.any():
        future = teacher.afterstate_values(np.ascontiguousarray(moved.afterstates[legal]))
        flat[legal] = moved.rewards[legal].astype(np.float64) + future.astype(np.float64)
    matrix = flat.reshape(len(boards), 4)
    result = np.zeros(len(boards), dtype=np.float64)
    has_legal = np.isfinite(matrix).any(axis=1)
    result[has_legal] = np.nanmax(matrix[has_legal], axis=1)
    return result


def decompose(teacher, boards, mover, iterations=500):
    totals = {k: 0.0 for k in [
        "repeat_actions", "move", "legal_compaction", "tuple", "reward", "max"
    ]}
    for _ in range(iterations):
        t=time.perf_counter()
        repeated=np.repeat(np.ascontiguousarray(boards),4,axis=0)
        actions=np.tile(np.arange(4,dtype=np.uint8),len(boards))
        totals["repeat_actions"] += time.perf_counter()-t
        t=time.perf_counter()
        moved=mover(repeated,actions)
        totals["move"] += time.perf_counter()-t
        t=time.perf_counter()
        legal=np.asarray(moved.moved,dtype=bool)
        legal_after=np.ascontiguousarray(moved.afterstates[legal])
        legal_rewards=moved.rewards[legal].astype(np.float64)
        totals["legal_compaction"] += time.perf_counter()-t
        t=time.perf_counter()
        future=teacher.afterstate_values(legal_after) if len(legal_after) else np.empty(0,np.float32)
        totals["tuple"] += time.perf_counter()-t
        t=time.perf_counter()
        flat=np.full(len(repeated),np.nan,dtype=np.float64)
        flat[legal]=legal_rewards+future.astype(np.float64)
        matrix=flat.reshape(len(boards),4)
        totals["reward"] += time.perf_counter()-t
        t=time.perf_counter()
        has=np.isfinite(matrix).any(axis=1)
        out=np.zeros(len(boards),dtype=np.float64)
        out[has]=np.nanmax(matrix[has],axis=1)
        totals["max"] += time.perf_counter()-t
    return {k:v/iterations for k,v in totals.items()}


def main():
    teacher = TupleTeacher(backend="cpp", cpp_prefetch=True)
    rng = np.random.default_rng(20260919)
    report={"sizes":{}}
    for n in SIZES:
        boards=np.ascontiguousarray(rng.integers(0,15,size=(n,16),dtype=np.uint8))
        a=formal_leaf_with(teacher,boards,numpy_move_batch)
        b=formal_leaf_with(teacher,boards,cpp_move_batch)
        diff=np.abs(a-b)
        py_sec,_=timed(lambda: formal_leaf_with(teacher,boards,numpy_move_batch))
        cpp_sec,_=timed(lambda: formal_leaf_with(teacher,boards,cpp_move_batch))
        row={
            "max_abs_error":float(diff.max(initial=0.0)),
            "bit_identical":bool(np.array_equal(a,b)),
            "numpy_move_formal_us_per_board":py_sec*1e6/n,
            "m2_cpp_move_formal_us_per_board":cpp_sec*1e6/n,
            "formal_speedup":py_sec/cpp_sec,
            "numpy_components":decompose(teacher,boards,numpy_move_batch,100),
            "m2_cpp_components":decompose(teacher,boards,cpp_move_batch,100),
        }
        report["sizes"][str(n)]=row
        print(f"p6 batch={n} speedup={row['formal_speedup']:.3f} exact={row['bit_identical']}",flush=True)
    out=ROOT/"artifacts"/"m3"/"m3_p6_reprofile.json"
    out.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
