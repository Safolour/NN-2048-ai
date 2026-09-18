"""Validate the real frozen M3 tuple adapter against the original executable."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from game2048.m3_tuple_teacher import DEFAULT_CHECKPOINT, EXPECTED_SHA256, TupleTeacher

UPSTREAM = Path(r"D:\CodexTasks\2048-ai")
AGENT = UPSTREAM / "build" / "2048_ai.exe"
SOURCE_FILES = [
    "src/ntuple/m6.cpp", "src/ntuple/m6.hpp", "src/ntuple/ntuple.cpp",
    "src/ntuple/ntuple.hpp", "src/agent/m6_agent.cpp",
    "build/ntuple_m6_tool.exe", "build/2048_ai.exe",
]

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def to_tile_values(board: np.ndarray) -> str:
    return ",".join("0" if int(e) == 0 else str(1 << int(e)) for e in board)

def upstream_values(board: np.ndarray) -> np.ndarray:
    proc = subprocess.run(
        [str(AGENT), "infer", "--agent", "m6", "--model", str(DEFAULT_CHECKPOINT),
         "--board", to_tile_values(board)],
        cwd=UPSTREAM, check=True, capture_output=True, text=True,
    )
    payload = json.loads(proc.stdout.strip())
    values = np.full(4, np.nan, dtype=np.float64)
    for item in payload["legal_actions"]:
        values[int(item["action_id"])] = float(item["value"])
    return values

def main() -> None:
    teacher = TupleTeacher()
    boards = [
        np.array([1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0], dtype=np.uint8),
        np.array([1,2,3,4,4,3,2,1,0,1,2,3,0,0,1,1], dtype=np.uint8),
        np.array([10,9,8,7,6,5,4,3,2,1,0,0,0,0,0,0], dtype=np.uint8),
        np.array([16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,0], dtype=np.uint8),
    ]
    rows = []
    max_abs = 0.0
    for board in boards:
        local1 = teacher.one_ply_action_values(board)
        local2 = teacher.one_ply_action_values(board)
        remote = upstream_values(board)
        diff = np.abs(local1 - remote)
        finite = np.isfinite(diff)
        max_abs = max(max_abs, float(diff[finite].max(initial=0.0)))
        rows.append({
            "board_exponents": board.tolist(),
            "local": [None if np.isnan(x) else float(x) for x in local1],
            "upstream": [None if np.isnan(x) else float(x) for x in remote],
            "deterministic": bool(np.allclose(local1, local2, equal_nan=True, rtol=0, atol=0)),
            "matches_upstream": bool(np.allclose(local1, remote, equal_nan=True, rtol=0, atol=1e-6)),
        })
    rng = np.random.default_rng(20260919)
    probes = []
    for _ in range(256):
        b = rng.integers(0, 15, size=16, dtype=np.uint8)
        probes.append(teacher.afterstate_value(b))
    report = {
        "result": "PASS" if all(r["deterministic"] and r["matches_upstream"] for r in rows) else "FAIL",
        "checkpoint": {
            "path": str(DEFAULT_CHECKPOINT.resolve()),
            "sha256": teacher.sha256,
            "expected_sha256": EXPECTED_SHA256,
            "size_bytes": DEFAULT_CHECKPOINT.stat().st_size,
            "metadata": teacher.metadata.__dict__,
        },
        "semantics": {
            "raw_tuple": "RAW_TUPLE_HEURISTIC",
            "formal_state_leaf": "FORMAL_STATE_TUPLE_HEURISTIC",
            "search_value": "SEARCH_VALUE_RAW_LEAF",
            "leaf_evaluator": "tuple_afterstate_greedy_1ply_adapter",
        },
        "differential": {"cases": rows, "max_abs_error": max_abs},
        "raw_value_probe": {
            "count": len(probes), "min": float(np.min(probes)), "max": float(np.max(probes)),
            "mean": float(np.mean(probes)), "std": float(np.std(probes)),
        },
        "upstream_provenance": {
            "git_commit": "172e7d1a1d501282899b8a5a2e251c174c43f98f",
            "git_dirty": True,
            "resolved_config_sha256": "6f4f6781eceeda496a3e600f754708ac644fc68be0057d4fe4bcb847ea0cc61f",
            "critical_sha256": {name: sha256(UPSTREAM / name) for name in SOURCE_FILES},
        },
    }
    (ROOT / "m3_teacher_semantics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
