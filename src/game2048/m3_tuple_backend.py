"""Python bridge for the M3 exact U2048NT6 C++ tuple evaluator."""
from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
_BUILD = _ROOT / "build" / "m3_tuple_backend"


def _load_extension():
    try:
        return importlib.import_module("_m3_tuple_backend")
    except ImportError:
        pass
    candidates: list[Path] = []
    if _BUILD.exists():
        for suffix in importlib.machinery.EXTENSION_SUFFIXES:
            candidates.extend(_BUILD.rglob(f"_m3_tuple_backend*{suffix}"))
    for candidate in candidates:
        spec = importlib.util.spec_from_file_location("_m3_tuple_backend", candidate)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules["_m3_tuple_backend"] = module
        spec.loader.exec_module(module)
        return module
    raise ImportError("M3 tuple backend is not built under build/m3_tuple_backend")


_ext = _load_extension()


def _boards(boards: np.ndarray) -> np.ndarray:
    arr = np.asarray(boards)
    if arr.ndim != 2 or arr.shape[1] != 16:
        raise ValueError(f"boards must have shape (N, 16), got {arr.shape}")
    if arr.dtype != np.uint8:
        raise ValueError(f"boards dtype must be uint8, got {arr.dtype}")
    if not arr.flags.c_contiguous:
        raise ValueError("boards must be C-contiguous")
    return arr


def _weights(weights: np.ndarray) -> np.ndarray:
    arr = np.asarray(weights)
    if arr.ndim != 2:
        raise ValueError("weights must be a 2D stage-major array")
    if arr.dtype != np.float32:
        raise ValueError(f"weights dtype must be float32, got {arr.dtype}")
    if not arr.flags.c_contiguous:
        raise ValueError("weights must be C-contiguous")
    return arr


def _thresholds(stage_thresholds) -> np.ndarray:
    arr = np.asarray(stage_thresholds, dtype=np.uint64)
    if arr.ndim != 1 or not 1 <= arr.shape[0] <= 4:
        raise ValueError("stage_thresholds must have shape (1..4,)")
    return np.ascontiguousarray(arr)


def feature_indices(boards: np.ndarray) -> np.ndarray:
    return _ext.feature_indices(_boards(boards))


def stage_indices(boards: np.ndarray, stage_thresholds) -> np.ndarray:
    return _ext.stage_indices(_boards(boards), _thresholds(stage_thresholds))


def afterstate_values(
    boards: np.ndarray,
    weights: np.ndarray,
    stage_thresholds,
    *,
    prefetch: bool = True,
) -> np.ndarray:
    b = _boards(boards)
    w = _weights(weights)
    thresholds = _thresholds(stage_thresholds)
    if thresholds.shape[0] != w.shape[0]:
        raise ValueError("stage_thresholds count must match weights stage_count")
    result = _ext.afterstate_values(b, w, thresholds, bool(prefetch))
    if result.dtype != np.float32 or result.shape != (len(b),):
        raise RuntimeError("C++ tuple backend returned invalid output")
    return result


def backend_info() -> dict:
    return {
        "kind": "scalar-prefetch",
        "module_file": str(getattr(_ext, "__file__", "")),
        "feature_count_per_pattern": int(_ext.FEATURE_COUNT_PER_PATTERN),
        "weight_count_per_stage": int(_ext.WEIGHT_COUNT_PER_STAGE),
    }


__all__ = ["afterstate_values", "backend_info", "feature_indices", "stage_indices"]
