"""Python bridge for the M2 scalar C++ movement/legal backend."""
from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import sys
from pathlib import Path

import numpy as np

from .fast_env import BatchMoveResult

_ROOT = Path(__file__).resolve().parents[2]
_BUILD = _ROOT / "build" / "m2_fast_backend"


def _load_extension():
    try:
        return importlib.import_module("_m2_fast_backend")
    except ImportError:
        pass

    candidates: list[Path] = []
    if _BUILD.exists():
        for suffix in importlib.machinery.EXTENSION_SUFFIXES:
            candidates.extend(_BUILD.rglob(f"_m2_fast_backend*{suffix}"))
    for candidate in candidates:
        spec = importlib.util.spec_from_file_location("_m2_fast_backend", candidate)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules["_m2_fast_backend"] = module
        spec.loader.exec_module(module)
        return module

    raise ImportError(
        "M2 C++ backend is not built. Configure/build cpp/m2_fast_backend into "
        "build/m2_fast_backend before importing game2048.m2_fast_backend."
    )


_ext = _load_extension()


def _require_boards(boards: np.ndarray) -> np.ndarray:
    arr = np.asarray(boards)
    if arr.ndim != 2 or arr.shape[1] != 16:
        raise ValueError(f"boards must have shape (N, 16), got {arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError("boards must contain at least one board")
    if arr.dtype != np.uint8:
        raise ValueError(f"boards dtype must be uint8, got {arr.dtype}")
    if not arr.flags.c_contiguous:
        raise ValueError("boards must be C-contiguous at the M2 C++ boundary")
    return arr


def _require_actions(actions: np.ndarray, count: int) -> np.ndarray:
    arr = np.asarray(actions)
    if arr.ndim != 1 or arr.shape[0] != count:
        raise ValueError(f"actions must have shape ({count},), got {arr.shape}")
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.bool_) or not np.issubdtype(arr.dtype, np.integer):
            raise ValueError(f"actions dtype must be an integer type, got {arr.dtype}")
        if arr.size:
            low = int(arr.min())
            high = int(arr.max())
            if low < 0 or high > 3:
                raise ValueError(f"actions must lie in 0..3, got {low}..{high}")
        arr = arr.astype(np.uint8)
    elif arr.size and int(arr.max()) > 3:
        raise ValueError(f"actions must lie in 0..3, got max {int(arr.max())}")
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)
    return arr


def legal_mask_batch(boards: np.ndarray) -> np.ndarray:
    arr = _require_boards(boards)
    result = _ext.legal_mask_batch(arr)
    if result.dtype != np.bool_ or result.shape != (arr.shape[0], 4):
        raise RuntimeError("C++ backend returned an invalid legal-mask shape/dtype")
    return result


def next_legal_batch(boards: np.ndarray) -> np.ndarray:
    return legal_mask_batch(boards)


def move_selected_batch(boards: np.ndarray, actions: np.ndarray) -> BatchMoveResult:
    board_array = _require_boards(boards)
    action_array = _require_actions(actions, board_array.shape[0])
    afterstates, rewards, moved = _ext.move_selected_batch(board_array, action_array)
    return BatchMoveResult(
        afterstates=afterstates,
        rewards=rewards,
        moved=moved,
    )


def backend_info() -> dict:
    return {
        "kind": str(_ext.BACKEND_KIND),
        "simd_used": bool(_ext.SIMD_USED),
        "lut_used": bool(_ext.LUT_USED),
        "module_file": str(getattr(_ext, "__file__", "")),
    }


__all__ = [
    "backend_info",
    "legal_mask_batch",
    "move_selected_batch",
    "next_legal_batch",
]
