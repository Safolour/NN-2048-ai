"""M3 read-only adapter for the frozen U2048NT6 Teacher checkpoint."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import struct
from typing import Final

import numpy as np

from .reference_env import ACTIONS, move_without_spawn

REPO_ROOT: Final = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT: Final = REPO_ROOT / "teacher_checkpoints" / "m3" / "ordinary_td_comparator_ep4800000_7192719323a0.bin"
EXPECTED_SHA256: Final = "7192719323a073ba2b6b19b62cb7d46ef4aa90ecc8c4ae6baf27ad0c51566a84"
MAGIC: Final = b"U2048NT6"
HEADER_BYTES: Final = 256
PATTERN_COUNT: Final = 8
TUPLE_LENGTH: Final = 6
SYMMETRY_COUNT: Final = 8
FEATURE_COUNT_PER_PATTERN: Final = 1 << 24
WEIGHT_COUNT_PER_STAGE: Final = PATTERN_COUNT * FEATURE_COUNT_PER_PATTERN
PATTERNS: Final = np.array([
    [0, 1, 2, 4, 5, 6],
    [4, 5, 6, 7, 8, 9],
    [0, 1, 2, 3, 4, 5],
    [2, 3, 4, 5, 6, 9],
    [0, 1, 2, 5, 9, 10],
    [3, 4, 5, 6, 7, 8],
    [1, 3, 4, 5, 6, 7],
    [0, 1, 4, 8, 9, 10],
], dtype=np.uint8)

RAW_TUPLE_HEURISTIC: Final = "RAW_TUPLE_HEURISTIC"
FORMAL_STATE_TUPLE_HEURISTIC: Final = "FORMAL_STATE_TUPLE_HEURISTIC"

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

def fnv1a64(data: bytes) -> int:
    value = 14695981039346656037
    for byte in data:
        value ^= byte
        value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return value

@dataclass(frozen=True)
class CheckpointMetadata:
    format_version: int
    stage_count: int
    stage_thresholds: tuple[int, ...]
    stage_policy_version: int
    rule: int
    phase: int
    alpha_normalizer: int
    global_episodes_completed: int
    otd_episodes_completed: int
    tc_episodes_completed: int
    otd_episode_budget: int
    tc_episode_budget: int
    training_seed: int
    weight_checksum_fnv1a64: int
    coherence_checksum_fnv1a64: int
    has_coherence_stats: bool
    patterns: tuple[tuple[int, ...], ...]

def parse_checkpoint_header(header: bytes) -> CheckpointMetadata:
    if len(header) != HEADER_BYTES:
        raise ValueError(f"checkpoint header must be {HEADER_BYTES} bytes")
    if header[:8] != MAGIC:
        raise ValueError("invalid U2048NT6 checkpoint magic")
    stored = struct.unpack_from("<Q", header, 248)[0]
    if fnv1a64(header[:248]) != stored:
        raise ValueError("checkpoint header FNV-1a checksum mismatch")
    version, endian, pc, tl, sc, alphabet = struct.unpack_from("<6I", header, 8)
    feature_count, weight_count = struct.unpack_from("<2Q", header, 32)
    if (version, endian, pc, tl, sc, alphabet) != (2, 0x01020304, 8, 6, 8, 4):
        raise ValueError("unsupported U2048NT6 schema constants")
    if feature_count != FEATURE_COUNT_PER_PATTERN or weight_count != WEIGHT_COUNT_PER_STAGE:
        raise ValueError("unsupported U2048NT6 feature/weight counts")
    stage_count, policy_version = struct.unpack_from("<2I", header, 48)
    if not 1 <= stage_count <= 4:
        raise ValueError("invalid stage count")
    thresholds4 = struct.unpack_from("<4Q", header, 56)
    rule, phase, normalizer, _optimistic = struct.unpack_from("<4I", header, 88)
    coherence_flag = struct.unpack_from("<I", header, 124)[0]
    counters = struct.unpack_from("<6Q", header, 128)
    weight_checksum, coherence_checksum = struct.unpack_from("<2Q", header, 176)
    stored_patterns = tuple(tuple(header[200 + p * 6:206 + p * 6]) for p in range(8))
    expected_patterns = tuple(tuple(int(x) for x in row) for row in PATTERNS)
    if stored_patterns != expected_patterns:
        raise ValueError("checkpoint tuple pattern schema mismatch")
    return CheckpointMetadata(
        format_version=version,
        stage_count=stage_count,
        stage_thresholds=tuple(int(x) for x in thresholds4[:stage_count]),
        stage_policy_version=policy_version,
        rule=rule,
        phase=phase,
        alpha_normalizer=normalizer,
        global_episodes_completed=counters[0],
        otd_episodes_completed=counters[1],
        tc_episodes_completed=counters[2],
        otd_episode_budget=counters[3],
        tc_episode_budget=counters[4],
        training_seed=counters[5],
        weight_checksum_fnv1a64=weight_checksum,
        coherence_checksum_fnv1a64=coherence_checksum,
        has_coherence_stats=bool(coherence_flag),
        patterns=stored_patterns,
    )

def _transform_coord(row: int, col: int, transform_id: int) -> tuple[int, int]:
    if not 0 <= transform_id < 8:
        raise ValueError("D4 transform id must be 0..7")
    if transform_id >= 4:
        col = 3 - col
        transform_id -= 4
    for _ in range(transform_id):
        row, col = col, 3 - row
    return row, col

def _build_source_cells() -> np.ndarray:
    result = np.empty((8, 8, 6), dtype=np.uint8)
    for symmetry in range(8):
        source_for_target = np.empty(16, dtype=np.uint8)
        for row in range(4):
            for col in range(4):
                tr, tc = _transform_coord(row, col, symmetry)
                source_for_target[tr * 4 + tc] = row * 4 + col
        for pattern_id in range(8):
            result[pattern_id, symmetry] = source_for_target[PATTERNS[pattern_id]]
    return result

_SOURCE_CELLS: Final = _build_source_cells()

def feature_indices(board: np.ndarray) -> np.ndarray:
    arr = np.asarray(board)
    if arr.shape != (16,) or arr.dtype != np.uint8:
        raise ValueError("board must be uint8 with shape (16,)")
    return feature_indices_batch(arr.reshape(1, 16))[0]

def feature_indices_batch(boards: np.ndarray) -> np.ndarray:
    arr = np.asarray(boards)
    if arr.ndim != 2 or arr.shape[1] != 16 or arr.dtype != np.uint8:
        raise ValueError("boards must be uint8 with shape (N, 16)")
    source = np.minimum(arr, np.uint8(15)).astype(np.uint32, copy=False)
    values = source[:, _SOURCE_CELLS.reshape(-1)].reshape(-1, 8, 8, 6)
    shifts = (np.arange(6, dtype=np.uint32) * 4).reshape(1, 1, 1, 6)
    return np.bitwise_or.reduce(values << shifts, axis=3).reshape(-1, 64)

class TupleTeacher:
    """Deterministic read-only mmap evaluator for one frozen checkpoint."""

    def __init__(self, path: Path | str = DEFAULT_CHECKPOINT, *, verify_sha256: bool = True) -> None:
        self.path = Path(path).resolve()
        with self.path.open("rb") as handle:
            header = handle.read(HEADER_BYTES)
        self.metadata = parse_checkpoint_header(header)
        if self.metadata.has_coherence_stats:
            raise ValueError("M3 baseline adapter currently accepts the frozen no-TC weight plane only")
        expected_size = HEADER_BYTES + self.metadata.stage_count * WEIGHT_COUNT_PER_STAGE * 4
        if self.path.stat().st_size != expected_size:
            raise ValueError(f"checkpoint size mismatch: {self.path.stat().st_size} != {expected_size}")
        self.sha256 = sha256_file(self.path) if verify_sha256 else None
        if verify_sha256 and self.path == DEFAULT_CHECKPOINT.resolve() and self.sha256 != EXPECTED_SHA256:
            raise ValueError("M3 frozen checkpoint SHA-256 mismatch")
        self._weights = np.memmap(
            self.path, dtype="<f4", mode="r", offset=HEADER_BYTES,
            shape=(self.metadata.stage_count, WEIGHT_COUNT_PER_STAGE),
        )
        self._feature_bases = np.repeat(
            np.arange(PATTERN_COUNT, dtype=np.uint32) * FEATURE_COUNT_PER_PATTERN,
            SYMMETRY_COUNT,
        )

    @property
    def value_semantics(self) -> str:
        return RAW_TUPLE_HEURISTIC

    def stage_for(self, board: np.ndarray) -> int:
        max_exp = int(np.asarray(board, dtype=np.uint8).max(initial=0))
        max_tile = 0 if max_exp == 0 else (1 << max_exp)
        stage = 0
        for i, threshold in enumerate(self.metadata.stage_thresholds):
            if max_tile >= threshold:
                stage = i
        return stage

    def afterstate_value(self, board: np.ndarray) -> float:
        arr = np.asarray(board)
        return float(self.afterstate_values(arr.reshape(1, 16))[0])

    def afterstate_values(self, boards: np.ndarray) -> np.ndarray:
        arr = np.asarray(boards)
        if arr.ndim != 2 or arr.shape[1] != 16 or arr.dtype != np.uint8:
            raise ValueError("boards must be uint8 with shape (N, 16)")
        features = feature_indices_batch(arr)
        absolute = self._feature_bases.reshape(1, 64) + features
        stages = np.fromiter((self.stage_for(board) for board in arr), dtype=np.intp, count=len(arr))
        values = self._weights[stages[:, None], absolute]
        return values.astype(np.float64).sum(axis=1).astype(np.float32)

    def one_ply_action_values(self, state: np.ndarray) -> np.ndarray:
        values = np.full(4, np.nan, dtype=np.float64)
        for action in ACTIONS:
            moved = move_without_spawn(state, action)
            if moved.moved:
                values[int(action)] = float(moved.reward) + self.afterstate_value(moved.afterstate)
        return values

    def formal_state_leaf(self, state: np.ndarray) -> float:
        return float(self.formal_state_leaf_batch(np.asarray(state).reshape(1, 16))[0])

    def formal_state_leaf_batch(self, states: np.ndarray) -> np.ndarray:
        boards = np.asarray(states)
        if boards.ndim != 2 or boards.shape[1] != 16 or boards.dtype != np.uint8:
            raise ValueError("states must be uint8 with shape (N, 16)")
        from .fast_env import move_batch
        repeated = np.repeat(np.ascontiguousarray(boards), 4, axis=0)
        actions = np.tile(np.arange(4, dtype=np.uint8), len(boards))
        moved = move_batch(repeated, actions)
        flat = np.full(len(repeated), np.nan, dtype=np.float64)
        legal = np.asarray(moved.moved, dtype=bool)
        if legal.any():
            future = self.afterstate_values(np.ascontiguousarray(moved.afterstates[legal]))
            flat[legal] = moved.rewards[legal].astype(np.float64) + future.astype(np.float64)
        matrix = flat.reshape(len(boards), 4)
        result = np.zeros(len(boards), dtype=np.float64)
        has_legal = np.isfinite(matrix).any(axis=1)
        result[has_legal] = np.nanmax(matrix[has_legal], axis=1)
        return result

__all__ = [
    "CheckpointMetadata", "DEFAULT_CHECKPOINT", "EXPECTED_SHA256",
    "FORMAL_STATE_TUPLE_HEURISTIC", "PATTERNS", "RAW_TUPLE_HEURISTIC",
    "TupleTeacher", "feature_indices", "fnv1a64", "parse_checkpoint_header", "sha256_file",
]
