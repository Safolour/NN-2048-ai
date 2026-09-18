from __future__ import annotations

import struct

import numpy as np
import pytest

from game2048.m3_tuple_teacher import (
    FEATURE_COUNT_PER_PATTERN,
    HEADER_BYTES,
    MAGIC,
    PATTERNS,
    WEIGHT_COUNT_PER_STAGE,
    feature_indices,
    fnv1a64,
    parse_checkpoint_header,
)

def _synthetic_header() -> bytes:
    h = bytearray(HEADER_BYTES)
    h[:8] = MAGIC
    struct.pack_into("<6I", h, 8, 2, 0x01020304, 8, 6, 8, 4)
    struct.pack_into("<2Q", h, 32, FEATURE_COUNT_PER_PATTERN, WEIGHT_COUNT_PER_STAGE)
    struct.pack_into("<2I", h, 48, 1, 1)
    struct.pack_into("<4Q", h, 56, 0, 0, 0, 0)
    struct.pack_into("<4I", h, 88, 0, 0, 64, 1)
    struct.pack_into("<5f", h, 104, 0.0, 0.0, 0.1, 1.0, np.finfo(np.float32).tiny)
    struct.pack_into("<I", h, 124, 0)
    struct.pack_into("<6Q", h, 128, 4_800_000, 4_800_000, 0, 10_000_000, 0, 1)
    struct.pack_into("<2Q", h, 176, 123, 0)
    h[200:248] = PATTERNS.tobytes()
    struct.pack_into("<Q", h, 248, fnv1a64(bytes(h[:248])))
    return bytes(h)

def test_parse_synthetic_v2_header():
    meta = parse_checkpoint_header(_synthetic_header())
    assert meta.format_version == 2
    assert meta.stage_count == 1
    assert meta.stage_thresholds == (0,)
    assert meta.global_episodes_completed == 4_800_000
    assert meta.alpha_normalizer == 64
    assert meta.has_coherence_stats is False

def test_header_checksum_corruption_is_rejected():
    h = bytearray(_synthetic_header())
    h[40] ^= 1
    with pytest.raises(ValueError, match="checksum"):
        parse_checkpoint_header(bytes(h))

def test_feature_indices_identity_slot_and_high_tile_clamp():
    board = np.arange(16, dtype=np.uint8)
    board[0] = 20
    features = feature_indices(board)
    assert features.shape == (64,)
    assert features.dtype == np.uint32
    expected = 0
    for i, cell in enumerate(PATTERNS[0]):
        expected |= min(int(board[int(cell)]), 15) << (4 * i)
    assert int(features[0]) == expected

def test_feature_indices_does_not_mutate_input():
    board = np.arange(16, dtype=np.uint8)
    before = board.copy()
    feature_indices(board)
    assert np.array_equal(board, before)

def test_feature_indices_batch_matches_scalar():
    rng = np.random.default_rng(20260919)
    boards = rng.integers(0, 30, size=(17, 16), dtype=np.uint8)
    from game2048.m3_tuple_teacher import feature_indices_batch
    batch = feature_indices_batch(boards)
    expected = np.stack([feature_indices(board) for board in boards])
    assert np.array_equal(batch, expected)
