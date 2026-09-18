import numpy as np
import pytest

from game2048.m3_tuple_backend import (
    backend_info,
    feature_indices as cpp_feature_indices,
    stage_indices as cpp_stage_indices,
)
from game2048.m3_tuple_teacher import feature_indices_batch


def test_feature_indices_match_python_random_and_high_exponents():
    rng = np.random.default_rng(20260919)
    random = rng.integers(0, 256, size=(1000, 16), dtype=np.uint8)
    edges = np.array([
        [0] * 16,
        [15] * 16,
        [16] * 16,
        [17] * 16,
        [20] * 16,
        [31] * 16,
        [63] * 16,
        [64] * 16,
        [127] * 16,
        [255] * 16,
    ], dtype=np.uint8)
    boards = np.ascontiguousarray(np.vstack([random, edges]))
    assert np.array_equal(cpp_feature_indices(boards), feature_indices_batch(boards))


def test_stage_indices_are_overflow_safe_at_high_exponents():
    exponents = [0, 13, 14, 15, 16, 63, 64, 127, 255]
    boards = np.zeros((len(exponents), 16), dtype=np.uint8)
    boards[:, 0] = np.asarray(exponents, dtype=np.uint8)
    thresholds = np.array([0, 16384, 32768, 65536], dtype=np.uint64)
    got = cpp_stage_indices(boards, thresholds)
    assert got.tolist() == [0, 0, 1, 2, 3, 3, 3, 3, 3]


def test_empty_batches_are_supported_for_pure_helpers():
    boards = np.empty((0, 16), dtype=np.uint8)
    assert cpp_feature_indices(boards).shape == (0, 64)
    assert cpp_stage_indices(boards, [0]).shape == (0,)


def test_wrapper_rejects_wrong_dtype_shape_and_noncontiguous():
    with pytest.raises(ValueError):
        cpp_feature_indices(np.zeros((2, 16), dtype=np.int16))
    with pytest.raises(ValueError):
        cpp_feature_indices(np.zeros((2, 15), dtype=np.uint8))
    base = np.zeros((4, 32), dtype=np.uint8)
    noncontiguous = base[:, ::2]
    assert noncontiguous.shape == (4, 16)
    assert not noncontiguous.flags.c_contiguous
    with pytest.raises(ValueError):
        cpp_feature_indices(noncontiguous)


def test_stage_threshold_count_and_power_of_two_validation():
    board = np.zeros((1, 16), dtype=np.uint8)
    with pytest.raises(ValueError):
        cpp_stage_indices(board, [])
    with pytest.raises(ValueError):
        cpp_stage_indices(board, [0, 1, 2, 4, 8])
    with pytest.raises(ValueError):
        cpp_stage_indices(board, np.zeros((2, 2), dtype=np.uint64))
    with pytest.raises((ValueError, RuntimeError)):
        cpp_stage_indices(board, [0, 3])


def test_backend_metadata_is_production_sized():
    info = backend_info()
    assert info["feature_count_per_pattern"] == 1 << 24
    assert info["weight_count_per_stage"] == 8 * (1 << 24)
