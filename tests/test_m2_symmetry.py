import numpy as np
import torch

from game2048.m2_symmetry import (
    inverse_transform_q_values,
    transform_action_batch,
    transform_board_batch,
    transform_q_values,
)
from game2048.symmetry import transform_action, transform_board


def test_batched_board_transform_matches_frozen_m0_oracle():
    rng = np.random.default_rng(20260918)
    boards_np = rng.integers(0, 256, size=(32, 16), dtype=np.uint8)
    boards = torch.from_numpy(boards_np)
    for tid in range(8):
        ids = torch.full((len(boards_np),), tid, dtype=torch.int64)
        actual = transform_board_batch(boards, ids).numpy()
        expected = np.stack([transform_board(board, tid) for board in boards_np])
        assert np.array_equal(actual, expected)


def test_batched_action_transform_matches_frozen_m0_oracle():
    actions = torch.tensor([a for _tid in range(8) for a in range(4)], dtype=torch.int64)
    tids = torch.tensor([tid for tid in range(8) for _a in range(4)], dtype=torch.int64)
    actual = transform_action_batch(actions, tids).tolist()
    expected = [
        int(transform_action(a, tid))
        for tid in range(8)
        for a in range(4)
    ]
    assert actual == expected


def test_q_transform_round_trip_is_exact():
    torch.manual_seed(4)
    q = torch.randn(64, 4)
    tids = torch.arange(64, dtype=torch.int64) % 8
    transformed = transform_q_values(q, tids)
    restored = inverse_transform_q_values(transformed, tids)
    assert torch.equal(restored, q)


def test_board_transform_preserves_device_dtype_and_batch():
    boards = torch.arange(32, dtype=torch.uint8).reshape(2, 16)
    tids = torch.tensor([0, 4], dtype=torch.int64)
    out = transform_board_batch(boards, tids)
    assert out.dtype == boards.dtype
    assert out.device == boards.device
    assert out.shape == boards.shape
