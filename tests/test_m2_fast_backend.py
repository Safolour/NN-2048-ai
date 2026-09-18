from __future__ import annotations

import numpy as np
import pytest

from game2048 import Action, is_terminal, legal_mask, move_without_spawn
from game2048.fast_env import (
    is_terminal_batch as numpy_terminal_batch,
    legal_mask_batch as numpy_legal_mask_batch,
    move_batch as numpy_move_batch,
)
from game2048.m2_fast_backend import (
    backend_info,
    legal_mask_batch,
    move_selected_batch,
)
from game2048.symmetry import transform_action, transform_board


SEED = 20260919
HIGH_EXPONENTS = (15, 16, 17, 20, 21, 31, 61, 62, 63, 126, 127, 128, 254, 255)


def ordinary_boards(count: int, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    boards = rng.integers(1, 18, size=(count, 16), dtype=np.uint8)
    boards[rng.random((count, 16)) < 0.5] = 0
    return np.ascontiguousarray(boards)


def assert_move_equal(cpp, numpy_result):
    assert np.array_equal(cpp.afterstates, numpy_result.afterstates)
    assert np.array_equal(cpp.rewards, numpy_result.rewards)
    assert np.array_equal(cpp.moved, numpy_result.moved)


def test_backend_identity():
    info = backend_info()
    assert info["kind"] == "scalar+row-lut"
    assert info["simd_used"] is False
    assert info["lut_used"] is True


def test_ordinary_10000_boards_all_four_actions_differential():
    boards = ordinary_boards(10_000)
    cpp_legal = legal_mask_batch(boards)
    np_legal = numpy_legal_mask_batch(boards)
    assert np.array_equal(cpp_legal, np_legal)
    assert np.array_equal(~cpp_legal.any(axis=1), numpy_terminal_batch(boards))

    for action in range(4):
        actions = np.full(boards.shape[0], action, dtype=np.uint8)
        assert_move_equal(
            move_selected_batch(boards, actions),
            numpy_move_batch(boards, actions),
        )


def test_direct_m0_oracle_anchor():
    boards = ordinary_boards(128, seed=SEED + 1)
    for i, board in enumerate(boards):
        cpp_legal = legal_mask_batch(board.reshape(1, 16))[0]
        assert np.array_equal(cpp_legal, np.asarray(legal_mask(board), dtype=bool))
        assert bool(~cpp_legal.any()) == bool(is_terminal(board))
        for action in range(4):
            cpp = move_selected_batch(
                board.reshape(1, 16), np.array([action], dtype=np.uint8)
            )
            ref = move_without_spawn(board, Action(action))
            assert np.array_equal(cpp.afterstates[0], ref.afterstate), (i, action)
            assert int(cpp.rewards[0]) == ref.reward, (i, action)
            assert bool(cpp.moved[0]) == ref.moved, (i, action)


def test_high_tile_nonmerging_boards_cover_required_exponents():
    boards = []
    for exponent in HIGH_EXPONENTS:
        board = np.zeros(16, dtype=np.uint8)
        board[[0, 5, 10, 15]] = exponent
        boards.append(board)

    rng = np.random.default_rng(SEED + 2)
    values = np.array(HIGH_EXPONENTS, dtype=np.uint8)
    for _ in range(64):
        board = np.zeros(16, dtype=np.uint8)
        positions = rng.choice(16, size=len(values), replace=False)
        shuffled = values.copy()
        rng.shuffle(shuffled)
        board[positions] = shuffled
        boards.append(board)

    batch = np.ascontiguousarray(np.stack(boards))
    assert np.array_equal(legal_mask_batch(batch), numpy_legal_mask_batch(batch))
    for action in range(4):
        actions = np.full(batch.shape[0], action, dtype=np.uint8)
        assert_move_equal(
            move_selected_batch(batch, actions),
            numpy_move_batch(batch, actions),
        )


@pytest.mark.parametrize(
    "board,action",
    [
        ([61, 61, 61, 61] + [0] * 12, 2),
        ([62, 62, 0, 0] + [0] * 12, 2),
        ([255, 255, 0, 0] + [0] * 12, 2),
    ],
)
def test_selected_move_overflow_matches_frozen_numpy(board, action):
    boards = np.asarray([board], dtype=np.uint8)
    actions = np.asarray([action], dtype=np.uint8)
    with pytest.raises(OverflowError):
        numpy_move_batch(boards, actions)
    with pytest.raises(OverflowError):
        move_selected_batch(boards, actions)


def test_legal_tile_exponent_overflow_matches_frozen_numpy():
    boards = np.zeros((1, 16), dtype=np.uint8)
    boards[0, :2] = 255
    with pytest.raises(OverflowError):
        numpy_legal_mask_batch(boards)
    with pytest.raises(OverflowError):
        legal_mask_batch(boards)


def test_all_eight_d4_symmetries_match_and_are_equivariant():
    boards = ordinary_boards(256, seed=SEED + 3)
    rng = np.random.default_rng(SEED + 4)
    actions = rng.integers(0, 4, size=boards.shape[0], dtype=np.uint8)
    base = move_selected_batch(boards, actions)

    for transform_id in range(8):
        transformed_boards = np.ascontiguousarray(
            np.stack([transform_board(board, transform_id) for board in boards])
        )
        transformed_actions = np.asarray(
            [
                int(transform_action(Action(int(action)), transform_id))
                for action in actions
            ],
            dtype=np.uint8,
        )
        cpp = move_selected_batch(transformed_boards, transformed_actions)
        frozen = numpy_move_batch(transformed_boards, transformed_actions)
        assert_move_equal(cpp, frozen)

        expected_after = np.ascontiguousarray(
            np.stack(
                [transform_board(board, transform_id) for board in base.afterstates]
            )
        )
        assert np.array_equal(cpp.afterstates, expected_after)
        assert np.array_equal(cpp.rewards, base.rewards)
        assert np.array_equal(cpp.moved, base.moved)
        assert np.array_equal(
            legal_mask_batch(transformed_boards),
            numpy_legal_mask_batch(transformed_boards),
        )


def test_backend_requires_contiguous_uint8_boards():
    good = ordinary_boards(8)
    with pytest.raises(ValueError):
        legal_mask_batch(good[:, ::-1])
    with pytest.raises(ValueError):
        legal_mask_batch(good.astype(np.int16))

