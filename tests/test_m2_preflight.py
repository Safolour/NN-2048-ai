"""M2 Preflight: the CPU -> GPU contiguous board-batch transfer contract.

Every test here is CPU-only and must never be skipped: this file is the executable
form of the frozen M2 boundary defined in ``game2048.m2_data``.

The contract under test:

    type   : numpy.ndarray
    shape  : (N, 16), N >= 1
    dtype  : uint8
    layout : C-contiguous

with three fixed rules:

* a C-contiguous input is returned **as the same object** (zero copy);
* a valid but non-contiguous input is converted **once**, here, at this boundary;
* a wrong dtype or a wrong shape is rejected -- never silently converted, never
  silently reshaped.

There is no GPU work and no network in this file; only the producer-side NumPy
boundary exists at M2 Preflight.
"""

from __future__ import annotations

import numpy as np
import pytest

from game2048 import Action
from game2048.fast_env import Fast2048BatchEnv
from game2048.m2_data import prepare_board_batch_for_transfer

from _m1_helpers import SEED

BOARD_WIDTH = 16


def _contiguous(rows: int = 4) -> np.ndarray:
    """A small C-contiguous ``(rows, 16)`` ``uint8`` batch with distinct values."""
    values = np.arange(rows * BOARD_WIDTH, dtype=np.uint8).reshape(rows, BOARD_WIDTH)
    array = np.ascontiguousarray(values)
    assert array.flags.c_contiguous
    return array


def _strided(rows: int = 4) -> np.ndarray:
    """A genuinely non-contiguous ``(rows, 16)`` ``uint8`` view.

    ``base[:, ::2]`` keeps every second column of a ``(rows, 32)`` buffer, so the
    shape is exactly ``(rows, 16)`` while the strides are not.  The layout is
    asserted here so this fixture can never silently become contiguous.
    """
    base = np.arange(rows * 2 * BOARD_WIDTH, dtype=np.uint8).reshape(
        rows, 2 * BOARD_WIDTH
    )
    view = base[:, ::2]
    assert view.shape == (rows, BOARD_WIDTH)
    assert not view.flags.c_contiguous
    return view


# --------------------------------------------------------------------------- #
# 1. valid contiguous input
# --------------------------------------------------------------------------- #


def test_valid_contiguous_batch_is_accepted():
    boards = _contiguous(8)
    result = prepare_board_batch_for_transfer(boards)
    assert isinstance(result, np.ndarray)
    assert result.shape == (8, BOARD_WIDTH)
    assert result.dtype == np.uint8


def test_single_board_batch_is_accepted():
    """``N >= 1``: a batch of exactly one board is valid."""
    boards = _contiguous(1)
    result = prepare_board_batch_for_transfer(boards)
    assert result.shape == (1, BOARD_WIDTH)
    assert result is boards


# --------------------------------------------------------------------------- #
# 2. contiguous input is zero-copy
# --------------------------------------------------------------------------- #


def test_contiguous_input_is_zero_copy():
    boards = _contiguous(16)
    result = prepare_board_batch_for_transfer(boards)

    # The identity requirement: not merely "equal", the very same object.
    assert result is boards
    assert np.shares_memory(result, boards)


def test_contiguous_input_is_not_copied_from_a_plain_zeros_batch():
    boards = np.zeros((32, BOARD_WIDTH), dtype=np.uint8)
    assert prepare_board_batch_for_transfer(boards) is boards


# --------------------------------------------------------------------------- #
# 3./4./5. non-contiguous input: accepted, converted once, values preserved
# --------------------------------------------------------------------------- #


def test_non_contiguous_batch_is_accepted():
    view = _strided(4)
    result = prepare_board_batch_for_transfer(view)
    assert isinstance(result, np.ndarray)
    assert result.shape == view.shape
    assert result.dtype == np.uint8


def test_non_contiguous_result_is_c_contiguous():
    view = _strided(4)
    result = prepare_board_batch_for_transfer(view)
    assert result.flags.c_contiguous


def test_non_contiguous_result_preserves_values_exactly():
    view = _strided(6)
    expected = np.array(view, copy=True)
    result = prepare_board_batch_for_transfer(view)
    assert np.array_equal(result, expected)
    assert result.tobytes() == expected.tobytes()


def test_non_contiguous_input_is_not_returned_as_the_same_object():
    """A strided view cannot be handed to the transfer boundary unchanged."""
    view = _strided(4)
    result = prepare_board_batch_for_transfer(view)
    assert result is not view


# --------------------------------------------------------------------------- #
# 6. invalid dtype is rejected, never converted
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "dtype",
    [np.int64, np.int32, np.int16, np.float32, np.float64, np.bool_],
)
def test_invalid_dtype_is_rejected(dtype):
    boards = np.zeros((4, BOARD_WIDTH), dtype=dtype)
    with pytest.raises(TypeError):
        prepare_board_batch_for_transfer(boards)


def test_invalid_dtype_is_not_silently_converted():
    """Even an all-zero ``int64`` batch must not be coerced to ``uint8``."""
    boards = np.zeros((4, BOARD_WIDTH), dtype=np.int64)
    with pytest.raises(TypeError):
        prepare_board_batch_for_transfer(boards)


def test_non_ndarray_is_rejected():
    with pytest.raises(TypeError):
        prepare_board_batch_for_transfer([[1, 2, 3]])


# --------------------------------------------------------------------------- #
# 7./8. invalid shape is rejected, never reshaped
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "shape",
    [
        (BOARD_WIDTH,),          # a flat (16,) vector is not a batch
        (2, 4, 4),               # (N, 4, 4) is not the contract shape
        (4, 15),                 # one cell short
        (4, 17),                 # one cell too many
        (4, 32),                 # a wider buffer is not silently narrowed
        (0, BOARD_WIDTH),        # an empty batch carries no work
        (4,),                    # (4,) is not (4, 16)
        (BOARD_WIDTH, 0),        # (16, 0) is not (N, 16)
    ],
)
def test_invalid_shape_is_rejected(shape):
    boards = np.zeros(shape, dtype=np.uint8)
    with pytest.raises(ValueError):
        prepare_board_batch_for_transfer(boards)


def test_empty_batch_is_rejected():
    boards = np.zeros((0, BOARD_WIDTH), dtype=np.uint8)
    with pytest.raises(ValueError):
        prepare_board_batch_for_transfer(boards)


def test_flat_vector_is_not_reshaped_into_a_batch():
    """``(16,)`` must fail rather than quietly becoming ``(1, 16)``."""
    flat = np.zeros(BOARD_WIDTH, dtype=np.uint8)
    with pytest.raises(ValueError):
        prepare_board_batch_for_transfer(flat)


# --------------------------------------------------------------------------- #
# 9. the real M1 environment satisfies the boundary contract
# --------------------------------------------------------------------------- #


def test_fast_env_boards_satisfy_the_boundary_contract():
    env = Fast2048BatchEnv(64, seed=SEED)
    env.reset(seed=SEED)

    boards = np.asarray(env.boards)
    assert boards.shape == (64, BOARD_WIDTH)
    assert boards.dtype == np.uint8
    assert boards.flags.c_contiguous

    result = prepare_board_batch_for_transfer(boards)
    assert result is boards, "the M1 production layout must cross for free"


def test_fast_env_boards_satisfy_the_contract_after_stepping():
    """The contract survives real transitions, not just ``reset``."""
    env = Fast2048BatchEnv(32, seed=SEED)
    env.reset(seed=SEED)
    env.step(np.full(32, int(Action.LEFT), dtype=np.uint8))
    env.step(np.full(32, int(Action.DOWN), dtype=np.uint8))

    boards = np.asarray(env.boards)
    assert boards.dtype == np.uint8
    assert boards.flags.c_contiguous
    assert prepare_board_batch_for_transfer(boards) is boards


def test_fast_env_boards_view_of_a_selection_needs_the_boundary_copy():
    """A strided selection is legal input, but it is genuinely not contiguous.

    This is exactly the case the boundary exists for: the caller may legitimately
    hold a strided view, and the conversion then happens once at this boundary
    instead of inside a later network forward.
    """
    env = Fast2048BatchEnv(16, seed=SEED)
    env.reset(seed=SEED)

    selection = env.boards[::2]
    assert selection.shape == (8, BOARD_WIDTH)
    assert not selection.flags.c_contiguous

    result = prepare_board_batch_for_transfer(selection)
    assert result.flags.c_contiguous
    assert np.array_equal(result, np.array(selection, copy=True))


# --------------------------------------------------------------------------- #
# 10. the caller's input is never mutated
# --------------------------------------------------------------------------- #


def test_contiguous_input_is_never_mutated():
    boards = _contiguous(4)
    before = boards.tobytes()
    prepare_board_batch_for_transfer(boards)
    assert boards.tobytes() == before


def test_non_contiguous_input_is_never_mutated():
    view = _strided(4)
    before = view.tobytes()
    result = prepare_board_batch_for_transfer(view)
    assert view.tobytes() == before
    assert not view.flags.c_contiguous, "the input view must keep its own layout"


def test_non_contiguous_input_backing_buffer_is_never_mutated():
    """Converting a view must not write into the buffer the view was cut from."""
    base = np.arange(4 * 2 * BOARD_WIDTH, dtype=np.uint8).reshape(4, 2 * BOARD_WIDTH)
    before = base.tobytes()
    view = base[:, ::2]
    prepare_board_batch_for_transfer(view)
    assert base.tobytes() == before


def test_writing_the_result_of_a_strided_input_does_not_touch_the_source():
    """The converted batch is an independent array, so it is safe to hand onward."""
    view = _strided(4)
    before = view.tobytes()
    result = prepare_board_batch_for_transfer(view)
    result[0, 0] = 200
    assert view.tobytes() == before
