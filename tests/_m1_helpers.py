"""Shared helpers for the M1 FastEnv test suite.

Every random test in this suite is driven by :data:`SEED` and documents its own
sample count and exponent range at the top of the file, so a failure is always
reproducible: the assertions print the sample index, the board, the action and
both results.
"""

from __future__ import annotations

import numpy as np

from game2048 import (
    Action,
    is_terminal,
    legal_mask,
    move_without_spawn,
)

#: Fixed seed for every random M1 test.  Deterministic, documented, reproducible.
SEED = 20260918

#: Exponent ceiling used by the "ordinary" differential tests.
ORDINARY_EXPONENT = 17
#: Exponent ceiling used by the high-tile differential tests.
HIGH_EXPONENT = 22


def batch(boards) -> np.ndarray:
    """Stack a sequence of 16-element rows into a C-contiguous ``(N, 16)`` batch."""
    array = np.asarray(boards, dtype=np.uint8)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    return np.ascontiguousarray(array)


def board(*rows) -> np.ndarray:
    """Build one ``(16,)`` ``uint8`` board from four rows of exponents."""
    return np.array(rows, dtype=np.uint8).reshape(16)


def row_board(values) -> np.ndarray:
    """Build a board whose only non-empty cells are ``values`` in row-major order."""
    board_array = np.zeros(16, dtype=np.uint8)
    board_array[: len(values)] = values
    return board_array


def random_boards(
    count: int,
    exponent_high: int,
    seed: int = SEED,
    *,
    empty_probability: float = 0.5,
) -> np.ndarray:
    """``(count, 16)`` ``uint8`` boards with exponents in ``0..exponent_high``.

    ``empty_probability`` controls how often a cell is left empty, which lets the
    tests cover both sparse and nearly-full boards.
    """
    rng = np.random.default_rng(seed)
    values = rng.integers(1, exponent_high + 1, size=(count, 16))
    blank = rng.random((count, 16)) < empty_probability
    values[blank] = 0
    return values.astype(np.uint8)


def board_str(board_array) -> str:
    """Compact single-line rendering of a board, for failure messages."""
    return "[" + " ".join(str(int(value)) for value in np.asarray(board_array).reshape(-1)) + "]"


def compare_move(fast_result, reference, entry, action: int, index: int) -> None:
    """Assert that one fast move reproduces ``move_without_spawn`` exactly.

    ``MISMATCH`` is reported with everything needed to replay the case, and
    ``OverflowError`` is only accepted when the reference raises it too.
    """
    try:
        expected = move_without_spawn(entry, Action(action))
    except OverflowError:
        # M0 refuses this synthetic high-exponent board.  The fast kernel has a
        # stricter ``int64`` reward limit and is checked separately in
        # ``test_m1_high_tiles.py``; no movement comparison is meaningful here.
        return

    context = f"sample={index} action={Action(action).name} board={board_str(entry)}"
    assert np.array_equal(
        fast_result.afterstates[index], expected.afterstate
    ), (
        f"{context}\n  fast afterstate = {board_str(fast_result.afterstates[index])}"
        f"\n  ref  afterstate = {board_str(expected.afterstate)}"
    )
    assert int(fast_result.rewards[index]) == expected.reward, (
        f"{context}\n  fast reward = {int(fast_result.rewards[index])}"
        f"\n  ref  reward = {expected.reward}"
    )
    assert bool(fast_result.moved[index]) == expected.moved, (
        f"{context}\n  fast moved = {bool(fast_result.moved[index])}"
        f"\n  ref  moved = {expected.moved}"
    )


def reference_legal_mask(entry) -> np.ndarray:
    """M0's ``legal_mask`` as a plain bool array."""
    return np.asarray(legal_mask(entry), dtype=bool)


def reference_terminal(entry) -> bool:
    """M0's ``is_terminal`` as a plain bool."""
    return bool(is_terminal(entry))


def reference_spawns(entry):
    """M0's exact spawn enumeration as a flat list of ``SpawnOutcome``."""
    from game2048 import enumerate_spawns

    return enumerate_spawns(entry)
