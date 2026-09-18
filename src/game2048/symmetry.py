"""D4 symmetry transforms for the 2048 reference environment (milestone M0).

The eight transforms are frozen and numbered exactly as follows:

===== ==========================================
 id    meaning
===== ==========================================
  0    Identity
  1    Rotate 90 degrees clockwise
  2    Rotate 180 degrees
  3    Rotate 270 degrees clockwise
  4    Mirror Left-Right (column 0 <-> column 3,
       column 1 <-> column 2)
  5    Rotate90(Mirror Left-Right(board))
  6    Rotate180(Mirror Left-Right(board))
  7    Rotate270(Mirror Left-Right(board))
===== ==========================================

``transform_action`` is derived from the geometry of the direction vectors
instead of a hand-written table: ``row`` grows downwards, ``col`` grows to the
right, and the direction of an action is mapped through the same linear map
that sends a board to its transform.  The resulting mapping is additionally
pinned by a regression test and by the movement-equivariance property test::

    T(move(s, a).afterstate) == move(T(s), T(a)).afterstate

Untransformed actions keep their canonical meaning::

    UP = (-1, 0)   DOWN = (1, 0)   LEFT = (0, -1)   RIGHT = (0, 1)
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .reference_env import (
    BOARD_SHAPE,
    BOARD_SIDE,
    CELL_COUNT,
    DIRECTION_VECTORS,
    Action,
    _as_cell_list,
    _coerce_action,
)

#: Number of D4 transforms (the dihedral group of the square).
TRANSFORM_COUNT: int = 8

_IDENTITY = 0
_ROTATE_90_CW = 1
_ROTATE_180 = 2
_ROTATE_270_CW = 3
_MIRROR_LR = 4

#: Inverse element of each transform in the D4 group.  Transforms 4..7 are
#: reflections, and every reflection is its own inverse.
_INVERSE_TRANSFORM_IDS: Tuple[int, ...] = (0, 3, 2, 1, 4, 5, 6, 7)

#: Action of each direction vector; inverse of ``DIRECTION_VECTORS``.
_ACTION_BY_DIRECTION: Dict[Tuple[int, int], Action] = {
    vector: action for action, vector in DIRECTION_VECTORS.items()
}


def _coerce_transform_id(transform_id) -> int:
    """Return ``transform_id`` as an ``int`` in ``0..7`` or raise ``ValueError``."""
    try:
        value = int(transform_id)
    except (TypeError, ValueError):
        raise ValueError(
            f"transform_id must be an integer in 0..{TRANSFORM_COUNT - 1}; "
            f"got {transform_id!r}"
        ) from None
    if value < 0 or value >= TRANSFORM_COUNT:
        raise ValueError(
            f"transform_id must be in 0..{TRANSFORM_COUNT - 1}; got {value}"
        )
    return value


def _as_grid(board) -> np.ndarray:
    """Return ``board`` as a fresh ``(4, 4)`` grid of exponents.

    The caller's array is never modified and the returned grid is independent
    of it, so transforms can never alias the input.
    """
    cells = _as_cell_list(board)
    return np.array(cells, dtype=np.uint8).reshape(BOARD_SIDE, BOARD_SIDE)


def _rotate_90_cw(grid: np.ndarray) -> np.ndarray:
    """Rotate the grid 90 degrees clockwise (visually, rows downwards)."""
    return np.rot90(grid, -1)


def _rotate_180(grid: np.ndarray) -> np.ndarray:
    """Rotate the grid 180 degrees."""
    return np.rot90(grid, 2)


def _rotate_270_cw(grid: np.ndarray) -> np.ndarray:
    """Rotate the grid 270 degrees clockwise (= 90 degrees counter-clockwise)."""
    return np.rot90(grid, 1)


def _mirror_left_right(grid: np.ndarray) -> np.ndarray:
    """Mirror the grid left-right (column 0 <-> 3, column 1 <-> 2)."""
    return np.fliplr(grid)


def transform_board(board, transform_id) -> np.ndarray:
    """Apply D4 transform ``transform_id`` to ``board``.

    Returns a new ``(16,)`` ``uint8`` array.  The input is never modified.
    """
    transform = _coerce_transform_id(transform_id)
    grid = _as_grid(board)

    if transform == _IDENTITY:
        transformed = grid
    elif transform == _ROTATE_90_CW:
        transformed = _rotate_90_cw(grid)
    elif transform == _ROTATE_180:
        transformed = _rotate_180(grid)
    elif transform == _ROTATE_270_CW:
        transformed = _rotate_270_cw(grid)
    elif transform == _MIRROR_LR:
        transformed = _mirror_left_right(grid)
    elif transform == 5:
        transformed = _rotate_90_cw(_mirror_left_right(grid))
    elif transform == 6:
        transformed = _rotate_180(_mirror_left_right(grid))
    else:  # transform == 7
        transformed = _rotate_270_cw(_mirror_left_right(grid))

    # np.array copies, so the result never aliases the caller's board.
    return np.array(transformed, dtype=np.uint8).reshape(CELL_COUNT)


def _transform_direction(
    direction: Tuple[int, int], transform_id: int
) -> Tuple[int, int]:
    """Map a direction vector through the linear part of ``transform_id``.

    ``(row, col)`` with rows growing downwards.  Rotating the board 90 degrees
    clockwise maps ``(r, c)`` to ``(c, -r)``, hence ``UP`` becomes ``RIGHT``.
    """
    row, col = direction
    if transform_id == _IDENTITY:
        return (row, col)
    if transform_id == _ROTATE_90_CW:
        return (col, -row)
    if transform_id == _ROTATE_180:
        return (-row, -col)
    if transform_id == _ROTATE_270_CW:
        return (-col, row)
    if transform_id == _MIRROR_LR:
        return (row, -col)
    # Transforms 5..7 are "mirror first, then rotate", so the direction is
    # mirrored first and then rotated, exactly like the board itself.
    mirrored = _transform_direction(direction, _MIRROR_LR)
    return _transform_direction(mirrored, transform_id - 4)


def transform_action(action, transform_id) -> Action:
    """Return the action that plays ``action`` inside the transformed board.

    It satisfies ``T(move(s, a).afterstate) == move(T(s), T(a)).afterstate``.
    """
    coerced_action = _coerce_action(action)
    transform = _coerce_transform_id(transform_id)
    direction = DIRECTION_VECTORS[coerced_action]
    return _ACTION_BY_DIRECTION[_transform_direction(direction, transform)]


def inverse_transform_id(transform_id) -> int:
    """Return the D4 transform that undoes ``transform_id``."""
    return _INVERSE_TRANSFORM_IDS[_coerce_transform_id(transform_id)]


__all__ = [
    "TRANSFORM_COUNT",
    "inverse_transform_id",
    "transform_action",
    "transform_board",
]
