"""Vectorized / batched 2048 environment (milestone M1).

``Fast2048BatchEnv`` = the high-throughput state producer
=========================================================

M0 (``reference_env.py`` / ``symmetry.py``) is the **permanently frozen golden
reference and correctness oracle**.  Everything in this module is a *pure
performance* implementation of exactly the same rules and must reproduce M0
exactly on boards, rewards, ``moved``, ``legal_mask`` and ``is_terminal``.
Whenever this module disagrees with M0, **this module is wrong**.

Scope (M1)
----------
* batch movement over a whole ``(N, 16)`` board batch;
* batch legal mask / batch terminal;
* exact spawn enumeration in a flat CSR layout;
* vectorized random spawn and deterministic spawn application;
* :class:`Fast2048BatchEnv`: one object holding ``N`` independent games.

Explicitly **out of scope** (M2 and later): networks, Q/V/A heads, trainer,
replay buffers, self-play, teacher, expectimax, search correction, CUDA.

Board layout (identical to M0)
------------------------------
Every board is 16 cells in row-major order::

     0   1   2   3
     4   5   6   7
     8   9  10  11
    12  13  14  15

A cell stores the tile's exponent: ``0`` = empty, ``1`` = 2, ``2`` = 4, ...,
``20`` = 1048576, ``21`` = 2097152, ``22`` = 4194304, ...  The exponent is stored
verbatim and is **never** clamped to a network overflow bucket.

A batch is one ``numpy.ndarray`` of shape ``(N, 16)``, dtype ``numpy.uint8``,
C-contiguous.

Actions (identical to M0)
-------------------------
``UP = 0``, ``DOWN = 1``, ``LEFT = 2``, ``RIGHT = 3``.  :class:`Action` is
imported from M0 -- this module never defines a second action numbering.

Movement
--------
The core is a fully vectorized transcription of M0's frozen single-line rule: the
whole batch is moved at once, with **no N-dependent Python loop** anywhere.

* A line is identified by its four exponents in **destination-first order**,
  which is exactly ``reference_env._line_indices`` -- the same table M0 uses, so
  the line layout can never drift.
* The batch is split into at most four action groups (four fixed iterations).
  Each group is gathered into canonical line layout with one fancy index, reshaped
  to ``(R, 4)`` ``uint8`` (``R = boards * 4``), and then packed, merged and
  re-packed entirely with array-wide masks; one scatter writes the afterstates
  back.  The only remaining Python loops are the fixed ``range(4)`` over the
  actions, the fixed ``range(3)`` over the merge boundaries and the fixed
  ``range(4)`` over the four lines of one board during reward aggregation.
* ``pack-left`` is a rank-based gather: the non-zero cells of each row are
  scattered to column ``cumsum(non-zero) - 1``.  That is exact in one pass.
* ``merge-left`` visits boundaries ``0, 1, 2`` and uses a per-row
  ``blocked_from_previous`` mask, which is what makes ``[1,1,1,1] -> [2,2,0,0]``
  and prevents a freshly created tile from merging again.
* There is no per-line cache: the previous ``_LINE_CACHE`` /
  ``_merge_line_cached`` / ``_shift_line`` / ``_line_changed`` scalar helpers are
  gone, because they were exactly the N-dependent Python loop this module avoids.

Reward (identical to M0)
------------------------
Merging two exponent-``e`` tiles gives exponent ``e + 1`` and reward
``2 ** (e + 1)``, accumulated in ``numpy.int64``::

    exp 16 + exp 16 -> exp 17, reward = 131072
    exp 20 + exp 20 -> exp 21, reward = 2 ** 21
    exp 21 + exp 21 -> exp 22, reward = 2 ** 22

M1's ``int64`` contract has **two** range limits, and both raise
:class:`OverflowError` rather than wrapping.  M0 is unaffected: its reward is an
arbitrary-precision Python ``int``.

1. **Single merge.**  ``e + e -> e + 1`` pays ``2 ** (e + 1)``, so the largest
   mergeable exponent is ``61`` (``2 ** 62``); ``e = 62`` needs ``2 ** 63`` and is
   rejected.  Exponents ``>= 62`` are therefore unmergeable in M1.
2. **Aggregate.**  A single action can merge on several boundaries of one line and
   on several lines of one board.  A sum of individually representable rewards can
   still exceed ``9223372036854775807`` (``2 ** 63 - 1``):

   * ``[61, 61, 61, 61]`` merges twice in one row: ``2**62 + 2**62 = 2**63``;
   * two rows of ``[61, 61]`` merge on two lines: ``2**62 + 2**62 = 2**63``.

   Both are rejected, and so is a score update whose running total would pass the
   ceiling.  The check is performed **before** the addition, because NumPy does
   not signal integer overflow for array operations or reductions --
   ``np.errstate(over="raise")`` catches *scalar* overflow only and silently
   returns a wrapped value for ``a.sum()``.  See
   :func:`_checked_add_nonnegative_int64`.

This is the single documented divergence between M1 and M0, and it is always an
explicit exception -- never a wrong number.  These boards are unreachable in real
play (a game that reached exponent 62 has produced far more tiles than the board
can hold) and remain fully supported by M0.

RNG (identical distribution, batched stream)
--------------------------------------------
Only ``numpy.random.Generator`` is used -- never the global ``numpy.random``
state, never the ``random`` module.  ``Fast2048BatchEnv`` draws one ``(k, 2)``
block per spawn group, so it is **not** bitwise-stream-identical to N independent
M0 environments (which draw twice per environment).  What *is* identical, and
tested: the support, the ``0.9 / 0.1`` tile probabilities, the uniform cell
choice and "an illegal action consumes no randomness".  Re-seeding one
``Fast2048BatchEnv`` and replaying the same actions always reproduces exactly the
same trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .reference_env import (
    Action,
    CELL_COUNT,
    INITIAL_TILE_COUNT,
    MAX_EXPONENT,
    SPAWN_EXPONENT_TILE_2,
    SPAWN_EXPONENT_TILE_4,
    SPAWN_PROB_TILE_2,
    _line_indices,
)

__all__ = [
    "ACTION_DOWN",
    "ACTION_LEFT",
    "ACTION_RIGHT",
    "ACTION_UP",
    "ACTIONS",
    "BOARD_COLUMNS",
    "MAX_SAFE_MERGE_EXPONENT",
    "MAX_SPAWN_OUTCOMES",
    "BatchMoveResult",
    "BatchSpawnEnumeration",
    "BatchSpawnResult",
    "BatchStepResult",
    "Fast2048BatchEnv",
    "apply_spawn_batch",
    "enumerate_spawns_batch",
    "is_terminal_batch",
    "legal_mask_batch",
    "move_batch",
    "spawn_random_batch",
]

# --------------------------------------------------------------------------- #
# Frozen constants
# --------------------------------------------------------------------------- #

#: Number of columns per board row.
BOARD_COLUMNS: int = 4

#: Actions in canonical M0 order ``UP, DOWN, LEFT, RIGHT``.
ACTIONS: Tuple[Action, ...] = (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT)

ACTION_UP: int = int(Action.UP)
ACTION_DOWN: int = int(Action.DOWN)
ACTION_LEFT: int = int(Action.LEFT)
ACTION_RIGHT: int = int(Action.RIGHT)

#: Largest exponent that can still be merged on the fast ``int64`` reward path.
#:
#: ``e + e -> e + 1`` pays ``2 ** (e + 1)`` and ``2 ** 63 - 1`` is the ``int64``
#: ceiling, hence ``e + 1 <= 62``.  A merge above this raises ``OverflowError``.
MAX_SAFE_MERGE_EXPONENT: int = 62

#: A board has 16 cells and each empty cell contributes at most two outcomes.
MAX_SPAWN_OUTCOMES: int = 2 * CELL_COUNT

_SPAWN_EXPONENT_TILE_2: int = int(SPAWN_EXPONENT_TILE_2)
_SPAWN_EXPONENT_TILE_4: int = int(SPAWN_EXPONENT_TILE_4)
_SPAWN_PROB_TILE_2: float = float(SPAWN_PROB_TILE_2)

#: ``log2(tile)`` for the two spawnable tiles: ``log2(2) = 1``, ``log2(4) = 2``.
#: Index 0 is the tile-2 branch (``draw < 0.9``) and index 1 the tile-4 branch, so
#: a single table lookup turns the draw into the exponent.
_LOG2_TILE = np.array([_SPAWN_EXPONENT_TILE_2, _SPAWN_EXPONENT_TILE_4], dtype=np.uint8)


def _spawn_exponents(draws: np.ndarray) -> np.ndarray:
    """Map uniform draws in ``[0, 1)`` to spawn exponents 1 or 2.

    Draw ``< 0.9`` selects the tile-2 branch (exponent 1); otherwise the tile-4
    branch (exponent 2) -- M0's frozen 90 / 10 split.
    """
    return _LOG2_TILE[(draws >= _SPAWN_PROB_TILE_2).astype(np.uint8)]


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #


def _as_boards(boards, *, name: str = "boards") -> np.ndarray:
    """Return ``boards`` as a ``(N, 16)`` ``uint8`` array; reject anything else.

    A ``uint8`` array of the right shape is returned unchanged (never copied).
    A different shape, a non-integer / bool / float dtype, or an exponent outside
    ``0..255`` raises ``ValueError`` instead of being reinterpreted.
    """
    arr = np.asarray(boards)
    if arr.ndim != 2 or arr.shape[1] != CELL_COUNT:
        raise ValueError(f"{name} must have shape (N, {CELL_COUNT}), got {arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one board, got {arr.shape}")
    if np.issubdtype(arr.dtype, np.bool_) or not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"{name} dtype must be an integer type, got {arr.dtype}")
    if arr.dtype != np.uint8:
        if int(arr.min()) < 0 or int(arr.max()) > MAX_EXPONENT:
            raise ValueError(
                f"{name} values must lie in 0..{MAX_EXPONENT} for dtype uint8"
            )
        arr = arr.astype(np.uint8)
    return arr


def _as_actions(actions, count: int) -> np.ndarray:
    """Return ``actions`` as an ``(N,)`` ``uint8`` array with values in ``0..3``.

    Deliberately strict: float, bool, string, wrong shape, wrong length and
    out-of-range values are rejected.  In particular ``1.9`` and ``True`` are
    *not* coerced to ``1``; that kind of silent coercion is forbidden on the M1
    interface.
    """
    arr = np.asarray(actions)
    if arr.ndim != 1:
        raise ValueError(f"actions must have shape (N,), got {arr.shape}")
    if arr.shape[0] != count:
        raise ValueError(
            f"actions length {arr.shape[0]} does not match the number of boards "
            f"{count}"
        )
    if np.issubdtype(arr.dtype, np.bool_) or not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(
            f"actions dtype must be an integer type (bool/float/str are rejected), "
            f"got {arr.dtype}"
        )
    if arr.size:
        low = int(arr.min())
        high = int(arr.max())
        if low < 0 or high > 3:
            raise ValueError(
                f"actions must lie in 0..3 (UP, DOWN, LEFT, RIGHT); got values in "
                f"{low}..{high}"
            )
    return arr.astype(np.uint8, copy=False)


def _as_vector(
    values,
    count: int,
    *,
    name: str,
    dtype,
    low: Optional[int] = None,
    high: Optional[int] = None,
) -> np.ndarray:
    """Return ``values`` as a ``(N,)`` integer array of ``dtype``, else raise."""
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{name} must have shape (N,), got {arr.shape}")
    if arr.shape[0] != count:
        raise ValueError(
            f"{name} length {arr.shape[0]} does not match the number of boards "
            f"{count}"
        )
    if np.issubdtype(arr.dtype, np.bool_) or not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"{name} dtype must be an integer type, got {arr.dtype}")
    if arr.size:
        observed_low = int(arr.min())
        observed_high = int(arr.max())
        if low is not None and observed_low < low:
            raise ValueError(f"{name} must be >= {low}; got {observed_low}")
        if high is not None and observed_high > high:
            raise ValueError(f"{name} must be <= {high}; got {observed_high}")
    return arr.astype(dtype, copy=False)


def _require_rng(rng) -> np.random.Generator:
    """Return ``rng`` if it is a ``numpy.random.Generator``, else raise."""
    if not isinstance(rng, np.random.Generator):
        raise TypeError(
            f"rng must be a numpy.random.Generator, got {type(rng).__name__}"
        )
    return rng


# --------------------------------------------------------------------------- #
# Movement core: vectorized across boards
# --------------------------------------------------------------------------- #
#
# It is M0's own ``_line_indices``, reused verbatim so the line layout can never
# drift away from the reference.
#
#: ``_LINE_ORDER[a]`` is the flat permutation of the 16 cells for action ``a``,
#: already in the canonical **destination-first** order used by the reference
#: merge routine (``_line_indices`` from M0).  ``_LINE_ORDER[a][k]`` is the board
#: cell that canonical position ``k`` reads from, so gathering a whole board with
#: it produces the canonical layout.
#:
#: ``_LINE_INVERSE[a]`` is the inverse permutation: writing a canonical value at
#: position ``k`` must touch ``_LINE_INVERSE[a][k]``.
#
#: Both are 1-D permutations of ``0..15`` (M0 returns them as a ``(4, 4)`` ragged
#: list, which is flattened here so that a whole ``(N, 16)`` batch can be
#: transformed by a single fancy-index gather and a single scatter).
#:
#: With these two tables the movement path is: gather every board into canonical
#: order, reshape to ``(N * 4, 4)`` lines, pack/merge/pack entirely with
#: array-wide masks, then scatter back.  There is no Python-level loop over boards
#: anywhere in the movement path.

_LINE_ORDER = np.array(
    [_line_indices(Action(entry)) for entry in range(4)], dtype=np.intp
).reshape(4, CELL_COUNT)

_LINE_INVERSE = np.empty((4, CELL_COUNT), dtype=np.intp)
for _entry in range(4):
    _LINE_INVERSE[_entry][_LINE_ORDER[_entry]] = np.arange(CELL_COUNT)
del _entry

#: Reward of one merge at exponent ``e``: ``2 ** (e + 1)``.
#:
#: Only entries up to :data:`MAX_SAFE_MERGE_EXPONENT` are ever indexed on the hot
#: path (larger exponents raise before this table is touched), so every reachable
#: entry is exact in ``int64``.
_MERGE_REWARD = np.zeros(MAX_SAFE_MERGE_EXPONENT + 1, dtype=np.int64)
for _exponent in range(MAX_SAFE_MERGE_EXPONENT + 1):
    _MERGE_REWARD[_exponent] = np.int64(1) << np.int64(_exponent + 1)
del _exponent

# --------------------------------------------------------------------------- #
# Checked int64 reward arithmetic
# --------------------------------------------------------------------------- #
#: Largest value representable in ``numpy.int64``.  Every M1 reward is a sum of
#: non-negative terms, so this is the only limit that has to be checked.
_INT64_MAX: np.int64 = np.iinfo(np.int64).max


def _checked_add_nonnegative_int64(
    total: np.ndarray,
    increment: np.ndarray,
    *,
    context: str,
) -> np.ndarray:
    """Add two arrays of non-negative ``int64`` values, refusing to wrap.

    Both operands must be ``int64`` arrays holding only values ``>= 0``; the
    result is likewise non-negative.  The sum is verified **before** it is
    computed::

        increment > (_INT64_MAX - total)   ->   OverflowError

    ``np.errstate(over="raise")`` is deliberately *not* used anywhere for this.
    NumPy only signals integer overflow for **scalar** operations; whole-array
    additions and integer reductions such as ``.sum(axis=1)`` wrap silently:

    >>> a = np.array([2 ** 62, 2 ** 62], dtype=np.int64)
    >>> with np.errstate(over="raise"):
    ...     a.sum()
    -9223372036854775808          # no exception

    The check is a single vectorized comparison, so it costs one pass over a
    ``(R,)`` array and adds no per-board Python work.

    Parameters
    ----------
    total:
        Running ``int64`` sum, every element ``>= 0``.
    increment:
        ``int64`` values to add, every element ``>= 0``, broadcastable to
        ``total``.
    context:
        Short description of the aggregation site, used in the error message.

    Returns
    -------
    np.ndarray
        ``total + increment`` as ``int64``, exact.  The inputs are never
        modified.

    Raises
    ------
    OverflowError
        The true mathematical sum does not fit in ``int64``.
    """
    if np.any(increment > (_INT64_MAX - total)):
        raise OverflowError(
            f"{context}: the true reward does not fit in numpy.int64 "
            f"(max {_INT64_MAX}); M1 refuses to emit a silently wrapped reward"
        )
    return total + increment


def _pack_left_rows(rows: np.ndarray) -> np.ndarray:
    """Move every non-zero tile to the left of its row, across the whole batch.

    ``rows`` is ``(R, 4)`` ``uint8``; the result is a new ``(R, 4)`` ``uint8``
    array with the same relative order of the non-zero tiles and all zeros on the
    right.  ``[0,1,0,2] -> [1,2,0,0]``, ``[0,0,0,5] -> [5,0,0,0]``.

    Implemented by computing, for every cell, its **rank among the non-zero cells
    of its row**, and then scattering the non-zero values into the positions
    ``(row, rank)``.  That is exact in one pass; shifting cell by cell would need
    up to three passes and is easy to get wrong for rows like ``[0,1,0,2]``.

    All operations are array-wide: the number of Python iterations is the fixed
    constant :data:`BOARD_COLUMNS`, never ``R``.
    """
    non_zero = rows != 0
    rank = np.cumsum(non_zero, axis=1, dtype=np.intp) - 1
    packed = np.zeros_like(rows)
    # ``np.nonzero`` returns the (row, column) coordinates of the non-zero cells;
    # each one is written to column ``rank`` of its own row.  The destination is
    # flattened by the boolean mask, so it is reshaped back afterwards.
    packed[
        np.nonzero(non_zero)[0], rank[non_zero]
    ] = rows.reshape(-1)[non_zero.reshape(-1)]
    return packed.reshape(rows.shape)


def _merge_left_rows(
    packed: np.ndarray, *, compute_rewards: bool
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Merge equal adjacent tiles towards column 0, across the whole batch.

    ``packed`` must already be pack-left (all zeros on the right).  Returns
    ``(merged, reward)``:

    * ``merged``: ``(R, 4)`` ``uint8``, the merged and re-packed rows;
    * ``reward``: ``(R,)`` ``int64``, the exact merge reward of each row, or
      ``None`` when ``compute_rewards`` is false.

    ``compute_rewards=False`` is the **movement-only** mode used by
    :func:`legal_mask_batch` and :func:`is_terminal_batch`.  The merge itself is
    performed exactly as in reward mode -- the same frozen rule, the same
    re-packing -- but the :data:`_MERGE_REWARD` lookup and the checked row
    aggregation are skipped entirely, because legality asks "did the board
    change" and never "does the reward fit in ``int64``".  Nothing is emulated
    with Python ``int`` or ``dtype=object``; the arithmetic is simply not needed.

    The frozen rule is applied with a per-row *one-merge-at-a-time* block mask:
    boundaries ``0``, ``1`` and ``2`` are visited left to right, and a row whose
    boundary just merged is blocked at the next boundary.  That is what makes
    ``[1,1,1,1] -> [2,2,0,0]`` instead of ``[2,1,1]``-style cascades.

    The caller must have run the audits first: the reward path indexes
    :data:`_MERGE_REWARD` with the pre-merge exponents, which is only defined
    below :data:`MAX_SAFE_MERGE_EXPONENT`, and both paths add one to an exponent,
    which is only defined below :data:`MAX_EXPONENT`.
    """
    work = np.array(packed, dtype=np.uint8, copy=True)
    row_count = work.shape[0]
    blocked = np.zeros(row_count, dtype=bool)
    reward = np.zeros(row_count, dtype=np.int64) if compute_rewards else None

    for _column in range(BOARD_COLUMNS - 1):
        can_merge = (
            (~blocked) & (work[:, _column] != 0) & (work[:, _column] == work[:, _column + 1])
        )
        if can_merge.any():
            left = work[:, _column]
            if compute_rewards:
                # A single row can merge at two different boundaries, so the *row
                # total* can exceed int64 even when every individual merge fits:
                # ``[61, 61, 61, 61]`` merges twice, and ``2**62 + 2**62 = 2**63``
                # does not fit.  The range check therefore runs **before** the
                # addition -- comparing against the post-check result would be too
                # late.  See :func:`_checked_add_nonnegative_int64` for why
                # ``np.errstate`` cannot be used for this.
                increment = _MERGE_REWARD[left[can_merge]]
                if np.any(increment > (_INT64_MAX - reward[can_merge])):
                    raise OverflowError(
                        "row merge reward aggregation: the true reward does not fit "
                        f"in numpy.int64 (max {_INT64_MAX}); M1 refuses to emit a "
                        "silently wrapped reward"
                    )
                reward[can_merge] += increment
            # ``left`` is a copy, so ``left + 1`` cannot wrap in the array; the
            # uint8 tile audit has already rejected any exponent that could.
            work[can_merge, _column] = left[can_merge] + np.uint8(1)
            work[can_merge, _column + 1] = 0
        blocked = can_merge

    return _pack_left_rows(work), reward


def _audit_tile_merge_overflow(packed: np.ndarray) -> None:
    """Reject any merge whose resulting *tile* does not fit in ``uint8``.

    ``255 + 255`` would need exponent ``256``.  M0 raises for that, so M1 must
    raise too and must never silently wrap to ``0``.

    This is part of the **tile movement semantics**, not a reward limit, so it
    runs in *both* movement modes -- including the movement-only mode behind
    :func:`legal_mask_batch` / :func:`is_terminal_batch`.

    Called only when the block contains an exponent of at least
    :data:`MAX_EXPONENT`, the one case in which this limit can be reached.
    """
    work = np.array(packed, dtype=np.uint8, copy=True)
    for _column in range(BOARD_COLUMNS - 1):
        left = work[:, _column]
        merging = (left != 0) & (left == work[:, _column + 1])
        if merging.any():
            exponent = int(left[merging].max())
            if exponent >= MAX_EXPONENT:
                raise OverflowError(
                    f"cannot merge two exponent-{exponent} tiles: the result "
                    f"exponent {exponent + 1} does not fit in numpy.uint8 "
                    f"(max {MAX_EXPONENT})"
                )
            # Consume the pair so the next boundary sees the same state as the
            # real merge would (a freshly created tile never merges again).
            work[merging, _column] = left[merging] + np.uint8(1)
            work[merging, _column + 1] = 0


def _audit_merge_reward_overflow(packed: np.ndarray) -> None:
    """Reject any merge whose *reward* does not fit in ``int64``.

    A single merge at exponent ``>= MAX_SAFE_MERGE_EXPONENT`` needs ``2 ** 63``
    or more.  M0 has no such limit (its reward is a Python ``int``), so this is a
    pure M1 representation limit -- which is exactly why it is checked only on
    the reward-producing path (:func:`move_batch`, :meth:`Fast2048BatchEnv.step`)
    and never on the legality path.

    Called only when the block contains an exponent of at least
    :data:`MAX_SAFE_MERGE_EXPONENT`.
    """
    work = np.array(packed, dtype=np.uint8, copy=True)
    for _column in range(BOARD_COLUMNS - 1):
        left = work[:, _column]
        merging = (left != 0) & (left == work[:, _column + 1])
        if merging.any():
            exponent = int(left[merging].max())
            if exponent >= MAX_SAFE_MERGE_EXPONENT:
                raise OverflowError(
                    f"merge of two exponent-{exponent} tiles needs a reward of "
                    f"2 ** {exponent + 1}, which does not fit in numpy.int64 "
                    f"(MAX_SAFE_MERGE_EXPONENT = {MAX_SAFE_MERGE_EXPONENT}); M1 "
                    f"refuses to emit a silently wrapped reward"
                )
            # Consume the pair so the next boundary sees the same state as the
            # real merge would (a freshly created tile never merges again).
            work[merging, _column] = left[merging] + np.uint8(1)
            work[merging, _column + 1] = 0


def _move_groups(
    boards: np.ndarray, group_actions: np.ndarray, *, compute_rewards: bool
):
    """Vectorized movement for one batch, optionally covering all four actions.

    ``group_actions`` is ``(N,)`` with values in ``0..3``.  Returns
    ``(afterstates, rewards, moved)`` shaped ``(N, 16)``, ``(N,)``, ``(N,)``;
    ``rewards`` is ``None`` when ``compute_rewards`` is false.

    There is exactly **one** movement kernel with two internal modes -- never a
    second movement implementation:

    * **reward-producing** (``compute_rewards=True``, used by :func:`move_batch`
      and :meth:`Fast2048BatchEnv.step`): the full ``int64`` reward contract,
      including the checked row and board aggregations;
    * **movement-only** (``compute_rewards=False``, used by
      :func:`legal_mask_batch` and :func:`is_terminal_batch`): identical packing
      and merging, and the identical ``uint8`` tile audit, but no reward
      arithmetic whatsoever.

    The distinction matters because legality is *defined* as "the move changed the
    board".  Whether the reward of that move happens to fit in ``int64`` is a
    separate, M1-specific question and must never turn a legal query into an
    ``OverflowError``.

    The batch is split into at most four action groups (four fixed iterations).
    Inside a group every board is processed at once: one gather to canonical
    order, one reshape to ``(R, 4)`` lines, vectorized pack/merge/pack, one
    scatter back.  No loop ever runs per board or per line.
    """
    board_count = boards.shape[0]
    afterstates = np.empty((board_count, CELL_COUNT), dtype=np.uint8)
    rewards = np.zeros(board_count, dtype=np.int64) if compute_rewards else None
    moved = np.zeros(board_count, dtype=bool)

    for action in range(4):
        selected = np.flatnonzero(group_actions == action)
        group_size = int(selected.size)
        if group_size == 0:
            continue
        group = boards[selected]

        # Canonical destination-first lines, four consecutive rows per board.
        # ``canonical_index`` is ``_LINE_ORDER[action]`` repeated once per board,
        # re-based on that board's own 16 cells.
        canonical_index = (
            _LINE_ORDER[action][None, :]
            + (np.arange(group_size, dtype=np.intp) * CELL_COUNT)[:, None]
        ).reshape(-1)
        rank = np.arange(group_size, dtype=np.intp)[:, None] * CELL_COUNT
        lines = group.reshape(-1)[canonical_index].reshape(-1, BOARD_COLUMNS)

        packed = _pack_left_rows(lines)
        # Both audits must run *before* the merge, because the merge itself indexes
        # the reward table with the pre-merge exponents and adds one to them.  The
        # uint8 tile audit is movement semantics and therefore runs in both modes;
        # the int64 reward audit is an M1 representation limit and runs only where
        # a reward is actually produced.  Each gate is ``>=``: an exponent of
        # exactly ``MAX_SAFE_MERGE_EXPONENT`` would already need a reward of
        # ``2 ** 63``, and two tiles of exactly ``MAX_EXPONENT`` would need
        # exponent ``256``.
        max_exponent = int(packed.max(initial=0))
        if max_exponent >= MAX_EXPONENT:
            _audit_tile_merge_overflow(packed)
        if compute_rewards and max_exponent >= MAX_SAFE_MERGE_EXPONENT:
            _audit_merge_reward_overflow(packed)

        merged, line_reward = _merge_left_rows(
            packed, compute_rewards=compute_rewards
        )

        if compute_rewards:
            # Lines ``4 * i .. 4 * i + 3`` all belong to board ``i`` of the group.
            # A board can merge on several lines at once, so the board total needs
            # the same checked aggregation as the row total: ``.sum(axis=1)`` would
            # wrap silently (see :func:`_checked_add_nonnegative_int64`).  The loop
            # is the fixed ``range(BOARD_COLUMNS)`` over the four lines of one
            # board, never over boards.  ``headroom`` is carried forward instead of
            # being recomputed, so each line costs one comparison and one add.
            line_rewards = line_reward.reshape(group_size, BOARD_COLUMNS)
            board_reward = np.zeros(group_size, dtype=np.int64)
            headroom = np.full(group_size, _INT64_MAX, dtype=np.int64)
            for column in range(BOARD_COLUMNS):
                line = line_rewards[:, column]
                if np.any(line > headroom):
                    raise OverflowError(
                        "board merge reward aggregation: the true reward does not fit "
                        f"in numpy.int64 (max {_INT64_MAX}); M1 refuses to emit a "
                        "silently wrapped reward"
                    )
                board_reward += line
                headroom -= line
            rewards[selected] = board_reward

        # Scatter the canonical board back to real board indices.  ``moved`` is
        # exactly "the board changed", so it is read straight off the afterstate.
        board_values = merged.reshape(-1)[
            (_LINE_INVERSE[action][None, :] + rank).reshape(-1)
        ].reshape(group_size, CELL_COUNT)
        afterstates[selected] = board_values
        moved[selected] = (board_values != group).any(axis=1)

    return afterstates, rewards, moved


@dataclass
class BatchMoveResult:
    """Result of :func:`move_batch` for a whole batch of boards.

    Attributes
    ----------
    afterstates:
        ``(N, 16)`` ``uint8`` board after sliding/merging, before any spawn.
    rewards:
        ``(N,)`` ``int64`` sum of all merge rewards of this move.
    moved:
        ``(N,)`` ``bool`` ``True`` iff the board actually changed.
    """

    afterstates: np.ndarray
    rewards: np.ndarray
    moved: np.ndarray


def move_batch(boards: np.ndarray, actions: np.ndarray) -> BatchMoveResult:
    """Apply one action per board, batched over the whole board array.

    Parameters
    ----------
    boards:
        ``(N, 16)`` ``uint8`` board batch.  Never modified.
    actions:
        ``(N,)`` integer array, every value in ``0..3``
        (``UP, DOWN, LEFT, RIGHT``).  Never modified.

    Returns
    -------
    BatchMoveResult
        ``afterstates`` ``(N, 16)`` ``uint8`` (freshly allocated, never aliasing
        the input), ``rewards`` ``(N,)`` ``int64``, ``moved`` ``(N,)`` ``bool``.

    Raises
    ------
    ValueError
        Wrong shapes, wrong dtypes (float/bool/str actions included), mismatched
        lengths, or actions outside ``0..3``.
    OverflowError
        The **true total** reward of one move cannot be represented exactly in
        ``numpy.int64`` (max ``9223372036854775807``).  Two distinct situations
        reach this, and both are checked **before** the offending addition, so a
        wrapped value is never produced:

        1. a single merge at exponent ``>= 62`` needs ``2 ** 63`` or more;
        2. several individually representable merges aggregate past the ceiling,
           either within one row (``[61, 61, 61, 61]`` -> ``2**62 + 2**62``) or
           across the lines of one board (two rows of ``[61, 61]``).

        ``np.errstate(over="raise")`` is deliberately not used for this: NumPy
        only signals *scalar* integer overflow, so array adds and reductions such
        as ``.sum(axis=1)`` wrap silently.
    """
    board_array = _as_boards(boards)
    action_array = _as_actions(actions, board_array.shape[0])
    return _move_batch(board_array, action_array)


def _move_batch(boards: np.ndarray, actions: np.ndarray) -> BatchMoveResult:
    """``move_batch`` core for already validated inputs.

    Delegates the whole batch to :func:`_move_groups` in **reward-producing**
    mode, which is fully vectorized across boards; the only Python loop in the
    movement path is the fixed four-iteration loop over actions.
    """
    afterstates, rewards, moved = _move_groups(
        boards, actions, compute_rewards=True
    )
    return BatchMoveResult(afterstates=afterstates, rewards=rewards, moved=moved)


def _move_all_actions_batch(boards: np.ndarray) -> np.ndarray:
    """``(N, 4)`` bool: "this action changed the board", for all four actions.

    Used by :func:`legal_mask_batch` and :func:`is_terminal_batch`, which are both
    defined purely in terms of "did the move change the board" -- so they share
    exactly the same vectorized movement core as :func:`move_batch` and never
    introduce a second legality rule.

    The core runs in **movement-only** mode on purpose.  Legality does not depend
    on the merge reward, so this path must not be able to raise ``OverflowError``
    because some hypothetical action's reward does not fit in ``int64``; that is a
    limit of M1's reward representation, not of the game.  The ``uint8`` tile audit
    still applies, since it is part of the movement semantics itself.
    """
    board_count = boards.shape[0]
    moved = np.zeros((board_count, 4), dtype=bool)
    for action in range(4):
        actions = np.full(board_count, action, dtype=np.uint8)
        _, _, action_moved = _move_groups(boards, actions, compute_rewards=False)
        moved[:, action] = action_moved
    return moved


# --------------------------------------------------------------------------- #
# Batch legal mask / terminal
# --------------------------------------------------------------------------- #


def legal_mask_batch(boards: np.ndarray) -> np.ndarray:
    """Boolean mask ``(N, 4)`` of legal actions, order ``UP, DOWN, LEFT, RIGHT``.

    Defined exactly as in M0 -- an action is legal iff its move changed the board::

        legal_mask_batch(s)[i, a] == move_without_spawn(s[i], a).moved

    There is deliberately **no** second, independent legality rule (no "has an
    empty cell" / "has an adjacent equal pair" shortcut).  All four moves are
    computed in full; correctness wins over the extra arithmetic.

    Legality is **independent of M1's int64 reward representation**: the shared
    movement core runs in movement-only mode here, so a board whose move would
    produce a reward beyond ``int64`` (for example ``[61, 61, 61, 61]``, whose
    move pays ``2 ** 63``) still answers the legality question normally instead of
    raising ``OverflowError``.  The ``uint8`` tile limit is *not* suspended: an
    actual ``255 + 255`` merge still raises, exactly as in M0.
    """
    board_array = _as_boards(boards)
    return _move_all_actions_batch(board_array)


def is_terminal_batch(boards: np.ndarray) -> np.ndarray:
    """``(N,)`` bool: ``True`` iff no action is legal (identical to M0).

    A full board that still contains two adjacent equal tiles is **not** terminal,
    so this is ``~legal_mask_batch(boards).any(axis=1)`` and never a "board is
    full" test.

    For the same reason as :func:`legal_mask_batch`, the terminal verdict never
    depends on whether some hypothetical action's reward would fit in ``int64``.
    """
    return ~legal_mask_batch(boards).any(axis=1)


# --------------------------------------------------------------------------- #
# Exact spawn enumeration
# --------------------------------------------------------------------------- #


@dataclass
class BatchSpawnEnumeration:
    """Every spawn outcome of a batch of afterstates, in CSR-style flat form.

    Attributes
    ----------
    states:
        ``(M, 16)`` ``uint8``: afterstate with exactly one tile added.
    probabilities:
        ``(M,)`` ``float64``: ``0.9 / n`` for exponent 1 and ``0.1 / n`` for
        exponent 2, where ``n`` is that parent's number of empty cells.
    parent_indices:
        ``(M,)`` ``int64``: row of the parent board inside the input batch.
    spawn_indices:
        ``(M,)`` ``int64``: flat cell index (``0..15``) receiving the tile.
    spawn_exponents:
        ``(M,)`` ``uint8``: ``1`` (tile 2) or ``2`` (tile 4).
    offsets:
        ``(N + 1,)`` ``int64``: outcome range of parent ``i`` is
        ``offsets[i] : offsets[i + 1]``; a full board yields an empty range.
    empty_counts:
        ``(N,)`` ``int64``: number of empty cells per parent.  Provided because it
        is the quantity that drives the output size (``2 * n`` outcomes).
    """

    states: np.ndarray
    probabilities: np.ndarray
    parent_indices: np.ndarray
    spawn_indices: np.ndarray
    spawn_exponents: np.ndarray
    offsets: np.ndarray
    empty_counts: np.ndarray


def enumerate_spawns_batch(afterstates: np.ndarray) -> BatchSpawnEnumeration:
    """Enumerate every spawn outcome of every afterstate, batched.

    Ordering per parent is frozen exactly as in M0's ``enumerate_spawns``: empty
    cells in ascending flat index, and for each cell first exponent ``1`` then
    exponent ``2``.  A parent with ``n`` empty cells therefore produces ``2 * n``
    outcomes (at most 32), and each parent's probabilities sum to ``1.0`` within
    floating-point tolerance.  A full board produces an empty range.

    Generate rows directly in parent order (one vectorized pass per cell
    position, never over boards), which is what makes the CSR ranges contiguous.
    Ordering per parent is frozen exactly as in M0's ``enumerate_spawns``: empty
    cells in ascending flat index, and for each cell first exponent ``1`` then
    exponent ``2``.

    The input is never modified.
    """
    boards = _as_boards(afterstates, name="afterstates")
    board_count = boards.shape[0]

    empty = boards == 0
    empty_counts = empty.sum(axis=1, dtype=np.int64)
    offsets = np.zeros(board_count + 1, dtype=np.int64)
    np.cumsum(empty_counts * 2, out=offsets[1:])
    total = int(offsets[-1])

    parent_indices = np.zeros(total, dtype=np.int64)
    spawn_indices = np.zeros(total, dtype=np.int64)
    spawn_exponents = np.zeros(total, dtype=np.uint8)
    probabilities = np.zeros(total, dtype=np.float64)
    states = np.zeros((total, CELL_COUNT), dtype=np.uint8)

    if total == 0:
        return BatchSpawnEnumeration(
            states=states,
            probabilities=probabilities,
            parent_indices=parent_indices,
            spawn_indices=spawn_indices,
            spawn_exponents=spawn_exponents,
            offsets=offsets,
            empty_counts=empty_counts,
        )

    # ``rank[i, c]`` numbers the empty cells of parent ``i``, so cell ``c`` is
    # that parent's ``rank``-th empty cell and its outcome rows are
    # ``offsets[i] + 2 * rank`` (exponent 1) and ``+1`` (exponent 2).
    rank = np.cumsum(empty, axis=1, dtype=np.int64) - 1
    first_row = offsets[:-1]

    for cell in range(CELL_COUNT):
        parents = np.flatnonzero(empty[:, cell])
        if parents.size == 0:
            continue
        slots = rank[parents, cell]
        index_one = first_row[parents] + 2 * slots
        index_two = index_one + 1
        counts = empty_counts[parents]

        parent_indices[index_one] = parents
        parent_indices[index_two] = parents
        spawn_indices[index_one] = cell
        spawn_indices[index_two] = cell
        spawn_exponents[index_one] = _SPAWN_EXPONENT_TILE_2
        spawn_exponents[index_two] = _SPAWN_EXPONENT_TILE_4
        probabilities[index_one] = _SPAWN_PROB_TILE_2 / counts
        probabilities[index_two] = (1.0 - _SPAWN_PROB_TILE_2) / counts
        states[index_one] = boards[parents]
        states[index_two] = boards[parents]
        states[index_one, cell] = _SPAWN_EXPONENT_TILE_2
        states[index_two, cell] = _SPAWN_EXPONENT_TILE_4

    return BatchSpawnEnumeration(
        states=states,
        probabilities=probabilities,
        parent_indices=parent_indices,
        spawn_indices=spawn_indices,
        spawn_exponents=spawn_exponents,
        offsets=offsets,
        empty_counts=empty_counts,
    )


# --------------------------------------------------------------------------- #
# Random spawn
# --------------------------------------------------------------------------- #


@dataclass
class BatchSpawnResult:
    """Result of :func:`spawn_random_batch`.

    Attributes
    ----------
    states:
        ``(N, 16)`` ``uint8``: afterstate with one random tile added.
    spawn_indices:
        ``(N,)`` ``int64``: flat cell index that received the tile.
    spawn_exponents:
        ``(N,)`` ``uint8``: ``1`` (tile 2) or ``2`` (tile 4).
    """

    states: np.ndarray
    spawn_indices: np.ndarray
    spawn_exponents: np.ndarray


def spawn_random_batch(
    afterstates: np.ndarray,
    rng: np.random.Generator,
) -> BatchSpawnResult:
    """Spawn one random tile per board, batched.

    Per board: choose one of its own empty cells uniformly, then place exponent
    ``1`` with probability ``0.9`` or exponent ``2`` with probability ``0.1`` --
    the frozen M0 spawn distribution.  The input is never modified.

    Exactly one ``(N, 2)`` draw block is consumed, so a seeded caller replays
    identically.

    Raises
    ------
    ValueError
        If any board has no empty cell; the message names the offending parent
        indices.
    TypeError
        If ``rng`` is not a ``numpy.random.Generator``.
    """
    boards = _as_boards(afterstates, name="afterstates")
    generator = _require_rng(rng)
    board_count = boards.shape[0]

    empty = boards == 0
    empty_counts = empty.sum(axis=1, dtype=np.int64)
    full = np.flatnonzero(empty_counts == 0)
    if full.size:
        preview = ", ".join(str(int(index)) for index in full[:8])
        more = "" if full.size <= 8 else f" (+{full.size - 8} more)"
        raise ValueError(
            f"cannot spawn on a board with no empty cell: parent indices "
            f"{preview}{more}"
        )

    draws = generator.random((board_count, 2))
    choices = np.minimum(
        (draws[:, 0] * empty_counts).astype(np.int64), empty_counts - 1
    )
    spawn_indices = _nth_empty_index(empty, choices)
    spawn_exponents = _spawn_exponents(draws[:, 1])

    states = boards.copy()
    states[np.arange(board_count), spawn_indices] = spawn_exponents
    return BatchSpawnResult(
        states=states,
        spawn_indices=spawn_indices,
        spawn_exponents=spawn_exponents,
    )


def _nth_empty_index(empty: np.ndarray, choices: np.ndarray) -> np.ndarray:
    """Flat index of the ``choices[i]``-th empty cell of every row of ``empty``.

    ``empty`` is a ``(N, 16)`` boolean "cell is empty" mask.  Rows are processed
    cell by cell with vectorized row selections, so no Python-level per-board work
    happens.
    """
    board_count = empty.shape[0]
    remaining = choices.astype(np.int64, copy=True)
    indices = np.zeros(board_count, dtype=np.int64)
    found = np.zeros(board_count, dtype=bool)
    for cell in range(CELL_COUNT):
        take = empty[:, cell] & (remaining == 0) & ~found
        indices[take] = cell
        found |= take
        remaining[empty[:, cell]] -= 1
    if not found.all():
        raise ValueError(
            "internal error: at least one board has no reachable empty cell"
        )
    return indices


def apply_spawn_batch(
    afterstates: np.ndarray,
    spawn_indices: np.ndarray,
    spawn_exponents: np.ndarray,
) -> np.ndarray:
    """Deterministically place one tile per board; no RNG, no side effects.

    This is the bridge that lets differential tests feed a *specific* M0
    ``SpawnOutcome`` into the fast path without involving any random stream.

    Parameters
    ----------
    afterstates:
        ``(N, 16)`` ``uint8`` boards, never modified.
    spawn_indices:
        ``(N,)`` integer flat cell indices in ``0..15``.
    spawn_exponents:
        ``(N,)`` integer exponents; only ``1`` (tile 2) and ``2`` (tile 4) are
        allowed, matching the frozen spawn rule.

    Returns
    -------
    numpy.ndarray
        A new ``(N, 16)`` ``uint8`` array.

    Raises
    ------
    ValueError
        Wrong shape/dtype/length, ``spawn_index`` outside ``0..15``, an exponent
        other than ``1`` or ``2``, or a target cell that is not empty.
    """
    boards = _as_boards(afterstates, name="afterstates")
    board_count = boards.shape[0]
    indices = _as_vector(
        spawn_indices,
        board_count,
        name="spawn_indices",
        dtype=np.int64,
        low=0,
        high=CELL_COUNT - 1,
    )
    exponents = _as_vector(
        spawn_exponents,
        board_count,
        name="spawn_exponents",
        dtype=np.uint8,
        low=_SPAWN_EXPONENT_TILE_2,
        high=_SPAWN_EXPONENT_TILE_4,
    )
    if exponents.size and int(exponents.max()) > _SPAWN_EXPONENT_TILE_4:
        raise ValueError("spawn_exponents may only be 1 (tile 2) or 2 (tile 4)")

    rows = np.arange(board_count)
    occupied = np.flatnonzero(boards[rows, indices] != 0)
    if occupied.size:
        preview = ", ".join(str(int(index)) for index in occupied[:8])
        more = "" if occupied.size <= 8 else f" (+{occupied.size - 8} more)"
        raise ValueError(
            f"spawn target cell must be empty, but board(s) {preview}{more} already "
            f"hold a tile at the requested cell"
        )

    states = boards.copy()
    states[rows, indices] = exponents
    return states


# --------------------------------------------------------------------------- #
# Batch environment
# --------------------------------------------------------------------------- #


@dataclass
class BatchStepResult:
    """Result of :meth:`Fast2048BatchEnv.step` for the whole batch.

    Attributes
    ----------
    states:
        ``(N, 16)`` ``uint8`` official board after the step (spawn included).
    afterstates:
        ``(N, 16)`` ``uint8`` board after the move, before the spawn.  For an
        illegal action this equals the previous state.
    rewards:
        ``(N,)`` ``int64`` merge reward of this step (``0`` for illegal moves).
    legal:
        ``(N,)`` ``bool`` whether the requested action was legal.
    terminated:
        ``(N,)`` ``bool`` terminal status of ``states``.
    spawn_indices:
        ``(N,)`` ``int64`` cell that received the tile, ``-1`` when no spawn
        happened.
    spawn_exponents:
        ``(N,)`` ``uint8`` exponent placed, ``0`` when no spawn happened.
    """

    states: np.ndarray
    afterstates: np.ndarray
    rewards: np.ndarray
    legal: np.ndarray
    terminated: np.ndarray
    spawn_indices: np.ndarray
    spawn_exponents: np.ndarray


class Fast2048BatchEnv:
    """``N`` independent 2048 games inside one object, with no per-game objects.

    The persistent **core game state** is three buffers::

        _boards : (N, 16) uint8, C-contiguous
        _scores : (N,)    int64
        _rng    : numpy.random.Generator

    The class additionally holds performance scratch buffers that are shared by
    the whole batch -- ``_empty_prefix`` (spawn prefix sums) and ``_rows_buffer``
    (reusable row indices).  They are plain arrays, never independent
    per-environment Python objects, and they carry no game semantics.

    It never creates ``Reference2048Env`` instances (nor any other per-game
    object); every transition is produced by the batch functions above.

    Semantics are M0's, evaluated for a whole batch at once:

    * ``reset`` starts every game from an empty board, sets ``score = 0`` and
      performs exactly two real spawns;
    * ``reset_where(mask)`` resets only the selected games -- the asynchrony
      primitive needed for continuous self-play;
    * ``step`` moves, spawns only on legal actions, adds the reward to the score
      and then evaluates terminal;
    * an illegal action leaves board, score and RNG untouched
      (``spawn_index = -1``, ``spawn_exponent = 0``);
    * ``step`` never auto-resets; the caller decides via :meth:`reset_where`;
    * ``step`` is **atomic** with respect to the ``int64`` reward/score range
      checks: they all run before the first state mutation, so an
      ``OverflowError`` leaves board, scores and RNG exactly as they were.
    """

    def __init__(self, num_envs: int, seed: Optional[int] = None) -> None:
        if isinstance(num_envs, bool) or not isinstance(num_envs, (int, np.integer)):
            raise TypeError(f"num_envs must be an int, got {type(num_envs).__name__}")
        if int(num_envs) <= 0:
            raise ValueError(f"num_envs must be positive, got {int(num_envs)}")

        self._num_envs = int(num_envs)
        self._boards = np.zeros((self._num_envs, CELL_COUNT), dtype=np.uint8)
        self._scores = np.zeros(self._num_envs, dtype=np.int64)
        self._rng = np.random.Generator(np.random.PCG64(seed))

        # Scratch reused by every spawn; only ``[:active]`` is ever touched, so
        # these are constant allocations regardless of the spawn-group count.
        self._empty_prefix = np.zeros((self._num_envs, CELL_COUNT + 1), dtype=np.int64)
        self._rows_buffer = np.empty(self._num_envs, dtype=np.int64)

    # -- state access ------------------------------------------------------ #

    @property
    def num_envs(self) -> int:
        """Number of parallel games."""
        return self._num_envs

    @property
    def boards(self) -> np.ndarray:
        """Read-only live view ``(N, 16)`` ``uint8`` of the internal board buffer.

        No per-environment Python object is created and nothing is copied.  The
        view is write-protected so external code cannot corrupt internal state;
        use :meth:`copy_boards` for an independent, writable snapshot.
        """
        view = self._boards.view()
        view.flags.writeable = False
        return view

    @property
    def scores(self) -> np.ndarray:
        """Read-only live view ``(N,)`` ``int64`` of the accumulated scores.

        "Live" is a real guarantee, not a figure of speech: the view aliases the
        environment's own backing array, and ``step`` / ``reset`` /
        ``reset_where`` all update that array **in place**.  A view taken here
        therefore keeps reflecting the current scores for the whole lifetime of
        the environment.  The view itself is write-protected; use
        :meth:`copy_scores` for an independent, writable snapshot.
        """
        view = self._scores.view()
        view.flags.writeable = False
        return view

    @property
    def rng(self) -> np.random.Generator:
        """The environment's own generator (exposed for checkpointing)."""
        return self._rng

    def copy_boards(self) -> np.ndarray:
        """Independent, writable ``(N, 16)`` ``uint8`` copy of the boards."""
        return self._boards.copy()

    def copy_scores(self) -> np.ndarray:
        """Independent, writable ``(N,)`` ``int64`` copy of the scores."""
        return self._scores.copy()

    # -- reset ------------------------------------------------------------- #

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset **all** games and return a fresh ``(N, 16)`` ``uint8`` board copy.

        ``seed`` re-seeds the generator, so the result depends only on the seed;
        ``seed=None`` continues the current random stream, like M0.  Every game
        starts from an empty board, gets exactly two real spawns and
        ``score = 0``.
        """
        if seed is not None:
            self._rng = np.random.Generator(np.random.PCG64(seed))
        self._scores[:] = 0
        self._boards[:] = 0
        self._spawn_group(self._all_rows(), INITIAL_TILE_COUNT)
        return self._boards.copy()

    def reset_where(self, mask: np.ndarray) -> None:
        """Reset only the games selected by ``mask`` (shape ``(N,)``, dtype bool).

        Boards and scores of unselected games are left bit-for-bit untouched.
        This is the primitive that makes fully asynchronous continuous self-play
        possible: there is no need to wait until every game is dead.
        """
        selection = np.asarray(mask)
        if selection.shape != (self._num_envs,):
            raise ValueError(
                f"mask must have shape ({self._num_envs},), got {selection.shape}"
            )
        if selection.dtype != np.bool_:
            raise ValueError(f"mask dtype must be bool, got {selection.dtype}")
        rows = np.flatnonzero(selection)
        if rows.size == 0:
            return
        self._scores[rows] = 0
        self._boards[rows] = 0
        self._spawn_group(rows, INITIAL_TILE_COUNT)

    # -- stepping ---------------------------------------------------------- #

    def step(self, actions: np.ndarray) -> BatchStepResult:
        """Advance every game by one action, exactly like M0's ``step``.

        Illegal actions are a no-op: the board is unchanged, the reward is ``0``,
        the score is unchanged, no spawn happens, no randomness is consumed, and
        ``spawn_indices`` / ``spawn_exponents`` are ``-1`` / ``0``.  Terminal games
        are *not* auto-reset; the caller decides via :meth:`reset_where`.

        Atomicity: every ``int64`` range check of a step happens **before** the
        first state mutation.  If the merge aggregation or the score update cannot
        be represented in ``int64``, ``OverflowError`` is raised while the board,
        the scores and the RNG are all still untouched -- a step either happens
        completely or not at all.

        The terminal calculation that runs *after* the spawn cannot reintroduce a
        failure of that kind: it is a legality query, and legality is computed by
        the movement-only path, which performs no reward arithmetic.  A step is
        therefore never aborted after the spawn by an ``int64`` reward limit of
        some hypothetical alternative action.  (That is a statement about the
        reward checks only; it is not a blanket "nothing can fail after a spawn".)

        Performance note: ``is_terminal_batch`` is a legality query served by the
        movement-only path, so it stays valid on boards whose rewards would not fit
        in ``int64``.
        """
        action_array = _as_actions(actions, self._num_envs)
        move = _move_batch(self._boards, action_array)
        moved = move.moved

        rewards = np.where(moved, move.rewards, np.int64(0)).astype(
            np.int64, copy=False
        )

        # Range-check the score update *before* the spawn commits anything.  The
        # check must not be hoisted after ``_spawn_group``: that call both writes
        # the board buffer and consumes randomness.
        new_scores = _checked_add_nonnegative_int64(
            self._scores,
            rewards,
            context="score accumulation",
        )

        spawn_indices = np.full(self._num_envs, -1, dtype=np.int64)
        spawn_exponents = np.zeros(self._num_envs, dtype=np.uint8)
        rows = np.flatnonzero(moved)
        if rows.size:
            indices, exponents = self._spawn_group(
                rows, 1, afterstate=move.afterstates
            )
            spawn_indices[rows] = indices
            spawn_exponents[rows] = exponents

        # Commit in place: ``self.scores`` is documented as a *live* view, so the
        # backing ndarray object must survive the step.  Rebinding ``self._scores``
        # to the fresh array would silently leave every previously handed-out view
        # pointing at the stale scores.
        self._scores[:] = new_scores

        # M0 evaluates terminal on ``s'``, the board *after* the spawn, so the
        # status is computed here on the current board buffer.
        terminated = is_terminal_batch(self._boards)

        return BatchStepResult(
            states=self._boards.copy(),
            afterstates=move.afterstates,
            rewards=rewards,
            legal=moved.copy(),
            terminated=terminated.copy(),
            spawn_indices=spawn_indices,
            spawn_exponents=spawn_exponents,
        )

    # -- spawn plumbing ---------------------------------------------------- #

    def _all_rows(self) -> np.ndarray:
        """``0..N-1`` as an ``int64`` array (reused buffer, no allocation)."""
        rows = self._rows_buffer[: self._num_envs]
        rows[:] = np.arange(self._num_envs, dtype=np.int64)
        return rows

    def _spawn_group(
        self,
        rows: np.ndarray,
        count: int,
        afterstate: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Spawn ``count`` tiles for each row of ``rows`` and commit them.

        ``afterstate`` is the board the spawns are applied to (defaults to the
        internal board); it is never modified.  Returns the first spawn's
        ``(indices, exponents)`` for ``rows``, which is what ``step`` reports.

        Sum over the spawn attempts of ``2 * count * num_envs`` random values.
        Drawing the *full* batch width (not just the legal rows) is deliberate: it
        makes the generator position depend only on the number of steps taken, so
        an illegal action truly cannot shift the stream and
        "illegal then legal" is bit-for-bit identical to "legal".
        """
        active = rows.size
        if active == 0:
            # No legal row => no spawn => **no randomness at all** is consumed,
            # exactly like M0's illegal action.
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.uint8)

        # Draw exactly two values per (row, attempt), indexed by row, so each
        # environment's draw depends only on *its own* spawn count.  An illegal
        # action therefore never contributes a draw, and a fully illegal step
        # leaves the generator state bit-for-bit unchanged.
        draws = self._rng.random((active, count, 2))
        # Work on a copy of the selected rows so the caller's array is never
        # modified: for ``step`` the caller's array is the freshly allocated
        # afterstate that is also returned to the user.
        scratch = (
            self._boards[rows].copy() if afterstate is None else afterstate[rows].copy()
        )
        first_indices = np.zeros(active, dtype=np.int64)
        first_exponents = np.zeros(active, dtype=np.uint8)

        for attempt in range(count):
            indices, exponents = self._place_tiles(
                scratch, draws[:, attempt], rows
            )
            if attempt == 0:
                first_indices = indices
                first_exponents = exponents
            self._boards[rows] = scratch

        return first_indices, first_exponents

    def _place_tiles(
        self,
        scratch: np.ndarray,
        draws: np.ndarray,
        rows: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Place one tile per scratch row (already copied); returns the draws used."""
        active = scratch.shape[0]
        empty = scratch == 0
        empty_counts = empty.sum(axis=1, dtype=np.int64)

        missing = np.flatnonzero(empty_counts == 0)
        if missing.size:
            offenders = rows[missing]
            preview = ", ".join(str(int(index)) for index in offenders[:8])
            more = "" if offenders.size <= 8 else f" (+{offenders.size - 8} more)"
            raise ValueError(
                f"cannot spawn on a board with no empty cell: parent indices "
                f"{preview}{more}"
            )

        choices = np.minimum(
            (draws[:, 0] * empty_counts).astype(np.int64), empty_counts - 1
        )
        prefix = self._empty_prefix[:active]
        prefix[:, 0] = 0
        np.cumsum(empty, axis=1, out=prefix[:, 1:])

        indices = np.zeros(active, dtype=np.int64)
        for cell in range(CELL_COUNT):
            take = empty[:, cell] & (choices == prefix[:, cell])
            if take.any():
                hit = np.flatnonzero(take)
                indices[hit] = cell
                scratch[hit, cell] = _spawn_exponents(draws[hit, 1])
        return indices, _spawn_exponents(draws[:, 1])
