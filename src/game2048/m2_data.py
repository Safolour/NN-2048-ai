"""M2 CPU batch boundary: normalise / validate a board batch for transfer.

This module is the **M2 Preflight data contract**.  It defines exactly one thing:
what a board batch must look like at the moment it leaves the CPU environment and
crosses the CPU -> GPU transfer boundary.

There is no GPU code here, and there is deliberately no ``torch`` import.  The
contract is expressed purely with ``numpy``, because the producer side of the
pipeline is the frozen M1 :class:`~game2048.fast_env.Fast2048BatchEnv`, which is
NumPy-only::

    Fast2048BatchEnv            (M1, frozen)
        |
        v
    prepare_board_batch_for_transfer   (this module: the explicit boundary)
        |
        v
    CPU -> GPU transfer          (M2 and later -- not implemented yet)

The formal contract
-------------------
A board batch at the transfer boundary is:

* type: ``numpy.ndarray``;
* shape: ``(N, 16)``, row-major 16-cell boards, ``N >= 1``;
* dtype: ``numpy.uint8`` (tile exponents, no negatives, no floats, no bools);
* layout: C-contiguous.

Why the boundary is explicit
----------------------------
The M1 production layout is already ``(N, 16)`` ``uint8`` C-contiguous, so the
normal path must be **free**: a correct batch is returned as the very same object,
never copied.  Only a genuinely non-contiguous input pays one conversion, and it
pays it here -- once, at a named boundary -- instead of paying an invisible copy
inside every network forward.

That is the rule the master plan states for this boundary: never let a
non-contiguous NumPy view trigger implicit copy / implicit re-layout /
unmeasurable data movement repeatedly in the training hot path.

Two refusals are deliberate and are the reason this function is strict:

* a wrong dtype is an error, not an ``astype``.  A silent conversion would hide a
  real dtype change (and its cost) in the hot path;
* a wrong shape is an error, not a reshape.  ``(16,)``, ``(N, 4, 4)``, ``(N, 15)``
  and ``(0, 16)`` are all rejected rather than repaired.

Caller input is never mutated: a contiguous batch is returned unchanged (same
object), a non-contiguous one is converted into a new array.

Explicitly **out of scope** for this module: ``torch``, ``torch.Tensor``, CUDA,
devices, pinned memory, ``non_blocking`` transfers, BF16 / FP16, the network
(Transformer / MLP), Q / V / A heads, forward passes and action selection.
"""

from __future__ import annotations

import numpy as np

from .fast_env import CELL_COUNT

__all__ = ["prepare_board_batch_for_transfer"]


def prepare_board_batch_for_transfer(boards: np.ndarray) -> np.ndarray:
    """Return ``boards`` as a transferable ``(N, 16)`` ``uint8`` C-contiguous batch.

    Parameters
    ----------
    boards:
        The batch produced by the environment.  Must be an array-like of shape
        ``(N, 16)`` with ``N >= 1`` and dtype ``uint8``.

    Returns
    -------
    numpy.ndarray
        * If ``boards`` is already C-contiguous: **the same object**, no copy.
          ``result is boards`` holds.
        * Otherwise: a new C-contiguous ``uint8`` array with the same shape and
          exactly the same values.  This is the only explicit copy this boundary
          is allowed to make.

    Raises
    ------
    TypeError
        If ``boards`` is not a ``numpy.ndarray`` (or array-like convertible to
        one), or if its dtype is not exactly ``uint8``.  A wrong dtype is never
        silently converted: ``int64``, ``int32``, ``float32``, ``float64`` and
        ``bool`` are all rejected.

    ValueError
        If the shape is not ``(N, 16)`` with ``N >= 1``.  A wrong shape is never
        silently reshaped: ``(16,)``, ``(N, 4, 4)``, ``(N, 15)``, ``(N, 17)`` and
        ``(0, 16)`` are all rejected.

    Notes
    -----
    The input array is never modified.
    """
    if not isinstance(boards, np.ndarray):
        raise TypeError(
            f"boards must be a numpy.ndarray, got {type(boards).__name__}"
        )

    if boards.dtype != np.uint8:
        raise TypeError(
            f"boards dtype must be exactly uint8 (no implicit conversion), "
            f"got {boards.dtype}"
        )

    if boards.ndim != 2 or boards.shape[1] != CELL_COUNT:
        raise ValueError(
            f"boards must have shape (N, {CELL_COUNT}) with N >= 1, got {boards.shape}"
        )
    if boards.shape[0] == 0:
        raise ValueError(
            f"boards must contain at least one board, got shape {boards.shape}"
        )

    if boards.flags.c_contiguous:
        # The M1 production layout: hand the caller's own object straight through.
        # No copy, no view, no dtype work -- this is the zero-cost path.
        return boards

    # Legal, but not in the production layout (for example a strided view such as
    # ``base[:, ::2]``).  Convert exactly once, here, at the named boundary.
    return np.ascontiguousarray(boards)
