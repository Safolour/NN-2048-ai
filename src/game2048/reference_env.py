"""2048 reference environment (milestone M0).

``Reference2048Env`` = correctness reference
============================================

This module is the **golden reference / correctness oracle** for 2048.  Later
milestones (M1) may add ``Fast2048Env`` / ``Vectorized2048Env`` / C++ / CUDA
backends, but every one of them must reproduce the semantics defined here.

Everything in this module is deliberately simple, explicit and auditable.  It is
*not* optimised for speed: bitboard tricks, lookup tables, SIMD and CUDA belong
to M1 and must never leak into this file.

Frozen facts (mirrored in ``docs/M0_ENVIRONMENT_SPEC.md``)
---------------------------------------------------------
* Board representation: ``numpy.ndarray`` with ``shape == (16,)`` and
  ``dtype == numpy.uint8``, row-major order::

      0   1   2   3
      4   5   6   7
      8   9  10  11
      12 13  14  15

* Each cell stores the *exponent* of the tile: ``0`` = empty, ``1`` = 2,
  ``2`` = 4, ``3`` = 8, ... ``20`` = 1048576.  The environment stores the real
  exponent and **never** clamps it to the network's overflow bucket; that
  encoding is a later (network) concern.
* Actions are fixed: ``UP = 0``, ``DOWN = 1``, ``LEFT = 2``, ``RIGHT = 3``.
* Merging two exponent-``e`` tiles gives exponent ``e + 1`` and reward
  ``2 ** (e + 1)`` computed as a Python ``int`` (never a ``uint8``).
* ``legal_mask`` is defined *exclusively* through ``move_without_spawn``.
* ``is_terminal`` is defined *exclusively* as "no action is legal".

The dataclasses in this module override ``__eq__`` so that board fields are
compared with ``numpy.array_equal`` instead of producing ambiguous arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

#: Shape of the canonical board array.
BOARD_SHAPE: Tuple[int, ...] = (16,)
#: Side length of the (square) board.
BOARD_SIDE: int = 4
#: Number of cells on the board.
CELL_COUNT: int = 16
#: Number of actions.
ACTION_COUNT: int = 4

#: Exponent stored in an empty cell.
EMPTY_EXPONENT: int = 0
#: Largest exponent representable by ``numpy.uint8``.
MAX_EXPONENT: int = 255

#: Probability that a spawn produces a tile-2 (exponent 1).
SPAWN_PROB_TILE_2: float = 0.9
#: Probability that a spawn produces a tile-4 (exponent 2).
SPAWN_PROB_TILE_4: float = 0.1
#: Exponent spawned with probability :data:`SPAWN_PROB_TILE_2`.
SPAWN_EXPONENT_TILE_2: int = 1
#: Exponent spawned with probability :data:`SPAWN_PROB_TILE_4`.
SPAWN_EXPONENT_TILE_4: int = 2

#: Number of tiles (2) placed by :meth:`Reference2048Env.reset`.
INITIAL_TILE_COUNT: int = 2


class Action(IntEnum):
    """The four 2048 actions.  The numbering is frozen and globally valid.

    A later Q head with a ``[4]`` output is interpreted as
    ``[UP, DOWN, LEFT, RIGHT]``.
    """

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


#: Actions in canonical order ``UP, DOWN, LEFT, RIGHT``.
ACTIONS: Tuple[Action, ...] = (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT)

#: Direction vector of each action.  ``row`` grows downwards, ``col`` grows to
#: the right, hence ``UP = (-1, 0)``.
DIRECTION_VECTORS = {
    Action.UP: (-1, 0),
    Action.DOWN: (1, 0),
    Action.LEFT: (0, -1),
    Action.RIGHT: (0, 1),
}


# --------------------------------------------------------------------------- #
# Input validation / coercion helpers
# --------------------------------------------------------------------------- #


def _as_cell_list(board) -> List[int]:
    """Return ``board`` as a list of 16 Python ``int`` exponents.

    The caller's array is never modified.  Raises ``ValueError`` for a wrong
    shape, a non-integer dtype or an out-of-range exponent.
    """
    arr = np.asarray(board)
    if arr.shape != BOARD_SHAPE:
        raise ValueError(
            f"board must have shape {BOARD_SHAPE}, got {arr.shape}"
        )
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(
            f"board dtype must be an integer type, got {arr.dtype}"
        )
    cells = [int(value) for value in arr.tolist()]
    for value in cells:
        if value < 0 or value > MAX_EXPONENT:
            raise ValueError(
                f"exponent {value} is outside the representable range "
                f"0..{MAX_EXPONENT}"
            )
    return cells


def _coerce_action(action) -> Action:
    """Return ``action`` as an :class:`Action`, accepting plain ints too."""
    if isinstance(action, Action):
        return action
    try:
        return Action(int(action))
    except (TypeError, ValueError):
        raise ValueError(
            f"action must be one of {[int(a) for a in ACTIONS]} "
            f"(UP, DOWN, LEFT, RIGHT); got {action!r}"
        ) from None


# --------------------------------------------------------------------------- #
# Result containers
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, eq=False)
class MoveResult:
    """Result of :func:`move_without_spawn`.

    Attributes
    ----------
    afterstate:
        Board after sliding/merging, **before** any random spawn.
    reward:
        Sum of all merge rewards produced by this move (Python ``int``).
    moved:
        ``True`` iff the board actually changed.
    """

    afterstate: np.ndarray
    reward: int
    moved: bool

    def __eq__(self, other) -> bool:
        if not isinstance(other, MoveResult):
            return NotImplemented
        return (
            self.reward == other.reward
            and self.moved == other.moved
            and np.array_equal(self.afterstate, other.afterstate)
        )

    __hash__ = None  # type: ignore[assignment]


@dataclass(frozen=True, eq=False)
class SpawnOutcome:
    """One possible result of a spawn on an afterstate.

    Attributes
    ----------
    state:
        Board with a single new tile added.
    probability:
        Probability of this outcome.  ``0.9 / n`` for a tile-2 and ``0.1 / n``
        for a tile-4, where ``n`` is the number of empty cells.
    spawn_index:
        Flat row-major index (``0..15``) of the cell that received the tile.
    spawn_exponent:
        Exponent placed at ``spawn_index`` (``1`` for tile 2, ``2`` for tile 4).
    """

    state: np.ndarray
    probability: float
    spawn_index: int
    spawn_exponent: int

    def __eq__(self, other) -> bool:
        if not isinstance(other, SpawnOutcome):
            return NotImplemented
        return (
            self.probability == other.probability
            and self.spawn_index == other.spawn_index
            and self.spawn_exponent == other.spawn_exponent
            and np.array_equal(self.state, other.state)
        )

    __hash__ = None  # type: ignore[assignment]


@dataclass(frozen=True, eq=False)
class StepResult:
    """Result of :meth:`Reference2048Env.step`.

    ``afterstate`` and ``state`` are never the same thing for a legal action:
    ``afterstate`` is the board before the spawn, ``state`` is the official
    board after the spawn.  For an illegal action the board is unchanged, so
    ``afterstate == state``, ``reward == 0`` and the spawn fields are ``None``.
    """

    state: np.ndarray
    afterstate: np.ndarray
    reward: int
    legal: bool
    terminated: bool
    spawn_index: Optional[int]
    spawn_exponent: Optional[int]

    def __eq__(self, other) -> bool:
        if not isinstance(other, StepResult):
            return NotImplemented
        return (
            self.reward == other.reward
            and self.legal == other.legal
            and self.terminated == other.terminated
            and self.spawn_index == other.spawn_index
            and self.spawn_exponent == other.spawn_exponent
            and np.array_equal(self.state, other.state)
            and np.array_equal(self.afterstate, other.afterstate)
        )

    __hash__ = None  # type: ignore[assignment]


# --------------------------------------------------------------------------- #
# Pure movement core
# --------------------------------------------------------------------------- #


def _line_indices(action: Action) -> List[List[int]]:
    """Flat indices of the four lines of ``action``, destination cell first.

    Element ``0`` of each returned line is the cell the tiles move *towards*,
    so the merge routine only ever has to look at one direction.
    """
    lines: List[List[int]] = []
    for k in range(BOARD_SIDE):
        if action == Action.LEFT:
            lines.append([k * BOARD_SIDE + j for j in range(BOARD_SIDE)])
        elif action == Action.RIGHT:
            lines.append([k * BOARD_SIDE + j for j in range(BOARD_SIDE - 1, -1, -1)])
        elif action == Action.UP:
            lines.append([i * BOARD_SIDE + k for i in range(BOARD_SIDE)])
        else:  # Action.DOWN
            lines.append([i * BOARD_SIDE + k for i in range(BOARD_SIDE - 1, -1, -1)])
    return lines


def _merge_line(line: Sequence[int]) -> Tuple[List[int], int]:
    """Slide and merge one line of exponents; index ``0`` is the destination.

    Implements the frozen rule: drop blanks, scan from the destination side,
    merge adjacent equal tiles, let every tile merge at most once, pad with
    zeros.  A freshly created tile never merges again in the same action.
    """
    compact = [exponent for exponent in line if exponent != EMPTY_EXPONENT]

    merged_line: List[int] = []
    reward = 0
    position = 0
    while position < len(compact):
        exponent = compact[position]
        if position + 1 < len(compact) and compact[position + 1] == exponent:
            if exponent >= MAX_EXPONENT:
                raise OverflowError(
                    f"cannot merge two exponent-{exponent} tiles: the result "
                    f"exponent {exponent + 1} does not fit in numpy.uint8 "
                    f"(max {MAX_EXPONENT})"
                )
            new_exponent = exponent + 1
            merged_line.append(new_exponent)
            # Reward is a Python int, never a uint8.
            reward += 2 ** new_exponent
            position += 2
        else:
            merged_line.append(exponent)
            position += 1

    merged_line.extend([EMPTY_EXPONENT] * (BOARD_SIDE - len(merged_line)))
    return merged_line, reward


def move_without_spawn(board, action) -> MoveResult:
    """Apply ``action`` to ``board`` and return the afterstate.

    Pure and deterministic: no randomness, no spawn, the input is never
    modified and identical inputs always produce identical outputs.
    """
    action = _coerce_action(action)
    cells = _as_cell_list(board)

    afterstate_cells = list(cells)
    reward = 0
    for line_indices in _line_indices(action):
        line = [cells[index] for index in line_indices]
        merged_line, line_reward = _merge_line(line)
        reward += line_reward
        for position, index in enumerate(line_indices):
            afterstate_cells[index] = merged_line[position]

    afterstate = np.array(afterstate_cells, dtype=np.uint8)
    moved = afterstate_cells != cells
    return MoveResult(afterstate=afterstate, reward=int(reward), moved=moved)


# --------------------------------------------------------------------------- #
# Derived predicates (defined only in terms of move_without_spawn)
# --------------------------------------------------------------------------- #


def legal_mask(board) -> np.ndarray:
    """Boolean mask of legal actions, shape ``(4,)``, order ``UP, DOWN, LEFT, RIGHT``.

    An action is legal iff ``move_without_spawn(board, action).moved`` is true.
    No second, independent legality rule exists on purpose.
    """
    return np.array(
        [move_without_spawn(board, action).moved for action in ACTIONS],
        dtype=bool,
    )


def is_terminal(board) -> bool:
    """``True`` iff no action is legal.

    A full board that still contains two adjacent equal tiles is **not**
    terminal.
    """
    return not bool(legal_mask(board).any())


# --------------------------------------------------------------------------- #
# Spawning
# --------------------------------------------------------------------------- #


def _empty_indices(cells: Sequence[int]) -> List[int]:
    """Flat indices of all empty cells, in ascending order."""
    return [index for index, exponent in enumerate(cells) if exponent == EMPTY_EXPONENT]


def enumerate_spawns(afterstate) -> List[SpawnOutcome]:
    """Enumerate every possible spawn outcome of ``afterstate``.

    The cell is chosen uniformly among the ``n`` empty cells, then it receives a
    tile-2 with probability ``0.9`` or a tile-4 with probability ``0.1``::

        P(tile 2 at a given cell) = 0.9 / n
        P(tile 4 at a given cell) = 0.1 / n

    At most ``16 * 2 = 32`` outcomes are returned, ordered by cell index and
    then by exponent.  A full board yields an empty list.  The input is never
    modified.
    """
    cells = _as_cell_list(afterstate)
    empty = _empty_indices(cells)
    if not empty:
        return []

    count = len(empty)
    prob_tile_2 = SPAWN_PROB_TILE_2 / count
    prob_tile_4 = SPAWN_PROB_TILE_4 / count

    outcomes: List[SpawnOutcome] = []
    for index in empty:
        for exponent, probability in (
            (SPAWN_EXPONENT_TILE_2, prob_tile_2),
            (SPAWN_EXPONENT_TILE_4, prob_tile_4),
        ):
            spawned = list(cells)
            spawned[index] = exponent
            outcomes.append(
                SpawnOutcome(
                    state=np.array(spawned, dtype=np.uint8),
                    probability=probability,
                    spawn_index=index,
                    spawn_exponent=exponent,
                )
            )
    return outcomes


def spawn_random(afterstate, rng: np.random.Generator) -> SpawnOutcome:
    """Spawn one random tile on ``afterstate`` and return the outcome.

    The only randomness source is the supplied ``numpy.random.Generator``; the
    global ``numpy.random`` state and the ``random`` module are never touched.
    Exactly two draws are consumed: one uniform cell choice in ``0..n-1`` and
    one uniform float for the 90% / 10% tile-2 / tile-4 decision.

    The input board is never modified.  Raises ``ValueError`` for a full board
    (which cannot happen after a legal move) and ``TypeError`` if ``rng`` is not
    a ``numpy.random.Generator``.
    """
    if not isinstance(rng, np.random.Generator):
        raise TypeError(
            f"rng must be a numpy.random.Generator, got {type(rng).__name__}"
        )

    cells = _as_cell_list(afterstate)
    empty = _empty_indices(cells)
    count = len(empty)
    if count == 0:
        raise ValueError("cannot spawn on a board with no empty cell")

    position = int(rng.integers(0, count))
    if float(rng.random()) < SPAWN_PROB_TILE_2:
        exponent = SPAWN_EXPONENT_TILE_2
    else:
        exponent = SPAWN_EXPONENT_TILE_4

    index = empty[position]
    spawned = list(cells)
    spawned[index] = exponent
    probability = (
        SPAWN_PROB_TILE_2 if exponent == SPAWN_EXPONENT_TILE_2 else SPAWN_PROB_TILE_4
    ) / count

    return SpawnOutcome(
        state=np.array(spawned, dtype=np.uint8),
        probability=probability,
        spawn_index=index,
        spawn_exponent=exponent,
    )


# --------------------------------------------------------------------------- #
# Reference environment
# --------------------------------------------------------------------------- #


class Reference2048Env:
    """Reference 2048 environment == the correctness reference.

    The class is intentionally small and literal.  It follows exactly the
    frozen ``step`` pipeline::

        current board s
        -> move_without_spawn
        -> if illegal: board unchanged, reward 0, no spawn, no RNG consumed
        -> if legal:   afterstate x, merge reward r, one spawn -> s'
                       score += r
        -> terminal evaluated on s'

    Notes
    -----
    * ``__init__`` seeds the RNG and leaves the board empty; call
      :meth:`reset` to start a real game.  Consequently
      ``Reference2048Env(seed=s).reset()`` and
      ``Reference2048Env().reset(seed=s)`` produce the identical starting board.
    * ``reset(seed=...)`` re-seeds the RNG; ``reset()`` continues the current
      random stream (gymnasium convention).
    * :attr:`board` returns a copy, so external code can never mutate the
      internal state by accident.
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        if rng is not None and seed is not None:
            raise ValueError("pass either seed or rng, not both")
        if rng is not None:
            if not isinstance(rng, np.random.Generator):
                raise TypeError(
                    f"rng must be a numpy.random.Generator, got {type(rng).__name__}"
                )
            self._rng = rng
        else:
            self._rng = np.random.Generator(np.random.PCG64(seed))

        self._board = np.zeros(CELL_COUNT, dtype=np.uint8)
        self._score = 0

    # -- state access ------------------------------------------------------ #

    @property
    def board(self) -> np.ndarray:
        """Copy of the current official board (shape ``(16,)``, ``uint8``)."""
        return self._board.copy()

    @property
    def score(self) -> int:
        """Accumulated merge reward (Python ``int``, never ``uint8``)."""
        return self._score

    @property
    def rng(self) -> np.random.Generator:
        """The environment's own random generator (exposed for checkpointing)."""
        return self._rng

    # -- environment API --------------------------------------------------- #

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Start a standard new game and return the new board.

        The game starts from an all-empty board, ``score = 0``, and then exactly
        two tiles are spawned with the normal spawn rule (90% tile-2, 10%
        tile-4, second spawn restricted to the remaining empty cells).
        """
        if seed is not None:
            self._rng = np.random.Generator(np.random.PCG64(seed))

        self._board = np.zeros(CELL_COUNT, dtype=np.uint8)
        self._score = 0

        for _ in range(INITIAL_TILE_COUNT):
            self._board = spawn_random(self._board, self._rng).state

        return self._board.copy()

    def step(self, action) -> StepResult:
        """Apply ``action`` and return a :class:`StepResult`."""
        action = _coerce_action(action)
        current = self._board
        move = move_without_spawn(current, action)

        if not move.moved:
            # Illegal action: no spawn, no score change, no RNG consumption.
            unchanged = current.copy()
            return StepResult(
                state=unchanged.copy(),
                afterstate=unchanged,
                reward=0,
                legal=False,
                terminated=is_terminal(current),
                spawn_index=None,
                spawn_exponent=None,
            )

        spawned = spawn_random(move.afterstate, self._rng)
        self._board = spawned.state
        self._score = self._score + int(move.reward)

        return StepResult(
            state=self._board.copy(),
            afterstate=move.afterstate.copy(),
            reward=int(move.reward),
            legal=True,
            terminated=is_terminal(self._board),
            spawn_index=spawned.spawn_index,
            spawn_exponent=spawned.spawn_exponent,
        )

    def legal_mask(self) -> np.ndarray:
        """Boolean mask ``(4,)`` of the legal actions in the current state."""
        return legal_mask(self._board)

    def is_terminal(self) -> bool:
        """``True`` iff no action is legal in the current state."""
        return is_terminal(self._board)

    def __repr__(self) -> str:
        grid = self._board.reshape(BOARD_SIDE, BOARD_SIDE).tolist()
        return f"Reference2048Env(score={self._score}, board={grid})"
