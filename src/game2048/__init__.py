"""``game2048`` -- M0 reference implementation of the 2048 game rules.

The public surface of milestone M0 is:

* the frozen :class:`Action` numbering (``UP=0, DOWN=1, LEFT=2, RIGHT=3``);
* the pure movement core :func:`move_without_spawn`;
* the derived predicates :func:`legal_mask` and :func:`is_terminal`;
* the exact spawn distribution :func:`enumerate_spawns` and the RNG-driven
  :func:`spawn_random`;
* the reference environment :class:`Reference2048Env`
  (``Reference2048Env`` = correctness reference);
* the eight D4 transforms :func:`transform_board`, :func:`transform_action`
  and :func:`inverse_transform_id`.

Nothing from later milestones (networks, replay, teacher, expectimax, trainer,
self-play, ...) lives here, by design.
"""

from .reference_env import (
    ACTION_COUNT,
    ACTIONS,
    BOARD_SHAPE,
    BOARD_SIDE,
    CELL_COUNT,
    DIRECTION_VECTORS,
    EMPTY_EXPONENT,
    INITIAL_TILE_COUNT,
    MAX_EXPONENT,
    SPAWN_EXPONENT_TILE_2,
    SPAWN_EXPONENT_TILE_4,
    SPAWN_PROB_TILE_2,
    SPAWN_PROB_TILE_4,
    Action,
    MoveResult,
    Reference2048Env,
    SpawnOutcome,
    StepResult,
    enumerate_spawns,
    is_terminal,
    legal_mask,
    move_without_spawn,
    spawn_random,
)
from .symmetry import (
    TRANSFORM_COUNT,
    inverse_transform_id,
    transform_action,
    transform_board,
)

__all__ = [
    # constants
    "ACTION_COUNT",
    "ACTIONS",
    "BOARD_SHAPE",
    "BOARD_SIDE",
    "CELL_COUNT",
    "DIRECTION_VECTORS",
    "EMPTY_EXPONENT",
    "INITIAL_TILE_COUNT",
    "MAX_EXPONENT",
    "SPAWN_EXPONENT_TILE_2",
    "SPAWN_EXPONENT_TILE_4",
    "SPAWN_PROB_TILE_2",
    "SPAWN_PROB_TILE_4",
    "TRANSFORM_COUNT",
    # actions and value types
    "Action",
    "MoveResult",
    "SpawnOutcome",
    "StepResult",
    # reference environment
    "Reference2048Env",
    # pure core
    "enumerate_spawns",
    "is_terminal",
    "legal_mask",
    "move_without_spawn",
    "spawn_random",
    # D4 symmetry
    "inverse_transform_id",
    "transform_action",
    "transform_board",
]
