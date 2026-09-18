"""M0 tests for the eight D4 transforms.

Three independent checks are applied to the action mapping:

1. a hand-written regression table (this file),
2. the movement-equivariance property ``T(move(s,a)) == move(T(s), T(a))``,
3. the group laws ``inverse(T)(T(s)) == s`` and ``T`` being a bijection.

The implementation derives the action mapping from direction vectors, so the
hand-written table is a genuinely independent oracle.
"""

import numpy as np
import pytest

from game2048 import (
    ACTIONS,
    Action,
    TRANSFORM_COUNT,
    inverse_transform_id,
    is_terminal,
    legal_mask,
    move_without_spawn,
    transform_action,
    transform_board,
)

# A board holding every exponent 0..15 exactly once, so any index mistake in a
# transform shows up immediately.
GRID = np.arange(16, dtype=np.uint8)

# Hand-computed images of GRID under the eight transforms.
EXPECTED_TRANSFORMS = {
    0: [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15],
    ],
    1: [  # rotate 90 clockwise: new[i][j] = old[3 - j][i]
        [12, 8, 4, 0],
        [13, 9, 5, 1],
        [14, 10, 6, 2],
        [15, 11, 7, 3],
    ],
    2: [  # rotate 180
        [15, 14, 13, 12],
        [11, 10, 9, 8],
        [7, 6, 5, 4],
        [3, 2, 1, 0],
    ],
    3: [  # rotate 270 clockwise: new[i][j] = old[j][3 - i]
        [3, 7, 11, 15],
        [2, 6, 10, 14],
        [1, 5, 9, 13],
        [0, 4, 8, 12],
    ],
    4: [  # mirror left-right
        [3, 2, 1, 0],
        [7, 6, 5, 4],
        [11, 10, 9, 8],
        [15, 14, 13, 12],
    ],
    5: [  # rotate 90 clockwise of the mirror
        [15, 11, 7, 3],
        [14, 10, 6, 2],
        [13, 9, 5, 1],
        [12, 8, 4, 0],
    ],
    6: [  # rotate 180 of the mirror
        [12, 13, 14, 15],
        [8, 9, 10, 11],
        [4, 5, 6, 7],
        [0, 1, 2, 3],
    ],
    7: [  # rotate 270 clockwise of the mirror
        [0, 4, 8, 12],
        [1, 5, 9, 13],
        [2, 6, 10, 14],
        [3, 7, 11, 15],
    ],
}

# Hand-derived action table: ACTION_TABLE[t][action] is the action to play on
# the transformed board.
ACTION_TABLE = {
    0: {Action.UP: Action.UP, Action.DOWN: Action.DOWN, Action.LEFT: Action.LEFT, Action.RIGHT: Action.RIGHT},
    1: {Action.UP: Action.RIGHT, Action.DOWN: Action.LEFT, Action.LEFT: Action.UP, Action.RIGHT: Action.DOWN},
    2: {Action.UP: Action.DOWN, Action.DOWN: Action.UP, Action.LEFT: Action.RIGHT, Action.RIGHT: Action.LEFT},
    3: {Action.UP: Action.LEFT, Action.DOWN: Action.RIGHT, Action.LEFT: Action.DOWN, Action.RIGHT: Action.UP},
    4: {Action.UP: Action.UP, Action.DOWN: Action.DOWN, Action.LEFT: Action.RIGHT, Action.RIGHT: Action.LEFT},
    5: {Action.UP: Action.RIGHT, Action.DOWN: Action.LEFT, Action.LEFT: Action.DOWN, Action.RIGHT: Action.UP},
    6: {Action.UP: Action.DOWN, Action.DOWN: Action.UP, Action.LEFT: Action.LEFT, Action.RIGHT: Action.RIGHT},
    7: {Action.UP: Action.LEFT, Action.DOWN: Action.RIGHT, Action.LEFT: Action.UP, Action.RIGHT: Action.DOWN},
}


def grid_of(board):
    return np.asarray(board, dtype=np.uint8).reshape(4, 4).tolist()


def random_board(rng, low=0, high=18):
    return rng.integers(low, high, size=16).astype(np.uint8)


# --------------------------------------------------------------------------- #
# transform_board: hand-computed regression vectors
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("transform_id", range(TRANSFORM_COUNT))
def test_transform_board_matches_hand_computed_vectors(transform_id):
    assert grid_of(transform_board(GRID, transform_id)) == EXPECTED_TRANSFORMS[transform_id]


@pytest.mark.parametrize("transform_id", range(TRANSFORM_COUNT))
def test_transform_board_shape_and_dtype(transform_id):
    result = transform_board(GRID, transform_id)
    assert result.shape == (16,)
    assert result.dtype == np.uint8


def test_identity_transform_is_a_no_op():
    board = np.arange(16, dtype=np.uint8)
    assert transform_board(board, 0).tolist() == board.tolist()


def test_rotations_compose():
    board = np.arange(16, dtype=np.uint8)
    once = transform_board(board, 1)
    twice = transform_board(once, 1)
    thrice = transform_board(twice, 1)
    assert twice.tolist() == transform_board(board, 2).tolist()
    assert thrice.tolist() == transform_board(board, 3).tolist()
    assert transform_board(thrice, 1).tolist() == board.tolist()


def test_mirror_is_an_involution():
    board = np.arange(16, dtype=np.uint8)
    once = transform_board(board, 4)
    assert transform_board(once, 4).tolist() == board.tolist()


def test_transforms_5_to_7_are_mirror_followed_by_rotation():
    board = np.arange(16, dtype=np.uint8)
    mirrored = transform_board(board, 4)
    for rotate_id in (1, 2, 3):
        assert (
            transform_board(board, 4 + rotate_id).tolist()
            == transform_board(mirrored, rotate_id).tolist()
        )


def test_transforms_are_permutations_of_the_cells():
    rng = np.random.Generator(np.random.PCG64(11))
    for _ in range(200):
        board = random_board(rng)
        for transform_id in range(TRANSFORM_COUNT):
            transformed = transform_board(board, transform_id)
            assert sorted(transformed.tolist()) == sorted(board.tolist())


def test_transform_board_does_not_modify_input():
    rng = np.random.Generator(np.random.PCG64(12))
    board = random_board(rng)
    before = board.copy()
    for transform_id in range(TRANSFORM_COUNT):
        transform_board(board, transform_id)
    assert np.array_equal(before, board)


def test_transform_board_result_is_independent():
    board = np.arange(16, dtype=np.uint8)
    for transform_id in range(TRANSFORM_COUNT):
        transformed = transform_board(board, transform_id)
        transformed[0] = 99
        assert board[0] == 0


def test_transform_board_rejects_bad_ids():
    for bad in (-1, 8, 100):
        with pytest.raises(ValueError):
            transform_board(GRID, bad)


# --------------------------------------------------------------------------- #
# transform_action: hand-written table + geometric consistency
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("transform_id", range(TRANSFORM_COUNT))
def test_transform_action_matches_hand_written_table(transform_id):
    expected = ACTION_TABLE[transform_id]
    for action in ACTIONS:
        assert transform_action(action, transform_id) == expected[action]


@pytest.mark.parametrize("transform_id", range(TRANSFORM_COUNT))
def test_transform_action_is_a_bijection(transform_id):
    images = [transform_action(action, transform_id) for action in ACTIONS]
    assert sorted(int(image) for image in images) == [0, 1, 2, 3]


def test_transform_action_with_identity_keeps_actions():
    for action in ACTIONS:
        assert transform_action(action, 0) == action


def test_transform_action_rejects_bad_ids():
    for bad in (-1, 8, 100):
        with pytest.raises(ValueError):
            transform_action(Action.UP, bad)


def test_transform_action_accepts_plain_ints():
    for raw in range(4):
        assert transform_action(raw, 3) == transform_action(Action(raw), 3)


# --------------------------------------------------------------------------- #
# inverse_transform_id
# --------------------------------------------------------------------------- #


def test_inverse_transform_ids_are_frozen():
    assert [inverse_transform_id(t) for t in range(TRANSFORM_COUNT)] == [0, 3, 2, 1, 4, 5, 6, 7]


@pytest.mark.parametrize("transform_id", range(TRANSFORM_COUNT))
def test_inverse_round_trips_on_random_boards(transform_id):
    rng = np.random.Generator(np.random.PCG64(1000 + transform_id))
    inverse = inverse_transform_id(transform_id)
    for _ in range(50):
        board = random_board(rng)
        round_tripped = transform_board(transform_board(board, transform_id), inverse)
        assert round_tripped.tolist() == board.tolist()


@pytest.mark.parametrize("transform_id", range(TRANSFORM_COUNT))
def test_inverse_is_symmetric(transform_id):
    assert inverse_transform_id(inverse_transform_id(transform_id)) == transform_id


def test_inverse_transform_id_rejects_bad_ids():
    for bad in (-1, 8):
        with pytest.raises(ValueError):
            inverse_transform_id(bad)


# --------------------------------------------------------------------------- #
# Core equivalence: T(move(s, a)) == move(T(s), T(a))
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("transform_id", range(TRANSFORM_COUNT))
@pytest.mark.parametrize("action", list(Action))
def test_movement_equivariance(transform_id, action):
    rng = np.random.Generator(np.random.PCG64(5000 + 10 * transform_id + int(action)))
    for _ in range(60):
        board = random_board(rng)
        transformed_action = transform_action(action, transform_id)

        original = move_without_spawn(board, action)
        reference = move_without_spawn(transform_board(board, transform_id), transformed_action)

        # afterstate
        assert (
            transform_board(original.afterstate, transform_id).tolist()
            == reference.afterstate.tolist()
        )
        # reward
        assert original.reward == reference.reward
        # legality
        assert original.moved == reference.moved


@pytest.mark.parametrize("transform_id", range(TRANSFORM_COUNT))
def test_legal_mask_equivariance(transform_id):
    rng = np.random.Generator(np.random.PCG64(9000 + transform_id))
    for _ in range(60):
        board = random_board(rng)
        mask = legal_mask(board)
        transformed_mask = legal_mask(transform_board(board, transform_id))
        for action in ACTIONS:
            assert bool(mask[action]) == bool(transformed_mask[transform_action(action, transform_id)])


@pytest.mark.parametrize("transform_id", range(TRANSFORM_COUNT))
def test_terminal_is_invariant(transform_id):
    rng = np.random.Generator(np.random.PCG64(13000 + transform_id))
    for _ in range(200):
        board = random_board(rng, low=0, high=5)
        assert is_terminal(transform_board(board, transform_id)) == is_terminal(board)


def test_terminal_is_invariant_on_jammed_boards():
    jammed = np.array(
        [1, 2, 1, 2, 2, 1, 2, 1, 1, 2, 1, 2, 2, 1, 2, 1], dtype=np.uint8
    )
    assert is_terminal(jammed) is True
    for transform_id in range(TRANSFORM_COUNT):
        transformed = transform_board(jammed, transform_id)
        assert is_terminal(transformed) is True
        assert legal_mask(transformed).tolist() == [False] * 4


def test_equivariance_holds_for_an_explicit_example():
    """A fully hand-checked instance of the equivariance law.

    Board: 2 at (0, 1) and 4 at (2, 3).  LEFT merges nothing but slides the 2 to
    (0, 0); the 4 is already flush left.
    """
    board = np.zeros(16, dtype=np.uint8)
    board[1] = 1
    board[11] = 2

    result = move_without_spawn(board, Action.LEFT)
    expected = np.zeros(16, dtype=np.uint8)
    expected[0] = 1
    expected[8] = 2
    assert result.afterstate.tolist() == expected.tolist()
    assert result.reward == 0
    assert result.moved is True

    for transform_id in range(TRANSFORM_COUNT):
        transformed_action = transform_action(Action.LEFT, transform_id)
        reference = move_without_spawn(transform_board(board, transform_id), transformed_action)
        assert (
            transform_board(result.afterstate, transform_id).tolist()
            == reference.afterstate.tolist()
        )
        assert reference.reward == 0
        assert reference.moved is True
