"""M0 property tests over large random samples.

Property tests use fixed seeds plus many random cases (no Hypothesis, as
required by the M0 plan).  The default sample size is 10,000 random boards with
exponents drawn from 0..17; a second sweep extends the range to 0..20 so that
high tiles are exercised as well.
"""

import numpy as np
import pytest

from game2048 import (
    ACTIONS,
    SPAWN_EXPONENT_TILE_2,
    SPAWN_EXPONENT_TILE_4,
    TRANSFORM_COUNT,
    Action,
    enumerate_spawns,
    inverse_transform_id,
    is_terminal,
    legal_mask,
    move_without_spawn,
    spawn_random,
    transform_action,
    transform_board,
)

#: Number of random boards used by the main sweep.
SAMPLE_COUNT = 10000
#: Number of random boards used by the extended high-exponent sweep.
HIGH_SAMPLE_COUNT = 2000
#: Number of random boards used by the (more expensive) D4 sweep.
D4_SAMPLE_COUNT = 2000

#: Fixed seed of the main sweep.
SEED = 20240517
#: Fixed seed of the extended sweep.
HIGH_SEED = 987654321
#: Fixed seed of the D4 sweep.
D4_SEED = 13571113


def tile_value(exponent):
    """Tile value of an exponent: ``0 -> 0``, ``e > 0 -> 2 ** e``."""
    return 0 if exponent == 0 else 2 ** exponent


def board_value_sum(board):
    """Sum of all tile values on the board, as a Python int."""
    return int(sum(tile_value(int(exponent)) for exponent in np.asarray(board).tolist()))


def random_boards(count, seed, low, high):
    rng = np.random.Generator(np.random.PCG64(seed))
    return rng.integers(low, high, size=(count, 16)).astype(np.uint8)


MAIN_BOARDS = random_boards(SAMPLE_COUNT, SEED, 0, 18)  # exponents 0..17
HIGH_BOARDS = random_boards(HIGH_SAMPLE_COUNT, HIGH_SEED, 0, 21)  # exponents 0..20
D4_BOARDS = random_boards(D4_SAMPLE_COUNT, D4_SEED, 0, 18)


# --------------------------------------------------------------------------- #
# Property A: the total tile value is conserved by a move (before the spawn)
# --------------------------------------------------------------------------- #


def test_property_a_tile_value_is_conserved_by_a_move():
    for board in MAIN_BOARDS:
        before = board_value_sum(board)
        for action in ACTIONS:
            result = move_without_spawn(board, action)
            assert board_value_sum(result.afterstate) == before


def test_property_a_holds_for_high_exponents_too():
    for board in HIGH_BOARDS:
        before = board_value_sum(board)
        for action in ACTIONS:
            result = move_without_spawn(board, action)
            assert board_value_sum(result.afterstate) == before


def test_property_a_holds_for_dense_boards():
    rng = np.random.Generator(np.random.PCG64(24680))
    for _ in range(2000):
        board = rng.integers(1, 6, size=16).astype(np.uint8)
        before = board_value_sum(board)
        for action in ACTIONS:
            assert board_value_sum(move_without_spawn(board, action).afterstate) == before


# --------------------------------------------------------------------------- #
# Property B: moved == False implies an untouched board and zero reward
# --------------------------------------------------------------------------- #


def test_property_b_immobile_move_changes_nothing():
    for board in MAIN_BOARDS:
        for action in ACTIONS:
            result = move_without_spawn(board, action)
            if not result.moved:
                assert result.reward == 0
                assert result.afterstate.tolist() == board.tolist()


def test_property_b_converse_reward_implies_moved():
    for board in MAIN_BOARDS[:3000]:
        for action in ACTIONS:
            result = move_without_spawn(board, action)
            if result.reward > 0:
                assert result.moved is True
            if result.moved:
                assert result.afterstate.tolist() != board.tolist()


# --------------------------------------------------------------------------- #
# Property C: legal_mask agrees with move_without_spawn, in both directions
# --------------------------------------------------------------------------- #


def test_property_c_legal_mask_matches_moved():
    for board in MAIN_BOARDS:
        mask = legal_mask(board)
        assert mask.shape == (4,)
        assert mask.dtype == bool
        for action in ACTIONS:
            moved = move_without_spawn(board, action).moved
            assert bool(mask[action]) is moved


def test_property_c_legal_mask_for_high_exponents():
    for board in HIGH_BOARDS:
        mask = legal_mask(board)
        for action in ACTIONS:
            assert bool(mask[action]) is move_without_spawn(board, action).moved


# --------------------------------------------------------------------------- #
# Property D + E: a spawn adds exactly one tile of 2 or 4 into an empty cell
# --------------------------------------------------------------------------- #


def test_property_d_and_e_spawn_adds_two_or_four_in_one_empty_cell():
    rng = np.random.Generator(np.random.PCG64(424242))
    for board in MAIN_BOARDS:
        # Spawning needs an empty cell; skip the (measure-zero) full boards.
        if not (board == 0).any():
            continue
        before = board_value_sum(board)
        outcome = spawn_random(board, rng)

        after = board_value_sum(outcome.state)
        assert after - before in (2, 4)
        assert after - before == tile_value(outcome.spawn_exponent)

        changed = np.flatnonzero(outcome.state != board)
        assert changed.size == 1
        assert int(changed[0]) == outcome.spawn_index
        assert board[outcome.spawn_index] == 0
        assert outcome.spawn_exponent in (SPAWN_EXPONENT_TILE_2, SPAWN_EXPONENT_TILE_4)


def test_property_d_and_e_for_every_enumerated_outcome():
    for board in MAIN_BOARDS[:400]:
        if not (board == 0).any():
            continue
        before = board_value_sum(board)
        for outcome in enumerate_spawns(board):
            assert board_value_sum(outcome.state) - before == tile_value(outcome.spawn_exponent)
            changed = np.flatnonzero(outcome.state != board)
            assert changed.tolist() == [outcome.spawn_index]


# --------------------------------------------------------------------------- #
# Property F: D4 movement equivariance and group laws
# --------------------------------------------------------------------------- #


def test_property_f_d4_movement_equivariance():
    for board in D4_BOARDS:
        for transform_id in range(TRANSFORM_COUNT):
            transformed_board = transform_board(board, transform_id)
            for action in ACTIONS:
                transformed_action = transform_action(action, transform_id)

                original = move_without_spawn(board, action)
                reference = move_without_spawn(transformed_board, transformed_action)

                assert (
                    transform_board(original.afterstate, transform_id).tolist()
                    == reference.afterstate.tolist()
                ), f"afterstate mismatch for transform {transform_id}, action {action}"
                assert original.reward == reference.reward
                assert original.moved == reference.moved


def test_property_f_d4_terminal_invariance():
    for board in D4_BOARDS:
        terminal = is_terminal(board)
        for transform_id in range(TRANSFORM_COUNT):
            assert is_terminal(transform_board(board, transform_id)) == terminal


def test_property_f_d4_inverse_round_trip():
    for board in D4_BOARDS:
        for transform_id in range(TRANSFORM_COUNT):
            inverse = inverse_transform_id(transform_id)
            round_tripped = transform_board(transform_board(board, transform_id), inverse)
            assert round_tripped.tolist() == board.tolist()


def test_property_f_d4_transforms_preserve_the_multiset_of_tiles():
    for board in D4_BOARDS[:500]:
        sorted_board = sorted(board.tolist())
        for transform_id in range(TRANSFORM_COUNT):
            assert sorted(transform_board(board, transform_id).tolist()) == sorted_board


def test_property_f_d4_identity_is_truly_identity():
    for board in D4_BOARDS:
        assert transform_board(board, 0).tolist() == board.tolist()
        for action in ACTIONS:
            assert transform_action(action, 0) == action


# --------------------------------------------------------------------------- #
# Property G: terminal == no legal action
# --------------------------------------------------------------------------- #


def test_property_g_terminal_equals_all_actions_illegal():
    for board in MAIN_BOARDS:
        expected = not any(move_without_spawn(board, action).moved for action in ACTIONS)
        assert is_terminal(board) == expected
        assert is_terminal(board) == (not legal_mask(board).any())


def test_property_g_terminal_on_nearly_full_boards():
    """Dense boards exercise the terminal predicate far more often."""
    rng = np.random.Generator(np.random.PCG64(555))
    terminal_count = 0
    for _ in range(3000):
        board = rng.integers(1, 8, size=16).astype(np.uint8)
        expected = not any(move_without_spawn(board, action).moved for action in ACTIONS)
        assert is_terminal(board) == expected
        terminal_count += int(is_terminal(board))
    assert terminal_count > 0, "expected some random dense boards to be terminal"

    # Guaranteed-terminal family: a full checkerboard of two distinct values has
    # no two adjacent equal tiles at all.
    for _ in range(200):
        first, second = rng.choice(np.arange(1, 12), size=2, replace=False)
        board = np.array(
            [
                first if (index // 4 + index % 4) % 2 == 0 else second
                for index in range(16)
            ],
            dtype=np.uint8,
        )
        assert is_terminal(board) is True
        assert legal_mask(board).tolist() == [False] * 4


# --------------------------------------------------------------------------- #
# Purity over the whole sweep
# --------------------------------------------------------------------------- #


def test_pure_functions_never_modify_their_input():
    for board in MAIN_BOARDS[:2000]:
        before = board.copy()
        for action in ACTIONS:
            move_without_spawn(board, action)
        legal_mask(board)
        is_terminal(board)
        for transform_id in range(TRANSFORM_COUNT):
            transform_board(board, transform_id)
        if (board == 0).any():
            enumerate_spawns(board)
        assert np.array_equal(before, board)


def test_pure_functions_do_not_share_memory_with_the_result():
    board = MAIN_BOARDS[17].copy()
    result = move_without_spawn(board, Action.LEFT).afterstate
    result[:] = 0
    assert np.array_equal(board, MAIN_BOARDS[17])


# --------------------------------------------------------------------------- #
# Independent oracle for the movement core itself
# --------------------------------------------------------------------------- #

#: Row/column step of each action: (delta_row, delta_col).
_DIRECTION_STEPS = {
    0: (-1, 0),  # UP
    1: (1, 0),  # DOWN
    2: (0, -1),  # LEFT
    3: (0, 1),  # RIGHT
}


def independent_move(board, action):
    """A deliberately *different* implementation of the frozen 2048 rules.

    Where :func:`move_without_spawn` compacts a line and then merges, this
    simulation moves one tile one cell at a time until the board is stable, runs
    a single merge pass in destination-first order (guarded by a per-tile
    "already merged" flag), and slides again.  It exists purely as an oracle for
    the reference core, so a systematic mistake in the reference algorithm shows
    up as a disagreement instead of silently passing every test.
    """
    delta_row, delta_col = _DIRECTION_STEPS[int(action)]
    grid = [[int(board[r * 4 + c]) for c in range(4)] for r in range(4)]
    reward = 0

    def slide():
        changed = False
        if delta_row:
            row_order = range(1, 4) if delta_row < 0 else range(2, -1, -1)
            for r in row_order:
                for c in range(4):
                    if grid[r][c] and grid[r + delta_row][c] == 0:
                        grid[r + delta_row][c] = grid[r][c]
                        grid[r][c] = 0
                        changed = True
        else:
            col_order = range(1, 4) if delta_col < 0 else range(2, -1, -1)
            for c in col_order:
                for r in range(4):
                    if grid[r][c] and grid[r][c + delta_col] == 0:
                        grid[r][c + delta_col] = grid[r][c]
                        grid[r][c] = 0
                        changed = True
        return changed

    while slide():
        pass

    merged = [[False] * 4 for _ in range(4)]
    if delta_row:
        row_order = range(4) if delta_row < 0 else range(3, -1, -1)
        for r in row_order:
            for c in range(4):
                partner_row = r - delta_row
                if not 0 <= partner_row < 4:
                    continue
                if grid[r][c] == 0 or grid[partner_row][c] != grid[r][c]:
                    continue
                if merged[r][c] or merged[partner_row][c]:
                    continue
                if grid[r][c] >= 255:
                    raise OverflowError("independent oracle: exponent overflow")
                grid[r][c] += 1
                grid[partner_row][c] = 0
                merged[r][c] = True
                reward += 2 ** grid[r][c]
    else:
        col_order = range(4) if delta_col < 0 else range(3, -1, -1)
        for c in col_order:
            for r in range(4):
                partner_col = c - delta_col
                if not 0 <= partner_col < 4:
                    continue
                if grid[r][c] == 0 or grid[r][partner_col] != grid[r][c]:
                    continue
                if merged[r][c] or merged[r][partner_col]:
                    continue
                if grid[r][c] >= 255:
                    raise OverflowError("independent oracle: exponent overflow")
                grid[r][c] += 1
                grid[r][partner_col] = 0
                merged[r][c] = True
                reward += 2 ** grid[r][c]

    while slide():
        pass

    afterstate = np.array([cell for row in grid for cell in row], dtype=np.uint8)
    return afterstate, reward, afterstate.tolist() != [int(v) for v in board.tolist()]


def test_independent_oracle_agrees_on_the_main_sweep():
    for board in MAIN_BOARDS:
        for action in ACTIONS:
            reference = move_without_spawn(board, action)
            afterstate, reward, moved = independent_move(board, action)

            assert afterstate.tolist() == reference.afterstate.tolist()
            assert reward == reference.reward
            assert moved == reference.moved


def test_independent_oracle_agrees_on_high_exponents():
    for board in HIGH_BOARDS:
        for action in ACTIONS:
            reference = move_without_spawn(board, action)
            afterstate, reward, moved = independent_move(board, action)

            assert afterstate.tolist() == reference.afterstate.tolist()
            assert reward == reference.reward
            assert moved == reference.moved


def test_independent_oracle_agrees_on_dense_boards():
    rng = np.random.Generator(np.random.PCG64(112233))
    for _ in range(3000):
        board = rng.integers(0, 4, size=16).astype(np.uint8)
        for action in ACTIONS:
            reference = move_without_spawn(board, action)
            afterstate, reward, moved = independent_move(board, action)
            assert afterstate.tolist() == reference.afterstate.tolist()
            assert reward == reference.reward
            assert moved == reference.moved


# --------------------------------------------------------------------------- #
# Reward sanity on the main sweep
# --------------------------------------------------------------------------- #


def test_rewards_are_python_ints_and_match_the_merge_count():
    for board in MAIN_BOARDS[:2000]:
        for action in ACTIONS:
            result = move_without_spawn(board, action)
            assert isinstance(result.reward, int)
            assert result.reward >= 0
            # Every merge at least doubles a 2, so each merge contributes >= 4.
            if result.reward:
                assert result.reward % 4 == 0
