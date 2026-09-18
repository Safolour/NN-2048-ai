"""M0 tests for the spawn model: exact enumeration and the RNG sampler.

Rule correctness is pinned by the *exact* probabilities of
:func:`enumerate_spawns`; the statistical checks on :func:`spawn_random` are
only sampler sanity checks.
"""

import numpy as np
import pytest

from game2048 import (
    SPAWN_EXPONENT_TILE_2,
    SPAWN_EXPONENT_TILE_4,
    SPAWN_PROB_TILE_2,
    SPAWN_PROB_TILE_4,
    enumerate_spawns,
    spawn_random,
)

EMPTY_ROW = (0, 0, 0, 0)


def B(*rows):
    """Build a ``(16,)`` ``uint8`` board from four rows of exponents."""
    return np.array(rows, dtype=np.uint8).reshape(16)


EMPTY_BOARD = B(EMPTY_ROW, EMPTY_ROW, EMPTY_ROW, EMPTY_ROW)

# One empty cell is at flat index 5: row 1, column 1.
ONE_HOLE = B((1, 2, 3, 4), (5, 0, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16))


# --------------------------------------------------------------------------- #
# enumerate_spawns: exact distribution
# --------------------------------------------------------------------------- #


def test_enumerate_spawns_count_is_two_per_empty_cell():
    outcomes = enumerate_spawns(EMPTY_BOARD)
    assert len(outcomes) == 16 * 2


def test_enumerate_spawns_single_hole():
    outcomes = enumerate_spawns(ONE_HOLE)
    assert len(outcomes) == 2

    by_exponent = {outcome.spawn_exponent: outcome for outcome in outcomes}
    assert set(by_exponent) == {SPAWN_EXPONENT_TILE_2, SPAWN_EXPONENT_TILE_4}

    for outcome in outcomes:
        assert outcome.spawn_index == 5
        expected = ONE_HOLE.tolist()
        expected[5] = outcome.spawn_exponent
        assert outcome.state.tolist() == expected

    assert by_exponent[SPAWN_EXPONENT_TILE_2].probability == pytest.approx(SPAWN_PROB_TILE_2)
    assert by_exponent[SPAWN_EXPONENT_TILE_4].probability == pytest.approx(SPAWN_PROB_TILE_4)


@pytest.mark.parametrize("empty_cells", [1, 2, 3, 5, 7, 11, 16])
def test_enumerate_spawns_probabilities_are_exact(empty_cells):
    cells = [3] * (16 - empty_cells) + [0] * empty_cells
    board = np.array(cells, dtype=np.uint8)

    outcomes = enumerate_spawns(board)
    assert len(outcomes) == 2 * empty_cells

    for outcome in outcomes:
        if outcome.spawn_exponent == SPAWN_EXPONENT_TILE_2:
            assert outcome.probability == pytest.approx(SPAWN_PROB_TILE_2 / empty_cells)
        else:
            assert outcome.probability == pytest.approx(SPAWN_PROB_TILE_4 / empty_cells)

    assert sum(outcome.probability for outcome in outcomes) == pytest.approx(1.0, abs=1e-12)


def test_enumerate_spawns_total_probability_is_one_full_board():
    outcomes = enumerate_spawns(EMPTY_BOARD)
    assert sum(outcome.probability for outcome in outcomes) == pytest.approx(1.0, abs=1e-12)


def test_enumerate_spawns_are_distinct_and_cover_every_cell():
    outcomes = enumerate_spawns(EMPTY_BOARD)

    seen = set()
    indices_two = set()
    indices_four = set()
    for outcome in outcomes:
        key = (outcome.spawn_index, outcome.spawn_exponent)
        assert key not in seen
        seen.add(key)
        assert outcome.spawn_exponent in (SPAWN_EXPONENT_TILE_2, SPAWN_EXPONENT_TILE_4)
        if outcome.spawn_exponent == SPAWN_EXPONENT_TILE_2:
            indices_two.add(outcome.spawn_index)
        else:
            indices_four.add(outcome.spawn_index)

    assert indices_two == set(range(16))
    assert indices_four == set(range(16))


def test_enumerate_spawns_changes_exactly_one_empty_cell():
    outcomes = enumerate_spawns(ONE_HOLE)
    for outcome in outcomes:
        difference = np.flatnonzero(outcome.state != ONE_HOLE)
        assert difference.tolist() == [outcome.spawn_index]
        assert ONE_HOLE[outcome.spawn_index] == 0
        assert outcome.state[outcome.spawn_index] == outcome.spawn_exponent
        assert outcome.state.dtype == np.uint8


def test_enumerate_spawns_on_full_board_is_empty():
    full = np.array([1] * 16, dtype=np.uint8)
    assert enumerate_spawns(full) == []


def test_enumerate_spawns_does_not_modify_input():
    board = ONE_HOLE.copy()
    before = board.copy()
    enumerate_spawns(board)
    assert np.array_equal(before, board)


# --------------------------------------------------------------------------- #
# spawn_random: sampler sanity
# --------------------------------------------------------------------------- #

SPAWN_SAMPLES = 20000


def _sample_spawns(board, seed, count):
    rng = np.random.Generator(np.random.PCG64(seed))
    exponents = []
    indices = []
    for _ in range(count):
        outcome = spawn_random(board, rng)
        exponents.append(outcome.spawn_exponent)
        indices.append(outcome.spawn_index)
    return np.array(exponents), np.array(indices)


def test_spawn_random_is_reproducible_for_a_fixed_seed():
    first = _sample_spawns(EMPTY_BOARD, seed=20240517, count=64)
    second = _sample_spawns(EMPTY_BOARD, seed=20240517, count=64)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])


def test_spawn_random_tile_four_ratio_is_about_ten_percent():
    exponents, _ = _sample_spawns(EMPTY_BOARD, seed=12345, count=SPAWN_SAMPLES)
    ratio_four = float(np.mean(exponents == SPAWN_EXPONENT_TILE_4))
    assert 0.08 <= ratio_four <= 0.12, f"tile-4 ratio was {ratio_four}"


def test_spawn_random_only_produces_tile_two_or_tile_four():
    exponents, _ = _sample_spawns(EMPTY_BOARD, seed=999, count=2000)
    assert set(np.unique(exponents).tolist()) == {1, 2}


def test_spawn_random_chooses_cells_uniformly():
    _, indices = _sample_spawns(EMPTY_BOARD, seed=777, count=SPAWN_SAMPLES)
    counts = np.bincount(indices, minlength=16)
    assert counts.sum() == SPAWN_SAMPLES

    expected = SPAWN_SAMPLES / 16
    # Loose sanity band; the exact rule is pinned by enumerate_spawns.
    assert counts.min() > expected * 0.8
    assert counts.max() < expected * 1.2


def test_spawn_random_respects_the_available_holes():
    """Only the single empty cell may receive the tile."""
    rng = np.random.Generator(np.random.PCG64(4242))
    for _ in range(200):
        outcome = spawn_random(ONE_HOLE, rng)
        assert outcome.spawn_index == 5
        assert outcome.spawn_exponent in (SPAWN_EXPONENT_TILE_2, SPAWN_EXPONENT_TILE_4)
        # n == 1: probabilities collapse to the raw spawn probabilities.
        expected = (
            SPAWN_PROB_TILE_2
            if outcome.spawn_exponent == SPAWN_EXPONENT_TILE_2
            else SPAWN_PROB_TILE_4
        )
        assert outcome.probability == pytest.approx(expected)


def test_spawn_random_agrees_with_enumerated_support():
    """Every sampled outcome must be one of the enumerated outcomes."""
    board = B((1, 1, 0, 0), (2, 0, 3, 0), (0, 0, 0, 4), (0, 5, 0, 0))
    enumerated = {
        (outcome.spawn_index, outcome.spawn_exponent): outcome
        for outcome in enumerate_spawns(board)
    }

    rng = np.random.Generator(np.random.PCG64(31337))
    for _ in range(500):
        outcome = spawn_random(board, rng)
        expected = enumerated[(outcome.spawn_index, outcome.spawn_exponent)]
        assert np.array_equal(outcome.state, expected.state)
        assert outcome.probability == pytest.approx(expected.probability)


def test_spawn_random_does_not_modify_input():
    board = ONE_HOLE.copy()
    before = board.copy()
    rng = np.random.Generator(np.random.PCG64(5))
    spawn_random(board, rng)
    assert np.array_equal(before, board)


def test_spawn_random_rejects_full_board():
    full = np.array([1] * 16, dtype=np.uint8)
    rng = np.random.Generator(np.random.PCG64(5))
    with pytest.raises(ValueError):
        spawn_random(full, rng)


def test_spawn_random_rejects_non_generator_rng():
    state = np.random.RandomState(0)
    with pytest.raises(TypeError):
        spawn_random(ONE_HOLE, state)


def test_spawn_random_does_not_touch_the_global_numpy_state():
    np.random.seed(2024)
    before = np.random.get_state()[1].copy()
    rng = np.random.Generator(np.random.PCG64(1))
    spawn_random(EMPTY_BOARD, rng)
    after = np.random.get_state()[1]
    assert np.array_equal(before, after)
