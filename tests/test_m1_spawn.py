"""M1: spawn enumeration and random spawn tests.

seed = 20260918 for every random case; sample counts are stated per test.
"""

from __future__ import annotations

import numpy as np
import pytest

from game2048.fast_env import (
    MAX_SPAWN_OUTCOMES,
    apply_spawn_batch,
    enumerate_spawns_batch,
    spawn_random_batch,
)

from _m1_helpers import SEED, batch, board, board_str, random_boards, reference_spawns

#: Floating-point tolerance for probability comparisons (spec: <= 1e-12).
PROBABILITY_TOLERANCE = 1e-12


def _sparse_boards(count: int, empty_cells: int, seed: int = SEED) -> np.ndarray:
    """Boards with exactly ``empty_cells`` empty cells each (spread evenly)."""
    rng = np.random.default_rng(seed)
    entries = np.ones((count, 16), dtype=np.uint8)
    for index in range(count):
        cells = rng.choice(16, size=empty_cells, replace=False)
        entries[index, cells] = 0
    return np.ascontiguousarray(entries)


def test_enumeration_matches_m0_on_random_boards():
    """2,000 random boards with varying empty counts: order, state, probability.

    seed = 20260918, sample count = 2,000, exponent range = 0..17.

    The CSR ``offsets`` give each parent's contiguous *outcome range*; the
    ``parent_indices`` column is what identifies a row, so the per-parent rows are
    selected by ``parent_indices`` and compared against M0 position by position.
    """
    entries = random_boards(2_000, 17, SEED, empty_probability=0.5)
    enumeration = enumerate_spawns_batch(entries)

    assert enumeration.states.shape[1] == 16
    assert np.array_equal(
        enumeration.offsets[:-1] + 2 * enumeration.empty_counts, enumeration.offsets[1:]
    )

    for index in range(entries.shape[0]):
        expected = reference_spawns(entries[index])
        start = int(enumeration.offsets[index])
        stop = int(enumeration.offsets[index + 1])
        assert stop - start == len(expected), (
            f"sample={index} board={board_str(entries[index])}\n"
            f"  fast outcomes = {stop - start}, ref outcomes = {len(expected)}"
        )
        # The range must hold exactly this parent's rows.
        row_parents = enumeration.parent_indices[start:stop]
        assert np.all(row_parents == index), (
            f"sample={index} CSR range {start}:{stop} holds parents "
            f"{np.unique(row_parents).tolist()}"
        )
        for position, outcome in enumerate(expected):
            slot = start + position
            assert int(enumeration.spawn_indices[slot]) == outcome.spawn_index, (
                f"sample={index} outcome={position} spawn_index mismatch: "
                f"{int(enumeration.spawn_indices[slot])} vs {outcome.spawn_index}"
            )
            assert int(enumeration.spawn_exponents[slot]) == outcome.spawn_exponent
            assert np.array_equal(enumeration.states[slot], outcome.state), (
                f"sample={index} outcome={position} state mismatch: "
                f"{board_str(enumeration.states[slot])} vs "
                f"{board_str(outcome.state)}"
            )
            assert (
                abs(float(enumeration.probabilities[slot]) - outcome.probability)
                <= PROBABILITY_TOLERANCE
            ), (
                f"sample={index} outcome={position} probability "
                f"{float(enumeration.probabilities[slot])} vs {outcome.probability}"
            )


def test_enumeration_order_is_flat_index_then_exponent():
    """Empty cells ascending, then exponent 1 before exponent 2 within a cell."""
    entries = batch(
        [
            board((0, 5, 0, 7), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)),
        ]
    )
    enumeration = enumerate_spawns_batch(entries)
    # 16 cells, two of them occupied.
    assert int(enumeration.empty_counts[0]) == 14
    assert int(enumeration.offsets[1]) == 28

    indices = enumeration.spawn_indices[0:28].tolist()
    exponents = enumeration.spawn_exponents[0:28].tolist()
    assert indices == [
        0, 0, 2, 2, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8,
        9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15,
    ]
    assert exponents == [1, 2] * 14


def test_enumeration_probabilities_sum_to_one_per_parent():
    """Each parent's outcome probabilities sum to 1.0 within tolerance."""
    entries = _sparse_boards(64, 4)
    enumeration = enumerate_spawns_batch(entries)
    for index in range(entries.shape[0]):
        start = int(enumeration.offsets[index])
        stop = int(enumeration.offsets[index + 1])
        total = float(enumeration.probabilities[start:stop].sum())
        assert abs(total - 1.0) <= PROBABILITY_TOLERANCE, (
            f"sample={index} probability sum = {total}"
        )


def test_enumeration_of_a_full_board_is_an_empty_range():
    """A full board yields exactly zero outcomes (``offset[i] == offset[i+1]``)."""
    full = np.ones((1, 16), dtype=np.uint8)
    enumeration = enumerate_spawns_batch(full)
    assert enumeration.states.shape[0] == 0
    assert int(enumeration.offsets[0]) == int(enumeration.offsets[1]) == 0
    assert int(enumeration.empty_counts[0]) == 0


def test_enumeration_outcome_count_is_at_most_32():
    """``2 * n`` outcomes, bounded by ``MAX_SPAWN_OUTCOMES``."""
    entries = _sparse_boards(32, 16)
    enumeration = enumerate_spawns_batch(entries)
    for index in range(entries.shape[0]):
        count = int(enumeration.offsets[index + 1] - enumeration.offsets[index])
        assert count == 2 * int(enumeration.empty_counts[index])
        assert count <= MAX_SPAWN_OUTCOMES


def test_enumeration_does_not_modify_its_input():
    entries = random_boards(128, 17, SEED)
    before = entries.copy()
    enumeration = enumerate_spawns_batch(entries)
    assert np.array_equal(entries, before)
    assert not np.shares_memory(enumeration.states, entries)


def test_random_spawn_stays_within_empty_cells():
    """Every sampled tile lands on a cell that was empty, and the input is intact."""
    entries = random_boards(500, 17, SEED, empty_probability=0.5)
    before = entries.copy()
    rng = np.random.default_rng(SEED)
    result = spawn_random_batch(entries, rng)

    assert np.array_equal(entries, before)
    rows = np.arange(entries.shape[0])
    empty_counts = (entries == 0).sum(axis=1)
    for index in range(entries.shape[0]):
        cell = int(result.spawn_indices[index])
        assert before[index, cell] == 0, (
            f"sample={index} board={board_str(before[index])} spawned at occupied "
            f"cell {cell}"
        )
        assert int(result.spawn_exponents[index]) in (1, 2)
        assert int(result.states[index, cell]) == int(result.spawn_exponents[index])
        # Exactly one cell changed, and only on that board.
        changed = np.flatnonzero(result.states[index] != before[index])
        assert changed.tolist() == [cell]
        assert int(empty_counts[index]) >= 1


def test_random_spawn_distribution_tile_four_ratio():
    """100,000 sampled spawns: the tile-4 share must land in 0.09..0.11.

    seed = 20260918, sample count = 100,000, exponents drawn from 1..6 with about
    40% of the cells empty.  Boards are generated by blanking cells rather than by
    filling them, so a full board (which cannot spawn) never occurs.
    """
    total = 0
    fours = 0
    rng = np.random.default_rng(SEED)
    for round_index in range(100):
        entries = random_boards(1_000, 6, SEED + round_index, empty_probability=0.6)
        result = spawn_random_batch(entries, rng)
        total += entries.shape[0]
        fours += int((result.spawn_exponents == 2).sum())
    ratio = fours / total
    assert total == 100_000
    assert 0.09 <= ratio <= 0.11, f"tile-4 ratio {ratio} outside 0.09..0.11"


def test_random_spawn_position_is_approximately_uniform():
    """Across 200 seeds, the per-cell rates stay near ``1 / empty_count``."""
    empty_cells = 4
    entry = np.ones((1, 16), dtype=np.uint8)
    entry[0, 0] = 0
    entry[0, 1] = 0
    entry[0, 2] = 0
    entry[0, 3] = 0

    totals = np.zeros(16, dtype=np.int64)
    seeds = 200
    samples = 2_000
    for offset in range(seeds):
        rng = np.random.default_rng(SEED + offset)
        tile = np.repeat(entry, samples, axis=0)
        result = spawn_random_batch(tile, rng)
        totals += np.bincount(result.spawn_indices, minlength=16)

    grand_total = int(totals.sum())
    expected = 1.0 / empty_cells
    for cell in range(4):
        observed = totals[cell] / grand_total
        assert abs(observed - expected) < 0.02, (
            f"cell {cell} sampled {observed:.4f}, expected about {expected:.4f}"
        )
    for cell in range(4, 16):
        assert totals[cell] == 0, f"tile landed on occupied cell {cell}"


def test_random_spawn_rejects_full_boards_and_names_the_parent():
    entries = np.ones((3, 16), dtype=np.uint8)
    entries[1, 0] = 0
    rng = np.random.default_rng(SEED)
    with pytest.raises(ValueError) as error:
        spawn_random_batch(entries, rng)
    message = str(error.value)
    assert "0" in message and "2" in message


def test_random_spawn_requires_a_generator():
    entries = random_boards(4, 17, SEED)
    with pytest.raises(TypeError):
        spawn_random_batch(entries, np.random)
    with pytest.raises(TypeError):
        spawn_random_batch(entries, 12345)


def test_random_spawn_is_reproducible_for_a_fixed_seed():
    entries = random_boards(256, 17, SEED, empty_probability=0.5)
    first = spawn_random_batch(entries, np.random.default_rng(SEED))
    second = spawn_random_batch(entries, np.random.default_rng(SEED))
    assert np.array_equal(first.spawn_indices, second.spawn_indices)
    assert np.array_equal(first.spawn_exponents, second.spawn_exponents)
    assert np.array_equal(first.states, second.states)


def test_apply_spawn_places_tiles_deterministically():
    entries = random_boards(128, 17, SEED, empty_probability=0.6)
    before = entries.copy()
    indices = np.zeros(entries.shape[0], dtype=np.int64)
    exponents = np.ones(entries.shape[0], dtype=np.uint8)
    for index in range(entries.shape[0]):
        empty = np.flatnonzero(entries[index] == 0)
        assert empty.size
        indices[index] = int(empty[index % empty.size])
        exponents[index] = 1 + (index % 2)

    states = apply_spawn_batch(entries, indices, exponents)
    assert np.array_equal(entries, before)
    assert not np.shares_memory(states, entries)
    for index in range(entries.shape[0]):
        assert int(states[index, indices[index]]) == int(exponents[index])
        changed = np.flatnonzero(states[index] != before[index])
        assert changed.tolist() == [int(indices[index])]


def test_apply_spawn_rejects_occupied_targets():
    entries = np.zeros((1, 16), dtype=np.uint8)
    entries[0, 5] = 4
    with pytest.raises(ValueError):
        apply_spawn_batch(
            entries, np.array([5], dtype=np.int64), np.array([1], dtype=np.uint8)
        )


def test_apply_spawn_rejects_invalid_exponents():
    entries = np.zeros((1, 16), dtype=np.uint8)
    for exponent in (0, 3, 7):
        with pytest.raises(ValueError):
            apply_spawn_batch(
                entries,
                np.array([0], dtype=np.int64),
                np.array([exponent], dtype=np.int64),
            )


def test_apply_spawn_rejects_out_of_range_indices():
    entries = np.zeros((2, 16), dtype=np.uint8)
    with pytest.raises(ValueError):
        apply_spawn_batch(
            entries,
            np.array([0, 16], dtype=np.int64),
            np.array([1, 1], dtype=np.uint8),
        )
    with pytest.raises(ValueError):
        apply_spawn_batch(
            entries,
            np.array([0, -1], dtype=np.int64),
            np.array([1, 1], dtype=np.uint8),
        )


def test_apply_spawn_rejects_float_and_bool_vectors():
    entries = np.zeros((2, 16), dtype=np.uint8)
    with pytest.raises(ValueError):
        apply_spawn_batch(
            entries, np.array([0.0, 1.0]), np.array([1, 1], dtype=np.uint8)
        )
    with pytest.raises(ValueError):
        apply_spawn_batch(
            entries, np.array([True, False]), np.array([1, 1], dtype=np.uint8)
        )


def test_enumerated_outcomes_can_be_applied_and_match_m0_states():
    """Feed each enumerated outcome back through ``apply_spawn_batch``."""
    entries = _sparse_boards(200, 3)
    enumeration = enumerate_spawns_batch(entries)
    assert enumeration.states.shape[0] > 0
    applied = apply_spawn_batch(
        entries[enumeration.parent_indices],
        enumeration.spawn_indices,
        enumeration.spawn_exponents,
    )
    assert np.array_equal(applied, enumeration.states)
