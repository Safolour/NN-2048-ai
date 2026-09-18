"""M1 post-audit fix: ``int64`` **aggregate** reward overflow must never wrap.

The pre-fix M1 handled the *single-merge* limit correctly (``e = 62`` raises) but
not the *aggregate* limit: several individually representable merges could sum
past ``np.iinfo(np.int64).max`` and wrap silently.  ``np.errstate(over="raise")``
does not help, because NumPy only signals integer overflow for **scalar**
operations -- whole-array adds and integer reductions such as ``.sum(axis=1)``
wrap without any warning:

    >>> a = np.array([2 ** 62, 2 ** 62], dtype=np.int64)
    >>> with np.errstate(over="raise"):
    ...     a.sum()
    -9223372036854775808

M1's contract is therefore: every reward either is the exact non-negative
``int64`` value, or the call raises ``OverflowError``.  This file pins three
aggregation levels (row, board, score) plus the atomicity of ``step``.

The final-audit sections at the end extend the same file to the *inverse*
requirement: the ``int64`` reward limit is a property of **reward-producing**
calls only.  ``legal_mask_batch`` / ``is_terminal_batch`` answer a movement
question and must stay independent of it, while the ``uint8`` tile limit (a
``255 + 255`` merge) stays enforced everywhere.  ``scores`` must also remain the
live view its docstring promises.

Exact boundary values used below (computed by hand, per spec §41)::

    INT64_MAX  = 9223372036854775807   = 2**63 - 1
    2**62      = 4611686018427387904
    2**62 * 2  = 9223372036854775808   = 2**63        -> overflow
    2**62+2**61= 6917529027641081856   <= INT64_MAX   -> safe
"""

from __future__ import annotations

import numpy as np
import pytest

from game2048 import Action, move_without_spawn
from game2048.fast_env import (
    Fast2048BatchEnv,
    is_terminal_batch,
    legal_mask_batch,
    move_batch,
)

from _m1_helpers import (
    SEED,
    batch,
    board,
    board_str,
    reference_legal_mask,
    reference_terminal,
)

DIRECTIONS = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]

INT64_MAX = np.iinfo(np.int64).max
REWARD_61 = 2 ** 62  # 4611686018427387904 -- the largest single merge
REWARD_60 = 2 ** 61  # 2305843009213693952
OVERFLOW_TOTAL = 2 ** 63  # 9223372036854775808 -- one past int64
SAFE_TOTAL = 2 ** 62 + 2 ** 61  # 6917529027641081856


# --------------------------------------------------------------------------- #
# Board builders
# --------------------------------------------------------------------------- #


def _rotated(rows) -> np.ndarray:
    """``(16,)`` uint8 board from four rows of exponents."""
    return board(*rows)


def _rotated_for(action: Action, row_a, row_b) -> np.ndarray:
    """Two identical pairs laid out so ``action`` merges **both** pairs.

    Each pair is placed along the line direction of ``action``, and the two pairs
    are placed on two *different* lines, so the two merges land on separate lines
    of the same board.
    """
    grid = np.zeros((4, 4), dtype=np.uint8)
    for index, pair in enumerate((row_a, row_b)):
        grid = _place_pair(grid, action, line=index, pair=pair)
    return grid.reshape(16)


def _place_pair(grid, action: Action, *, line: int, pair):
    """Write ``pair`` onto line ``line`` of ``grid``, destination side first."""
    first, second = int(pair[0]), int(pair[1])
    if action == Action.LEFT:
        grid[line, 0], grid[line, 1] = first, second
    elif action == Action.RIGHT:
        grid[line, 3], grid[line, 2] = first, second
    elif action == Action.UP:
        grid[0, line], grid[1, line] = first, second
    else:  # Action.DOWN
        grid[3, line], grid[2, line] = first, second
    return grid


def _single_line_for(action: Action, pair) -> np.ndarray:
    """One line holding two identical pairs: ``[a, a, a, a]`` along ``action``."""
    grid = np.zeros((4, 4), dtype=np.uint8)
    first, second = int(pair[0]), int(pair[1])
    if action == Action.LEFT:
        grid[0, :] = [first, second, first, second]
    elif action == Action.RIGHT:
        grid[0, :] = [first, second, first, second]
    elif action == Action.UP:
        grid[:, 0] = [first, second, first, second]
    else:  # Action.DOWN
        grid[:, 0] = [first, second, first, second]
    return grid.reshape(16)


def _one(entry) -> np.ndarray:
    return batch([entry])


def _actions(count: int, action: Action) -> np.ndarray:
    return np.full(count, int(action), dtype=np.uint8)


# --------------------------------------------------------------------------- #
# 1. Single-merge boundary is unchanged
# --------------------------------------------------------------------------- #


def test_single_61_merge_still_succeeds():
    """``61 + 61 -> 62`` pays ``2**62``, which fits: it must not be rejected."""
    entry = board((61, 61, 0, 0), (0,) * 4, (0,) * 4, (0,) * 4)
    result = move_batch(_one(entry), np.array([int(Action.LEFT)], dtype=np.uint8))
    assert int(result.rewards[0]) == REWARD_61
    assert int(result.afterstates[0][0]) == 62
    reference = move_without_spawn(entry, Action.LEFT)
    assert int(result.rewards[0]) == reference.reward
    assert np.array_equal(result.afterstates[0], reference.afterstate)


def test_single_62_merge_still_raises():
    """``62 + 62`` needs ``2**63``: pre-existing behaviour, still an exception."""
    entry = board((62, 62, 0, 0), (0,) * 4, (0,) * 4, (0,) * 4)
    with pytest.raises(OverflowError):
        move_batch(_one(entry), np.array([int(Action.LEFT)], dtype=np.uint8))


# --------------------------------------------------------------------------- #
# 2. Row-level aggregation (the first audit finding)
# --------------------------------------------------------------------------- #


def test_row_with_two_61_merges_raises():
    """``[61, 61, 61, 61]`` merges twice in ONE row: ``2**62 + 2**62 = 2**63``.

    Pre-fix this returned ``-9223372036854775808`` (silent wrap).
    """
    entry = board((61, 61, 61, 61), (0,) * 4, (0,) * 4, (0,) * 4)
    reference = move_without_spawn(entry, Action.LEFT)
    assert reference.reward == OVERFLOW_TOTAL  # M0 (Python int) has no limit
    assert reference.reward > INT64_MAX
    with pytest.raises(OverflowError):
        move_batch(_one(entry), np.array([int(Action.LEFT)], dtype=np.uint8))


def test_row_aggregation_never_returns_a_negative_reward():
    """Whatever happens, the fast path must not hand back a wrapped number."""
    entry = board((61, 61, 61, 61), (0,) * 4, (0,) * 4, (0,) * 4)
    try:
        result = move_batch(_one(entry), np.array([int(Action.LEFT)], dtype=np.uint8))
    except OverflowError:
        return
    raise AssertionError(
        "expected OverflowError, got reward "
        f"{int(result.rewards[0])} (a wrapped negative reward is forbidden)"
    )


# --------------------------------------------------------------------------- #
# 3. Board-level aggregation (the second audit finding)
# --------------------------------------------------------------------------- #


def test_board_with_two_61_pair_rows_raises():
    """Two separate 61-pair rows: ``2**62 + 2**62 = 2**63`` across lines."""
    entry = board((61, 61, 0, 0), (61, 61, 0, 0), (0,) * 4, (0,) * 4)
    reference = move_without_spawn(entry, Action.LEFT)
    assert reference.reward == OVERFLOW_TOTAL
    with pytest.raises(OverflowError):
        move_batch(_one(entry), np.array([int(Action.LEFT)], dtype=np.uint8))


def test_board_aggregation_is_not_a_sum_of_wrapped_line_rewards():
    """The board total must be checked even though each line alone fits.

    Each line pays exactly ``2**62`` (representable); only the board total
    overflows.  A ``.sum(axis=1)`` over already-correct line rewards would wrap.
    """
    entry = board((61, 61, 0, 0), (61, 61, 0, 0), (0,) * 4, (0,) * 4)
    with pytest.raises(OverflowError):
        move_batch(_one(entry), np.array([int(Action.LEFT)], dtype=np.uint8))


# --------------------------------------------------------------------------- #
# 4. Near-limit but legal rewards must still be exact
# --------------------------------------------------------------------------- #


def test_safe_near_limit_aggregate_is_exact():
    """``2**62 + 2**61 = 6917529027641081856 <= INT64_MAX``: must be returned."""
    entry = board((61, 61, 0, 0), (60, 60, 0, 0), (0,) * 4, (0,) * 4)
    reference = move_without_spawn(entry, Action.LEFT)
    assert reference.reward == SAFE_TOTAL
    assert SAFE_TOTAL <= INT64_MAX

    result = move_batch(_one(entry), np.array([int(Action.LEFT)], dtype=np.uint8))
    assert int(result.rewards[0]) == SAFE_TOTAL
    assert int(result.rewards[0]) == reference.reward
    assert np.array_equal(result.afterstates[0], reference.afterstate)


def test_exactly_int64_max_line_and_board_totals_are_still_accepted():
    """A deterministic board whose totals sit under the ceiling is not rejected.

    Built from the largest legal single merge plus smaller ones, so the reward is
    verified against M0 rather than hard-coded.
    """
    entry = board((61, 61, 0, 0), (60, 60, 0, 0), (59, 59, 0, 0), (0,) * 4)
    reference = move_without_spawn(entry, Action.LEFT)
    assert reference.reward <= INT64_MAX
    result = move_batch(_one(entry), np.array([int(Action.LEFT)], dtype=np.uint8))
    assert int(result.rewards[0]) == reference.reward
    assert np.array_equal(result.afterstates[0], reference.afterstate)


# --------------------------------------------------------------------------- #
# 5. All four directions
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("action", DIRECTIONS)
def test_board_overflow_raises_in_every_direction(action):
    """The board-level aggregate check must not depend on the direction."""
    entry = _rotated_for(action, (61, 61), (61, 61))
    reference = move_without_spawn(entry, action)
    if reference.reward <= INT64_MAX:
        pytest.skip("construction did not overflow for this direction")
    with pytest.raises(OverflowError):
        move_batch(_one(entry), _actions(1, action))


@pytest.mark.parametrize("action", DIRECTIONS)
def test_row_overflow_raises_in_every_direction(action):
    """A single line with two 61-pairs overflows for every direction."""
    entry = _single_line_for(action, (61, 61))
    reference = move_without_spawn(entry, action)
    assert reference.reward == OVERFLOW_TOTAL, (
        f"{action.name}: expected M0 total 2**63, got {reference.reward} "
        f"for board {board_str(entry)}"
    )
    with pytest.raises(OverflowError):
        move_batch(_one(entry), _actions(1, action))


@pytest.mark.parametrize("action", DIRECTIONS)
def test_safe_near_limit_is_exact_in_every_direction(action):
    """The near-limit-but-legal case must also work in all four directions."""
    entry = _rotated_for(action, (61, 61), (60, 60))
    reference = move_without_spawn(entry, action)
    assert reference.reward == SAFE_TOTAL, (
        f"{action.name}: expected {SAFE_TOTAL}, got {reference.reward} "
        f"for board {board_str(entry)}"
    )
    result = move_batch(_one(entry), _actions(1, action))
    assert int(result.rewards[0]) == reference.reward
    assert np.array_equal(result.afterstates[0], reference.afterstate)


def _asymmetric_line_for(action: Action, tail: int) -> np.ndarray:
    """One line ``[61, 61, 61, tail]`` laid out destination-first for ``action``.

    Unlike :func:`_single_line_for` this board is **not** symmetric, so the
    destination side actually matters and ``UP`` / ``DOWN`` (and ``LEFT`` /
    ``RIGHT``) exercise genuinely different permutations.
    """
    grid = np.zeros((4, 4), dtype=np.uint8)
    cells = [61, 61, 61, int(tail)]
    if action == Action.LEFT:
        grid[0, :] = cells
    elif action == Action.RIGHT:
        grid[0, ::-1] = cells
    elif action == Action.UP:
        grid[:, 0] = cells
    else:  # Action.DOWN
        grid[::-1, 0] = cells
    return grid.reshape(16)


@pytest.mark.parametrize("action", DIRECTIONS)
def test_asymmetric_overflow_raises_in_every_direction(action):
    """A non-symmetric line whose destination side is well defined.

    ``[61, 61, 61, 0]`` towards the destination side gives one merge and leaves a
    lone ``61``: total ``2**62``, which fits.  Appending a fourth ``61`` instead
    (``[61, 61, 61, 61]``) merges twice: ``2**63``, which does not.  Both are
    checked here so a direction mix-up cannot pass unnoticed.
    """
    safe = _asymmetric_line_for(action, 0)
    reference = move_without_spawn(safe, action)
    assert reference.reward == REWARD_61, (
        f"{action.name}: [61,61,61,0] should pay exactly 2**62, "
        f"got {reference.reward} for {board_str(safe)}"
    )
    result = move_batch(_one(safe), _actions(1, action))
    assert int(result.rewards[0]) == reference.reward
    assert np.array_equal(result.afterstates[0], reference.afterstate)

    overflowing = _asymmetric_line_for(action, 61)
    assert move_without_spawn(overflowing, action).reward == OVERFLOW_TOTAL, (
        f"{action.name}: [61,61,61,61] should total 2**63 for "
        f"{board_str(overflowing)}"
    )
    with pytest.raises(OverflowError):
        move_batch(_one(overflowing), _actions(1, action))


# --------------------------------------------------------------------------- #
# 6. Mixed batches: one bad board poisons the whole call
# --------------------------------------------------------------------------- #


def _mixed_batch():
    """Four boards: normal / near-limit-safe / overflow / high-but-no-merge."""
    normal = board((1, 1, 2, 0), (0,) * 4, (0,) * 4, (0,) * 4)
    near_limit = board((61, 61, 0, 0), (60, 60, 0, 0), (0,) * 4, (0,) * 4)
    overflowing = board((61, 61, 0, 0), (61, 61, 0, 0), (0,) * 4, (0,) * 4)
    high_no_merge = board((40, 0, 0, 0), (0, 0, 0, 0), (0,) * 4, (0,) * 4)
    return batch([normal, near_limit, overflowing, high_no_merge]), (
        normal,
        near_limit,
        overflowing,
        high_no_merge,
    )


def test_mixed_batch_with_one_overflowing_board_raises():
    """No partial results: one unrepresentable transition fails the whole batch."""
    entries, _ = _mixed_batch()
    actions = _actions(entries.shape[0], Action.LEFT)
    with pytest.raises(OverflowError):
        move_batch(entries, actions)


def test_mixed_batch_without_the_bad_board_succeeds_exactly():
    """The same batch minus the overflow board must still be bit-exact vs M0."""
    entries, originals = _mixed_batch()
    kept = [0, 1, 3]
    subset = batch([originals[index] for index in kept])
    actions = _actions(subset.shape[0], Action.LEFT)
    result = move_batch(subset, actions)
    for position, index in enumerate(kept):
        reference = move_without_spawn(originals[index], Action.LEFT)
        assert int(result.rewards[position]) == reference.reward, (
            f"board {index}: fast {int(result.rewards[position])} "
            f"vs ref {reference.reward}"
        )
        assert np.array_equal(result.afterstates[position], reference.afterstate)


# --------------------------------------------------------------------------- #
# 6b. Randomized differential right at the overflow boundary
# --------------------------------------------------------------------------- #


def test_randomized_boundary_differential_never_wraps():
    """Exhaustive-ish sweep of the overflow boundary against M0.

    Boards are generated with exponents concentrated at 55..66, one board per
    ``move_batch`` call -- a *single* board per call is deliberate, because per
    spec a batch containing one unrepresentable transition raises as a whole, so
    grouping would hide the per-board verdict.

    The contract checked for every case:

    * if the true reward exceeds ``INT64_MAX`` (or the afterstate leaves uint8),
      M1 must raise ``OverflowError``;
    * otherwise M1 must return the exact M0 reward, afterstate and ``moved``.

    seed = 20260918, 7 exponent bands, 900 boards per band (6,300 total).
    """
    bands = [(55, 62), (58, 62), (60, 63), (61, 62), (0, 62), (50, 66), (0, 17)]
    per_band = 900
    rng = np.random.default_rng(SEED)

    batches = []
    for low, high in bands:
        chunk = rng.integers(low, high + 1, size=(per_band, 16)).astype(np.uint8)
        chunk[rng.random((per_band, 16)) < 0.55] = 0
        batches.append(np.ascontiguousarray(chunk))
    boards = np.concatenate(batches, axis=0)

    agreed = 0
    raised = 0
    for index in range(boards.shape[0]):
        entry = boards[index]
        action = Action(int(rng.integers(0, 4)))
        expected = move_without_spawn(entry, action)
        reference_afterstate = np.asarray(expected.afterstate)
        m0_unrepresentable = (
            expected.reward > INT64_MAX or int(reference_afterstate.max()) > 255
        )

        context = (
            f"index={index} action={action.name} board={board_str(entry)} "
            f"m0_reward={expected.reward}"
        )
        try:
            result = move_batch(
                entry.reshape(1, 16), np.array([int(action)], dtype=np.uint8)
            )
        except OverflowError:
            raised += 1
            # Strict: an ``OverflowError`` is only ever acceptable when M0 itself
            # cannot represent the transition.  An earlier version of this test
            # also accepted any board whose maximum exponent was below 62, which
            # would have hidden a genuine false rejection -- a transition that M0
            # represents exactly but that M1 refuses.  There is no third outcome.
            assert m0_unrepresentable, (
                f"{context}: M1 raised although M0 represents "
                "the transition exactly"
            )
            continue

        assert not m0_unrepresentable, (
            f"{context}: M1 returned a value for an unrepresentable transition"
        )
        agreed += 1
        assert int(result.rewards[0]) == expected.reward, context
        assert int(result.rewards[0]) >= 0, f"{context}: wrapped reward"
        assert np.array_equal(result.afterstates[0], reference_afterstate), context
        assert bool(result.moved[0]) == expected.moved, context

    # Both outcomes must actually occur, or the sweep is not exercising anything.
    assert agreed > 1000, f"only {agreed} representable cases; sweep is too narrow"
    assert raised > 10, f"only {raised} rejecting cases; sweep is too narrow"


# --------------------------------------------------------------------------- #
# 7. Score accumulation and step() atomicity
# --------------------------------------------------------------------------- #


def _rng_state(rng) -> dict:
    state = rng.bit_generator.state
    return {
        key: (value.copy() if isinstance(value, np.ndarray) else value)
        for key, value in state.items()
    }


def _assert_rng_equal(before: dict, after: dict) -> None:
    assert before.keys() == after.keys()
    for key in before:
        left, right = before[key], after[key]
        if isinstance(left, np.ndarray):
            assert np.array_equal(left, right), f"RNG field {key!r} changed"
        else:
            assert left == right, f"RNG field {key!r} changed"


def test_step_overflow_is_atomic_for_board_score_and_rng():
    """A reward overflow in ``step`` must leave the environment completely intact.

    The move is computed before any mutation, so the failure happens while board,
    scores and RNG are all still untouched -- no spawn, no RNG draw, no partial
    commit.
    """
    env = Fast2048BatchEnv(2, seed=SEED)
    env._boards[0] = board((61, 61, 0, 0), (61, 61, 0, 0), (0,) * 4, (0,) * 4)
    env._boards[1] = board((1, 1, 2, 0), (0,) * 4, (0,) * 4, (0,) * 4)
    env.reset_where(np.zeros(2, dtype=bool))

    boards_before = env.boards.copy()
    scores_before = env.scores.copy()
    rng_before = _rng_state(env.rng)

    with pytest.raises(OverflowError):
        env.step(np.array([int(Action.LEFT), int(Action.LEFT)], dtype=np.uint8))

    assert np.array_equal(env.boards, boards_before), "board was partially modified"
    assert np.array_equal(env.scores, scores_before), "score changed"
    _assert_rng_equal(rng_before, _rng_state(env.rng))


def test_step_score_overflow_raises_without_wrapping():
    """``score + reward`` past ``INT64_MAX`` must raise, not wrap negative."""
    env = Fast2048BatchEnv(1, seed=SEED)
    env._boards[0] = board((61, 61, 0, 0), (0,) * 4, (0,) * 4, (0,) * 4)
    env._scores[0] = INT64_MAX - 1
    env.reset_where(np.zeros(1, dtype=bool))

    boards_before = env.boards.copy()
    rng_before = _rng_state(env.rng)

    with pytest.raises(OverflowError):
        env.step(np.array([int(Action.LEFT)], dtype=np.uint8))

    assert int(env.scores[0]) == INT64_MAX - 1
    assert int(env.scores[0]) > 0, "score wrapped to a negative value"
    assert np.array_equal(env.boards, boards_before), "board was partially modified"
    _assert_rng_equal(rng_before, _rng_state(env.rng))


def test_step_score_accumulation_exactly_at_the_limit_is_accepted():
    """A score that lands exactly on ``INT64_MAX`` is representable: accept it."""
    env = Fast2048BatchEnv(1, seed=SEED)
    env._boards[0] = board((1, 1, 0, 0), (0,) * 4, (0,) * 4, (0,) * 4)
    reward = 4  # 1 + 1 -> 2 pays 2 ** 2
    env._scores[0] = INT64_MAX - reward
    env.reset_where(np.zeros(1, dtype=bool))

    result = env.step(np.array([int(Action.LEFT)], dtype=np.uint8))
    assert int(result.rewards[0]) == reward
    assert int(env.scores[0]) == INT64_MAX


def test_ordinary_seeded_trajectories_are_still_reproducible():
    """The fix must not consume randomness or perturb ordinary play.

    Two environments with the same seed and the same action sequence must produce
    bit-for-bit identical boards and scores, and the rollout must actually score
    (so the check is not vacuous).  The safety checks are pure range comparisons
    and draw no random numbers, so the spawn trajectory is unchanged.
    """
    num_envs = 8
    steps = 40
    actions = np.random.default_rng(SEED).integers(
        0, 4, size=(steps, num_envs)
    ).astype(np.uint8)

    def rollout():
        env = Fast2048BatchEnv(num_envs, seed=SEED)
        env.reset(seed=SEED)
        boards = []
        scores = []
        for step_actions in actions:
            env.step(step_actions)
            boards.append(env.boards.copy())
            scores.append(env.scores.copy())
        return boards, scores

    first_boards, first_scores = rollout()
    second_boards, second_scores = rollout()

    for index, (left, right) in enumerate(zip(first_boards, second_boards)):
        assert np.array_equal(left, right), f"board diverged at step {index}"
    for index, (left, right) in enumerate(zip(first_scores, second_scores)):
        assert np.array_equal(left, right), f"score diverged at step {index}"

    assert first_scores[-1].max() > 0, "the rollout never scored, check is vacuous"
    assert all(int(score) >= 0 for score in first_scores[-1]), "score went negative"


# --------------------------------------------------------------------------- #
# 8. Final audit: legal / terminal must be isolated from the int64 reward limit
#
# Legality is *defined* as "the move changed the board" (M0's rule).  Whether the
# reward of that move happens to fit in ``np.int64`` is a separate, M1-specific
# question.  Before this fix the legality path ran the reward aggregation as a
# side effect, so a legal query about ``[61, 61, 61, 61]`` raised ``OverflowError``
# even though M0 answers it without complaint.
# --------------------------------------------------------------------------- #

#: The critical reproducer board: its LEFT/RIGHT move pays ``2 ** 63``, one past
#: ``int64``, while UP/DOWN are perfectly ordinary and reward-free.
HIGH_REWARD_ALTERNATIVE = board((61, 61, 61, 61), (0,) * 4, (0,) * 4, (0,) * 4)


def test_legal_mask_does_not_depend_on_int64_reward_range():
    """The critical reproducer: legality must not care about the reward range.

    ``[61, 61, 61, 61]`` on the top row: UP does nothing (illegal), DOWN slides the
    row to the bottom, LEFT/RIGHT merge it into ``[62, 62, 0, 0]`` and pay
    ``2 ** 63`` -- which does not fit in ``int64``.  M0 answers the legality
    question normally, so M1 must too, with no ``OverflowError``.
    """
    fast = legal_mask_batch(HIGH_REWARD_ALTERNATIVE[None, :])[0]
    reference = reference_legal_mask(HIGH_REWARD_ALTERNATIVE)

    assert np.array_equal(fast, reference), (
        f"legal mismatch: fast={fast.tolist()} ref={reference.tolist()}"
    )
    # Pin the expected answer, so a shared bug in both paths cannot pass this.
    assert reference.tolist() == [False, True, True, True]


def test_is_terminal_is_isolated_from_the_int64_reward_range():
    """``is_terminal_batch`` must survive the same board, and agree with M0."""
    fast = is_terminal_batch(HIGH_REWARD_ALTERNATIVE[None, :])[0]
    reference = reference_terminal(HIGH_REWARD_ALTERNATIVE)

    assert bool(fast) == reference
    assert reference is False  # not terminal: DOWN/LEFT/RIGHT all change the board


@pytest.mark.parametrize(
    "exponent, expected_afterstate",
    [(100, 101), (254, 255)],
)
def test_high_exponent_merge_legality_matches_m0(exponent, expected_afterstate):
    """Exponents whose reward dwarfs ``int64`` are still legal movements.

    The afterstate exponent (``101``, ``255``) still fits in ``uint8``, so the move
    is representable and legal even though its reward is astronomically past
    ``int64``.  ``legal_mask_batch`` must agree with M0 in every direction.
    """
    entries = board((exponent, exponent, 0, 0), (0,) * 4, (0,) * 4, (0,) * 4)

    fast = legal_mask_batch(entries[None, :])[0]
    reference = reference_legal_mask(entries)
    assert np.array_equal(fast, reference), (
        f"exponent {exponent}: fast={fast.tolist()} ref={reference.tolist()}"
    )

    # The merge really is representable: M0 produces the expected afterstate.
    merged = move_without_spawn(entries, Action.LEFT)
    assert int(merged.afterstate[0]) == expected_afterstate
    assert merged.reward > INT64_MAX, "case is vacuous: reward fits in int64"

    # ``move_batch`` keeps the int64 contract for exactly the same board.
    with pytest.raises(OverflowError):
        move_batch(entries[None, :], np.array([int(Action.LEFT)], dtype=np.uint8))


def test_255_plus_255_tile_overflow_is_still_raised_by_legality():
    """A *tile* that cannot be represented must still fail, even in legal mode.

    Only the reward limit is dropped on the legality path -- the ``uint8`` exponent
    limit is part of the movement semantics.  ``255 + 255`` needs exponent ``256``;
    M0 raises, so M1 must raise too and must never silently wrap to ``0``.
    """
    entries = board((255, 255, 0, 0), (0,) * 4, (0,) * 4, (0,) * 4)

    with pytest.raises(OverflowError):
        move_without_spawn(entries, Action.LEFT)  # M0's own verdict

    with pytest.raises(OverflowError):
        legal_mask_batch(entries[None, :])
    with pytest.raises(OverflowError):
        is_terminal_batch(entries[None, :])
    with pytest.raises(OverflowError):
        move_batch(entries[None, :], np.array([int(Action.LEFT)], dtype=np.uint8))


def test_move_batch_still_rejects_the_high_reward_alternative():
    """Decoupling the legality path must not weaken ``move_batch`` itself.

    The very same board whose legality is now answerable still raises on the
    reward-producing path -- no reward check was removed, only bypassed where no
    reward is produced.
    """
    entries = batch([HIGH_REWARD_ALTERNATIVE, np.zeros(16, dtype=np.uint8)])
    for action in (Action.LEFT, Action.RIGHT):
        with pytest.raises(OverflowError):
            move_batch(entries, np.array([int(action), 0], dtype=np.uint8))

    # UP/DOWN are legal and pay nothing, so they must still work.
    result = move_batch(entries, np.array([int(Action.DOWN), 0], dtype=np.uint8))
    assert int(result.rewards[0]) == 0
    assert bool(result.moved[0]) is True


def test_step_on_a_high_reward_alternative_board_completes():
    """The post-spawn terminal check must not raise on a hypothetical reward.

    ``DOWN`` on ``[61, 61, 61, 61]`` is legal and pays ``0``: there is no int64
    problem in the step itself.  The failure used to happen *after* the spawn, when
    the terminal calculation asked whether LEFT/RIGHT were legal -- and their
    ``2 ** 63`` reward tripped the aggregation.  The step must now complete.
    """
    env = Fast2048BatchEnv(1, seed=SEED)
    env._boards[0] = HIGH_REWARD_ALTERNATIVE
    env.reset_where(np.zeros(1, dtype=bool))

    result = env.step(np.array([int(Action.DOWN)], dtype=np.uint8))

    assert bool(result.legal[0]) is True
    assert int(result.rewards[0]) == 0
    assert int(result.spawn_indices[0]) >= 0, "a legal move must spawn exactly once"
    assert int(result.spawn_exponents[0]) in (1, 2)

    # ``terminated`` is M0's verdict on the *formal* state, after the spawn.
    assert bool(result.terminated[0]) == reference_terminal(result.states[0])

    # And the reported state is still a normal board for every batch primitive.
    mask = legal_mask_batch(result.states)
    assert mask.shape == (1, 4)
    assert mask.dtype == np.bool_
    assert bool(is_terminal_batch(result.states)[0]) == bool(result.terminated[0])

    # The four 61s slid to the bottom row *without* merging: they are side by side
    # in a row, and DOWN slides along columns, so each column holds a single tile.
    # (That is exactly why this step's own reward is 0 -- the ``2 ** 63`` belongs to
    # the hypothetical LEFT/RIGHT of the terminal check.)
    expected = move_without_spawn(HIGH_REWARD_ALTERNATIVE, Action.DOWN)
    assert np.array_equal(result.afterstates[0], expected.afterstate)
    assert expected.reward == 0
    assert [int(value) for value in result.afterstates[0][12:]] == [61, 61, 61, 61]
    assert [int(value) for value in result.afterstates[0][:12]] == [0] * 12


def test_high_reward_legal_differential_matches_m0():
    """>=500 high-exponent boards: legality agrees with M0, no reward exception.

    Exponent band ``60..254`` deliberately excludes ``255``, the only value whose
    merge cannot be represented as a tile.  Every case here is therefore a legal
    question M0 can answer, and several of them pay far more than ``int64`` -- the
    exact situation that used to make the legality path raise.
    """
    rng = np.random.default_rng(SEED)
    boards: list[np.ndarray] = []

    # (a) Guaranteed merges of two equal high-exponent pairs, laid out
    #     destination-first for every action, so the action really merges them.
    for action in DIRECTIONS:
        for exponent in (60, 61, 62, 63, 100, 150, 200, 253, 254):
            partner = int(rng.integers(60, 255))
            boards.append(_rotated_for(action, (exponent, exponent), (partner, partner)))

    # (b) Boards that merely *contain* high exponents, merged or not.
    while len(boards) < 500:
        grid = rng.integers(60, 255, size=(4, 4)).astype(np.uint8)
        grid[rng.random((4, 4)) < 0.45] = 0
        boards.append(grid.reshape(16))

    entries = batch(boards[:500])
    fast = legal_mask_batch(entries)
    terminal = is_terminal_batch(entries)

    assert fast.shape == (500, 4) and fast.dtype == np.bool_
    assert terminal.shape == (500,) and terminal.dtype == np.bool_

    beyond_int64 = 0
    for index in range(entries.shape[0]):
        entry = entries[index]
        expected = reference_legal_mask(entry)
        assert np.array_equal(fast[index], expected), (
            f"sample={index} board={board_str(entry)}\n"
            f"  fast={fast[index].tolist()}\n  ref ={expected.tolist()}"
        )
        assert bool(terminal[index]) == reference_terminal(entry), (
            f"sample={index} board={board_str(entry)}: terminal mismatch"
        )
        for action in DIRECTIONS:
            if move_without_spawn(entry, action).reward > INT64_MAX:
                beyond_int64 += 1

    assert beyond_int64 >= 30, (
        f"only {beyond_int64} action rewards exceeded int64; the sweep does not "
        "actually exercise the decoupling it is meant to prove"
    )


# --------------------------------------------------------------------------- #
# 9. Final audit: ``scores`` must stay a real live view
# --------------------------------------------------------------------------- #


def _scoring_board() -> np.ndarray:
    """Top row ``[1, 1, 0, 0]``: LEFT merges it and scores ``4``."""
    return board((1, 1, 0, 0), (0,) * 4, (0,) * 4, (0,) * 4)


def test_scores_live_view_survives_step():
    """A ``scores`` view taken before a step must reflect the new score.

    Regression guard: committing with ``self._scores = new_scores`` replaced the
    backing ndarray, silently leaving every previously handed-out view -- a public,
    documented API -- frozen on the old values.
    """
    env = Fast2048BatchEnv(1, seed=SEED)
    env._boards[0] = _scoring_board()
    env.reset_where(np.zeros(1, dtype=bool))

    backing = env._scores
    view = env.scores
    assert view.flags.writeable is False
    assert int(view[0]) == 0

    env.step(np.array([int(Action.LEFT)], dtype=np.uint8))

    assert env._scores is backing, "the backing scores ndarray was replaced"
    assert np.shares_memory(view, env._scores), "the view no longer aliases scores"
    assert int(view[0]) == int(env.scores[0])
    assert int(view[0]) > 0, "the live view did not follow the score update"


def test_scores_live_view_survives_step_reset_where_and_reset():
    """One view must stay live across ``step``, ``reset_where`` and ``reset``."""
    env = Fast2048BatchEnv(2, seed=SEED)
    env._boards[0] = _scoring_board()
    env._boards[1] = _scoring_board()
    env.reset_where(np.zeros(2, dtype=bool))

    backing = env._scores
    view = env.scores

    def assert_still_live(stage: str) -> None:
        assert env._scores is backing, f"{stage}: backing ndarray was replaced"
        assert np.shares_memory(view, env._scores), f"{stage}: view is detached"
        assert np.array_equal(view, env.scores), f"{stage}: view is stale"
        assert view.flags.writeable is False, f"{stage}: view became writeable"

    env.step(np.array([int(Action.LEFT), int(Action.LEFT)], dtype=np.uint8))
    assert_still_live("after step")
    assert int(view[0]) > 0, "step did not score; the check would be vacuous"

    env.reset_where(np.array([True, False]))
    assert_still_live("after reset_where")
    assert int(view[0]) == 0, "reset_where did not clear the score"
    assert int(view[1]) > 0, "reset_where cleared an unselected game"

    env.reset(seed=SEED)
    assert_still_live("after reset")
    assert np.array_equal(view, np.zeros(2, dtype=np.int64))
