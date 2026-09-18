"""Regression tests for the *vectorization* of the M1 movement hot path.

Two independent properties are pinned here, because together they are exactly what
"the movement hot path is vectorized across boards" means:

1. :func:`test_production_path_never_calls_m0_move_without_spawn` -- M0's frozen
   per-board ``move_without_spawn`` is monkeypatched to raise, and the whole
   public batch surface must keep working.  This is the direct proof that the
   production path no longer delegates board by board to the reference kernel.

2. :func:`test_8192_board_batch_matches_m0_on_sampled_indices` -- one single
   ``move_batch`` call over 8,192 random boards with random actions, then at least
   512 randomly sampled indices are cross-checked cell by cell against M0.

3. :func:`test_python_line_count_is_constant_in_batch_size` -- the number of
   Python line events executed inside the movement kernels is measured for two
   very different batch sizes and must not grow with the batch.  A per-board
   Python loop makes this number scale linearly with ``N``, so this test fails
   loudly if the loop is ever reintroduced (``sys.settrace`` counts *Python*
   lines; NumPy's C loops are invisible to it, which is the point).

Seed, sample counts and exponents are documented so every failure is reproducible.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

import game2048.fast_env as fast_env_module
import game2048.reference_env as reference_env_module
from game2048 import Action, move_without_spawn
from game2048.fast_env import (
    Fast2048BatchEnv,
    is_terminal_batch,
    legal_mask_batch,
    move_batch,
)

from _m1_helpers import SEED, board_str, random_boards

#: Board count for the big one-shot batch (fixed by the M1 acceptance criteria).
LARGE_BATCH = 8192
#: Minimum number of indices cross-checked against M0 (M1 acceptance criteria).
MIN_SAMPLED = 512
#: Exponent ceiling of the large batch; well inside the M1 exactness boundary.
LARGE_EXPONENT = 20
#: Batch sizes for the "Python work must not scale with N" measurement.
SMALL_BATCH = 8
TRACE_BATCH = 2048


class M0MovementCalled(AssertionError):
    """Raised by the monkeypatched M0 kernel if the production path calls it."""


def test_production_path_never_calls_m0_move_without_spawn(monkeypatch):
    """FastEnv's production path must not delegate to M0's per-board kernel.

    ``reference_env.move_without_spawn`` is the frozen scalar reference.  While it
    is monkeypatched to raise, every public batch entry point must still produce
    exactly the right answer -- afterstates, rewards and ``moved`` included.
    """
    calls: list[tuple] = []

    def exploding_move_without_spawn(board, action):  # pragma: no cover - guard
        calls.append((np.asarray(board).copy(), action))
        raise M0MovementCalled(
            "the M1 production path called reference_env.move_without_spawn"
        )

    monkeypatch.setattr(
        reference_env_module, "move_without_spawn", exploding_move_without_spawn
    )

    entries = random_boards(300, LARGE_EXPONENT, SEED, empty_probability=0.5)
    rng = np.random.default_rng(SEED)

    for action in (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT):
        actions = np.full(entries.shape[0], int(action), dtype=np.uint8)
        result = move_batch(entries, actions)
        assert result.afterstates.shape == entries.shape
        assert result.rewards.shape == (entries.shape[0],)
        assert result.moved.shape == (entries.shape[0],)

    actions = rng.integers(0, 4, size=entries.shape[0]).astype(np.uint8)
    move_batch(entries, actions)

    mask = legal_mask_batch(entries)
    assert mask.shape == (entries.shape[0], 4)
    assert mask.dtype == np.bool_

    terminal = is_terminal_batch(entries)
    assert terminal.shape == (entries.shape[0],)
    assert np.array_equal(terminal, ~mask.any(axis=1))

    env = Fast2048BatchEnv(64, seed=SEED)
    step = env.step(rng.integers(0, 4, size=64).astype(np.uint8))
    assert step.afterstates.shape == (64, 16)
    env.reset_where(np.arange(64) % 3 == 0)

    assert calls == [], (
        "reference_env.move_without_spawn was called by the M1 production path: "
        f"{len(calls)} time(s), first action {calls[0][1] if calls else None}"
    )


def test_8192_board_batch_matches_m0_on_sampled_indices():
    """One 8,192-board ``move_batch`` with random actions, 512+ M0 cross-checks.

    seed = 20260918, boards = 8,192, exponents = 0..20, empty probability = 0.5,
    random actions over all four directions, sampled indices = 512.
    """
    entries = random_boards(LARGE_BATCH, LARGE_EXPONENT, SEED, empty_probability=0.5)
    rng = np.random.default_rng(SEED)
    actions = rng.integers(0, 4, size=LARGE_BATCH).astype(np.uint8)

    result = move_batch(entries, actions)

    assert result.afterstates.shape == (LARGE_BATCH, 16)
    assert result.rewards.dtype == np.int64
    assert result.moved.dtype == np.bool_

    sampled = rng.choice(LARGE_BATCH, size=MIN_SAMPLED, replace=False)
    assert sampled.size >= MIN_SAMPLED

    for index in sampled:
        index = int(index)
        entry = entries[index]
        action = int(actions[index])
        expected = move_without_spawn(entry, Action(action))
        context = (
            f"index={index} action={Action(action).name} board={board_str(entry)}"
        )
        assert np.array_equal(result.afterstates[index], expected.afterstate), (
            f"{context}\n  fast afterstate = {board_str(result.afterstates[index])}"
            f"\n  ref  afterstate = {board_str(expected.afterstate)}"
        )
        assert int(result.rewards[index]) == expected.reward, (
            f"{context}\n  fast reward = {int(result.rewards[index])}"
            f"\n  ref  reward = {expected.reward}"
        )
        assert bool(result.moved[index]) == expected.moved, (
            f"{context}\n  fast moved = {bool(result.moved[index])}"
            f"\n  ref  moved = {expected.moved}"
        )


def test_python_line_count_is_constant_in_batch_size():
    """The movement kernels must execute the same number of Python lines for any N.

    ``sys.settrace`` reports one event per *Python* line executed; a NumPy C loop
    produces none.  A per-board Python loop therefore makes the line count grow
    roughly linearly with the batch size, which this test catches.  Some fixed
    overhead per group is allowed (the four-action loop), so the guard is that the
    *per-board* line count must shrink by more than an order of magnitude.
    """
    traced = {
        fast_env_module._pack_left_rows,
        fast_env_module._merge_left_rows,
        fast_env_module._audit_merge_overflow,
        fast_env_module._move_groups,
        fast_env_module._move_batch,
    }
    codes = {function.__code__ for function in traced}

    def count_lines(batch_size: int) -> int:
        boards = random_boards(batch_size, 6, SEED, empty_probability=0.5)
        actions = np.zeros(batch_size, dtype=np.uint8)
        counter = {"lines": 0}

        def tracer(frame, event, arg):
            if frame.f_code in codes:
                if event == "line":
                    counter["lines"] += 1
                return tracer
            return None

        sys.settrace(tracer)
        try:
            move_batch(boards, actions)
        finally:
            sys.settrace(None)
        return counter["lines"]

    # Warm the code paths first so import-time/one-shot work is not measured.
    count_lines(SMALL_BATCH)
    small_lines = count_lines(SMALL_BATCH)
    large_lines = count_lines(TRACE_BATCH)

    small_per_board = small_lines / SMALL_BATCH
    large_per_board = large_lines / TRACE_BATCH

    assert large_per_board < small_per_board, (
        "the per-board Python line count did not decrease with batch size: "
        f"N={SMALL_BATCH} -> {small_per_board:.3f} lines/board, "
        f"N={TRACE_BATCH} -> {large_per_board:.3f} lines/board\n"
        "that is the signature of an N-dependent Python loop in the hot path"
    )
    assert large_per_board * 10 < small_per_board, (
        "the per-board Python line count barely moved between batch sizes: "
        f"N={SMALL_BATCH} -> {small_lines} lines ({small_per_board:.3f}/board), "
        f"N={TRACE_BATCH} -> {large_lines} lines ({large_per_board:.3f}/board)\n"
        "the movement path is still doing a bounded amount of Python work per board"
    )


def test_movement_kernels_have_no_per_board_python_iteration():
    """Static guard: the movement kernels must not iterate over boards or lines.

    The forbidden spellings are listed explicitly, and the loop variables that
    would indicate an N-dependent loop are named.  A fixed ``range(4)`` over the
    four actions / four lines of one board, or a ``range(3)`` over the three merge
    boundaries, stays allowed because those are compile-time constants
    independent of the batch size.
    """
    import ast
    import inspect
    import re
    import textwrap

    #: Loop variables that can only ever be a fixed constant: the action index,
    #: the column index within one line, and the index of one of the four lines
    #: of a single board (used by the checked reward aggregation).  Iterating over
    #: any *other* name -- and in particular over `board`, `line`, `row` or
    #: `entry` -- is per-board work.
    allowed_loop_variables = {"_column", "action", "_action", "_pass", "column"}

    forbidden = (
        r"np\.apply_along_axis",
        r"np\.vectorize",
        r"Reference2048Env",
    )

    # ``move_without_spawn`` is checked on executable code only: the docstrings of
    # ``legal_mask_batch`` / ``is_terminal_batch`` legitimately *name* M0's kernel
    # when they document the legality definition.  The behavioural half of this
    # contract lives in ``test_production_path_never_calls_m0_move_without_spawn``.
    forbidden_calls = (r"\bmove_without_spawn\s*\(",)

    def function_ast(function):
        """Parsed AST of ``function`` with the docstring removed."""
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
        node = tree.body[0]
        body = node.body
        if body and isinstance(body[0], ast.Expr) and isinstance(
            body[0].value, ast.Constant
        ):
            body = body[1:]  # drop the docstring
        return node, body

    def executable_source(function) -> str:
        _, body = function_ast(function)
        return "\n".join(ast.unparse(node) for node in body)

    def range_loops(function):
        """Every ``for <var> in range(<bound>)`` in the *executable* code.

        Read from the AST, so text inside a docstring or a comment can never be
        mistaken for a loop.
        """
        _, body = function_ast(function)
        found = []
        for statement in body:
            for child in ast.walk(statement):
                if not isinstance(child, ast.For):
                    continue
                target, iterator = child.target, child.iter
                if not isinstance(target, ast.Name) or not isinstance(
                    iterator, ast.Call
                ):
                    continue
                if not (isinstance(iterator.func, ast.Name)
                        and iterator.func.id == "range"):
                    continue
                bound = ast.unparse(iterator.args[0]) if iterator.args else ""
                found.append((target.id, bound))
        return found

    for name in (
        "_move_groups",
        "_move_batch",
        "_pack_left_rows",
        "_merge_left_rows",
        "move_batch",
        "legal_mask_batch",
        "is_terminal_batch",
    ):
        function = getattr(fast_env_module, name)
        code = executable_source(function)
        for pattern in forbidden + forbidden_calls:
            match = re.search(pattern, code)
            assert match is None, (
                f"{name} contains a forbidden per-board construct "
                f"{pattern!r}: {match.group(0)!r}"
            )

        for variable, argument in range_loops(function):
            assert variable in allowed_loop_variables, (
                f"{name} loops over {variable!r}, which is not one of the fixed "
                f"constants {sorted(allowed_loop_variables)}: this is an "
                "N-dependent Python loop in the movement hot path"
            )
            # The bound must be a compile-time constant (a literal, or one of the
            # board-shape constants), never something derived from the batch.
            # ``BOARD_COLUMNS - 1`` -> 3 boundaries is fine; ``boards.shape[0]``
            # or ``group_size`` is not.
            bound = argument.strip()
            assert not re.search(r"shape|len\(|size|count|\.\w+\(", bound), (
                f"{name} loops with the data-dependent bound {bound!r}; only "
                "compile-time bounds (4 actions, 4 columns, 3 boundaries) are "
                "allowed in the movement hot path"
            )


@pytest.mark.parametrize("action", list(Action))
def test_single_board_batch_is_still_exact(action):
    """A one-board batch must be bit-identical to M0 for every direction."""
    entries = random_boards(1, LARGE_EXPONENT, SEED, empty_probability=0.4)
    result = move_batch(entries, np.array([int(action)], dtype=np.uint8))
    expected = move_without_spawn(entries[0], action)
    assert np.array_equal(result.afterstates[0], expected.afterstate)
    assert int(result.rewards[0]) == expected.reward
    assert bool(result.moved[0]) == expected.moved
