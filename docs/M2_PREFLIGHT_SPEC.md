# M2 Preflight Specification

> **Scope:** this document records only what M2 Preflight actually changed.
> It is **not** a second master plan. The single authoritative plan remains
> `prompts/2048_AI_正式执行计划_M0_M1冻结_CI增强版_2026-09-18.md`.
> If this file ever disagrees with the master plan, the master plan wins.
>
> M2 Preflight does **not** implement the network. No Transformer, no Residual
> MLP, no Q / V / A head, no trainer, no replay, no self-play, no teacher, no
> C++, no CUDA, no GPU benchmark.

---

## 1. Frozen milestones (unchanged by this work)

| Milestone | Status | Tag | Commit |
| --- | --- | --- | --- |
| M0 — reference environment | completed, permanently frozen | `m0-reference-pass` | `3f2def1d95f56eff776e671143188947bf64485b` |
| M1 — fast / vectorized environment | FINAL AUDITED PASS, frozen | `m1-fastenv-audited-pass` | `e5486017a90eeec4fb9814de7880b3c413dc06dd` |

Both tags are **not moved** by M2 Preflight. `Reference2048Env` remains the golden
reference / correctness oracle, and M1 is not reopened.

The M2 Preflight cleanup commits on top of HEAD are M2 Preflight work — **not** a
revised M1 pass.

### M0 / M1 representation divergence (frozen, not revisited)

* M0 reward: arbitrary-precision Python `int`.
* M1 reward / score: `np.int64`, and an unrepresentable single-action reward or
  running score raises `OverflowError` — never a silent wrap.
* That reward limit must **not** contaminate movement legality: `legal_mask`,
  terminal and movement legality keep using the reward-independent movement path.
* `255 + 255 -> exponent 256` is a *tile representation* overflow, not a reward
  exception; M0 and M1 both raise `OverflowError` there.

This is the audited P1-A rule and is untouched by M2 Preflight.

---

## 2. The four Preflight items

Fixed order, all four required:

1. establish GitHub Actions CPU CI;
2. clean up the `Fast2048BatchEnv._terminated` fake cache and its misleading
   comments;
3. fix the always-true `assert np.any(env.scores != 0) or True`;
4. freeze the M2 CPU → GPU C-contiguous batch data contract.

Only when all four are done, the complete M0 + M1 suite is green **and** the
GitHub Actions run is green may the formal M2 network implementation begin.

---

## 3. Item 1 — GitHub Actions CPU CI

* Workflow: `.github/workflows/ci.yml`.
* Triggers: `push` and `pull_request`.
* Runner: `ubuntu-latest` (no Windows / macOS / OS matrix in the first version).
* Python: `3.12` (single version, via `actions/setup-python@v5`; checkout via
  `actions/checkout@v4`).
* Dependencies: the repository has **no** packaging / dependency manifest, so CI
  installs only what the CPU suite actually needs — `numpy` and `pytest`. No
  Poetry, no uv, no Conda, no packaging layer is introduced.
* Import path: the suite relies on the root `conftest.py`, which inserts `src/`
  into `sys.path`; CI therefore needs no `PYTHONPATH` override and no change to
  any frozen M0 / M1 package semantics.
* Test command: `python -m pytest` — the **complete** CPU-testable suite, never a
  subset.

### What the CPU CI is for

```
correctness
CPU regression
API regression
frozen-stage protection
```

### What the CPU CI is explicitly not for

No CUDA, no fake GPU, no `RTX 5060`, no GPU benchmark, no long training, no
self-play, no Teacher search, no long profiling. GPU milestone acceptance keeps
running on the local RTX 5060.

M0 and M1 frozen regression is now a permanent CI gate. From here on it may
**never** be weakened to make a milestone pass: no deleting old tests, no `skip`,
no `xfail`, no `pytest --ignore`, no collection exclusion, no config that hides a
frozen suite. If a frozen test fails after a new change, the default assumption is
that the change is wrong.

---

## 4. Item 2 — `_terminated` audit conclusion and removal

### Audit conclusion (verified against the frozen source)

In `src/game2048/fast_env.py`, `Fast2048BatchEnv._terminated` was:

* initialised in `__init__` (`self._terminated: Optional[np.ndarray] = None`);
* invalidated with `self._terminated = None` in `reset()` and `reset_where()`;
* written by `step()` through the `_refresh_terminated()` helper.

But **no later step ever read the previous value**. There was no reuse anywhere in
the module: every `step()` recomputed terminal from scratch. So the object was

```
cache write exists
cache reuse does not exist
```

while the source comments claimed the opposite — that the mask was *pipelined*,
*cached for the next step* and *reused*. That was misleading documentation of a
cache that did not exist.

### Decision taken

The master plan's minimal option was chosen: **delete the fake cache**. No cache
system was designed around it.

Removed:

* the `self._terminated` attribute from `__init__` and its comment;
* the `self._terminated = None` invalidation in `reset()`;
* the `self._terminated = None` invalidation in `reset_where()`;
* the whole `_refresh_terminated()` method (a pointless wrapper around a single
  call, with no remaining callers).

Changed in `step()`:

```python
# before
terminated = self._refresh_terminated()

# after
terminated = is_terminal_batch(self._boards)
```

### Corrected documentation

The class docstring, the `step()` docstring and the surrounding comments no longer
describe a pipelined terminal cache, a reused previous-step result or a value
cached for the next step. The remaining wording states the truth: `step` evaluates
`is_terminal_batch` on the current official board batch **after** the spawn has
completed, and that verdict is this step's `BatchStepResult.terminated`.

### Terminal semantics preserved

Terminal is still evaluated on

```
move
  -> spawn
  -> official next state s'
```

It is **not** afterstate terminal, and nothing about the verdict changed.

### Semantics explicitly verified unchanged

```
illegal action      : board unchanged, reward 0, score unchanged,
                      no spawn, no RNG consumption
terminal            : no legal actions
legal               : reward-independent movement-only path
reward / score      : explicit OverflowError, never silent wrap
tile exponent       : frozen M0 / M1 rule, still OverflowError
scores live view    : still a live in-place view
step atomicity      : still all-or-nothing
```

---

## 5. Item 3 — the always-true `or True` assertion

`tests/test_m1_batch_env.py::test_reset_zeroes_the_scores` contained

```python
assert np.any(env.scores != 0) or True
```

which can never fail and therefore proved nothing about `reset`.

### Why it could not simply be "fixed" by dropping `or True`

A freshly spawned random board does not reliably contain a mergeable pair, so
`reset()` followed by `LEFT` may legitimately produce zero reward. Asserting
`np.any(env.scores != 0)` on a random opening position would be unreliable or
flaky, and would not be a real regression test.

### Deterministic fix

The test now manufactures a real non-zero score through the **normal step path**:

1. create `Fast2048BatchEnv`;
2. place the same deliberately mergeable board into the internal board buffer for
   every game:
   `[1, 1, 0, 0] / [0, 0, 0, 0] / [0, 0, 0, 0] / [0, 0, 0, 0]`;
3. run a normal `env.step(LEFT)`, which merges `1 + 1 -> exponent 2` and pays
   exactly `2 ** 2 = 4`;
4. assert `legal`, assert `rewards == 4`, then assert `scores == 4`;
5. call `env.reset(seed=SEED)`;
6. assert every score is `0`.

The score is **not** faked by writing `env._scores` directly: the point is that the
real accumulation path produced a non-zero score first, so the test now proves
that `reset` clears a score that a genuine transition accumulated.

No other M1 test was modified: no reduced differential sample counts, no relaxed
equality, no changed overflow or D4 expectations, no deleted test, no `skip`, no
`xfail`, no seed changed to dodge a failure.

---

## 6. Item 4 — the frozen M2 CPU → GPU contiguous batch contract

### Module

`src/game2048/m2_data.py` — a new module dedicated to the M2 CPU batch boundary.

It deliberately does **not** extend the frozen M1 `fast_env.py`; M1 is frozen and
is not reopened. It contains no `torch`, no CUDA, no device, no pinned memory, no
`non_blocking`, no BF16 / FP16, no network, no forward pass and no action
selection. It only normalises and validates a NumPy CPU batch at the boundary
before the transfer.

### API

```python
def prepare_board_batch_for_transfer(boards: np.ndarray) -> np.ndarray: ...
```

One function, not a class. No config object, no allocator abstraction, no backend
or device abstraction.

### Official transfer boundary contract

```
type   : numpy.ndarray
shape  : (N, 16), N >= 1
dtype  : uint8
layout : C-contiguous
```

```text
Fast2048BatchEnv                     (M1, frozen)
    -> CPU board batch
    -> prepare_board_batch_for_transfer   (explicit boundary, this module)
    -> CPU -> GPU transfer            (M2 and later; not implemented yet)
    -> NN
```

### Contiguous input — zero copy

If `boards.flags.c_contiguous` is `True`, the **same object** is returned: no
`copy()`, no `ascontiguousarray`, no per-step duplication of the FastEnv batch.
The tests assert object identity (`result is boards`), not merely equality.

### Non-contiguous input — one explicit boundary copy

If the shape and dtype are legal but the array is not C-contiguous, the boundary
performs exactly one `np.ascontiguousarray(boards)`. The result keeps the same
shape and dtype, preserves the values exactly, and reports
`C_CONTIGUOUS = True`. This is the **only** copy this boundary is allowed to make.

The rule it enforces: a non-contiguous NumPy view must never make every network
forward perform implicit copies, implicit re-layouts or unmeasurable data
movement inside the training hot path.

### Invalid dtype — rejected, never converted

`int64`, `int32`, `int16`, `float32`, `float64`, `bool` (and any non-`uint8`
dtype) raise `TypeError`. There is no `astype(np.uint8)` shortcut: the M2 hot path
is not allowed to hide a dtype conversion, a copy or a memory re-layout.

### Invalid shape — rejected, never reshaped

`(16,)`, `(N, 4, 4)`, `(N, 15)`, `(N, 17)`, `(N, 32)`, `(16, 0)`, any other
`ndim != 2` shape, and the empty batch `(0, 16)` all raise `ValueError`. The
contract is exactly `(N, 16)` with `N >= 1`; nothing is silently reshaped into it.

### Input mutation

None. A contiguous input is handed back unchanged as the same object; a
non-contiguous input is converted into a new array. The caller's values — and, for
a strided view, the buffer it was cut from — are never modified.

### M1 API not reopened

`move_batch`, `legal_mask_batch`, `is_terminal_batch`, the `Fast2048BatchEnv`
public API and `_as_boards` are **not** modified to force every M1 external input
to be C-contiguous. Contiguity is a requirement of the **M2 pipeline boundary**,
not a reopening of the M1 API.

---

## 7. Preflight tests

`tests/test_m2_preflight.py` — CPU-only, never skipped, covering at least:

```
1.  valid (N,16) uint8 contiguous input accepted
2.  contiguous input zero-copy (result is input)
3.  non-contiguous valid uint8 input accepted
4.  result of non-contiguous input is C-contiguous
5.  values preserved exactly
6.  invalid dtype rejected
7.  invalid shapes rejected
8.  empty (0,16) rejected
9.  Fast2048BatchEnv.boards satisfies the boundary contract
10. caller input is never mutated
```

The non-contiguous fixture is genuinely non-contiguous (`base[:, ::2]` of a
`(N, 32)` buffer), and each such test asserts `not view.flags.c_contiguous` before
exercising the boundary, so a "non-contiguous" test can never silently become a
contiguous one.

---

## 8. Decisions that remain deferred

### C++ decision remains DEFER

A high movement share in the M1 profile is **not** grounds to migrate to C++,
pybind11, SIMD or a LUT now. The fixed decision order is:

```
finish the M2 network
  -> run the real closed loop
     Env -> CPU batch -> GPU -> NN -> Action -> Env
  -> end-to-end profiling
```

Only if that real closed loop shows CPU environment / Python / batch assembly /
H2D clearly starving the GPU may C++, pybind11, SIMD, a LUT, worker pipelines,
double buffering, pinned memory or async H2D be discussed. M2 Preflight does none
of this.

### Other deferred work

* The formal M2 network (Transformer, Residual MLP, Q / V / A) — **not started**.
* GPU throughput benchmark, batch-size sweep, BF16 / FP16 benchmark,
  `torch.compile` benchmark, M1 + M2 closed-loop benchmark — **not started**.
* The user's existing tuple 8×6 checkpoint — **not used**. It is reserved for the
  later Teacher stage.

Any performance work still follows the fixed order:

```
correct
  -> measure
  -> find the largest real bottleneck
  -> optimise that bottleneck
  -> re-benchmark
```

not a technology swap chosen by intuition.
