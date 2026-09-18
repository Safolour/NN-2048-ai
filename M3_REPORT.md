# M3 Teacher 小规模验证报告

## 1. RESULT

**M3 PERFORMANCE INCOMPLETE**

按 `prompts/M3_IMPLEMENTATION_PROMPT.md` §26 的性能 blocker 规则停止。本轮没有把 M3 标记为 AUDITED PASS，也没有进入 M4/M5。

## 2. STARTING STATE

- M3 closeout ancestor: `4139b3b3c9c11216e7048bad46147e2fc5cc4e92`
- 开工 HEAD / origin/main: `7713855d442f7db60ae8582a8b8feb9561c39978`
- ancestor gate: PASS
- 4139b3b 之后提交仅为 planning/prompts/master-plan docs 更新；未发现已提交的 src/cpp/tests/benchmarks 冻结实现变更。
- M0/M1/M2 保持冻结；本轮未修改 M2 网络结构或 frozen backend。

## 3. CHECKPOINT

- upstream: `D:\CodexTasks\2048-ai\oneclick-m6-report\snapshots\COMPARATOR\ep4800000_7192719323a0.bin`
- local runtime: `teacher_checkpoints/m3/ordinary_td_comparator_ep4800000_7192719323a0.bin`
- size: 536,871,168 bytes
- SHA-256: `7192719323a073ba2b6b19b62cb7d46ef4aa90ecc8c4ae6baf27ad0c51566a84`
- upstream/local/expected SHA: identical
- format: `U2048NT6`, format_version=2, 8 patterns × 6 tuple, D4, float32 weights
- local binary is Git-ignored and remains untracked.

## 4. TUPLE EVALUATOR

- board encoding: `uint8[16]` exponent board; tuple feature input clamps exponents to 4-bit feature alphabet as required by the original checkpoint schema.
- deterministic read-only mmap loader implemented in `src/game2048/m3_tuple_teacher.py`.
- differential against upstream `2048_ai.exe infer --agent m6`: 4 representative boards, max absolute error **0.0**.
- raw evaluator remains classified as `RAW_TUPLE_HEURISTIC`.
- formal-state adapter remains `FORMAL_STATE_TUPLE_HEURISTIC`.
- upstream dirty-run provenance is supplemented with SHA-256 of critical source/executable files in `m3_teacher_semantics.json`.

## 5. SEARCH SEMANTICS

- search: formal-state Expectimax
- decision_depth: **3**
- chance nodes do not consume depth
- chance: uniform empty cell × 90% tile-2 / 10% tile-4
- action edge: immediate merge reward is added
- leaf: post-spawn formal state
- leaf evaluator: `tuple_afterstate_greedy_1ply_adapter`
- search value classification: `SEARCH_VALUE_RAW_LEAF`
- transposition key includes node type, remaining decision depth, and board bytes.

## 6. SEARCH CORRECTNESS

M3 unit suite currently passes:

`14 passed in 0.27s`

Coverage includes loader/header/schema, deterministic evaluation, search backup/depth/chance semantics, cache correctness differential, sample schema and game-level split helpers.

## 7. TEACHER SAMPLE SCHEMA

Implemented four-action schema with `teacher_value[4]`, `reward[4]`, `afterstate[4,16]`, `legal_mask[4]`, game/step/score/max-tile provenance, checkpoint SHA, decision depth, data source and value/search semantics. Illegal actions use NaN/zero sentinels and are masked.

## 8. DATASET

Semantic/profile corpus exists at `artifacts/m3/semantic_profile_states.npz`:

- 256 formal states
- 16 complete game IDs
- sourced from frozen tuple 1-ply complete games

The required 8192-state / 64-game small Teacher validation dataset was **not generated**, because §26 performance blocker fired first. Shrinking the dataset to claim PASS is explicitly forbidden.

## 9. VALUE SEMANTICS

Kept distinct:

- tuple raw: `RAW_TUPLE_HEURISTIC`
- formal-state leaf: `FORMAL_STATE_TUPLE_HEURISTIC`
- depth-3 search values: `SEARCH_VALUE_RAW_LEAF`
- realized future score: not equated to any raw/search value

No raw tuple/search value was used as absolute Student Q/V/A truth.

## 10. CALIBRATION

Not executed after performance STOP. Therefore no Pearson/Spearman/affine/R²/MAE/RMSE claim is made and no value is promoted to `FUTURE_SCORE` or `CALIBRATED_FUTURE_SCORE`.

## 11. STUDENT SANITY

Not executed after performance STOP. M3 did not proceed to Student training.

## 12. SEARCH PROFILE

Exact saved 256-state corpus, 16 games, depth=3:

- pre-batch: **0.704252 root decisions/s**
- low-risk leaf batching: **1.127172 root decisions/s**
- speedup: **1.6005×**
- optimized wall: **227.117 s / 256 roots**
- player nodes: 145,588
- chance nodes: 477,658
- leaf calls: 1,600,220
- cache hit rate: **47.48%**
- tuple evaluator time: **196.667 s**
- total Search time: **227.050 s**
- tuple evaluator share: approximately **86.6%**
- recursive Python/orchestration residual: **9.563 s**

This is still a severe evaluator/Search throughput bottleneck after the only low-risk batching cleanup allowed by the first-round M3 prompt.

## 13. PERFORMANCE GATE

**FAIL / STOP**

The prompt requires `M3 PERFORMANCE INCOMPLETE` when Search dominates wall-clock and Python/evaluator path still has a serious performance loss after correctness baseline + profiling + low-risk cleanup. That condition is met.

No C++ Search rewrite was attempted because the current prompt explicitly forbids doing that without a dedicated M3 Search Performance Unblock work order.

## 14. PYTEST

- M3-only suite: **14 passed / 0 failed**
- A later full-pytest attempt used the system Python 3.12 environment and failed during collection because that interpreter lacks `torch` and cannot load the already-built M2 extension DLL dependencies.
- This collection failure is an environment/interpreter mismatch, not evidence of an M3 regression.
- Because the performance STOP had already fired, no environment repair or full-suite qualification was used to override the STOP.

## 15. CI

Not run for an M3 candidate because M3 did not reach candidate-complete state.

## 16. FROZEN VERIFICATION

M0/M1/M2 frozen implementation was not intentionally modified. Frozen tags were not moved.

## 17. FILES CHANGED / ADDED

Current M3 worktree includes:
- `.gitignore`
- `teacher_checkpoints/README.md`
- `teacher_checkpoints/manifest.json`
- `src/game2048/m3_tuple_teacher.py`
- `src/game2048/m3_search.py`
- `src/game2048/m3_teacher_data.py`
- `tests/test_m3_tuple_teacher.py`
- `tests/test_m3_search.py`
- `tests/test_m3_teacher_data.py`
- `benchmarks/validate_m3_teacher_semantics.py`
- `benchmarks/benchmark_m3_teacher_search.py`
- `m3_teacher_semantics.json`
- `m3_teacher_profile.json`
- this report

## 18. ARTIFACTS

Local-only artifacts include semantic/profile states, pre-batch profile, original C++ evaluation game records, and the frozen 512 MiB checkpoint. The checkpoint is ignored by Git.

## 19. FINAL GIT STATE

No M3 audited tag is created. No M4/M5 work is started.

## 20. REMAINING BLOCKERS

Primary blocker: depth-3 Expectimax Teacher throughput. A dedicated **M3 Search Performance Unblock**施工单 is required before continuing calibration, the 8192-state dataset, Student sanity, full candidate CI, and independent audit.
