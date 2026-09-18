# M3 Teacher 小规模验证报告

## 1. RESULT

**M3 SEARCH PERFORMANCE UNBLOCK PASS**

本结论只表示 M3 的 depth-3 Expectimax Search 性能阻塞已经解除，不表示 M3 candidate / AUDITED PASS。没有创建 M3 tag，没有冻结 M3，也没有进入 M4/M5。

## 2. STARTING STATE

- Performance Unblock 起点：`240fa3c8b3fcf3fa497d9f11c2aea262d31b46d6`
- WIP baseline 之前的 M3 correctness/performance-stop 工作已保留。
- 固定 checkpoint：`teacher_checkpoints/m3/ordinary_td_comparator_ep4800000_7192719323a0.bin`
- checkpoint SHA-256：`7192719323a073ba2b6b19b62cb7d46ef4aa90ecc8c4ae6baf27ad0c51566a84`
- 历史 Search baseline：1.127171768 roots/s，227.117 s / 256 roots，formal-leaf/evaluator share ~86.6%。
- 本轮仅做 exact performance engineering；Search semantics、checkpoint、depth、chance distribution、backup 均未改变。

## 3. WIP BASELINE COMMIT

WIP baseline commit：

`a433d1e4d73eea2a62df0df994487e4e015da1c4`

message：`m3: checkpoint correctness-pass search-performance-blocked baseline`

该 commit 固化了性能解锁前的 correctness-pass / performance-blocked 状态，后续优化均从此基础增量进行。

## 4. OFFICIAL PYTHON / FULL PYTEST BASELINE

最终指定解释器：

`D:\sd-webui-forge-aki-v1.0\python\python.exe`

Performance Unblock 收口后的完整 pytest：

- **525 passed**
- **0 failed**
- **0 skipped**
- **0 xfailed**
- 6 warnings，均为既有 PyTorch Transformer nested-tensor warning
- wall：24.80 s

## 5. EVALUATOR DECOMPOSITION

P1 对 `formal_state_leaf_batch` 做了阶段拆分。2048-board / 100 iterations 的代表性累计时间：

- M1 NumPy `move_batch`：1.2262 s
- legal compaction：0.0237 s
- stage selection：0.0268 s
- feature packing：1.5860 s
- random weight lookup：1.2784 s
- weight accumulation：0.1632 s
- reward + continuation：0.0086 s
- max legal：0.0279 s
- Python call/allocation residual：2.1462 s
- formal total reference：6.4869 s

这证明原先 86.6% 的“tuple evaluator”包裹时间并非只有随机大表 lookup，还混有 NumPy movement 与 Python/array orchestration。

## 6. LEAF BATCH DISTRIBUTION

同一 256-root / 16-game / depth=3 corpus：

- formal leaf states：1,600,220
- Python leaf batch calls：176,814
- mean batch：9.05
- median：8
- p90：14
- p99：26
- max：28

histogram：

- 2–4：28,143
- 5–8：78,306
- 9–16：59,563
- 17–32：10,802

真实工作负载以小 batch 为主，因此 primitive 优化必须在 batch 8–32 区间真正有效。

## 7. LEAF DUPLICATION

exact board-bytes 统计：

- total leaf formal states：1,600,220
- unique leaf states：1,279,347
- within-root duplicate count：148,976
- within-root duplicate ratio：9.31%
- global/cross-root duplicate count：320,873
- global duplicate ratio：20.05%

存在可观 duplicate，但本轮最终 throughput 已满足 gate，因此没有继续为了 dedup 进行额外工程化。

## 8. UPSTREAM M6 REFERENCE

上游参考：

- `m6.cpp` SHA-256：`3152d41c16b5f69122d2629e892a4bf8784eca877a2ea49ea1294077bc7f52c6`
- `m6.hpp` SHA-256：`9c88ad5fbd0a391b39f86ec3e1d5f2306f6a83b6e03df5709131ec1c257fb23d`

256-board upstream `2048_ai.exe infer --agent m6` anchor differential：

- max abs error：0.0
- action disagreement：0

外部 dirty repo 仅作为语义/实现参考，没有形成 build dependency。

## 9. C++ TUPLE BACKEND

新增独立 backend：

- `cpp/m3_tuple_backend/`
- `src/game2048/m3_tuple_backend.py`

生产语义保持：

- U2048NT6
- 8 patterns × 6 cells
- D4 8 symmetries
- 4-bit exponent saturation
- pattern-major -> symmetry-minor 64-slot 累加顺序
- float32 weights
- read-only NumPy memmap/buffer
- stage selection 支持 uint8 exponent 0..255
- scalar prefetch variant

Python evaluator 保留为 correctness oracle。

## 10. DIFFERENTIAL CORRECTNESS

C++ tuple evaluator：

- 10,019 boards Python vs C++ no-prefetch：bit-identical，max abs error 0.0
- 10,019 boards Python vs C++ prefetch：bit-identical，max abs error 0.0
- no-prefetch vs prefetch：bit-identical
- 256 formal leaves：bit-identical
- 10,000 formal-state leaves，在复用 frozen M2 movement 后：bit-identical，max/mean abs error 0.0

最终 exact Search differential：

- 256 roots
- 911 legal action values
- legal mask equal：true
- action values：bit-identical
- max abs error：0.0
- mean abs error：0.0
- best-action disagreement：0
- player/chance/leaf/move/chance-outcome/cache lookup/cache hit counts：全部一致

## 11. PRIMITIVE A/B

C++ tuple primitive 的关键 steady-state speedup：

- batch 1：~5.40×
- batch 16：**11.57×**
- batch 32：**10.99×**
- batch 128：~8.34×
- batch 2048：~4.05×
- batch 8192：~3.96×

真实关键 batch 16–32 明显超过施工单 >=3× 要求。

memory-safety gate：

- N=1：PASS
- empty batch：PASS
- read-only weights：PASS
- repeated calls exact：PASS
- independent memmap owner lifetime：PASS

## 12. SEARCH A/B

P5 只替换 tuple evaluator、其余 Search 不变时：

Python contemporaneous baseline：

- 1.6191 roots/s
- 158.112 s / 256 roots

C++ no-prefetch：

- 2.2375 roots/s
- 114.416 s / 256 roots

C++ prefetch：

- **2.3751 roots/s**
- 107.783 s / 256 roots

P5 correctness 全部 bit-identical，但 throughput 仍低于 5 roots/s，因此按施工单进入 evidence-driven P6。

## 13. REPROFILE

P5 C++ prefetch 后 profile 仍显示 formal-leaf wrapper 占主导，其中实际新热点是 formal leaf 内部仍调用 M1 NumPy movement。

P6 后最终 256-root profile：

- total Search：40.4215 s
- wall：40.5036 s
- tuple/formal-leaf：17.9425 s，约 44.4%
- move generation：10.7185 s，约 26.5%
- chance expansion：5.4905 s，约 13.6%
- hash：0.4278 s
- recursive Python orchestration residual：5.8422 s，约 14.5%

最终已不存在单项 >=50% 或明显可直接修复的 Python/NumPy 数量级损失。

## 14. SECOND-STAGE OPTIMIZATION

P6 仅做了一项由 profile 直接证明的低风险改动：

当 `TupleTeacher.backend == "cpp"` 时，`formal_state_leaf_batch` 的四动作展开复用已经 frozen / audited 的 M2 C++ `scalar+row-lut` movement primitive；Python oracle 路径保持原 M1 NumPy movement，不修改 frozen M2 source。

P6 micro A/B：

- batch 2：11.81×
- batch 4：16.09×
- batch 8：**11.76×**
- batch 9：9.78×
- batch 14：10.36×
- batch 16：9.00×
- batch 28：7.16×
- batch 32：7.16×
- batch 128：3.65×

所有测试 batch 均 bit-identical。

## 15. FINAL THROUGHPUT

最终 exact 256-root corpus：

- **6.320418629 roots/s**
- wall：**40.5036462 s**
- mean：0.157902 s/root
- median：0.142790 s/root
- p95：0.373139 s/root
- max：0.519936 s/root
- nodes/s：54,895.45
- cache hit rate：47.4849%

相对历史 1.127171768 roots/s baseline：

- **5.6073× speedup**

满足 >=5.0 roots/s Performance Unblock gate。

## 16. 8192 PROJECTION

按最终 exact 256-root throughput：

- projected 8192 Search time：**1296.116678 s**
- 即约 **21.60 分钟**

要求 <=30 分钟，PASS。

## 17. INVALIDATED DEPTH3 ROLLOUT DIAGNOSTIC

上一版施工单曾错误把 calibration continuation 解释成“每步 repeated depth-3 Expectimax 一直 rollout 到 terminal”，并把 2048 rollout 总 wall-clock 当作 Performance Unblock gate。

最新版施工单明确废止该解释：

- 此诊断**不是** Search Unblock PASS 条件
- 不得继续补跑
- 不得用于 calibration fit
- 普通 M3 正式 calibration continuation 改为 frozen checkpoint 原生 `greedy_1ply` policy

遗留诊断进度保存在：

`artifacts/m3/calibration_viability_progress/`

恢复时状态：

- 8 slots
- 7 complete
- 1 interrupted
- interrupted index 0 已到 10,304 decisions / 1384.5 s，仍未 terminal

这些文件只保留为“错误 repeated depth-3 continuation 会导致极长完整对局”的诊断 evidence，不计入正式 M3 calibration 完成度。

## 18. PYTEST

最终指定官方 Python full regression：

**525 passed, 0 failed, 0 skipped, 0 xfailed**

本地 correctness gate：PASS。

## 19. CI

已更新 `.github/workflows/ci.yml`，Ubuntu CI 将同时：

1. build frozen M2 C++ fast backend
2. build M3 C++ tuple backend
3. run full pytest

CI 不依赖真实 512 MiB checkpoint。

当前报告写入时：**等待本轮 implementation commit push 后触发远端 GitHub Actions；最终 CI 结果将在收口后回写。**

## 20. FROZEN VERIFICATION

frozen tags 解引用到 commit：

- M0 `m0-reference-pass^{}` -> `3f2def1d95f56eff776e671143188947bf64485b`
- M1 `m1-fastenv-audited-pass^{}` -> `e5486017a90eeec4fb9814de7880b3c413dc06dd`
- M2 `m2-network-audited-pass^{}` -> `6a9da5b1bf7c72207acd6e89bc667d439d4823a8`

本轮没有移动 frozen tags，没有修改 frozen M2 backend source。

## 21. FILES CHANGED

Performance Unblock implementation 主要包括：

- `.github/workflows/ci.yml`
- `cpp/m3_tuple_backend/CMakeLists.txt`
- `cpp/m3_tuple_backend/m3_tuple_backend.cpp`
- `src/game2048/m3_tuple_backend.py`
- `src/game2048/m3_tuple_teacher.py`
- `tests/test_m3_tuple_backend.py`
- `benchmarks/benchmark_m3_tuple_evaluator.py`
- `benchmarks/benchmark_m3_tuple_backend.py`
- `benchmarks/benchmark_m3_search_unblock.py`
- `benchmarks/benchmark_m3_p6_reprofile.py`
- `benchmarks/benchmark_m3_p6_search.py`
- `m3_tuple_backend_benchmark.json`
- `m3_search_performance_unblock.json`
- `m3_teacher_profile.json`
- `M3_REPORT.md`

用户主动修改的主计划 / M3 prompts，以及用户主动删除的旧 M1/M2 临时 JSON，不属于本轮 implementation commit。

## 22. ARTIFACTS

权威/保留 artifacts：

- `m3_tuple_backend_benchmark.json`
- `m3_search_performance_unblock.json`
- `m3_teacher_profile.json`
- `artifacts/m3/m3_tuple_evaluator_profile.json`
- `artifacts/m3/m3_p6_reprofile.json`
- `artifacts/m3/semantic_profile_states.npz`
- `artifacts/m3/search_ab_*_values.npz`
- `artifacts/m3/calibration_viability_progress/`（已废止 diagnostic，仅保留证据）

512 MiB checkpoint 继续保持 Git-ignored，不进入 Git history。

## 23. FINAL GIT STATE

本报告写入时处于 closeout implementation worktree，尚未创建 M3 tag，尚未把 M3 标记为 frozen/AUDITED PASS。

Search Performance Unblock implementation commit / push / CI 完成后，本节将补充最终 commit 与远端状态。

## 24. REMAINING BLOCKERS

**Search 性能不再是 blocker。**

剩余工作属于普通 M3，而不是 Search Performance Unblock：

- value calibration（按正式 `greedy_1ply` continuation）
- 8192-state / >=64-game Teacher dataset
- game-level split
- Student sanity
- final M3 candidate report
- candidate full pytest / remote CI
- independent audit

## 25. NEXT M3 RESUME POINT

Performance Unblock PASS 后必须立即停止性能工程。

恢复 `prompts/M3_IMPLEMENTATION_PROMPT.md`，从原性能 STOP 点之后继续：

`value calibration -> 8192-state/64-game dataset -> game-level split -> Student sanity -> final M3 report -> full pytest -> candidate commit/push -> GitHub Actions -> STOP for independent audit`

不得重做 checkpoint copy、loader format gate、Search semantics、256-state corpus 或本轮已通过的性能 correctness 工作。
