# M3 Search Performance Unblock — 严格施工提示词

> 项目：Safolour/NN-2048-ai
> 正式工作区：D:\CodexTasks\NN-2048-ai
> 当前阶段：M3 Teacher 小规模验证中的性能解阻子阶段
> 本文件不是新的总计划。唯一权威总计划与 M3_IMPLEMENTATION_PROMPT.md 继续有效。
> 本任务只解除 depth=3 Expectimax / tuple leaf evaluator 的性能 blocker。

# 0. 开工前必须读取

完整读取：
1. prompts/2048_AI_正式执行计划_M0_M1冻结_CI增强版_2026-09-18.md
2. prompts/M3_IMPLEMENTATION_PROMPT.md
3. 本文件
4. M3_REPORT.md
5. m3_teacher_profile.json
6. m3_teacher_semantics.json

若真实冲突：STOP；报告冲突；不得自行裁决。

# 1. 冻结基础

M0: m0-reference-pass -> 3f2def1d95f56eff776e671143188947bf64485b
M1: m1-fastenv-audited-pass -> e5486017a90eeec4fb9814de7880b3c413dc06dd
M2: m2-network-audited-pass -> 6a9da5b1bf7c72207acd6e89bc667d439d4823a8

绝对不得移动这三个 tag；不得修改 frozen M0/M1/M2 semantics。

# 2. 当前 M3 已完成事实

不得重做以下已经通过的工作，除非出现新的 correctness evidence：

- 本地 checkpoint: teacher_checkpoints/m3/ordinary_td_comparator_ep4800000_7192719323a0.bin
- SHA256: 7192719323a073ba2b6b19b62cb7d46ef4aa90ecc8c4ae6baf27ad0c51566a84
- size: 536,871,168 bytes
- format: U2048NT6 format_version=2
- upstream/local/expected SHA 已一致
- Python loader 已与上游 2048_ai.exe infer --agent m6 做 differential，max abs error = 0.0
- existing Python loader 现在是 M3 tuple evaluator correctness oracle
- decision_depth=3 formal-state Expectimax correctness baseline 已建立
- M3 专项测试：14 passed / 0 failed
- 固定 corpus: artifacts/m3/semantic_profile_states.npz
- corpus = 256 formal states / 16 complete game IDs / depth=3

# 3. 正式性能 baseline

优化前 root throughput: 0.704252 root decisions/s
现有 low-risk leaf batching 后: 1.127171768 root decisions/s
256 roots wall: 227.117 s

Profile：
- player nodes: 145,588
- chance nodes: 477,658
- leaf calls: 1,600,220
- cache lookups: 623,246
- cache hits: 295,948
- cache hit rate: 47.4849%
- formal-leaf/evaluator time: 196.667 s
- total Search: 227.050 s
- evaluator share: ~86.6%
- move generation: 13.034 s
- chance expansion: 7.139 s
- Python/orchestration residual: 9.563 s

这个 baseline 必须永久保留并作为所有 A/B 的 before。

# 4. 本阶段唯一目标

只解除 exact depth=3 Expectimax + tuple leaf evaluator 的数量级性能 blocker。

禁止：
- 改 Teacher 算法
- 改 decision_depth
- 把 depth=3 偷换成 depth=1
- 缩 dataset
- 换 checkpoint
- 改 Search backup/chance 概率
- 开始 calibration
- 开始 8192-state 正式数据
- 开始 Student sanity
- 进入 M4/M5

Performance Unblock PASS 后才回普通 M3。

# 5. 先纠正 profiling 口径

现有 tuple_evaluator_seconds 实际包住 formal_state_leaf_batch，不等于纯 weight lookup。
第一轮必须拆出：
A. M2 move_batch 四动作
B. legal afterstate compaction
C. stage selection
D. 8 patterns × 8 symmetries × 6 cells feature packing
E. 64 random weight lookups
F. 64-weight accumulation
G. reward + continuation
H. max legal action
I. Python call/allocation overhead

# 6. P0 — 保全当前 M3 WIP

当前 working tree 尚未形成 M3 candidate。
禁止 git reset --hard / git clean / git stash / git restore / 删除 artifacts / 覆盖 semantic corpus。

先记录：git status --short、HEAD、origin/main、三个 frozen tag resolution。
.vs/ 不得 stage；512 MiB checkpoint 必须继续 Git ignored。

# 7. P0.1 — 修正 full pytest 执行环境

上轮 full pytest 的 collection failure 是因为误用系统 Python 3.12，不是 M3 regression。
本轮固定使用：D:\sd-webui-forge-aki-v1.0\python\python.exe
该环境应为 Python 3.11.9 / PyTorch 2.9.1+cu130。

允许做 DLL/PATH/build 环境准备；禁止为了错误解释器去改 frozen M2 source。

运行 full pytest。预期至少 505 frozen tests + 14 current M3 tests，即至少 519 collected。
要求 0 failed / 0 skipped / 0 xfailed。
若正确正式 Python 下仍是真 regression：STOP。

# 8. P0.2 — 保存 performance-blocked WIP baseline

只有 full pytest green 后，创建普通 WIP commit：
m3: checkpoint correctness-pass search-performance-blocked baseline

允许包含当前 M3 source/tests/benchmarks/report/profile/semantics/manifest/.gitignore。
默认不 commit semantic corpus、大型 artifacts、512 MiB checkpoint。
这个 commit 不是 M3 candidate，不创建 M3 tag。

# 9. P1 — evaluator 精细 microprofile

新增 benchmarks/benchmark_m3_tuple_evaluator.py。

afterstate_values batch 固定测：1,16,32,128,512,2048,8192。
formal_state_leaf_batch batch 固定测：1,8,16,32,128,512,2048。

warmup >=20；正式 >=100 iterations 或 >=2 sec，取更严格者。
记录 boards/s、us/board、每子阶段 wall-time、RSS。

# 10. P1.1 — 真实 leaf batch 分布

在完全相同的 256-root corpus 上记录 formal_state_leaf_batch 的调用次数和 batch-size histogram：
1 / 2-4 / 5-8 / 9-16 / 17-32 / 33-64 / >64。
同时记录 mean/median/p90/p99/max。
必须回答 1,600,220 leaf states 实际由多少次 Python batch call 承载。

# 11. P1.2 — leaf duplicate analysis

统计：total leaf formal states、unique leaf formal states、within-root duplicate ratio、cross-root duplicate ratio。
使用 exact board bytes。
只诊断，不改变 Search semantics。
只有 duplicate ratio 明显，后续才允许做 leaf dedup/cache。

# 12. P2 — C++ tuple evaluator 决策

由于 evaluator/formal leaf wrapper = 86.6% Search wall-clock，且上游训练项目已有真实 C++ M6 evaluator，本施工单固定：C++ batch tuple evaluator = YES。

第一版只迁：feature packing、stage selection、64 random weight lookup、64-weight accumulation。
禁止第一步就把整个 Expectimax 搬进 C++。

# 13. 上游 C++ 只作为语义/性能参考

上游参考：
D:\CodexTasks\2048-ai\src\ntuple\m6.cpp
D:\CodexTasks\2048-ai\src\ntuple\m6.hpp

已确认上游 sum_stage 会先构造 64 个 weight cell 地址，执行 prefetch，再按固定 pattern-major / symmetry-minor 顺序累加。

M3 可以参考这一访问顺序，但禁止运行时链接 dirty 外部 repo、include 外部绝对路径、把另一个 repo 变成 build dependency。

必须记录实际参考的 m6.cpp / m6.hpp SHA256。

# 14. P2 backend 固定目录

新增：
cpp/m3_tuple_backend/CMakeLists.txt
cpp/m3_tuple_backend/m3_tuple_backend.cpp
src/game2048/m3_tuple_backend.py
tests/test_m3_tuple_backend.py
benchmarks/benchmark_m3_tuple_backend.py

不得修改 cpp/m2_fast_backend、m2_fast_backend.py、m2_rollout_env.py。

# 15. C++ evaluator API

至少提供 afterstate_values(boards, weights, stage_thresholds) -> float32[N]。
boards 必须 uint8 (N,16) C-contiguous。
weights 使用现有 read-only NumPy memmap/buffer view，禁止每次复制 512 MiB、每次重新 mmap、每次重新 parse header。

如果实际 comparator stage_count=1，可以有 stage_count==1 fast path，但不得硬编码格式永远只有 stage 0。

# 16. 数值语义锁死

访问/累加顺序固定：pattern-major -> symmetry-minor，共 64 slots。
exponent >15 必须 clamp 到15。
不得改 tuple patterns、D4 order、nibble packing、stage policy、float32 checkpoint interpretation。

建议与上游一致：读取 float32 weights，double 累加，finite/range check，最后 cast float32。

# 17. prefetch

第一版允许 scalar no-prefetch 与 scalar prefetch 两种模式。
只在 end-to-end 有收益时默认启用 prefetch。
禁止第一版就上 AVX2 gather / AVX-512 / GPU table / weight compression。

# 18. P3 differential correctness

任何 Search benchmark 前先 PASS。

Python oracle vs C++：至少 10,000 random afterstates，另覆盖 all-zero、single tile、dense、exp 0..15、16/17/20/31/63/127/255、stage threshold boundary、全部 D4。

目标 bit-identical float32。
如果因严格确认的 reduction 实现细节无法 bit-identical，max abs error 必须 <=1e-5，并证明 one-ply ranking、formal leaf ranking、Search best action 不改变。

另取至少 256 representative boards 与上游 2048_ai.exe infer --agent m6 做 anchor differential。
记录 max abs error / mean abs error / action disagreement。
action disagreement 必须 = 0。

同一 boards 在 batch=1/16/128/2048 结果必须一致。

# 19. memory safety

测试 N=1、wrong dtype、wrong shape、empty batch、non-contiguous policy、read-only weights、repeated calls、memmap owner lifetime。
禁止 C++ 保存无 owner lifetime 的裸 NumPy pointer。

# 20. P4 primitive A/B

correctness PASS 后比较：Python NumPy/memmap、C++ scalar no-prefetch、C++ scalar prefetch。
batch: 1,16,32,128,512,2048,8192。
记录 boards/s、us/board、speedup、RSS、first-call vs warm steady-state、prefetch gain。

代表性 batch 16~32 上 C++ 若没有至少 3.0x primitive speedup：先查重映射/复制/GIL/分配/prefetch问题；确认无实现错误仍 <3x，则 M3 SEARCH PERFORMANCE INCOMPLETE，STOP。

# 21. GIL

纯 C++ batch 计算开始后必须释放 GIL；buffer/Python object validation 完成后再释放。
释放后不得调用 Python API。

# 22. P5 — 先只替换 evaluator backend

第一轮只替换 TupleTeacher.afterstate_values 的计算 backend。
formal_state_leaf_batch、ExpectimaxTeacher、backup、chance、cache、decision_depth 全部不改。

保留明确 selector：python / cpp。
Python oracle不得删除。

# 23. exact 256-root Search A/B

必须使用同一 artifacts/m3/semantic_profile_states.npz、256 roots、16 games、depth=3、cache=True。

比较 current Python baseline vs C++ evaluator backend。
记录 root decisions/s、wall、tuple evaluator time、move/chance/orchestration、cache hit、全部 node counts。

A/B 的 player_nodes、chance_nodes、leaf_calls、cache lookups、cache hits 必须一致。
全部 256 roots 的 legal mask 和 best action 必须一致。

# 24. 第一性能 gate

如果 exact 256-root corpus >=5.0 root decisions/s，则达到本轮 operational floor。
5 roots/s 对应 8192 roots 约 27.4 分钟。

若 <5.0 roots/s，不得立刻写 C++ Expectimax，必须先执行 P6 reprofile。

# 25. P6 — 只优化新的最大热点

固定分支：
- evaluator/formal leaf 仍 >=50%：先优化 evaluator/batching/dedup。
- Python recursion/orchestration >=30%：才允许 frontier batching。
- move generation >=30%：优先复用 frozen M2 C++ movement primitive。
- chance expansion >=30%：优化 outcome representation/allocation，但概率语义不变。

不满足阈值时只优化实际最大热点。

# 26. P6.1 cross-chance / root-level leaf batching

若真实 batch 过小，允许 two-phase exact leaf evaluation：收集 leaf formal states -> exact duplicate 可 dedup -> 大 batch C++ evaluator -> 写回 -> 原树结构/概率 backup。

禁止 chance sampling、beam、probability truncation、top-k、leaf subsampling、降 depth。
必须对 256 exact roots 与原 recursive Search 做逐 root value/action differential。

# 27. P6.2 frontier batching

只有 C++ evaluator PASS、reprofile证明 Python orchestration >=30%、且仍 <5 roots/s 时才允许 explicit frontier/DAG evaluation。

仍必须 exact Expectimax：same probabilities、same depth、same node semantics、same backup。
优先 Python/NumPy orchestration + 已有 C++ primitives。

# 28. 整套 C++ Search 授权条件

只有同时满足以下六项才允许考虑 C++ Expectimax core：
1. C++ tuple evaluator correctness PASS
2. primitive >=3x
3. cross-chance/root batching 已测试
4. frontier batching 已测试（如适用）
5. exact 256-root throughput 仍 <5 roots/s
6. reprofile 显示 Python recursion/orchestration仍 >=30%

条件不全，禁止实现整套 C++ Search。

# 29. 如果最终需要 C++ Search

新增 cpp/m3_search_backend/，不得塞进 M2 backend。
必须 exact depth=3、chance不减depth、90/10、empty-cell uniform、immediate reward exactly once、formal-state leaf、same tuple adapter、terminal future=0、完整 transposition key。

Python Expectimax必须保留为 correctness oracle。

# 30. C++ Search correctness gate

若进入：比较 256 semantic roots + 10,000 generated legal formal states。
Python vs C++ 比 legal mask、四 action values、best action、terminal、depth semantics。
best action disagreement = 0。
cache-on/cache-off 都必须一致。

# 31. 绝对禁止的伪性能优化

禁止 chance sampling、stochastic approximation、beam、top-k spawn、降 depth、heuristic early cut、pruning低概率 spawn、float16/quantize weights、改4-bit saturation、只算部分 action、换 checkpoint、换 corpus。

只能让 exact same computation 执行得更快。

# 32. 最终 Performance Unblock PASS gate

Correctness：evaluator differential PASS；Search differential PASS；full pytest PASS；0 skipped；0 xfailed。

Throughput：exact 256-root corpus >=5.0 root decisions/s，且 8192-state projected Search time <=30 minutes。

最终 profile不得还存在一个已知、可直接修复的 Python/NumPy数量级损失却被当成正常成本。

如果最终最大项变成随机 512 MiB 大表内存访问本身，且已经是 differential 通过的 C++ prefetch evaluator，可以作为硬件/算法成本接受。

# 33. Stretch target

10 root decisions/s，不是 PASS 必需。
达到 10/s 后如果继续优化只有小幅收益，停止，不得无限工程化。

# 34. 正式性能测试方法

Search A/B 固定同一 256-state corpus、同一 checkpoint、depth=3、cache配置一致，完整跑完 256 roots。
记录 mean/median/p95/max latency。
禁止只测前10个简单 state。

# 35. artifacts

新增：m3_tuple_backend_benchmark.json、m3_search_performance_unblock.json。
更新：m3_teacher_profile.json、M3_REPORT.md。

必须保留历史 baseline：1.127171768 roots/s、227.117 s/256、86.6% evaluator share。

# 36. M3_REPORT 追加章节

SEARCH PERFORMANCE UNBLOCK 至少包含：starting baseline、evaluator decomposition、leaf batch distribution、duplicate analysis、上游 M6 source hashes、C++ backend、differential、primitive A/B、exact Search A/B、reprofile、second-stage optimization、final roots/s、8192 projection、remaining blocker。

# 37. full pytest

最终必须用 D:\sd-webui-forge-aki-v1.0\python\python.exe 跑完整 pytest。
所有 505 frozen tests、现有 M3 tests、新 backend tests全部保留。
要求 0 failed / 0 skipped / 0 xfailed。

# 38. CI

若新增 cpp/m3_tuple_backend，GitHub Actions 必须在 Ubuntu 同时构建 M2 C++ backend 和 M3 tuple backend，再运行 full pytest。
CI 不得依赖真实 512 MiB checkpoint；backend unit tests用 tiny deterministic synthetic fixture。

# 39. Git / 当前 M3 WIP

允许普通 WIP/implementation commits并 push，但不得创建 M3 tag、不得标记 M3 AUDITED PASS、不得更新主计划为 M3 frozen、不得进入 M4/M5。

M3 candidate 仍必须等回原 M3施工单完成 calibration、8192-state dataset、Student sanity、final report、remote CI。

# 40. PASS 后衔接

一旦 M3 SEARCH PERFORMANCE UNBLOCK PASS，立即停止性能工程。
回 prompts/M3_IMPLEMENTATION_PROMPT.md，从性能 STOP 点之后继续：
value calibration -> 8192-state/64-game dataset -> game-level split -> Student sanity -> final M3 report -> full pytest -> candidate commit/push -> GitHub Actions -> STOP for independent audit。

不要重做 checkpoint copy、loader format gate、Search semantics、256-state corpus、已通过 correctness tests。

# 41. 结果枚举

只允许：
M3 SEARCH PERFORMANCE UNBLOCK PASS
M3 SEARCH PERFORMANCE INCOMPLETE
M3 SEARCH PERFORMANCE FAIL

PASS = correctness全绿 + >=5 roots/s + projected8192<=30min + 无明显可修复数量级热点。
INCOMPLETE = correctness可用，但 exact优化后仍<5 roots/s或需要新的重大路线决定。
FAIL = 无法保持 evaluator/Search semantics 或 frozen regression。

# 42. 禁止越界

本轮禁止 calibration 正式执行、8192-state正式 Teacher dataset、Student training、M4、M5、Teacher promotion、换 checkpoint、Transformer vs MLP、自博弈、Replay、Double-Q、Target、Search Correction、修改 frozen M2 backend。

# 43. 最终报告固定格式

1 RESULT
2 STARTING STATE
3 WIP BASELINE COMMIT
4 OFFICIAL PYTHON / FULL PYTEST BASELINE
5 EVALUATOR DECOMPOSITION
6 LEAF BATCH DISTRIBUTION
7 LEAF DUPLICATION
8 UPSTREAM M6 REFERENCE
9 C++ TUPLE BACKEND
10 DIFFERENTIAL CORRECTNESS
11 PRIMITIVE A/B
12 SEARCH A/B
13 REPROFILE
14 SECOND-STAGE OPTIMIZATION
15 FINAL THROUGHPUT
16 8192 PROJECTION
17 PYTEST
18 CI
19 FROZEN VERIFICATION
20 FILES CHANGED
21 ARTIFACTS
22 FINAL GIT STATE
23 REMAINING BLOCKERS
24 NEXT M3 RESUME POINT

# 44. 开工第一动作

必须先报告：
- current HEAD
- current working tree
- frozen M0/M1/M2 tag resolutions
- fixed checkpoint SHA 7192719323a073ba2b6b19b62cb7d46ef4aa90ecc8c4ae6baf27ad0c51566a84
- baseline 1.127171768 roots/s
- 227.117 s / 256 roots
- 86.6% formal-leaf/evaluator share
- 本轮 exact performance only / no M4/M5

然后严格按：
P0 official-Python full regression -> WIP baseline commit -> P1 evaluator decomposition -> P2 C++ tuple evaluator -> P3 differential -> P4 primitive A/B -> P5 exact Search A/B -> evidence-driven P6 only if needed -> final gate -> STOP / resume normal M3。

不得跳步。