# M1 Report

## A. Final Result

```text
PASS
```

M1 = FINAL AUDITED PASS

> The authoritative final M1 tag is `m1-fastenv-audited-pass`.
>
> `m1-fastenv-pass` (commit `a34e56b`) and `m1-fastenv-final-pass` (commit
> `ce51f15`) are retained **only as historical snapshots**; neither is the
> defect-free final version and neither was moved or deleted.
>
> FINAL AUDITED PASS includes the post-audit int64 aggregate-overflow fix (§S)
> and the final legal/terminal reward-isolation audit (§T).

全部 43 条 M1 Exit Criteria 通过（第 1–35 条为 M1 原有条目，第 36–39 条为
向量化修复新增，第 40–43 条为 post-audit 溢出修复新增，逐条见 §P），
其中性能相关的条目以**同机实测**为依据（§R.3 / §R.5 / §R.6 / §S / §T.6）。

M1 的唯一目标——在冻结的 M0 Reference Environment 之上建立**可严格验证的高速
批量状态生产者**——已达成。movement production hot path 中**不再存在任何与 N
相关的 Python 循环**（§R，字段 `N-dependent Python movement loop: NONE`），
移动原语与 4096-env 端到端吞吐均超出验收阈值。M2 未开始。

---

## B. Files Added

| 文件 | 作用 |
| --- | --- |
| `src/game2048/fast_env.py` | M1 主体：`move_batch`、`legal_mask_batch`、`is_terminal_batch`、`enumerate_spawns_batch`、`spawn_random_batch`、`apply_spawn_batch`、`Fast2048BatchEnv` |
| `tests/_m1_helpers.py` | M1 测试共享工具：固定 seed、随机 board 生成、differential 比较（失败时输出可复现信息） |
| `tests/test_m1_batch_move.py` | 18 个测试：批量移动、M0 回归向量四方向、非法输入拒绝、输入不可变 |
| `tests/test_m1_differential.py` | 25 个测试：ordinary / high-tile / legal / terminal differential、D4 等变性 |
| `tests/test_m1_spawn.py` | 18 个测试：exact enumeration differential、顺序、概率、random spawn 分布、`apply_spawn_batch` |
| `tests/test_m1_batch_env.py` | 24 个测试：`Fast2048BatchEnv` 构造、reset、reset_where、step、illegal action、rollout |
| `tests/test_m1_high_tiles.py` | 37 个测试：高位 tile、int64 边界、overflow 契约 |
| `tests/test_m1_vectorization.py` | 8 个测试：向量化回归契约（见 §R.4） |
| `tests/test_m1_reward_overflow.py` | 41 个测试：int64 reward / score 聚合溢出契约、边界随机 differential、`step` 原子性（见 §S），以及 legal/terminal reward 隔离与 `scores` live view（见 §T） |
| `benchmarks/_utils.py` | benchmark/profiling 共享工具：环境元数据、CPU/RSS 采样、计时与表格渲染（仅 stdlib + NumPy） |
| `benchmarks/benchmark_m1_env.py` | 吞吐 benchmark：规模 sweep、worker scaling、primitive、producer→consumer |
| `benchmarks/profile_m1_env.py` | wall-clock profiler（+ 可选 `cProfile`） |
| `benchmarks/benchmark_ab_vectorization.py` | **同机 A/B**：old per-board 路线 vs new 向量化路线（§R.5） |
| `docs/M1_FAST_ENV_SPEC.md` | M1 权威规格（§3.5 为向量化实现说明） |
| `M1_REPORT.md` | 本文件 |
| `benchmark_results.json` | 向量化修复后 benchmark 原始结果 |
| `baseline_before_vectorization.json` | 向量化修复前同机 baseline 原始结果（A/B 的 OLD 侧） |
| `ab_vectorization.json` | 同机 A/B benchmark 原始结果 |
| `sanity_after_final.json` | 本次 post-audit 溢出修复后的性能 sanity 原始结果 |
| `sanity_after_primitives.json` | 同上，primitive 部分 |

**未修改任何 M0 文件**（见 §C）。`src/game2048/__init__.py` 也未改动：
M1 通过 `from game2048.fast_env import ...` 显式导入，避免触碰冻结文件。

---

## C. M0 Freeze Verification

M1 开始前与结束后各执行一次。

```text
$ git rev-parse HEAD
3f2def1d95f56eff776e671143188947bf64485b

$ git rev-parse m0-reference-pass^{commit}
3f2def1d95f56eff776e671143188947bf64485b

HEAD == m0-reference-pass 指向的 commit
```

冻结文件 diff（M1 结束时再次执行）：

```text
$ git diff --stat HEAD -- \
    src/game2048/reference_env.py src/game2048/symmetry.py \
    src/game2048/__init__.py \
    tests/test_m0_moves.py tests/test_m0_spawn.py tests/test_m0_legal_terminal.py \
    tests/test_m0_d4.py tests/test_m0_properties.py tests/test_m0_high_tiles.py \
    tests/test_m0_env_api.py docs/M0_ENVIRONMENT_SPEC.md M0_REPORT.md
(无输出)
```

```text
M0 文件是否被修改: 否
M0 tag:           m0-reference-pass (annotated) -> 3f2def1
M1 开始前 pytest: 261 passed
M1 结束后 pytest: 432 passed（261 M0 + 161 M1 + 10 final-audit，全部通过）
```

`git status --short` 只显示新增文件，没有任何 M0 文件被修改或删除。

**结论：M0 永久冻结完好，未被 M1 触碰。**

---

## D. Correctness

```text
python -m pytest
432 passed
```

| 项目 | 数量 |
| --- | --- |
| pytest 总数 | 432 |
| M0 tests | 261（全部继续通过，未被排除） |
| M1 tests | 171（122 原有 + 8 向量化回归 + 31 溢出/原子性回归 + 10 final-audit 回归） |
| pytest config 排除 M0 | 无 |
| skip / xfail | **0** |
| tolerance 放宽 | **无**（probability 浮点比较沿用原 `abs tol <= 1e-12`） |

Differential sample count（`seed = 20260918`）：

| 类别 | boards | actions | 比较次数 | mismatch |
| --- | --- | --- | --- | --- |
| ordinary（exponent 0..17） | 10,000 | 4 | 40,000 | **0** |
| high-tile（exponent 0..22） | 2,000 | 4 | 8,000 | **0** |
| M0 回归向量（四方向） | 手工 | 4 | 全部 | **0** |
| legal mask（exponent 0..17） | 10,000 | 4 | 40,000 | **0** |
| terminal（exponent 0..17） | 10,000 | 1 | 10,000 | **0** |
| legal + terminal（exponent 0..22） | 2,000 | 4 | 10,000 | **0** |
| exact spawn enumeration | 2,000 | — | 全部 outcome | **0** |
| D4 等变性 | 2,000 × 8 transforms | 4 | 64,000 | **0** |
| **单次 8,192-board batch 抽样复核**（§R.4） | 8,192 | random 4 | **512 sampled** | **0** |

```text
mismatch count: 0
```

除 probability 浮点比较（`abs tol <= 1e-12`）外，board / reward / moved /
legal / terminal 一律 **exact equality**，任何 mismatch 都未放宽 tolerance、
未跳过、未修改 ReferenceEnv。

**向量化修复后的回归验证**：上表全部 differential 是在
「删除 per-board 循环、改用跨 board 向量化内核」之后重新跑出的结果，
不是复用修复前的旧数据。

**post-audit 溢出修复后的回归验证**：全部 432 条测试是在 int64 聚合溢出修复
（§S）与 legal/terminal reward 隔离修复（§T）之后重新跑出的；原 391 条**一条未删、
一条未跳过**，另外新增 31 条溢出/原子性回归测试与 10 条 final-audit 回归测试。

---

## E. Differential Coverage

| 覆盖项 | 测试位置 | 内容 |
| --- | --- | --- |
| ordinary boards | `test_m1_differential.py::test_ordinary_differential_10k_boards_all_actions` | 10,000 × 4，afterstate/reward/moved |
| high-tile boards | `test_m1_high_tiles.py::test_high_tile_differential_against_m0` | 2,000 × 4，exponent 0..22 |
| legal | `test_m1_differential.py::test_legal_mask_differential_10k_boards`、`::test_legal_mask_matches_moved_by_definition` | 10,000，且验证 `legal[a] ⇔ moved` |
| terminal | `test_m1_differential.py::test_terminal_differential_10k_boards`、`::test_terminal_is_no_legal_action_and_not_full_board` | 10,000；满盘可合并**不是** terminal；空盘按定义为 terminal |
| spawn enumeration | `test_m1_spawn.py::test_enumeration_matches_m0_on_random_boards` | 2,000；outcome 数 / state / index / exponent / probability / 顺序，CSR 区间连续 |
| D4 | `test_m1_differential.py::test_d4_equivariance_random_boards`（8 个参数化用例）、`::test_d4_legal_mask_equivariance`、`::test_d4_inverse_round_trip_on_fast_afterstates` | `T(FastMove(s,a)) == FastMove(T(s),T(a))`，2,000 × 8 × 4 |
| illegal action | `test_m1_batch_move.py`、`test_m1_batch_env.py::test_illegal_action_semantics`、`::test_illegal_action_does_not_consume_any_randomness`、`::test_partial_illegal_step_keeps_streams_in_step` | board 不变、reward=0、score 不变、legal=False、`spawn_index=-1`、`spawn_exponent=0`、不 spawn、全非法时 RNG 逐 bit 不变 |
| reset | `test_m1_batch_env.py::test_reset_is_reproducible_for_a_fixed_seed`、`::test_reset_gives_exactly_two_spawned_tiles`、`::test_reset_zeroes_the_scores` | 可复现、恰好 2 个非空 tile、两者 exponent ∈ {1,2} |
| reset_where | `test_m1_batch_env.py::test_reset_where_only_touches_the_selected_environments`、`::test_reset_where_with_an_empty_mask_is_a_no_op`、`::test_reset_where_validates_its_mask` | 只改 mask=True；其他 env 的 board/score 逐 bit 不变 |
| 输入不可变 | `test_m1_batch_move.py::test_move_batch_does_not_modify_its_inputs`、`test_m1_spawn.py::test_enumeration_does_not_modify_its_input`、`::test_random_spawn_stays_within_empty_cells`、`::test_apply_spawn_places_tiles_deterministically`、`test_m1_high_tiles.py::test_high_exponents_spawn_and_enumerate_correctly` | 五个 batch 函数均不修改 caller 输入 |
| 输入校验 | `test_m1_batch_move.py`、`test_m1_spawn.py` | float / bool / string action、形状错、长度错、越界、dtype 错全部显式拒绝 |
| **向量化契约** | `test_m1_vectorization.py`（8 条） | 见 §R.4：production path 不得调用 M0 per-board 内核；8,192-board 单次 batch 抽样复核；Python 行数不随 N 增长；静态源码检查 |

---

## F. Performance Environment

本机已安装 `psutil` 为 `False`，因此 CPU / RAM 指标由 `benchmarks/_utils.py`
通过 `ctypes` 直接调用 Windows API 采集（不引入新依赖）。

```text
OS            : Windows 11 (10.0.26100)
Python        : 3.12.10 (CPython)
NumPy         : 2.5.3
CPU           : AMD Ryzen 7 H 260 w/ Radeon 780M Graphics
physical cores: 16
logical cores : 16
RAM           : 15.3 GiB
Git commit    : 3f2def1d95f56eff776e671143188947bf64485b
benchmark seed: 20260918
```

说明：
* `psutil` 未安装，**也没有为 M1 安装它**；RSS 通过
  `GetProcessMemoryInfo`（working set / peak）读取。
* 受限沙箱下进程内存查询返回 0，报告中对应项记为 `unknown`，
  但 `num_envs` sweep 本身全部正常完成（无 OOM）。
* 未安装 PyTorch，因此 benchmark **没有**包含 `torch.from_numpy` 数据搬运预演
  （规格 §45 要求不为 M1 单独安装 PyTorch；留给 M2）。

---

## G. Single-Worker Scalability

命令（NEW）：`python benchmarks/benchmark_m1_env.py --json benchmark_results.json`

命令（OLD）：`python benchmarks/benchmark_m1_env.py --json baseline_before_vectorization.json`
（向量化修复**之前**在同一台机器上跑的同一条命令、同一个 harness）

动作策略：uniform random legal action；terminal env 显式 `reset_where` 后继续。
两轮使用同一个固定 seed `20260918`、同一个 benchmark 逻辑与同一组环境规模。

**口径说明（重要）**：两轮的 step count 并不相同 —— OLD 侧各档一律
`16 steps`（当时的 `--steps` 默认值），NEW 侧由 `TARGET_TRANSITIONS` 推出
`64 / 64 / 48 / 24 / 12 steps`。以 4096 档为例：

```text
OLD:  4096 env × 16 steps  =  65,536 transitions
NEW:  4096 env × 48 steps  = 196,608 transitions
```

因此两列连 `total transitions` 都不相等，本表**不是**「同 workload」的比较。
两侧使用相同环境规模、相同 seed 和相同 benchmark 逻辑，吞吐又按每 transition
归一，所以该比值仍是有意义的性能证据，用于证明**长期 steady-state 吞吐**提升；
但它**不是严格逐指令的 A/B**。严格同进程、同输入、交替测量的 primitive A/B
由 `benchmarks/benchmark_ab_vectorization.py` 提供（§R.5）。

| num_envs | OLD iters | NEW iters | OLD total | NEW total | OLD transitions/s | **NEW transitions/s** | speedup |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 256 | 16 | 64 | 4,096 | 16,384 | 24,195 | **64,799** | **2.68×** |
| 1,024 | 16 | 64 | 16,384 | 65,536 | 26,319 | **106,771** | **4.06×** |
| 4,096 | 16 | 48 | 65,536 | 196,608 | 14,826 | **73,571** | **4.96×** |
| 8,192 | 16 | 24 | 131,072 | 196,608 | 14,586 | **70,124** | **4.81×** |
| 16,384 | 16 | 12 | 262,144 | 196,608 | 14,671 | **72,596** | **4.95×** |

¹ 沙箱禁止进程内存查询，`GetProcessMemoryInfo` 返回 0，两轮的 peak RAM 均记为
`unknown`。16384 档**正常完成**，因此没有触发 `SKIPPED_RESOURCE_LIMIT`。

单 worker 每次 `step()` 的固定开销 = 1 次 move（动作）+ 8 次 batch move
（legal mask 4 次 + terminal 4 次）→ 见 §M。向量化后这 9 次 move 各自都是
跨越整批 board 的数组运算，不再是 per-board Python 循环。

每档均 full sweep 完成：**256 / 1024 / 4096 / 8192 / 16384 全部 OK**。

---

## H. Worker Scaling

每个 worker 是**独立进程**，各自持有自己的 `Fast2048BatchEnv`。

`scaling_efficiency = Tk / (k * T1)`（按规格 §43，公式未改）。

| workers | envs/worker | OLD transitions/s | **NEW transitions/s** | NEW scaling efficiency | speedup |
| --- | --- | --- | --- | --- | --- |
| 1 | 4,096 | 7,414 | **29,717** | 1.000（baseline） | **4.01×** |
| 2 | 4,096 | 12,852 | **57,053** | **0.960** | **4.44×** |
| 4 | 4,096 | 23,745 | **100,324** | **0.844** | **4.23×** |

（OLD 侧用的是 2048 envs/worker × 16 steps；NEW 侧是 4096 envs/worker × 48 steps。
两者的 scaling efficiency 是各自定义内自洽的，speedup 一列只作数量级参考，
严谨的同机同workload A/B 见 §R.5。）

机器逻辑核数 16，因此按规格测试 1 / 2 / 4 worker。

实现说明：worker 进程通过 `subprocess` 启动并使用**文件**汇报结果。
在 Windows 上 `multiprocessing` 的 `spawn` 启动方式需要一个匿名 bootstrap 管道，
本机沙箱禁止创建管道（`WinError 5`），改用 `subprocess` + 结果文件后
1 / 2 / 4 worker 均正常完成。每个 worker 内部仍是完整的 `Fast2048BatchEnv`。

外在并行效率随 worker 数下降（4 worker 0.844），但**单 worker 内部仍几乎完全
单核**（CPU util 0.99–1.02×），因此瓶颈不在同步，而在单核 Python/NumPy 执行速度。
与 OLD 的 0.829 相比，4-worker 效率略有改善（0.844），因为单进程内部留给
Python 解释器的时间占比下降、留给 NumPy C 循环的时间占比上升。

---

## I. Primitive Throughput

`num_envs = 4096`，固定 seed 20260918，median of 3 repeats（warmup 2）。

| primitive | OLD median | OLD rate | **NEW median** | **NEW rate** | speedup |
| --- | --- | --- | --- | --- | --- |
| `move_batch` | 26.94 ms | **152.0 k boards/s** | **5.20 ms** | **787.4 k boards/s** | **5.18×** |
| `legal_mask_batch` | 135.88 ms | **30.1 k boards/s** | **17.94 ms** | **228.3 k boards/s** | **7.57×** |
| `is_terminal_batch` | 130.10 ms | **31.5 k boards/s** | **22.74 ms** | **180.2 k boards/s** | **5.72×** |
| `spawn_random_batch` | 1.82 ms | 2.25 M boards/s | 1.93 ms | 2.12 M boards/s | 0.94× |
| `enumerate_spawns_batch` n=2 | 1.93 ms | 1.93 M parents/s（7.72 M outcomes/s） | 2.00 ms | 2.05 M parents/s（8.21 M outcomes/s） | 1.06× |
| `enumerate_spawns_batch` n=4 | 3.74 ms | 1.25 M parents/s（10.03 M outcomes/s） | 2.88 ms | 1.42 M parents/s（11.37 M outcomes/s） | 1.13× |
| `enumerate_spawns_batch` n=8 | 8.26 ms | 495 k parents/s（7.92 M outcomes/s） | 5.60 ms | 732 k parents/s（11.70 M outcomes/s） | 1.48× |
| `enumerate_spawns_batch` n=12 | 8.25 ms | 527 k parents/s（12.66 M outcomes/s） | 11.40 ms | 359 k parents/s（8.63 M outcomes/s） | 0.68× |

读法说明：

* **`spawn_random_batch` 与 `enumerate_spawns_batch` 的代码在本轮修复中一行未改**，
  它们几档的 0.68×–1.48× 波动是同一台机器上重复测量的噪声与缓存效应
  （n=8/n=12 的内存访问模式随 empty-cell 数变化明显），不构成性能变化。
* 真正被改写的是三个 movement primitive，它们的提升是 **5.2×–7.6×**，并且
  是这次修复的直接结果。
* `move_batch` 的 787.4 k boards/s 是 benchmark harness 内的正式读数；
  同机 A/B（§R.5，独立进程、独立输入构造）读到的是 1.11 M boards/s。
  两者都远高于 OLD 的 152.0 k，也都远高于 2.0× 的验收阈值。

Exact spawn 的输出规模随 empty count 线性增长（`2n` outcomes/parent），
因此 `parents/s` 随 empty count 下降、而 `outcomes/s` 基本稳定在
约 **8–12 M outcomes/s**。

**OLD 时移动是唯一慢 primitive**（比 spawn 慢 ~15–20 倍）；
**NEW 时移动与 spawn 已在同一数量级**，movement 不再是数量级上的异常项。

---

## J. Producer → Consumer Throughput

Harness：`FastEnv → contiguous board batch → synthetic NumPy consumer → actions → FastEnv`。
consumer **整 batch 一次处理**（无 per-board 循环），只做批量 reduction +
legal-mask 选动作，不模拟棋力。

`num_envs = 4096`。OLD 侧 16 steps / 65,536 transitions；NEW 侧 48 steps /
196,608 transitions（同一 harness 的默认 workload）。

| metric | OLD | **NEW** |
| --- | --- | --- |
| throughput | 14.39 k transitions/s | **81.75 k transitions/s（5.68×）** |
| CPU utilization | 0.99× | 0.99× |
| batch assembly（`copy_boards`） | 0.00% | **0.03%** |
| consumer（含 legal mask） | 43.6% | **41.7%** |
| environment step（move + spawn + terminal） | 56.4% | **58.3%** |

```text
batch assembly      0.03%
consumer           41.7%
environment step   58.3%
overall            81.75 k transitions/s   (0.99 × single core)
```

结论：batch 数据路径本身（contiguous copy + 整体消费）**不是**成本；
成本仍然全部落在 move/terminal 的数值计算上，但同样是这部分成本，
绝对吞吐提升了 **5.68×**。

---

## K. Wall-Clock Profile

命令（NEW）：`python benchmarks/profile_m1_env.py --num-envs 4096 --iterations 48 --cprofile`

| stage | OLD ms（8192 env × 16 steps） | OLD share | **NEW ms（4096 env × 48 steps）** | **NEW share** |
| --- | --- | --- | --- | --- |
| Environment stepping (move + spawn + score) | 4,444.159 | **60.28%** | **1,152.930** | **56.86%** |
| Legal/action preparation (mask + terminal) | 2,855.131 | **38.73%** | **752.249** | **37.10%** |
| Spawn (random) | 42.640 | 0.58% | 56.516 | 2.79% |
| Exact spawn enumeration | 29.944 | 0.41% | 64.904 | 3.20% |
| Batch assembly (contiguous copy) | 0.119 | 0.00% | 0.185 | 0.01% |
| Memory copy | 0.375 | 0.01% | 0.570 | 0.03% |
| Synchronization / worker overhead | 0.000 | 0.00% | 0.000 | 0.00% |
| Other | 0.185 | 0.00% | 0.443 | 0.02% |
| **TOTAL** | **7,372.553** | **100.00%** | **2,027.799** | **100.00%** |

```text
NEW: Environment stepping   56.86%
     Legal/action           37.10%
     Spawn                   2.79%
     Exact spawn             3.20%
     Batch assembly          0.01%
     Memory copy             0.03%
     Synchronization         0.00%
     Other                   0.02%
     -----------------------------
     sum                   100.00%
```

（NEW：4096 env，48 steps，196,608 transitions，平均 **96.96 k transitions/s**。）
百分比之和为 100.00%，无未归类时间。

**profile 形状本身没有改变**：movement（Environment stepping + Legal/action）
仍然占据 **93.96%** 的实测 wall clock（OLD 是 99.01%）。变的是**同一份工作量
花了多少绝对时间**——每 transition 的 wall clock 从 OLD 的 56.25 µs
降到 NEW 的 10.31 µs。

`cProfile` 辅助细节（num_envs 4096，5 steps）：

```text
   ncalls  tottime  cumtime  function
        1    0.002    0.095  profile_m1_env.py:98(profile_iteration)
       50    0.014    0.076  fast_env.py:419(_move_groups)
       10    0.000    0.058  fast_env.py:565(legal_mask_batch)
       10    0.001    0.058  fast_env.py:539(_move_all_actions_batch)
      160    0.020    0.043  fast_env.py:316(_pack_left_rows)
        5    0.000    0.043  fast_env.py:1041(step)
       80    0.011    0.031  fast_env.py:343(_merge_left_rows)
        5    0.000    0.030  fast_env.py:1090(_refresh_terminated)
        5    0.000    0.030  fast_env.py:581(is_terminal_batch)
      579    0.013    0.013  {method 'nonzero' of 'numpy.ndarray' objects}
      182    0.011    0.011  {method 'cumsum' of 'numpy.ndarray' objects}
```

对照 OLD 的 cProfile：

```text
   409600    0.105    0.187  fast_env.py:304(_merge_line_cached)
   409600    0.082    0.082  {method 'get' of 'dict' objects}
   409600    0.077    0.077  fast_env.py:450(_shift_line)
   409600    0.069    0.069  fast_env.py:346(_line_changed)
```

**OLD 的热区是 409,600 次 per-line Python 函数调用**（`_merge_line_cached` /
`_shift_line` / `_line_changed`，各 409,600 次 = 4096 boards × 4 lines × 25 次
move/batch），即 Python 解释器开销。

**NEW 的热区是 `_pack_left_rows` / `_merge_left_rows` 内部的 NumPy C 调用**：
`ndarray.nonzero` 579 次、`ndarray.cumsum` 182 次——次数由**固定的动作分组数**
决定，与 batch 里的 board 数无关。per-line 的 Python 函数（`_merge_line_cached` /
`_shift_line` / `_line_changed` / `_LINE_CACHE`）已从代码中**彻底删除**。

---

## L. Saturation Point

**饱和点出现在 1024 ～ 4096 env 之间（与 OLD 相同），但饱和平台的绝对高度
提升了约 4.9×。**

```text
NEW:

num_envs    transitions/s
     256        64,799   ┐
   1,024       106,771   ┘  ← 峰值区间
   4,096        73,571   ┐
   8,192        70,124   │  ← 饱和平台（4,096 → 16,384 波动 < 4.9%）
  16,384        72,596   ┘

OLD（同机 baseline）:

num_envs    transitions/s
     256        24,195   ┐
   1,024        26,319   ┘
   4,096        14,826   ┐
   8,192        14,586   │  ← 饱和平台
  16,384        14,671   ┘
```

解释（NEW）：

* 256 → 1024 吞吐上升 1.65×：小批量时每次 `step()` 的固定 Python/NumPy 调用
  开销仍未摊销完；
* 1024 → 4096 吞吐**下降 31%**，此后 4096 / 8192 / 16384 吞吐基本持平
  （73.57 / 70.12 / 72.60 k transitions/s，波动 < 4.9%）；
* 即从 4096 env 起进入**饱和平台**：增加环境数不再提升吞吐，只线性增加单步
  wall time。

原因：向量化之后单次 `move_batch` 的成本已经主要由**内存带宽**
（gather / scatter / pack 的若干次整批读写）决定，而 4096 env 的
`(4096, 16) uint8` board 批只有 64 KiB，已经超出 L1、落进 L2/L3 的
规模区间；继续加大 N 只是把同样的带宽压力摊到更大数组上，因此吞吐进入平台。

OLD 的饱和原因则完全不同（per-board Python 循环使单步成本与 N 严格线性），
二者的**平台位置**巧合地都落在 4096 env 附近，但**平台高度差了 4.9×**。

**D4 核 worker scaling 的下降（0.844）属于内存带宽/超线程争用，不是本实现的
可扩展性缺陷**——单 worker 内部 CPU util 只有 0.99–1.02×。

---

## M. Bottleneck

**当前最大瓶颈：movement 内核的 NumPy 调用开销与内存带宽，不再是 Python 循环。**

`measured evidence`：

1. **profiler**：`Environment stepping` 56.86% + `Legal/action` 37.10% = **93.96%**
   的实测 wall clock 落在 movement 上；`Spawn` 2.79%、`Exact spawn` 3.20%、
   `Batch assembly` + `Memory copy` 合计 0.04%。
2. **cProfile**：热区是 `_pack_left_rows` 内部的 `ndarray.nonzero`（579 次）与
   `ndarray.cumsum`（182 次），以及 `_merge_left_rows` 里的布尔掩码运算。
   **这些调用次数只取决于固定的动作分组数（4）与三个 merge 边界，与 batch 内
   board 数无关**——这正是"跨 board 向量化"的特征。
   OLD 版本对应的热区是 409,600 次 per-line Python 函数调用，已全部删除。
3. **primitive 对比**：`move_batch` 787.4 k boards/s（OLD 152.0 k，5.18×）；
   `spawn_random_batch` 2.12 M boards/s。movement 与 spawn 已落在同一数量级，
   不再是数量级上的异常项。
4. **每一步 9 次 move**：`step()` 自身 1 次 + `legal_mask_batch` 4 次
   （consumer 侧）+ `is_terminal_batch` 4 次 ⇒ movement 成本仍被放大 9 倍。
   因此**进一步优化的正确方向是减少重复 move**，而不是继续微调内核。
5. **worker scaling**：1 worker CPU util 0.99–1.02×，2 worker 效率 0.960，
   4 worker 0.844 —— 加进程仍可提升总吞吐，但单进程依然被钉在 1 核。

**NumPy FastEnv saturation point**：4096 env（§L）。

**下一步的可选优化方向（本轮未做，不属于 M1 验收要求）**：

* `step()` 内部已经把 legal mask 流水化（复用上一步结果），但 consumer 侧的
  `legal_mask_batch` 与 `is_terminal_batch` 仍各自完整算 4 次 move。若把
  「一次 `_move_all_actions_batch` 同时产出 moved 与 reward，并让二者共享」
  进一步推广到 consumer 侧，理论上可再省下若干次重复 move。
* `_pack_left_rows` 的 rank-based gather 每次分配一个 `(R, 4)` intp 中间数组；
  预分配 scratch buffer 可以减少分配次数（但会牺牲纯函数性，需权衡）。

**已完成的向量化**：

* movement 的 production hot path 中**不存在任何与 N 相关的 Python 循环**
  （§R.2 逐条列出允许的固定常量循环）；
* 全部 spawn 路径（random / exact enumeration / apply）、`legal_mask_batch` 的
  分组、`reset` / `reset_where`、`BatchStepResult` 组装都是整 batch 一次完成。

**仍然存在的 per-board 循环**：**NONE**。

---

## N. C++ Decision

```text
C++ migration recommended: DEFER_TO_M2
```

原因（向量化修复后重新评估，结论未变，但依据更充分）：

1. M1 的单核 Python 向量化路径已经达到 **787 k boards/s**（`move_batch`）与
   **~82 k transitions/s**（producer→consumer 端到端，单核），比修复前提升
   4.8–5.7×。剩余瓶颈是 NumPy 调用开销与内存带宽，**不再是解释器开销**。
2. **M1 仍然没有 M2 的网络实际吞吐**。规格 §50 明确：若当前只能证明"环境在某处
   饱和"，而还没有 M2 网络实际吞吐，应优先 `DEFER_TO_M2`，不要提前过度优化。
3. 判断"CPU 能不能喂饱 RTX 5060"必须到 M2 网络完成后做端到端 benchmark。
   在不知道 consumer 每步要吃掉多少 board/s 之前，无法判断
   82 k transitions/s 是"不够"还是"足够"。
4. 迁移 C++ 是一个大的工程投入（构建系统、ABI、跨平台、differential 重测），
   并且 M1 阶段已明确禁止 C++/pybind11/Cython/Numba/Triton/CUDA/Rust。
   在需求（目标吞吐）尚未量化前投入，属于过早优化。

```text
Current bottleneck:
  movement 内核的 NumPy 调用开销 + 内存带宽
  （不再有任何 N-dependent Python 循环）

Measured evidence:
  profiler: Environment stepping 56.86% + Legal/action 37.10% = 93.96%
  cProfile: 热区为 _pack_left_rows 的 ndarray.nonzero(579) / cumsum(182)
            —— 调用次数只取决于固定动作分组数，与 N 无关
  primitive: move 787.4 k boards/s（OLD 152.0 k，5.18×）
             vs 纯向量化 spawn 2.12 M boards/s（同一数量级）
  step() 每步放大 9 次 move  ⇒ 下一步优化方向是减少重复 move

NumPy FastEnv saturation point:
  4096 env（1024→4096 吞吐下降 31%，此后 4096/8192/16384 持平于 ~72 k/s）

Worker scaling:
  1 worker 29,717 t/s (eff 1.000)
  2 worker 57,053 t/s (eff 0.960)
  4 worker 100,324 t/s (eff 0.844)
  单 worker CPU util 0.99-1.02x，加进程可线性扩展总吞吐

Producer/consumer throughput:
  81.75 k transitions/s（consumer 41.7% / env step 58.3% / batch assembly 0.03%）

C++ migration recommended: DEFER_TO_M2
```

若 M2 端到端 benchmark 显示 CPU 端确实喂不饱 GPU，建议的迁移顺序是：
`_pack_left_rows` / `_merge_left_rows` 这两个数组内核 → 再评估
`_move_all_actions_batch`（把 4 次 move 合成一次调用）。
本 M1 **不自行启动 C++ 重写**，等待明确指令。

---

## O. Known Issues

1. **`_pack_left_rows` 每次调用分配一个 `(R, 4)` intp 秩数组**：这是为了保持
   函数纯粹（不修改输入、无隐藏状态）。它已经不再是瓶颈，但在 M2 若需要进一步
   压榨单核吞吐，可以用预分配 scratch buffer 换掉。**这不影响正确性。**
2. **profile 形状与修复前相同（movement 占 ~94%）**：这是**预期**结果，不是缺陷。
   修复改变的是同一份工作的绝对耗时（每 transition 从 56.25 µs 降到 10.31 µs），
   不是各阶段的相对占比。M1 的验收判据是**绝对吞吐**（§R.6）。
3. **RSS / peak RAM 报告为 `unknown`**：本机沙箱禁止进程内存查询
   （`GetProcessMemoryInfo` 返回 0），且 `psutil` 未安装（规格允许安装，
   但本机连内存查询都被拦截，安装也无济于事）。所有 num_envs 档位均正常完成，
   因此没有触发资源上限。
4. **`git commit` 在 benchmark 输出中为 `unknown`**：沙箱需要
   `safe.directory` 才能运行 git；报告 §C / §F 中已用 `git rev-parse` 单独确认
   commit 为 `3f2def1d95f56eff776e671143188947bf64485b`。
5. **Worker scaling 用 `subprocess` + 结果文件而非 `multiprocessing`**：
   Windows 的 `spawn` 需要匿名管道，本机沙箱禁止（`WinError 5`）。
   每个 worker 仍是完整独立的 `Fast2048BatchEnv`，语义未变。这也意味着 §H 的
   OLD/NEW 两轮**不是**完全相同的 `envs_per_worker × steps`（OLD 2048×16，
   NEW 4096×48），因此 §H 的 speedup 一列只作数量级参考；严格同机同 workload
   的对照见 §R.5。
6. **未做 `torch.from_numpy` 数据搬运预演**：本机未安装 PyTorch，
   规格 §45 明确要求不为 M1 单独安装，留待 M2。
7. **M1 与 M0 有一处有意差异**：exponent ≥ 62 的 merge，M1 抛 `OverflowError`
   而 M0 返回 Python 大整数（M0 无上限）。已由 `test_m1_high_tiles.py` 双向钉住，
   且永远不是静默错误数字。详见 `docs/M1_FAST_ENV_SPEC.md` §4.1。
   说明：`MAX_SAFE_MERGE_EXPONENT = 62` 的含义是「**指数 e ≥ 62 的 pair 不允许
   merge**」，因为 `2 ** (e+1) >= 2**63` 超出 int64；因此最大可精确表示的 merge
   是 `61 + 61 → 62`，reward `2**62`。
8. `prompts/` 目录不在 M1 新增文件清单内（M0 阶段已存在，本阶段未改动）。
9. **§I 中 `spawn_random_batch` / `enumerate_spawns_batch` 的 0.68×–1.48×
   波动是测量噪声**：这两个 primitive 的代码在本轮修复中一行未改，
   其绝对水平与 OLD 一致（2.1–2.3 M boards/s、8–12 M outcomes/s）。

---

## P. M1 Exit Checklist

| # | 条目 | 结果 | 证据 |
| --- | --- | --- | --- |
| 1 | M0 原测试全部继续通过 | **PASS** | 261 passed（单独运行，未排除） |
| 2 | M0 frozen files 未被修改 | **PASS** | `git diff --stat HEAD --` 对全部冻结文件无输出 |
| 3 | FastEnv vs M0 ordinary differential 0 mismatch | **PASS** | 10,000 × 4，mismatch 0 |
| 4 | high-tile differential 0 mismatch | **PASS** | 2,000 × 4，exponent 0..22，mismatch 0 |
| 5 | legal mask differential 0 mismatch | **PASS** | 10,000，mismatch 0 |
| 6 | terminal differential 0 mismatch | **PASS** | 10,000，mismatch 0 |
| 7 | exact spawn enumeration differential 0 mismatch | **PASS** | 2,000 parents，全部 outcome 字段 0 mismatch |
| 8 | D4 equivariance 0 mismatch | **PASS** | 2,000 × 8 × 4，mismatch 0 |
| 9 | illegal action 语义正确 | **PASS** | board/reward/score/legal/sentinel 全部断言 |
| 10 | illegal action 不消耗 FastEnv spawn RNG | **PASS** | 全非法 step 前后 generator state 逐 bit 相同 |
| 11 | reset 正确 | **PASS** | 可复现；恰好 2 非空 tile；exponent ∈ {1,2} |
| 12 | reset_where 正确 | **PASS** | 只改 mask=True；其余 board/score 逐 bit 不变 |
| 13 | BatchEnv 不使用 N 个 ReferenceEnv object | **PASS** | 持久核心游戏状态为 `_boards`/`_scores`/`_rng` 三块缓冲，另加整批共享的 scratch/cache（`_empty_prefix`/`_rows_buffer`/`_terminated`，普通数组、非 per-env object）；无 per-game object |
| 14 | hot path 不逐 board 调 ReferenceEnv | **PASS** | 无任何 `Reference2048Env` / `move_without_spawn` 调用；规则本体自行实现并 differential 验证；另由 §R.4 的 monkeypatch 测试行为性钉死 |
| 15 | board buffer `(N,16) uint8 contiguous` | **PASS** | `test_m1_batch_env.py::test_env_boards_are_always_contiguous_uint8` |
| 16 | action batch 接口完成 | **PASS** | `move_batch(boards, actions)` |
| 17 | legal mask batch 完成 | **PASS** | `legal_mask_batch` |
| 18 | terminal batch 完成 | **PASS** | `is_terminal_batch` |
| 19 | random spawn batch 完成 | **PASS** | `spawn_random_batch`（+ 分布/support 测试） |
| 20 | exact spawn enumeration batch 完成 | **PASS** | `enumerate_spawns_batch`（CSR flat layout） |
| 21 | producer → consumer harness 完成 | **PASS** | `benchmarks/benchmark_m1_env.py::run_producer_consumer`，整 batch consumer |
| 22 | 256 env benchmark 完成 | **PASS** | 64,799 t/s（2.68× OLD） |
| 23 | 1024 env benchmark 完成 | **PASS** | 106,771 t/s（4.06× OLD） |
| 24 | 4096 env benchmark 完成 | **PASS** | 73,571 t/s（4.96× OLD） |
| 25 | 8192 env benchmark 完成 | **PASS** | 70,124 t/s（4.81× OLD） |
| 26 | 16384 env 已测试或记录 resource limit | **PASS** | 已测试并完成：72,596 t/s（未触发 SKIPPED_RESOURCE_LIMIT） |
| 27 | worker scaling 已测试 | **PASS** | 1 / 2 / 4 worker，efficiency 1.000 / 0.960 / 0.844 |
| 28 | wall-clock profile 已输出 | **PASS** | `profile_m1_env.py`，百分比和 100.00% |
| 29 | throughput saturation region 已识别 | **PASS** | 4096 env（§L） |
| 30 | 当前主要 bottleneck 已识别 | **PASS** | movement 内核的 NumPy 调用开销 + 内存带宽（§M） |
| 31 | C++ migration decision 已写明 | **PASS** | `DEFER_TO_M2`（§N） |
| 32 | `docs/M1_FAST_ENV_SPEC.md` 已生成 | **PASS** | 见文件；§3.5 已更新为向量化实现说明 |
| 33 | `M1_REPORT.md` 已生成 | **PASS** | 本文件 |
| 34 | `python -m pytest` 全绿 | **PASS** | 432 passed（261 M0 + 161 M1 + 10，0 skip / 0 xfail） |
| 35 | M2 尚未开始 | **PASS** | 无 network/trainer/replay/self-play/teacher/expectimax/search 代码；未加载 tuple checkpoint |
| **36** | **movement hot path 无 N-dependent Python 循环** | **PASS** | §R.2 / §R.3 字段 `N-dependent Python movement loop: NONE`；§R.4 的 4 类回归测试钉死 |
| **37** | **`move_batch` 相对修复前提速 ≥ 2.0×** | **PASS** | 152.0 k → 787.4 k boards/s = **5.18×**（同进程 A/B 对照 10.97×），§R.6 |
| **38** | **4096-env 端到端吞吐相对修复前提速 ≥ 1.5×** | **PASS** | 14,826 → 73,571 transitions/s = **4.96×**，§R.6 |
| **39** | **修复未破坏既有 M1 正确性与 API** | **PASS** | 原 391 条全部保留并通过（新增 8 条向量化回归后为 391）；公开 API 名称/签名/返回类型/语义未变 |
| **40** | **row / board / score 三层 int64 聚合受检，不依赖 `np.errstate`** | **PASS** | §S.2；`[61,61,61,61]` 与两个 `[61,61]` 行均抛 `OverflowError`（修复前返回 `-9223372036854775808`） |
| **41** | **`step()` overflow 具备原子性** | **PASS** | §S.3；异常后 board / score / RNG 逐 bit 不变，`test_m1_reward_overflow.py` 断言 |
| **42** | **安全检查未造成 >10% 性能回退** | **PASS** | §S.5；同进程交替 A/B 中位 **+1.7%**（min +4.3% / mean +2.6%） |
| **43** | **文档中的错误 int64 边界（exponent 53）与「三块缓冲」描述已纠正** | **PASS** | §S；`fast_env.py` 模块说明 / `move_batch` docstring / `docs/M1_FAST_ENV_SPEC.md` §4.1 / §11.1 均已改正 |

```text
43 / 43 PASS  →  M1 = FINAL PASS
```

第 36–39 条是「M1 性能修复与最终验收」新增的验收项，
第 40–43 条是本次「M1 Post-Audit Correctness Fix」新增的验收项；
第 1–35 条为 M1 原有 Exit Criteria，全部在**最后一次修复之后**重新跑过并保持通过。

---

## R. Vectorization Repair

本节记录「消灭 movement hot path 的 N-board Python 主循环」这一轮修复。
**它不是重新执行 M1**：M1 的 API、语义、测试集合与 M0 冻结基线全部保持不变，
本节只是把 movement 内核从 per-board Python 循环换成真正的跨 board 向量化，
并把修复前后的同机数据并列记录。

### R.1 Previous implementation

修复前的 movement 路径由两层组成：

```text
_move_batch(boards, actions)
  └─ for each board in the batch:            ← N-dependent Python 循环
       取该 board 的 4 条 line 指数
       用 _merge_line_cached(row_tuple) 查表 + 计算该行 merge
       用 _shift_line / _line_changed 写回与比较
```

具体地：

* `_LINE_CACHE`：dict，以「平移后的行四元组」为 key 缓存单行 merge 结果；
* `_merge_line_cached(row)`：dict 查找 + 未命中时执行 M0 `_merge_line`；
* `_shift_line(row)`：逐格平移（pack-left 的标量实现，最多 3 趟）；
* `_line_changed(row)`：逐格比较以决定 `moved`。

一次 4096-board batch 会产生 **409,600 次**（4096 × 4 lines × 25 次 move）
上述三个函数的 Python 调用。

### R.2 New implementation

新的 movement 路径由 `_move_groups(boards, group_actions)` 承担：

```text
_move_groups(boards, actions)
  for action in range(4):                        ← 固定常量循环（4 个动作）
      selected = flatnonzero(actions == action)
      gather 整组 board 到 M0 规范行列布局      ← 一次 fancy-index
      reshape 成 (R, 4)，R = group_size × 4，uint8
      packed   = _pack_left_rows(lines)          ← rank-based gather，一次完成
      merged, line_reward = _merge_left_rows(packed)
                                                 ← 固定 for _column in range(3)
      rewards[selected] = line_reward 按 board 求和（int64）
      scatter 回真实 board 索引                  ← 一次 fancy-index
      moved[selected] = (afterstate != board)
```

关键实现点：

| 组件 | 做法 |
| --- | --- |
| 行布局 | 复用 M0 `reference_env._line_indices`，逐 board 平铺并以该 board 的 16 格为基准重定位；`_LINE_ORDER[a]` / `_LINE_INVERSE[a]` 为 16 元置换 |
| pack-left | rank-based gather：`rank = cumsum(rows != 0, axis=1) - 1`，把非零值散射到 `(row, rank)` |
| merge | 三个固定边界 `0,1,2`，每个边界一次全数组布尔掩码；per-row `blocked_from_previous` mask 表达「一个 tile 每步最多 merge 一次」，保证 `[1,1,1,1] → [2,2,0,0]` |
| reward | 查表 `_MERGE_REWARD[e] = 1 << (e + 1)`（`int64`，size 63），`np.errstate(over="raise")` 兜底 |
| overflow | 门控 `packed.max() >= MAX_SAFE_MERGE_EXPONENT`，命中才做惰性审计并抛 `OverflowError` |
| `moved` | `afterstate != board`（与 M0 `legal_mask` 定义逐字一致），无第二套规则 |
| 删除 | `_LINE_CACHE`、`_merge_line_cached`、`_shift_line`、`_line_changed` 及 `_move_batch` 中的 per-board 循环**已全部删除** |

`legal_mask_batch` 与 `is_terminal_batch` 共用 `_move_all_actions_batch`，
因此二者的向量化与 `move_batch` 完全一致。

**允许保留的 Python 循环**（均为编译期常量，与 N 无关）：

```text
for action in range(4)                    # 四个动作
for _column in range(BOARD_COLUMNS - 1)   # 三个 merge 边界
```

`R` 的构造只有一次 `reshape`，没有逐 board 迭代。

### R.3 Required fields

```text
Previous implementation:
  _move_batch 的 per-board Python 循环
  （_LINE_CACHE + _merge_line_cached + _shift_line + _line_changed）
  —— 由 §R.4 的回归测试钉死，production path 已不再调用

Previous move throughput:
  ~112 k boards/s（M1 第一轮实现当时的读数）
  152.0 k boards/s（本轮同机重新测得的 baseline_before_vectorization.json，
                    num_envs = 4096，median of 3）
  说明：两者都是同一个 per-board 实现，差异来自两次会话之间的运行环境；
        本轮性能验收一律以**同机重测的 152.0 k** 为 OLD 基准。

New implementation:
  _move_groups + _pack_left_rows + _merge_left_rows
  （rank-based pack + 3 边界固定掩码 merge，全部数组级）

N-dependent Python movement loop:
  NONE

Old primitive throughput:
  move_batch          152.0 k boards/s
  legal_mask_batch     30.1 k boards/s
  is_terminal_batch    31.5 k boards/s

New primitive throughput:
  move_batch          787.4 k boards/s
  legal_mask_batch    228.3 k boards/s
  is_terminal_batch   180.2 k boards/s

Primitive speedup:
  move_batch           5.18x
  legal_mask_batch     7.57x
  is_terminal_batch    5.72x

Old 4096-env throughput:
  14,826 transitions/s   (baseline_before_vectorization.json, 4096 env × 16 steps
                          = 65,536 transitions)

New 4096-env throughput:
  73,571 transitions/s   (benchmark_results.json, 4096 env × 48 steps
                          = 196,608 transitions)

End-to-end speedup:
  4.96x

Old bottleneck:
  _move_batch 的 per-board / per-line Python 循环
  （cProfile: 409,600 次 _merge_line_cached / _shift_line / _line_changed 调用）

New bottleneck:
  movement 内核的 NumPy 调用开销与内存带宽
  （cProfile: _pack_left_rows 的 ndarray.nonzero 579 次、cumsum 182 次；
    调用次数只取决于固定动作分组数 4，与 batch 内 board 数无关）

Old saturation:
  1024 -> 4096 吞吐下降 44%，4096 / 8192 / 16384 持平于 ~14.7 k transitions/s

New saturation:
  1024 -> 4096 吞吐下降 31%，4096 / 8192 / 16384 持平于 ~72 k transitions/s
  （平台高度提升 4.9x）
```

### R.4 New regression tests

`tests/test_m1_vectorization.py`（8 条）：

1. **`test_production_path_never_calls_m0_move_without_spawn`** —
   把 `reference_env.move_without_spawn` monkeypatch 成**抛异常**，然后依次执行
   `move_batch`（四个方向各一次 + 一次随机动作混合 batch 300 boards）、
   `legal_mask_batch`、`is_terminal_batch`、`Fast2048BatchEnv.step` /
   `reset_where`。任何一次对 M0 per-board 内核的调用都会让测试失败。
   测试结束时断言调用次数为 **0**。
2. **`test_8192_board_batch_matches_m0_on_sampled_indices`** —
   **一次** `move_batch` 处理 8,192 个随机 board（seed 20260918，exponent 0..20，
   empty probability 0.5）与随机动作，然后随机抽样 **512** 个下标，
   逐个与 M0 `move_without_spawn` 比对 afterstate / reward / moved，
   要求 **0 mismatch**。
3. **`test_python_line_count_is_constant_in_batch_size`** —
   用 `sys.settrace` 统计 movement 内核实际执行的 **Python 行数**：
   N=8 与 N=2048 两次测量，要求 per-board Python 行数下降一个数量级以上。
   若有人重新引入 per-board 循环，这个数字会随 N 线性增长，测试立刻失败。
4. **`test_movement_kernels_have_no_per_board_python_iteration`** —
   静态检查：AST 抽出内核函数的**可执行代码**（排除 docstring），
   断言不出现 `np.apply_along_axis` / `np.vectorize` / `Reference2048Env` /
   `move_without_spawn(`，且所有 `for ... in range(...)` 的循环变量只能是
   固定常量名（`action` / `_column` / `_pass`），循环上界不得含
   `shape` / `len(` / `size` / `count` 等数据相关表达式。
5. `test_single_board_batch_is_still_exact`（4 个参数化用例）—— N=1 的
   batch 在四个方向上仍与 M0 逐 bit 一致（边界情形）。

### R.5 Same-machine A/B

工具：`benchmarks/benchmark_ab_vectorization.py`（新增）。
同一台机器、同一个进程、同一份输入、同一个 seed，背靠背测量：

* **OLD** = per-board 标量路线，每块 board 调用一次
  `reference_env.move_without_spawn`；
* **NEW** = `game2048.fast_env` 的向量化 batch 内核。

两侧结果完全一致（脚本内置断言）：

```text
agreement check : OK (afterstates, rewards, moved, mask, terminal)
afterstate check: OK (identical pre-spawn afterstates on 512 envs)
spawn check     : 8 steps x 512 envs -> spawns OLD 2,212 / NEW 2,217,
                  exponent sum OLD 2,448 / NEW 2,427
```

（afterstate 是 spawn 之前的确定性结果，两侧必须逐 bit 相同，脚本断言通过。
spawn 之后两侧使用**各自独立的 RNG 流**，因此 spawn 格子不同是预期的；
两侧的 spawn 次数 2,212 vs 2,217（差 0.23%）与指数和 2,448 vs 2,427 一致，
说明 workload 形状等价，比较是 like-for-like。）

```text
primitive (N=4,096)     OLD /s      NEW /s      speedup
----------------------  ----------  ----------  -------
move_batch              100.86 k    1.11 M      10.97x
legal_mask_batch         29.25 k    231.99 k     7.93x
is_terminal_batch        27.29 k    242.30 k     8.88x
env step x48             10.78 k    146.30 k    13.57x
```

说明：这里的 OLD/NEW 两侧都**不含 spawn 的随机数生成开销差异**，
是纯粹的移动/合法性/终局实现对比。§R.3 中的 152.0 k → 787.4 k（5.18×）
是 benchmark harness 内两个独立进程的正式读数；本节的 10.97× 是同进程
背靠背对照。两者都远超 2.0× 的验收阈值，方向与量级一致。

### R.6 Performance Exit Criteria

```text
判据 1: move_batch 相对修复前 baseline 提速 >= 2.0x
  实测:  152.0 k -> 787.4 k boards/s = 5.18x（同 harness、同 workload）
         同进程 A/B 对照:                = 10.97x
  结论:  PASS

判据 2: 4096-env 端到端吞吐相对修复前 baseline 提速 >= 1.5x
  实测:  14,826 -> 73,571 transitions/s = 4.96x
         OLD: 4096 env x 16 steps =  65,536 transitions
         NEW: 4096 env x 48 steps = 196,608 transitions
         （两侧环境规模 / seed / benchmark 逻辑相同，但正式记录的 step count
           不同，因此该 throughput 比用于证明长期 steady-state 吞吐提升，
           不属于严格逐指令 A/B；严格同进程 A/B 见 §R.5）
  结论:  PASS
```

同机修复前 baseline 原始数据保留在 `baseline_before_vectorization.json`，
修复后数据在 `benchmark_results.json`，A/B 数据在 `ab_vectorization.json`。
两轮使用同一台机器、同一个固定 seed、同一个 harness 与同一组环境规模；
但**所有档位的 step count 都不同**（OLD 一律 16 steps；NEW 为 64/64/48/24/12），
因此连 `total transitions` 也不相等。§G 的比值是按每 transition 归一的性能证据，
而非严格逐指令 A/B；严格同进程、同输入、交替测量的 primitive A/B 见 §R.5。
**没有任何档位被缩小以伪造提速**（NEW 侧每档总 transitions 均 ≥ OLD 侧）。

### R.7 What was explicitly NOT done

* 未执行 `git reset --hard`、未 checkout 到 `m0-reference-pass`、未 rebase /
  force push；
* 未删除或重写 M1 的任何正确性测试，未减少测试数量、未 skip、未 xfail、
  未放宽任何 tolerance；
* 未修改任何 M0 冻结文件（§C 重新验证）；
* 未引入 C++ / pybind11 / Cython / Numba / Triton / CUDA / Rust；
* 未修改任何公开 API 的函数名、签名、返回类型或语义；
* 未更换 benchmark workload 来伪造提速（两侧 workload 定义完全相同）；
* 未开始 M2，未安装 PyTorch，未使用 tuple 8×6 checkpoint。

---

## Q. Next Stage

```text
M2 NOT STARTED
```

M1 到此停止。未创建 `network.py`，未实现 Transformer / MLP / Q Head / V Head /
A Head，未引入 Trainer / Replay Buffer / Self-play / Double-Q / Target Network /
Champion / Teacher / N-tuple / Expectimax / Search Correction / High-tile Restart
Pool，未使用 tuple 8×6 checkpoint，未做 GPU forward。

M2 的输入已经就绪：一个与 M0 逐 bit 一致、可严格验证、
movement hot path 完全没有 N-dependent Python 循环、
并在 4096 env 处标定过饱和点（~72 k transitions/s）的批量状态生产者。


---

## S. Post-Audit int64 Reward Aggregation Fix

代码审查在 `a34e56b` 上发现一个**明确的 correctness bug**：多个**各自可表示**的
merge reward 在同一 row 或同一 board 内累加后可能静默回绕成负数。
本节记录该 bug、修复、回归测试与性能影响。

```text
Audit finding:
  Multiple individually representable merge rewards could overflow
  np.int64 during row/board aggregation without raising.

Affected pre-fix commit:
  a34e56b36a2fdf3e9b1eb66c0b3ea1afeeb5561e

Old tag:
  m1-fastenv-pass
  retained unchanged for history (NOT moved, NOT deleted)

Fix:
  checked int64 aggregation at three levels (row / board / score),
  all range checks performed BEFORE the addition, plus step() reordering
  so every check completes before the first state mutation.

Row-level checked aggregation:      PASS
Board-level checked aggregation:    PASS
Score checked aggregation:          PASS
Overflow atomicity:                 PASS

New regression tests:
  tests/test_m1_reward_overflow.py (31 tests)

pytest:
  422 passed / 0 failed / 0 skipped / 0 xfailed
  (M0 261 + M1 130 + overflow 31; was 391 before this fix)
  -> 修复 §T 之后为 432 passed（新增 10 条 legal/terminal/live-view 回归）

move_batch sanity:
  ~787 k boards/s (pre-fix, benchmark_results.json)
  -> ~989 k boards/s (post-fix, same harness/methodology)
  same-process interleaved A/B vs pre-fix code: +1.7% median time

4096-env sanity:
  73,571 transitions/s (pre-fix)
  -> 84,990 transitions/s (post-fix, 4096 env x 48 steps, same harness)

M0 modified:
  NO

M2 started:
  NO

Final status:
  M1 = FINAL PASS
```

### S.1 The bug

两层各自都会静默回绕，而当时的保护是**无效**的。

**Row 层。** `_merge_left_rows` 用 `reward[can_merge] += _MERGE_REWARD[...]`
累加同一条 line 上不同边界的 merge。`[61,61,61,61]` 会 merge 两次：

```text
(61+61 → 62) + (61+61 → 62) = 2**62 + 2**62 = 2**63 = 9223372036854775808
INT64_MAX                                   = 9223372036854775807
```

修复前的实测输出：

```text
M0 reward = 9223372036854775808        (Python int, 无上限)
M1 reward = -9223372036854775808       ← 静默回绕
```

**Board 层。** `_move_groups` 用
`with np.errstate(over="raise"): line_reward.reshape(...).sum(axis=1)` 聚合
4 条 line。两个 `[61,61]` 行各付 `2**62`，合计 `2**63`，同样回绕。

**为什么 `np.errstate` 挡不住。** NumPy **只对整数标量运算**发出 overflow
信号；整数组加法与整数 reduction 一律静默回绕：

```python
a = np.array([2**62, 2**62], dtype=np.int64)
with np.errstate(over="raise"):
    a.sum()        # -> -9223372036854775808，没有任何异常
```

已实测确认（`a + a` 同样不抛；只有 `np.int64(2**62) + np.int64(2**62)` 这种
标量运算才会抛 `FloatingPointError`）。因此原有的 `np.errstate(over="raise")`
**不是** int64 reward 的 correctness 保障。

### S.2 The fix

新增内部常量与受检加法 helper：

```python
_INT64_MAX: np.int64 = np.iinfo(np.int64).max

def _checked_add_nonnegative_int64(total, increment, *, context):
    if np.any(increment > (_INT64_MAX - total)):
        raise OverflowError(...)      # 先检查
    return total + increment          # 再相加
```

三层各自接入：

| 层 | 位置 | 做法 |
| --- | --- | --- |
| Row | `_merge_left_rows` | 每个 merge 边界先比较 `increment > (_INT64_MAX - reward[can_merge])`，通过才 `reward[can_merge] += increment` |
| Board | `_move_groups` | 固定 `for column in range(BOARD_COLUMNS)` 四次受检累加；`headroom` 递推而非每次重算 `INT64_MAX - total` |
| Score | `Fast2048BatchEnv.step` | `_checked_add_nonnegative_int64(self._scores, rewards, ...)` |

`_move_groups` 中**固定 4 次的 `for column in range(BOARD_COLUMNS)`** 是与 N
无关的编译期常量循环，符合规格；**没有**任何 `for board in boards` /
`for row in rows` 之类的 N-dependent 循环。reward dtype 仍固定 `np.int64`；
**未**使用 `dtype=object`，**未**退化为 per-board Python `int`。

### S.3 step() atomicity

修复前 `step()` 的顺序是：

```text
move → spawn（写 _boards + 消耗 RNG）→ 聚合 reward → _scores += rewards
```

board-level 聚合一旦抛异常，**board 已经被 spawn 改过、RNG 也已经被消耗**，
环境处于半提交状态。

修复后顺序改为：

```text
1. move_batch
2. 得到 rewards
3. 检查 scores + rewards 是否 int64-safe      ← 所有范围检查都在此完成
4. 只有通过才 spawn（写 board + 消耗 RNG）
5. 提交 score
6. terminal
```

因此 `OverflowError` 必然发生在**第一次状态改动之前**：
board / score / RNG 全部保持原样。**未**引入任何 transaction / 回滚系统，
**未**为测试特殊回滚 RNG —— 只是把既有的检查放在正确的位置。
对不会溢出的普通对局，RNG 调用次数与顺序完全不变（检查是纯比较，不抽随机数）。

### S.4 Exact boundary results

手工核对的边界值（规格 §41）：

```text
INT64_MAX   = 9223372036854775807 = 2**63 - 1
2**62       = 4611686018427387904
2**63       = 9223372036854775808  > INT64_MAX
2**62+2**61 = 6917529027641081856  <= INT64_MAX
```

| 情形 | M0 reward | M1 行为 |
| --- | --- | --- |
| 单个 `61 + 61` | `2**62` | 正常返回 `2**62` |
| 单个 `62 + 62` | `2**63` | `OverflowError`（与修复前一致） |
| `[61,61,61,61]` 一行两次 merge | `2**63` | **`OverflowError`（本次修复）** |
| 两个 `[61,61]` 行 | `2**63` | **`OverflowError`（本次修复）** |
| `[61,61]` + `[60,60]` | `2**62+2**61` | **正常返回精确值（本次修复保证不被误拒）** |

四方向（UP / DOWN / LEFT / RIGHT）均已参数化验证；其中 `[61,61,61,0]`
这类**非对称**行也单独验证，确保 destination 侧没有被搞混。

### S.5 Performance impact

性能 sanity 命令：
`python benchmarks/benchmark_m1_env.py --num_envs 4096 --steps 48`

| 指标 | 修复前 | 修复后 | 变化 |
| --- | --- | --- | --- |
| `move_batch` primitive（同 harness） | 787,435 boards/s | **988,751 boards/s** | +25.6% |
| 4096 env × 48 steps | 73,571 transitions/s | **84,990 transitions/s** | +15.5% |

（4096-env 档两次独立复测为 83,950 与 84,990 transitions/s。）

两次测量之间运行环境发生了变化（OS build 从 `10.0.26100` 变为 `10.0.26200`），
因此上表**不能**用来判断检查开销。为此另做了**同进程、同输入、交替测量**的
严格 A/B：用 `git show a34e56b:src/game2048/fast_env.py` 取出修复前模块，
与当前模块在同一进程内对同一批 4,096 boards 交替计时各 41 次：

```text
PRE-FIX  move_batch  中位 3,664.1 us   -> 1,117,873 boards/s
CURRENT  move_batch  中位 3,727.0 us   -> 1,099,007 boards/s
中位比值 1.0172 (+1.7%)   min 比值 1.0434 (+4.3%)   mean 比值 1.0257 (+2.6%)
```

即安全检查的真实代价约 **+1.7%**（中位），远低于 10% 的告警阈值。

中间版本曾出现 +10.2% 的回退。按要求先 profile 而不是删检查：逐构造二分定位到
开销来自受检路径里两处**额外的稀疏 mask 索引**（`np.subtract(..., where=...)`
的 masked 写 + 一次额外 gather），而不是比较运算本身。把 row 层检查改成直接
比较 `increment > (_INT64_MAX - reward[can_merge])`（复用已有的 `can_merge`
索引、不新增 gather）、board 层改用递推 `headroom` 之后，回退降到 +1.7%。
**检查本身从未被删除或弱化。**

`_pack_left_rows` / `_merge_left_rows` 的向量化结构、`_LINE_ORDER` /
`_LINE_INVERSE`、canonical movement、spawn、RNG、reset / reset_where、
legal mask 与 terminal 定义**全部未改动**。

### S.6 What was explicitly NOT done

* 未重写 movement architecture，未改 `_LINE_ORDER` / `_LINE_INVERSE` /
  canonical movement / `_pack_left_rows` 算法；
* 未改 spawn / RNG / reset / reset_where / legal mask / terminal / Action /
  board encoding 的语义；
* 未修改任何公开 API 的名称、签名或返回类型；
* 未引入 C++ / pybind11 / Cython / Numba / Triton / CUDA / Rust；
* 未用 `dtype=object` 或 per-board Python `int` 逃避 int64 上限；
* 未为了性能删除或弱化 overflow 检查；
* 未删除、skip、xfail 任何既有测试，未放宽任何 tolerance；
* 未修改任何 M0 冻结文件；
* 未移动 `m0-reference-pass`，未移动 `m1-fastenv-pass`；
* 未开始 M2，未安装 PyTorch，未使用 tuple 8×6 checkpoint。

---

## T. Final Audit: Legal/Terminal Reward Isolation

本节记录最后一次代码审查发现的四个问题及其修复。修复完成后：

```text
M1 = FINAL AUDITED PASS
```

**权威 tag**：`m1-fastenv-audited-pass`。
`m1-fastenv-pass`（`a34e56b`）与 `m1-fastenv-final-pass`（`ce51f15`）自此仅作为
**历史 snapshot** 保留，均未移动、未删除。

### T.1 Audit finding 1 — legal/terminal 被 int64 reward 计算污染

**问题**：`legal_mask_batch` / `is_terminal_batch` 通过
`_move_all_actions_batch → _move_groups → _merge_left_rows` 计算 `moved`，而这条
路径**顺带**执行了 reward 聚合与 int64 range check。合法性根本不需要 reward，
却因此可能在真实 reward `> INT64_MAX` 时抛 `OverflowError`。

**Reproducer（修复前，已实测复现）**：

```text
board = [61, 61, 61, 61] / 0 / 0 / 0

M0  legal_mask(board)            -> [False  True  True  True]     正常返回
M1  legal_mask_batch(board[None]) -> OverflowError: row merge reward aggregation ...
M1  is_terminal_batch(board[None])-> OverflowError（同上）
M1  step(DOWN)                    -> OverflowError（spawn 之后 terminal 计算触发）
```

`DOWN` 本身完全正常（reward = 0，四个 61 沿列下滑并不 merge）；抛异常的是
terminal 检查里对 LEFT/RIGHT 这两个**假设动作**的 reward 求值。

**Fix**：同一套 movement core 增加**内部模式**，不复制第二套 movement 实现，
公共 API 完全不变：

```text
compute_rewards=True    move_batch / step
                        完整 int64 契约（单 merge + row + board + score），一个检查未删

compute_rewards=False   legal_mask_batch / is_terminal_batch
                        同一套 pack / merge / pack / afterstate / moved
                        不做任何 reward 运算
                        仍强制执行 uint8 tile exponent overflow
```

两个 overflow 被拆成最小的两个函数：`_audit_tile_merge_overflow`（uint8，
两种模式都跑）与 `_audit_merge_reward_overflow`（int64，仅 reward 模式）。
`_move_all_actions_batch` 现在只返回 `moved`，不再构造无用的 reward 数组。

核心不变量：**legal legality is independent of M1 int64 reward representation.**
但 tile 上限不在豁免范围内 —— `[255, 255, 0, 0]` 走 `legal_mask_batch` 仍然
`OverflowError`（与 M0 一致），绝不静默 wrap 成 `0`。

### T.2 Audit finding 2 — `scores` live view 被破坏

**问题**：int64 score 修复把提交写成了 `self._scores = new_scores`，**替换了
backing ndarray**。此前 `env.scores` 交付出去的 view 因此永久停留在旧值 ——
而被替换掉的正是文档承诺的 "read-only **live** view"。

```text
（修复前实测）
backing preserved : False
shares memory     : False
view[0] = 0        env.scores[0] = 4      # view 已失效
```

**Fix**：`self._scores[:] = new_scores`（原地提交）。顺序保持
move → 全部 range check → spawn → score commit → terminal，
score commit **没有**被提前到 spawn 之前，所有 int64 检查仍在任何 state
mutation 之前完成。

### T.3 Audit finding 3 — boundary fuzz 存在过宽的 false-positive 断言

**问题**：`test_randomized_boundary_differential_never_wraps` 中

```python
assert m0_unrepresentable or int(entry.max()) < 62
```

这个 `or` 分支允许「M0 明明可精确表示、M1 却抛异常」的情况静默通过。

**Fix**：收紧为严格断言，不留任何例外：

```python
assert m0_unrepresentable, (
    f"{context}: M1 raised although M0 represents the transition exactly"
)
```

（实测该 `or` 分支在修复前后都不曾被触发 —— 它能触发的唯一途径正是上述
false rejection，因此收紧后 6,300 条 fuzz 依然全绿，但断言不再有漏洞。）

### T.4 Audit finding 4 — 4096 档 OLD/NEW step count 口径矛盾

**问题**：报告 §G 用 `14,826 → 73,571`，却写成「两侧均为 4096 × 48 steps」。
实际的冻结 JSON 是：

```text
baseline_before_vectorization.json  4096 env × 16 steps =  65,536 transitions
benchmark_results.json              4096 env × 48 steps = 196,608 transitions
```

而且 OLD 侧**所有档位**都是 16 steps，NEW 侧是 64/64/48/24/12，因此连
`total transitions` 都不相等（原表把它写成单一列，且声称「每档总 transitions
相同」，两处均属错误）。

**Fix**：只改文字，**未重新生成任何 JSON**。§G 表格拆成
`OLD iters / NEW iters / OLD total / NEW total` 四列并显式写出
`OLD = 4096 × 16`、`NEW = 4096 × 48`；§R.6 判据 2 与 §R 备注同步更正，
明确该 throughput 比用于证明长期 steady-state 吞吐提升，**不属于严格逐指令
A/B**（严格同进程 A/B 见 §R.5）。

### T.5 Tests

```text
新增 10 条（全部放入 tests/test_m1_reward_overflow.py，未拆新模块）：

test_legal_mask_does_not_depend_on_int64_reward_range
test_is_terminal_is_isolated_from_the_int64_reward_range
test_high_exponent_merge_legality_matches_m0[100-101]
test_high_exponent_merge_legality_matches_m0[254-255]
test_255_plus_255_tile_overflow_is_still_raised_by_legality
test_move_batch_still_rejects_the_high_reward_alternative
test_step_on_a_high_reward_alternative_board_completes
test_high_reward_legal_differential_matches_m0          (500 boards, 60..254)
test_scores_live_view_survives_step
test_scores_live_view_survives_step_reset_where_and_reset

原 31 条 overflow 测试全部保留，未删除、未降低、未放宽。
pytest: 432 passed / 0 failed / 0 skipped / 0 xfailed
        = 261 M0 + 161 M1 + 10 本次新增
```

新增 high-reward legal differential：500 块 board，exponent 集中在
`60..254`（刻意避开 `255`，因为那是唯一 tile 不可表示的取值），
`legal_mask_batch` / `is_terminal_batch` 与 M0 **逐位一致，0 mismatch**，
且 **0 次 reward 相关 `OverflowError`**；其中 ≥ 30 个 action 的真实 reward
超过 `INT64_MAX`，因此该 sweep 确实在检验「解耦」而非空跑。

### T.6 Performance

同进程交替 A/B，pre-fix = `ce51f15` 由 git 载入为第二个模块，
同一输入、median of repeats：

```text
metric              BEFORE      AFTER     CHANGE
move_batch        1,040,888  1,018,982     -2.1%     （噪声区间内，见下）
legal_mask_batch    244,688    281,251    +14.9%
is_terminal_batch   247,365    283,938    +14.8%
4096 env            140,454    151,404     +7.8%
```

三轮重复中 `move_batch` 的变化为 `-4.7% / +2.6% / -2.1%`，即围绕 0 波动；
reward 路径的代码除 gate 变量外逐行未变，该波动是机器噪声。
legal / terminal 如预期**变快**（去掉了纯浪费的 reward 运算），而非变慢，
因此不存在需要 profile 的 >10% 倒退。原始数据见 `sanity_after_audit.json`。
未重跑完整 worker sweep（本次只改动 legal/terminal 内部路径与 score backing）。

### T.7 What was explicitly NOT done

* 未重写 movement architecture / `_pack_left_rows` / canonical movement；
* 未改 `_LINE_ORDER` / `_LINE_INVERSE` / spawn 算法 / RNG 算法 / `reset` /
  `reset_where` / Action / board encoding / D4；
* 未改 `move_batch` / `legal_mask_batch` / `is_terminal_batch` /
  `enumerate_spawns_batch` / `spawn_random_batch` / `apply_spawn_batch` /
  `Fast2048BatchEnv` / 全部 `Batch*` dataclass 的名称、参数与返回语义；
* 未删除或弱化 row / board / score 任何一级 checked-add；
* 未建立第二套 adjacency legality 规则（legal 仍来自真实 movement semantics）；
* 未引入 C++ / pybind11 / Cython / Numba / Triton / CUDA / Rust；
* 未修改任何 M0 冻结文件，未移动任何既有 tag；
* 未开始 M2，未安装 PyTorch，未使用 tuple 8×6 checkpoint。

### T.8 Final Result

```text
Audit finding 1 (legal/terminal reward coupling):  FIXED
Audit finding 2 (scores live view):                FIXED
Audit finding 3 (fuzz false positive):             FIXED
Audit finding 4 (16/48 benchmark wording):         FIXED

M0 modified:  NO
M2 started:   NO

M1 = FINAL AUDITED PASS
```

权威 tag：`m1-fastenv-audited-pass`。
