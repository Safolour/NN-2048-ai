# M0 报告：2048 Reference Environment

**状态：M0 = PASS**
**范围：仅 M0（游戏环境）。M1 及以后未开始，且不得自行开始。**

- 日期：2026-02（本机执行）
- 环境：Windows / CPython **3.12.10** / NumPy **2.5.3** / pytest **9.1.1**
- 仓库根目录：`D:\CodexTasks\NN-2048-ai`（执行 M0 前为空仓库）

---

## A. 实现文件

### 新增源码

| 文件 | 内容 |
| --- | --- |
| `src/game2048/__init__.py` | 公共导出（Action、纯函数、`Reference2048Env`、D4 变换） |
| `src/game2048/reference_env.py` | 棋盘/动作常量、`move_without_spawn`、`legal_mask`、`is_terminal`、`enumerate_spawns`、`spawn_random`、`Reference2048Env`、`MoveResult` / `SpawnOutcome` / `StepResult` |
| `src/game2048/symmetry.py` | `transform_board`、`transform_action`、`inverse_transform_id`（D4 八变换，动作映射由方向向量几何导出） |
| `docs/M0_ENVIRONMENT_SPEC.md` | 冻结规格（board 表示、exponent 语义、动作编号、move/reward/spawn/illegal/legal/terminal/reset/RNG/D4、高位不 clamp） |

### 新增测试

| 文件 | 测试数 |
| --- | --- |
| `tests/test_m0_moves.py` | 32 |
| `tests/test_m0_spawn.py` | 24 |
| `tests/test_m0_legal_terminal.py` | 23 |
| `tests/test_m0_d4.py` | 111 |
| `tests/test_m0_properties.py` | 22 |
| `tests/test_m0_high_tiles.py` | 20 |
| `tests/test_m0_env_api.py` | 29 |
| **合计** | **261** |

### 新增工程辅助文件

| 文件 | 说明 |
| --- | --- |
| `conftest.py` | 把 `src/` 加入 `sys.path`，使 `python -m pytest` 在未安装包的情况下即可运行（纯实现细节，不改变任何 M0 语义） |

### 明确未创建

`network.py` / `replay.py` / `teacher.py` / `expectimax.py` / `trainer.py` / `self_play.py`
以及任何神经网络、Replay Buffer、Self-play、Double-Q、Target Network、Champion、
Restart Pool、Search Correction、CUDA 相关代码 **均未创建**。

### 环境准备说明

仓库初始为空且系统 Python 中**没有 pytest**。已安装 `pytest 9.1.1`
（及其依赖 `pluggy` / `iniconfig` / `packaging` / `pygments` / `colorama`）。
未引入 Hypothesis 或任何其它非必要依赖。

---

## B. Rule tests（人工明确答案的 regression vectors）

`tests/test_m0_moves.py`（32 项）覆盖：四方向移动、单次 merge、双 merge、三相同 tile、
四相同 tile、中间有空格、禁止连锁二次 merge、无 merge 纯移动、illegal move、
多 merge reward 求和、输入不被修改、确定性、dtype/shape、动作编号冻结、非法 action 取值。

方向与合并顺序的关键回归向量（全部通过）：

| 输入（指数） | 方向 | 期望 afterstate | 期望 reward |
| --- | --- | --- | --- |
| `[1,1,1,1]` | LEFT | `[2,2,0,0]` | 8 |
| `[1,1,2,0]` | LEFT | `[2,2,0,0]`（**不是** `[3,0,0,0]`） | 4 |
| `[2,2,2,0]` | LEFT | `[3,2,0,0]` | 8 |
| `[1,1,1,0]` | LEFT | `[2,1,0,0]` | 4 |
| `[1,0,1,1]` | LEFT | `[2,1,0,0]` | 4 |
| `[1,1,2,2]` | LEFT | `[2,3,0,0]` | 12 |
| `[3,3,3,3]` | RIGHT | `[0,0,4,4]` | 32 |
| `[0,1,1,1]` | RIGHT | `[0,0,1,2]` | 4 |
| `[1,1,0,1]` | RIGHT | `[0,0,1,2]` | 4 |
| 列 `[1,1,0,0]` ×4 | UP | 顶行 `[2,2,2,2]` | 16 |
| 列 `[0,0,1,1]` ×4 | DOWN | 底行 `[2,2,2,2]` | 16 |

`tests/test_m0_legal_terminal.py`（23 项）覆盖：`legal_mask` shape/dtype/顺序、
与 `move_without_spawn(...).moved` 在 500 个随机盘面上逐位一致、
满盘无相邻相同 ⇒ terminal、满盘**有**相邻相同 ⇒ **不** terminal、
单 tile / 两 tile 永不 terminal、全空盘按定义 terminal（不可达状态，已在 SPEC 说明）、
`terminal == not legal_mask.any()` 在 500 个随机盘面上成立。

`tests/test_m0_env_api.py`（29 项）覆盖 `reset` / `step` / `board` / `score` /
`legal_mask()` / `is_terminal()`，以及 illegal action 的全部契约（见下方清单）。

### 补充：独立 oracle 交叉验证

`tests/test_m0_properties.py` 内含一份**刻意采用不同算法**的第二实现
（逐格单步滑动至稳定 → 单趟合并 + `already merged` 标记 → 再次滑动），
与 `move_without_spawn` 在 **60,000 次** (board, action) 上逐项比对
afterstate / reward / moved，**零分歧**。
这用于排除「参考实现自身算法写错却让所有自测通过」的系统性错误。

---

## C. Property tests（随机测试数量与 seed）

`tests/test_m0_properties.py`：固定 seed + 大量随机案例（无 Hypothesis）。

| 数据集 | 数量 | 随机区间 | seed |
| --- | --- | --- | --- |
| `MAIN_BOARDS` | **10,000** | exponent `0..17` | `20240517` |
| `HIGH_BOARDS` | 2,000 | exponent `0..20` | `987654321` |
| `D4_BOARDS` | 2,000 | exponent `0..17` | `13571113` |

逐条 Property 与实测规模：

| Property | 内容 | 实测规模 |
| --- | --- | --- |
| A | 移动前后（spawn 之前）tile 数值之和完全不变 | 10,000×4 + 2,000×4 + 2,000×4 = 56,000 |
| B | `moved == False` ⇒ `afterstate == original` 且 `reward == 0`（及逆否） | 10,000×4 + 3,000×4 |
| C | `legal_mask[a] == move_without_spawn(...).moved`（双向） | 10,000×4 + 2,000×4 |
| D | spawn 后 tile 数值总和只增加 2 或 4 | 约 6,000 次 spawn |
| E | 一次 spawn 恰好只改变一个原先为空的位置 | 同上 + 400 盘面全枚举 |
| F | D4 movement equivariance / terminal 不变性 / 逆变换回环 | 2,000×8×4 = 64,000 等变检查 |
| G | `terminal == 四个动作全部 illegal` | 10,000 + 3,000 稠密 + 200 保证 terminal 的棋盘 |
| — | 纯函数不修改输入 | 2,000 盘面 × (4 action + mask + terminal + 8 transform + enumerate) |

其它随机 seed：`20240518`、`777`（legal/terminal 对照），
`11/12/1000+t/5000+10t+a/9000+t/13000+t`（D4 各 transform），
`4242/31337/12345/999/777/5/2024`（spawn 与 RNG 契约），
`0..199`（reset 契约），`2024/99/1/2/0/3/8/21`（完整对局）。

### Spawn 精确性（以 `enumerate_spawns` 为准）

* 空格数 `n = 1..16` 全覆盖：outcome 数 = `2n`（`n=16` 时 32，符合上限）；
  概率 `0.9/n` / `0.1/n` 逐项精确匹配；
  `max |Σp − 1| = 2.220e-16`（实测，1..16 全扫）。
* `spawn_random` 抽样 sanity（20,000 次，seed `12345`）：
  tile-4 比例 **0.10140**、tile-2 比例 **0.89860**（判定区间 `0.08 ~ 0.12`）；
  16 格位置计数 min/max = **1197 / 1328**（期望 1250，最大相对偏差 6.24%）。

---

## D. D4 tests（验证的 invariant）

`tests/test_m0_d4.py`（111 项）+ `tests/test_m0_properties.py` 中的 D4 部分：

1. **八变换正确性**：以 `0..15` 全互异棋盘为输入，8 张**手工推算**的期望矩阵逐格比对通过。
2. **结构性质**：`t0` 恒等；旋转复合（`t1∘t1 = t2`、`t1∘t1∘t1 = t3`、四次回环）；`t4` 是对合；
   `t5..t7 == rotate(t4(board))`；每个变换都是格子置换（多重集不变）。
3. **动作映射正确性**：
   * 8×4 **手写回归表**逐项比对通过；
   * 每个 transform 下 `transform_action` 是 `{UP,DOWN,LEFT,RIGHT}` 的双射；
   * 另有独立几何复算：在 `0..15` 棋盘上 8×4 全部满足 `T(move(s,a)) == move(T(s), T(a))`。
4. **Movement equivariance（核心等价关系）**：
   `T(move(s,a).afterstate) == move(T(s), T(a)).afterstate`，
   且 **reward 完全相等**、**legal（moved）完全相等**；
   参数化 8 transform × 4 action 各 60 盘 + 主 sweep 2,000 盘 × 8 × 4。
5. **legal_mask 等变**：`mask[a] == transformed_mask[transform_action(a, t)]`，8 transform 全覆盖。
6. **terminal 不变性**：`terminal(T(s)) == terminal(s)`，8 transform × 200 随机盘面 + 2,000 盘主 sweep + 死盘。
7. **逆变换**：`inverse_transform_id == (0,3,2,1,4,5,6,7)` 冻结测试；
   `inverse(T)(T(s)) == s` 8 transform 全覆盖；逆的对称性。
8. **纯度**：`transform_board` 不修改输入，且返回值不与输入共享内存。

---

## E. High-tile tests（65536 / 131072 / 2^20+）

`tests/test_m0_high_tiles.py`（20 项）**全部通过**：

| 项目 | 结果 |
| --- | --- |
| `65536 + 65536 -> 131072`（`16 + 16 -> 17`），reward `131072` | PASS |
| `2^20 + 2^20 -> 2^21`（`20 + 20 -> 21`），reward `2**21` | PASS |
| `2^21 + 2^21 -> 2^22`（`21 + 21 -> 22`），reward `2**22` | PASS |
| 高位 tile 在四方向均正确合并（LEFT/RIGHT/UP/DOWN） | PASS |
| `[2^20,2^20,2^20,2^20] LEFT -> [2^21,2^21,0,0]` | PASS |
| 高低位混合 reward 求和 `[1,1,20,20] LEFT -> reward 4 + 2**21` | PASS |
| **不 clamp**：`exp 20`/`exp 21` 原样保存、原样滑动，无任何改写为 21 | PASS |
| 指数 `1,15,16,17,20,21,22,30,100,254,255` 原值穿过移动 | PASS |
| reward 是 Python `int`（非 `np.uint8`），`2097152 > 255` 且 `% 256 != 自身` | PASS |
| 环境 `score` 可累计 `2**21 + 2**21`（远超 uint8 范围） | PASS |
| `255 + 255` 在四个方向均抛 **`OverflowError`**（禁止静默回绕为 0） | PASS |
| `254 + 254 -> 255` 正常，reward `2**255` | PASS |
| `2**255` tile 可正常滑动（无 merge 时不报错） | PASS |
| 全 `255` 棋盘上做 `legal_mask` / `is_terminal` 同样抛 `OverflowError`（不静默回绕） | PASS |

---

## F. Test command（可重新执行全部 M0 tests 的确切命令）

在仓库根目录 `D:\CodexTasks\NN-2048-ai` 执行：

```powershell
python -m pytest
```

等价的显式写法：

```powershell
python -m pytest tests/test_m0_moves.py tests/test_m0_spawn.py tests/test_m0_legal_terminal.py tests/test_m0_d4.py tests/test_m0_properties.py tests/test_m0_high_tiles.py tests/test_m0_env_api.py
```

本机解释器（若 `python` 不在 PATH 上）：

```powershell
& "C:\Users\Safolour\AppData\Local\Programs\Python\Python312\python.exe" -m pytest
```

---

## G. Test result

```text
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-9.1.1, pluggy-1.6.0
rootdir: D:\CodexTasks\NN-2048-ai
collected 261 items

tests\test_m0_d4.py .................................................... [ 19%]
...........................................................              [ 42%]
tests\test_m0_env_api.py .............................                   [ 53%]
tests\test_m0_high_tiles.py ....................                         [ 61%]
tests\test_m0_legal_terminal.py .......................                  [ 70%]
tests\test_m0_moves.py ................................                  [ 82%]
tests\test_m0_properties.py ......................                       [ 90%]
tests\test_m0_spawn.py ........................                          [100%]

============================ 261 passed in 11.84s =============================
```

* **passed = 261**
* **failed = 0**
* **skipped / xfail = 0**
* pytest 最终摘要：`261 passed in 11.84s`

（首轮运行有 4 项失败，全部是**测试内手工推算的期望值写错**，
实现本身经独立 oracle 交叉验证正确；已修正期望值后全绿。
其中 1 项为随机稠密盘面样本恰好未命中 terminal，已改为固定 seed +
确定性构造的 terminal 棋盘族以消除偶然性。）

---

## 验收门槛逐条核对（必须全部满足）

| # | 门槛 | 结果 | 证据 |
| --- | --- | --- | --- |
| 1 | 四方向规则测试全部通过 | PASS | `test_m0_moves.py` |
| 2 | 合并顺序测试全部通过 | PASS | 见 B 节回归向量 |
| 3 | 单 tile 一步只能 merge 一次 | PASS | `test_left_two_two_four_does_not_chain` 等 + oracle |
| 4 | merge reward 正确 | PASS | `test_m0_moves.py` / `test_m0_high_tiles.py` |
| 5 | illegal action 不改变 board | PASS | `test_illegal_action_leaves_everything_unchanged` |
| 6 | illegal action reward = 0 | PASS | 同上 |
| 7 | illegal action 不 spawn | PASS | `test_illegal_action_does_not_spawn` |
| 8 | illegal action 不消耗 spawn RNG | PASS | `test_illegal_action_does_not_consume_spawn_rng`、`test_repeated_illegal_actions_are_completely_inert`（比对 `rng.bit_generator.state`） |
| 9 | legal action 才 spawn | PASS | `test_legal_step_spawns_exactly_one_tile` |
| 10 | spawn 位置均匀 | PASS | `test_spawn_random_chooses_cells_uniformly` + 枚举精确覆盖 |
| 11 | tile 2 概率 90% | PASS | `test_enumerate_spawns_probabilities_are_exact`（精确）+ 统计 sanity |
| 12 | tile 4 概率 10% | PASS | 同上（实测 0.10140） |
| 13 | legal_mask 正确 | PASS | `test_m0_legal_terminal.py` |
| 14 | terminal 正确 | PASS | 同上 |
| 15 | reset 正确 | PASS | `test_reset_starts_with_exactly_two_tiles` 等 |
| 16 | afterstate 语义正确 | PASS | `test_step_afterstate_matches_the_pure_core` |
| 17 | D4 八变换正确 | PASS | `test_transform_board_matches_hand_computed_vectors` |
| 18 | D4 action mapping 正确 | PASS | 手写表 + 双射 + 几何复算 |
| 19 | D4 movement equivariance 正确 | PASS | 8×4 参数化 + 2,000 盘主 sweep |
| 20 | Property tests 全部通过 | PASS | `test_m0_properties.py` |
| 21 | 10,000+ 随机 board 测试通过 | PASS | `SAMPLE_COUNT = 10000` |
| 22 | 65536 正常 | PASS | `test_65536_plus_65536_gives_131072` |
| 23 | 131072 正常 | PASS | 同上 |
| 24 | exponent 20 -> 21 正常 | PASS | `test_exp20_plus_exp20_gives_exp21` |
| 25 | exponent 21 -> 22 正常 | PASS | `test_exp21_plus_exp21_gives_exp22` |
| 26 | 输入 board 不被纯函数修改 | PASS | 6 个专项测试 + 2,000 盘面总扫 |
| 27 | 全部 pytest 通过 | PASS | 261 passed / 0 failed |
| 28 | `M0_ENVIRONMENT_SPEC.md` 已生成 | PASS | `docs/M0_ENVIRONMENT_SPEC.md` |
| 29 | `M0_REPORT.md` 已生成 | PASS | 本文件 |

---

## H. 已知问题

**None.**

以下两条**不是问题**，而是冻结定义推出的必然结论，已写入 SPEC 并配有测试，
供 M1 参考（避免 M1 误判为 bug）：

1. 全空棋盘没有合法动作，因此按 `terminal := not legal_mask.any()` 的定义是 terminal。
   该状态在真实对局中**不可达**（`reset` 必定生成 2 个 tile，且 tile 数不会降到 0）。
2. 只有 1 个 tile 时永远不是 terminal（tile 总能向至少一个方向滑动，
   已在 16 个位置逐一验证）。

另注：本机在 M0 开始前未安装 pytest，已安装 `pytest 9.1.1`；这是环境准备，不属于 M0 缺陷。

---

## 停止位置

M0 全部验收门槛已通过，**在此停止**。

* M1（高速并行环境 / golden reference 对照）**未开始**。
* 未创建任何 M1+ 模块，未实现任何网络、Teacher、Expectimax、Replay Buffer、
  Self-play、Double-Q、Target Network、Champion、Restart Pool、Search Correction、CUDA 代码。
* M1 必须等待新的明确指令。
