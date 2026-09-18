# M1 Fast / Vectorized Environment 规格说明

> 本文件是 **M1 Fast Environment 的权威规格**。
> 它描述的实现位于 `src/game2048/fast_env.py`，其唯一正确性来源是
> **永久冻结的 M0 Reference Environment**（`M0_REPORT.md`、`docs/M0_ENVIRONMENT_SPEC.md`、git tag `m0-reference-pass`）。
> 二者冲突时，**M1 是错的一方**。

---

## 0. 模块定位

```text
M0 Reference Environment      ← correctness oracle / Golden Reference（冻结）
        ↓
M1 Fast / Vectorized Environment   ← 本文件描述的对象
        ↓
为 M2 提供高速、连续、可批量消费的 state
```

`src/game2048/fast_env.py` 的依赖边界是固定的：

| 允许依赖 | 说明 |
| --- | --- |
| Python 标准库 | `dataclasses`、`typing`、`operator` 等 |
| NumPy | 唯一的数值依赖 |
| `game2048.reference_env` | 复用 `Action`、常量与 `_line_indices` |

FastEnv core **不依赖** `psutil`、`torch`、`pytest` 或任何 benchmark 工具。
`benchmarks/` 下的东西只服务于测量。

M1 **不包含**：神经网络、Q/V/A Head、Trainer、Replay Buffer、Self-play、
Teacher、N-Tuple、Expectimax、Search Correction、高位 Restart Pool、CUDA 环境。

---

## 1. 棋盘表示

| 项目 | 固定值 |
| --- | --- |
| 类型 | `numpy.ndarray` |
| batch shape | `(N, 16)` |
| dtype | `numpy.uint8` |
| 内存布局 | C_CONTIGUOUS |
| 顺序 | row-major |

单棋盘展开顺序与 M0 完全一致：

```text
 0  1  2  3
 4  5  6  7
 8  9 10 11
12 13 14 15
```

cell 存 **tile 的 2 的指数**：`0` = 空，`1` = 2，`2` = 4，…，`20` = 1048576，
`21` = 2097152，`22` = 4194304，…，`255` 为 uint8 上限。

**指数永不 clamp。** 「把 > 2^20 的 tile 编码进网络 overflow bucket」是后续网络
输入阶段的事情，与 M1 环境无关。

---

## 2. Action 编号

编号直接取自 M0，M1 **不定义第二套 Action**：

```python
from game2048.reference_env import Action   # UP=0, DOWN=1, LEFT=2, RIGHT=3
```

`legal_mask_batch` / `BatchStepResult.legal` 的第 `i` 位对应 `Action(i)`。

方向语义（与 M0 完全一致，`row` 向下增加）：

```text
UP    = (-1, 0)      DOWN  = ( 1, 0)
LEFT  = ( 0,-1)      RIGHT = ( 0, 1)
```

---

## 3. 批量移动

### 3.1 `BatchMoveResult`

```python
@dataclass
class BatchMoveResult:
    afterstates: np.ndarray   # (N,16) uint8，新分配，绝不与输入共享内存
    rewards:     np.ndarray   # (N,)   int64
    moved:       np.ndarray   # (N,)   bool
```

### 3.2 `move_batch(boards, actions)`

| 参数 | 约束 |
| --- | --- |
| `boards` | `(N,16)`，整数 dtype，值域 `0..255`，不被修改 |
| `actions` | `(N,)`，整数 dtype，值域 `0..3`，不被修改 |

显式拒绝（抛 `ValueError`）：

* board 形状不是 `(N,16)`（包括 `(16,)`、`(N,4,4)`）；
* board dtype 是 bool / float；
* board 值域越界（当 dtype 非 uint8 时检查 `0..255`）；
* action 形状不是 `(N,)`；
* action 长度与 board 数量不一致；
* action dtype 是 **float / bool / string**；
* action 取值不在 `0..3`。

**禁止** `int(action)` 式静默强制转换：`1.9` 与 `True` 必须被拒绝，而不是变成 `1`。

### 3.3 行布局：复用 M0 的 `_line_indices`

M1 不重新推导四方向的行布局，而是直接使用 M0 的
`reference_env._line_indices(Action(a))`：

```text
UP    : line i = [i, i+4, i+8, i+12]                 （列，自上而下）
DOWN  : line i = [i+12, i+8, i+4, i]                 （列，自下而上）
LEFT  : line i = [4i, 4i+1, 4i+2, 4i+3]              （行，自左而右）
RIGHT : line i = [4i+3, 4i+2, 4i+1, 4i]              （行，自右而左）
```

每行第 0 位恒为**目的格**，因此四个方向共用同一个「向 0 号位合并」内核，
不存在四套独立 merge 算法。

### 3.4 单行 merge 语义（完全继承 M0）

对一行指数：

1. 去掉 0；
2. 非零 tile 保持原顺序；
3. 从 0 号位（目的侧）向右扫描；
4. 相邻且相等才 merge；
5. 一个 tile 每步最多 merge 一次；
6. 新生成的 tile 本步不得再次 merge；
7. 末尾补 0。

必须成立：

```text
[1,1,1,1] → [2,2,0,0]   reward = 8
[1,1,2,0] → [2,2,0,0]   reward = 4      （禁止 [3,0,0,0]）
[2,2,2,0] → [3,2,0,0]   reward = 8
[1,1,1,0] → [2,1,0,0]   reward = 4
[1,0,1,1] → [2,1,0,0]   reward = 4
```

### 3.5 实现方式：跨 board 全向量化

移动内核是**跨 board 完全向量化**的，production hot path 中**没有任何与 N 相关的
Python 循环**。允许出现的 Python 循环只有三个编译期常量：

```text
for action in range(4)                  # 四个动作
for _column in range(BOARD_COLUMNS - 1) # 三个 merge 边界
```

一个 batch 的移动分两步：

**第一步：按动作分组（最多 4 次固定迭代）。**
`_move_groups(boards, actions)` 把 `(N,)` 的 action 数组拆成至多四个组，每组
用一个 fancy-index gather 一次性把整组 board 搬到 M0 的规范行列布局
（`_LINE_ORDER[a]`，逐 board 平铺、以该 board 的 16 格为基准重定位），
reshape 成 `(R, 4)`（`R = group_size × 4`，dtype `uint8`，C-contiguous）。

**第二步：`(R, 4)` 行批上的数组级 pack → merge → pack。**

* `_pack_left_rows(rows)`：rank-based gather。对每一行算非零元素的累计秩
  `rank = cumsum(rows != 0, axis=1) - 1`，再把非零值散射到 `(row, rank)`。
  一次完成，不是逐格平移。
* `_merge_left_rows(packed)`：三个固定边界 `0, 1, 2` 各一次全数组 mask 操作。
  「一个 tile 每步最多 merge 一次」由 per-row 的 `blocked_from_previous` bool
  mask 表达，即 `[1,1,1,1] → [2,2,0,0]`。reward 通过查表
  `_MERGE_REWARD[e] = 1 << (e + 1)`（`int64`）累加。
* 完成后用 `_LINE_INVERSE[a]` 一次性 scatter 回真实 board 索引。

**`moved` 的定义没有改变**：`moved = (afterstate != board)`，与 M0
`legal_mask` 的定义逐字一致（因此 `legal[a] ⇔ move(a).moved` 仍然成立）。

`legal_mask_batch` 与 `is_terminal_batch` 复用同一个 `_move_all_actions_batch`
向量化内核，不引入第二套合法性规则。

**为什么不需要记忆化。** `_LINE_CACHE` / `_merge_line_cached` / `_shift_line` /
`_line_changed` 这套 per-line 记忆化实现已在向量化重写中**删除**：它们正是
N-dependent Python 循环的来源。删除后正确性由 differential test 重新验证
（0 mismatch），性能反而提升数倍（见 `M1_REPORT.md` §R）。

---

## 4. reward 与 overflow

两个指数为 `e` 的 tile 合并：

```text
e + e → e + 1        reward = 2 ** (e + 1)
```

一轮 action 内所有 merge 的 reward 求和，累加 dtype 固定为 `numpy.int64`。

**禁止 `uint8` 保存 reward。**

必须通过：

```text
16 + 16 → 17   reward = 131072
20 + 20 → 21   reward = 2**21
21 + 21 → 22   reward = 2**22
```

### 4.1 高位支持范围与 divergence

M0 的 reward 是 **Python 任意精度 int**，M1 是 **int64**。

| | 支持范围 |
| --- | --- |
| M0 | 任意（`254 + 254 → 255`，reward = `2**255`） |
| M1 | 最高 exponent `61`（reward `2**62`） |

`MAX_SAFE_MERGE_EXPONENT = 62`：指数 `e >= 62` 的 merge 需要 reward `2**(e+1) >= 2**63`，
超出 int64。

M1 **绝不静默回绕**。当某行的 merge 无法精确表示时：

* 抛出 **`OverflowError`**（不会返回错误数字，也不会 wrap）；
* 错误信息给出触发 merge 的指数与原因。

这是 **M1 与 M0 唯一一处有意的语义差异**，并且永远是显式异常。
该范围的 board 在真实对局中不可达（到 exponent 53 之前 tile 数早已爆掉），
M0 仍然完整支持它们。

非 merge 的高位 tile 只是滑动时**不会**抛异常（不涉及 reward）：

```text
board[0] = 255, action = RIGHT  →  正常滑动，reward = 0，无异常
```

---

## 5. `legal_mask_batch`

```python
def legal_mask_batch(boards) -> np.ndarray    # (N,4) bool，顺序 UP, DOWN, LEFT, RIGHT
```

定义与 M0 完全一致：

```text
legal_mask_batch(s)[i, a]  ⇔  move_without_spawn(s[i], a).moved
```

M1 **不允许**另写一套「有没有空格 / 有没有相邻相同」的独立判定。
实现上四个方向都真算一遍：正确性优先于少量额外算力。

---

## 6. `is_terminal_batch`

```python
def is_terminal_batch(boards) -> np.ndarray   # (N,) bool
```

定义固定：

```text
terminal = 没有合法动作 = ~legal_mask_batch(boards).any(axis=1)
```

**禁止**用「棋盘已满」代替 terminal：满盘但有相邻相同 tile **不是** terminal。

---

## 7. Exact spawn enumeration（CSR flat layout）

### 7.1 `BatchSpawnEnumeration`

```python
@dataclass
class BatchSpawnEnumeration:
    states:          np.ndarray   # (M,16) uint8
    probabilities:   np.ndarray   # (M,)   float64
    parent_indices:  np.ndarray   # (M,)   int64
    spawn_indices:   np.ndarray   # (M,)   int64
    spawn_exponents: np.ndarray   # (M,)   uint8
    offsets:         np.ndarray   # (N+1,) int64
    empty_counts:    np.ndarray   # (N,)   int64
```

`empty_counts` 是附加字段：输出规模就是它驱动的（`2 * n`），报告与 benchmark 都需要。

**禁止**返回 `list[list[np.ndarray]]` 作为正式高速接口。

### 7.2 顺序（冻结）

对每个 parent：

* 空格按 **flat index 升序**；
* 每个空格依次输出 **exponent 1**、**exponent 2**。

因此 parent 的 outcome 数量为 `2n`，最多 `32`。满盘为 `0`。

### 7.3 CSR 语义

`offsets` 是真正的行指针：parent `i` **连续**拥有行 `offsets[i] : offsets[i+1]`。
实现按「空格数相同」分块生成后散列回 parent 序，因此既没有 per-board 循环，
也保证每个 parent 的区间连续。

### 7.4 概率

```text
P(exponent 1 at a cell) = 0.9 / n
P(exponent 2 at a cell) = 0.1 / n
```

每个 parent 的 probability 和在浮点容差内为 `1.0`。

---

## 8. Random batch spawn

### 8.1 `BatchSpawnResult`

```python
@dataclass
class BatchSpawnResult:
    states:          np.ndarray   # (N,16) uint8
    spawn_indices:   np.ndarray   # (N,)   int64
    spawn_exponents: np.ndarray   # (N,)   uint8
```

### 8.2 `spawn_random_batch(afterstates, rng)`

每个 board：

1. 只从**自己的** empty cells 中选位；
2. empty position 均匀；
3. tile 2 / exponent 1 概率 `0.9`；
4. tile 4 / exponent 2 概率 `0.1`；
5. 输入 board 不修改。

任何 board 没有空格 → 显式 `ValueError`，并指出 parent index。
`rng` 不是 `numpy.random.Generator` → `TypeError`。

---

## 9. Deterministic spawn application

```python
def apply_spawn_batch(afterstates, spawn_indices, spawn_exponents) -> np.ndarray
```

* **无 RNG、无副作用**；
* 只允许 `spawn_exponent ∈ {1, 2}`，否则 `ValueError`；
* 目标 cell 必须为 0，否则 `ValueError` 并指出 parent index；
* 返回新的 `(N,16)` uint8 数组。

它存在的唯一目的：让 differential test 能把 M0 的**某个明确 SpawnOutcome**
喂给 fast path，而不引入任何随机流。

---

## 10. RNG 语义

### 10.1 允许的随机源

只允许 `np.random.Generator`：

* **禁止** 全局 `np.random.*`；
* **禁止** Python `random`。

### 10.2 什么必须一致，什么不要求一致

| 项目 | 要求 |
| --- | --- |
| spawn 的 support（哪些 (cell, exponent) 可能） | 必须与 M0 完全一致 |
| 概率分布（`0.9/n` / `0.1/n`，位置均匀） | 必须与 M0 完全一致 |
| `step` 语义（何时 spawn、何时不 spawn） | 必须与 M0 完全一致 |
| 相同 seed 下与逐环境 M0 PCG64 的**逐 bit 序列** | **不要求**一致 |

原因：M1 使用向量化 RNG 批处理（一次 `(k,2)` draw block），而 M0 逐环境各抽两次。
测试因此**不得**依赖两者随机流恰好同步，而是使用：

* exact enumeration（§7）；
* predetermined spawn（§9）；
* distribution test（§11）。

### 10.3 可复现性

对同一个 `Fast2048BatchEnv`：

* 相同 seed + 相同初始 state + 相同 action sequence ⇒ **完全相同的 trajectory**；
* `reset(seed=x)` 结果只由 `x` 决定；
* 每步消耗的随机量只由「本步需要 spawn 的行数」决定，与该行是否合法无关。

### 10.4 illegal action 与随机数

* 全部 action 都非法 ⇒ **完全不消耗随机数**（generator state 逐 bit 不变）；
* 部分非法 ⇒ 只有合法行消耗各自的 2 个抽样；非法行既不 spawn 也不占抽样。

这使「相同 seed 的两个 env 走相同 action 序列」始终一致。

---

## 11. `Fast2048BatchEnv`

### 11.1 构造

```python
Fast2048BatchEnv(num_envs: int, seed: int | None = None)
```

内部状态只有三块缓冲：

```text
_boards : (N,16) uint8, C-contiguous
_scores : (N,)   int64
_rng    : np.random.Generator
```

**禁止**创建 N 个 `Reference2048Env`；**禁止**创建 N 个 Python env object。

`num_envs` 校验：非正整数 → `ValueError`；非 int（含 `bool`/`float`）→ `TypeError`。

### 11.2 属性

```python
num_envs  -> int
boards    -> (N,16) uint8 只读 live view（不复制、不创建 per-env object）
scores    -> (N,)   int64 只读 live view
rng       -> np.random.Generator
copy_boards()  -> 真正的 (N,16) 可写副本
copy_scores()  -> 真正的 (N,)  可写副本
```

`boards` / `scores` 返回的 view 被设为 `writeable = False`，
外部调用者无法直接破坏内部 state。

### 11.3 `reset(seed=None)`

* 重置**全部** env：board 清零、`score = 0`；
* 每个 game 连续做 **2 次真实 spawn**（正常 spawn 规则）；
* `seed is not None` ⇒ 重新播种，结果只由 seed 决定；
* `seed is None` ⇒ 延续当前随机流（gymnasium 约定，与 M0 一致）。

因此 `reset()` 之后**恰好两个非空格**，且两者 exponent ∈ {1,2}。

### 11.4 `reset_where(mask)`

* `mask`：`(N,)` bool；
* 只重置 `mask=True` 的 env：board 清零、`score = 0`、两次 spawn；
* 其余 env 的 board / score **逐 bit 不变**；
* 空 mask 是 no-op；
* shape 或 dtype 不符 → `ValueError`。

这是后续「大量异步游戏持续 self-play」的必需接口：
**不得**要求所有游戏死光以后一起 reset。

### 11.5 `BatchStepResult`

```python
@dataclass
class BatchStepResult:
    states:          np.ndarray   # (N,16) uint8  spawn 后的正式 state s'
    afterstates:     np.ndarray   # (N,16) uint8  spawn 前
    rewards:         np.ndarray   # (N,)   int64
    legal:           np.ndarray   # (N,)   bool
    terminated:      np.ndarray   # (N,)   bool
    spawn_indices:   np.ndarray   # (N,)   int64
    spawn_exponents: np.ndarray   # (N,)   uint8
```

### 11.6 `step(actions)`

固定流程（与 M0 逐条对齐）：

```text
当前正式棋盘 s
↓
move（= move_without_spawn 语义）
↓
illegal :  state = s（不变）、afterstate = s、reward = 0、legal = False
           score 不变、不 spawn、不消耗随机数
           spawn_indices = -1、spawn_exponents = 0
legal   :  state = afterstate + 一次随机 spawn、score += reward
↓
在 s' 上计算 terminated
```

* **禁止偷偷 auto-reset**：terminal 只体现为 `terminated = True`；
  何时 reset 由调用者 `reset_where(...)` 显式决定。
* `actions` 的校验规则与 `move_batch` 相同（拒绝 float / bool / string / 越界 / 形状错）。
* 返回的数组都是独立缓冲，调用者修改它们不会影响内部 state。

---

## 12. Benchmark 与 differential test 方法

### 12.1 正确性由 pytest 负责

* `tests/test_m1_*.py` 只判断 correctness；
* benchmark **不写** `assert steps_per_second > X` 这类武断阈值。

### 12.2 differential 方法

| 测试 | 规模 | 比较内容 |
| --- | --- | --- |
| ordinary differential | 10,000 boards × exponent 0..17 × 4 actions | afterstate / reward / moved |
| high-tile differential | 2,000 boards × exponent 0..22 × 4 actions | 同上 + legal mask + terminal |
| legal differential | 10,000 boards | `legal_mask_batch` vs M0 `legal_mask` |
| terminal differential | 10,000 boards | `is_terminal_batch` vs M0 `is_terminal` |
| spawn enumeration | 2,000 boards | outcome count / state / index / exponent / probability / 顺序 |
| D4 | 2,000 boards × 8 transforms × 4 actions | afterstate / reward / moved |
| illegal action | 手工 + 随机 | board / reward / score / legal / sentinel / RNG |
| reset / reset_where | — | 可复现、恰好 2 tile、只改 mask、其他不变 |

随机测试 seed 固定为 **20260918**，每个文件顶部写明 seed / sample count /
exponent range；失败时输出 sample index、board、action、两侧结果。

除 probability 浮点比较外，board / reward / moved / legal / terminal
一律要求 **exact equality**。

### 12.3 benchmark 方法

`benchmarks/benchmark_m1_env.py`：

* 环境规模 sweep：256 / 1024 / 4096 / 8192 / 16384；
* primitive throughput：move / legal mask / terminal / random spawn / exact spawn；
* exact spawn 分别测 empty = 2 / 4 / 8 / 12；
* producer → consumer harness（轻量 NumPy synthetic consumer，整 batch 处理）；
* worker scaling：1 / 2 / 4 个独立进程，每个持有自己的 `Fast2048BatchEnv`。

固定规则：

* 动作策略 = **uniform random legal action**，使用与 env spawn RNG **分离**的
  benchmark RNG；
* terminal 的 env 显式 `reset_where(terminated)` 后继续；
* 每个正式测量前先 warmup（默认 5 次迭代），正式至少 3 次 repeat，
  报告 median / min / max，吞吐比较主要看 **median**；
* 固定 seed `20260918`。

### 12.4 profiling 方法

`benchmarks/profile_m1_env.py` 用 `time.perf_counter()` 对明确阶段打点，
输出**人类可读的 wall-clock breakdown**（绝对时间 + 占实测 wall clock 百分比，
百分比之和约 100%）。`cProfile` 作为辅助细节另行打印。

阶段划分：

```text
Environment stepping (move + spawn + score)
Legal/action preparation (mask + terminal)
Spawn (random)
Exact spawn enumeration
Batch assembly (contiguous copy)
Memory copy
Synchronization / worker overhead
Other
```

---

## 13. 与 M2 的边界

M1 到此为止。以下**不属于** M1，也不得在 M1 中开始：

* Transformer / Residual MLP / Q Head / V Head / A Head；
* PyTorch Trainer、Replay Buffer、Self-play、Double-Q、Target Network、Champion；
* Teacher、N-Tuple、Expectimax、Search Correction、High-tile Restart Pool；
* CUDA 神经网络、GPU throughput sweep、任何用 GPU utilization 作为 M1 PASS 条件的判据；
* 用户已有的 tuple 8×6 checkpoint（不加载、不转换、不研究、不写接口）。

M1 交付的是：**与 M0 逐 bit 一致的高速批量状态生产者**，以及它的
differential test、scalability benchmark、profiler 与瓶颈结论。

关于是否迁移 C++：M1 **只记录结论**（见 `M1_REPORT.md` §N），
不自行启动重写。
