你现在负责本项目的 **M1：高速并行 2048 环境**。

这是一个已经完成 M0、准备进入性能工程阶段的 2048 AI 项目。

你的任务不是重新设计项目，不是重新设计 2048 规则，不是开始神经网络训练，也不是“顺便把后面的东西做了”。

你的唯一任务是：

```
> **在永久冻结的 M0 Reference Environment 之上，实现一个高吞吐、批量化、可严格验证的 Fast / Vectorized Environment，并完成 differential test、scalability benchmark、profiling 和 M1 报告。**
```

完成 M1 后必须停止。

**禁止开始 M2。**

---

# 0\. 最高优先级规则

执行优先级固定为：

1. 本提示词中的明确规定；
2. M0 已冻结的实际代码和测试语义；
3. 当前仓库中不与以上内容冲突的工程约定；
4. 其余纯实现细节采用最简单、最容易验证的方案。

不得自行重新设计：

- 2048 规则；
- board 编码；
- Action 编号；
- reward；
- spawn；
- terminal；
- legal mask；
- D4 定义；
- M0 API 语义；
- 后续网络架构；
- RL 算法；
- Teacher；
- Search。

如果你认为 M0 有 bug：

**不得直接修改。**

必须：

1. 给出最小复现 board；
2. 给出 M0 实际结果；
3. 给出冻结规则要求的结果；
4. 证明二者确实矛盾；
5. 在 `M1_REPORT.md` 中标记 `M0_BLOCKER`；
6. 停止该问题相关工作。

除此以外：

FastEnv 与 M0 不一致时，

**默认 FastEnv 是错的。**

---

# 1\. M0 永久冻结

仓库中的 M0 是 Golden Reference / Correctness Oracle。

如果仓库中存在：

`m0-reference-pass`

Git tag，

它就是 M0 冻结节点。

必须保留。

以下现有 M0 文件禁止修改：

``` text
src/game2048/reference_env.py
src/game2048/symmetry.py
src/game2048/__init__.py

tests/test_m0_moves.py
tests/test_m0_spawn.py
tests/test_m0_legal_terminal.py
tests/test_m0_d4.py
tests/test_m0_properties.py
tests/test_m0_high_tiles.py
tests/test_m0_env_api.py

docs/M0_ENVIRONMENT_SPEC.md
M0_REPORT.md
```

除非用户之后明确授权修 M0，否则：

**M1 不得编辑以上文件。**

M1 开始前记录：

``` bash
git status
git rev-parse HEAD
git rev-parse m0-reference-pass
```

M1 结束时再次检查。

必须证明：

M0 文件没有被 M1 修改。

---

# 2\. M1 明确不做什么

本阶段禁止实现：

- Transformer；
- Residual MLP；
- Q Head；
- V Head；
- A Head；
- PyTorch Trainer；
- Replay Buffer；
- Self-play RL；
- Double-Q；
- Target Network；
- Champion；
- Teacher；
- N-tuple；
- Expectimax；
- Search Correction；
- High-tile Restart Pool；
- CUDA neural network；
- M2 GPU throughput sweep。

也禁止使用用户已有的：

**tuple 8×6 checkpoint。**

这个 checkpoint 与 M1 无关。

不要加载。

不要转换。

不要研究。

不要给它写接口。

留给后面的 Teacher 阶段。

---

# 3\. M1 唯一目标

实现：

``` text
M0 Reference Environment
        ↓
correctness oracle

M1 Fast / Vectorized Environment
        ↓
大量 board 批量推进
        ↓
连续 batch
        ↓
为 M2 提供高速状态生产能力
```

核心要求：

**正确性第一。**

然后：

**吞吐第二。**

任何：

“更快但是和 M0 有一个 case 不一样”

的实现：

直接失败。

---

# 4\. 第一版技术路线固定

第一版 M1：

**使用 Python + NumPy 实现向量化 FastEnv。**

不要一开始写：

- C++ extension；
- pybind11；
- Cython；
- CUDA kernel；
- Rust；
- Numba；
- Triton；
- 自定义汇编。

原因：

M1 首先需要建立：

- 正确高速 baseline；
- differential test；
- benchmark harness；
- profiler；
- 数据布局。

只有 benchmark / profiling 以后才能决定是否值得迁移热点。

因此：

**第一版强制 NumPy Vectorized FastEnv。**

完成 NumPy baseline 后：

如果 profiling 显示存在明确的 Python / NumPy 热点，

只在 `M1_REPORT.md` 中记录：

``` text
C++ migration recommended: YES/NO
Reason:
Measured bottleneck:
Expected target:
```

**本次 M1 不自行启动 C++ 重写。**

等待用户下一条明确指令。

这样禁止 Agent 在 M1 中无限扩张工程范围。

---

# 5\. 新文件结构固定

新增：

``` text
src/game2048/fast_env.py

tests/test_m1_batch_move.py
tests/test_m1_differential.py
tests/test_m1_spawn.py
tests/test_m1_batch_env.py
tests/test_m1_high_tiles.py

benchmarks/benchmark_m1_env.py
benchmarks/profile_m1_env.py

docs/M1_FAST_ENV_SPEC.md
M1_REPORT.md
```

允许为了 benchmark 增加少量：

``` text
benchmarks/_utils.py
```

如果确有必要。

除此之外：

不要自行拆几十个模块。

不要建立复杂框架。

不要重构 M0。

---

# 6\. FastEnv board 表示固定

FastEnv batch board：

``` python
np.ndarray
shape = (N, 16)
dtype = np.uint8
C_CONTIGUOUS = True
```

仍然使用：

``` text
0  = empty
1  = 2
2  = 4
...
16 = 65536
17 = 131072
20 = 1048576
21 = 2097152
22 = 4194304
...
```

FastEnv：

**禁止把 exponent \> 20 clamp 为网络 overflow bucket。**

网络 overflow 编码：

不是 M1 的事情。

所有 board：

row-major。

索引仍然：

``` text
 0  1  2  3
 4  5  6  7
 8  9 10 11
12 13 14 15
```

---

# 7\. Action 编号固定

必须直接采用：

``` python
UP    = 0
DOWN  = 1
LEFT  = 2
RIGHT = 3
```

不得创造第二套 Action。

FastEnv 可以 import：

``` python
from game2048.reference_env import Action
```

M1 自己不得重新定义不同编号。

---

# 8\. Batch API 固定

`src/game2048/fast_env.py`

至少提供：

``` python
@dataclass
class BatchMoveResult:
    afterstates: np.ndarray   # (N,16), uint8
    rewards: np.ndarray       # (N,), int64
    moved: np.ndarray         # (N,), bool
```

以及：

``` python
def move_batch(
    boards: np.ndarray,
    actions: np.ndarray,
) -> BatchMoveResult
```

输入：

``` text
boards:
    shape (N,16)
    dtype uint8

actions:
    shape (N,)
    integer dtype
    each value ∈ {0,1,2,3}
```

错误输入必须显式拒绝。

禁止：

- float action；
- bool action；
- string action；
- shape 错误；
- board 非 `(N,16)`；
- action 数量与 board 数量不一致。

不要使用：

``` python
int(action)
```

这种会把：

``` text
1.9 -> 1
True -> 1
```

悄悄吞掉错误的写法。

---

# 9\. Batch movement 的实现方向固定

禁止：

``` python
for board in boards:
    move_without_spawn(board, action)
```

作为生产 FastEnv。

M0 reference 只能用于：

**test / differential oracle。**

正式 batch move 必须：

**跨 board vectorize。**

允许最多：

- 对 4 个 action 分组；
- 对固定的 4 行/列做小常数循环。

禁止：

对 N 个 board 做 Python-level per-board 主循环。

---

# 10\. 四方向统一为 Canonical LEFT

内部推荐并优先使用这一固定方案：

把不同动作的 board 转换成：

**朝 LEFT 合并的 canonical row representation。**

规则：

``` text
LEFT:
    原棋盘

RIGHT:
    每行左右翻转

UP:
    transpose，使原来的 column 变成 row

DOWN:
    transpose 后再左右翻转
```

统一执行：

``` text
compress left
→ merge left
→ compress
```

然后做逆变换恢复原方向。

不要为：

UP / DOWN / LEFT / RIGHT

维护四套彼此独立的 merge 算法。

这样减少：

规则分叉。

---

# 11\. 单行 merge 语义完全继承 M0

canonical LEFT row：

``` text
[a,b,c,d]
```

处理顺序：

1. 去掉 0；
2. 非零 tile 保持原顺序；
3. 从左向右；
4. 相邻且相等才 merge；
5. 一个 tile 每步最多 merge 一次；
6. 新生成 tile 本步不得再次 merge；
7. 末尾补 0。

必须正确：

``` text
[1,1,1,1]
→ [2,2,0,0]
reward = 8
```

``` text
[1,1,2,0]
→ [2,2,0,0]
reward = 4
```

禁止：

``` text
[1,1,2,0]
→ [3,0,0,0]
```

---

# 12\. FastEnv reward 表示

M1 实际训练高速路径：

``` python
dtype = np.int64
```

必须正确覆盖本项目实际支持和训练会使用的高 tile 范围，包括至少：

``` text
exp 16
exp 17
exp 20
exp 21
exp 22
```

必须通过：

``` text
16 + 16 -> 17
reward = 131072

20 + 20 -> 21
reward = 2**21

21 + 21 -> 22
reward = 2**22
```

禁止使用：

`uint8`

保存 reward。

如果检测到某个极端 synthetic board 的 merge reward 已超出 int64 可表示范围：

不得静默 overflow。

必须：

**显式抛出** **`OverflowError`。**

M0 仍负责这种极端 arbitrary-uint8 board 的 Python-int 完整语义。

M1 性能 fast path 不允许静默产生错误数字。

---

# 13\. Batch legal mask 固定

必须提供：

``` python
def legal_mask_batch(
    boards: np.ndarray
) -> np.ndarray
```

返回：

``` text
shape = (N,4)
dtype = bool
order = UP, DOWN, LEFT, RIGHT
```

定义必须仍然是：

``` text
legal[a] ⇔ move(board,a).moved
```

不要自己再写第二套：

“看有没有空格 / 看有没有相邻相同”

的独立 legal 判定。

FastEnv 可以同时计算四方向 move 来获得 moved。

正确性优先于少量额外算力。

以后如果优化：

也必须 differential test 证明等价。

---

# 14\. Batch terminal 固定

必须提供：

``` python
def is_terminal_batch(
    boards: np.ndarray
) -> np.ndarray
```

返回：

``` text
shape = (N,)
dtype = bool
```

定义：

``` text
terminal = no legal action
```

即：

``` python
~legal_mask_batch(boards).any(axis=1)
```

禁止直接使用：

``` text
board full => terminal
```

因为：

满盘仍可能 merge。

---

# 15\. Deterministic Batch Spawn Enumeration 固定

必须提供：

``` python
@dataclass
class BatchSpawnEnumeration:
    states: np.ndarray
    probabilities: np.ndarray
    parent_indices: np.ndarray
    spawn_indices: np.ndarray
    spawn_exponents: np.ndarray
    offsets: np.ndarray
```

固定 shape / dtype：

``` text
states:
    (M,16) uint8

probabilities:
    (M,) float64

parent_indices:
    (M,) integer

spawn_indices:
    (M,) integer

spawn_exponents:
    (M,) uint8

offsets:
    (N+1,) integer
```

这是 CSR-style flat representation。

禁止返回：

``` text
list[list[np.ndarray]]
```

作为正式高速接口。

---

# 16\. Spawn enumeration 顺序固定

对每个 parent board：

空格：

按 flat index：

**从小到大。**

每个空格依次输出：

``` text
exponent 1
exponent 2
```

对应：

``` text
P(exp1 at cell) = 0.9 / n
P(exp2 at cell) = 0.1 / n
```

如果：

``` text
n = number of empty cells
```

则：

``` text
outcome count = 2n
```

最多：

``` text
32
```

full board：

``` text
0 outcome
```

对应：

``` text
offset[i] == offset[i+1]
```

每个 parent 的 probability sum：

浮点容差内：

``` text
1.0
```

---

# 17\. Random Batch Spawn 固定

必须提供：

``` python
@dataclass
class BatchSpawnResult:
    states: np.ndarray
    spawn_indices: np.ndarray
    spawn_exponents: np.ndarray
```

以及：

``` python
def spawn_random_batch(
    afterstates: np.ndarray,
    rng: np.random.Generator,
) -> BatchSpawnResult
```

要求：

每个 board：

1. 只从自己的 empty cells 中选择；
2. empty position 均匀；
3. tile 2 / exponent 1 概率 0.9；
4. tile 4 / exponent 2 概率 0.1；
5. 输入 board 不修改。

如果任何输入 board 没有 empty cell：

显式报错并指出 parent index。

---

# 18\. Fast batch RNG 规则

FastEnv 使用：

``` python
np.random.Generator
```

禁止：

- global `np.random.*`
- Python `random`

同一个：

seed + 相同初始 state + 相同 action sequence

必须在 FastEnv 内部：

**完全复现相同 trajectory。**

M1 differential correctness 要求：

FastEnv 与 M0 的：

**概率规则、support、step 语义完全相同。**

不要求：

Fast batch sampler 在相同 seed 下与逐环境 M0 PCG64 调用产生逐 bit 完全相同的 spawn 序列。

原因：

M1 使用向量化 RNG 批处理。

但是：

概率分布必须一致。

测试 FastEnv 和 M0 时：

不要依赖两者 random stream 恰好同步。

使用：

- exact enumeration；
- predetermined spawn；
- distribution test

进行验证。

---

# 19\. 必须提供 Deterministic Spawn Application

为了 differential test 不被 RNG 干扰，

必须提供：

``` python
def apply_spawn_batch(
    afterstates: np.ndarray,
    spawn_indices: np.ndarray,
    spawn_exponents: np.ndarray,
) -> np.ndarray
```

只允许：

``` text
spawn_exponent = 1 or 2
```

目标 cell：

必须为 0。

否则：

显式报错。

这个函数：

无 RNG。

无副作用。

方便把：

M0 的某个明确 SpawnOutcome

应用到 FastEnv。

---

# 20\. Fast2048BatchEnv 类固定

必须提供：

``` python
class Fast2048BatchEnv:
```

构造：

``` python
Fast2048BatchEnv(
    num_envs: int,
    seed: int | None = None,
)
```

内部至少保存：

``` text
_boards:
    (N,16) uint8 contiguous

_scores:
    (N,) int64

_rng:
    np.random.Generator
```

禁止：

创建 N 个：

``` python
Reference2048Env
```

对象。

禁止：

创建 N 个 Python Env object。

这是 batch environment。

---

# 21\. reset API 固定

必须支持：

``` python
reset(seed=None)
```

重置全部环境。

返回：

``` text
(N,16) uint8
```

每个 game：

从空棋盘开始，

连续做两次真实 spawn。

score：

重置为 0。

同时必须支持：

``` python
reset_where(mask)
```

输入：

``` text
shape (N,)
dtype bool
```

只重置 terminal / 指定环境。

这是后续大量异步游戏持续 self-play 必需接口。

不得要求：

“所有游戏死光以后一起 reset。”

---

# 22\. step API 固定

必须提供：

``` python
@dataclass
class BatchStepResult:
    states: np.ndarray
    afterstates: np.ndarray
    rewards: np.ndarray
    legal: np.ndarray
    terminated: np.ndarray
    spawn_indices: np.ndarray
    spawn_exponents: np.ndarray
```

`step(actions)`：

输入：

``` text
actions shape (N,)
```

输出固定：

``` text
states:
    (N,16) uint8

afterstates:
    (N,16) uint8

rewards:
    (N,) int64

legal:
    (N,) bool

terminated:
    (N,) bool

spawn_indices:
    (N,) integer

spawn_exponents:
    (N,) integer
```

非法动作：

``` text
state = old state
afterstate = old state
reward = 0
legal = False
score unchanged
spawn_index = -1
spawn_exponent = 0
```

并且：

**不得 spawn。**

合法动作：

``` text
move
→ afterstate
→ one random spawn
→ state
→ score += reward
→ calculate terminal
```

---

# 23\. FastEnv 不自动 reset terminal game

`step()`：

**禁止偷偷 auto-reset。**

terminal：

只是：

``` text
terminated = True
```

是否 reset：

由调用者显式：

``` python
reset_where(...)
```

控制。

否则以后 trajectory boundary 会变得不清楚。

---

# 24\. FastEnv property 规则

至少提供：

``` python
@property
def num_envs(self) -> int
```

``` python
@property
def boards(self) -> np.ndarray
```

``` python
@property
def scores(self) -> np.ndarray
```

`boards` / `scores`：

不得要求每次访问都创建：

N 个 Python object。

允许返回：

**read-only ndarray view。**

禁止外部调用者直接破坏内部 state。

另提供必要时的：

``` python
copy_boards()
```

返回真正 copy。

---

# 25\. M1 Differential Test 主体固定

`tests/test_m1_differential.py`

至少运行：

### 普通随机 board

``` text
10,000 boards
exponent = 0..17
fixed seed
```

全部：

4 actions。

共至少：

``` text
40,000 board-action comparisons
```

比较：

- afterstate
- reward
- moved

必须：

**0 mismatch。**

---

# 26\. 高位 differential 固定

另外：

``` text
2,000 boards
exponent = 0..22
fixed seed
```

全部四动作。

比较：

- afterstate
- reward
- moved
- legal mask
- terminal

还必须显式测试：

``` text
16+16 -> 17
20+20 -> 21
21+21 -> 22
```

四个方向全部覆盖。

不能只测 LEFT。

---

# 27\. M0 回归向量必须再次经过 FastEnv

至少复用 / 重写这些输入：

``` text
[1,1,1,1]
[1,1,2,0]
[2,2,2,0]
[1,1,1,0]
[1,0,1,1]
[1,1,2,2]
```

LEFT / RIGHT。

并包含：

UP / DOWN。

FastEnv 必须与：

``` python
move_without_spawn
```

完全一致。

---

# 28\. legal / terminal differential

至少：

``` text
10,000 random boards
```

比较：

``` python
fast_legal[i]
```

与：

``` python
reference legal_mask(board_i)
```

要求：

100%。

同时：

``` python
fast_terminal[i]
```

与：

``` python
reference is_terminal(board_i)
```

100%。

---

# 29\. Spawn enumeration differential

至少：

``` text
2,000 random boards
```

包含不同 empty count。

逐 parent 比较：

- outcome count；
- state；
- spawn\_index；
- spawn\_exponent；
- probability；
- ordering。

要求：

100%。

float probability 使用：

``` text
abs tolerance <= 1e-12
```

---

# 30\. Random spawn statistical test

固定 seed。

至少：

``` text
100,000 sampled spawns
```

必须验证：

tile 4 总比例：

落在：

``` text
0.09 ～ 0.11
```

同时使用多个不同 empty-count board，

验证位置：

近似均匀。

统计测试只用于：

sampler sanity。

真正规则正确性：

以 exact enumeration 为准。

---

# 31\. Batch step differential 不允许靠同步 RNG

测试 BatchEnv step 时：

不要：

“FastEnv 和 ReferenceEnv 用同 seed，然后期待随机结果一样。”

正确方式：

测试 deterministic transition：

``` text
state
→ action
→ reference afterstate
→ predetermined spawn outcome
```

和：

Fast batch move + `apply_spawn_batch`

比较。

随机 sampler：

单独测试分布。

---

# 32\. Illegal action 必须专项测试

随机和手工 board 中：

测试非法动作。

必须验证：

- board 完全不变；
- reward = 0；
- score 不变；
- legal = False；
- spawn\_index = -1；
- spawn\_exponent = 0。

同时必须证明：

对 FastEnv 自身而言，

执行非法 action：

**不消耗 spawn RNG。**

测试方法：

两个相同 seed 的 FastEnv：

A：

``` text
illegal
→ legal
```

B：

``` text
legal
```

最终：

合法 step 的 spawn outcome

必须一致。

---

# 33\. reset / reset\_where 专项测试

必须验证：

`reset(seed=x)`：

可复现。

必须验证：

全部环境：

恰好两个非空 tile。

必须验证：

每个 tile：

只能 exponent 1 / 2。

必须验证：

`reset_where(mask)`：

只改变 mask=True 的 env。

其他 env：

board / score 不变。

---

# 34\. D4 测试

M1 不需要重新实现 D4。

继续使用：

M0：

``` python
transform_board
transform_action
```

对 Fast move 做：

``` text
T(FastMove(s,a))
==
FastMove(T(s),T(a))
```

随机至少：

``` text
2,000 boards
× 8 transforms
× 4 actions
```

至少比较：

- afterstate
- reward
- moved。

---

# 35\. Pure batch function 不得修改输入

必须专项测试：

- `move_batch`
- `legal_mask_batch`
- `is_terminal_batch`
- `enumerate_spawns_batch`
- `spawn_random_batch`
- `apply_spawn_batch`

不得修改：

caller 提供的 input board。

---

# 36\. 性能 hot path 禁止行为

正式 FastEnv hot path 禁止：

``` text
for each board:
    call Reference2048Env
```

禁止：

``` text
for each board:
    create ndarray
```

禁止：

``` text
for each board:
    create dataclass
```

禁止：

大量：

``` text
list[board]
```

禁止：

每 step：

重复构建不必要的：

- transform tables；
- index tables；
- constant masks。

常量：

模块初始化一次。

---

# 37\. Benchmark 不进入 pytest PASS/FAIL

性能 benchmark：

不要写成：

``` python
assert steps_per_second > 某个武断数字
```

因为不同机器性能不同。

正确性：

pytest 负责。

性能：

benchmark + report 负责。

因此：

M1 pytest：

只判断 correctness。

M1 throughput：

记录真实结果。

---

# 38\. Benchmark 使用真实 FastEnv

必须建立：

``` text
benchmarks/benchmark_m1_env.py
```

命令行至少支持：

``` bash
python benchmarks/benchmark_m1_env.py
```

以及参数：

``` text
--num-envs
--steps
--seed
--workers
```

没有参数时：

自动执行完整 M1 sweep。

---

# 39\. 单 worker 环境规模 sweep 固定

至少 benchmark：

``` text
256
1024
4096
8192
```

资源允许：

``` text
16384
```

如果 16384：

OOM / RAM 明显不足，

记录：

``` text
SKIPPED_RESOURCE_LIMIT
```

不要因此失败。

---

# 40\. 每个规模至少记录

必须记录：

``` text
num_envs
iterations
total_transitions
wall_time_s
transitions_per_s
boards_per_s
CPU utilization
peak/process RAM
```

还必须单独记录：

``` text
legal-mask evaluations/s
random spawn boards/s
exact-spawn parent boards/s
exact-spawn outcomes/s
```

---

# 41\. Benchmark 必须有 warmup

每个正式性能 measurement 前：

先 warmup。

默认：

``` text
5 warmup iterations
```

然后正式：

至少：

``` text
3 repeats
```

报告：

- median；
- min；
- max。

最终吞吐比较主要使用：

**median。**

---

# 42\. Worker scaling benchmark 固定

不要把多线程逻辑塞进 FastEnv 本身。

M1 worker scaling：

使用多个独立 worker，

每个 worker：

拥有自己的 Fast2048BatchEnv。

Windows：

必须兼容：

``` python
if __name__ == "__main__":
```

和：

`spawn`

multiprocessing mode。

至少测试：

``` text
1 worker
2 workers
4 workers
```

如果机器 CPU 不足：

最多测试到：

``` text
min(4, logical_cpu_count)
```

每 worker：

默认：

``` text
4096 env
```

如果资源不足：

降为：

``` text
1024 env / worker
```

并在报告写明。

---

# 43\. Scaling efficiency 固定公式

如果：

单 worker throughput：

``` text
T1
```

k worker：

``` text
Tk
```

则：

``` text
scaling_efficiency = Tk / (k * T1)
```

记录：

- 2-worker；
- 4-worker。

不要为了让数字好看改公式。

---

# 44\. Producer → Consumer Harness 固定

M1 必须模拟：

``` text
FastEnv
→ contiguous board batch
→ consumer
→ actions
→ FastEnv
```

M1 不允许创建真实 NN。

consumer 第一版固定为：

**轻量级 NumPy synthetic consumer。**

它只需要：

1. 读取完整 board batch；
2. 做一个简单批量 reduction / deterministic pseudo-score；
3. 根据 legal mask 选择合法 action；
4. 返回 action batch。

目的：

不是模拟棋力。

目的：

验证：

**batch 数据路径。**

---

# 45\. 如果 PyTorch 已安装

如果当前环境：

已经有 PyTorch，

允许 benchmark：

``` python
torch.from_numpy(...)
```

或等价 CPU tensor conversion。

只作为：

M2 前的数据搬运预演。

如果没有 PyTorch：

**不要为了 M1 单独安装完整 PyTorch。**

M2 再处理。

M1 不做 GPU forward。

---

# 46\. 禁止逐棋盘 consumer

Harness 中禁止：

``` python
for board in boards:
    consumer(board)
```

consumer：

必须整个 batch 一次处理。

---

# 47\. Profiling 固定

必须实现：

``` text
benchmarks/profile_m1_env.py
```

至少测量：

``` text
Environment stepping
Legal/action preparation
Spawn
Exact spawn enumeration
Batch assembly
Memory copy
Synchronization / worker overhead
Other
```

结果输出：

``` text
absolute time
percentage of measured wall-clock
```

百分比总和：

应约等于：

100%。

允许计时噪声造成小误差。

---

# 48\. Profiling 方法

优先使用：

``` python
time.perf_counter()
```

对明确阶段打点。

可以辅助使用：

``` text
cProfile
```

但：

最终 `M1_REPORT.md`

必须给出：

人类可读的 wall-clock breakdown。

不要只丢一份 profiler raw output。

---

# 49\. CPU / RAM 指标

如果环境已有：

`psutil`

使用它。

如果没有：

允许安装：

**psutil 作为 benchmark/dev dependency。**

psutil：

不得成为 FastEnv runtime correctness 所必需依赖。

FastEnv core：

只能依赖：

- Python stdlib；
- NumPy；
- 项目 M0。

---

# 50\. 性能判断不写死 C++ 门槛

M1 不允许 Agent 自己说：

“我觉得应该上 C++”

然后开始重写。

M1 结束必须给出：

``` text
Current bottleneck:
Measured evidence:
NumPy FastEnv saturation point:
Worker scaling:
Producer/consumer throughput:
C++ migration recommended: YES / NO / DEFER_TO_M2
```

推荐规则：

如果当前 M1 只能证明：

“环境在某处饱和”

但还没有 M2 网络实际吞吐，

优先：

``` text
DEFER_TO_M2
```

不要提前过度优化。

---

# 51\. M1 不要求 GPU 100%

M1 不使用正式网络。

因此：

**不要拿 GPU utilization 作为 M1 PASS 条件。**

M1 的任务：

生产高速、连续、可批量消费的 state。

真正：

“CPU 能不能喂饱 RTX 5060”

必须到：

M2 网络完成后，

进行端到端 benchmark 才能最终判断。

---

# 52\. M1 PASS 不使用武断绝对 steps/s

不要自行发明：

``` text
必须 100 万 steps/s
```

或者：

``` text
必须 500 万 steps/s
```

这种计划没有规定的硬门槛。

M1 PASS 取决于：

- correctness；
- vectorization；
- scalability benchmark；
- saturation point；
- profiler；
- batch pipeline；
- 已知瓶颈。

绝对吞吐：

记录。

但不擅自设定。

---

# 53\. M1 Benchmark 必须可复现

固定 benchmark seed。

至少：

``` text
seed = 20260918
```

完整 benchmark：

同一机器重复执行时，

吞吐允许波动，

但：

逻辑 workload 必须一致。

报告记录：

- OS；
- Python version；
- NumPy version；
- CPU model；
- physical cores；
- logical cores；
- RAM；
- Git commit hash。

---

# 54\. Benchmark action policy

不要用：

永远 LEFT

这种极端 workload。

benchmark full-env step 时：

固定采用：

**uniform random legal action。**

动作选择使用：

独立 benchmark RNG。

与 environment spawn RNG 分开。

如果某 env terminal：

显式：

``` python
reset_where(terminated)
```

继续 benchmark。

这样 workload：

持续包含真实移动、merge、spawn、terminal/reset。

---

# 55\. Exact Spawn benchmark workload

Exact Spawn benchmark：

不要只测：

全空棋盘。

至少分别构造代表：

``` text
2 empty cells
4 empty cells
8 empty cells
12 empty cells
```

的 batch。

记录：

``` text
parent boards/s
outcomes/s
```

因为 empty count 会显著影响：

输出规模。

---

# 56\. M1 测试必须保持固定 seed

随机 correctness test：

必须固定 seed。

至少在测试文件顶部明确写：

``` text
seed
sample count
exponent range
```

失败时：

必须输出：

- sample index；
- board；
- action；
- Fast result；
- Reference result。

这样错误可复现。

---

# 57\. Differential failure 行为固定

任何 mismatch：

不要：

扩大 tolerance。

不要：

跳过。

不要：

改 ReferenceEnv。

立即：

输出最小失败 case。

然后：

修 FastEnv。

除 probability floating-point 外：

board/reward/moved/legal/terminal：

要求：

**exact equality。**

---

# 58\. Test 命令固定

最终必须保证：

``` bash
python -m pytest
```

全部通过。

这意味着：

原来所有：

M0 tests

仍然必须通过。

同时：

M1 tests

全部通过。

禁止：

用 pytest config 把旧 M0 tests 排除掉。

---

# 59\. M1\_FAST\_ENV\_SPEC.md 必须记录

至少写清楚：

- M0 是 Golden Reference；
- FastEnv board layout；
- Action 编号；
- BatchMoveResult；
- legal\_mask\_batch；
- is\_terminal\_batch；
- spawn enumeration CSR layout；
- random batch spawn；
- BatchStepResult；
- reset；
- reset\_where；
- RNG 语义；
- reward dtype；
- high-tile 支持范围；
- Overflow 行为；
- benchmark 方法；
- differential test 方法；
- 哪些 RNG 行为要求分布一致而不是 bitwise stream 一致；
- M1 与 M2 的边界。

---

# 60\. M1\_REPORT.md 固定章节

必须生成：

``` text
# M1 Report

## A. Final Result
PASS / FAIL

## B. Files Added
新增文件和作用

## C. M0 Freeze Verification
M0 tag / commit
M0 文件是否被修改

## D. Correctness
pytest 总数
M0 tests
M1 tests
differential sample count
mismatch count

## E. Differential Coverage
ordinary boards
high-tile boards
legal
terminal
spawn enumeration
D4
illegal action
reset

## F. Performance Environment
OS
Python
NumPy
CPU
cores
RAM
Git commit

## G. Single-Worker Scalability
256
1024
4096
8192
16384 if possible

## H. Worker Scaling
1 / 2 / 4 workers
throughput
scaling efficiency

## I. Primitive Throughput
move
legal mask
random spawn
exact spawn enumeration

## J. Producer → Consumer Throughput
batch assembly
consumer
overall

## K. Wall-Clock Profile
Environment stepping
Legal/action
Spawn
Exact spawn
Batch assembly
Memory copy
Synchronization
Other

## L. Saturation Point
找到的吞吐饱和区域

## M. Bottleneck
当前最大瓶颈及证据

## N. C++ Decision
YES / NO / DEFER_TO_M2
原因

## O. Known Issues
没有则写 None

## P. M1 Exit Checklist
逐条 PASS / FAIL

## Q. Next Stage
M2 NOT STARTED
```

---

# 61\. M1 Exit Criteria 固定

以下全部满足：

M1 才能标记：

**PASS。**

必须逐条检查：

1. M0 原测试全部继续通过；
2. M0 frozen files 未被修改；
3. FastEnv 与 M0 ordinary differential 0 mismatch；
4. high-tile differential 0 mismatch；
5. legal mask differential 0 mismatch；
6. terminal differential 0 mismatch；
7. exact spawn enumeration differential 0 mismatch；
8. D4 equivariance 0 mismatch；
9. illegal action 语义正确；
10. illegal action 不消耗 FastEnv spawn RNG；
11. reset 正确；
12. reset\_where 正确；
13. BatchEnv 不使用 N 个 ReferenceEnv object；
14. hot path 不逐 board 调 ReferenceEnv；
15. board buffer `(N,16) uint8 contiguous`；
16. action batch 接口完成；
17. legal mask batch 完成；
18. terminal batch 完成；
19. random spawn batch 完成；
20. exact spawn enumeration batch 完成；
21. producer → consumer harness 完成；
22. 256 env benchmark 完成；
23. 1024 env benchmark 完成；
24. 4096 env benchmark 完成；
25. 8192 env benchmark 完成；
26. 16384 env 已测试或明确记录 resource limit；
27. worker scaling 已测试；
28. wall-clock profile 已输出；
29. throughput saturation region 已识别；
30. 当前主要 bottleneck 已识别；
31. C++ migration decision 已写明；
32. `docs/M1_FAST_ENV_SPEC.md` 已生成；
33. `M1_REPORT.md` 已生成；
34. `python -m pytest` 全绿；
35. M2 尚未开始。

任一 correctness 项失败：

**M1 = FAIL。**

Benchmark 某档因机器资源不足跳过：

可以记录：

`SKIPPED_RESOURCE_LIMIT`

不因此单独判 FAIL，

但：

256 / 1024 / 4096 / 8192

原则上必须完成。

如果 8192 也无法运行：

必须解释：

具体 RAM / 内存 / 实现原因，

M1 不得假装已经达到“几千环境并行”。

---

# 62\. M1 完成后的停止规则

全部 M1 Exit Criteria 通过以后：

立即停止。

不要：

- 创建 network.py；
- 实现 Transformer；
- 实现 MLP；
- 跑 M2；
- 加 Teacher；
- 加 N-tuple；
- 加 Expectimax；
- 加 Replay；
- 加 self-play；
- 用 tuple checkpoint。

最终回复用户只报告：

1. M1 PASS / FAIL；
2. 新增了什么；
3. pytest 结果；
4. differential test 数量 / mismatch；
5. benchmark 关键结果；
6. saturation point；
7. 当前 bottleneck；
8. C++ decision；
9. known issues；
10. M2 是否保持未开始。

---

# 63\. 实际执行顺序固定

严格按以下顺序执行：

### Step 1

检查 repository。

确认：

M0 文件存在。

运行当前：

``` bash
python -m pytest
```

确保开始 M1 前：

M0 全绿。

如果 M0 开始前就失败：

停止。

不要在 M1 偷偷修 M0。

### Step 2

记录：

Git commit / M0 tag / environment metadata。

### Step 3

实现：

`move_batch`

并先通过 movement differential。

### Step 4

实现：

`legal_mask_batch`

和：

`is_terminal_batch`

并通过 differential。

### Step 5

实现：

exact spawn enumeration batch。

通过 differential。

### Step 6

实现：

random spawn batch。

完成：

support + distribution tests。

### Step 7

实现：

`apply_spawn_batch`

用于 deterministic testing。

### Step 8

实现：

`Fast2048BatchEnv`

包括：

reset / reset\_where / step / score。

### Step 9

完成：

BatchEnv differential / illegal RNG / D4 / high tile tests。

### Step 10

运行：

完整 pytest。

必须全绿。

### Step 11

建立：

benchmark harness。

### Step 12

执行：

256 / 1024 / 4096 / 8192 / 16384-if-possible

规模 sweep。

### Step 13

执行：

1 / 2 / 4 worker scaling。

### Step 14

执行：

producer → consumer benchmark。

### Step 15

执行：

wall-clock profiling。

### Step 16

分析：

- saturation；
- bottleneck；
- C++ 是否应该现在做。

但是：

**不要自行开始 C++ rewrite。**

### Step 17

生成：

`docs/M1_FAST_ENV_SPEC.md`

和：

`M1_REPORT.md`

### Step 18

再次执行：

``` bash
python -m pytest
```

### Step 19

检查：

M0 frozen files 没有被修改。

### Step 20

如果全部 Exit Criteria PASS：

标记：

**M1 = PASS**

然后：

**立即停止。**

---

# 64\. 最后提醒

你没有权限：

“为了性能稍微改一下 M0。”

你没有权限：

“顺便把网络搭了。”

你没有权限：

“觉得 C++ 更快所以直接重写。”

你没有权限：

“因为 benchmark 不好看就减少测试规模。”

你没有权限：

“因为 Fast 和 Reference 有差异就放宽测试。”

本阶段的目标只有两个：

**第一，FastEnv 和 M0 一模一样地玩 2048。**

**第二，在不破坏第一条的前提下，把大量环境的状态生产做成真正的批量高速流水线。**

除此之外：

不要扩展项目范围。