# M0 环境规格说明（冻结）

> 本文件是 **M0 Reference Environment 的权威规格**。
> 后续 M1（Fast / Vectorized / C++ / CUDA 后端）**不允许重新猜测或重新定义**这里的任何语义。
> M1 的实现必须与本文件（以及 `src/game2048/`）逐条对齐，并以 M0 作为 correctness oracle 对照。

---

## 0. 模块与文件布局

```text
src/game2048/__init__.py         # 公共导出
src/game2048/reference_env.py    # 棋盘/动作/移动核心/spawn/Reference2048Env
src/game2048/symmetry.py         # D4 八种变换
conftest.py                      # 仅用于让 src/ 布局可被 pytest 直接导入
tests/test_m0_*.py               # 7 个 M0 测试文件
```

`Reference2048Env` = **correctness reference**。
本层刻意牺牲速度，换取「简单、可读、可人工审计、易于写 oracle 测试」。

M0 明确**不包含**：神经网络、Transformer、MLP、Teacher、Expectimax、Replay Buffer、Self-play、Double-Q、Target Network、Champion、高位 Restart Pool、Search Correction、CUDA 环境。

---

## 1. 棋盘表示（board representation）

| 项目 | 固定值 |
| --- | --- |
| 类型 | `numpy.ndarray` |
| shape | `(16,)` |
| dtype | `numpy.uint8` |
| 顺序 | row-major |

row-major 展开顺序固定为：

```text
 0   1   2   3
 4   5   6   7
 8   9  10  11
12  13  14  15
```

即 `flat_index = row * 4 + col`，`row` 向下增加，`col` 向右增加。

所有纯函数（`move_without_spawn` / `legal_mask` / `is_terminal` / `transform_board` / `enumerate_spawns`）
**绝不修改调用者传入的 board**；返回值总是新数组，不与输入共享内存。

## 2. exponent 语义

每个格子保存 **tile 的 2 的指数**：

```text
0  = empty
1  = 2
2  = 4
3  = 8
...
16 = 65536
17 = 131072
20 = 1048576
```

`exponent e > 0` 对应的 tile 数值为 `2 ** e`；`e == 0` 表示空，数值为 `0`。

### 2.1 高位 tile **绝不 clamp**（强制规定）

* 环境内部**保存真实指数**，`exp 20 + exp 20 -> exp 21`、`exp 21 + exp 21 -> exp 22` 必须正常工作。
* 「把大于 `2^20` 的 tile 编码进网络 overflow bucket」是**以后网络输入阶段**的事情，
  **不得污染游戏环境**。
* 若某个 tile 的指数已经等于 `255`（`uint8` 上限）并再次参与 merge，
  必须显式抛出 `OverflowError`，**禁止 uint8 静默回绕成 0**。
* merge reward 与累计 score 一律使用 Python `int`，不因 board dtype 是 `uint8` 而溢出。

## 3. 动作编号（Action numbering，冻结）

```python
class Action(IntEnum):
    UP    = 0
    DOWN  = 1
    LEFT  = 2
    RIGHT = 3
```

* 任何地方都不得使用另一套动作顺序。
* Q Head 的 `[4]` 输出按 `[UP, DOWN, LEFT, RIGHT]` 解释。
* `legal_mask` 的第 `i` 位对应 `Action(i)`。

方向向量（`row` 向下增加，`col` 向右增加）：

```text
UP    = (-1, 0)
DOWN  = ( 1, 0)
LEFT  = ( 0,-1)
RIGHT = ( 0, 1)
```

## 4. move 语义（单行合并规则，唯一）

`move_without_spawn(board, action) -> MoveResult` 是**纯函数**：
无随机数、无 spawn、不修改输入、相同输入永远得到完全相同结果。

对每一行/列，按移动方向执行：

1. 去掉空格；
2. 从**移动方向一侧**开始扫描；
3. 相邻且相等的两个 tile 合并；
4. 一个 tile 在一次 action 中**最多参与一次 merge**；
5. 合并完成后补零；
6. 刚生成的新 tile 在同一步**不得再次参与 merge**。

实现上等价于「先紧凑到目的侧 → 从目的侧成对合并 → 顺序写出 → 补零」。
`action` 决定 4 条 line 的取向：line 的第 `0` 位始终是**目的格**。

### 4.1 必须成立的回归向量（tile 数值 / 指数）

| 输入 | 方向 | 输出 | reward |
| --- | --- | --- | --- |
| `[2,2,2,2]` = `[1,1,1,1]` | LEFT | `[4,4,0,0]` = `[2,2,0,0]` | `8` |
| `[2,2,4,0]` = `[1,1,2,0]` | LEFT | `[4,4,0,0]` = `[2,2,0,0]`（**禁止** `[8,0,0,0]`） | `4` |
| `[4,4,4,0]` = `[2,2,2,0]` | LEFT | `[8,4,0,0]` = `[3,2,0,0]` | `8` |
| `[2,2,2,0]` = `[1,1,1,0]` | LEFT | `[4,2,0,0]` = `[2,1,0,0]` | `4` |
| `[2,0,2,2]` = `[1,0,1,1]` | LEFT | `[4,2,0,0]` = `[2,1,0,0]` | `4` |
| `[8,8,8,8]` = `[3,3,3,3]` | RIGHT | `[0,0,16,16]` = `[0,0,4,4]` | `32` |

上下方向有独立测试（`tests/test_m0_moves.py`）。

## 5. afterstate 定义

**afterstate = 执行移动与合并之后、还没有随机生成 2 / 4 之前的棋盘。**

`afterstate` 与 spawn 之后的正式 state 是**两个不同的东西**，任何接口都不得混用：

* `MoveResult.afterstate`：移动核心的输出，**未 spawn**；
* `StepResult.afterstate`：本步移动后的 afterstate，**未 spawn**；
* `StepResult.state`：spawn 之后的正式棋盘 `s'`。

## 6. reward 定义

两个指数为 `e` 的 tile 合并：

```text
e + e -> e + 1
reward = 2 ** (e + 1)          # Python int
```

一轮 action 内所有 merge 的 reward **求和**。

* `[2,2,2,2] -> [4,4,0,0]`：两次 `2+2 -> 4`，`reward = 4 + 4 = 8`；
* `exp 16 + exp 16 -> exp 17`：`reward = 131072`；
* `exp 20 + exp 20 -> exp 21`：`reward = 2**21`。

环境累计 `score` **只累计 merge reward**；**spawn 不产生 reward**。

## 7. spawn 规则（完全冻结）

只有**合法动作完成以后**才允许 spawn。

设 afterstate 有 `n` 个空格：

1. 从 `n` 个空格中**均匀**选择一个位置；
2. 在该位置生成：
   * exponent `1`（tile 2），概率 `0.9`；
   * exponent `2`（tile 4），概率 `0.1`。

因此每个空位置：

```text
P(tile 2 at this cell) = 0.9 / n
P(tile 4 at this cell) = 0.1 / n
```

`enumerate_spawns(afterstate)` 返回全部 spawn outcome（最多 `16 × 2 = 32` 个），
每项包含 `state` / `probability` / `spawn_index` / `spawn_exponent`，
枚举顺序为「空格 flat index 升序，然后 exponent 1、2」。
概率和在浮点容差内等于 `1.0`。满盘返回空列表。

> 事实：任何**合法**动作之后的 afterstate 一定至少有一个空格
> （空盘情形：tile 数不增加；满盘情形：合法动作必然发生 merge，tile 数减少）。
> 因此合法 step 一定能 spawn。

## 8. RNG 方案

* 随机源固定为 `numpy.random.Generator`。
* 默认构造方式固定为：

```python
np.random.Generator(np.random.PCG64(seed))
```

* **禁止**使用全局 `np.random.random(...)` 等全局状态，**禁止**使用标准库 `random`。
* 环境随机性全部由明确传入 / 保存的 `Generator` 控制，便于 checkpoint 与复现实验。
* `spawn_random(afterstate, rng)` 每次调用**恰好消耗两次抽样**，顺序固定：
  1. `position = int(rng.integers(0, n))` —— 在 `n` 个空格中均匀选位；
  2. `rng.random() < 0.9` —— 决定 exponent 1 / 2。

  M1 若要与 M0 逐步对齐，必须复现这个消耗顺序。

`Reference2048Env(seed=None, rng=None)`：

* 传 `rng` 则直接使用该 `Generator`（与 `seed` 互斥，同时传会抛 `ValueError`）；
* 传 `seed` 则 `Generator(PCG64(seed))`；
* 都不传则由系统熵构造 `Generator(PCG64(None))`。

## 9. illegal action 规则

若调用非法动作：

* 棋盘**完全不变**；
* `reward = 0`；
* **不 spawn**；
* **不修改 score**；
* **不消耗 spawn 随机数**（RNG state 逐位不变）；
* 不得假装执行成功（`StepResult.legal is False`，`spawn_index = spawn_exponent = None`）。

验收测试要求：在相同初始 RNG state 下，
「先执行 illegal action 再执行合法 action」与「直接执行该合法 action」
必须得到**完全相同的棋盘与完全相同的 RNG state**。

## 10. legal mask 定义

```python
legal_mask(board) -> np.ndarray    # shape (4,), dtype bool, 顺序 UP DOWN LEFT RIGHT
```

某动作合法 **当且仅当**：

```python
move_without_spawn(board, action).moved == True
```

**禁止**另外维护一套可能与真实移动逻辑不一致的合法性判断。
M0 是 reference implementation：宁可慢，也不要造成语义分叉。

## 11. terminal 定义

```python
is_terminal(board) -> bool
```

定义固定为：

```text
四个动作全部 illegal
等价于 not legal_mask(board).any()
```

* **不得**用「棋盘已满」直接代替 terminal；
* 满盘但仍有相邻相同 tile ⇒ **不是** terminal。

> 由该定义推出的两个退化事实（不可达 / 非 bug）：
> * 全空棋盘没有合法动作，因此按定义是 terminal；
>   实际对局不可达（`reset` 必定生成 2 个 tile，且 tile 数不会降到 0）。
> * 只有 1 个 tile 时**永远不是** terminal（tile 总能向至少一个方向滑动）。

## 12. reset 定义

标准新游戏固定为：

1. 创建全空 4×4 board；
2. `score = 0`；
3. 按第 7 节正常 spawn 规则生成**第一个** tile；
4. 再按第 7 节正常 spawn 规则生成**第二个** tile（只能选剩余空格）。

因此 `reset()` 之后**恰好有两个非空格**，两个 spawn 都服从 `2: 90% / 4: 10%`。

`reset(seed=None)`：

* `seed is not None` ⇒ 用 `Generator(PCG64(seed))` **重新播种**，结果只由 `seed` 决定；
* `seed is None` ⇒ **延续当前随机流**（gymnasium 约定）。

`Reference2048Env.__init__` 只播种并置空棋盘，**不**自动 spawn；
`Reference2048Env(seed=s).reset()` 与 `Reference2048Env().reset(seed=s)` 得到完全相同的初始棋盘。

## 13. step 流程（冻结）

```text
当前正式棋盘 s
↓
move_without_spawn(s, action)
↓
如果 illegal：
    board 不变；reward = 0；不 spawn；不消耗 RNG
↓
如果 legal：
    得到 afterstate x
    获得 merge reward r
    spawn 一次 → 下一正式棋盘 s'
    score += r
↓
在 s' 上计算 terminal
```

返回值 `StepResult` 至少包含：

```text
state, afterstate, reward, legal, terminated, spawn_index, spawn_exponent
```

illegal 时 `spawn_index = None`、`spawn_exponent = None`。

## 14. 环境 API

```python
class Reference2048Env:
    def __init__(self, seed=None, rng=None) -> None
    def reset(self, seed=None) -> np.ndarray      # 返回新棋盘（副本）
    def step(self, action) -> StepResult
    @property board -> np.ndarray                 # 副本，逐次独立
    @property score -> int                        # Python int
    @property rng -> np.random.Generator
    def legal_mask(self) -> np.ndarray            # (4,) bool
    def is_terminal(self) -> bool
```

`step` / `move_without_spawn` 的 `action` 接受 `Action` 或 `int`；
取值不在 `0..3` 时抛 `ValueError`（注意：非法**取值**与非法**动作**是两回事）。

## 15. D4 对称（八种变换，冻结）

| id | 含义 |
| --- | --- |
| 0 | Identity |
| 1 | Rotate 90° clockwise |
| 2 | Rotate 180° |
| 3 | Rotate 270° clockwise |
| 4 | Mirror Left-Right（第 0 列 ↔ 第 3 列，第 1 列 ↔ 第 2 列） |
| 5 | Rotate90(Mirror Left-Right(board)) |
| 6 | Rotate180(Mirror Left-Right(board)) |
| 7 | Rotate270(Mirror Left-Right(board)) |

接口：

```python
transform_board(board, transform_id) -> np.ndarray   # 新的 (16,) uint8
transform_action(action, transform_id) -> Action
inverse_transform_id(transform_id) -> int            # (0, 3, 2, 1, 4, 5, 6, 7)
```

`transform_action` **由方向向量的几何变换导出**（不是手写八张表）：
把方向向量经与该 transform 相同的线性映射送出。
旋转 90° clockwise 把格子 `(r, c)` 映到 `(c, -r)`（平移部分与方向无关），
因此 `UP -> RIGHT`。

几何导出的动作表（同时被回归测试与等变性测试双重约束）：

| id | UP | DOWN | LEFT | RIGHT |
| --- | --- | --- | --- | --- |
| 0 | UP | DOWN | LEFT | RIGHT |
| 1 | RIGHT | LEFT | UP | DOWN |
| 2 | DOWN | UP | RIGHT | LEFT |
| 3 | LEFT | RIGHT | DOWN | UP |
| 4 | UP | DOWN | RIGHT | LEFT |
| 5 | RIGHT | LEFT | DOWN | UP |
| 6 | DOWN | UP | LEFT | RIGHT |
| 7 | LEFT | RIGHT | UP | DOWN |

对任意 board `s`、action `a`、transform `T`，必须成立：

```text
T(move(s, a).afterstate) == move(T(s), T(a)).afterstate
reward 完全相等
legal / moved 完全相等
terminal(T(s)) == terminal(s)
inverse(T)(T(s)) == s
```

八种 D4 全部测试（见 `tests/test_m0_d4.py`、`tests/test_m0_properties.py`）。

## 16. 运行测试

```bash
python -m pytest
```

或只跑 M0：

```bash
python -m pytest tests/test_m0_moves.py tests/test_m0_spawn.py tests/test_m0_legal_terminal.py tests/test_m0_d4.py tests/test_m0_properties.py tests/test_m0_high_tiles.py tests/test_m0_env_api.py
```

---

## 17. 冻结清单（M1 必须逐条复用）

1. board = `ndarray (16,) uint8`，row-major，cell 存 exponent；
2. `UP=0, DOWN=1, LEFT=2, RIGHT=3`；
3. 移动 = 去空格 → 从移动侧成对合并 → 一个 tile 一步最多合并一次 → 补零；
4. afterstate = 移动后、spawn 前；
5. reward = `Σ 2 ** (e + 1)`，Python int；
6. spawn = 空格均匀选位 + `0.9 / 0.1` 出 2 / 4，每次消耗 2 次 RNG 抽样；
7. illegal action 完全无副作用（含不消耗 RNG）；
8. `legal_mask[a] := move_without_spawn(s, a).moved`；
9. `terminal := not legal_mask(s).any()`；
10. `reset` = 2 次标准 spawn，`score = 0`；
11. `exp > 20` **绝不 clamp**，`exp 255 + 255` 抛 `OverflowError`；
12. D4 编号 0..7 与动作映射表如第 15 节。
