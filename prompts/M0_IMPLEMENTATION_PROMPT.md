你现在负责执行《2048 单次神经网络 Agent 完整训练计划》的 **M0：游戏环境**。

你的任务不是重新设计项目，也不是提出替代方案，而是**严格完成 M0 的代码实现、自动测试和验收**。

本提示词是 M0 的具体施工规范。

---

# 0. 最高原则

你没有权限修改总计划的算法设计。

你没有权限提前开始 M1、M2 或任何后续阶段。

你没有权限因为“这样更快”“这样更优雅”“业界一般这么做”而自行改变这里规定的数据结构、动作编号、游戏语义、测试标准或 API。

优先级固定为：

1. 本提示词中的明确规定；  
2. 当前仓库中不与本提示词冲突的既有工程约定；  
3. 其余未规定的纯实现细节，以最简单、最容易验证、不影响后续接口的方式处理。

如果发现前两项之间存在真正无法同时满足的冲突：

**不要自行选择一种解释。**

记录：
- 冲突位置；
- 两种要求；
- 为什么无法同时满足；

然后停止对应冲突项。

除此以外，不要把正常实现细节重新抛回给用户决定。

---

# 1. M0 唯一目标

实现一个：

**正确、确定、容易验证的 2048 Reference Environment。**

这个环境以后将作为 M1 高速并行环境的：

**golden reference / correctness oracle。**

M0 不追求最高速度。

M0 首先追求：

**规则绝对正确。**

禁止为了性能使用难以检查的 bitboard 黑魔法、CUDA kernel、C++ SIMD、lookup-table 大规模优化等。

这些属于 M1。

---

# 2. 技术栈锁定

如果当前仓库已经明确是 Python/PyTorch 项目：

继续使用现有 Python 环境。

如果仓库还是空的或没有既定实现：

固定使用：

- Python 3.11+
- NumPy
- pytest

M0 不新增 Hypothesis 等非必要依赖。

Property test 使用：

**固定随机 seed + 大量随机案例**

实现。

不要为了 M0 新建复杂框架。

---

# 3. 棋盘内部表示

棋盘固定表示为：

`numpy.ndarray`

shape：

`(16,)`

dtype：

`np.uint8`

顺序固定为 row-major：

```text
0   1   2   3
4   5   6   7
8   9  10  11
12 13  14  15
```

每个格子保存：

**tile 的 2 的指数。**

固定：

```text
0 = empty
1 = 2
2 = 4
3 = 8
...
16 = 65536
17 = 131072
20 = 1048576
```

非常重要：

总计划中网络输入以后会把超过 `2^20` 的 tile 编码到 overflow bucket。

但是：

**M0 游戏环境内部绝对不能把 exponent > 20 截断成 21。**

环境保存真实指数。

例如：

```text
exp 20 + exp 20 -> exp 21
exp 21 + exp 21 -> exp 22
```

必须正常工作。

网络输入的 overflow 编码属于以后网络阶段。

不要污染游戏环境本身。

如果 exponent 已经等于 `255` 并尝试继续 merge：

必须显式抛出 `OverflowError`。

禁止 uint8 静默回绕成 0。

merge reward 使用 Python `int`。

禁止因为棋盘 dtype 是 uint8 而让 reward 溢出。

---

# 4. 动作编号完全锁死

动作固定为：

```python
UP    = 0
DOWN  = 1
LEFT  = 2
RIGHT = 3
```

必须提供 `IntEnum`：

```python
class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
```

任何地方都不得使用另一套动作顺序。

以后 Q Head 的 `[4]` 输出也按照：

```text
[UP, DOWN, LEFT, RIGHT]
```

解释。

---

# 5. 必须首先实现纯确定性移动核心

必须存在一个纯函数，语义等价于：

```python
move_without_spawn(board, action) -> MoveResult
```

`MoveResult` 至少包含：

```python
afterstate
reward
moved
```

其中：

### afterstate

执行移动和合并完成以后，

**还没有随机生成 2 / 4 的棋盘。**

### reward

此次 action 所产生的全部 merge reward。

### moved

棋盘是否实际发生变化。

这个函数：

- 不允许随机数；
- 不允许 spawn；
- 不允许修改输入 board；
- 相同输入必须永远得到完全相同结果。

---

# 6. 单行合并规则必须唯一

实现移动时，每一行/列按照移动方向执行：

1. 去掉空格；
2. 从移动方向一侧开始扫描；
3. 相邻且相等的两个 tile 合并；
4. 一个 tile 在一次 action 中最多参与一次 merge；
5. 合并完成后补零；
6. 不允许刚生成的新 tile 在同一步再次参与 merge。

典型回归测试必须包括：

```text
[2,2,2,2] LEFT
-> [4,4,0,0]
reward = 8
```

指数表示：

```text
[1,1,1,1]
-> [2,2,0,0]
reward = 8
```

还必须测试：

```text
[2,2,4,0] LEFT
-> [4,4,0,0]
reward = 4
```

禁止错误得到：

```text
[8,0,0,0]
```

还必须包含：

```text
[4,4,4,0] LEFT
-> [8,4,0,0]

[2,2,2,0] LEFT
-> [4,2,0,0]

[2,0,2,2] LEFT
-> [4,2,0,0]

[8,8,8,8] RIGHT
-> [0,0,16,16]
reward = 32
```

上述展示的是 tile 数值；

测试代码内部使用指数表示。

上下方向同样必须有独立测试。

---

# 7. Merge reward 精确定义

两个 exponent 为 `e` 的 tile 合并：

```text
e + e -> e + 1
```

此次 merge 的 reward：

```python
2 ** (e + 1)
```

一轮 action 有多个 merge：

reward 为所有 merge reward 之和。

例如：

```text
[2,2,2,2]
-> [4,4,0,0]
```

发生两个：

```text
2 + 2 -> 4
```

所以：

```text
reward = 4 + 4 = 8
```

环境累计 score：

只累计 merge reward。

spawn 不产生 reward。

---

# 8. Legal action 精确定义

必须实现：

```python
legal_mask(board) -> np.ndarray
```

结果：

```text
shape = (4,)
dtype = bool
```

顺序：

```text
UP DOWN LEFT RIGHT
```

某动作合法，当且仅当：

```python
move_without_spawn(board, action).moved == True
```

禁止另外维护一套可能和真实移动逻辑不一致的合法性判断。

M0 是 reference implementation。

宁可慢，也不要复制另一套复杂判断造成语义分叉。

---

# 9. Terminal 精确定义

必须实现：

```python
is_terminal(board) -> bool
```

定义固定：

```text
四个动作全部 illegal
```

即：

```python
not legal_mask(board).any()
```

不要用“棋盘已满”直接代替 terminal。

棋盘已满但仍有相邻相同 tile：

仍然不是 terminal。

---

# 10. Spawn 规则完全锁死

只有：

**合法动作完成以后**

才允许 spawn。

spawn 固定规则：

假设 afterstate 有 `n` 个空格：

1. 从 n 个空格中均匀选择一个；
2. 该位置生成：
   - exponent 1，也就是 tile 2：概率 0.9
   - exponent 2，也就是 tile 4：概率 0.1

因此每个空位置对应：

```text
P(tile 2 at this cell) = 0.9 / n
P(tile 4 at this cell) = 0.1 / n
```

必须实现确定性的：

```python
enumerate_spawns(afterstate)
```

返回所有可能 spawn outcome。

每项至少包含：

```text
state
probability
spawn_index
spawn_exponent
```

概率必须精确符合上面的规则。

概率和必须在浮点容差内等于：

```text
1.0
```

一个 4×4 棋盘最多：

```text
16 × 2 = 32
```

种 spawn outcome。

---

# 11. 随机 spawn

还必须实现：

```python
spawn_random(afterstate, rng)
```

随机数源固定使用：

```python
numpy.random.Generator
```

默认构造方式：

```python
np.random.Generator(np.random.PCG64(seed))
```

不要调用全局：

```python
np.random.random(...)
```

不要偷偷使用：

```python
random
```

环境随机性必须全部由明确传入/保存的 `Generator` 控制。

这样 checkpoint 和复现实验才有可靠基础。

---

# 12. Illegal action 规则

如果玩家调用非法动作：

棋盘：

**完全不变。**

reward：

```text
0
```

不得：

- spawn；
- 修改 score；
- 消耗一次 spawn 随机数；
- 修改棋盘；
- 假装执行成功。

必须在测试中验证：

相同 RNG state 下，

先执行一次 illegal action 再执行合法 action，

和直接执行该合法 action，

spawn 结果必须一致。

这用来证明 illegal action 没有偷偷消耗 RNG。

---

# 13. step() API

实现一个 reference environment class。

名称可固定为：

```python
Reference2048Env
```

至少提供：

```python
reset(seed=None)
step(action)
board
score
legal_mask()
is_terminal()
```

`step(action)` 的流程必须严格为：

```text
当前正式棋盘 s
↓
move_without_spawn
↓
如果 illegal：
    board 不变
    reward = 0
    不 spawn
↓
如果 legal：
    得到 afterstate x
    获得 merge reward r
    spawn 一次
    得到下一正式棋盘 s'
    score += r
↓
在 s' 上计算 terminal
```

建议返回结构化 `StepResult`，至少包含：

```text
state
afterstate
reward
legal
terminated
spawn_index
spawn_exponent
```

如果 illegal：

```text
spawn_index = None
spawn_exponent = None
```

不要把 afterstate 和 spawn 后 state 混为一谈。

---

# 14. reset() 固定规则

标准新游戏固定：

1. 创建全空 4×4 board；
2. score = 0；
3. 按正常 spawn 规则生成第一个 tile；
4. 再按正常 spawn 规则生成第二个 tile。

所以 reset 后：

必须恰好有两个非空格。

两个 spawn 都服从：

```text
2: 90%
4: 10%
```

第二次 spawn 只能选择剩余空格。

---

# 15. D4 对称必须在 M0 完整实现

必须实现八种棋盘变换。

固定编号：

```text
0 = Identity
1 = Rotate 90° clockwise
2 = Rotate 180°
3 = Rotate 270° clockwise
4 = Mirror Left-Right
5 = Rotate90(Mirror Left-Right(board))
6 = Rotate180(Mirror Left-Right(board))
7 = Rotate270(Mirror Left-Right(board))
```

这里：

`Mirror Left-Right`

表示左右翻转，也就是：

```text
第 0 列 <-> 第 3 列
第 1 列 <-> 第 2 列
```

必须实现：

```python
transform_board(board, transform_id)
transform_action(action, transform_id)
inverse_transform_id(transform_id)
```

动作转换不能靠随手写八张容易出错的表然后不验证。

必须按照方向向量的几何变换实现，或者使用经过完整测试的固定映射。

坐标约定：

```text
row 向下增加
col 向右增加
```

方向向量：

```text
UP    = (-1, 0)
DOWN  = ( 1, 0)
LEFT  = ( 0,-1)
RIGHT = ( 0, 1)
```

---

# 16. D4 必须满足的核心等价关系

对任意：

```text
board s
action a
transform T
```

必须验证：

```text
T(move(s,a).afterstate)
==
move(T(s), T(a)).afterstate
```

并且：

```text
reward 完全相等
```

以及：

```text
legal 性完全相等
```

还必须验证：

```text
terminal(T(s)) == terminal(s)
```

和：

```text
inverse(T)(T(s)) == s
```

八种 D4 都必须测试。

---

# 17. 65536+ 必须作为正式回归测试

至少明确测试：

```text
65536 + 65536 -> 131072
```

指数：

```text
16 + 16 -> 17
```

reward：

```text
131072
```

还必须测试：

```text
2^20 + 2^20 -> 2^21
```

也就是：

```text
20 + 20 -> exponent 21
```

以及继续：

```text
21 + 21 -> exponent 22
```

这项测试专门防止有人错误地把环境内部指数提前 clamp 到网络的 overflow bucket。

---

# 18. 输入不得被偷偷修改

所有纯函数：

```text
move_without_spawn
legal_mask
is_terminal
transform_board
enumerate_spawns
```

必须保证：

**不修改调用者传入的 board。**

测试中必须保存输入副本并进行：

```python
np.array_equal(before, after)
```

验证。

---

# 19. 必须建立固定 regression tests

不要只写随机测试。

需要一组人工明确答案的 regression vectors。

至少覆盖：

- 四个方向移动；
- 单次 merge；
- 双 merge；
- 三相同 tile；
- 四相同 tile；
- 中间有空格；
- 不允许连锁二次 merge；
- 无 merge 纯移动；
- illegal move；
- 满棋盘但可 merge；
- 满棋盘且 terminal；
- 高 exponent；
- 多 merge reward；
- spawn；
- reset；
- legal mask；
- terminal；
- D4。

这些测试是以后 M1 的长期回归基准。

---

# 20. Property / randomized tests

使用固定 seed。

至少生成：

```text
10,000 个随机 board
```

随机 exponent 建议覆盖：

```text
0～17
```

对所有 board 和动作检查核心 invariant。

必须检查：

### Property A

移动前后、spawn 以前：

```text
所有 tile 数值之和完全不变
```

tile 数值：

```python
0 exponent -> 0
e > 0 -> 2**e
```

### Property B

如果：

```text
moved == False
```

则：

```text
afterstate == original
reward == 0
```

### Property C

如果：

```text
legal_mask[a] == True
```

则：

```text
move_without_spawn(...).moved == True
```

反之亦然。

### Property D

spawn 后：

棋盘 tile 数值总和只能增加：

```text
2 或 4
```

### Property E

一次 spawn：

恰好只改变一个原先为空的位置。

### Property F

D4 movement equivariance 全部成立。

### Property G

terminal：

与四个动作全部 illegal 完全一致。

---

# 21. 随机 spawn 测试

由于 `enumerate_spawns()` 是精确枚举：

必须直接验证每个 outcome 的理论概率。

此外对 `spawn_random()` 做固定 seed 的统计 sanity test。

例如：

累计至少：

```text
20,000 次
```

spawn。

要求生成 4 的比例大致位于：

```text
0.08 ～ 0.12
```

这里只是 sampler sanity check。

真正的规则正确性以：

`enumerate_spawns()`

的精确概率测试为主。

---

# 22. Reference 与生产环境职责必须分离

M0 代码中明确标注：

```text
Reference2048Env = correctness reference
```

以后 M1 可以写：

```text
Fast2048Env
Vectorized2048Env
C++ backend
CUDA backend
```

但 M1 必须拿结果和 M0 reference 对照。

因此 M0：

不要为了未来速度把代码写得极其晦涩。

优先：

- 简单；
- 可读；
- 容易人工审计；
- 容易写 oracle tests。

---

# 23. 推荐文件结构固定

如果现有仓库没有自己的明确结构，使用：

```text
src/
  game2048/
    __init__.py
    reference_env.py
    symmetry.py

tests/
  test_m0_moves.py
  test_m0_spawn.py
  test_m0_legal_terminal.py
  test_m0_d4.py
  test_m0_properties.py
  test_m0_high_tiles.py
  test_m0_env_api.py

docs/
  M0_ENVIRONMENT_SPEC.md

reports/m0/M0_REPORT.md
```

不要在 M0 创建：

```text
network.py
replay.py
teacher.py
expectimax.py
trainer.py
self_play.py
```

这些不是 M0。

---

# 24. M0_ENVIRONMENT_SPEC.md 必须记录

文档必须明确写出：

- board 表示；
- exponent 语义；
- row-major 索引；
- Action 编号；
- move 语义；
- afterstate 定义；
- reward 定义；
- spawn 规则；
- illegal action 规则；
- legal mask 定义；
- terminal 定义；
- reset 定义；
- RNG 方案；
- D4 编号和含义；
- 高 tile 不 clamp 的规定。

以后 M1 不允许重新猜这些定义。

---

# 25. reports/m0/M0_REPORT.md

完成以后生成：

```text
reports/m0/M0_REPORT.md
```

里面必须包含：

### A. 实现文件

实际新增/修改了哪些文件。

### B. Rule tests

运行了什么测试。

### C. Property tests

随机测试数量和 seed。

### D. D4 tests

验证了哪些 invariant。

### E. High-tile tests

65536 / 131072 / `2^20+` 是否通过。

### F. Test command

给出可以重新执行全部 M0 tests 的确切命令。

### G. Test result

必须记录：

```text
passed / failed
```

以及 pytest 最终摘要。

### H. 已知问题

如果没有：

明确写：

```text
None
```

不要为了显得谨慎凭空编造问题。

---

# 26. 禁止事项

M0 严禁：

- 开始神经网络；
- 实现 Transformer；
- 实现 MLP；
- 实现 Teacher；
- 实现 Expectimax；
- 实现 Replay Buffer；
- 实现 Self-play；
- 实现 Double-Q；
- 实现 Target Network；
- 实现 Champion；
- 实现高位 Restart Pool；
- 实现 Search Correction；
- 实现 CUDA 环境；
- 为性能牺牲 reference clarity；
- 修改总计划；
- 自行改变动作顺序；
- 自行改变 spawn 概率；
- 自行改变 reward；
- 自行改变 terminal 定义；
- 自行改变 board 编码；
- 把 exponent >20 提前 clamp；
- 使用特殊高 tile 奖励。

即使你认为后面的东西“顺便写了更方便”，

**也不允许。**

---

# 27. 验收门槛

M0 只有同时满足以下条件才算完成：

```text
[ ] 四方向规则测试全部通过
[ ] 合并顺序测试全部通过
[ ] 单 tile 一步只能 merge 一次
[ ] merge reward 正确
[ ] illegal action 不改变 board
[ ] illegal action reward = 0
[ ] illegal action 不 spawn
[ ] illegal action 不消耗 spawn RNG
[ ] legal action 才 spawn
[ ] spawn 位置均匀
[ ] tile 2 概率 90%
[ ] tile 4 概率 10%
[ ] legal_mask 正确
[ ] terminal 正确
[ ] reset 正确
[ ] afterstate 语义正确
[ ] D4 八变换正确
[ ] D4 action mapping 正确
[ ] D4 movement equivariance 正确
[ ] Property tests 全部通过
[ ] 10,000+ 随机 board 测试通过
[ ] 65536 正常
[ ] 131072 正常
[ ] exponent 20 -> 21 正常
[ ] exponent 21 -> 22 正常
[ ] 输入 board 不被纯函数修改
[ ] 全部 pytest 通过
[ ] M0_ENVIRONMENT_SPEC.md 已生成
[ ] reports/m0/M0_REPORT.md 已生成
```

只要其中任何一项失败：

**M0 = FAILED。**

不得开始 M1。

---

# 28. 实际执行方式

不要只回复我：

“我建议这样实现”。

不要只给伪代码。

不要只生成计划。

你需要：

1. 检查当前仓库；
2. 按上面的规范实际创建/修改代码；
3. 写完整测试；
4. 实际运行测试；
5. 如果测试失败，定位并修复；
6. 重复运行直到全部通过，或者出现无法继续的真实外部阻塞；
7. 生成 `M0_ENVIRONMENT_SPEC.md`；
8. 生成 `reports/m0/M0_REPORT.md`；
9. 最后向我汇报实际测试结果。

不要因为第一次 pytest 通过就跳过其他验收检查。

---

# 29. 最终停止位置

当 M0 所有验收门槛通过以后：

**立即停止。**

不要自行开始 M1。

最终回复只需要明确告诉我：

- M0 是否 PASS；
- 实现了什么；
- 测试数量/结果；
- 是否存在已知问题；
- 下一阶段是否已经被保持为未开始状态。

M1 必须等待新的明确指令。

---

现在开始执行 M0。