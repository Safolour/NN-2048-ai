你现在负责本项目的：

# M1 性能修复与最终验收

注意：

**这不是重新执行 M1。**

当前 M1 已经完成：

- FastEnv API；
- batch environment；
- differential tests；
- spawn enumeration；
- RNG；
- D4；
- benchmark harness；
- profiler；
- 文档；
- M1_REPORT。

当前正确性已经基本确认：

- M0 tests：261 passed；
- M1 tests：122 passed；
- 总计：383 passed；
- ordinary differential：40,000 comparisons，0 mismatch；
- high-tile differential：8,000 comparisons，0 mismatch；
- legal / terminal：0 mismatch；
- exact spawn enumeration：0 mismatch；
- D4：64,000 comparisons，0 mismatch。

这些成果全部保留。

---

# 0. 本任务为什么存在

当前 M1 报告暴露了一个没有满足原施工规范的硬问题：

`_move_batch`

虽然接口是 batch，

但内部仍存在：

**Python-level per-board loop。**

当前 profiling 已经明确：

```text
Environment stepping    60.28%
Legal/action            38.73%
合计                    99.01%
```

cProfile 热点：

大量：

```text
_merge_line_cached
_shift_line
_line_changed
```

标量 Python 调用。

当前 benchmark：

```text
256 env      ≈ 25.5k transitions/s
1024 env     ≈ 26.1k transitions/s
4096 env     ≈ 14.6k transitions/s
8192 env     ≈ 14.9k transitions/s
16384 env    ≈ 14.8k transitions/s
```

说明：

FastEnv 虽然正确，

但没有真正完成：

**跨 board vectorized movement。**

因此当前状态固定定义为：

```text
M1 correctness baseline = PASS
M1 performance implementation = INCOMPLETE
M1 overall = NOT YET FINAL PASS
```

本任务唯一目标：

> **在不破坏现有 M1 正确性和 API 的前提下，消灭 movement hot path 的 N-board Python 主循环，实现真正跨 board NumPy vectorization，然后重新进行完整 correctness + performance 验收。**

---

# 1. 绝对禁止重做项目

必须在：

**当前已经包含 M1 实现的工作区**

继续修改。

禁止：

```bash
git reset --hard
```

禁止：

checkout 回：

```text
m0-reference-pass
```

禁止：

删除现有 M1 再重写。

禁止：

rebase / force push / 改写历史。

开始前先运行：

```bash
git status
git diff --stat
git rev-parse HEAD
git rev-parse m0-reference-pass
```

记录当前状态。

如果现有 M1 文件还没有 commit：

**保持它们，不得丢失。**

---

# 2. M0 继续永久冻结

以下 M0 文件：

仍然绝对禁止修改：

```text
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

M0 仍然是：

**Golden Reference / Correctness Oracle。**

FastEnv 和 M0 不一致时：

默认：

**FastEnv 错。**

禁止为了性能修改 M0。

---

# 3. 本次允许修改的范围

主要允许修改：

```text
src/game2048/fast_env.py
```

允许按需要修改：

```text
tests/test_m1_batch_move.py
tests/test_m1_differential.py
tests/test_m1_batch_env.py
tests/test_m1_high_tiles.py
tests/test_m1_spawn.py
tests/_m1_helpers.py

benchmarks/benchmark_m1_env.py
benchmarks/profile_m1_env.py
benchmarks/_utils.py

docs/M1_FAST_ENV_SPEC.md
M1_REPORT.md
```

只有为了：

- 验证真正 vectorization；
- benchmark 新实现；
- 更新报告；

才允许修改这些文件。

禁止借这个任务：

重构项目其他模块。

---

# 4. 本次仍然禁止

禁止实现：

- Transformer；
- Residual MLP；
- Q/V/A Head；
- M2；
- GPU network benchmark；
- Trainer；
- Replay；
- Self-play RL；
- Teacher；
- N-tuple；
- Expectimax；
- Search；
- C++ extension；
- pybind11；
- Cython；
- Numba；
- Triton；
- CUDA kernel；
- Rust。

尤其：

**禁止因为 NumPy 当前慢，就自行改写成 C++。**

本次任务首先必须完成：

**真正的 NumPy batch vectorization。**

C++ 是否需要：

留到 M2 端到端测试后决定。

---

# 5. 不得使用 tuple checkpoint

用户已有：

**tuple 8×6 checkpoint。**

本任务：

完全不用。

禁止：

- 加载；
- 转换；
- benchmark；
- 接口适配；
- Teacher 集成。

这不是本阶段内容。

---

# 6. 现有公共 API 必须保持兼容

除非存在无法实现的真实 bug，

以下现有 API 的：

- 名称；
- 参数语义；
- 返回 shape；
- dtype；
- 规则；

不得改变。

包括：

```python
move_batch(...)
legal_mask_batch(...)
is_terminal_batch(...)
enumerate_spawns_batch(...)
spawn_random_batch(...)
apply_spawn_batch(...)
Fast2048BatchEnv
```

以及已有：

```text
BatchMoveResult
BatchSpawnEnumeration
BatchSpawnResult
BatchStepResult
```

禁止为了写 vectorization：

破坏现有 API。

---

# 7. 本次必须解决的唯一核心问题

当前：

`move_batch`

内部不能再存在：

```python
for board in boards:
    ...
```

或者任何等价的：

```text
N boards
→ N 次 Python scalar merge
```

正式 hot path 必须：

**跨 board NumPy vectorized。**

允许存在的 Python loop 仅限：

**与 batch size N 无关的固定小常数循环。**

例如允许：

```python
for action in range(4):
```

允许：

```python
for col in range(3):
```

允许：

固定最多 3 次 compression pass。

这些循环次数固定，

不会随着：

```text
N = 256
1024
4096
8192
16384
```

增长。

---

# 8. 严格禁止的 movement 实现

禁止：

```python
for i in range(len(boards)):
```

禁止：

```python
for board in boards:
```

禁止：

```python
[_merge_line(row) for row in rows]
```

禁止：

```python
np.apply_along_axis(...)
```

如果其内部实际逐 row 回 Python。

禁止：

```python
np.vectorize(...)
```

因为：

`np.vectorize`

不是实际性能 vectorization。

禁止：

每 board 调：

```python
Reference2048Env
move_without_spawn
```

禁止：

每 board 创建 Python：

- list；
- tuple；
- dataclass；
- temporary ndarray。

---

# 9. Canonical LEFT 路线继续固定

不要重新设计四套移动算法。

继续使用：

**四方向 → canonical LEFT → inverse transform**

思路。

固定：

```text
LEFT
原 board

RIGHT
每行 reverse

UP
transpose

DOWN
transpose + reverse
```

然后全部进入：

```text
vectorized pack-left
→ vectorized merge-left
→ vectorized pack-left
```

最后恢复原方向。

---

# 10. 本次 vectorized row 输入固定

内部 row batch：

```python
rows.shape == (R, 4)
rows.dtype == np.uint8
```

其中：

```text
R = 某组 board 数量 × 4
```

`R`

可以非常大。

所有操作：

必须同时作用于：

整个 `(R,4)`。

禁止：

逐 row Python 调用。

---

# 11. Pack-left 实现方向锁定

实现一个内部 helper，语义类似：

```python
_pack_left_rows(rows)
```

输入：

```text
(R,4) uint8
```

输出：

```text
(R,4) uint8
```

必须：

保持非零 tile 相对顺序，

所有 0 移动到右侧。

例如：

```text
[0,1,0,2]
→
[1,2,0,0]
```

```text
[0,0,0,5]
→
[5,0,0,0]
```

实现必须：

**跨全部 R rows 使用 NumPy boolean indexing / advanced indexing。**

允许：

固定 3 次 shift pass。

例如整体思路允许：

```text
最多做 3 次：

对 column 0..2：
    mask = 当前格 == 0 且右边格 != 0
    对所有 rows 一次性移动
```

这是：

固定：

```text
3 × 3
```

个 mask 操作。

这是允许的。

禁止：

对 `R` 循环。

---

# 12. Merge-left 实现方向锁定

实现内部 helper：

```python
_merge_left_rows(rows)
```

输入：

已经 pack-left 的：

```text
(R,4)
```

必须跨全部 rows 同时 merge。

固定语义：

从左往右。

一个 tile：

每 step 最多参与一次 merge。

新 tile：

本 step 不能再次 merge。

推荐并优先采用：

固定 column loop：

```text
c = 0
c = 1
c = 2
```

每个 `c`：

使用长度：

```text
R
```

的 boolean mask，

一次性找出所有：

```text
rows[:,c] == rows[:,c+1]
```

且：

- 非零；
- 当前位置没有因为前一个 merge 被 block；

的 rows。

必须维护：

**per-row one-step block mask。**

不能写成：

先把所有相等邻居同时 merge，

否则：

```text
[1,1,1,1]
```

容易错误。

必须继续得到：

```text
[2,2,0,0]
```

而不是其他结果。

---

# 13. 推荐的 merge control flow

允许固定为：

```text
packed = pack_left(rows)

blocked_from_previous = bool[R] all False
reward = int64[R] all zero

for c in 0,1,2:

    can_merge =
        not blocked_from_previous
        AND packed[:,c] != 0
        AND packed[:,c] == packed[:,c+1]

    对 can_merge 的所有 rows：
        packed[:,c] += 1
        packed[:,c+1] = 0
        reward += 2 ** new_exponent

    blocked_from_previous = can_merge

packed = pack_left(packed)
```

注意：

这里的：

```python
for c in range(3)
```

是：

**固定 3 次。**

允许。

禁止把它改成：

`for row in rows`。

---

# 14. Reward overflow 必须继续安全

现有要求不变：

FastEnv reward：

```python
np.int64
```

必须正确支持至少：

```text
16+16 -> 17
20+20 -> 21
21+21 -> 22
```

不得 silent overflow。

单次 merge reward：

如果超过：

```text
np.int64
```

范围：

显式：

```python
OverflowError
```

同一 row / board：

多个 reward 相加导致 int64 overflow：

也必须显式检测。

禁止：

为了 vectorization 删除 overflow 检查。

---

# 15. uint8 tile overflow 同样不得回绕

如果：

```text
255 + 255
```

需要产生 exponent：

```text
256
```

则不能：

uint8 wrap。

必须保持现有约定：

显式：

```python
OverflowError
```

禁止：

为了快而：

```text
255 -> 0
```

---

# 16. move_batch 动作分组固定

`move_batch(boards, actions)`

允许：

固定循环四个 action：

```python
for action in (UP, DOWN, LEFT, RIGHT):
```

对：

```python
actions == action
```

一次得到：

整个 board subset。

然后：

一次性 canonical transform，

一次性 reshape 成：

```text
(R,4)
```

一次性：

pack / merge / pack，

一次性 inverse transform，

一次性写回输出。

这属于：

**真正跨 board vectorization。**

---

# 17. 禁止每个 action subset 再逐 board

允许：

```text
4 action groups
```

不允许：

```text
action group
↓
for each board
↓
scalar merge
```

action group 内：

必须是：

NumPy batch。

---

# 18. legal_mask_batch 继续由 move 语义产生

禁止重新发明：

“检查空格和相邻 tile”

这种第二套 legal 规则。

legal 仍然必须满足：

```text
legal[a]
⇔
move(a).moved
```

为了避免重复代码：

允许增加内部：

```python
_move_all_actions_batch(boards)
```

类似 helper。

它可以一次计算：

```text
all_afterstates
all_rewards
all_moved
```

shape 可为：

```text
afterstates: (N,4,16)
rewards:     (N,4)
moved:       (N,4)
```

然后：

```text
legal_mask_batch
```

直接使用：

`moved`。

但：

不要为了性能改变 legal 定义。

---

# 19. 是否加入 _move_all_actions_batch

推荐：

**加入。**

因为：

当前 benchmark 中：

Legal/action preparation

占：

38.73%。

如果：

`legal_mask_batch`

仍然触发大量独立 scalar movement，

会浪费这次优化。

因此推荐最终结构：

```text
vectorized canonical move core
        ↑
        │
move_batch
        │
legal_mask_batch
        │
is_terminal_batch
```

共享同一批量 movement semantics。

但：

公共 API 不变。

---

# 20. Fast2048BatchEnv.step 不得改变语义

`step(actions)`：

语义完全保持当前版本。

非法：

```text
state unchanged
afterstate unchanged
reward = 0
legal = False
score unchanged
no spawn
```

合法：

```text
move
→ afterstate
→ spawn once
→ state
→ score += reward
→ terminal
```

不要因为优化：

改变：

- RNG；
- spawn；
- terminal；
- score；
- reset；
- reset_where。

---

# 21. Spawn 代码本次原则上不重写

当前报告：

```text
spawn_random ≈ 2.23M boards/s
exact spawn ≈ 7–12M outcomes/s
```

它们不是当前最大瓶颈。

因此：

**本任务原则上禁止大改 spawn implementation。**

除非：

修改 movement 后重新 profiling，

证明 spawn 成为新的主要瓶颈。

即便如此：

本任务也只记录结果，

不要继续无限优化。

本轮唯一必须修：

movement hot path。

---

# 22. 不允许为了 benchmark 好看减少 correctness

禁止：

- 缓存错误结果；
- 减少 high-tile support；
- clamp exponent；
- 跳过 illegal；
- 少算 legal；
- 少算 terminal；
- 降低 spawn correctness；
- 修改 workload；
- benchmark 永远 LEFT；
- benchmark 不 reset terminal；
- 修改 test sample 数量。

Benchmark workload：

继续保持现有 M1 规范。

---

# 23. 现有 383 tests 是最低下限

修改以后：

必须首先保证现有：

```text
383 tests
```

全部继续通过。

测试数可以：

增加。

不能：

减少。

不得删除：

旧测试。

不得：

把失败测试：

skip / xfail。

不得：

修改 expected result 适应新实现。

---

# 24. 必须新增“真正 vectorized”防回归测试

新增至少一项：

防止以后又有人把：

per-board Python loop

塞回来。

测试重点：

### A. Reference isolation

monkeypatch：

M0 scalar movement：

```python
move_without_spawn
```

使其调用立即抛异常。

然后：

FastEnv batch move

仍必须正常工作。

证明：

production FastEnv：

不调用 M0 scalar implementation。

---

# 25. 必须新增大 Batch correctness test

至少：

```text
N = 8192
```

随机 boards。

随机 actions。

调用：

一次：

```python
move_batch(...)
```

然后抽取：

至少：

```text
512
```

个随机 indices，

逐个使用：

M0 Reference

交叉验证。

要求：

0 mismatch。

---

# 26. 必须检查源码中不存在 N-dependent Python movement loop

在完成后：

人工审查：

```text
move_batch
vectorized movement helpers
legal_mask_batch
```

不得存在：

```python
for i in range(len(boards))
```

```python
for board in boards
```

```python
for row in rows
```

或者等价逻辑。

允许的循环必须能够明确证明：

循环次数只与：

```text
4 actions
4 cells
3 merge boundaries
```

有关，

与：

`N`

无关。

在：

`M1_REPORT.md`

明确写：

```text
N-dependent Python movement loop: NONE
```

---

# 27. Differential Test 全部重新运行

优化以后：

重新运行原有全部：

### Ordinary

```text
10,000 boards × 4 actions
= 40,000
```

0 mismatch。

### High tile

```text
2,000 × 4
= 8,000
```

0 mismatch。

### Legal

```text
10,000
```

0 mismatch。

### Terminal

```text
10,000
```

0 mismatch。

### Spawn enumeration

```text
2,000 parents
```

0 mismatch。

### D4

```text
2,000 × 8 × 4
= 64,000
```

0 mismatch。

除原先已经冻结的：

int64 reward representational limit

以外，

不得新增：

Fast vs Reference 差异。

---

# 28. M0 仍必须全绿

最终：

```text
python -m pytest
```

必须同时包含：

M0 + M1。

M0：

261 tests

仍然全部 PASS。

禁止修改 pytest 配置：

排除 M0。

---

# 29. 性能测试必须和旧结果直接比较

旧 baseline 已知：

```text
256      25.5k transitions/s
1024     26.1k
4096     14.6k
8192     14.9k
16384    14.8k
```

primitive：

```text
move_batch ≈ 112k boards/s
```

producer → consumer：

```text
≈14.45k transitions/s
```

修改以后：

使用：

**完全相同 benchmark workload**

重新测试。

不要：

改 workload 来制造提升。

---

# 30. 重新执行完整规模 sweep

必须重新测试：

```text
256
1024
4096
8192
16384
```

仍然：

- warmup；
- ≥3 repeats；
- median；
- min；
- max。

记录：

```text
old throughput
new throughput
speedup
```

speedup：

固定：

```text
new / old
```

---

# 31. Primitive move benchmark 必须重新测

旧：

```text
≈112k boards/s
```

新：

必须重新测。

记录：

```text
old
new
speedup
```

本任务不能只证明：

“循环没了。”

还必须证明：

**真实 throughput 变快。**

---

# 32. 性能修复最低验收线

为了避免：

“技术上 vectorized 了，但实际更慢”

本轮加入明确性能 Exit Criteria。

在：

**同一机器 / 同一 benchmark workload**

条件下：

### 必须满足 A

`move_batch boards/s`

至少达到旧 baseline：

```text
112k
```

的：

**2.0×**

也就是：

至少明显超过：

```text
224k boards/s
```

如果未达到：

本项 FAIL。

---

# 33. End-to-end M1 throughput 最低验收

4096 env：

旧：

```text
≈14.6k transitions/s
```

新实现：

至少达到：

旧值的：

**1.5×**

即：

约：

```text
21.9k transitions/s
```

以上。

否则：

M1 不得最终 PASS。

如果机器或运行环境和旧 benchmark 明显不同：

不得直接拿绝对值判。

必须：

在同一当前环境内：

运行：

旧实现 baseline

和：

新实现

做 A/B。

如果没有办法运行旧实现：

明确记录原因，

并以：

结构要求 + primitive move 2× + 新 profiler

进行验收。

不要伪造比较。

---

# 34. 性能目标不是只过最低线

上面的：

2× primitive

和：

1.5× end-to-end

只是：

**最低验收门槛。**

如果简单 NumPy vectorization：

自然能获得更高提升，

不要人为限速。

但也不要为了追求漂亮数字：

引入本阶段禁止的：

- C++；
- Numba；
- CUDA；
- 算法语义变化。

---

# 35. Scaling 应重新观察

旧实现：

1024 → 4096

吞吐下降：

约 44%。

修改后记录：

```text
256
1024
4096
8192
16384
```

新的 saturation region。

不要预先要求：

必须：

16384 最快。

正确目标：

找到：

**新的真实性能饱和点。**

如果 vectorization 后：

最佳规模仍然在较小 N，

这是允许的，

只要原因已经通过 profiling 解释。

---

# 36. Worker Scaling 重新运行

继续测试：

```text
1 worker
2 workers
4 workers
```

记录：

- throughput；
- scaling efficiency。

不要为了这次任务：

改 worker architecture。

现有：

subprocess + file

如果是环境限制导致，

可以继续保持。

本轮重点不是：

重写 worker IPC。

---

# 37. Producer → Consumer Benchmark 重新运行

旧：

```text
≈14.45k transitions/s
```

重新测。

继续分解：

```text
consumer %
env step %
batch assembly %
```

观察：

movement vectorization 后，

瓶颈是否：

从 Env 转移到 consumer。

这是重要结果。

---

# 38. Profiling 必须重新做

旧 profile：

```text
Environment stepping 60.28%
Legal/action          38.73%
≈99.01%
```

新 profile：

必须重新输出。

至少：

```text
Environment stepping
Legal/action preparation
Spawn
Exact spawn enumeration
Batch assembly
Memory copy
Synchronization
Consumer
Other
```

并给出：

absolute time + percentage。

---

# 39. 必须重新看 cProfile 热点

旧热点：

```text
_merge_line_cached
_shift_line
_line_changed
```

修改以后：

这些标量 per-row helper：

不得继续作为：

主要 hot path。

如果还在：

并且调用数继续随：

N × rows

线性暴涨，

说明：

本任务没有真正完成。

---

# 40. 不要为了保留旧 helper 强行使用它们

如果旧：

```text
_merge_line_cached
_shift_line
_line_changed
```

只适合：

scalar row，

可以：

删除，

或者：

保留为非 hot-path helper。

但是：

生产 `move_batch`

不能为了“少改代码”

继续调用它们几十万次。

---

# 41. 可以删除已经无用的 M1 scalar cache

如果当前 fast_env.py 中存在：

只服务于旧 per-board movement 的：

- scalar cache；
- line cache；
- Python tuple lookup；
- per-line memoization；

在新 vectorized implementation 完成并测试通过后：

允许删除。

但：

先确认没有其他 API 使用。

不要保留两套生产 movement 路径造成维护分叉。

---

# 42. 不得建立第二套规则语义

虽然 movement core 会重写，

但它仍必须：

只有一套 batch movement semantics。

不要留下：

```text
old scalar fast move
new vectorized fast move
```

然后不同 API 各用一套。

最终：

```text
move_batch
legal_mask_batch
terminal logic
Fast2048BatchEnv
```

必须尽量共享：

同一 vectorized movement core。

---

# 43. 优化顺序固定

严格按以下顺序。

### Step 1

不要改代码。

先重新运行：

```bash
python -m pytest
```

记录：

当前 baseline。

### Step 2

重新运行一次当前：

```text
move primitive benchmark
4096 env benchmark
```

确认旧 baseline 大致可复现。

如果差异很大：

记录机器 / 环境变化。

### Step 3

检查：

`fast_env.py`

明确定位：

所有：

N-dependent Python movement loops。

记录到：

临时开发笔记 / M1 report。

### Step 4

实现：

batch：

`_pack_left_rows`

禁止 per-row loop。

### Step 5

为 `_pack_left_rows`

单独测试。

至少覆盖：

```text
[0,0,0,0]
[1,0,0,0]
[0,1,0,0]
[0,0,0,1]
[0,1,0,2]
[1,0,2,0]
[0,1,2,3]
```

### Step 6

实现：

vectorized：

`_merge_left_rows`

禁止 per-row loop。

### Step 7

专项验证：

```text
[1,1,1,1]
[1,1,2,0]
[1,1,1,0]
[1,0,1,1]
[2,2,2,2]
[2,2,3,3]
```

### Step 8

接入：

canonical four-direction movement。

### Step 9

运行：

movement differential。

必须：

0 mismatch。

### Step 10

运行：

high-tile differential。

必须：

0 mismatch。

### Step 11

接入：

legal_mask_batch / terminal。

优先共享：

vectorized movement core。

### Step 12

运行：

legal / terminal differential。

0 mismatch。

### Step 13

运行：

完整：

```bash
python -m pytest
```

### Step 14

确认：

至少原：

383 tests

全部存在且通过。

### Step 15

新增：

large-batch vectorization regression test。

### Step 16

重新运行：

完整 pytest。

### Step 17

运行：

move primitive benchmark。

### Step 18

如果：

move primitive < old × 2：

不要宣布成功。

profile：

vectorized movement core，

继续优化：

NumPy implementation。

仍然：

禁止 C++。

### Step 19

达到 primitive 最低线后：

运行完整 env-count sweep。

### Step 20

运行：

worker scaling。

### Step 21

运行：

producer → consumer benchmark。

### Step 22

运行：

wall-clock profiling + cProfile。

### Step 23

确认：

标量 per-board merge helpers

不再是主要热点。

### Step 24

更新：

`docs/M1_FAST_ENV_SPEC.md`

### Step 25

更新：

`M1_REPORT.md`

### Step 26

最终重新运行：

```bash
python -m pytest
```

### Step 27

检查：

M0 frozen files：

无 diff。

### Step 28

逐条执行：

本提示词最终 Exit Criteria。

### Step 29

如果全部通过：

**M1 FINAL PASS。**

### Step 30

立即停止。

**禁止开始 M2。**

---

# 44. 最终 M1 Exit Criteria

必须同时满足以下全部条件：

```text
[ ] M0 261 tests 全部继续通过
[ ] 原 M1 tests 全部继续通过
[ ] 总测试数不得低于原来的 383
[ ] ordinary differential 40,000 comparisons = 0 mismatch
[ ] high-tile differential 8,000 comparisons = 0 mismatch
[ ] legal differential = 0 mismatch
[ ] terminal differential = 0 mismatch
[ ] spawn enumeration differential = 0 mismatch
[ ] D4 differential = 0 mismatch

[ ] move_batch production hot path 不存在 N-dependent Python loop
[ ] vectorized row core 不存在 per-row Python loop
[ ] 禁止 np.vectorize 冒充 vectorization
[ ] 禁止 np.apply_along_axis 调 scalar Python merge
[ ] 禁止 FastEnv 调 M0 scalar move 作为生产实现

[ ] canonical LEFT 语义正确
[ ] reward int64 安全检查仍然存在
[ ] uint8 tile overflow 安全检查仍然存在
[ ] illegal action 语义未改变
[ ] RNG 语义未改变
[ ] spawn 语义未改变
[ ] reset/reset_where 未改变

[ ] N=8192 batch correctness test 已加入
[ ] 大 batch 抽样 reference comparison = 0 mismatch

[ ] move primitive throughput ≥ old baseline × 2
[ ] 4096-env throughput ≥ old baseline × 1.5
    或在运行环境变化时完成同机 A/B 并证明等价提升

[ ] 256 benchmark 完成
[ ] 1024 benchmark 完成
[ ] 4096 benchmark 完成
[ ] 8192 benchmark 完成
[ ] 16384 benchmark 完成

[ ] worker scaling 重新完成
[ ] producer→consumer benchmark 重新完成
[ ] wall-clock profile 重新完成
[ ] cProfile 重新完成

[ ] 旧 per-board scalar merge 不再是主要热点
[ ] 新 saturation point 已记录
[ ] 新 bottleneck 已记录

[ ] 未实现 C++
[ ] 未实现 M2
[ ] 未使用 tuple checkpoint
[ ] 未修改 M0 frozen files

[ ] M1_FAST_ENV_SPEC.md 已更新
[ ] M1_REPORT.md 已更新
```

其中任何：

correctness 条目失败：

立即：

```text
M1 = FAIL
```

其中：

```text
N-dependent Python movement loop
```

仍存在：

```text
M1 = FAIL
```

其中：

性能最低线未达到：

```text
M1 = PERFORMANCE INCOMPLETE
```

不得写：

PASS。

---

# 45. M1_REPORT.md 必须修改最终结论

原报告中：

```text
M1 = PASS
```

必须根据本轮实际结果重新判定。

如果本轮成功：

改为：

```text
M1 = FINAL PASS
```

并增加：

```text
## R. Vectorization Repair

Previous implementation:
    per-board Python movement loop

Previous move throughput:
    ~112k boards/s

New implementation:
    ...

N-dependent Python movement loop:
    NONE

Old primitive throughput:
    ...

New primitive throughput:
    ...

Primitive speedup:
    ...x

Old 4096-env throughput:
    ~14.6k transitions/s

New 4096-env throughput:
    ...

End-to-end speedup:
    ...x

Old bottleneck:
    ...

New bottleneck:
    ...

Old saturation:
    ...

New saturation:
    ...
```

---

# 46. Known Issues 不能继续写旧问题而仍然 PASS

如果最后：

`_move_batch`

仍然是：

per-board Python loop，

则：

Known Issues

里不能只是写：

“未做到字面 vectorize”

然后同时：

```text
M1 = PASS
```

这是禁止的。

本次任务就是：

解决这个问题。

---

# 47. 最终 C++ Decision

修复 NumPy vectorization 后：

重新填写：

```text
C++ migration recommended:
YES / NO / DEFER_TO_M2
```

本轮即使结果是：

```text
YES
```

也：

**不要实现 C++。**

只记录。

实际上如果：

NumPy vectorization 后 M1 已经不再明显是瓶颈，

优先：

```text
DEFER_TO_M2
```

等 M2 的真实神经网络出来以后：

再判断 CPU 是否喂不饱 RTX 5060。

---

# 48. 最终回复格式固定

最后只报告：

```text
1. M1 FINAL PASS / FAIL / PERFORMANCE INCOMPLETE

2. Movement implementation
旧：
新：

3. Python per-board loop
NONE / STILL PRESENT

4. pytest
总数
passed
failed

5. Differential
ordinary
high-tile
legal
terminal
spawn
D4
mismatch

6. Performance
                OLD      NEW      SPEEDUP
move_batch
256 env
1024 env
4096 env
8192 env
16384 env
producer/consumer

7. Saturation point
旧：
新：

8. Profiling
旧 bottleneck：
新 bottleneck：

9. C++ decision
YES / NO / DEFER_TO_M2

10. Known issues

11. M0 frozen files modified?
NO 必须

12. M2 started?
NO 必须
```

不要写大段无关解释。

---

# 49. 最终停止规则

当：

M1 FINAL PASS

以后：

立即停止。

禁止：

“既然性能好了，我顺手搭一下网络。”

禁止：

“顺便安装 PyTorch。”

禁止：

“顺便试一下 5060。”

禁止：

“顺便把 tuple checkpoint 接进来。”

禁止：

“顺便开始 M2。”

M2：

必须等待用户下一条明确指令。

---

# 50. 最终原则

这次不是重新设计 M1。

不是重新设计 2048。

不是追求无限优化。

不是开始 C++。

不是开始 M2。

只做一件事：

> **把当前已经正确的 FastEnv，从“batch API 包着 Python 标量循环”，修成真正跨 board 的 NumPy batch movement，并用原有 Golden Reference 证明它仍然 100% 正确，再用相同 benchmark 证明性能确实得到实质提升。**

完成以后：

才允许把 M1 正式标记为：

**FINAL PASS。**