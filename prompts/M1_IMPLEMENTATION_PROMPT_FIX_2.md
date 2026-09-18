你现在负责：

# M1 Post-Audit Correctness Fix：int64 Reward 聚合溢出

这是一个**极小范围的 M1 修复任务**。

这不是：

- 重做 M1；
- 重写 FastEnv；
- 再次优化 movement；
- 上 C++；
- 开始 M2；
- 修改训练算法。

当前仓库已经完成：

```text
M0:
m0-reference-pass
→ 3f2def1d95f56eff776e671143188947bf64485b

M1 当前已提交版本:
a34e56b36a2fdf3e9b1eb66c0b3ea1afeeb5561e

旧 M1 tag:
m1-fastenv-pass
```

M1 当前已有：

- 真正跨 board vectorized movement；
- 无 N-dependent Python movement loop；
- 391 tests 全绿；
- differential 全部 0 mismatch；
- 约 5× FastEnv 性能提升；
- M0 frozen files 未修改；
- M2 未开始。

以上成果全部保留。

但是代码审查发现：

**当前 M1 存在一个明确 correctness bug。**

本任务只修这个 bug。

---

# 0. 最终目标

必须修复：

> 多个单独可表示的 merge reward，在同一 row 或同一 board 内累加以后超过 `np.int64` 上限时，当前代码可能静默整数回绕，而不是抛出 `OverflowError`。

修复完成后必须保证：

```text
任何 M1 reward：

要么得到完全正确的非负 np.int64 数值；

要么在真实结果无法用 np.int64 精确表示时显式抛 OverflowError。

绝对禁止 silent wrap。
```

---

# 1. 当前发现的两个具体溢出位置

不是只有 board-level sum 一个位置。

必须同时修复以下两层。

---

## 1.1 Row 内多个 merge 的 reward 聚合

当前 `_merge_left_rows()` 中类似：

```python
reward[can_merge] += _MERGE_REWARD[left[can_merge]]
```

的问题：

单次：

```text
61 + 61 → 62
reward = 2**62
```

可以放进 int64。

但是一行：

```text
[61, 61, 61, 61]
```

向 LEFT：

会发生两个 merge：

```text
61 + 61 → 62
61 + 61 → 62
```

真实 reward：

```text
2**62 + 2**62
= 2**63
```

已经超过：

```text
np.iinfo(np.int64).max
= 2**63 - 1
```

所以这一行：

**必须抛 `OverflowError`。**

不得：

wrap 成负数。

---

## 1.2 Board 内多个 line reward 聚合

当前 `_move_groups()` 中类似：

```python
with np.errstate(over="raise"):
    rewards[selected] = line_reward.reshape(
        group_size, BOARD_COLUMNS
    ).sum(axis=1)
```

这是错误的保护方式。

原因：

**NumPy 整数 reduction 溢出不会可靠地被 `np.errstate(over="raise")` 捕获。**

例如棋盘：

```text
61 61  0  0
61 61  0  0
 0  0  0  0
 0  0  0  0
```

向 LEFT。

每一行分别：

```text
reward = 2**62
```

两行合计：

```text
2**63
```

同样超 int64。

M0：

使用 Python arbitrary-precision int，

可以返回：

```text
9223372036854775808
```

M1：

因为固定使用 int64，

必须：

```python
raise OverflowError
```

绝不能返回：

```text
-9223372036854775808
```

或其他 wrap 后数字。

---

# 2. 本次禁止范围扩张

本任务禁止：

- 重写 movement architecture；
- 改 `_LINE_ORDER`；
- 改 `_LINE_INVERSE`；
- 改 canonical movement；
- 改 `_pack_left_rows` 算法；
- 改 spawn；
- 改 RNG；
- 改 reset；
- 改 reset_where；
- 改 legal mask 定义；
- 改 terminal 定义；
- 改 Action；
- 改 board encoding；
- C++；
- pybind11；
- Cython；
- Numba；
- Triton；
- CUDA；
- Rust；
- M2；
- PyTorch；
- Transformer；
- MLP；
- tuple checkpoint；
- Teacher；
- Expectimax。

本任务不是性能重构。

只修：

**reward overflow correctness。**

---

# 3. M0 仍然永久冻结

以下 M0 文件禁止修改：

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

**Golden Reference。**

当前：

```text
m0-reference-pass
```

必须保持原样。

禁止移动 tag。

禁止 retag。

禁止修改历史。

---

# 4. 旧 M1 tag 也不得移动

现有：

```text
m1-fastenv-pass
```

已经指向：

```text
a34e56b36a2fdf3e9b1eb66c0b3ea1afeeb5561e
```

虽然现在发现了 post-audit bug，

但：

**禁止删除、移动或覆盖这个 tag。**

它保留为：

“审计前 M1 snapshot”。

本次修复完成以后：

创建新的 tag：

```text
m1-fastenv-final-pass
```

指向新的修复 commit。

---

# 5. 开始前固定检查

先执行：

```bash
git status
git rev-parse HEAD
git rev-parse m0-reference-pass^{commit}
git rev-parse m1-fastenv-pass^{commit}
```

预期：

```text
HEAD = a34e56b...
m0-reference-pass = 3f2def1...
m1-fastenv-pass = a34e56b...
```

如果 HEAD 已经包含用户后续提交：

不要 reset。

在当前最新 HEAD 上继续。

禁止：

```bash
git reset --hard
```

禁止：

rebase。

禁止：

force push。

---

# 6. 允许修改的代码文件

本任务主要允许修改：

```text
src/game2048/fast_env.py
tests/test_m1_high_tiles.py
docs/M1_FAST_ENV_SPEC.md
M1_REPORT.md
```

如果确实需要：

允许新增一个非常小的专项测试文件：

```text
tests/test_m1_reward_overflow.py
```

二选一：

- 直接补进 `test_m1_high_tiles.py`；
- 或新增 `test_m1_reward_overflow.py`。

不要两个都做重复测试。

---

# 7. 必须增加统一的安全 int64 加法逻辑

禁止继续依赖：

```python
np.errstate(over="raise")
```

捕获整数 overflow。

它不能作为本项目 int64 reward 的 correctness 保障。

推荐固定新增内部常量：

```python
_INT64_MAX = np.iinfo(np.int64).max
```

推荐新增内部 helper：

```python
def _checked_add_nonnegative_int64(
    total: np.ndarray,
    increment: np.ndarray,
    *,
    context: str,
) -> np.ndarray:
    ...
```

或者等价内部实现。

语义固定：

输入：

```text
total:
    np.int64 array
    全部 >= 0

increment:
    np.int64 array
    全部 >= 0
```

在执行：

```text
total + increment
```

之前：

必须先检测：

```python
increment > (_INT64_MAX - total)
```

如果任何元素满足：

```python
raise OverflowError(...)
```

否则：

安全执行：

```python
total += increment
```

---

# 8. 禁止用 Python object dtype 逃避问题

禁止把生产 FastEnv reward 改成：

```python
dtype=object
```

禁止：

每个 board 用 Python `int`。

禁止：

逐 board Python 循环。

M1 fast path 继续固定：

```text
reward dtype = np.int64
```

超范围：

**OverflowError。**

---

# 9. 修复 `_merge_left_rows`

当前：

```python
reward[can_merge] += ...
```

必须替换为：

**先检查，再相加。**

固定逻辑应等价于：

```python
increment = _MERGE_REWARD[left[can_merge]]
current = reward[can_merge]

if np.any(increment > (_INT64_MAX - current)):
    raise OverflowError(
        "row merge reward does not fit in numpy.int64"
    )

reward[can_merge] = current + increment
```

可以使用统一 helper。

不得：

先发生 int64 overflow，

再检查结果是否为负。

必须：

**在加法发生以前检测。**

---

# 10. 修复 board-level reward 聚合

当前：

```python
line_reward.reshape(group_size, BOARD_COLUMNS).sum(axis=1)
```

不能继续作为无保护的总 reward 聚合。

必须使用：

固定 4 条 line 的 checked accumulation。

允许：

```python
line_rewards = line_reward.reshape(
    group_size,
    BOARD_COLUMNS,
)

total = np.zeros(group_size, dtype=np.int64)

for column in range(BOARD_COLUMNS):
    total = checked_add(total, line_rewards[:, column])
```

这里：

```python
for column in range(4)
```

是固定常数循环。

**允许。**

它与 batch size N 无关。

禁止：

```python
for board in boards:
```

---

# 11. 允许增加低成本 fast path

如果你确认不会增加复杂错误，

允许：

```python
max_line_reward = ...
```

如果可以数学证明：

4 条 line 的最大可能总和仍然不超过：

```text
INT64_MAX
```

则使用：

```python
sum(axis=1)
```

快速路径。

否则：

进入 checked accumulation。

但是：

**这不是必需项。**

第一优先级：

正确。

如果简单的固定 4 次 checked accumulation：

性能影响很小，

优先保持简单。

---

# 12. 必须保持 movement 完全 vectorized

修复以后：

以下仍然必须成立：

```text
N-dependent Python movement loop = NONE
```

允许循环：

```python
for action in range(4)
```

```python
for column in range(4)
```

```python
for _column in range(3)
```

禁止：

```python
for board in boards
```

禁止：

```python
for row in rows
```

禁止：

```python
for i in range(group_size)
```

禁止：

`np.vectorize`

禁止：

`np.apply_along_axis`

---

# 13. 必须新增 Row-level overflow 回归测试

新增测试：

```text
board:

61 61 61 61
 0  0  0  0
 0  0  0  0
 0  0  0  0

action = LEFT
```

M0：

reward：

```text
2**63
```

M1：

必须：

```python
with pytest.raises(OverflowError):
    move_batch(...)
```

这个测试用于证明：

**同一 line 内多个合法单 merge 累加后超 int64 会显式失败。**

---

# 14. 必须新增 Board-level overflow 回归测试

新增：

```text
61 61  0  0
61 61  0  0
 0  0  0  0
 0  0  0  0
```

action：

```text
LEFT
```

M0 reward：

```text
2**63
```

M1：

必须：

```python
OverflowError
```

这个测试用于证明：

**不同 line 的 reward 聚合也不能 wrap。**

---

# 15. 必须增加 safe near-limit 测试

不能只测试“会溢出”。

还必须证明：

**接近上限但仍然合法的 reward 不会被错误拒绝。**

使用：

```text
61 61  0  0
60 60  0  0
 0  0  0  0
 0  0  0  0
```

LEFT。

真实 reward：

```text
2**62 + 2**61
```

它满足：

```text
< 2**63 - 1
```

M1：

必须正常返回。

并且：

```python
int(fast_reward) == reference_reward
```

---

# 16. 原有单 merge 边界必须继续成立

已有语义保持：

```text
61 + 61 → 62
reward = 2**62
```

必须成功。

```text
62 + 62
```

必须：

```python
OverflowError
```

不得因为修复 total overflow：

把：

61 merge

也全部禁止。

---

# 17. 必须覆盖四个方向

至少对：

board-level overflow case

参数化：

```text
UP
DOWN
LEFT
RIGHT
```

都验证：

当真实总 reward > int64：

M1：

```text
OverflowError
```

不要只测 LEFT。

允许通过构造等价旋转 board 实现。

---

# 18. 必须测试 Batch 混合情况

构造一个 batch：

```text
board 0:
普通低位正常 move

board 1:
near-limit 但合法

board 2:
reward overflow

board 3:
普通高位但不 merge
```

调用：

```python
move_batch(...)
```

由于 batch 中包含：

一个无法精确表示的 transition，

整个调用：

必须：

```python
OverflowError
```

禁止：

只返回部分 batch 的错误数据。

---

# 19. Fast2048BatchEnv.step 必须保持原子性

这是必须新增的测试。

准备一个：

会触发 reward overflow 的 board。

记录调用前：

```text
board
score
RNG bit_generator.state
```

然后：

```python
with pytest.raises(OverflowError):
    env.step(actions)
```

异常以后必须验证：

```text
board unchanged
score unchanged
RNG state unchanged
```

原因：

overflow 发生在：

move / reward 计算阶段。

不得：

先 spawn，

再发现 reward 错。

不得：

污染 environment state。

---

# 20. 不得修改 spawn/RNG 来通过 atomicity test

正常调用顺序本来就是：

```text
move
→ reward
→ spawn
```

所以正确修复 reward 检测以后：

OverflowError 应在 spawn 前发生。

禁止：

为了测试：

特殊回滚 RNG。

不要增加复杂 transaction system。

只保证：

错误在 state mutation 前被发现。

---

# 21. 必须检查 `_scores` 的 int64 累加边界

代码审查中顺便检查：

```python
self._scores += rewards
```

是否可能出现：

当前 score + 本步 reward > int64 max

而 silent overflow。

虽然真实 2048 对局不可能接近该范围，

但是 M1 已经明确规定：

**禁止 silent integer overflow。**

因此本任务必须明确决定：

### 推荐方案

在 `_scores += rewards` 前执行同样的 checked nonnegative int64 addition。

如果：

```text
score + reward > INT64_MAX
```

显式：

```python
OverflowError
```

不得 wrap。

---

# 22. Score overflow 必须有测试

允许直接人工设置：

```python
env._scores[0] = np.iinfo(np.int64).max - 1
```

再构造一个合法：

reward >= 2

的 move。

调用：

```python
env.step(...)
```

必须：

```python
OverflowError
```

并验证：

```text
score 没有 wrap 成负数
```

如果实现能够保证异常发生在 board spawn 之前：

最好同时验证：

board / RNG 不变。

如果 score overflow 只能在 move 后才能知道：

仍然要求：

不得出现部分提交。

---

# 23. Step 必须先验证 score 是否可累加，再 spawn

为了保证 atomicity，

推荐调整 `step()` 顺序为：

```text
1. move_batch
2. 得到 rewards
3. 检查 scores + rewards 是否全部 int64-safe
4. 只有检查通过：
   才执行 spawn
5. 更新 score
6. terminal
```

禁止：

```text
spawn
→ 然后才检查 score overflow
```

否则异常会留下已经改变的 board / RNG。

正常低位行为必须与原来完全一致。

---

# 24. 不要改变正常 RNG 序列

对所有不会 overflow 的普通游戏：

相同：

```text
seed
initial board
action sequence
```

修复前和修复后：

spawn trajectory

应保持一致。

不要为了安全检查：

额外调用 RNG。

---

# 25. 原有所有测试必须继续通过

当前 baseline：

```text
391 tests passed
0 failed
0 skipped
0 xfail
```

修复后：

总测试数：

```text
>= 391
```

并且：

全部 PASS。

禁止：

- 删除旧测试；
- skip；
- xfail；
- 改 expected result；
- 放宽 differential tolerance。

---

# 26. M0 261 tests 必须继续全部通过

最终完整：

```bash
python -m pytest
```

必须包含：

M0 + M1。

M0：

```text
261 passed
```

不能减少。

---

# 27. Differential 主体无需缩减

原有：

```text
ordinary       40,000
high-tile       8,000
legal           10,000
terminal        10,000
spawn            2,000 parents
D4              64,000
8192 batch sample 512
```

全部继续保留。

不要因为本次只是 overflow fix：

删除任何 differential。

---

# 28. 文档中的错误边界必须统一纠正

搜索仓库 M1 文件中的：

```text
53
exponent 53
e >= 53
```

以及所有关于：

int64 reward boundary

的说明。

不得留下错误说法：

```text
“exponent 53 以上就放不下 int64”
```

正确：

### 单个 merge

```text
e = 61:
reward = 2**62
可表示

e = 62:
reward = 2**63
不可表示
```

但是：

### 整个 action 总 reward

即使所有单 merge 都满足：

```text
e <= 61
```

多个 merge 的总和：

仍然可能超过：

```text
INT64_MAX
```

此时同样：

```text
OverflowError
```

---

# 29. M1 与 M0 divergence 的正式定义必须修改

禁止继续写：

> “M1 与 M0 唯一差异是 exponent >= 62 的 merge 抛异常。”

准确改成：

> M0 reward 使用 Python arbitrary-precision int；M1 FastEnv reward 使用 `np.int64`。因此，只要**单次 action 的真实总 reward**无法精确表示为非负 int64，M1 就显式抛出 `OverflowError`。这包括但不限于单个 exponent ≥ 62 merge，也包括多个 exponent ≤ 61 merge 的 reward 聚合后超过 `INT64_MAX`。M1 永远不得静默回绕。

这才是正式 contract。

---

# 30. 修正 `fast_env.py` 顶部模块说明

当前模块说明里存在旧的：

```text
exponent 53
```

边界描述。

必须改。

同时：

`move_batch()` docstring

如果仍写：

```text
exponent >= 53
```

也必须改。

---

# 31. 修正 `docs/M1_FAST_ENV_SPEC.md`

重点修改：

```text
§4 reward 与 overflow
§4.1 高位支持范围与 divergence
```

明确区分：

```text
single merge representability
```

和：

```text
whole-move aggregate representability
```

必须补上两个示例：

```text
[61,61,61,61]
→ total reward = 2**63
→ M1 OverflowError
```

以及：

```text
row 1: 61,61
row 2: 61,61
→ total reward = 2**63
→ M1 OverflowError
```

---

# 32. 修正错误的“三块缓冲”描述

当前文档 / 报告如果声称：

`Fast2048BatchEnv`

只持有：

```text
_boards
_scores
_rng
```

三块 buffer，

但实际还有：

```text
_empty_prefix
_rows_buffer
_terminated
```

则改成准确表述：

> 三块持久核心游戏状态为 `_boards / _scores / _rng`；此外允许持有性能 scratch/cache buffer，例如 `_empty_prefix / _rows_buffer / _terminated`，它们不是独立 per-env Python object。

不要继续写：

“类里面只有三块缓冲”。

---

# 33. 修正 benchmark 描述口径

不要修改已有原始 JSON 数据。

保留：

```text
baseline_before_vectorization.json
benchmark_results.json
ab_vectorization.json
```

但 `M1_REPORT.md`

不得继续把旧/新 env sweep 描述成：

**完全相同 step count workload**

因为旧：

部分数据用了：

```text
16 steps
```

新：

4096 档用了：

```text
48 steps
```

正确写法：

> env sweep 使用相同 benchmark 逻辑、seed 与环境规模，但部分正式测量的 step count 不同，因此其吞吐比可作为性能证据，但不是严格逐指令 A/B。严格同进程 primitive A/B 由 `benchmark_ab_vectorization.py` 提供。

---

# 34. A/B 脚本说明必须保持诚实

当前：

`benchmark_ab_vectorization.py`

OLD：

其实是：

```text
M0 scalar per-board route
```

不是：

已经删除的旧 M1 `_LINE_CACHE` 实现。

这一点脚本已经写明。

不要修改成：

“精确恢复旧 M1”。

`M1_REPORT.md`

也必须保持同样措辞。

---

# 35. 不需要完整重跑全部性能工程

本任务是 correctness fix。

不需要重新：

- 256/1024/8192/16384 完整 sweep；
- worker 1/2/4 scaling；
- producer/consumer 全套；
- full profiler。

只需要：

### Correctness

完整：

```bash
python -m pytest
```

### Performance sanity

重新跑：

```text
move_batch primitive
4096-env benchmark
```

用于确认：

安全 checked-add

没有造成明显性能回退。

---

# 36. 性能 sanity 验收

参考当前：

```text
move_batch:
~787k boards/s 正式 harness

4096 env:
~73.6k transitions/s
```

性能存在测量波动。

要求：

**不得出现明显数量级回退。**

如果同机 median 相比当前数据：

下降超过：

```text
10%
```

则：

先 profile。

确认：

checked overflow protection

是否被错误写成 per-board Python logic。

如果只是：

正常小幅波动：

记录即可。

不要为了追回 1～5%：

扩展本任务范围。

---

# 37. 不得为了性能删除 overflow 检查

即使 benchmark 下降：

禁止：

“真实游戏不会遇到，所以把检查删了。”

本项目规格已经明确：

**不能 silent overflow。**

安全检查是硬 requirement。

---

# 38. M1_REPORT 必须新增 Post-Audit Fix 章节

在：

`M1_REPORT.md`

新增：

```text
## S. Post-Audit int64 Reward Aggregation Fix
```

至少记录：

```text
Audit finding:
  Multiple individually representable merge rewards could overflow
  np.int64 during row/board aggregation without raising.

Affected pre-fix commit:
  a34e56b36a2fdf3e9b1eb66c0b3ea1afeeb5561e

Old tag:
  m1-fastenv-pass
  retained unchanged for history

Fix:
  ...

Row-level checked aggregation:
  PASS

Board-level checked aggregation:
  PASS

Score checked aggregation:
  PASS

Overflow atomicity:
  PASS

New regression tests:
  ...

pytest:
  ...

move_batch sanity:
  ...

4096-env sanity:
  ...

M0 modified:
  NO

M2 started:
  NO

Final status:
  M1 = FINAL PASS
```

---

# 39. 原来的 FINAL PASS 状态需注明历史修正

不要删除历史。

但报告开头应该明确写：

```text
M1 = FINAL PASS
```

并紧接一句：

> Final PASS includes the post-audit int64 aggregate-overflow fix documented in §S. The earlier `m1-fastenv-pass` tag predates this fix and is retained only as a historical snapshot.

避免以后别人误以为：

旧 tag 就是最终无缺陷版本。

---

# 40. 本次最终测试至少包含

必须明确确认：

```text
[ ] single 61+61 merge succeeds
[ ] single 62+62 merge raises
[ ] [61,61,61,61] row aggregate raises
[ ] two separate 61-pair rows board aggregate raises
[ ] safe 61-pair + 60-pair returns exact reward
[ ] overflow works in UP
[ ] overflow works in DOWN
[ ] overflow works in LEFT
[ ] overflow works in RIGHT
[ ] mixed batch containing overflow raises
[ ] env.step overflow leaves board unchanged
[ ] env.step overflow leaves score unchanged
[ ] env.step overflow leaves RNG unchanged
[ ] score accumulation overflow raises
[ ] ordinary seeded trajectories remain reproducible
[ ] all old 391 tests still pass
[ ] all new tests pass
```

---

# 41. 必须人工计算几个边界值

测试中不要自己写错数字。

固定：

```text
INT64_MAX
= 9223372036854775807
= 2**63 - 1

2**62
= 4611686018427387904

2**62 + 2**62
= 9223372036854775808
= 2**63
→ overflow

2**62 + 2**61
= 6917529027641081856
→ safe
```

---

# 42. 不要把 MAX_SAFE_MERGE_EXPONENT 改成 61

现有：

```python
MAX_SAFE_MERGE_EXPONENT = 62
```

其语义是：

```text
e >= 62 的 pair 禁止 merge
```

因此：

最大允许输入 merge exponent：

```text
61
```

这个常量名略容易误解，

但本次：

**不要为了命名重构 API。**

保持值：

```text
62
```

只把文档解释写清楚。

---

# 43. 修复后 Git 操作

只有在：

所有 correctness tests PASS

且：

性能 sanity 无明显回退

以后，

才执行：

```bash
git add src/game2048/fast_env.py
git add tests/
git add docs/M1_FAST_ENV_SPEC.md
git add M1_REPORT.md
```

检查：

```bash
git diff --cached
```

确认没有：

M0 frozen file。

然后：

```bash
git commit -m "Fix M1 int64 reward aggregation overflow"
```

---

# 44. 新 tag 固定

修复 commit 成功后：

创建：

```bash
git tag -a m1-fastenv-final-pass -m "M1 FastEnv final pass after int64 overflow fix"
```

然后：

```bash
git push
git push origin m1-fastenv-final-pass
```

**禁止移动：**

```text
m0-reference-pass
m1-fastenv-pass
```

---

# 45. Push 失败时的行为

如果 sandbox / 权限导致：

push 失败，

不要：

- force；
- 改 remote；
- 重建 repo；
- 删除 tag。

只报告：

```text
local commit created
local tag created
push failed due to environment restriction
```

代码修复本身仍然完成。

---

# 46. 最终 Exit Criteria

只有以下全部满足：

才能重新宣布：

```text
M1 = FINAL PASS
```

检查表：

```text
[ ] M0 frozen files 未修改
[ ] m0-reference-pass 未移动
[ ] old m1-fastenv-pass 未移动

[ ] row-level reward overflow 已修
[ ] board-level reward overflow 已修
[ ] score accumulation overflow 已修
[ ] 不再依赖 np.errstate 捕获整数 sum overflow

[ ] single merge e=61 正确
[ ] single merge e=62 OverflowError
[ ] multiple e=61 merges aggregate OverflowError
[ ] safe near-limit aggregate 返回准确 reward
[ ] overflow 不产生负数 / wrap

[ ] env.step 异常具备 atomicity
[ ] board 不被部分修改
[ ] score 不被部分修改
[ ] RNG 不被部分消耗

[ ] production movement 仍无 N-dependent Python loop
[ ] 无 np.vectorize
[ ] 无 apply_along_axis
[ ] API 未改变
[ ] spawn/RNG/reset/legal/terminal 语义未改变

[ ] 原 391 tests 全部仍通过
[ ] 新 overflow tests 全部通过
[ ] 0 skip
[ ] 0 xfail

[ ] move_batch performance sanity 完成
[ ] 4096-env performance sanity 完成
[ ] 无 >10% 无解释性能回退

[ ] fast_env.py overflow 文档已纠正
[ ] M1_FAST_ENV_SPEC.md 已纠正
[ ] M1_REPORT.md 已新增 §S
[ ] “只有三块 buffer”错误描述已纠正
[ ] benchmark workload 描述已纠正

[ ] M2 未开始
[ ] tuple checkpoint 未使用
```

任一 correctness 项失败：

```text
M1 = FAIL
```

不得：

PASS WITH KNOWN ISSUE。

---

# 47. 最终回复格式固定

最终只报告：

```text
1. Final status
M1 = FINAL PASS / FAIL

2. Bug fixed
Row-level aggregate:
Board-level aggregate:
Score aggregate:

3. Exact boundary results
61+61:
62+62:
[61,61,61,61]:
two 61-pair rows:
61-pair + 60-pair:

4. Atomicity
board unchanged on overflow:
score unchanged:
RNG unchanged:

5. pytest
total:
passed:
failed:
skipped:
xfailed:

6. Differential
ordinary:
high-tile:
legal:
terminal:
spawn:
D4:
mismatch:

7. Vectorization
N-dependent Python movement loop:
must be NONE

8. Performance sanity
move_batch before:
move_batch after:
difference:

4096 env before:
4096 env after:
difference:

9. Documentation corrections
fast_env.py:
M1_FAST_ENV_SPEC.md:
M1_REPORT.md:

10. Git
new commit:
old m0 tag unchanged:
old m1 tag unchanged:
new tag:

11. M0 modified?
NO

12. M2 started?
NO

13. Known issues
None / exact remaining issue
```

不要输出大段自我表扬。

---

# 48. 最终停止规则

完成：

```text
m1-fastenv-final-pass
```

以后：

立即停止。

禁止：

- 开始 M2；
- 安装 PyTorch；
- 测 RTX 5060；
- 接 tuple checkpoint；
- 上 C++；
- 继续优化 FastEnv；
- 开始 Teacher。

下一阶段必须等：

用户明确给出 M2 指令。

---

# 49. 一句话任务定义

你这次只需要做到：

> **保证 FastEnv 的所有 reward 与 score 都满足：能精确放进 np.int64 就返回完全正确的值；放不进去就在任何状态/RNG 被提交前显式抛出 OverflowError。然后补齐回归测试、纠正文档，并保持现有向量化性能架构完全不动。**

除此之外：

不要自由发挥。