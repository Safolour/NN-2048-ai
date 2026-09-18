你现在负责：

# M1 Final Audit Fix：Legal / Terminal Overflow Isolation

这是一个**极小范围的 M1 最终审计补丁**。

当前项目已经完成：

```text
M0:
m0-reference-pass
→ 3f2def1d95f56eff776e671143188947bf64485b

M1 vectorization snapshot:
m1-fastenv-pass
→ a34e56b36a2fdf3e9b1eb66c0b3ea1afeeb5561e

M1 int64 aggregation fix:
m1-fastenv-final-pass
→ ce51f15e423f49f611c7703f2b9e4e6175a5f377
```

当前 M1 已经确认：

```text
movement vectorization        PASS
N-dependent Python loop       NONE
row reward overflow           FIXED
board reward overflow         FIXED
score overflow                FIXED
step reward-overflow atomicity FIXED
422 tests                     PASS
M0 frozen                     PASS
M2                            NOT STARTED
```

不要推翻这些成果。

本任务只解决最后一次代码审查发现的几个小范围问题。

---

# 0. 当前最终状态

当前不得视为真正封板。

固定状态：

```text
M1 core movement correctness       PASS
M1 vectorization                   PASS
M1 reward overflow fix             PASS
M1 legal/terminal isolation        INCOMPLETE
M1 API live-view regression        INCOMPLETE

M1 overall:
PASS WITH FIXES
```

只有完成本提示词以后：

才能重新宣布：

```text
M1 = FINAL AUDITED PASS
```

---

# 1. 本次只允许解决四件事

本任务只能处理：

```text
A. legal_mask_batch / is_terminal_batch
   不得因为 reward 超出 int64 而失败

B. Fast2048BatchEnv.scores
   必须恢复真正的 read-only live-view 语义

C. boundary fuzz
   删除错误放宽的 false-positive 条件

D. M1_REPORT
   修正旧 4096 benchmark 的 16 steps / 新版 48 steps 口径矛盾
```

除此之外：

**不要自由发挥。**

---

# 2. 绝对禁止范围扩张

禁止：

```text
重写 movement architecture
重写 _pack_left_rows
重写 canonical movement
改变 _LINE_ORDER
改变 _LINE_INVERSE
重新设计 FastEnv API
修改 spawn 算法
修改 RNG 算法
修改 reset
修改 reset_where
修改 Action
修改 board encoding
修改 D4
修改 M0

C++
pybind11
Cython
Numba
Triton
CUDA
Rust

PyTorch
Transformer
MLP
M2
Teacher
Expectimax
Replay
Self-play
tuple checkpoint
```

这不是性能优化阶段。

不要借机继续压榨 FastEnv。

---

# 3. M0 继续永久冻结

以下文件禁止修改：

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

禁止移动：

```text
m0-reference-pass
```

---

# 4. 两个旧 M1 tag 都禁止移动

禁止移动或删除：

```text
m1-fastenv-pass
m1-fastenv-final-pass
```

它们是历史 snapshot。

本轮最终通过后创建新的：

```text
m1-fastenv-audited-pass
```

不要覆盖旧 tag。

---

# 5. 开始前固定检查

首先执行：

```bash
git status
git rev-parse HEAD
git rev-parse m0-reference-pass^{commit}
git rev-parse m1-fastenv-pass^{commit}
git rev-parse m1-fastenv-final-pass^{commit}
```

如果当前仓库没有后续提交，

预期：

```text
HEAD
= ce51f15e423f49f611c7703f2b9e4e6175a5f377

m0-reference-pass
= 3f2def1d95f56eff776e671143188947bf64485b

m1-fastenv-pass
= a34e56b36a2fdf3e9b1eb66c0b3ea1afeeb5561e

m1-fastenv-final-pass
= ce51f15e423f49f611c7703f2b9e4e6175a5f377
```

如果 HEAD 已经比 ce51f15 更新：

不要 reset。

在当前最新 HEAD 上继续。

禁止：

```bash
git reset --hard
git rebase
git push --force
```

---

# 6. 第一个核心 bug：legal mask 被 reward 上限污染

当前：

```python
legal_mask_batch(...)
```

通过：

```text
_move_all_actions_batch
→ _move_groups
→ _merge_left_rows
→ reward int64 checks
```

来判断：

```text
moved
```

问题：

legal mask 根本不需要 reward。

但是现在只要某个方向的真实 reward：

```text
> INT64_MAX
```

就可能：

```python
legal_mask_batch(...)
```

直接抛：

```python
OverflowError
```

这是错误的。

---

# 7. legal 的正式语义

M0 冻结定义：

```text
legal[a]
⇔
move_without_spawn(board, a).moved
```

因此 legal 只回答：

> 这个动作会不会改变棋盘？

它不回答：

> 这个动作的 reward 能不能塞进 M1 的 int64？

这两个问题必须分开。

---

# 8. 必须复现当前 bug

修代码以前先新增或临时运行以下 reproduction：

```text
61 61 61 61
 0  0  0  0
 0  0  0  0
 0  0  0  0
```

调用：

```python
legal_mask_batch(board[None, :])
```

M0：

```python
legal_mask(board)
```

可以正常返回。

当前 M1 如果：

```text
OverflowError
```

即证明 bug。

必须把 reproduction 记录进：

```text
M1_REPORT.md
```

---

# 9. is_terminal 同样必须与 reward 解耦

定义仍然：

```python
is_terminal_batch(boards)
=
~legal_mask_batch(boards).any(axis=1)
```

因此：

`is_terminal_batch`

也不得因为：

```text
reward > INT64_MAX
```

失败。

---

# 10. 不允许通过写第二套 legal 规则修复

严格禁止改成：

```text
if empty cell exists:
    legal = ...
```

或者：

```text
check adjacent equal tile
```

这种独立 shortcut。

原规则保持：

```text
legal
来自真实 movement semantics
```

只是不计算不需要的 reward。

---

# 11. 正确实现方向固定：movement core 增加内部 reward mode

推荐并要求采用：

**同一个 movement core，两种内部模式。**

不要复制第二套 movement 实现。

推荐接口：

```python
_move_groups(
    boards,
    group_actions,
    *,
    compute_rewards: bool,
)
```

或者语义完全等价的内部参数。

公共 API：

**禁止改变。**

---

# 12. compute_rewards=True 的语义

以下调用：

```python
move_batch(...)
Fast2048BatchEnv.step(...)
```

必须继续：

```text
compute_rewards = True
```

并保持现在所有规则：

```text
single merge int64 check
row aggregate int64 check
board aggregate int64 check
score aggregate int64 check
```

一个检查都不能删。

---

# 13. compute_rewards=False 的语义

以下调用：

```python
legal_mask_batch(...)
is_terminal_batch(...)
```

必须使用：

```text
compute_rewards = False
```

在该模式中：

仍然必须真正执行：

```text
pack
merge
pack
afterstate / moved
```

但是：

不得：

```text
建立 merge reward table lookup
累计 row reward
累计 board reward
进行 int64 reward overflow check
```

因为：

这些结果根本不会被使用。

---

# 14. compute_rewards=False 不能关闭 uint8 tile overflow

非常重要。

只是不计算：

**reward。**

不是：

“不检查任何 overflow”。

例如：

```text
255 + 255
```

需要产生：

```text
exponent 256
```

无法放入：

```text
uint8
```

M0：

对此会：

```python
OverflowError
```

因此 M1 legal movement 路径：

仍然必须保持相同的：

**tile exponent overflow 语义。**

---

# 15. 必须拆开两种 overflow

当前 `_audit_merge_overflow()` 把：

```text
A. uint8 exponent overflow
B. int64 reward overflow
```

放在一起。

本次允许做最小拆分。

推荐：

```python
_audit_tile_merge_overflow(...)
```

只负责：

```text
255 + 255 -> 256
```

这种 tile representation overflow。

reward overflow：

只在：

```text
compute_rewards=True
```

路径中检查。

不要建立复杂 class。

不要过度抽象。

---

# 16. legal-only movement 必须允许高 reward merge

例如：

```text
61 61 61 61
```

LEFT。

真实 reward：

```text
2**63
```

M1 `move_batch`：

仍然必须：

```python
OverflowError
```

但是：

```python
legal_mask_batch
```

必须：

正常返回：

```text
LEFT = legal
RIGHT = legal
...
```

与 M0 exact equality。

---

# 17. legal-only movement 还必须允许 exponent 62..254 merge

例如：

```text
100 100 0 0
```

LEFT。

M0 可以生成：

```text
101 0 0 0
```

reward 很大，

但 legal 本身完全可定义。

因此：

```python
legal_mask_batch(...)
```

必须正常。

同理：

```python
is_terminal_batch(...)
```

必须正常。

只要：

tile exponent 本身没有超过 uint8。

---

# 18. 255+255 行为不能被错误放开

例如：

```text
255 255 0 0
```

LEFT。

M0 movement：

必须 OverflowError。

因此：

M1 的 movement-based legality：

不得静默：

```text
255+255 -> 0
```

也不得 wrap。

必须保持：

```python
OverflowError
```

这条测试必须新增。

---

# 19. `_merge_left_rows` 的推荐改法

推荐改为：

```python
_merge_left_rows(
    packed,
    *,
    compute_rewards: bool,
)
```

无论：

```text
True / False
```

都执行：

```text
merge semantics
tile overflow safety
```

只有：

```text
compute_rewards=True
```

时才：

```text
查 _MERGE_REWARD
checked row reward accumulation
```

如果：

```text
compute_rewards=False
```

则：

完全跳过 reward arithmetic。

---

# 20. 不要返回 object reward

即使：

```text
compute_rewards=False
```

也禁止：

Python int / dtype object。

内部如果函数签名必须返回 reward：

允许返回：

```text
None
```

或者：

```text
全零 int64
```

但：

不要为了兼容强行执行无用 reward 计算。

推荐：

返回：

```python
merged, reward_or_none
```

其中：

```text
compute_rewards=False
→ reward_or_none = None
```

这是内部函数。

不影响公共 API。

---

# 21. `_move_groups` 必须继续只有固定常量循环

修复以后：

仍然禁止：

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

允许：

```python
for action in range(4)
```

允许：

```python
for _column in range(3)
```

允许：

```python
for column in range(4)
```

但最后这个：

board reward aggregation

只有：

```text
compute_rewards=True
```

才需要执行。

---

# 22. `_move_all_actions_batch` 必须改为 reward-free

当前：

```python
_move_all_actions_batch
```

会保存：

```text
moved
rewards
```

但：

`legal_mask_batch`

和：

`is_terminal_batch`

根本不用 reward。

因此建议固定改成：

```text
_move_all_actions_batch
→ 只需要 moved
```

或者：

仍返回 tuple，

但不计算真实 reward。

禁止为了保持旧内部返回值：

继续执行 int64 reward path。

---

# 23. legal_mask_batch 公共 API 禁止变化

仍然：

```python
legal_mask_batch(
    boards
) -> np.ndarray
```

固定：

```text
shape = (N,4)
dtype = bool
order = UP, DOWN, LEFT, RIGHT
```

---

# 24. is_terminal_batch 公共 API 禁止变化

仍然：

```python
is_terminal_batch(
    boards
) -> np.ndarray
```

固定：

```text
shape = (N,)
dtype = bool
```

---

# 25. 必须新增 legal overflow isolation 测试

至少新增：

```python
def test_legal_mask_does_not_depend_on_int64_reward_range():
```

棋盘：

```text
61 61 61 61
 0  0  0  0
 0  0  0  0
 0  0  0  0
```

要求：

```python
fast = legal_mask_batch(board[None, :])[0]
reference = legal_mask(board)

assert np.array_equal(fast, reference)
```

并且：

**不得 OverflowError。**

---

# 26. 必须新增 high exponent legal 测试

至少使用：

```text
100 100 0 0
```

和：

```text
254 254 0 0
```

验证：

```python
legal_mask_batch
```

与：

```python
M0 legal_mask
```

一致。

因为：

reward 无法进入 int64，

但 afterstate exponent：

```text
101
255
```

仍然能进入 uint8。

---

# 27. 必须新增 255+255 representation overflow 测试

使用：

```text
255 255 0 0
```

验证：

M0：

```text
OverflowError
```

M1：

同样：

```text
OverflowError
```

即使走：

```text
legal_mask_batch
```

也不能 silent wrap。

---

# 28. 必须新增 terminal isolation 测试

对：

```text
61 61 61 61
0 0 0 0
0 0 0 0
0 0 0 0
```

运行：

```python
is_terminal_batch(...)
```

与：

```python
M0 is_terminal(...)
```

比较。

必须：

exact equality。

不得：

OverflowError。

---

# 29. 必须新增 step 的关键回归测试

这是本轮最重要测试之一。

初始 board：

```text
61 61 61 61
0  0  0  0
0  0  0  0
0  0  0  0
```

执行：

```text
DOWN
```

这个动作：

```text
legal = True
reward = 0
```

本身不存在 int64 reward overflow。

它会把四个 61：

移动到底部。

然后正常：

spawn。

step 最后需要：

terminal calculation。

旧 bug 下：

terminal 检查可能因为 LEFT/RIGHT 假想动作的 reward：

```text
2**63
```

而抛异常。

修复后：

```python
result = env.step(DOWN)
```

必须完整成功。

---

# 30. 上述 step 测试必须检查

至少验证：

```text
result.legal == True
result.reward == 0
spawn_index >= 0
spawn_exponent in {1,2}
result.terminated 与 M0 对正式 state 的 is_terminal 一致
```

并验证：

```python
legal_mask_batch(result.states)
```

可以正常运行。

---

# 31. move_batch reward overflow 仍必须保持

同样的：

```text
61 61 61 61
```

执行：

```text
LEFT
```

时：

```python
move_batch(...)
```

仍必须：

```text
OverflowError
```

不能因为 legal path 解耦：

把真实 reward overflow 检查删除。

---

# 32. 原 31 个 overflow tests 全部保留

当前：

```text
tests/test_m1_reward_overflow.py
```

已有：

31 tests。

禁止删除。

禁止降低。

新增 legal/terminal isolation tests：

可以继续放：

```text
test_m1_reward_overflow.py
```

或者新增：

```text
test_m1_legal_overflow_isolation.py
```

优先：

直接放现有文件，

避免再拆模块。

---

# 33. 第二个 bug：scores live-view 被破坏

当前 `scores` property 的文档承诺：

```text
read-only live view
```

例如：

```python
view = env.scores
```

之后：

```python
env.step(...)
```

`view`

应该仍然反映：

最新 score。

---

# 34. 当前问题

修复 score overflow 后：

`step()` 现在执行：

```python
self._scores = new_scores
```

这会：

替换 backing ndarray。

因此旧：

```python
view = env.scores
```

会继续指向：

旧 array。

不再 live。

---

# 35. scores 修复固定

必须改成：

```python
self._scores[:] = new_scores
```

禁止：

```python
self._scores = new_scores
```

这样：

```text
backing ndarray object
```

保持不变。

---

# 36. scores 修复不得破坏 atomicity

顺序继续保持：

```text
move
↓
所有 reward / score range check
↓
spawn
↓
self._scores[:] = new_scores
↓
terminal
```

不要把：

score commit

提前到：

spawn 前。

当前 int64 overflow 检查：

仍然必须在任何 state mutation 前完成。

---

# 37. 必须新增 scores live-view 测试

固定：

```python
env = Fast2048BatchEnv(...)
env._boards[...] = scoring_board

backing = env._scores
view = env.scores

env.step(scoring_action)

assert env._scores is backing
assert np.shares_memory(view, env._scores)
assert view[0] == env.scores[0]
assert view[0] > 0
```

同时验证：

```python
view.flags.writeable == False
```

---

# 38. reset / reset_where 也必须保持 live-view

同一个：

```python
view = env.scores
```

依次执行：

```text
step
reset_where
reset
```

以后：

view

都必须指向：

同一个 backing `_scores`。

至少增加一个综合测试。

---

# 39. 第三个问题：boundary fuzz 错误放宽

当前：

```python
except OverflowError:
    assert m0_unrepresentable or int(entry.max()) < 62
```

这是不允许的。

它会让：

某些 M0 明明可表示的 case

被 M1 错误拒绝，

但测试仍然 PASS。

---

# 40. fuzz 断言固定修复

必须改为：

```python
except OverflowError:
    raised += 1
    assert m0_unrepresentable, (
        f"{context}: M1 raised although M0 represents "
        "the transition exactly"
    )
```

不得保留任何：

```text
or entry.max() < ...
```

例外。

---

# 41. fuzz 的正式规则

对于每个随机 case：

如果 M0：

```text
afterstate 可表示
且
reward <= INT64_MAX
```

则：

M1：

**必须返回 exact result。**

如果 M0：

```text
reward > INT64_MAX
```

或者：

tile result 超出 M1/M0 board representation

则：

M1：

允许 / 必须：

```python
OverflowError
```

不能存在第三种：

“反正 exponent 不高，所以随便允许异常”。

---

# 42. 第四个问题：M1_REPORT benchmark 文案

当前报告仍有一处类似：

```text
14,826 -> 73,571
两侧均为 4096 × 48
```

但：

保存的旧：

```text
baseline_before_vectorization.json
```

4096 档实际上：

```text
4096 × 16
```

而新版：

```text
4096 × 48
```

---

# 43. 报告固定修正

不得修改：

原始 JSON。

只改文字。

正确写法：

```text
OLD:
4096 env × 16 steps

NEW:
4096 env × 48 steps

两侧使用相同环境规模、seed 和 benchmark 逻辑，
但正式记录的 step count 不同，因此该 throughput 比用于证明
长期 steady-state 吞吐提升，不属于严格逐指令 A/B。

严格同进程 A/B 见 benchmark_ab_vectorization.py。
```

---

# 44. 不要修改旧 benchmark 数字

禁止：

为了让报告漂亮：

重生成旧 JSON。

保留：

```text
baseline_before_vectorization.json
benchmark_results.json
ab_vectorization.json
sanity_after_final.json
sanity_after_primitives.json
```

历史证据不动。

---

# 45. 允许修改的文件范围

主要允许：

```text
src/game2048/fast_env.py

tests/test_m1_reward_overflow.py
tests/test_m1_batch_env.py
tests/test_m1_legal_terminal.py
tests/test_m1_vectorization.py

docs/M1_FAST_ENV_SPEC.md
M1_REPORT.md
```

不必每个都修改。

只修改真正需要的。

禁止碰：

M0 frozen files。

---

# 46. 必须保持公共 API 完全不变

禁止改变：

```text
move_batch
legal_mask_batch
is_terminal_batch
enumerate_spawns_batch
spawn_random_batch
apply_spawn_batch
Fast2048BatchEnv
BatchMoveResult
BatchSpawnEnumeration
BatchSpawnResult
BatchStepResult
```

函数名、参数和返回语义：

全部保持。

---

# 47. vectorization 防回退测试必须继续通过

当前：

```text
test_m1_vectorization.py
```

中的：

```text
sys.settrace scaling guard
AST loop guard
M0 monkeypatch isolation
8192 batch test
```

全部继续保留。

如果内部增加：

```text
compute_rewards
```

不能让 AST guard 误判固定循环。

必要时：

只允许最小修改：

allowed fixed-loop variable 名单。

禁止：

放宽成：

“什么 loop 都可以”。

---

# 48. legal-only path 仍然不得调用 M0

生产：

```python
legal_mask_batch
```

禁止：

逐 board 调：

```python
move_without_spawn
```

M0 只能用于：

tests。

---

# 49. 修改顺序固定

严格按下面顺序施工：

```text
STEP 1
运行当前 python -m pytest
确认修改前 baseline 全绿

STEP 2
新增 reproducer：
legal_mask_batch([61,61,61,61]) 当前问题

STEP 3
新增 step(DOWN) reproducer

STEP 4
重构内部 movement core：
增加 compute_rewards 模式
不改变 public API

STEP 5
让 legal_mask_batch / is_terminal_batch
使用 compute_rewards=False

STEP 6
确认 move_batch / step
仍然 compute_rewards=True

STEP 7
确保 no-reward path 仍然检查 255+255 tile overflow

STEP 8
修 self._scores = new_scores
为 self._scores[:] = new_scores

STEP 9
收紧 boundary fuzz assertion

STEP 10
修 M1_REPORT benchmark 16/48 wording

STEP 11
更新 M1_FAST_ENV_SPEC

STEP 12
运行 targeted tests

STEP 13
运行完整 python -m pytest

STEP 14
运行最小 performance sanity

STEP 15
检查 M0 frozen files diff

STEP 16
更新 M1_REPORT 最终审计章节

STEP 17
commit

STEP 18
创建 m1-fastenv-audited-pass

STEP 19
push

STEP 20
立即停止，不开始 M2
```

不要改变顺序。

---

# 50. Targeted tests 固定

完整 pytest 之前至少单独运行：

```text
test_m1_reward_overflow.py
test_m1_batch_env.py
test_m1_legal_terminal.py
test_m1_vectorization.py
test_m1_high_tiles.py
```

如果真实文件名不同：

使用当前对应文件。

---

# 51. 完整 correctness baseline

当前：

```text
422 passed
```

修复以后：

```text
total >= 422
failed = 0
skip = 0
xfail = 0
```

新测试只会增加。

不能减少。

---

# 52. Differential 全部继续保留

最终仍然必须保持：

```text
ordinary       40,000       0 mismatch
high-tile       8,000       0 mismatch
legal          10,000       0 mismatch
terminal       10,000       0 mismatch
spawn           2,000       0 mismatch
D4             64,000       0 mismatch
8192 batch        512       0 mismatch
boundary fuzz    6,300      0 mismatch / false reject
```

---

# 53. 新增 high-reward legal differential

额外构造一批：

```text
exponents concentrated in 60..254
```

但避开：

```text
255+255
```

导致的 tile overflow。

至少：

```text
500 boards
```

比较：

```text
legal_mask_batch
vs
M0 legal_mask
```

要求：

```text
0 mismatch
0 reward-related OverflowError
```

目的：

证明 legal 和 reward int64 range 已真正解耦。

---

# 54. 性能不需要重做整套 M1

本次只修改：

legal/terminal internal path

和：

score backing array。

不要重跑：

完整 worker sweep。

只需要：

```text
move_batch primitive
legal_mask_batch primitive
is_terminal_batch primitive
4096-env sanity
```

---

# 55. 性能 baseline

当前记录大致：

```text
move_batch:
~988,751 boards/s

legal_mask_batch:
~240,577 boards/s

is_terminal_batch:
~239,918 boards/s

4096 env:
~84,990 transitions/s
```

这些只是参考。

由于机器有抖动：

不要以 1～5% 波动判失败。

---

# 56. 性能验收

同一当前机器环境下：

修改前先运行一次 baseline。

修改后：

使用相同命令再跑。

如果：

```text
legal_mask_batch
is_terminal_batch
4096 env
```

下降：

```text
> 10%
```

先 profile。

不能直接接受。

理论上：

reward-free legal path

应该：

相同或更快，

而不是明显变慢。

---

# 57. 禁止为了性能重新加入 reward path

即使 legal-only path benchmark 变慢：

也禁止：

重新打开 int64 reward calculation。

先检查：

- 不必要 allocations；
- 不必要 return reward arrays；
- 重复 movement；
- 新增 mask copies。

但：

不要扩大本次工程范围。

---

# 58. M1_FAST_ENV_SPEC 必须明确新内部语义

文档增加：

```text
Movement has two internal modes:

reward-producing mode:
    move_batch / step
    full int64 reward contract

movement-only mode:
    legal_mask_batch / is_terminal_batch
    same tile movement semantics
    no reward arithmetic
    still enforces uint8 tile exponent overflow
```

并明确：

```text
legal legality is independent of M1 int64 reward representation.
```

---

# 59. step atomicity 文档必须准确

原先：

```text
reward / score overflow
```

atomicity：

继续成立。

新增说明：

terminal calculation

不再因为：

```text
hypothetical alternative action reward > INT64_MAX
```

导致 post-spawn exception。

不要扩大成：

“任何宇宙中的异常都绝对 atomic”。

只写真实保证。

---

# 60. M1_REPORT 新增最终审计章节

新增：

```text
## T. Final Audit: Legal/Terminal Reward Isolation
```

内容至少记录：

```text
Audit finding 1:
legal_mask_batch/is_terminal_batch were coupled to int64 reward computation.

Reproducer:
[61,61,61,61] legal query

Fix:
shared movement core gained movement-only mode;
reward checks remain enabled only for reward-producing APIs.

Audit finding 2:
self._scores = new_scores replaced the backing ndarray
and broke the documented live-view property.

Fix:
self._scores[:] = new_scores

Audit finding 3:
boundary fuzz contained an overly-permissive false-positive assertion.

Fix:
assert m0_unrepresentable strictly.

Audit finding 4:
4096 OLD/NEW step count wording inconsistent.

Fix:
OLD=16, NEW=48 explicitly documented.

Tests:
...

Performance:
...

M0 modified:
NO

M2 started:
NO

Final result:
M1 = FINAL AUDITED PASS
```

---

# 61. 必须修改报告顶部状态

如果本轮全部通过：

报告顶部写：

```text
M1 = FINAL AUDITED PASS
```

并说明：

```text
The authoritative final M1 tag is:
m1-fastenv-audited-pass
```

旧：

```text
m1-fastenv-pass
m1-fastenv-final-pass
```

只作为历史 snapshot。

---

# 62. 新 tag 固定

所有测试和 sanity 通过后：

commit message 固定建议：

```bash
git commit -m "Fix M1 legal-terminal reward isolation"
```

然后：

```bash
git tag -a m1-fastenv-audited-pass \
  -m "M1 FastEnv final audited pass"
```

然后：

```bash
git push
git push origin m1-fastenv-audited-pass
```

---

# 63. 禁止移动已有 tags

最终再次验证：

```bash
git rev-parse m0-reference-pass^{commit}
git rev-parse m1-fastenv-pass^{commit}
git rev-parse m1-fastenv-final-pass^{commit}
git rev-parse m1-fastenv-audited-pass^{commit}
```

前三个：

必须保持原 commit。

---

# 64. 最终 Exit Criteria

只有下面全部满足：

才能：

```text
M1 = FINAL AUDITED PASS
```

```text
[ ] M0 261 tests 仍全绿
[ ] 总测试数 >= 422
[ ] 0 failed
[ ] 0 skipped
[ ] 0 xfailed

[ ] legal_mask_batch 不因 int64 reward overflow 失败
[ ] is_terminal_batch 不因 int64 reward overflow 失败
[ ] exp 100/254 merge legality 与 M0 一致
[ ] 255+255 tile overflow 仍显式异常

[ ] move_batch reward overflow contract 未改变
[ ] row reward checked-add 未删除
[ ] board reward checked-add 未删除
[ ] score checked-add 未删除

[ ] step(DOWN) 高 reward-alternative board 能完整成功
[ ] terminal 在 spawn 后正常计算
[ ] 不发生 post-spawn int64 reward exception

[ ] scores backing ndarray 在 step 前后不被替换
[ ] scores live view 在 step 后仍实时更新
[ ] scores live view 在 reset_where 后仍实时更新
[ ] scores live view 在 reset 后仍实时更新
[ ] scores view 仍 read-only

[ ] boundary fuzz 只接受真正 m0_unrepresentable 的 OverflowError
[ ] 6,300 boundary fuzz 继续通过
[ ] high-reward legal differential >=500 boards，0 mismatch

[ ] movement production path 仍无 N-dependent Python loop
[ ] np.vectorize 未引入
[ ] np.apply_along_axis 未引入
[ ] production path 不调用 M0 scalar movement

[ ] legal_mask / terminal 仍来自 movement semantics
[ ] 没有建立第二套 adjacency legality 规则

[ ] move_batch primitive sanity 完成
[ ] legal_mask primitive sanity 完成
[ ] terminal primitive sanity 完成
[ ] 4096-env sanity 完成
[ ] 无 >10% 无解释性能倒退

[ ] M1_FAST_ENV_SPEC 已更新
[ ] M1_REPORT §T 已更新
[ ] benchmark 16/48 文案已纠正

[ ] M0 frozen files 未修改
[ ] m0-reference-pass 未移动
[ ] m1-fastenv-pass 未移动
[ ] m1-fastenv-final-pass 未移动

[ ] M2 未开始
[ ] PyTorch 未安装
[ ] tuple checkpoint 未使用
```

任意 correctness 项失败：

```text
M1 = FAIL
```

不得写：

```text
PASS WITH KNOWN ISSUE
```

---

# 65. 最终回复格式固定

最终只回复：

```text
1. Final status
M1 = FINAL AUDITED PASS / FAIL

2. Legal/terminal isolation
legal_mask reward-independent:
is_terminal reward-independent:
255+255 tile overflow preserved:

3. Critical reproducer
[61,61,61,61] legal_mask:
[61,61,61,61] is_terminal:
DOWN step on high-reward-alternative board:

4. scores live-view
backing ndarray preserved:
old view updates after step:
after reset_where:
after reset:
read-only:

5. Reward overflow
row check:
board check:
score check:
all still enabled:

6. Boundary fuzz
cases:
false rejects:
false accepts:
failures:

7. Differential
ordinary:
high-tile:
legal:
terminal:
spawn:
D4:
high-reward legal:
mismatch:

8. pytest
total:
passed:
failed:
skipped:
xfailed:

9. Vectorization
N-dependent Python loop:
must be NONE

10. Performance sanity
metric              BEFORE    AFTER    CHANGE
move_batch
legal_mask_batch
is_terminal_batch
4096 env

11. Documentation
M1_FAST_ENV_SPEC:
M1_REPORT:

12. Git
new commit:
new tag:
old m0 tag unchanged:
old m1-fastenv-pass unchanged:
old m1-fastenv-final-pass unchanged:

13. M0 modified?
NO

14. M2 started?
NO

15. Known issues
None / exact remaining blocker
```

不要输出无关长文。

---

# 66. 最终停止规则

一旦：

```text
m1-fastenv-audited-pass
```

创建完成，

立即停止。

禁止：

```text
开始 M2
安装 PyTorch
测试 RTX 5060
继续优化 FastEnv
上 C++
接 tuple 8×6 checkpoint
写网络
写 Teacher
```

等待用户下一条明确命令。

---

# 67. 一句话任务定义

你这次唯一要做的是：

> **让 legal/terminal 复用同一套真实 movement 语义，但完全脱离 int64 reward 表示限制；恢复 scores 的真正 live-view；收紧 boundary fuzz；纠正 benchmark 文案。保持现有 reward overflow、vectorization、spawn、RNG、M0 和全部公共 API 不变。**

不要做除此之外的任何设计。