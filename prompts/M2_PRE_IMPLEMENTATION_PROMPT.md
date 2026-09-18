你现在接手一个连续开发中的 2048 AI 项目。

GitHub 仓库：

`https://github.com/Safolour/NN-2048-ai`

当前对话中应同时提供一份最新的 **唯一权威总计划**。

你必须先完整理解其中与以下内容有关的条款：

- 当前 M0 / M1 冻结状态；
- 性能工程总规范；
- GitHub Actions / CI 长期规范；
- 当前执行顺序；
- 第 99 节 M2；
- M2 Preflight；
- M2 正式 Exit Criteria；
- C++ / profiling 决策原则。

## 极重要

**总计划是项目最高层唯一权威规范。**

本提示词是：

**M2 Preflight 的详细施工单。**

两者关系固定为：

```text
总计划
    ↓
定义项目路线、冻结语义和 milestone 要求

本提示词
    ↓
把 M2 Preflight 的具体实施方式、修改范围、测试方式和禁止事项锁死
```

你不得：

- 绕过总计划；
- 重新设计项目路线；
- 根据自己的偏好修改总计划；
- 把本提示词理解为允许你推翻总计划。

如果：

**仓库实际状态 / 本提示词 / 最新总计划**

之间存在真实冲突：

**停止施工并报告冲突。**

不得自行裁决。

如果当前对话中根本没有最新唯一权威总计划文件：

**不要开始施工。**

直接报告：

`BLOCKED: authoritative master plan is missing`

不要根据本提示词猜整个项目。

---

# 一、当前正式冻结状态

以下状态必须首先通过 GitHub 实际检查验证。

## M0

authoritative tag：

```text
m0-reference-pass
```

commit：

```text
3f2def1d95f56eff776e671143188947bf64485b
```

M0：

**已完成并永久冻结。**

`Reference2048Env` 是长期 Golden Reference / Correctness Oracle。

不得为了后续实现方便修改 M0 游戏规则。

---

## M1

authoritative tag：

```text
m1-fastenv-audited-pass
```

commit：

```text
e5486017a90eeec4fb9814de7880b3c413dc06dd
```

正式最终测试：

```text
432 passed
0 failed
0 skipped
0 xfailed
```

M1：

**FINAL AUDITED PASS / FROZEN**

已经完成：

```text
Fast / Vectorized NumPy Environment

move_batch
legal_mask_batch
is_terminal_batch
enumerate_spawns_batch
spawn_random_batch
apply_spawn_batch
Fast2048BatchEnv

ordinary differential
high-tile differential
legal differential
terminal differential
spawn differential
D4 differential
overflow regression
atomicity regression
vectorization regression

scalability benchmark
worker scaling
producer→consumer benchmark
profiling
```

production movement hot path：

```text
N-dependent Python per-board loop = NONE
```

M1 不重新执行。

M1 authoritative tag：

**绝对不能移动。**

---

# 二、M0 / M1 representation divergence 已经正式审计

不得重新争论、重新设计或“修复”为另一套东西。

正式规则：

```text
M0 reward:
arbitrary precision Python int

M1 reward / score:
np.int64
```

如果真实：

```text
single-action reward
```

或者：

```text
running score + reward
```

无法非负、精确地表示在：

```text
0 .. INT64_MAX
```

中：

M1 必须：

```python
raise OverflowError
```

禁止：

- silent wrap；
- 负数回绕；
- 截断；
- dtype=object；
- per-board Python int fallback。

但：

**reward 不可表示不得污染 movement legality。**

所以：

```text
legal_mask
terminal
movement legality
```

继续使用 reward-independent movement path。

另外：

```text
255 + 255 -> exponent 256
```

属于 tile representation overflow，不属于 reward exception。

该行为继续遵守冻结 M0 / M1 规则。

本任务禁止重新实现这一部分。

---

# 三、现阶段是什么

当前：

```text
M0 = frozen
M1 = frozen
M2 = 尚未正式开始
```

下一步固定为：

```text
M2 Preflight
```

不是：

```text
直接写 Transformer
```

也不是：

```text
直接写 Residual MLP
```

M2 Preflight 完成并重新证明：

```text
完整 M0 + M1 tests 全绿
+
GitHub Actions 全绿
```

以后：

才允许进入正式 M2 network implementation。

---

# 四、本次任务严格只有 M2 Preflight

M2 Preflight 固定四项，顺序固定：

```text
1. 建立 GitHub Actions CPU CI

2. 清理 Fast2048BatchEnv._terminated
   没有实际复用价值的假 cache
   以及相关误导注释

3. 修复 tests/test_m1_batch_env.py 中
   assert np.any(env.scores != 0) or True
   这个永远为真的无效断言

4. 冻结 M2 CPU→GPU C-contiguous batch 数据契约
```

四项完成以后：

```text
focused tests
↓
完整 python -m pytest
↓
GitHub Actions 实际验证
↓
停止
```

**不要继续正式 M2 网络实现。**

---

# 五、开工前仓库审计

在修改任何文件前必须先检查并记录：

```bash
git status --short
git rev-parse HEAD
git rev-parse m0-reference-pass^{commit}
git rev-parse m1-fastenv-audited-pass^{commit}
```

还必须确认：

```text
m0-reference-pass
=
3f2def1d95f56eff776e671143188947bf64485b
```

以及：

```text
m1-fastenv-audited-pass
=
e5486017a90eeec4fb9814de7880b3c413dc06dd
```

检查当前默认开发分支相对 frozen M1 的状态。

如果当前 main / working HEAD 仍然就是 frozen M1：

正常。

如果已经存在后续提交：

先审查这些提交是什么。

不得盲目覆盖。

---

# 六、保护用户已有工作

如果：

```bash
git status --short
```

显示已有未提交修改：

禁止：

```text
git reset --hard
git clean
git checkout .
git restore .
自动 stash
覆盖用户修改
删除未知文件
```

不得破坏用户工作。

如果已有修改与本任务发生冲突：

停止并报告。

---

# 七、禁止事项——这是硬约束

本任务期间禁止实现：

```text
Transformer
Residual MLP
Q Head
Value Head
Afterstate Value Head

正式训练器
optimizer pipeline
Teacher
tuple evaluator
Expectimax
Replay Buffer
Self-play
Double-Q
EMA Target
Champion
Search Correction

GPU throughput benchmark
batch-size GPU sweep
BF16 / FP16 benchmark
torch.compile benchmark
M1+M2 GPU closed-loop benchmark

C++
pybind11
CUDA FastEnv
SIMD 新实现
LUT 新环境
movement rewrite
```

用户已有：

```text
tuple 8×6 checkpoint
```

M2 Preflight：

**完全不使用。**

它留给后续 Teacher 阶段。

---

# 八、禁止“顺手优化”

你不是来重构项目的。

禁止因为：

```text
顺便整理代码
这样更优雅
以后可能需要
这样理论上更快
我觉得架构应该这样
```

而扩展 scope。

性能优化原则仍然是：

```text
正确
↓
测量
↓
找最大真实瓶颈
↓
优化最大瓶颈
↓
重新 benchmark
```

不是凭感觉提前换技术栈。

---

# 九、Preflight 1：建立 GitHub Actions CPU CI

创建：

```text
.github/workflows/ci.yml
```

第一版保持简单。

不得建立复杂 workflow 网络。

---

## CI Trigger

至少：

```yaml
on:
  push:
  pull_request:
```

---

## Runner

第一版：

```text
ubuntu-latest
```

本任务不要建立：

- Windows matrix；
- macOS matrix；
- 大型 OS matrix。

---

## Python

第一版使用：

```text
Python 3.12
```

不要建立十几个 Python 版本的 matrix。

使用：

```text
actions/checkout@v4
actions/setup-python@v5
```

---

## CI dependencies

检查仓库现有依赖管理方式。

如果仓库已有正式 dependency manifest：

优先使用现有方式。

如果仓库目前没有完整 packaging / dependency manifest：

不要为了这一项自行建立：

```text
Poetry
uv 架构
Conda 环境体系
复杂 packaging
```

只安装完整 CPU test suite 真实所需的最小依赖。

至少通常包括：

```text
numpy
pytest
```

但必须以仓库实际测试需求为准。

---

## Import path

项目代码当前位于：

```text
src/game2048
```

必须确保 CI 可以正确 import。

如果仓库当前测试方式依赖：

```text
PYTHONPATH=src
```

则在 CI 明确配置。

不要通过修改 frozen M0/M1 package 语义来解决 CI import。

---

## CI 测试命令

最终必须运行：

```bash
python -m pytest
```

完整 suite。

禁止：

```text
只跑 M2
只跑 smoke tests
只跑 M1
排除 M0
```

M0 / M1 frozen regression：

从现在开始长期进入 CI。

---

## CI 禁止弱化 frozen tests

禁止：

```text
删除旧 test
skip
xfail
pytest --ignore
修改 pytest collection 排除 frozen suite
改配置绕过失败
```

如果 frozen test 因新改动失败：

默认先认为：

**你的修改有问题。**

---

## CI 不承担 GPU 性能验收

不要在 GitHub hosted CPU runner 上：

```text
伪造 CUDA
要求 RTX 5060
运行 GPU benchmark
运行长时间 self-play
运行 Teacher Search
```

CI 当前职责：

```text
correctness
CPU regression
API regression
frozen-stage protection
```

GPU milestone gate 以后在用户 RTX 5060 本机执行。

---

# 十、Preflight 2：清理 `_terminated` 假 cache

当前 frozen M1 中：

```python
Fast2048BatchEnv._terminated
```

存在。

你必须先实际阅读当前：

```text
src/game2048/fast_env.py
```

确认现状。

已知审计结论是：

```text
_terminated 被初始化
reset / reset_where 会置 None
step 最后会更新
_refresh_terminated 会写入
```

但是：

**下一次 step 并没有读取旧 `_terminated` 来复用结果。**

也就是说：

```text
cache write exists
cache reuse does not exist
```

同时源码注释却描述了：

```text
pipelined
cached for next step
reuse previous result
```

这是误导。

---

# 十一、`_terminated` 固定处理方式

不要让自己重新决定设计。

总计划已经偏向最小方案：

**删除假 cache。**

除非仓库当前状态已经与冻结 M1 不同且你能提出明确冲突证据，否则按以下方案执行。

---

## 从 `__init__` 删除

删除：

```python
self._terminated: Optional[np.ndarray] = None
```

以及只为这个假 cache 服务的注释。

---

## 从 `reset()` 删除

删除：

```python
self._terminated = None
```

---

## 从 `reset_where()` 删除

删除：

```python
self._terminated = None
```

---

## 删除 `_refresh_terminated()`

如果当前函数仍只是类似：

```python
self._terminated = is_terminal_batch(self._boards)
return self._terminated
```

则整个函数删除。

不要保留无意义 wrapper。

---

## `step()` 改为直接计算

当前如果是：

```python
terminated = self._refresh_terminated()
```

改为：

```python
terminated = is_terminal_batch(self._boards)
```

必须保持：

terminal 是在：

```text
move
↓
spawn
↓
正式下一 state s'
```

上计算。

不得改为 afterstate terminal。

不得改变 terminal 语义。

---

# 十二、同步修正误导注释

检查：

```text
Fast2048BatchEnv class docstring
step() docstring
_refresh_terminated docstring
附近 comments
```

删除以下虚假描述：

```text
pipelined terminal cache
reuse previous step result
cached for next step
```

新的描述只需真实反映：

```text
step 在 spawn 完成以后，
对当前正式 board batch 调用 is_terminal_batch，
生成本次 BatchStepResult.terminated。
```

不要趁机重写整个模块文档。

---

# 十三、删除假 cache 后禁止改变的语义

必须保持：

```text
illegal action:
board unchanged
reward = 0
score unchanged
no spawn
no RNG consumption

terminal:
no legal actions

legal:
reward-independent movement-only path

reward / score overflow:
explicit OverflowError

tile exponent overflow:
冻结 M0/M1 规则

scores live view:
继续有效

step atomicity:
继续有效
```

如果清理 `_terminated` 导致这些任何一项回归：

Preflight 失败。

---

# 十四、Preflight 3：修复 `or True`

目标：

```text
tests/test_m1_batch_env.py
```

当前存在：

```python
assert np.any(env.scores != 0) or True
```

这是无效测试。

必须删除。

---

# 十五、不要换成另一个随机假测试

问题来源是：

如果：

```text
reset random initial board
↓
执行 LEFT
```

不能保证一定产生 merge reward。

所以不能简单改成：

```python
assert np.any(env.scores != 0)
```

然后继续依赖随机局面。

那样测试可能 flaky 或逻辑不可靠。

---

# 十六、固定修法：确定性制造真实非零 score

修改：

```python
test_reset_zeroes_the_scores
```

使它明确：

1. 创建 Fast2048BatchEnv；
2. 人工给环境内部 board buffer 放入确定可合并棋盘；
3. 通过正常 `env.step()` 产生真实 reward；
4. 先断言 score 确实变为预期非零值；
5. 调用 `reset()`；
6. 断言全部 score 被清零。

推荐局面：

```text
[1, 1, 0, 0]
[0, 0, 0, 0]
[0, 0, 0, 0]
[0, 0, 0, 0]
```

执行：

```text
LEFT
```

结果 merge：

```text
1 + 1 -> exponent 2
reward = 4
```

因此如果所有测试 env 都使用同样局面：

可以明确验证：

```python
assert np.all(env.scores == 4)
```

随后：

```python
env.reset(seed=SEED)
assert np.all(env.scores == 0)
```

这样真正证明：

**reset 清除了由正常 step 累积出来的非零 score。**

---

# 十七、不要用直接改 `_scores` 偷懒

不要使用：

```python
env._scores[:] = 123
```

然后 reset。

这里应该通过：

```text
真实 step
```

产生非零 score。

这样测试同时证明 score accumulation 的真实路径确实发生过。

---

# 十八、旧测试修改权限非常窄

除了本条明确授权修复：

```text
test_reset_zeroes_the_scores
```

不要为了 Preflight 大规模修改 M1 tests。

禁止：

```text
减少 differential sample
放宽 equality
改 overflow expectation
改 D4 expectation
删除 test
skip
xfail
改 seed 躲问题
```

---

# 十九、Preflight 4：冻结 M2 CPU→GPU contiguous contract

M1 正式高速生产布局：

```text
shape = (N,16)
dtype = np.uint8
C-contiguous
```

M2 正式路径未来会是：

```text
FastEnv
↓
CPU board batch
↓
明确 transfer boundary
↓
CPU → GPU
↓
NN
```

Preflight 当前只定义：

**transfer boundary 之前的 CPU batch contract。**

不要开始 GPU 实现。

---

# 二十、新建专用 M2 数据边界模块

创建：

```text
src/game2048/m2_data.py
```

不要把这个逻辑塞回 M1：

```text
fast_env.py
```

M1 已冻结。

这个新模块仅负责：

```text
M2 CPU batch boundary normalization / validation
```

现在：

**不要 import torch。**

---

# 二十一、固定 API

实现：

```python
def prepare_board_batch_for_transfer(boards: np.ndarray) -> np.ndarray:
    ...
```

不要改成大型 class。

不要引入：

```text
config object
allocator abstraction
backend abstraction
device abstraction
```

本阶段没必要。

---

# 二十二、输入 contract

合法输入：

```text
type:
numpy.ndarray

shape:
(N,16)

N:
>= 1

dtype:
np.uint8
```

---

# 二十三、C-contiguous 输入必须 zero-copy

如果：

```python
boards.flags.c_contiguous
```

为：

```text
True
```

则：

直接返回同一个对象。

测试必须能够证明：

```python
result is boards
```

禁止无意义：

```python
boards.copy()
np.ascontiguousarray(boards)
```

每步复制 FastEnv 正式 batch。

---

# 二十四、non-contiguous 合法输入

如果：

```text
shape 正确
dtype = uint8
```

但：

```python
boards.flags.c_contiguous == False
```

则只在这个明确边界：

```python
np.ascontiguousarray(boards)
```

一次。

输出必须：

```text
shape 相同
dtype = uint8
数值完全相同
C_CONTIGUOUS = True
```

这就是：

**允许的唯一显式 boundary copy。**

---

# 二十五、dtype 错误不要自动修

例如：

```text
int64
int32
float32
float64
bool
```

禁止：

```python
.astype(np.uint8)
```

偷偷转换。

直接报错。

原因：

M2 hot path 不允许隐藏：

```text
dtype conversion
copy
memory re-layout
```

---

# 二十六、shape 错误不要自动 reshape

拒绝：

```text
(16,)
(N,4,4)
(N,15)
(N,17)
(0,16)
```

不要自动 reshape。

正式 contract 就是：

```text
(N,16)
N >= 1
```

---

# 二十七、函数不得修改 caller input

contiguous：

```text
返回 same object
```

non-contiguous：

```text
返回新的 contiguous array
```

无论哪一种：

输入数值不得被修改。

---

# 二十八、新增独立 Preflight tests

创建：

```text
tests/test_m2_preflight.py
```

全部 CPU-only。

不得 skip。

至少覆盖：

```text
1. valid (N,16) uint8 contiguous input accepted

2. contiguous input zero-copy
   result is input

3. non-contiguous valid uint8 input accepted

4. result of non-contiguous input is C-contiguous

5. values preserved exactly

6. invalid dtype rejected

7. invalid shapes rejected

8. empty (0,16) rejected

9. Fast2048BatchEnv.boards satisfies boundary contract

10. caller input is never mutated
```

---

# 二十九、non-contiguous fixture 必须真的不连续

例如：

```python
base = np.zeros((N, 32), dtype=np.uint8)
view = base[:, ::2]
```

此时：

```text
view.shape == (N,16)
```

测试首先必须：

```python
assert not view.flags.c_contiguous
```

然后再测试：

```python
prepare_board_batch_for_transfer(view)
```

禁止写一个所谓 non-contiguous test：

实际 fixture 却是 contiguous。

---

# 三十、`m2_data.py` 现在禁止出现这些内容

禁止：

```text
torch
torch.Tensor
CUDA
device
pin_memory
non_blocking
BF16
FP16
GradScaler
model
Transformer
MLP
Q
V
A
forward
action selection
```

当前只完成：

```text
NumPy CPU producer
→
明确 contiguous transfer boundary
```

---

# 三十一、Preflight 文档

创建：

```text
docs/M2_PREFLIGHT_SPEC.md
```

不要写成第二份总计划。

只记录已经实施的事实。

至少包括：

```text
M0 frozen tag + commit

M1 frozen tag + commit

M2 Preflight 四项

CPU CI 的职责

_terminated 假 cache 的审计结论和删除方案

or True 的测试修复原则

正式 transfer boundary:
(N,16)
uint8
C-contiguous

合法 contiguous input:
zero-copy

合法 non-contiguous input:
boundary explicit copy once

invalid dtype:
reject

invalid shape:
reject

禁止 network forward 内隐式 copy

C++ decision remains deferred

tuple 8×6 checkpoint not used
```

---

# 三十二、不要修改 M1 API 来强迫 contiguous

非常重要。

不要为了 M2 boundary contract 回头修改：

```text
move_batch
legal_mask_batch
is_terminal_batch
Fast2048BatchEnv public API
_as_boards
```

来强制所有 M1 external input 都必须 C-contiguous。

总计划已经把规则定义成：

**这是 M2 pipeline boundary 的要求。**

不是：

**重新打开 M1 API。**

---

# 三十三、静态检查

修改完成后至少运行：

```bash
git grep -n "_terminated"
git grep -n "or True"
```

判断结果。

production FastEnv 中不应继续存在：

```text
self._terminated
```

作为假 cache。

测试中不应继续存在：

```text
assert ... or True
```

如果文档为了说明历史问题提到这些字符串：

可以保留。

不要机械删除正确的文档说明。

---

# 三十四、先跑 focused tests

至少运行：

```bash
python -m pytest tests/test_m1_batch_env.py
python -m pytest tests/test_m2_preflight.py
```

如果某个文件路径 / test collection 因仓库实际结构不同：

按实际合理路径执行等价 test。

不要跳过。

---

# 三十五、然后必须跑完整 suite

最终强制：

```bash
python -m pytest
```

必须跑完整 CPU-testable suite。

要求：

```text
0 failed
0 skipped
0 xfailed
```

新增了 Preflight tests 后：

总 passed 数应该：

```text
> 432
```

完全正常。

不要为了维持原数字 432：

删除新增测试。

真正要求是：

**原冻结 432 tests 必须继续存在并全部通过。**

---

# 三十六、必须确认 M0 / M1 没被弱化

检查：

```text
M0 tests still collected
M1 tests still collected
no old tests deleted
no skip added
no xfail added
no collection exclusion added
```

任何通过弱化 frozen regression 得来的绿色结果：

**一律视为失败。**

---

# 三十七、GitHub Actions 实际验证

实现：

```text
.github/workflows/ci.yml
```

并在有权限和条件时：

实际检查 GitHub Actions run。

如果 workflow 已经 push：

必须查看对应 run。

只有：

```text
actual GitHub Actions run = PASS
```

才算 CI gate 已真正通过。

---

# 三十八、禁止伪造 CI PASS

如果当前执行环境只完成了：

```text
workflow file created
local pytest passed
```

但：

workflow 还没 push

或者：

没有产生 GitHub Actions run

或者：

你无法查看 run

则绝对禁止写：

```text
GitHub Actions PASS
```

必须写：

```text
GitHub Actions status:
NOT YET VERIFIED
```

此时整个 M2 Preflight：

不能报正式 PASS。

---

# 三十九、C++ 决策继续 DEFER

即使你看到 M1 profiling 中：

```text
movement 占比较高
```

也禁止现在迁 C++。

正式决策顺序固定：

```text
先完成 M2 网络
↓
真实运行
Env
→ CPU batch
→ GPU
→ NN
→ Action
→ Env
↓
端到端 profiling
```

只有如果真实闭环证明：

```text
CPU environment
Python
batch assembly
H2D
```

明显造成 GPU starvation：

以后才允许讨论：

```text
C++
pybind11
SIMD
LUT
worker pipeline
double buffering
pinned memory
async H2D
```

Preflight 不做。

---

# 四十、关于 commit / tag

M0 tag：

```text
m0-reference-pass
```

必须仍然指向：

```text
3f2def1d95f56eff776e671143188947bf64485b
```

M1 tag：

```text
m1-fastenv-audited-pass
```

必须仍然指向：

```text
e5486017a90eeec4fb9814de7880b3c413dc06dd
```

禁止：

```text
force tag
delete + recreate tag
move M1 tag
```

本次 HEAD 上的 cleanup：

属于：

```text
M2 Preflight work
```

不是：

```text
M1 revised pass
```

---

# 四十一、修改范围应保持非常集中

正常情况下，核心变更应主要集中在：

```text
.github/workflows/ci.yml

src/game2048/fast_env.py
仅 `_terminated` cleanup + 对应注释

tests/test_m1_batch_env.py
仅修无效 reset score test

src/game2048/m2_data.py

tests/test_m2_preflight.py

docs/M2_PREFLIGHT_SPEC.md
```

如果为了最小依赖配置确实需要新增或修改少量 CI dependency 文件：

必须解释原因。

如果发现自己改了十几个无关文件：

暂停。

检查 scope creep。

---

# 四十二、禁止修改这些冻结核心文件

除非总计划明确另有授权，否则本任务不应修改：

```text
src/game2048/reference_env.py
src/game2048/symmetry.py
```

以及其他 M0 frozen correctness 文件。

如果你认为必须修改：

不要直接改。

报告原因。

---

# 四十三、最终 diff 审计

在结束前检查：

```bash
git status --short
git diff --stat
git diff
```

确认不存在：

- 意外重构；
- 无关格式化；
- frozen semantics 修改；
- 测试弱化；
- 网络提前实现；
- C++ 提前实现。

---

# 四十四、本次不是让你宣布正式 M2 完成

即使 Preflight 全部成功：

不要继续写网络。

不要自动开始：

```text
Transformer
Residual MLP
```

不要询问后自动顺手继续。

做到这里：

**停止。**

后续会由独立审计决定是否允许正式进入 M2 network implementation。

---

# 四十五、Preflight PASS 条件

只有以下全部同时成立：

```text
四项 Preflight 全部完成

GitHub Actions CPU CI 已建立

_terminated 假 cache 已正确清理

or True 永真断言已被真正有意义的确定性测试替代

CPU→GPU contiguous boundary 已冻结

新增 Preflight tests PASS

原 M0 frozen tests 全部 PASS

原 M1 frozen tests 全部 PASS

0 failed
0 skipped
0 xfailed

实际 GitHub Actions run PASS

M0 tag 未移动

M1 tag 未移动

没有 frozen tests 被删除

没有 frozen tests 被弱化

没有改变 M0/M1 游戏语义

没有实现正式 M2 network

没有使用 tuple 8×6 checkpoint

没有提前实现 C++
```

才允许最终写：

```text
M2 PREFLIGHT PASS
```

---

# 四十六、如果只差远程 CI

如果：

```text
local full pytest = PASS
workflow implemented
```

但：

```text
actual GitHub Actions run 尚未产生 / 无法确认
```

最终状态必须是：

```text
M2 PREFLIGHT INCOMPLETE
```

剩余 blocker：

```text
remote mandatory GitHub Actions green run
```

不是 PASS。

---

# 四十七、如果发现真实冲突

如果最新总计划、冻结 tag 或仓库实际源码表明：

本提示词某个具体实现细节已经不适用：

不要自行大改方案。

报告：

```text
CONFLICT FOUND
```

并精确给出：

```text
plan clause
repository file / code
prompt requirement
why they conflict
```

停止施工。

---

# 四十八、最终报告格式固定

最终回答必须按照下面结构。

不要用宣传性语言代替证据。

```text
1. RESULT

M2 PREFLIGHT PASS
或
M2 PREFLIGHT FAIL
或
M2 PREFLIGHT INCOMPLETE


2. AUTHORITATIVE PLAN

- authoritative plan file read:
- relevant M2 / CI clauses checked:
- any conflict found:


3. STARTING REPOSITORY STATE

- initial HEAD:
- initial git status:
- m0-reference-pass:
- m1-fastenv-audited-pass:
- main relative to M1 tag:


4. CI

- workflow:
- triggers:
- runner:
- Python:
- dependencies:
- exact full test command:
- actual remote Actions status:
  PASS / FAIL / NOT YET VERIFIED


5. _terminated CLEANUP

- file:
- removed attribute:
- removed invalidation writes:
- removed method:
- step terminal computation now:
- docstrings/comments corrected:
- semantic changes:
  NONE / explain


6. INVALID ASSERT CLEANUP

- file:
- old invalid assertion:
- new deterministic setup:
- expected pre-reset score:
- post-reset assertion:
- why test is now meaningful:


7. CONTIGUOUS CONTRACT

- module:
- function:
- accepted type:
- accepted shape:
- accepted dtype:
- contiguous input behavior:
- non-contiguous input behavior:
- invalid dtype behavior:
- invalid shape behavior:
- input mutation:
- torch dependency:
  NO


8. TEST RESULTS

Focused:

[commands + exact result]

Full:

python -m pytest

- passed:
- failed:
- skipped:
- xfailed:


9. FROZEN REGRESSION VERIFICATION

- original M0 tests preserved:
- original M1 tests preserved:
- frozen tests deleted:
- skip added:
- xfail added:
- collection exclusions added:


10. TAG VERIFICATION

- m0-reference-pass final target:
- unchanged:
- m1-fastenv-audited-pass final target:
- unchanged:


11. OUT-OF-SCOPE VERIFICATION

- Transformer implemented: NO
- Residual MLP implemented: NO
- Q/V/A implemented: NO
- Teacher implemented: NO
- tuple 8x6 checkpoint used: NO
- Replay implemented: NO
- Self-play implemented: NO
- C++ implemented: NO
- CUDA implemented: NO
- GPU benchmark started: NO


12. FILES CHANGED

For every file:
- path
- exact reason


13. FINAL DIFF

git status --short

[output]

git diff --stat

[output]


14. REMAINING BLOCKERS

NONE

or exact blockers.
```

---

# 四十九、执行顺序再次锁死

现在按以下顺序执行：

```text
读取最新唯一权威总计划
↓
审计相关 M2 / CI 条款
↓
审计仓库 / tags / working tree
↓
建立 CPU CI
↓
清理 _terminated 假 cache
↓
修复 or True
↓
实现 contiguous transfer boundary
↓
写 Preflight tests
↓
写简洁 Preflight spec
↓
focused tests
↓
完整 python -m pytest
↓
检查 frozen regression
↓
检查 tags
↓
检查 diff
↓
实际 GitHub Actions 验证
↓
按固定格式报告
↓
STOP
```

不要进入正式 M2 网络。

不要提前上 C++。

不要使用 tuple checkpoint。

不要移动 frozen tags。

不要通过弱化测试制造 PASS。

**现在开始 M2 Preflight。**