# M2 Performance Unblock 正式执行计划

> 项目：`Safolour/NN-2048-ai`
> 正式工作区：`D:\CodexTasks\NN-2048-ai`
> 本文件用途：**M2 正式网络实现之后、M3 之前的性能解阻施工计划**
> 本文件不是新的总计划。若与唯一权威总计划冲突，**唯一权威总计划优先**。
> Agent 不得自行扩展范围、替换技术路线、调整验收门槛或提前进入 M3。

---

# 0. 当前项目状态

当前正式状态固定为：

```text
M0
FINAL PASS / FROZEN
tag: m0-reference-pass
commit: 3f2def1d95f56eff776e671143188947bf64485b

M1
FINAL AUDITED PASS / FROZEN
tag: m1-fastenv-audited-pass
commit: e5486017a90eeec4fb9814de7880b3c413dc06dd

M2 Preflight
AUDITED PASS
audited implementation commit:
356cedd443d6f56ac513f97e723e9addb78a28a9

M2 Formal Network Implementation
CURRENT STAGE
status: INCOMPLETE

M3
FORBIDDEN UNTIL M2 EXIT CRITERIA PASS
```

当前正式工作区 HEAD 在最近一次 M2 执行报告中为：

```text
ea3c2167fb3d8b7cb0a205007ea0b7ca49d5afc3
```

该 HEAD 相对 M2 Preflight closeout 基线只包含 prompt 同步提交。

当前 M2 源码、tests、benchmarks、artifact、report 均仍为未提交 working-tree 工作。

---

# 1. 为什么现在必须执行本计划

M2 已经完成：

- Transformer2048
- ResidualMLP2048
- Q / V / A 三个独立 Head
- learned absolute position embedding
- D4 Torch batch utilities
- legal mask / random tie-safe greedy policy
- CPU forward / backward correctness
- GPU forward / backward correctness
- tiny-overfit
- FP32 / BF16 GPU benchmark
- batch 256 / 512 / 1024 / 2048 / 4096 / 8192 sweep
- Transformer closed-loop profiling
- Residual MLP closed-loop profiling
- 一轮安全 pipeline A/B tuning

当前完整 CPU regression：

```text
487 collected
487 passed
0 failed
0 skipped
0 xfailed
```

当前关键 GPU 数据：

## Transformer2048

```text
model-only best:
BF16 / eager / batch 1024
64,168.50 states/s

best closed-loop:
4096 envs
35,719.40 decisions/s

GPU mean:
63.40%

CPU/H2D feed:
32.44%

starvation:
PASS
A=False
B=False
C=False
```

## ResidualMLP2048

```text
model-only best:
BF16 / eager / batch 8192
1,509,056.29 states/s

baseline best closed-loop:
4096 envs
91,638.80 decisions/s

GPU mean:
10.04%

CPU/H2D feed:
89.98%

closed-loop / model-only:
6.07%

starvation:
FAIL
A=True
B=False
C=True
```

安全 pipeline tuning 后：

```text
2 workers:
101,193.45 decisions/s
+10.43%

GPU mean:
11.25%

CPU/H2D feed:
87.15%

closed-loop / model-only:
6.71%

starvation:
STILL FAIL
A=True
B=False
C=True
```

所以 M2 当前不能通过 Exit Criteria。

---

# 2. 权威总计划对本阶段的要求

唯一权威总计划已经明确规定：

```text
Python / NumPy vectorization
↓
profiling
↓
如果环境明显成为主要瓶颈
↓
将真正热点迁移至：
C++ / SIMD / LUT / multithreading
```

同时 M2 GPU Starvation Gate 明确规定：

如果 GPU 长期明显等待：

- environment
- Python
- batch assembly
- H2D

则：

```text
不得直接进入 M3
必须回退优化 M1 / M2 pipeline
```

允许手段包括：

- 更多环境并行
- vectorization
- worker pipeline
- double buffering
- pinned memory
- async H2D
- 减少 object allocation
- C++ FastEnv
- SIMD / LUT
- batch size 调整

当前 profiling 已经满足“环境成为主要瓶颈”的触发条件。

因此：

```text
C++ decision:
不再是 DEFER

当前状态：
ELIGIBLE FOR EVIDENCE-BASED DECISION
```

但仍禁止：

```text
因为“C++ 理论更快”
直接重写整个环境。
```

必须先完成本计划规定的 hotspot attribution。

---

# 3. 本阶段唯一目标

本阶段只解决一件事：

> **解除 ResidualMLP2048 的 CPU environment / legal-mask GPU starvation，使 M2 满足性能 Exit Criteria。**

本阶段不是：

- M3 Teacher；
- tuple checkpoint 接入；
- Teacher Search；
- Self-play；
- Replay；
- Double-Q；
- Target Network；
- Search Correction；
- 高位训练；
- 网络结构研究；
- 棋力比较。

---

# 4. 强制执行顺序

Agent 必须按以下顺序执行：

```text
P0 保存当前可工作的 M2 baseline
↓
P1 精细拆解 CPU 热点
↓
P2 证明重复 movement / legal / terminal 成本
↓
P3 检查 Windows C++ toolchain
↓
P4 最后一轮受控低风险 Python/NumPy A/B
↓
P5 固定 C++ YES / NO 决策
↓
若 NO：
    用证据说明为何环境已足够，直接进入 P9
若 YES：
    P6 实现独立 M2 C++ hotspot backend
    ↓
    P7 differential correctness
    ↓
    P8 primitive + closed-loop A/B
↓
P9 重跑 GPU starvation gate
↓
P10 补 M2 benchmark 缺口
↓
P11 完整 regression
↓
P12 candidate commit + push + GitHub Actions
↓
P13 STOP，交给独立审计
```

不得跳步。

---

# 5. P0 — 保存当前 M2 baseline

## 5.1 开工前检查

必须运行：

```powershell
Set-Location 'D:\CodexTasks\NN-2048-ai'

git status --short
git rev-parse HEAD
git rev-parse main
git rev-parse 'm0-reference-pass^{commit}'
git rev-parse 'm1-fastenv-audited-pass^{commit}'

python -m pytest
```

必须确认：

```text
M0 tag =
3f2def1d95f56eff776e671143188947bf64485b

M1 tag =
e5486017a90eeec4fb9814de7880b3c413dc06dd
```

pytest 必须：

```text
487 passed
0 failed
0 skipped
0 xfailed
```

若不是：

```text
STOP
```

不得开始性能修改。

---

## 5.2 当前 baseline 必须先形成 WIP checkpoint commit

当前已经存在并通过 correctness 的 M2 文件必须先保存。

允许 checkpoint 包含：

```text
.github/workflows/ci.yml

src/game2048/m2_models.py
src/game2048/m2_policy.py
src/game2048/m2_symmetry.py

tests/test_m2_models.py
tests/test_m2_policy.py
tests/test_m2_symmetry.py

benchmarks/_m2_utils.py
benchmarks/validate_m2_models.py
benchmarks/benchmark_m2_gpu.py
benchmarks/benchmark_m2_closed_loop.py
benchmarks/benchmark_m2_pipeline_tuning.py

reports/m2/m2_correctness.json
reports/m2/m2_gpu_benchmark.json
reports/m2/m2_closed_loop_benchmark.json
reports/m2/M2_REPORT.md
```

若 working tree 还有用户自己的 prompt 文件变更：

```text
不得 stage
不得 commit
不得 restore
不得删除
```

checkpoint commit message 固定：

```text
m2: checkpoint correctness-pass performance-blocked baseline
```

这个 commit：

```text
不是 M2 PASS
不是 M2 candidate
不是 frozen tag
```

不得创建 tag。

---

# 6. P1 — 精细拆解 CPU 热点

当前 profile 只知道：

```text
Residual MLP tuned:
environment_step ≈ 54%
legal_mask       ≈ 32%
```

这还不够决定 C++ 应该迁什么。

必须创建：

```text
benchmarks/benchmark_m2_cpu_hotspots.py
```

该脚本只做 profiling / benchmark。

不得修改 frozen M1。

---

## 6.1 必须拆出的项目

至少单独测：

```text
A. current-state legal_mask_batch

B. selected-action movement core

C. reward calculation / reward aggregation

D. score overflow check / score update

E. spawn preparation

F. random draw generation

G. spawn placement

H. post-spawn terminal calculation

I. BatchStepResult / output copy allocation

J. reset_where

K. total env.step
```

其中 B～I 必须尽量归因 `env.step()` 内部真实成本。

不得只给：

```text
env.step = 54%
```

这种结果。

---

## 6.2 测试规模固定

至少测试：

```text
4096 boards
8192 boards
16384 boards
```

每个规模：

```text
warmup >= 20 iterations

正式测量：
>= 100 iterations
或
>= 2 seconds

取更严格者
```

输出：

```text
absolute us / iteration
boards/s
占一个完整 decision 的 %
allocation count（可合理测量时）
```

---

## 6.3 必须同时使用两种方法交叉确认

方法 1：

```text
显式 wall-clock microbench
```

方法 2：

```text
cProfile / 等价函数级 profiler
```

如果二者排序明显冲突：

```text
STOP
调查测量方法
```

不得挑一个更符合预期的结果。

---

# 7. P2 — 证明重复 movement 成本

当前生产闭环必须明确记录一个 decision 实际做了多少 movement work。

当前结构大致是：

```text
current state
↓
legal_mask_batch
↓
4-action movement-only

GPU Q forward
↓
action selected

env.step
↓
selected-action reward-producing move
↓
spawn
↓
is_terminal_batch
↓
4-action movement-only on next state
```

而下一次 decision 又会：

```text
对同一个 next state 再次 legal_mask_batch
```

所以必须量化：

```text
current-state legal:
4 directional movement evaluations

selected move:
1 directional movement evaluation

post-spawn terminal:
4 directional movement evaluations

next iteration legal:
再次 4 directional movement evaluations
```

必须回答：

```text
一个 steady-state decision 中，
有多少 movement work 属于可消除的重复计算？
```

---

## 7.1 必须验证 next-state terminal / next-state legal 的重复

要求建立只读 diagnostic：

```text
step 后 is_terminal_batch(s')
与
下一轮 legal_mask_batch(s')
```

证明：

```text
terminal = ~next_legal.any(axis=1)
```

并确认：

```text
next_legal 本身可以同时用于：
1. 当前 terminal verdict
2. 下一轮 policy illegal-action masking
```

这一步只做验证。

不得修改 M1 API。

---

# 8. P3 — Windows C++ Toolchain Gate

在写任何正式 C++ backend 之前：

必须检查：

```text
MSVC Build Tools
cl.exe
Windows SDK
CMake
Python development headers
pybind11 build capability
```

当前已知：

```text
CMake 已安装

普通 shell 当前未发现 cl.exe

Visual Studio C++ toolchain 尚未确认
```

---

## 8.1 固定检查命令

至少执行：

```powershell
Get-Command cmake.exe
Get-Command cl.exe -ErrorAction SilentlyContinue

$vswhere='C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe'
if (Test-Path $vswhere) {
    & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
}
```

还必须尝试：

```text
Developer Command Prompt / VsDevCmd.bat
```

激活后重新检查 `cl.exe`。

---

## 8.2 如果 MSVC 不存在

不得：

- 切 MinGW；
- 切 Rust；
- 切 Cython；
- 切 Numba；
- 下载未知 compiler；
- 改成 custom CUDA；
- 绕开计划。

最终状态：

```text
BLOCKED_TOOLCHAIN
```

并向用户明确要求安装：

```text
Visual Studio Build Tools 2022

Desktop development with C++

MSVC v143 x64/x86 build tools
Windows 10/11 SDK
C++ CMake tools
```

STOP。

---

## 8.3 Toolchain smoke test

正式 backend 前必须能编译并 import 一个最小 pybind11 extension：

```text
m2_cpp_smoke
```

内容只允许：

```text
int add(int a, int b)
```

验证：

```text
import 成功
add(2, 3) == 5
```

smoke test 文件完成后可删除。

禁止把 smoke extension 留成生产 dependency。

---

# 9. P4 — 最后一轮低风险 Python / NumPy A/B

这一步的目的不是无限优化 Python。

只允许验证 3 类问题：

```text
1. 是否存在明显重复 allocation
2. 是否能避免重复 legal/terminal 计算
3. 是否有合理的批处理组织仍未测试
```

---

## 9.1 禁止重复测试

已有证据：

```text
ThreadPool 2 workers:
+10.43%

4 workers:
明显倒退

8 workers:
明显倒退
```

所以禁止再次：

```text
把 4 / 8 ThreadPool 作为“新方案”重复跑
```

除非代码路径已经发生实质变化。

---

## 9.2 允许测试的低风险项

最多允许：

### A. buffer reuse

消除 profile 明确显示的大型重复临时 allocation。

### B. cached next legal prototype

只在独立 M2 prototype / benchmark 层：

```text
计算一次 next legal
同时用于：
terminal
下一轮 legal mask
```

不得修改 M1 frozen semantics。

### C. process-based producer probe

只有在：

```text
ThreadPool 被 GIL / NumPy 调度限制
```

有 profiler 证据时，

允许做一个：

```text
2-process producer
```

的短 A/B probe。

不得构建复杂生产 multiprocessing framework。

---

## 9.3 P4 停止阈值

如果所有允许的 Python/NumPy 调整中：

最佳方案相对当前 tuned 101,193.45 decisions/s：

```text
提升 < 20%
```

并且 starvation A 或 C 仍为 True：

则：

```text
Python-only route exhausted
C++ = YES
```

进入 P6。

如果提升 >=20%，必须重新跑完整 starvation gate。

只有所有 gate 都解除，才允许 C++ = NO。

---

# 10. P5 — C++ YES / NO 决策

Agent 不得自行写长篇主观论证。

必须输出固定表：

```text
C++ DECISION

Current best closed-loop:
...

GPU util:
...

CPU/env/legal share:
...

Top 3 CPU hotspots:
1.
2.
3.

Python safe optimization best improvement:
...

Starvation A:
True/False

Starvation B:
True/False

Starvation C:
True/False

Decision:
YES / NO
```

---

## 10.1 YES 的固定条件

满足以下全部：

```text
1. starvation A 或 C 仍为 True

2. CPU environment/legal/movement
   仍为主要 wall-clock bottleneck

3. P4 最佳安全 Python 优化
   无法解除 starvation

4. hotspot 已被归因到明确 primitive
```

则：

```text
C++ = YES
```

不允许继续拖延。

---

## 10.2 NO 的固定条件

只有：

```text
starvation A=False
starvation B=False
starvation C=False
```

且：

```text
继续优化环境预计已无法明显提高端到端 throughput
```

才能：

```text
C++ = NO
```

然后直接进入 P9 / P10。

---

# 11. P6 — C++ backend 的固定边界

如果 C++ = YES：

禁止：

```text
重写整个项目
重写网络
重写训练器
修改 M0
修改 frozen M1 semantics
把 PyTorch 搬进 C++
把 Teacher 搬进 C++
写 CUDA FastEnv
```

正式原则：

```text
Python + PyTorch:
继续负责 orchestration / NN / experiment management

C++:
只负责 profiling 已证明的 CPU hotspot
```

---

# 12. C++ backend 的目录固定

新增：

```text
cpp/m2_fast_backend/
    CMakeLists.txt
    m2_fast_backend.cpp
    movement_core.h
    movement_core.cpp
```

Python bridge：

```text
src/game2048/m2_fast_backend.py
```

M2 rollout wrapper：

```text
src/game2048/m2_rollout_env.py
```

tests：

```text
tests/test_m2_fast_backend.py
tests/test_m2_rollout_env.py
```

benchmark：

```text
benchmarks/benchmark_m2_fast_backend.py
```

不得：

```text
把 C++ 写进 src/game2048/fast_env.py
```

M1 frozen source默认不修改。

---

# 13. C++ 第一版只允许实现这些 primitive

第一优先级：

```text
1. movement-only all-actions legality
2. selected-action move
3. next-state legal calculation
```

允许根据 P1 hotspot 数据加入：

```text
4. terminal reduction
```

如果 P1 证明 spawn 不是热点：

```text
spawn 保持 NumPy / Python
```

不得因为“顺手”把 RNG 全迁 C++。

---

# 14. RNG 固定策略

第一版 C++ backend：

```text
禁止自行实现新的 RNG
```

原因：

M1 已冻结 NumPy Generator 语义。

随机数仍由：

```text
numpy.random.Generator
```

产生。

禁止：

```text
std::mt19937
std::random_device
PCG 自己重新实现
```

替换 NumPy RNG。

这样避免：

```text
seed replay
illegal action no RNG consumption
spawn sequence
```

被破坏。

---

# 15. C++ movement 表示

输入 board 固定：

```text
uint8
shape (N,16)
C contiguous
```

Action：

```text
uint8
shape (N,)
0..3
```

输出不得使用 Python object per board。

必须是 contiguous NumPy-compatible buffers。

---

# 16. C++ 第一版禁止引入 4-bit board

总计划的正式 board representation：

```text
uint8[16]
```

所以本阶段：

```text
禁止改成 64-bit nibble board
禁止把 exponent 截到 4 bit
```

可在内部临时构造 LUT key，

但正式输入 / 输出语义必须仍支持：

```text
0..255 exponent
```

尤其：

```text
65536+
```

不得回归。

---

# 17. LUT 规则

只有 P1 profiling 证明：

```text
row movement
```

是主要 hotspot，

才允许为：

```text
4-cell row movement
```

设计 LUT。

LUT 必须满足：

```text
uint8 exponent semantics
high tile correctness
reward overflow semantics
255+255 tile exponent overflow semantics
```

禁止使用只能覆盖 exponent 0..15 的传统 65536-entry 2048 LUT 作为完整正式语义。

如果使用分层 LUT / fast common-path：

高 exponent 必须：

```text
fallback to exact generic path
```

且 differential tests 必须覆盖。

---

# 18. SIMD 规则

SIMD 只允许在：

```text
scalar C++ baseline correctness PASS
```

之后。

顺序：

```text
scalar C++
↓
benchmark
↓
如果 movement 仍是 hotspot
↓
SIMD
```

禁止直接写一版难审计 SIMD 并把 scalar oracle 省掉。

---

# 19. 新的 M2 rollout 环境行为

如果 C++ backend 用于生产闭环：

必须新建：

```text
M2RolloutBatchEnv
```

不得替换：

```text
Fast2048BatchEnv
```

M1 FastEnv 继续保留作为：

```text
frozen NumPy oracle
```

M2 rollout wrapper 必须：

```text
保持当前 boards / scores / RNG 状态
```

并支持：

```text
reset
reset_where
step
legal mask
terminal
```

---

# 20. 关键性能设计：next legal 只算一次

新 M2 rollout path 必须支持：

```text
current_legal
```

缓存。

每一步固定：

```text
当前缓存 current_legal
↓
GPU Q
↓
mask + action selection
↓
selected move
↓
spawn
↓
计算 next_legal 一次
↓
terminated = ~next_legal.any(axis=1)
↓
把 next_legal 保存为下一轮 current_legal
```

禁止：

```text
同一个 s'
先算 terminal 的 4-action legal
下一轮再重复算一次 4-action legal
```

这是本阶段必须消除的重复工作。

---

# 21. reset / reset_where 后 legal cache

`reset()`：

```text
完成两个 spawn
↓
计算 legal
↓
建立 current_legal cache
```

`reset_where(mask)`：

只允许重新计算：

```text
被 reset rows
```

对应 legal。

未 reset rows：

```text
legal cache 不得重新全量计算
```

除非 benchmark 证明局部更新反而更慢。

如果全量更快：

必须有 A/B 数据后才能采用。

---

# 22. illegal action

正式 policy 已经 legal-mask，

但 backend 仍必须保持 M1 语义：

illegal action：

```text
board unchanged
reward = 0
score unchanged
no spawn
no RNG consumption
```

C++ 不得假设：

```text
“policy 永远不会传 illegal，所以可以不实现”
```

测试必须覆盖。

---

# 23. reward / score overflow

冻结语义继续：

```text
reward / score:
np.int64 representable only

overflow:
explicit OverflowError
```

C++ 不得：

```text
wrap
saturate
clip
```

所有状态 mutation 之前：

必须完成必要 range check。

保持：

```text
atomicity
```

---

# 24. tile exponent overflow

冻结语义继续。

例如：

```text
255 + 255
```

产生 exponent 256，

无法表示在 uint8。

必须和 M0/M1 当前行为一致。

不得 silent wrap。

---

# 25. P7 — C++ differential correctness

C++ backend 在任何性能 benchmark 之前：

必须通过 differential。

---

## 25.1 ordinary boards

至少：

```text
10,000 random boards
```

覆盖四方向。

比较：

```text
afterstate
moved
reward
legal mask
terminal
```

---

## 25.2 high-tile boards

至少覆盖：

```text
15
16
17
20
21
31
61
62
63
126
127
128
254
255
```

以及随机组合。

---

## 25.3 D4

所有：

```text
8 symmetries
```

必须与 frozen oracle 一致。

---

## 25.4 RNG / spawn

同 seed：

必须验证：

```text
spawn location sequence
spawn exponent sequence
illegal action RNG non-consumption
reset
reset_where
```

如果 C++ 不负责 RNG：

仍必须验证整个 M2 rollout wrapper 的实际序列。

---

## 25.5 overflow / atomicity

必须覆盖：

```text
reward near INT64_MAX
score near INT64_MAX
unrepresentable action reward
score accumulation overflow
tile exponent overflow
```

发生异常后：

```text
board unchanged
score unchanged
RNG unchanged
```

---

## 25.6 live-view contract

如果 M2RolloutBatchEnv 暴露：

```text
boards
scores
```

则必须明确其 view contract。

第一版建议保持与 M1 相同：

```text
read-only live view
```

不得无说明改变语义。

---

# 26. correctness oracle 顺序

正式 differential oracle：

```text
M0 Reference2048Env
↓
M1 Fast2048BatchEnv
↓
M2 C++ backend
```

出现冲突：

```text
M0 优先判定游戏规则
```

M1 frozen behavior 若与 M0 已有正式审计差异：

仅允许总计划已明确接受的：

```text
int64 representation divergence
```

不得自行发明新 divergence。

---

# 27. P8 — 性能 A/B

C++ correctness PASS 后：

必须对同一台机器、同样 batch、同样 workload 做：

```text
NumPy baseline
vs
M2 new backend
```

---

## 27.1 primitive benchmark

至少记录：

```text
legal mask boards/s

selected move boards/s

terminal boards/s

step transitions/s

next-legal fused path decisions/s
```

batch：

```text
1024
4096
8192
16384
```

---

## 27.2 端到端 benchmark 才是主标准

最终性能判断：

```text
ResidualMLP closed-loop decisions/s
```

不是：

```text
C++ kernel 自己快多少倍
```

必须重跑：

```text
1024
4096
8192
16384 envs
```

---

## 27.3 Closed-loop adaptive scale extension

正式 P8/P9 的 closed-loop benchmark 不得机械地在 `16384 envs` 停止。

固定先测：

```text
1024
4096
8192
16384
```

然后按以下固定规则自适应扩展：

```text
if 16384 未 OOM
and RAM / VRAM 无明显资源压力
and decisions/s(16384) 相比 decisions/s(8192) 提升 >= 5%
then:
    必须追加测试 32768 envs
```

如果：
```text
32768 未 OOM
and RAM / VRAM 仍有安全余量
and decisions/s(32768) 相比 decisions/s(16384) 提升 >= 5%
then:
    必须追加测试 65536 envs
```

之后继续采用同样原则：只有前一档相对上一档仍提升 >= 5%，且资源安全，才允许继续扩大。

出现以下任一情况立即停止继续扩大：

```text
relative throughput gain < 5%
throughput 下降
OOM
RAM 压力明显
VRAM 压力明显
paging / swapping
稳定性下降
```

最终正式 systems config 必须选择：

```text
最高稳定端到端 decisions/s
```

对应的 env count，而不是固定选择 16384，也不是机械选择能够运行的最大规模。

该规则只作用于：

```text
当前/最终生产 backend 的正式 P8 closed-loop A/B
最终 P9 starvation gate
```

不要求为了补本规则而重新执行已经完成的旧 scalar baseline 32768/65536。

每个扩展点必须记录：

```text
env_count
decisions/s
relative gain vs previous scale
GPU utilization
CPU utilization
RAM
VRAM
starvation A/B/C
```

---

# 28. C++ 优化保留阈值

某项复杂优化若：

```text
primitive 快很多
但
closed-loop 提升 < 5%
```

默认：

```text
不保留
```

除非它是后续更大优化的必要基础，

且必须在报告中说明。

---

# 29. P9 — GPU starvation gate 重跑

保持之前 gate 定义不变。

不得中途改判据。

---

## Gate A

```text
steady-state mean GPU utilization < 60%
AND
CPU environment + batch preparation + H2D >= 40%
```

则：

```text
A=True
```

---

## Gate B

```text
pipeline stall >= 20%
```

则：

```text
B=True
```

---

## Gate C

```text
closed-loop / model-only < 70%
AND
CPU/H2D feed >= 40%
```

则：

```text
C=True
```

---

# 30. M2 性能解阻 PASS 条件

Transformer：

```text
A=False
B=False
C=False
```

Residual MLP：

```text
A=False
B=False
C=False
```

并且：

```text
主要瓶颈已经明确
不存在明显 GPU starvation
继续优化环境预计无法再明显提高端到端 throughput
```

才允许：

```text
M2 PERFORMANCE UNBLOCK PASS
```

---

# 31. 如果 C++ 后仍 FAIL

不得继续无限优化。

如果：

```text
C++ hotspot backend 已完成
correctness PASS
端到端 A/B 已完成
```

但 MLP 仍：

```text
A=True 或 C=True
```

则只允许再进行一轮：

```text
P1 profile repeat
```

找新的最大瓶颈。

如果新瓶颈：

```text
仍是已实现 C++ movement/legal
```

才允许：

```text
scalar → SIMD / LUT
```

如果新瓶颈已经变成：

```text
Python orchestration
process synchronization
memory copy
action selection
```

则只能优化新的真实热点。

禁止继续凭惯性堆 C++。

---

# 32. P10 — 补 M2 GPU benchmark 两个缺口

在 performance unblock 之后补。

---

## 32.1 MLP 16384 GPU-only inference

现有数据：

```text
BF16 4096:
1.355M states/s

BF16 8192:
1.509M states/s
```

8192 相比 4096：

```text
仍明显增长
```

且 VRAM 很低。

因此必须补：

```text
ResidualMLP
BF16
eager
batch 16384
GPU-only inference
```

如果：

```text
16384 vs 8192 gain >= 5%
```

且资源足够：

再测：

```text
32768
```

否则停止扩大。

---

## 32.2 training torch.compile 记录修正

现有 raw artifact 中：

```text
torch.compile(model).forward_state
torch.compile(model).forward_afterstate
```

实际仍绑定 original model。

所以旧 training compile PASS：

```text
不得算有效 compile evidence
```

固定处理：

优先建立一个只为 benchmark 的：

```text
compiled training wrapper module
```

其 `forward(...)` 一次返回：

```text
Q
V
A
```

并真正让 `torch.compile` 包裹 `forward`。

如果：

```text
TritonMissing
```

仍出现：

记录：

```text
FAILED / UNSUPPORTED
```

即可。

不得安装额外 Triton stack 只为让 benchmark 变绿。

eager 仍允许作为正式配置。

---

# 33. 不重新做网络结构研究

本阶段禁止：

```text
改 Transformer hidden
改 block 数
改 attention heads
改 FFN
改 MLP hidden
改 MLP block 数
改 Q/V/A head
改 position embedding
```

除非发现 correctness bug。

性能解阻目标：

```text
pipeline
```

不是：

```text
让网络变慢从而“提高 GPU utilization”
```

---

# 34. 禁止伪造 starvation PASS

以下方式全部禁止：

```text
故意扩大网络
故意增加无用 GPU work
重复 forward
降低 environment throughput
人为 sleep
只选择 GPU util 更高但 decisions/s 更低的配置
```

主指标：

```text
端到端有效 decisions/s
```

GPU utilization 只是诊断指标。

---

# 35. P11 — 完整 regression

最终必须运行：

```powershell
python -m pytest
```

要求：

```text
0 failed
0 skipped
0 xfailed
```

原有：

```text
487 tests
```

必须全部保留。

新增 C++ / rollout tests 后：

```text
总数必须 > 487
```

---

# 36. CI 必须构建 C++ backend

如果 C++ backend 成为正式 M2 path：

GitHub Actions CPU CI 必须：

```text
在 ubuntu-latest
编译 C++ extension
运行全部 CPU tests
```

CI 不允许：

```text
因为没有 C++ extension 就 skip 新 backend tests
```

---

# 37. CI 依赖固定

若使用 pybind11：

CI 允许新增：

```text
pybind11
cmake
```

优先使用 runner 已有 C++ compiler。

不得引入：

```text
Conda
Poetry
大型 build framework
```

仅为这个 extension 服务。

---

# 38. Windows build 必须可重复

不得只靠某个 Agent shell 的临时状态。

必须提供：

```text
docs/M2_FAST_BACKEND_BUILD.md
```

只记录：

```text
依赖
MSVC requirement
CMake configure
CMake build
extension output
如何运行 pytest
```

不要写第二份总计划。

---

# 39. P12 — final artifacts

必须更新：

```text
reports/m2/M2_REPORT.md
reports/m2/m2_closed_loop_benchmark.json
reports/m2/m2_gpu_benchmark.json
```

新增：

```text
m2_cpu_hotspots.json
reports/m2/m2_fast_backend_benchmark.json
```

如果 C++ = NO：

```text
reports/m2/m2_fast_backend_benchmark.json
```

可不存在，

但报告必须有完整 NO 证据。

---

# 40. M2_REPORT 必须新增章节

至少：

```text
PERFORMANCE UNBLOCK

1. baseline checkpoint
2. fine-grained hotspot attribution
3. repeated legal/terminal analysis
4. toolchain result
5. Python safe A/B
6. C++ decision
7. C++ architecture（如果 YES）
8. differential correctness
9. primitive A/B
10. closed-loop A/B
11. starvation gate before
12. starvation gate after
13. remaining bottleneck
14. MLP 16384 GPU-only result
15. compile-training validation correction
```

---

# 41. Candidate 条件

只有以下全部成立：

```text
网络 correctness PASS

M0 frozen regressions PASS
M1 frozen regressions PASS
M2 Preflight tests PASS
M2 network CPU tests PASS
M2 fast-backend differential PASS

precision benchmark complete
batch sweep complete
compile status correctly recorded
inference states/s recorded
training samples/s recorded
VRAM recorded

closed-loop benchmark complete

Transformer starvation:
PASS

Residual MLP starvation:
PASS

current main bottleneck identified

no obvious GPU starvation

full pytest:
0 failed
0 skipped
0 xfailed
```

才允许产生：

```text
M2 candidate commit
```

---

# 42. Git / tag 规则

绝对禁止移动：

```text
m0-reference-pass
m1-fastenv-audited-pass
```

不得创建：

```text
m2-pass
m2-final
m2-performance-pass
```

直到独立审计以后。

---

# 43. Candidate commit

最终候选 commit message：

```text
m2: unblock pipeline performance and complete candidate
```

push 后：

必须实际等待 GitHub Actions。

只有：

```text
actual remote CI = PASS
```

才允许报告：

```text
M2 CANDIDATE COMPLETE
```

不是：

```text
M2 AUDITED PASS
```

---

# 44. 主计划暂时不改 PASS

本任务过程中：

```text
prompts/2048_AI_正式执行计划_M0_M1冻结_CI增强版_2026-09-18.md
```

默认：

```text
READ ONLY
```

禁止自行写：

```text
M2 PASS
M2 FROZEN
进入 M3
```

只有独立审计后再 closeout。

---

# 45. tuple checkpoint

本阶段：

```text
禁止读取
禁止加载
禁止转换
禁止 benchmark
```

tuple 8×6 checkpoint 继续留给：

```text
M3 Teacher
```

---

# 46. 不得开始后续阶段

本计划执行期间禁止：

```text
M3 Teacher
M4 Transformer vs MLP 棋力比较
M5 Teacher pretraining
M6 Student correction
M7 Self-play
M8 high-tile
M9 Search Correction
M10 iteration
M11 32768
M12 65536
```

---

# 47. Performance Unblock 最终报告格式

Agent 最终必须严格按下面格式报告：

```text
1. RESULT

M2 PERFORMANCE UNBLOCK PASS
或
M2 PERFORMANCE UNBLOCK INCOMPLETE
或
BLOCKED_TOOLCHAIN
或
FAIL


2. STARTING STATE

- initial HEAD
- initial git status
- M0 tag
- M1 tag
- baseline pytest
- baseline artifact hashes


3. BASELINE CHECKPOINT

- checkpoint commit
- files included
- files intentionally excluded
- tag created: NO


4. CPU HOTSPOT PROFILE

4096:
- current legal
- selected move
- reward
- score checks
- RNG
- spawn
- terminal
- allocation/copy
- reset
- total

8192:
...

16384:
...

Top 3 hotspots:
1.
2.
3.


5. DUPLICATE MOVEMENT ANALYSIS

- current-state legal movement work
- selected movement work
- post-spawn terminal movement work
- duplicated next-state legal work
- avoidable share


6. TOOLCHAIN

- cmake
- MSVC
- cl.exe
- Windows SDK
- pybind11
- smoke extension result


7. PYTHON/NUMPY FINAL A/B

For every probe:
- exact change
- before decisions/s
- after decisions/s
- improvement
- starvation A/B/C


8. C++ DECISION

YES / NO

Evidence:
...


9. C++ BACKEND
(if YES)

- files
- exported API
- RNG strategy
- legal caching strategy
- SIMD used: YES/NO
- LUT used: YES/NO


10. DIFFERENTIAL CORRECTNESS

- ordinary
- high tile
- D4
- legal
- terminal
- reward
- score
- spawn
- RNG
- overflow
- atomicity
- reset/reset_where


11. PRIMITIVE A/B

NumPy vs new backend:
- legal
- selected move
- terminal
- step
- fused next-legal


12. CLOSED LOOP A/B

Residual MLP:

1024
4096
8192
16384

For each:
- decisions/s
- GPU mean
- CPU util
- feed %
- model-only ratio
- VRAM


13. STARVATION GATE

Transformer:
A=
B=
C=
PASS/FAIL

Residual MLP:
A=
B=
C=
PASS/FAIL


14. MLP GPU-ONLY EXTENDED BATCH

8192:
...

16384:
...

32768 if required:
...


15. TORCH.COMPILE TRAINING VALIDATION

- old artifact status
- corrected benchmark method
- actual result
- exact failure if unsupported


16. PYTEST

collected:
passed:
failed:
skipped:
xfailed:


17. CI

candidate commit:
Actions run:
actual conclusion:
collected:
passed:


18. FROZEN VERIFICATION

M0 tag unchanged:
M1 tag unchanged:
frozen tests deleted:
skip added:
xfail added:
M1 source modified:


19. OUT OF SCOPE

Teacher: NO
tuple checkpoint: NO
Replay: NO
Self-play: NO
M3: NO
custom CUDA FastEnv: NO


20. FILES CHANGED

逐文件说明。


21. FINAL GIT STATE

git status --short

git rev-parse HEAD


22. REMAINING BLOCKERS

NONE
或精确 blocker。
```

---

# 48. STOP 条件

以下任何情况立即 STOP：

```text
M0/M1 tag 不匹配

baseline 487 tests 不全绿

用户未提交文件与本任务发生冲突

MSVC toolchain 缺失且需要 C++

C++ differential correctness 失败且无法在本轮明确修复

出现新的 frozen semantic conflict

M2 starvation 在允许的优化后仍无法解除
```

不得：

```text
为了完成任务而绕过 gate
```

---

# 49. 本阶段成功后的正式流程

如果本计划最终得到：

```text
M2 CANDIDATE COMPLETE
```

下一步不是直接 M3。

必须：

```text
candidate commit
+
M2_REPORT
+
benchmark JSON
+
GitHub Actions
↓
独立审计
↓
M2 Closeout
↓
更新唯一权威主计划
↓
才允许 M3
```

---

# 50. M2 通过之后的后续项目路线

唯一权威总计划后续仍然是：

```text
M3
Teacher 小规模验证

M4
Transformer vs MLP

M5
Teacher 预训练

M6
Student State Correction

M7
正式 Self-play Baseline

M8
高位专项

M9
Search Correction

M10
迭代提升

M11
32768 阶段

M12
65536 阶段

最终部署
```

所以：

```text
M2 Performance Unblock
不是整个项目最后一步。
```

它只是：

```text
正式大规模训练开始前
最后一个系统性能门槛。
```

---

# 51. 最终施工原则

Agent 必须始终遵守：

```text
Correctness first
↓
Measure
↓
Identify largest wall-clock bottleneck
↓
Optimize only that bottleneck
↓
Differential test
↓
End-to-end benchmark
↓
Repeat only when evidence requires
```

禁止：

```text
“顺手优化”
“我觉得这样更好”
“C++ 应该更快”
“先重写再测”
```

最终验收的是：

```text
真实端到端 decisions/s
+
正确性
+
M2 Exit Criteria
```

不是：

```text
漂亮的局部 kernel benchmark。
```
