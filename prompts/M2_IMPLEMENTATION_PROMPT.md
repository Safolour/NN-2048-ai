你现在接手项目：

`https://github.com/Safolour/NN-2048-ai`

这是一个新的执行 Agent 对话。

当前对话中用户会同时提供：

**最新唯一权威总计划：**

`2048_AI_正式执行计划_M0_M1冻结_CI增强版_2026-09-18.md`

你必须首先阅读完整总计划，尤其是：

```text
网络输入编码
Q / V / A 三个 Head 的正式语义
Transformer Position Embedding
Afterstate 输入方式
Transformer / Residual MLP baseline
性能工程 P1～P17
持续集成 CI.1～CI.8
第 99 节 M2
M2 Correctness Gate
M2 GPU Throughput Benchmark
M1 + M2 End-to-End Benchmark
GPU Starvation Gate
M2 Exit Criteria
```

总计划是：

**项目最高层唯一权威规范。**

本提示词是：

**正式 M2 的详细施工单。**

如果本提示词和总计划存在真实冲突：

```text
STOP
报告冲突
不要自行裁决
```

如果当前对话没有提供最新总计划：

```text
BLOCKED:
authoritative master plan missing
```

不要开工。

---

# 0. 当前权威仓库状态

正式 M2 开工基线：

```text
main / expected starting HEAD:

827f30c8ccd635420c8ef1c2dfea00a9e54197b2

commit message:
docs: close M2 preflight and advance plan to formal M2
```

M2 Preflight 已经完成并通过独立审计：

```text
audited Preflight implementation commit:

356cedd443d6f56ac513f97e723e9addb78a28a9
```

Preflight 后：

```text
465 passed
0 failed
0 skipped
0 xfailed
```

GitHub Actions：

```text
Preflight implementation:
run 35337724192 PASS

Preflight closeout:
run 35338812369 PASS
```

---

# 1. Frozen Milestones

## M0

```text
tag:
m0-reference-pass

commit:
3f2def1d95f56eff776e671143188947bf64485b
```

永久冻结。

`Reference2048Env` 永久是：

```text
Golden Reference
Correctness Oracle
```

---

## M1

```text
tag:
m1-fastenv-audited-pass

commit:
e5486017a90eeec4fb9814de7880b3c413dc06dd
```

永久冻结。

M1 已完成：

```text
Fast2048BatchEnv
move_batch
legal_mask_batch
is_terminal_batch
spawn
exact spawn enumeration
D4 differential
overflow handling
batch benchmarks
producer→consumer harness
profiling
```

不得重新实现 M1。

---

# 2. M2 Preflight 已关闭

以下已经正式完成：

```text
GitHub Actions CPU CI              PASS
_terminated fake-cache cleanup     PASS
invalid or True cleanup            PASS
CPU→GPU contiguous contract        PASS
```

当前正式 boundary：

```text
numpy.ndarray
shape = (N,16)
dtype = uint8
N >= 1
C-contiguous
```

模块：

```text
src/game2048/m2_data.py
```

函数：

```python
prepare_board_batch_for_transfer(...)
```

M2 必须使用它。

不要重新设计这套 contract。

---

# 3. 本任务是什么

本任务：

# M2 正式网络实现

必须完成：

```text
Transformer baseline
+
Residual MLP baseline
+
Q Head
+
Value Head
+
Afterstate Value Head
+
M2 correctness
+
GPU throughput benchmark
+
batch-size sweep
+
precision benchmark
+
eager / torch.compile benchmark
+
M1 + M2 closed-loop benchmark
+
profiling
+
GPU starvation gate
```

完成这些并达到 M2 Exit Criteria 后：

才可以形成：

```text
M2 candidate
```

但：

**你本人不得宣布其经过独立审计。**

完成后停止，交给外部审计。

---

# 4. 本任务明确禁止

M2 不允许提前实现：

```text
Teacher
tuple evaluator
tuple 8×6 checkpoint
Expectimax
Replay Buffer
Self-play training system
Double-Q
Target Network
Champion
Search Correction
high-tile restart pool
M3 Teacher pipeline
M4+
```

用户拥有现成：

```text
tuple 8×6 checkpoint
```

本阶段：

# 禁止读取、加载、转换或使用。

---

# 5. C++ 继续 DEFER

本阶段初始禁止实现：

```text
C++
pybind11
Cython
custom CUDA FastEnv
SIMD FastEnv rewrite
LUT FastEnv rewrite
```

正式顺序：

```text
先建网络
↓
跑真实闭环
↓
profile
↓
判断 CPU / Python / H2D 是否真的饿死 GPU
```

只有 profiling 证明存在严重 starvation：

才有资格提出 C++。

但本任务中：

## 即使发现 C++ 可能必要，也不要自行实现 C++。

如果经过本文允许的安全 pipeline 优化后仍明显 starvation：

```text
M2 RESULT = INCOMPLETE
```

报告数据并停止。

等待外部审计决定是否给专门的 C++ 优化施工单。

---

# 6. 开工前仓库检查

任何修改之前执行：

```bash
git status --short
git rev-parse HEAD
git rev-parse main
git rev-parse m0-reference-pass^{commit}
git rev-parse m1-fastenv-audited-pass^{commit}
```

预期：

```text
HEAD/main:
827f30c8ccd635420c8ef1c2dfea00a9e54197b2
```

M0/M1 tags 必须仍分别等于：

```text
3f2def1d95f56eff776e671143188947bf64485b

e5486017a90eeec4fb9814de7880b3c413dc06dd
```

---

# 7. 用户未提交 prompt 文件

此前本地 working tree 曾存在用户自己的：

```text
D  prompts/M2_IMPLEMENTATION_PROMPT.md
?? prompts/M2_PRE_IMPLEMENTATION_PROMPT.md
```

如果现在仍存在：

它们属于：

```text
USER-OWNED WORK
```

不得：

```text
删除
恢复
重命名
修改
stash
reset
clean
stage
commit
```

你的 M2 commit 不得夹带它们。

如果当前环境是全新 clone、这些文件不存在：

正常。

---

# 8. 不得更新 authoritative master plan 状态

本次：

```text
prompts/2048_AI_正式执行计划_M0_M1冻结_CI增强版_2026-09-18.md
```

默认：

# READ ONLY

它已经正确写明：

```text
M2 = CURRENT EXECUTION STAGE
```

不要在本次实现结束后自行写：

```text
M2 PASS
M2 FROZEN
M2 COMPLETE
```

外部审计通过后才会有单独 Closeout 指令。

---

# 9. 输入编码固定

网络输入来自：

```text
(N,16) uint8 exponent board
```

2048 exponent：

```text
empty = 0
2     = 1
4     = 2
...
2^20  = 20
>2^20 = overflow token 21
```

网络 vocabulary：

```text
22
```

固定 token ID：

```text
0..20:
保持原 exponent

21..255:
全部映射到 token 21
```

即：

```python
token = min(exponent, 21)
```

注意：

这只是：

```text
NN input vocabulary mapping
```

不是修改环境中的真实 exponent。

禁止把环境 board 本身 clamp 成 21。

---

# 10. Torch model 输入 contract

模型只接受：

```text
torch.Tensor

shape:
(B,16)

B >= 1

dtype:
torch.uint8
```

不要自动接受：

```text
numpy array
(B,4,4)
(16,)
float tensor
int64 board tensor
```

这些 preprocessing 属于边界，而不是模型内部随意兼容层。

在模型 tokenization 中明确执行：

```text
uint8 exponent
↓
clamp max 21
↓
convert embedding index dtype
↓
Embedding
```

embedding index dtype 转换属于：

```text
显式 NN preprocessing
```

不要隐藏 CPU-side dtype copy。

---

# 11. 新建模型模块

创建：

```text
src/game2048/m2_models.py
```

该模块包含正式：

```text
Transformer2048
ResidualMLP2048
```

以及必要的小型共用工具。

不要创建十几个网络文件。

---

# 12. 三个 Head 的语义固定

所有网络必须保留：

```text
Q Head
Value Head
Afterstate Value Head
```

## Q

```text
Q(s,a)
```

表示：

```text
当前正式 state s
执行 a
之后按当前目标策略继续游戏
从现在开始预计还能获得的总游戏分数
```

包括：

```text
当前动作 immediate merge reward
```

---

## V

```text
V(s)
```

表示：

```text
从正式 state s 开始，
之后按 greedy policy，
预计还能获得的未来游戏分数。
```

---

## A

```text
A(x)
```

其中：

```text
x = afterstate
```

表示：

```text
动作已经移动/合并完成，
但随机 tile 尚未 spawn 时，
从这个 afterstate 开始预计还有多少未来收益。
```

A 不包含：

```text
刚刚那个动作已经获得的 immediate reward
```

因为：

```text
Q ≈ reward + A
```

只是其价值关系。

---

# 13. 不允许把 Head 硬绑定

禁止 forward 中强制：

```python
Q = reward + A
V = max(Q)
```

三个 Head：

```text
是独立预测
```

只共享 backbone。

正式训练以后才通过 loss / targets 拉近语义关系。

---

# 14. 模型 API 固定

两个模型都实现相同 API：

```python
def encode(self, boards: torch.Tensor) -> torch.Tensor:
    ...

def forward(self, boards: torch.Tensor) -> torch.Tensor:
    """
    Final-inference path.
    Return Q only.
    shape: (B,4)
    """

def forward_state(
    self,
    boards: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return:
        q: (B,4)
        value: (B,)
    """

def forward_afterstate(
    self,
    afterstates: torch.Tensor,
) -> torch.Tensor:
    """
    Return:
        afterstate_value: (B,)
    """
```

这样：

```text
正式推理：
model(board)
→ Q only
```

训练：

```text
state
→ same backbone
→ Q + V

afterstate
→ same backbone
→ A
```

符合总计划。

---

# 15. Transformer baseline 固定结构

类：

```python
Transformer2048
```

固定 baseline：

```text
tile vocabulary          22
tokens                   16
hidden                   256
blocks                   6
attention heads          8
FFN                      1024
position embeddings      16 learned absolute
dropout                  0.0
activation               GELU
batch_first              True
norm_first               True
```

输入 token：

```text
TileEmbedding(token_i)
+
LearnedPositionEmbedding(i)
```

Position Embedding 必须是真正：

```text
nn.Parameter / trainable embedding
```

shape：

```text
(1,16,256)
```

或语义完全等价的：

```text
(16,256)
```

禁止省略。

---

# 16. Transformer backbone 固定实现方向

使用标准 PyTorch：

```text
nn.Embedding
nn.TransformerEncoderLayer
nn.TransformerEncoder
nn.LayerNorm
```

不要自己手写 attention kernel。

不要自己实现 FlashAttention。

不要引入：

```text
xformers
flash-attn
第三方 Transformer 库
```

第一版 baseline 保持标准 PyTorch。

---

# 17. Transformer pooling 固定

Transformer 最后一层得到：

```text
(B,16,256)
```

正式 baseline 使用：

```text
mean pooling over 16 tokens
```

得到：

```text
(B,256)
```

然后分别接：

```text
Q Linear: 256 → 4
V Linear: 256 → 1
A Linear: 256 → 1
```

V/A 最终 squeeze 为：

```text
(B,)
```

不要加入 CLS token。

总计划写的是：

```text
16 tile tokens
```

因此第一版不要偷偷加第 17 个 CLS token。

---

# 18. Transformer 输出禁止激活

Q/V/A 最后：

```text
raw real-valued scalar
```

禁止：

```text
softmax
sigmoid
tanh
ReLU clipping
```

因为价值可以覆盖很大的 future-score 数值空间。

---

# 19. Residual MLP baseline 固定

类：

```python
ResidualMLP2048
```

为了匹配 Transformer 约 5M 参数量，固定：

```text
tile embedding dim       32
16 cells
flatten input            16 × 32 = 512
hidden                   640
residual blocks          6
activation               GELU
dropout                  0.0
```

输入：

```text
22-class TileEmbedding
↓
(B,16,32)
↓
flatten
↓
(B,512)
↓
Linear 512→640
↓
GELU
```

---

# 20. Residual MLP block 固定

每个 block：

```text
x
↓
LayerNorm(640)
↓
Linear 640→640
↓
GELU
↓
Linear 640→640
↓
+
residual x
```

共：

```text
6 blocks
```

最后：

```text
LayerNorm(640)
```

然后：

```text
Q Linear: 640→4
V Linear: 640→1
A Linear: 640→1
```

不要擅自加入：

```text
convolution
attention
gating
MoE
recurrent layers
```

---

# 21. 参数量 gate

必须记录两个模型：

```text
trainable parameters
```

预期：

```text
Transformer ≈ 5M
Residual MLP ≈ 5M
```

Residual MLP 与 Transformer 参数量差距要求：

```text
<= 15%
```

如果实际超出：

先检查实现。

不要自行扩大网络。

---

# 22. Weight initialization

第一版保持简单。

允许：

```text
PyTorch Linear / LayerNorm 默认初始化
```

Embedding 和 Transformer position embedding：

使用：

```python
normal_(mean=0.0, std=0.02)
```

不要实现复杂初始化研究。

---

# 23. D4 必须建立正式 batched Torch 工具

新建：

```text
src/game2048/m2_symmetry.py
```

禁止修改冻结：

```text
src/game2048/symmetry.py
```

M2 symmetry 必须以冻结 M0 D4 为 oracle。

---

# 24. D4 batch API

实现至少：

```python
transform_board_batch(
    boards: torch.Tensor,
    transform_ids: torch.Tensor,
) -> torch.Tensor
```

输入：

```text
boards:
(B,16)

transform_ids:
(B,)
values 0..7
```

输出：

```text
(B,16)
```

必须保持：

```text
device
dtype
batch size
```

---

实现：

```python
transform_action_batch(
    actions: torch.Tensor,
    transform_ids: torch.Tensor,
) -> torch.Tensor
```

以及：

```python
transform_q_values(...)
inverse_transform_q_values(...)
```

用于：

```text
D4 label mapping
D4 consistency evaluation
```

---

# 25. D4 permutation 来源

不要手猜。

使用冻结：

```text
game2048.transform_board
game2048.transform_action
```

生成一次正式 permutation table。

可以在 import / constant construction 时生成：

```text
8 × 16 board permutation

8 × 4 action permutation
```

正式 batch transformation：

必须通过：

```text
Torch indexing / gather
```

完成。

禁止训练 hot path：

```python
for board in boards:
    transform_board(...)
```

---

# 26. D4 correctness test

对所有：

```text
8 transforms
```

以及多个随机 board，

Torch batched transform 必须逐项等于：

```text
冻结 M0 transform_board
```

action mapping：

所有：

```text
8 × 4
```

组合必须 exact equality：

```text
transform_action
```

Q permutation：

必须 round-trip：

```text
Q
→ transform
→ inverse transform
→ original Q
```

完全一致。

---

# 27. Legal mask / action selection 模块

创建：

```text
src/game2048/m2_policy.py
```

实现：

```python
apply_legal_mask(...)
select_greedy_actions(...)
```

不要重新实现游戏 legal rules。

legal mask 来源仍然是冻结 M1：

```text
legal_mask_batch
```

---

# 28. Legal masking

Q shape：

```text
(B,4)
```

legal mask：

```text
(B,4) bool
```

非法动作：

```text
必须不能被 argmax
```

mask 使用：

```text
-inf
```

或语义严格等价方式。

如果某 row：

```text
4 个动作全非法
```

不要偷偷选择 UP。

直接拒绝 / 明确报错。

正式 closed-loop 必须在动作选择前处理 terminal reset。

---

# 29. Q tie-breaking

禁止固定：

```text
UP > LEFT > DOWN > RIGHT
```

第一版固定：

```text
tie_atol = 1e-6
tie_rtol = 0
```

如果多个合法动作满足：

```text
abs(Q_i - Q_best) <= 1e-6
```

从这些合法并列动作：

```text
uniform random
```

选择。

API 必须允许传入：

```text
torch.Generator
```

以便 benchmark / evaluation 使用固定 seed。

---

# 30. Policy tests

至少测试：

```text
非法最高 Q 永远不会被选
单一合法动作必选
单一最大 Q 必选
exact tie 只从 tied legal actions 中选
近似 tie <= tolerance 生效
超出 tolerance 不视为 tie
terminal/all-false legal mask 被拒绝
固定 Generator 可复现
```

CPU pytest 即可。

---

# 31. 不要把正式 M2 变成 RL Trainer

M2 需要：

```text
backward correctness
tiny overfit
training throughput benchmark
```

但：

不要因此实现：

```text
Replay
TD
Double-Q
Target
EMA
Self-play learner
Teacher targets
```

M2 使用：

```text
synthetic supervised targets
```

只用于：

```text
验证 gradient
验证可过拟合
测 training throughput
```

必须在报告明确：

```text
这些 synthetic targets 没有棋力语义，
不是正式训练算法。
```

---

# 32. CPU correctness tests

创建：

```text
tests/test_m2_models.py
tests/test_m2_symmetry.py
tests/test_m2_policy.py
```

如果需要少量 pipeline CPU test：

```text
tests/test_m2_pipeline.py
```

所有 pytest：

```text
CPU 可运行
不依赖 CUDA
不 skip
不 xfail
```

---

# 33. `test_m2_models.py` 必须覆盖

两个模型都测试：

```text
input shape validation
input dtype validation
B=1
B>1
Q shape = (B,4)
V shape = (B,)
A shape = (B,)
forward == Q inference path
forward_state Q 与 forward Q 相同
finite output
backward works
embedding receives gradient
backbone receives gradient
Q head receives gradient
V head receives gradient
A head receives gradient
```

Transformer 额外：

```text
position embedding exists
shape correct
requires_grad = True
position embedding receives gradient
```

---

# 34. overflow token mapping test

明确测试：

```text
0..20 → unchanged

21 → 21
22 → 21
100 → 21
255 → 21
```

这是 NN vocabulary 规则。

不要测试成环境 exponent 被修改。

---

# 35. CPU CI 必须扩展到 M2

当前：

```text
.github/workflows/ci.yml
```

已经是 mandatory gate。

M2 新增 CPU Torch tests 后：

CI 必须能够实际运行它们。

因此：

需要给 CI 安装：

```text
CPU PyTorch
```

---

# 36. CI PyTorch 版本规则

先检查本机正式 CUDA PyTorch：

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

记录真实版本。

CI 使用：

```text
相同 base PyTorch version 的 CPU wheel
```

例如本地：

```text
X.Y.Z+cuXYZ
```

CI pin：

```text
torch==X.Y.Z
```

CPU wheel 从官方 PyTorch CPU index 安装。

不要：

```text
在 GitHub CPU runner 安装 CUDA wheel
```

如果对应 CPU wheel 真不存在：

报告并采用与本机兼容的同 minor CPU version，

但必须在 M2_REPORT 中解释。

---

# 37. CI 必须继续完整运行

最终 CI：

```bash
python -m pytest
```

必须执行：

```text
全部 M0
全部 M1
M2 Preflight
全部新 M2 CPU tests
```

要求：

```text
0 failed
0 skipped
0 xfailed
```

禁止通过：

```text
skip
xfail
ignore
pytest marker exclusion
```

躲 GPU。

GPU validation：

不要放进 pytest。

---

# 38. GPU 环境 gate

正式 GPU benchmark 前必须记录：

```text
GPU model
driver
torch version
CUDA runtime
Python version
OS
CPU
RAM
```

必须确认：

```python
torch.cuda.is_available()
```

为 True。

并确认实际设备是用户：

```text
RTX 5060
```

如果 CUDA 不可用：

```text
M2 INCOMPLETE
```

不得用 CPU benchmark 假装 GPU throughput。

---

# 39. 新建 GPU correctness runner

创建：

```text
benchmarks/validate_m2_models.py
```

职责只有：

```text
actual baseline model forward
actual baseline model backward
gradient finite
tiny dataset overfit
```

对：

```text
Transformer
Residual MLP
```

分别执行。

---

# 40. Tiny overfit 固定方法

每个模型使用固定 seed。

建立：

```text
128 state samples
128 afterstate samples
```

board：

```text
随机 uint8 exponent
主要范围 0..17
少量包含 18..255
```

合成固定：

```text
Q targets       (128,4)
V targets       (128,)
A targets       (128,)
```

targets：

使用固定 RNG 生成，

仅作为 memorization target。

不要调用 Teacher。

---

# 41. Tiny overfit loss

使用：

```text
MSE(Q)
+
0.5 * MSE(V)
+
0.5 * MSE(A)
```

Optimizer：

```text
AdamW
lr = 3e-4
weight_decay = 1e-4
```

Gradient clip：

```text
global norm = 1.0
```

先用：

```text
FP32 eager
```

进行 correctness overfit。

---

# 42. Tiny overfit PASS

必须同时满足：

```text
loss finite throughout
no NaN
no Inf
```

并且最终总 loss：

```text
<= initial loss × 0.20
```

即至少：

```text
80% reduction
```

同时：

```text
Q loss下降
V loss下降
A loss下降
```

如果 500 optimizer steps 内无法达到：

不要无限训练。

最多：

```text
1000 optimizer steps
```

仍无法达到：

```text
Correctness FAIL
```

调查模型/优化问题。

---

# 43. GPU correctness 输出

写：

```text
reports/m2/m2_correctness.json
```

至少记录：

```text
model
parameter_count
initial_total_loss
final_total_loss
Q loss before/after
V loss before/after
A loss before/after
steps
peak VRAM
NaN/Inf
PASS/FAIL
```

---

# 44. GPU throughput benchmark

创建：

```text
benchmarks/benchmark_m2_gpu.py
```

输出：

```text
reports/m2/m2_gpu_benchmark.json
```

必须测试两个模型：

```text
Transformer
Residual MLP
```

---

# 45. Batch-size sweep 固定

每个模型至少：

```text
256
512
1024
2048
4096
8192
```

每个 batch 必须分别测：

```text
Inference
Training
```

不得遗漏某档然后声称完整 sweep。

如果某档 OOM：

记录：

```text
OOM
```

不要 crash 后把结果删掉。

---

# 46. Inference benchmark 的定义

Inference 测：

```python
model(boards)
```

即：

```text
Q-only final inference path
```

因为最终正式 Agent：

```text
当前 board
→ one NN forward
→ Q(4)
```

不要把：

```text
afterstate forward
training heads
```

算进正式 inference throughput。

记录：

```text
states/s
batch latency
GPU utilization
allocated VRAM
reserved VRAM
```

---

# 47. Training throughput 定义

Training benchmark 每个样本定义为：

```text
1 state
+
1 afterstate
+
Q target
+
V target
+
A target
```

一次 optimizer step：

```text
forward_state(state)
forward_afterstate(afterstate)
loss Q + 0.5 V + 0.5 A
backward
global grad clip 1.0
AdamW step
zero_grad(set_to_none=True)
```

记录：

```text
training samples/s
optimizer step wall time
GPU utilization
peak allocated VRAM
peak reserved VRAM
```

`samples/s`：

按：

```text
B transition-pairs / second
```

计算。

不是：

```text
2B board forwards/s
```

可额外记录 board forwards/s，

但不能替代 samples/s。

---

# 48. Precision benchmark

必须测试：

## FP32

强制。

## BF16

如果：

```python
torch.cuda.is_bf16_supported()
```

为 True：

强制测试。

使用：

```text
torch.autocast(device_type="cuda", dtype=torch.bfloat16)
```

Optimizer 参数仍保持正常 FP32 master weights。

---

# 49. FP16 规则

如果：

```text
BF16 不支持
BF16 不稳定
BF16 correctness 有问题
```

则必须测试：

```text
FP16 + GradScaler
```

如果 BF16 在 RTX 5060 上稳定且可用：

FP16+GradScaler：

可以作为额外比较，

但不是强制替代 BF16。

必须在结果中说明：

```text
FP16 required?
YES / NO
```

---

# 50. 不允许通过降低 correctness 换吞吐

任何 precision 下如果出现：

```text
NaN
Inf
gradient explosion
loss non-finite
obvious invalid output
```

该配置：

```text
INVALID
```

不得因为 states/s 高而选择。

---

# 51. PyTorch eager vs compile

必须测试：

```text
eager
torch.compile
```

但不用对 6 个 batch × 所有 precision 做完整笛卡尔积。

固定方法：

对每个模型：

先通过 eager sweep 找到：

```text
最佳稳定 precision
最佳 throughput batch
```

然后至少在：

```text
batch 1024
+
该模型最佳 batch
```

比较：

```text
eager
vs
torch.compile
```

Inference 和 Training 都测。

---

# 52. `torch.compile` 失败处理

如果当前：

```text
PyTorch
Windows/Linux
driver
RTX 5060
```

组合导致：

```text
compile unsupported
compile exception
compile hang
compile regression
```

不要安装一堆第三方编译工具绕路。

记录：

```text
exact exception
environment
result = UNSUPPORTED / FAILED / REGRESSION
```

然后：

```text
允许正式使用 eager
```

总计划已经允许。

---

# 53. benchmark timing 规则

CUDA benchmark 必须：

```text
warmup
CUDA synchronize
multiple iterations
steady state
```

禁止：

```text
只测一次 forward
拿第一次 compile 时间算吞吐
异步 CUDA 未 synchronize 就读 wall-clock
```

---

# 54. 推荐 benchmark timing

每个配置：

```text
warmup >= 20 iterations
```

测量阶段：

至少满足：

```text
>= 100 iterations
或
>= 2 seconds steady-state
```

二者取更严格者。

如果某配置非常慢：

允许合理减少 iterations，

但测量窗口仍至少：

```text
2 seconds
```

---

# 55. CUDA latency

单 batch latency：

使用：

```text
CUDA Events
```

或经过同步的可靠等价方法。

End-to-end wall：

使用：

```text
time.perf_counter
+
正确 synchronize
```

不要混淆。

---

# 56. VRAM

记录：

```text
torch.cuda.max_memory_allocated()
torch.cuda.max_memory_reserved()
```

每个 benchmark point 前：

```text
reset_peak_memory_stats
```

---

# 57. GPU utilization

GPU utilization 必须真实采样。

优先使用：

```text
nvidia-smi
```

持续采样 workload 期间的：

```text
GPU utilization
VRAM
```

不要只在 benchmark 结束后读一次瞬时数字。

结果记录：

```text
mean GPU utilization
可获得时：
min / max
sample count
```

如果采样方法不可用：

不要伪造。

报告：

```text
UNAVAILABLE + reason
```

但由于用户机器为 RTX 5060，正式 M2 Exit 前应尽力取得真实数据。

---

# 58. 不根据单 batch latency 选配置

每个模型选择候选配置的主要标准：

```text
stable
+
correct
+
长期 throughput 最大
```

Inference：

```text
states/s
```

Training：

```text
samples/s
```

不能因为：

```text
batch 256 latency 最低
```

就选择 batch 256。

---

# 59. 超过 8192 的规则

8192 后不是强制。

如果：

```text
8192 无 OOM
且
peak reserved VRAM < 70% total VRAM
且
吞吐仍明显上升
```

则继续测试：

```text
16384
```

如果 16384 相比 8192：

```text
throughput gain >= 5%
```

且仍有显存：

可测试：

```text
32768
```

否则停止扩大。

不要为了“大 batch 看起来厉害”无限扩大。

---

# 60. 不在 M2 选择棋力赢家

M2 的网络尚未经过正式 Teacher / Self-play 训练。

因此禁止写：

```text
Transformer 棋力更强
MLP 棋力更强
```

M2 只能比较：

```text
correctness
parameter count
GPU throughput
VRAM
latency
closed-loop throughput
```

真实棋力比较属于后续训练阶段。

两个模型都必须保留。

---

# 61. 建立 M1 + M2 closed-loop benchmark

创建：

```text
benchmarks/benchmark_m2_closed_loop.py
```

输出：

```text
reports/m2/m2_closed_loop_benchmark.json
```

必须使用：

```text
Fast2048BatchEnv
+
真实 legal_mask_batch
+
M2 contiguous boundary
+
GPU NN
+
legal masking
+
greedy/tie-random action selection
+
action batch 回 CPU
+
FastEnv.step
+
reset_where
```

---

# 62. 正式 closed-loop

循环固定：

```text
Fast2048BatchEnv
↓
current board batch
↓
prepare_board_batch_for_transfer
↓
CPU→GPU
↓
model Q forward
↓
legal mask
↓
mask illegal actions
↓
random tie-safe greedy action selection
↓
action GPU→CPU
↓
FastEnv.step
↓
reset terminal rows
↓
next iteration
```

不得使用：

```text
随机动作 policy
Teacher
search
```

closed-loop 必须真的是：

```text
NN controlling FastEnv
```

---

# 63. Legal mask 路径

当前 legal mask：

使用冻结：

```python
legal_mask_batch(...)
```

不要重新写 Torch 版 2048 movement 来算 legal。

M2 不重新实现环境。

---

# 64. terminal reset 顺序

每次 step 返回：

```text
terminated
```

下一次动作选择之前：

```text
reset_where(terminated)
```

确保送给 policy 的 board：

不是：

```text
4 actions all illegal
```

的 terminal board。

---

# 65. closed-loop batch/environment sweep

两个模型分别至少测试：

```text
1024 envs
4096 envs
8192 envs
```

如果：

```text
8192 仍明显受小 batch 限制
且资源允许
```

加：

```text
16384
```

不要求为了形式跑超过实际价值的规模。

---

# 66. closed-loop 模型配置

每个模型使用：

```text
benchmark_m2_gpu.py
确定的最佳稳定 inference precision
+
最佳稳定 execution mode
```

不要人为给 Transformer 和 MLP 不同质量的 benchmark 条件。

各自都用自己在 RTX 5060 上的合理最佳配置。

---

# 67. closed-loop 至少记录

每个配置：

```text
complete agent decisions/s
environment transitions/s
NN states/s
CPU→GPU transfer time
CPU→GPU effective bandwidth
GPU→CPU action transfer time
GPU utilization
CPU utilization
peak VRAM
pipeline stall time
```

---

# 68. wall-clock breakdown

必须输出：

```text
Environment stepping        xx%
Legal mask                  xx%
CPU batch preparation       xx%
CPU → GPU transfer          xx%
NN inference                xx%
legal mask / action select  xx%
GPU → CPU action transfer   xx%
reset_where                 xx%
Other                       xx%
```

训练 benchmark 另外记录：

```text
NN backward
Optimizer
```

不要把所有时间都叫：

```text
Other
```

---

# 69. profiling 与 throughput 分两种模式

非常重要。

## Throughput mode

目标：

```text
最大真实 end-to-end throughput
```

尽量少做：

```text
额外 synchronize
额外 logging
额外 instrumentation
```

---

## Profile mode

允许：

```text
CUDA synchronize
stage timers
CUDA events
```

来拆 wall-clock。

不要把高度 instrumentation 的 profile mode：

误当正式吞吐数字。

报告中要区分。

---

# 70. CPU utilization

使用现有 M1 benchmark utility：

```text
benchmarks/_utils.py
```

能复用的：

```text
environment metadata
CPU timing
RSS
format helpers
```

优先复用。

不要复制一整套新的 benchmark util。

---

# 71. 允许的 M2 pipeline 优化

如果第一次 closed-loop 出现 starvation，

在不重写 M1 的前提下，允许按 evidence 做：

```text
提高 batch/env count
减少临时 tensor allocation
复用 tensor buffer
合理 pinned staging buffer
non_blocking H2D
减少 CPU→GPU transfer 次数
减少不必要 dtype conversion
```

每做一项：

必须：

```text
before benchmark
↓
change
↓
after benchmark
```

记录真实 improvement。

---

# 72. 不允许凭感觉 double-buffer

以下属于更强 pipeline 变化：

```text
worker pipeline
double buffering
复杂异步 producer/consumer
```

只有 profile 明确证明：

```text
等待 CPU producer / H2D
```

是主要瓶颈时才允许。

实施前在报告记录：

```text
why
baseline %
expected target
```

实现后必须 A/B benchmark。

---

# 73. 明显 GPU starvation 的 M2 operational gate

不要仅凭：

```text
GPU utilization 一个数字
```

判断。

如果同时出现以下任一组：

### A

```text
steady-state mean GPU utilization < 60%
并且
CPU environment + batch preparation + transfer
>= 40% profiled wall-clock
```

### B

```text
measured pipeline stall >= 20% total wall-clock
```

### C

```text
model-only inference capacity
明显高于 closed-loop NN feed rate
且
closed-loop 只能达到 model-only states/s 的 < 70%
并且差距由 CPU/H2D/profile 解释
```

则视为：

```text
OBVIOUS GPU STARVATION
```

---

# 74. starvation 处理

如果触发：

先只做本文允许的安全 M2 pipeline 优化。

然后重新测。

如果优化后仍触发：

```text
M2 RESULT = INCOMPLETE
```

报告：

```text
主要瓶颈
wall-clock %
GPU util
model-only capacity
closed-loop capacity
已尝试的优化
A/B improvement
```

然后 STOP。

不要自行上 C++。

---

# 75. 如果没有明显 starvation

如果：

```text
closed-loop 吞吐稳定
profile 不显示 CPU/H2D 明显饿 GPU
继续优化环境预计收益已经很低
```

则：

```text
GPU starvation gate PASS
```

不要为了追求 GPU 99% utilization：

无意义重写环境。

---

# 76. M2 benchmark artifact

最终至少交付：

```text
reports/m2/m2_correctness.json
reports/m2/m2_gpu_benchmark.json
reports/m2/m2_closed_loop_benchmark.json
reports/m2/M2_REPORT.md
```

不要覆盖 M1：

```text
reports/m1/M1_REPORT.md
benchmark_results.json
```

---

# 77. `reports/m2/M2_REPORT.md`

必须包含：

```text
hardware/software environment
starting commit
M0/M1 frozen tags
M2 Preflight status
files changed
model architectures
parameter counts
input encoding
Q/V/A APIs and semantics
D4 test result
legal mask test result
forward/backward result
tiny overfit result
precision benchmark
batch sweep
eager/compile result
inference throughput
training throughput
VRAM
closed-loop throughput
wall-clock profile
GPU starvation analysis
optimizations attempted
full pytest
GitHub Actions status
remaining blockers
```

---

# 78. 不允许改主目标

不要新增：

```text
16384 bonus
32768 bonus
65536 bonus
```

M2 根本不做棋力训练。

主项目目标仍然是：

```text
最大化平均最终游戏分数
```

---

# 79. 不允许偷偷训练网络打 2048

M2 当前模型：

```text
只做 correctness + systems benchmark
```

不要因为能跑环境就开始：

```text
百万局 self-play
TD learning
Teacher imitation
```

这属于后续 milestone。

---

# 80. M0/M1 回归永远必须绿

任何 M2 change 后：

最终必须：

```bash
python -m pytest
```

要求：

```text
全部现有 M0
全部 M1
33 条 M2 Preflight
全部新增 M2 CPU tests

0 failed
0 skipped
0 xfailed
```

如果 frozen regression 失败：

不要改 frozen tests 来适配 M2。

修 M2。

---

# 81. 禁止 frozen test 弱化

不得：

```text
删除
skip
xfail
减少 sample count
放宽 assertion
修改 expected value
pytest ignore
修改 collection
```

制造绿色。

---

# 82. GitHub Actions mandatory

M2 candidate push 后：

必须等实际 GitHub Actions 运行。

必须：

```text
PASS
```

如果：

```text
local tests PASS
remote Actions FAIL
```

则：

```text
M2 FAIL / INCOMPLETE
```

不能宣布 candidate complete。

---

# 83. GPU benchmark 不能放 CI

GitHub Actions CPU CI：

只跑：

```text
CPU-testable correctness
```

GPU benchmark：

用户 RTX 5060 本机执行。

不要让 CI 因无 GPU 而：

```text
skip GPU pytest
```

正确做法：

```text
GPU benchmark 不属于 pytest suite
```

---

# 84. 建议文件修改范围

正式 M2 正常新增：

```text
src/game2048/m2_models.py
src/game2048/m2_symmetry.py
src/game2048/m2_policy.py

tests/test_m2_models.py
tests/test_m2_symmetry.py
tests/test_m2_policy.py

benchmarks/validate_m2_models.py
benchmarks/benchmark_m2_gpu.py
benchmarks/benchmark_m2_closed_loop.py

reports/m2/M2_REPORT.md
reports/m2/m2_correctness.json
reports/m2/m2_gpu_benchmark.json
reports/m2/m2_closed_loop_benchmark.json
```

根据真实需要可增加：

```text
一个很小的 M2 benchmark helper
或
一个明确的 M2 pipeline helper
```

但不要建立庞大框架。

---

# 85. 正常允许修改

允许修改：

```text
.github/workflows/ci.yml
```

用于：

```text
安装 CPU PyTorch
继续运行完整 pytest
```

---

# 86. 默认不允许修改

除非发现真实 blocker，否则不要修改：

```text
src/game2048/reference_env.py
src/game2048/symmetry.py
src/game2048/fast_env.py
src/game2048/m2_data.py

M0 tests
M1 tests
tests/test_m2_preflight.py

authoritative master plan
```

如果认为必须修改：

先停止并在报告写：

```text
UNEXPECTED BLOCKER
```

不要自行打开 frozen stage。

---

# 87. `game2048.__init__` 不要污染

当前：

```text
game2048.__init__
```

明确主要暴露 frozen M0 API。

不要把：

```text
Transformer
MLP
benchmark
training
```

全部塞入 package root。

正式 M2 代码使用：

```python
from game2048.m2_models import ...
```

等明确模块路径。

---

# 88. 测试开发顺序固定

实现顺序：

```text
A. model input/tokenization
↓
B. Transformer
↓
C. Residual MLP
↓
D. Q/V/A API
↓
E. D4 batch utilities
↓
F. legal mask/action selection
↓
G. CPU unit tests
↓
H. full pytest + CI-compatible state
↓
I. GPU correctness / overfit
↓
J. GPU throughput sweep
↓
K. eager/compile benchmark
↓
L. closed-loop benchmark
↓
M. profile / starvation analysis
↓
N. only evidence-based safe pipeline tuning if needed
↓
O. rerun everything
↓
P. report
↓
STOP
```

不要颠倒成：

```text
先 benchmark 半成品
先写 C++
先写 Trainer
```

---

# 89. 每个阶段发现 failure 时

Correctness failure：

```text
先修 correctness
```

不得用：

```text
benchmark 更快
```

掩盖。

GPU OOM：

```text
记录 OOM
继续其他合法 batch
```

不得静默跳过。

Compile failure：

```text
记录
允许 eager
```

Starvation：

```text
profile
安全优化
复测
```

持续严重：

```text
INCOMPLETE + STOP
```

---

# 90. 不创建 M2 tag

你当前只产生：

```text
M2 candidate commit
```

不要自行创建：

```text
m2-pass
m2-final
m2-network-pass
```

外部独立审计之后再决定 tag。

绝对不要移动：

```text
m0-reference-pass
m1-fastenv-audited-pass
```

---

# 91. Commit 规则

可以有开发过程 commit，

但最终 push 到 main 的 M2 candidate 必须：

```text
源码
CPU tests
benchmark scripts
benchmark result JSON
reports/m2/M2_REPORT.md
CI changes
```

彼此一致。

不要把：

```text
用户自己的 prompt rename/delete
```

夹入 M2 commit。

---

# 92. 最终本地复验

结束前至少执行：

```bash
python -m pytest
```

以及：

```text
validate_m2_models.py
benchmark_m2_gpu.py
benchmark_m2_closed_loop.py
```

正式 run。

不要拿：

```text
--quick
debug
reduced batches
short smoke mode
```

的结果冒充正式 M2 benchmark。

---

# 93. 最终远程复验

push M2 candidate 后：

确认：

```text
main == pushed candidate
```

GitHub Actions：

```text
CPU test suite PASS
```

读取真实 log，

记录：

```text
collected N items
N passed
```

不得只说：

```text
green icon
```

---

# 94. M2 PASS / INCOMPLETE 判定

你只有在以下全部成立时：

```text
Transformer implemented
Residual MLP implemented
Q/V/A all implemented
Position Embedding present
parameter-count gate satisfied
CPU forward tests PASS
CPU backward tests PASS
D4 tests PASS
legal-mask/action-selection tests PASS
GPU forward PASS
GPU backward PASS
both models tiny-overfit PASS
precision benchmark complete
required batch sweep complete
eager/compile benchmark complete or compile failure properly documented
inference states/s recorded
training samples/s recorded
VRAM recorded
closed-loop benchmark complete
wall-clock breakdown complete
main bottleneck identified
no obvious GPU starvation
full pytest PASS
0 failed
0 skipped
0 xfailed
GitHub Actions actual run PASS
M0/M1 tags unchanged
no frozen tests weakened
no Teacher/tuple/replay/self-play/C++ scope creep
```

才可以写：

```text
M2 CANDIDATE COMPLETE
```

注意：

不是：

```text
M2 AUDITED PASS
```

外部审计还没发生。

---

# 95. 如果任何必需 GPU 项缺失

例如：

```text
CUDA unavailable
GPU benchmark incomplete
8192 没测且不是 OOM
closed-loop 未完成
GPU utilization 没记录且没有合理原因
starvation 尚未解决
remote CI 未绿
```

则：

```text
M2 RESULT = INCOMPLETE
```

不要包装成 PASS。

---

# 96. 最终报告格式固定

最终回答必须包含：

```text
1. RESULT

M2 CANDIDATE COMPLETE
或
M2 INCOMPLETE
或
M2 FAIL


2. STARTING STATE

- initial HEAD
- initial git status
- master plan read
- m0 tag
- m1 tag
- M2 Preflight audited commit


3. ENVIRONMENT

- OS
- Python
- CPU
- RAM
- GPU
- driver
- torch
- CUDA
- BF16 support


4. FILES CHANGED

逐文件：
- path
- exact purpose


5. MODELS

Transformer:
- architecture
- params
- position embedding
- Q/V/A API

Residual MLP:
- architecture
- params
- parameter ratio


6. INPUT CONTRACT

- NumPy boundary
- Torch board shape/dtype
- overflow token mapping


7. CORRECTNESS

- CPU forward
- CPU backward
- D4
- legal mask
- tie handling
- Q/V/A shape
- Transformer position embedding gradient
- tiny overfit Transformer
- tiny overfit MLP


8. PYTEST

python -m pytest

- collected
- passed
- failed
- skipped
- xfailed


9. GPU PRECISION

FP32:
...

BF16:
...

FP16 + GradScaler:
required/not required/result


10. BATCH SWEEP

For each model:
256
512
1024
2048
4096
8192
optional larger

Inference:
- states/s
- latency
- GPU util
- VRAM

Training:
- samples/s
- step time
- GPU util
- VRAM


11. EAGER VS COMPILE

Transformer:
...

MLP:
...

compile failure if any:
exact error


12. SELECTED M2 SYSTEMS CONFIG

Transformer:
- precision
- batch
- eager/compile
- reason

MLP:
- precision
- batch
- eager/compile
- reason


13. CLOSED LOOP

For each model / env count:
- decisions/s
- env transitions/s
- NN states/s
- H2D
- D2H
- GPU util
- CPU util
- VRAM
- stall


14. WALL CLOCK PROFILE

Environment stepping
Legal mask
CPU batch prep
H2D
NN
action selection
D2H
reset
other


15. GPU STARVATION GATE

- PASS / FAIL
- evidence
- model-only throughput
- closed-loop throughput
- pipeline stall
- main bottleneck


16. PERFORMANCE OPTIMIZATIONS

For every optimization:
- evidence before
- exact change
- benchmark after
- improvement


17. CI

- workflow
- candidate commit
- Actions run id
- actual log
- collected
- passed
- conclusion


18. FROZEN VERIFICATION

- m0 tag unchanged
- m1 tag unchanged
- frozen tests deleted: NONE
- skip added: NONE
- xfail added: NONE


19. OUT OF SCOPE

- Teacher: NO
- tuple checkpoint: NO
- Expectimax: NO
- Replay: NO
- Self-play learner: NO
- Double-Q: NO
- Target network: NO
- C++: NO
- pybind11: NO
- custom CUDA env: NO
- M3 started: NO


20. ARTIFACTS

- reports/m2/M2_REPORT.md
- reports/m2/m2_correctness.json
- reports/m2/m2_gpu_benchmark.json
- reports/m2/m2_closed_loop_benchmark.json


21. FINAL GIT STATE

git status --short

git rev-parse HEAD

git diff --stat <starting-head>..HEAD


22. REMAINING BLOCKERS

NONE

or exact blockers.
```

---

# 97. 最终行为

完成报告以后：

# STOP

不要：

```text
更新 master plan 为 M2 PASS
创建 M2 tag
进入 M3
加载 tuple checkpoint
开始 Teacher
开始 Self-play
```

用户会将：

```text
M2 report
+
GitHub commit
+
benchmark artifacts
```

交给独立审计。

只有独立审计通过以后：

才会继续后续阶段。

---

现在开始。

第一件事：

```text
完整阅读 authoritative master plan
+
核对 HEAD / tags / working tree
```

然后严格按本施工单执行正式 M2。