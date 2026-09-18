# M3 Teacher 小规模验证 — 严格施工提示词

> **状态：HISTORICAL / FINAL AUDITED PASS / FROZEN。M3 已完成独立审计，本施工单保留用于复现与审计，不得作为当前阶段重新执行。当前阶段见 authoritative master plan。**
>
> 项目：Safolour/NN-2048-ai
> 正式工作区：D:\CodexTasks\NN-2048-ai
> 当前阶段：M3 Teacher 小规模验证
> 本文件是 M3 的详细施工单，不是新的总计划。
> 唯一权威总计划始终优先。

---

# 0. 必须先读的权威文件

Agent 开工前必须完整读取：

1. `prompts/2048_AI_正式执行计划_M0_M1冻结_CI增强版_2026-09-18.md`
2. 本文件

M2 已完成，不得重新执行。

如果本文件与总计划发生真实冲突：

```text
STOP
报告冲突
不得自行裁决
```

---

# 1. 当前冻结状态

M2 closeout parent 固定为：

```text
4139b3b3c9c11216e7048bad46147e2fc5cc4e92
docs: close M2 and advance plan to M3
```

本 M3 prompt / master-plan Teacher baseline 回写会产生新的 docs-only HEAD，因此 M3 开工时不得机械要求 HEAD 等于 4139b3b。

必须验证：

```text
git merge-base --is-ancestor 4139b3b3c9c11216e7048bad46147e2fc5cc4e92 HEAD
```

并确认 4139b3b 之后、M3 开工之前只有 planning / prompts / master-plan docs 更新。

若 4139b3b 之后存在未经审计的 src/ / cpp/ / tests/ / benchmarks/ 实现变更：

```text
STOP
```

冻结里程碑：

```text
M0
tag: m0-reference-pass
commit: 3f2def1d95f56eff776e671143188947bf64485b

M1
tag: m1-fastenv-audited-pass
commit: e5486017a90eeec4fb9814de7880b3c413dc06dd

M2
tag: m2-network-audited-pass
audited implementation commit:
6a9da5b1bf7c72207acd6e89bc667d439d4823a8
```

M2 final:
- full pytest: 505 / 505
- GitHub Actions closeout run #10 / 35376846598: PASS
- production fast backend: scalar+row-lut
- SIMD=False
- LUT=True
- selected rollout scale: 32768 envs
- measured closed-loop: 704,818.03 decisions/s
- starvation A=False / B=False / C=False

这些全部视为冻结基础设施。

---

# 2. 本任务唯一目标

M3 只验证：

```text
现有 tuple 8×6 Teacher checkpoint
+
Expectimax Teacher Search
```

是否满足后续 M5 大规模 Teacher 预训练的前置条件。

M3 必须回答四件事：

1. tuple checkpoint 能否被确定无歧义地读取与复现；
2. Teacher evaluator / Search value 的数值单位到底是什么；
3. 小规模 Teacher 数据能否让 Student 稳定学习；
4. Teacher / Search wall-clock 组成是否已经可测量，是否存在数量级 Python 性能损失。

M3 不做正式大规模 Teacher training。

---

# 3. M3 固定 Teacher checkpoint

M3 baseline 已由外部规划/审计阶段选定。

禁止 Agent 自行重新选模型。

固定 snapshot：

```text
D:\CodexTasks\2048-ai\oneclick-m6-report\snapshots\COMPARATOR\ep4800000_7192719323a0.bin
```

固定 SHA-256：

```text
7192719323a073ba2b6b19b62cb7d46ef4aa90ecc8c4ae6baf27ad0c51566a84
```

source run：

```text
D:\CodexTasks\2048-ai\runs\20260916T182634Z_m6-ordinary-td-comparator-seed1
```

source project：

```text
D:\CodexTasks\2048-ai
```

source run metadata：

```text
git commit:
172e7d1a1d501282899b8a5a2e251c174c43f98f

git dirty:
True

resolved config SHA-256:
6f4f6781eceeda496a3e600f754708ac644fc68be0057d4fe4bcb847ea0cc61f
```

因为 source run 记录为 `dirty=true`，
M3 如果调用原项目 loader / evaluator，
除了记录 git commit 之外，
还必须记录实际使用的关键 source / executable 的 SHA256；
不得只靠 git commit 声称完全可复现。

checkpoint metadata 已知：

```text
episode: 4,800,000 / 10,000,000
size: 512 MiB verified snapshot
format magic: U2048NT6
format_version: 2
feature_schema: 4bit_exponent_6tuple
patterns: 8
symmetry: D4
algorithm: afterstate_td0
action_value: move_reward + V(afterstate)
policy: greedy_1ply
search: none
```

## 3.1 为什么固定这一版

相同 dev-fast 2,000 frozen seeds：

```text
COMPARATOR ep4.8M mean:
224,947

FORMAL ep6.7M mean:
200,840

paired delta:
+24,107

95% CI:
[+19,077, +29,049]

wins:
COMPARATOR 1,193
FORMAL 805
ties 2
```

因此 M3 baseline 固定选择 COMPARATOR ep4.8M。

同时必须保留 trade-off：

```text
p10:
COMPARATOR 80,361
FORMAL 89,919

reach >=8192:
COMPARATOR 88.40%
FORMAL 89.50%
```

禁止把“paired mean 显著更强”改写成“所有指标全面更强”。

## 3.2 不追 latest.bin

M3 整个 milestone：

```text
禁止自动读取 latest.bin
禁止训练一推进就换 checkpoint
禁止同一 dataset 混入不同 Teacher checkpoint
```

即使 source run 后续到达 10M：

```text
M3 仍使用上述固定 SHA snapshot
```

只有：

```text
M3 独立审计完成
+
准备进入 M5
```

才允许执行 master plan §31.4 的 Teacher Checkpoint Promotion Gate。

Agent 在 M3 中不得自行 promotion。

## 3.3 snapshot 缺失或 hash 不符

如果固定 snapshot 不存在：

```text
BLOCKED_CHECKPOINT_PATH
STOP
```

如果实际 SHA256 不等于：

```text
7192719323a073ba2b6b19b62cb7d46ef4aa90ecc8c4ae6baf27ad0c51566a84
```

则：

```text
BLOCKED_CHECKPOINT_MISMATCH
STOP
```

禁止 fallback 到 latest.bin 或其他 checkpoint。

---

# 4. checkpoint 本地物化与保护规则

上游 source snapshot 永远只读：

```text
D:\CodexTasks\2048-ai\oneclick-m6-report\snapshots\COMPARATOR\ep4800000_7192719323a0.bin
```

M3 不允许长期直接依赖另一个训练项目的运行目录。

因此 M3 Agent 开工时必须把已经冻结并校验过的 source snapshot
做一次**字节级只读复制**到本项目自己的本地 checkpoint 区。

固定本项目结构：

```text
D:\CodexTasks\NN-2048-ai
│
├─ teacher_checkpoints/
│  ├─ README.md
│  ├─ manifest.json
│  │
│  └─ m3/
│     └─ ordinary_td_comparator_ep4800000_7192719323a0.bin
│
└─ artifacts/
   └─ m3/
```

正式 M3 runtime checkpoint 固定为：

```text
D:\CodexTasks\NN-2048-ai\teacher_checkpoints\m3\ordinary_td_comparator_ep4800000_7192719323a0.bin
```

## 4.1 物化顺序固定

必须严格按以下顺序：

```text
1. 验证 upstream source snapshot 存在
2. 计算 upstream SHA256
3. upstream SHA 必须等于固定 expected SHA
4. 创建 teacher_checkpoints/m3/
5. 创建/更新 teacher_checkpoints/README.md
6. 创建/更新 teacher_checkpoints/manifest.json
7. 在 .gitignore 中加入 binary ignore 规则
8. 将 upstream snapshot 复制到固定 local path
9. 重新计算 local copy SHA256
10. 比较 source size / local size
11. local SHA 必须与 source SHA、expected SHA 三者完全一致
12. 验证 local .bin 被 Git ignore
13. 从此 M3 runtime 只读取 local copy
```

expected SHA256：

```text
7192719323a073ba2b6b19b62cb7d46ef4aa90ecc8c4ae6baf27ad0c51566a84
```

禁止：

```text
move
rename upstream
overwrite upstream
convert upstream in place
normalize weights in place
hardlink / junction / symlink 代替独立副本
fallback 到 latest.bin
```

要求是真正的独立文件副本。

## 4.2 destination 已存在时

如果 local destination 已存在：

先计算 SHA256。

如果：

```text
local SHA == expected SHA
```

则：

```text
复用现有 local copy
不得重复覆盖
```

如果：

```text
local SHA != expected SHA
```

则：

```text
BLOCKED_LOCAL_CHECKPOINT_MISMATCH
STOP
```

不得自动删除、覆盖或“修复”现有文件。

## 4.3 Git 跟踪规则

真实 `.bin` checkpoint：

```text
必须存在于项目目录
但必须保持 Git ignored / untracked
```

`.gitignore` 至少加入：

```gitignore
# Local external Teacher checkpoints
teacher_checkpoints/**/*.bin
```

必须实际验证：

```text
git check-ignore -v teacher_checkpoints/m3/ordinary_td_comparator_ep4800000_7192719323a0.bin
```

且 `git status --short` 中不得出现该 `.bin`。

允许并要求提交 Git 的只有：

```text
teacher_checkpoints/README.md
teacher_checkpoints/manifest.json
.gitignore 的对应规则
```

## 4.4 manifest.json 固定最小字段

`teacher_checkpoints/manifest.json` 至少必须保存：

```json
{
  "m3_teacher_baseline": {
    "relative_path": "m3/ordinary_td_comparator_ep4800000_7192719323a0.bin",
    "sha256": "7192719323a073ba2b6b19b62cb7d46ef4aa90ecc8c4ae6baf27ad0c51566a84",
    "episode": 4800000,
    "format": "U2048NT6",
    "format_version": 2,
    "feature_schema": "4bit_exponent_6tuple",
    "algorithm": "afterstate_td0",
    "role": "M3 frozen teacher baseline",
    "source_snapshot": "D:\\CodexTasks\\2048-ai\\oneclick-m6-report\\snapshots\\COMPARATOR\\ep4800000_7192719323a0.bin",
    "source_run": "D:\\CodexTasks\\2048-ai\\runs\\20260916T182634Z_m6-ordinary-td-comparator-seed1"
  }
}
```

允许增加 provenance 字段，
不得删除上面的核心身份字段。

## 4.5 README.md 必须说明

至少说明：

- binary checkpoint 不进 Git；
- 固定文件名；
- 固定 SHA256；
- M3 为什么固定这一版；
- 如何验证 SHA；
- 上游 source snapshot 只用于 provenance / initial copy；
- M3 runtime 只读本地副本；
- 后续 Teacher promotion 不得覆盖 M3 baseline。

## 4.6 运行时路径规则

完成本地物化以后：

所有 M3：

```text
loader
search
calibration
dataset generation
student sanity
profiling
```

默认 checkpoint path 一律使用：

```text
teacher_checkpoints/m3/ordinary_td_comparator_ep4800000_7192719323a0.bin
```

不得继续把上游 `D:\CodexTasks\2048-ai\...` 路径作为日常运行依赖。

上游路径只保留在 manifest / report 里作为 provenance。

## 4.7 派生产物目录

任何：

```text
index cache
converted read-only cache
profile output
teacher dataset
calibration output
```

必须放：

```text
artifacts/m3/
```

或施工单明确允许的新 M3 artifact 路径。

不得写回：

```text
teacher_checkpoints/m3/*.bin
```

---

# 5. checkpoint format gate

必须先确定 checkpoint 的真实格式和 evaluator 语义。

允许：
- 使用用户提供的原始 loader；
- 对明确、自描述格式编写最小只读 loader；
- 对照用户提供的原实现做输出 differential。

禁止盲目 reverse-engineer 一个无法确认字段语义的二进制格式后继续训练。

如果无法可靠确定：
- tuple pattern
- weight dtype
- indexing / symmetry rule
- board exponent encoding
- value output definition

则：

```text
BLOCKED_CHECKPOINT_FORMAT
STOP
```

不要进入 Search。

---

# 6. tuple evaluator adapter

若 checkpoint 可可靠读取，新建 M3 专用 adapter。

建议模块：

```text
src/game2048/m3_tuple_teacher.py
```

只负责：
- checkpoint load
- board -> tuple evaluator raw value
- batch evaluator（若原格式允许）
- metadata
- deterministic read-only inference

不得修改 M2 网络或 M2 C++ backend。

输入 board 继续使用：
`uint8[16]` exponent representation。

不得偷偷改成 4-bit board 作为正式状态表示。

---

# 7. raw evaluator 的语义默认 UNKNOWN

最重要的默认规则：

```text
Teacher raw value != future-score
```

除非有证据证明二者语义一致。

禁止因为：
- 数值大小看起来像分数；
- checkpoint 名字里有 value；
- 相关项目曾经用 TD；
就直接把 raw value 当 Student Q target。

必须给 raw evaluator metadata 标注：

```text
value_semantics =
UNKNOWN_RAW_HEURISTIC
```

直到 M3 calibration gate 明确通过。

# 8. Search baseline 固定语义

M3 第一版 Teacher Search 固定为 Expectimax。

节点语义：

```text
formal state s
↓ player action
afterstate x
↓ chance spawn
next formal state s'
```

Player node：
- 只考虑合法动作；
- 取 max。

Chance node：
- 空格位置均匀；
- 2 = 90%；
- 4 = 10%；
- 按真实概率求期望。

Action edge：
- 必须加真实 immediate merge reward。

Terminal：
- future value = 0。

---

# 9. Search depth 定义锁死

M3 使用：

```text
decision depth
```

Chance node 不计 depth。

```text
3-depth
= 3 次玩家决策
```

不得把：
- player node + chance node
算成 2 层。

Search API / report 必须使用：
`decision_depth`
这个字段名。

---

# 10. M3 第一版深度固定

M3 correctness / semantics baseline：

```text
decision_depth = 3
```

不得自由选择 2/4/6。

只允许在 profiling 完成后额外做少量：
`depth=5`
scaling probe。

M3 不要求 depth=7 正式生产数据。

如果 depth=3 已存在数量级性能问题：
不得用降低到 depth=1 冒充 M3 PASS。

---

# 11. Search leaf 固定

现有 tuple checkpoint 的原生 evaluator 是：

```text
afterstate value:
V_tuple(x)
```

所以不能假装它直接提供 neural-style `V(s)`。

M3 formal-state leaf adapter 固定为：

```text
L_tuple(s)
=
max over legal a [
    immediate_reward(s,a)
    +
    V_tuple(afterstate(s,a))
]
```

M3 Teacher Search 的显式叶子仍必须落在：

```text
正式 post-spawn state s
```

最后一次玩家动作后：
- 先形成 afterstate；
- 完整展开下一 chance node；
- 得到 formal state；
- Search 在该 formal state 停止；
- 调用 `L_tuple(s)` 作为 leaf heuristic。

固定 metadata：

```text
leaf_evaluator = tuple_afterstate_greedy_1ply_adapter
```

重要：

```text
checkpoint 的 greedy_1ply
不是 Search decision_depth 上限
```

Search 的：

```text
decision_depth = 3
```

只计算显式 Expectimax 中的三次玩家决策。

leaf adapter 内部为了把 afterstate evaluator 适配成 formal-state heuristic
所做的一次 greedy action scan：

```text
不额外计入 Search decision_depth
```

禁止：
- 在 afterstate 直接停并仍声称是同一种 formal-state search；
- 把 `L_tuple(s)` 误称 neural `V(s)`；
- 把 1-ply tuple 当成只能做 depth=1 Teacher。

---

# 12. Search correctness oracle

正式 Teacher Search 必须先有 CPU correctness tests。

至少覆盖：
- legal action filtering
- terminal
- immediate reward addition
- chance probability sum = 1
- spawn 90/10
- empty-cell uniformity
- depth=1/2/3 的 node-count / semantic checks
- deterministic evaluator fixture
- handcrafted board expected backup
- high tile exponent
- illegal action never selected
- no board mutation

M0 继续作为游戏规则 Golden Reference。

M2 C++ backend可作为高速 primitive，
但不得为了 Search 修改 frozen M2 semantics。

---

# 13. Teacher action sample schema 锁死

每个 formal state 必须保存四动作维度。

建议正式 schema：

```text
state                 uint8[16]
teacher_value         float64[4]
reward                int64[4]
afterstate            uint8[4,16]
legal_mask            bool[4]

game_id               int64
step_index             int32
current_score          int64
max_tile_exp           uint8

teacher_version        string
checkpoint_sha256      string
decision_depth         int32
data_source            enum/string
value_semantics        enum/string
search_mode            string
```

非法动作固定：
- legal_mask = False
- teacher_value = NaN
- reward = 0
- afterstate = all-zero sentinel

非法项不得参与 loss、argmax 或 ranking metric。

---

# 14. M3 小规模数据预算固定

M3 分两级。

## A. semantic/profile set

固定收集：

```text
至少 256 formal states
来自至少 16 个完整 game_id
```

用途：
- value semantics
- Search correctness
- profiling
- calibration probe

## B. small Teacher validation set

只有 A 通过后才生成：

```text
目标 8192 formal states
来自至少 64 个完整 game_id
decision_depth = 3
```

如果 Search 性能 blocker 导致该数据量在合理单机时间内无法完成：
不得偷偷缩成 100 个 state 并宣布 PASS。

应返回：
`M3 PERFORMANCE INCOMPLETE`。

---

# 15. 数据来源固定

优先让现有 tuple Teacher / 3-depth Expectimax 自己推进完整游戏。

不得只从随机盘面生成主数据。

允许 semantic/profile set 加少量：
- hand-crafted states
- high-tile fixtures
- random legal states

但必须标记 `data_source`。

---

# 16. game-level split 固定

生成数据后先按 game_id 切：

```text
train 80%
validation 10%
test 10%
seed = 20260919
```

同一 game_id 绝对不得跨 split。

split 后才允许 D4 augmentation。

M3 原始 artifact 必须保存 canonical、未增强数据；
D4 不覆盖原始 sample。

---

# 17. Teacher value semantics gate

M3 必须把以下三种概念分开：

1. tuple raw evaluator value
2. Expectimax action value with raw tuple leaf
3. true future game score

不得混名。

固定 enum：

```text
RAW_TUPLE_HEURISTIC
FORMAL_STATE_TUPLE_HEURISTIC
SEARCH_VALUE_RAW_LEAF
FUTURE_SCORE
CALIBRATED_FUTURE_SCORE
```

默认前两者不能当 Student Q 的绝对真值。

# 18. future-score calibration probe

必须从 semantic/profile set 中固定抽样：

```text
128 states
```

对每个 state 使用固定 seed 集，完整 rollout 到 terminal，估计从该 state 之后真实获得的 future score。

这里的 **calibration continuation policy 固定为 checkpoint 的原生 policy**：

```text
policy = greedy_1ply
Q_tuple(s,a) = immediate_reward(s,a) + V_tuple(afterstate(s,a))
```

动作顺序 / tie-break 必须复用已通过 upstream anchor differential 的 frozen tuple adapter 语义；不得引入随机 tie-break。允许复用已经 differential PASS 的 M2 C++ movement primitive 与 M3 C++ tuple evaluator 加速 rollout，但不得调用 depth-3 Search 作为 continuation。

也就是说：每个 rollout 的每一步都按 frozen tuple checkpoint 的 greedy_1ply policy 选动作；spawn 仍按真实 90%/10% 与空格均匀随机规则采样。禁止把 continuation policy 偷换成“每一步重新运行 decision_depth=3 Expectimax 到 terminal”。

原因是本 gate 要校准的是 frozen tuple evaluator / formal-state tuple adapter 与真实 future score 的数值关系；checkpoint metadata 已明确其原生训练/推理 policy 为 `greedy_1ply`。Repeated depth-3 Search rollout 属于另一个更昂贵的 policy evaluation 问题，不是本 calibration gate 的必要条件。

至少：

```text
16 stochastic rollouts / state
```

固定定义：

```text
realized_future_score
= 从该 formal state 开始直到 terminal 的后续 merge reward 总和
```

不得把进入该 state 之前已经累计的 `current_score` 重复计入 future score。

每个 state 至少记录：
- raw tuple afterstate value（在有明确对应 afterstate 时）
- formal-state leaf adapter `L_tuple(s)`
- depth-3 search best action value（diagnostic only）
- mean realized future score under frozen greedy_1ply continuation
- std
- Pearson
- Spearman
- affine fit slope/intercept
- validation R²
- MAE / RMSE

必须明确区分两种用途：

1. `V_tuple / L_tuple` vs greedy_1ply realized future score：允许用于本节的 calibration mapping 证据；
2. `depth-3 search best action value` vs上述 realized future score：只作为 diagnostic correlation 记录。因为 rollout continuation 不是 repeated depth-3 Search policy，所以这组相关性不得单独证明 Search value = FUTURE_SCORE，也不得因此把 `SEARCH_VALUE_RAW_LEAF` 自动升级为 `FUTURE_SCORE` / `CALIBRATED_FUTURE_SCORE`。

若后续阶段确实需要 repeated depth-3 Search policy 的 on-policy return，应另立明确预算与专门 gate；不得在 M3 calibration probe 中暗含执行。

所有 calibration fit 必须 train/validation 分开，
不得在同一批 state 上拟合又宣布验证通过。

---

# 19. 绝对 future-score 判定规则

对 `V_tuple / L_tuple`，只有以下两类证据之一成立，才允许标成 FUTURE_SCORE：

A. 用户提供的原始训练/loader/算法代码可直接证明其 target 定义就是“从当前 state 开始、在对应原生 policy 下的真实 future game score”；

或

B. 按 §18 固定 `greedy_1ply` continuation 得到的独立 holdout calibration 明确支持近似 identity，并且 Agent 在报告中给出完整数值证据。

对 `depth-3 search best action value`，§18 的 greedy_1ply rollout **不是 repeated depth-3 Search 的 on-policy return**，因此无论相关性多高，都不得凭 §18 单独把它标成 `FUTURE_SCORE` / `CALIBRATED_FUTURE_SCORE`。M3 默认保持：

```text
SEARCH_VALUE_RAW_LEAF
```

只有另有明确数学证明，或未来专门执行 budgeted repeated depth-3 Search on-policy return gate，才允许重新审议 Search value semantics；本 M3 不要求该 gate。

若 tuple evidence 也不足：

```text
V_tuple / L_tuple value_semantics remains RAW
search value_semantics remains SEARCH_VALUE_RAW_LEAF
```

不要硬判 FUTURE_SCORE。

M3 允许完成而不把任何 raw/search value 校准成 future-score；此时 Student sanity 只能使用 policy/ranking supervision。

---

# 20. calibration mapping

如果 `V_tuple / L_tuple` 与 §18 的 greedy_1ply realized future score 高相关但不是 identity，允许拟合一个显式 calibration mapping。

第一版只允许：

```text
affine:
future_score_hat = a * raw_value + b
```

禁止一上来训练另一个深网络做 calibration。

mapping 必须只用 train split 拟合，在 validation/test split 独立报告。

只有 validation/test 稳定后，且 mapping 的目标 policy 明确记录为 `greedy_1ply`，对应 tuple/formal-state value 才允许标：

```text
CALIBRATED_FUTURE_SCORE
```

该 affine mapping **不得自动套到 depth-3 Search action values**；Search action value 默认继续标 `SEARCH_VALUE_RAW_LEAF`。

---

# 21. Student learning sanity

M3 只验证“Student 可以稳定学习”，
不做 M4 模型胜负比较。

固定使用现有：

```text
ResidualMLP2048
```

作为 M3 sanity student。

理由只是 throughput 快；
不得据此宣布 MLP 比 Transformer 强。

---

# 22. Student supervision 分支

## 如果 value semantics 仍未校准

只允许：
- teacher best-action cross entropy / classification
- legal-action ranking metric

禁止：
- raw teacher value -> Q MSE
- raw teacher value -> V MSE
- raw teacher value -> A MSE

## 如果某组 teacher action target 自身已证明 FUTURE_SCORE / CALIBRATED_FUTURE_SCORE

才允许对那一组合法动作做 Q regression。tuple/formal-state calibration 不自动赋予 depth-3 Search action values 相同语义；若 Search action values 仍是 `SEARCH_VALUE_RAW_LEAF`，不得对它们做 absolute Q regression。

V/A 的绝对 target 只有在数学语义可由 action value / reward 正确推导时才使用。

不得为了“让三个 head 都有 loss”而伪造 target。

---

# 23. Student sanity 固定验收

至少记录：
- train loss
- validation loss
- teacher-best-action accuracy
- legal action ranking accuracy
- D4 consistency diagnostic

如果是 absolute regression：
再记录：
- Q MAE/RMSE on legal actions
- value scale statistics

PASS 要求：
- train metric 明显改善；
- validation 不发散；
- teacher-best-action accuracy 明显高于合法动作随机基线；
- fixed-seed rerun 可复现同方向结果。

不得用 test split 调参。

---

# 24. Search profiling 必做

创建独立 profiler / benchmark。

至少拆分：
- move generation
- legal-action generation
- chance expansion
- board transform
- tuple evaluator lookup
- transposition lookup
- hash
- recursive search overhead
- Python object allocation / orchestration
- NN leaf batching（若未使用则明确 N/A）

固定测试：
- depth=3 formal run
- 至少 256 states
- warmup 后计时
- total states/s
- root decisions/s
- nodes/s
- node counts
- cache hit rate（若有 transposition table）

---

# 25. transposition table 规则

M3 baseline 允许建立最小 transposition table，
但 key/value 语义必须明确包含足够信息：
- board
- remaining decision depth
- node type / evaluator mode（如需要）

不得因为错误 cache key 复用不同深度的值。

必须有 cache-on / cache-off correctness differential。

---

# 26. 性能 blocker 判定

如果 profiling 证明：
- Search 占 M3 wall-clock 主体；
且
- Python recursion / object overhead / move generation / evaluator lookup
存在明显数量级损失；

则 M3 不得假装“先凑合”。

本施工单第一轮只允许：
- correctness baseline
- profiling
- 小范围 low-risk batching / allocation cleanup

不得直接大规模重写 C++ Search。

若仍严重：

```text
M3 PERFORMANCE INCOMPLETE
STOP
```

把热点证据交回外部审计，
再发专门 M3 Search Performance Unblock 施工单。

---

# 27. 禁止提前进入 M4/M5

M3 不允许：
- 正式 Transformer vs MLP 比赛
- 大规模 Teacher 预训练
- 大规模 self-play
- Replay
- Double-Q
- Target Network
- Champion
- Search Correction
- M4+
- 改网络结构
- 改 frozen M2 backend

# 28. 建议文件边界

M3 Agent 允许新增/修改：

```text
.gitignore

teacher_checkpoints/README.md
teacher_checkpoints/manifest.json

src/game2048/m3_tuple_teacher.py
src/game2048/m3_search.py
src/game2048/m3_teacher_data.py

tests/test_m3_tuple_teacher.py
tests/test_m3_search.py
tests/test_m3_teacher_data.py

benchmarks/benchmark_m3_teacher_search.py
benchmarks/validate_m3_teacher_semantics.py
benchmarks/validate_m3_student_sanity.py

reports/m3/M3_REPORT.md
reports/m3/m3_teacher_semantics.json
reports/m3/m3_teacher_profile.json
reports/m3/m3_student_sanity.json
```

本地还必须创建但绝对不得 Git 跟踪：

```text
teacher_checkpoints/m3/ordinary_td_comparator_ep4800000_7192719323a0.bin
```

实际 checkpoint 格式若要求一个极小 adapter 文件，
可以新增，但必须在报告解释。

不要创建大型 framework。

---

# 29. CI

所有不需要真实 checkpoint 大文件的 unit tests 必须进入 GitHub Actions。

CI 不得依赖用户本地 checkpoint。

checkpoint-dependent tests：
- 使用 tiny deterministic synthetic fixture；
- 或从真实 loader 逻辑构造最小测试 checkpoint。

不得 skip/xfail 来逃避。

最终 full pytest：
- 0 failed
- 0 skipped
- 0 xfailed

---

# 30. checkpoint 本地保存但禁止 commit

本阶段规则不是“checkpoint 不得复制进 repo”。

正确规则是：

```text
必须复制进本项目工作目录
+
必须被 .gitignore 排除
+
绝对不得进入 Git object history
```

正式 local path：

```text
teacher_checkpoints/m3/ordinary_td_comparator_ep4800000_7192719323a0.bin
```

禁止：

- `git add -f` checkpoint；
- commit checkpoint；
- Git LFS 上传 checkpoint；
- GitHub Release 上传 checkpoint；
- 把 checkpoint 转成源码数组后 commit；
- 把 512 MiB binary 塞进任何 tracked archive；
- 覆盖 M3 frozen baseline；
- 把 `latest.bin` 复制到这个文件名冒充固定 baseline。

允许提交：

```text
teacher_checkpoints/README.md
teacher_checkpoints/manifest.json
.gitignore
```

source code 不得硬编码用户上游绝对路径。

正式 runtime 应从 repo root 解析：

```text
teacher_checkpoints/m3/ordinary_td_comparator_ep4800000_7192719323a0.bin
```

CLI 仍允许显式：

```text
--checkpoint <path>
```

但 M3 正式运行必须记录实际 resolved path 与 SHA256。

CI 不得要求真实 512 MiB checkpoint。

---

# 31. Git 保护

绝对不得移动：

```text
m0-reference-pass
m1-fastenv-audited-pass
m2-network-audited-pass
```

不得修改 M0/M1/M2 frozen source，
除非发现真实 correctness bug；
若发现则 STOP，走独立修复流程。

用户自己的 prompt 文件仍不得 stage/commit/restore/delete/rename。

---

# 32. M3 RESULT 判定

只有以下全部成立才允许：

```text
M3 CANDIDATE COMPLETE
```

要求：
- checkpoint format / loader 无歧义；
- checkpoint SHA256 已记录；
- tuple evaluator deterministic；
- Search correctness PASS；
- Teacher action schema 完整；
- raw/search/future-score 语义明确区分；
- calibration probe 完成；
- 未校准 raw value 没被错误用于 absolute Q/V/A regression；
- 至少 8192-state / 64-game 小规模 Teacher dataset 完成，或总计划允许的等价正式小规模数据已明确记录；
- game-level split 正确；
- Student sanity 稳定学习；
- Search profile 完整；
- 没有已知数量级 Python 性能损失被当成正常成本接受；
- full pytest green；
- remote GitHub Actions green；
- frozen tags unchanged。

如果 Search 有严重性能 blocker：

```text
M3 PERFORMANCE INCOMPLETE
```

如果 checkpoint 无法确定格式/语义：

```text
BLOCKED_CHECKPOINT_FORMAT
```

如果 checkpoint path 缺失：

```text
BLOCKED_CHECKPOINT_PATH
```

---

# 33. 最终报告固定格式

1. RESULT
2. STARTING STATE
3. CHECKPOINT
   - path
   - size
   - SHA256
   - format
   - loader source
4. TUPLE EVALUATOR
   - board encoding
   - raw output stats
   - deterministic checks
5. SEARCH SEMANTICS
   - depth
   - leaf
   - backup
6. SEARCH CORRECTNESS
7. TEACHER SAMPLE SCHEMA
8. DATASET
   - games
   - states
   - split by game_id
9. VALUE SEMANTICS
   - raw
   - search
   - future score
10. CALIBRATION
    - Pearson/Spearman
    - slope/intercept
    - R²
    - MAE/RMSE
    - final classification
11. STUDENT SANITY
12. SEARCH PROFILE
13. PERFORMANCE GATE
14. PYTEST
15. CI
16. FROZEN VERIFICATION
17. FILES CHANGED
18. ARTIFACTS
19. FINAL GIT STATE
20. REMAINING BLOCKERS

---

# 34. STOP 行为

完成 M3 candidate/report 后：

STOP。

不要：
- 标记 M3 AUDITED PASS；
- 创建 M3 tag；
- 更新主计划为 M3 FROZEN；
- 进入 M4；
- 开始 M5 Teacher 预训练。

交回外部独立审计。

---

# 35. 开工第一句话

先复述固定 provenance 与目标 local copy：

```text
upstream source snapshot:
D:\CodexTasks\2048-ai\oneclick-m6-report\snapshots\COMPARATOR\ep4800000_7192719323a0.bin

expected SHA256:
7192719323a073ba2b6b19b62cb7d46ef4aa90ecc8c4ae6baf27ad0c51566a84

target local checkpoint:
D:\CodexTasks\NN-2048-ai\teacher_checkpoints\m3\ordinary_td_comparator_ep4800000_7192719323a0.bin
```

同时复述：

- current HEAD / origin-main
- M0/M1/M2 authoritative tags
- 本轮只做 M3 Teacher 小规模验证
- M3 不追 `latest.bin`
- 二进制必须复制进项目，但必须 Git ignored

然后严格执行：

```text
A. repository / frozen-tag verification
B. upstream source snapshot existence check
C. upstream SHA256 verification
D. teacher_checkpoints/ structure creation
E. README.md / manifest.json / .gitignore preparation
F. byte-for-byte local copy
G. local SHA256 + size re-verification
H. git check-ignore verification
I. local-copy-only checkpoint format gate
J. 后续 M3 correctness / semantics / search / dataset / student sanity / profiling
```

若 upstream source 缺失：

```text
BLOCKED_CHECKPOINT_PATH
STOP
```

若 upstream SHA 不匹配：

```text
BLOCKED_CHECKPOINT_MISMATCH
STOP
```

若已存在 local copy 但 SHA 不匹配：

```text
BLOCKED_LOCAL_CHECKPOINT_MISMATCH
STOP
```

禁止 fallback 到：

```text
latest.bin
FORMAL checkpoint
其他 episode
其他 tuple model
```

不得在这三个 blocker 情况下继续。
