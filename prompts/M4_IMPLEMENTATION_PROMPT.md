# M4 Transformer vs MLP — 严格施工提示词

> 项目：Safolour/NN-2048-ai
> 正式工作区：D:\CodexTasks\NN-2048-ai
> 当前阶段：M4 Transformer vs MLP
> 本文件是 M4 的唯一详细施工单；authoritative master plan 始终优先。

# 0. 权威输入

开工前完整读取：
1. `prompts/2048_AI_正式执行计划_M0_M1冻结_CI增强版_2026-09-18.md`
2. `M3_AUDIT_REPORT.md`
3. `M3_REPORT.md`
4. 本文件

如有真实冲突：STOP，报告冲突，不得自行裁决。

# 1. Frozen provenance

必须验证 annotated tags（必须用 `^{}` 解引用）：
- `m0-reference-pass^{}` -> `3f2def1d95f56eff776e671143188947bf64485b`
- `m1-fastenv-audited-pass^{}` -> `e5486017a90eeec4fb9814de7880b3c413dc06dd`
- `m2-network-audited-pass^{}` -> `6a9da5b1bf7c72207acd6e89bc667d439d4823a8`
- `m3-teacher-audited-pass^{}` -> `7c95f9c5f543065b220fb5f9e9114521732cdc0b`

M3 audit docs closeout parent：
`2ed31b084d8e73f31ea12d69b6dc23e293c7e024`

M4 开工 HEAD 可以是上述 commit 的 planning/docs-only descendant。
不得修改 M0/M1/M2/M3 frozen implementation 或移动 frozen tags。

# 2. 当前工作区已知非 M4 变化

以下是用户主动清理/本地状态，不是 regression：
- 旧 M1/M2 benchmark/sanity JSON 的 tracked deletion；
- `.vs/`；
- 本地 `artifacts/`。

禁止自动 restore；禁止 `git add -A` / `git add .`。
M4 commit 必须显式 file allowlist，不得吸收这些变化。

# 3. M4 唯一目标

只回答一个问题：

**在约 5M 参数、相同 frozen M3 Teacher 数据、相同训练协议和相同硬件预算下，Transformer2048 与 ResidualMLP2048 哪个更适合作为后续 M5+ 的主网络架构？**

M4 比较：
- Teacher imitation；
- 实际 greedy 游戏平均分；
- 推理速度；
- 训练速度；
- 参数量；
- GPU/VRAM/CPU 计算成本；
- D4 consistency diagnostic。

M4 不做：
- M5 大规模 Teacher 数据生成/预训练；
- Teacher checkpoint promotion；
- self-play；
- Replay / Double-Q / Target；
- Search Correction；
- 网络规模 2M/10M/15M sweep；
- 修改 Transformer/MLP architecture；
- 修改 M3 Teacher/Search。

# 4. 固定模型

必须直接复用 M2 frozen architecture：
- `Transformer2048()`
- `ResidualMLP2048()`

当前权威参数量：
- Transformer2048 = **4,750,342**
- ResidualMLP2048 = **5,264,710**

两者差约 10.8%，视为当前约 5M parameter-matched baseline。
禁止为了“更公平”临时改 hidden/blocks/heads/embedding。

Transformer 固定：
- tile vocab 22
- hidden 256
- 6 blocks
- 8 heads
- FFN 1024
- 16 learned absolute position embeddings
- dropout 0

ResidualMLP 固定：
- tile embedding 32
- hidden 640
- 6 residual blocks

# 5. 固定 Teacher dataset

M4 只使用 M3 audited canonical dataset：

`artifacts/m3/m3_teacher_validation_8192.npz`

开工必须验证：
- states = 8192
- games = 64
- train = 51 games / 6528 states
- validation = 6 / 768
- test = 7 / 896
- game overlap = 0
- decision_depth unique = 3
- checkpoint SHA = `7192719323a073ba2b6b19b62cb7d46ef4aa90ecc8c4ae6baf27ad0c51566a84`
- value_semantics = `SEARCH_VALUE_RAW_LEAF`
- canonical_unaugmented = true

artifact 存在时禁止重新生成。
如果缺失：STOP，报告 `BLOCKED_M3_DATASET_MISSING`；不得在 M4 偷跑 M3 数据生成。

# 6. Supervision 锁死

M3 已证明 Search values 不能当 absolute future-score target。

因此 M4 两个架构都固定：
- loss = teacher-best-action 4-class cross entropy
- target = `nanargmax(teacher_value)`
- absolute Q MSE = 禁止
- V MSE = 禁止
- Afterstate MSE = 禁止
- pairwise ranking = evaluation metric，不额外加入 loss

不得给某个架构单独增加 auxiliary loss。

# 7. 数据增强与 batch plan 公平性

train split 才允许 random D4 augmentation。
validation/test 永远 canonical。

M4 必须为每个 training seed 预生成同一套 deterministic training plan：
- 每个 optimizer step 的 sample indices；
- 每个 sample 的 D4 transform id；
- epoch/shuffle order。

同一 seed 下 Transformer 与 MLP 必须复用完全相同的 plan。
每个 seed 的 plan 必须保存到 `artifacts/m4/`，记录 sample-index / transform-id arrays 的 SHA256；两架构报告中的 plan SHA 必须相同。
不得让两个架构各自随机 shuffle / augmentation 后声称公平。

# 8. 固定训练 seeds

固定三个独立 training seeds：

`20260919, 20260920, 20260921`

每个 seed 两个架构都各跑一次，共 6 个 primary runs。

# 9. Primary equal-update training budget

固定：
- epochs = 30
- batch_size = 1024
- optimizer = AdamW
- lr = 3e-4
- weight_decay = 1e-4
- gradient clip global norm = 1.0
- scheduler = none
- early stopping = 禁止

每个架构每个 seed 完全相同 optimizer update 数、看到相同样本与 D4 transforms。

AMP：
1. 先对两个架构共同做 BF16 finite smoke test；
2. 若两者都稳定，则 primary 全部使用 BF16；
3. 若任一架构 BF16 非 finite / correctness 不稳定，则两个架构都统一回退 FP32。
禁止一个 BF16、另一个 FP32。

记录实际 precision。

# 10. 每个 primary run 必须记录

训练前：
- initial train CE
- initial validation CE
- initial legal best-action accuracy
- initial legal pairwise ranking

每 epoch：
- train CE
- validation CE
- validation legal best-action accuracy
- validation legal pairwise ranking
- wall seconds
- optimizer steps
- samples/s
- peak VRAM
- GPU utilization diagnostic（能可靠取得则记录；取不到明确 N/A）

训练结束：
- final train metrics
- final validation metrics
- D4 consistency diagnostic
- total training wall
- mean/median step time
- Q head/backbone parameter updates confirmed
- V head / Afterstate head parameters与初始化快照 bit-identical（因为本阶段禁止给它们 fabricated loss）

test split 只有全部 6 个 primary run 完成、协议不得再修改后才读取。

# 11. Test Teacher metrics

对每个 final model 记录：
- test CE
- legal best-action accuracy
- legal pairwise ranking accuracy
- raw best-action accuracy
- D4 legal-argmax consistency
- centered-logit D4 MAE

test 不得用于改 LR/epoch/batch/architecture。

# 12. 实际游戏评测固定

每个 final model 必须进行 pure-NN greedy closed-loop evaluation。

禁止：
- Teacher
- tuple evaluator
- Search
- exploration
- beam/rollout
- test-time D4 ensemble

决策：
- 单次 NN Q/logit forward
- mask illegal actions
- 选择最高合法 logit
- exact ties / 差值 <= 1e-7 的合法动作使用独立 tie-break RNG
- tie-break RNG 与 environment spawn RNG 分离。

game evaluator 必须保证**每一局拥有独立、由该 game seed 决定的 spawn RNG stream**；不得用一个共享 batch RNG 导致某架构因提前终局而改变其他游戏后续随机数分配。
允许复用 M2 C++ movement primitive 加速，但 movement/reward/legal/terminal 语义必须保持 M0 等价。
正式 2000-game 前必须用至少 32 个固定 game seeds 做 evaluator determinism / scalar-reference spot check；同 model + 同 seed 重跑 final score/moves/max tile 必须完全一致。

固定 paired game seeds：

`20261001 + i, i = 0..1999`

即每个 trained model **2000 games**。
两个架构、三个 training seeds 都使用完全相同 2000 game seeds。

每局记录：
- final score
- max tile exponent
- moves

每个 model 汇总：
- mean score（主指标）
- median
- p10 / p90
- mean moves
- reach 2048/4096/8192/16384/32768/65536
- max-tile distribution

# 13. 棋力统计规则

对每个 training seed，计算 paired：
`Transformer score - MLP score`

用固定 bootstrap seed `20261001`、10,000 paired bootstrap resamples，报告 mean delta 95% CI。

再做 aggregate：
对同一个 evaluation seed，先取三个 Transformer training seeds 的 mean score；
MLP 同理；
再对 2000 个 evaluation seeds 做 paired bootstrap 95% CI。

只有满足：
1. aggregate paired 95% CI 不包含 0；
2. 三个 per-training-seed delta 至少 2/3 与 aggregate 同方向；

才允许宣布某架构在 M4 的**实际棋力显著更高**。

否则固定结论：
`ARCHITECTURE_STRENGTH_NOT_STATISTICALLY_RESOLVED`。

# 14. 无法确认棋力差异时的固定 tie-break

严格按总计划“更简单、更快”处理。

先比较 **8192-env closed-loop decisions/s**：
- 使用 frozen M2 `scalar+row-lut` rollout backend；
- 同一机器、同一环境配置；
- warmup 后至少 3 个正式重复；
- 取 median。

若 median closed-loop throughput 差异 >=10%：
选择更快者。

若差异 <10%：
选择 trainable parameter count 更少者。

该 tie-break 只在实际棋力未统计确认时使用。

# 15. 性能 benchmark

两个 final architecture 至少测。用于 §14 selection tie-break 的性能结果固定使用 **PyTorch eager + 两架构共同 precision**；不得一个使用 `torch.compile`、另一个 eager。可额外报告“两者都成功 compile”时的 secondary 数据，但不改变 §14 tie-break。

Model-only inference batch：
- 1
- 256
- 1024
- 4096
- 8192

记录：
- states/s
- batch latency
- peak VRAM

Closed-loop：
- 2048 envs
- 8192 envs
- 若两者均安全再测 16384
- 32768 只有两者均满足 RAM/VRAM safety 才测，不得为凑表 OOM

记录：
- decisions/s
- GPU utilization
- CPU usage
- VRAM
- wall
- pipeline stall diagnostic

Training：
- samples/s
- step time
- peak VRAM。

# 16. Equal-wall-clock secondary

总计划要求重要比较最好同时提供 same wall-clock/GPU-hour。

Primary 6 runs 完成后计算两架构 primary training wall 的 median。

若较慢 / 较快 median wall ratio <=1.20：
记录“compute cost sufficiently close”，不再额外 rerun。

若 ratio >1.20：
必须执行 equal-wall-clock secondary：
- common budget = 两架构 primary median wall 中较大的那个；
- 两架构均从随机初始化重新开始；
- 使用 training seed 20260919；
- 相同 optimizer/lr/batch/precision/data-plan循环；
- 不 early-stop；
- 运行到不超过 common budget 的最后一个完整 optimizer step；
- 记录完成 steps、samples、Teacher metrics；
- 再用同一 2000 paired game seeds 做 closed-loop score。

secondary 只作为 compute-normalized evidence，不覆盖 primary selection rule，除非 primary strength unresolved；若 primary unresolved，可作为 tie-break 的附加解释，但最终仍按 §14 的固定 speed/parameter tie-break 选架构。

# 17. 禁止不公平调参

M4 禁止 architecture-specific：
- LR
- weight decay
- batch
- epoch
- scheduler
- loss
- augmentation
- precision
- label smoothing
- dropout change
- gradient clip
- early stopping
- hand-picked checkpoint epoch

如果共同协议导致**两个架构都**明显失败，可 STOP 并报告 protocol blocker。
如果只有一个架构失败，不得单独给它调参救场；失败本身属于 baseline comparison 结果。

# 18. Checkpoint / artifact 边界

正式本地 artifact：
`artifacts/m4/`

保存：
- per-run checkpoints
- training histories
- raw 2000-game score arrays
- performance raw measurements

这些默认不 Git track。

Git 可提交：
- M4 harness/scripts
- CPU-testable tests
- compact JSON summary
- `M4_REPORT.md`

不得提交 GPU model binary / 大型 raw score dump。

# 19. 建议新增文件

允许：
- `benchmarks/run_m4_architecture_compare.py`
- `benchmarks/benchmark_m4_architecture_performance.py`
- `tests/test_m4_architecture_compare.py`
- `m4_architecture_compare.json`
- `M4_REPORT.md`

如需一个极小公共 helper，可新增 `src/game2048/m4_compare.py`。
禁止大 framework，禁止改 M2 model architecture。

# 20. Tests / correctness

至少新增 CPU tests：
- both models receive same batch plan
- D4 action transform correctness reused
- illegal mask in gameplay
- exact tie-break RNG independent from spawn RNG
- paired bootstrap deterministic
- game score aggregation
- no test split used during training protocol
- policy-only loss does not touch V/A heads

existing M0-M3 tests不得删除/skip/xfail。

# 21. 开工 P0

第一动作报告：
- HEAD / origin/main
- working tree
- four frozen tag resolutions
- M3 dataset path/shape/splits/SHA metadata
- GPU / torch / CUDA
- exact model parameter counts
- pre-existing user cleanup deletions

然后用官方 Python：
`D:\sd-webui-forge-aki-v1.0\python\python.exe -m pytest -q`

要求至少：
- 525 passed
- 0 failed
- 0 skipped
- 0 xfailed

失败则 STOP，不得用 M4 修改 frozen source 来救。

# 22. 长任务可观测性

所有 6 个训练 run 至少每 epoch 输出：architecture / seed / epoch / train CE / validation CE / val accuracy / elapsed / samples/s。
2000-game evaluation 至少每完成 100 games 输出：architecture / training seed / completed / elapsed / games/s / running mean score。
性能 benchmark 每个 batch/env size 完成后立即持久化 progress artifact。
进程异常中断后必须优先从已完成 run/game shard 恢复，禁止无条件从 P0 重跑。

# 23. 执行顺序锁死

P0 frozen/preflight/full pytest
→ P1 implement deterministic paired training/eval harness + CPU tests
→ P2 BF16 shared smoke / precision decision
→ P3 6 primary equal-update training runs
→ P4 teacher-metric final/test evaluation
→ P5 6 × 2000 paired greedy game evaluation
→ P6 paired statistical analysis
→ P7 inference/training/closed-loop performance benchmark
→ P8 conditional equal-wall-clock secondary only if §16 triggers
→ P9 deterministic selection rule
→ P10 final full pytest
→ candidate implementation commit/push
→ candidate GitHub Actions green
→ 如需把 CI run ID/conclusion 回写 M4_REPORT，只允许一个 docs/report-only closeout descendant
→ 若创建该 closeout descendant，再等它自己的 GitHub Actions green
→ STOP for independent M4 audit

不得跳步。

# 24. M4 selection result enum

只允许最终 architecture decision：
- `M4_SELECT_TRANSFORMER`
- `M4_SELECT_RESIDUAL_MLP`
- `M4_BLOCKED`

同时必须单独记录 strength conclusion：
- `TRANSFORMER_STRENGTH_SIGNIFICANT`
- `MLP_STRENGTH_SIGNIFICANT`
- `ARCHITECTURE_STRENGTH_NOT_STATISTICALLY_RESOLVED`

如果 strength unresolved，selection 必须按 §14 tie-break 得出，不能凭主观判断。

# 25. Candidate complete gate

只有全部成立才允许：
`M4 CANDIDATE COMPLETE`

要求：
- M0/M1/M2/M3 frozen tags unchanged
- frozen M2 architectures unmodified
- same M3 dataset / split
- same primary training budget
- 3 paired training seeds
- 2000 paired game seeds/model
- Teacher metrics complete
- actual score metrics complete
- paired CIs complete
- inference/training/closed-loop performance complete
- selection rule mechanically satisfied
- no architecture-specific tuning
- full pytest green
- remote CI green
- artifacts organized / temp files cleaned

# 26. STOP / Git

M4 candidate 必须创建普通 implementation candidate commit并 push。该 commit 是 M4 candidate implementation 的权威候选 SHA。

candidate CI green 后，如 `M4_REPORT.md` 需要记录实际 CI run ID/conclusion，允许再创建**至多一个** docs/report-only closeout descendant；该 descendant 不得修改 src/cpp/tests/benchmarks/result JSON/model selection，且不得冒充 implementation candidate commit。若创建，必须再次等待它自己的 CI green。

禁止：
- 创建 M4 audited tag
- 标记 M4 FROZEN
- 进入 M5
- 执行 Teacher Checkpoint Promotion Gate

candidate 完成后 STOP，交给独立审计。

commit 必须显式 allowlist。
禁止 `git add -A` / `git add .`。
用户旧 JSON deletion、`.vs/`、M3/M4 local artifacts 不得混入 commit。

# 27. CI

GitHub Actions 必须继续：
- build M2 C++ backend
- build M3 C++ tuple backend
- full CPU-testable pytest

M4 unit tests不得依赖：
- RTX GPU
- 512 MiB Teacher checkpoint
- 本地 8192-state artifact

需要数据时使用 tiny deterministic synthetic fixture。
M4 大型训练/游戏评测不进 CI。

# 28. 最终报告固定格式

1. RESULT
2. STARTING STATE
3. FROZEN VERIFICATION
4. DATASET VERIFICATION
5. MODEL DEFINITIONS / PARAM COUNTS
6. FAIRNESS CONTRACT
7. PRECISION DECISION
8. PRIMARY TRAINING RESULTS
9. VALIDATION / TEST TEACHER METRICS
10. GREEDY GAME SCORE RESULTS
11. PAIRED STATISTICS
12. D4 DIAGNOSTIC
13. INFERENCE PERFORMANCE
14. TRAINING PERFORMANCE
15. CLOSED-LOOP PERFORMANCE
16. EQUAL-WALL-CLOCK SECONDARY
17. ARCHITECTURE SELECTION
18. PYTEST
19. CI
20. FILES CHANGED
21. ARTIFACTS / CLEANUP
22. FINAL GIT STATE
23. REMAINING BLOCKERS

# 29. 禁止越界

M4 的 selected architecture 只是给 M5+ 使用的 architecture baseline。
不得把 M4 的 8192-state policy-only imitation model 宣布为最终 Champion。
不得因为某个模型在 M4 胜出就删除另一个 frozen M2 architecture 实现。
不得开始 M5。

# 30. 最终行为

完成 candidate 后输出：
`M4 CANDIDATE COMPLETE`

并明确：
- selected architecture
- strength conclusion
- selection依据（score significance 或 speed/size tie-break）
- candidate implementation commit
- report-only closeout commit（若有；否则 N/A）
- candidate CI run / closeout CI run（如适用）
- no M4 tag
- no M5

然后 STOP。