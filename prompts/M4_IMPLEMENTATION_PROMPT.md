# M4 Transformer vs MLP — 严格施工提示词

> **状态：HISTORICAL / FINAL AUDITED PASS / FROZEN。M4 已完成独立审计；本施工单仅用于复现与审计，不得作为当前阶段重新执行。当前阶段见 authoritative master plan（M5）。**
>
> authoritative tag：`m4-architecture-audited-pass` -> `300d51a818fa55394ec56de7507bcade111a06d1`
> selected architecture：`ResidualMLP2048`
>
> 项目：Safolour/NN-2048-ai
> 正式工作区：D:\CodexTasks\NN-2048-ai
> 历史阶段：M4 Transformer vs MLP
> 本文件是 M4 的冻结施工单；authoritative master plan 始终优先。
> Work-order version: `M4_WO_CLOSURE_V2`。

# 0. 权威输入

开工前完整读取：
1. `prompts/2048_AI_正式执行计划_M0_M1冻结_CI增强版_2026-09-18.md`
2. `reports/m3/M3_AUDIT_REPORT.md`
3. `reports/m3/M3_REPORT.md`
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

每次进入 `run M4`，先固定执行：

```text
git fetch origin
git rev-parse HEAD
git rev-parse origin/main
git status --porcelain=v1 -uall
git diff --cached --quiet
```

并计算：
- 当前 `prompts/M4_IMPLEMENTATION_PROMPT.md` 文件 SHA-256，记为 `prompt_sha256`；
- `artifacts/m4/progress/session.json` 是否存在。

启动模式只允许两种：

### FRESH

仅当 `session.json` 不存在时进入 FRESH。硬门：
- `HEAD == origin/main`；
- `git status --porcelain=v1 -uall` 为空；
- staged diff 为空；
- `git merge-base --is-ancestor "m3-teacher-audited-pass^{}" HEAD` exit 0；
- 根目录不存在任何 `M*_REPORT.md` 或 `m*_*.json` milestone 产物；
- `git diff --quiet "m2-network-audited-pass^{}" -- src/game2048/m2_models.py src/game2048/m2_symmetry.py src/game2048/m2_policy.py src/game2048/m2_rollout_env.py src/game2048/m2_fast_backend.py cpp/m2_fast_backend` exit 0。

任一不满足：STOP，输出 `M4_BLOCKED_START_STATE`；不得自行 pull/rebase/reset/restore/delete。

FRESH P0 全部通过后，原子创建 `artifacts/m4/progress/session.json`，固定至少包含：
`schema_version=1, work_order_version="M4_WO_CLOSURE_V2", base_head, prompt_sha256, dataset_sha256, state="P0_PASSED", p0_pytest`。

### RESUME

`session.json` 存在时必须进入 RESUME，禁止按 FRESH 重新开始。先验证：
- `schema_version == 1`；
- `work_order_version == "M4_WO_CLOSURE_V2"`；
- session 中 `prompt_sha256` 等于当前 prompt SHA；
- session 中 `dataset_sha256` 等于 §5 固定 SHA；
- four frozen tags 仍精确匹配；
- frozen M2 source diff gate 仍 exit 0。

任一不满足：STOP，输出 `M4_BLOCKED_RESUME_METADATA_MISMATCH`，不得删除 session 后假装 FRESH。

RESUME 时**先执行 §22 的 Git/session 窄窗口恢复，再应用下列 Git 门**。session `state` 只允许：
- pre-candidate set = `{P0_PASSED,P1_READY,PRIMARY_RUNNING,PRIMARY_DONE,PERFORMANCE_RUNNING,PERFORMANCE_DONE,FINALIZE_RUNNING,FINALIZED,P10_RUNNING,P10_DONE,REPORT_WRITTEN}`：必须 `HEAD == origin/main == base_head`；staged diff 必须为空；dirty/untracked 路径只能是 §19 六个 tracked 路径的子集，除此之外任一 dirty path → STOP `M4_BLOCKED_RESUME_DIRTY_SCOPE`；不重跑 P0 full pytest；
- `CANDIDATE_PUSHED`：必须 `HEAD == origin/main == candidate_sha` 且 worktree clean；只恢复 candidate CI 阶段；
- `CLOSEOUT_REPORT_READY`：必须 `HEAD == origin/main == candidate_sha`；staged diff 必须为空；dirty path 只能为空或精确等于 `reports/m4/M4_REPORT.md`；只恢复 mandatory closeout report/commit；
- `CLOSEOUT_PUSHED`：必须 `HEAD == origin/main == closeout_sha` 且 worktree clean；只恢复 closeout CI 阶段；
- `COMPLETE`：不得重跑任何实验，直接报告已有 complete evidence 并 STOP。

不得修改 M0/M1/M2/M3 frozen implementation 或移动 frozen tags。

# 2. 工作区边界

当前 main 已完成 post-M3 cleanup；FRESH 必须从 clean working tree 开始，RESUME 只能按 §1 的 session 状态机继续。

允许本地存在被 `.gitignore` 忽略的 `artifacts/`、checkpoint、build/cache；它们不属于 dirty worktree。
禁止 `git add -A` / `git add .`。
所有 M4 commit 必须使用显式 file allowlist。

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
- 网络规模 2M/10M/15M sweep；**这只是 M4 控制变量限制，不得解释为取消 master plan 后续约 2M/5M/10M/15M scale experiment 或其“有收益则继续扩大、无收益才停止”的强度上限机制**；
- 修改 Transformer/MLP architecture；
- 修改 M3 Teacher/Search。

# 4. 固定模型

必须直接复用 M2 frozen architecture。M4 的 canonical `architecture_id` 只允许两个精确字符串：`Transformer2048` 与 `ResidualMLP2048`；本文所有 `<architecture>` 占位符、artifact 文件名与 summary JSON model key 都必须使用这两个字符串。
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

开工先计算文件 SHA-256，必须精确等于：

`F0E5D806A3E17A27284998AAD52A683245A3C74EBDD15B23EFC13448B9D96582`

不一致：STOP，输出 `M4_BLOCKED_M3_DATASET_SHA_MISMATCH`。

然后验证：
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

上述 metadata/schema 任一不满足：STOP，输出 `M4_BLOCKED_M3_DATASET_METADATA`。

artifact 存在时禁止重新生成。
如果缺失：STOP，报告 `M4_BLOCKED_M3_DATASET_MISSING`；不得在 M4 偷跑 M3 数据生成。

# 6. Supervision 锁死

M3 已证明 Search values 不能当 absolute future-score target。M3 calibration validation 仅 1 game / 8 states，affine mapping validation/test 不稳定；M4 不得复用该 affine mapping 或其 slope/intercept 作为 absolute target。

因此 M4 两个架构都固定：
- loss = teacher-best-action 4-class cross entropy
- target = `nanargmax(teacher_value)`
- absolute Q MSE = 禁止
- V MSE = 禁止
- Afterstate MSE = 禁止
- pairwise ranking = evaluation metric，不额外加入 loss

不得给某个架构单独增加 auxiliary loss。

# 7. 数据增强与 batch plan 公平性

train split **必须**按本节 deterministic plan 使用 D4 augmentation；不得关闭。validation/test 永远 canonical，禁止 D4 augmentation。

M4 必须为每个 training seed 预生成同一套 deterministic training plan，算法固定如下：

1. 取 train split 的 6528 个 canonical row index，按升序作为 `train_indices`；
2. 对 seed `S` 创建独立 `numpy.random.Generator(numpy.random.PCG64(S))`，该 RNG 只负责 data plan；
3. 对 epoch `0..29`：先调用一次 `rng.permutation(train_indices)` 得到该 epoch 完整 sample order；再调用一次 `rng.integers(0, 8, size=6528, dtype=np.uint8)` 得到与该 order 一一对应的 D4 transform id；
4. `drop_last = False`，按 order 连续切 batch：前 6 个 batch 各 1024，最后 1 个 batch 384；因此固定 **7 optimizer steps/epoch，30 epochs = 210 steps/run**；
5. board 使用 `src/game2048/m2_symmetry.py::transform_board_batch`；teacher-best-action label 使用 `transform_action_batch` 做同一 transform；
6. validation/test 禁止 D4 augmentation；
7. 每个 seed 固定保存为 `artifacts/m4/plans/plan_<seed>.npz`，其中 key 只能为 `sample_indices` 与 `transform_ids`；
8. plan SHA 算法固定：分别对 `np.ascontiguousarray(sample_indices).tobytes()` 与 `np.ascontiguousarray(transform_ids).tobytes()` 做 SHA-256，再对 ASCII 字符串 `sample_sha + ":" + transform_sha` 做一次 SHA-256，最终结果记为 `plan_sha256`。

同一 seed 下 Transformer 与 MLP 必须读取同一个 plan 文件，禁止各自重新生成。训练不得使用 DataLoader shuffle/worker RNG；直接按 plan 中的 row indices 对常驻 CPU tensor 索引形成 batch，再转 GPU。报告中两架构对应的 `plan_sha256` 必须完全相同。

# 8. 固定训练 seeds

固定三个独立 training seeds：

`20260919, 20260920, 20260921`

每个 seed 两个架构都各跑一次，共 6 个 primary runs。

# 9. Primary equal-update training budget

固定：
- epochs = 30
- batch_size = 1024
- optimizer = `torch.optim.AdamW`
- lr = 3e-4
- weight_decay = 1e-4
- betas = (0.9, 0.999)
- eps = 1e-8
- amsgrad = False
- foreach = False
- fused = False
- capturable = False
- 每 step 固定 `optimizer.zero_grad(set_to_none=True)` → forward/CE(mean) → backward → `clip_grad_norm_(all_trainable_parameters, 1.0)` → optimizer.step()
- gradient clip global norm = 1.0
- scheduler = none
- early stopping = 禁止

每个架构每个 seed 完全相同 optimizer update 数、看到相同样本与 D4 transforms。

每个 primary run 开始前固定执行：
- `random.seed(S)`
- `numpy.random.seed(S)`
- `torch.manual_seed(S)`
- `torch.cuda.manual_seed_all(S)`
- `torch.backends.cudnn.benchmark = False`
- `torch.backends.cudnn.deterministic = True`
- `torch.backends.cuda.matmul.allow_tf32 = False`
- `torch.backends.cudnn.allow_tf32 = False`
- `torch.set_float32_matmul_precision("highest")`
- `torch.use_deterministic_algorithms(True, warn_only=True)`

模型必须在设置这些 seed 后重新实例化；禁止从 M3 Student checkpoint warm-start。

Precision 决策只有一次，而且必须在 §7 三个正式 plan 已生成并校验 SHA 后执行。算法固定：
1. 先调用 `torch.cuda.is_bf16_supported()`；返回 False 时两个架构的 BF16 smoke 状态都固定记 `NOT_SUPPORTED`，直接进入第 4 步；返回 True 时继续；
2. BF16 smoke 架构顺序固定 `Transformer2048 → ResidualMLP2048`。每个架构开始 smoke 前都重新执行 §9 的 seed/determinism 设置，使用 `S=20260919`，然后全新实例化 model + AdamW；读取 `artifacts/m4/plans/plan_20260919.npz` 的前 20 个正式 optimizer step，做 **20 steps**，禁止临时另造数据顺序；对 BF16 smoke 内部捕获 CUDA/PyTorch exception并把该架构标记 `FAIL_EXCEPTION`，不得让它冒泡成 §17 runtime blocker；
3. 每一步必须满足：loss finite、所有存在的 gradient finite、optimizer step 后所有 trainable parameter finite；任一不满足标记该架构 `FAIL_NONFINITE`。无论第一个架构成功/失败，都必须继续完成另一个架构的 BF16 smoke。两个架构均 20/20 PASS → primary precision 固定为 `BF16 autocast`，smoke 模型全部丢弃；
4. BF16 不支持或任一架构 BF16 非 PASS → 两个架构统一使用 FP32，并固定按 `Transformer2048 → ResidualMLP2048` 做 FP32 smoke。每个架构前再次重置 §9 seed/determinism、全新 model+AdamW，使用同一 plan 的**前 5 个正式 optimizer step**，不得复用 BF16 smoke 的 model/optimizer state；
5. FP32 smoke 每一步使用与第 3 步同样 finite 条件；任一架构 5-step FP32 非 PASS/exception → STOP，输出 `M4_BLOCKED_NUMERIC_SMOKE`；两者 PASS → primary precision 固定 FP32。

`artifacts/m4/progress/precision.json` 必须原子写入：BF16 support bool、两个架构 BF16 状态/完成 steps、是否 fallback、两个架构 FP32 状态/完成 steps（未运行则 `NOT_RUN`）、最终 selected precision。只有该文件完整后才开始 primary run；中断在 precision smoke 内时丢弃 partial smoke 模型并从整个 precision decision 第 1 步重跑。

禁止 FP16，禁止 GradScaler，禁止一个架构 BF16、另一个 FP32。最终 precision 决策写入 summary。

# 10. 每个 primary run 必须记录

§10 的所有 initial / per-epoch validation / final train / final validation eval 一律 `model.eval()` + `torch.inference_mode()` + FP32 inference（autocast disabled）。训练前必须在 **canonical、无 D4 augmentation** 的完整 train/validation split 上 eval：
- initial train CE
- initial validation CE
- initial raw best-action accuracy
- initial legal best-action accuracy
- initial legal pairwise ranking

metric 公式固定复用 M3 定义：
- CE = `F.cross_entropy(logits, teacher_best_action, reduction="sum") / samples`；
- raw accuracy = `logits.argmax == target`；
- legal accuracy = illegal logits 填 `-inf` 后 argmax 与 target 比较；
- pairwise ranking = 仅统计两个动作都 legal 且 teacher values 不相等的 action pair，比较 `(teacher_a > teacher_b)` 与 `(logit_a > logit_b)` 是否同号；
- eval batch size 固定 1024。

每 epoch：
- train CE = 该 epoch 7 个 augmented training batch 的 sample-weighted CE
- validation CE = 完整 canonical validation split CE
- validation legal best-action accuracy
- validation legal pairwise ranking
- training compute wall seconds：每个 epoch 只包围 7 个 training steps；epoch validation/checkpoint/日志不计入；epoch training 段前 `torch.cuda.synchronize()` + `time.perf_counter()`，7 steps 后再次 synchronize 并累加 elapsed
- optimizer steps
- samples/s = `6528 / 该 epoch training_compute_wall_seconds`
- peak VRAM：每个 primary run 在 initial eval 完成、正式 epoch 1 开始前执行 `torch.cuda.reset_peak_memory_stats()`；run 结束取 `torch.cuda.max_memory_allocated()`，因此记录的是整个 30-epoch train/validation 期间峰值（模型本身已在 GPU）
- GPU utilization diagnostic：P0 固定执行一次 `nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits`；若 exit 0，则**仅在每个 epoch 的 7-step training segment**期间用 1 Hz sampler，所有 epoch 有效样本合并后报告 mean/p95 utilization 与 peak memory.used；若整个 run 有效样本数=0 写 `N/A_NO_SAMPLES`；若 P0 命令非 0 写 `N/A_NVIDIA_SMI_UNAVAILABLE`；该 diagnostic 不参与 PASS/selection

训练结束固定使用 **epoch 30 最终模型**；禁止选择 validation 最好的历史 epoch。固定保存 `artifacts/m4/checkpoints/<architecture>_<seed>_final.pt`（model state + architecture + seed + precision + dataset SHA + plan SHA），计算整个 checkpoint 文件 SHA-256 并写入 summary。然后记录：
- final canonical train metrics
- final validation metrics
- D4 consistency diagnostic **此处只登记为 DEFERRED_TO_§11；不得在单个 run 结束时消费/索引任何 test row**
- total training compute wall = 30 个 epoch training-only elapsed 之和
- 210 个 step 的 GPU step time：每 step 用一对 `torch.cuda.Event(enable_timing=True)` 包围 forward/backward/clip/optimizer，epoch 末 synchronize 后读取 milliseconds；报告 mean/median/p95
- Q head/backbone parameter updates confirmed：训练前 clone 对应参数；epoch 30 后必须至少一个 backbone tensor 与至少一个 q_head tensor `torch.equal == False`，否则 STOP `M4_BLOCKED_NO_PARAMETER_UPDATE`
- V head / Afterstate head 参数与训练前 clone 必须对每个 tensor `torch.equal == True`；任一变化 → STOP `M4_BLOCKED_VALUE_HEAD_UPDATED`

dataset 文件/数组可以在程序启动时整体加载，但在全部 6 个 primary run 完成前，**禁止用 `split==test` 选出 test rows、禁止把 test row 转成训练/eval tensor、禁止把 test row 输入模型或任何 metric**。只有全部 6 个 primary run 完成后才允许消费 test rows。

# 11. Test Teacher metrics

全部 6 个 primary run 完成后才允许进入本节并首次消费 `split==test` 的 896 rows。Teacher train/validation/test metrics 与 D4 diagnostic 一律 `model.eval()` + `torch.inference_mode()` + **FP32 inference（autocast disabled）**；对每个 final model 按 §10 完全相同 metric 公式、batch size=1024 记录：
- test CE
- legal best-action accuracy
- legal pairwise ranking accuracy
- raw best-action accuracy
- D4 legal-argmax consistency
- centered-logit D4 MAE

D4 diagnostic **固定只在完整 canonical test split 896 states 上计算**。公式固定与 M3 一致：canonical logits 减去每行 4-action mean；对 transform id 1..7 使用 `transform_board_batch`、NN forward、`inverse_transform_q_values` 回原方向，再减每行 mean；报告每个 transform id 的 legal argmax consistency、7 个 id 合并后的 overall consistency，以及所有 restored-centered 与 canonical-centered 元素绝对误差的全局 mean。

test 不得用于改 LR/epoch/batch/architecture。

# 12. 实际游戏评测固定

每个 final model 必须进行 pure-NN greedy closed-loop evaluation。棋力评测一律 `model.eval()` + `torch.inference_mode()` + **FP32 inference（autocast disabled）**，不受训练 precision 决策影响。

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
- 令合法动作最大 logit 为 `qmax`；候选集合固定为所有 legal 且 `qmax - q[a] <= 1e-7` 的动作；1 个候选直接选，>1 个候选使用该局独立 tie-break RNG 做 uniform integer choice
- tie-break RNG 与 environment spawn RNG 分离。

game evaluator 实现固定，不得自行选择：

- movement/legal：使用 frozen `src/game2048/m2_fast_backend.py::move_selected_batch` / `legal_mask_batch`；
- 每个 game seed `G` 单独创建 `spawn_rng = numpy.random.Generator(numpy.random.PCG64(G))`；初始 board 固定为 `np.zeros(16, dtype=np.uint8)`，然后严格调用两次 `reference_env.spawn_random(board, spawn_rng).state` 生成起始两 tile；以后所有 spawn 继续只消费该局自己的 `spawn_rng`；初始 score=0；
- 每个 game seed `G` 单独创建 `tie_rng = numpy.random.Generator(numpy.random.PCG64(G ^ 0x9E3779B9))`；它只用于 Q tie-break，不得用于 spawn；
- evaluator **必须**把当前 evaluation shard 内所有 active boards 按 game_seed 升序合并成一个 batch 做 NN forward，并用 `move_selected_batch` 批量 movement；formal 2000-game evaluation 的 shard size 固定为 100，§12 correctness gate 的 gate batch 固定为 32；随后仅 spawn 阶段按 game_seed 升序逐局消费对应独立 RNG；某局提前 terminal 不得影响其他局 RNG；
- 每个 loop 先对 current boards 计算 legal mask；`legal.any(axis=1)==False` 的 row 立即 finalize 并从 active set 删除，禁止送入 NN；只对剩余 nonterminal rows forward；其所有 legal logits 必须 finite，任一 legal logit 非 finite → STOP `M4_BLOCKED_EVAL_NONFINITE`；
- tie candidate 固定为按 action id 升序的 `np.flatnonzero(legal & (qmax - q <= 1e-7))`；多个候选时使用 `candidate_index = tie_rng.integers(0, len(candidates))`，再取 `candidates[candidate_index]`；
- selected action 必须 legal，且对应 `move_selected_batch.moved` 必须 True，否则 STOP `M4_BLOCKED_GAME_EVALUATOR_CORRECTNESS`；每次成功 action 后 `moves += 1`；score 只累加 merge reward；terminal 后停止该局。

正式 2000-game 前固定用 **Transformer2048/20260919 final checkpoint 与 ResidualMLP2048/20260919 final checkpoint** 两个模型，各自对 evaluation seeds `20261001..20261032` 做 correctness gate：
1. 每个 model + 每个 seed 连续运行两次，final score / moves / max_tile_exp 必须完全一致；
2. 在第一次运行中，对每一步模型实际选择的 action，在 spawn 前同时计算 fast/reference：设该 row 为 `k`，必须 `np.array_equal(fast.afterstates[k], ref.afterstate)`、`int(fast.rewards[k]) == int(ref.reward)`、`bool(fast.moved[k]) == bool(ref.moved)`；三项逐步完全一致；
3. 任一失败 → STOP，输出 `M4_BLOCKED_GAME_EVALUATOR_CORRECTNESS`。

固定 paired game seeds：

`20261001 + i, i = 0..1999`

即每个 trained model **2000 games**。
两个架构、三个 training seeds 都使用完全相同 2000 game seeds。为保证断线恢复不改变 batch composition，2000 seeds 固定切成 **20 个连续 shard，每 shard 恰好 100 seeds**：shard `j=0..19` 包含 `20261001 + 100*j .. 20261001 + 100*j + 99`。每个 shard 从空盘独立开始并在该 shard 内动态 batch active games；禁止跨 shard 合批。game evaluation 的 6-model 执行顺序固定复用 §23 的 primary run 顺序，不得重排；每个 model 内 shard 必须按 j=0..19 顺序。

每局记录：
- final score
- max tile exponent
- moves

每个 model 汇总：
- mean score（主指标）
- median
- p10 / p90
- mean moves
- reach 2048/4096/8192/16384/32768/65536：分别定义为 `max_tile_exp >= 11/12/13/14/15/16` 的 game 比例；
- max-tile distribution：按 terminal `max_tile_exp` 的整数值做 `{exp: count}` 频数表；
- p10 / p90 均使用 `np.quantile(scores, [0.1, 0.9])` 默认 linear method。

# 13. 棋力统计规则

对每个 training seed，计算 paired：
`Transformer score - MLP score`

paired bootstrap 算法固定：
- delta vector = 同一 evaluation seed 的 `Transformer score - MLP score`；
- training seeds 按 `20260919, 20260920, 20260921` 顺序，三个 per-seed bootstrap RNG 分别固定为 `PCG64(20261101) / PCG64(20261102) / PCG64(20261103)`；每个 CI 都新建自己的 Generator；
- 恰好 10,000 次 resample；每次从 2000 个 paired index 中有放回抽取 2000 个 index并计算 mean delta；
- 95% CI = bootstrap mean delta 的 `np.quantile([0.025, 0.975])`（NumPy 默认 linear method）；
- 每个 training seed 分别报告 paired mean delta / CI。

aggregate 固定：对每个 evaluation seed，先分别对三个 Transformer training-seed score 求算术平均、三个 MLP training-seed score 求算术平均，再形成 2000 个 aggregate paired delta；使用全新 `numpy.random.Generator(numpy.random.PCG64(20261104))` 按同一算法做 10,000 bootstrap。

strength 判定机械执行：
1. aggregate CI lower > 0 且 3 个 per-training-seed **mean delta** 至少 2 个 > 0 → `TRANSFORMER_STRENGTH_SIGNIFICANT`；
2. aggregate CI upper < 0 且 3 个 per-training-seed **mean delta** 至少 2 个 < 0 → `MLP_STRENGTH_SIGNIFICANT`；
3. 其他所有情况 → `ARCHITECTURE_STRENGTH_NOT_STATISTICALLY_RESOLVED`。

# 14. 无法确认棋力差异时的固定 tie-break

严格按总计划“更简单、更快”处理。

§14 **不得另跑性能 benchmark**。唯一性能真源是 §15；直接读取 §15 产出的 8192-env closed-loop median：
- `T_median = performance.closed_loop["8192"]["Transformer2048"]["median_decisions_per_s"]`
- `M_median = performance.closed_loop["8192"]["ResidualMLP2048"]["median_decisions_per_s"]`

令 `fast = max(T_median, M_median)`，`slow = min(T_median, M_median)`，`speed_ratio = fast / slow`。

- 若 `speed_ratio >= 1.10`：选择 median throughput 更高者；
- 若 `speed_ratio < 1.10`：选择 trainable parameter count 更少者，即 Transformer2048（4,750,342 < 5,264,710）。

不存在第三种 tie-break，不得综合主观“复杂度”评分。

该 tie-break 只在实际棋力未统计确认时使用。

# 15. 性能 benchmark

性能 benchmark 每个架构固定使用 **training seed 20260919 的 epoch-30 final checkpoint**；权重数值不参与选择 checkpoint。执行路径固定 **PyTorch eager + §9 共同 primary precision**；M4 禁止 `torch.compile` benchmark，避免引入第二套执行路径。每次只把当前被测模型保留在 GPU，上一个模型删除后执行 `gc.collect(); torch.cuda.empty_cache()`。

Model-only inference 固定 batch：`1, 256, 1024, 2048, 4096, 8192`。每个 batch size 的 5 个 repeat 测量顺序固定：repeat 0/2/4 为 Transformer2048→ResidualMLP2048，repeat 1/3 为 ResidualMLP2048→Transformer2048；模型加载/删除不计入 timed 区间。每个 architecture/batch size：
- 必须加载该 architecture 的 training seed 20260919 final checkpoint，并验证 checkpoint metadata 的 architecture/seed/precision/dataset SHA 与当前 M4 evidence 一致；SHA 必须等于 primary_training 中记录的 final checkpoint SHA，否则 STOP `M4_BLOCKED_PERFORMANCE_CHECKPOINT_MISMATCH`；
- 固定 synthetic input RNG = `numpy.random.Generator(numpy.random.PCG64(20261300 + batch_size))`；生成 shape `(batch_size,16)`、dtype `uint8`、每格均匀整数 `[0,21]`（即 `integers(0,22)`）并一次性传到 GPU；timed loop 复用同一 tensor，不包含 H2D；
- inference context 固定为 `model.eval()` + `torch.inference_mode()`；若 §9 primary precision=`BF16 autocast`，则用 `torch.autocast(device_type="cuda", dtype=torch.bfloat16)`；若为 FP32，则 autocast disabled；
- 每个 architecture/batch size 恰好做 5 个 repeat；每个 repeat 开始前 `torch.cuda.reset_peak_memory_stats()`，50 iterations warmup，再 `torch.cuda.synchronize(); t0=time.perf_counter()`，连续 200 forward，随后 synchronize并取 elapsed；
- 每 repeat 固定 `states_per_s = batch_size * 200 / elapsed`，`mean_latency_s = elapsed / 200`，`peak_vram_bytes = torch.cuda.max_memory_allocated()`；
- 最终 `median_states_per_s=np.median(repeat_states_per_s)`、`median_latency_s=np.median(repeat_latency_s)`、`peak_vram_bytes=max(repeat_peak_vram_bytes)`。

Closed-loop 只测固定两个规模：`2048` 与 `8192` envs。§15 是该 benchmark 的唯一协议真源。

对每个 `env_count E in [2048, 8192]`、每个 `repeat_index r in [0,1,2,3,4]`、每个架构：
- 架构执行顺序固定：r=0/2/4 为 Transformer2048→ResidualMLP2048，r=1/3 为 ResidualMLP2048→Transformer2048；
- 加载/验证 checkpoint 规则与本节 model-only 完全相同；inference context 同样使用 §9 primary precision；
- 构造 `M2RolloutBatchEnv(E, seed=20261200+r)`，随后显式 `reset(seed=20261200+r)`；
- 每个 closed-loop step 的数据路径固定：`boards_np = prepare_board_batch_for_transfer(env.boards)`；`legal_np = np.ascontiguousarray(env.current_legal)`；同步 `torch.from_numpy(...).to("cuda", non_blocking=False)` 得到 boards/legal；在固定 precision inference context 下 `q=model(boards)`；然后 `select_greedy_actions(q.float(), legal, generator=generator, tie_atol=1e-6)`；actions 同步 `.to("cpu").numpy().astype(np.uint8, copy=False)`；调用 `env.step(actions_np)`；若 `step.terminated.any()` 则 `env.reset_where(step.terminated)`；
- `generator = torch.Generator(device="cuda").manual_seed(20261400 + E*10 + r)`，从该 repeat 第一个 warmup step 一直连续使用到 500 个 timed step结束，不重置；
- 当前架构 checkpoint 加载完成后先做 20 个上述 closed-loop warmup steps，不计时；
- 每个 repeat timed 区间开始前记录 `cpu0=time.process_time()`，`torch.cuda.synchronize()` 后 `t0=time.perf_counter()`；计时恰好 500 个完整 closed-loop steps（**包括 CPU board/legal 准备、H2D、NN、action selection、D2H、env.step、terminal reset**），随后 synchronize；
- `elapsed_seconds = perf_counter()-t0`，`cpu_seconds=time.process_time()-cpu0`，`decisions_per_s = E * 500 / elapsed_seconds`，`normalized_cpu_percent = 100 * cpu_seconds / elapsed_seconds / os.cpu_count()`；
- 每个 repeat 开始前 `torch.cuda.reset_peak_memory_stats()`；模型加载/删除与 warmup 不计入 timed wall；`peak_vram_bytes=torch.cuda.max_memory_allocated()`。

每个 E / architecture 恰好得到 5 个 decisions/s，并报告 median。**M4 不测 16384/32768**，不得自行扩展 benchmark 矩阵。

每个 closed-loop point 固定记录：
- 5 个 repeat decisions/s + median；
- wall seconds；
- peak VRAM = `torch.cuda.max_memory_allocated()`；
- GPU utilization：若 P0 `nvidia-smi` 可用，则每个 timed repeat 开始时启动 1 Hz sampler、timed repeat 结束立即停止；有 >=1 样本则该 repeat 记录 mean utilization，5 repeats 的 point 值取所有有效样本合并后的 mean/p95；若某 point 总有效样本数为 0，则写 `N/A_NO_SAMPLES`；若 P0 不可用则写 `N/A_NVIDIA_SMI_UNAVAILABLE`；该字段纯 diagnostic，不参与 PASS/selection；
- normalized CPU process utilization：记录每个 repeat 的 `normalized_cpu_percent`，point 值取 5 个 repeat 的 median；
- pipeline ratio = 同一 `E` 的 model-only batch=`E` states/s median ÷ closed-loop decisions/s median；因此只对 E=2048 与 E=8192 计算。

任一 required model-only 或 closed-loop point OOM → STOP，`M4_BLOCKED_PERFORMANCE_OOM`；任一 throughput/latency 非 finite 或 <=0 → STOP，`M4_BLOCKED_PERFORMANCE_INVALID`。禁止缩小 required batch/env 或删除 point。

Training 性能不另起一套训练任务，直接从 6 个 primary runs 统计：
- 每 run 的 `6528 * 30 / total_training_compute_wall` 作为 training-only samples/s；
- 210 个 optimizer step 的 CUDA-event mean/median/p95 step time；
- peak VRAM；
- 每架构对 3 个 seed 再报告上述指标的 median。

# 16. Equal-wall-clock secondary

M4 对 same wall-clock/GPU-hour 的处理固定使用本节机械 trigger；Agent 不得自行决定是否追加其他 compute-normalized 实验。

Primary 6 runs 完成后，使用 §10 定义的 `total_training_compute_wall`，分别取 Transformer 三个 seed 与 ResidualMLP 三个 seed 的 median，得到 `T_primary_wall_median` / `M_primary_wall_median`。

令 `wall_ratio = max(T_primary_wall_median, M_primary_wall_median) / min(T_primary_wall_median, M_primary_wall_median)`。initial/final/epoch validation、checkpoint I/O、report/logging 均不计入该 wall。

- `wall_ratio <= 1.20`：`equal_wall_clock_secondary.status = NOT_TRIGGERED`，不得额外训练；
- `wall_ratio > 1.20`：必须执行 secondary，禁止跳过。

secondary 固定协议：
- common budget seconds = `max(T_primary_wall_median, M_primary_wall_median)`；
- 两架构执行顺序固定 `Transformer2048 → ResidualMLP2048`；二者均重新 random init，training seed 固定 `20260919`；每个架构开始前都重新执行 §9 的全部 seed/determinism 设置；模型初始化/optimizer/precision 与 primary 完全相同；
- secondary 数据流与 primary plan 分离，使用**无上限、可恢复的 stateless per-epoch plan**：secondary epoch `e=0,1,2,...` 时创建 `rng = numpy.random.Generator(numpy.random.PCG64(20262000 + e))`，先 `rng.permutation(train_indices)`，再 `rng.integers(0,8,size=6528,dtype=np.uint8)`；batch 切分仍为 1024×6 + 384×1；两架构对同一 e 必须得到完全相同 plan；
- `secondary_plan_id` 固定为 `FBDCB22E4A9A16359073740890F3FBC1F5C91031656F385487AEB3F507B48642`，即 ASCII 字符串 `M4_SECONDARY_STATELESS_PCG64_20262000_V1` 的 SHA-256；不使用 primary `plan_sha256` 代替；
- secondary 的 `training_only_wall` 计时口径必须与 primary §10 完全一致：每个完整 epoch 的 7 个 training steps 作为一个连续 training segment，segment 前 synchronize + perf_counter，7 steps 完成后 synchronize 并累加 wall；validation/checkpoint/logging 不计入；
- 每个 secondary epoch 开始前检查累计 `training_only_wall`；若已 `>= common_budget` 则停止，否则执行完整 7-step epoch；因此最终累计最多超预算一个完整 epoch；
- secondary 必须每完成一个完整 epoch就原子保存 `artifacts/m4/secondary/<architecture>_latest.pt` 与 `artifacts/m4/secondary/<architecture>_progress.json`，至少含 model/optimizer state、next_epoch、completed_steps、training_only_wall、secondary_plan_id、precision、dataset SHA；中断在 epoch 内时从最近完整 epoch checkpoint 重做该未完成 epoch，禁止重放已标记完成 epoch；
- 停止后保存 `artifacts/m4/secondary/<architecture>_final.pt`，记录 final checkpoint SHA；
- 不做 early stopping；
- 最终 validation Teacher metrics 按 §10 FP32 canonical validation 公式计算；
- 两个 secondary final model 都使用 §12 完全相同的 2000 game seeds 与 FP32 game evaluator，raw 结果分别保存 `artifacts/m4/secondary/games/<architecture>.npz`；
- secondary 只报告两架构 mean score、median、reach rates 与 paired mean delta；paired bootstrap 95% CI 固定使用全新 `PCG64(20261105)`、10,000 resamples、与 §13 相同算法。

secondary **永远不改变 §13 strength conclusion，也不改变 §14 selection algorithm**；它只写入报告作为 compute-normalized diagnostic，不得作为额外主观 tie-break。

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

禁止使用“表现差 / 学得慢 / 看起来没收敛”作为 STOP 或调参理由。只按以下技术失败处理：

- 任一 required primary run 出现 CUDA OOM → STOP，`M4_BLOCKED_PRIMARY_OOM`；禁止减 batch；
- 任一 required primary run 出现 non-finite loss/gradient/parameter → STOP，`M4_BLOCKED_PRIMARY_NUMERIC`；
- 任一 required script 自身 exit code 非 0 / uncaught exception → 立即 STOP，`M4_BLOCKED_RUNTIME_ERROR`；不得自动改代码、改配置或重跑掩盖错误；
- 指标有限但很差 → 仍必须完成全部 6 runs、2000-game eval、统计与 selection；不得救场调参。

工具连接/会话中断但本地脚本仍正常运行，不属于 runtime error；重新连接后按 §1/§22 session 状态机继续读取/恢复，不得启动第二个重复任务。

# 18. Checkpoint / artifact 边界

正式本地 artifact：
`artifacts/m4/`

保存：
- per-run checkpoints
- training histories
- raw 2000-game score arrays
- performance raw measurements

这些默认不 Git track。

# 19. Tracked file scope

Git 中 M4 **必须且只允许**新增/修改以下 tracked files：
- `src/game2048/m4_compare.py`
- `benchmarks/run_m4_architecture_compare.py`
- `benchmarks/benchmark_m4_architecture_performance.py`
- `tests/test_m4_architecture_compare.py`
- `reports/m4/m4_architecture_compare.json`
- `reports/m4/M4_REPORT.md`

除上述 6 个路径外，不得修改任何 tracked source/test/prompt/CI/config 文件；若发现必须改第 7 个 tracked 文件才能继续，STOP，输出 `M4_BLOCKED_SCOPE_CHANGE_REQUIRED`。

职责固定：
- `src/game2048/m4_compare.py`：deterministic plan、metric、game evaluator、bootstrap、selection 的纯函数/helper；
- `benchmarks/run_m4_architecture_compare.py`：precision smoke、6 primary runs、Teacher metrics、2000-game eval、statistics、conditional equal-wall secondary；
- `benchmarks/benchmark_m4_architecture_performance.py`：model-only 与 2048/8192 closed-loop benchmark；
- `tests/test_m4_architecture_compare.py`：§20 固定 CPU tests；
- `reports/m4/m4_architecture_compare.json`：机器可读最终 evidence；
- `reports/m4/M4_REPORT.md`：人类可读最终报告。

不得提交 GPU model binary / raw 2000-game arrays / training histories；它们只放 `artifacts/m4/`。
仓库根目录不得产生 M4 文件。

机器可读 summary 顶层 key 固定为：
`schema_version, result, dataset, models, fairness, precision, plans, primary_training, teacher_metrics, game_evaluation, paired_statistics, d4, performance, equal_wall_clock_secondary, selection, regression`。
`schema_version` 固定为 `1`；`result` 在 candidate report 写入时固定为 `M4_CANDIDATE_EVIDENCE_COMPLETE`（它不表示 CI 已完成）；禁止自行改顶层 key 名。

最低子字段固定：
- `dataset`: `path, sha256, checkpoint_sha256, states, games, split_counts`；
- `models`: 两架构的 `parameter_count`；
- `fairness`: `work_order_version, prompt_sha256, base_head, epochs, batch_size, optimizer, lr, weight_decay, grad_clip, training_seeds, evaluation_seed_start, evaluation_games, game_shard_size, tie_tolerance`；
- `precision`: selected precision + 两架构 smoke pass/fail 细节；
- `plans`: 3 个 seed 的 `plan_sha256`；
- `primary_training`: 6 个 run 的 final checkpoint path/SHA、wall、step-time、samples/s；
- `teacher_metrics`: 6 个 run 的 canonical train/validation/test metrics；
- `game_evaluation`: 6 个 run 的 2000-game summary + raw NPZ path；
- `paired_statistics`: 3 个 per-seed delta/CI + aggregate delta/CI；
- `d4`: 6 个 final model 的 consistency/MAE；
- `performance.model_only`: batch key 固定为字符串 `"1","256","1024","2048","4096","8192"`；每个 batch 下固定 `Transformer2048/ResidualMLP2048`，各含 `repeat_states_per_s[5], repeat_latency_s[5], median_states_per_s, median_latency_s, peak_vram_bytes`；
- `performance.closed_loop`: env key 固定为字符串 `"2048","8192"`；每个 env 下固定 `Transformer2048/ResidualMLP2048`，各含 `repeat_decisions_per_s[5], median_decisions_per_s, repeat_wall_s[5], peak_vram_bytes, gpu_utilization, cpu_utilization, pipeline_ratio`；
- `performance.training`: 两架构各自 3-run training-only samples/s 与 median、step-time median/p95、peak VRAM；
- `equal_wall_clock_secondary`: `status` 只允许 `NOT_TRIGGERED` 或 `TRIGGERED_COMPLETE`；触发时必须含 `secondary_plan_id, common_budget_seconds, Transformer2048, ResidualMLP2048, paired_delta`；
- `selection`: `strength_conclusion, selected_architecture, selection_rule`；其中 `selection_rule` 只允许 `SCORE_SIGNIFICANCE / 8192_SPEED / PARAMETER_COUNT` 三个字符串；
- `regression`: local pytest counts。

# 20. Tests / correctness

`tests/test_m4_architecture_compare.py` 固定必须包含并通过以下 8 个 test function（名字也固定）：
- `test_training_plan_is_identical_across_architectures`
- `test_d4_board_and_action_transform_alignment`
- `test_gameplay_masks_illegal_actions`
- `test_spawn_rng_is_independent_from_tie_rng`
- `test_paired_bootstrap_is_deterministic`
- `test_game_score_aggregation`
- `test_training_path_never_consumes_test_split`
- `test_policy_only_loss_leaves_value_heads_untouched`

existing M0-M3 tests不得删除/skip/xfail。

# 21. 开工 P0

FRESH P0 第一动作必须按顺序报告：
- HEAD / origin/main（必须相等）
- working tree（必须 clean）
- root layout check（不得有 milestone report/result JSON）
- four frozen tag resolutions
- frozen M2 source diff gate = 0
- M3 dataset file SHA-256 + shape/splits/checkpoint-SHA metadata
- official Python version
- torch / CUDA / GPU name
- `nvidia-smi` availability status
- exact model parameter counts：Transformer 必须 `4,750,342`、ResidualMLP 必须 `5,264,710`；任一不符 → STOP `M4_BLOCKED_MODEL_DRIFT`

仅 FRESH P0 使用官方 Python运行 baseline：
`D:\sd-webui-forge-aki-v1.0\python\python.exe -m pytest -q`

结果必须精确为：
- 525 passed
- 0 failed
- 0 skipped
- 0 xfailed

失败则 STOP，不得用 M4 修改 frozen source 来救。RESUME 不重跑这条 baseline，而是读取 session 中已冻结的 `p0_pytest`。

# 22. 长任务可观测性与断线恢复

固定 session state 枚举：
`P0_PASSED, P1_READY, PRIMARY_RUNNING, PRIMARY_DONE, PERFORMANCE_RUNNING, PERFORMANCE_DONE, FINALIZE_RUNNING, FINALIZED, P10_RUNNING, P10_DONE, REPORT_WRITTEN, CANDIDATE_PUSHED, CLOSEOUT_REPORT_READY, CLOSEOUT_PUSHED, COMPLETE`。

固定 artifact：
- `artifacts/m4/progress/session.json`
- `artifacts/m4/progress/precision.json`
- `artifacts/m4/progress/primary.json`
- `artifacts/m4/progress/train_<architecture>_<seed>.json`
- `artifacts/m4/checkpoints/<architecture>_<seed>_latest.pt`
- `artifacts/m4/checkpoints/<architecture>_<seed>_final.pt`
- `artifacts/m4/games/<architecture>_<seed>.npz`
- `artifacts/m4/progress/performance.json`
- `artifacts/m4/progress/finalize.json`
- `artifacts/m4/pytest.xml`
- §16 触发时的 `artifacts/m4/secondary/...`。

session 更新必须原子写临时文件再 rename；每个 phase 启动前先写 `*_RUNNING`，成功结束后写对应 `*_DONE/FINALIZED`。

Primary 训练：每个 epoch 结束必须原子更新 `train_<architecture>_<seed>.json`，并覆盖 `latest.pt`（包含 model state、optimizer state、next_epoch、seed、precision、dataset SHA、plan SHA）；同时输出 architecture / seed / epoch / train CE / validation CE / val accuracy / elapsed / samples/s。中断在 epoch 内时，恢复从最近一个已完整保存 epoch 的 checkpoint 开始，重新执行该未完成 epoch；不得重放已标记完成的 epoch。

每个 primary run 完成后，其 `train_<architecture>_<seed>.json` 必须追加并冻结：`completed=true, final_checkpoint_path, final_checkpoint_sha256, total_training_compute_wall, step_time_mean_ms, step_time_median_ms, step_time_p95_ms, peak_vram_bytes, final_train_metrics, final_validation_metrics`。

2000-game evaluation：严格以 §12 的 20 个固定 100-game shard 为恢复单位。只有一个 shard 100 局全部 terminal 后才允许把该 shard 结果并入 NPZ；中断在 shard 内时丢弃该 shard 的内存中部分结果并从该 shard 第一个 seed 重新跑，禁止把 partial shard 与新 batch composition 拼接。NPZ key 固定为 `game_seed, final_score, max_tile_exp, moves`，数组只包含完整 shard、按 game_seed 升序。每完成一个 shard 原子更新 NPZ，并输出 architecture / training seed / completed_shards / completed_games / elapsed / games/s / running mean score。

P2～P6 全部完成后，`--phase primary` 必须最后原子生成 `artifacts/m4/progress/primary.json`，它是 P7/P8/P9 的唯一 primary evidence manifest。固定顶层 key：`schema_version, work_order_version, dataset, precision, plans, primary_training, teacher_metrics, d4, game_evaluation, paired_statistics`；`schema_version=1`。它必须引用 6 个 final checkpoint 路径/SHA、6 个 raw game NPZ 路径及其 2000-game 完整性、3 个 plan SHA、§13 全部统计。只有 primary.json 完整写入后才允许 session=`PRIMARY_DONE`。

性能 benchmark：P7 必须只从 `primary.json` 读取 selected precision 与 seed=20260919 两个 final checkpoint path/SHA。执行顺序固定：先 model-only batch 按 `1→256→1024→2048→4096→8192`，每个 batch 内 r=0..4；再 closed-loop env 按 `2048→8192`，每个 env 内 r=0..4。

**performance resume 的原子单位是 architecture pair**：同一个 `(kind, size, repeat_index)` 的两个架构必须按 §15 固定顺序全部成功后，才允许把这一 pair 的两份结果一起原子写入 `artifacts/m4/progress/performance.json`。中断/异常发生在 pair 中间时，丢弃该 pair 已测但尚未 commit 的内存/temp 结果，恢复时按原固定架构顺序重跑整个 pair；已完整 commit 的 pair 禁止重跑。P7 完成时 performance.json 必须满足 §19 performance schema 后才允许 session=`PERFORMANCE_DONE`。

Primary resume metadata 固定检查：
- architecture
- training seed
- precision
- dataset SHA
- 对应 primary `plan_sha256`
- final checkpoint 预期 architecture/seed。
任一不一致 → STOP `M4_BLOCKED_RESUME_METADATA_MISMATCH`。

若 session/progress 宣称某 plan/checkpoint/game NPZ/manifest/repeat 已完成，但对应文件缺失、无法解析、数组 shape/key 不符、或已有记录 SHA 与实际文件不符 → STOP `M4_BLOCKED_RESUME_ARTIFACT_CORRUPT`；不得自行删除该 evidence 后重算。

Secondary resume 只按 §16 的 `secondary_plan_id + dataset SHA + precision + architecture + training seed` 检查，不得拿 primary `plan_sha256` 代替。

Git/session 窄窗口恢复也固定，且**先于 §1 的 state Git gate 执行**：

1. candidate commit 窗口（session.state=`REPORT_WRITTEN`）：
   - `HEAD==origin/main==base_head`：说明尚未 commit，保持 REPORT_WRITTEN；
   - 若 worktree clean、`HEAD^==base_head`、commit message 精确为 `m4: compare Transformer and ResidualMLP`、changed-path set 精确等于 §19 六个路径：
     - 若 `origin/main==base_head`：说明 commit 已完成但 push 未完成；固定执行一次普通 `git push origin main`；push 非 0 → STOP `M4_BLOCKED_RESUME_GIT_STATE`；
     - push 后或原本就满足 `origin/main==HEAD`：写 `candidate_sha=HEAD, state=CANDIDATE_PUSHED`；
   - 其他情况 → STOP `M4_BLOCKED_RESUME_GIT_STATE`。

2. closeout report 编辑窗口（session.state=`CLOSEOUT_REPORT_READY`）：
   - `HEAD==origin/main==candidate_sha` 且 dirty path 为空或仅 `reports/m4/M4_REPORT.md`：继续 closeout edit/commit；
   - 若 worktree clean、`HEAD^==candidate_sha`、commit message 精确为 `docs: close M4 candidate report`、changed-path set 精确等于 `reports/m4/M4_REPORT.md`：
     - 若 `origin/main==candidate_sha`：说明 closeout commit 已完成但 push 未完成；固定执行一次普通 `git push origin main`；push 非 0 → STOP `M4_BLOCKED_RESUME_GIT_STATE`；
     - push 后或原本就满足 `origin/main==HEAD`：写 `closeout_sha=HEAD, state=CLOSEOUT_PUSHED`；
   - 其他情况 → STOP `M4_BLOCKED_RESUME_GIT_STATE`。

3. 若 session.state=`CANDIDATE_PUSHED` 但发现 HEAD 已是合法 closeout commit，只有在**上述第 2 条**的合法 closeout commit 条件全部成立时才允许自动升级为 `CLOSEOUT_PUSHED`；否则 STOP。

4. 其他任何 HEAD/session 不一致均 STOP；不得 reset/rebase/restore/force-push 猜测恢复。

# 23. 执行顺序锁死

FRESH 从 P0 开始；RESUME 根据 session.state **跳到第一个未完成 phase**，不得重跑已经成功完成的 phase。

P0 FRESH frozen/preflight/full pytest
→ 成功后 session=`P0_PASSED`
→ P1 只创建 §19 的 4 个 code/test files，完成 helper/harness/performance harness/8 CPU tests；`run_m4_architecture_compare.py` CLI 必须支持 `--phase {primary,finalize,report}` + `--resume`，`benchmark_m4_architecture_performance.py` 必须支持 `--resume`
→ P1.1 固定执行 `D:\sd-webui-forge-aki-v1.0\python\python.exe -m pytest tests/test_m4_architecture_compare.py -q`；必须精确 `8 passed / 0 failed / 0 skipped / 0 xfailed`，禁止 parametrization 产生额外 collected cases
→ 成功后 session=`P1_READY`
→ 写 session=`PRIMARY_RUNNING`
→ P2～P6 固定执行：`D:\sd-webui-forge-aki-v1.0\python\python.exe benchmarks/run_m4_architecture_compare.py --phase primary --resume`
→ `--phase primary` 内部固定顺序：**生成/验证 3 个 §7 primary plan → precision smoke → 6 primary runs → 全 6 run 完成后首次消费 test rows并执行 §11 Teacher/D4 metrics → §12 evaluator correctness gate → 6 × 2000 games → §13 paired statistics**
→ 6 primary run 顺序固定为：Transformer/20260919 → ResidualMLP/20260919 → ResidualMLP/20260920 → Transformer/20260920 → Transformer/20260921 → ResidualMLP/20260921
→ primary 成功后 session=`PRIMARY_DONE`
→ 写 session=`PERFORMANCE_RUNNING`
→ P7 固定执行：`D:\sd-webui-forge-aki-v1.0\python\python.exe benchmarks/benchmark_m4_architecture_performance.py --resume`
→ performance 成功后 session=`PERFORMANCE_DONE`
→ 写 session=`FINALIZE_RUNNING`
→ P8～P9 固定执行：`D:\sd-webui-forge-aki-v1.0\python\python.exe benchmarks/run_m4_architecture_compare.py --phase finalize --resume`
→ `--phase finalize` **只读取** `primary.json + performance.json`，做 §16 conditional equal-wall secondary（机械触发时）与 §24 selection，并把结果原子写入 `artifacts/m4/progress/finalize.json`；不得重算 primary/P7、不得先写 tracked report；`finalize.json` 固定至少含 `schema_version=1, wall_ratio, equal_wall_clock_secondary, strength_conclusion, speed_ratio, selected_architecture, selection_rule`
→ finalize 成功后 session=`FINALIZED`
→ 执行 candidate artifact cleanup：保留 §25 要求的 final/evidence 文件，删除所有 primary/secondary `*_latest.pt`、临时 `.tmp/.lock`；cleanup 必须幂等，完成后验证所有 required final artifacts 可读且 SHA 与 manifest 一致
→ 写 session=`P10_RUNNING`
→ P10 固定执行：`D:\sd-webui-forge-aki-v1.0\python\python.exe -m pytest -q --junitxml=artifacts/m4/pytest.xml`
→ 成功且 JUnit/pytest 均确认 `533 passed / 0 failed / 0 errors / 0 skipped / 0 xfailed` 后 session=`P10_DONE`
→ 固定执行：`D:\sd-webui-forge-aki-v1.0\python\python.exe benchmarks/run_m4_architecture_compare.py --phase report --resume`
→ `--phase report` 的数据输入**只能**是 `session.json + primary.json + performance.json + finalize.json + pytest.xml`（以及这些 manifest 引用的路径字符串/SHA，不得重新打开 raw game/checkpoint 做新计算）；不得训练、评测或改 selection；必须再次确认 tests=533 且 0 failures/errors/skipped，然后机械合并写 `reports/m4/m4_architecture_compare.json` + candidate version of `reports/m4/M4_REPORT.md`；candidate report 的 Git/CI 字段固定写 `candidate_sha=PENDING_BY_DESIGN, candidate_ci=PENDING_BY_DESIGN, closeout_commit_sha=SELF_NOT_EMBEDDABLE_BY_DESIGN, closeout_ci=PENDING_BY_DESIGN`
→ report 成功后 session=`REPORT_WRITTEN`
→ candidate implementation commit/push
→ push 成功后立即写 `candidate_sha=HEAD, state=CANDIDATE_PUSHED`
→ candidate GitHub Actions green
→ candidate CI success 后先写 session=`CLOSEOUT_REPORT_READY`
→ mandatory report-only closeout：只修改 `reports/m4/M4_REPORT.md`；把 `candidate_sha` 与 `candidate_ci` 回写真实值；`closeout_parent` 固定等于 candidate_sha；`closeout_commit_sha` 固定保持 `SELF_NOT_EMBEDDABLE_BY_DESIGN`，禁止 amend 自引用；`closeout_ci` 固定写 `PENDING_BY_DESIGN_AT_REPORT_COMMIT`
→ report-only closeout commit/push
→ push 成功后立即写 `closeout_sha=HEAD, state=CLOSEOUT_PUSHED`
→ closeout GitHub Actions green
→ session=`COMPLETE`
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

selection 映射固定：
- `TRANSFORMER_STRENGTH_SIGNIFICANT` → `M4_SELECT_TRANSFORMER`，`selection_rule=SCORE_SIGNIFICANCE`；
- `MLP_STRENGTH_SIGNIFICANT` → `M4_SELECT_RESIDUAL_MLP`，`selection_rule=SCORE_SIGNIFICANCE`；
- `ARCHITECTURE_STRENGTH_NOT_STATISTICALLY_RESOLVED` 且 `speed_ratio >= 1.10` → 选择 §14 median throughput 更高者，`selection_rule=8192_SPEED`；
- `ARCHITECTURE_STRENGTH_NOT_STATISTICALLY_RESOLVED` 且 `speed_ratio < 1.10` → `M4_SELECT_TRANSFORMER`，`selection_rule=PARAMETER_COUNT`；
- 只有施工单明确的 blocker 才允许 `M4_BLOCKED`。

不得凭主观判断覆盖该映射。

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
- inference/training/closed-loop performance complete且所有 required point finite、无 OOM
- selection rule mechanically satisfied
- no architecture-specific tuning
- final full pytest：精确 533 passed / 0 failed / 0 skipped / 0 xfailed
- candidate CI green
- mandatory report-only closeout CI green
- `session.state == COMPLETE`
- artifacts organized / temp files cleaned：始终保留 6 个 primary final checkpoint、3 个 primary plan、6 个 primary raw game NPZ、performance evidence、pytest.xml、session/progress JSON；若 §16 触发还保留 2 个 secondary final checkpoint、2 个 secondary raw game NPZ、secondary progress/evidence；删除所有 primary/secondary `*_latest.pt` 与仅用于运行中的临时 scratch/lock 文件

# 26. STOP / Git

M4 必须创建普通 implementation candidate commit并 push。commit 前的事务门固定：
- `git status --porcelain=v1 -uall` 中只能出现 §19 六个路径；
- 执行下方显式 `git add -- ...` 后，`git diff --name-only` 必须为空（无 unstaged tracked 修改）；
- `git ls-files --others --exclude-standard` 必须为空（无未跟踪非 ignored 文件）；
- `git diff --cached --name-only` 的集合必须**精确等于** §19 六个路径，缺一或多一都 STOP `M4_BLOCKED_CANDIDATE_STAGE_SET`；
- `git diff --cached --check` 必须 exit 0。

固定命令语义：

```text
git add -- src/game2048/m4_compare.py benchmarks/run_m4_architecture_compare.py benchmarks/benchmark_m4_architecture_performance.py tests/test_m4_architecture_compare.py reports/m4/m4_architecture_compare.json reports/m4/M4_REPORT.md
git diff --cached --check
git commit -m "m4: compare Transformer and ResidualMLP"
git push origin main
```

该 commit 是 M4 candidate implementation 的权威候选 SHA；commit 后立即记录 `git rev-parse HEAD`。

candidate CI 固定按 `headSha == candidate_sha`、workflow name=`CI`、event=`push` 过滤；若出现多个匹配 run，固定取 databaseId 最大者；轮询直到 completed。

- conclusion=`success` → 继续；
- 任何其他 conclusion（failure/cancelled/timed_out/action_required/stale/neutral/skipped 等）→ 立即 STOP `M4_BLOCKED_CANDIDATE_CI`；执行 Agent 不得 rerun CI、不得修改代码、不得创建第二个 candidate commit。

candidate CI green 后**必须**创建恰好一个 report-only closeout descendant。进入 commit 前固定检查：dirty/untracked path 只能是 `reports/m4/M4_REPORT.md`；执行显式 add 后 `git diff --name-only` 与 `git ls-files --others --exclude-standard` 都必须为空；`git diff --cached --name-only` 必须精确只有 `reports/m4/M4_REPORT.md`；`git diff --cached --check` 必须 exit 0，否则 STOP `M4_BLOCKED_CLOSEOUT_STAGE_SET`。

closeout 只修改 `reports/m4/M4_REPORT.md`，写入最终 candidate SHA、candidate CI run ID/conclusion、closeout parent。固定：

```text
git add -- reports/m4/M4_REPORT.md
git diff --cached --check
git commit -m "docs: close M4 candidate report"
git push origin main
```

closeout commit push 后按 `headSha == closeout_sha`、workflow name=`CI`、event=`push` 过滤；多个匹配 run 固定取 databaseId 最大者并轮询到 completed。conclusion 只有 `success` 才通过；任何其他 conclusion → 立即 STOP `M4_BLOCKED_CLOSEOUT_CI`，不得 rerun、不得修改 implementation/report、不得创建第三个 commit。只有 closeout CI success 后才写 session=`COMPLETE` 并允许输出 `M4 CANDIDATE COMPLETE`。

禁止：
- 创建 M4 audited tag
- 标记 M4 FROZEN
- 进入 M5
- 执行 Teacher Checkpoint Promotion Gate

candidate 完成后 STOP，交给独立审计。

禁止 `git add -A` / `git add .`。
`artifacts/m4/`、checkpoint、raw game arrays 不得进入任何 commit。

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

任何阶段触发 `M4_BLOCKED_*` 或 `M4_BLOCKED` 时，统一立即 STOP，并输出：`RESULT=M4_BLOCKED`、精确 blocker code、当前 session state、触发条件实际值、已完成 evidence/artifact 路径、`git status --short`、HEAD/origin/main；不得继续后续 phase，不得创建 M4 candidate/tag。

只有 §25 全部成立后才算完成 candidate，并输出：
`M4 CANDIDATE COMPLETE`

并明确：
- selected architecture
- strength conclusion
- selection依据：必须原样输出 `selection_rule`（`SCORE_SIGNIFICANCE / 8192_SPEED / PARAMETER_COUNT` 之一）及其对应关键数值
- candidate implementation commit
- mandatory report-only closeout commit
- candidate CI run
- closeout CI run
- no M4 tag
- no M5

然后 STOP。