# M4 Transformer vs MLP — 严格施工提示词

> 项目：Safolour/NN-2048-ai
> 正式工作区：D:\CodexTasks\NN-2048-ai
> 当前阶段：M4 Transformer vs MLP
> 本文件是 M4 的唯一详细施工单；authoritative master plan 始终优先。

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

开工固定执行：

```text
git fetch origin
git rev-parse HEAD
git rev-parse origin/main
git status --short
```

硬门：
- `HEAD == origin/main`；
- `git status --short` 必须为空；
- `git merge-base --is-ancestor "m3-teacher-audited-pass^{}" HEAD` 必须 exit 0；
- 根目录不得存在任何 `M*_REPORT.md` 或 `m*_*.json` milestone 产物。

任一不满足：STOP，输出 `M4_BLOCKED_START_STATE`，不得自行 pull/rebase/reset/restore/delete。
不得修改 M0/M1/M2/M3 frozen implementation 或移动 frozen tags。

# 2. 工作区边界

当前 main 已完成 post-M3 cleanup；M4 从 clean working tree 开始。

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

开工先计算文件 SHA-256，必须精确等于：

`F0E5D806A3E17A27284998AAD52A683245A3C74EBDD15B23EFC13448B9D96582`

不一致：STOP，输出 `BLOCKED_M3_DATASET_SHA_MISMATCH`。

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

artifact 存在时禁止重新生成。
如果缺失：STOP，报告 `BLOCKED_M3_DATASET_MISSING`；不得在 M4 偷跑 M3 数据生成。

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
- `torch.use_deterministic_algorithms(True, warn_only=True)`

模型必须在设置这些 seed 后重新实例化；禁止从 M3 Student checkpoint warm-start。

Precision 决策只有一次，算法固定：
1. 使用 training seed `20260919`，两个架构分别从随机初始化做 **20 个 optimizer step** 的 BF16 smoke；data plan 使用该 seed 的前 20 个正式 step；smoke 结束后丢弃模型，不计入 primary；
2. BF16 smoke 每一步必须同时满足：loss finite、所有存在的 gradient finite、optimizer step 后所有 trainable parameter finite、无 CUDA/PyTorch exception；
3. 两个架构 20/20 steps 全满足 → M4 primary precision 固定为 `BF16 autocast`；
4. 只要任一架构任一步不满足，或当前 CUDA/PyTorch 不支持 BF16 → 两个架构统一使用 FP32；随后两个架构各做 5-step FP32 finite smoke；
5. 若 FP32 smoke 任一架构仍出现 non-finite/exception → STOP，输出 `M4_BLOCKED_NUMERIC_SMOKE`。

禁止 FP16，禁止 GradScaler，禁止一个架构 BF16、另一个 FP32。最终 precision 决策写入 summary。

# 10. 每个 primary run 必须记录

训练前必须在 **canonical、无 D4 augmentation** 的完整 train/validation split 上 eval：
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
- samples/s
- peak VRAM
- GPU utilization diagnostic：P0 固定执行一次 `nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits`；若 exit 0，则训练期间每 1 秒采样并报告 mean/p95 GPU utilization 与 peak memory.used；若 P0 命令非 0，则本轮所有 GPU-util 字段统一写 `N/A (nvidia-smi unavailable)`，不得换监控方案

训练结束固定使用 **epoch 30 最终模型**；禁止选择 validation 最好的历史 epoch。固定保存 `artifacts/m4/checkpoints/<architecture>_<seed>_final.pt`（model state + architecture + seed + precision + dataset SHA + plan SHA），计算整个 checkpoint 文件 SHA-256 并写入 summary。然后记录：
- final canonical train metrics
- final validation metrics
- D4 consistency diagnostic
- total training compute wall = 30 个 epoch training-only elapsed 之和
- 210 个 step 的 GPU step time：每 step 用一对 `torch.cuda.Event(enable_timing=True)` 包围 forward/backward/clip/optimizer，epoch 末 synchronize 后读取 milliseconds；报告 mean/median/p95
- Q head/backbone parameter updates confirmed：训练前 clone 对应参数；epoch 30 后必须至少一个 backbone tensor 与至少一个 q_head tensor `torch.equal == False`，否则 STOP `M4_BLOCKED_NO_PARAMETER_UPDATE`
- V head / Afterstate head 参数与训练前 clone 必须对每个 tensor `torch.equal == True`；任一变化 → STOP `M4_BLOCKED_VALUE_HEAD_UPDATED`

test split 只有全部 6 个 primary run 完成、协议不得再修改后才读取。

# 11. Test Teacher metrics

Teacher train/validation/test metrics 与 D4 diagnostic 一律 `model.eval()` + `torch.inference_mode()` + **FP32 inference（autocast disabled）**；对每个 final model 按 §10 完全相同 metric 公式、batch size=1024 记录：
- test CE
- legal best-action accuracy
- legal pairwise ranking accuracy
- raw best-action accuracy
- D4 legal-argmax consistency
- centered-logit D4 MAE

D4 diagnostic 公式固定与 M3 一致：canonical logits 减去每行 4-action mean；对 transform id 1..7 做 board transform、NN forward、inverse-transform Q 回原方向，再减每行 mean；报告 7 个变换上的 legal argmax consistency，以及所有 restored-centered 与 canonical-centered 元素绝对误差的全局 mean。

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
- 每个 game seed `G` 单独创建 `spawn_rng = numpy.random.Generator(numpy.random.PCG64(G))`；初始两个 tile 与以后所有 spawn 都只消费该局自己的 `spawn_rng`；
- 每个 game seed `G` 单独创建 `tie_rng = numpy.random.Generator(numpy.random.PCG64(G ^ 0x9E3779B9))`；它只用于 Q tie-break，不得用于 spawn；
- evaluator **必须**把当前所有 active boards 合并成一个 batch 做 NN forward，并用 `move_selected_batch` 批量 movement；随后仅 spawn 阶段按 game id 逐局消费对应独立 RNG；某局提前 terminal 不得影响其他局 RNG；
- score 只累加 merge reward；terminal 后停止该局。

正式 2000-game 前固定用 evaluation seeds `20261001..20261032` 做 correctness gate：
1. 同一个 model + 同一个 seed 连续运行两次，final score / moves / max_tile_exp 必须完全一致；
2. 对同一动作序列逐步将 `move_selected_batch` 与 M0 `move_without_spawn` 比较，afterstate/reward/moved 必须逐步完全一致；
3. 任一失败 → STOP，输出 `M4_BLOCKED_GAME_EVALUATOR_CORRECTNESS`。

固定 paired game seeds：

`20261001 + i, i = 0..1999`

即每个 trained model **2000 games**。
两个架构、三个 training seeds 都使用完全相同 2000 game seeds。game evaluation 的 6-model 执行顺序固定复用 §23 的 primary run 顺序，不得重排。

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

paired bootstrap 算法固定：
- delta vector = 同一 evaluation seed 的 `Transformer score - MLP score`；
- RNG = `numpy.random.Generator(numpy.random.PCG64(20261001))`；
- 恰好 10,000 次 resample；每次从 2000 个 paired index 中有放回抽取 2000 个 index并计算 mean delta；
- 95% CI = bootstrap mean delta 的 `np.quantile([0.025, 0.975])`（NumPy 默认 linear method）；
- 每个 training seed 分别执行一次并报告 paired mean delta / CI。

aggregate 固定：对每个 evaluation seed，先分别对三个 Transformer training-seed score 求算术平均、三个 MLP training-seed score 求算术平均，再形成 2000 个 aggregate paired delta；用同一算法但 RNG seed `20261002` 做 10,000 bootstrap。

strength 判定机械执行：
1. aggregate CI lower > 0 且 3 个 per-training-seed **mean delta** 至少 2 个 > 0 → `TRANSFORMER_STRENGTH_SIGNIFICANT`；
2. aggregate CI upper < 0 且 3 个 per-training-seed **mean delta** 至少 2 个 < 0 → `MLP_STRENGTH_SIGNIFICANT`；
3. 其他所有情况 → `ARCHITECTURE_STRENGTH_NOT_STATISTICALLY_RESOLVED`。

# 14. 无法确认棋力差异时的固定 tie-break

严格按总计划“更简单、更快”处理。

先比较 **8192-env closed-loop decisions/s**，协议固定：
- 使用 frozen M2 `scalar+row-lut` rollout backend / `M2RolloutBatchEnv`；
- PyTorch eager；precision 使用 §9 的共同 primary precision；
- env_count = 8192；
- 每个架构先做 100 个 closed-loop batch steps warmup，不计时；
- 然后恰好做 5 个正式重复；每个重复重新构造 env，seed 固定为 `20261200 + repeat_index`，warmup 20 steps，然后计时 500 个 closed-loop batch steps；
- 每个 timed repeat 前后执行 `torch.cuda.synchronize()`；
- decisions/s = `8192 * 500 / elapsed_seconds`；
- §14 使用 5 个 decisions/s 的 median。

令 `fast = max(T_median, M_median)`，`slow = min(T_median, M_median)`，`speed_ratio = fast / slow`。

- 若 `speed_ratio >= 1.10`：选择 median throughput 更高者；
- 若 `speed_ratio < 1.10`：选择 trainable parameter count 更少者，即 Transformer2048（4,750,342 < 5,264,710）。

不存在第三种 tie-break，不得综合主观“复杂度”评分。

该 tie-break 只在实际棋力未统计确认时使用。

# 15. 性能 benchmark

性能 benchmark 每个架构固定使用 **training seed 20260919 的 epoch-30 final checkpoint**；权重数值不参与选择 checkpoint。执行路径固定 **PyTorch eager + §9 共同 primary precision**；M4 禁止 `torch.compile` benchmark，避免引入第二套执行路径。每次只把当前被测模型保留在 GPU，上一个模型删除后执行 `gc.collect(); torch.cuda.empty_cache()`。

Model-only inference 固定 batch：`1, 256, 1024, 2048, 4096, 8192`。每个 batch size 的 5 个 repeat 测量顺序固定：repeat 0/2/4 为 Transformer→ResidualMLP，repeat 1/3 为 ResidualMLP→Transformer；模型加载/删除不计入 timed 区间。每个 architecture/batch size：
- 固定 synthetic input RNG = `numpy.random.Generator(numpy.random.PCG64(20261300 + batch_size))`；生成 shape `(batch_size,16)`、dtype `uint8`、每格均匀整数 `[0,21]`（即 `integers(0,22)`）并一次性传到 GPU；timed loop 复用同一 tensor，不包含 H2D；
- 每个 architecture/batch size 恰好做 5 个 repeat；每个 repeat 开始前 `torch.cuda.reset_peak_memory_stats()`，50 iterations warmup，再 200 timed iterations；
- timed 区间前后 `torch.cuda.synchronize()`；
- 每 repeat 报告 states/s、mean latency、peak VRAM；最终报告 5 个 states/s 的 median 与 5 个 latency 的 median。

Closed-loop 只测固定两个规模：
- 2048 envs
- 8192 envs

每个 env size 的 5 个 repeat 同样固定交替架构顺序：repeat 0/2/4 Transformer→ResidualMLP，repeat 1/3 ResidualMLP→Transformer。每个规模均按 §14 的 `100-step global warmup + 5 repeats × (20 warmup + 500 timed steps)` 协议执行。**M4 不测 16384/32768**，不得自行扩展 benchmark 矩阵。

每个 closed-loop point 固定记录：
- 5 个 repeat decisions/s + median；
- wall seconds；
- peak VRAM = `torch.cuda.max_memory_allocated()`；
- GPU utilization = §10 固定 `nvidia-smi` 采样规则；
- normalized CPU process utilization = `100 * (process_time_delta / wall_delta) / os.cpu_count()`；
- pipeline ratio = 对应 batch/env size 的 model-only states/s median ÷ closed-loop decisions/s median。

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
- 两架构均重新 random init，training seed 固定 `20260919`；
- optimizer/lr/batch/precision 与 primary 完全相同；
- data-plan RNG 重新用 `PCG64(20260919)`，按 §7 算法连续生成 epoch，不限制 30 epochs；
- secondary budget 也只计算 training steps：每个 step 前 synchronize 并读取累计 training-only wall；若累计 `>= common_budget` 则停止，否则执行一个完整 step并把该 step 的 synchronized wall 加入累计；允许最终累计超预算最多一个完整 step；validation/checkpoint/logging 不计入 budget；
- 不做 early stopping；
- 记录完整 steps、samples、实际 wall、最终 validation Teacher metrics；
- 使用固定 2000 game seeds 再评测一次实际分数。

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
- 任一 required script exception/进程非 0 且一次原样重跑仍复现 → STOP，`M4_BLOCKED_RUNTIME_ERROR`；
- 指标有限但很差 → 仍必须完成全部 6 runs、2000-game eval、统计与 selection；不得救场调参。

“原样重跑”只允许相同代码、相同配置、相同 seed 重跑一次用于排除偶发进程/driver 中断，不得修改任何超参。

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
`schema_version` 固定为 `1`；禁止自行改顶层 key 名。

最低子字段固定：
- `dataset`: `path, sha256, checkpoint_sha256, states, games, split_counts`；
- `models`: 两架构的 `parameter_count`；
- `fairness`: `epochs, batch_size, optimizer, lr, weight_decay, grad_clip, training_seeds, evaluation_seed_start, evaluation_games, tie_tolerance`；
- `precision`: selected precision + 两架构 smoke pass/fail 细节；
- `plans`: 3 个 seed 的 `plan_sha256`；
- `primary_training`: 6 个 run 的 final checkpoint path/SHA、wall、step-time、samples/s；
- `teacher_metrics`: 6 个 run 的 canonical train/validation/test metrics；
- `game_evaluation`: 6 个 run 的 2000-game summary + raw NPZ path；
- `paired_statistics`: 3 个 per-seed delta/CI + aggregate delta/CI；
- `d4`: 6 个 final model 的 consistency/MAE；
- `performance`: model-only、closed-loop、training；
- `equal_wall_clock_secondary`: `status` + 若触发则两架构结果；
- `selection`: `strength_conclusion, selected_architecture, selection_rule`；
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

第一动作必须按顺序报告：
- HEAD / origin/main（必须相等）
- working tree（必须 clean）
- root layout check（不得有 milestone report/result JSON）
- four frozen tag resolutions
- M3 dataset file SHA-256 + shape/splits/checkpoint-SHA metadata
- official Python version
- torch / CUDA / GPU name
- `nvidia-smi` availability status
- exact model parameter counts

然后用官方 Python：
`D:\sd-webui-forge-aki-v1.0\python\python.exe -m pytest -q`

P0 baseline 必须精确为：
- 525 passed
- 0 failed
- 0 skipped
- 0 xfailed

失败则 STOP，不得用 M4 修改 frozen source 来救。

# 22. 长任务可观测性与断线恢复

固定 artifact：
- `artifacts/m4/progress/precision.json`
- `artifacts/m4/progress/train_<architecture>_<seed>.json`
- `artifacts/m4/checkpoints/<architecture>_<seed>_latest.pt`
- `artifacts/m4/games/<architecture>_<seed>.npz`
- `artifacts/m4/progress/performance.json`

训练：每个 epoch 结束必须原子更新 progress JSON，并覆盖 `latest.pt`（包含 model state、optimizer state、next_epoch、seed、precision、data-plan SHA）；同时输出 architecture / seed / epoch / train CE / validation CE / val accuracy / elapsed / samples/s。

2000-game evaluation：每完成恰好 100 个 game seeds 就更新一次 NPZ；NPZ key 固定为 `game_seed, final_score, max_tile_exp, moves`，数组长度等于当前已完成 games 且按 game_seed 升序；因为每局 RNG 独立，恢复时只运行尚未存在结果的 seed。每 100 games 输出 architecture / training seed / completed / elapsed / games/s / running mean score。

性能 benchmark：每完成一个 model-only batch size 或一个 closed-loop env size 的一个 repeat，立即更新 `performance.json`。

恢复规则机械执行：
1. 若 progress/checkpoint 的 architecture、seed、precision、dataset SHA、plan SHA 与当前施工单不完全一致 → STOP，`M4_BLOCKED_RESUME_METADATA_MISMATCH`；
2. 一致 → 从 `next_epoch` / 缺失 game seed / 缺失 performance repeat 继续；
3. 已完成 evidence 禁止重跑，除非对应 artifact 读取失败；
4. 不得因工具断线从 P0 无条件重跑。

# 23. 执行顺序锁死

P0 frozen/preflight/full pytest
→ P1 只创建 §19 的 4 个 code/test files，完成 helper/harness/performance harness/8 CPU tests；`run_m4_architecture_compare.py` 的 CLI 必须支持 `--phase {primary,finalize,report}` + `--resume`，`benchmark_m4_architecture_performance.py` 的 CLI 必须支持 `--resume`
→ P1.1 `D:\sd-webui-forge-aki-v1.0\python\python.exe -m pytest tests/test_m4_architecture_compare.py -q`
→ P2～P6 固定执行：`D:\sd-webui-forge-aki-v1.0\python\python.exe benchmarks/run_m4_architecture_compare.py --phase primary --resume`
→ `--phase primary` 内部固定顺序：precision smoke → 生成/验证 3 个 plan → 6 primary runs → final train/validation/test Teacher metrics → 6 × 2000 games → paired statistics
→ 6 primary run 顺序固定为：Transformer/20260919 → ResidualMLP/20260919 → ResidualMLP/20260920 → Transformer/20260920 → Transformer/20260921 → ResidualMLP/20260921（中间 seed 反转先后，禁止改顺序）
→ P7 固定执行：`D:\sd-webui-forge-aki-v1.0\python\python.exe benchmarks/benchmark_m4_architecture_performance.py --resume`
→ P8～P9 固定执行：`D:\sd-webui-forge-aki-v1.0\python\python.exe benchmarks/run_m4_architecture_compare.py --phase finalize --resume`
→ `--phase finalize` 只做 conditional equal-wall secondary（若 §16 机械触发）与 selection，并把结果写入 `artifacts/m4/progress/finalize.json`；不得先写 tracked report
→ P10 固定执行：`D:\sd-webui-forge-aki-v1.0\python\python.exe -m pytest -q --junitxml=artifacts/m4/pytest.xml`
→ P10 成功后固定执行：`D:\sd-webui-forge-aki-v1.0\python\python.exe benchmarks/run_m4_architecture_compare.py --phase report --resume`
→ `--phase report` 必须解析 `artifacts/m4/pytest.xml`，确认 0 failures / 0 errors / 0 skipped 且 tests >=533，然后写 `reports/m4/m4_architecture_compare.json` + candidate version of `reports/m4/M4_REPORT.md`
→ candidate implementation commit/push
→ candidate GitHub Actions green
→ mandatory report-only closeout：只允许修改 `reports/m4/M4_REPORT.md`，回写 candidate SHA / candidate CI run ID+conclusion
→ report-only closeout commit/push
→ closeout GitHub Actions green
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
- `TRANSFORMER_STRENGTH_SIGNIFICANT` → `M4_SELECT_TRANSFORMER`；
- `MLP_STRENGTH_SIGNIFICANT` → `M4_SELECT_RESIDUAL_MLP`；
- `ARCHITECTURE_STRENGTH_NOT_STATISTICALLY_RESOLVED` → 严格执行 §14，得到 Transformer 或 ResidualMLP；
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
- final full pytest：0 failed / 0 skipped / 0 xfailed，passed count >= 533
- candidate CI green
- mandatory report-only closeout CI green
- artifacts organized / temp files cleaned：6 个 final checkpoint、3 个 plan、6 个 raw game NPZ、performance evidence、pytest.xml 保留；所有 `<architecture>_<seed>_latest.pt` 与仅用于运行中的临时 scratch/progress lock 文件删除

# 26. STOP / Git

M4 必须创建普通 implementation candidate commit并 push。固定命令语义：

```text
git add -- src/game2048/m4_compare.py benchmarks/run_m4_architecture_compare.py benchmarks/benchmark_m4_architecture_performance.py tests/test_m4_architecture_compare.py reports/m4/m4_architecture_compare.json reports/m4/M4_REPORT.md
git diff --cached --check
git commit -m "m4: compare Transformer and ResidualMLP"
git push origin main
```

该 commit 是 M4 candidate implementation 的权威候选 SHA；commit 后立即记录 `git rev-parse HEAD`。

candidate CI 固定用 candidate SHA 查询对应 GitHub Actions `CI` run，并等待完成。

- CI success → 继续；
- CI 因 checkout/setup/dependency/runner infrastructure 失败 → 只允许原样 rerun 一次；第二次仍非 success → STOP `M4_BLOCKED_CI_INFRA`；
- CI 在 M2 build / M3 build / pytest 阶段失败 → 不得改协议；只允许在 §19 的 4 个 code/test files 内修 correctness bug，重新跑本地 full pytest，创建**新的** candidate implementation commit并重新走 candidate CI；旧 candidate 作废，以最后 green 的 implementation SHA 为准。

candidate CI green 后**必须**创建恰好一个 report-only closeout descendant：只修改 `reports/m4/M4_REPORT.md`，写入最终 candidate SHA、candidate CI run ID/conclusion、closeout parent。固定：

```text
git add -- reports/m4/M4_REPORT.md
git diff --cached --check
git commit -m "docs: close M4 candidate report"
git push origin main
```

closeout commit push 后等待它自己的 CI。closeout CI 若 infrastructure 失败只原样 rerun 一次；若 build/pytest 失败则 STOP `M4_BLOCKED_CLOSEOUT_CI`（docs-only closeout 不允许顺手改 implementation）。只有 closeout CI success 后才允许输出 `M4 CANDIDATE COMPLETE`。

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

完成 candidate 后输出：
`M4 CANDIDATE COMPLETE`

并明确：
- selected architecture
- strength conclusion
- selection依据（score significance 或 speed/size tie-break）
- candidate implementation commit
- mandatory report-only closeout commit
- candidate CI run
- closeout CI run
- no M4 tag
- no M5

然后 STOP。