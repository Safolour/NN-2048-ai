# M5 Teacher 预训练 — 严格施工提示词

> 项目：Safolour/NN-2048-ai
> 正式工作区：D:\CodexTasks\NN-2048-ai
> 当前阶段：M5 Teacher Pretraining
> Work-order version: `M5_WO_CLOSURE_V1`
> 本文件是 M5 的唯一详细施工单；authoritative master plan 始终优先。

# 0. 权威输入

开工前必须完整读取：
1. `prompts/2048_AI_正式执行计划_M0_M1冻结_CI增强版_2026-09-18.md`
2. `reports/m4/M4_AUDIT_REPORT.md`
3. `reports/m4/M4_REPORT.md`
4. `reports/m5/M5_TEACHER_PROMOTION_AUDIT.md`
5. 本文件

如有真实冲突：STOP，输出 `M5_BLOCKED_SPEC_CONFLICT`，不得自行裁决。

# 1. Frozen provenance / start state

必须验证 peeled tags：
- `m0-reference-pass^{}` = `3f2def1d95f56eff776e671143188947bf64485b`
- `m1-fastenv-audited-pass^{}` = `e5486017a90eeec4fb9814de7880b3c413dc06dd`
- `m2-network-audited-pass^{}` = `6a9da5b1bf7c72207acd6e89bc667d439d4823a8`
- `m3-teacher-audited-pass^{}` = `7c95f9c5f543065b220fb5f9e9114521732cdc0b`
- `m4-architecture-audited-pass^{}` = `300d51a818fa55394ec56de7507bcade111a06d1`

M4 audit docs closeout ancestor：
`93a72cc1d0ea9c854b69039733da2aba29485ea3`

每次 `run M5` 先执行：
```text
git fetch origin
git rev-parse HEAD
git rev-parse origin/main
git status --porcelain=v1 -uall
```

并计算当前本 prompt SHA-256，记为 `prompt_sha256`。

FRESH：仅当 `artifacts/m5/progress/session.json` 不存在。
Git/frozen 硬门：HEAD==origin/main、worktree clean、上述五个 tag 精确匹配、M4 audit closeout 是 HEAD ancestor。
Git/frozen 硬门任一失败：STOP `M5_BLOCKED_START_STATE`；不得 pull/rebase/reset/restore/delete 猜测修复。

正式 runtime 还必须精确验证：
- Python: `D:\sd-webui-forge-aki-v1.0\python\python.exe`
- Python 3.11.9
- PyTorch `2.9.1+cu130`
- CUDA runtime `13.0`
- GPU name = `NVIDIA GeForce RTX 5060 Laptop GPU`
- `nvidia-smi` 可执行

runtime 任一不符：STOP `M5_BLOCKED_RUNTIME_ENVIRONMENT`；不得临时换 Python/CUDA 环境。

FRESH P0 成功后原子创建 `artifacts/m5/progress/session.json`，至少保存：
`schema_version=1, work_order_version=M5_WO_CLOSURE_V1, base_head, prompt_sha256, teacher_sha256, m4_primary_sha256, state=P0_PASSED`。

RESUME：session 存在时禁止重新 FRESH。必须验证 work-order version、prompt SHA、Teacher SHA、M4 primary SHA、五个 frozen tags；不一致 STOP `M5_BLOCKED_RESUME_METADATA_MISMATCH`。

# 2. Frozen source protection

M5 禁止修改 M0/M1/M2/M3/M4 frozen implementation。
开工必须确认以下 diff gate 为 0：
- M2 frozen model/env/backend paths vs `m2-network-audited-pass^{}`
- M3 tuple/search/data paths vs `m3-teacher-audited-pass^{}`
- M4 architecture compare implementation/test paths vs `m4-architecture-audited-pass^{}`

M5 只能新增自己的 harness/helper/test/report；不得为了 M5 方便回写 frozen implementation。

# 3. M5 唯一目标

M5 只回答：在 M4 已选定的 **ResidualMLP2048** 架构上，使用冻结 N-tuple + depth-3 Expectimax Teacher，扩大高质量 Teacher 数据量进行 supervised pretraining，能否显著提高 pure-NN 实际游戏强度，以及数据量继续扩大是否仍有收益。

M5 不做：
- Transformer 重赛；
- 2M/10M/15M 模型规模 sweep；
- M6 Student State Correction；
- M7 Self-play / Replay / Double-Q / Target；
- Q+V/Q+A/Q+V+A head 实验；
- multi-step / TD(lambda)；
- Search depth sweep；
- absolute Q/V/A regression；
- M3 affine calibration reuse；
- Teacher checkpoint 自动追随 latest；
- M6/M7 任何实现。

注意：这些“不做”只限制 M5 控制变量，不取消 master plan 后续 mandatory 2M/5M/10M/15M、Head、multi-step、Teacher、Replay、Target 等强度实验。

# 4. Architecture 固定

M5 唯一网络：`ResidualMLP2048()`。
必须复用 frozen M2 定义，参数量必须精确 `5,264,710`；不符 STOP `M5_BLOCKED_MODEL_DRIFT`。

M5 不允许改 hidden width/depth/embedding/head shape/dropout/norm/residual topology。

# 5. Teacher promotion 已独立完成

`reports/m5/M5_TEACHER_PROMOTION_AUDIT.md` 是本阶段前置独立 promotion gate。
结论固定：`NO PROMOTION`。

M5 Teacher checkpoint 固定：
`teacher_checkpoints/m3/ordinary_td_comparator_ep4800000_7192719323a0.bin`
SHA-256：
`7192719323a073ba2b6b19b62cb7d46ef4aa90ecc8c4ae6baf27ad0c51566a84`

禁止在 M5 使用：
- COMPARATOR 6.4M / SHA `5EDC31BEB45806F3E73DA96FB01BCA1B6CF1D94F5331173683B1F715240792A9`；
- FORMAL 8.4M / SHA `846B76FE64A0B62D64D5BC295EC4BB997DFB5A958B05BD3FF6251FC896D620DC`；
- 更早已被本次 refreshed audit supersede 的 5.1M / 7.0M candidate；
- 任何 `latest.bin`；
- promotion audit 之后出现的新 checkpoint。

FORMAL 8.4M 的 p10 / reach8192 / reach16384 相对 retained baseline 明显改善，但 paired mean-score 显著下降；该 trade-off 只作为后续 mandatory Teacher/high-tile experiment evidence 保留。M5 不得把两个 Teacher 混合标注，也不得因为 tail 指标好而绕过 §31.4 mean-score promotion gate。

若本地 frozen Teacher 缺失或 SHA 不符：STOP `M5_BLOCKED_TEACHER_CHECKPOINT`。

# 6. Teacher value semantics

M5 继承 M3 audited semantics：
- `V_tuple`: `RAW_TUPLE_HEURISTIC`
- formal leaf: `FORMAL_STATE_TUPLE_HEURISTIC`
- depth-3 root action value: `SEARCH_VALUE_RAW_LEAF`

M3 calibration validation 仅 1 game / 8 states，affine mapping 不得作为绝对标尺。

因此 M5 supervision 固定：
- target = legal action 中 `argmax(teacher_value)`；
- training loss = 4-class policy cross entropy；
- illegal action 不参与 target；
- pairwise ranking 仅作 metric；
- 禁止 raw absolute Q MSE；
- 禁止 V MSE；
- 禁止 Afterstate MSE；
- 禁止复用 M3 affine slope/intercept。

# 7. Search 定义固定

Teacher label search 固定：
- decision_depth = 3；
- chance nodes 不计 depth；
- use_cache = True；
- formal-state leaf = frozen tuple afterstate greedy-1ply adapter；
- action value = immediate reward + downstream Expectimax value；
- tie = `np.nanargmax` 最低 action id；
- output semantics = `SEARCH_VALUE_RAW_LEAF`。

不得把 native greedy_1ply 当 Search depth 上限，也不得偷偷改 depth=5/7。

# 8. Search re-profile gate

M5 在生成大规模 labels 前必须重新 profile 当前 frozen Teacher + current audited Search path。

固定输入：
- `artifacts/m3/semantic_profile_states.npz`
- SHA-256 `9AFD3A503819C8BD777C7F48D3CCDCA2B1C2F02A7D28DE7DF8997735EE68E512`
- 恰好 256 roots
- reference values: `artifacts/m3/search_ab_p6_cpp_values.npz`
- SHA-256 `8E09F0D7774BCEC55C496A29550F2362FB83DEE142643F69472F12AE14963C51`

canonical constants：`profile_repeat_count = 3`, `roots_per_repeat = 256`。
执行 3 个完整 repeats；每 repeat 对 256 roots 全部重新构造 Search。
必须记录：root/s、wall、player/chance/leaf/move counts、cache hit rate、hash/move/chance/tuple/orchestration seconds、per-root p50/p95/max。

correctness gate：三个 repeats 的 legal mask、action values、best action、node/cache counts 必须与 M3 P6 reference 精确一致；任一失败 STOP `M5_BLOCKED_SEARCH_CORRECTNESS`。

performance gate：3-repeat median root/s 必须 >= `5.688376766`（M3 audited 6.320418629 × 0.90）。否则 STOP `M5_BLOCKED_SEARCH_PERFORMANCE_REGRESSION`。

将三个 repeats 的 `recursive_python_orchestration_seconds` 求和后除以三个 repeats 的 `total_seconds` 总和，得到唯一 orchestration_share。若 `orchestration_share > 0.50`，STOP `M5_BLOCKED_SEARCH_PERFORMANCE_UNBLOCK_REQUIRED`；M5 execution Agent 不得现场重写 Search，另开独立 unblock。

M3 已经完成 C++ tuple evaluator + frozen M2 C++ movement 的正式 Search performance unblock。M5 的本节职责是确认该已审计优化没有回退；只要 correctness、90% throughput gate、50% orchestration gate 全过，就视为已满足 master plan“Search 为主要成本时优先处理真实热点”的前置要求，不为优化而重写系统。

profile 完成后报告：
- projected 32,768 label wall
- projected 131,072 label wall
- projected 524,288 label wall

这些 projection 只是容量/进度信息，不是 M5 强度上限。

# 9. 数据规模阶梯固定

每个 complete source game 固定抽取 128 个 canonical formal states，采样算法与 M3 相同：
`np.rint(np.linspace(0, moves-1, 128)).astype(np.int64)`。
必须保证 128 个 position 唯一；不足 128 decision states 的 game 属于 correctness error，STOP。

固定 source-policy：同一 frozen Teacher checkpoint 的 native `greedy_1ply`。
每局独立 spawn RNG 固定为 `np.random.Generator(np.random.PCG64(game_seed))`；初始 board=`np.zeros(16,dtype=np.uint8)`，随后严格调用两次 frozen M0 `spawn_random(board,rng).state` 生成起始两 tile；以后该局所有 spawn 只继续消费自己的同一个 rng。greedy action 固定为 legal `np.nanargmax(reward + V_tuple(afterstate))`，tie 取最低 action id。某局提前 terminal 不得改变其他 game RNG stream。

固定 split / seed：
- train: game_id 0..4095；game_seed = `20265001 + game_id`
- validation: game_id 100000..100063；game_seed = `20275001 + local_index`
- test: game_id 200000..200063；game_seed = `20276001 + local_index`

固定 training rungs：
- `32k`: first 256 train games = 32,768 states
- `131k`: first 1,024 train games = 131,072 states
- `524k`: first 4,096 train games = 524,288 states

固定 validation：64 games = 8,192 states。
固定 test：64 games = 8,192 states。

32k 和 131k **mandatory**。
524k 仅按 §18 mechanical scale gate 触发。
524k 不是项目最终上限；只是 M5 本 milestone 最大 rung。

# 10. Dataset shard contract

固定 manifest 路径：
- `artifacts/m5/datasets/source_train_manifest.json`
- `artifacts/m5/datasets/source_validation_manifest.json`
- `artifacts/m5/datasets/source_test_manifest.json`（仅 FINAL_SCALE_SELECTED 后生成）
- `artifacts/m5/datasets/label_manifest.json`

Search label shard 固定 2,048 states = 16 whole games。
禁止一个 game 跨 shard。

因此：
- 32k train = 16 shards
- 131k train = 64 shards
- 524k train = 256 shards
- validation = 4 shards
- test = 4 shards

每个 completed shard 固定保存：
`artifacts/m5/datasets/<split>/shard_<index:04d>.npz`
并立即记录 SHA-256 到 manifest。

每 shard 至少包含：
- `state [N,16] uint8`
- `teacher_value [N,4] float64`
- `reward [N,4] int32`
- `afterstate [N,4,16] uint8`
- `legal_mask [N,4] bool`
- `game_id [N] int64`
- `game_seed [N] int64`
- `step_index [N] int32`
- `current_score [N] int64`
- `max_tile_exp [N] uint8`
- `teacher_version`
- `checkpoint_sha256`
- `decision_depth`
- `value_semantics`
- `data_source`

illegal action：`teacher_value=NaN`、reward/afterstate 可用零 sentinel，但 `legal_mask=False`，不得参加 loss/argmax。

所有 dataset 原始 board 永远 canonical_unaugmented=True；D4 只在 training batch 动态执行。

# 11. Test isolation

M5 在 scale/champion 选择完成之前：
- 可以知道 test split 的 seed contract；
- 禁止生成 test Search labels；
- 禁止把 test board 输入模型；
- 禁止计算 test metric；
- 禁止根据 test 调 scale/seed/hyperparameter。

只有 `FINAL_SCALE_SELECTED` 后才允许生成 4 个 test shards，并且只对最终 champion checkpoint计算一次 Teacher test metrics。

# 12. Label generation / resume

source trajectory 与 Search label 分离。

source trajectory generation 必须保存 complete-game summary；每 game 记录 seed/moves/final score/max tile/sampled positions。

Search labels：
- worker count 固定 8；
- workers 只 mmap/read frozen Teacher；
- 每 root 独立 `ExpectimaxTeacher(decision_depth=3,use_cache=True)`；
- 每完成 64 roots 输出 progress / elapsed / roots/s / ETA；
- 只有 2,048-root shard 全部完成后才 atomic rename 成正式 NPZ 并写 SHA；
- 中断在 shard 内：丢弃 partial shard，恢复时重做该 shard；
- completed+SHA-valid shard 禁止重跑。

若 completed shard 缺失、schema/shape 不符或 SHA 不符：STOP `M5_BLOCKED_DATASET_ARTIFACT_CORRUPT`，不得自行删除后重算。

# 13. Training protocol 固定

每个 rung 都从随机初始化重新训练 3 个 ResidualMLP2048；禁止从上一 rung warm-start，避免把“更多数据”和“额外训练历史”混在一起。

training seeds：
`20262101, 20262102, 20262103`

统一 protocol：
- epochs = 30
- batch_size = 1024
- drop_last = False
- optimizer = `torch.optim.AdamW`
- lr = 3e-4
- weight_decay = 1e-4
- betas = (0.9,0.999)
- eps = 1e-8
- amsgrad=False / foreach=False / fused=False / capturable=False
- gradient clip global norm = 1.0
- no scheduler
- no early stopping
- no label smoothing
- no architecture-specific tuning

每 step：
`zero_grad(set_to_none=True) -> forward -> CE(mean) -> backward -> clip_grad_norm_ -> optimizer.step()`。

canonical step constants：
- 32k: `steps_per_epoch=32`, `optimizer_steps_per_run=960`
- 131k: `steps_per_epoch=128`, `optimizer_steps_per_run=3840`
- 524k: `steps_per_epoch=512`, `optimizer_steps_per_run=15360`

等价计算：
- 32k: 32 steps/epoch ×30 = 960
- 131k: 128 ×30 = 3,840
- 524k: 512 ×30 = 15,360

# 14. Deterministic D4 training plan

不保存巨型 permutation 文件；使用 stateless per-epoch plan。

scale offset：
- 32k = 0
- 131k = 100000
- 524k = 200000

对 training seed S、epoch e：
`rng = np.random.Generator(np.random.PCG64(S + scale_offset + e))`

顺序固定：
1. `rng.permutation(train_row_indices)`
2. `rng.integers(0,8,size=N,dtype=np.uint8)`
3. 按 permutation 顺序将 transform id 一一应用 board 与 teacher-best-action label

board 使用 frozen `transform_board_batch`；action 使用 frozen `transform_action_batch`。
validation/test 永远 canonical，无 augmentation。

每 rung/seed 的 `plan_id` 固定为 ASCII：
`M5_PLAN_V1:<scale>:<seed>:PCG64_STATELESS_EPOCH`
并记录其 SHA-256。

checkpoint 只在完整 epoch 后保存；中断在 epoch 内重做该未完成 epoch。

# 15. Precision decision

M5 precision 只决定一次，之后所有 rungs/seeds 相同。

使用 32k rung、training seed 20262101 的前 20 正式 steps做 BF16 smoke。
若 `torch.cuda.is_bf16_supported()==False` 或任一步 loss/gradient/parameter non-finite / exception：统一 fallback FP32，并从全新模型做前 5 正式 steps FP32 smoke。
FP32 smoke 仍失败：STOP `M5_BLOCKED_NUMERIC_SMOKE`。

BF16 PASS：全部 training 使用 BF16 autocast。
禁止 FP16 / GradScaler。
Teacher metrics / D4 / gameplay 一律 FP32 inference、autocast disabled。

# 16. Teacher metrics 固定

每个 run 在 canonical validation 8,192 states 上记录：
- CE
- raw best-action accuracy
- legal best-action accuracy
- legal pairwise ranking accuracy
- ranking pair count
- D4 legal-argmax consistency
- centered-logit D4 MAE

metric 公式固定复用 M4 audited 定义；eval batch size=1024。

validation random-legal baseline 必须按每个 state 的 legal action 数精确计算：对每 state baseline=`1 / legal_action_count`，split baseline 为这些值的算术平均；不允许写死 25%。high-tile stratum 同样在该 stratum 内重新计算。

高位 validation strata 固定按 current board max tile exponent：
- >=11 (2048+)
- >=12 (4096+)
- >=13 (8192+)

每 stratum 报 samples、random-legal baseline、legal best-action accuracy、pairwise ranking。
若某 stratum samples >=128，则最终 selected rung 的三 seed median legal accuracy 必须 >= 该 stratum random-legal baseline + 0.05；否则 STOP `M5_BLOCKED_HIGH_TILE_TEACHER_FIT`。
样本 <128 的 stratum 仅 diagnostic，不做 gate。

# 17. Pure-NN development gameplay

M5 development gameplay 必须复用 M4 audited pure-NN evaluator semantics：
- single ResidualMLP forward
- no Teacher / Search / tuple / D4 ensemble / exploration
- legal mask
- FP32 inference
- tie tolerance = 1e-7
- spawn RNG 与 tie RNG 独立
- fast movement 必须保持 M0 exact semantics

固定 dev game seeds：**复用 M4 的 2,000 seeds `20261001..20263000`**。
这样 M5 可以和 M4 frozen ResidualMLP baseline 做严格 paired comparison。

M4 baseline evidence 固定：
`artifacts/m4/progress/primary.json`
SHA-256：
`2196DCADEBE1661E8BBF5AADA6B7C1F9D1F199C6944FF70F0CD1C5C9B5B1BBD5`

M4 baseline 的 per-game aggregate 固定定义为：对同一个 game seed，将 M4 三个 ResidualMLP2048 training-seed raw score 求算术平均；得到 2,000 个 baseline aggregate scores。禁止只取 M4 最好 seed。

M4 ResidualMLP 三 seed mean score：
- 20260919: 2164.192
- 20260920: 2253.604
- 20260921: 1963.802
- architecture aggregate mean = 2127.1993333333335

P0 必须验证 M4 primary manifest、三个 ResidualMLP raw game NPZ 与 manifest SHA；缺失或不一致 STOP `M5_BLOCKED_M4_BASELINE_EVIDENCE`。

每个 M5 rung/seed 必须在相同 2,000 dev seeds 上保存 raw：
`artifacts/m5/games/<scale>/<seed>.npz`
key 固定：`game_seed, final_score, max_tile_exp, moves`。

# 18. Paired statistics / scale gate

对每个 rung，先对同一 dev game seed 的三个 training-seed score 求算术平均，得到 2,000 个 architecture-aggregate scores。

paired bootstrap：
- 10,000 resamples
- 每 replicate 有放回抽 2,000 paired indices
- quantile [0.025,0.975], NumPy default linear

固定 seeds：
- M4 baseline vs 32k: 20265201
- 32k vs 131k: 20265202
- selected pre-524k vs 524k: 20265203
- final selected rung vs M4 baseline: 20265204

32k 与 131k 都 mandatory，不由 gate 跳过。

本节所有 `median_val_*` 固定定义为：同一 rung 三个 training seeds 对应 validation metric 的数值 median。

131k -> 524k trigger 固定：

`score_positive_signal = (mean_delta_131k_minus_32k > 0) AND (CI95_upper > 0)`

`teacher_positive_signal = ANY(`
- `median_val_CE_131k <= 0.99 * median_val_CE_32k`
- `median_val_legal_acc_131k >= median_val_legal_acc_32k + 0.005`
- `median_val_pairwise_131k >= median_val_pairwise_32k + 0.005`
`)`

若 `score_positive_signal OR teacher_positive_signal`：必须生成/训练 524k。
否则：524k `NOT_TRIGGERED`，禁止为了“看看”继续烧 Search。

final scale selection 机械执行：
1. 先比较 32k vs 131k：
   - CI lower >0 -> 131k 赢；
   - CI upper <0 -> 32k 赢；
   - CI 跨 0 -> 若 131k teacher_positive_signal=True 则 131k，否则 32k。
2. 若 524k triggered，再把 524k 与上一步 winner 比：
   - CI lower >0 -> 524k 赢；
   - CI upper <0 -> 保留 previous winner；
   - CI 跨 0 -> 若 524k median validation CE 至少再降 1% 或 legal/pairwise 任一再升 0.5pp，则 524k，否则保留 previous winner。

不得按“数据越大越高级”自动选最大 rung。

# 19. M5 strength pass gate

最终 selected rung 的 3-seed aggregate 必须相对 M4 ResidualMLP aggregate 在同一 2,000 dev seeds 上满足：

`paired mean-score CI95 lower bound > 0`

否则 M5 不能宣布 Teacher pretraining strength PASS，STOP：
`M5_BLOCKED_PRETRAIN_GAIN_NOT_CONFIRMED`。

同时 selected rung validation 学习 gate：
- median legal best-action accuracy >= random-legal baseline + 0.10
- median pairwise ranking >= 0.65
- median CE finite

任一失败：STOP `M5_BLOCKED_TEACHER_LEARNING`。

D4 consistency/MAE 必须 finite并完整报告；M5 不设置未经证据支持的 D4 percentage threshold。

# 20. Final champion seed

scale 选定后，在该 scale 三个 training seeds 中，用 2,000 dev-game mean score 选择唯一 champion：
- mean score 最大者胜；
- 精确相等时 seed 数值更小者胜。

不得看 Teacher test 或 final gameplay test 后再换 champion。

记录 champion final checkpoint：
`artifacts/m5/checkpoints/<scale>/<seed>_final.pt`
及文件 SHA-256。

final checkpoint 必须含：architecture, scale, seed, precision, teacher_sha256, dataset manifest SHA, plan_id, epoch=30, model_state。

Q head + backbone 必须至少一个 tensor 相对 init 改变；value_head/afterstate_head 必须逐 tensor bit-identical to init，否则分别 STOP `M5_BLOCKED_NO_PARAMETER_UPDATE` / `M5_BLOCKED_VALUE_HEAD_UPDATED`。

# 21. Teacher test

只在 champion 固定后：
1. 生成 64 test source games；
2. 完成 4 个 depth-3 test label shards；
3. champion 仅一次计算完整 8,192-state Teacher test metrics；
4. 报 overall + high-tile strata；
5. test 不再影响 scale / seed / hyperparameter。

# 22. Final gameplay test

champion 固定后做一次 independent final gameplay：
- seeds = `20277001..20287000`
- exactly 10,000 games
- pure-NN evaluator same as §17
- shard = 100 games；partial shard 中断则整 shard 重跑

保存 raw：
`artifacts/m5/final/final_games_10000.npz`

报告：mean / median / p10 / p90 / mean moves / max-tile distribution / reach 2048/4096/8192/16384/32768/65536。

固定公式：
- p10/p90 = `np.quantile(scores,[0.1,0.9])` default linear
- reach threshold 分别为 `max_tile_exp >= 11/12/13/14/15/16`
- max-tile distribution = `{exp: count}` terminal frequency table

这 10,000 final games 是 final report，不允许再反向调 M5。

# 23. Training observability

每个 epoch 必须输出并原子持久化：
architecture / scale / seed / epoch / train CE / validation CE / validation legal accuracy / elapsed training wall / samples/s / peak VRAM。

固定 progress：
`artifacts/m5/progress/train_<scale>_<seed>.json`

checkpoint：
- running: `artifacts/m5/checkpoints/<scale>/<seed>_latest.pt`
- final: `artifacts/m5/checkpoints/<scale>/<seed>_final.pt`

每个完整 epoch 后保存 latest；resume 从 `next_epoch`。
已完成 epoch 禁止重跑；中断 epoch 允许从该 epoch 开头重做。

# 24. Session states / evidence DAG

session states 固定：
`P0_PASSED, P1_READY, PROFILE_RUNNING, PROFILE_DONE, DATA_32K_RUNNING, DATA_32K_DONE, TRAIN_32K_RUNNING, TRAIN_32K_DONE, DATA_131K_RUNNING, DATA_131K_DONE, TRAIN_131K_RUNNING, TRAIN_131K_DONE, SCALE_GATE_DONE, DATA_524K_RUNNING, DATA_524K_DONE, TRAIN_524K_RUNNING, TRAIN_524K_DONE, FINAL_SCALE_SELECTED, TEST_RUNNING, FINAL_TEST_DONE, P10_RUNNING, P10_DONE, REPORT_WRITTEN, CANDIDATE_PUSHED, CLOSEOUT_REPORT_READY, CLOSEOUT_PUSHED, COMPLETE`。

固定 evidence DAG：
`session.json`
-> `search_profile.json`
-> `precision.json`
-> source/dataset shard manifests
-> per-run progress + final checkpoints
-> `development.json`
-> `scale_selection.json`
-> `teacher_test.json`
-> `final_gameplay.json`
-> `pytest.xml`
-> tracked summary/report

所有 JSON/NPZ 写入必须 temp + atomic rename。
manifest 宣称 complete 但文件缺失/schema/SHA 不符：STOP `M5_BLOCKED_RESUME_ARTIFACT_CORRUPT`。

# 25. RESUME 规则

pre-candidate RESUME（state <= REPORT_WRITTEN）：
- HEAD==origin/main==base_head；
- staged diff 必须为空；
- dirty/untracked 非 ignored 路径只能是 §27 允许的 M5 code/test/report paths 子集；
- frozen diff gates 仍为 0；
- 不重跑 P0 full pytest；
- 从 session state 的第一个未完成 phase 继续。

post-candidate RESUME：
- `CANDIDATE_PUSHED`: HEAD==origin/main==candidate_sha 且 worktree clean，只恢复 candidate CI；
- candidate CI success 后先写 `CLOSEOUT_REPORT_READY`；该 state 下 HEAD==origin/main==candidate_sha，dirty path 只能为空或精确 `reports/m5/M5_REPORT.md`；
- `CLOSEOUT_PUSHED`: HEAD==origin/main==closeout_sha 且 worktree clean，只恢复 closeout CI；
- `COMPLETE`: 不得重跑实验，直接报告已有 complete evidence并 STOP。

若 session 与 Git 恰好断在 candidate/closeout commit 已产生但 session 尚未更新的窄窗口，只允许在 commit parent、固定 commit message、changed-path set 全部精确匹配 §32 时自动补写 candidate_sha/closeout_sha；其他情况 STOP `M5_BLOCKED_RESUME_GIT_STATE`，禁止 reset/rebase/force-push 猜测恢复。

若长任务仍有同一个正式进程运行：只读取进度，不得启动第二个相同任务。
completed shard/run/final game shard 禁止无条件重跑。

# 26. Artifact cleanup

candidate 前必须保留：
- `artifacts/m5/promotion_planning/` 的 pre-start promotion raw evidence（若仍存在）
- search profile
- all source/dataset manifests
- completed label shards
- selected/compared final checkpoints
- all dev raw game NPZ
- champion Teacher test evidence
- 10,000 final raw games
- session/progress JSON
- pytest.xml

删除：
- all `*_latest.pt`
- `.tmp/.lock`
- superseded partial shard scratch

不得删除可供独立 M5 audit 重算的数据。

# 27. Tracked file scope

M5 execution candidate 只允许新增/修改以下 7 个 tracked paths：
- `src/game2048/m5_pretrain.py`
- `benchmarks/profile_m5_teacher_search.py`
- `benchmarks/generate_m5_teacher_dataset.py`
- `benchmarks/run_m5_teacher_pretrain.py`
- `tests/test_m5_teacher_pretrain.py`
- `reports/m5/m5_teacher_pretrain.json`
- `reports/m5/M5_REPORT.md`

除此之外需要第 8 个 tracked path 才能继续：STOP `M5_BLOCKED_SCOPE_CHANGE_REQUIRED`。
禁止修改 CI、M0-M4 source/test、master plan、promotion audit。

职责固定：
- `m5_pretrain.py`: deterministic data/train/eval/stat pure helpers
- `profile_m5_teacher_search.py`: §8 only
- `generate_m5_teacher_dataset.py`: source games + sharded labels + manifests
- `run_m5_teacher_pretrain.py`: precision/train/dev eval/scale gate/final test/report orchestration
- test file: §28 exact tests
- JSON/MD: final machine/human evidence

# 28. 固定 CPU tests

`tests/test_m5_teacher_pretrain.py` 必须恰好包含 10 个非 parametrized test functions：
1. `test_stateless_epoch_plan_is_deterministic`
2. `test_scale_row_counts_and_game_boundaries`
3. `test_teacher_target_ignores_illegal_actions`
4. `test_d4_board_action_alignment`
5. `test_training_path_never_consumes_test_split`
6. `test_policy_only_loss_leaves_value_heads_untouched`
7. `test_scale_gate_is_deterministic`
8. `test_paired_bootstrap_is_deterministic`
9. `test_game_shard_resume_is_seed_stable`
10. `test_manifest_rejects_corrupt_completed_shard`

P1.1 固定命令：
`D:\sd-webui-forge-aki-v1.0\python\python.exe -m pytest tests/test_m5_teacher_pretrain.py -q`
必须精确：`10 passed / 0 failed / 0 skipped / 0 xfailed`。

FRESH P0 baseline 固定命令：
`D:\sd-webui-forge-aki-v1.0\python\python.exe -m pytest -q`
必须精确：`533 passed / 0 failed / 0 skipped / 0 xfailed`。

M5 final full pytest 固定命令：
`D:\sd-webui-forge-aki-v1.0\python\python.exe -m pytest -q --junitxml=artifacts/m5/pytest.xml`
必须精确：`543 passed / 0 failed / 0 skipped / 0 xfailed`；JUnit 同时必须 543 tests / 0 failures / 0 errors / 0 skipped。

# 29. CLI contract / execution order

固定 CLI：
- `profile_m5_teacher_search.py --resume`
- `generate_m5_teacher_dataset.py --phase {validation,train-32k,train-131k,train-524k,test} --resume`
- `run_m5_teacher_pretrain.py --phase {precision,train-32k,train-131k,scale-gate,train-524k,finalize,test,report} --resume`

固定顺序：
P0 FRESH preflight + official runtime check + 533 baseline
-> P1 implement 5 code/test files
-> P1.1 exact 10 tests
-> P2 profile Search; session PROFILE_RUNNING -> PROFILE_DONE
-> P3 validation source+labels and train-32k source+labels
-> P4 precision decision, train 32k three seeds, 2,000 dev games each
-> P5 train-131k incremental source+labels
-> P6 train 131k three seeds, 2,000 dev games each
-> P7 run scale-gate
-> if 524k NOT_TRIGGERED: skip its phases mechanically
-> if TRIGGERED: generate only additional train games/shards needed to reach 524k; train three seeds; 2,000 dev games each
-> P8 finalize scale selection + champion seed; session FINAL_SCALE_SELECTED
-> P9 generate test labels, champion Teacher test, champion 10,000 final games; session FINAL_TEST_DONE
-> candidate artifact cleanup
-> P10 full pytest exact 543 + `--junitxml=artifacts/m5/pytest.xml`
-> report phase
-> candidate implementation commit/push/CI
-> mandatory report-only closeout commit/push/CI
-> session COMPLETE
-> STOP independent M5 audit

不得跳步；未触发 524k 是唯一允许的 conditional skip。

# 30. Machine summary schema

`reports/m5/m5_teacher_pretrain.json` schema_version=1。
顶层 key 固定：
`schema_version, result, provenance, teacher, search_profile, dataset, precision, training, validation, development_gameplay, scale_gate, selection, teacher_test, final_gameplay, performance, regression`。

最低要求：
- provenance: base_head, prompt_sha256, five peeled tags
- teacher: path/SHA/semantics/promotion_audit/result
- search_profile: 3 repeats + median + component timing + projections
- dataset: split seeds/counts/shard size/shard manifests/SHA
- precision: smoke details + selected precision
- training: every scale/seed checkpoint SHA, epochs, steps, wall, samples/s, VRAM
- validation: every scale/seed Teacher metrics + high-tile strata + D4
- development_gameplay: raw NPZ path/SHA + summaries
- scale_gate: all paired deltas/CIs/signals/trigger/status
- selection: selected_scale, champion_seed, champion_checkpoint/SHA
- teacher_test: one-time champion test metrics
- final_gameplay: 10,000 summary + raw path/SHA
- performance: label roots/s, projected/actual label wall, training throughput
- regression: pytest counts

candidate summary `result = M5_CANDIDATE_EVIDENCE_COMPLETE`；不写假 CI success。

`--phase report` 只能消费已完成的 session/search-profile/dataset-manifest/training-progress/development/scale-selection/teacher-test/final-gameplay/pytest evidence，禁止重新训练、重新 Search、重新跑游戏或改变 selection。

`reports/m5/M5_REPORT.md` 必须是实际多行 Markdown：物理行数 >= 50，且不得出现“整份报告由字面量 `\\n` 串联成单行”的 M4 历史缺陷；不满足 STOP `M5_BLOCKED_REPORT_RENDERING`。`reports/m5/m5_teacher_pretrain.json` 必须可被 `json.load` 完整解析。

# 31. Candidate gate

只有全部满足才允许 candidate commit：
- five frozen tags unchanged
- M4 baseline evidence verified
- Teacher promotion audit = NO PROMOTION; frozen 4.8M SHA unchanged
- Search profile correctness PASS + performance gate PASS
- 32k and 131k complete
- 524k trigger mechanically obeyed
- all completed dataset shards SHA-valid
- all required 3-seed runs complete
- scale selection mechanical
- selected rung Teacher-learning gate PASS
- selected rung high-tile gate PASS
- selected rung vs M4 aggregate paired score CI lower >0
- champion chosen before test
- one-time Teacher test complete
- 10,000 final gameplay complete
- value/afterstate heads unchanged
- artifact cleanup PASS
- exact final 543 pytest
- report rendering / JSON parse gate PASS
- no tracked changes outside §27

否则禁止 `M5 CANDIDATE COMPLETE`。

# 32. Git / CI

candidate 前：
- `git status --porcelain=v1 -uall` 只能出现 §27 七个路径
- staged 为空
- 显式 `git add --` 七个 paths
- `git diff --name-only` 为空
- nonignored untracked 为空
- cached path set 精确等于七个 paths
- `git diff --cached --check` exit 0

candidate commit message 固定：
`m5: teacher pretraining candidate`

commit 前再次 `git fetch origin`，必须确认 `origin/main == base_head`；若远端已推进 STOP `M5_BLOCKED_REMOTE_ADVANCED`，不得 rebase/merge/force-push。

固定 candidate commit：
`git commit -m "m5: teacher pretraining candidate"`
随后 `git push origin main`；push 成功后记录 `candidate_sha=HEAD`，原子写 session=`CANDIDATE_PUSHED`。

按 `headSha==candidate_sha`、workflow name=`CI`、event=`push` 查询；若多个匹配 run，固定取 databaseId 最大者；若暂时 0 个，按 5 秒间隔轮询，最多 24 次（120 秒），仍无 run 则 STOP `M5_BLOCKED_CANDIDATE_CI_NOT_FOUND`。出现后轮询到 completed；只有 conclusion=`success` 通过。
任何非 success：STOP `M5_BLOCKED_CANDIDATE_CI`；执行 Agent 不得 rerun CI、不得现场修第二个 candidate。

candidate report 在 candidate commit 前固定写：`candidate_sha=PENDING_BY_DESIGN`, `candidate_ci=PENDING_BY_DESIGN`, `closeout_commit_sha=SELF_NOT_EMBEDDABLE_BY_DESIGN`, `closeout_ci=PENDING_BY_DESIGN`。

candidate CI green 后先写 session=`CLOSEOUT_REPORT_READY`；mandatory closeout 只修改 `reports/m5/M5_REPORT.md`，回写 candidate SHA / candidate CI run ID+conclusion；`closeout_commit_sha` 保持 `SELF_NOT_EMBEDDABLE_BY_DESIGN`，`closeout_ci=PENDING_BY_DESIGN_AT_REPORT_COMMIT`，禁止 amend 自引用。

closeout commit 前：
- `git fetch origin` 后必须 `origin/main==candidate_sha`
- dirty/nonignored untracked path 只能精确为 `reports/m5/M5_REPORT.md`
- 显式 `git add -- reports/m5/M5_REPORT.md`
- unstaged tracked 为空、nonignored untracked 为空
- cached path set 精确只有 `reports/m5/M5_REPORT.md`
- `git diff --cached --check` exit 0
否则 STOP `M5_BLOCKED_CLOSEOUT_STAGE_SET` 或 `M5_BLOCKED_REMOTE_ADVANCED`。

固定 closeout commit：
`git commit -m "docs: close M5 candidate report"`
然后 `git push origin main`；push 成功后记录 `closeout_sha=HEAD`，session=`CLOSEOUT_PUSHED`。

按 `headSha==closeout_sha`、workflow name=`CI`、event=`push` 查询；多个匹配取 databaseId 最大者；0 个时 5 秒间隔最多轮询 24 次。找不到 STOP `M5_BLOCKED_CLOSEOUT_CI_NOT_FOUND`；找到后轮询 completed，只有 conclusion=`success` 通过，否则 STOP `M5_BLOCKED_CLOSEOUT_CI`。不得 rerun/amend/第三个 commit。

M5 execution Agent **不得创建 M5 audited tag**，不得把 master plan 推进 M6。

# 33. STOP / blocker behavior

任意 `M5_BLOCKED_*`：立即 STOP，输出：
- exact blocker code
- session state
- actual triggering values
- HEAD/origin/main
- `git status --short`
- completed evidence paths

不得为了绕 gate 修改 protocol。

script exception / CUDA OOM / non-finite primary training：分别 STOP `M5_BLOCKED_RUNTIME_ERROR` / `M5_BLOCKED_OOM` / `M5_BLOCKED_NUMERIC`；禁止自动减 batch/改 LR/改 precision。

# 34. 最终输出

全部 candidate gates + candidate CI + closeout CI 成功后：

`M5 CANDIDATE COMPLETE`

同时输出：
- selected data scale: 32k / 131k / 524k
- champion seed
- champion checkpoint SHA
- selected-scale vs M4 aggregate paired mean delta + CI95
- validation Teacher metrics
- final 10,000-game mean/median/p10/p90/reach
- Search profile median roots/s
- actual Teacher label wall
- candidate implementation SHA
- candidate CI run
- report-only closeout SHA
- closeout CI run

然后 STOP for independent M5 audit。
不得进入 M6。
