# M6 Student State Correction — 严格施工提示词

> 状态：READY FOR EXECUTION / NOT STARTED。
>
> 项目：Safolour/NN-2048-ai
> 正式工作区：D:\CodexTasks\NN-2048-ai
> 当前阶段：M6 Student State Correction
> Work-order version：M6_WO_STATE_CORRECTION_V1
> 本文件是 M6 正式施工单；authoritative master plan 始终优先。
> M6 只允许在本施工单对应 planning commit 已 push 且 CI success 后启动。
> 本施工单的生成不等于 M6 training 已开始。

# 0. 权威输入与冲突处理

开工前必须完整读取：
1. prompts/2048_AI_正式执行计划_M0_M1冻结_CI增强版_2026-09-18.md
2. prompts/M5_IMPLEMENTATION_PROMPT.md
3. reports/m5/M5_AUDIT_REPORT.md
4. reports/m5/M5_REPORT.md
5. 本文件 prompts/M6_IMPLEMENTATION_PROMPT.md

如有真实冲突：STOP M6_BLOCKED_SPEC_CONFLICT。
不得由 execution Agent 自行判断“哪个更合理”后继续。

本施工单已经替 M6 锁死研究级决策。
Execution Agent 只能实现、运行、记录、按机械 gate 分支。
不得自行改变：
- 数据来源
- Student rollout 策略
- Teacher
- Search depth
- correction dataset 规模
- train/validation/test split
- training mixture
- optimizer / lr / epoch / batch
- training seeds
- development / validation / final test seeds
- promotion 规则
- bootstrap 规则
- tracked scope
- candidate / closeout 状态机
- audit 边界

# 1. Frozen provenance / start state

M6 开工必须验证 peeled tags：
- m0-reference-pass^{} = 3f2def1d95f56eff776e671143188947bf64485b
- m1-fastenv-audited-pass^{} = e5486017a90eeec4fb9814de7880b3c413dc06dd
- m2-network-audited-pass^{} = 6a9da5b1bf7c72207acd6e89bc667d439d4823a8
- m3-teacher-audited-pass^{} = 7c95f9c5f543065b220fb5f9e9114521732cdc0b
- m4-architecture-audited-pass^{} = 300d51a818fa55394ec56de7507bcade111a06d1
- m5-pretrain-audited-pass^{} = 36fe93cb211b08916d40df3a7e80bdea0b04a658

M5 independent audit report 必须存在：
reports/m5/M5_AUDIT_REPORT.md

M6 planning lineage 必须包含当前 M5 audit/master-plan closeout HEAD：
4b74343325d9994df957298499f9764807c366e5
作为 ancestor。

每次 run M6 先执行：
- git fetch origin
- git rev-parse HEAD
- git rev-parse origin/main
- git status --porcelain=v1 -uall
- 计算本 prompt SHA-256，记为 prompt_sha256

FRESH 仅当 artifacts/m6/progress/session.json 不存在。

FRESH Git 硬门：
- HEAD == origin/main
- worktree clean
- 六个 frozen tags 精确匹配
- 4b74343325d9994df957298499f9764807c366e5 是 HEAD ancestor
- 不存在任何已有 M6 正式进程

任一失败：STOP M6_BLOCKED_START_STATE。
禁止 pull/rebase/reset/restore/delete 猜测修复。

正式 runtime 固定：
- Python = D:\sd-webui-forge-aki-v1.0\python\python.exe
- Python 3.11.9
- PyTorch 2.9.1+cu130
- CUDA runtime 13.0
- GPU = NVIDIA GeForce RTX 5060 Laptop GPU
- nvidia-smi 可执行

任一不符：STOP M6_BLOCKED_RUNTIME_ENVIRONMENT。
不得自动换 Python / CUDA / GPU profile。

FRESH P0 通过后原子创建：
artifacts/m6/progress/session.json

至少保存：
schema_version=1
work_order_version=M6_WO_STATE_CORRECTION_V1
base_head=<FRESH 时 HEAD>
prompt_sha256
m5_tag_sha
m5_student_checkpoint_sha256
teacher_sha256
state=P0_PASSED

RESUME 时禁止重新 FRESH。
必须验证上述 metadata、六个 frozen tags 与 base lineage。
不一致：STOP M6_BLOCKED_RESUME_METADATA_MISMATCH。

# 2. Frozen M5 protection

M0-M5 均视为 frozen input。
M6 不允许为了实现方便回写任何 M0-M5 frozen implementation、prompt、report、audited artifact 或 tag。

特别禁止：
- 重新训练 M5 32k / 131k / 524k
- 重新选择 M5 scale
- 重新选择 M5 champion seed
- 修改 M5 final checkpoint
- 修改 M5 source/label shards
- 修改 M5 development/final gameplay raw
- 重新运行 M5 Teacher promotion gate
- 把 M5 10,000-game final result 当作 M6 paired gate 的替代品

M5 historical final mean 9812.2184 只作历史参考。
M6 promotion 必须在 M6 自己冻结的新 seed split 上重新 paired evaluate 同一个 M5 champion。

若未来独立发现 M5 correctness bug：
必须另开独立 audit-fix 流程。
M6 execution Agent 不得自行修 M5。

# 3. M6 唯一研究问题

M6 只回答一个问题：
让 frozen M5 champion 按自己的 pure-NN policy 实际玩游戏，
收集它真正访问的 on-policy states，
再由 frozen depth-3 Teacher 重标注这些 states，
从同一个 M5 champion checkpoint 出发做一次监督式 state correction，
能否在新的独立 gameplay seeds 上显著提高 pure-NN 平均最终分数。

M6 不是：
- M5 replay
- Teacher checkpoint promotion
- architecture sweep
- model-size sweep
- Search depth sweep
- disagreement-only mining experiment
- loss/head sweep
- LR / epoch / batch sweep
- RL / Q-learning
- Double-Q / EMA Target / Replay Buffer
- exploration / exact-spawn experiment
- high-tile restart pool
- M9 Search Correction
- M10 iterative Champion/Learner loop
- M7 implementation

M6 只做一次固定 protocol correction。
不得把一次 M6 自动扩展成多轮 correction。

# 4. Frozen Student baseline

M6 Student baseline 唯一固定为 M5 selected champion：
- architecture：ResidualMLP2048
- parameter count：5,264,710
- M5 selected scale：524k
- M5 champion seed：20262103
- checkpoint：artifacts/m5/checkpoints/524k/20262103_final.pt
- checkpoint SHA-256：4C1313E9085E3A0C1FEC4C6AAA1200A0A5A4377F3EAE0176659F80E9EEAE5784

P0 必须重新计算 checkpoint SHA。
不一致：STOP M6_BLOCKED_M5_BASELINE_CHECKPOINT。

所有 M6 fine-tune run 都必须从这一个 exact checkpoint 的 model_state 开始。
禁止：
- 从 M5 其他 seed 开始
- 从随机初始化开始
- 从某个 M6 run warm-start 另一个 M6 run
- 加载 M5 optimizer state
- 在三 run 间传递 optimizer / scheduler state

M6 每个 training seed：
同一 model_state 起点 + fresh optimizer。

# 5. Frozen Teacher / Search semantics

M6 Teacher 继续固定为 M3/M5 retained Teacher：
- checkpoint：teacher_checkpoints/m3/ordinary_td_comparator_ep4800000_7192719323a0.bin
- SHA-256：7192719323A073BA2B6B19B62CB7D46EF4AA90ECC8C4AE6BAF27AD0C51566A84

不得使用：
- latest.bin
- COMPARATOR 6.4M
- FORMAL 8.4M
- 新出现的 tuple checkpoint
- 混合 Teacher

M6 不重新打开 Teacher promotion。
Teacher promotion = M5 frozen NO PROMOTION。

value semantics 继续冻结：
- V_tuple = RAW_TUPLE_HEURISTIC
- formal leaf = FORMAL_STATE_TUPLE_HEURISTIC
- depth-3 root values = SEARCH_VALUE_RAW_LEAF

Search 固定：
- decision_depth = 3
- chance nodes 不计 decision depth
- use_cache = True
- formal-state leaf = frozen tuple afterstate greedy-1ply adapter
- action value = immediate reward + downstream Expectimax value

Teacher best action：
只在 legal actions 中取 np.nanargmax；exact Search tie 取最低 action id。

因此 M6 supervision 仍然只能把 Search 当 policy / ranking Teacher。
禁止：
- raw teacher_value 做 absolute Q MSE
- V MSE
- Afterstate MSE
- M3 affine calibration
- 把 SEARCH_VALUE_RAW_LEAF 宣称为 FUTURE_SCORE

# 6. M6 冻结实验设计总表

A. Frozen Student on-policy source
- train：1,024 logical games × 128 states = 131,072 states
- Teacher validation：64 logical games × 128 = 8,192 states
- one-time Teacher test：64 logical games × 128 = 8,192 states

B. Teacher relabel
- 所有 sampled states 全部 depth-3 relabel
- 不按 Student/Teacher disagreement 过滤
- disagreement 只作为诊断字段

C. Correction training
- 3 个 training seeds
- 全部从 exact M5 champion 起步
- batch = 1,024
- 每 batch 精确 512 M6 correction + 512 M5 anchor

- 30 epochs
- policy CE only
- backbone + Q head 可更新
- Value / Afterstate heads 必须保持相对 M5 baseline bit-identical

D. Selection / validation / final
- 2,000 fresh development games：选唯一 M6 candidate seed
- 5,000 fresh independent gameplay validation：promotion validation gate
- 10,000 fresh final paired games：one-time final confirmation
- 三套 gameplay seeds 与所有 source seeds互不重叠

除此之外没有隐藏 sweep。

# 7. Student rollout policy 精确冻结

Student source trajectory 必须由 frozen M5 champion 自己产生。

正式 rollout policy：
- single ResidualMLP2048 forward
- exact M5 champion checkpoint
- model.eval()
- FP32 inference
- autocast disabled
- no Teacher / Search / tuple
- no D4 ensemble
- no exploration
- legal mask
- tie tolerance = 1e-7
- fast movement 必须保持 frozen M0 exact semantics

必须直接复用 M4/M5 audited pure-NN action semantics。

每个 game_seed：
spawn_rng = PCG64(game_seed)
tie_rng = PCG64(game_seed XOR 0x9E3779B9)

初始 board：
zeros uint8[16]
-> frozen M0 spawn_random 两次
-> 后续同一个 spawn_rng 连续消费

每一步：
1. 对当前 formal state 做 Student forward
2. legal mask
3. 选择所有 legal Q/logit 与最大值差 <=1e-7 的 candidates
4. 只有一个 candidate：直接选
5. 多个 candidate：用该局独立 tie_rng 均匀抽一个
6. 执行动作
7. legal move 后才 spawn
8. terminal 正常结束

禁止把 tie RNG 与 spawn RNG 混在一个 stream。

# 8. Student source splits / seed contract

train logical games：
- game_id = 0..1023
- canonical_game_seed = 20288001 + game_id

Teacher-validation logical games：
- game_id = 100000..100063
- canonical_game_seed = 20290001 + local_index

Teacher-test logical games：
- game_id = 200000..200063
- canonical_game_seed = 20290101 + local_index

train / validation / test seeds 不得重叠。
它们也不得与 §18/§19/§21 gameplay seeds 重叠。

每个 accepted game 固定抽 128 个 canonical formal states：
np.rint(np.linspace(0, moves-1, 128)).astype(np.int64)

必须 128 个 position 全唯一。
禁止 duplicate/pad。

若 Student game 天然 moves <128：
整个 attempt rejected，
其任何 state 不进入正式 dataset。

deterministic retry：
- retry_index=0：effective_game_seed = canonical_game_seed
- retry_index>0：effective_game_seed = canonical_game_seed + retry_index * 1_000_000_000
- 最多 retry_index=32

0..32 全部不足 128：STOP M6_BLOCKED_SOURCE_GAME_RETRY_EXHAUSTED。

第一个 moves>=128 的 attempt accepted。
logical game_id 不变。
dataset 的 game_seed 保存 accepted effective seed。

source manifest 必须保存：
- logical game_id
- canonical_game_seed
- accepted game_seed
- retry_index
- rejected attempts 的 seed/moves/final_score/max_tile_exp
- accepted moves/final_score/max_tile_exp
- sampled step indices

# 9. Search re-profile gate

M6 在任何正式 correction label 前必须重新确认 frozen Search 没有 correctness / throughput 回退。

固定 profile 输入：
artifacts/m3/semantic_profile_states.npz
SHA-256：9AFD3A503819C8BD777C7F48D3CCDCA2B1C2F02A7D28DE7DF8997735EE68E512

reference：
artifacts/m3/search_ab_p6_cpp_values.npz
SHA-256：8E09F0D7774BCEC55C496A29550F2362FB83DEE142643F69472F12AE14963C51

固定：
- roots = 256
- repeats = 3
- 每 repeat 对全部 256 roots 重新构造 Search

correctness：三个 repeat 的 legal mask、action values、best action、node/cache counts 必须与 reference 精确一致。
失败：STOP M6_BLOCKED_SEARCH_CORRECTNESS。

M5 audited Search median：9.781657434455234 roots/s
M6 performance gate：3-repeat median roots/s >= 8.803491691009711
即 M5 audited median 的 90%。

低于：STOP M6_BLOCKED_SEARCH_PERFORMANCE_REGRESSION。

orchestration_share：
三个 repeat 的 recursive_python_orchestration_seconds 总和 / 三个 repeat total_seconds 总和。
必须 <= 0.50。
超过 0.50：STOP M6_BLOCKED_SEARCH_PERFORMANCE_UNBLOCK_REQUIRED。

Execution Agent 不得现场重构 Search；需要优化时另开独立 unblock。

必须报告 projected wall：
- 131,072 train labels
- 8,192 validation labels
- 8,192 test labels
- total

# 10. Correction dataset / Teacher relabel contract

source trajectory 与 Teacher Search relabel 必须分离。
固定 source manifests：
- artifacts/m6/datasets/source_train_manifest.json
- artifacts/m6/datasets/source_validation_manifest.json
- artifacts/m6/datasets/source_test_manifest.json

固定 label manifest：
artifacts/m6/datasets/label_manifest.json

Teacher-test source/labels 只有 CANDIDATE_LOCKED 之后允许生成。

Search label shard：2,048 states = 16 whole games。
禁止 game 跨 shard。

因此：
- train = 64 shards
- validation = 4 shards
- test = 4 shards

shard path：
artifacts/m6/datasets/<split>/shard_<index:04d>.npz

每个 completed shard 立即计算 SHA-256 并原子更新 manifest。

每 shard 至少包含：
- state [N,16] uint8
- teacher_value [N,4] float64
- reward [N,4] int32
- afterstate [N,4,16] uint8
- legal_mask [N,4] bool
- student_logits [N,4] float32
- student_action [N] uint8
- teacher_best_action [N] uint8
- disagreement [N] bool
- game_id [N] int64
- game_seed [N] int64
- step_index [N] int32
- current_score [N] int64
- max_tile_exp [N] uint8
- student_checkpoint_sha256
- teacher_checkpoint_sha256
- decision_depth
- value_semantics
- data_source

student_logits 只表示 frozen M5 policy logits，不得伪造 absolute Q semantics。

Teacher illegal action：
- teacher_value = NaN
- legal_mask = False
- reward / afterstate 可用零 sentinel
- 不参加 target / metric / loss

teacher_best_action 只在 legal actions 中取 argmax。
disagreement = teacher_best_action != frozen Student rollout action。

所有 source board：canonical_unaugmented=True。
D4 只在 training batch 动态执行。
M6 训练使用全部 train rows，禁止 disagreement-only / confidence / score / tile 过滤。
# 11. Label generation / resume / observability

Teacher label worker count 固定 8。
workers：
- 只读 frozen Teacher
- 每 root 独立 ExpectimaxTeacher(decision_depth=3,use_cache=True)

每完成 64 roots 必须输出并持久化：
- split
- shard
- completed roots / total
- elapsed
- roots/s
- ETA
- current shard status

只有 2,048-root shard 全部完成后：
temp -> atomic rename 正式 NPZ -> SHA -> manifest complete。

中断在 shard 内：丢弃该 partial shard；恢复时只重做当前 shard。
completed + SHA-valid shard：禁止重跑。

manifest 声称 complete 但文件缺失、schema/shape 不符或 SHA 不符：
STOP M6_BLOCKED_DATASET_ARTIFACT_CORRUPT。
不得删掉后偷偷重算。

# 12. M5 anchor dataset 固定

M6 correction 不允许 correction-only 训练。
固定使用 M5 anchor 防止在原 Teacher distribution 上灾难性遗忘。
anchor 唯一来源：M5 524k TRAIN labels。

读取：
- artifacts/m5/datasets/label_manifest.json
- manifest SHA-256 必须精确为 AE6BD51FCF0815A3EB9B919CC4D5980E41EC614CC01ECE174B58F20E3AFE1551
- manifest 中全部 train shard references

只允许 M5 train split。
禁止把 M5 validation 或 test rows 用于 training。

P0/P2 必须验证：
- M5 label manifest 存在
- train 256 shards complete
- 524,288 train rows
- 每个 shard SHA-valid
- Teacher SHA 与 frozen M3 Teacher 一致
- value_semantics = SEARCH_VALUE_RAW_LEAF

失败：STOP M6_BLOCKED_M5_ANCHOR_EVIDENCE。

M6 不复制、不改写、不重新 relabel M5 anchor；只读。

# 13. Training mixture 精确冻结

M6 每个 optimizer batch：总 batch_size = 1024。
其中精确：
- 512 M6 correction rows
- 512 M5 anchor rows

禁止动态调比例、loss reweight、disagreement weighting、prioritized sampling、hard mining、curriculum。

每个 epoch：M6 correction 131,072 rows 全部恰好使用一次。
因此：correction_half_batch = 512，steps_per_epoch = 131072 / 512 = 256。

global row id 定义固定：
- M6 correction：按 train shard_index 升序连接，各 shard 内保持原始 row 顺序，得到 global ids 0..131071
- M5 anchor：按 M5 train shard_index 升序连接，各 shard 内保持原始 row 顺序，得到 global ids 0..524287
禁止依赖 filesystem 枚举顺序。

每 epoch anchor：
从 524,288 M5 train rows 中确定性无放回抽 131,072 rows；每 row 本 epoch最多一次。
不同 epoch 允许重新抽不同 subset。

每个 step：
- correction permutation 当前连续 512 rows 作为 batch 前半
- anchor subset/permutation 当前连续 512 rows 作为 batch 后半
- 固定按 correction-first / anchor-second concat 成 1,024 rows
- concat 后不再额外 shuffle
- 一次统一 forward / CE mean

target：Teacher legal best action。
loss：4-class policy cross entropy。
禁止 absolute value regression。

# 14. Training seeds / stateless plan

固定 M6 training seeds：
- 20263101
- 20263102
- 20263103

三个 run：同一 M5 checkpoint 起点，fresh AdamW optimizer。
每个 run 开头必须先调用项目既有 deterministic seeding helper 等价设置 Python/NumPy/PyTorch/CUDA 的 seed=S 与 deterministic flags；不得另选 seed 派生规则。
然后 load exact M5 model_state，再创建 fresh optimizer；禁止先做任何随机参数修改。
epoch = 0..29，共 30 epochs。

对 seed S、epoch e：
correction_rng = PCG64(S + 300000 + e)
anchor_rng = PCG64(S + 400000 + e)

correction_rng 固定顺序：
1. permutation 131,072 correction row ids
2. integers(0,8,size=131072,dtype=uint8) 生成 correction D4 ids

anchor_rng 固定顺序：
1. permutation 524,288 M5 anchor row ids
2. 取前 131,072 作为本 epoch anchor subset
3. integers(0,8,size=131072,dtype=uint8) 生成 anchor D4 ids

D4 必须同步变换 board + Teacher action label。
validation/test/gameplay：canonical，不做 augmentation。

plan_id：
M6_PLAN_V1:<seed>:CORR50_ANCHOR50:PCG64_STATELESS_EPOCH

必须记录 plan_id SHA-256。

checkpoint 只在完整 epoch 后保存。
中断 epoch：从该 epoch 开头重做。
已完成 epoch：禁止重跑。

# 15. Optimizer / precision / parameter update

模型 architecture 不变。
禁止改 width/depth/embedding/head shape/residual topology。

optimizer 固定：torch.optim.AdamW
- lr = 3e-4
- weight_decay = 1e-4
- betas = (0.9,0.999)
- eps = 1e-8
- amsgrad = False
- foreach = False
- fused = False
- capturable = False
- no scheduler / no early stopping / no label smoothing
- gradient clip global norm = 1.0
每 step：zero_grad(set_to_none=True) -> forward -> CE(mean) -> backward -> clip_grad_norm_ -> optimizer.step()。

precision 固定沿用 M5 audited selection：BF16 autocast for training。
禁止 FP16 / GradScaler / 自动 fallback 改 protocol。

P1 先做 5-step BF16 numeric smoke，仍从 exact M5 checkpoint copy 开始；smoke 不计正式 optimizer steps。
smoke 必须使用独立临时 model copy + 独立临时 optimizer；smoke 结束后无论 PASS/FAIL 都不得保存为正式 training checkpoint，也不得复用其 optimizer/model state。
BF16 unsupported / loss / grad / parameter non-finite：STOP M6_BLOCKED_NUMERIC_SMOKE。
smoke PASS 后，三个正式 run 必须再次各自从 exact M5 checkpoint 重新 load model_state，并各自新建 fresh optimizer；smoke 的 5 steps 对正式 run 的参数、optimizer steps、RNG plan 均为零影响。

正式 run：30 × 256 = 7,680 optimizer steps / seed。
三个 run 都必须完整 7,680 steps，不得 early stop。

训练后：
- backbone 至少一个 tensor 相对 M5 checkpoint 改变
- q_head 至少一个 tensor 相对 M5 checkpoint 改变
- value_head 每个 tensor 相对 M5 checkpoint bit-identical
- afterstate_head 每个 tensor 相对 M5 checkpoint bit-identical

失败分别：
- M6_BLOCKED_NO_PARAMETER_UPDATE
- M6_BLOCKED_VALUE_HEAD_UPDATED

# 16. Training observability

每个 epoch 必须可读输出：
- seed
- epoch / 30
- train total CE
- correction CE
- anchor CE
- Teacher-validation CE
- Teacher-validation legal accuracy
- elapsed training wall
- samples/s
- peak VRAM
- estimated remaining training wall

每个 epoch 后原子写：artifacts/m6/progress/train_<seed>.json

running checkpoint：artifacts/m6/checkpoints/<seed>_latest.pt
final checkpoint：artifacts/m6/checkpoints/<seed>_final.pt

每个 final checkpoint 至少含：
- architecture
- seed
- work_order_version
- prompt_sha256
- m5_baseline_checkpoint_sha256
- teacher_sha256
- correction manifest SHA
- M5 anchor manifest SHA
- plan_id
- epoch=30
- optimizer_steps=7680
- precision
- model_state

# 17. Teacher-validation metrics

固定 validation：64 Student rollout games / 8,192 states，与 train seed完全隔离。
在同一 8,192 states 上先评估 frozen M5 baseline，再评估每个 M6 corrected run。

记录：
- CE
- raw best-action accuracy
- legal best-action accuracy
- legal pairwise ranking accuracy
- ranking pair count
- random-legal baseline
- frozen-Student/Teacher disagreement rate
- accuracy on baseline-disagreement subset
- D4 legal-argmax consistency
- centered-logit D4 MAE

high-tile diagnostic strata：
- >=11 (2048+)
- >=12 (4096+)
- >=13 (8192+)

每 stratum 报 samples / accuracy / pairwise。
M6 不对稀疏 high-tile stratum 设置 promotion gate。

Teacher-validation learning sanity：每个 M6 final run必须：
- CE finite
- legal accuracy finite
- pairwise finite
- D4 metrics finite
- legal accuracy >= split random-legal baseline + 0.10
- pairwise >= 0.65

任一失败：STOP M6_BLOCKED_CORRECTION_LEARNING_SANITY。

这些 Teacher metrics不决定三个 seed 中谁成为 locked M6 candidate。
locked M6 candidate 只由 §18 development gameplay 机械选择；independent audit 前禁止把它称为 authoritative Champion。

# 18. Pure-NN development gameplay / seed selection

M6 development seeds：20291001..20293000，exactly 2,000 games。
该 split 从未用于 M5 selection，且不得与任何 M6 source/validation/final split 重叠。

先用 frozen M5 champion 跑 baseline：
artifacts/m6/games/development/m5_baseline.npz
然后三个 M6 final checkpoints 各跑同一 2,000 seeds：
artifacts/m6/games/development/<seed>.npz

pure-NN evaluator 固定：
- single forward
- FP32
- no Teacher / Search / tuple / D4 ensemble / exploration
- legal mask
- tie tolerance 1e-7
- spawn/tie RNG exact §7 semantics
- same fast env semantics

raw key：game_seed, final_score, max_tile_exp, moves

对每个 M6 seed分别与 M5 baseline做 paired bootstrap：
- 10,000 resamples
- 每 replicate 有放回抽 2,000 paired indices
- NumPy quantile [0.025,0.975] default linear

bootstrap seeds：
- 20263101 vs M5 = 20266201
- 20263102 vs M5 = 20266202
- 20263103 vs M5 = 20266203

记录每个 seed：
- mean / median / p10 / p90
- paired mean delta + CI95
- wins / losses / ties
- reach 2048/4096/8192/16384/32768/65536

唯一 candidate seed：development mean score 最大者；精确相等时 seed 数值更小者胜。
禁止：
- 按 Teacher accuracy 选 seed
- 按 high-tile reach 选 seed
- 看 gameplay validation 后换 seed
- 看 final test 后换 seed

candidate seed 选定后：session = CANDIDATE_LOCKED。
后续不得重新选择。

dev_positive：locked candidate 的 paired mean delta > 0。
dev_positive 是最终 promotion 必要条件；development CI 不作为最终显著性门，因为该 split承担 seed selection。

# 19. Independent gameplay validation

CANDIDATE_LOCKED 后，只评估：
- frozen M5 baseline
- locked M6 candidate

固定 validation gameplay seeds：20294001..20299000，exactly 5,000 games。

raw：
- artifacts/m6/games/validation/m5_baseline.npz
- artifacts/m6/games/validation/m6_candidate.npz

evaluator 仍为 §18 exact pure-NN semantics。

paired bootstrap：
- 10,000 resamples
- 5,000 paired indices
- bootstrap seed = 20266210
- quantile [0.025,0.975] default linear

validation_gain_pass：paired mean delta > 0 AND CI95 lower bound > 0。
validation result一旦写入：不得再训练、换 seed 或改 hyperparameter。
validation fail 不是 runtime blocker；它是合法研究结果 NO_PROMOTION 的证据之一。

# 20. One-time Teacher test isolation

M6 Teacher-test split：64 games / 8,192 states，只允许在 CANDIDATE_LOCKED 后生成 source + depth-3 labels。

在 CANDIDATE_LOCKED 前禁止：
- 生成 test source
- 生成 test Search labels
- 把 test board 输入 M6 model
- 计算 test Teacher metrics

locked candidate 只计算一次 test Teacher metrics。
M5 baseline 同一 test split也计算一次供诊断 comparison。

Teacher-test 永远不能：
- 改 candidate seed
- 触发 retrain
- 改 training mixture
- 改 epoch/lr
- 改 promotion gate

违反：STOP M6_BLOCKED_TEST_ISOLATION。

# 21. Final paired 10,000-game test

locked candidate + gameplay validation 完成后，执行一次 final paired gameplay。

固定 seeds：20300001..20310000，exactly 10,000 games。

必须同时跑 frozen M5 baseline 和 locked M6 candidate。
raw：
- artifacts/m6/final/m5_baseline_10000.npz
- artifacts/m6/final/m6_candidate_10000.npz

每 100 games 一个原子 shard。
partial shard 中断：整 shard重跑。
completed shard禁止重跑。

报告双方：
- mean
- median
- p10
- p90
- mean moves
- max-tile distribution
- reach 2048/4096/8192/16384/32768/65536

paired：
- mean delta
- wins/losses/ties
- 10,000-resample paired bootstrap
- bootstrap seed = 20266220
- CI95 default linear quantile

final_gain_pass：paired mean delta > 0 AND CI95 lower bound > 0。

final test 只消费一次。
无论结果如何：禁止 retrain / reselection / rerun with new seed。

# 22. Promotion decision

M6 promotion_result 只有两种：
- PROMOTION_ELIGIBLE
- NO_PROMOTION
PROMOTION_ELIGIBLE 必须同时满足：
1. dev_positive = True
2. validation_gain_pass = True
3. final_gain_pass = True
4. correction learning sanity PASS
5. 所有 provenance/data/test-isolation/regression gates PASS

否则：NO_PROMOTION。

NO_PROMOTION 不是 M6 runtime failure，也不是允许 Agent 现场改方案的理由。
NO_PROMOTION 时：
- M6 protocol仍可 CANDIDATE COMPLETE
- frozen Student baseline继续保持 M5 champion
- 不得再试另一个 lr/epoch/seed/ratio
- 不得进入 M7

PROMOTION_ELIGIBLE 时：
locked M6 checkpoint 只是提交 independent M6 audit 的 promotion candidate。
在 independent audit 通过前，它不是新的 authoritative frozen Student baseline。

Execution Agent 不得创建 M6 audited tag。

# 23. Session states / evidence DAG

session states 固定：
P0_PASSED
P1_READY
PROFILE_RUNNING
PROFILE_DONE
SOURCE_TRAIN_RUNNING
SOURCE_TRAIN_DONE
LABEL_TRAIN_RUNNING
LABEL_TRAIN_DONE
SOURCE_VALIDATION_RUNNING
SOURCE_VALIDATION_DONE
LABEL_VALIDATION_RUNNING
LABEL_VALIDATION_DONE
DATA_READY
TRAIN_RUNNING
TRAIN_DONE
DEV_RUNNING
DEV_DONE
CANDIDATE_LOCKED
GAMEPLAY_VALIDATION_RUNNING
GAMEPLAY_VALIDATION_DONE
TEST_DATA_RUNNING
TEACHER_TEST_DONE
FINAL_GAMEPLAY_RUNNING
FINAL_GAMEPLAY_DONE
PROMOTION_DECIDED
P10_RUNNING
P10_DONE
REPORT_WRITTEN
CANDIDATE_PUSHED
CLOSEOUT_REPORT_READY
CLOSEOUT_PUSHED
COMPLETE

固定 evidence DAG 与精确路径：
artifacts/m6/progress/session.json
-> artifacts/m6/provenance.json
-> artifacts/m6/search_profile.json
-> artifacts/m6/datasets/source_train_manifest.json / source_validation_manifest.json / source_test_manifest.json
-> artifacts/m6/datasets/label_manifest.json + label shards
-> artifacts/m6/anchor_verification.json
-> artifacts/m6/progress/train_<seed>.json + artifacts/m6/checkpoints/<seed>_final.pt
-> artifacts/m6/teacher_validation.json
-> artifacts/m6/development.json
-> artifacts/m6/candidate_selection.json
-> artifacts/m6/gameplay_validation.json
-> artifacts/m6/teacher_test.json
-> artifacts/m6/final_gameplay.json
-> artifacts/m6/promotion.json
-> artifacts/m6/pytest.xml
-> reports/m6/m6_state_correction.json + reports/m6/M6_REPORT.md

所有 JSON/NPZ：temp + atomic rename。

# 24. RESUME 规则

pre-candidate RESUME：
- HEAD==origin/main==base_head
- staged diff 必须为空
- dirty/untracked 非 ignored路径只能属于 §27 M6 execution tracked scope 的尚未提交文件
- 六个 frozen tags仍精确
- M5 baseline SHA / Teacher SHA / prompt SHA仍精确

不得重跑：
- completed source logical games
- SHA-valid label shards
- completed epochs
- completed dev game shards
- completed validation game shards
- completed final game shards

若同一正式长任务进程仍在运行：只读 progress，不得启动第二个同任务。

post-candidate RESUME：
- CANDIDATE_PUSHED：HEAD==origin/main==candidate_sha，worktree clean，只恢复 candidate CI
- CLOSEOUT_REPORT_READY：HEAD==origin/main==candidate_sha，dirty path 只能为空或精确 reports/m6/M6_REPORT.md
- CLOSEOUT_PUSHED：HEAD==origin/main==closeout_sha，worktree clean，只恢复 closeout CI
- COMPLETE：禁止重跑实验，直接报告现有 complete evidence并 STOP
若 session 与 Git 断在 commit 已创建但 session 未更新的窄窗口：
只有 parent、fixed commit message、changed-path set 全部精确符合 §33 时，才允许机械补写 SHA。

否则：STOP M6_BLOCKED_RESUME_GIT_STATE。
禁止 reset/rebase/force-push 猜测恢复。

# 25. Artifact cleanup

candidate 前必须保留可供 independent audit 重算的正式 evidence：
- artifacts/m6/progress/session.json
- provenance / search profile
- source manifests
- completed correction label shards
- train progress
- 三个 final checkpoints
- M5 anchor verification
- development raw games
- gameplay validation raw games
- Teacher test evidence
- final 10,000 paired raw games
- promotion evidence
- pytest.xml

必须删除：
- 所有 *_latest.pt
- .tmp
- .lock
- partial/scratch shards
- 一次性 debug dump
- superseded 临时 benchmark 输出

不得把任何 M6 milestone result 放仓库根目录。
有审计价值的大型/原始数据只放 artifacts/m6/。
tracked 报告只放 reports/m6/。
# 26. 长任务可观测性

以下任务必须有持续可读输出：
- Student rollout source generation
- Teacher Search relabel
- training
- 2,000 dev gameplay
- 5,000 validation gameplay
- 10,000 final gameplay
- full pytest / CI waiting

至少输出：
- phase
- completed / total
- elapsed
- throughput
- ETA
- 当前 shard/game/epoch

训练另外输出关键 metric。
Search另外输出 roots/s。
gameplay另外输出 games/s 与 rolling mean。

禁止长时间静默黑盒运行。

# 27. Tracked file scope

M6 execution candidate 只允许新增/修改以下 7 个 tracked paths：
1. src/game2048/m6_state_correction.py
2. benchmarks/profile_m6_teacher_search.py
3. benchmarks/generate_m6_correction_dataset.py
4. benchmarks/run_m6_state_correction.py
5. tests/test_m6_state_correction.py
6. reports/m6/m6_state_correction.json
7. reports/m6/M6_REPORT.md
本 planning prompt：prompts/M6_IMPLEMENTATION_PROMPT.md 已属于 M6 planning base，execution Agent不得修改。
master plan：execution Agent不得修改。
M0-M5任何 tracked path：execution Agent不得修改。

需要第 8 个 execution tracked path 才能继续：
STOP M6_BLOCKED_SCOPE_CHANGE_REQUIRED。
不得自行扩大 scope。

职责固定：
- m6_state_correction.py：deterministic source/train/eval/stat pure helpers
- profile_m6_teacher_search.py：§9 only
- generate_m6_correction_dataset.py：Student source + Teacher relabel + manifests
- run_m6_state_correction.py：anchor verify/train/dev/validation/test/promotion/report orchestration
- test file：§28 exact tests
- JSON/MD：final machine/human evidence

# 28. 固定 CPU tests

tests/test_m6_state_correction.py 必须恰好包含 10 个 non-parametrized test functions：
1. test_student_rollout_is_seed_stable
2. test_student_rollout_uses_frozen_policy_and_independent_rngs
3. test_teacher_relabel_ignores_illegal_actions
4. test_source_splits_are_disjoint_and_test_isolated
5. test_correction_anchor_batch_is_exactly_half_and_deterministic
6. test_finetune_starts_from_exact_m5_checkpoint_with_fresh_optimizer
7. test_policy_only_correction_leaves_value_heads_untouched
8. test_candidate_selection_and_paired_bootstrap_are_deterministic
9. test_final_test_cannot_run_before_candidate_lock
10. test_manifest_rejects_corrupt_completed_shard

P0 baseline command：D:\sd-webui-forge-aki-v1.0\python\python.exe -m pytest -q
P0 baseline 必须精确：
- 543 passed
- 0 failed
- 0 skipped
- 0 xfailed

P1.1 command：
D:\sd-webui-forge-aki-v1.0\python\python.exe -m pytest tests/test_m6_state_correction.py -q

必须精确：
- 10 passed
- 0 failed
- 0 skipped
- 0 xfailed

M6 final P10：
D:\sd-webui-forge-aki-v1.0\python\python.exe -m pytest -q --junitxml=artifacts/m6/pytest.xml

必须精确：
- 553 passed
- 0 failed
- 0 skipped
- 0 xfailed

JUnit 聚合实际 testsuite 后必须：
- 553 tests
- 0 failures
- 0 errors
- 0 skipped

# 29. CLI contract

固定 CLI：
- benchmarks/profile_m6_teacher_search.py --resume
- benchmarks/generate_m6_correction_dataset.py --phase train --resume
- benchmarks/generate_m6_correction_dataset.py --phase validation --resume
- benchmarks/generate_m6_correction_dataset.py --phase test --resume
- benchmarks/run_m6_state_correction.py --phase anchor-check --resume
- benchmarks/run_m6_state_correction.py --phase smoke --resume
- benchmarks/run_m6_state_correction.py --phase train --resume
- benchmarks/run_m6_state_correction.py --phase development --resume
- benchmarks/run_m6_state_correction.py --phase select --resume
- benchmarks/run_m6_state_correction.py --phase gameplay-validation --resume
- benchmarks/run_m6_state_correction.py --phase teacher-test --resume
- benchmarks/run_m6_state_correction.py --phase final-gameplay --resume
- benchmarks/run_m6_state_correction.py --phase promote --resume
- benchmarks/run_m6_state_correction.py --phase report --resume

# 30. P0 -> P10 固定执行顺序

P0：
- frozen provenance
- runtime
- M5 checkpoint SHA
- Teacher SHA
- M5 anchor manifest/SHA
- exact 543 baseline pytest
- create session

P1：
- 只实现 §27 五个 code/test paths
- 不开始正式 training
- BF16 5-step smoke helper ready

P1.1：
- exact M6 10 tests PASS

P2：
- Search re-profile
- correctness/performance/orchestration gates
- report projections

P3：
- generate train Student source
- generate validation Student source
- manifests complete

P4：
- Teacher relabel train 64 shards
- Teacher relabel validation 4 shards
- DATA_READY

P5：
- anchor-check
- BF16 numeric smoke
- 三个 seeds 从 exact M5 checkpoint 正式 correction training
- 30 epochs / 7,680 steps each
- Teacher-validation metrics
- TRAIN_DONE

P6：
- frozen M5 baseline 2,000 dev games
- 三个 M6 runs 同一 2,000 dev games
- paired stats
- mechanically select unique candidate
- CANDIDATE_LOCKED

P7：
- frozen M5 baseline + locked candidate
- 5,000 independent gameplay validation
- write validation_gain_pass
- 不得重训

P8：
- generate one-time Teacher-test source
- 4 Teacher-test Search shards
- evaluate M5 baseline + locked candidate Teacher metrics once
- 不得影响 selection

P9：
- frozen M5 baseline + locked candidate
- 10,000 paired final gameplay
- compute final_gain_pass
- mechanically compute PROMOTION_ELIGIBLE / NO_PROMOTION
- PROMOTION_DECIDED

P10：
- artifact cleanup
- exact 553 full pytest + JUnit
- validate every required raw/manifest SHA
- validate report JSON schema
- validate Markdown rendering
- candidate gate
- candidate commit/push/CI
- mandatory report-only closeout commit/push/CI
- COMPLETE
- STOP for independent M6 audit

不得跳步。
不得在任何 P 阶段进入 M7。

# 31. Machine summary / report

reports/m6/m6_state_correction.json：schema_version=1。

顶层 key 固定：
schema_version, result, provenance, student_baseline, teacher, search_profile,
student_rollout, correction_dataset, anchor, training, teacher_validation,
development_gameplay, selection, gameplay_validation, teacher_test,
final_gameplay, promotion, performance, regression
最低要求：

provenance：base_head, prompt_sha256, six peeled tags。
student_baseline：checkpoint path/SHA, scale, seed, architecture。
teacher：path/SHA/semantics/search depth。
search_profile：3 repeats, median, component timing, projections。
student_rollout：split seeds/counts/retries/pure-NN policy contract。
correction_dataset：manifests/shards/SHAs/disagreement rate。
anchor：M5 manifest SHA + verified 524,288 train rows。
training：3 seeds, exact start SHA, plan_id, epochs, steps, wall, throughput, VRAM, final checkpoint SHA。
teacher_validation：M5 baseline + each M6 seed metrics + strata。
development_gameplay：all raw NPZ paths/SHAs/summaries/paired stats。
selection：candidate seed/checkpoint/SHA, CANDIDATE_LOCKED evidence。
gameplay_validation：paired 5,000 raw + delta + CI + pass。
teacher_test：one-time 8,192 metrics。
final_gameplay：paired 10,000 raw + summaries + delta + CI + pass。
promotion：dev_positive, validation_gain_pass, final_gain_pass, final result。
performance：Search roots/s, label wall, training throughput, gameplay throughput。
regression：pytest/JUnit counts。

candidate summary result 只能是：
- M6_CANDIDATE_EVIDENCE_COMPLETE_PROMOTION_ELIGIBLE
- M6_CANDIDATE_EVIDENCE_COMPLETE_NO_PROMOTION

不得把 candidate 写成 FINAL AUDITED PASS。

report phase 只能消费已完成 evidence。
禁止 report phase 触发 training / Search / gameplay。

reports/m6/M6_REPORT.md 必须真实多行 Markdown，物理行数 >=50。
不得是字面量 \n 拼成一行。
JSON 必须 json.load 成功。
失败：STOP M6_BLOCKED_REPORT_RENDERING。
# 32. Candidate gate

只有全部满足才允许 M6 candidate commit：
- six frozen tags unchanged
- M5 champion checkpoint SHA exact
- M5 anchor evidence exact
- Teacher SHA exact
- Search profile correctness PASS
- Search performance PASS
- 131,072 train Student states complete
- 8,192 Teacher-validation states complete
- all train/validation label shards SHA-valid
- three correction runs complete
- each run exact 7,680 optimizer steps
- all run exact M5 start model state
- fresh optimizer per run
- 50/50 correction/anchor mixture exact
- correction learning sanity PASS
- value/afterstate heads bit-identical to M5 baseline
- 2,000 dev baseline + three M6 raw complete
- candidate selected mechanically before validation/test
- 5,000 gameplay validation complete
- one-time Teacher test complete
- paired 10,000 final games complete
- promotion result mechanically derived
- NO_PROMOTION accepted as valid outcome if gates say so
- artifact cleanup PASS
- exact final 553 pytest
- JUnit exact
- report rendering / JSON parse PASS
- no tracked changes outside §27
- no M7 implementation/artifact

否则禁止 M6 CANDIDATE COMPLETE。
# 33. Git / CI candidate state machine

candidate 前：git status --porcelain=v1 -uall 只能出现 §27 七个 execution paths。
staged 必须为空。

显式 git add --：
- src/game2048/m6_state_correction.py
- benchmarks/profile_m6_teacher_search.py
- benchmarks/generate_m6_correction_dataset.py
- benchmarks/run_m6_state_correction.py
- tests/test_m6_state_correction.py
- reports/m6/m6_state_correction.json
- reports/m6/M6_REPORT.md

然后必须满足：
- unstaged tracked 为空
- nonignored untracked 为空
- cached path set 精确等于七个 paths
- git diff --cached --check exit 0

commit 前 git fetch origin。
必须 origin/main == base_head。
若远端推进：STOP M6_BLOCKED_REMOTE_ADVANCED。
不得 rebase/merge/force-push。

candidate commit message 固定：
m6: student state correction candidate

commit / push 后：
- candidate_sha = HEAD
- session = CANDIDATE_PUSHED

按以下条件查 GitHub Actions：
- headSha == candidate_sha
- workflow name == CI
- event == push
若多个匹配 run：取 databaseId 最大者。
暂时 0 个：每 5 秒轮询，最多 24 次。
120 秒仍无：STOP M6_BLOCKED_CANDIDATE_CI_NOT_FOUND。

找到后轮询到 completed。
只有 conclusion=success PASS。
其他 conclusion：STOP M6_BLOCKED_CANDIDATE_CI。
不得 rerun CI；不得现场修第二个 candidate。

candidate report 在 commit 前固定写：
- candidate_sha=PENDING_BY_DESIGN
- candidate_ci=PENDING_BY_DESIGN
- closeout_commit_sha=SELF_NOT_EMBEDDABLE_BY_DESIGN
- closeout_ci=PENDING_BY_DESIGN

candidate CI green 后：session = CLOSEOUT_REPORT_READY。
mandatory closeout 只修改 reports/m6/M6_REPORT.md。
回写 candidate SHA + candidate CI run ID/conclusion。
closeout_commit_sha 保持 SELF_NOT_EMBEDDABLE_BY_DESIGN。
closeout_ci 保持 PENDING_BY_DESIGN_AT_REPORT_COMMIT。
禁止 amend 自引用。
closeout commit 前：
- git fetch origin
- 必须 origin/main == candidate_sha
- dirty/nonignored untracked path 只能精确 reports/m6/M6_REPORT.md
- staged path 必须精确只有 reports/m6/M6_REPORT.md
- git diff --cached --check exit 0

固定 closeout commit message：docs: close M6 candidate report

完成 closeout commit 并 push 后：
- closeout_sha=HEAD
- session=CLOSEOUT_PUSHED
closeout CI 查询固定：
- headSha == closeout_sha
- workflow name == CI
- event == push

0 个时每 5 秒轮询，最多 24 次。
找不到：STOP M6_BLOCKED_CLOSEOUT_CI_NOT_FOUND。
非 success：STOP M6_BLOCKED_CLOSEOUT_CI。
不得 rerun / amend / 第三个 commit。

closeout CI success：session=COMPLETE。

Execution Agent：
- 不得创建 M6 audited tag
- 不得改 master plan 推进 M7
- 不得开始 M7

# 34. STOP / blocker behavior

任意 M6_BLOCKED_*：立即 STOP。
输出必须包含：
- exact blocker code
- session state
- actual triggering values
- HEAD / origin/main
- git status --short
- completed evidence paths
- 正在运行进程状态（若有）
不得为了绕 gate 改 protocol。

runtime exception：M6_BLOCKED_RUNTIME_ERROR
CUDA OOM：M6_BLOCKED_OOM
non-finite：M6_BLOCKED_NUMERIC

禁止自动：
- 减 batch
- 改 lr
- 改 precision
- 少跑 game
- 少跑 seed
- 少跑 epoch
- 改 Teacher
- 降 Search depth

研究结果 NO_PROMOTION 不是 blocker。
不得把 NO_PROMOTION 包装成 runtime fail，也不得为了做出正结果现场调参。

# 35. Independent audit 边界

Execution Agent 完成后必须 STOP for independent M6 audit。

独立 audit 至少重新核对：
1. 六个 frozen tags / M5 checkpoint / Teacher SHA
2. M5 frozen selection 完全未重开
3. Student source states 确由 exact M5 champion pure-NN rollout产生
4. spawn/tie RNG semantics正确且 seed splits无重叠
5. train/validation/test source sampling与 retry provenance
6. all Search label shard schema/count/SHA
7. Teacher depth/semantics未漂移
8. M5 anchor只使用 train split
9. 每 batch 512 correction + 512 anchor
10. 三 run exact same M5 model start + fresh optimizer
11. exact 30 epochs / 7,680 steps
12. value/afterstate heads bit-identical
13. candidate在 validation/test之前锁定
14. dev/validation/final raw gameplay可独立重算
15. paired bootstrap / CI / promotion logic可独立重算
16. one-time Teacher test isolation
17. NO_PROMOTION 或 PROMOTION_ELIGIBLE 与 raw evidence一致
18. exact 553 regression
19. candidate / closeout Git scope + CI
20. artifact cleanup
21. 无 M7 tracked implementation / artifacts/m7

如果 promotion_result = PROMOTION_ELIGIBLE：
只有 independent audit通过后，后续才允许把该 M6 checkpoint提升为新的 frozen Student baseline。

如果 promotion_result = NO_PROMOTION：
independent audit仍可把 M6 作为完整负结果封板，
但后续 frozen Student baseline继续是 M5 champion。

M6 authoritative audited tag 由 independent audit阶段决定和创建。
Execution Agent不得创建。

# 36. 最终输出

全部 execution protocol + candidate CI + closeout CI 成功后，只输出：
M6 CANDIDATE COMPLETE

并列出：
- result = PROMOTION_ELIGIBLE 或 NO_PROMOTION
- M5 baseline checkpoint SHA
- correction train states
- M6 candidate seed
- M6 candidate checkpoint SHA
- Teacher-validation metrics
- development candidate vs M5 mean delta
- gameplay-validation paired mean delta + CI95
- final 10,000 paired mean delta + CI95
- baseline/candidate final mean/median/p10/p90/reach
- Search profile median roots/s
- actual Teacher label wall
- three training walls / samples/s
- exact pytest count
- candidate implementation SHA
- candidate CI run
- report-only closeout SHA
- closeout CI run

然后：STOP for independent M6 audit。
不得进入 M7。
