# M7 正式 Self-play Baseline — 严格施工提示词

> 状态：READY FOR EXECUTION AFTER PLANNING CI / NOT STARTED。
>
> 项目：Safolour/NN-2048-ai
> 正式工作区：D:\CodexTasks\NN-2048-ai
> 当前阶段：M7 正式 Self-play Baseline
> Work-order version：`M7_WO_SELFPLAY_BASELINE_V1`
> planning parent：`ca0d22b109e99db87cf16004a0f4a569fac6cec3`
> 本文件是 M7 的唯一详细施工单；authoritative master plan 始终优先。
> M7 只允许在包含本文件的 planning commit 已 push 且 matching CI success 后启动。
> 本施工单生成、commit、CI 均不等于 M7 training 已开始。

# 0. M7 的唯一目的

M7 是项目第一次正式运行完整 RL / TD self-play baseline。

固定目标：
- 从已审计冻结的 M6 Student baseline 开始；
- 运行 1-step off-policy greedy Double-Q；
- 使用 Learner + EMA Target + Replay + epsilon exploration；
- 对 5% training rows 使用 exact spawn expectation；
- 首次真正训练 Q / V / A 的 value semantics；
- 用独立 gameplay 判断 self-play 是否提升棋力。
M7 不是：
- Teacher promotion；
- Search Correction；
- 高位 Restart Pool；
- architecture sweep；
- model-size sweep；
- reward shaping；
- n-step / TD(lambda)；
- Target hard-update sweep；
- epsilon sweep；
- LR / batch / loss-weight sweep；
- M8 / M9 的提前实现。

Execution Agent 不得自行把“可能更好”的实验塞进 M7。
任何未在本施工单明确允许的研究决策，默认禁止。

# 1. 开工前权威输入

必须完整读取：
1. authoritative master plan；
2. `prompts/M6_IMPLEMENTATION_PROMPT.md`；
3. `reports/m6/M6_AUDIT_REPORT.md`；
4. `reports/m6/M6_REPORT.md`；
5. `src/game2048/m2_models.py`；
6. `src/game2048/m2_rollout_env.py`；
7. `src/game2048/m2_fast_backend.py`；
8. `src/game2048/m2_symmetry.py`；
9. `src/game2048/m4_compare.py`；
10. 本文件。
如存在真实规格冲突：立即 STOP `M7_BLOCKED_SPEC_CONFLICT`。
不得由执行 Agent 自行“挑一个更合理的解释”。

# 2. Frozen provenance

开工必须验证：

`m6-state-correction-audited-pass^{}`
精确解析到：
`bf3b939eef92f575541f134fcdad545c0d2a9fb5`

M6 audit/docs closeout HEAD：
`ca0d22b109e99db87cf16004a0f4a569fac6cec3`

Frozen M6 Student：
- architecture：`ResidualMLP2048`
- parameters：`5,264,710`
- seed：`20263101`
- checkpoint：`artifacts/m6/checkpoints/20263101_final.pt`
- SHA-256：`EF779ADFB3C2BE21631F73BE400149D511CC7903B9DECBC309948F77F6F94FAB`

M0-M6 全部视为 frozen input。
不得回写任何 frozen implementation / report / prompt / tag。
特别禁止：
- 修改 M6 checkpoint；
- 改 M6 candidate selection；
- 改 M6 Teacher provenance；
- 改 M2 C++ movement/LUT；
- 改 M0/M1 environment semantics；
- 重新训练 M5/M6；
- 重新运行 M6 final gameplay 来“确认一下”。

# 3. M7 最关键的 semantic bridge

必须明确区分：

M6 的 `q_head` 虽然结构上叫 Q Head，
但 M5/M6 只用 policy cross-entropy / ranking supervision，
因此 M6 checkpoint 的数值输出只被验证为**动作排序 logits**，
没有被校准为绝对 future-score Q 值。

同样：
M6 的 `value_head` 与 `afterstate_head`
相对初始化保持 bit-identical，从未学过正式 value semantics。

因此 M7 不得：
- 把 M6 logits 数值直接写入 replay 当 frozen Q target；
- 把 M6 V/A 数值当 ground truth；
- 把 Teacher raw Search value 当 FUTURE_SCORE。
M7 的处理方式固定：

1. Learner 参数整体从 exact M6 checkpoint warm-start；
2. Target 参数初始精确复制 Learner；
3. 不重置 backbone / Q / V / A heads；
4. M6 Q 的已有排序能力仅作为有用初始化；
5. 从 M7 第一个 optimizer step 开始，由正式 TD target 逐步赋予 Q/V/A value semantics；
6. promotion 依据仍然是 independent gameplay，不是假设初始 value 已校准。

# 4. Formal value semantics

仍严格沿用 master plan：

状态：
`s -> action a -> afterstate x -> random spawn -> next state s'`

Q：
`Q(s,a)` = 从当前动作开始，之后按当前 greedy target policy 玩，预计还能获得的总游戏分数，包含当前 merge reward。

V：
`V(s)` = 当前正式棋盘按当前 greedy policy 继续玩的未来总分。

A：
`A(x)` = move/merge 已完成但 spawn 尚未发生时的未来总分期望。
固定关系：
- `Q = reward + A`
- `A = E_spawn[V(next policy state)]`
- `V = max_legal Q`

网络三个 head 不做硬公式绑定；通过 targets/loss 学习这些关系。

`gamma = 1.0`，固定，不实验。

terminal next state：
future value = 0，禁止 bootstrap。

非法动作：
任何 action selection / target selection / eval 中都必须 mask。

# 5. Value representation：M7 锁定 symlog

M7 不做 raw/log/symlog sweep。
本 milestone 唯一 baseline 锁定：

`symlog(x) = sign(x) * log(1 + abs(x))`
`symexp(z) = sign(z) * (exp(abs(z)) - 1)`

原因不是改变 value semantics，而是稳定第一次从 policy-logit scale 过渡到数万/数十万 future-score scale。
所有 Bellman 算术必须先回到真实 score 空间：

Target network 输出 `z_Q`
→ `symexp(z_Q)`
→ 在真实分数空间做 spawn expectation / reward addition
→ 最后对完整 target 做 `symlog`
→ 与网络输出计算 loss。

严禁：
- 在 symlog 空间直接做 `reward + transformed_future`；
- 对 transformed values 做概率 expectation 后声称等价于真实 expectation。

由于 symlog 单调，
greedy action selection 可以直接比较网络输出 z，不需要先 symexp。

checkpoint metadata 必须写：
`value_transform = SYMLOG_V1`

后续 M8+ 读取 M7 checkpoint 时必须尊重该 metadata。

# 6. Learner / Target / Champion

Learner：
- 唯一接受梯度更新的网络。
Target：
- 初始化 = exact Learner；
- 无 optimizer；
- 无独立 self-play；
- 每个 optimizer step 后做 EMA；
- `tau = 0.005`；
- 更新：`theta_T = (1-tau)*theta_T + tau*theta_L`；
- stop-gradient。

Champion：
- 整个 M7 execution 期间 frozen Champion 是 M6 baseline；
- 不在 formal training 中途“自动晋级”；
- candidate 只能在全部 formal development 完成后锁定；
- independent validation/final 通过后才得到 `PROMOTION_ELIGIBLE`。

为保护 semantic transition：
前 `2,000 optimizer steps` 的 self-play behavior actor 固定使用 frozen M6 Champion + epsilon exploration。
从 optimizer step 2,000 之后，behavior actor 切换为当前 Learner + epsilon exploration。
该切换点固定，不实验。

# 7. Environment / reward / policy

正式 collector 必须复用已审计 M2 production movement/legal path：
`scalar+row-lut`
不得退回逐 board Python movement。
正式 collector 使用 `M2RolloutBatchEnv` 或等价、但不得修改 frozen M2 文件的 M7 wrapper。

固定 collector env count：
`16,384`

如果该固定配置 OOM / 系统内存不足：
STOP `M7_BLOCKED_RESOURCE_LIMIT`。
不得自动改成 8192 或偷偷换配置。

reward：
只允许真实 2048 merge score。
禁止 survival reward、empty-cell reward、corner reward、monotonicity reward 等 shaping。

training behavior：
epsilon-greedy。

`epsilon = 0.05`，整个 M7 formal training 固定，不 decay。

epsilon 随机动作：
在当前 legal actions 中均匀随机。

greedy tie tolerance：
`1e-7`
并列 legal action 均匀随机。

Validation / Final：
epsilon = 0。
# 8. RNG 分离

每个 formal training seed S 必须使用独立 RNG streams，禁止一个 RNG 同时承担所有用途。

至少分离：
- environment spawn RNG；
- exploration RNG；
- greedy tie RNG；
- replay sampling RNG；
- D4 augmentation RNG；
- exact-spawn-row selection RNG。

推荐固定 derivation：
- spawn：`S + 1000`
- exploration：`S + 2000`
- tie：`S + 3000`
- replay：`S + 4000`
- D4：`S + 5000`
- exact-row：`S + 6000`

统一使用 NumPy `PCG64`。
PyTorch CPU/CUDA RNG 也必须按 S 初始化并 checkpoint。

Resume 必须恢复所有 RNG state，而不是只重新 seed。

# 9. Replay Buffer

M7 baseline：
`capacity = 5,000,000 transitions`

固定 FIFO / Ring Buffer。
禁止 prioritized replay、reservoir sampling、importance weights。

存储必须是 compact struct-of-arrays，禁止 5M 个 Python object。

建议使用 `artifacts/m7/replay/<seed>/` 下的 NumPy memmap arrays，至少保存：
- state uint8 [capacity,16]
- action uint8
- reward int64
- afterstate uint8 [capacity,16]
- next_state uint8 [capacity,16]
- terminal bool
- game_id int64
- producer_step int64
- source_type uint8
- max_tile_exp uint8
- insertion_id int64

source_type 在 M7 formal TD replay 固定为 SELFPLAY。

不得永久保存旧模型 Q/V/A 数值作为真理。
每次 replay sample 都使用当前 Learner + Target 重算 bootstrap targets。

# 10. Replay durability / resume

Replay files 是 artifacts，不进 Git。
checkpoint 记录：
- capacity
- count
- write pointer
- insertion counter
- replay schema/version
- replay file sizes / paths
- 一个完整 environment batch 的 minimum-age boundary。

memmap 在 checkpoint 时 flush。
若 crash 后磁盘中存在 checkpoint pointer 之后的残留写入：
恢复时以 checkpoint pointer/count 为权威，后续覆盖即可。
不得把 checkpoint 之后未提交的 replay rows 自动算作正式数据。

已完成 checkpoint 的 replay metadata 与文件 size/schema 不一致：
STOP `M7_BLOCKED_REPLAY_RESUME_MISMATCH`。

# 11. Replay warmup / minimum age / UTD

正式 gradient update 前至少：
`100,000 transitions`

由于 collector batch 固定为 16,384，formal warmup 必须完整收集 7 个 batch：
`114,688 transitions`
然后才允许第一次 optimizer.step。
不得在第 7 个 batch 中途截断到“刚好 100,000”。

warmup 阶段：
- 只 self-play collection；
- 不允许 optimizer.step；
- Learner/Target 不变；
- behavior = frozen M6 Champion + epsilon 0.05。

Minimum Replay Age：
最新插入的**一个完整 collector batch**不得进入普通 replay sampling。

稳定训练后 UTD 固定：
`1.0 training sample per newly inserted environment transition`
实现方式：
- batch size = 1024；
- 每次插入 collector batch 后累计 new transitions；
- 每累计 1024 new transitions，执行 1 optimizer step；
- 不因机器快慢动态改变 UTD；
- 不额外“补训练”。

# 12. M7 batch source

Master plan 的 80/10/10 baseline 中：
- Teacher/Search Correction 在 M7 不启用；
- High-tile/Restart Pool 在 M7 不启用（属于 M8）；
- 缺口按 master plan 回填普通 Replay。

因此 M7 formal TD batch 最终固定为：
`100% Self-play Replay`

不得把 M6 Teacher raw-search values 混入绝对 TD regression。
不得加入 M6 policy CE anchor。
不得加入 M5 anchor CE。
这样 M7 测量的是完整 self-play TD baseline 本身，而不是再次做 Teacher distillation。

# 13. Replay sampling / D4

每个 optimizer batch：
`1024 rows`

从 eligible replay rows 中 uniform sample without replacement within the batch。
不同 optimizer steps 之间允许重复采到历史 row。
每个 sampled transition 独立随机 D4 transform id 0..7。

同一个 transition 的：
- state
- action
- afterstate
- next_state
必须使用同一个 D4 transform；
reward / terminal 不变。

必须复用已审计 `m2_symmetry` action transform。
禁止只变 board 不变 action。

Evaluation 不做 D4 ensemble。

# 14. Double-Q target：普通 spawn row

对于 sampled transition `(s,a,r,x,s',terminal)`：

在 next state `s'`：
1. Learner 输出 `z_L(s',.)`
2. legal mask
3. `a* = argmax_legal z_L(s',.)`
4. Target 输出 `z_T(s',a*)`
5. 若 terminal：`G_real = 0`
6. 否则：`G_real = symexp(z_T(s',a*))`

普通 sampled-spawn row：
`y_A_real = G_real`
`y_Q_real = r + y_A_real`

再：
`y_A = symlog(y_A_real)`
`y_Q = symlog(y_Q_real)`
# 15. Exact Spawn 5%

每个 1024-row optimizer batch：
精确选择 `51 rows` 做 exact spawn expectation。
这是固定约 5% baseline，不做比例 sweep。

51 rows 由 exact-row PCG64 在 batch row indices 中 sample without replacement。
其余 973 rows 使用 stored sampled next_state。

对 exact row 的 afterstate x：
- 枚举全部合法 spawn outcomes；
- 位置均匀；
- tile 2 = 0.9；
- tile 4 = 0.1；
- outcome probability 精确归一；
- 对每个 next state 独立做 Learner legal argmax + Target evaluation；
- terminal outcome future = 0；
- 在真实 score 空间做概率加权。

`y_A_real = sum_i p_i * G_real(s'_i)`
`y_Q_real = r + y_A_real`

必须批处理所有 exact outcomes，禁止每个 outcome 单独 GPU forward。
# 16. Value target

对当前 state s：

Learner：
`a_s* = argmax_legal z_L(s,.)`

Target：
`V_real_target = symexp(z_T(s,a_s*))`

然后：
`y_V = symlog(V_real_target)`

Target forward / action selection 全部 stop-gradient。

注意：
V target 不使用当前 behavior action；
它跟随当前 greedy policy 定义。

# 17. Loss / optimizer

Q loss：
只监督 replay 中实际 executed legal action 的 `z_Q(s,a)` vs `y_Q`。

V loss：
`z_V(s)` vs `y_V`。

A loss：
`z_A(x)` vs `y_A`。

三个基础 loss 全部使用：
`SmoothL1Loss(beta=1.0)`
总 loss 固定：
`L = 1.0*L_Q + 0.5*L_V + 0.5*L_A`

M7 不 sweep loss weights。

Optimizer 固定：
- AdamW
- lr = `3e-4`
- weight_decay = `1e-4`
- betas = `(0.9, 0.999)`
- eps = `1e-8`
- batch = `1024`
- grad clip global norm = `1.0`
- no scheduler
- no early stopping
- foreach=False
- fused=False
- capturable=False

Gradient training：
BF16 autocast。
target arithmetic / symlog/symexp / loss accumulation 使用 FP32。

Behavior actor inference：
FP32 + `torch.inference_mode()`。
# 18. EMA

Target 初始 state_dict 必须与 Learner bit-identical。

每个 successful optimizer.step 后恰好执行一次 EMA。
optimizer.step 失败/non-finite 时不得推进 EMA/global step。

EMA：
`tau=0.005`

所有 floating parameters 一致更新。
无可训练 running-stat buffer；若未来发现 buffer，必须显式复制/定义，不得静默忽略。

Target 始终 `eval()` / no-grad。

# 19. Formal training seeds / budget

正式 training seeds：
- `20264101`
- `20264102`
- `20264103`

每个 seed：
- exact M6 checkpoint fresh load；
- fresh optimizer；
- fresh Target = Learner；
- fresh replay；
- fresh RNG streams；
- 不允许跨 seed warm-start；
- 不允许共享 replay。

每个 seed 固定：
`60,000 optimizer steps`
正式 milestone checkpoints：
- step 20,000
- step 40,000
- step 60,000

为 resume：
每 2,000 optimizer steps 写一次 atomic latest checkpoint。

latest 只用于同一 formal run resume。
milestone checkpoint 永久保留，不得被 latest 覆盖。

不得因为 development 表现好/坏提前结束 60k formal budget。
不得追加到 80k/100k“再看看”。

# 20. Checkpoint exact-resume contract

每个 latest/milestone checkpoint 至少保存：
- Learner model_state
- Target model_state
- optimizer state
- global optimizer step
- global inserted transitions
- UTD accumulator
- actor mode（M6 Champion / Learner）
- environment boards/scores/legal
- current per-env game_id
- episode returns/moves/max-tile diagnostics
- environment spawn RNG state
- exploration/tie/replay/D4/exact-row RNG states
- Python / NumPy / torch CPU / torch CUDA RNG states
- replay count/write pointer/insertion counter/schema
- formal seed
- config
- work-order version
- M6 tag/checkpoint SHA
- source commit
- value_transform
- latest completed development checkpoint info

atomic temp -> rename。

Resume：
- metadata 全匹配才允许；
- 只从最后一个完整 checkpoint；
- 不重复已完成 optimizer step；
- 不跳过应执行 step；
- incomplete temp checkpoint 删除后从上一完整 checkpoint继续。

metadata mismatch：
STOP `M7_BLOCKED_RESUME_METADATA_MISMATCH`。

# 21. Learning sanity

三个 formal runs 都必须：
- all losses finite；
- gradients finite；
- backbone updated；
- Q head updated；
- V head updated；
- A head updated；
- Target 与 Learner 不完全相同但持续 EMA 跟随；
- replay sample legal action consistency 100%；
- terminal bootstrap 0；
- no illegal argmax。

不得用“V/A 没用到 final inference”作为允许它们不训练的理由。
# 22. Development evaluation

固定 M7 development game seeds：
`20311001..20313000`
恰好 2,000 games。

先对 frozen M6 baseline 只运行一次 2,000 games。

然后对每个 formal seed 的：
- 20k
- 40k
- 60k
三个 milestone checkpoint 各完整运行 2,000 games。

总计 9 个 M7 milestone development results。

Evaluation contract：
- pure NN single forward；
- Q head only；
- transformed output直接用于 ranking；
- FP32；
- epsilon=0；
- no Teacher/Search；
- no D4 ensemble；
- legal mask；
- tie tolerance `1e-7`；
- M4/M5/M6 canonical independent spawn/tie RNG contract。

所有模型使用完全相同的 paired game seeds。
# 23. Candidate selection

只有 9 个完整 M7 development results 全部完成后才允许锁 candidate。

机械规则：
1. 最大完整 2,000-game development mean；
2. exact mean tie：较小 training seed；
3. 同 seed exact tie：较小 optimizer step。

禁止按：
- training loss；
- Q/V/A calibration；
- high tile rate；
- 某个 rolling window；
- validation/final；
- wall-clock
改选 candidate。

candidate lock 后永久固定，禁止 reselection / retraining。

Development gate：
candidate mean 必须严格 > frozen M6 baseline mean（同 2,000 seeds）。

若不满足：
result = `NO_PROMOTION`
不消耗 validation/final seeds；禁止重训/换 candidate。
随后直接进入 P8/P9/P10 + Git/CI closeout，把完整负结果封成 M7 candidate evidence。
# 24. Independent Champion Validation

只有 candidate lock + positive development gate 后执行。

Validation seeds：
`20314001..20364000`
恰好 50,000 paired games。

只评：
- frozen M6 Champion
- locked M7 candidate

bootstrap：
- paired score delta
- 10,000 resamples
- PCG64 seed = `20267210`

Validation PASS：
- paired mean delta > 0
- bootstrap CI95 lower bound > 0

若 FAIL：
result = `NO_PROMOTION`
跳过 Final；不得调参、换 candidate、追加训练、查看别的 checkpoint 后重新选。
随后进入 P8/P9/P10 + Git/CI closeout。

# 25. Final paired gameplay

只有 Validation PASS 后执行。

Final seeds：
`20370001..20380000`
恰好 10,000 paired games。
bootstrap：
- 10,000 resamples
- PCG64 seed = `20267220`

Final PASS：
- paired mean delta > 0
- CI95 lower bound > 0

记录：
- mean / median / p10 / p90
- mean moves
- max tile distribution
- reach 2048/4096/8192/16384/32768/65536
- wins/losses/ties
- paired delta + CI95

若 Final FAIL：
result = `NO_PROMOTION`，随后照常完成 P8/P9/P10 + Git/CI closeout。

只有：
- development positive
- validation PASS
- final PASS
- all learning sanity
- correctness/provenance/regression/cleanup PASS
才可：
`PROMOTION_ELIGIBLE`

PROMOTION_ELIGIBLE 仍只是 candidate，必须 STOP for independent M7 audit。
# 26. No Teacher / Search / high-tile leakage

M7 formal training / selection 禁止调用：
- M6 promoted Teacher
- Expectimax
- M3 tuple heuristic
- M6 correction label values
- high-tile restart pool
- M8/M9 数据

Teacher checkpoint provenance仍记录在项目中，但 M7 baseline不消费它。

如果代码路径意外触发 Teacher/Search：
STOP `M7_BLOCKED_SCOPE_LEAK`。

# 27. Performance / progress

长任务必须有可读 progress。

Training 至少每 1,000 optimizer steps输出：
- formal seed
- optimizer step / 60,000
- inserted transitions
- replay count / capacity
- actor mode
- elapsed
- optimizer steps/s
- environment transitions/s
- training samples/s
- rolling completed-game mean
- Q/V/A loss
- total loss
- ETA
每个 development/validation/final gameplay：
completed/total、elapsed、games/s、rolling mean、ETA。

正式训练开始前做一次 bounded performance smoke：
- 固定 16,384 envs；
- scratch replay；
- 不写 formal evidence；
- 至少覆盖 collection、replay insert/sample、ordinary target、exact-spawn target、backprop、EMA；
- 5 optimizer steps correctness smoke；
- 再做 200 optimizer-step timing profile。

profile 必须记录：
- actor forward wall
- env step wall
- replay I/O wall
- target forward wall
- exact-spawn wall
- train forward/backward wall
- optimizer/EMA wall
- residual Python orchestration wall
- peak RAM / VRAM

`orchestration_share <= 0.50`
否则 STOP `M7_BLOCKED_PERFORMANCE_PROFILE`。

不允许 Agent 自动改 env count/batch/precision 来绕 gate。

P2 PASS 后立即计算并写入 provenance：
- `src/game2048/m7_selfplay.py` SHA-256
- `benchmarks/profile_m7_selfplay.py` SHA-256
- `benchmarks/run_m7_selfplay.py` SHA-256
- `tests/test_m7_selfplay.py` SHA-256

从第一个 formal replay row 开始，上述四个文件进入 execution freeze。
每个 formal seed 启动/恢复、每次 development/validation/final、candidate commit 前都必须重验 SHA。
任何变化：STOP `M7_BLOCKED_CODE_DRIFT`。
不得用 seed1 的数据配合修改后的 seed2 实现。

恢复长任务前必须检查 session 与活跃 PID/process。
若原进程仍正常运行：只监控，禁止启动第二个重复任务。
若已退出：只从完整 checkpoint/session 精确 RESUME。

# 28. Formal artifacts

所有 M7 大型/中间结果放：
`artifacts/m7/`

建议：
- `artifacts/m7/progress/session.json`
- `artifacts/m7/provenance.json`
- `artifacts/m7/profile.json`
- `artifacts/m7/replay/<seed>/`
- `artifacts/m7/checkpoints/<seed>/`
- `artifacts/m7/games/development/`
- `artifacts/m7/games/validation/`
- `artifacts/m7/games/final/`
- `artifacts/m7/selection.json`
- `artifacts/m7/promotion.json`
- `artifacts/m7/pytest.xml`

tracked reports：
- `reports/m7/m7_selfplay.json`
- `reports/m7/M7_REPORT.md`

禁止在仓库根目录创建：
- M7_REPORT.md
- m7_*.json
- scratch/temp/debug output

所有正式 JSON/NPZ/PT：
temp -> atomic rename。
# 29. Session states

固定高层状态：
- P0_PASSED
- P1_READY
- P2_PROFILED
- WARMUP_RUNNING
- TRAINING_RUNNING
- TRAINING_DONE
- DEVELOPMENT_RUNNING
- DEVELOPMENT_DONE
- CANDIDATE_LOCKED
- VALIDATION_RUNNING
- VALIDATION_DONE
- FINAL_RUNNING
- FINAL_DONE
- REPORT_WRITTEN
- CANDIDATE_PUSHED
- CANDIDATE_CI_PASSED
- CLOSEOUT_PUSHED
- COMPLETE

NO_PROMOTION 时必须记录触发 gate，并 STOP；不得继续消耗未授权 test seeds。

# 30. Exact tracked execution scope

M7 execution 只允许新增/修改以下 6 个 tracked paths：

1. `src/game2048/m7_selfplay.py`
2. `benchmarks/profile_m7_selfplay.py`
3. `benchmarks/run_m7_selfplay.py`
4. `tests/test_m7_selfplay.py`
5. `reports/m7/m7_selfplay.json`
6. `reports/m7/M7_REPORT.md`

Execution Agent 禁止修改：
- 本 planning prompt
- master plan
- M0-M6 source/tests/reports/prompts
若需要第 7 个 execution tracked path：
STOP `M7_BLOCKED_TRACKED_SCOPE`。
不得自行扩大 scope。
# 31. M7 exact tests

`tests/test_m7_selfplay.py` 必须精确包含 15 个非 parametrized tests：

1. `test_symlog_roundtrip_and_real_score_target_arithmetic`
2. `test_double_q_selects_with_learner_and_evaluates_with_target`
3. `test_terminal_transition_has_zero_bootstrap`
4. `test_exact_spawn_expectation_matches_reference_enumeration`
5. `test_d4_transition_transform_keeps_action_consistent`
6. `test_epsilon_policy_is_legal_and_seed_stable`
7. `test_replay_fifo_wrap_and_minimum_age`
8. `test_replay_resume_metadata_and_pointer_are_exact`
9. `test_warmup_performs_no_optimizer_step`
10. `test_utd_schedule_is_exactly_one_sample_per_inserted_transition`
11. `test_ema_target_update_matches_tau`
12. `test_fresh_run_loads_exact_m6_learner_and_target_with_fresh_optimizer`
13. `test_checkpoint_resume_restores_rng_env_replay_and_steps`
14. `test_candidate_selection_and_paired_bootstrap_are_deterministic`
15. `test_validation_and_final_cannot_run_before_candidate_lock`

不得用 parametrize 改变 collected count。
# 32. P0 -> P10 execution order

P0：
- verify six prior audited tags + M6 tag；
- verify M6 checkpoint SHA；
- HEAD/origin planning commit一致；
- worktree clean；
- full baseline pytest：必须 exact `553 passed`。

P1：
实现仅前四个 execution code/test paths。
不得开始 formal training。

P1.1：
只运行 M7 test file；
必须 exact `15 passed`。

P2：
scratch correctness/performance smoke；
5-step correctness + 200-step timing；
不得复用 scratch model/replay 到 formal training。

P3：
建立 provenance/session/replay schema；
重新 load exact M6；
开始三个 formal seeds。

P4：
完成 seed 20264101 / 20264102 / 20264103 各 60k。

P5：
完成 M6 baseline + 9 milestone development eval；
机械 candidate lock。
P6：
若 development positive，执行 50k paired Validation；
否则锁定 NO_PROMOTION，跳过 P7，直接进入 P8/P9/P10。

P7：
若 Validation PASS，执行 10k final paired gameplay；
否则锁定 NO_PROMOTION，跳过 final，进入 P8/P9/P10。

P8：
机械 promotion decision。

P9：
生成 machine report + Markdown report；
cleanup scratch/latest temp residue。
正式 milestone/candidate checkpoints保留。

P10：
full pytest 必须 exact：
`568 passed`
`0 failed / 0 errors / 0 skipped / 0 xfailed`

写 JUnit 并独立解析计数。

# 33. Git candidate / closeout

正式 execution candidate commit 前：
- `git diff --check`
- dirty set 精确只有允许的 6 paths
- fetch origin
- origin/main 仍精确等于 M7 planning commit

candidate commit message 固定：
`m7: self-play baseline candidate`
push 后必须查询 matching：
- workflow = CI
- event = push
- headSha = candidate SHA
- conclusion = success

CI fail：
STOP，不得 amend formal evidence 后假装同一 candidate。

candidate CI success 后：
只允许 report closeout update：
`reports/m7/M7_REPORT.md`

closeout commit message：
`docs: close M7 candidate report`

push + matching CI success。

Execution Agent：
- 不得创建 M7 audited tag；
- 不得修改 master plan 推进 M8；
- 不得进入 M8。

# 34. NO_PROMOTION / blockers

`NO_PROMOTION` 是完整研究结果，不允许自动重训。
触发后必须保留 evidence，STOP for independent audit / M7 revision decision。

硬 blocker 示例：
- `M7_BLOCKED_SPEC_CONFLICT`
- `M7_BLOCKED_FROZEN_PROVENANCE`
- `M7_BLOCKED_RESOURCE_LIMIT`
- `M7_BLOCKED_SCOPE_LEAK`
- `M7_BLOCKED_REPLAY_RESUME_MISMATCH`
- `M7_BLOCKED_RESUME_METADATA_MISMATCH`
- `M7_BLOCKED_PERFORMANCE_PROFILE`
- `M7_BLOCKED_CODE_DRIFT`
- `M7_BLOCKED_STEP_ACCOUNTING`
- `M7_BLOCKED_TRACKED_SCOPE`
- `M7_BLOCKED_NUMERIC`
- `M7_BLOCKED_OOM`
- `M7_BLOCKED_RUNTIME_ERROR`
- `M7_BLOCKED_CI`

任何 blocker：
立即输出 exact state / values / paths / HEAD / origin/main / git status。
不得自行降低 gate。

# 35. 最终输出

全部 protocol + candidate CI + closeout CI 成功后：
`M7 CANDIDATE COMPLETE`

必须报告：
- result = PROMOTION_ELIGIBLE 或 NO_PROMOTION
- frozen M6 checkpoint SHA
- selected M7 training seed + optimizer step
- candidate checkpoint SHA
- development baseline/candidate means
- validation 50k paired delta + CI95（若执行）
- final 10k paired delta + CI95（若执行）
- final mean/median/p10/p90/reach
- Q/V/A learning sanity
- replay capacity / peak count
- total inserted transitions
- exact-spawn fraction
- three training walls / transitions/s / samples/s
- performance orchestration share
- exact pytest/JUnit counts
- candidate implementation SHA + CI run
- report closeout SHA + CI run
- final git status

然后：
STOP for independent M7 audit。

不得进入 M8。
