# M3 独立审计报告

## RESULT

**M3 FINAL AUDITED PASS / FROZEN**

本报告是对 M3 candidate 的独立收口审计，不是 candidate 自述的重复。
权威 M3 implementation commit：

`7c95f9c5f543065b220fb5f9e9114521732cdc0b`

candidate report-only closeout descendant：

`66be70cc08162c61e86d42cc7bfee5a5d49ec186`

authoritative tag：

`m3-teacher-audited-pass`

tag 必须指向上述 implementation commit，而不是 docs-only closeout commit。

## 审计范围

独立核对：
- 总计划与 M3 implementation prompt 的全部 candidate gate；
- checkpoint identity / loader / tuple evaluator；
- exact depth-3 Expectimax correctness；
- Search Performance Unblock；
- calibration；
- 8192-state / 64-game canonical Teacher dataset；
- Student sanity；
- local full pytest；
- GitHub Actions；
- frozen M0/M1/M2 tags 与 Git 边界。

## 独立复算证据

### Checkpoint

本地 checkpoint 重新计算 SHA-256：

`7192719323a073ba2b6b19b62cb7d46ef4aa90ecc8c4ae6baf27ad0c51566a84`

与固定 expected SHA 完全一致。
checkpoint 继续被 `.gitignore` 的 `teacher_checkpoints/**/*.bin` 命中，未进入 Git history。

### Frozen milestones

annotated tags 使用 `^{}` 解引用：
- M0 -> `3f2def1d95f56eff776e671143188947bf64485b`
- M1 -> `e5486017a90eeec4fb9814de7880b3c413dc06dd`
- M2 -> `6a9da5b1bf7c72207acd6e89bc667d439d4823a8`

均未移动。

### Source games 全量重放

使用当前 frozen checkpoint、固定 seeds 20260919..20260982、当前 greedy_1ply policy 独立重放 64 个完整 source games。

结果：
- total decisions / moves: **583,332**
- 8192 / 8192 sampled source states 逐字节一致
- sampled current_score mismatch: **0**
- 64 / 64 game moves 一致
- 64 / 64 final_score 一致
- 64 / 64 terminal max_tile_exp 一致

### Canonical dataset

对 `artifacts/m3/m3_teacher_validation_8192.npz` 独立检查：
- states = **8192**
- games = **64**
- state shape/dtype 正确
- legal_mask 与实际 movement 全量一致
- legal reward 全量一致
- legal afterstate 全量一致
- illegal teacher_value 全为 NaN
- illegal reward 全为 0
- illegal afterstate 全为 zero sentinel
- 所有 legal teacher values finite
- 所有 canonical states 均非 terminal
- decision_depth 唯一值 = 3
- value_semantics 唯一值 = SEARCH_VALUE_RAW_LEAF
- checkpoint SHA metadata 全部一致

split：
- train 51 games / 6528 states
- validation 6 games / 768 states
- test 7 games / 896 states
- game_id overlap = **0**
- split labels 与固定 split function 一致

另外均匀抽取 32 个 canonical states，重新运行 exact decision_depth=3 Expectimax：
- legal masks: equal
- action value max abs error: **0.0**
- best-action disagreement: **0**

### Calibration

从正式 rollout artifact 独立复算：
- states = **128**
- source games = **16**
- rollouts/state = **16**
- total rollouts = **2048**
- unique rollout seeds = **2048**
- rollout policy decisions = **12,219,131**
- continuation = frozen checkpoint native greedy_1ply

L_tuple vs realized future score：
- Pearson = **0.9371498188**
- Spearman = **0.9107060357**

train-only affine fit：
- slope = **1.3018450985**
- intercept = **-69881.4718323**
- validation R² = **0.0442735933**
- test R² = **0.9297664817**

validation/test 稳定性不足，因此不晋升到 CALIBRATED_FUTURE_SCORE。
最终语义保持：
- RAW_TUPLE_HEURISTIC
- FORMAL_STATE_TUPLE_HEURISTIC
- SEARCH_VALUE_RAW_LEAF

该保守结论符合总计划与 M3 prompt。

### Student sanity

确认训练脚本只使用 teacher-best-action cross entropy：
- absolute Q regression = false
- absolute V regression = false
- absolute afterstate regression = false

两次 fixed seed 均满足 M3 固定 PASS 条件：
- train metric 明显改善
- validation 不发散
- validation best-action accuracy 明显高于合法动作随机基线
- 两次 seed 复现同方向学习

D4 consistency 作为 diagnostic 记录，不存在 M3 硬阈值，因此不构成 blocker。

### Search / performance

正式 Search correctness evidence：
- 256 roots / 16 games
- 911 legal action values bit-identical
- max / mean abs error = 0
- best-action disagreement = 0
- player/chance/leaf/cache counts equal

最终 throughput：
- **6.320418629 roots/s**
- historical baseline = 1.127171768 roots/s
- speedup = **5.6073x**
- projected 8192 serial Search = **1296.12 s ~= 21.60 min**

满足 Search Performance Unblock gate。

### Regression / CI

独立审计时重新运行官方本地解释器：

`D:\sd-webui-forge-aki-v1.0\python\python.exe -m pytest -q`

结果：
- **525 passed**
- 0 failed
- 0 skipped
- 0 xfailed
- 6 existing PyTorch Transformer warnings

GitHub Actions：
- candidate run #17 / ID 35397080143: success
- report-only closeout run #18 / ID 35397257131: success

两个 run 的实际 job 均确认：
- Build M2 C++ fast backend: success
- Build M3 C++ tuple backend: success
- Run full pytest suite: success

## AUDIT DECISION

M3 的 checkpoint semantics、Teacher/Search correctness、value-semantics guard、calibration probe、8192-state/64-game dataset、game-level split、Student sanity、Search profiling/performance、local regression、remote CI 与 frozen-stage protection 全部达到正式施工单要求。

未发现需要返工 M3 implementation 的 correctness blocker。

因此：

**M3 FINAL AUDITED PASS / FROZEN**

M4 可以在完成本次 docs/tag closeout并确认 CI green 后成为下一执行阶段。