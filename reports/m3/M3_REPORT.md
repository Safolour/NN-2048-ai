# M3 Teacher 小规模验证报告

## 1. RESULT

**M3 CANDIDATE COMPLETE**

本结论表示 M3 施工单要求的 checkpoint/Teacher semantics、exact depth-3 Search、calibration probe、8192-state / 64-game 小规模 Teacher dataset、game-level split、ResidualMLP Student sanity、Search profile、full regression 与 remote CI gate 已具备候选完成条件。

这**不是** M3 AUDITED PASS。本阶段不创建 M3 tag、不把 M3 标记为 frozen、不进入 M4/M5；candidate 提交后必须 STOP，交回独立审计。

## 2. STARTING STATE

- 本轮从 Search Performance Unblock closeout 后恢复。
- resume HEAD / origin/main：`962652d91610f246c589fc0ff06c0ac856653ac2`
- Search Performance Unblock implementation：`4ceebef9a028ec084c18248d41d827c804e333de`
- frozen M0/M1/M2 均未移动。
- 本轮没有重做 checkpoint copy、loader format gate、Search semantics 或 256-state corpus。
- 用户自己的主计划 / prompt 修改及旧 M1/M2 临时 JSON 删除保持未 stage、未 restore。

## 3. CHECKPOINT

- runtime path：`teacher_checkpoints/m3/ordinary_td_comparator_ep4800000_7192719323a0.bin`
- size：536,871,168 bytes
- SHA-256：`7192719323a073ba2b6b19b62cb7d46ef4aa90ecc8c4ae6baf27ad0c51566a84`
- expected/local/upstream SHA：一致
- format：`U2048NT6` v2
- stage_count：1
- stage_thresholds：`[0]`
- tuple schema：8 patterns × 6 cells，D4 8 symmetries，4-bit feature alphabet，float32 weights
- training provenance：ordinary-TD comparator episode 4,800,000
- loader source：`src/game2048/m3_tuple_teacher.py`
- performance backend：`cpp/m3_tuple_backend/` + `src/game2048/m3_tuple_backend.py`
- binary 保持 Git-ignored，未进入 Git history。

## 4. TUPLE EVALUATOR

board encoding：

- formal board：`uint8[16]` exponent board
- tuple feature：每个 exponent 按 checkpoint ABI 饱和到 4-bit 0..15
- pattern-major -> symmetry-minor 固定 64-slot 累加顺序

256-state raw probe：

- count：256
- min：-19,792.9238
- max：27,937.7227
- mean：-901.7502
- std：6,177.5561

deterministic / differential：

- 10,019-board Python vs C++ evaluator：bit-identical，max abs error 0.0
- C++ no-prefetch vs prefetch：bit-identical
- upstream 256-board `2048_ai.exe infer --agent m6` anchor：max abs error 0.0，action disagreement 0
- exponent 63/64/127/255 safety gate：PASS
- batch 1/16/128/2048 consistency：PASS
- memory-owner/read-only/repeated-call safety：PASS

## 5. SEARCH SEMANTICS

- algorithm：formal-state Expectimax
- decision_depth：3
- chance nodes 不消耗 decision depth
- spawn：空格均匀 × 90% exponent-1 / 10% exponent-2
- action edge：immediate merge reward exactly once
- leaf：最后 chance 后的 formal state
- leaf evaluator：`tuple_afterstate_greedy_1ply_adapter`
- leaf adapter：`L_tuple(s)=max_a [reward(s,a)+V_tuple(afterstate(s,a))]`
- terminal future：0
- cache key：node type + remaining decision depth + board bytes
- Search value classification：`SEARCH_VALUE_RAW_LEAF`

没有 chance sampling、beam、top-k spawn、低概率裁剪、降 depth、quantization 或 checkpoint 替换。

## 6. SEARCH CORRECTNESS

固定 256-root / 16-game semantic corpus：

- 911 legal action values
- legal mask：完全一致
- Python reference vs final C++-accelerated path：bit-identical
- max abs error：0.0
- mean abs error：0.0
- best-action disagreement：0
- player_nodes：145,588
- chance_nodes：477,658
- leaf_calls：1,600,220
- move_calls：542,996
- chance_outcomes：1,745,808
- cache lookups：623,246
- cache hits：295,948

所有 node/cache counts 与 Python oracle 一致。

## 7. TEACHER SAMPLE SCHEMA

canonical sample 保存：

- `state uint8[16]`
- `teacher_value float64[4]`
- `reward int64[4]`
- `afterstate uint8[4,16]`
- `legal_mask bool[4]`
- `game_id int64`
- `step_index int32`
- `current_score int64`
- `max_tile_exp uint8`
- teacher/checkpoint/depth/data-source/value/search/leaf metadata

非法动作固定：

- legal_mask=False
- teacher_value=NaN
- reward=0
- afterstate=all-zero sentinel

8192-state artifact 已逐项验证上述 sentinel 与合法 value finite 规则。

## 8. DATASET

canonical artifact：

`artifacts/m3/m3_teacher_validation_8192.npz`

来源：

1. frozen tuple checkpoint 原生 `greedy_1ply` policy 推进 64 个完整游戏到 terminal；
2. 每局从完整轨迹均匀抽取 128 个非终局 formal states；
3. 对选出的 8192 states 使用 exact decision_depth=3 Expectimax 打 Teacher label。

规模：

- states：8,192
- complete source games：64
- states/game：128
- source-game moves：min 1,065 / median 10,641 / max 13,893
- canonical artifact：未做 D4 augmentation

game-level split，seed=20260919：

- train：51 games / 6,528 states
- validation：6 games / 768 states
- test：7 games / 896 states
- train/validation/test game_id overlap：0

D4 augmentation 只在 Student train split 之后在线进行，不覆盖 canonical artifact。

## 9. VALUE SEMANTICS

最终保持严格区分：

- tuple raw：`RAW_TUPLE_HEURISTIC`
- formal-state leaf：`FORMAL_STATE_TUPLE_HEURISTIC`
- depth-3 Search：`SEARCH_VALUE_RAW_LEAF`
- realized rollout return：`FUTURE_SCORE`

没有把 raw tuple / formal leaf / Search action value 偷换成绝对 Student Q/V/A truth。

最终没有任何 raw/search value 被升级成 `FUTURE_SCORE` 或 `CALIBRATED_FUTURE_SCORE`。

## 10. CALIBRATION

§18 固定 probe：

- 128 semantic/profile states
- 来自 16 game_ids，每局取 8 个 evenly-spaced slots
- 16 stochastic rollouts/state
- total rollouts：2,048
- continuation policy：frozen checkpoint 原生 `greedy_1ply`
- 每一步 policy：argmax immediate_reward + V_tuple(afterstate)
- tie-break：最低 action index，与 frozen adapter 一致
- spawn：真实 90/10 + 空格均匀
- rollout policy decisions：12,219,131
- wall：126.19 s
- throughput：96,834 policy decisions/s

all-128 correlation：

- raw selected-afterstate V vs realized future score：Pearson 0.9371 / Spearman 0.9107
- formal leaf L_tuple vs realized future score：Pearson 0.9371 / Spearman 0.9107
- depth-3 best Search value vs greedy_1ply return（diagnostic only）：Pearson 0.9367 / Spearman 0.9086

formal-leaf affine fit，仅 train split 拟合：

`future_score_hat = 1.3018451 * L_tuple - 69881.4718`

train：
- R² 0.8541
- MAE 17,503.65
- RMSE 22,949.67

validation：
- Pearson 0.6046
- Spearman 0.6190
- R² **0.0443**
- MAE 16,726.93
- RMSE 21,651.72

test：
- Pearson 0.9716
- Spearman 0.9567
- R² 0.9298
- MAE 11,314.34
- RMSE 16,069.65

结论：全局相关性很高，但独立 game-level validation 与 test 稳定性不一致，因此 affine mapping **不晋升为 CALIBRATED_FUTURE_SCORE**。tuple/formal/Search 语义保持 RAW；M3 按施工单转入 policy/ranking-only Student supervision。

旧版 repeated depth-3-to-terminal rollout diagnostic 已明确废止；其遗留 7 complete + 1 interrupted evidence 保存在 `artifacts/m3/calibration_viability_progress/`，不用于本 calibration fit。


## 11. STUDENT SANITY

固定模型：现有 `ResidualMLP2048`，没有修改 architecture。

由于 Search values 仍为 `SEARCH_VALUE_RAW_LEAF`：

- supervision：teacher-best-action cross entropy / classification
- absolute Q regression：禁止并未执行
- absolute V regression：禁止并未执行
- absolute afterstate regression：禁止并未执行
- legal-action ranking：evaluation metric

训练协议：

- device：RTX 5060 Laptop GPU
- PyTorch：2.9.1+cu130
- epochs：30
- batch：1024
- optimizer：AdamW
- lr：3e-4
- weight decay：1e-4
- train：split 后 random D4 augmentation
- validation/test：canonical data
- fixed seeds：20260919、20260920
- test split 只在两次固定训练完成后读取，不参与调参

合法动作随机基线：

- train：29.43%
- validation：29.34%
- test：29.28%

seed 20260919：

- final validation CE：1.1138
- validation teacher-best legal-action accuracy：45.18%
- validation legal pairwise ranking accuracy：78.37%
- test best legal-action accuracy：46.76%
- test legal pairwise ranking accuracy：78.11%
- validation D4 legal-argmax consistency：61.38%

seed 20260920：

- final validation CE：1.1262
- validation teacher-best legal-action accuracy：41.67%
- validation legal pairwise ranking accuracy：77.01%
- test best legal-action accuracy：45.65%
- test legal pairwise ranking accuracy：76.92%
- validation D4 legal-argmax consistency：59.17%

两次 seed 均满足：

- train metric 明显改善
- validation 不发散
- validation accuracy 明显高于合法动作随机基线
- fixed-seed rerun 复现同方向结果

因此：**M3_STUDENT_SANITY_PASS**。

CUDA 训练请求了 deterministic algorithms，但 cuBLAS 提示未设置 bitwise-deterministic workspace；本 gate 只声明两次固定 seed 的“同方向可复现”，不声称逐 bit 相同。

## 12. SEARCH PROFILE

最终 exact 256-root profile：

- wall：40.5036 s
- root decisions/s：**6.320418629**
- nodes/s：54,895.45
- mean：0.157902 s/root
- median：0.142790 s/root
- p95：0.373139 s/root
- max：0.519936 s/root
- cache hit rate：47.4849%

分项：

- formal-leaf / tuple evaluator：17.9425 s，约 44.4%
- move generation：10.7185 s，约 26.5%
- chance expansion：5.4905 s，约 13.6%
- recursive Python orchestration residual：5.8422 s，约 14.5%
- hash：0.4278 s

历史 baseline：

- 1.127171768 roots/s
- 227.117 s / 256 roots
- formal-leaf/evaluator share ~86.6%

最终 speedup：**5.6073×**。

没有已知的 Python/NumPy 数量级损失继续被当成正常成本接受。

## 13. PERFORMANCE GATE

Search Performance Unblock：**PASS**

- exact 256-root throughput：6.3204 roots/s >= 5.0
- projected 8192-state serial Search：1296.12 s ≈ 21.60 min <= 30 min
- production dataset generation 使用 8-worker exact Search，实际 labeling steady-state 约 42 roots/s
- depth、chance、leaf、backup、checkpoint 均未改变

M3 Search 性能不再是 blocker。

## 14. PYTEST

最终 candidate full pytest 必须使用：

`D:\sd-webui-forge-aki-v1.0\python\python.exe`

candidate full regression：

- **525 passed**
- **0 failed**
- **0 skipped**
- **0 xfailed**
- 6 warnings（既有 PyTorch Transformer nested-tensor warning）
- wall：23.55 s

本地 candidate regression gate：PASS。

## 15. CI

CI workflow 已在 Search Unblock 阶段支持：

1. build frozen M2 C++ fast backend
2. build M3 C++ tuple backend
3. run full pytest on Ubuntu / Python 3.12

Search Unblock implementation 与 docs closeout CI 均已 green。

M3 candidate remote CI：

- candidate commit：`7c95f9c5f543065b220fb5f9e9114521732cdc0b`
- GitHub Actions：CI run #17
- run ID：`35397080143`
- event：push
- conclusion：**success**
- M2 C++ fast backend build：success
- M3 C++ tuple backend build：success
- remote full pytest：success

因此 candidate remote CI gate：PASS。

## 16. FROZEN VERIFICATION

权威 frozen tags 必须保持：

- `m0-reference-pass^{}` -> `3f2def1d95f56eff776e671143188947bf64485b`
- `m1-fastenv-audited-pass^{}` -> `e5486017a90eeec4fb9814de7880b3c413dc06dd`
- `m2-network-audited-pass^{}` -> `6a9da5b1bf7c72207acd6e89bc667d439d4823a8`

本轮没有修改 frozen M2 backend source，没有移动任何 frozen tag。

candidate 收口后再次核验。

## 17. FILES CHANGED

普通 M3 恢复阶段新增/更新的主要文件：

- `benchmarks/validate_m3_value_calibration.py`
- `benchmarks/generate_m3_teacher_dataset.py`
- `benchmarks/validate_m3_student_sanity.py`
- `reports/m3/m3_teacher_calibration.json`
- `reports/m3/m3_teacher_dataset.json`
- `reports/m3/m3_student_sanity.json`
- `reports/m3/M3_REPORT.md`

Search Performance Unblock 已提交的 M3 C++ evaluator / benchmark / profile 文件继续保留。

用户自己的 prompt/master-plan 修改、旧 M1/M2 临时 JSON 删除、`.vs/` 不属于本 candidate commit。

## 18. ARTIFACTS

正式本地 M3 artifacts：

- `artifacts/m3/semantic_profile_states.npz`
- `artifacts/m3/m3_calibration_rollouts.npz`
- `artifacts/m3/m3_teacher_source_states_8192.npz`
- `artifacts/m3/m3_teacher_source_games_64.json`
- `artifacts/m3/m3_teacher_validation_8192.npz`
- `artifacts/m3/m3_tuple_evaluator_profile.json`
- `artifacts/m3/m3_p6_reprofile.json`
- `artifacts/m3/search_ab_*_values.npz`
- `artifacts/m3/calibration_viability_progress/`（已废止 depth-3 rollout diagnostic，仅历史证据）
- `artifacts/m3/diagnostics/`

临时 Teacher label chunks / generation progress / Search A/B progress 已在阶段结束时清理。

512 MiB Teacher checkpoint 继续 Git-ignored，不进入 Git history。

canonical 8192-state dataset 保持未增强；D4 仅训练时在线使用。

## 19. FINAL GIT STATE

M3 implementation candidate commit：

`7c95f9c5f543065b220fb5f9e9114521732cdc0b`

该 commit 已 push 到 `origin/main`，GitHub Actions run #17（ID `35397080143`）结论为 **success**。

candidate implementation 已停止变更。随后只允许一个 docs-only report closeout descendant 记录上述 CI 事实；它不改变 checkpoint、Search、dataset、calibration 或 Student 实现。

- 不创建 M3 tag
- 不标记 M3 AUDITED PASS / FROZEN
- 不进入 M4/M5
- 用户 prompts/master plan、旧 M1/M2 JSON 删除、`.vs/` 与本地 artifacts 不混入 M3 commit

## 20. REMAINING BLOCKERS

就 M3 施工单 candidate gate 而言，功能性 blocker 已解除：

- checkpoint semantics：PASS
- tuple evaluator：PASS
- Search correctness：PASS
- calibration probe：COMPLETE，且保守保持 RAW semantics
- 8192-state / 64-game dataset：COMPLETE
- game-level split：PASS
- Student sanity：PASS
- Search profile/performance：PASS

剩余只允许执行 candidate 收口：

`full pytest -> candidate commit/push -> remote CI -> frozen/tag verification -> STOP for independent audit`

独立审计之前不得进入 M4/M5，不得创建 M3 tag，不得将 M3 标记为 AUDITED PASS。
