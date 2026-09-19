# M4 独立审计报告

## RESULT

**M4 FINAL AUDITED PASS / FROZEN**

本报告是对 M4 candidate 的独立收口审计，不是 candidate 自述的重复。

权威 M4 implementation commit：

`300d51a818fa55394ec56de7507bcade111a06d1`

candidate report-only closeout descendant：

`2378761afe49ae6db8323035a367811699be9f5e`

authoritative tag：

`m4-architecture-audited-pass`

tag 必须指向上述 implementation commit，而不是 report-only closeout 或 audit docs closeout commit。

## 审计范围

独立核对：
- M4_WO_CLOSURE_V2 全部 candidate gate；
- Git / frozen provenance / tracked scope；
- audited M3 dataset identity；
- deterministic D4 training plans 与 plan SHA；
- BF16 precision gate；
- 6 个 primary training runs；
- checkpoint identity / metadata / parameter-update semantics；
- canonical Teacher metrics 与 D4 diagnostics；
- pure-NN closed-loop evaluator correctness；
- 6 × 2000 paired games 与 raw result SHA；
- paired bootstrap / strength conclusion；
- model-only / closed-loop / training performance；
- equal-wall-clock secondary；
- mechanical architecture selection；
- local full pytest 与 GitHub Actions；
- artifact cleanup。

## Git / frozen provenance

审计时：
- HEAD / origin/main = `2378761afe49ae6db8323035a367811699be9f5e`
- worktree clean
- M0 -> `3f2def1d95f56eff776e671143188947bf64485b`
- M1 -> `e5486017a90eeec4fb9814de7880b3c413dc06dd`
- M2 -> `6a9da5b1bf7c72207acd6e89bc667d439d4823a8`
- M3 -> `7c95f9c5f543065b220fb5f9e9114521732cdc0b`

frozen M2 source diff gate = 0。

candidate 相对 base `cdc250477c...` 的 changed-path set 精确为施工单允许的 6 个 M4 文件。
report-only closeout 只修改 `reports/m4/M4_REPORT.md`。

## Dataset / plans

固定 dataset：
`artifacts/m3/m3_teacher_validation_8192.npz`
SHA-256：
`F0E5D806A3E17A27284998AAD52A683245A3C74EBDD15B23EFC13448B9D96582`

独立确认：
- 8192 states / 64 games
- train 51 games / 6528 states
- validation 6 games / 768 states
- test 7 games / 896 states
- decision_depth = 3
- value_semantics = SEARCH_VALUE_RAW_LEAF
- checkpoint SHA metadata 与 M3 audited checkpoint 一致

三个 primary plan 均独立按 PCG64 固定算法重建，sample_indices / transform_ids 逐元素一致。

plan SHA：
- 20260919: `61C94248D18649C528B06EFD2B64738D6546C4AB02D195008C84D597F4B67667`
- 20260920: `7EF700494F05549A56C233D44D110AD4BF49E6A7EAD421A6F13F24F1F17C14DB`
- 20260921: `0B1E9EEF0C3E4668A2318834C7924C984ABB43C64E6B6250C63C1A3230FD2231`

全部与 evidence 精确一致。

## Precision / primary training

BF16 support = true。
Transformer2048 / ResidualMLP2048 的 BF16 smoke 均为 20 / 20 PASS。
因此 primary precision 固定为 `BF16 autocast`，未触发 FP32 fallback。
6 个 primary runs 全部满足：
- 30 epochs
- 210 optimizer steps
- AdamW / lr 3e-4 / weight_decay 1e-4
- batch 1024，最后 batch 384
- same deterministic per-seed D4 plan
- no scheduler / no early stopping / no architecture-specific tuning

6 个 final checkpoint 文件 SHA 全部与 manifest 一致。
checkpoint metadata 的 architecture / seed / precision / dataset SHA / plan SHA 全部正确。

独立重建同 seed 随机初始化后逐 tensor 比较：
- q_head：已更新
- backbone：已更新
- value_head：bit-identical to init
- afterstate_head：bit-identical to init

因此 policy-only loss 没有偷偷训练 V / Afterstate head。

## Teacher / D4 independent recompute

代码结构确认 test rows 只有在 6 个 primary runs 全部完成后才被消费。

从 6 个 final checkpoint + audited M3 dataset 独立重算完整 896-state test：
- raw best-action accuracy：6 / 6 精确一致
- legal best-action accuracy：6 / 6 精确一致
- legal pairwise ranking：6 / 6 精确一致
- D4 per-transform consistency：6 / 6 一致
- D4 overall consistency：6 / 6 一致
- centered-logit MAE：6 / 6 一致
Teacher CE 复算 5 / 6 精确一致；Transformer2048/20260921 的重复 GPU FP32 forward 与记录值仅相差约 `6.8e-8`，属于浮点累加级差异，不影响任何 action / ranking / gate。

## Closed-loop games

6 个 primary raw game NPZ：
- 每个均为 2000 games
- seeds 精确为 20261001..20263000
- 文件 SHA 全部匹配 manifest
- mean / median / p10 / p90 / mean moves / reach / max-tile distribution 全部独立重算一致

另外独立抽取 seeds 20261001..20261003，对 6 个 final checkpoint 共 18 局重新 closed-loop replay：
- final score：18 / 18 精确一致
- moves：18 / 18 精确一致
- max_tile_exp：18 / 18 精确一致
- 每一步 fast movement vs M0 reference afterstate/reward/moved：全部一致

spawn RNG 与 tie RNG 分离、tie tolerance = 1e-7、pure FP32 game inference 与施工单一致。

## Primary statistics

Transformer - ResidualMLP mean-score delta：
- seed 20260919: +171.462
- seed 20260920: -671.672
- seed 20260921: +77.578

独立 10,000 paired bootstrap：
- 20260919 CI95 = [93.2612, 247.3902]
- 20260920 CI95 = [-746.9514, -597.2157]
- 20260921 CI95 = [0.80355, 151.26015]

aggregate:
- mean delta = -140.8773333333
- CI95 = [-184.3488, -96.4407833333]

虽然 aggregate CI 偏向 MLP，但三个 training-seed mean 中只有一个为负，不满足施工单的 >=2/3 同向 gate。

因此独立结论为：

`ARCHITECTURE_STRENGTH_NOT_STATISTICALLY_RESOLVED`

与 candidate evidence 完全一致。

## Performance / selection

8192-env closed-loop 5-repeat median：
- Transformer2048 = **50,267.472287 decisions/s**
- ResidualMLP2048 = **728,105.446166 decisions/s**

独立重算 speed ratio：
**14.4846242119x**

primary training compute-wall medians：
- Transformer2048 = 25.3767847 s
- ResidualMLP2048 = 10.3726655 s
- wall ratio = 2.4465056450

因此 equal-wall secondary 必须触发。
secondary 完成：
- Transformer: 29 epochs / 203 steps / 25.6025005 s
- ResidualMLP: 75 epochs / 525 steps / 25.6157702 s
- secondary Transformer-MLP mean delta = +80.52
- CI95 = [-5.1946, 169.3504]

secondary CI 跨 0，且施工单明确 secondary 只作 diagnostic，不得覆盖 primary strength / selection。

primary strength unresolved 且 8192 speed ratio >= 1.10，因此 mechanical selection 唯一结果：

`M4_SELECT_RESIDUAL_MLP`

selection_rule：

`8192_SPEED`

## Regression / CI

独立审计时重新运行官方本地解释器：

`D:\sd-webui-forge-aki-v1.0\python\python.exe -m pytest -q`

结果：
- **533 passed**
- 0 failed
- 0 skipped
- 0 xfailed
- 6 existing PyTorch Transformer warnings

GitHub Actions 独立核对：
- candidate run #26 / ID 35425929852 / head 300d51a... / success
- report-only closeout run #27 / ID 35426048354 / head 2378761... / success

两个 run 的实际 job 均确认：
- Build M2 C++ fast backend: success
- Build M3 C++ tuple backend: success
- Run full pytest suite: success

## Artifact cleanup

审计确认：
- primary final checkpoints = 6
- primary plans = 3
- primary raw game NPZ = 6
- secondary final checkpoints = 2
- secondary raw game NPZ = 2
- required session / primary / performance / finalize / pytest evidence 存在
- `*_latest.pt` = 0
- runtime `.tmp/.lock` = 0

## Non-blocking finding

candidate implementation 的 `_report_markdown()` 使用了字面 `"\\n"` join，导致 candidate 版 Markdown 初次生成时换行为字面量 `\n`。

mandatory report-only closeout 已只修改 `reports/m4/M4_REPORT.md` 修正排版，并回写 candidate CI；机器 evidence、selection、implementation training/evaluation semantics 均未改变。

该问题是历史 report-rendering defect，不影响 M4 architecture decision，因此不构成 M4 correctness blocker。冻结后不得为了美观重写 M4 experiment semantics。

## AUDIT DECISION

未发现需要返工 M4 experiment implementation、raw evidence、statistics 或 selection 的 correctness blocker。
M4 的公平训练、test isolation、game evaluator、paired statistics、performance measurement、equal-wall secondary、mechanical selection、local regression、remote CI 与 frozen-stage protection全部达到施工单要求。

因此：

**M4 FINAL AUDITED PASS / FROZEN**

权威 selected architecture：

**ResidualMLP2048**

M5 可以在本次 audit docs/tag closeout完成并确认 CI green 后成为下一执行阶段。
