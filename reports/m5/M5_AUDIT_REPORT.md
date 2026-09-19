# M5 独立审计报告

## RESULT

**M5 FINAL AUDITED PASS / FROZEN**

本报告是对 `M5_WO_CLOSURE_V2` candidate 的独立收口审计，不复用 execution Agent 的结论作为证据。

权威 M5 implementation commit：
`36fe93cb211b08916d40df3a7e80bdea0b04a658`

candidate report-only closeout descendant：
`bf2d9005c958792d5621d7835ee2c77fea3aa144`

权威 audited tag：
`m5-pretrain-audited-pass`

该 tag 必须指向上述 implementation commit，而不是 report-only closeout 或本 audit docs commit。

## 审计范围

独立核对：
- M0-M4 frozen provenance / tags / M5 tracked scope
- V1 -> V2 short-source-game migration contract
- deterministic rejection-and-retry
- 4096-game / 524,288-state source dataset
- 256 train + 4 validation + 4 test Search-label shards
- Teacher checkpoint / Search profile
- 32k / 131k / 524k 三档三 seed fresh-init training
- validation / high-tile / Teacher fitting evidence
- 9 × 2000 development games
- scale gate / final scale selection / champion selection
- one-time Teacher test isolation
- final 10,000 pure-NN games
- P10 exact test count / JUnit parser
- candidate / closeout Git scope and GitHub Actions
- report / machine summary consistency
- artifact cleanup / no M6 entry

## Git / frozen provenance

审计起点：
- HEAD = `bf2d9005c958792d5621d7835ee2c77fea3aa144`
- origin/main = `bf2d9005c958792d5621d7835ee2c77fea3aa144`
- worktree clean

五个 peeled frozen tags：
- M0 `m0-reference-pass` -> `3f2def1d95f56eff776e671143188947bf64485b`
- M1 `m1-fastenv-audited-pass` -> `e5486017a90eeec4fb9814de7880b3c413dc06dd`
- M2 `m2-network-audited-pass` -> `6a9da5b1bf7c72207acd6e89bc667d439d4823a8`
- M3 `m3-teacher-audited-pass` -> `7c95f9c5f543065b220fb5f9e9114521732cdc0b`
- M4 `m4-architecture-audited-pass` -> `300d51a818fa55394ec56de7507bcade111a06d1`

M5 candidate parent 精确为 V2 planning commit：
`0058fa117a9ccc6d999597796fa50387348bf987`

candidate commit changed-path set 精确为施工单允许的 7 个路径：
- `src/game2048/m5_pretrain.py`
- `benchmarks/profile_m5_teacher_search.py`
- `benchmarks/generate_m5_teacher_dataset.py`
- `benchmarks/run_m5_teacher_pretrain.py`
- `tests/test_m5_teacher_pretrain.py`
- `reports/m5/m5_teacher_pretrain.json`
- `reports/m5/M5_REPORT.md`

closeout commit parent 精确为 candidate SHA，且只修改：
`reports/m5/M5_REPORT.md`

从 M4 authoritative implementation 到当前 source/tests/benchmarks 的新增 implementation 仅为 M5 自有文件；未发现 M5 回写 M0-M4 frozen implementation。

## V1 -> V2 migration / source games

V2 规划 commit `0058fa1...` 是 V1 planning commit `9c2aab4...` 的后继，只修补 short-source-game protocol / master plan；M5 execution 后续以：
- `work_order_version=M5_WO_CLOSURE_V2`
- `base_head=0058fa117a9ccc6d999597796fa50387348bf987`
- prompt SHA `C31E9B74A24EE65BFBD3C134679991B8573D4733C74593041B2E167B21A89ACF`
继续。

独立检查 current source evidence：
- train：4096 logical games / 524,288 states
- validation：64 games / 8,192 states
- test：64 games / 8,192 states
- 每个 accepted game 恰好 128 rows
- 每个 game 的 128 `step_index` 全唯一
- 每个 game 只有一个 accepted effective seed
- 无 duplicate/pad

train 唯一三个 retry cases：
- game 2682：canonical seed `20267683`，119 moves rejected；retry 1 seed `1020267683` accepted，13884 moves
- game 2897：canonical seed `20267898`，112 moves rejected；retry 1 seed `1020267898` accepted，7032 moves
- game 2996：canonical seed `20267997`，54 moves rejected；retry 1 seed `1020267997` accepted，10965 moves

全部符合 `effective_seed = canonical_seed + retry_index * 1_000_000_000`，且 rejected attempt 未进入 accepted source rows。

## Search labels

独立遍历 264 个完成 shard：
- train 256 / 256
- validation 4 / 4
- test 4 / 4
- 每 shard 2048 rows
- total label rows = 540,672
- 每个文件 SHA-256 与 manifest 一致
- required schema keys 全部一致
- `state=(2048,16)` / `teacher_value=(2048,4)` / `afterstate=(2048,4,16)`
- `decision_depth=3`
- `value_semantics=SEARCH_VALUE_RAW_LEAF`
- Teacher version 全部为 `ordinary_td_comparator_ep4800000_7192719323a0`
- Teacher checkpoint SHA hex 值全部为 `719271...66a84`；NPZ 字段使用小写十六进制，和 frozen SHA 字节值完全相同
- 所有 target rows 均至少有一个 legal action

未发现 SHA、schema、row-count 或 semantic metadata corruption。

## Teacher / Search profile

实际 frozen Teacher checkpoint：
`teacher_checkpoints/m3/ordinary_td_comparator_ep4800000_7192719323a0.bin`

独立计算 SHA-256：
`7192719323A073BA2B6B19B62CB7D46EF4AA90ECC8C4AE6BAF27AD0C51566A84`

与 M3 frozen Teacher / promotion audit 的 NO PROMOTION 结果一致。

Search profile 输入 SHA 独立核对：
- semantic corpus：`9AFD3A503819C8BD777C7F48D3CCDCA2B1C2F02A7D28DE7DF8997735EE68E512`
- P6 reference values：`8E09F0D7774BCEC55C496A29550F2362FB83DEE142643F69472F12AE14963C51`

3 × 256 roots 全部满足：
- action values bit-identical
- legal mask equal
- best action equal
- node/cache counts equal

性能：
- median = `9.781657434455234 roots/s`
- gate = `5.688376766 roots/s`
- orchestration share = `0.14840051784293348` < 0.50

`search_profile.json` 保留 V1 work-order 字样，因为该 phase 在 V1 已完成且 V2 migration 明确禁止重跑已完成阶段；这不影响 V2 source/data/training provenance。

## Training protocol / checkpoints

源码审计确认每个 `(scale, seed)` run：
- 先 `set_determinism(seed)`
- fresh `ResidualMLP2048()` initialization
- 只有同一 run RESUME 时才加载其 `_latest.pt`
- 不存在跨 scale / 跨 seed warm-start
- AdamW lr `3e-4`, weight decay `1e-4`, betas `(0.9,0.999)`, eps `1e-8`
- epochs = 30
- batch = 1024
- grad clip = 1.0
- policy-only cross entropy
- deterministic stateless D4 epoch plan

9 个 progress 均：`completed=true`, `next_epoch=30`。

optimizer steps：
- 32k：960 / run
- 131k：3840 / run
- 524k：15360 / run

precision 全部为 `BF16 autocast`。

9 个 final checkpoint：
- 文件 SHA 全部与 progress JSON 精确一致
- architecture/scale/seed/precision/Teacher SHA/epoch metadata 全部一致
- `q_head_updated=true`
- `backbone_updated=true`
- value / afterstate heads 保持未更新

因此未发现非法 warm-start、训练预算漂移或 head 语义污染。

## Validation / Teacher fitting

三档 validation 随数据量增加持续改善。

32k 三 seed final legal accuracy：约 0.4995-0.5033；CE 约 1.006-1.018。

131k：legal accuracy 约 0.5382-0.5477；CE 约 0.8951-0.9023。

524k：
- seed 20262101：CE `0.8280642852`, legal acc `0.580810546875`
- seed 20262102：CE `0.8200413957`, legal acc `0.59130859375`
- seed 20262103：CE `0.8217614889`, legal acc `0.587646484375`

selected 524k median：
- CE `0.8217614889144897`
- legal best-action accuracy `0.587646484375`
- pairwise ranking `0.8347887287229883`

满足施工单 validation/high-tile gates。

## Development games / scale selection

9 个 raw development NPZ 均独立验证：
- 2000 games
- seeds 精确 `20261001..20263000`
- mean / median / p10 / p90 与 `development.json` 一致

per-seed mean score：

32k：
- 20262101 = 3059.452
- 20262102 = 3131.780
- 20262103 = 2956.676
- aggregate = `3049.302666666667`

131k：
- 5370.304 / 5245.608 / 5063.996
- aggregate = `5226.636`

524k：
- 9392.450 / 9367.040 / 10003.648
- aggregate = `9587.712666666666`

独立重算 10,000 paired bootstrap：
- 131k - 32k mean = `+2177.333333333333`; CI95 `[2080.268766666667, 2274.0055]`
- 524k - 131k mean = `+4361.076666666666`; CI95 `[4185.468133333332, 4535.305516666667]`
- selected 524k - M4 mean = `+7460.513333333334`; CI95 `[7307.345816666666, 7616.9902833333335]`

全部与 recorded evidence 一致。

机械 selection 唯一结果：`524k`。

524k 三 seed dev mean 最大者为 `20262103`，因此 champion seed 机械得到 `20262103`。

champion checkpoint SHA：
`4C1313E9085E3A0C1FEC4C6AAA1200A0A5A4377F3EAE0176659F80E9EEAE5784`

## One-time Teacher test

代码硬性限制 test dataset 只能在 `FINAL_SCALE_SELECTED` / `TEST_RUNNING` 后生成。

artifact timestamps 顺序：
- scale selection：22:24:28
- test source：22:25:18
- first test label shard：22:26:21
- Teacher test：22:29:54

test split：64 complete games / 8192 states / 4 shards。

使用 champion checkpoint + raw test shards 独立重算 core Teacher metrics：
- samples = 8192
- CE = `0.8076148480176926`
- raw best-action accuracy = `0.5875244140625`
- legal best-action accuracy = `0.587890625`
- pairwise ranking = `0.8354601802177197`
- ranking pairs = 38398
- random legal baseline = `0.2937215169270833`

与 `teacher_test.json` 精确一致。

未发现提前消费 test 或重复生成/调参回流证据。

## Final 10,000-game gameplay

raw：`artifacts/m5/final/final_games_10000.npz`

SHA-256：
`917F1D814F47DB7C90A3582BE6B2287B47136C2C9F62947B7D758169B7FED67A`

独立从 raw NPZ 复算：
- games = 10000
- seeds 精确、连续 `20277001..20287000`
- mean = `9812.2184`
- median = `8482.0`
- p10 = `3040.0`
- p90 = `17884.0`
- mean moves = `631.1613`
- reach 2048 = `0.0325`
- reach 4096/8192/16384/32768/65536 = `0`
- max observed exponent = 11 (2048)

全部与 `final_gameplay.json` / machine summary 一致。

## P10 / regression

candidate JUnit XML 根节点为 `<testsuites>`，外层无 tests/failures count；内部 `<testsuite>` 才含计数。

独立 XML 聚合得到：
- tests = 543
- failures = 0
- errors = 0
- skipped = 0

因此 M5 对 JUnit parser 的内部 suite 聚合修复是正确且必要的。

审计期间另外独立重新执行 full pytest：
`543 passed, 6 warnings in 43.16s`

无 test failure。

## GitHub Actions / closeout

candidate：
- SHA `36fe93cb211b08916d40df3a7e80bdea0b04a658`
- commit message `m5: teacher pretraining candidate`
- GitHub Actions run `35449808850`
- workflow `CI`
- event `push`
- head SHA 精确匹配 candidate
- conclusion `success`

closeout：
- SHA `bf2d9005c958792d5621d7835ee2c77fea3aa144`
- parent 精确为 candidate SHA
- changed path 仅 `reports/m5/M5_REPORT.md`
- commit message `docs: close M5 candidate report`
- GitHub Actions run `35450597283`
- workflow `CI`
- event `push`
- head SHA 精确匹配 closeout
- conclusion `success`

## Reports / cleanup

`reports/m5/m5_teacher_pretrain.json` 可解析，result = `M5_CANDIDATE_EVIDENCE_COMPLETE`。

machine summary 中 selected scale / champion / checkpoint SHA / Teacher test / final gameplay / regression 与 raw evidence 一致。

`M5_REPORT.md` closeout 已写 candidate SHA + candidate CI；closeout self-SHA / closeout CI 保持施工单规定的 `SELF_NOT_EMBEDDABLE_BY_DESIGN` / `PENDING_BY_DESIGN_AT_REPORT_COMMIT`，由 session 与 Git history 提供权威 closeout provenance。

cleanup 独立检查：
- 无 `*_latest.pt`
- 无 `.tmp` / `.lock`
- 无 partial / scratch
- root 无 milestone 临时输出
- session = `COMPLETE`
- 未发现 M6 tracked implementation / `artifacts/m6`
- 审计开始前不存在任何 M5 audited tag

## FINAL CONCLUSION

M5 的 provenance、dataset、short-game recovery、Search labels、training、development gameplay、scale/champion selection、one-time Teacher test、10k final gameplay、P10、candidate/closeout Git/CI 和 cleanup 均满足 `M5_WO_CLOSURE_V2`。

独立审计没有发现需要重新训练、重新标注或修复 implementation 的 correctness blocker。

因此最终审计结论：

**M5 FINAL AUDITED PASS / FROZEN**

权威 audited implementation：
`36fe93cb211b08916d40df3a7e80bdea0b04a658`

权威 tag：
`m5-pretrain-audited-pass`

下一阶段允许进入 M6 Student State Correction，但 M5 frozen evidence / selection / champion 不得重新打开，除非未来独立发现真实 correctness bug。