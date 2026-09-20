# M6 独立审计报告

## RESULT

**M6 FINAL AUDITED PASS / FROZEN**

本报告记录对 `M6_WO_STATE_CORRECTION_V2` candidate 的独立收口审计。审计不把 execution Agent 的结论本身作为证据，不重新执行 training、Teacher labeling、development / validation / final gameplay 或 Teacher-test generation。

权威 M6 implementation commit：
`bf3b939eef92f575541f134fcdad545c0d2a9fb5`

candidate report-only closeout descendant：
`936b981746486e84bbb7ea8521428ec35e7aaef7`

权威 audited tag：
`m6-state-correction-audited-pass`

该 tag 必须指向上述 implementation commit，而不是 report-only closeout 或本 audit docs commit。

Promoted frozen Student baseline：
- training seed：`20263101`
- checkpoint SHA-256：`EF779ADFB3C2BE21631F73BE400149D511CC7903B9DECBC309948F77F6F94FAB`

## 审计边界

本次独立审计保持只读实验边界：
- 未重跑 M6 training
- 未重跑 Teacher labeling
- 未重跑 development / validation / final gameplay
- 未重新生成 Teacher-test
- 未修改任何实验结果以“修”审计
- 未进入 M7
- 审计开始时不存在 M6 audited tag
## Git / frozen provenance

审计起点：
- HEAD = `936b981746486e84bbb7ea8521428ec35e7aaef7`
- origin/main = `936b981746486e84bbb7ea8521428ec35e7aaef7`
- worktree clean

六个既有 frozen audited tags 独立核对后均未发生漂移：
- M0 `m0-reference-pass` -> `3f2def1d95f56eff776e671143188947bf64485b`
- M1 `m1-fastenv-audited-pass` -> `e5486017a90eeec4fb9814de7880b3c413dc06dd`
- M2 `m2-network-audited-pass` -> `6a9da5b1bf7c72207acd6e89bc667d439d4823a8`
- M3 `m3-teacher-audited-pass` -> `7c95f9c5f543065b220fb5f9e9114521732cdc0b`
- M4 `m4-architecture-audited-pass` -> `300d51a818fa55394ec56de7507bcade111a06d1`
- M5 `m5-pretrain-audited-pass` -> `36fe93cb211b08916d40df3a7e80bdea0b04a658`

M6 planning base：
`6a9b7614e1f7e971e246ae9820f3d052e07875ed`

planning base 后精确只有两个 execution/closeout commits：
1. `bf3b939eef92f575541f134fcdad545c0d2a9fb5` — `m6: student state correction candidate`
2. `936b981746486e84bbb7ea8521428ec35e7aaef7` — `docs: close M6 candidate report`

candidate commit changed-path set 精确为施工单允许的 7 个 execution paths；closeout commit 只修改 `reports/m6/M6_REPORT.md`。

## Frozen inputs / Teacher provenance

M5 frozen Student baseline checkpoint 独立 SHA-256：
`4C1313E9085E3A0C1FEC4C6AAA1200A0A5A4377F3EAE0176659F80E9EEAE5784`
M5 historical Teacher provenance 保持为：
`7192719323A073BA2B6B19B62CB7D46EF4AA90ECC8C4AE6BAF27AD0C51566A84`

M6 correction Teacher：
`teacher_checkpoints/m6/formal_afterstate_td0_tc_ep10000000_908ba8b8d01a.bin`

独立重算完整 3.22 GB 文件 SHA-256：
`908BA8B8D01A4AFF76D32BE65220D61B6ED702696AA83E10B41594EB5A25AACC`

header 独立核对：
- format version = 2
- stage count = 2
- thresholds = `(0, 16384)`
- global episodes = 10,000,000
- OTD = 9,000,000
- TC = 1,000,000
- training seed = 1
- alpha normalizer = 128

M5 historical Teacher `719271...` 与 M6 correction Teacher `908BA8...` 在 checkpoint / anchor / active V2 provenance 中保持严格分离。

V1 -> V2 archive 存在并保留旧 Teacher provenance；active V2 session 明确记录合法 migration，没有覆盖历史 evidence。

## P2 Search correctness / performance

冻结 semantic-profile states SHA：
`9AFD3A503819C8BD777C7F48D3CCDCA2B1C2F02A7D28DE7DF8997735EE68E512`

审计再次确认：
- legacy legal mask 一致
- frozen traversal counts 一致
- promoted Teacher C++ / Python formal leaf bit-identical
- 三个 promoted-Teacher Search repeat action values bit-identical
正式 P2 evidence：
- repeat 1 = `14.842951...` roots/s
- repeat 2 = `14.044621...` roots/s
- repeat 3 = `15.096675245667344` roots/s
- median = `14.84295131260611` roots/s
- required minimum = `8.803491691009711` roots/s
- orchestration share = `0.1319098161`

独立审计环境额外只读 Search 复跑约为：
- `8.659`
- `8.886`
- `8.930` roots/s
- median = `8.8863` > gate
- orchestration share = `0.1381` < 0.50

因此 Search correctness 与性能 gate 在独立审计中仍成立；未降低门槛。

## Student source / correction labels

Student source split 的 canonical seeds、retry stride、sampling positions 与 split isolation 均符合 V2 contract。

独立核对 retry：
- train：32 次合法 retry
- validation：0
- test：2
- accepted seed sets 彼此不重叠，并与 gameplay seed ranges 隔离

Correction label dataset：
- train：64 shards / 131,072 rows
- validation：4 shards / 8,192 rows
- test：4 shards / 8,192 rows

所有 completed shard 的 SHA、schema、dtype/shape、legal-value semantics、Teacher best action、disagreement、game boundary 与 metadata 均通过。
Label shards 中保存的 Student state / logits / action / game_seed / step_index 等字段与各自 source NPZ 逐数组一致。

M5 anchor：
- manifest SHA-256 = `AE6BD51FCF0815A3EB9B919CC4D5980E41EC614CC01ECE174B58F20E3AFE1551`
- 256 train shards
- 524,288 rows
- validation/test 未用于 training
- historical Teacher 仍为 `719271...`

## Correction training

三个正式 correction runs：
- `20263101`
- `20263102`
- `20263103`

源码与 checkpoint/progress evidence 独立确认：
- 每个 run 从 exact M5 model_state 开始
- fresh optimizer
- 30 epochs
- 256 optimizer steps / epoch
- 7,680 optimizer steps / run
- 每 batch 严格 512 correction + 512 M5 anchor
- objective 为单一 policy cross entropy
- backbone updated
- Q head updated
- value head 相对 M5 bit-identical
- afterstate head 相对 M5 bit-identical
- 三个 run `learning_sanity=true`

Teacher-validation legal-best accuracy：
- M5 = `0.492310`
- 20263101 = `0.542236`
- 20263102 = `0.548462`
- 20263103 = `0.546875`

Teacher-validation pairwise ranking：
- 20263101 = `0.777277`
- 20263102 = `0.781553`
- 20263103 = `0.776923`
## Development / candidate lock

四方 development raw gameplay 均使用完全相同的 `20291001..20293000` 2,000 seeds。

独立 raw means：
- M5 baseline = `9823.184`
- 20263101 = `14570.972`
- 20263102 = `14481.978`
- 20263103 = `14172.956`

按冻结规则“完整 development mean 最大；精确 tie 取较小 training seed”，唯一机械 candidate 为：
`20263101`

candidate checkpoint：
`artifacts/m6/checkpoints/20263101_final.pt`

SHA-256：
`EF779ADFB3C2BE21631F73BE400149D511CC7903B9DECBC309948F77F6F94FAB`

状态机与 artifact 时间顺序确认：
candidate lock → independent validation → test split / Teacher-test → final gameplay。

不存在 validation / Teacher-test / final result 参与 reselection 或 retraining 的路径。

## Independent gameplay validation

5,000 paired validation 独立从 raw evidence 重算：
- M5 mean = `9700.6816`
- candidate mean = `14341.816`
- paired delta = `+4641.1344`
- wins/losses/ties = `3313 / 1686 / 1`
- bootstrap CI95 = `[4348.53444, 4939.57606]`

CI lower bound > 0，因此 frozen validation gain gate PASS。

## One-time Teacher test

Teacher-test 只在 candidate lock 后执行，且仅评估 M5 + locked candidate；`affected_selection=false`、`one_time_test_consumed=true`。
Teacher-test metrics：
- M5 legal-best accuracy = `0.485352`
- M6 candidate legal-best accuracy = `0.534302`
- M5 pairwise = `0.749035`
- M6 candidate pairwise = `0.767759`

Teacher-test 没有参与 candidate selection。

## Final 10,000 paired gameplay

final raw gameplay seeds：
`20300001..20310000`

两个 raw NPZ 的 recorded SHA 均经独立核对匹配。

独立重算：
- pairs = `10,000`
- M5 mean = `9789.302`
- M6 candidate mean = `14598.8872`
- paired delta = `+4809.5852`
- bootstrap CI95 = `[4606.754690000001, 5016.3123]`
- wins/losses/ties = `6741 / 3257 / 2`

formal report 的舍入 CI `[4606.75469, 5016.3123]` 与独立重算一致。

因此 `final_gain_pass=true`。

## Promotion logic

机械 promotion inputs：
- development delta > 0
- independent validation delta > 0
- validation CI lower > 0
- final delta > 0
- final CI lower > 0
- 三个 formal runs learning sanity PASS
- provenance / split isolation / selection isolation PASS
- correctness / regression / cleanup gates PASS

因此独立审计机械推出：
`PROMOTION_ELIGIBLE`
## Regression / reports / cleanup

P10 JUnit 原件独立解析：
- tests = `553`
- failures = `0`
- errors = `0`
- skipped = `0`

formal pytest result：
`553 passed / 0 failed / 0 skipped / 0 xfailed`

M6 test file 包含施工单规定的 10 个 M6 tests。

cleanup 独立扫描未发现：
- `*_latest.pt`
- `.tmp`
- `.lock`
- partial / scratch / debug / superseded residue
- root-level stray M6 result
- M7 tracked implementation
- `artifacts/m7`

`reports/m6/m6_state_correction.json` 与关键 raw evidence 一致；`M6_REPORT.md` 中 candidate、validation、final、promotion、regression、Git/CI 信息与独立重算一致。

Execution report 明确保留 “M6 is not audited or frozen by this execution” / “No M6 audited tag was created”，因此 execution 没有越权把 candidate 自行升级为 audited baseline。

## GitHub Actions

candidate：
- SHA `bf3b939eef92f575541f134fcdad545c0d2a9fb5`
- message `m6: student state correction candidate`
- run `35501899266`
- workflow `CI`
- event `push`
- conclusion `success`

closeout：
- SHA `936b981746486e84bbb7ea8521428ec35e7aaef7`
- message `docs: close M6 candidate report`
- run `35502192278`
- workflow `CI`
- event `push`
- conclusion `success`
## FINAL CONCLUSION

未发现足以使 M6 promotion 失效的：
- provenance drift
- selection leakage
- test leakage
- resume corruption
- duplicate experimental execution
- protocol drift
- checkpoint/head contamination
- Git scope drift

因此最终审计结论：

**M6 FINAL AUDITED PASS / FROZEN**

`PROMOTION_ELIGIBLE` 已被 independent audit 验证。

权威 audited implementation：
`bf3b939eef92f575541f134fcdad545c0d2a9fb5`

权威 tag：
`m6-state-correction-audited-pass`

新的 frozen Student baseline：
- seed `20263101`
- checkpoint SHA-256 `EF779ADFB3C2BE21631F73BE400149D511CC7903B9DECBC309948F77F6F94FAB`

后续 milestone 必须以该 checkpoint 作为冻结 Student baseline；不得重新打开 M6 correction selection / validation / final promotion，除非未来独立发现真实 correctness bug。

M7 尚未执行。下一步仅允许进入 M7 planning / implementation work-order preparation。
