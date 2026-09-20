# M6 Student State Correction Candidate Report

## 1. RESULT

**M6_CANDIDATE_EVIDENCE_COMPLETE_PROMOTION_ELIGIBLE**

Candidate evidence only; independent M6 audit is still required.

## 2. STARTING STATE

- planning base: 6a9b7614e1f7e971e246ae9820f3d052e07875ed
- work order: M6_WO_STATE_CORRECTION_V2
- prompt SHA-256: 70BD2D2C67DC56C37290BF05A9254AB1F7FFF40155C68F7514904A1265CFCBA3

## 3. FROZEN PROVENANCE

- M0-M5 authoritative frozen tags were verified unchanged.
- M5 Student SHA-256: 4C1313E9085E3A0C1FEC4C6AAA1200A0A5A4377F3EAE0176659F80E9EEAE5784
- Teacher SHA-256: 908BA8B8D01A4AFF76D32BE65220D61B6ED702696AA83E10B41594EB5A25AACC

## 4. SEARCH PROFILE

- median roots/s: 14.842951
- orchestration share: 0.131910
- all three 256-root repeats passed exact correctness.

## 5. STUDENT ROLLOUT

- source policy: frozen M5 pure-NN single-forward FP32.
- spawn RNG and tie RNG are independent and seed-stable.
- 128 canonical states were sampled per accepted game.

## 6. CORRECTION DATASET

- train states: 131072
- validation states: 8192
- test states: 8192
- Teacher relabel depth: 3, SEARCH_VALUE_RAW_LEAF.

## 7. ANCHOR

- M5 anchor manifest SHA-256: AE6BD51FCF0815A3EB9B919CC4D5980E41EC614CC01ECE174B58F20E3AFE1551
- anchor rows: 524,288 M5 train rows only.

## 8. TRAINING

- architecture: ResidualMLP2048
- three seeds: 20263101 / 20263102 / 20263103
- exact mixture: 512 correction + 512 anchor per batch.
- 30 epochs / 7,680 optimizer steps per seed.
- value/afterstate heads remained bit-identical.

## 9. CANDIDATE SELECTION

- locked candidate seed: 20263101
- candidate checkpoint SHA-256: EF779ADFB3C2BE21631F73BE400149D511CC7903B9DECBC309948F77F6F94FAB
- dev_positive: True

## 10. GAMEPLAY VALIDATION

- paired mean delta: 4641.134400
- CI95: [4348.53444, 4939.576059999999]
- validation_gain_pass: True

## 11. TEACHER TEST

- one-time Teacher-test split consumed only after candidate lock.
- Teacher test did not alter selection or training.

## 12. FINAL 10,000-GAME TEST

- paired mean delta: 4809.585200
- CI95: [4606.754690000001, 5016.3123]
- final_gain_pass: True

- M5 mean / median: 9789.302000 / 8382.000000
- M6 mean / median: 14598.887200 / 13834.000000

## 13. PROMOTION

- result: **PROMOTION_ELIGIBLE**
- NO_PROMOTION is a valid completed research outcome.

## 14. REGRESSION

- pytest: 553 passed
- failures / errors / skipped: 0 / 0 / 0

## 15. GIT / CI

- candidate_sha=bf3b939eef92f575541f134fcdad545c0d2a9fb5
- candidate_ci=35501899266 / success
- closeout_commit_sha=SELF_NOT_EMBEDDABLE_BY_DESIGN
- closeout_ci=PENDING_BY_DESIGN_AT_REPORT_COMMIT

## 16. AUDIT STATUS

M6 is not audited or frozen by this execution.
No M6 audited tag was created.
M7 was not entered.
