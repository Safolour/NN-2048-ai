# M5 Teacher Pretraining Candidate Report

## 1. RESULT

**M5_CANDIDATE_EVIDENCE_COMPLETE**

This is candidate evidence, not M5 audited/frozen status.

## 2. STARTING STATE

- base HEAD: 0058fa117a9ccc6d999597796fa50387348bf987
- work order: M5_WO_CLOSURE_V2
- prompt SHA-256: C31E9B74A24EE65BFBD3C134679991B8573D4733C74593041B2E167B21A89ACF

## 3. FROZEN PROVENANCE

- M0/M1/M2/M3/M4 authoritative tags verified.
- Frozen M2/M3/M4 implementation diff gates remained zero.

## 4. TEACHER

- promotion audit result: **NO PROMOTION**
- retained Teacher SHA-256: 7192719323A073BA2B6B19B62CB7D46EF4AA90ECC8C4AE6BAF27AD0C51566A84
- Search root values retain SEARCH_VALUE_RAW_LEAF semantics.
- M5 used policy CE only; no absolute Q/V/A regression.

## 5. SEARCH PROFILE

- median roots/s: 9.781657
- orchestration share: 0.148401
- three complete 256-root repeats passed exact correctness gates.

## 6. DATASET

- 32k and 131k rungs completed.
- 524k trigger: TRIGGERED
- canonical source boards remained unaugmented.
- D4 augmentation was applied only in training batches.
- validation contains 64 complete source games / 8192 states.
- test labels were generated only after champion selection.

## 7. PRECISION

- selected precision: BF16 autocast
- Teacher metrics, D4, and gameplay used FP32 inference.

## 8. TRAINING

- architecture: ResidualMLP2048
- parameter count: 5,264,710
- epochs per run: 30
- batch size: 1024
- optimizer: AdamW, lr 3e-4, weight decay 1e-4
- value and afterstate heads remained bit-identical to initialization.

## 9. SCALE SELECTION

- selected scale: **524k**
- champion seed: **20262103**
- champion checkpoint SHA-256: 4C1313E9085E3A0C1FEC4C6AAA1200A0A5A4377F3EAE0176659F80E9EEAE5784
- selected vs M4 paired mean delta: 7460.513333
- selected vs M4 CI95: [7307.345817, 7616.990283]

## 10. VALIDATION

- median CE: 0.821761
- median legal best-action accuracy: 0.587646
- median pairwise ranking accuracy: 0.834789
- random-legal baseline: 0.293121

## 11. TEACHER TEST

- champion Teacher test was consumed once after champion selection.
- test samples: 8192

## 12. FINAL GAMEPLAY

- games: 10000
- mean score: 9812.218400
- median score: 8482.000000
- p10 score: 3040.000000
- p90 score: 17884.000000
- mean moves: 631.161300

- reach 2048: 0.032500
- reach 4096: 0.000000
- reach 8192: 0.000000
- reach 16384: 0.000000
- reach 32768: 0.000000
- reach 65536: 0.000000

## 13. PERFORMANCE

- actual Teacher label wall: 11263.736s

## 14. REGRESSION

- pytest passed: 543
- pytest failures/errors/skipped: 0/0/0

## 15. GIT / CI

- candidate_sha=PENDING_BY_DESIGN
- candidate_ci=PENDING_BY_DESIGN
- closeout_commit_sha=SELF_NOT_EMBEDDABLE_BY_DESIGN
- closeout_ci=PENDING_BY_DESIGN

## 16. AUDIT STATUS

M5 remains candidate evidence only.
Independent M5 audit is still required before any audited tag.
M6 has not been entered.
