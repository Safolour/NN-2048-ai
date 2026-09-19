# M4 Transformer vs ResidualMLP Candidate Report

## 1. RESULT

**M4_CANDIDATE_EVIDENCE_COMPLETE**

This is candidate evidence, not M4 audited/frozen status.

## 2. STARTING STATE

- base HEAD: `cdc250477cce21fbe388871f50c013990a749521`
- work order: `M4_WO_CLOSURE_V2`
- prompt SHA-256: `3EB37DFD85999F1D2344A7922F74A4E566582968AAD11ECF402BFE10B602605B`

## 3. FROZEN VERIFICATION

M0/M1/M2/M3 authoritative tags were verified at FRESH P0. Frozen M2 source diff gate was zero.

## 4. DATASET VERIFICATION

- dataset SHA-256: `F0E5D806A3E17A27284998AAD52A683245A3C74EBDD15B23EFC13448B9D96582`
- 8192 states / 64 games
- train 51 games / 6528 states
- validation 6 games / 768 states
- test 7 games / 896 states
- value semantics: `SEARCH_VALUE_RAW_LEAF`

## 5. MODEL DEFINITIONS / PARAM COUNTS

- Transformer2048: 4,750,342
- ResidualMLP2048: 5,264,710

## 6. FAIRNESS CONTRACT

Same frozen dataset, deterministic D4 plans, three paired training seeds, 30 epochs / 210 optimizer steps per run, AdamW 3e-4, batch 1024, no architecture-specific tuning.

## 7. PRECISION DECISION

Selected primary precision: `BF16 autocast`.

## 8. PRIMARY TRAINING RESULTS

All six epoch-30 final checkpoints completed. Full per-run training metrics, checkpoint SHA values and timing are in `m4_architecture_compare.json`.

## 9. VALIDATION / TEST TEACHER METRICS

Canonical FP32 train/validation/test Teacher metrics were measured for all six final models only after all primary runs completed.

## 10. GREEDY GAME SCORE RESULTS

- Transformer2048/20260919: mean=2335.654, median=2144.000, p10=912.000, p90=3816.000
- Transformer2048/20260920: mean=1581.932, median=1348.000, p10=471.600, p90=3128.800
- Transformer2048/20260921: mean=2041.380, median=1740.000, p10=627.600, p90=3544.000
- ResidualMLP2048/20260919: mean=2164.192, median=1812.000, p10=680.000, p90=3869.600
- ResidualMLP2048/20260920: mean=2253.604, median=1960.000, p10=728.000, p90=3869.200
- ResidualMLP2048/20260921: mean=1963.802, median=1704.000, p10=656.000, p90=3472.400

## 11. PAIRED STATISTICS

Strength conclusion: **ARCHITECTURE_STRENGTH_NOT_STATISTICALLY_RESOLVED**
Aggregate paired result: `{'bootstrap_resamples': 10000, 'bootstrap_seed': 20261104, 'ci95': [-184.3488, -96.44078333333334], 'mean_delta': -140.87733333333335, 'pairs': 2000}`

## 12. D4 DIAGNOSTIC

D4 legal-argmax consistency by transform id, overall consistency and centered-logit MAE were measured on the complete 896-state canonical test split.

## 13. INFERENCE PERFORMANCE

Required eager model-only batches 1/256/1024/2048/4096/8192 completed for both architectures.

## 14. TRAINING PERFORMANCE

Training-only samples/s, CUDA-event step timings and peak VRAM were recorded for all six primary runs.

## 15. CLOSED-LOOP PERFORMANCE

Required 2048-env and 8192-env closed-loop points completed with five paired-order repeats per architecture.

## 16. EQUAL-WALL-CLOCK SECONDARY

Status: `TRIGGERED_COMPLETE`

## 17. ARCHITECTURE SELECTION

- selected: **M4_SELECT_RESIDUAL_MLP**
- strength: **ARCHITECTURE_STRENGTH_NOT_STATISTICALLY_RESOLVED**
- selection_rule: **8192_SPEED**
- speed_ratio: 14.484624

## 18. PYTEST

533 passed / 0 failed / 0 errors / 0 skipped / 0 xfailed.

## 19. CI

- candidate_sha=300d51a818fa55394ec56de7507bcade111a06d1
- candidate_ci=CI run #26 / ID 35425929852 / success
- closeout_parent=300d51a818fa55394ec56de7507bcade111a06d1
- closeout_commit_sha=SELF_NOT_EMBEDDABLE_BY_DESIGN
- closeout_ci=PENDING_BY_DESIGN_AT_REPORT_COMMIT

## 20. FILES CHANGED

Only the six M4 tracked paths allowed by the work order.

## 21. ARTIFACTS / CLEANUP

Primary final checkpoints, plans, raw game NPZ files, progress/session manifests, performance evidence and pytest.xml are retained. Runtime latest/temp/lock files are removed before P10.

## 22. FINAL GIT STATE

Candidate implementation commit: `300d51a818fa55394ec56de7507bcade111a06d1`. Mandatory report-only closeout parent: `300d51a818fa55394ec56de7507bcade111a06d1`.

## 23. REMAINING BLOCKERS

Local candidate evidence is complete and candidate CI is green. Mandatory report-only closeout CI remains before `M4 CANDIDATE COMPLETE`.
