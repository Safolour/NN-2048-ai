# M5 Teacher Checkpoint Promotion Gate — Independent Audit

## RESULT

**NO PROMOTION. M5 MUST CONTINUE WITH THE M3 FROZEN TEACHER BASELINE.**

Authoritative retained Teacher checkpoint:

`teacher_checkpoints/m3/ordinary_td_comparator_ep4800000_7192719323a0.bin`

SHA-256:

`7192719323a073ba2b6b19b62cb7d46ef4aa90ecc8c4ae6baf27ad0c51566a84`

This audit refreshes and supersedes the earlier same-session 5.1M/7.0M candidate check. The current immutable candidates available before M5 sealing are COMPARATOR 6.4M and FORMAL 8.4M.

## Frozen development seed set

Upstream benchmark root:

`D:\CodexTasks\2048-ai`

dev-fast seed set:

`benchmarks/seeds/dev-fast.csv`

count: 2,000

SHA-256:

`7a287318233f5e4577a1edbd19271a68e6e1aefa3f387c9a406a8cb06748418c`

The dev-strong 10,000-seed set was **not opened**, because neither current candidate passed the mandatory dev-fast promotion gate.

## Evaluator

Formal evaluator:

`D:\CodexTasks\2048-ai\build\ntuple_m6_tool.exe`

SHA-256:

`EA033133065448B3D128D049FF02F10D6CFF75104F08ED38F96D7F9696FC68A5`

The same evaluator was used for baseline and both candidates on the same frozen 2,000 seeds.

Bootstrap:
- paired by `pair_id`
- 10,000 resamples
- NumPy PCG64
- candidate-specific deterministic seeds 20260940 / 20260941
- 95% interval = default linear quantiles [0.025, 0.975]

## Baseline — COMPARATOR 4.8M

Snapshot:

`D:\CodexTasks\2048-ai\oneclick-m6-report\snapshots\COMPARATOR\ep4800000_7192719323a0.bin`

episode: 4,800,000

SHA-256:

`7192719323a073ba2b6b19b62cb7d46ef4aa90ecc8c4ae6baf27ad0c51566a84`

dev-fast:
- mean score: **224,947.046**
- median score: 245,634
- p10: 80,360.8
- p90: 333,914.4
- reach >=8192: 88.40%
- reach >=16384: 55.20%
- reach >=32768: 0%

Raw evidence:

`artifacts/m5/promotion_planning/devfast_baseline.jsonl`

SHA-256:

`6D91D550EE4880D43B6C36BBF9D0AF9D7775C3B995284082A873889DA6B211E3`

## Candidate A — COMPARATOR 6.4M

Snapshot:

`D:\CodexTasks\2048-ai\oneclick-m6-report\snapshots\COMPARATOR\ep6400000_5edc31beb458.bin`

episode: 6,400,000

SHA-256:

`5EDC31BEB45806F3E73DA96FB01BCA1B6CF1D94F5331173683B1F715240792A9`

dev-fast:
- mean score: **225,342.570**
- median score: 242,316
- p10: 79,822.0
- p90: 340,946.0
- reach >=8192: 87.70%
- reach >=16384: 54.00%
- reach >=32768: 0%

Candidate - baseline paired mean score delta:

**+395.524**

95% paired bootstrap CI:

**[-5,479.49305, +6,250.74465]**

wins / losses / ties:

1,012 / 986 / 2

Master-plan §31.4 requires the CI lower bound to be > 0.

Result:

**FAIL DEV-FAST PROMOTION GATE.**

Interpretation: 6.4M has a slightly higher sample mean than 4.8M, but the paired evidence does not establish a real mean-score improvement.

Raw evidence:

`artifacts/m5/promotion_planning/devfast_comparator640.jsonl`

SHA-256:

`8DE39B0AE09842D98BFCA6FAFBFCB664C66502A8F185B0FECBD00AE5564C2F53`

## Candidate B — FORMAL 8.4M

Snapshot:

`D:\CodexTasks\2048-ai\oneclick-m6-report\snapshots\FORMAL\ep8400000_846b76fe64a0.bin`

episode: 8,400,000

SHA-256:

`846B76FE64A0B62D64D5BC295EC4BB997DFB5A958B05BD3FF6251FC896D620DC`

dev-fast:
- mean score: **211,434.094**
- median score: 242,728
- p10: **113,244.8**
- p90: 276,160.0
- reach >=8192: **91.30%**
- reach >=16384: **59.80%**
- reach >=32768: 0%

Candidate - baseline paired mean score delta:

**-13,512.952**

95% paired bootstrap CI:

**[-18,533.8834, -8,423.36625]**

wins / losses / ties:

869 / 1,131 / 0

Result:

**FAIL DEV-FAST PROMOTION GATE.**

FORMAL 8.4M is important evidence, not a useless checkpoint: it materially improves lower-tail score and high-tile reach versus the retained 4.8M baseline, while significantly reducing paired mean score. That trade-off must be preserved for later mandatory Teacher/high-tile experiments. It does **not** satisfy the frozen M5 baseline-promotion rule and therefore must not silently replace the M3 Teacher or be mixed into the M5 dataset.

Raw evidence:

`artifacts/m5/promotion_planning/devfast_formal840.jsonl`

SHA-256:

`00752ABC10259B0C4D67D42FC3C367F1308FCE144721F89C1E38584DB850C8A8`

## Promotion decision

Neither current immutable candidate satisfies:

`paired mean-score CI95 lower bound > 0`

on the frozen 2,000-game dev-fast set.

Therefore:
- dev-strong 10,000 is not opened;
- no candidate is eligible for promotion;
- M5 retains COMPARATOR 4.8M as its single frozen Teacher baseline;
- M5 must not use COMPARATOR 6.4M, FORMAL 8.4M, any older superseded candidate, or any moving `latest.bin`;
- FORMAL 8.4M tail/high-tile gains are explicitly preserved as evidence for later Teacher/high-tile experiments, not discarded.

This review satisfies the §31.4 pre-M5 promotion requirement for the current immutable candidates available before M5 sealing.

A future checkpoint can only replace the retained baseline through a new independent promotion gate using the same master-plan rules.
