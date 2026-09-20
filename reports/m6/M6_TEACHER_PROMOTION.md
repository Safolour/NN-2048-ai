# M6 Teacher Promotion Record

Status: PROMOTED FOR M6 PLANNING BASE

This record freezes the Teacher change that happened after M5 was already audited and frozen.
It does not rewrite the historical M5 Teacher-promotion decision.
M5 remains frozen exactly as audited.

## Promoted Teacher identity

Local runtime checkpoint:
`teacher_checkpoints/m6/formal_afterstate_td0_tc_ep10000000_908ba8b8d01a.bin`

Source surviving model:
`D:\CodexTasks\2048-ai\final.bin`

SHA-256:
`908BA8B8D01A4AFF76D32BE65220D61B6ED702696AA83E10B41594EB5A25AACC`

File size:
`3,221,225,728 bytes`

The source project was intentionally reduced after training; only the final model remains there.
The M6 runtime uses the immutable local copy above, not the source path.

## Header metadata

Format: U2048NT6 / version 2
Feature schema: 4-bit exponent 8x6 N-tuple
Stage count: 2
Stage thresholds: 0, 16384
Phase: 2
Global episodes: 10,000,000
OTD episodes: 9,000,000
TC episodes: 1,000,000
OTD budget: 9,000,000
TC budget: 1,000,000
Training seed: 1
has_coherence_stats: true
alpha_normalizer: 128

The checkpoint layout is:
256-byte header + inference weight plane + TC coherence-statistics tail.

The existing M3 loader rejected TC checkpoints because it only accepted the old no-TC layout.
M3 frozen code is not modified.
M6 uses an M6-only read-only loader that maps only the inference weight plane and ignores the training-only TC tail.

## Promotion evidence

The completed paired evaluation supplied before the source-project cleanup used:
- FORMAL final.bin at 10,000,000 episodes, TC complete
- ordinary control snapshot at 9,300,000 episodes
- the same frozen 2,000 game seeds
- the same `ntuple_m6_tool eval` path
- 4 workers

FORMAL mean score: 265,299
Control mean score: 230,338
FORMAL minus control mean: +34,961
FORMAL median: 285,290
Control median: 256,576
FORMAL p10: 164,290
Control p10: 80,489
FORMAL standard deviation: 79,315
Control standard deviation: 95,015
FORMAL wins / control wins / ties: 1,194 / 805 / 1
FORMAL per-seed win rate: 59.7%

Reported paired statistic was control minus FORMAL:
-34,961 with 95% CI [-40,489, -29,345].
Equivalently, FORMAL minus control is positive and the paired interval excludes zero.

Reach >=16384:
FORMAL 73.65% versus control 57.40%, +16.25 percentage points.

Reach >=8192:
FORMAL 96.15% versus control 89.10%, +7.05 percentage points.

No 32768 was reached by either side in this 2,000-game evaluation.

The original raw evaluation artifacts are no longer present because the source project was deliberately cleaned down to the final model.
Therefore this report preserves the supplied promotion result as planning evidence, while the exact promoted model identity is independently protected by the full SHA-256 above.

## Compatibility validation

The promoted checkpoint header parsed successfully with the existing U2048NT6 parser.

A read-only M6 compatibility smoke mapped the inference weight plane and used the existing C++ tuple backend.
Four representative afterstates produced finite values.

On all 256 frozen M3 semantic-profile states:
M6 promoted Teacher C++ formal-state leaf values were bit-identical to the existing Python tuple backend values.

A depth-3 `ExpectimaxTeacher(decision_depth=3, use_cache=True)` smoke completed successfully with finite legal root values and normal node/cache accounting.

Search semantics remain:
- V_tuple = RAW_TUPLE_HEURISTIC
- formal leaf = FORMAL_STATE_TUPLE_HEURISTIC
- depth-3 root action values = SEARCH_VALUE_RAW_LEAF

No absolute-score calibration is introduced.
M6 supervision remains policy/ranking-only.

## Frozen boundary after promotion

The M5 Student checkpoint is unchanged.
The M5 anchor dataset is unchanged.
Their historical Teacher provenance remains the old M3/M5 Teacher SHA:
`7192719323A073BA2B6B19B62CB7D46EF4AA90ECC8C4AE6BAF27AD0C51566A84`.

Only M6 correction relabeling uses the promoted Teacher SHA:
`908BA8B8D01A4AFF76D32BE65220D61B6ED702696AA83E10B41594EB5A25AACC`.

M6 must still pass its own exact Search correctness/throughput gate before any correction labels are generated.
This promotion does not waive or lower the frozen performance threshold.
