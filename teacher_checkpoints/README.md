# M3 Teacher checkpoint

The real Teacher checkpoint is stored locally but is never committed to Git.

Frozen M3 filename: `m3/ordinary_td_comparator_ep4800000_7192719323a0.bin`

Expected SHA-256: `7192719323a073ba2b6b19b62cb7d46ef4aa90ecc8c4ae6baf27ad0c51566a84`

M3 freezes the ordinary-TD comparator episode 4,800,000 snapshot because the preregistered paired dev-fast comparison showed a higher mean score than the contemporaneous FORMAL snapshot. Tail metrics are not claimed to be uniformly better.

Verify with PowerShell: `Get-FileHash teacher_checkpoints\m3\ordinary_td_comparator_ep4800000_7192719323a0.bin -Algorithm SHA256`.

The upstream snapshot is provenance and initial-copy input only. M3 runtime reads the local copy. A future Teacher promotion must create a new immutable baseline and must not overwrite this file.


## M6 promoted Teacher checkpoint

M6 correction Teacher filename: `m6/formal_afterstate_td0_tc_ep10000000_908ba8b8d01a.bin`

Expected SHA-256: `908ba8b8d01a4aff76d32be65220d61b6ed702696aa83e10b41594eb5a25aacc`

This is the completed FORMAL 10,000,000-episode checkpoint (9M OTD + 1M TC).
It is a U2048NT6 v2 checkpoint with TC coherence statistics appended after the inference weight plane.

M3 remains frozen and continues to use the old 4.8M comparator file.
M5 Student/anchor provenance also remains tied to that old Teacher.
Only M6 correction relabeling uses this promoted checkpoint.

The M6 runtime uses an M6-only read-only loader that maps the inference weight plane and ignores the training-only coherence-statistics tail.
The M3 loader is intentionally unchanged.

Verify with PowerShell: `Get-FileHash teacher_checkpoints\m6\formal_afterstate_td0_tc_ep10000000_908ba8b8d01a.bin -Algorithm SHA256`.

Promotion evidence and compatibility details are recorded in `reports/m6/M6_TEACHER_PROMOTION.md`.
