# M3 Teacher checkpoint

The real Teacher checkpoint is stored locally but is never committed to Git.

Frozen M3 filename: `m3/ordinary_td_comparator_ep4800000_7192719323a0.bin`

Expected SHA-256: `7192719323a073ba2b6b19b62cb7d46ef4aa90ecc8c4ae6baf27ad0c51566a84`

M3 freezes the ordinary-TD comparator episode 4,800,000 snapshot because the preregistered paired dev-fast comparison showed a higher mean score than the contemporaneous FORMAL snapshot. Tail metrics are not claimed to be uniformly better.

Verify with PowerShell: `Get-FileHash teacher_checkpoints\m3\ordinary_td_comparator_ep4800000_7192719323a0.bin -Algorithm SHA256`.

The upstream snapshot is provenance and initial-copy input only. M3 runtime reads the local copy. A future Teacher promotion must create a new immutable baseline and must not overwrite this file.
