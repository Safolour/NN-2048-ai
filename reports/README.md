# Milestone Reports

This directory is the committed report/compact-summary home for all milestones on the current branch.

Canonical layout:
- `reports/m0/`: M0 report.
- `reports/m1/`: M1 report.
- `reports/m2/`: M2 report and committed compact benchmark/correctness summaries.
- `reports/m3/`: M3 candidate/audit reports and committed compact Teacher/Search summaries.
- `reports/m4/` and later: the same pattern for future milestones.
- `artifacts/mN/`: raw, large, intermediate, checkpoint and per-game/per-run artifacts; local-only by default.
- `prompts/`: implementation work orders.
- `docs/`: specifications and engineering documentation.

The repository root must not contain milestone reports or milestone result JSON files.

Historical frozen tags still preserve the old pre-cleanup paths exactly. The current branch intentionally uses this unified layout and updates current scripts/prompts to the canonical paths above.
