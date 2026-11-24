# Verification report

6 independently executed notebooks; 125 code cells; 0 errors; 65 passing automated tests. 14 plots and 8 audio players verified. Research acceptance remains incomplete.

Recomputed delivered-waveform metrics: {"cases": 4, "before_mae_cents": 27.44190742699078, "after_mae_cents": 14.663040953839834, "mean_paired_improvement_cents": 12.778866473150943, "relative_reduction_percent": 46.56697610087467}. These values retain every scoreable failure/withhold; exclusions and independent singer/recording counts remain explicit. No historical result, checkpoint, partition or acceptance threshold was changed by notebook cleanup.

| Check | Status | Observation |
|---|---|---|
| Calibration research acceptance | FAIL | 0 |
| Two-singer pilot acceptance | FAIL | False |
| Natural intended-note ground truth | BLOCKED | not present for current pilot |
| Blinded listening judgments | BLOCKED | automated fixtures only; no human approval claimed |
| Downstream extraction F1 | BLOCKED | MIDI/CSV connection only; reference unavailable |
| Population singer confidence interval | NOT APPLICABLE | one scoreable pilot singer; no population CI reported |
| Cross-file acoustic overlap beyond exact hashes | BLOCKED | exact decoded duplicates and source/singer partitions checked; near-duplicate/session overlap not established |

Archived notebook bytes and protected experiment hashes are checked separately from current notebook execution. Original recordings remain unchanged. Historical source locations are retained in the provenance manifests; superseded notebook sources can be restored from `results/notebook_cleanup/current.json`.
