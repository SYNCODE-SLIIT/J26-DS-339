# Training-only intervention coverage audit - 13 October 2026

Gate B has started. The new `bpdr_input.coverage` tool compares candidate geometries and probes real rendered shifts without changing audit v0.3, research eligibility, manifests, reviews or model training.

## What was measured

The active Vocadito run supplies **30 training crops from 14 singers**. Only training crop/reference files were opened; development, PP1-evaluation and final-holdout recordings were not read. Crop/reference checksums were verified before and after; manifest/review hashes stayed unchanged.

All candidates retain the six shifts (-30, -20, -10, +10, +20, +30 cents), fine/coarse pitch tolerances, tracker agreement, minimum audited frames, coverage and outside-drift limits. Annotation boundary guards remain 0.10 seconds. Candidates only alter renderer ramp, analysis guard and minimum plateau. Reduced 0.05-second analysis guards exceed the YIN window's 0.04644-second half duration at 22,050 Hz; that calculation is not a substitute for actual acoustic auditing.

| Policy | Required reliable endpoint span | Ramp / analysis guard / minimum plateau | Reference-eligible crops / sources / singers | Probed singers with any passing shift | Probed singers passing all six |
|---|---:|---|---|---:|---:|
| Active v0.3 | 0.86 s | 0.08 / 0.10 / 0.30 s | 3 / 2 / 1 | 1 | 1 |
| Candidate 0.66 | 0.66 s | 0.08 / 0.05 / 0.30 s | 5 / 4 / 3 | 1 | 1 |
| Candidate 0.52 | 0.52 s | 0.06 / 0.05 / 0.20 s | 10 / 8 / 6 | 3 | 1 |
| Candidate 0.50 | 0.50 s | 0.05 / 0.05 / 0.20 s | 10 / 8 / 6 | 3 | 1 |

Reference eligibility also requires sufficient plateau and outside-anchor frame counts; it is an upper bound before acoustic tracker agreement or reviewer approval. The earlier informal 0.5-second span count of 11 crops was not an executable support/anchor check; this tool finds 10 candidates under the stricter condition.

The bounded acoustic probe selected **one crop per eligible singer**, using longest reference support and phrase ID, before any shift results. Six singers were probed, not all eligible crops. The table is therefore a pilot yield, not a dataset-wide acoustic-coverage estimate.

## Realized-shift findings

- S29 / `vocadito_37_p00`: all six shifts pass for all four geometries.
- S23 / `vocadito_29_p00` and S28 / `vocadito_36_p00`: five shifts pass for both shorter candidates; -10 cents fails the current coarse PESTO median-error limit.
- S10 / `vocadito_13_p01`: shorter support is available, but all shifts fail the full audit.
- S1 / `vocadito_3_p01`: no common reliable region after acoustic masking.
- S5 / `vocadito_6_p00`: common reliable region is too short even for the shorter candidates.

The -10 failures on S23/S28 require interpretation: independent YIN median errors are 2.01/1.58 cents and p90 errors 4.69/7.44 cents, with zero measured outside drift. PESTO reports approximately -33.33 cents, giving a 23.33-cent median error against a 20-cent coarse limit. This is consistent with the known PESTO pitch-bin stepping. The failure is an audit rejection, not proof that the audio shift itself is wrong. **No tolerance was relaxed**; any revised coarse-consistency rule needs a separate training-only calibration and re-audit, including wrong-shift and artifact cases.

Each passing shift also passed sample-count preservation, exact outside-support samples, zero-change identity, plateau/outside coverage and drift checks. Rendered WAVs and full per-shift audits are retained for inspection. None of this establishes musical acceptability or human intention.

## Decision and next work

Keep audit v0.3 active. The two shorter candidates widen potential support, but neither establishes the planned >=5 independently sourced voices with fully audited six-shift examples. Gate B remains open; no candidate is declared a production policy or approved training reference.

1. Diagnose/calibrate coarse PESTO consistency using the saved cases and independent evidence; retain deliberate wrong-shift/transition failure checks before any policy revision. Inspect other eligible crops rather than interpreting one crop as a singer-wide exclusion.
2. Build the common Vocadito/Jamendo source adapters and manifest/screening path on the existing stems. Add screened sources to address genuine reliable-support shortages; do not assume more songs fix tracker quantization.
3. Review suitable reference intervals independently with two musically experienced reviewers. Research eligibility remains zero until actual approvals and applicable audits pass.

## Evidence and reproduction

- Complete run: `runs/coverage/1993b73598b94feeb4de15d3588f65e9/summary.json` and `probes/`.
- Input data run: `runs/phase2/15fb678d1f44406ab4da3fddd7c2f809`.
- Initial interrupted attempt: `runs/coverage/05febbb30078419ca280b384d02263ee`; a new-tool checksum field-name mismatch was corrected before the complete run. This attempt is not credited as a completed probe.
- Regression: **51 tests pass** (the existing 48 plus three coverage safety checks), including frozen PESTO smoke. Log: `runs/coverage/regression_2026-10-13.txt`.

From this component folder:

```powershell
& '.\.venv\Scripts\python.exe' -B -m bpdr_input.coverage --probe-singers 6
```

Use `--probe-singers 0` for the reference-only geometry report. Every run gets a new folder; no active data/review pointers are replaced. Outputs are labelled `exploratory_not_research`. This is a coverage/audit-engineering result, not BPDR model accuracy.
