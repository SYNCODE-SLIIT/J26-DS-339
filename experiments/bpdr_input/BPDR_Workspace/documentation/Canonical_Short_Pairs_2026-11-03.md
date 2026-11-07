# V55 canonical short pairs: frontend-matched rebuild adds little data

3 November 2026. Training-data engineering only. No model fit, activation, nontraining audio access or threshold change. MOD-03 remains ACTIVE; usable accuracy is unproved.

## What ran
`tools/build_canonical_short_pairs.py` rebuilt interventions on the V54 canonical (app-frontend-matched) working copies. It reused historical pairs only where the parent PCM24 was byte-identical and rebuilt the rest with the unchanged three-tracker audit. All 528 attempts (33 recordings x 4 widths x 4 offsets) remain counted. Output: `results/transformer_usability/2026-11-02_v55_canonical_short_pairs/`.

## Counts
| Item | Count |
|---|---:|
| Attempted pairs | 528 |
| Valid pairs | 188 (V45 had 115 plus the HumTrans pack) |
| Reused unchanged / newly built | 176 / 12 |
| Valid originals | 32 of 33 |
| Excluded: no single reliable region with 150 ms context | 304 |
| Excluded: three-tracker physical audit failed | 20 |
| Excluded: source shorter than five seconds | 16 |

Valid pairs by group: S5 10, S7 11, S11 8, S13 4, S22 4, S23 24, S25 2, S28 24, S29 32, HumTrans F02 9, F03 16, F04 28, F05 8, M01 8.

## Interpretation
- Frontend parity recovers four recordings but only 12 new valid pairs. The dominant limit is the lack of stable single regions with 150 ms context on both sides (304 exclusions), not preprocessing.
- S25 still has 2 valid pairs and S13/S22 have 4, so per-group coverage targets cannot be met. Another fit would again be dominated by a few groups (S29, F04, S23, S28).
- Decision: **no new fit** is justified by +12 pairs. Retain V46/V51 results as the comparators. A fit on this pack should be declared only together with a change that addresses region availability or objective (the full BPDR vs augmentation-only vs frame-only comparison), not an optimizer sweep.
- Not done: an independent replay verifier for V55 (no `verify_` tool exists yet). The pack is therefore unverified and not research-eligible.
