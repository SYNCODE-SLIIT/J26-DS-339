# Fixed-mask segment-phase comparison — 15 October 2026

**Gate F — renderer validation remains open.** The declared diagnostic is implemented and executed on all ten recordings and eight synthetic controls. Results are mixed: seven Jamendo interval gains and one loss, no additional interval reaching 0.86 s, and synthetic regressions. No method is adopted; no waveform correction, harmonic fit, repaired package or research-ready example. Gate D/J4 remains deferred by owner.

## Declaration, implementation and seal

Run: `runs/segment_phase/7cb66194012549168a6765ed034b855d/evidence.json`. Candidate `src/bpdr_input/segment_phase.py`, version `segment-local-phase-diagnostic-0.1`; runner `segment_phase_experiment.py`; separate post-seal scorer `segment_phase_scoring.py`. The prospective [protocol](../runs/support_fragmentation/fc21ac54b04a4ee8b623f684c374f613/segment_comparison_protocol.json), SHA-256 `eb4fab68c50dce995810f0319ab1ede1e2e8b63ae7e5e475bcbb2eb1767938f9`, remains unchanged. Code/test snapshots and hashes in the execution declaration precede source processing. All 18 source-only predictions and NPZ hashes are saved in `prediction_seal.json` before importing the synthetic scorer or reading construction profiles. No truth, labels, paths or requested curves enter the candidate API.

Exact frozen sample reliability, guards and retained masks; local median from unique represented reliable YIN frames; order-four Butterworth band [0.7,1.3] times local center; odd SOS padding explicitly 27; local Hilbert/unwrap/gradient; raw islands <=4,410 samples recorded short. No bridging, alternate filter, tuning, fallback, bad-sample removal or new amplitude gate. Fixed support is a comparison domain, not validated local-estimator/resynthesis support. Phase origins are independent across islands.

Initial run `dd8239cb29424956a7e683c66f822a3b` failed on duplicate input/output prediction-hash field names before recording any completed comparison row or accessing truth. Its failure record, first prediction artifact and code snapshot are preserved. The successful execution fixes only that bookkeeping name; candidate/scorer numerical code and declared parameters are unchanged. Execution logs are separate (`runs/segment_phase_execution_2026-10-15.txt`, `runs/segment_phase_execution_retry_2026-10-15.txt`).

## Every source and refusal retained

A preserves original v0.1 status/arrays. B checks saved global phase on whole frozen retained intervals. C checks locally filtered phase on the same intervals. An interval is phase-feasible only if every derived frequency is finite/positive; a single invalid sample fails the whole interval. A/B is refusal scope only; B/C compares local phase on an identical domain. These diagnostics do not confer source acceptance.

| Source | A status | Frozen retained intervals | B / C phase-feasible | C gains / losses | B / C feasible >=0.86 s |
| --- | --- | ---: | ---: | ---: | ---: |
| vocadito_vocadito_10_p00 | refused | 21 | 21 / 21 | 0 / 0 | 0 / 0 |
| vocadito_vocadito_13_p00 | refused | 21 | 21 / 21 | 0 / 0 | 0 / 0 |
| vocadito_vocadito_14_p00 | refused | 13 | 13 / 13 | 0 / 0 | 0 / 0 |
| jamendo_1071484_s4189500 | refused | 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| jamendo_1086669_s5953500 | refused | 22 | 22 / 21 | 0 / 1 | 0 / 0 |
| jamendo_1086669_s7056000 | refused | 18 | 18 / 18 | 0 / 0 | 1 / 1 |
| jamendo_1101810_s1984500 | refused | 60 | 60 / 60 | 0 / 0 | 2 / 2 |
| jamendo_1101810_s11466000 | refused | 3 | 3 / 3 | 0 / 0 | 3 / 3 |
| jamendo_1113639_s1323000 | refused | 50 | 45 / 50 | 5 / 0 | 1 / 1 |
| jamendo_1113639_s8820000 | refused | 29 | 27 / 29 | 2 / 0 | 0 / 0 |
| constant_110 | estimated | 1 | 1 / 1 | 0 / 0 | 1 / 1 |
| constant_220 | estimated | 1 | 1 / 1 | 0 / 0 | 1 / 1 |
| constant_440 | estimated | 1 | 1 / 1 | 0 / 0 | 1 / 1 |
| constant_880 | estimated | 1 | 1 / 1 | 0 / 0 | 1 / 1 |
| vibrato_220 | estimated | 1 | 1 / 1 | 0 / 0 | 1 / 1 |
| vibrato_880 | refused | 1 | 1 / 1 | 0 / 0 | 1 / 1 |
| glide_220 | refused | 1 | 1 / 1 | 0 / 0 | 1 / 1 |
| rested_220 | refused | 2 | 2 / 2 | 0 / 0 | 2 / 2 |

Vocadito: **55/55 both**, zero long intervals. Jamendo: **175/182 B versus 181/182 C**, seven gains and one loss. B has 606,927 whole-feasible retained samples; C has 617,605, against the same 622,897 retained samples. There are seven >=0.86-s feasible Jamendo intervals in both arms, across the same four crops, so C adds no long interval. All 390 short real raw islands stay short; the empty-support crop stays empty. All ten original recording refusals remain unchanged (nine global nonpositive-frequency refusals, one empty guarded support). Diagnostic B/C status is separate from A; no refusal is reclassified.

C reduces nonpositive estimates inside Jamendo retained support from 54 to 14. Its new failure is `jamendo_1086669_s5953500`, [53,582,58,874) samples (0.24 s), within raw island [51,377,61,079) (0.44 s), local median 331.1924594636341 Hz. Fourteen nonpositive retained estimates fail that whole interval; B had no invalid retained frequency there. The seven gained intervals belong to `1113639_s1323000` (five) and `1113639_s8820000` (two). No cause for the new failure has yet been established.

Three Vocadito crops represent three verified singer groups; seven Jamendo crops represent four artist proxies, not verified singers. This is a bounded development sample, not held-out or population validation. Tracker agreement, positivity and geometry are not human-voicing coverage, correct intended pitch or musical accuracy.

## Synthetic truth scoring, only after predictions sealed

Exact source reconstruction is checked by the separate scorer. Across all eight controls, both arms have nine feasible retained intervals, 707,584 retained samples, zero invalid retained frequency values and zero retained samples in constructed silence. Common-valid voiced domain equals the full retained domain in every row; invalid counts and full denominators are saved anyway. Descriptive frequency error is absolute cents, with no phase alignment or harmonic fit.

| Profile | Retained/common-valid samples | B median / p95 / max, cents | C median / p95 / max, cents |
| --- | ---: | --- | --- |
| constant_110 | 83349 | 0.302764 / 1.937492 / 5.394208 | 0.253345 / 0.370008 / 0.632700 |
| constant_220 | 83349 | 0.229938 / 0.942597 / 2.670428 | 0.225461 / 0.545015 / 1.445270 |
| constant_440 | 83790 | 0.219046 / 0.598209 / 1.653205 | 0.219046 / 0.598209 / 1.653205 |
| constant_880 | 83349 | 0.208048 / 0.398635 / 0.834539 | 0.201499 / 0.514971 / 1.410591 |
| vibrato_220 | 83349 | 0.232874 / 0.959480 / 2.745659 | 0.223107 / 0.577990 / 1.473888 |
| vibrato_880 | 83569 | 0.201811 / 0.493071 / 1.206325 | 0.199679 / 0.637913 / 1.810240 |
| glide_220 | 83349 | 0.244942 / 1.227907 / 3.209815 | 0.205375 / 0.670720 / 1.244810 |
| rested_220 | 123480 | 0.213096 / 0.796586 / 2.674890 | 0.235487 / 0.478088 / 1.463495 |

The 880-Hz constant and vibrato p95/maxima worsen. The rested-220 median also worsens; constant-440 is identical. Improvements elsewhere do not establish uniform gain or justify adopting C. Synthetic small errors do not establish phase closure, natural-vocal amplitude/harmonic fidelity, audible quality or strict renderer success. No research accuracy claim is made for recordings without a validated reference.

## Checks and next gate action

Ten independent toy-signal tests verify gap preservation, guards/short islands, local centers, immutable inputs, invalid bands/filter failures/nonfinite analytic signals and whole-interval refusal. Three scorer tests verify full denominators, paired common support and silent/invalid prediction accounting. Full component regression, independent saved-array replay, unchanged source/review hashes, and fresh execution/visual checks of both maintained notebooks are recorded in `final_verification.json` in this run. All **194 component regression tests passed** (`runs/segment_phase_regression_2026-10-15.txt`). Independent replay exactly matches all 148 computed local islands; 390 short islands remain excluded. All 455 frozen hashes, 18 source/mask records, ten original recording refusals, both review files and eight synthetic score rows verify. Both notebooks completed with zero errors, 24 plots and 40 audio widgets; the new figure was visually checked. These are engineering/diagnostic checks, not musical accuracy. Historical notebooks and failed attempts remain evidence.

**Next — Gate F (open):** Diagnose the newly failed 0.24-s Jamendo interval and the constant/vibrato 880-Hz synthetic regressions using saved A/B/C arrays before declaring any revised phase method or downstream decomposition test. Preserve masks and all failures; no vocal correction. **Gate D/J4 remains deferred.**
