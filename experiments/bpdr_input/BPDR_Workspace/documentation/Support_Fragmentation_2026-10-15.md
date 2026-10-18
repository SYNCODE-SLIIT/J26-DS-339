# Frozen support diagnosis and segment-phase declaration — 15 October 2026

**Gate F — usable delivered audio / renderer validation remains open.** Completed: read-only diagnosis of ten frozen source predictions and a precise prospective comparison declaration. Planned: candidate implementation and 18-source execution. All ten v0.1 recording refusals remain unchanged. No vocal correction, decomposition, musical approval, repaired package, training-ready example or adoption. Gate D/J4 is deferred by owner.

## Saved evidence and measured findings

Run: `runs/support_fragmentation/fc21ac54b04a4ee8b623f684c374f613/evidence.json`. The saved masks exactly recompute from the confidence >=0.8, valid-positive PESTO/YIN, <=50-cent agreement, nearest-frame mapping, 2,205-sample guards and original amplitude floor. No tracker rerun or new audio processing. All prior tracked hashes and prediction artifacts verify.

| Measurement | Vocadito, 3 crops | Jamendo, 7 crops |
| --- | ---: | ---: |
| All tracker frames | 2,002 | 7,007 |
| PESTO/confidence excluded | 552 | 2,672 |
| Invalid YIN after PESTO check | 0 | 0 |
| Disagreement >50 cents after validity | 196 | 754 |
| Reliable frames | 1,254 | 3,581 |
| Predicted reliable samples | 553,014 | 1,579,001 |
| Samples removed by guards | 350,154 (63.3%) | 901,845 (57.1%) |
| Samples removed next by amplitude floor | 8,842 | 54,259 |
| Retained samples / all decoded samples | 194,018 / 882,689 | 622,897 / 3,087,000 |
| Raw reliable islands | 124 | 405 |
| Raw islands <=0.20 s | 81 | 309 |
| Retained runs >=0.86 s | 0 | 7 |
| Reliable YIN frames outside global filter band | 160 / 1,254 | 592 / 3,581 |

Frame exclusions are exclusive in evaluation order. Saved low-confidence F0 is missing, so the combined PESTO/confidence category cannot identify separate physical causes. YIN validity after PESTO is not an all-source YIN assessment. The 390/529 islands at most 0.20 s have no interior after both existing 0.10-s end guards. The empty-support crop `jamendo_1071484_s4189500` has five islands, all at most 0.08 s. Guards are measured losses, not faulty behavior to remove without validation.

The out-of-band YIN count describes the frozen global [0.7,1.3] × median-Hz filter against reliable tracker estimates. It motivates local-filter testing but does not prove a cause for nonpositive analytic-phase derivatives, tracker correctness or dataset-wide quality. Intermediate mask coverage is against decoded samples, not human voicing. Three Vocadito crops are three verified singer groups; seven Jamendo crops are four artist proxies, not seven verified singers. No human Jamendo F0 reference or held-out generalization claim.

## Declared next experiment, not executed

The machine-readable [protocol](../runs/support_fragmentation/fc21ac54b04a4ee8b623f684c374f613/segment_comparison_protocol.json) is `frozen-mask-segment-phase-comparison-0.1`; candidate version `segment-local-phase-diagnostic-0.1`. It fixes all ten recording audio/NPZ hashes and all eight existing synthetic controls. No selection by outcomes or new extraction.

1. **A, frozen v0.1:** preserve status, global-frequency refusal and arrays exactly.
2. **B, global phase with segment-scoped diagnostic:** inspect saved frequency on each whole frozen retained interval. Positive finite frequency throughout is phase feasibility only; preserve A's refusal.
3. **C, local phase with fixed support:** filter separately within each maximal raw reliable island. Evaluate the exact same retained intervals as B; do not split around bad samples or bridge gaps.

A/B isolates refusal scope descriptively. B/C compares global versus local phase on an identical domain. No successful diagnostic is adopted as an acceptance policy.

The candidate accepts only decoded mono float32 audio at 22050 Hz and frozen source-only arrays, with no paths, dataset/profile labels, human notes/F0 or correction curves. Local center is median YIN over unique nearest-frame indices represented by the raw reliable island; these must be reliable. Islands <=4,410 samples are recorded as short without filtering. Longer islands use fourth-order Butterworth SOS band [0.7,1.3] × local center, odd padding explicitly 27, Hilbert/unwrapped phase, sine phase +pi/2, frequency gradient ×22050/(2pi). Invalid band, filter errors or nonfinite outputs are recorded failures; no fallback/retry. No phase continuity across islands is claimed.

Freeze the existing guards and retained masks exactly. No new amplitude floor is fitted: support is a fixed comparison domain, not validated local-estimator or resynthesis support. Report local amplitude descriptively. A whole retained interval is phase-feasible only when every derived frequency is finite/positive. Separately count intervals at least 18,963 samples (0.86 s); this is renderer geometry, not permission or a successful strict audit. Local phase cannot lengthen the frozen Vocadito longest retained intervals (0.54/0.40/0.82 s).

Implement and test with independent toy fixtures, then freeze/hash code in a separate execution declaration **before** processing these 18 sources. Save/hash source-only predictions before a separate synthetic scorer accesses construction truth. Report per-source/per-interval failures, local centers, invalid counts, full retained denominator and paired B/C feasibility. Synthetic absolute frequency cents errors use median/p95/max on positive finite predictions; report the common-valid intersection and invalid counts/full denominator so refusals cannot improve accuracy by selective omission. No phase-offset fitting, harmonic reconstruction or waveform correction is in scope. Recordings have descriptive feasibility only.

Even a successful result leads only to a separately declared decomposition/amplitude/strict-renderer test. If support is too short, report that block; do not lower 0.86 s, bridge gaps, tune per-source parameters or call development-source success generalization. No harmonic fitting, candidate WAV, learner training, review modification, policy adoption or musical approval.

## Verification

Six focused tests cover exclusive frame accounting, observed zero support, mask tampering, amplitude loss, half-open intervals and no array mutation. All **181 component regression tests passed**, recorded in `runs/support_fragmentation_regression_2026-10-15.txt`. Fresh execution/hash verification of both maintained panel notebooks is recorded in `final_verification.json` in this run. These checks establish diagnostic execution, not musical accuracy. Historical executed notebook copies and raw evidence are retained.

**Next — Gate F (open):** Implement and freeze the declared segment-local phase diagnostic, then execute its A/B/C comparison on all ten recordings and eight synthetic controls using unchanged masks; seal predictions before synthetic truth scoring. No vocal correction. **Gate D/J4 remains deferred.**
