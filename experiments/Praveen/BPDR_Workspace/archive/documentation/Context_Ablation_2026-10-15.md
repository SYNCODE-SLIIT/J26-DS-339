# Fixed-center filter/Hilbert context ablation — 15 October 2026

**Gate F — renderer validation remains open.** Completed: declared B/C/D/E comparison on all ten recordings and eight synthetic controls, with all predictions sealed before truth scoring. No uniform winner, new long support, correction, fit, adoption or musical approval. Original ten recording refusals remain unchanged; repaired-package and research-ready counts remain zero. Gate D/J4 is deferred by owner.

Run: `runs/context_ablation/354b67b2137f4237a67078327dcf7fb1/evidence.json`; implementation `src/bpdr_input/context_ablation.py`, `context_ablation_scoring.py`, `context_ablation_experiment.py`. The prior protocol remains unchanged at `runs/phase_regression_diagnostics/9a019681b7184066b1a15a5c78d1645e/context_ablation_protocol.json`, SHA-256 `10319621d2777ee4686e9c97fe906dbcc8dee5429400d1dceb90f954edc3ef64`. Runtime versions, source/test snapshots and code hashes were declared before processing. All 18 D/E NPZ artifacts and source-only result records seal before synthetic scorer import/construction truth. Every source, short island, failed interval and denominator remains recorded.

## Fixed computational contrasts

| Arm | Band center | Filtering domain | Hilbert domain |
| --- | --- | --- | --- |
| B, unchanged replay | global median | full decoded clip | full clip |
| C, unchanged replay | saved local median | raw reliable island | island |
| D, new | identical saved local median | full clip | full clip |
| E, new | same local median as D/C | identical full-clip filtered array from D | slice to island first |

No center/mask recomputation, padding choice, amplitude gate, phase alignment, support expansion or bad-sample removal. Butterworth order4, band[0.7,1.3], explicit odd SOS pad27, Hilbert/unwrap/gradient are fixed. Short raw islands <=4,410 samples are recorded without D/E filtering. Phase origins are independent across islands. Full context intentionally includes unreliable/gap audio; it is offline computational context, not validated streaming or resynthesis support. Whole intervals fail on any nonpositive/nonfinite frequency; >=0.86 s is geometry, not permission.

## Every group and contrast

| Group | Frozen intervals | B feasible | C feasible | D feasible | E feasible | Long feasible in every arm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Vocadito, 3 crops / 3 singer groups | 55 | 55 | 55 | 55 | 55 | 0 |
| Jamendo, 7 crops / 4 artist proxies | 182 | 175 | 181 | 181 | 181 | 7 |
| Synthetic, 8 controls | 9 | 9 | 9 | 9 | 9 | 9 |

B→D changes only band center with full filtering/Hilbert context and reproduces Jamendo's seven gains/one loss. D→E and E→C change no whole-interval outcome. The local-center choice is therefore sufficient to reproduce the new computational failure without cropping the filter or Hilbert domain. This does not establish human pitch correctness or a general causal explanation for vocals.

The failed [53,582,58,874) interval in `jamendo_1086669_s5953500` remains failed in C/D/E: B/C/D/E retained nonpositive counts 0/14/14/13. Across all Jamendo support counts are 54/14/14/13. The existing guards and masks remain unchanged; none of these values is removed. All arms retain the same seven long Jamendo intervals across the same four crops; Vocadito still has none. No extra usable-duration support is gained. Jamendo artist identity is not verified singer identity; this is a bounded development sample, not held-out or population validation.

## Synthetic frequency errors and tradeoffs

All arms have 707,584 retained/common-four-arm voiced samples, zero retained silence and zero invalid retained predictions. Median/p95/max for every control and all fixed edge bins are in notebook section 14.15 and `evidence.json`. No success-only scoring or best-arm selection.

| Control | Arm | Median cents | p95 cents | Maximum cents |
| --- | --- | ---: | ---: | ---: |
| constant_110 | B | 0.302764 | 1.937492 | 5.394208 |
| constant_110 | C | 0.253345 | 0.370008 | 0.632700 |
| constant_110 | D | 0.302764 | 1.937492 | 5.394208 |
| constant_110 | E | 0.289318 | 1.826843 | 5.331413 |
| constant_220 | B | 0.229938 | 0.942597 | 2.670428 |
| constant_220 | C | 0.225461 | 0.545015 | 1.445270 |
| constant_220 | D | 0.229938 | 0.942597 | 2.670428 |
| constant_220 | E | 0.241925 | 0.383297 | 0.722410 |
| constant_440 | B | 0.219046 | 0.598209 | 1.653205 |
| constant_440 | C | 0.219046 | 0.598209 | 1.653205 |
| constant_440 | D | 0.219046 | 0.598209 | 1.653205 |
| constant_440 | E | 0.219046 | 0.598209 | 1.653205 |
| constant_880 | B | 0.208048 | 0.398635 | 0.834539 |
| constant_880 | C | 0.201499 | 0.514971 | 1.410591 |
| constant_880 | D | 0.208048 | 0.398635 | 0.834539 |
| constant_880 | E | 0.211492 | 0.385765 | 1.013375 |
| vibrato_220 | B | 0.232874 | 0.959480 | 2.745659 |
| vibrato_220 | C | 0.223107 | 0.577990 | 1.473888 |
| vibrato_220 | D | 0.232874 | 0.959480 | 2.745659 |
| vibrato_220 | E | 0.228328 | 0.857452 | 2.632221 |
| vibrato_880 | B | 0.201811 | 0.493071 | 1.206325 |
| vibrato_880 | C | 0.199679 | 0.637913 | 1.810240 |
| vibrato_880 | D | 0.201811 | 0.493071 | 1.206325 |
| vibrato_880 | E | 0.202888 | 0.473532 | 1.233759 |
| glide_220 | B | 0.244942 | 1.227907 | 3.209815 |
| glide_220 | C | 0.205375 | 0.670720 | 1.244810 |
| glide_220 | D | 0.244942 | 1.227907 | 3.209815 |
| glide_220 | E | 0.215656 | 0.894205 | 2.264957 |
| rested_220 | B | 0.213096 | 0.796586 | 2.674890 |
| rested_220 | C | 0.235487 | 0.478088 | 1.463495 |
| rested_220 | D | 0.213096 | 0.796586 | 2.674890 |
| rested_220 | E | 0.221752 | 0.656601 | 1.788905 |

For both 880-Hz controls, D is exactly B on retained frequency samples. Holding the full filtered signal fixed while changing Hilbert domain (D→E) improves p95 but increases the maximum. Changing filtering context with the island Hilbert domain fixed (E→C) worsens both p95 and maximum. Both operations affect tails, with a tradeoff rather than a universal fix. Constant-440 is identical across all arms; other controls also show mixed outcomes. These constructed three-harmonic signals do not establish natural-vocal fidelity, phase closure, audible quality, intended-note accuracy or adoption.

## Declared next stage, not implemented/executed

The [unchanged-source harmonic feasibility protocol](../runs/context_ablation/354b67b2137f4237a67078327dcf7fb1/harmonic_feasibility_protocol.json) fixes all four arms and 18 sources. It reuses saved phase/amplitude/frequency and masks, reports every interval including short/invalid/empty cases, and preserves original A refusals. Eligible whole intervals require >=18,963 samples, finite phase/amplitude and positive finite frequency throughout. No new amplitude gate or center tuning.

Fit the six sin/cos coefficients for harmonics 1–3 to original audio's first floor(N/2) interval samples with `np.linalg.lstsq(rcond=None)`, require rank6/finite coefficients, and evaluate the rest. Save coefficients, model, fundamental-only model and residual in NPZ, not WAV. Record rank/singular values/conditioning, source/residual RMS, raw relative RMSE/explained energy and unforced float64/float32 recombination discrepancies. Source zero RMS is explicit unavailable/refusal, not an invented zero or epsilon denominator. Phase used full context, so this is a coefficient sample split, not independent validation.

Analytic amplitude is descriptive. Seal every source-only fit and disposition before synthetic fundamental/amplitude scoring with full/valid/voiced/silent counts. The constructed controls match the model family; natural-vocal correctness remains unproven. No requested curve, pitch correction, truth-based fitting, waveform output or policy adoption. Interpret representation/amplitude limitations before separately declaring any nonzero renderer test.

## Verification

Eight focused toy-signal tests cover shared filtering and distinct Hilbert domains, full-clip equality, short islands, filter refusal, immutable masks/gaps, fixed centers and four-arm common/invalid/silent denominators. Full component regression, independent saved-array replay and scoring checks, unchanged source/review hashes, fresh kernels for both maintained notebooks and visual inspection are recorded in `final_verification.json` in this run. All **207 component regression tests passed** (`runs/context_ablation_regression_2026-10-15.txt`). Independent replay exactly matches 148 full filters and 296 D/E island array sets; all 390 short islands, 484 frozen hashes, 18 source/mask records, ten original recording refusals and both review files are preserved. Whole-interval outcomes and all eight four-arm synthetic score/edge-bin tables independently verify. Both notebooks freshly executed with zero errors, 26 plots and 40 audio widgets; the new figure was visually inspected. These establish execution, not musical accuracy. Historical runs and notebooks remain evidence.

**Next — Gate F (open):** Implement and freeze the declared unchanged-source amplitude/harmonic feasibility diagnostic across B/C/D/E and all 18 sources; preserve every short/invalid interval and seal source-only fits before synthetic scoring. No waveform correction or adoption. **Gate D/J4 remains deferred.**
