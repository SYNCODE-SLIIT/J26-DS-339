# Saved phase-regression diagnosis — 15 October 2026

**Gate F — renderer validation remains open.** Completed: read-only diagnosis of all 18 sealed source results and all eight saved synthetic truth arrays. Planned: a separately declared filter/Hilbert-context ablation. No new filtering, prediction, extraction, mask change, fitting, corrected/candidate WAV or adoption. All ten original recording refusals remain; repaired-package and research-ready counts stay zero. Gate D/J4 is deferred by owner.

Run: `runs/phase_regression_diagnostics/9a019681b7184066b1a15a5c78d1645e/evidence.json`. The prior comparison is `runs/segment_phase/7cb66194012549168a6765ed034b855d`; its arrays, seal, source hashes and numerical code are unchanged. Diagnostic source/test snapshots and tracked input hashes accompany this run. Error-distance bins are descriptive, not new acceptance thresholds.

## Jamendo: amplitude/phase association, not established causation

`jamendo_1086669_s5953500` has 14 retained nonpositive local-frequency samples: [57,809,57,818) and [58,028,58,033), inside the unchanged failed [53,582,58,874) interval. They lie 138.140590–148.253968 ms from the nearest raw boundary, beyond the existing 100-ms guard. C's derived frequency there spans −3452.771242 to −2.293087 Hz; B remains positive at 316.945636–332.133243 Hz. Nonpositive analytic-phase derivatives are estimator failures, not physically negative vocal pitch.

Local analytic amplitude on those samples is 0.000197815–0.001797595; its median is 0.001194057, against guarded-island median 0.036392992. Thus bad-sample amplitudes are 0.543553–4.939398% of that median (median ratio 3.281008%). B's corresponding amplitudes are 29.330430–36.245829% of its own guarded-island median. This low filtered analytic amplitude co-occurs with unstable local phase; it does not establish physical silence, true voicing or a causal explanation. No new amplitude floor or bad-sample deletion is applied. B/C also differ in band center here: global 276.797492 Hz versus local 331.192459 Hz (+310.606745 cents), as well as processing extent. A controlled contrast is needed to separate those choices.

## 880-Hz controls: centers identical, context differs

| Control | Global/local center Hz | B extent | C raw extent |
| --- | ---: | --- | --- |
| constant-880 | 880.8533107247177, exactly equal | [0,88200) | [221,87980) |
| vibrato-880 | 881.1351521414963, exactly equal | [0,88200) | [221,88200) |
| constant-440 consistency control | 440.2232981632686, exactly equal | [0,88200) | [0,88200) |

With equal band centers and the same declared SOS/Hilbert/gradient procedure, changed processing extent is the remaining computational difference for both 880-Hz cases. Saved B/C cannot separate filter-context/padding effects from finite Hilbert-window effects. Constant-440 has matching extent/center and matching B/C error statistics; this is a consistency control, not general proof.

Distance is `min(i-start, end_exclusive-1-i)` from a raw reliable island boundary. Fixed bins are [0,100), [100,200), [200,500), [500,infinity) ms. All eight controls are tabulated in notebook section 14.14 with full retained/common-valid denominators and invalid/silence counts; no control is omitted. The largest C errors occur 100.181406 ms from an edge for constant-880 and 101.723356 ms for vibrato-880. Yet p95 error remains worse at >=500 ms: constant-880 0.353403→0.398951 cents; vibrato-880 0.419351→0.459320 cents. A larger guard is not established as a sufficient fix and is not applied. Samplewise differences are descriptive, not independent samples for significance tests or human musical accuracy.

## Prospective context ablation, not executed

The [machine-readable declaration](../runs/phase_regression_diagnostics/9a019681b7184066b1a15a5c78d1645e/context_ablation_protocol.json) is `fixed-center-filter-hilbert-context-ablation-0.1`. All 18 sources and every frozen raw/guarded/retained mask, local/global center, short-island rule, order-four band [0.7,1.3], odd padding 27, derivative calculation and 0.86-s geometry reference stay fixed.

- **B:** replay existing full-clip filter/Hilbert at the global center.
- **C:** replay existing island filter/Hilbert at the saved local center, including every failure.
- **D:** full-clip filter and full-clip Hilbert at the exact saved local center for each island; evaluate only its frozen intervals.
- **E:** use the identical full-clip filtered array from D, slice to the raw island, then Hilbert/gradient on that island; same frozen intervals.

B/D isolates band-center choice with full context. D/E isolates Hilbert extent on identical filtered values. E/C isolates filtering context while keeping the island Hilbert domain and local center. Full-context arms intentionally use original audio outside reliable islands, including gaps; this is an offline analysis-context diagnostic, not validated streaming, resynthesis support or permission to bridge edited regions. No cross-island phase-continuity or human-truth claim.

Implement/test independent toy fixtures and freeze/hash code before processing all 18 sources. Seal all predictions/results before separate synthetic scoring. Retain every whole-interval failure, invalid/full denominator and >=0.86-s count; report eight-control median/p95/max errors and common valid voiced intersection across all four arms, plus the same diagnostic edge bins. No support rescue by increasing guards, adding an amplitude floor, deleting bad samples, tuning per-source settings, fitting harmonics or generating WAVs. Interpret the fixed contrasts before separately declaring any revised estimator or downstream decomposition test. Positivity alone never establishes musical correctness or adoption.

## Verification

Five focused tests cover read-only behavior, bad-run endpoints, amplitude association, incomplete truth/mask rejection and full edge-bin/common-domain accounting. Regression logs, independent scalar checks, unchanged frozen hashes/reviews, fresh notebook execution and visual checks are recorded in `final_verification.json` in this run. All **199 component regression tests passed** (`runs/phase_regression_diagnostics_regression_2026-10-15.txt`). Independent checks verify 478 frozen input hashes, all 18 source/mask records, ten original recording refusals, both review files, all 14 bad samples and eight synthetic edge-bin results. Both notebooks executed from fresh kernels with zero errors, 25 plots and 40 audio widgets; the new diagnostic figure was visually inspected. These checks establish engineering execution, not musical accuracy. Historical runs remain evidence.

**Next — Gate F (open):** Implement and freeze the declared filter/Hilbert-context ablation, then compare saved B/C with new D/E on all 18 fixed sources; seal predictions before synthetic scoring. Preserve every mask/failure; no vocal correction or adoption. **Gate D/J4 remains deferred.**
