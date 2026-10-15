# Audio-only source-estimation feasibility protocol — 15 October 2026

**Gate F open. Status at declaration: declared, not executed.**

**Execution update, 15 October:** this unchanged protocol has now run; [five estimated and three refused](Source_Estimation_2026-10-15.md). The original machine-readable declaration/hash is preserved as the prospective specification. This protocol follows the [frozen control diagnosis](Known_Control_Diagnosis_2026-10-15.md). It is the next bounded engineering check before defining a practical renderer; it is not a promised vocal correction method.

Machine-readable declaration: [source_estimation_protocol.json](../runs/known_control_diagnostics/bf53253259254eef89f9bb3fe09e24bd/source_estimation_protocol.json). Its selection, fixed parameters and source hashes are frozen before the future experiment. The declaration SHA-256 is recorded in [diagnosis final verification](../runs/known_control_diagnostics/bf53253259254eef89f9bb3fe09e24bd/final_verification.json). A runner must verify the declaration hash before and after measuring and snapshot estimator/scorer code separately. Any method change needs a separately labelled declaration/run; do not tune and overwrite this one.

## Question and input separation

Can source fundamental phase, frequency and amplitude be estimated from the eight already available constructed source WAVs, with no construction truth passed to the estimator? The runner loads all eight frozen sources. Its only estimator inputs are decoded float32 mono audio and sample rate. It must not pass a filename, profile ID, exact center frequency, construction F0/phase, supplied correction or prior mathematical solution. Profile labels and construction metadata are reserved for the scorer **after estimation**. This check generates no corrected waveform and makes no real-vocal generalization.

## Fixed estimator

1. Analyze the source with the existing frozen PESTO and YIN-2048, 441-sample hop. Use source frames with PESTO confidence at least 0.8, both finite positive estimates and agreement within 50 cents. Do not use known silence or the profile to set the estimator's reliable mask.
2. Set the filter center to median YIN Hz over those source-only reliable frames. Refuse explicitly if no such frames exist or the resulting band is invalid. Do not choose the center from construction labels.
3. Apply the existing fourth-order Butterworth bandpass at 0.7–1.3 times that estimated center, default odd-padded forward/backward SOS filtering, then Hilbert analysis. Estimate sine phase as `unwrap(angle(analytic)) + pi/2`, amplitude as `abs(analytic)`, and instantaneous frequency using `numpy.gradient(phase) * sr / (2*pi)`. Do not fit a global phase offset to oracle truth.
4. Map source-only reliable frames to sample times using nearest frame, later frame on ties. Exclude 100 ms at clip ends and either side of predicted reliable/unreliable transitions. On the remaining predicted reliable samples, retain analytic amplitude above 10% of its median. Report coverage and exclusions; never hide invalid samples or use truth to rescue prediction.
5. Refuse explicitly for no reliable frames, invalid Nyquist band, empty guarded support, nonfinite analytic values or nonpositive estimated frequency. Preserve the refusal row and all eight source rows; no fallback estimator.

The frequency band and signal model are engineering hypotheses for these constructed sources. They are not calibrated voice-support or edit-permission policies. A source-only result cannot close natural-vocal support, harmonic/residual reconstruction or strict phase-closure requirements.

## Fixed scoring and limits

After predictions are sealed, the separate scorer loads construction F0, sine phase and fundamental amplitude. Report predicted reliable/retained samples, true-voiced coverage, excluded true-voiced samples and any false support in constructed silence separately. Score only jointly retained positive-amplitude construction samples; do not discard other source rows from the report.

Report the estimated center, median/p90/max absolute frequency error in cents, median/p90/max absolute circular phase error without oracle alignment, and median/p10/p90 estimated-to-true fundamental amplitude ratio. Retain arrays, code, hashes and all refusals. These are descriptive feasibility metrics, with **no new production/musical threshold or automatic adoption**.

The eight sources are distinct from the eight correction conditions and 24 withheld requests; this experiment does not reopen those correction decisions. No candidate WAV, new correction solve, model fit, corpus expansion, research pair or review alteration is permitted in this stage. Experienced musical review remains deferred (Gate D/J4), research-ready examples and accepted delivery remain zero, and dual-OS integration remains planned.
