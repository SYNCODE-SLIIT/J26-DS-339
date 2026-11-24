# Audit diagnosis and independent review - 14 October 2026

The existing extracted material was reused. All eighteen saved v0.3 detunings replayed with the same five passes. A declared renderer comparison improves the observed count to ten passes on these three training crops, but loses two previously passing cases and still produces no crop passing all six shifts. The active renderer, geometry, confidence/error thresholds, reviews and eligibility remain unchanged. **Zero research-ready examples exist.** The future portable pipeline remains deferred.

## What explains the failures

The [diagnostic replay](../runs/audit_diagnostics/c71f8e700e0b43aeade92c6832f075e2/diagnostics.json) classifies all active requirements, retaining multiple causes per variant:

| Failure requirement | Variants failing |
|---|---:|
| Fine YIN median error >5 cents | 9 |
| Fine YIN p90 error >10 cents | 11 |
| Wrong fine shift direction | 1 |
| Common plateau coverage <75% | 4 |
| Coarse PESTO error/direction | 0 |
| Protected outside drift or sample preservation | 0 |

Counts overlap; they are not thirteen separate additive failures. This sample's thirteen failed variants are not fixed simply by changing coarse PESTO tolerance.

At track 1113639, crop offset 30 s, +20/+30 lose 9/11 of 32 plateau frames after rendering. Those frames are rejected by PESTO's fixed 0.8 confidence/voicing gate. Their median after-confidence is about 0.578/0.565. Their YIN p90 shift errors are about 2.89/2.15 cents, but YIN is not independent human ground truth or a replacement for the required common coverage. Both variants remain rejected. Other sources also show substantial fine shift errors; confidence loss alone does not explain every failure.

For all eighteen variants, each centered 2,048-sample YIN audit window lies wholly inside the unit-blend plateau and has **zero sample difference** from the corresponding whole-recording pitch shift. PESTO's plateau voiced mask is also identical for local versus whole-recording shift. This rules out the local crossfade as the cause of these measured interior differences under the tested configurations. The whole-recording shift is a diagnostic contrast only: it alters protected outside audio and cannot replace the local renderer.

## Controlled signals distinguish an estimator limit from a renderer limit

The replay includes harmonic constant-440-Hz and 220-Hz vibrato signals (35-cent depth, 5 Hz), with six requested shifts each. An analytically generated waveform scales the known F0 contour at the same timestamps; the existing phase-vocoder path is measured against the same requested shift. Analysis uses the interior 1-3 seconds of a four-second signal.

All analytic controls have fine p90 error below 1.97 cents. The existing renderer passes the fine limits for all six constant-tone shifts, with p90 below 0.91 cents. Its vibrato -30/-20 shifts have p90 errors of 13.35/10.57 cents, exceeding the unchanged 10-cent limit. Thus known nonstationary pitch provides evidence of renderer deviation; YIN can measure the matched analytic reference much more accurately. This does not determine the true F0 or audible acceptability of separated real vocals. It supports testing renderer behavior on expressive contours rather than shortening guards or lowering confidence requirements to increase counts.

## Declared candidate comparison

The [comparison manifest](../runs/renderer_candidates/966a7845435e4a198ab02af828714b39/comparison.json) froze three configurations before measuring them. All use the same three source crops, supports, masks, six shifts, fixed audit thresholds, local envelope and frozen analyzers. No held-out audio or new separation/decoding was used. Candidates were evaluated in memory; this run creates no training pairs or accepted repaired outputs.

| Renderer settings | Audit passes /18 | Crops with any pass | Artist proxies | Crops passing all six |
|---|---:|---:|---:|---:|
| Active default FFT 2048 / hop 512 | 5 | 2 | 2 | 0 |
| FFT 2048 / hop 128 | 6 | 3 | 2 | 0 |
| FFT 1024 / hop 128 | 10 | 3 | 2 | 0 |
| FFT 1024 / hop 256 | 9 | 3 | 2 | 0 |

FFT 1024/hop 128 retains only three of the five active passes, loses two and adds seven. Its vibrato-control fine p90 error is at most 3.71 cents across six shifts, versus 13.35 cents for the default. FFT 1024/hop 256 retains four active passes and loses one; its vibrato-control p90 is at most 6.86 cents. Aggregate improvement is not dominance or demonstrated BPDR accuracy. Fine errors and PESTO coverage failures remain on real vocal material.

**Decision:** retain v0.3 and the active renderer. Treat FFT 1024/hop 128 and hop 256 as candidates for a wider, versioned training-only calibration and audible-artifact comparison, including losses of currently passing cases. Do not select a different renderer per shift/crop to create an apparent union of successes. Freeze a justified choice before development/evaluation; human approvals remain mandatory.

## Human review is now prepared

The [independent review bundle](../runs/review_bundles/5d35fe91ab1846ee8ab56e16289879ed/manifest.json) links the nine unchanged candidate WAVs without copying them. It contains:

- [Listening page](../runs/review_bundles/5d35fe91ab1846ee8ab56e16289879ed/listen.html).
- [Instructions](../runs/review_bundles/5d35fe91ab1846ee8ab56e16289879ed/instructions.md).
- [First reviewer sheet](../runs/review_bundles/5d35fe91ab1846ee8ab56e16289879ed/reviewer_1.csv) and [second reviewer sheet](../runs/review_bundles/5d35fe91ab1846ee8ab56e16289879ed/reviewer_2.csv), each with nine pending rows and empty reviewer identities/timestamps.

Three crops retain Vocadito human F0/A1/A2 references; six Jamendo crops from three artists retain `model_estimate` provenance. Reviewers independently assess acceptable single-voice reference singing, leakage/backing voices, unclear transitions and crop edges, then supply actual identity/experience, decisions, reasons, UTC timestamps and crop-relative stable intervals. No one was contacted and no approvals were inferred. The added observation fields supplement the existing review fields; the bundle has no automatic importer. A generic preparation/review manifest still cannot bypass the laboratory loader's approval, interval, integrity and intervention gates.

Complete working copies of the assigned sheets and return those copies; preserve the original pending templates and bundle hashes. Filled submissions will require their own recorded hashes and review validation in the next collection version.

## Validation and next action

All **78 regression checks passed**, including the new diagnostic classification, confidence-mask selection, full-window comparison and review-selection checks. [Regression log](../runs/audit_diagnostics_regression_2026-10-14.txt). The real replay reproduces all eighteen baseline decisions, and the declared comparison evaluates 54 real-crop variants plus 36 matched constructed controls. Source/code snapshots and saved audio/review hashes are verified; existing source runs and review sheets remain untouched.

Next: obtain the two actual independent reference reviews while extending the declared renderer comparison to more diverse existing training references and expressive controls. Assess audible artifacts and losses of previously passing cases, then decide whether a new renderer policy is justified. Keep research fitting, corpus expansion and portable-pipeline integration behind their existing gates.

Reproduce from this component folder (new immutable evidence directories are created):

```powershell
$env:OMP_NUM_THREADS='8'
$env:MKL_NUM_THREADS='8'
& '.\.venv\Scripts\python.exe' -B -m bpdr_input.audit_diagnostics '.\runs\jamendo_yield\de2f85a01b194c4eab009aa95b14e496\evidence.json'
& '.\.venv\Scripts\python.exe' -B -m bpdr_input.renderer_candidates '.\runs\audit_diagnostics\c71f8e700e0b43aeade92c6832f075e2\diagnostics.json'
& '.\.venv\Scripts\python.exe' -B -m bpdr_input.review_bundle '.\runs\preparation\d274ed728d3d46f3b2661ba5291e61da'
```
