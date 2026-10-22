# Wider renderer calibration - 14 October 2026

The wider comparison is complete. FFT 1024/hop 128 improves the measured Jamendo and expressive-control results, but loses active successes on both datasets. **Retain the active renderer and v0.3 policy.** This experiment creates no research training pairs, approved references or delivered repairs. Research-ready examples remain **zero**.

## Frozen comparison and source limits

[Calibration manifest](../runs/renderer_calibration/7b8a3cfaab7349f5b574db6d7d552683/calibration.json). The selection, three configurations, six shifts and twelve control profiles were recorded before rendering. The same supports, masks, local envelope, fine/coarse thresholds and analyzers apply within every real-crop comparison. Held-out partitions were not opened.

- Vocadito: all three training crops admitted by the earlier **active-policy reference geometry**, not crops chosen after candidate outcomes. They are `vocadito_37_p00`, `vocadito_37_p01` and `vocadito_39_p00`: two recordings, **one singer (S29)**. Human F0/A1/A2 annotations supply screening references; actual musical approval is still pending. All three also pass the acoustic support screen. Existing preparation was reused for the first crop; the other two existing crops received unchanged technical preparation, with no new separation or GAME decoding.
- Jamendo: the same three previously inspected training crops, two songs and **two artist proxies**, from the saved yield experiment. This reproduces the previous comparison; it is not independent validation or a new voice sample. Model-estimate provenance stays explicit.
- Controls: harmonic signals with amplitude modulation, constant pitch and 35-cent/5-Hz vibrato at 110/220/440/880 Hz; 70-cent vibrato at 220 Hz with 3/7-Hz rates; rising/falling 200-cent glides over four seconds. Six shifts per profile give **72 distinct analytic shift cases**, with three renderers giving **216 renderer measurements**. Analytic references preserve the same contour timestamps, harmonics and amplitude envelope. Audit frames lie in the interior 1-3 seconds. These controls are diagnostic signals, not independent singers or research examples.

## Real-audio results

There are 108 real-crop measurements: six crops x six shifts x three conditions. Counts describe this selected calibration sample; they are not correction accuracy or a population estimate.

| Dataset / renderer | Passing shifts /18 | Crops passing all six | Active passes lost | New passes added |
|---|---:|---:|---:|---:|
| Vocadito / active FFT 2048, hop 512 | 17 | 2 | 0 | 0 |
| Vocadito / candidate FFT 1024, hop 128 | 17 | 2 | 1 | 1 |
| Vocadito / candidate FFT 1024, hop 256 | 17 | 2 | 1 | 1 |
| Jamendo / active FFT 2048, hop 512 | 5 | 0 | 0 | 0 |
| Jamendo / candidate FFT 1024, hop 128 | 10 | 0 | 2 | 7 |
| Jamendo / candidate FFT 1024, hop 256 | 9 | 0 | 1 | 5 |

Both candidates lose Vocadito `37_p01` at -20 cents: common plateau coverage falls to about 41.1%/19.6%, below the fixed 75% requirement, even though fine YIN p90 error on surviving frames is about 3.46/3.67 cents. Both recover `39_p00` at -30 cents, which the active renderer rejects for 12.01-cent fine p90 error. Equal totals therefore conceal a different failed case. Jamendo decisions reproduce the preceding calibration, including its regressions and zero all-six crops.

All 108 variants preserve exact outside-support samples and sample count, and each source/configuration zero-shift identity check passes. This verifies those mechanics on the tested material; it does not establish audible transition quality or acceptable musical repair. Passing injected-shift audits does not grant human reference approval.

## Constructed expressive controls

These numbers apply the existing **fine error/frame requirements only**. They are not full PESTO/reference/preservation eligibility audits.

| Renderer | Fine passes /72 | Maximum fine p90 error, cents |
|---|---:|---:|
| Active FFT 2048 / hop 512 | 47 | 31.32 |
| Candidate FFT 1024 / hop 128 | 70 | 10.87 |
| Candidate FFT 1024 / hop 256 | 63 | 15.06 |

All 72 analytic oracle cases meet the fine limits; their maximum p90 error is 3.73 cents. All three renderers pass all constant-pitch and glide cases. Vibrato accounts for the remaining failures. The hop-128 candidate fails the wide 7-Hz vibrato at -30/-20 cents, with p90 errors about 10.87/10.85 cents. Thus it is substantially better on this constructed set, but the wider test reveals an expressive limit missed by the previous two-profile comparison. No limits were relaxed to admit those cases.

## Listening and actual reviews

[Open the matched listening page](../runs/renderer_calibration/7b8a3cfaab7349f5b574db6d7d552683/listen.html). It links unchanged originals and 36 saved float-WAV comparisons: both extreme shifts, all six crops and all three renderers. Each saved WAV was reloaded and matched the generated samples exactly. The page labels configuration and automated audit status; it is a **diagnostic, unblinded comparison**, not a completed listening experiment. No audible-quality verdict has been recorded.

Listen against the unchanged original at consistent playback levels. Check vibrato shape, timbre/metallic artifacts, edit edges, timing/rests and naturalness. Include failed variants and losses of active successes; do not choose a favorable renderer separately for each crop or shift. Musical suitability of the original and quality of the generated variant are separate decisions.

The [existing reference-review bundle](../runs/review_bundles/5d35fe91ab1846ee8ab56e16289879ed/instructions.md) remains unchanged: nine original candidates, two independent sheets and eighteen pending rows. Those forms assess the original references, not these rendered variants. Reviewers must return real working-copy decisions, identities, timestamps and stable intervals. Vocadito annotations and two agreeing trackers are not substitutes for those decisions; reviewed Jamendo still retains model-estimate provenance. The newly probed S29 crops remain unreviewed in the original Phase 2 queue; the nine-crop bundle does not approve them automatically.

## Decision and remaining gates

1. Keep the current renderer and all v0.3 support/error/confidence/preservation settings. Candidate improvement is not dominance; no automatic winner or union of per-case successes is adopted.
2. Complete the independent original-reference reviews and record audible observations for these matched comparisons. Resolve disagreement with documented reasons and intersect approved stable intervals before pair generation.
3. Address the limited support across independently identified singers. More variants of S29 do not solve this. Any further geometry or renderer study must declare a separate version and selection before outcomes; preserve held-out material for a later frozen protocol.
4. Only after reviewed supports, successful intervention audits and a justified frozen policy are available can research pair generation and matched learning advance. Controller/rollback fixtures can continue in engineering mode while these gates remain open. Corpus expansion and future Windows/macOS pipeline integration stay deferred.

## Verification and reproduction

All **81 regression checks passed**: [verified log](../runs/renderer_calibration_regression_verified_2026-10-14.txt). New checks verify that eligible geometry cannot select held-out audio, aggregate gains still report lost baseline cases, and missing/insufficient controlled measurements cannot pass. The restricted initial test run failed on Windows temporary-folder permissions; the verified rerun had that access. This is execution evidence, not research validity.

The run stores input/review hashes, original accepted hashes, analyzer metadata, source-code snapshots, software versions, per-case failures and saved listening-artifact hashes. Inputs and code were rechecked at completion; existing manifests/reviews and the active renderer were preserved. See [final integrity verification](../runs/renderer_calibration/7b8a3cfaab7349f5b574db6d7d552683/final_verification.json).

From this component folder, a repeat creates a new evidence directory:

```powershell
$env:OMP_NUM_THREADS='8'
$env:MKL_NUM_THREADS='8'
& '.\.venv\Scripts\python.exe' -B -m bpdr_input.renderer_calibration '.\runs\coverage\1993b73598b94feeb4de15d3588f65e9\summary.json' '.\runs\jamendo_yield\de2f85a01b194c4eab009aa95b14e496\evidence.json'
```
