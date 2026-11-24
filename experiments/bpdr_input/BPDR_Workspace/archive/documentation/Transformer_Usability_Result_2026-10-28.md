# Transformer usability result — 28 October 2026

**The requested usable Transformer has not yet been achieved. MOD-03 remains
active. The new candidate is not enabled in the live demo.**

This iteration completed broader source preparation, multiple declared model
conditions, actual-WAV development and independent-singer evaluation, a guarded
project adapter and a repeatable accuracy command. Working inference and a
successful selected example do not close the remaining coverage/generalization
failure.

## Measured outcome

The final candidate is a genuine two-layer Transformer operating on three
engineered, input-only acoustic-step measurements. It learns signed correction
and activity from audited real-WAV interventions. No reference audio, known
intervention interval or source identity enters inference. The classifier does
not infer intended notes or a key. This engineering variant is distinct from
the original BPDR architecture; boundary invariance is by input construction.

| Check | Previously inspected development | Four reserved evaluation singers |
|---|---:|---:|
| Attempted interventions | 48 | 48 |
| Scoreable interventions | 24 | 40 |
| Scoreable / attempted singers | 2 / 4 | 4 / 4 |
| Verified delivered improvements | 9 | 5 |
| Improved / all attempts | 18.75% | 10.42% |
| Improved / scoreable cases | 37.50% | 12.50% |
| Raw correction MAE | 3.014 cents | 13.740 cents |
| Raw scored frames within 10 cents, singer averaged | 100.00% | 51.74% |
| Actual delivered error, before -> after | 24.846 -> 16.970 cents | 24.797 -> 22.219 cents |
| Original WAV controls unchanged | 12 / 12 | 12 / 12 |
| Scoped practical gate | Pass for represented scope | **Fail** |

The final singer-averaged error falls about 10.4%, below the declared 25%
per-singer requirement. No final singer passes all repair criteria. The model
preserves originals much more reliably than the earlier candidate but cannot
yet repair enough eligible cases. A tolerance percentage is not general singing
accuracy, and the favorable two-singer development result did not generalize.

The broader training-side cross-validation is consistent with that limitation:
MAE13.341 cents, original request0.160 cents and response gain0.587 over eight
scoreable held-out singers. All eight pass the limited prediction gate, which
requires response>=.25 and improvement over zero correction, not a full
delivered-repair pass. There are108 scoreable interventions /200 attempts from
27 scoreable recordings /50 attempted originals. Two attempted training singers
have no valid repair labels. Only45 originals have reliable frames available
for unchanged-audio anchor sampling; unavailable anchors remain documented.

## What was implemented and checked

- Downloaded42 specific additional VocalSet sources plus eight prospectively
  declared evaluation i-vowel sources, with bounded ZIP reads, CRC checks,
  retained originals and attribution. No dataset original was edited.
- Completed 47 saved neural fits in this work session across the declared
  conditions. The final guarded run reuses the V6 weights byte-for-byte and
  does not add three extra fits to that count. Earlier27 October fits are
  separate historical executions.
- Added learned activity/correction heads, wider-context conditions, original
  anchors even when synthetic interventions were unscoreable, and controlled
  input-localization tests. Failed conditions and technical corrections remain
  in their own folders; no failed result was promoted as success.
- Fixed acoustic eligibility for continuously voiced multi-pitch phrases by
  separately assessing large measured pitch regions. Added multiple candidates
  and a stricter candidate-local step-fit guard of.75. The guard is a measured
  shape-fit criterion, not calibrated confidence. It was chosen on historical
  development and fixed before the reserved evaluation.
- Retained30-cent maximum correction, .86s minimum support, 25% changed-duration
  budget, voicing/protection checks, dual-tracker audits and rollback. No
  interval was padded to pass duration. User protection still vetoes the whole
  overlapping candidate.
- Added the guarded BPDR adapter and candidate-loader checks. A known,
  already-inspected development example passed the actual project processing
  path and encoded-WAV audit. This was an internal diagnostic, not a public
  model activation, user approval or an accepted prepared handoff.
- **57 targeted tests passed.** This is not a newly executed full project or
  notebook suite. The activation command was also checked against this actual
  failed candidate and refused it; no active Transformer card was created.

The original ExtraTrees model, its selection/export files and the user's
pre-existing notebook edits were preserved. `pp1_demo` and sibling components
were not modified. Selective-review UI remains unimplemented.

## Evaluation counting and integrity

The written protocol says **per represented independent singer**. The initial
gate implementation incorrectly required repair-performance results from
wholly construction-unscoreable singers. A separate, recorded correction now
marks them unassessed, keeps all attempts in cohort yield, and requires every
original to remain unchanged. At least two scoreable independent singers and
both available dataset voice categories are required for the scoped gate.
Numerical error, coverage and preservation thresholds did not change.

That correction allowed the development result to be assessed on its two
scoreable singers; it does **not** change the final conclusion. All four final
singers have valid reference cases and all four fail repair requirements. The
failed unmodified gate receipts and the counting amendment are retained.

The final cohort was accessed only after weights, the.75 guard, source
selection and scoring rule were fixed. It must now be treated as an inspected
evaluation/development benchmark for future revisions. Repeating its command
does not create a new untouched test. No threshold or model was selected from
its outcome in this iteration.

## Why it still fails

Among40 scoreable final cases, five technically verified outputs improve,
three attempted outputs roll back, and32 remain identity outputs. Candidate
traces record13 protected-region rejections, six weak-shape rejections and one
insufficient-support rejection; these are candidate counts and may overlap at
case level. Some inputs produce no candidate at all. One final singer has no
verified improvement.

Thus simply reducing the minimum duration would not resolve the main failure.
The narrow step-evidence representation and acoustic eligibility miss too many
voice/contour variations. Missing input evidence cannot be repaired by a better
displayed training-loss number. The original-preservation and audit guards
must remain in place.

## Check the actual latest candidate

From `BPDR_Workspace`:

```powershell
& ..\.venv\Scripts\python.exe -X utf8 -B tools\check_transformer_accuracy.py --latest --render
```

This now resolves the latest completed **candidate**, even when it is not
activated. It prints `FAILED_SCOPED_USABILITY_GATES`, verifies the model/policy
identities, refuses fitting overlap and writes a separate replay report. It
does not activate a failed model or silently check the older model instead.

The independent invocation reproduced the final summary and outcomes:
[repeat report](../results/transformer_usability/accuracy_checks/a68958268a754036a7972c4b054ca1f7/index.html).

## Evidence and next model work

- [Readable comparison and actual-audio report](../results/transformer_usability/2026-10-28_v7_guard/index.html)
- [Unchanged failed independent-evaluation receipt](../results/transformer_usability/2026-10-28_v7_guard/final_evaluation/evidence.json)
- [Per-singer final gates](../results/transformer_usability/2026-10-28_v7_guard/final_evaluation/gates.json)
- [Original controls](../results/transformer_usability/2026-10-28_v7_guard/final_evaluation/original_controls.json)
- [Guard calibration and provenance](../results/transformer_usability/2026-10-28_v7_guard/execution_provenance.json)
- [Actual activation refusal](../results/transformer_usability/2026-10-28_v7_guard/technical_execution/activation_refusal.json)
- [Known-example project integration](../results/transformer_usability/2026-10-28_v7_guard/technical_execution/integration_check.json)
- [Verification receipt](../results/transformer_usability/2026-10-28_v7_guard/technical_execution/verification.json)

Continue MOD-03, not the later feature tasks. First audit and reserve a new
independent confirmation cohort before further tuning. Compare the narrow
three-feature condition with a joint pitch-context + step-evidence Transformer;
the joint input should retain the broader measured context when the step fitter
abstains. Add declared positive local-step and negative gradual-drift/vibrato
training controls, clearly distinguishing feature-only augmentation from
real-WAV evidence. Keep singer/source descendants together, unchanged-original
anchors broad, and all audits fixed. Evaluate candidate selection against the
actual protected/reliable supports without padding or removing protections.

That is a next experiment, not an implemented or successful result. The original
matched BPDR/augmentation-only/frame-only hypothesis, independent listening,
novice references and downstream extraction benefit remain unestablished.
