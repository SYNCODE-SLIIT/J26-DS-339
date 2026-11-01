# Research gap and implementation status

**Updated: 28 October 2026.** Current decisions and task order are in the [master plan](BPDR_Master_Plan.md); status is in the [tracker](BPDR_Implementation_Tracker.md). The [planning review](Planning_Review_2026-10-27.md) records source authority and deviations.

## Main question

Can BPDR ignore false split/merge/timing changes in provisional boundaries while making useful local counter-detuning corrections, improving delivered repair and subsequent extraction without more harmful alteration at matched acceptance coverage?

The primary learned comparison remains full BPDR versus augmentation-only, with a matched frame-only Transformer testing the simpler boundary-free alternative. Original audio and conservative rule correction are practical references. ExtraTrees is supplementary engineering evidence.

| Requirement / proposal source | Implemented or measured | Remaining work |
|---|---|---|
| Main objective, p15 | Technical preparation, controlled learning/evaluation and export paths exist | Held-out delivered repair, preservation and downstream benefit together |
| Compact correction Transformer, pp21–22 | Historical 71-feature model; new 32/31-context-feature two-layer model with saved weights | Useful permitted output; explicit architecture/feature-version deviations |
| Four views and objective, pp17–22 | Historical preparation/losses/comparisons; new context-model experiment | Matched inputs/budgets/selection; proposal MAE response definition in the proposal-aligned study |
| Boundary consistency | Same-audio alternative boundary views and numerical comparisons | Benefit beyond augmentation-only, coupled with useful response and natural-analyser controls |
| Counter-detuning | New BPDR-condition raw MAE 5.839 cents on 10/16 calibration attempts | Delivered-WAV gain and preservation; raw tolerance percentage is not general singing accuracy |
| Preservation anchors | MOD-01 adds broader original/outside anchors; CV requests 5.312 -> 3.688 cents while correction error rises 9.857 -> 10.568 | All-singer preservation/response gate still fails; no general accuracy improvement established |
| Practical comparison | ExtraTrees raw MAE 10.031; augmentation-only 5.644; frame-only 5.652; BPDR 5.839 | No BPDR-loss advantage established by this seed; retain simpler alternatives and negative results |
| Permission and region selection | Scope, largest-span selector and fixed limits exist | Seven scored regions entirely excluded by scope; multiple candidates and explicit eligibility experiment |
| Rendering / rollback, p26 | MOD-02 runs new-model actual-WAV checks: 3 unique technical passes, 1 rollback, 0/16 useful gains; two original controls altered | Useful repair, original preservation and actual listening remain unmet |
| FR1–FR3: input/feedback, p28 | Upload/recording, source retention, technical statuses, provisional acoustic review | Independent user evidence and calibrated usability judgments |
| FR4–FR6: local edits/export, p28 | Whole-recording comparison and separate reviewed experimental package | Protected regions, per-candidate Accept/Keep and final composition/reverification: planned new feature |
| FR7: diagnostics, p28 | Repeatable lineage-checked accuracy command; complete legacy-policy raw/scope/selection/permission/render trace with exclusions | Extend tracing to the future versioned selective runtime; diagnostics do not pass repair gates |
| Data/partitions, pp16–17 | Vocadito, exploratory Jamendo and supplemental VocalSet/HumTrans preparation | Fresh reserved singers, prior-use/overlap checks, approved references; augmented variants are not new participants |
| References/participants, pp16–17,39–40 | Typed performed-pitch references and automated audits | Actual reviewers, ethics/consent and guide-melody/free-hum collection; no completed recruitment claimed |
| Extraction benefit, pp18–20 | Exact accepted WAV can produce GAME MIDI/CSV | Independent-reference before/after note precision/recall/F1 and repeat variation |
| Listening/uncertainty, pp19–20 | Saved clips and source-aware summaries | Actual blinded judgments, disagreement, matched coverage, singer-level uncertainty and separate seed variation |
| Deployment, p29 | Local runtime, failure handling and provenance | Median/p95 latency against the proposed 30-second target, memory and accessibility checks |

## How the selective feature supports the main goal

Protected intervals and individual review extend preservation, changed-interval inspection and verified export. They do not infer notes or transfer key/beat/chord responsibilities into BPDR. User acceptance can remove edits but cannot authorize a technically failed one. Compose the final WAV from the original plus selected specifications and verify it after encoding.

This is an engineering feature. Evidence for the proposed learning method still requires the matched scientific comparison. A model without a boundary column is invariant by construction; a zero-output model cannot demonstrate useful repair through low sensitivity alone.

The completed [five-objective model-first study](Transformer_Improvement_2026-10-27.md) adds a repeatable [after-training accuracy workflow](Transformer_Accuracy_Workflow.md). Its broader seven-singer CV and separate two-singer calibration must remain distinct from the earlier recovery results above. The fixed default calibration has 6.670-cent raw error / 79.87% within 10 cents but zero delivered improvements. No objective passes every CV singer. This supports further model/data work and a separate controller comparison, not adoption or a claim that the main research hypothesis has been validated.

## Evidence rules

- Keep the two sixteen-attempt experiments separate: four scoreable cases in the earlier delivered ExtraTrees pilot versus ten in the new Transformer calibration.
- A validly labelled, scope-excluded case remains a scored withheld/unchanged outcome; it does not become an intervention-construction exclusion.
- Score actual delivered WAVs against references in controlled evaluation. For uploads, verify requested movement and preservation without inventing intended-pitch truth.
- Retain all failures, source/singer counts, direction coverage and no-change controls. Frames, variants and seeds are not additional singers.
- Participant collection, independent review and downstream musical benefit remain separate from working interface execution. A user checkbox is not independent approval.
- Preserve dataset permissions/attribution and originals. Uploads are not automatically reused for training/research.

**Next: MOD-03**, continue declared model/data improvement before later features. Reserve another independent cohort before changing the model; retain the failed final result. The [28 October usability result](Transformer_Usability_Result_2026-10-28.md) records 47 completed neural fits, a guarded project adapter and 57 passing targeted tests. The locked four-singer evaluation improved **5/48 attempted interventions (5/40 scoreable)** and preserved **12/12 original controls**. All four singers failed the coverage/gain criteria. The candidate remains inactive; MOD-03 stays ACTIVE and later feature work stays on hold. These final recordings have now been inspected and cannot be reused as a new untouched test.
