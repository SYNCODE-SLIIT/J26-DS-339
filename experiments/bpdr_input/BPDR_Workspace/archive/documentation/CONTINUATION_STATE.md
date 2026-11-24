# BPDR continuation state

Updated 8 November 2026. Current status only; historical snapshots and old
job handles are preserved in [the continuation archive](Continuation_History_Through_2026-11-08.md).
Read the user objective before work:
`C:/Users/lnipu/.codex/attachments/1c65460a-93b4-43cc-966e-247b11d04002/goal-objective.md`.

## Goal, method and boundaries

Complete the vocal-input component for January 2027, evaluated demonstration
by 31 December 2026. Free original 5-15-second melodic humming with short
notes is essential. Optional guide mode does not replace it. No short-hum
pitch-repair candidate is qualified. No new candidate has been activated.
The historical ExtraTrees preview is experimental, not a general winner.

Keep engineering completion, useful delivered audio and support for the
BPDR hypothesis separate. The hypothesis concerns boundary-invariance and
counter-detuning objectives with preservation anchors against matched
augmentation-only/frame-only controls. Cleanup is a separate workstream.
BPDR owns audio preparation, local relative repair, review, protection,
rollback and accepted-WAV provenance. Other components own final notes,
durations, tempo, key and harmony. Do not change the shared extractor.

Authority: [January plan](Continuation_Method_and_January_Plan_2026-11-07.md),
[frozen experiment protocol](Next_Experiment_Protocol_2026-11-07.md),
[master plan](BPDR_Master_Plan.md) and
[implementation tracker](BPDR_Implementation_Tracker.md).

## Live jobs: poll before any restart

- Session **36082**: `tools/resume_balanced_short_objectives.py --mode fit`,
  then sequential `--mode verify`. Original session 35400 stopped (exit 1)
  at a two-ULP duplicate-view assertion after five intact fits (four
  augmentation-only, one frame-only). The numerical diagnosis exactly
  reproduces identical inputs and repeated outputs, with a 2.33e-10-cent
  within-batch difference. The separately declared wrapper permits up to
  four float32 ULPs only for that construction assertion, retains all
  predictions and requires exact input/archived prediction equality.
  Scientific gates, fitting and budgets are unchanged; no fit is repeated.
  All twelve checkpoint/receipt hashes are now verified (four per arm).
  First-pass aggregation now exists: BPDR positive singer-macro raw MAE
  21.387088 cents, augmentation-only 23.246071 cents and frame-only
  24.905712 cents. BPDR fails the raw gate
  in 11 of 17 scoreable groups; `advance_to_delivery` is false. This is
  provisional until the independent twelve-checkpoint replay finishes.
  Jitter boundary sensitivity also worsens versus augmentation-only
  (0.257566 versus 0.108847 cents), failing another advance prerequisite.
  Session 36082 was polled live again on 8 November; its replay receipt
  is still absent. Do not restart it or report these as replayed results.
  Twelve fits total, four source-disjoint folds, seed 339,
  1200 updates each; split/merge/jitter receive 400 updates each. All arms
  share the corrected balanced anchor, original smoothness, initialization,
  query traces and capacity. Full BPDR alone adds its unchanged response
  terms after 200 warm-up updates. Smoke check passed. Output:
  `results/transformer_usability/2026-11-07_balanced_short_objectives/`.
  Do not interpret intermediate training losses as accuracy.

UI checks are complete: twenty-seven tests in session 51819, and legacy
recovery UI in 14445, both exit 0. Earlier sandboxed UI attempts were intentionally
interrupted after a traceback showed asyncio loopback blocking before app
startup. Do not poll those stopped handles or infer application failure.

The added time-specific feedback has eight focused passing tests and a
four-state UI/fresh-process replay (session 35926, exit 0). Current export
hashes equal the already consumed original/prepared WAVs. Full regression
including feedback completes in session 5430 (35 tests, exit 0).

The request/permission/selected-WAV display now independently replays six
state/role pairs (25090 exit 0); original selections show zero changed
samples. Current full regression passes 43 tests (74089 exit 0). No new
model or consumer inference is claimed by that display check.

The required held identity-resynthesis extension is implemented and
predeclared in `tools/check_balanced_identity_controls.py --mode declare`
(26747 exit 0), using all eight existing control cases. Run create then
verify only after the parent twelve-checkpoint replay receipt matches its
result. It evaluates correct held folds across all arms and perturbations,
retains unsupported-boundary exclusions and produces no fit or promotion.
Declaration: `2026-11-07_balanced_short_objectives/identity_controls/`.

The new read-only `tools/report_balanced_short_objectives.py` is syntax
checked but not yet executed. Run create then fresh-process verify after
the parent receipt exists. It reconstructs all nine saved arm/kind scores,
checks gates independently from per-record diagnostics, retains exclusions
and original-only groups, and reports domain/singer/duration/direction/edit
width and original activation tails. No fit, rendering or promotion.

All other handles in the archived continuation text are historical.
A poll timeout does not establish termination. Inspect receipts and the
actual process before restarting missing work.

## Verified findings that determine next actions

- Six historical objective fits remain stopped in
  `2026-11-06_short_objective_comparison`; do not mix them with new fits.
  Four augmentation-only fits were nearly zero; two frame-only fits exist.
- Fixed-batch diagnostics measured nonzero targets at one sixth of anchor
  mass. Original weighting, full BPDR response terms and bypassing the
  activity multiplier each failed the training-fit prerequisite. Gradient
  audit found zero-target norm about 173 times nonzero-target norm at the
  saved model. No exact-token contradictory labels in that batch.
- Equal-class anchor probe reached 3.062-cent positive MAE and 0.8406 signed
  response after 200 updates, zero-target mean request 1.184 cents. Exact
  replay matched. This supports the corrected comparison, not usability.
  Split/merge/jitter were audited across all 498 records with acoustic
  channels fixed. Frame-only equality is a construction property.
- The actual shared consumer is frozen in
  `2026-11-07_primary_extractor_freeze/manifest.json`: five source hashes,
  peak normalization to 0.95, no trimming and exact-end-frame exclusion.
  Historical cleanup helpers omitted the latter preprocessing details.
- Three-case actual-consumer check independently reproduced six calls.
  Track 1 clean/rain10/rain0: neural PCM24 improves RPA50 while onset/pitch
  note F1 falls for both annotators. One source is insufficient for a
  general conclusion. The full development first pass also fails to
  demonstrate always-on B2 benefit: 0 dB A1/A2 onset/pitch gains
  +0.00333/+0.01293, both singer intervals crossing zero; harms 35/84,
  36/84; clean harms 4/21, 3/21. All twelve saved-metric summaries
  independently reproduce (maximum error 2.43e-17); all 378 inference
  calls independently replayed exactly in session 88786, exit 0,
  with aggregate error 6.94e-18. Receipt and all raw output hashes
  rechecked. RPA50 0 dB improves .59796 -> .80829,
  demonstrating why frame accuracy alone is insufficient.
- All 189 neural exports have verified exact PCM24 samples, unchanged
  counts/rates and headroom. Full 567-estimate export rescore completed;
  aggregate replay matched. One PESTO score changed by -0.002219756 versus
  FLOAT. The old gate still fails (18/84 PESTO harms at 0 dB, one accepted
  clean harm). Never relabel that gate as passed.
- Clean-control alignment: 103/105 full/local windows peak at zero lag in
  both encodings; two local exceptions remain. Internal enhancer delay
  configuration is unverified. Do not shift audio from these observations.
- V71 delivered 22/445 benefits, 89/445 harms, 8/53 changed originals;
  replay complete. Spectral-context comparison failed its raw gates.
  Longer-hum V68 transfer improved raw predictions but delivered no useful
  repair. Two endpoint-cleared diagnostics passed encoded checks while
  causing known-zero harm. Identity-resynthesis controls show one large
  model-response change in eight cases. These controls still need assessment
  with new held checkpoints before research/repair qualification.

See [result details](HumTrans_Transfer_Result_2026-11-02.md) and
[cleanup results](Usable_Input_Plan_Status_2026-11-05.md). Preserve all
1152 physical attempts, 707 exclusions, 445 scoreable positives, 53 originals
and unassessed groups; derivatives are not independent participants.

## Implementation and external work still required

No qualified cleanup configuration or short-hum repair yet. Compare the
format-only, declared conservative spectral and frozen neural options,
then a fixed runtime-observable policy. No architecture/threshold sweep.
Keep reserved singers, sources and validation noises closed until frozen
confirmation. Never infer intended pitch from a semitone grid or guide MIDI.

Complete the recording/upload, diagnostics, original/prepared playback,
protection/rejection/undo, combined-edit verification and selected-WAV export
workflow. Prove accepted-only actual-pYIN integration by waveform hashes;
the shared worker currently forwards unchanged audio beside a preview.
The new opt-in `repair/primary_handoff.py` proves exact selected-WAV
delivery to frozen pYIN on synthetic original/prepared/protected-original
exports and six real calls including independent repeats. Current receipt
and source snapshots: `results/demo_sessions/2026-11-08_primary_selected_handoff_v3/`.
Use `./Run-Review-Demo.ps1` for `application/review_demo_app.py`. Historical
`demo_app.py` and `demo_review.py` retain exact source-lock bytes; current
protection metadata uses the new `repair/review_export.py` wrapper.
The UI now supports decoded-sample protection, whole-preview veto,
local-edit subset selection/rejection and undo, with fresh final decoded
combination auditing and export/result/confirmation invalidation.
Twenty-seven tests pass. The manual two-edit synthetic control independently
replays all three selections and parent under unchanged real acoustic
limits (eight audits); the combined export reaches pYIN with exact hashes
in two child calls. Session 90864 is complete, exit 0; receipt/source
snapshots are under `results/demo_sessions/2026-11-08_local_combination_audio_v2/`.
The [real-hum UI workflow check](Real_Hum_Workflow_Check_2026-11-08.md)
also completes and independently replays in session 21082 (exit 0): upload,
simulated recording bytes, controlled-rain preparation/protection/undo and
silence refusal. Eight route executions and six actual pYIN calls cover one
already opened eight-second training hum and its derivative. The spectral
example delivers +2.45525 dB paired SNR with its existing preservation audit;
all selected/consumer hashes match. Confirmations are explicitly automated
fixtures. This is not live microphone, human listening, fresh confirmation
or general cleanup qualification. Evidence/source snapshots:
`results/demo_sessions/2026-11-08_real_hum_workflow/`.
Qualified real-hum review and complete shared PP1 orchestration remain open.
The [recording feedback check](Recording_Feedback_Check_2026-11-08.md)
adds timestamped quiet/peak/uncertain-measurement observations with neutral
pause/expression guidance. It does not infer intended-note mistakes or
change acceptance. Feedback is bound to source/working/features/code hashes
and included in export metadata. Its four-state UI replay and export hash
checks complete under `results/demo_sessions/2026-11-08_recording_feedback_ui/`.
The [request/delivery summary check](Request_Delivery_Check_2026-11-08.md)
separates model requests, effective preview permission and measured selected
WAV changes. Summary metadata is bound to the selected role/hash; stale
selection metadata cannot be exported. Six state/role pairs replay exactly
with unchanged export checksums under `2026-11-08_request_delivery_ui`.
This is engineering
evidence only, not completed shared PP1 orchestration or audio benefit.
See [the handoff result record](Accepted_Pyin_Handoff_Engineering_2026-11-08.md).
Rejected statuses must have no accepted-audio path. Failed candidates stay
inactive; unchanged output is not a qualified pitch repair.

The owner can arrange two musical reviewers and will provide the rubric
later. [Verified four-page brief](../output/pdf/BPDR_Independent_Reviewer_Brief_2026-11-07.pdf)
has reviewer instructions/form (pages 1-3) and coordinator/academic tasks
(page 4). Audio packet, consent/recruitment, performer intention feedback
and actual independent ratings remain pending. Do not fabricate approvals.
Use the [collection plan](Review_and_Collection_Plan_2026-11-01.md).

December targets: method decision by 11 December; workflow/integration by
20 December; evaluation, four honest demonstrations and rehearsal by
31 December. No completion percentage is established. Essential unqualified
pitch repair cannot be counted complete or silently removed from scope.

## Environment

Canonical root: this BPDR_Workspace. PowerShell, Python
`../.venv/Scripts/python.exe -X utf8 -B`, `PYTHONPATH=source;tools`.
Use the local results directory for TEMP/TMP where required.
mir_eval 0.8.2 is isolated in
`results/transformer_usability/evaluation_dependencies/mir_eval_0_8_2`.
No commits, publishing, reserved-data access or model promotion authorized
by the existence of these engineering checks.
