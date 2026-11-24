# Requests, permission and the selected waveform

8 November 2026. The opt-in review UI now separates the maximum model
request, the preview permitted by existing checks, and samples actually
changed in the version selected for handoff. `application/request_delivery.py`
reads the saved request grid and compares decoded WAVs without editing them.

Choosing the original reports zero changed samples, even when a numerical
preview is available. Local selection uses only the currently permitted
intervals, including a reduced subset. A withheld or rolled-back preview
does not gain permission because its file still exists. Whole-clip cleanup
is identified separately; it is not presented as a pitch-correction request.
Unavailable pitch requests remain unavailable, rather than being invented
as zero. File checksums and source/selected sample grids are verified.

The metadata records the selected role and waveform checksum. Export refuses
a summary associated with a different selection. The ordinary confirmation
and export checks remain in place. Counts of changed samples measure waveform
difference; they do not establish pitch accuracy, musical quality or intended
note recovery. Selection is not a downstream handoff until an accepted package
is created and consumed.

## Verified examples

The existing failed-generalization legacy pilot has a maximum model and
permitted request of 26.448 cents. Choosing its original changes zero samples;
choosing its preview changes 23,371 samples. This is a readout of that saved
diagnostic example, not promotion of the pilot or a general repair result.

On the existing HumTrans controlled-rain example, original selection changes
zero samples; prepared selection changes 194,039 samples and is labelled
whole-clip cleanup. On a synthetic withheld state made from the legacy
example, the saved model request remains visible while permission and selected
changes are zero. The unprocessed training hum also remains unchanged.

## Evidence

Eight focused tests cover request/permission/selection separation, direct
decoded sample counts, rolled-back files, subset maxima, cleanup, unavailable
requests, checksum/grid failures and stale-summary export rejection.
All **43** affected feedback/review/export/UI tests pass (session 74089,
exit 0).

The declared six-state/role UI check and fresh-process replay also pass
(session 25090, exit 0). Six automated fixture exports per process retain
exact selection summaries and WAV checksums. The test reuses already opened
pilot/training states and performs no new model or consumer inference.
Confirmations are explicitly automated fixtures, never human judgments.

Tool: `tools/check_request_delivery_ui.py`. Declaration, source snapshots,
results and hash-bound replay receipt:
`results/demo_sessions/2026-11-08_request_delivery_ui/`.

This completes the tested display/provenance behavior for these states.
Qualified short-hum repair, fresh confirmation, actual user listening and
the complete shared PP1 orchestration remain separate unfinished requirements.
