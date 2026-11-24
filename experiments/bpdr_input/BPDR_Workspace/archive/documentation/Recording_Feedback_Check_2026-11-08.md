# Time-specific input recording feedback

8 November 2026. Implemented in `application/recording_feedback.py` and
the opt-in `application/review_demo_app.py`.

The review UI now lists measured time ranges with practical guidance:

- Quiet ranges are described as possible intentional pauses. Guidance to
  rerecord applies only if the performer expected audible voice there.
- Near-full-scale peaks prompt listening for distortion and lowering input
  level if needed. Reaching the numerical threshold alone is not labelled
  proof of audible clipping.
- Disagreement between pitch measurements is uncertainty, not an incorrect
  intended note. Slides and expressive changes remain allowed.

Level observations use preserved input samples in approximately 20 ms bins.
Pitch observations use comparable voiced frames on the working-audio grid.
Sample-free endpoint frames are excluded. A 0.12-second display floor for
quiet/disagreement spans reuses the existing contiguous-support value,
alongside the existing 50-cent tracker comparison convention. These choices do not change
acceptance, permission, rendering, model selection or rerecord thresholds.

The cache is bound to source, working WAV, feature and feedback-code hashes.
Changed inputs cannot reuse earlier feedback. Metadata accompanies accepted
exports; WAV samples are unchanged by feedback. Corrupt or unreadable source
states show feedback as unavailable while preserving the existing technical
failure/refusal flow.

## Verification

Eight focused tests pass: silence timing, neutral pauses, peak timing,
uncertain pitch spans, short/unvoiced intervals, sample-free endpoints,
invalid grids and source-bound caching. Tests confirm no mutation of samples
or features and no inferred intended notes or acceptance changes.

The declared UI check in
`results/demo_sessions/2026-11-08_recording_feedback_ui/` renders two saved
real-hum workflow states and two synthetic refused states. A fresh process
reproduces all four results exactly (session 35926, exit 0). The real-hum
states produce no listed spans under these limited checks; this is not a
claim of musical correctness. Synthetic silence is localized to [0,1),
and malformed WAV feedback is handled without an application exception.

Two fixture exports in each process preserve their WAV checksums exactly
against the previously consumed original/prepared workflow WAVs and retain
feedback metadata. No new consumer inference or participant evaluation is
claimed in this UI check. Automated exports retain `user_confirmed=False`.
Source snapshots, the declaration and hash-bound replay receipt are saved
beside the result. The full review regression suite also passes after this
UI addition: 35 tests, session 5430 exit 0.

## Remaining limits

These are input observations, not a noise classifier, a musical-error
detector or a validated usability decision. Human feedback, real microphone
capture, unfamiliar-user confirmation and qualified short-hum repair remain
open. No research-validation or completion percentage follows from this feature.
