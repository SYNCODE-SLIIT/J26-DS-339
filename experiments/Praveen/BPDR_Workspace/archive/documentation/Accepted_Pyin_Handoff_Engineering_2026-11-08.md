# Selected-WAV delivery to the frozen pYIN consumer

8 November 2026. Engineering result, not audio-quality qualification.

The BPDR demo can now send a separately reviewed accepted package to the
existing shared melody component's frozen `analyze_vocal(..., trim=False)`
implementation. The prior standalone GAME option remains available. This
does not change the shared PP1 orchestrator or its unchanged-audio adapter.

Start the current opt-in workflow with `./Run-Review-Demo.ps1` (local port
8503). It uses `application/review_demo_app.py`. Historical `demo_app.py`
and `repair/demo_review.py` are restored byte-for-byte to their source-lock
hashes; the new metadata extension lives in `repair/review_export.py`.
The historical panel/default launchers are unchanged.

## Implemented behavior

`source/bpdr_input/repair/primary_handoff.py` verifies:

- Package is inside the workspace and has ACCEPT_UNCHANGED or ACCEPT_PREPARED.
- Selected WAV is inside the reviewed package, with valid metadata, finite
  samples, permitted amplitude and the declared checksum.
- Confirmation belongs to the selected waveform and original/prepared role.
  Automated fixtures require `allow_test_fixture=True`; normal UI calls
  require an explicitly user-confirmed package.
- Frozen owning source hashes and dependency versions still match.
- A separate bounded child process consumes a byte-identical copied WAV.
- Selected, copied and consumer-reported hashes match after extraction;
  the selection manifest and review also remain unchanged.

Failure or timeout preserves the accepted file and does not report consumer
success. Logs and receipts stay with each attempt. The worker imports the
owning extractor; BPDR adds no transcription or key-estimation algorithm.

The demo defaults to this shared consumer after the user creates an accepted
package. Changing the selected version, unchecking confirmation or losing
all eligible versions clears the pending export and consumer result.
Changing the consumer clears its previous result. Existing exported files
remain historical artifacts.

## Verified scope

`tools/check_primary_selected_handoff.py` declares two synthetic five-second
signals: an original and a gain-prepared file with different hashes. It
exports both with `confirmation_kind='automated_test_fixture'` and makes
four real consumer calls, two per selection in separate child processes.
Every selected/copied/consumer hash matches; repeated output matches after
excluding inference runtime. Each call produces six note events. The count
is an execution observation, not a transcription-accuracy claim.

Receipt: `results/demo_sessions/2026-11-08_primary_selected_handoff/integrity_receipt.json`.
The declaration binds the test tool, handoff adapter, review exporter and
consumer freeze. No human confirmation, useful-audio improvement, research
qualification or shared PP1 end-to-end completion is claimed.

Twenty review/handoff/protection tests pass (session 61878, exit 0),
including refusal of rejected packages, changed audio, mismatched review,
external paths, changed copies, timeout, stale-selection invalidation,
protected sample veto, half-open ranges and UI undo. Tests simulate widget
interaction without creating user-review evidence. Windows AppTest needs
loopback access: sandboxed attempts blocked before app startup, confirmed
by `ui_stack.log` in the first handoff result folder. Those attempts were
explicitly interrupted; the subsequent run with socket access completed.
The legacy recovery UI check also passed in session 14445 (27.1 seconds).

## Protection and current exported provenance

`repair/review_protection.py` checks the actual decoded preview against the
original in user-selected time ranges. If any protected sample changes,
the whole preview is vetoed, its review eligibility is removed and the
original remains available. It creates no sample splice or new pitch target.
Undo restores the prior review state. Either action invalidates pending
exports, consumer results and the current confirmation checkbox.

The six-call current-source check under
`results/demo_sessions/2026-11-08_primary_selected_handoff_v3/` completed
in session 29115, exit 0. It covers original, prepared and protected-original
selections, each with an independent child-process repeat. All hashes and
repeated results match; protection metadata survives into the exported
package. Source snapshots are retained. The earlier suffix-free and v2
folders remain historical checks of their declared exporter versions.

The protection action is a whole-preview veto. Local-edit subset review
and final combination auditing are now implemented separately below.

## Local-edit selection and final combination verification

`repair/local_review.py` accepts only indices of an existing, permitted
local pitch preview. It verifies the permission/source/prediction bindings
and exact sample support. Users can retain one edit, several edits or none;
they cannot draw arbitrary replacement pitch or edit intervals. Whole-clip
denoising remains a whole-preview decision. Existing acoustic limits are
unchanged.

The selected combination is encoded to PCM24, decoded and re-audited with
the existing framewise PESTO/YIN auditor. Samples outside chosen support and
in protected ranges remain exact. The new waveform receives a fresh
provisional acoustic assessment. A failed audit leaves the previous usable
review unchanged and records the failed attempt. Undo restores the previous
selection; confirmation/export/consumer results are invalidated on review
changes. The UI distinguishes pending edit choices from edits actually
included in the current preview.

Twenty-seven backend/UI tests pass (session 51819, exit 0), including
subset/combined selection, rejection of all edits, reselection from the
original permitted preview, failed-audit retention, provenance and undo.
Unit tests use explicitly mocked synthetic audits for state behavior.

Real acoustic checks are separate:
`results/demo_sessions/2026-11-08_local_combination_audio_v2/` contains a
declared ten-second generated harmonic signal with two manual +/-20-cent
requests on [2,3) and [6,7). One parent and three selected combinations
pass the unchanged acoustic limits in both the creation process and an
independent verification process: eight successful decoded-WAV audits.
All repeated waveform hashes and acoustic audit results match exactly.
The combined reviewed fixture is exported with local-selection metadata
and reaches frozen pYIN in two separate child calls with matching hashes.
Session 90864 completed both phases, exit 0. No human confirmation is
claimed; there are zero independent participants and no trained-model
accuracy result in this check.

The earlier suffix-free fixture attempt stopped on JSON tuple/list
comparison in the test's metadata assertion after its audio audits passed.
Its artifacts remain. The v2 tool canonicalizes JSON representation; no
waveform, acoustic limit or selection policy changed to obtain the result.

## Remaining acceptance work

Prove the full recording/upload-to-selection workflow and actual shared PP1
handoff with a qualified useful real-hum output. Broaden these controlled
review checks to that qualified complete workflow. Preserve the current public
statuses. These engineering checks do not fill the outstanding cleanup,
short-hum repair, human listening or confirmation-data requirements.
