# BPDR scope and interface decisions

**Current scope: 12 October 2026. Owner: Liyanage L. N. P., IT23141506.** Earlier milestone/contract examples are preserved in the recovery snapshot. Use the [master plan](PP1_Master_Plan.md), [execution gates](PP1_Phase_Execution_Plan.md) and [evidence tracker](PP1_Evidence_Tracker.md) together.

## Scope

- Input: 0.25-20-second WAV/FLAC, intended as a single unaccompanied singing/humming voice. Phase 1 currently validates technical integrity, not monophony or musical acceptability.
- Current output: preserved original, checked mono 22,050 Hz PCM24 WAV, acoustics and diagnostics; no musical correction.
- Planned output: justified minimal pitch repair with permission checks, duration/rest preservation, post-render verification and rollback. No key/scale snapping, rhythm replacement or automatic recovery of badly clipped/multisinger input.
- BPDR's proposed contribution: correction-response training under split/merge boundary views and known local detuning, tested against matched controls. Runtime tools and pipeline reuse are supporting engineering.
- Jamendo separation/cropping belongs to offline research preparation; it does not enlarge the production-input promise.

## Proposed hand-off

| Producer | Consumer | Data boundary |
|---|---|---|
| BPDR / IT23141506 | Melody understanding / IT23439078 | Accepted WAV and source/time mapping, status, metadata and preparation/repair diagnostics |
| Melody understanding | Planner / IT23148086 | Timed notes/rests, key/mode, rhythm/phrase representation and agreed context |
| Melody understanding + planner | Refinement / IT23157132 | Melody and timed relative-root/family plan with agreed context |

Only BPDR's local technical package is implemented. These arrows are proposed interfaces, not verified live integration. The four supplied reports contain conflicting older component numbering and chord-family examples; see [the audit](Project_Audit_2026-10-12.md).

`ProcessingResult` schema 0.1 carries request/source IDs, status, optional accepted-audio path/hash/rate/count/channels, reasons and policy version. Per-run diagnostics currently carry operation logs and edit intervals. A versioned diagnostics reference, crop/time mapping and model/controller/renderer provenance must be agreed as the integrated package evolves.

| Status | Audio allowed downstream? | Meaning |
|---|---|---|
| `ACCEPT_UNCHANGED` | Yes, validated accepted WAV | No musical change; technical preparation remains logged; suitability must be established for the eventual product |
| `ACCEPT_PREPARED` | Yes, after future repair checks | Verified permitted repair and preservation |
| `RERECORD_REQUIRED` | No | Reasoned unusable/out-of-scope input |
| `TECHNICAL_FAILURE` | No | Decoding, invalid-sample, analyzer or processing failure |

The consumer calls `require_accepted_audio` and validates the actual package/file before decoding. Never infer acceptance from file existence, a successful extractor exit or an empty result labelled complete.

## Working decisions

1. Use both datasets, with Vocadito reference/evaluation and Jamendo screened exploratory material. Participant recording remains a later proposal commitment.
2. Reuse pinned separator/GAME models, keep the team mirror unedited, and build source-specific adapters here. Read GAME `model_midi_pitch`; do not import instrumental/combined melody/chords/key into BPDR.

   **13 October:** the common adapters execute on both datasets. The [fresh five-artist sample](Jamendo_Five_Artist_Sample_2026-10-13.md) has individual separator receipts and yields six Jamendo screen crops from three artists, plus three Vocadito candidates: eighteen pending reviews, zero research-ready examples. Only five of eighteen tested shifts pass v0.3 across two artists; fine shift errors and voiced audit coverage need diagnosis alongside the earlier quantization issue. Human references and model estimates remain distinct; musical-suitability calibration and independent approval remain open.
3. Retain two independent musically experienced reviews and waveform/shift audits for research anchors. Automatic screening prioritizes review; it cannot replace it.
4. Investigate intervention geometry on training data under a new audit version; retain quality tolerances unless evidence justifies a revision. No broad training claim from the current single-singer pilot.
5. Freeze controller/renderer settings before held-out evaluation; preserve usable original audio when proposed edits fail.
6. Keep the Oct 25 working target distinct from the proposal's Oct 26-28 prototype figure and its later full-study schedule.

## External evidence still needed

- Named reviewers and their actual independent decisions; supervisor agreement if the proposed review method changes.
- Melody-owner agreement on extractor, accepted WAV format, timing origin and JSON/Python contract. Current H1 is technical compatibility evidence, not that agreement.
- Ethics/consent and feasible sample size for participant study; dataset attribution/track licenses for any released audio.
- Official prototype/final assessment dates and slots; no dates are confirmed by this local plan.
- Team agreement on coarse chord families versus enriched labels and harmonic-function provenance. BPDR owns none of those chord-model changes.

Independent development may continue against contract fixtures while these are open. A fixture or a supervisor-directed future task is not evidence that live agreement or a research result already exists.
