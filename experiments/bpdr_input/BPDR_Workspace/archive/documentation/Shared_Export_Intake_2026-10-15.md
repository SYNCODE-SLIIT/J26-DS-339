# Shared extraction outputs: BPDR intake and revised next step — 15 October 2026

**Current work: Gates B/C — dataset intake and training readiness. Gate F remains open.** The common pipeline was built/run by another member and is shared read-only. BPDR only imports/adapts its data within `experiments/bpdr_input`. The older harmonic-feasibility experiment remains planned/unexecuted; this dataset handoff takes priority. Reviewer deferral remains unchanged.

## What the supplied archive actually contains

| Item | Count |
| --- | ---: |
| Shared track folders / artist proxies | 1,156 / 183 |
| GAME vocal melody outputs | 1,156 |
| Chord output pairs | 1,155 |
| Combined/SheetSage2/beat/key/Roman outputs, each | 525 |
| BPDR WAV/features/boundaries/masks packages | 16 |
| BPDR artist proxies | 2 |
| Tracks without BPDR record | 1,140 |
| Substantive imported files | 15,521 |
| macOS metadata files omitted from local copy | 22,367 |
| Substantive bytes | 316,618,614 |

Archive `C:/Users/lnipu/Downloads/outputs.zip` remains unchanged. The local copy is `data/shared_exports/6432fb33030a539a0e67`; `import_receipt.json` records original archive hash and every copied file hash. Paths/duplicate destinations are checked before writing; original metadata and artifacts in the ZIP are preserved. No dataset rows or failed crops are deleted. There is no root manifest, original full MP3 or full stem in this archive; it is a partial BPDR extraction snapshot.

Run: `runs/shared_export_intake/1998c76291664a6aa51253424e630a54/evidence.json`. The reader is `src/bpdr_input/shared_exports.py`; the auditable importer/catalog is `shared_export_audit.py`. Full track and crop catalogs are in `evidence.json` and `source_inputs.json`. No common-pipeline code or settings are altered/executed. Only lightweight training-crop YIN replay is performed locally; no separator, GAME or PESTO rerun.

## Compatibility and quality are separate

All 16 packages pass 64 artifact hashes and mono-22050 crop format/time mapping, finite frame grid/71-feature matrix, masked NaN F0, crop-local unshifted note/column-70 boundary and saved-mask/screen-count checks. All 13 training reliable masks replay exactly. The reader exposes the existing 70-feature slice and 71-feature matrix, preserving GAME crop-edge features separately from the acoustic boundary heuristic. It reads fractional `model_midi_pitch`; it never uses the published +12 pitch, combined melody gap fills, instrumental notes, chords or key as intended-note/correction targets.

The common acoustic extractor has the same parsed syntax tree as the BPDR extractor. A verified historical 30-update engineering checkpoint successfully forwards one 990×71 **training** input to 990 finite outputs, with no training, normalizer refit, audio edit or holdout forward. The chosen source remains screen-withheld. This validates I/O, not model accuracy or repair permission. See `input_probe.json` and its declaration/artifact.

Producer-code provenance is not identical to this checkout: CRLF/LF normalization accounts for analysis/crop/worker hash differences. The external GAME `infer.py` matches wherever recorded after resolving its asset path; some records contain fewer code entries (see `provenance_addendum.json`). The exact `export.py`/`selection.py` producing revision remains unmatched. Preserve the supplied record hashes and request the exact producing revision/snapshot; do not edit shared code or rewrite metadata to make them match. Full MP3/stem hashes are recorded upstream but cannot be byte-verified from this ZIP. These provenance limits do not erase the successful artifact integrity checks.

## Actual training readiness

| Group | Crops | Frozen BPDR partition | Exporter screen |
| --- | ---: | --- | --- |
| artist_000020 | 13 | train | all withheld |
| artist_003241 | 3 | final_holdout | all review_candidate |

The artist map comes from the existing frozen preparation manifest, not a new split recomputed on these 16 crops. Artist identity is not verified singer identity. No final-holdout crop receives model inference, new screen computation, tuning or fitting; integrity checks and copied exporter metrics are kept distinct. All six imported review rows remain pending, and both original review files are unchanged.

All 13 training crops carry `INSUFFICIENT_RELIABLE_PITCH`: their saved independent reliable-pitch support inside predicted notes is below the existing screen requirement. One also has low voicing and four have uncertain crop edges. Many still have substantial predicted voicing, so silence alone does not explain the screen result. The exact training-mask replay confirms the saved screen inputs, not musical correctness. Keep these inputs for diagnosis; do not delete them or relax the screen merely to increase the training count.

These are **16 usable source-input packages, not 16 approved training pairs**. They contain no known correction targets or aligned A/B/C/D intervention views required by `examples.load_pairs`. Therefore zero new paired-training and research-ready examples are claimed; no training is launched. Deferring experienced review does not authorize changing screens, source splits or target semantics. Vocadito remains the reference/evaluation source; Jamendo remains supplementary predicted-source/robustness data. The original ten pilot crops were technically accepted unchanged; a later experimental source estimator refused them. That was never a finding that either dataset was unusable.

## Concrete data handoff, without pipeline changes

The [priority request](../runs/shared_export_intake/1998c76291664a6aa51253424e630a54/bpdr_export_priority.json) identifies **407 missing policy-eligible training exports across 64 artist groups**. It prioritizes one lowest-ID missing track per frozen training artist, using metadata/license policy rather than pitch outcomes. Ask the common-pipeline operator to supply those existing pipeline `bpdr/` packages as extraction completes, plus the original root manifest and producing revision. No pipeline modification is needed. Full-song notes/keys/chords alone cannot replace the missing vocal WAV/features.

1. Import further archives with the same component adapter; retain hashes, source offset, labels and original records.
2. Screen source support and diversity on the frozen training partition; keep holdout untouched for development.
3. Build declared known-target, audited paired examples for an engineering comparison; do not turn GAME predictions into human truth.
4. Run matched BPDR/augmentation-only/frame-only fits only on eligible pairs, with explicit exploratory/research status. Collect real review later for research claims.
5. Return to a bounded end-to-end Gate F repair/rollback decision after source readiness; do not automatically extend the old phase-diagnostic branch.

Repeat intake (no shared pipeline execution):

```powershell
& '.venv/Scripts/python.exe' -B -m bpdr_input.shared_export_audit 'C:/Users/lnipu/Downloads/outputs.zip'
```

The generic reader/catalog supports further batches; dated verification scripts and recorded counts describe this archive only.

## Verification and panel explanation

Nine focused tests cover unsafe/duplicate archive paths, metadata-only archives, corrupt artifacts, rehashed incorrect boundary semantics, invalid F0/voicing and serialized training claims. Full component regression, full imported-file hash checks, common-pipeline before/after hashes, input-probe checks and fresh execution/visual verification of both maintained notebooks are recorded in `final_verification.json` in this run. Existing data, review rows, prior failures and executed notebooks remain historical evidence.

**Next — Gates B/C (open):** Use the 64-artist priority list to obtain missing BPDR exports from the existing frozen training partition; import the next batch with the component adapter, then build known-target audited pairs before new model training. Do not modify the common pipeline or consume final holdout for training. Gate F remains open. **Gate D/J4 remains deferred.**

**Verified completion:** All **216 component regression tests pass**. Independent verification checks all 15,521 imported file hashes and confirms the original archive and 28 common-pipeline files are unchanged. Both maintained notebooks execute with zero cell errors and verified source/output/index hashes, producing 27 plots and 41 audio widgets. The new four-panel intake figure was visually inspected. Latest panel: `runs/notebooks/329677675d3f4a6d98c464fc7d742008/02_data_review_and_paired_experiments.html`. The successful final execution follows a preserved run stopped by the source-change guard during a provenance-note edit; the stable sources were rerun. These are engineering and evidence-integrity checks, not musical accuracy.
