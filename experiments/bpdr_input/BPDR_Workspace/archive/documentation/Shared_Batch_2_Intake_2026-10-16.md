# Second shared-output batch: saved data and readiness — 16 October 2026

**Current Gates B/C remain open. Data intake is complete; next is a Gate C known-target pair audit.** The supplied `outputs 2.zip` is saved as a separate, content-hashed extracted dataset under `data/shared_exports/e6acc9d5b1dcd16ed122`. No first-batch file is overwritten and no original ZIP or shared-pipeline file is changed. The common pipeline is read-only and was not executed locally.

| Item | Measured result |
| --- | ---: |
| Track folders | 1,156 |
| BPDR records / artist proxies | 762 / 114 |
| Valid complete exported packages | 663 |
| Valid upstream withheld records, without crop artifacts | 99 |
| Quarantined records | 0 |
| Exported training crops / artist proxies | 312 / 51 |
| Training screen candidates | 163 |
| Training screen candidates with current support geometry / artist proxies | 71 / 23 |
| New audited training pairs / research-ready examples | 0 / 0 |

Archive SHA-256: `e6acc9d5b1dcd16ed122027a9f572337bfc184a3f822d2031558676dd715999a`. The clean copy preserves 25,827 substantive files (1,428,225,902 bytes) and omits 34,365 macOS metadata entries. The original archive remains at `C:/Users/lnipu/Downloads/outputs 2.zip`. Per-file hashes and original source location are recorded in `import_receipt.json`. Raw records, predicted notes, crop offsets, unshifted `model_midi_pitch`, masks, provenance, failures and reviews remain intact. No instructions embedded in archive files are executed.

## What is usable now

The component reader checks every exported package's artifact hashes, audio format/timing, 71-feature grid, pitch/voicing semantics, crop-local GAME boundary column and saved masks. Expected `status=withheld` records have reasons and no crop artifacts; they are retained as upstream selection outcomes rather than treated as broken audio packages. No schema, shared exporter or threshold is changed to accept them.

Frozen BPDR artist partitions come from the original preparation manifest. Exported counts by split: `{'train': 312, 'final_holdout': 25, 'pp1_evaluation': 136, 'development': 190}`. Screen candidates by split: `{'final_holdout': 18, 'pp1_evaluation': 95, 'development': 74, 'train': 163}`. The producer's `split=unassigned` field is preserved; the BPDR catalog assigns existing component partitions separately. Artist groups are proxies, not verified singers.

All 312 exported training crops receive lightweight YIN replay, with 312 exact saved-mask matches. All other partitions receive integrity checks only: no new pitch inference, model forward, support-geometry selection, fitting or tuning. Training candidates must pass the existing exporter screen and license/source policy plus existing v0.3 contiguous-support geometry. The candidate queue is **prospective**, not successful intervention evidence. No audio is edited and no correction targets, A/B/C/D views or model fit are generated in this step.

The supplied predictions are not human note or intended-pitch labels. Known injected relative corrections must be constructed and independently audited before supervised examples exist. Vocadito remains the human-reference dataset; Jamendo adds supplementary predicted-source and robustness cases. Experienced musical review remains deferred by the owner, with no invented approvals.

## Batch reconciliation and provenance

All **15,521 files shared with the first import are identical**; this batch adds 10,306 files and removes none. The 663 distinct incoming crop hashes include 16 previous crops, so **647 crop hashes are new relative to the first batch**. Both versions remain preserved; repeated crops are not counted as independent material.

Of the 64 previously requested priority tracks, **46 have valid exported packages, 4 have valid withheld responses, and 14 are not supplied**. A withheld response is a received outcome rather than a missing file. The prospective 71-crop training queue now permits a bounded pair audit; the remaining export/provenance requests do not trigger automatic retries or shared-pipeline changes.

There is still no original root manifest or complete source MP3/full stem in this ZIP. Recorded upstream audio hashes cannot be independently byte-verified without those files. The exact producing revision remains a provenance item; `independent_verification.json` records code-hash comparisons without rewriting producer metadata. These limitations remain visible separately from artifact integrity and exploratory pair-audit eligibility.

## Next action

Gate C: prospectively audit known relative-detuning A/B/C/D pairs on all 71 frozen-training candidates with existing screen and v0.3 support geometry. Preserve source/artist partitions and masks, record every pass/failure, and retain model-estimate reference types. No training until valid audited targets exist. Gate B independent source suitability, Gate F real-audio repair and future Gate D/J4 review remain open.

Run evidence: `runs/shared_batch_intake/2fa025b548444bc3ad9a515168ec91e8/evidence.json`. `source_inputs.json` is the unpaired source catalog; `pair_audit_candidates.json` is the complete declared eligible training queue. No new learner training or musical accuracy is credited. The previous harmonic diagnostic remains unexecuted and is not automatically resumed.

For a further archive with the same known-track metadata mapping, use the component's batch reader, which handles expected upstream withheld records:

```powershell
& '.venv/Scripts/python.exe' -B -m bpdr_input.shared_batch_intake 'C:/Users/lnipu/Downloads/outputs 2.zip'
```

The already imported copy is reused and checked; this command does not run the common extraction pipeline. Additional unknown track IDs require explicit source/partition mapping before acceptance. The first-batch-only historical audit command should not be used for records without crop artifacts.

## Completed verification

All **228 component tests pass**, including six new expected-withholding/metadata guards. Independent verification rehashes all 25,827 imported files and checks 2,652 exported artifact hashes, 99 expected withheld records and 312 exact training-screen replays. All eligible support choices independently reproduce. There are 663 distinct incoming crop hashes and 0 duplicate incoming audio-hash groups. Original review decisions and all 28 shared-pipeline files remain unchanged. No held-out pitch inference, new pair or learner fit is performed.

Review rows supplied in this batch: `{'pending': 700}`. These are decisions as stored; pending rows are not approvals. Producer-code comparisons and unresolved producing-revision/source-audio provenance remain in `independent_verification.json`. The new data does not close musical validity or delivered-repair gates.

The separate `provenance_resolution.json` resolves all known common-source filenames plus the external GAME asset path used by the first intake: 5 of 5 recorded `infer.py` hashes match after LF normalization. Across the batch, **756 of 762 records match all their supplied code-hash entries**; 6 retain unresolved/different entries. Records have different code-hash coverage, so this does not establish an exact producing Git revision. Original producer records and the independent comparison remain unchanged. The exact revision and complete source-manifest request remain outstanding.

Both maintained notebooks freshly execute with zero cell errors, **29 plots and 41 audio widgets**, with source/executed/HTML/index hashes verified. The new second-batch coverage figure was visually inspected. Latest panel: `runs/notebooks/238f6dded22d4a6c939bf8b3653966d4/02_data_review_and_paired_experiments.html`. Full checks: `final_verification.json` in this intake run. All older runs, input files and failed/withheld cases remain preserved.
