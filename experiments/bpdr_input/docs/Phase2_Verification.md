# Phase 2 implementation and evidence — 30 September 2026

**Engineering implemented:** source/annotation validation, fixed singer partitions, mapped crops, pending human review, identity/local-detuning renderer, split/merge/jitter metadata, audited A/B/C/D inputs and executable notebooks. **Research gate remains open:** actual reviewer decisions are not yet available, so training-ready examples remain zero.

**30 September recheck:** the existing manifest and review sheet were inspected again. All 114 decisions remain `pending`; the empty `reviewed` run does not signify human approval. The 27 September v0.1/v0.2 pilots remain saved as historical engineering evidence. A fresh v0.3 exploratory run and notebook execution are recorded below; neither is a newly collected human review or learned research result.

## Observed data results

- All 40 recordings and their F0/A1/A2 files were read; metadata identifies 29 singers.
- `vocadito_30`: A1's final note extends about 2.77 ms beyond the recording. `vocadito_31`: both note streams extend about 8.78 ms beyond the recording. These sources are flagged and excluded from crop generation under the one-source-sample bounds tolerance. Original files are not altered.
- `vocadito_21`: overlapping A1 notes are recorded as a warning; overlap frames are ineligible for reliable supervision.
- 38 sources yield 57 mapped, non-overlapping crops <=20 seconds. All 57 decoded crops match their exact original source-rate sample intervals; coverage has no omitted or duplicated samples for eligible sources.
- Singer partition seed 339 uses stable SHA-256 ordering: 14 train, 6 development, 6 PP1 evaluation, 3 final holdout. Full source counts before exclusions are 21/8/8/3 respectively. Descendants retain source/singer/split IDs.
- After the two source exclusions, the 57 eligible **crops** are 30 train, 14 development, 9 PP1 evaluation and 4 final holdout. The independent denominator remains singers/sources, not crops, shifts or boundary variants; the PP1 evaluation and final holdout are small pilot groups.
- 114 pending review rows (two per crop). No approvals are invented. A1/A2 dataset annotations are not treated as the required two new human reviews.

`runs/phase2/latest.json` points to the current run. Its `manifest.json`, `data_summary.json`, `reviews.csv` and `crops/` contain the source hashes, partition map, annotation findings, crop offsets, shifted reference timestamps and partial-note flags. Metadata, raw references, crops and generated reference checksums are verified before intervention generation; changed data invalidates the manifest.

## Pilot, audit history and interpretation

The current pilot is **one training phrase from one singer**, selected by a deterministic reference-reliability ranking. It applies -30/-20/-10/+10/+20/+30 cents and combines each with false split, merge and jitter tracks, producing 18 paired examples. Six rendered waveforms are audited; repeated boundary views are not independent recordings.

Every A/B pair references the same original WAV and every C/D pair the same detuned WAV. A/B and C/D acoustic features match exactly; A/C and B/D timing tracks match exactly. Detuned acoustics are recomputed from the waveform. Input arrays contain only times and 70-/71-feature inputs; private reference masks and correction targets are separate. All 18 pair integrity checks pass.

The nominal 71st feature in the current pairs is a sharp timing-only boundary track made from **A1 annotation onsets and offsets** (the earlier v0.2 pilot used onsets and the final offset only), while Phase 1 runtime analysis produces an audio-derived **provisional boundary score**. They are not interchangeable evidence of natural boundary-error robustness. This controlled laboratory input tests the proposed contrast; a separate frozen-model evaluation must use the runtime score and naturally estimated boundary errors before making an end-to-end robustness claim.

**Audit v0.1 failed all six render candidates:** it required both PESTO and YIN to meet the 5-cent median/10-cent 90th-percentile fine-shift limits. PESTO responses in this phrase often behaved in approximately 33.33-cent steps, consistent with its three bins per semitone and sharply peaked activations. Independent YIN measured the injected small changes substantially more closely. The failure is retained in `runs/phase2/audit_history/v0.1.json`.

**Audit v0.2 was frozen before the next pilot:** YIN retains the original fine limits (median error <=5 cents, 90th-percentile error <=10 cents, correct sign). PESTO is a coarser consistency check (median <=20 cents, 90th percentile <=40 cents, no opposing median sign). At least 10 jointly reliable audit frames are required; outside-support median drift must be <=5 cents. Both estimators and failure counts remain visible. This protocol revision uses training pilot evidence, not evaluation/holdout data, and must be disclosed in later comparisons. It does not establish general analyzer accuracy.

All six candidates pass v0.2; all 18 derived pairs are exploratory and **zero are training eligible**. An exact passthrough identity render preserves all samples and measured F0. Samples outside local support are preserved exactly; length, finite values and clipping are checked. Raised-cosine crossfades are not assumed to cause a linear cents curve. Guarded plateau frames get negative injected-cents targets only after the realized-shift audit passes; crossfade/guard frames are unsupervised; reliable outside frames have zero anchors. The common reliable/anchor masks do not change with boundary corruption.

**Audit v0.3 and current run (30 September):** the policy additionally requires at least 75% coverage of eligible plateau frames, at least 10 outside-support audited frames, at least 75% outside coverage, and outside-support 90th-percentile pitch drift <=20 cents. These checks prevent a small jointly voiced subset or a drift tail from passing on median error alone. The earlier six rendered waveforms were independently re-audited under v0.3 and all six passed; the stricter F0 coverage rule changed none of the 40 source annotation statuses. A new run, `runs/phase2/a57b0dea3d344ee4b8a5dcd40746cce6/exploratory/86be8e8d7eb34c1a966c2321edd320ad`, then regenerated the A1 onset-plus-offset boundary views with current code: six shifts passed, 18 paired examples passed integrity checks, one phrase/singer was counted, no exclusions were recorded, and **training-eligible pairs remained zero**. Plateau coverage ranged from 0.83 to 1.00; outside coverage was 1.00 and outside drift p90 was 0.00 cents for each shift. The manifest, review sheet and six relevant source modules are hashed in the run summary; each model-input archive is hashed in its pair record.

The current review-gated command was also executed: `runs/phase2/a57b0dea3d344ee4b8a5dcd40746cce6/reviewed/60d849bd3dac4c7eb217d6eeb21e2ade` contains **zero** phrases, pairs and training-eligible examples while the review sheet is pending. This is the intended gate behavior, not a failed attempt to count unreviewed material.

The loader now independently checks the training split, crop identity against the validated source manifest, two current independent approvals and their overlapping support, locked audit policy and pass result, waveform/target/review/input hashes, and four-view input relationships. A forged positive flag on a pending review sheet fails. Review-sheet edits intentionally invalidate earlier records; generate a new reviewed run after reviewer decisions are finalized.

On the current phrase, PESTO's original v0.1 audit measured some requested 10-cent changes as zero median cents while YIN resolved them. The v0.2/v0.3 audits verify **rendering under their stated protocols**; they do not prove that the 70-feature model input can resolve every 10-cent shift. Check input separability by shift size on approved training material and report the result with the later baselines.

The current small pilot demonstrates engineering execution only. It is not a musical-quality listening study, natural-error benchmark, learned repair or BPDR superiority result. Phase 4 must implement and audit the actual repair renderer/controller separately; this local constant-shift intervention renderer is not that finished component.

## Tests and notebooks

32 automated tests pass on the updated repository source. New checks also reject sparse/short F0 references, incomplete renderer-audit coverage and drift tails, and forged training eligibility with pending reviews or altered model inputs. The earlier 28-test result remains a historical baseline.

Both maintained notebooks ran from a fresh component-local kernel without errors on 30 September. The current execution is `runs/notebooks/011f5a7c9c914e3da18d44df92ff8e23`: notebook 01 has two plots/two playback controls; notebook 02 has four plots/three playback controls. Clean notebook sources are tracked under `notebooks/`; executed copies, HTML exports and cell-level verification are saved under `runs/notebooks/`, with a `latest.json` pointer. Use the README command to regenerate them; saved outputs are not manually manufactured.

Notebook 01 explains Phase 1 acceptance/failure paths and measured acoustics. Notebook 02 shows splits, source quality findings, review fields, reference overlays, paired boundary/audio identity, measured realized shifts and the actual empty-review gate. Development/evaluation/final-holdout crops are not used for renderer pilot selection or threshold revision.

## Assessment evidence and remaining gate

E03/E04/E07 now have executed MIR/data/intervention code, source-separated partitions, versioned audit policies and reproducible notebook outputs. E01 can use the same-waveform false-boundary example to explain the experimental problem. E02 still lacks the proposed trained mechanism and fair learned comparison. W3 has validated data/crops and an exploratory intervention pilot; reviewed targets and the research collection remain outstanding. No university mark or project completion percentage is calculated.

Before Phase 2 becomes training-ready: obtain actual independent reviews with agreed reliable intervals; resolve or justify the two annotation exclusions; expand approved pilot coverage across singers without changing the fixed groups; retain all audit failures/exclusions. Prioritize a declared multi-singer training subset and development material by the 4 October checkpoint; if reviewers are unavailable, model/loss/metric code can be tested on constructed fixtures but unreviewed anchors cannot support a trained research claim. Keep final holdout untouched for later research. Broader intervention quality, runtime-boundary testing and listening checks remain required; one singer cannot establish accuracy/generalization. Dataset redistribution terms are not yet recorded, so raw and generated audio stay untracked until provenance is settled.
