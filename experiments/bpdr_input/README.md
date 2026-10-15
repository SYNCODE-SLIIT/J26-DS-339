# BPDR vocal input component

Owner: Liyanage L. N. P., IT23141506. Updated 14 October 2026.

BPDR means Boundary-Pitch Disentangled Repair. This component checks a short solo singing/humming recording, preserves usable input, and is intended to make only small justified pitch repairs before melody understanding. All component work stays here in `experiments/bpdr_input`.

## Start here

1. Read the [current master plan](docs/PP1_Master_Plan.md) for scope, dataset roles and output-quality requirements.
2. Follow the [execution plan](docs/PP1_Phase_Execution_Plan.md) in order. Its gates define when work may advance.
3. Check the [evidence tracker](docs/PP1_Evidence_Tracker.md) before claiming a feature or research result.
4. The [four-proposal audit](docs/Project_Audit_2026-10-12.md) explains the whole project and source disagreements. The [cleanup record](docs/Cleanup_2026-10-12.md) records what was removed and how to recover it.

## What works today

- Original retention, technical checks, frozen PESTO analysis and checked mono 22,050 Hz PCM24 WAV export. Supported input is WAV/FLAC, 0.25-20 seconds. Duration, time origin and internal rests are preserved.
- Vocadito inventory, singer partitions, 57 mapped crops, review sheets, audited artificial detuning and four-view boundary experiments.
- Model/loss/metric/training code for BPDR, augmentation-only and frame-only conditions. Existing 30-update CPU/XPU runs are engineering smoke runs on one singer's repeated examples.
- Component-local Jamendo mirror and pinned extraction tools, eight saved vocal stems, and pilot GAME/audio-format checks.

**What remains:** musical-usability validation, two-reviewer approved training anchors, broader intervention coverage, representative Jamendo yield, independent model selection/evaluation, repair controller/renderer/rollback and live team integration. The current saved data summary has **zero training-ready examples**. Technical acceptance and training fit are not repair accuracy.

## Folder guide

| Folder | Purpose |
|---|---|
| `src/bpdr_input`, `tests` | Maintained implementation and regression checks |
| `docs` | Current plans, review instructions and dated verification records |
| `notebooks` | Two maintained learning/inspection notebooks |
| `extraction` | Platform adapter and saved-check reproduction tools |
| `data` | Ignored Vocadito, Jamendo metadata/audio/mirror and retained vocal stems |
| `checkpoints`, `.venv`, `.tools`, `.jupyter` | Local model/runtime dependencies; keep them |
| `runs` | Ignored manifests, reviews, audit history and actual execution evidence |
| `private` | Supplied proposals, related paper, rubric and recovery archive |

## Run the existing foundation

**Owner update — reviewers deferred:** experienced musical reviewers are unavailable. Continue automated checks, controlled comparisons and provisional engineering now; add independent musical review later. Gate D/J4 is **deferred/open**, with pending records preserved, rather than a task that stops this engineering sequence. See [the revised no-reviewer workflow](docs/Reviewer_Deferral_2026-10-14.md). No review approvals or research-ready examples are inferred.

**Current work gate: Gate F — usable delivered audio (experimental renderer validation). Gate F is open.** When reporting the next step, name the gate, the action and what evidence would close it. This describes the current engineering work; it does not imply earlier research gates are complete. Gate D/J4 (actual independent reference approvals) is deferred until reviewers are available, Gate B (independent voice/support coverage) is open, and Gate E (matched research fits/evaluation) is not complete. Research-ready examples remain zero; engineering continues now.

Gate B started on 13 October: see the [coverage audit and measured limitations](docs/Coverage_Audit_2026-10-13.md). The active rendering policy and research review gate remain unchanged.

From this component folder in PowerShell:

```powershell
.\Run-Phase1.ps1 -InputAudio '.\data\vocadito\Audio\vocadito_10.wav'
& '.\.venv\Scripts\python.exe' -B -m unittest discover -s tests -v
.\Run-Notebooks.ps1
```

Open the generated `report.html` for playback and diagnostics. `ACCEPT_UNCHANGED` means no musical correction; channel selection, resampling and encoding are logged. `ACCEPT_PREPARED` is reserved for a future verified repair path. Rejected/failed outputs carry no accepted WAV.

## Keep the panel notebooks current

Maintain these two notebooks in presentation order:

1. [Input validation and analysis](notebooks/01_input_validation_and_analysis.ipynb): a fresh unchanged technical demonstration, original/working-copy playback, measured features, timing and controlled stops.
2. [Datasets, experiments and research readiness](notebooks/02_data_review_and_paired_experiments.ipynb): Vocadito/Jamendo roles, review gates, A/B/C/D construction, measured coverage/renderer limits, learning scaffold and historical smoke records, accepted-only hand-off and future pipeline scope.

After every completed BPDR step, update the affected cells and [explicit evidence index](notebooks/evidence_index.json). Clear stale outputs and execute both from fresh kernels. The executor records source/executed/HTML hashes and updates `runs/notebooks/latest.json` only after both succeed. Present the latest verified executed notebooks or their HTML exports; edit the maintained files above. Missing/changed evidence stops instead of silently rerunning extraction, interventions or model fitting. Notebook 1's fresh preparation/stop fixtures remain an explicit technical demonstration.

Keep implemented behavior, historical engineering execution, actual research validity and planned features distinct. Source-revision differences in old training records are visible. Review approvals, repair accuracy, delivered repaired audio and cross-platform integration must not be credited before verification. Keep failed/pending gates and independent singer/source counts visible.

The 14 October inventory found two maintained notebooks and four dated executed copies, with no redundant working notebook. Historical copies and before-edit backups are evidence/recovery material, not current presentation sources. Remove a notebook only after checking dependencies and redundancy, preserving recovery, and staying within this component. Continue updating the existing two instead of creating a separate notebook for each experiment.

The project uses Python 3.12+ and a component-local uv lock. On a fresh machine, install uv and run `uv sync --locked` here; `uv sync --locked --group notebooks` includes the notebook tools. Existing launchers use the ignored portable uv in `.tools` when present.

## Use the datasets correctly

**Vocadito** is the original development/reference collection: 40 recordings from 29 singers with F0 and two note annotations. Preserve its source/singer split and untouched evaluation partitions. Its annotations describe performed singing and are not intended-note correction targets.

**MTG-Jamendo** adds accompanied songs for exploratory vocal preparation. Keep artist/source groups separated, filter licenses, retain separated vocal WAVs, crop to <=20 seconds, and inspect leakage, multiple voices and pitch reliability. There are 1,604 local tracks and eight retained stems, including five fresh sources with individual receipts. The [five-artist sample](docs/Jamendo_Five_Artist_Sample_2026-10-13.md) prepared seven Jamendo crops: six screen candidates from three artists. Together with three Vocadito candidates, the current run has eighteen pending review rows and zero research-ready examples. Five of eighteen tested shifts pass v0.3 across two artists; review, renderer accuracy and voiced audit coverage remain open.

```powershell
& '.\.venv\Scripts\python.exe' -B -m bpdr_input.data_builder verify '.\runs\preparation\d274ed728d3d46f3b2661ba5291e61da'
& '.\.venv\Scripts\python.exe' -B -m bpdr_input.data_builder build --vocadito-singers 3 --jamendo-artists 5 --jamendo-tracks 1101810,1071484,86270,1113639,1086669
& '.\.venv\Scripts\python.exe' -B -m bpdr_input.separation_adapter plan --count 5
```

Build outputs live in `runs/preparation/<run-id>` with a manifest, summary and pending review sheet. A local listening page can be generated with `python -m bpdr_input.preparation_report '<run-path>'`. Each run records typed references and both accepted-crop GAME decodes; the existing Phase 2 review sheets are preserved.

Use the [current listening page](runs/preparation/d274ed728d3d46f3b2661ba5291e61da/review.html). `jamendo_batch --selection '<selection-json>' --resume '<batch-path>'` reuses verified completed receipts and refuses changed selection/settings. `jamendo_yield '<completed-batch-path>'` measures the fixed v0.3 support and shifts in a new evidence directory; it does not create eligible training pairs. Historical runs retain their original hashes; new extraction can change live metadata/history, so do not rewrite old manifests to make them appear current.

The [14 October diagnosis and reviewer packet](docs/Audit_Diagnostics_and_Review_2026-10-14.md) reuse those extracted crops. All eighteen baseline decisions replay. A declared renderer candidate gives 10/18 passes versus the active 5/18, but loses two existing successes and yields no all-six crop; it is not adopted. Two independent blank reviewer sheets and a listening page are ready. This initial diagnostic stage passed 78 checks; the wider calibration below updates the current evidence. Actual reviews and research eligibility remain pending.

The [wider calibration](docs/Renderer_Calibration_2026-10-14.md) is now complete: 108 real-crop measurements across both datasets and 216 constructed-control measurements. Candidates improve Jamendo/control totals but each loses a Vocadito success. Those three Vocadito crops represent only one singer. Keep the active renderer; 36 matched audio comparisons are ready for listening. Actual independent reviews and research eligibility remain pending. That calibration stage passed **81 checks**.

The [permission and rollback fixture step](docs/Controller_Fixtures_2026-10-14.md) verifies nine constructed controller outcomes; that fixture stage passed **93 checks**. Both notebooks explain the permission rules, exact rollback and measured constructed pitch case. These are engineering fixtures with illustrative bounds and supplied proposals, not a calibrated runtime controller or accepted repaired output. Actual reviews and research eligibility remain pending.

The [prediction-to-permission adapter](docs/Prediction_Adapter_2026-10-14.md) now reads the saved engineering checkpoint, constructs candidate intervals and withholds runtime edits when uncertainty/expressivity/permission are uncalibrated. The seen S29 crop produces 940-frame predictions below the provisional request threshold; no audio is edited. Six constructed adapter cases verify interval grouping/refusal, and that adapter stage passed **102 checks**. Both maintained notebooks include this step; useful repair, real reviews and the framewise renderer remain open.

The [framewise renderer candidate](docs/Framewise_Renderer_2026-10-14.md) is now tested on 32 constructed changing/constant-curve cases: 26 full audits pass, with all fine-error checks passing and six stricter coarse-direction rejects retained. Eight identities, exact timing/outside/rest preservation and one supplied-fixture transaction verify. That renderer stage passed **108 checks** and both notebooks include the results. The optional `rendering` dependency is pinned; the candidate is unadopted, real vocal quality/reviews remain open, and no accepted repaired package exists.

The same WAVs show an additional transition limitation: five YIN peaks above 30 cents in excluded ramp/guard windows, maximum about 42.42 cents. This is an estimator diagnostic, not a musical verdict. A guarded audit pass does not validate transitions or authorize delivery; the current notebook/report keeps this open gate visible.

**Completed Gate F diagnosis:** [direction and transition diagnosis](docs/Framewise_Diagnostics_2026-10-14.md) replays all 32 audits and reproduces all saved local waveforms exactly. All 17 opposing coarse changes in the six rejects are below 0.001 cent; the six baseline rejects remain. Full/local/oracle contrasts show both a blending contribution to measured peaks and YIN-window sensitivity. That stage passed **112 checks**; both maintained notebooks retain its evidence.

**Prior completed step — Gate F:** [controlled boundary-taper comparison](docs/Boundary_Comparison_2026-10-14.md) gives 27/32 full passes versus 26/32, all fine/preservation checks passing and eight exact zero identities. It loses one baseline pass and worsens the shorter-window maximum peak (58.24 to 59.40 cents), so the declared advancement filter refuses it. It remains unadopted; no accepted repaired package or research example exists. That stage passed **119 checks** and both maintained notebooks retain its regressions. Do not treat the taper candidate as ready for the bounded vocal comparison.

**Prior completed step — Gate F:** [phase/amplitude feasibility](docs/Phase_Feasibility_2026-10-14.md) reconstructs all 64 saved local waveforms exactly and measures 72 ramps per existing method. Destructive fundamental mixing appears in the controls: measured ratios below half the phase-aligned reference occur in 25 baseline and 45 taper ramps. These are narrowband diagnostics, not auditory quality scores. That stage passed **123 checks** and is retained in both notebooks.

**Prior completed step — Gate F:** [one phase-aware polar join](docs/Phase_Join_Comparison_2026-10-14.md) removes the measured fundamental dips (0/72 below the diagnostic ratio markers), but worsens transition YIN peaks from 42.42/58.24 to 80.81/127.94 cents. All 32 full guarded audits exactly match baseline, with 26 passes/six rejects; physical/fine checks and eight identities pass. The unchanged advancement filter refuses the candidate. That stage passed **129 checks** and remains in both notebooks.

**Current completed step — Gate F, 15 October:** [fixed-center context ablation](docs/Context_Ablation_2026-10-15.md) executes B/C/D/E on all 18 sources with predictions sealed before synthetic scoring. B→D changes only band center and reproduces Jamendo seven gains/one loss; D/E/C have the same 181/182 phase-feasible intervals, with 14/13/14 nonpositive retained estimates. Context changes add no whole-interval success or long interval. Vocadito remains 55/55 but zero >=0.86 s; Jamendo remains seven long feasible intervals. At 880 Hz, full-context D reproduces B; island-Hilbert E lowers p95 but increases maxima, and island-filter C further worsens both tails. No uniform winner or adoption; all original refusals/masks remain. The unchanged-source amplitude/harmonic follow-up is **declared, not implemented/executed**. **Next — Gate F (open):** Implement and freeze the declared unchanged-source amplitude/harmonic feasibility diagnostic across B/C/D/E and all 18 sources; preserve every short/invalid interval and seal source-only fits before synthetic scoring. No waveform correction or adoption. **Gate D/J4 remains deferred; repaired-package and research-ready counts stay zero.**

**Prior completed step — Gate F, 15 October:** [saved-array phase-regression diagnosis](docs/Phase_Regression_Diagnosis_2026-10-15.md) finds that the 14 newly nonpositive Jamendo estimates occur at local analytic amplitudes 0.54–4.94% of their guarded-island median and 138–148 ms from the nearest raw boundary, beyond the 100-ms guard. This is association, not human voicing or proven cause. Both 880-Hz controls have exactly equal B/C band centers; altered processing extent, rather than center choice, is the remaining computational difference. Their error peaks lie about 100–102 ms from raw boundaries, but p95 also worsens at >=500 ms. No guard increase, new amplitude mask or estimator change is made. A separate D/E filter-versus-Hilbert context ablation is **declared, not implemented/executed**. All original refusals remain.

**Prior completed step — Gate F, 15 October:** [fixed-mask segment-phase comparison](docs/Segment_Phase_Comparison_2026-10-15.md) implements and executes the declared A/B/C diagnostic on all ten recordings and eight synthetic controls, with predictions sealed before synthetic scoring. Local phase gains seven Jamendo intervals but loses one: B 175/182 versus C 181/182; the new 0.24-s failure contains 14 nonpositive estimates. Vocadito stays 55/55 phase-feasible intervals with zero reaching 0.86 s; Jamendo stays seven phase-feasible intervals reaching that duration in both arms. Synthetic retained intervals are 9/9 in both arms; 880-Hz constant/vibrato p95 and maximum frequency errors worsen. This is mixed diagnostic evidence, not adoption or musical accuracy. All ten original v0.1 recording refusals and masks remain unchanged; no fitting, generated vocal, repaired package or research-ready example.

**Prior completed step — Gate F, 15 October:** [frozen support-fragmentation diagnosis](docs/Support_Fragmentation_2026-10-15.md) finds 390/529 raw reliable islands no longer than the two 100-ms guards combined. Guards remove 350,154/553,014 predicted reliable Vocadito samples and 901,845/1,579,001 Jamendo samples; subsequent amplitude screening removes 8,842 and 54,259. Reliable YIN estimates outside the global filter band: 160/1,254 and 592/3,581 frames. These are tracker/filter geometry, not human truth or a proven failure cause. **All ten v0.1 refusals remain unchanged.** A separately versioned three-arm segment-phase comparison is declared for the same ten recordings and eight synthetic controls; **candidate implementation and comparison execution remain next**, not completed. Fixed masks mean local phase alone cannot make any current Vocadito interval reach the 0.86-s renderer minimum. No generated vocal, adoption or research-ready example.

**Prior completed step — Gate F, 15 October:** [source-only Vocadito/Jamendo transfer](docs/Source_Transfer_2026-10-15.md) screens all ten frozen training crops (three Vocadito/seven Jamendo) without changing estimation or refusal rules. **All ten are refused:** nine for nonpositive derived frequency, one for empty guarded support. No decomposition or corrected vocal output is produced. Two Jamendo crops have 30/24 nonpositive estimates inside retained support; the three Vocadito crops have longest retained segments 0.54/0.40/0.82 s, below the current 0.86-s renderer minimum. Geometry is descriptive and changes no refusals. Synthetic passes therefore do not establish transfer. All **175 regression checks pass**. The datasets are not declared unusable from this bounded method failure.

**Prior completed step — Gate F, 15 October:** [estimated-harmonic correction trial](docs/Estimated_Harmonic_Correction_2026-10-15.md) retains all 20 fixed requests: **five crossing-curve candidates rendered and all five pass the unchanged full audit; 15 requests withheld**, with the three prior source refusals separate. The matched baseline passes four of these same five cases; constant-880 gains a pass and none is lost. Fine/physical checks and five zero identities pass. Both aggregate transition-change maxima decrease, but three individual 2048-window peaks increase; these remain raw-change diagnostics, not target errors. Runtime uses frozen audio-only phase/frequency and source-fitted harmonics with the same residual input, not construction truth. All **170 regression checks pass**. No adoption, research learner fitting or accepted repair. The subsequent recording transfer result is recorded above.

**Prior completed step — Gate F, 15 October:** [source-estimated harmonic/residual separation](docs/Harmonic_Separation_2026-10-15.md) decomposes all five eligible profiles and withholds the three prior refusals. Six waveform coefficients per source are fitted from source audio/estimated phase before midpoint; all fits are sealed before construction scoring. Phase estimation used the full clip, so the second-half evaluation is not independent prediction. Harmonic-model evaluation relative RMSE ranges 0.000068835–0.000577455. Raw harmonic-plus-residual recombination and the separate zero-edit bypass reproduce all five sources exactly, including excluded samples. Exact recombination alone is an accounting check, not correction quality. All **162 regression checks pass**. No candidate WAV, pitch shift, learning-model training or adoption. The subsequent nonzero synthetic trial is recorded above.

**Prior completed step — Gate F, 15 October:** [audio-only source estimation](docs/Source_Estimation_2026-10-15.md) executes the frozen eight-source protocol with all predictions saved and hashed before construction-truth scoring. **Five sources return estimates; three remain refused** (vibrato-880, glide-220, rested-220) for nonpositive frequencies outside retained support. No post-outcome relaxation is adopted. On the declared joint support, per-profile median frequency errors are 0.202–0.303 cents; these descriptive scores also include refused rows and are not repair accuracy. The rested source has 441 predicted false-support samples in silence before guards and zero after guarding. All **154 regression checks pass**. No corrected WAV, model fit or renderer is created. The subsequent harmonic/residual result is recorded above.

**Prior completed step — Gate F, 15 October:** [frozen control diagnosis](docs/Known_Control_Diagnosis_2026-10-15.md) reproduces all **16 guarded audits and 32 transition peaks exactly**. The glide reject is one positive PESTO change of +0.000413 cents at 1.24 s against a -15.6-cent request; its sign remains positive in float64, and native/aligned timestamps agree. YIN measures -17.420747 cents. The reject remains; no epsilon is adopted. Existing transition peaks measure pitch-change magnitude, not target error. Separate known-control center-law residual maxima are 5.422835/7.289706 cents at the fixed 2048/1024 windows; these remain finite-window diagnostics, not auditory thresholds. All **147 regression checks pass**. No WAV or renderer is created or changed; all eight measured and 24 withheld cases remain. The subsequent audio-only estimation result is recorded above.

**Prior completed step — Gate F, 15 October:** [known-source analytical audio controls](docs/Known_Source_Control_2026-10-15.md) render the eight preselected strict cases and explicitly withhold 24. Seven of eight full audits pass, matching the baseline count on those same eight cases, but the passing cases differ: constant-880 is gained and glide-220 is lost to one opposing coarse-direction frame. All eight fine-pitch/preservation checks and eight exact identities pass. Maximum transition estimates improve on the matched subset, yet four individual 2048-window cases and glide at 1024 worsen. Source-relative amplitude dips improve across 18 ramps; this uses a different denominator from the older branch-relative diagnostic. All **141 regression checks pass**. These controls require known construction F0/phase and are not a practical vocal renderer or musical accuracy. The subsequent frozen diagnosis is recorded above.

**Prior completed step — Gate F, 15 October:** [source-anchored mathematical feasibility](docs/Anchor_Feasibility_2026-10-15.md) checks all 32 cases/36 supports before another waveform trial. The fixed sign/magnitude-preserving bridge solves nine supports across eight crossing-curve cases. Eighteen supports fail the conditional necessary budget; nine more remain unresolved by this basis. A relaxed +/-30-cent solution can reverse a requested correction and is diagnostic only. All **134 regression checks pass**; no new WAV/renderer is issued. The subsequent known-source audio control is measured above.

The existing working extraction route is:

```powershell
.\Run-Extraction.ps1 -IndexOnly
.\Run-Extraction.ps1 -First '1079195,1348174' -Gpu -VocalOnly -KeepStems -Threads 8
```

It requires the installed component tools. Fresh setup uses `Setup-Jamendo.ps1` and `Setup-ExtractionTools.ps1`; do not rerun setup or full-volume extraction merely to inspect existing evidence. The Intel Arc environment lives under `checkpoints/melody-extraction/.venv-xpu`.

Reuse the team's pinned separator and GAME through component-owned adapters. Read GAME's **`model_midi_pitch`**; the published pitch is +12 semitones. Instrumental/combined melody, chords and key are not BPDR inputs. The newer nine-stage melody/chord runner is not a drop-in replacement for this launcher. See the [extraction assessment](docs/Jamendo_Extraction_Verification.md).

Research training remains review-gated. Engineering mode may use audited unreviewed examples but stamps every output `exploratory_not_research`. The downstream component receives a validated accepted WAV, status and provenance, not these dataset pseudo-labels.
