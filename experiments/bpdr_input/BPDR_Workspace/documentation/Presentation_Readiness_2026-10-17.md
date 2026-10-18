# BPDR presentation handover - 17 October 2026

**Status: verified engineering prototype; the main real-user research study is not complete.**

The selected Transformer remains the working mixed Vocadito/Jamendo/synthetic checkpoint from the sealed presentation release. The completion pass fixed real-input handling, used HumTrans in a separate training attempt, added and checked optional colored-noise cleaning, checked the pinned downstream extractor, and updated exactly two maintained notebooks. Failed models and earlier runs are retained. Neither new learning attempt justified replacing the selected model.

## Run and demonstrate

From this component folder, run `./Run-Presentation.ps1`, then open `http://127.0.0.1:8501`. For dependency reproduction use `.tools/uv.exe sync --locked --extra rendering --group notebooks --group presentation`; the notebook group now includes `mido==1.3.3` for MIDI diagnostics.

1. Choose the labelled synthetic 20-cent-detuned phrase and request the separate rendered candidate. Explain the known constructed reference and the numerical repair audit.
2. Choose **Real HumTrans humming + recorded rain - controlled mixture**. Confirm the noise-only prefix only because this labelled fixture has a known 0.7-second noise-only interval. Analyse it and play the original and separate cleaned preview. This demonstrates cleaning of real humming with added recorded noise, not recovery of arbitrary musical intent.
3. Show the unchanged-original example and a silent-recording stop. Uncertain or unsupported edits must not be forced.
4. Show Recorded experiments, including every failed real/humming attempt. The eight evidence tables are measured results; they are not eight independent studies proving accuracy.
5. Explain the two notebooks in order: technical input handling, then model/data/experiments. Notebook 2 sections 10-12 contain the completion evidence and section 13 gives the permitted claims.

Upload supports WAV/FLAC or local microphone capture, 0.25-20 seconds total, solo voice without accompaniment. Include at least 0.7 seconds of genuine room tone before humming if using optional cleaning. Do not declare speech, breathing, music or humming as noise-only. Files stay in component-local runs and are not automatically reused for training. Physical microphone capture, native file-picker interaction and Mac execution still need manual verification; automated UI checks do not establish them.

The accepted downstream package retains the original working copy. Cleaned/rendered candidates are separate comparisons and downloads. Prepared downstream handoff remains off under the recorded owner decision. No user needs to enter note names, a key or music theory.

## Selected model and rejected candidates

| Checkpoint / experiment | Result and decision |
| --- | --- |
| Selected mixed-data BPDR, seed 339 | Synthetic pretraining plus 300 real+synthetic adaptation updates. Synthetic MAE 4.47 cents, median response 1.03; real response remains approximately zero. Retained for the experimental presentation preview, not adopted as validated real repair. |
| New A4-safe real-input learning pilot | Eight fixed Vocadito/Jamendo training variants, 200 updates, balanced correction/preservation anchor and output-gain30 optimization. Jamendo response approximately zero; Vocadito response .0028 and original alteration 1.18 cents. Training gate failed; larger matched fit stayed locked. |
| New humming-learning pilot | Twelve real-humming training variants from two people mixed with synthetic training variants, 200 BPDR updates. Training MAE 13.20 cents versus zero-output 20, response .426, original alteration 1.07 cents. On six development variants from one other recording/person: MAE 20.30 versus zero-output 20, response .115, original alteration 11.25 cents. Failed development and preservation checks; retained for diagnosis, not selected. |
| Humming candidate on synthetic heldout data | MAE 10.90 cents, worse than selected model's 4.47. No automatic replacement of working weights. |

`runs/completion_model_decision.json` records the choice and selected checksum. The original sealed release and all nine synthetic reference checkpoints, including failed seed 340, remain unchanged. The new checkpoint files are real executed artifacts; their existence is not evidence of deployment or musical validity.

The new objectives/parameterization are documented engineering changes. The three original matched Transformer conditions and their completed experiments remain the methodological controls. A further matched comparison is not unlocked by an unsuccessful small learning check. No BPDR advantage over the controls is established.

## Dataset use and the reason for failure

Real datasets are already used. The selected model includes Vocadito and Jamendo adaptation; the separate candidate also uses HumTrans training recordings. Failure to undo a known added detuning does **not** show that original music is bad. Technical acceptance, musical reference approval and learned repair are different decisions.

Real pitch contours include transitions, expression and tracking variation. In the saved training diagnosis, median adjacent voiced-frame changes are .91 cents in synthetic data, 7.78 in Vocadito and 4.61 in Jamendo. These are not intonation-error labels. Median labelled correction support is 47, 10 and 40 frames respectively. Much longer preservation exposure, synthetic shortcuts, limited suitable real coverage, domain differences and seed instability are plausible difficulties, not isolated causal findings.

### Versioned input fix

`runs/real_learning_v2/10b5b8ea10ca416c89f2e8a5e19248a4` rebuilds all 995 training and 304 development variants. Valid zero-cent A4-bin pitch is retained using explicit voicing and the unchanged 50-cent YIN/PESTO agreement guard. Other acoustic/boundary columns and original target files remain unchanged. Phrase centering and training-only hierarchical normalization are saved; pitch/pitch-change scales are declared as 100/30 cents. Restored-frame totals count views/variants, not people. Correcting this bug did not establish real learning.

The 200-update pilot uses the same fixed eight records throughout. Its objective balances correction and preservation regions within C/D while preserving A/B; output-gain30 changes optimization, with the saved head converted back to physical cents. The normalizer remap is checked to preserve the pretrained function at initialization. This is an overfit diagnostic, not a held-out accuracy claim.

### HumTrans intake and suitability

[HumTrans](https://huggingface.co/datasets/dadinghh2/HumTrans) data/card/split metadata is pinned at revision `c3c8504280cbdef0f00bb8119f16f034bea2bd9f`. The retained card lists CC BY-NC 4.0. Initial audio retrieval transferred about 16 MB from the 14.7-GB archive using standard byte ranges; an additional prospective long-note sample was also retrieved. Raw recordings, references, hashes, declarations and failures are component-owned under `data/humtrans`.

The initial 16 recordings used separate person/composition groups for six training and two development people, with two further people reserved unseen. All selected audio came from official TRAIN. The official split metadata includes all ten people in each split, so it does not establish unseen-person performance. The component split is separate from that benchmark and is not a representative demographic sample.

Only two initial training recordings yielded qualified long-note pairs; none of the initial development recordings did. Eight additional development recordings were selected by a declared MIDI-held-note-duration rule before audio/model scoring, preserving the same development people and distinct compositions. One yielded qualified pairs. All **24 source recordings** remain reported: three yielded 18 audited variants (12 train, six development), and 21 yielded no usable audited pair. Suitability is specific to the current long stable-note repair protocol, not a judgement that the other humming is bad.

The [authors' paper](https://arxiv.org/html/2309.09623v2) describes guided Da-Da-Da recording by musically experienced participants and quality filtering. It does not validate inexperienced free humming, every closed-mouth style or noisy novices. MIDI is score-derived; lag/octave discrepancies are diagnostic only, not corrected labels, model inputs or ideal musical targets. A [later study](https://github.com/shubham-gupta-30/humming_transcription) also reports timing-label problems; automatically corrected subsets are not human BPDR approvals.

`runs/humming_experiment/14d6b846d0bd487d95c048b8cd641143` reuses both preparation runs without regenerating earlier outputs. Its six development variants come from one prospectively selected recording/person: report point measurements, not population confidence or general accuracy. `runs/humming_learning/6a24bbda0fbf497d8aa564e02c685f0a` contains the declared training attempt, saved history, weights and all endpoint results. No official validation/test audio or reserved final-test audio was processed.

## Optional noise cleaning

Proposal page 26 permits limited attenuation only with a reliable noise-only interval, original phase, rejection of vocal damage and feature re-analysis. `noise_preview.py` implements this separate optional path. It does not treat low pitch confidence as noise, does not trim pauses or shift the timeline, and does not remove overlapping speech. Caller declaration is required; screening cannot prove it is correct.

The first white/fan-proxy study is retained: 13/24 real-source controls and 8/8 constructed controls passed; the failed real controls were withheld despite SNR improvements. This earlier policy used a high flatness floor and a 6-dB attenuation cap.

The first recorded-noise screening diagnosis rejected a stationary rain reference primarily for its colored spectrum. `ColoredNoisePolicy` version 2 permits flatness down to .01 and reduces the attenuation cap to **3 dB**. Periodicity/stationarity and voice-preservation limits are unchanged. These are provisional fixed engineering values, not perceptual production calibration.

The [ESC-50](https://github.com/karolpiczak/ESC-50) source/attribution files and audio hashes are retained. Previously unused fold-2 noises were mixed with six real training-source excerpts: two each from Vocadito, Jamendo and HumTrans. Both policies were evaluated on the same mixtures. Original policy passed 6/24 rain/vacuum controls; revised policy passed **15/24**. All 12 clock controls were withheld. Six breathing controls had no eligible declared noise reference and remained exact identity; this is not evidence of automatic breath detection. Failures and correlated variants are retained.

These are mixtures made from recorded sources with a known baseline, **not naturally co-recorded novice/clean pairs**. Numerical pitch/voicing preservation does not certify breaths, consonants, transients, listening quality or intended notes. A different waveform is useful only with supported improvement and preservation.

The app uses version 2 for its optional preview. A labelled real-humming/rain example passed actual encoded-waveform checks, retained sample count and original handoff, recalculated features and fed that separate candidate to inference. `runs/humming_noise_live_check.json` and the UI verification identify the artifacts.

### Fixed downstream extractor diagnostic

`runs/completion_handoff_check/8a9fb545567d4c14921631b392b73e78` runs the pinned component-local GAME checkpoint on baseline, repeated baseline, noisy and cleaned preview, CPU-only. Baseline-repeat onset/pitch agreement was 1.0; noisy and cleaned agreement were both .9444. All outputs had 18 notes. No extraction gain was observed. These are agreements with GAME's baseline predictions, not human transcription accuracy. No shared extraction pipeline was executed or changed and no candidate was delivered as accepted prepared audio.

## Proposal traceability and gates

| Requirement | Implemented evidence / remaining limit |
| --- | --- |
| Four views, compact Transformer, L_A/L_B/L_D/L_S, three matched conditions | Implemented and executed; balanced/scaled diagnostic changes declared. Consistent BPDR advantage not established. |
| Public real data and grouped partitions before augmentation | Vocadito/Jamendo preparation and adaptation; HumTrans grouped intake and separate training attempt. Jamendo groups are artist proxies. Reserved final test untouched. |
| Original preservation, technical/rerecord/failure states, accepted-only contract | Working technical pipeline and checks. Musical acceptance is separate. |
| Bounded correction, timing/rest preservation, uncertainty and rollback | Verified numerical fixtures and candidate rendering; trustworthy real musical pitch repair remains unvalidated. |
| Conditional limited noise removal, original phase and re-analysis | Implemented optional preview and recorded controls; natural-noise/novice/listening validation pending. |
| Recording/upload, comparison playback, plots and clear feedback | Local app and maintained notebooks; automated UI check passes, physical microphone and other-OS checks unverified. |
| Independent suitability review and listener evaluation | Deferred by owner because experienced reviewers are unavailable. No generated approvals or participant responses. |
| Consented novice participants and downstream benefit | No participant study; one fixed-extractor consistency diagnostic shows no extraction gain. |
| Reproducibility, checkpoints, timing and source provenance | Saved inputs, declarations, hashes, histories, all failures, selected/rejected weights, fresh notebook execution and regression/UI checks. Timing remains a limited earlier constructed-phrase sample. |

Gate E remains open for reliable real-voice correction and the methodological comparison. Its bounded input/learning investigations are completed, but the research gate did not pass. Gate F has engineering candidates and noise checks, without validated real musical repair. D/J4 remains deferred. Research-ready examples remain zero. More training or independent review alone is not claimed to solve the technical generalisation failure.

## Panel explanation and verification

Use this wording: **BPDR is a working, traceable experimental vocal-input prototype. The Transformer demonstrates local repair on controlled humming. Real datasets were used and unsuccessful learning attempts are reported. Optional cleaning improves measured noise on some real-source controls while preserving measured voice features. General real-user pitch repair and the proposed BPDR advantage are not established.**

The complete component suite passed **448 tests** (`runs/completion_regression.log`). Targeted checks cover explicit A4 voicing, hierarchical weights, balanced supervision, initialization remapping, bounded ZIP intake, MIDI tempo/offset parsing, noise-reference rejection and vocal-damage rejection. Earlier sealed-release test counts are historical.

`runs/completion_app_verification.json` records a zero-exception AppTest run with eight evidence tables and a labelled real-humming noise-preview download. It is not a physical microphone or visual-browser verification. The latest notebook run is pinned in `runs/notebooks/latest.json`; both notebooks were executed in fresh kernels without fitting. The final evidence bundle records their paths/hashes and the exact code/model versions.

Existing review records remain unchanged. `docs/Reference_Review_Guide.md` points to the retained human review bundle. Future participation needs the project's actual ethics/consent process; no records were fabricated. Future research needs stronger reviewed real/humming coverage, a newly declared learning design, blinded evaluation and a reserved final test. Those are unfulfilled research requirements, not hidden completed stages.

All changes are confined to `experiments/bpdr_input`; the 20 shared-pipeline source hashes and selected-model checksum are checked unchanged. Old runs, notebooks, reviews and the sealed release remain recoverable. The completion archive is a review/evidence package for this workspace, not a complete portable dataset/runtime distribution.

Implementation and explanatory documentation used AI assistance. Reported measurements come from saved local execution. The research owner must review the implementation and defend its methodology and limitations.
