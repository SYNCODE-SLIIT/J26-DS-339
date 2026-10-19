# Working vocal-input demo

Run `Run-Demo.ps1` inside `BPDR_Workspace` and open the displayed localhost URL. `-Port 8503` selects the port used during verification.

## Current flow

1. Start with the experimental learned repair example (a real VocalSet voice with a controlled +30-cent shift), its unshifted preservation example, real HumTrans humming with added rain, or your own WAV/FLAC or microphone recording. Generated pitch controls remain secondary examples.
2. For your own recording, say whether you intended a sung/hummed/chanted melody, ordinary spoken text without a tune, or are unsure. Speech-like delivery alone is not a rejection rule.
3. Process the recording. Technical checks preserve the original. A separate provisional review checks agreeing pitch evidence, including pauses and repeated notes. It does not claim to identify correct intended notes or classify all speech.
4. Compare the original and any verified prepared preview. Noise reduction requires a genuine declared voice-free first 0.7 seconds. Failed pitch/noise previews cannot be offered as successful processed output.
5. If the selected version has sufficient provisional acoustic support, listen and confirm that it preserves your idea. Create a separate accepted demo package. Changing the selected version clears confirmation.
6. Download the WAV with metadata, or try that exact accepted waveform with the existing pinned GAME melody extractor. Successful extraction provides MIDI/CSV estimates. It does not certify musical correctness or prove improved extraction accuracy.

If reliable pitch support cannot be recovered, the interface explains the observation and asks for a clearer sung/hummed take. Unclear intent requires confirmation. Ordinary spoken text without an intended tune is preserved for technical analysis and receives a request to sing or hum the wanted idea; no melody is invented.

The shared original handoff and model-adoption flags remain unchanged. The separate export policy is `demo-user-reviewed-0.1`, and its acoustic thresholds remain unvalidated engineering settings. User review is an operational selection, not independent research approval.

## Model status

The latest trained demo option is **ExtraTrees**, not a Transformer. A bounded official VocalSet subset contains 24 unmodified source WAVs from 12 singers: 16 recordings/8 singers for training preparation, 4/2 for calibration, and 4/2 for the development pilot. Only 43 of 64 attempted training interventions passed the construction audit and supplied labels. ExtraTrees and histogram gradient boosting completed their small fits; the selected locked ExtraTrees checkpoint has 96 trees and 31 input-audio context features. A mechanical summary error required repeating the same deterministic fits from cached prepared data; its technical receipt is retained.

Actual delivered-WAV replay found two improvements among four scoreable development interventions, both +30-cent shifts: local MAE 29.202 to 5.425 cents and 30.243 to 2.905 cents. Across all four scoreable cases, including withholding and a non-improving edit, MAE was 27.442 to 14.663 cents (46.6% error reduction). This is not general singing accuracy: only one evaluation singer supplied valid labels, 12 of 16 interventions were unscorable, and the same development singer identities had been inspected in the failed first preparation attempt. All four unshifted working originals were preserved.

Broader calibration and two-singer criteria **failed and remain failed**. The historical research model is not replaced. A separately guarded card permits only an explicitly labelled experimental preview, with a post-hoc selected verified example, complete-pilot metrics and user review before separate export. It is not research-model adoption or a production accuracy claim. Runtime never receives the clean reference, injected support or target mask; these are used for supervision and post-inference scoring only. Existing magnitude, support, protected-region, timing, clipping and rollback checks remain active, with a second audit of exact encoded PCM24 bytes.

Source: [official VocalSet 1.0](https://zenodo.org/records/1193957), CC BY 4.0. Original hashes, ZIP-entry CRC32, attribution, failed intervention audits and full receipts remain under `datasets/vocalset_pilot` and `results/model_recovery`. No full corpus was downloaded or trained. The demo exposes genuine learned repair on selected examples; general natural-voice repair remains unfinished.

The following paragraphs describe the earlier pilot retained in notebook 06.

The historical BPDR Transformer remains an experimental option. The conservative stable-note baseline and declared-reference noise reduction also remain available. No failed checkpoint is silently substituted for the current model.

A single prospective original-data pilot trained a compact `LocalContextMLP` on **32 real-audio variants from 9 original recordings and 8 source groups**. Development used **12 variants from 8 other recordings and 6 different groups**. This used Vocadito/Jamendo recordings and existing controlled interventions, with no generated-voice training, full-corpus fitting, final-test access or new separation. Each of the augmentation-only, frame-only and BPDR-loss arms completed 180 updates within its 90-second limit.

Every arm failed at least one declared per-dataset response/preservation criterion. The BPDR arm improved known-shift error on Jamendo, but altered originals by 2.449 cents, above the 1-cent preservation gate. Vocadito response failed. The attempt stopped without retries, threshold changes or model adoption. These are real-recording controlled-intervention results, not natural intended-note accuracy.

The pilot and its saved predictions are in `results/small_real_pilot`. Candidate loading refuses its failed checkpoints. A good general real-singing repair model remains unfinished; this package must not be described as perfectly accurate or production validated.

## Notebook and verification

Use `notebooks/07_learned_stable_note_recovery.ipynb` for the latest model, all calibration settings, retained failed attempts, source and model hashes, fixed-denominator delivered scores, original preservation and real input/output/reference audio. Fresh-kernel execution completed 10 code cells with zero errors, four plots and five audio players. `tools/execute_recovery_notebook.py` executes it without training or reconstructing interventions.

Notebook 06 retains the earlier small-data experiment, provisional assessment and verified next-component connection. `tools/execute_usefulness_notebook.py` reruns that earlier notebook alone.

Notebook 05 remains the short controlled-demo explanation. Notebooks 03/04 explain the historical Transformer and evaluation; notebook 02 preserves the detailed experiment history.

Automated UI tests checked original `ACCEPT_UNCHANGED`, prepared-noise `ACCEPT_PREPARED`, confirmation reset after selecting a different version, and silent-input `RERECORD_REQUIRED`. These automated fixtures explicitly record that no human musical judgment occurred. A separate original HumTrans integration fixture successfully produced GAME MIDI and CSV from the exact reviewed WAV.

The latest live learned-model UI check processed the selected real controlled-shift example, displayed 29.20 to 5.42 cents, passed the exact delivered-file audit and provisional acoustic review, and preserved the unshifted example. A separate `automated_test_fixture` export produced `ACCEPT_PREPARED` with a checksum identical to the candidate; no human listening judgment was claimed. Receipt: `results/demo_sessions/ui_recovery_check.json`. Reproduce with `tools/check_recovery_demo.py`.

Remaining research work includes independently reviewed usability labels, natural novice singing/speech-like examples, calibrated acceptance coverage, blinded listening, and trustworthy real-voice learned repair. More training should follow a justified revised protocol, not an unbounded attempt to force a positive demonstration.
