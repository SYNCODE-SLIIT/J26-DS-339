# Transformer usability work

**28 October 2026 — iteration complete, usability goal still active.** Read the
[measured result](Transformer_Usability_Result_2026-10-28.md): the guarded model
fails the independent-singer repair gate and remains inactive. The owner prioritized making the
Transformer useful in BPDR before selective editing or other presentation
features. Completion requires measured delivered-audio benefit and original
preservation, not merely training completion or a favorable frame percentage.

## Scope and source reservation

Use the existing BPDR app, source-preserving handoff and user review. Do not
modify `pp1_demo` or another member's feature. The historical ExtraTrees export
and failed Transformer evidence remain intact. A new guarded Transformer
adapter is hidden unless its separately saved output-validation gates pass.

The official [VocalSet 1.0 deposit](https://zenodo.org/records/1193957) supplies
the existing CC BY 4.0 source data. The bounded downloader fetched 42 specific
additional files, with an 80 MiB transfer cap, exact ZIP ranges and per-member
CRC/hash verification. It did not fetch the full archive. Originals and
attribution are retained in `datasets/vocalset_expansion_2026-10-28`.

IDs 1–4 remain training; their additional i/o/u vowels and all five straight
vowels from female7/male7 expand training to 50 original recordings. Existing
singers 5/6 remain previously inspected development. Before extracting any
new evaluation features, an explicit amendment reserves **female8/female9 and
male8/male9** as four final singers, aligning with the master plan. This
amendment precedes acoustic analysis and uses no new evaluation outcomes.
Downloaded source inventory/duration checks are not acoustic evaluation.

## Executed and active conditions

| Condition | Change | Current result |
|---|---|---|
| Prior 27 October model | 81-frame context and five objectives | No useful default calibration repairs; retained as historical baseline |
| 28 October v1 | Four-second context, 51 tokens, two-layer width-32 Transformer; signed bounded amplitude plus learned activity; broader audited training | 8 grouped fits and 3 final refits completed. 121/200 interventions scoreable; 9 scoreable singers / 39 source recordings. Gated CV MAE 8.601 cents; original mean 2.646 cents, but all-singer gate fails. Most proposed supports remain below the unchanged 0.86s minimum |
| Longer-deviation v2 | Actual 1.52–1.84s deviations rendered with Praat and independently audited; six-second context, same token count/model size | 100/200 training interventions scoreable; 8 grouped fits + 3 refits completed. All-singer gate failed; no development/final access or activation. Longer actual deviations are a different scope, not padded runtime candidates |
| Drift-aware v3 | All available original-only anchors, including recordings without valid intervention labels; outer-flank drift normalization | 8 grouped fits + 3 refits completed. Gated CV MAE 17.989 cents, original mean 2.043; gate failed. An empty-anchor sampler failure was corrected with a retained technical receipt. Training diagnostics reveal cross-note context saturation and partial absorption of the injected shift by broad flank medians |
| Pitch-neighborhood v4 | Same continuous measured voiced region and +/-70-cent neighborhood; two outer flank observations; invalid context masked from attention | Completed: four singer folds at3200 updates, three refits at4800. Gate failed. Larger optimization budget is declared; no isolated feature-only causal attribution |
| v4 actual development replay | Frozen model/policy on previously inspected singers5/6, vowels a/e/i | Failed: 1/48 attempted interventions improved (24 scoreable), all12 originals unchanged. Two singers had no scoreable cases under the longer-scope rule. No final access or activation |
| v5 data-only context check | Longer untouched context on both sides of an actual local deviation | 44/200 attempts scoreable; insufficient preparation yield. No neural fit or final access |
| v6 step-conditioned Transformer | Genuine two-layer self-attention model learns signed correction/activity from three engineered, input-only local-step measurements; version2 acoustic-region eligibility | Training-side prediction gate **passed for all8 scoreable held-out singers**. 108/200 interventions scoreable, 27 scoreable recordings /50 attempted. Raw MAE13.341c, original request0.160c, response0.587. Other training singers have no valid repair labels and are not counted as validated |
| v7 frozen model + stricter shape guard | Exact V6 weights, development-calibrated candidate-local fit gain>=.75; all physical limits retained | Development9/48 verified improvements, all12 originals unchanged. Independent four-singer result5/48 improvements, all12 originals unchanged: **failed scoped usability**. No activation |

The new network receives only measured acoustic features from the current
recording. Targets, clean originals, source IDs and known intervention supports
are absent from inference. Signed corrections are learned by a Transformer;
the activity head learns unchanged behavior. The feature redesign does not
establish the proposal's BPDR-objective advantage.

**The v6 model is an engineering variant, not the original model renamed.**
Its three features measure a local plateau level, fit improvement and residual
scale against the current recording's own drift. A Transformer calibrates these
features using independently audited real-WAV intervention labels. The feature
algorithm is not a reference melody and never receives intervention locations.
Provisional boundary metadata is absent, so invariance is by construction.
This cannot demonstrate a learned BPDR-invariance advantage. The primary
proposal-aligned comparison and a bare-feature versus neural comparison remain
separate scientific obligations. The bare training-data feature measurement
had MAE12.850c versus the model's13.341c on these folds; do not claim the neural
model improves raw MAE over that feature estimate. The learned output has no
above-30c requests in this comparison; the bare measurement has some.

The version2 eligibility fix separates large measured pitch changes within
continuously voiced audio before testing held-region stability. The previous
version could reject a whole multi-pitch phrase despite stable held regions.
Small20–30c changes stay inside the region; large transitions and protection
still veto full candidates. Feature-only short-gap context must never be
confused with permission to edit or join unreliable audio.

The v6 data declaration uses actual1.16–1.40s local deviations, >=.26s measured
unshifted context on each side, at most70% of the held region and24% of the full
recording. Median pitch is used only to assess/measure context; original WAVs
are never smoothed. All source attempts and construction failures are retained.
The training-only feature check passed its pre-fit signal/preservation gate:
48.6% lower controlled correction error, response0.586 and zero requests on
available original-anchor controls. This is a diagnostic ceiling, not delivered
audio success or evidence from fresh people.

The separate held-voice policy keeps all supported sign-consistent candidates
and distinguishes small pitch-only boundaries from acoustic onsets/large
transitions. Unreliable voice, user-protected overlap and real transitions still
veto an entire candidate. Candidates are never extended just to meet duration.
Maximum 30 cents, minimum 0.86s support, maximum 25% changed duration, agreement
limits and all existing encoded-WAV audits remain fixed. The training-only
policy grid uses activation 8/12/16 cents, release at half activation and gaps
0/40/80ms; never bridge unreliable/opposing/protected evidence.

## Completion checks

Select models and policy on training-side singer folds only. Freeze all three
seed endpoints (339/340/341), with 339 fixed in advance, before development.
Development must pass before final singers are analyzed. Do not substitute
failed or unscoreable singers based on outcome. Keep all attempts/exclusions.

For each evaluation singer require raw-original requests <=3 cents, exact
original preservation after the real processing path, at least 25% of all
attempts yielding verified improved WAVs, and at least 25% delivered error
reduction. Both intervention directions must be represented by actual gains.
Verify decoded PCM24, protected/outside samples, timing, sample count and
rollback. A scoped practical preview does not certify natural intended notes,
independent listening or the broader research hypothesis.

Initial tests passed for exact center/batch alignment, translation-invariant
input, magnitude bounds, full protected-candidate veto, multiple candidates,
unreliable-gap protection and distinction between small pitch steps and large
transitions. Additional runtime-loader, output and integration checks will be
recorded with the final evidence. No full-suite claim is made here.

The added cross-note regression test explicitly checks that a neighboring
400-cent pitch level cannot saturate away a 20-cent local shift. An all-unvoiced
input must remain finite and request identity. Old checkpoints retain their
serialized earlier centering contract; executed source snapshots preserve the
earlier failed conditions.

## Evidence and commands

- `results/transformer_usability/2026-10-28_v1/`: declaration, all folds,
  final refits, locked diagnostic selection and executed source snapshots.
- `results/transformer_usability/2026-10-28_long_v2/`: real-WAV training data,
  input/source hashes, folds and four-final-singer partition amendment.
- `tools/prepare_transformer_expansion.py`: bounded source download/resumption.
- `tools/prepare_held_voice_training.py`: expanded original condition.
- `tools/prepare_long_held_voice_pairs.py`: independently audited longer deviations.
- `tools/train_long_held_voice_transformer.py`: grouped comparison/refits.
- `tools/train_neighborhood_transformer.py`: current input-localized gated model.
- `tools/train_step_feature_transformer.py`: current v6 step-conditioned model;
  four held-out folds at1000 updates and fixed seeds339/340/341 at2000 updates.
- `tools/evaluate_held_voice_transformer.py`: training-only policy selection,
  locked development and conditionally unlocked final evaluation.
- `tools/check_held_voice_accuracy.py`: independent after-training repeat without
  refitting; `--latest` requires a passed, integrity-checked scoped preview card.
- `tools/activate_held_voice_transformer.py`: refuses activation unless both
  locked development and four-singer final actual-WAV gates pass.
- `tools/evaluate_step_feature_transformer.py`: v6 policy selection and locked
  development/final evaluation, using the same numerical/encoded-WAV audits.

This report preserves the execution history. The [result report](Transformer_Usability_Result_2026-10-28.md)
contains the final outcome and integration/refusal receipts. The model task
remains active; a hidden adapter or completed fit is not an available validated
repair feature.
