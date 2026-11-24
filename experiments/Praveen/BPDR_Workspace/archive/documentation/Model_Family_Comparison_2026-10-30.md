# Matched acoustic model-family comparison

## Why this comparison

The owner's30 October update asks whether Transformer choice is the problem.
V26 confirms that one more Transformer variant can improve raw error while
losing delivered repair. V27/V28 therefore compare sequence encoders and a
classical baseline against the authenticated reference. No model is selected
because of its reputation or a historical demo metric.

## Frozen conditions

All conditions use the same200 attempted cases,108 valid reference-controlled
interventions,50 original recordings, original singer/source folds and existing
joint acoustic context. Only8 attempted singers have scoreable interventions;
the other2 remain unassessed. No new audio, people or final-test exposure.

| Condition | Inputs / structure | Training |
|---|---|---|
| Reference Transformer |51x8 acoustic tokens, presence mask,32 hidden,2 encoder layers, original center correction/activity heads | Existing authenticated V12 matched-baseline fits,1200 updates |
| Bidirectional GRU | Same projection/positions/center heads,2 GRU layers,16 units per direction; packed present tokens | Same V12 fit function, sample digest,1200 updates, losses and optimizer |
| Temporal convolution | Same projection/positions/center heads,4 residual blocks,2 symmetric kernel3 convolutions per block, dilation1/2/4/8 | Same V12 fit function, sample digest,1200 updates, losses and optimizer |
| Fresh ExtraTrees | Same tokens flattened plus presence mask,459 features,256 trees,depth18,min leaf8 | Exact same43200 sampled rows/fold and original/positive weights; weighted squared-error objective |

GRU/TCN replace only the encoder and checkpoint implementation in isolated
workers; common historical source files remain unchanged. Actual sampler,
training source and singer digests must match each frozen V12 receipt.
The classical model has a different loss and fitting budget; neither equal
wall-clock time nor equal parameter counts are claimed. Record both before
interpreting the practical comparison. Seeds/frames/variants are not people.

No IDs, provisional boundary fields, reference pitches, keys, notes or target
intervals enter prediction features. The contextual presence mask is input
geometry, not intervention supervision. All outputs stay bounded30c and zero
on unvoiced frames. Tree regression has no calibrated activity probability.

## Execution and qualification

Implemented tests cover gradients, padding-edge inputs, checkpoint reload,
bounded output, unvoiced identity and invariance to excluded boundary metadata.
V27 eight neural fits and both audio evaluations finish. V28 four tree fits,
recovered verification and audio evaluation also finish. Existing shared
demo/selection and6 protected artifacts stay intact.

| Condition | Raw MAE,cents | Original prediction magnitude,cents | Raw gate | Verified repairs /200 | Delivered after MAE,cents | Original WAVs unchanged |
|---|---:|---:|---|---:|---:|---:|
| Reference Transformer | 11.901 | 0.789 | Pass | 23 | 20.342 | 50/50 |
| Bidirectional GRU | 12.751 | 2.423 | Fail | 18 | 21.626 | 50/50 |
| Temporal convolution | 11.893 | 0.797 | Pass | 19 | 21.281 | 50/50 |
| Fresh context ExtraTrees | 12.033 | 1.439 | Fail | 20 | 21.217 | 50/50 |

GRU original predictions exceed3c for female2 (8.284c) and male7 (7.684c),
and male7 response0.229 is below0.25. TCN raw error is effectively tied with
the reference and loses actual delivery. **Neither neural alternative is
selected.** These are single frozen configurations under the declared budget;
do not generalize the outcome to all recurrent/convolutional models or claim
statistical Transformer superiority. TCN has25,506 parameters and GRU10,018;
recorded wall time includes concurrent execution and is not a controlled
inference-speed benchmark.

ExtraTrees also loses the delivery comparison and exceeds the raw-original
3c limit for male7 (4.428c). It is not selected. Independent replay verifies
its4 trained checkpoints and sampler/source lineages,216 input views,20
encoded WAVs,50 original identities and6 protected artifacts.

Next run actual median11/Praat delivery with unchanged original controls and
every existing numerical, support, preservation and encoded-audio audit. Compare
raw error, per-singer response/original diagnostics and remaining delivered
error. Select only after all-singer raw gates and all50 original identities;
lowest delivered singer-macro MAE wins, with reference retained on ties.
Repair count alone cannot determine selection.

Independent sequence replay passes:8 checkpoint lineages and matched sampling
digests,216 held-out input views per family (432 total), all37 encoded verified
WAVs,50 original identities per family,6 protected files and absent activation.
No qualified new model follows; the reference remains retained. All tree
fitting, evaluation and verification also complete. Fresh confirmation and
musical review remain separate. Implementation or a running fit is not an
accuracy or usability result.

Commands from the workspace environment:
`compare_sequence_models.py declare/cv/delivery` and
`compare_context_tree.py declare/cv/delivery` (one stage argument at a time).
Evidence roots are `2026-10-30_v27_sequence_comparison` and
`2026-10-30_v28_context_tree` under `results/transformer_usability`.

The tree runner terminates at its bitwise checkpoint-prediction check after
saving all4 fits: threaded float64 summation differs by at most6.94e-18 in
the first observed128-row probe. The original terminal failure is retained.
After confirming terminal status and all4 checkpoints, the separate
`resume_context_tree_verification.py --original-cv-terminal` command completes
bitwise-exact serial reload, parallel agreement to normalized1e-14 and unchanged
sampler/source hashes, recording explicit amendments. It never refits or changes
prediction rules, edit limits or encoded-WAV checks. Independent verification
passes afterward. No failure or training history is overwritten as a success.

The [feasibility diagnosis](Repair_Feasibility_Diagnosis_2026-10-30.md) adds
known-label realization and outside-injection preservation diagnostics and
defines the next measurement/localization investigation. No generic model-family
sweep is recommended from these results.

Primary research and component-safe usability suggestions are in
[model choice and input usability](Model_Choice_and_Input_Usability_2026-10-30.md).
