# Gate E fixed feature-weighting test - 16 October 2026

The declared feature-group weighting test is complete. It reuses the verified saved **balanced-loss control** from the previous multi-source run and fits only the treatment for **100 updates**. The treatment divides the first 64 standardized log-mel inputs by sqrt(64)=8; all seven scalar inputs, stored feature archives, training-only standardizer, masks, detuning targets, model settings and optimizer settings stay unchanged. This is a fixed design choice from group width, not a searched gain or new audio information.

Control reuse is verified against the producing implementation, source/shift records, weighted batch, exact initial model tensors, all 32 unscaled initial prediction vectors, Torch version, interpreter and device. The training seed and four-view/dropout call schedule match. The control is historical, not a concurrent rerun. All eight pairs (one Vocadito recording and three Jamendo crops from distinct artist proxies, each with both +/-30-cent injections) remain in the comparison, with equal total dataset weight. Final update 100 is reported; intermediate checkpoints are descriptive only.

| Condition | Plateau error (cents) | Original plateau alteration (cents) | Outside alteration (cents) | Pairs passing all provisional limits |
| --- | ---: | ---: | ---: | ---: |
| Saved unchanged-input control | 17.90727 | 0.33434 | 0.17250 | 5/8 |
| Fixed group-size treatment | 29.99966 | 0.01576 | 0.01657 | 0/8 |


The fixed group-size treatment does not improve the combined recovery/preservation result over the saved control in this bounded test. It is not adopted. This tests this specific scaling and budget, not every possible use of pitch features.

The provisional per-pair limits remain plateau error <=27 cents, original plateau alteration <=1 cent and detuned outside alteration <=1 cent. They are training triage limits, not musical approvals. The prior input diagnosis remains valid: the failed crop contains the injected pitch information. This outcome does not justify removing that source, changing shared extraction or claiming a missing pitch signal. It also does not prove that feature weighting caused the original model failure.

All **288 regression tests pass**. Independent checks authenticate the 71-column transform, exact common initialization, historical-control protocol, raw prediction metrics and model reloads. The checkpoint explicitly records its required transform and remains diagnostic-only. No model/input transform is adopted, no original audio or target is changed, and no learned waveform is rendered. The original comparison remains 900 updates; previous diagnostics 700; this step adds 100 (800 diagnostic updates total), with zero control retraining. One seed and eight repeated training pairs do not establish independent accuracy. Human-reference paired development remains unavailable; research-ready examples remain zero, Gate E open, D/J4 deferred and F open.

## Per-source and direction outcomes

### Saved unchanged-input control

| Source | Injected cents | Plateau error | Original plateau alteration | Outside alteration | Boundary sensitivity | All limits pass |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| vocadito_37 | -30 | 15.403394 | 0.224252 | 0.149046 | 0.000158 | True |
| vocadito_37 | +30 | 15.566585 | 0.227111 | 0.146372 | 0.000140 | True |
| track_0076663 | -30 | 29.879608 | 0.119996 | 0.161781 | 0.000126 | False |
| track_0076663 | +30 | 30.137166 | 0.119875 | 0.161053 | 0.000138 | False |
| track_0347861 | -30 | 15.407716 | 1.002188 | 0.160486 | 0.003559 | False |
| track_0347861 | +30 | 15.565191 | 0.909191 | 0.164735 | 0.003522 | True |
| track_0368745 | -30 | 15.409271 | 0.252681 | 0.283092 | 0.000261 | True |
| track_0368745 | +30 | 15.578322 | 0.254039 | 0.252632 | 0.000288 | True |

Dataset-balanced boundary sensitivity: 0.00073249 cents. This is a recorded diagnostic, not a newly invented acceptance threshold.

Dataset summaries: `{"vocadito": {"repair_error_cents": 15.484989643096924, "original_plateau_alteration_cents": 0.22568166255950928, "original_audited_alteration_cents": 0.15661289542913437, "outside_alteration_cents": 0.14770928770303726, "original_joint_voiced_no_change_cents": 0.291443906724453, "counter_detuning_error_cents": 2.2467710375785828, "plateau_response_gain": 0.48388130962848663}, "jamendo": {"repair_error_cents": 20.329545895258587, "original_plateau_alteration_cents": 0.44299495841066044, "original_audited_alteration_cents": 0.2159241996705532, "outside_alteration_cents": 0.19729652752478918, "original_joint_voiced_no_change_cents": 0.4064670043687026, "counter_detuning_error_cents": 3.4727235436439514, "plateau_response_gain": 0.32183868144193184}}`.

Direction summaries: `{"-30.0": {"repair_error_cents": 17.81779638926188, "original_plateau_alteration_cents": 0.3412703797221184, "original_audited_alteration_cents": 0.18629883850614232, "outside_alteration_cents": 0.1754162038366, "original_joint_voiced_no_change_cents": 0.35856349393725395, "counter_detuning_error_cents": 2.8899976809819536, "plateau_response_gain": 0.3946977668591293}, "30.0": {"repair_error_cents": 17.996739149093628, "original_plateau_alteration_cents": 0.3274062412480513, "original_audited_alteration_cents": 0.18623825659354526, "outside_alteration_cents": 0.16958961139122647, "original_joint_voiced_no_change_cents": 0.33934741715590155, "counter_detuning_error_cents": 2.8294969002405805, "plateau_response_gain": 0.4110222242112892}}`.
### Fixed group-size treatment

| Source | Injected cents | Plateau error | Original plateau alteration | Outside alteration | Boundary sensitivity | All limits pass |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| vocadito_37 | -30 | 30.014475 | 0.014869 | 0.016366 | 0.000007 | False |
| vocadito_37 | +30 | 29.984632 | 0.014881 | 0.016354 | 0.000007 | False |
| track_0076663 | -30 | 30.017219 | 0.017144 | 0.016234 | 0.000009 | False |
| track_0076663 | +30 | 29.982743 | 0.017173 | 0.016219 | 0.000009 | False |
| track_0347861 | -30 | 30.017254 | 0.017375 | 0.016977 | 0.000029 | False |
| track_0347861 | +30 | 29.982661 | 0.017352 | 0.016975 | 0.000028 | False |
| track_0368745 | -30 | 30.014445 | 0.015403 | 0.017179 | 0.000015 | False |
| track_0368745 | +30 | 29.984229 | 0.015379 | 0.017142 | 0.000015 | False |

Dataset-balanced boundary sensitivity: 0.00001210 cents. This is a recorded diagnostic, not a newly invented acceptance threshold.

Dataset summaries: `{"vocadito": {"repair_error_cents": 29.999553203582764, "original_plateau_alteration_cents": 0.014875429682433605, "original_audited_alteration_cents": 0.016149811912328005, "outside_alteration_cents": 0.016359897796064615, "original_joint_voiced_no_change_cents": 0.016103819012641907, "counter_detuning_error_cents": 4.347659945487976, "plateau_response_gain": 1.4630368241341785e-05}, "jamendo": {"repair_error_cents": 29.99975856145223, "original_plateau_alteration_cents": 0.016637563550223906, "original_audited_alteration_cents": 0.0168010164052248, "outside_alteration_cents": 0.016787791935106117, "original_joint_voiced_no_change_cents": 0.016475692546616, "counter_detuning_error_cents": 4.59911725918452, "plateau_response_gain": 8.156657505272355e-06}}`.

Direction summaries: `{"-30.0": {"repair_error_cents": 30.015390396118164, "original_plateau_alteration_cents": 0.01575501449406147, "original_audited_alteration_cents": 0.016474418031672634, "outside_alteration_cents": 0.016581429479022823, "original_joint_voiced_no_change_cents": 0.016288871721675, "counter_detuning_error_cents": 4.462119499842325, "plateau_response_gain": 1.2116932036102906e-05}, "30.0": {"repair_error_cents": 29.983921368916832, "original_plateau_alteration_cents": 0.015757978738596044, "original_audited_alteration_cents": 0.01647641028588017, "outside_alteration_cents": 0.016566260252147913, "original_joint_voiced_no_change_cents": 0.01629063983758291, "counter_detuning_error_cents": 4.48465770483017, "plateau_response_gain": 1.0670093710511233e-05}}`.

## Protocol and scope

The immutable declaration predates treatment fitting. Sources, order, loss, sample weights, seed 339, 100 updates, no warmup, AdamW lr .001 / weight decay .01, gradient clip 1 and full phrase batch eight match the saved control. The existing four-layer, hidden-128, four-head model with feed-forward width 512 and dropout .1 is unchanged. No development or reserved acoustics are processed by this diagnostic. No original/target-derived feature enters the input. The treatment changes the post-standardizer gain only; rerunning a standardizer fit after scaling would be a different, undeclared experiment. At reload, apply the recorded transform exactly once. It is not a production export.

Original source/config/control receipts and all directly used inputs/checkpoints are pinned. Prior full-corpus verification remains historical; no new full-corpus extraction or audit is claimed. The main model comparison and all earlier diagnostics are preserved.

Run: `runs/feature_weighting_control/46ea1f01c42b40338e609cceaf68c5a0`. Protocol: `declaration.json`; reuse proof: `control_compatibility.json`; raw history/predictions: `history.json`, `final_predictions.npz`; source snapshot: `feature_weighting_control_at_declaration.py`; replay: `independent_verification.json`. Regression log: `runs/feature_weighting_control_regression_2026-10-16.txt`.

## Next - Gate E remains open

Gate E validation prerequisite (Gate C support design): investigate reference-feasible support geometry on the frozen Vocadito training/development partitions before another fit. Start with a read-only geometric design: can one waveform intervention span several trusted note interiors while keeping existing trusted-frame masks, edge guards, minimum audited-frame counts and independent shift audits intact? Retain the existing continuous-support refusals and failed Jamendo crop; do not lower thresholds or relabel targets to fill coverage. Declare any new support protocol before generating audio/pairs; keep PP1/final holdouts reserved. D/J4 review deferred, F open. No further gain search is declared.

This next step addresses a separate gate prerequisite that gain/loss tuning cannot solve: the original human-reference dataset currently has no paired development support under the existing continuous-reliable-span rule. A multi-interior waveform support is a design candidate only. Trusted-frame masks and independent audit rules remain authoritative; coverage must be measured and every refusal retained before any new intervention is declared or generated.

## Final notebook verification

Both maintained notebooks freshly execute with zero errors, 37 plots and 44 audio controls. Source/executed/HTML/index hashes verify. All 288 regression tests pass. All 128 initial/final control/treatment prediction vectors reload exactly and 16 final metric rows replay. The historical control is not retrained. All 28 shared-pipeline files remain unchanged; the new figure was visually inspected. Notebook 2 final sections now reflect the current Gate E evidence and original-dataset support work, replacing stale first-batch acquisition instructions. No model/input transform is adopted.

Latest panel: `runs/notebooks/4c38662141ff4591a88215e16da6ce0d/02_data_review_and_paired_experiments.html`. Proof: `runs/feature_weighting_control/46ea1f01c42b40338e609cceaf68c5a0/final_verification.json`.
