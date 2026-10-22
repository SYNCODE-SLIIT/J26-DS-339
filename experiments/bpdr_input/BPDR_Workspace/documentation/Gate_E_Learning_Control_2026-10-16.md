# Gate E controlled learning diagnosis - 16 October 2026

The bounded Gate E learning diagnosis is complete. Four declared 100-update tests reuse one existing audited Vocadito training pair (-30-cent injection), with the same initial model, frozen normalizer, full phrase and all four views. The real-input existing-loss model reduces plateau error from 30 to **20.585 cents** and learns response gain **0.310**. This establishes nonzero learning in one controlled training case; it does not establish Jamendo learnability or development accuracy. The completed 900-update comparison and its selected checkpoints remain unchanged.

Equal plateau/outside loss weighting reduces real-input plateau error further, to **12.435 cents**, but original plateau alteration rises from **0.108 to 13.190 cents** (original audited-mask average: 0.144 to 2.114). Its relative response gain of 1.025 therefore does not mean correct absolute repair. The balanced loss is **not adopted**. The deliberately privileged answer-cue controls are private-label diagnostics, unavailable at runtime and never added to the corpus. All four tests pass their limited predeclared nonzero-response rule; that rule is not an accuracy or preservation gate.

Output gradients have the correct compensation sign for both injection directions in unit checks; measured model/input gradients are finite and all four parameter states change. Analytic zero losses, raw predictions, initial/final state and fixed-update histories are saved. Independent replay authenticates the frozen evidence and replays metrics plus 16 GPU prediction vectors. A separate CPU replay exceeded its attempted 0.0001-cent tolerance; it is retained as a failed check, and these diagnostic states are not verified CPU exports. The original comparison's separately verified CPU exports are unchanged.

All 267 regression tests pass, including six new isolation/gradient/reduction checks. This is one training source, one injected shift and one seed, with response terms active from update 1 and no warmup. It cannot be compared as a new matched corpus result. The 14 Vocadito development support refusals and missing paired human-reference validation remain unresolved. Research-ready examples remain zero; Gate D/J4 review is deferred and Gate F delivered repair remains open.

## Fixed final-update results

| Input / loss | Plateau error (cents) | Relative response gain | Original plateau alteration (cents) | Original audited alteration (cents) |
| --- | ---: | ---: | ---: | ---: |
| Real / existing loss | 20.585 | 0.310 | 0.108 | 0.144 |
| Answer cue / existing loss | 19.712 | 0.350 | 0.212 | 0.057 |
| Real / balanced regions | 12.435 | 1.025 | 13.190 | 2.114 |
| Answer cue / balanced regions | 12.380 | 1.017 | 12.880 | 2.057 |

Analytic zero-correction plateau error is 30 cents; original alteration is zero. The existing-loss zero objective is 6.601240 and balanced-loss zero objective is 22.5. Compare each model only to its own objective baseline, because the reductions differ. Final real-input losses are 4.697100 and 6.901204 respectively. There are 71 audited plateau and 413 audited outside frames in the 940-frame phrase; 92.665% of view-weighted anchor targets are zero. Masks and reference/audio targets remain unchanged.

## What the diagnosis establishes

The real-input model can learn a nonzero correction on this selected-by-order source. This rules out universally disconnected gradients or an inability to learn any nonzero response. It does not rule out weak cues elsewhere, insufficient optimization, source diversity effects or unstable learning across seeds. Increasing region weight changes the tradeoff in this bounded control, but does not establish target imbalance as the cause of corpus collapse. Strong relative response with substantial original alteration demonstrates why counter-detuning gain alone is insufficient. No learning control is selected for use.

## Declaration and evidence

The immutable declaration was written before optimization. The first record in the frozen training catalog is used; no outcome-based source selection. Each arm has exactly 100 updates, AdamW learning rate .001 / weight decay .01, gradient clipping 1, seed 339, the existing 128-hidden / four-layer model with dropout .1, full 940-frame context and the existing training-only normalizer. Physical/effective batch is one pair; response is active from update 1, unlike the original comparison warmup. Only the answer-cue column and/or declared region reduction differ across arms. Intermediate updates 25/50/75 are descriptive; final update 100 is used without checkpoint selection. Four hundred additional single-pair diagnostic updates are recorded separately from the original 900 corpus-comparison updates. No development, PP1 evaluation or final-holdout acoustics are processed.

The answer cue deliberately replaces normalized column 70 with the private requested correction divided by 30 for C/D and zero for A/B. It is an invalid runtime feature used only to test optimization; the loader inputs on disk remain byte-for-byte unchanged. Diagnostic-only states carry false training/production eligibility. No learned WAV is rendered and no common-pipeline file is modified or executed.

The initial CPU reload attempt exceeded its 1e-4-cent tolerance on the first arm (maximum difference 0.000286579 cents); the attempt log is preserved. Verification then reloaded the models on the original XPU environment at the stricter 1e-5 absolute/relative tolerance and independently recomputed stored metrics. This verifies the recorded GPU experiment, not CPU portability.

Run: `runs/learning_control/0cf06946734d482c89a4646bfa040635`. Declaration: `declaration.json`; raw experiment: `evidence.json`; replay: `independent_verification.json`; figure and extra preservation measures: `panel_evidence.json`. Test log: `runs/learning_control_regression_2026-10-16.txt`.

## Next - Gate E remains open

Gate E: declare a bounded multi-source training diagnostic using the existing audited inputs, both shift directions and separate original-plateau, outside-region and repair measurements. Test preservation-aware region weighting before changing the corpus training loss. Separately design reference-feasible supports retaining trusted human-annotation interiors and explicit audits; do not remove guards to manufacture coverage. Repeat the main matched comparison only after these checks justify a new declaration. D/J4 deferred; F open.

## Final notebook verification

Both maintained notebooks freshly execute with zero errors, 34 plots and 44 audio controls. Source/executed/HTML/index hashes verify. All 267 regression tests pass. All 16 diagnostic GPU prediction vectors reload exactly; all 28 shared-pipeline files remain unchanged. The new figure was visually inspected. No loss/model adoption or learned audio rendering occurs.

Latest panel: `runs/notebooks/f1d792b1e7774280ae13898e2d32f59c/02_data_review_and_paired_experiments.html`. Final proof: `runs/learning_control/0cf06946734d482c89a4646bfa040635/final_verification.json`.
