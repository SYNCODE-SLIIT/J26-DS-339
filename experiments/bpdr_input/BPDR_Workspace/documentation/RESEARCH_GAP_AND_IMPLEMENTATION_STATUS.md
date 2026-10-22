# Research gap and implementation status


## Proposed research, implemented work and remaining work

| Requirement | Proposal | Existing implementation / evidence | Remaining work |
|---|---|---|---|
| Transformer | Compact bidirectional correction encoder | `modeling/model.py`; saved historical checkpoint loads | Reliable real-voice correction response |
| Four-view learning | Shared A/B/C/D model | Pair preparation, losses and matched historical experiments exist | Independent natural-voice validation of the methodological advantage |
| Boundary score | 71st input for full and augmentation-only | Present in Transformer; absent from tree's 31-feature interface | Calibrated natural analyser errors and held-out split/merge/jitter evaluation |
| Boundary consistency | Same audio should get stable repair | `boundary_invariance`; numerical tests and historical training | Demonstrate benefit beyond low response / constant output |
| Detuning response | Known added e requires response −e | `counter_detuning`; numerical sign tests | Useful paired response and original preservation on held-out voices |
| Anchors | Accepted originals zero; audited detuning reverse target | Training rows and masks reconstruct from cached files | Independently reviewed natural reference material |
| Current ExtraTrees | Additional engineering alternative | 96 trees / 31 own-audio features, optional experimental preview | Does not implement the original BPDR losses |
| Dataset | Permitted material, source-separated partitions, planned participants | Public VocalSet/Vocadito/Jamendo exploratory experiments | Planned recruitment was not completed by generating variants |
| Validation | Matched neural comparisons and held-out natural material | Historical synthetic comparisons; current calibration/pilot gates fail | Untouched singers/melodies, listening and confidence intervals |
| Permission / rendering | Bounded changes, preservation, rollback | Controller, Praat renderer, exact encoded-WAV re-audit | Threshold calibration and audible-quality validation |
| Downstream | Fixed extractor before/after comparison | Exact accepted WAV → GAME MIDI/CSV receipt | Reference-note precision/recall/F1 benefit |

**A model that excludes boundary metadata is invariant to it by construction. That alone does not prove learned boundary-pitch disentanglement.** The frame-only Transformer is therefore a necessary control, not a failed version of BPDR.

## Limitations and a realistic sequence of work

1. **Immediate:** keep this executed pack and local assets available; rehearse code search and audio; show failed criteria beside favorable demonstrations. Restore missing artifacts before dependent cells rather than substituting outputs.
2. **Near term:** predeclare a revised diagnosis of Transformer response using existing training-side evidence; broaden audited positive/negative shift coverage and compare an independent corruption renderer. Do not infer a cause from architectural comparisons where features and training changed together.
3. **Main study:** finish independently reviewed references and permitted participant collection; reserve untouched singers and melodies; run the existing full four-view, augmentation-only and matched frame-only implementations with equal budgets and selection. Historical code/training is complete; real-voice contribution validation is not.
4. Evaluate natural analyser errors, identity controls, no-change phrases and expressive transitions at matched acceptance coverage. Keep harmful edits, withheld cases, unscorable interventions and render failures visible.
5. Add blinded tune-preservation and rendering-quality listening, user recognition of free-hummed ideas, and fixed-extractor note precision/recall/F1 against independent annotations. Report singer-level uncertainty and training-seed variation separately.

Professional sustained vowels do not represent novice melodies, ordinary reading, overlap or accompaniment. Known detuning does not label private intention. Current improvement coverage is directional; estimator agreement is not truth. Calibration selection is not a final test. Exact-hash checks cannot exclude every near-duplicate recording or shared session. Numerical rendering passes cannot certify artifact-free listening. A single scoreable pilot singer does not support population confidence intervals. Full latency/memory profiling remains a deployment task; a single run time is descriptive only.

Use licensed or explicitly consented recordings, retain attribution and coded source IDs, and keep personal contact/consent records outside demo artifacts. No participant recruitment or ethics approval is claimed by this pack. Keep unmodified recordings and failed results. No automatic training, large download, external service or cloud charge occurs during the fast presentation route.


The primary literature comparison is in notebook 09, block LITERATURE; source identities and links are in panel_source_manifest.json.
