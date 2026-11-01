# Transformer improvement and accuracy verification

27 October 2026. Completed MOD-01/MOD-02 and the legacy-policy trace in SVR-01.

**Result: preservation improved, but a usable Transformer has not been demonstrated.**
The complete declared experiment and an independent invocation of the accuracy
command finished. No cross-validation objective passed every singer's prediction
gate. The selected checkpoint is diagnostic only; the working ExtraTrees demo
and its selection artifacts were preserved.

## What changed

The new training path compares five objectives: the existing context-model
baseline, stronger original/outside preservation, adjacent-frame continuity,
an over-30-cent penalty, and their combination. It retains the same model,
paired sampling, 300 updates and seeds. Preservation adds original-only samples;
continuity adds adjacent windows, so compute and supervision budgets differ.
This is an engineering objective comparison, not the matched proposal study.

The new accuracy command verifies fitting lineage, rejects overlapping singers
or original source hashes, and writes a separate report for every invocation.
It measures correction MAE/RMSE, tolerance percentages, original/outside requests,
response gain, permission outcomes and actual encoded-WAV benefit. All
construction exclusions and withheld cases remain visible. References enter
training/scoring only; callable predictions use one recording's context.

## Training-side results

Four grouped folds and three seeds produced **60 cross-validation fits**.
There were 64 attempted interventions, 43 scoreable, from 16 attempted source
recordings. Scoreable material represents **seven singers and 13 recordings**.
One wholly unscoreable singer remains in the denominator. Repeating seeds gives
192 case evaluations / 129 scoreable evaluations per objective; it does not
create more independent recordings or singers.

Metrics average cases within seed/singer, seeds within singer, then singers.
All numbers below are predictions in cents, not applied audio movement.

| Objective | Correction MAE ↓ | Within 10 cents ↑ | Original request ↓ | Outside request ↓ | All-singer prediction gate |
|---|---:|---:|---:|---:|---|
| Baseline | 9.857 | 62.64% | 5.312 | 6.026 | Fail |
| Preservation | 10.905 | 55.91% | 3.675 | 4.272 | Fail |
| Continuity | 9.789 | 63.43% | 5.326 | 5.992 | Fail |
| Correction limit | 9.817 | 62.99% | 5.254 | 5.965 | Fail |
| Combined | 10.568 | 57.59% | 3.688 | 4.231 | Fail |

Continuity has the lowest correction MAE, only about 0.7% below baseline, while
original requests remain high. Combined reduces original requests by **30.6%**
and outside requests by **29.8%**, but correction error increases **7.2%**.
Its predeclared weighted development score falls from 7.010 to 6.178 cents
(11.9%); that tradeoff selected it **for diagnosis only**. It does not establish
an overall accuracy win. The gate requires original mean <=3 cents, improvement
over unchanged correction, and response gain >=0.25 for every scoreable singer.

The selected objective was refitted on all training data for all three seeds.
All checkpoints were locked before calibration features were loaded. Seed 339
was declared as the default in advance; no calibration seed was selected after
seeing its result.

## Separate calibration and actual audio

These are the already inspected development singers, not an untouched final
test: 16 attempted interventions, ten scoreable, two scoreable singers and three
scoreable original recordings (four source recordings attempted).

| Final seed | Raw MAE | Within 10 cents | Original request | Prediction gate | Actual-WAV evaluation |
|---|---:|---:|---:|---|---|
| 339, fixed default | 6.670 | 79.87% | 3.020 | Fail | Completed |
| 340 | 6.823 | 69.91% | 2.224 | Pass on these two singers only | Not requested |
| 341 | 7.026 | 73.74% | 3.192 | Fail | Not requested |

For default seed 339 the unchanged controller allowed four unique render
attempts across original and detuned inputs. Three unique encoded candidates
passed technical audits: **two unchanged-original controls were altered and
one detuned candidate was edited outside the scored plateau**. The remaining
detuned attempt rolled back because one coarse-tracker frame opposed the
requested direction (`COARSE_DIRECTION`). This is retained as a failure even
though its fine-tracker audit was within tolerance.

On the ten scoreable detuned cases, outcomes were six withheld, two identity,
one rollback and one verified technical candidate with no scored-region gain.
**Zero of ten scoreable cases and zero of sixteen attempts improved.**
Actual measured plateau error remained **25.975 -> 25.975 cents**. This uses
re-extracted pitch from delivered audio; the known-target unchanged-correction
baseline is a different measurement, 25.833 cents. Seven scoreable cases still
had no scope-eligible scored frames. Technical encoding success is therefore
not evidence of useful pitch repair or original preservation.

The prior 5.839-cent calibration result belongs to the earlier checkpoint and
experiment. Do not compare it to the new 9.857-cent cross-validation baseline
as though they were the same evaluation protocol. The broader validation here
reveals singer variation that a two-singer calibration summary cannot resolve.

## How to check accuracy after training

From `BPDR_Workspace`:

```powershell
& ..\.venv\Scripts\python.exe -X utf8 -B tools\check_transformer_accuracy.py --latest --render
```

This checks the fixed default checkpoint from the latest **completed**
improvement experiment. It does not refit, activate a model or choose a better
calibration seed. For another audited, independent-in-fit manifest use explicit
`--checkpoint` and `--pairs-manifest`; see the
[accuracy workflow](Transformer_Accuracy_Workflow.md).

The standalone rendered recheck reproduced all summary measurements and saved
predictions from the internal seed-339 evaluation. Its separate report is
[here](../results/transformer_improvement/accuracy_checks/c0ae34adf10c487aae080e84a852fa32/index.html).
An earlier command check also reproduced a training-side held-out fold; an
intentional overlapping-manifest check was rejected. Twenty-four targeted
tests passed for losses, masks, grouping, overlap rejection, renderer/delivery
rollback, evidence denominators and technical resumption. This is not a new
full-suite or notebook-suite claim.

## Evidence and recovery

- [Full comparison report](../results/transformer_improvement/d6e8b933b23546c78a53aa2292c03c1a/index.html)
- [Prospective declaration](../results/transformer_improvement/d6e8b933b23546c78a53aa2292c03c1a/declaration.json)
- [Locked selection](../results/transformer_improvement/d6e8b933b23546c78a53aa2292c03c1a/locked_selection.json)
- [Completed-run evidence](../results/transformer_improvement/d6e8b933b23546c78a53aa2292c03c1a/evidence.json)
- [Default calibration report](../results/transformer_improvement/d6e8b933b23546c78a53aa2292c03c1a/calibration/seed_339/index.html)
- [Final verification receipt](../results/transformer_improvement/d6e8b933b23546c78a53aa2292c03c1a/technical_execution/final_verification.json)

The sequential job was stopped after 26 completed fits and resumed with three
isolated workers. A first resume launcher had a run-allocation error; its
snapshot and failed launch receipt are retained. All remaining fits were saved
under the original declaration. The corrected launcher reused hash-verified
fits, then ran the original selection/refit/calibration path. No outcome,
threshold, source or fit configuration was changed. Cached-loop timing is not
training time; per-fit receipts record fit durations. Historical incomplete
folders remain history and are not promoted as completed experiments.

## Decision and next work

Do not replace the current demo with this candidate. Model/policy/source
preservation checks passed, but useful delivered repair and consistent raw
original preservation did not. Tolerance percentages measure controlled
correction frames, not general singing correctness; independent listening and
natural intended-note accuracy remain unassessed.

Continue **SVR-02**: audit prior use, reserve fresh final sources, and declare a
broader training/development protocol before additional tuning. The next model
study should address the preservation/response tradeoff with more independent
source coverage and retain the baseline and continuity alternative. Keep its
held-out selection separate from final testing. In parallel, SVR-03 can compare
the frozen checkpoint under versioned eligibility/multiple-candidate selection,
with unchanged correction limits and preservation controls. These are separate
model and controller questions; neither can be resolved by displaying a higher
raw tolerance percentage. Selective UI and the proposal-aligned study remain
open tasks in the [master plan](BPDR_Master_Plan.md).
