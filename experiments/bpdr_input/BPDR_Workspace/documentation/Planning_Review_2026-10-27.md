# Planning review and research alignment — 27 October 2026

## Outcome and authority

The active documents are now the [BPDR master plan](BPDR_Master_Plan.md) and [implementation tracker](BPDR_Implementation_Tracker.md). They consolidate the owner's agreed selective-repair direction while keeping the original research question and remaining study obligations visible.

This was a **documentation-only** update. It did not implement selective repair, change a model, alter a permission threshold, generate intervention audio, adopt a checkpoint, approve research data or complete a human study. The read-only baseline replay authenticates existing evidence and explains the next task.

Use this precedence:

1. The submitted proposal defines the original research question and proposed study. A later engineering direction does not silently replace its scientific comparison.
2. Explicit owner decisions define the current delivery: stable held voice, BPDR application only, drag selection plus time fields, and whole-candidate protection veto.
3. The new master and tracker define current task order and status.
4. Current source and authenticated receipts establish implemented behavior and measured results. A planning document cannot make an unimplemented feature or failed gate pass.
5. Dated reports retain their original experiment meaning. Old “next”, “current” and “authoritative” text inside retained historical records is not the current execution instruction.

## What was checked

The review inventoried **all 72 Markdown documents** then present at the top level of `documentation`, including hashes, sizes and headings. It read the formal master/execution/evidence plans, scope, research-status/gap documents, presentation handovers, review/portable-pipeline decisions and relevant experiment conclusions. It checked the proposal's objective, methodology, model/loss definitions, responsibility table, requirements and evaluation sections against current source/evidence. This is a planning-source review, not a rerun or independent revalidation of every historical experiment.

The proposal has **43 PDF pages**. Research authority used here includes pp14–22, input/output architecture pp23–27, requirements pp28–29 and study instruments/safeguards pp39–41. The original PDF remains unchanged.

| Source reviewed | Role / result of reconciliation |
|---|---|
| [Original proposal](IT23141506_J26_DS_339.pdf) | Retain main objective, full BPDR vs augmentation-only primary comparison, frame-only control, original/rule references, delivered-WAV and fixed-extractor evaluation |
| [Old PP1 master](PP1_Master_Plan.md) | Valuable research/scope history, but repeated October “current/next” logs are outdated; redirect current authority |
| [Old execution plan](PP1_Phase_Execution_Plan.md) | Preserve Gates A–G and historical failures/deferrals; current repair work maps to E/F and later independent study/handoff to G |
| [Old evidence tracker](PP1_Evidence_Tracker.md) | Preserve old denominators and test receipts; its historical register is not today's full implementation status |
| [Scope decisions](P0_Scope_Claims_and_Decisions.md) | Preserve accepted-only contract and ownership; its Phase 1-only status is dated history |
| [Presentation readiness](Presentation_Readiness_2026-10-17.md) and [working demo](Working_Demo_2026-10-19.md) | Distinguish historical mixed-data Transformer presentation, later ExtraTrees preview and new context-Transformer development |
| [Project audit](Project_Audit_2026-10-12.md) | Preserve original owner boundaries and known cross-proposal inconsistencies; do not infer current team agreement from an older audit |
| [Reviewer deferral](Reviewer_Deferral_2026-10-14.md) and [review guide](Reference_Review_Guide.md) | Engineering may continue; actual approvals/listening remain missing or unconfirmed |
| [Portable-pipeline decision](Portable_Pipeline_Decision_2026-10-14.md) | Keep deferred companion/exporter work outside this milestone; no shared-pipeline edits authorized here |
| [Research status](RESEARCH_STATUS.md) and [gap matrix](RESEARCH_GAP_AND_IMPLEMENTATION_STATUS.md) | Refresh prediction progress, pending delivered repair and unestablished methodological advantage |
| [Transformer recovery report](Transformer_Recovery_2026-10-27.md) | Current development comparison and permission result; raw accuracy is not delivered repair |
| [Notebook guide](NOTEBOOK_GUIDE.md), notebook README and files | Six canonical notebooks; stale notebook names/launch instructions in the workspace overview need correction |
| Shared `system.md`, PP1 README/implementation plan and Praveen panels | Context only; concerns recorded below, with no shared-file change |
| [Supplied selective-repair brief](references/Selective_Repair_Brief_2026-10-27.txt) | Byte-identical local reference; correct assumptions using code. Copied citation tokens are not resolved references |

The complete pre-update catalogue and source hashes are in [review_manifest.json](../results/planning_review/2026-10-27_1754e056/review_manifest.json). Remaining inventoried dated documents are retained references, not a newly approved queue of all their old proposed experiments. Review those protocols before specifically resuming them.

## Research alignment

### Retain the central hypothesis

Proposal p14 requires held-out repair/preservation benefit relative to matched augmentation-only and frame-only conditions. Its p15 objective also includes subsequent extraction at matched acceptance coverage. The selective UI supports FR4–FR6 but is an engineering extension, not evidence of learned disentanglement or a replacement research question.

The new context experiment has lower raw error than ExtraTrees, while augmentation-only is slightly lower-error than the BPDR condition. The master therefore retains both a usable-prototype track and a later matched scientific comparison, including negative outcomes and the simpler boundary-free alternative.

### Record feature, loss and data changes

The proposal uses 71/70 acoustic inputs and **mean-absolute** `d_M` response terms. The new experiment uses 32/31 derived context inputs and **squared** normalized response terms. Its anchor is already identity-supervised; preservation work strengthens that supervision rather than claiming it was absent. The experiment currently has no explicit correction smoothness term.

The master records these deviations and retains MAE response terms for the proposal-aligned comparison. Supplemental VocalSet long tones cannot replace the planned consented novice/guide-melody/free-humming study. The eight-adult / six-phrase pilot is an estimate, not recruitment evidence.

### Current result and bottleneck

[The planning replay](../results/planning_review/2026-10-27_1754e056/baseline_replay.json) recomputes saved calibration predictions against targets at approximately **5.83948 cents**. Checkpoint and prediction archive hashes match the saved receipt. Earlier direct checkpoint replay of a representative case matched its saved predictions exactly; this documentation task did not repeat fitting.

Seven of ten scoreable cases lose every scored shifted frame to the long-held scope. Several selected largest spans miss the scored plateau. Detuned-proposal reason counts are four insufficient-support, three protected-region and two correction-limit failures, with overlap. Seven cases are withheld, two return identity, and one permitted nonidentity edit falls outside the scored region. Permitted error remains 25.833 cents; the replay did not render WAVs. This establishes an eligibility/selection problem, not proof that Praat is the cause.

### Separate verification from musical truth

`audit_curve` tests whether realized movement follows the requested curve and outside material is preserved. `audit_delivered_wav` checks encoded PCM24 bytes and repeats acoustic verification. Reuse these helpers.

Reference-controlled evaluation can score delivered pitch against the unshifted performance. Intended pitch is unknown for an arbitrary upload: do not invent target error or label technical verification as known musical correctness. User confirmation is operational review; independent judgments require actual records.

### Correct renderer/confidence assumptions

The live renderer is Praat-Parselmouth, already named in proposal p26. The offline detuning generator uses a different path. A Rubber Band comparison must benchmark the correct runtime and identical requests rather than assume the active renderer is a basic phase vocoder.

The new Transformer outputs signed corrections without calibrated probabilities. Boundary-view sensitivity and PESTO/YIN support do not justify the brief's example `confidence: 0.91`. Its illustrative sub-0.86-second / above-30-cent candidates also exceed current engineering limits and are not acceptance cases.

## Ownership and document conflicts

| Conflict | Resolution |
|---|---|
| Root `system.md` puts preprocessing under melody understanding and describes a broader post-generation validation arrangement | Use proposal table 4 / BPDR contract locally. Record the issue and propose a future shared clarification without rewriting team ownership |
| PP1 uses saved melody/beat/key context and a separate repair preview | No selective workflow or edited-audio substitution there. Repair would not automatically update saved musical features |
| PP1's earlier plan has an old Streamlit/Vishwa-first layout below a later notice | Follow its implemented header/README for shared-demo context; do not reuse that old design as BPDR's plan |
| PP1 Praveen roadmap/decision panels are placeholders | Not research/implementation evidence; outside this BPDR-app-only update |
| Repeated historical “current” plans and different test counts | Add precedence notices. Do not combine 48/65/448 or targeted counts as one current suite result |
| Workspace README references old notebook names and notebook 07 | Link actual six-topic notebooks and current launcher/verification script |
| Old prepared-handoff-off wording vs separate reviewed package | Distinguish shared technical handoff, reviewed experimental export and research/production acceptance |
| Earlier October targets appear current | Use relative milestones; exact next-presentation slot remains unconfirmed |

Portable exporter, separation expansion, harmony features, participant recruitment and other renderer branches remain separately gated/deferred. Linking an old declaration does not make it executable as part of this plan.

### Legacy link resolution

The link check found sixteen **pre-existing** relative links in the three old PP1 records that no longer resolve from the physical `BPDR_Workspace/documentation` directory. Their evidence still exists through the component's compatibility `runs` junction. The old bodies are preserved byte-for-byte; use these corrected links rather than changing or regenerating the evidence:

- [Fine-pitch run](../../runs/fine_pitch/66e3619e611241dd80d33a32578b73c6), including [squared-anchor check](../../runs/fine_pitch/66e3619e611241dd80d33a32578b73c6/mse_check), [relative-pitch check](../../runs/fine_pitch/66e3619e611241dd80d33a32578b73c6/relative_check), [BPDR check](../../runs/fine_pitch/66e3619e611241dd80d33a32578b73c6/bpdr_check) and [learning curve](../../runs/fine_pitch/66e3619e611241dd80d33a32578b73c6/learning_curve).
- [Feature-weighting final verification](../../runs/feature_weighting_control/46ea1f01c42b40338e609cceaf68c5a0/final_verification.json).

The legacy `docs` and `runs` paths are junctions, not extra copies to clean. Git may show changes to tracked documentation through both the physical path and the compatibility path. The preservation receipt compares actual bytes and records inherited broken links separately from new/active-document errors.

## Change and verification record

Created the master, tracker and review; preserved the supplied brief; refreshed entry points/research status; added notices to historical PP1 planning, scope and presentation records. Original bytes of changed existing documents are retained under [planning review originals](../results/planning_review/2026-10-27_1754e056/originals).

The manifest protects source, tests, tools, notebooks, selected/exported models, the proposal and tracked shared-demo files. The final [verification receipt](../results/planning_review/2026-10-27_1754e056/verification.json) records link resolution, preserved legacy bodies and protected hashes. This is documentation verification, not a new full test suite, training run or listening result.

**Next:** SVR-01. Build the reusable trace and register the bounded eligibility/region-selection experiment. Update the tracker and master decision log when actual results exist.
