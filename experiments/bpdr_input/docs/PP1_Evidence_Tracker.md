# PP1 evidence and readiness tracker

**Original planning baseline:** 27 September 2026. **Current inspected snapshot:** 30 September 2026. **Owner:** Liyanage L. N. P., IT23141506.

Current repository location: `J26-DS-339/experiments/bpdr_input`. Saved [Phase 1](Phase1_Verification.md) and [Phase 2](Phase2_Verification.md) verification records report 28 passing tests and two executable notebooks. Phase 1 technical processing is verified. Phase 2 has audited sources, fixed singer partitions, 57 mapped crops from 38 eligible sources, pending review sheets and an exploratory four-view renderer pilot. Its six audited shifts yield 18 derived pairs from one singer, **not** 18 independent performances. On this snapshot all 114 review rows are pending, zero pairs are training eligible, and the P2 trustworthy-target gate remains open. General musical accuracy, learned controls, BPDR repair/comparison and team integration remain outstanding.

Use this register while executing [the phase plan](PP1_Phase_Execution_Plan.md). It records evidence and implementation state; it does not calculate university marks or claim present readiness.

## Status rules

- **Documented:** a design, requirement or claim has a source or draft artifact.
- **Scaffolded:** implementation structure exists, but behavior is not verified.
- **Implemented:** behavior executes; acceptance checks may remain.
- **Verified:** declared acceptance checks pass with saved evidence.
- **Integrated:** verified behavior works across the agreed component interface.
- **Demo-ready:** verified/integrated artifacts have passed a rehearsal on the demo machine.
- **Unverified:** no sufficient artifact was inspected to credit implementation.

Use separate fields for implementation state and research outcome. "Verified experiment execution; inconclusive effect" is a legitimate entry. "Model trained" does not mean "novelty confirmed."

## Evidence register

Add a concrete artifact path, version/run ID, date, result and reviewer to each row as work proceeds. The current-state column below supersedes the historical 27 September foundation baseline; it credits executed engineering without implying a research outcome or university mark.

| ID | Rubric area / points | Claim or deliverable | Current state at 30 September | Phase |
|---|---|---|---|---|
| E01 | LO1 problem / 3 | Proposal-faithful problem/gap and sourced related-work distinction. | Documented; notebook 02 illustrates same-waveform boundary corruption and separate detuning. No trained response difference yet. | P0, P2 |
| E02 | LO1 proof of concept / 7 | Executing BPDR mechanism, real paired examples and rendered output. | Exploratory pairs/rendering execute; BPDR model, repair controller and comparative learned proof of concept are unverified. | P4-P5 |
| E03 | LO2 pillars / 7.5 | Applied MIR, ML/data design, controlled experiments and evaluation. | Frozen MIR analysis, singer-separated data and controlled intervention engineering execute; learned/evaluation outcomes pending. | P1-P5 |
| E04 | LO2 technologies / 17.5 | Actual features, losses, model inference, controller/renderer and technical explanation. | Frozen features and intervention renderer execute; trained losses, model inference and repair renderer/controller pending. | P1-P5 |
| E05 | LO3 design / 8 | Modular flow, original preservation, statuses, rollback and accepted-only contract. | Original preservation, technical statuses and accepted-file validation execute; musical repair rollback and live teammate hand-off pending. | P0-P5 |
| E06 | LO3 completion / 12 | Accepted deliverables against baseline and actual corrective actions. | P1 and P2 engineering recorded against the dated baseline; review gate blocks research training; percentage unverified. | All |
| E07 | LO3 practices / 8 | Split/permission manifests, versions, reproducibility and meaningful tests. | Checksums, fixed singer splits, crop provenance, audit-policy history and 28 saved passing tests; permissions/review and model runs pending. | P0-P5 |
| E08 | LO3 requirements / 8 | FR1-FR7 linked to acceptance evidence. | Technical portions of FR1/FR2/FR6/FR7 execute; musical decision, repair and full user feedback remain unverified. | P0-P5 |
| E09 | LO3 risks / 4 | Logged risks and executed mitigation or concrete execution plan. | Annotation exclusions, strengthened review/input gate and v0.1-to-v0.3 audit history are documented; reviewer delay and A1/runtime boundary shift need action. | All |
| E10 | LO4 communication / 9 | Rehearsed structure, accurate explanations and evidence-based Q&A. | Narrative planned; rehearsal unverified. | P6 |
| E11 | LO4 presentation / 6 | Readable visuals, clear delivery and time management. | Planned; actual deck/delivery unverified. | P6 |
| E12 | LO5 potential / 10 | Supported achievable benefits, intended users, alternatives and feasible cost. | Proposal business assumptions documented; observed user/cost evidence unverified. | P0-P6 |

For each evidence item append:

`Artifact | Version/date | Acceptance criterion | Actual result | Independent denominator | Limitations | Remaining work | Reviewer`

## Functional acceptance register

### Phase 2 update (27 September 2026)

- E01: same-waveform boundary corruptions and a separately detuned waveform are visible in notebook 02. This illustrates the experiment, not a trained-model advantage.
- E03/E04/E07: annotation validation, immutable crop provenance, singer-separated partitions, renderer/audit policies, A/B/C/D identities, target/input separation and executable notebooks are verified.
- E09: annotation overruns are excluded; overlaps/uncertainty are masked; fine versus coarse pitch measurement is explicit; empty reviews and failed audits block training eligibility.
- W3: 40 sources audited; 38 yield 57 mapped crops. Six v0.2-audited shifts produce 18 exploratory pairs from one phrase/singer. Human reviews remain pending; training-ready count is zero.
- E02/W4/W7: no BPDR checkpoint, comparative learned result or general accuracy study exists yet. W6 live team integration remains open.
- Human review is pending by user instruction; see [the review guide](Reference_Review_Guide.md). No completion percentage is inferred.

### 30 September research-gate check

- The saved 57-crop manifest contains 30 training, 14 development, 9 PP1-evaluation and 4 final-holdout crops. These counts are **crops**, whereas independent singer counts are 14/6/6/3. The two annotation-bounds exclusions remain visible; do not turn repeated shifts or boundary variants into additional independent subjects.
- The current review sheet still has 114 pending rows. A reviewed-run directory with zero eligible examples is a valid empty-gate check, not an approved training collection. P3 architecture/tests can proceed on constructed fixtures, but research fitting must wait for independently reviewed and audited anchors.
- The four-view pilot uses A1 annotation-onset tracks; operational analysis emits a provisional audio-derived boundary score. Record these as different feature provenances and evaluate runtime-like boundaries separately before claiming deployed robustness.
- The original v0.1 renderer audit failed all six candidates; v0.2 relaxed only PESTO's coarse consistency limits while retaining the YIN fine-shift limits. The revision was made on exploratory training material and must accompany every later result description. Neither audit establishes general accuracy.
- A fresh 30 September v0.3 exploratory run added plateau/outside coverage and outside-drift-tail checks, regenerated A1 onset-plus-offset boundary tracks, and again yielded six audited shifts/18 paired views from one singer. The loader rechecks actual approvals and all input hashes. The updated repository source passes 32 tests; both notebooks execute with plots and playback. See [the exact run IDs and metrics](Phase2_Verification.md). These engineering checks do not close the two-reviewer gate or prove BPDR accuracy.
- The [consolidated P0 record](P0_Scope_Claims_and_Decisions.md) now makes the local scope, gap, draft interface, actual contract examples, resources and dated decisions explicit. Team ownership confirmation, dataset redistribution terms, repair/controller checkpoints, comparative evaluation, business evidence and presentation rehearsal remain open. No project completion percentage or university mark is inferred.

### Phase 1 update (27 September 2026)

- E03/E04: acoustic extraction executes with real frozen PESTO; learned BPDR losses/models remain unverified.
- E05/E07: original preservation, timing tolerance, actual exported-file checks, rejection propagation, locked dependencies and meaningful failure tests verified.
- E08: foundational FR1/FR2/FR6/FR7 behavior executes; see the requirement-by-requirement limits in `Phase1_Verification.md`.
- E09: stereo cancellation, malformed input, missing analyzer and unchecked downstream input have exercised mitigations.
- E02 remains unverified: Phase 1 does not establish the proposed training advantage.
- W5 now has a verified technical unchanged path and acoustic/report outputs; renderer/controller/repair remain outstanding. W4/W6/W7 retain their research/integration gates. No completion percentage is assigned.
- Human musical-fixture review and presentation-machine listening/rehearsal remain open.

| Proposal requirement | What must work | Acceptance evidence to save |
|---|---|---|
| FR1 | Short supported recording/upload with understandable instructions. | Valid upload/capture and supported-limit checks. |
| FR2 | Original retained; technical failure distinguished from re-record request. | Source checksum and exercised failure branches. |
| FR3 | Assess/preserve/prepare from measured evidence. | Reviewed usable fixture, diagnostics and justified outcome. |
| FR4 | Supported local edits preserve timing, rests and event order. | Interval log, time/sample checks and expressive no-change cases. |
| FR5 | Original/prepared playback, intervals and practical feedback. | Working comparison screen and observed task evidence if collected. |
| FR6 | Export only checked accepted audio and diagnostics. | Package validation, render rollback and downstream rejection test. |
| FR7 | Aligned measured-pitch plots and traceable operation logs. | Plots labelled by units/source/run and type of reference/prediction. |

## Completion register: preserve the proposal baseline

Proposal Table 11 allocates these planning hours. They are weights/estimates, not verified hours spent. Do not enter elapsed time or download counts as completed deliverables.

| WBS | Work package | Proposal weight | Inspected deliverables and open gate at 30 September | Current completion credit |
|---|---|---:|---|---|
| W1 | Literature and gap | 50 h | Proposal and sourced gap/related-work distinction exist; acceptance against the original W1 scope not signed off. | Not calculated. |
| W2 | Design, protocol and ethics | 50 h | Consolidated P0 scope/claims and PP1 protocol/contract documented; team acknowledgment, human review and approval status remain open. | Not calculated. |
| W3 | Data and interventions | 120 h | Forty sources read, 57 mapped crops, singer splits and one-singer audited exploratory pairs execute; human-reviewed targets and broader training corpus absent. | Not calculated. |
| W4 | Models and experiments | 140 h | No learned control or BPDR checkpoint/comparison verified. | Not calculated. |
| W5 | Prototype and preparation | 90 h | Technical input validation, frozen acoustics, report/playback and accepted-file checks execute; musical repair/controller and rollback absent. | Not calculated. |
| W6 | Integration | 50 h | Draft accepted-only contract and package checks execute; real cross-component extraction not verified. | Not calculated. |
| W7 | Evaluation and analysis | 100 h | Renderer and data-pipeline audits execute; independent BPDR research evaluation absent. | Not calculated. |
| W8 | Documentation and assessment | 40 h | Plans, verification records and two runnable notebooks exist; result-backed deck/rehearsal not verified. | Not calculated. |
| **Total** | | **640 h** | | **No unsupported completion percentage.** |

If the team chooses a weighted milestone indicator, first define and review acceptance weights within each WBS package. Then calculate `sum(package weight x accepted milestone fraction) / 640 x 100`. This is a team-defined planning indicator, not the rubric's official mark formula, actual labor or an automatic 50% certificate. Retain its baseline date and do not reweight completed items retroactively.

Track evidence coverage separately from project completion. Filling all E01-E12 rows does not itself establish 50% of research completed. The rubric's completion qualification and timing judgment still apply.

## Experiment record template

| Field | Value to enter from the actual run |
|---|---|
| Run ID / date / commit or version | |
| Source manifest / singer groups / crop map | |
| Train / development / evaluation split IDs | |
| Method / checkpoint / seed / parameter count | |
| Training and selection budget / hardware | |
| Intervention / renderer / identity-audit version | |
| Independent mask / region / metric configuration | |
| Controller version / raw versus delivered output | |
| Singer/source counts / exclusions / failures | |
| Observed results / uncertainty / limitations | |

## Final demonstration checklist

- [ ] The problem and gap can be explained from a real example.
- [ ] A/B share the same waveform; C/D share the same waveform.
- [ ] Model inputs exclude known intervention targets and reviewer reference labels.
- [ ] All three learned conditions have actual reloadable checkpoints and comparable budgets.
- [ ] The experiment includes repair and preservation, not boundary invariance alone.
- [ ] Renderer identity/shift audits and controller outcomes are traceable.
- [ ] Every status is exercised; non-accepted packages stop downstream processing.
- [ ] Result plots/table come from the declared independent collection with denominators/failures.
- [ ] Teammate outputs identify proposed model, baseline, heuristic or fixture provenance.
- [ ] Completion claims have evidence, baseline and limitations.
- [ ] Business/user evidence distinguishes observed findings from assumptions.
- [ ] Live demo and labelled offline replay have been rehearsed.
- [ ] Q&A and timing are practiced against the confirmed presentation slot.

Leave unchecked items visible until their acceptance evidence exists.
