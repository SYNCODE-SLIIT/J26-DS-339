# PP1 evidence and readiness tracker

**Snapshot baseline:** 27 September 2026. **Owner:** Liyanage L. N. P., IT23141506.

Current repository location: `J26-DS-339/experiments/bpdr_input`. Phase 0 now includes a standalone package, draft contracts and a read-only dataset-audit command. Five foundation tests pass; the audit reads all 40 recordings with no missing-reference or audio-read errors. These checks do not establish audio acceptance, annotation accuracy or repair performance.

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

Add a concrete artifact path, version/run ID, date, result and reviewer to each row as work proceeds. Initial entries below credit existing documents/inventory only, not a finished prototype.

| ID | Rubric area / points | Claim or deliverable | Initial state | Phase |
|---|---|---|---|---|
| E01 | LO1 problem / 3 | Proposal-faithful problem/gap and sourced related-work distinction. | Documented in proposal/master plan; live problem demonstration unverified. | P0, P2 |
| E02 | LO1 proof of concept / 7 | Executing BPDR mechanism, real paired examples and rendered output. | Unverified. | P4-P5 |
| E03 | LO2 pillars / 7.5 | Applied MIR, ML/data design, controlled experiments and evaluation. | Method documented; execution unverified. | P1-P5 |
| E04 | LO2 technologies / 17.5 | Actual features, losses, model inference, controller/renderer and technical explanation. | Proposed architecture documented; code/checkpoints unverified. | P1-P5 |
| E05 | LO3 design / 8 | Modular flow, original preservation, statuses, rollback and accepted-only contract. | Draft contract invariants verified; real audio, rollback and integrated paths unverified. | P0-P5 |
| E06 | LO3 completion / 12 | Accepted deliverables against baseline and actual corrective actions. | Original WBS and new PP1 schedule documented; percentage unverified. | All |
| E07 | LO3 practices / 8 | Split/permission manifests, versions, reproducibility and meaningful tests. | Audio readability, checksums, metadata/reference presence and foundation tests verified; label quality, splits and model runs unverified. | P0-P5 |
| E08 | LO3 requirements / 8 | FR1-FR7 linked to acceptance evidence. | Proposal requirements documented; functional acceptance unverified. | P0-P5 |
| E09 | LO3 risks / 4 | Logged risks and executed mitigation or concrete execution plan. | Risk/contingency plan documented; actions require evidence. | All |
| E10 | LO4 communication / 9 | Rehearsed structure, accurate explanations and evidence-based Q&A. | Narrative planned; rehearsal unverified. | P6 |
| E11 | LO4 presentation / 6 | Readable visuals, clear delivery and time management. | Planned; actual deck/delivery unverified. | P6 |
| E12 | LO5 potential / 10 | Supported achievable benefits, intended users, alternatives and feasible cost. | Proposal business assumptions documented; observed user/cost evidence unverified. | P0-P6 |

For each evidence item append:

`Artifact | Version/date | Acceptance criterion | Actual result | Independent denominator | Limitations | Remaining work | Reviewer`

## Functional acceptance register

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

| WBS | Work package | Proposal weight | Verified accepted deliverables | Current completion credit |
|---|---|---:|---|---|
| W1 | Literature and gap | 50 h | Source-backed proposal/related-work documents exist; verify acceptance against the original W1 scope. | Not calculated. |
| W2 | Design, protocol and ethics | 50 h | Design documents exist; protocol/approval status not verified. | Not calculated. |
| W3 | Data and interventions | 120 h | Dataset inventory/audit executes on 40 files; reference quality, partitions and interventions remain unverified. | Not calculated. |
| W4 | Models and experiments | 140 h | No BPDR checkpoints/results verified. | Not calculated. |
| W5 | Prototype and preparation | 90 h | Package and draft contract scaffold verified; working audio-validation/repair prototype remains unverified. | Not calculated. |
| W6 | Integration | 50 h | Draft accepted-only contract checks pass; real cross-component execution not verified. | Not calculated. |
| W7 | Evaluation and analysis | 100 h | No executed BPDR study/results verified. | Not calculated. |
| W8 | Documentation and assessment | 40 h | Proposal/master/phase documents exist; demonstration/rehearsal not verified. | Not calculated. |
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
