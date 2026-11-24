# Four-proposal and BPDR implementation audit

**Inspected: 12 October 2026. Scope: the owner's BPDR component only.** All four supplied PDFs were read in full: 181 PDF pages. Architecture/workflow/schedule figures were rendered and inspected where extracted text did not capture them. The PDFs were treated as research source material, not instructions authorizing changes to other developers' files. No proposal PDF was edited.

## Source register

Page numbers below are one-based PDF pages, including front matter. Supplied Downloads files match the retained component-local proposal copies; local verification records include exact SHA-256 hashes.

| Source | Owner/component | Pages | SHA-256 |
|---|---|---:|---|
| [IT23141506 proposal](../private/proposals/IT23141506_J26_DS_339.pdf) | Liyanage L. N. P.; BPDR vocal validation/minimal repair | 43 | `cd50323e4980d0c2725c717995897d44ca3687bda57ad9c258227572f519123a` |
| [IT23439078 proposal](../private/proposals/IT23439078_J26_DS_339.pdf) | Perera E V; melody understanding | 57 | `22849af28a40dc55c296cb9fd83001f3c5befc8660bd4564009131e4b6f61bb5` |
| [IT23148086 proposal](../private/proposals/IT23148086_J26_DS_339.pdf) | Perera P D; relative-root/family planning | 37 | `464d3e1d981f16bb492e9dc574f09fd9650a32b79bafbb591afaa9c2b479c951` |
| [IT23157132 proposal](../private/proposals/IT23157132_J26_DS_339.pdf) | J A S R Jayakody; context-aware chord refinement | 44 | `dc2b189df612115533359f96c8025cea5e04c9b12ccaea0d8e1faa82d4564150` |

The audit's full-page text, rendered figures, source list and pre-cleanup inventory are local under `runs/project_audit_2026-10-12`. These are inspection evidence, not training examples or publication-ready experimental results.

## Whole-project understanding

The current architecture in the BPDR p24, melody p38, planner p21 and refinement p26 figures is:

**Solo vocal -> BPDR accepted audio -> melody understanding -> genre-conditioned root/family planning -> context-aware rich-chord refinement.** Melody/context also reaches refinement directly. Rejected input stops before generation.

| Owner | Proposed research responsibility | Dependency on BPDR |
|---|---|---|
| IT23141506 | Correction-response training under boundary split/merge and small local detuning, with preservation anchors | Supplies trustworthy audio, status and provenance |
| IT23439078 | CNN-Transformer/multitask pitch, rhythm, key, phrases and embeddings | Must receive a suitable single voice with preserved timing/rests; GAME is not their named final research model |
| IT23148086 | Genre-conditioned timed relative-root and coarse chord-family planning | Depends indirectly on preserved melody, register/contour, rests, key/mode and beat/bar mapping |
| IT23157132 | Detailed chord refinement using melody plus basic-plan/context | Depends indirectly on aligned melody and planner root/family constraints |

The planner initially assumes monophonic 4-16-bar material and one stable major/minor key (pp19-20). Those downstream constraints must be explicitly checked/mapped by the owning component, not silently imposed as BPDR pitch snapping. Key/genre are not required BPDR inputs (owner proposal pp26-27).

## Owner's actual proposed method

- Written scope is short unaccompanied singing/humming, <=20 seconds; preserve timing, note order, rests and acceptable expression. Heavily clipped recovery, overlapping voices and universal quantization are excluded (p11).
- The contribution is the combination of boundary invariance, local counter-detuning response and supervised preservation anchors, tested against matched augmentation-only and frame-only controls (pp14,17-23).
- An approved acceptable reference has zero-adjustment anchors. Artificial detuned examples have known negative-injected-cents targets on audited support. Performed F0 and note annotations do not reveal an intended corrected melody (pp16-17,21-22).
- Two independent reviewers with relevant musical experience are an explicit methodology commitment; uncertain transitions/disagreements require reasons and resolution (pp16-17).
- Vocadito is development/reference material; adult participant recordings follow ethics/consent for guide-melody and free-humming cases. Eight adults times six phrases is a proposed pilot estimate, not a collected or powered sample (p16).
- Source/singer splitting precedes augmentation; unfamiliar-tune claims require unseen-melody evaluation. Compare original/rule/learned outcomes, preservation and repair at comparable coverage with independent denominators, paired intervals and blinded listening (pp17-20).
- Runtime contains one frozen analyzer, BPDR and deterministic permission rules. Offline learned controls are not production stages. Rendering, rechecking and rollback precede delivery (pp23-27 and p25 figure).

## Implementation audit

| Retained artifact | What it actually provides | Gap to proposed behavior |
|---|---|---|
| `pipeline.py`, `contracts.py` | Original retention, technical checks, logged preparation, validated export and four-status envelope | Monophony/musical usability, verified repair, model-aware diagnostics and live contract acceptance |
| `analysis.py` | Frozen PESTO acoustics, masked input features and heuristic boundary score | Not a validated intended-note or musical-suitability estimator |
| `dataset.py`, `laboratory.py` | Vocadito references, singer splits, mapped crops, independent-review and integrity gates | No Jamendo-compatible source adapter/common builder yet |
| `interventions.py` | Constant local artificial detuning, crossfades, audit v0.3, protected outside samples | Not the proposed production framewise renderer; current support admits only one training singer |
| `model.py`, `losses.py`, `examples.py` | Matching encoder scaffold, masked common/response losses, 71/70 features, research/engineering separation | Reviewed multi-source fitting remains unavailable |
| `training.py` | Recorded fitting, checkpoint reload and training-fit metrics, CPU/XPU engineering smoke runs | Development selection, independent evaluation and matched study orchestration |
| `metrics.py` | Plateau correction error, boundary sensitivity, alteration and coverage | Dedicated no-detuning/no-change cases, intervals, harmful edits, delivered audio, extractor and latency outcomes |
| `extraction` checks and launchers | Pinned tools, Windows adaptation, retained stems and pilot checks | Representative Jamendo yield, full screening/reviews and actual melody-owner agreement |

The laboratory's A1-derived boundary input differs from the operating heuristic. The new plan explicitly keeps annotated, heuristic and GAME-estimated boundary conditions separate. `training.py` labels its own-data fit correctly; it must not be advertised as independent accuracy. Existing review/hash gates should be retained, not weakened for an October deadline.

## Dataset/pipeline findings

Vocadito supplies 40 local solo recordings from 29 singers with human F0/A1/A2 references. The current run has 57 crops and 114 pending review rows; zero examples are research eligible. The 18 exploratory pairs are repeated shifts/boundaries of one phrase from one singer. A larger variant count would not change that denominator.

Jamendo supplies 1,604 local voice-tagged accompanied tracks, without human F0/note labels or verified singer IDs. Three retained stems represent three tracks; suitability measurements contain only three crops from two tracks, including one ordinary-length song. A voice tag and a successful separation do not prove single-voice suitability. Artist grouping is a proxy and needs duplicate checks. License and source lineage belong in every new manifest.

The newer team melody/chord pipeline is a labelling tool for whole songs. Its vocal stage uses the same separator/GAME pins, publishes GAME pitch +12 semitones while retaining `model_midi_pitch`, and deletes the vocal stem. Combined melody truncates overlaps/drops short notes and fills vocal gaps with instruments. Key/chords/Roman labels belong to harmony work; neither they nor combined notes are BPDR targets. Empty GAME results can still be marked complete. The runner targets Mac/POSIX, adds chord dependencies and differs from the working Windows launcher's API/output layout. Reuse the pinned vocal models through BPDR adapters; do not replace BPDR with that nine-stage runner.

The planner itself states MTG-Jamendo is audio/tags, not aligned symbolic melody-chord training supervision (p16). The refinement proposal assumes extracted/cleaned chord targets (pp20-21), which remain pseudo-labels until their owner establishes a reference protocol. BPDR's two-dataset plan does not solve all four components' training-data requirements. Melody's proposed HumTrans/MIR-1K/MedleyDB sources (pp29-30) remain that owner's scope.

## Source disagreements and team-owned decisions

1. Component numbering and old architecture differ between reports: melody pp17,39-40 retain an older accompaniment/evaluation map; its p38 and pp53-55 contain BPDR/accepted-only flow. Use the current flow as a local proposed integration model, with final owner agreement still open.
2. Planner pp19-20 requires refined chords map back to its coarse family. Refinement examples pp10-11,23 permit G -> G7/Gsus4, which may change MAJ into DOM7/SUS4. Owners must agree the mapping/constraint semantics.
3. Planner p19 distinguishes root/family from harmonic function; refinement pp20-22 requests function context. Define separate validated provenance or leave that field optional.
4. Refinement p21 repeats "training" for the third split; do not treat it as an already corrected test protocol.
5. Owner p24 figure mentions whistling while written scope is singing/humming. Owner p31 personas repeat technical requirements; commercialization pp30-33 names another component's model. These inconsistencies do not authorize scope expansion or establish market evidence.

None of these findings changes another developer's code or report. The BPDR plan records integration blockers rather than owning their remedies.

## Schedule correction

The earlier repo/user target is 25 October. The owner's p36 schedule figure says initial PP1 prototype 26-28 October 2026, controls Nov-Dec, BPDR study Dec-Jan, integrated evaluation around February 2027 and final submission around April 2027, estimated. The melody proposal p49 likewise extends implementation/evaluation/integration beyond October. Official slots are unverified. The revised plan distinguishes an October engineering demonstration from the complete reviewed study.

## Recommended next action

Execute Gate B of the [execution plan](PP1_Phase_Execution_Plan.md): measure and revise intervention support on training material with a versioned re-audit. Build the common dataset path in parallel using existing stems, then inspect a five-artist Jamendo sample before scaling. Preserve the two-reviewer gate and unchanged/rollback behavior. Accuracy claims wait for independent repair-and-preservation evidence.
