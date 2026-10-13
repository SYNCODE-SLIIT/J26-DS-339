# BPDR current evidence and readiness

**Latest update: 13 October 2026; foundation snapshot: 12 October. Owner: Liyanage L. N. P., IT23141506.** This file records current implementation and saved evidence; it does not infer a completion percentage, marks or research accuracy. Dated verification documents remain historical records. Previous tracker/plans are in the verified recovery archive.

**13 October coverage:** Gate B has an executable [training-only coverage audit](Coverage_Audit_2026-10-13.md). Shorter geometries yield reference support across six singers, but a selected six-singer acoustic probe gives three singers with at least one fully audited shift and one with all six. Coarse PESTO -10-cent rejections need diagnosis; v0.3 stays active and research eligibility remains zero. The coverage-stage regression suite passed 51 tests.

**13 October preparation:** the [initial shared-builder pilot](Data_Preparation_Verification_2026-10-13.md) is historical. The [fresh five-artist sample](Jamendo_Five_Artist_Sample_2026-10-13.md) completed GPU separation and full-song GAME with five individual receipts. The current common run has ten unchanged technical outputs: three Vocadito candidates from S7/S10/S11 and seven Jamendo crops, six candidates from three artists. Eighteen review rows are pending; research-ready examples remain zero. Ten accepted packages, 34 source hashes, decode/reference/mask artifacts, both source snapshots and eighteen probe WAVs verify. Original Phase 2 reviews/pointer are unchanged. The full suite passed 68 tests; focused additions/guards bring the total to 72 unique checks. Musical-suitability calibration, J4 approvals and training pairs remain open.

**13 October intervention yield:** three Jamendo crops from two artists have v0.3 geometry. Five of eighteen tested shifts pass across two crops/two artists; no crop passes all six. Identity, sample count and exact outside samples pass for all tested variants. Fine YIN error and loss of common voiced audit coverage explain new failures; the earlier coarse PESTO issue is not the only bottleneck. Gate B remains open and v0.3 stays active.

## Current register

| Area | State | Evidence / independent denominator | Still required |
|---|---|---|---|
| Original/technical audio preservation | Implemented and regression-verified | `pipeline.py`, `contracts.py`; full replay of 40 sources, 21 unchanged and 19 duration stops | Musical suitability calibration |
| Frozen acoustics | Implemented; real PESTO smoke passes | Masked 70 features plus provisional boundary column; original timing/rest checks | Natural-boundary/runtime validation |
| Vocadito data | Audited local collection | 40 recordings, 29 singers; 57 mapped crops, two annotation exclusions | Reference approval and broader support |
| Human review | Pending | Original 114 rows unchanged; current preparation has 18 additional pending rows (12 Jamendo, 6 Vocadito) | Actual two-reviewer approvals and intervals; queues are not independent source counts |
| Research-ready material | Not available | Data summary: 0 training-ready examples; current paired summary: 0 eligible | Review AND intervention audits |
| Four-view interventions | Engineering execution verified | 18 audited exploratory pairs; 1 phrase, 1 singer | Revised multi-source geometry audit |
| Three learned conditions | Code implemented, fixture-verified | `model/losses/examples/metrics/training.py`; two 30-update BPDR CPU/XPU smoke records | Matched fits, dev selection, held-out comparison |
| Jamendo preparation | Five-artist execution/yield measured | 1,604 tracks, eight stems; seven new crops, six candidates from three artists, zero eligible pairs | Calibrated suitability, broader audited source support and reviewed anchors |
| GAME reference check (J2) | Pilot passed | 28 recordings, 20 training/development singers; onset+pitch F1 0.63 vs A1, 0.77 vs A2 | Owner agreement; no BPDR benefit inferred |
| Accepted-audio format (H1) | Technical pilot passed | 14 Vocadito crops and 3 Jamendo crops; unchanged technical preparation only | Edited-region-aware check after actual repair |
| Excerpt separation (J5) | Experimental/open | One window of one track | Multiple artists/windows before adopting |
| Musical repair/controller/rollback | Not implemented | No accepted repaired output was credited | Gate F of execution plan |
| Independent effect/listening | Not demonstrated | Training fit is explicitly own-data; no held-out BPDR advantage | Gate G and adequate independent sources |
| Live team integration | Not agreed/verified | Proposed package and baseline GAME adapter | Melody-owner acceptance and actual consumer |
| Regression after cleanup | **48 tests passed** | `runs/project_audit_2026-10-12/regression_tests.txt`; includes real frozen-pitch smoke | Execution proof, not research accuracy |

## Active artifact locations

- Current source/review run: `runs/phase2/15fb678d1f44406ab4da3fddd7c2f809` (`manifest.json`, `data_summary.json`, `reviews.csv`, `crops`).
- Current exploratory pairs: `runs/phase2/15fb678d1f44406ab4da3fddd7c2f809/exploratory/fda19aa33ede4aba8fa4b4087bb3afdc`; reviewed pointer remains a verified zero-example gate result.
- Five-artist batch: `runs/jamendo_sample/bf912553c00c40598b8f0d64ae266de0`; receipt/source preservation and final verification.
- Current common preparation/listening/reviews: `runs/preparation/d274ed728d3d46f3b2661ba5291e61da`; ten accepted technical outputs, eighteen pending rows, zero research-ready examples.
- Jamendo intervention yield: `runs/jamendo_yield/de2f85a01b194c4eab009aa95b14e496`; eighteen tested variants, five audit passes across two artists, no eligible training pairs.
- Latest executed notebooks: `runs/notebooks/bcfca999f47741b0a4c5f40c6dad98a7`, verified pointer; two notebooks, zero saved cell errors. They were preserved, not re-executed in this audit.
- Full technical replay: `runs/phase1_dataset/summary.json`; all 40 referenced run directories exist. Old example IDs in the dated Phase 1 note are historical; use this summary for current paths.
- J2: `runs/extraction_checks/0e9888a443b74de28b55a0150ab9735e/game_on_vocadito.json`.
- J3: `runs/extraction_checks/3fa67dcd34ee4245ba24de232bd8c3ec/stems_vs_vocadito.json` and `ed4b3433e8554f65bdc963dc5f250749/stems_vs_vocadito.json`.
- H1/crop check: `runs/extraction_checks/be8e05a22853497193474c22c8c6118e/handoff_and_crop_check.json`.
- J5: `runs/extraction_checks/d0a5668191734830bf22948b6bb2a3cd/excerpt_comparison.json`.
- Engineering checkpoints/records: `runs/engineering`; both are 30-update BPDR fits labelled `exploratory_not_research`, not research-ready models.
- Cleanup/source/hash verification: `runs/project_audit_2026-10-12` and `private/cleanup_2026-10-12`.

The older `a57b...` laboratory run, audit history and original source hashes are retained as historical evidence, including its pre-relocation absolute paths. Current launchers use the active `15fb...` run; do not silently rewrite historical hashed manifests.

## Quality gates to close

- [ ] Gate B: training-only geometry yields a justified independent-source collection, or report a limited pilot.
- [ ] Gate C/J3: common two-dataset builder and representative screened yield, with source/reference/license provenance.
- [ ] Gate D/J4: real independent reference approvals plus verified intervention support.
- [ ] Gate E: fair controls, development selection, separate no-change cases and frozen evaluation protocol.
- [ ] Gate F: permission, actual rendering, post-checks, rollback and four-status behavior.
- [ ] Gate G/H1: no unintended rest/timing/non-target damage; edited pitch matches the declared correction, with independent evaluation and consumer validation.
- [ ] Prototype: rehearse actual implemented behavior, show remaining gates and confirm official slot.

Use [the execution plan](PP1_Phase_Execution_Plan.md) to close these gates. Do not change a pending box because of downloads, more variants, fixtures or elapsed dates.
