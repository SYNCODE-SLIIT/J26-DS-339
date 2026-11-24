# Component 1 Deliverables — Research Findings, Manuscripts & Benchmarks

**Project:** `J26-DS-339` — *Genre-Aware Automatic Vocal Melody Harmonization Using Hierarchical Deep Learning*  
**Component 1:** *Vocal Input Validation, Boundary-Robust Modeling, and Active Intent Confirmation*  
**Researcher:** Liyanage L. N. P. (`IT23141506`)  
**Updated:** November 24, 2026

---

## 1. Research Scope & Architectural Evolution

Component 1 prepares solo amateur vocal and hummed recordings before downstream melody extraction and harmonic chord planning. Following the falsifiable protocol in the research proposal (`IT23141506_J26_DS_339.pdf`), experimental work progressed through three phases:

1. **Hypothesis 1A (Blind Reference-Free Micro-Pitch Repair — Honest Negative Result):**  
   Controlled evaluations across four model families (Transformer, ExtraTrees, TCN, GRU) and a 12-fold weight-matched 3-arm control study (`Candidate R2`) demonstrated that blind reference-free repair of small ($\pm 10\text{–}30\text{ c}$) pitch deviations is **information-limited**. Because natural expressive singing sits $19\text{–}25\text{ c}$ from equal-tempered targets, a $25\text{ c}$ error is acoustically indistinguishable from intended expression without an external reference.
2. **Hypothesis 1B (`FastTNet` Boundary Detection & Do-No-Harm Preparation):**  
   While micro-pitch intent cannot be inferred blindly, note-transition boundaries (onsets, re-articulations, pitch steps, and voiced/unvoiced edges) carry verifiable physical cues. `FastTNet` (`studio_v1`, `boundary_transformer.pt`) paired with Do-No-Harm acoustic validation and segment-median note formation improves development Note F1 and maintains onset robustness under boundary timing jitter.
3. **Active Intent Confirmation (`AIC v0.2`):**  
   To supply the missing pitch/boundary reference at minimal user cost, Component 1 implements an 8-screen interactive web workflow (`web/`, `source/bpdr_input/web/`) that queries the singer on high-value boundary and pitch ambiguities, calibrates per-span uncertainty via leave-one-singer-out scaling, and executes review-gated, post-render-verified span repairs (`studio/span_repair.py`).

---

## 2. Summary of Validated Experimental Results

### 2.1 Candidate R2: Weight-Matched 3-Arm Control (`2026-11-17_r2_weight_matched_control/results.json`)
- **Setup:** 4-fold singer/group CV (17 groups), `1.5855x` post-warmup positive-anchor weight matching, **12/12 healthy fits**.
- **Outcome:** `advance_to_delivery = False`, `production_eligible = False` across all arms.

| Arm | Raw Positive MAE (c) | Raw Original False Correction (c, max group) | Corrupted Response Error (c) | Raw Delivery Gate |
| :--- | :---: | :---: | :---: | :---: |
| `augmentation_only` (`1.5855x`) | `19.917` | `1.153` (max `5.037`, `S29`) | `20.452` | **FAIL** |
| `frame_only` (`1.5855x`) | `19.234` | `1.193` (max `4.730`, `S29`) | `18.747` | **FAIL** |
| `bpdr` (`1.0x` + response terms) | `18.986` | `1.002` (max `4.732`, `S29`) | `18.997` | **FAIL** |

*(Uncorrected baseline: `24.906 c`.)*

### 2.2 Real-Audio Studio Vocadito Benchmark (`studio_vocadito_benchmark.json`)
Evaluated on 39 Vocadito training-role windows across 14 singers (1,000-resample singer-macro 95% bootstrap CIs):

| Method | Note F1 `[95% CI]` | Onset F1 `[95% CI]` | Notes |
| :--- | :---: | :---: | :--- |
| `pyin_4096` (shared baseline) | `0.301 [0.256, 0.352]` | `0.419 [0.360, 0.483]` | ~88% of raw difference vs PESTO is frame length (4096 vs 2048) |
| `pesto_notes_from_track` | `0.486 [0.431, 0.546]` | `0.617 [0.572, 0.664]` | Pitch track only |
| `segment_median_extractor_boundaries` | `0.602` | — | Note-rule ablation (`aic_dev_results.json`): `+0.115` from rule + heuristic edges |
| **`studio_v1` (`FastTNet`, default)** | **`0.631 [0.585, 0.676]`** | **`0.767 [0.732, 0.799]`** | `+0.145 [+0.087, +0.201]` vs PESTO; `+0.029 [-0.015, +0.077]` over heuristic edges |

Under boundary timing jitter (`e2e_fit_results.json`), `FastTNet` maintains Note F1 stability over extractor edges (`+0.050 [+0.018, +0.084]`).

### 2.3 Active Intent Confirmation (`aic_dev_evaluation_v0.2.json`, `downstream_harmony_impact.json`)
- **Calibration (Leave-One-Out):** Boundary ECE `0.195 -> 0.026`; Pitch ECE `0.543 -> 0.018`.
- **H-I3 Do-No-Harm (Untouched Originals, $n=53$):** `0.0%` edited under oracle listener; `19.6% [7.3, 33.1]` under a simulated 10%-flip noisy listener; `92.8% [83.0, 100.0]` under automatic repair of every candidate.
- **EIG vs. Lowest-Confidence:** Paired CIs include `0` on boundary Note F1 (`oracle K=3: -0.0095 [-0.024, +0.002]`; `noisy K=3: +0.0024 [-0.002, +0.009]`).
- **Downstream Chords (`R-3`, 39 windows):** No detectable chord-agreement difference (`pyin_4096: 0.479`, `studio_v1: +0.030 [-0.079, +0.151]`).

---

## 3. Scientific Claims Ledger

### May Claim (With Stated Limits)
- **Do-no-harm acceptance & located guidance:** `0` of `96` sung/hummed inputs falsely rejected; located clipping/silence/speech feedback. (Room-check thresholds `-55/-40 dB` remain untested on real rooms.)
- **Development Note F1 gain (`+0.145` vs PESTO track):** `studio_v1` `0.631` vs `0.486` on 39 Vocadito development windows (`~80%` attributable to the segment-median note rule + heuristic edges; learned model adds `+0.029 [-0.015, +0.077]` clean, `+0.050 [+0.018, +0.084]` under timing jitter).
- **Honest negative on blind micro-pitch repair:** Blind detection AUROC `0.62–0.64`; 4 model families tie at `~10%`; Candidate R2 (`12/12` healthy) fails delivery gates in all arms.
- **AIC v0.2 simulation:** Calibrated uncertainty and reduced harm vs automatic repair in simulation; human pilot requires ethics approval.

### Do Not Claim
- That the Transformer blindly repairs pitch, or that BPDR's custom loss terms beat `frame_only` / `augmentation_only` controls.
- `"+110%"` Note F1 gain (confounds 4096→2048 frame-length effect with learned boundary modeling).
- `"0% harmful distortion, guaranteed"`.
- That EIG question selection outperforms lowest-confidence selection.
- That `trust_curve` is calibrated (current values are heuristic) or that Component 1 improves live downstream chords before teammate adoption.

---

## 4. Active Model Checkpoints & Provenance

| Artifact | Path | Role |
| :--- | :--- | :--- |
| Primary Boundary Model (`v1`) | `../results/studio/r1/boundary_transformer.pt` | Frozen default Studio FastTNet checkpoint (`thr = 0.80`) |
| Singing-Delta Boundary Model (`v5`) | `../results/studio/r1/boundary_transformer_v5.pt` | `FastTNetV5` 16-channel delta candidate (HumTrans `dev_M02: 0.804`) |
| BPDR 4-View Repair Model | `../results/studio/r1/bpdr_repair_transformer.pt` | Reference-conditioned / multi-view repair research checkpoint |

---

## 5. Contents of `deliverables/`

### `deliverables/reports/`
- [Component1_Intent_Redesign_Brief_2026-11-22.md](../documentation/Component1_Intent_Redesign_Brief_2026-11-22.md) — Active AIC + BPDR redesign brief, Claims Ledger, and `R-1..R-5` evaluation results (22–23 Nov 2026; single source in `documentation/`)
- [BPDR_Feasibility_and_Component_Plan_2026-11-13.md](../documentation/BPDR_Feasibility_and_Component_Plan_2026-11-13.md) — Master feasibility review, R1/R2 verdicts, and FastTNet `v1–v5` benchmarks (single source in `documentation/`)
- *Research_Justification_and_Methodological_Pivot.md* — withdrawn draft, moved to `../archive/documentation/` because it contains inflated figures; do not cite.
- [FastTNet_Statistical_Analysis_Report.md](reports/FastTNet_Statistical_Analysis_Report.md) — Full bootstrap statistical report across `v1–v4` and Vocadito
- [IT23141506_J26_DS_339_Proposal.pdf](reports/IT23141506_J26_DS_339_Proposal.pdf) — Official Component 1 Research Proposal

### `deliverables/benchmarks/`
- [aic_dev_evaluation_v0.2.json](benchmarks/aic_dev_evaluation_v0.2.json)
- [downstream_harmony_impact.json](benchmarks/downstream_harmony_impact.json)
- [studio_vocadito_benchmark_clean.json](benchmarks/studio_vocadito_benchmark_clean.json)
- [studio_vocadito_benchmark_noisy.json](benchmarks/studio_vocadito_benchmark_noisy.json)
- [bpdr_multiview_repair_benchmark.json](benchmarks/bpdr_multiview_repair_benchmark.json)
- [r2_weight_matched_control_results.json](benchmarks/r2_weight_matched_control_results.json)
