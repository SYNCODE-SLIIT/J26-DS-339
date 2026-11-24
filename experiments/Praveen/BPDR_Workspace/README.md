# Component 1 Workspace (`BPDR_Workspace`)

**Project:** `J26-DS-339` — *Genre-Aware Automatic Vocal Melody Harmonization Using Hierarchical Deep Learning*  
**Component 1:** Vocal Input Validation, Boundary-Robust Modeling, and Active Intent Confirmation  
**Researcher:** Liyanage L. N. P. (Praveen, `IT23141506`)  
**Updated:** November 24, 2026

---

## Start Here

- **Latest Findings, Claims Ledger & Key Benchmarks:** [deliverables/README.md](deliverables/README.md)
- **Active Redesign Brief (AIC v0.2, 22–23 Nov 2026):** [documentation/Component1_Intent_Redesign_Brief_2026-11-22.md](documentation/Component1_Intent_Redesign_Brief_2026-11-22.md)
- **Feasibility Review & R1/R2/v5 Verdicts (13–20 Nov 2026):** [documentation/BPDR_Feasibility_and_Component_Plan_2026-11-13.md](documentation/BPDR_Feasibility_and_Component_Plan_2026-11-13.md)

---

## Quick-Start Launchers

| Command | Application | Port |
| :--- | :--- | :---: |
| `.\Run-Component1.ps1` | **Active Intent Confirmation + BPDR Web Service** (Starlette/FastAPI + HTML/JS 8-screen flow & `/research` view) | `8505` |
| `.\Run-Component1-Panel.ps1` | **Component 1 Assessment Panel App** (Streamlit standalone evaluation & FastTNet inspector) | `8515` |
| `.\Run-Studio.ps1` | **Vocal Rescue Studio** (Streamlit development & listening interface) | `8504` |

---

## Repository Architecture

| Directory | Contents |
| :--- | :--- |
| `deliverables/` | Curated latest research reports, proposal PDF, Claims Ledger, and verified benchmark JSONs |
| `documentation/` | Active master plans, feasibility reviews, redesign brief, and official proposal PDF |
| `source/bpdr_input/` | Core Python package (`studio/`, `web/`, `application/`, `audio/`, `modeling/`, `repair/`, `preparation/`, `experiments/`, `reporting/`, `packaging/`) |
| `web/` | Static HTML/CSS/ES-module frontend (`index.html`, `app.js`, `audio-worklet.js`, `research.html`, `research.js`) |
| `results/` | Active model checkpoints (`results/studio/r1/`), AIC evaluation outputs (`results/studio/aic/`), and R2 control runs (`results/transformer_usability/`) |
| `tests/` | Active unit, contract, and integration tests (`test_studio.py`, `test_web_aic.py`, `test_evaluator_and_repair.py`, `test_analysis.py`, `test_workspace_layout.py`, etc.) |
| `tools/` | Active training, evaluation, benchmarking, and probing scripts |
| `datasets/` | Local audio datasets (`vocadito`, `humtrans`, `humtrans_r1`, `mir_1k`, `environmental_noise`, etc.) |
| `models_and_tools/` | Pinned acoustic/separator/GAME model artifacts |
| `notebooks/` | Canonical six-notebook research index (`01`–`06`) |
| `selected_model/` | Historical baseline checkpoint (`bpdr.pt`, `model.json`) |
| `archive/` | Non-destructive archive of historical October/early-November reports, experiment runs, legacy scripts, superseded tests, and temporary artifacts |

---

## Verification

Run the active Component 1 verification suite from `BPDR_Workspace`:

```powershell
..\.venv\Scripts\python.exe -X utf8 -B -m unittest tests/test_studio.py tests/test_web_aic.py tests/test_evaluator_and_repair.py tests/test_analysis.py tests/test_workspace_layout.py
```

---

## Data & Research Integrity Rules

- **Reserved Splits:** `HumTrans M03–M05`, `VALID`, and `TEST` remain strictly untouched until threshold freeze and supervisor authorization.
- **Claims Discipline:** All public claims in UI or reports must adhere to the Claims Ledger in [deliverables/README.md](deliverables/README.md).
