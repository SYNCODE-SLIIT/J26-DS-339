# BPDR vocal input component

Owner: Liyanage L. N. P., IT23141506. Updated 12 October 2026.

BPDR means Boundary-Pitch Disentangled Repair. This component checks a short solo singing/humming recording, preserves usable input, and is intended to make only small justified pitch repairs before melody understanding. All component work stays here in `experiments/bpdr_input`.

## Start here

1. Read the [current master plan](docs/PP1_Master_Plan.md) for scope, dataset roles and output-quality requirements.
2. Follow the [execution plan](docs/PP1_Phase_Execution_Plan.md) in order. Its gates define when work may advance.
3. Check the [evidence tracker](docs/PP1_Evidence_Tracker.md) before claiming a feature or research result.
4. The [four-proposal audit](docs/Project_Audit_2026-10-12.md) explains the whole project and source disagreements. The [cleanup record](docs/Cleanup_2026-10-12.md) records what was removed and how to recover it.

## What works today

- Original retention, technical checks, frozen PESTO analysis and checked mono 22,050 Hz PCM24 WAV export. Supported input is WAV/FLAC, 0.25-20 seconds. Duration, time origin and internal rests are preserved.
- Vocadito inventory, singer partitions, 57 mapped crops, review sheets, audited artificial detuning and four-view boundary experiments.
- Model/loss/metric/training code for BPDR, augmentation-only and frame-only conditions. Existing 30-update CPU/XPU runs are engineering smoke runs on one singer's repeated examples.
- Component-local Jamendo mirror and pinned extraction tools, three saved vocal stems, and pilot GAME/audio-format checks.

**What remains:** musical-usability validation, two-reviewer approved training anchors, broader intervention coverage, the common two-dataset builder, independent model selection/evaluation, repair controller/renderer/rollback and live team integration. The current saved data summary has **zero training-ready examples**. Technical acceptance and training fit are not repair accuracy.

## Folder guide

| Folder | Purpose |
|---|---|
| `src/bpdr_input`, `tests` | Maintained implementation and regression checks |
| `docs` | Current plans, review instructions and dated verification records |
| `notebooks` | Two maintained learning/inspection notebooks |
| `extraction` | Platform adapter and saved-check reproduction tools |
| `data` | Ignored Vocadito, Jamendo metadata/audio/mirror and retained vocal stems |
| `checkpoints`, `.venv`, `.tools`, `.jupyter` | Local model/runtime dependencies; keep them |
| `runs` | Ignored manifests, reviews, audit history and actual execution evidence |
| `private` | Supplied proposals, related paper, rubric and recovery archive |

## Run the existing foundation

Gate B started on 13 October: see the [coverage audit and measured limitations](docs/Coverage_Audit_2026-10-13.md). The active rendering policy and research review gate remain unchanged.

From this component folder in PowerShell:

```powershell
.\Run-Phase1.ps1 -InputAudio '.\data\vocadito\Audio\vocadito_10.wav'
& '.\.venv\Scripts\python.exe' -B -m unittest discover -s tests -v
.\Run-Notebooks.ps1
```

Open the generated `report.html` for playback and diagnostics. `ACCEPT_UNCHANGED` means no musical correction; channel selection, resampling and encoding are logged. `ACCEPT_PREPARED` is reserved for a future verified repair path. Rejected/failed outputs carry no accepted WAV.

The project uses Python 3.12+ and a component-local uv lock. On a fresh machine, install uv and run `uv sync --locked` here; `uv sync --locked --group notebooks` includes the notebook tools. Existing launchers use the ignored portable uv in `.tools` when present.

## Use the datasets correctly

**Vocadito** is the original development/reference collection: 40 recordings from 29 singers with F0 and two note annotations. Preserve its source/singer split and untouched evaluation partitions. Its annotations describe performed singing and are not intended-note correction targets.

**MTG-Jamendo** adds accompanied songs for exploratory vocal preparation. Keep artist/source groups separated, filter licenses, retain separated vocal WAVs, crop to <=20 seconds, and screen leakage, multiple voices and pitch reliability. There are 1,604 local tracks and three retained stems; a common builder is planned, not implemented. See the master plan before generating a new collection.

The existing working extraction route is:

```powershell
.\Run-Extraction.ps1 -IndexOnly
.\Run-Extraction.ps1 -First '1079195,1348174' -Gpu -VocalOnly -KeepStems -Threads 8
```

It requires the installed component tools. Fresh setup uses `Setup-Jamendo.ps1` and `Setup-ExtractionTools.ps1`; do not rerun setup or full-volume extraction merely to inspect existing evidence. The Intel Arc environment lives under `checkpoints/melody-extraction/.venv-xpu`.

Reuse the team's pinned separator and GAME through component-owned adapters. Read GAME's **`model_midi_pitch`**; the published pitch is +12 semitones. Instrumental/combined melody, chords and key are not BPDR inputs. The newer nine-stage melody/chord runner is not a drop-in replacement for this launcher. See the [extraction assessment](docs/Jamendo_Extraction_Verification.md).

Research training remains review-gated. Engineering mode may use audited unreviewed examples but stamps every output `exploratory_not_research`. The downstream component receives a validated accepted WAV, status and provenance, not these dataset pseudo-labels.
