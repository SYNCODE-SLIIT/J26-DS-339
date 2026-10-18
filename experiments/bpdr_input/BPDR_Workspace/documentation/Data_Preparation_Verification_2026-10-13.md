# Shared dataset preparation - 13 October 2026

The common Vocadito/Jamendo preparation path is implemented and has completed a real training-only pilot. It does not repair pitch, create training pairs, replace prior reviews or establish research accuracy. Representative stem suitability (J3), reference approval (J4) and multi-source intervention coverage (Gate B) remain open.

## Implemented path

`Typed source selection -> license/group/checksum checks -> mapped crop -> Phase 1 accepted WAV -> two pinned GAME decodes -> reliability screen -> immutable manifest and pending review queue.`

- `data_sources.py` reuses the Vocadito manifest with human F0/A1/A2 references, filtering training singers before crop/reference reads. Jamendo uses training-artist groups, derivative-permitting licenses and one song per artist; artist identity remains a singer proxy.
- `data_builder.py` verifies the pinned GAME commit/clean tracked source and the GAME/separator checkpoint hashes. GAME decodes each accepted WAV. Raw numeric pitch becomes `model_midi_pitch`, without the pipeline's octave shift. Two raw CSV/MIDI/log sets are retained and hashed; repeated-decode diagnostics are not calibrated accuracy.
- The common record contains original-source identity, sample/time mapping, accepted hash/format, typed human or estimated references, model/policy/source-code versions, masks, exclusions and review state. Human references never become intended-note correction targets or model input arrays.
- The provisional screen combines note interiors, voiced PESTO, YIN agreement and estimated-note pitch agreement, intersecting human-reference masks for Vocadito. Empty, weak and overlapping predictions are withheld. Single-voice identity, leakage, musical acceptability and ambiguous transitions remain manual checks.
- Exact selected-source hashes and normalized-envelope similarity flag possible duplicates. This is not a corpus-wide or exhaustive cross-partition duplicate audit.
- Two pending review rows are written per candidate, keyed to accepted WAV hashes. Review edits do not create audited training pairs; existing Phase 2 reviews and pointers remain intact.
- `preparation_report.py` verifies the run and generates a local listening page linking original/accepted audio, human/model notes, screen reasons and provenance.

## Real pilot

Run: `runs/preparation/d3a6761a634446fcbb14eacdc21a9bcb`.

| Dataset | Prepared crops | Represented groups | Screen candidates | Candidate groups |
|---|---:|---:|---:|---:|
| Vocadito | 3 | 3 known singers | 3 | 3 singers |
| Jamendo | 3 | 2 artists | 2 | 1 artist |

All six crops are `ACCEPT_UNCHANGED`: technical preparation only. Jamendo candidates are two windows from track 1038720 at 55 and 125 seconds. The 1079195 window is withheld for low voicing/reliable pitch. Track 1348174 has no window with the required five predicted vocal seconds and was excluded. This deterministic inspection sample is not representative J3 evidence; five candidate crops are not five independent voices.

Ten review rows are pending; **zero training-ready examples** were created. Screen settings (0.25 voiced fraction, 0.35 reliable share inside notes, ten reliable frames, maximum 0.05 predicted overlap) are versioned engineering hypotheses, not calibrated usability thresholds.

Reopened-run verification passed for six accepted packages, typed references, masks, all raw decode artifacts and 23 source-file hashes. The exact seven-module source used for the pilot is retained under `source_code_at_run/` and matches the manifest hashes. Later receipt support is safety-tested; the pilot only uses legacy stems. Development/evaluation/holdout recordings were not opened.

## Separation and provenance

`separation_adapter.py` selects new licensed training-artist songs and delegates one fresh track to the existing working vocal launcher. It records Intel GPU, overlap 4, eight threads, model/source/stem hashes and a per-track receipt. Existing stems/receipts/cached vocal outputs are refused; retry-failed handling and the fresh named track prevent accidental selection of another song. Receipt loading rejects changed source/stem hashes or different model pins.

The old three stems have no individual receipts, which is recorded explicitly. Checking today's model does not verify past settings. The new wrapper has selection/overwrite/cache/receipt tests, but **no fresh GPU separation ran during this pilot**. Runtime/yield validation is the next step; receipts alone do not prove exact separator alignment or musical usability.

The declared next sample is five new training artists: tracks 1101810, 1071484, 86270, 1113639 and 1086669. They are selected, not extracted; `next_sample.json` retains full metadata. Recheck actual eligibility/source hashes at launch.

## Verification

The full suite passed 63 tests, including the frozen PESTO smoke and typed two-source fixture. After the final cache-selection safeguard, 13 focused safety tests passed, including one additional case: 64 unique tests overall. Fixture GAME predictions are explicitly fake; the six-crop pilot uses the real pinned GAME model.

Checks cover protected partitions, source/reference/model corruption, licenses, traversal, duplicates, empty/overlapping predictions, unchanged original reviews, time mapping, no implicit research promotion, tampered accepted audio and stale receipts/cache selection. Logs: `runs/preparation_regression_2026-10-13.txt` and `runs/preparation_final_safety_2026-10-13.txt`.

## Reproduction

From this component folder:

```powershell
& '.\.venv\Scripts\python.exe' -B -m bpdr_input.data_builder build --vocadito-singers 3 --jamendo-artists 3
& '.\.venv\Scripts\python.exe' -B -m bpdr_input.data_builder verify '.\runs\preparation\d3a6761a634446fcbb14eacdc21a9bcb'
& '.\.venv\Scripts\python.exe' -B -m bpdr_input.preparation_report '.\runs\preparation\d3a6761a634446fcbb14eacdc21a9bcb'
& '.\.venv\Scripts\python.exe' -B -m bpdr_input.separation_adapter plan --count 5
# Full-song GPU job, creates a new stem/receipt without overwriting old files:
& '.\.venv\Scripts\python.exe' -B -m bpdr_input.separation_adapter separate 1101810
```

Builds create new UUID directories and preserve completed/failed evidence, never replacing Phase 2 pointers. Changed sources invalidate verification; preserve snapshots and create a new run instead of rewriting provenance. GAME exposes no seed, so later decodes need not reproduce boundaries exactly.

Next: extract/screen the declared five-artist sample with receipts, measure independent-source yield before expanding, obtain real reference reviews and resolve the coarse PESTO audit issue before research training.

## Later update: five-artist sample completed

The above describes the initial plumbing pilot. The [fresh five-artist report](Jamendo_Five_Artist_Sample_2026-10-13.md) now supplies current evidence: five new receipts, ten unchanged common outputs, six Jamendo screen candidates from three artists plus three Vocadito candidates, eighteen pending reviews and zero research-ready examples. Five of eighteen v0.3 test shifts pass across two artists. Fine shift error and loss of voiced audit coverage require diagnosis alongside the earlier PESTO issue.

Use `runs/preparation/d274ed728d3d46f3b2661ba5291e61da` for current validation/listening. New extraction changed the live metadata/history indices tracked by this initial pilot; its strict live-source verification is consequently historical. Preserve its original manifest/hashes and archived source rather than rewriting them. All five fresh sources have individual receipts; the legacy-receipt warning above applies to the initial three stems.
