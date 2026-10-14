# Permission and rollback fixtures - 14 October 2026

The next engineering step implements edit-permission and rollback mechanics while actual original-reference reviews remain pending. **The active pipeline and renderer are unchanged. No accepted repaired audio or research-ready example is created.**

## Implemented scope

`repair_controller.py` provides a separate engineering-only planner/transaction. It withholds the whole supplied proposal when correction magnitude, alternate-boundary disagreement, supplied uncertainty, unreliable voicing, protected regions, insufficient support or affected fraction fails. It never clips an excessive correction to obtain permission. Decisions include source/sample metadata and a decision checksum; changed source audio or permission content cannot reuse the decision.

The transaction renders in a disposable copy and independently checks sample format/count, finite/unclipped samples and exact outside-support equality before its post-render audit. Failed renders/audits return the preserved source and discard candidate samples. Zero correction bypasses rendering and returns exact identity. Every outcome has `accepted_audio=None` and `production_eligible=False`; it is not a `ProcessingResult` or `ACCEPT_PREPARED` hand-off.

The fixture policy predeclares 30-cent magnitude, 5-cent disagreement/uncertainty, 25% affected samples and 0.86-second minimum support. These are illustrative engineering bounds, **not development-calibrated runtime rules**. Supplied uncertainties/alternate views are test inputs; model uncertainty estimation and runtime proposal generation remain open.

## Constructed evidence

[Recorded fixture run](../runs/controller_fixtures/77e32684041d41fab54d9cd74416e662/evidence.json). A four-second harmonic tone has a known +20-cent pitch relative to 440 Hz. The fixture supplies a -20-cent correction over 1.0-1.9 seconds. It reuses the unchanged laboratory constant-shift renderer and full v0.3 PESTO/YIN post-shift audit. It does not infer an intended note from real vocal audio or implement the planned framewise production renderer.

| Constructed case | Verified outcome |
|---|---|
| Zero correction | Exact identity; no renderer/auditor callback |
| Bounded supplied -20-cent correction | Verified engineering candidate after the unchanged pitch audit |
| Over-limit magnitude | Whole proposal withheld |
| Disagreeing boundary-view proposal | Whole proposal withheld |
| Excess supplied uncertainty | Whole proposal withheld |
| Edit touches protected frames | Whole proposal withheld |
| Renderer alters an outside sample | Rollback to exact source |
| Renderer changes sample count | Rollback to exact source |
| Candidate fails to change pitch | Post-render audit fails; rollback to exact source |

All nine expected outcomes were observed. Only the constructed bounded candidate is saved, explicitly named `verified_fixture_candidate.wav`; it is not an accepted repair package. Original/candidate files, controller/analyzer source snapshots and hashes are recorded. Withheld/identity/rolled-back returned buffers equal the source exactly. Unit checks additionally cover missing uncertainty, unvoiced frames, excessive affected fraction, short support, nonfinite candidates, changed decisions/sources and callback buffer isolation.

All **93 regression checks passed**: [final log](../runs/controller_final_regression_2026-10-14.txt). This establishes these engineering branches, not musical quality, uncertainty calibration, research accuracy or production readiness.

## Notebooks and remaining work

Both maintained notebooks now include this step. Notebook 1 explains that its unchanged technical path is unaffected. Notebook 2 adds the permission/rollback table, measured post-render audit, constructed source/candidate playback and pitch plot, with supplied test values and production limitations explicit. Its evidence index selects the completed run above; Run All reads this evidence without regenerating fixtures or training.

Next: collect the actual independent original-reference decisions and matched real-audio listening observations. Continue engineering of runtime proposal/uncertainty handling and the duration-preserving framewise renderer without adopting uncalibrated fixture thresholds. Calibrate permission rules on separate development material after the required data gates; verify expressive/no-change preservation, transitions, rollback and accepted-only delivery before granting `ACCEPT_PREPARED`. Keep source/voice coverage and research fitting gates intact.

From the component folder, a fixture replay creates a new evidence directory:

```powershell
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
& '.\.venv\Scripts\python.exe' -B -m bpdr_input.controller_fixtures
```
