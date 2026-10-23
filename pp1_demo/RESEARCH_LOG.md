# PP1 application decisions and execution log

## 2026-10-22 — M0: integration setup

- Use a separate `pp1_demo/` application and existing project uv environment. Add only missing UI packages; preserve root dependency manifests and teammates' source.
- User authorized implementation, evaluation and incremental local commits. No push requested.
- Use Pasan's temporal weighted best checkpoint, not the function3 checkpoint. Decode its root/family heads and retain original architecture.
- Rahul receives mode `major` with `demo_constant` provenance on every run.
- BPDR reproduction depends on missing local manifests, historical model and audited pairs. Execute the notebook setup to record the real failure. Never manufacture training evidence or silently replace its model.
- Store uploaded recordings, executed notebooks and generated media only under ignored `runtime/`.

## 2026-10-23 — Local website and three-model integration

- User requested all work stay inside `pp1_demo/`, Vishwa deferred, dataset excerpts
  and saved musical features used, Rahul mode fixed to major.
- Replaced the partial Streamlit entry point with a local HTTP website and offline
  HTML/CSS/JS frontend. Added stage cards, live progress, source waveform/playback,
  piano roll, actual BPDR pitch/correction plots, chord comparison and audio/MIDI export.
- Verified the new complete Praveen export and used its unchanged installer and guarded
  loader in a private application-local source snapshot. Initial audio execution
  stopped on missing matplotlib in the isolated environment; resolved by reading
  the main environment's optional plotting packages after BPDR's own packages.
- Strictly loaded Pasan latest (epoch 9). Replaced the four-bar demo shortcut with
  sixteen complete saved-context bars. Kept crop offsets and original source notes.
- Ran all six recordings through real BPDR → Pasan → Rahul. BPDR reported IDENTITY /
  ACCEPT_UNCHANGED, all model stages completed, and no rendering warnings occurred.
  Rehearsals took about 5.7–6.7 seconds each before presentation holds.
- No training, no external source/checkpoint/dataset edits and no environment installs.
- Final verification: seven focused unit/contract tests passed. The localhost HTTP
  check passed, including all live model stages, completed-handoff visibility,
  byte-range audio, every output artifact, malformed input and path boundaries.
- Browser rehearsal verified recording/genre selection, source provenance dialog,
  live processing, actual chord output, original vocal playback and combined-audio
  playback. No JavaScript errors were reported. At 390 px, caught and corrected a
  grid-child intrinsic-width overflow; the verified document width now matches the
  viewport. Restored the normal desktop viewport afterward.
- Saved screenshots inside `runtime/screenshots/`. Local app runs on loopback port
  8765; startup is `./pp1_demo/start.sh`. Model/runtime source edits remain confined
  to this application folder.

## 2026-10-23 — Expanded recording library

- Removed the six-recording cap and source-major-only restriction. Discover all
  compatible local crops: 811 recordings (333 major, 478 minor) from 1,102 vocals.
- Added recording/genre/key search. Load waveform and full preview data only for
  the selected recording; catalogue metadata is approximately 485 KB and scans in
  1.6 seconds locally. The model pipeline and fixed Rahul mode are unchanged.

## 2026-10-23 — Vishwa restored as a live component

- Supersedes the earlier deferred-stage decision. Pulled team merge `4f60115`.
- Exported existing CNN/hybrid checkpoints and TRAIN normalization (approximately
  211 KB), with original report hash verification. No retraining or new accuracy claim.
- Live BPDR accepted audio now enters Vishwa: pYIN notes plus both learned score-class
  models. Live notes replace the audible crop; saved notes only supply outside-crop
  context for the existing sixteen-bar Pasan model. Rahul consumes the same merged
  melody and keeps the requested constant major mode.
- Preserved the team's current Pasan latest/epoch-9 selection and all teammate
  model source. The earlier best-checkpoint plan was superseded by their merged app.
- Added the melody stage, live plots, experiment probability maps, research panels,
  manifest provenance, portable setup and Windows runtime-path handling.
- Real track_0000382: 19.790 seconds, 57 live notes, 42 basic/refined chord intervals.
  Browser run 16.745 s including display holds, Vishwa 1.042 s. HTTP run 16.024 s.
- Twelve tests passed; HTTP artifact/range/security/handoff check passed; frontend
  JavaScript syntax and visible browser results checked. Only 11 compatible recordings
  are available on this machine, so catalogue tests now check availability rather
  than incorrectly require the previous machine's >100 recordings.
- Cross-platform setup is documented; only macOS execution has been verified.

Second real-recording check: `track_0000383` completed in 11.986 s without display holds, with 41 live notes and 41 excerpt chord predictions. Both learned checkpoints ran; the minor source key remained saved metadata and Rahul mode remained the explicit major constant.

Checkpoint export commit: `56d2eb5` (`feat: export portable Vishwa melody checkpoints`).
