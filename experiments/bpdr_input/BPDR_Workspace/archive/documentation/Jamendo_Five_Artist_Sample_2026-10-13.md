# Five-artist Jamendo sample - 13 October 2026

The declared five-track sample completed full-song vocal separation and GAME extraction on the Intel GPU/CPU path. The common builder then prepared ten unchanged technical outputs from both datasets. Six Jamendo crops from three artists passed the provisional screen. Only five of eighteen tested detuning variants passed the existing v0.3 audit, across two crops/two artists. **Zero research-ready examples or training pairs were created.**

This establishes a working preparation path and preliminary source yield. It does not establish musical acceptability, accurate repair, population-level suitability or a completed J3/J4 gate.

## Frozen selection and evidence

Five distinct training artists were selected before extraction using the existing seed, voice tag, duration >=60 seconds and derivative-permitting license allowlist. The selected sample was not replaced after observing its results. Artist identity remains a proxy: five artists are not five verified independent singers.

- [Frozen selection](../runs/preparation/d3a6761a634446fcbb14eacdc21a9bcb/next_sample.json).
- [Completed batch and track receipts](../runs/jamendo_sample/bf912553c00c40598b8f0d64ae266de0/batch.json).
- [Current preparation manifest](../runs/preparation/d274ed728d3d46f3b2661ba5291e61da/manifest.json), [listening page](../runs/preparation/d274ed728d3d46f3b2661ba5291e61da/review.html) and [pending review sheet](../runs/preparation/d274ed728d3d46f3b2661ba5291e61da/reviews.csv).
- [Intervention measurements and eighteen retained artifacts](../runs/jamendo_yield/de2f85a01b194c4eab009aa95b14e496/evidence.json).
- [Final preservation/integrity checks](../runs/jamendo_sample/bf912553c00c40598b8f0d64ae266de0/final_verification.json).

All five fresh stems have contemporaneous receipts linking original MP3/stem hashes, pinned model hashes, launcher, logs and settings: XPU, overlap 4, eight separator threads, full-song separation. Accepted-crop GAME uses the unchanged pinned model, numeric `model_midi_pitch`, zero octave shift and two saved decodes. The batch keeps the machine awake only for its process and restores the normal state afterward. The synthetic probe used eight CPU threads and the unchanged v0.3 thresholds.

The old pipeline reports these songs as partial because the instrumental/chord stage was deliberately skipped. Their vocal outputs completed. No chord-stage output is used by BPDR.

## Measured source yield

| Track | Artist | License | Full-song estimated notes | Prepared crops | Screen candidates | Crop offsets (seconds) |
|---|---|---|---:|---:|---:|---|
| 1101810 | artist_428449 | BY-NC 3.0 | 93 | 2 | 2 | 45, 260 |
| 1071484 | artist_359617 | BY-SA 3.0 | 29 | 1 | 0 | 95 |
| 86270 | artist_003799 | BY-NC-SA 3.0 | 6 | 0 | 0 | None |
| 1113639 | artist_442950 | BY-NC-SA 3.0 | 298 | 2 | 2 | 30, 200 |
| 1086669 | artist_439729 | BY-SA 3.0 | 342 | 2 | 2 | 135, 160 |

All seven Jamendo crops are 20 seconds and technically `ACCEPT_UNCHANGED`. Track 86270 had no fixed window containing five predicted vocal seconds; this is an extractor-based exclusion, not proof that the song contains no voice. The 1071484 crop is withheld: about 1.0% voiced frames and 5.5% reliable share inside predicted notes. Candidate reliable shares range from 44.7% to 75.7%; these are screening measurements, not accuracy or calibrated confidence.

The same preparation run retains three Vocadito crops from known singers S7, S10 and S11, all screen candidates, with their human F0/A1/A2 references kept distinct from GAME estimates. Total: ten technical outputs, nine screen candidates and eighteen pending review rows (twelve Jamendo, six Vocadito). Existing Phase 2 reviews remain unchanged.

Descriptive Jamendo yield is six of seven prepared crops, but only three of five selected artists reach the review queue. Multiple windows and synthetic variants from one track do not count as independent performances. The sample is small and filtered; these fractions are not a population benchmark.

## Intervention support is still a bottleneck

The existing v0.3 policy was retained. Three Jamendo candidate crops from two artists had both the required plateau and protected outside frames. Their zero-shift identity checks passed, and all eighteen nonzero variants preserved sample count and exact samples outside the support.

| Track / crop offset | Tested shifts | Passing shifts (cents) | Result |
|---|---:|---|---|
| 1101810 / 45 s | 6 | None | Fine YIN errors exceed the fixed median/p90 tolerances |
| 1101810 / 260 s | 6 | +10 | Other shifts fail fine error and/or common voiced audit coverage |
| 1113639 / 30 s | 6 | -30, -20, -10, +10 | +20/+30 lose required common audit coverage |

At 1113639/30 s, +20/+30 have small fine YIN errors but only 71.9%/65.6% common plateau coverage, below the fixed 75% requirement. Thus this sample's failures cannot all be explained by the previously observed coarse PESTO pitch-bin issue. Fine renderer errors and after-render voicing/coverage also require diagnosis. Protected outside pitch drift was zero in these measurements. Mathematical preservation and tracker audits still do not certify audible quality or approve the original performance.

Five passing variants across two artists are limited engineering evidence. No crop passes all six shifts; no approved anchor or research training eligibility follows from these probes.

## Verification and practical cost

The completed preparation verifier checked ten accepted packages, typed references, masks, decode artifacts, receipts and 34 tracked source hashes. Both source-code snapshots match their recorded hashes; all eighteen probe WAV hashes verify. Original Phase 2 review/pointer hashes are unchanged. Repository changes remain confined to `experiments/bpdr_input`.

The full regression run passed 68 checks. Four focused batch/yield checks then passed, including two additional cases. Final resume hardening passed 21 focused checks, including two more cases: **72 unique checks overall**. Logs: [full regression](../runs/jamendo_batch_regression_2026-10-13.txt), [yield checks](../runs/jamendo_yield_safety_2026-10-13.txt) and [final resume checks](../runs/jamendo_resume_safety_2026-10-13.txt). Constructed fixtures validate mechanics; the dataset run uses real pinned models.

After measurement, resume validation was tightened to require the exact separator settings and unchanged frozen selection file. The archived source snapshots retain the versions actually used for the dataset/probe runs; recorded evidence was not rewritten. The five real receipts also pass the final stricter settings check.

Per-track extraction took about 33.1, 14.9, 12.5, 23.6 and 17.0 minutes, including verification, launch, separation and full-song decoding: about 101 minutes total. This is offline preparation cost, not BPDR product latency. Memory pressure was observed; subsequent model jobs were kept sequential. The retained five-track receipts allow completed stems to be reused.

The initial pilot's live metadata/history hashes refer to the earlier extraction state. New extraction changed those indices, so its strict live-source verification is now historical. Its manifest/hashes must not be rewritten to appear current. The current ten-crop run above is the validation/listening reference. Generic legacy-stem warnings in the builder summary concern the older retained stems; all five sources used in this sample have individual receipts.

## Next work

1. Obtain two independent musical reviews of the nine queued crops: single voice, accompaniment leakage, acceptability and agreed stable intervals. Jamendo pitch references remain `model_estimate`; approval does not invent human note annotations.
2. Continue Gate B on training material: diagnose fine rendering error and loss of voiced audit coverage alongside the earlier PESTO quantization issue. Compare justified renderer/support candidates with fixed preservation/error requirements; retain v0.3 until evidence supports a new version.
3. Wire approved reference intervals and audit-passing interventions into the common pair builder. Require source/partition/hashes, reviewed anchors and realized-shift audits before research eligibility. Keep denoising disabled for the initial comparisons.
4. Decide on a 20-30-artist expansion only after manual quality and independent-source intervention yield are assessed. Six automatically screened windows alone do not justify research training or processing the full corpus.

Vocadito remains the human-reference/evaluation dataset. Jamendo is now proven usable through the shared engineering preparation path and supplies review candidates; its research-anchor role remains conditional on the outstanding gates.

## Reproduce or verify

From this component folder:

```powershell
& '.\.venv\Scripts\python.exe' -B -m bpdr_input.data_builder verify '.\runs\preparation\d274ed728d3d46f3b2661ba5291e61da'
& '.\.venv\Scripts\python.exe' -B -m bpdr_input.preparation_report '.\runs\preparation\d274ed728d3d46f3b2661ba5291e61da'
# New immutable preparation using exactly the five retained sources:
& '.\.venv\Scripts\python.exe' -B -m bpdr_input.data_builder build --vocadito-singers 3 --jamendo-artists 5 --jamendo-tracks 1101810,1071484,86270,1113639,1086669
# A genuinely interrupted batch can resume its declared selection and verified receipts:
& '.\.venv\Scripts\python.exe' -B -m bpdr_input.jamendo_batch --selection '.\runs\preparation\d3a6761a634446fcbb14eacdc21a9bcb\next_sample.json' --resume '.\runs\jamendo_sample\bf912553c00c40598b8f0d64ae266de0'
```

Completed-batch resumption creates another preparation run; it does not reproduce the original decodes bit-for-bit. Builds/probes create new directories and retain failed evidence. GAME exposes no seed. Further extraction that changes live metadata/history can invalidate strict verification of an earlier run; preserve historical evidence and build a new current run.
