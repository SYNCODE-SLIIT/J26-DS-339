# Usable-input plan and status (5 Nov 2026)

Status key: DONE = run, recorded, rule declared before scores. PARTIAL = run with stated limits. TODO = not started. BLOCKED = needs a decision/data/person.
None of the results below has been independently replayed. Nothing is committed. Shared code, defaults and the demo are unchanged.

## Purpose and claim (decided 5 Nov, scope widened by the user)
BPDR = input-preparation gate for hums/sung melodies before melody extraction: assess usability, apply bounded reversible repairs, accept a repair only if the downstream
extractor evidence improves, otherwise refuse with time-localised re-record guidance, and keep original + provenance. The boundary-invariance / counter-detuning objective (BPDR)
stays as the bounded pitch-repair operator; its matched comparison with augmentation-only and frame-only models is still required. No worldwide priority claim.
Note: the approved proposal (IT23141506) excludes heavily clipped recovery; widening beyond it must be told to the supervisors.

## Data and models available
| Item | Status | Use |
|---|---|---|
| Vocadito (local `datasets/vocadito`) | usable | real phone/computer singing, human F0 + notes; only the 21 training-role recordings (14 singers) may be used; the rest stays closed |
| HumTrans (local, 24 WAVs + guide MIDI) | usable with limits | real humming; guide MIDI is not performed-pitch truth |
| ESC-50 (local `datasets/environmental_noise`, CC BY-NC) | usable | 4 root clips (rain, vacuum, breathing, clock tick) used as REAL noise; 4 `validation` clips kept closed |
| Smule DAMP / Sing!300x30x2 | BLOCKED | license request required; no pitch annotations found; not downloaded |
| MIR-ST500, MIR-QBSH, MAMI | not obtained | need request/download; not verified |
| pYIN (librosa), PESTO 2.0.1, Praat-AC | installed, in use | frozen trackers for damage map, gate and real-data test |
| Team extractor `src/melody_understanding` (pYIN baseline) | in use | stand-in downstream extractor |
| DeepFilterNet (MIT/Apache), Voicefixer/ResembleEnhance (MIT), SGMSE+ (MIT) | NOT installed | candidate neural denoisers; installing them changes the locked environment, ask first |
| ROSVOT / RMVPE / Basic Pitch | not installed; checkpoint/license unverified | candidate stronger extractors |

## Workstreams
| # | Item | Status | Evidence (folder / report) | Result in one line |
|---|---|---|---|---|
| 1 | Learned Transformer for short-note repair | DONE (negative) | `2026-11-04_v70_detection_diagnosis`..`v73_edge_shape`, `v72_conservative_gate` | pitch-only input cannot separate injected detunes from natural movement (AUROC .32-.64); gate protects but catches <1% |
| 2 | Exact-reference repair upper bound | DONE (oracle) | `v74`..`v78` folders | macro error 24.9 -> 8.6c, 294/445 beneficial; audit and cap limits identified; not deployable |
| 3 | Realistic reference (semitone snap) | DONE | `2026-11-05_v79_realistic_reference` | 33% natural spans falsely flagged -> pitch repair must be review-only |
| 4 | Downstream note test (stand-in extractor) | DONE (oracle) | `2026-11-05_downstream_note_test` | in-span label agreement 82.6% -> 93.8% with exact reference |
| 5 | Prior-art / novelty report | PARTIAL | `reports/Usable vocal input restoration novelty.md` | claim the verified gate, not restoration; evidence is snippet-level |
| 6 | Damage map (synthetic, 3 trackers) | DONE | `2026-11-05_damage_map` | noise, clipping, low level, reverb, band-pass hurt; low-pass does not; PESTO fails at -60/-70 dB |
| 7 | Restoration + closed-loop gate, round 1 | DONE (rule failed) | `2026-11-05_restoration_gate` | PESTO +0.263 recall, pYIN/Praat ~0; gate cuts harm from 10-14% to ~3% |
| 8 | Heavy-clip refusal + dereverb, round 2 | DONE (rule failed) | `2026-11-05_restoration_gate2` | refusal works; own dereverb does not help |
| 9 | Real noise + human F0 (Vocadito x ESC-50) | DONE (declared rule passed, effect small for 2 of 3 trackers) | `2026-11-05_real_noise_human_f0` | at 0 dB the gate raises RPA50 vs human F0: PESTO .829->.867 (recovers ~73% of the loss), Praat .684->.705 (~7%), pYIN .598->.605 (~3%); gated harm 0%; accept rate 30% |
| 9b | Neural denoiser arm (DeepFilterNet3 in an isolated env, same real-noise test) | DONE (declared rule FAILED because of PESTO; big win for pYIN and Praat) | `2026-11-05_real_noise_neural` | at 0 dB gated RPA vs human F0: pYIN .598->.799 (88% of loss recovered; spectral gate 3%), Praat .684->.932 (85%; spectral 7%), PESTO .829->.864 (68%; spectral 73%) but PESTO is harmed in 21% of gated cases (harm 32% ungated) |
| 9c | Per-tracker denoiser choice (spectral vs neural, picked by the input-only gate) | TODO | - | new hypothesis, needs its own declared rule; do not tune on 9b's scores |
| 10 | Independent replay of key results (v76, downstream, gate) | TODO | - | required before citing |
| 11 | Matched BPDR vs augmentation-only vs frame-only comparison | TODO | - | the proposal's primary scientific test; current evidence leans against an advantage |
| 12 | Stronger dereverb / neural denoise (WPE, DeepFilterNet) | TODO / needs approval | - | needs a dependency decision |
| 13 | Time-localised re-record guidance (UI/report) | TODO | - | refusal logic exists in `restoration_gate2`; no guidance text/time ranges surfaced to a user |
| 14 | Real phone recordings with two annotators + ethics approval | BLOCKED | `Review_and_Collection_Plan_2026-11-01.md` | participants, consent, annotators needed |
| 15 | Tell supervisors about the scope change | BLOCKED (owner action) | - | proposal excludes heavy-clip recovery |
| 16 | Handoff contract for note/key targets from other components | BLOCKED | - | required for any pitch flagging |

## Next steps (ordered)
1. (DONE) Result 9 read and marked. It is the first evidence on real recordings, real noise and human F0, but only PESTO recovers much; the spectral-gate denoiser is the weak link for pYIN/Praat at 0 dB.
2. Replay the key results independently (10).
3. Decide on installing a neural denoiser (DeepFilterNet etc.) in an ISOLATED environment and re-run result 9 with it (12); this is the most likely route to recovering more of the 0 dB loss.
4. Run the matched BPDR comparison (11) so the proposal's stated novelty is honestly tested.
5. Build the guidance output (13) and collect real recordings (14).
