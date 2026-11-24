# Usable-input plan and status (5 Nov 2026)

## 8 November full actual-consumer development result

All 378 calls completed for 189 opened cases: 21 training-role Vocadito
recordings, 14 singers, first eight seconds, four root noise clips at
10/0 dB plus clean controls. B0 is peak-safe format-only PCM24; B2 is
the frozen saved neural output exported as PCM24. The owning shared
pYIN consumer and preprocessing remain unchanged. A1 and A2 are scored
separately against independent note annotations. These are development
digital mixtures of singing, not new real-hum confirmation participants.

`tools/report_primary_cleanup_development.py` independently reconstructs
all twelve summaries from saved per-case metrics; maximum difference
2.43e-17. Input hashes match the prior audit/export receipt. The full
fresh-process 378-call inference replay completed in **session 88786**,
exit 0. All metrics and preprocessing reproduce exactly; the verifier's
independent aggregate error is 6.94e-18. All 378 raw consumer outputs
are retained and match their receipt hashes. This is full development
inference replay, not fresh confirmation or listening qualification.
Results, aggregate receipt and caches are in
`results/transformer_usability/2026-11-08_primary_cleanup_development/`.

| Slice | A1 onset/pitch F1, B0 -> B2 | A2 onset/pitch F1, B0 -> B2 | A1 harms / benefits | A2 harms / benefits |
|---|---|---|---|---|
| Clean (21 cases) | .3000 -> .2898 | .2903 -> .2830 | 4 / 2 | 3 / 0 |
| 10 dB (84 cases) | .3024 -> .2938 | .2928 -> .2793 | 40 / 26 | 41 / 19 |
| 0 dB (84 cases) | .2832 -> .2866 | .2637 -> .2766 | 35 / 32 | 36 / 31 |

F1 entries are singer-macro means. Harm/benefit counts use per-case
changes below -0.02 / above +0.02; these are descriptive development
counts, not independent population observations. At 0 dB the A1/A2
paired gains are +0.00333 / +0.01293, with singer-bootstrap 95% intervals
[-0.01747, +0.02447] / [-0.00792, +0.03711]. Neither excludes zero.
Both 10 dB intervals also cross zero. The A2 clean interval is negative
[-0.01313, -0.00158]. Offset-aware results are retained: at 0 dB A1
gain is -0.00512 and A2 +0.01762, both intervals crossing zero.

RPA50 B0 -> B2 is .59796 -> .80829 at 0 dB, .78096 -> .82044 at
10 dB and .82762 -> .82405 on clean inputs. Thus the large noisy-frame
pitch benefit does not establish useful note-event improvement for
this actual consumer. At 0 dB mean estimated event count rises from
25.38 to 28.36; counts alone cannot establish fragmentation as the cause.
Raw replay outputs are retained for event-level diagnosis.

**Decision:** always-on B2 remains inactive and has no demonstrated
qualification. No new selector is fitted or inferred from these labels.
This does not evaluate B1 or a declared B3 runtime policy, and does not
alter the historical failed three-tracker gate or the BPDR comparison.

## 7 November encoded-output follow-up

### 8 November actual-consumer compatibility check

The fixed track-1 clean/rain-10/rain-0 check now calls the owning
`melody_understanding.analyze_vocal(path, trim=False)` on format-only
PCM24 and neural PCM24 inputs. Its normalization and final-frame handling
match the five pinned consumer source files. Mir_eval 0.8.2 is isolated
under the experiment dependencies and scores both independent note
annotators separately with the declared tolerances. Reference intervals
are intersected with the eight-second crop, with truncated-note counts
retained.

| Case | Human-F0 RPA50 original → neural | A1 onset/pitch F1 | A2 onset/pitch F1 |
|---|---|---|---|
| Rain 0 dB | 71.41% → 85.11% | 21.62% → 19.05% | 25.00% → 17.78% |
| Rain 10 dB | 82.42% → 85.33% | 27.03% → 20.51% | 25.00% → 19.05% |
| Clean | 85.11% → 85.22% | 25.64% → 24.39% | 23.81% → 22.73% |

All six calls reproduced in a separate invocation, including note
metrics and preprocessing. Offset-aware scores are retained separately
and are mixed in the 0 dB case. This one-source development result
does not establish population performance, but it directly shows why
frame-pitch gains cannot qualify the actual consumer's note output.
The neural candidate has not passed the primary usefulness criterion.
Results and replay receipt are under
`2026-11-08_primary_cleanup_compatibility/`.

The metric uses 50 ms onset and 50-cent pitch tolerance, with the
offset-aware version additionally using max(20% reference duration,
50 ms). [Official mir_eval transcription documentation](https://mir-eval.readthedocs.io/latest/api/transcription.html)

All 189 saved neural candidates now have deterministic, peak-safe PCM24
exports in `2026-11-07_neural_delivery_encoding_v2`. All 567 estimator
scores were recomputed on those decoded files. The seven formerly
over-full-scale FLOAT inputs reproduced all 21 historical estimates.
Across the full export set, 566 scores match the archived FLOAT value.
One PESTO score falls by 0.002219756. Separate aggregate calculations
reproduce the new summary.

With the historical gate decisions held fixed, 0 dB singer-macro RPA50
is pYIN 0.59796 to 0.79918, Praat 0.68429 to 0.93227 and PESTO
0.82879 to 0.86419. PESTO still has 18 harmful outputs among 84
0 dB cases and one harmful accepted clean control among 21 clean
sources. The historical gate remains failed. This study isolates
encoding and does not recompute the runtime decision after export.
Delay configuration, listening and fresh confirmation remain open.
The full result and aggregate receipt are in `full_rescore/`.

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
