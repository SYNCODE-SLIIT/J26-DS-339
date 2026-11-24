# Short-support timing: real-voice feasibility result

Completed 2 November 2026. Experimental rendering/measurement only; no correction
model was trained, evaluated or activated. This does not establish usable accuracy.

## What changed and why

V38 showed that the existing audit loses all interior measurements on the shortest
edits. V39 tests a specific temporal-support explanation on the same previously
inspected twelve sources, with first-at-most-ten-second crops, and on synthetic
six-second melodic fixtures. No source was replaced after seeing a result.

Inspection of the installed frozen PESTO kernels finds nonzero sample offsets
-1465 through +1464 at 22,050 Hz: at most **66.440 ms** on either side. Its installed
inference treats time as batch; its encoder and normalization do not mix adjacent
time frames. The current YIN window is 2048 samples, a half-width of **46.440 ms**.
A **70 ms guard** therefore contains both analysis windows on this exact aligned
20 ms grid. This is a finite-window containment argument, not a proof of pitch
accuracy, musical suitability, independent frames or future estimator versions.

The declaration hashes the installed acoustic code and checkpoint. The three
experimental arms are:

| Arm | Render ramp | Analysis guard |
|---|---:|---:|
| Baseline timing | 80 ms | 100 ms |
| Window-matched guard | 80 ms | 70 ms |
| Short ramp and guard | 20 ms | 70 ms |

All three use a **diagnostic-only 0.24 s renderer minimum** to permit measuring
short requests. This is not the default runtime, which still rejects supports
shorter than 0.86 s. The controller was not changed and no user export was granted.
All non-timing audit gates, including ten common frames, 75% coverage, fine/coarse
error limits, outside drift and strict direction checks, remain unchanged.

## Predeclared attempts and results

Twelve real sources × four durations × two shift directions = **96 planned requests
per arm**. Support selection uses the center of the longest source-only reliable
acoustic region and requires 70 ms of original context on both sides. There are
**48 constructed requests and 48 source-context exclusions per arm**. Exclusions
remain in the all-attempt denominator. Controlled shifts are ±30 cents.

| Support | Real attempts per arm | Rendered per arm | Baseline passes | Guard-only passes | Short-ramp passes |
|---|---:|---:|---:|---:|---:|
| 0.24 s | 24 | 16 | 0 | 0 | 0 |
| 0.36 s | 24 | 14 | 0 | 0 | 6 |
| 0.50 s | 24 | 10 | 0 | 9 | 9 |
| 0.70 s | 24 | 8 | 8 | 8 | 8 |
| Total | **96** | **48** | **8** | **17** | **23** |

The counts describe successfully measured **forward fixture shifts**, not learned
corrections, recovered original audio, participant success or independent singers.
Repeated shifts/crops are not independent people. No uncertainty interval or
population accuracy estimate is asserted from this small diagnostic cohort.

| Source performer | Baseline passes | Guard-only passes | Short-ramp passes |
|---|---:|---:|---:|
| VocalSet male10 | 2 | 4 | 6 |
| VocalSet male11 | 4 | 7 | 9 |
| CSD female source 1 | 2 | 6 | 8 |
| JSUT-song female source | 0 | 0 | 0 |

Synthetic fixtures use two registers (120/240 Hz), four durations, two support
offsets (0/10 ms) and both directions: 32 attempts per arm. Baseline passes 8/32;
guard-only and short-ramp pass 16/32 each. Every 0.50/0.70 s synthetic case passes
in the latter two arms, but no 0.24/0.36 s synthetic case passes.

At 0.36 s the short-ramp real cases have nine or ten common frames; only the six
ten-frame cases pass. All eight synthetic cases have fewer than ten. This is a
grid/endpoint-sensitive duration, not reliable general support for 0.36 s notes.
We did not round endpoints, reduce the frame count or insert a tolerance to pass
these cases. The 0.24 s measurement gap remains unresolved. The JSUT failures
also remain unresolved; this result must not narrow the owner's short-hum target
to the source groups or durations that happen to pass.

## Independent verification

`tools/verify_short_support_timing.py` independently recreates all 28 original
fixtures/crops, replays **240 decoded audits**, reproduces **144 source-context
exclusions** across arms and rechecks **240 opposite-request controls**, none of
which passes. Exact untouched samples, finite unclipped same-length outputs,
unchanged non-timing audit limits and V37's complete locked runtime closure pass.
These opposite-direction controls do not cover all possible harmful edits or
prove calibration on noise, expression or ambiguous intentions.

The initial verifier needed JSON tuple/list normalization when comparing saved
policies; after that comparison fix, all measurements were replayed unchanged.
No output from that failed verifier run was treated as a completed receipt.

Evidence: `results/transformer_usability/2026-11-02_v39_short_timing/` contains the
prospective declaration, every request/exclusion, per-source results, private
waveforms, complete results and `integrity_receipt.json`. JSUT source and derivative
audio remain private, ignored and excluded from shared/public demo artifacts.

## What this changes next

1. A shorter guard has a measured acoustic-window basis and improves this diagnostic
   protocol without altering error/direction limits. It remains an experimental
   option; there is no production timing selection or Transformer promotion.
2. The existing model feature is also duration-limited: `step_context.py` requires
   at least 63 interior frames and searches pulse widths from 43 frames (0.86 s).
   Its cues are not designed to describe short-note deviations. The next model
   iteration needs short-scale context features and matching physical training
   examples, rather than a blind increase in epochs on held-tone data.
3. V40 reserves the existing Vocadito melodic sources without changing any singer
   split: **21 training recordings/14 singers**, 8 development/6, 8 PP1 evaluation/6,
   and 3 final-holdout/3. Every source hash matches the existing Phase 2 manifest.
   No new acoustic decode or annotation loading occurred during reservation.
   The next construction step may access training sources only. Planned 336 training
   attempts are not generated examples or completed training. Prior dataset access
   remains disclosed; these are not newly untouched research sources.
4. Dataset-specific project reference reviews remain pending. Engineering
   controlled-recovery work is separate from research-eligible training. Human F0
   annotations describe performed pitch, not unknown intended notes.

Vocadito contains short solo singing recordings with musician annotations, according
to the [publisher record](https://zenodo.org/records/5578807). Its [official mirdata
loader](https://mirdata.readthedocs.io/en/1.0.0/_modules/mirdata/datasets/vocadito.html)
identifies CC BY 4.0 licensing. Attribution and source links are retained in V40's
reservation; annotations will not be added to callable model inputs.
