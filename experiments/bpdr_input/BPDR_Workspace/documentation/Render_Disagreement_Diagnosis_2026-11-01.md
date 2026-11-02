# Post-render pitch disagreement diagnosis

Declared 1 November 2026. Status: all60 candidate waveforms, tracks and existing audits independently
replayed;21 opposing frames reproduced; six protected artifacts unchanged.
Two diagnostic mask tests pass. No model selected or activated.

The V32 nearby-zero sampling arm passes raw prediction gates but worsens
verified repairs from23/200 to20/200. Four lost successes and one new success
all differ at single-frame coarse-direction vetoes with fine direction passing.
The existing result identifies a stage, not whether a tracker is wrong.

Reconstruct all60 permitted nonidentity candidates across the reference and V32
arms, including successful controls. Preserve both arms' full200 attempted,
108 scoreable and92 excluded denominator. Use exact frozen requested curves,
source audio and permission supports. Reproduce the existing post-render audit;
save native/aligned before/after PESTO, YIN, confidences, requested shifts,
reliable/common/direction masks, private candidate arrays and opposing-frame
measurements. Save no rejected audio as accepted output. No new training,
threshold/support changes, musical labels, development or confirmation access.

An independent replay must reconstruct all60 candidate waveforms, tracks,
strict-sign masks, audits and unchanged samples outside supports. Tiny opposing
measurements must remain included under the current rule; diagnostic magnitudes
must not be silently converted into an acceptance tolerance. Both pitch
measurements are estimates. Fine agreement does not prove the coarse estimate
wrong or the performer's intended melody preserved.

Use `run_render_disagreement_serialized.py diagnose`, then its `verify` stage
(the representation amendment below preserves the original tools and failure).
Results: `results/transformer_usability/2026-11-01_render_disagreement`.
Any further algorithm change needs a separately declared matched comparison and
the existing practical usability/independent-review requirements.


## Execution amendment and interpretation guard

The first run (session96590) stops at reference case9 before saving a completed
receipt: the audit policy has a tuple in memory and a list after JSON loading.
Printed numerical metrics and reasons agree. The original declaration and
sources remain unchanged. `run_render_disagreement_serialized.py diagnose`
performs an explicit JSON round-trip on audit dictionaries before equality;
the wrapper and original declaration are bound in `serialization_amendment.json`.
No numeric rounding, accepted-state change or model retraining is involved.
Run its `verify` stage for the same representation-aware independent replay.

The installed PESTO source uses3 bins per semitone and argmax-local weighted
averaging, which yields fractional-bin estimates. Do not infer that every
opposing measurement is a hard-bin quantization artifact. Its returned
confidence classifier estimates voicedness; a large value is not proof of
small pitch error. The [official inference repository](https://github.com/SonyCSLParis/pesto)
describes this decoding, and the [authors' paper](https://arxiv.org/abs/2508.01488)
describes the pitch-distribution/equivariance method. These sources explain the
estimator; they do not establish the cause of any BPDR disagreement.


## Completed reconstruction

Reference:30 candidates,23 existing audit passes,7 rollbacks and9 opposing
coarse frames. V32:30 candidates,20 passes,10 rollbacks and12 opposing coarse
frames. Across all21 opposing frames, absolute coarse shifts range from
0.0006191374 to0.0009287065 cents. Every opposing frame remains included and the
old pass/rollback outcome is preserved. The completed independent replay covers
all60 waveforms, native/aligned measurements, strict masks and exact outside
samples, not just the previously noticed five changing cases.

The separate [decoder precision diagnostic](Pesto_Precision_Diagnostic_2026-11-01.md)
independently completes90-track/60-audit replay. It changes only arithmetic on the same frozen model distributions;
it cannot approve delivered audio or establish intended pitch. Retain the
JSON tuple/list amendment as execution history. Also retain the confirmed
reference-coverage limitation: female7 has4 valid cases out of20, so the current
pack cannot demonstrate the declared25% per-group coverage regardless of model
accuracy. See the [review/collection plan](Review_and_Collection_Plan_2026-11-01.md).
