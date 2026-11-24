# Existing HumTrans recovery and short-note training preparation

**Current result:** [V51 transfer and V53 frontend replay](HumTrans_Transfer_Result_2026-11-02.md)
supersedes the running-fit status below. Fitting/prediction verification and exact
frontend replay complete; both raw model arms fail qualification. V52 actual-output
evaluation is still running. Historical V50 preprocessing/exclusions remain fixed.

2 November 2026. This supersedes V48's assertion that HumTrans data was unavailable
and its provisional person split. MOD-03 remains ACTIVE. No Transformer is qualified
or activated. The owner's target remains a 5–15-second melodic hum with short notes.

## Correction to the recent plan

The workspace already contains an October HumTrans intake, official metadata and
a development extension under `datasets/humtrans/`. The recent V48 work failed to
inventory that location before proposing new acquisition. Network attempts failed,
but that did not mean local data was absent. The existing
[17 October record](Presentation_Readiness_2026-10-17.md) already documents the
intake and an unsuccessful long-note training experiment.

Historical receipt paths refer to the former `bpdr_input/data/humtrans/` location.
`tools/reconcile_local_humtrans.py` maps those paths to the current dataset folder
without changing the original receipts, source bytes or source roles. It verifies
24 WAV hashes and 24 individual MIDI hashes, the retained license card, MIDI
archive, repository revision, intake declaration and extension-parent linkage.
Hashing reference files does not decode notes or pass them to the repair model.

## Roles and prior use that must remain authoritative

| Role | People | Existing source recordings | Status |
|---|---|---:|---|
| Training | F01, F02, F03, F04, F05, M01 | 12 | Previously used for exploratory engineering; not fresh confirmation |
| Development | M02, M03 | 12 | Four initial plus eight score-selected long-note extension sources; previously inspected |
| Reserved unseen | M04, M05 | No acquired source in this intake | Keep closed; no outcome claim |

Training/development people and composition IDs do not overlap. The extension is
selected for long score notes and is not representative of all humming. Twelve
training recordings from six people remain a small engineering set. Their
`training_eligible=False` research status is retained: public permission and
controlled recovery do not supply missing human musical-reference approval.

The recently proposed V48 balanced F/M split conflicts with these established
roles: for example, F05 has already been used for training. It cannot become a
fresh confirmation person. Both V48 command entry points now refuse a new
reservation/acquisition when the historical declaration exists. The pure metadata
auditor has an inventory-only mode that does not assign new roles. Thirteen focused
metadata/range/acquisition contract tests pass after adding this guard behavior.

## Actual official-metadata findings

The locally retained official JSON has 13,080 TRAIN, 765 VALID and 769 TEST keys.
All ten people occur in each official split. Every pair of official splits has
zero shared composition IDs and zero shared person/composition/segment families.
Thus official metadata does provide composition-ID separation, while it does not
provide singer separation. This actual audit replaces the earlier unverified
assumption of random-file composition leakage. Distinct composition IDs still
do not prove musical-content nonduplication.

The official repository snapshot and intake both pin revision
`c3c8504280cbdef0f00bb8119f16f034bea2bd9f`; the retained card is CC BY-NC4.0.
Source attribution/restrictions and score-versus-performed-reference limitations
remain in the earlier record. Do not publish source recordings in shared demo
artifacts or use guide MIDI at runtime.

## Current implementation step

`tools/prepare_humtrans_short_training.py` declares 192 physical interventions
before running: twelve existing training recordings, four support widths
(0.24/0.28/0.36/0.50 seconds), four signed shifts (-30/-20/+20/+30 cents).
It uses the first ten seconds without padding, the highest-energy native channel,
22050Hz resampling and PCM24 working audio, with no normalization. Invalid input
remains excluded rather than being replaced.

Support is centered in the longest input-only reliable acoustic region, with
150ms source context on each side. The previously calibrated experimental V44
three-tracker protocol audits decoded interventions. At least three retained
plateau observations and three input-C anchors on each side are required. Preserve
known-zero frames separately from the injected plateau; runtime inputs never
contain source reference, support labels or guide notes.

The current construction process uses training audio only. It does not reopen
development or reserved-unseen audio and does not decode guide notes. Construction
success measures the ability to make an audited training example, not model accuracy.

**Completed construction and independent replay:** 61/192 pairs are valid, from
F02 (9), F03 (16), F04 (28) and F05 (8). By support width, valid/attempted counts
are20/48 at0.24s,18/48 at0.28s,15/48 at0.36s and8/48 at0.50s. There are80
source-context exclusions,48 invalid-source variants and3 physical-audit failures.
Nine of twelve originals are valid. F01's second recording lasts4.936s; both M01
recordings have full-scale samples already at native44.1kHz and overshoot further
after resampling. No gain change, clipping repair, padding or favorable replacement
was introduced. F01 and M01 have no valid positive pair and remain unassessed.

Independent replay recreates nine working originals,128 construction exclusions,
64 decoded audits,128 unchanged/opposite controls,1,441 arrays and61 label files.
The receipt is `2026-11-02_v50_humtrans_short_training/integrity_receipt.json`.

An independent replay tool, `tools/verify_humtrans_short_training.py`, recreates
preprocessing, source/context arrays, decoded three-tracker audits, unchanged and
opposite-request controls, geometry exclusions and supervision masks before fitting.
The completed receipt supports these construction checks only; it does not supply
human reference approval, model accuracy or usable-hum qualification.

## Matched transfer comparison now running

`tools/compare_humtrans_short_transfer.py` declares four new fits using the same
21,666-parameter Transformer, seed339,1,200 updates, optimizer and paired-response/
preservation loss as V46's physical arm. It reuses the four verified V46 checkpoints
as frozen Vocadito-only parents rather than refitting them. The new arm adds the
61 verified real-hum pairs to115 Vocadito pairs, preserving all528 physical attempts
and33 attempted originals (28 valid controls). The only experimental change is
the training pool and its resulting singer-balanced sample draws.

The four existing Vocadito group folds are retained. Sorted historical hum training
identities are assigned round-robin to those folds without using outcomes; the new
arm excludes each held-out hum person as well as the corresponding Vocadito people.
Frozen parents have never fitted these HumTrans pairs. Prior historical HumTrans
engineering use is retained, so this is not fresh unfamiliar confirmation.

Report HumTrans and Vocadito outcomes separately as well as jointly. Do not hide
F01/M01 or other source groups with no positive pair. Reload all eight checkpoints,
replay four new sampler traces and reinfer held-out arrays before accepting even
a raw prediction result. Actual delivered-WAV, known-zero preservation and original
identity remain separate qualification requirements. No result from this running
comparison is claimed yet.

V52's delivered-output policy was locked before final V51 raw comparison results.
It retains the V47 input-only proposal policy,20ms ramp/70ms guard,three-tracker
audit, known-zero harm definition and original-identity requirement. Both arms
will use the identical expanded528-attempt cohort (176 scoreable cases,352
construction exclusions,28 original controls). Raw-score improvements cannot
replace this delivered-output check. The evaluator and independent re-render/
re-measure verifier are implemented; execution waits for the completed, independently
verified V51 checkpoints. Neither this declaration nor the code proves an output gain.

## Evidence and next decision

- `results/transformer_usability/2026-11-02_v49_local_humtrans/reconciliation.json`
  records verified hashes, remapped paths, historical roles and official overlap counts.
- `results/transformer_usability/2026-11-02_v50_humtrans_short_training/` records
  the prospective declaration, all attempted cases, preprocessing, exclusions and
  completed independent replay receipt.
- `results/transformer_usability/2026-11-02_v51_humtrans_transfer/` records the
  matched comparison declaration and new grouped fitting outputs as they complete.
- `results/transformer_usability/2026-11-02_v52_humtrans_delivery/declaration.json`
  fixes the actual-output policy before final raw scores.
- V48's failed-download status remains historical evidence. It is not a current
  claim that the whole workspace lacks humming data.

After replay, inspect duration/person coverage and expressive untouched controls.
Only then declare a matched retraining comparison with the new real-hum training
examples. Preserve the old Vocadito roles and historical development use. A new
fit must still qualify on actual delivered audio and unchanged originals; the
Transformer research hypothesis and natural-intention/listening evidence remain
unproven. Network recovery may be needed for additional recordings later, but is
not a prerequisite for this existing-data construction step.
