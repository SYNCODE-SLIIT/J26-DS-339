# HumTrans preparation for BPDR short-hum repair

**SUPERSEDED:** [existing local HumTrans recovery](Local_HumTrans_Recovery_2026-11-02.md)
found and hash-verified24 previously acquired recordings and their established
roles. The V48 acquisition-only plan below is historical. Its provisional person
split must not be used: F04/F05 were already training people. Command entry points
now refuse that conflicting reservation. Use local training sources first; network
failure does not block their use. Actual official metadata has composition-ID
separation but shares all ten singers across splits.

2 November 2026. Status: primary sources inspected; bounded metadata auditor,
ZIP64 range reader and reserved-pilot acquirer implemented. Twelve focused local
contract checks pass. **No complete metadata
download, actual split audit, reservation, audio acquisition or training is
claimed.** Direct HTTPS acquisition and a fresh retry failed with DNS/connection-reset errors;
browser-readable official pages were accessible, but the browser acquisition
fallback timed out. This is a network acquisition
failure, not an automatic approval rejection.

## Why this source fits the next experiment

The [dataset paper](https://arxiv.org/abs/2309.09623) describes ten musically
experienced participants and short guided melodic hums, closer to the owner's
target than held vowels. Most clips last 7–16 seconds. It does not establish
performance on novice free-form ideas. The
[official card](https://huggingface.co/datasets/dadinghh2/HumTrans/blob/main/README.md)
lists 14,614 recordings, composition/segment/repetition identity in filenames,
and CC BY-NC 4.0. Keep attribution and academic noncommercial use restrictions.
Do not redistribute source audio in the demo or repository.

The supplied MIDI describes the guide melody, not manually verified performed
F0 or a gold record of a free hummer's intention. Use it only for offline
reference-suitability/alignment checks after permission and partition checks.
Intentional octave suffixes are not errors to correct. Runtime repair receives
audio only. This does not add a symbolic melody/transcription feature owned by
another member.

## Metadata-first protocol

Pinned official repository commit:
`c3c8504280cbdef0f00bb8119f16f034bea2bd9f`, verified through the
[official commit page](https://huggingface.co/datasets/dadinghh2/HumTrans/commit/c3c8504280cbdef0f00bb8119f16f034bea2bd9f).
The full audio archive is approximately 14.7 GB; no full download is planned.

`tools/audit_humtrans_metadata.py` downloads only the pinned README (16 KiB cap)
and official split JSON (1 MiB cap), hashes exact bytes and checks filename/schema
contracts. It reports person, composition and repeated-segment family overlap
between official splits. The official random-file split must not be assumed to
provide singer/song holdout. Actual overlap counts remain uncomputed until the
real JSON is acquired; passing synthetic tests is not a dataset audit.

Before acoustic inspection, the new BPDR protocol fixes:

- Training identities F01–F03/M01–M03; development F04/M04; confirmation F05/M05.
- Composition assignment by seeded SHA-256 identity hash, 80/10/10 buckets.
  Both person and composition must match a role; conflicting cells are excluded.
  Repetitions, segments and octave versions of a composition cannot cross roles.
- An initial metadata-selected pilot of eight distinct compositions per training
  person and four per development/confirmation person: intended 48/8/8 clips.
  A shortage or unexpected ID schema stops acquisition for review; it does not
  silently replace people or select favorable audio.
- This is a new grouped BPDR split, not a result on the official benchmark.
  Preserve each source's official split label. No existing Vocadito role changes.
- Reserve confirmation IDs but do not fetch, decode, inspect or tune on their
  waveforms. Source names alone are not successful acoustic validation.

Selection hashes and counts are written only after actual metadata passes the
auditor. An existing different reservation cannot be overwritten. Real audio
duration/readiness failures remain exclusions; do not replace them based on
model outcomes. Confirm MIDI-content/song duplicates before declaring composition
independence; distinct IDs alone cannot prove musical nonduplication.

## After metadata passes

The separately bounded ZIP64/range reader is now implemented in
`tools/bounded_remote_zip.py`, with strict content-range, CRC/hash and safe-entry-path
checks. `tools/acquire_humtrans_pilot.py` replays the metadata reservation and
checks its hash and license snapshot before any network access. It permits only
the 48 training/eight development entries, never confirmation, and enforces a
cumulative128 MiB transfer budget through its acquisition receipt. Acquire
only reserved training and development entries initially. If range requests are
ignored, abort rather than download the full archive. Log exact downloaded bytes,
archive identity, selected entry names and failed requests. The metadata tool
itself does not acquire audio. Live end-to-end acquisition remains untested
because actual pinned metadata has not downloaded.

Local verification completes twelve tests: three metadata/schema/grouping checks,
six archive checks and three acquisition guards. The ZIP64 fixture locates its
central directory beyond4 GiB while transferring less than70,000 bytes; it is a
local sparse fixture, not a successful HumTrans transfer. Other checks exercise
ignored/mismatched ranges, changed archive identity, transfer/file limits, unsafe
names, corrupted CRCs, missing metadata and tampered/confirmation reservations.
An initial unsafe-name fixture exposed Windows `zipfile` normalizing backslashes;
validation now checks the original remote filename as well as its normalized form.

Run from this workspace with its existing Python environment:

```powershell
& ..\.venv\Scripts\python.exe -X utf8 -B tools/audit_humtrans_metadata.py
# Only after the actual metadata audit succeeds:
& ..\.venv\Scripts\python.exe -X utf8 -B tools/acquire_humtrans_pilot.py
```

No invented reservation or synthetic fixture is a substitute for the first step.
Source audio stays in ignored private results, outside shared demo artifacts.

Then measure original acoustic reliability and usable short support, retaining
all planned clip/person/duration denominators. Build permitted physical local
detuning pairs and untouched expressive controls from training only. Do not
require a constant note or use guide MIDI at inference. Compare the frozen
Transformer and a predeclared paired-preservation candidate using singer/source
holdouts before development. Do not alter gates after looking at development.

Qualification still requires preservation, delivered improvement and coverage
for every assessable group, then locked unfamiliar confirmation and independent
listening. Ten experienced participants remain a limited population even if
the pilot succeeds. Follow the separate consent/reviewer collection plan for
novice intended-idea evidence.
