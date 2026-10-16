"""Bounded training-only screen diagnosis and portable export request; no fitting."""
import csv
import json
from pathlib import Path
import uuid
import zipfile

import numpy as np

from .shared_exports import digest, load_export
from .shared_export_audit import pipeline_hashes


STAGES = ('inside_note', 'single_note', 'voiced', 'finite_trackers', 'tracker_agreement', 'note_agreement')


def decompose(loaded, yin):
    """Replay fixed 50/100-cent rules with exclusive first-failure accounting."""
    f0 = loaded['features']['f0_hz']
    yin = np.asarray(yin)
    if yin.shape != f0.shape:
        raise ValueError('Independent pitch grid mismatch')
    note = loaded['note_hz']
    stages = [loaded['inside_notes'].copy()]
    stages.append(stages[-1] & (loaded['overlap'] == 1))
    stages.append(stages[-1] & loaded['model_inputs']['voiced_mask'])
    stages.append(stages[-1] & np.isfinite(yin) & (yin > 0) & np.isfinite(f0) & (f0 > 0) & (note > 0))
    tracker_delta = np.full(len(f0), np.nan)
    note_delta = np.full(len(f0), np.nan)
    valid = stages[-1]
    tracker_delta[valid] = 1200 * np.log2(yin[valid] / f0[valid])
    note_delta[valid] = 1200 * np.log2(f0[valid] / note[valid])
    stages.append(stages[-1] & (abs(tracker_delta) <= 50))
    stages.append(stages[-1] & (abs(note_delta) <= 100))
    if not np.array_equal(stages[-1], loaded['masks']['reliable_pitch_screen']):
        raise ValueError('Saved reliability mask does not replay exactly')
    counts = {key: int(mask.sum()) for key, mask in zip(STAGES, stages)}
    losses = {STAGES[i]: int((stages[i-1] & ~stages[i]).sum()) for i in range(1, len(stages))}
    assert sum(losses.values()) + counts['note_agreement'] == counts['inside_note']
    changes = np.diff(np.r_[False, stages[-1], False].astype(int))
    lengths = np.flatnonzero(changes == -1) - np.flatnonzero(changes == 1)
    longest = int(lengths.max()) if len(lengths) else 0
    return dict(counts=counts, first_failure_losses=losses, mask_exact=True,
                longest_reliable_run_frames=longest,
                longest_reliable_endpoint_span_seconds=max(0, longest-1)*.02), dict(
                    yin_hz=yin, tracker_delta_cents=tracker_delta, note_delta_cents=note_delta,
                    **{key: mask for key, mask in zip(STAGES, stages)})


def diagnose_training_crop(row):
    # Guard before opening any crop or calling a tracker.
    if row['split'] != 'train':
        raise ValueError('Screen diagnosis is restricted to the frozen training partition')
    from .interventions import independent_pitch
    loaded = load_export(Path(row['record_path']).parent)
    result, arrays = decompose(loaded, independent_pitch(loaded['audio'], loaded['sample_rate_hz']))
    return loaded, result, arrays


def missing_training_tracks(evidence):
    candidates = [r for r in evidence['tracks'] if r['split'] == 'train' and r['policy_export_eligible']
                  and not r['bpdr_record_present'] and not r['issues']]
    candidates.sort(key=lambda r: int(r['track_id']))
    first = {}
    for row in candidates:
        first.setdefault(row['artist_id'], row['mtg_track_id'])
    return candidates, set(first.values())


def run():
    root = Path(__file__).resolve().parents[2]
    intake = json.loads((root/'runs/shared_export_intake_current.json').read_text())
    intake = Path(intake['path'])
    source = intake/'evidence.json'
    evidence = json.loads(source.read_text())
    if evidence['run_status'] != 'complete':
        raise ValueError('Intake must be complete')
    pipeline = root.parents[1]/'datasets/mtg-jamendo/melody-chord-extraction-pipeline'
    baseline = pipeline_hashes(pipeline)
    if baseline != evidence['shared_pipeline_files']:
        raise ValueError('Common pipeline differs from intake snapshot')
    reviews = [root/'runs/phase2/15fb678d1f44406ab4da3fddd7c2f809/reviews.csv',
               root/'runs/preparation/d274ed728d3d46f3b2661ba5291e61da/reviews.csv']
    frozen = {str(p): digest(p) for p in [source, Path(__file__), *reviews]}
    annotation_path = next(Path(p) for p in evidence['tracked_inputs'] if Path(p).name == 'autotagging.tsv')
    frozen[str(annotation_path)] = evidence['tracked_inputs'][str(annotation_path)]
    frozen.update({r['record_path']: r['record_sha256'] for r in evidence['crops']})
    for path, sha in frozen.items():
        if digest(path) != sha:
            raise ValueError('Frozen source changed: '+path)
    destination = root/'runs/shared_screen_diagnostics'/uuid.uuid4().hex
    destination.mkdir(parents=True)
    def write(name, data):
        (destination/name).write_text(json.dumps(data, indent=2)+'\n', encoding='utf-8')
    write('declaration.json', dict(source=str(source), source_sha256=digest(source),
        frozen_inputs=frozen, pipeline_files=baseline, partitions='training only; exclude holdout before tracker call',
        stages=STAGES, tracker_agreement_cents=50, note_agreement_cents=100,
        tasks=['Replay existing YIN screen, decompose fixed mask losses, retain arrays.',
               'Prepare training-only data handoff without running/changing common pipeline.'],
        no_training=True, no_threshold_changes=True, no_new_pairs=True, no_audio_edits=True))
    rows = []
    artifacts = {}
    for row in evidence['crops']:
        if row['split'] != 'train':
            continue
        loaded, result, arrays = diagnose_training_crop(row)
        path = destination/(row['mtg_track_id']+'.npz')
        np.savez_compressed(path, **arrays)
        artifacts[path.name] = digest(path)
        rows.append(dict(mtg_track_id=row['mtg_track_id'], artist_id=row['artist_id'], split='train',
            record_path=row['record_path'], record_sha256=row['record_sha256'],
            audio_sha256=loaded['record']['artifacts']['vocal.wav'],
            screen=loaded['record']['screen'], arrays=path.name, **result))
        print('REPLAYED', row['mtg_track_id'], result['counts'], flush=True)
    candidates, priority = missing_training_tracks(evidence)
    with annotation_path.open(encoding='utf-8', newline='') as f:
        reader = csv.reader(f, delimiter='\t');next(reader)
        annotation_paths = {r[0]: r[3] for r in reader}
    previous = json.loads((intake/'bpdr_export_priority.json').read_text())
    if priority != {r['mtg_track_id'] for r in previous['tracks']}:
        raise ValueError('Priority differs from previously declared metadata-only choice')
    bundle = destination/'handoff'
    bundle.mkdir()
    for filename, selected in [('priority_track_ids.txt', [r for r in candidates if r['mtg_track_id'] in priority]),
                                ('all_missing_training_track_ids.txt', candidates)]:
        (bundle/filename).write_text(''.join(str(r['track_id'])+'\n' for r in selected), encoding='utf-8')
    fields = ['track_id','mtg_track_id','artist_id','split','priority_first_per_artist','annotation_path','license_url']
    with (bundle/'tracks.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields);writer.writeheader()
        for row in candidates:
            number = int(row['track_id'])
            writer.writerow(dict(track_id=number, mtg_track_id=row['mtg_track_id'], artist_id=row['artist_id'],
                split='train', priority_first_per_artist=row['mtg_track_id'] in priority,
                annotation_path=annotation_paths[row['mtg_track_id']], license_url=row['license']['url']))
    instructions = f'''# BPDR data export request — 15 October 2026

Use the existing common extraction pipeline on the capable device. Do not change its code, settings, pitch conventions or other members' outputs for this request.

Start with `priority_track_ids.txt`: {len(priority)} tracks, one lowest numeric missing track per frozen BPDR training artist. `all_missing_training_track_ids.txt` lists all {len(candidates)} missing eligible training tracks as fallback/remaining coverage. This is a data request, not a new official split. Artist IDs are proxies, not verified singer IDs. The selection is fixed by metadata, not pitch outcomes. `tracks.csv` lists IDs, original annotation paths, partitions and recorded license URLs; resolve local MP3 paths through your original manifest.

For each requested track, return the unchanged complete `outputs/track_XXXXXXX/bpdr/` package (`record.json`, `vocal.wav`, `features.npz`, `boundaries.json`, `masks.npz`, and reviews if produced), plus `melody/game_vocal.json` needed to verify its recorded hash. Preserve original folder names, full-song crop offsets, original code/artifact/source hashes and unshifted `model_midi_pitch`. Keep withheld/failed records; if no package is produced, include a separate failure log keyed by track ID. Do not edit status, reviews, masks, targets or split assignments to obtain more candidates.

Include the original root manifest and exact producing Git revision/source snapshot, including `export.py` and `selection.py`, plus the GAME asset revision/hash. The previous ZIP lacked the root manifest and its exact export/selection producing revision could not be matched. Include source MP3/full-stem checksums; source audio is not required merely for this request but missing full audio limits independent byte verification. Keep license attribution information with the data.

Do not rerun melody, chord, key or beat stages that already have valid outputs merely for this handoff. Use the existing pipeline's supported BPDR export/resume operation. No operator command is fabricated here: confirm its invocation in your producing revision. If the expected package cannot be produced without code changes, return that limitation instead of altering the shared pipeline.

Archive outputs with the same relative `outputs/track_XXXXXXX/` layout. Supply the original manifest/revision/failure log alongside the outputs ZIP (the current BPDR importer expects only track outputs inside it). Deliver packages regardless of screening outcome; passing a screen is not musical approval. BPDR will import, verify, screen training-only support and build known-target audited pairs separately. No evaluation artist is requested. Nothing in this bundle has been sent automatically.
'''
    (bundle/'REQUEST.md').write_text(instructions, encoding='utf-8')
    bundle_files = {p.name: digest(p) for p in sorted(bundle.iterdir())}
    write('handoff_manifest.json', dict(files=bundle_files, priority_tracks=len(priority),
        missing_training_tracks=len(candidates), sent=False, common_pipeline_changes=False))
    with zipfile.ZipFile(destination/'bpdr_training_export_request.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(bundle.iterdir()):
            z.write(p, p.name)
    artifacts['bpdr_training_export_request.zip'] = digest(destination/'bpdr_training_export_request.zip')
    with zipfile.ZipFile(destination/'bpdr_training_export_request.zip') as z:
        import hashlib
        assert {name:hashlib.sha256(z.read(name)).hexdigest() for name in z.namelist()} == bundle_files
    counts = {key: sum(r['counts'][key] for r in rows) for key in STAGES}
    losses = {key: sum(r['first_failure_losses'][key] for r in rows) for key in STAGES[1:]}
    for path, sha in frozen.items():
        if digest(path) != sha:
            raise ValueError('Frozen input changed during diagnosis')
    if pipeline_hashes(pipeline) != baseline:
        raise ValueError('Common pipeline changed')
    write('evidence.json', dict(status='complete', declaration_sha256=digest(destination/'declaration.json'),
        source=str(source), source_sha256=digest(source), crops=rows, stage_counts=counts, first_failure_losses=losses,
        training_crops=len(rows), training_artist_proxies=len({r['artist_id'] for r in rows}),
        excluded_nontraining_crops=len(evidence['crops'])-len(rows), all_masks_exact=all(r['mask_exact'] for r in rows),
        priority_training_tracks=len(priority), missing_training_tracks=len(candidates), bundle_files=bundle_files,
        artifacts=artifacts, shared_pipeline_files_unchanged=len(baseline), frozen_inputs=frozen,
        training_updates=0,new_pair_examples=0,research_ready_examples=0,heldout_pitch_inference=False,
        screens_changed=False,adopted=False,
        limitations=['First-failure counts depend on declared gate order; losses are not independent causes.',
                    'Trackers and GAME are predictions; disagreement does not identify which is correct.',
                    'Thirteen crops represent one artist proxy; no population or singer-level inference.',
                    'No threshold relaxation, octave correction, audio intervention, fit or musical approval.']))
    (root/'runs/shared_screen_diagnostics_current.json').write_text(json.dumps(dict(path=str(destination))), encoding='utf-8')
    print('DIAGNOSTICS_RUN:', destination)
    return destination


if __name__ == '__main__':
    run()
