"""Declared five-artist extraction/preparation batch with persistent progress and receipts."""
import argparse
import ctypes
import json
from pathlib import Path
import time
import uuid
from .data_sources import inside, stem_receipt
from .separation_adapter import eligible, separate
from .data_builder import build, validate_run
from .preparation_report import render
from .pipeline import checksum


def frozen_selection(component, selection):
    rows = json.loads(Path(selection).read_text())['tracks']
    if len(rows) != 5 or len({r['track_id'] for r in rows}) != 5 or len({r['artist_id'] for r in rows}) != 5:
        raise ValueError('Exactly five distinct tracks/artists required')
    live = {row['track_id']: row for row in eligible(component/'data/mtg-jamendo')}
    for row in rows:
        current = live.get(row['track_id'])
        if current is None or current['artist_id'] != row['artist_id'] or current['license'] != row['license']:
            raise ValueError('Declared selection no longer matches training artists/licenses')
    return rows


def run(component, selection, resume=None):
    component = Path(component).resolve()
    rows = frozen_selection(component, selection)
    root = Path(resume).resolve() if resume else component/'runs/jamendo_sample'/uuid.uuid4().hex
    if not root.is_relative_to(component):
        raise ValueError('Batch evidence must remain inside BPDR')
    root.mkdir(parents=True, exist_ok=bool(resume))
    path = root/'batch.json'
    if resume:
        batch = json.loads(path.read_text())
        if batch['track_ids'] != [row['track_id'] for row in rows]:
            raise ValueError('Cannot change the frozen sample when resuming')
        if checksum(selection) != batch['selection_sha256']:
            raise ValueError('Frozen selection metadata changed; cannot resume')
    else:
        batch = dict(run_status='running', data_status='exploratory_not_research', track_ids=[row['track_id'] for row in rows],
                     selection_path=str(Path(selection).resolve()), selection_sha256=checksum(selection),
                     tracks=[dict(track_id=row['track_id'], artist_id=row['artist_id'], license=row['license'],
                                  source_path=str(inside(component/'data/mtg-jamendo', row['source_mp3'])),
                                  source_sha256=checksum(inside(component/'data/mtg-jamendo', row['source_mp3'])),
                                  status='pending') for row in rows], training_ready_examples=0)
    def save():
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(batch, indent=2), encoding='utf-8')
        temporary.replace(path)
    awake = False
    try:
        awake = bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
    except (AttributeError, OSError):
        pass
    batch['process_keep_awake'] = awake
    batch['run_status'] = 'running'
    save()
    print('BATCH_RUN:', root, flush=True)
    try:
        for item in batch['tracks']:
            track_id = item['track_id']
            if checksum(item['source_path']) != item['source_sha256']:
                raise ValueError('Declared source changed: '+track_id)
            stem = component/'data/jamendo-vocal-stems'/(track_id+'.wav')
            if stem.with_suffix('.receipt.json').exists():
                receipt = stem_receipt(stem, item['source_path'], track_id)
            else:
                item['status'] = 'running'
                save()
                print('START_TRACK:', track_id, flush=True)
                started = time.perf_counter()
                receipt = separate(component, track_id)
                item['elapsed_seconds'] = time.perf_counter()-started
            item.update(status='complete', receipt_path=str(receipt), receipt_sha256=checksum(receipt))
            save()
            print('COMPLETE_TRACK:', track_id, flush=True)
        batch['run_status'] = 'preparing'
        save()
        data_run = Path(json.loads((component/'runs/phase2/latest.json').read_text())['path'])
        prepared, _ = build(component, data_run, vocadito_singers=3, jamendo_artists=5, jamendo_track_ids=batch['track_ids'])
        batch['preparation_run'] = str(prepared)
        batch['verification'] = validate_run(prepared)
        batch['listening_page'] = render(prepared)['path']
        batch['run_status'] = 'complete'
        save()
        print('COMPLETE_BATCH:', root, flush=True)
    except BaseException as exc:
        batch.update(run_status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                     failure=dict(type=type(exc).__name__, reason=str(exc)))
        save()
        raise
    finally:
        if awake:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
    return root


def main():
    component = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    run(component, args.selection, args.resume)


if __name__ == '__main__':
    main()
