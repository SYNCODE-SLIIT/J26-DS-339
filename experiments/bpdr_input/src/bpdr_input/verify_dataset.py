"""Replay Phase 1 on every full recording; no cropping or accuracy claims."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time
import numpy as np
from .pipeline import process, validate_package, checksum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, required=True)
    component = Path(__file__).resolve().parents[2]
    parser.add_argument('--output-root', type=Path, default=component/'runs'/'phase1_dataset')
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for source in sorted((args.dataset_root/'Audio').glob('*.wav')):
        if source.name.startswith('._'):
            continue
        start = time.perf_counter()
        result, directory = process(source, args.output_root, component/'checkpoints')
        checks = dict(original_unchanged=checksum(source) == checksum(directory/'original.wav'))
        if result.accepted_audio:
            audio = validate_package(result)
            with np.load(directory/'features.npz') as f:
                checks['feature_shapes'] = f['frame_features_70'].shape[1] == 70 and f['boundary_features_71'].shape[1] == 71
                checks['finite_model_features'] = bool(np.isfinite(f['boundary_features_71']).all())
            diagnostics = json.loads((directory/'diagnostics.json').read_text())
            checks['duration_within_one_sample'] = abs(audio.duration_seconds-diagnostics['input']['duration_seconds']) <= 1/audio.sample_rate_hz
        else:
            checks['no_accepted_export'] = not (directory/'accepted.wav').exists()
        row = dict(source=source.name, status=result.status.value, reasons=result.reason_codes,
                   run_directory=str(directory), checks=checks, elapsed_seconds=time.perf_counter()-start)
        rows.append(row)
        print(f'{source.name}: {result.status.value}', flush=True)
        summary = dict(recording_count=len(rows), statuses=dict(Counter(r['status'] for r in rows)),
                       integrity_checks_pass=all(all(r['checks'].values()) for r in rows),
                       interpretation='Execution/integrity evidence only; not musical acceptance or repair accuracy',
                       recordings=rows)
        (args.output_root/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return 0 if rows and summary['integrity_checks_pass'] and not summary['statuses'].get('TECHNICAL_FAILURE') else 1


if __name__ == '__main__':
    raise SystemExit(main())
