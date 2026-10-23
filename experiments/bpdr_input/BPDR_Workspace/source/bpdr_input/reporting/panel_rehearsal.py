"""Rehearse existing inference paths once; never fit or replace a model."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import uuid

import numpy as np
import soundfile as sf

from bpdr_input import COMPONENT_ROOT
from bpdr_input.application.saved_results import checksum, noise_example, pitch_example


def run():
    from bpdr_input.repair.model_runtime import default_checkpoint, run as infer
    from bpdr_input.repair.noise_preview import run as clean
    from bpdr_input.repair.noise_policy_v2 import ColoredNoisePolicy

    workspace = COMPONENT_ROOT / 'BPDR_Workspace'
    destination = workspace / 'results/panel_demo/rehearsal' / uuid.uuid4().hex
    destination.mkdir(parents=True)
    checkpoint = default_checkpoint()
    original_hash = checksum(checkpoint)
    pitch, noise = pitch_example(), noise_example()
    silence = destination / 'silent_input.wav'
    sf.write(silence, np.zeros(22050, dtype=np.float32), 22050, subtype='PCM_24')
    cases = [
        ('known_detuning', pitch['input'], lambda source: infer(source, destination / 'pitch', render_preview=True), 'VERIFIED_FIXTURE_CANDIDATE'),
        ('real_humming_added_rain', noise['input'], lambda source: clean(source, (0., .7), output_root=destination / 'noise', noise_policy=ColoredNoisePolicy()), 'NOISE_PREVIEW_PASSED'),
        ('unchanged_generated_original', pitch['reference'], lambda source: infer(source, destination / 'unchanged', render_preview=True), 'IDENTITY'),
        ('silence', silence, lambda source: infer(source, destination / 'silence'), 'NOT_RUN_TECHNICAL_STOP'),
    ]
    rows = []
    for label, source, operation, expected in cases:
        started = time.perf_counter()
        before = checksum(source)
        try:
            directory, report = operation(source)
            row = dict(id=label, directory=str(directory), outcome=report['outcome'], expected=expected,
                       handoff_status=report['handoff']['status'], source_preserved=checksum(source) == before,
                       elapsed_seconds=round(time.perf_counter() - started, 3),
                       report_sha256=checksum(directory / 'model_report.json'))
            row['passed'] = row['outcome'] == expected and row['source_preserved']
        except Exception as exc:
            row = dict(id=label, expected=expected, passed=False, error_type=type(exc).__name__, reason=str(exc))
        rows.append(row)
        print(json.dumps(row), flush=True)
    model_unchanged = checksum(checkpoint) == original_hash
    summary = dict(created_at=datetime.now(timezone.utc).isoformat(), scope='Fresh engineering rehearsal; no training or musical approval',
                   checkpoint_sha256=original_hash, model_unchanged=model_unchanged, cases=rows,
                   passed=model_unchanged and all(row['passed'] for row in rows))
    (destination / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (workspace / 'results/panel_demo/rehearsal_current.json').write_text(json.dumps(dict(path=str(destination)), indent=2), encoding='utf-8')
    print(json.dumps(dict(summary=str(destination / 'summary.json'), passed=summary['passed'])), flush=True)
    return summary


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    raise SystemExit(0 if run()['passed'] else 1)
