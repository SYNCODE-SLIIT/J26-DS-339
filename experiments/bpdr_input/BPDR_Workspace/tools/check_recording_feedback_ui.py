"""Render feedback on existing real-hum states; do not rerun their processing."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch
from streamlit.testing.v1 import AppTest

from bpdr_input.application import recording_feedback as feedback_module
from bpdr_input.audio.pipeline import checksum
from bpdr_input.repair.demo_runtime import METHODS
from bpdr_input.repair.primary_handoff import WORKSPACE
from bpdr_input.repair.review_export import export_reviewed


ROOT = WORKSPACE / 'results/demo_sessions/2026-11-08_recording_feedback_ui'
ENTRY = WORKSPACE / 'source/bpdr_input/application/review_demo_app.py'
WORKFLOW = ROOT.parent / '2026-11-08_real_hum_workflow'


def canonical(value):
    return json.loads(json.dumps(value, allow_nan=False))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def main(mode):
    torch.set_num_threads(2)
    ROOT.mkdir(exist_ok=True)
    prior = json.loads((WORKFLOW / 'results.json').read_text())
    receipt = json.loads((WORKFLOW / 'integrity_receipt.json').read_text())
    assert receipt['results_sha256'] == checksum(WORKFLOW / 'results.json')
    cases = []
    for index, role in [(0, 'original'), (2, 'preview')]:
        row = prior['cases'][index]
        directory = Path(row['handoff']['package']).parents[1]
        report = json.loads((directory / 'model_report.json').read_text())
        review = json.loads((directory / 'usability_review.json').read_text())
        cases.append(dict(name=row['route'], directory=directory, report=report, review=review,
                          role=role, expected_sha256=row['handoff']['accepted_sha256']))
    silent = ROOT / 'silence.wav'
    invalid = ROOT / 'invalid.wav'
    if not silent.exists():
        sf.write(silent, np.zeros(22050, dtype=np.float32), 22050, subtype='PCM_24')
    if not invalid.exists():
        invalid.write_bytes(b'Explicit malformed-WAV UI fixture')
    for name, path, status in [('Synthetic silent state', silent, 'RERECORD_REQUIRED'),
                               ('Malformed source state', invalid, 'TECHNICAL_FAILURE')]:
        cases.append(dict(name=name, directory=ROOT / path.stem,
            report=dict(source=str(path), outcome='NOT_RUN_TECHNICAL_STOP', handoff=dict(
                status=status, accepted_audio=None, user_message='Synthetic refused-state rendering fixture.')),
            review=dict(original=dict(decision='NOT_ASSESSED', feedback='Synthetic refused-state rendering fixture.')),
            role=None, expected_sha256=None))
    files = [Path(__file__), ENTRY, Path(feedback_module.__file__),
             WORKSPACE / 'source/bpdr_input/repair/review_export.py']
    inputs = {}
    for case in cases:
        inputs[str(Path(case['report']['source']))] = checksum(case['report']['source'])
        for name in ['model_report.json', 'usability_review.json', 'features.npz']:
            path = case['directory'] / name
            if path.exists():
                inputs[str(path)] = checksum(path)
    protocol = dict(source_hashes={str(p): checksum(p) for p in files}, input_hashes=inputs,
        prior_workflow_result_sha256=checksum(WORKFLOW / 'results.json'),
        hypothesis='Input-only time feedback renders without changing statuses or previously consumed WAVs; metadata survives export.',
        data='Two saved real-hum states from one training phrase and its controlled rain derivative; two synthetic refused UI states.',
        budget='Four UI state renders/two fixture exports per create/verify process. No pipeline rerun, new consumer inference, fitting or listening.',
        criteria='Times inside source support; quiet silence localized; malformed source handled; no acceptance changes; unchanged export checksums; exact feedback replay.',
        stopping='One fixed test. No source or threshold selection.',
        human_judgment=False, production_qualified=False)
    declaration = ROOT / 'declaration.json'
    if declaration.exists():
        assert json.loads(declaration.read_text()) == protocol
    else:
        assert mode == 'create'
        write(declaration, protocol)
        snapshots = ROOT / 'source_snapshots'
        snapshots.mkdir(exist_ok=True)
        for path in files:
            (snapshots / path.name).write_bytes(path.read_bytes())
    observations = []
    for case in cases:
        path = Path(case['report']['source'])
        with ExitStack() as stack:
            stack.enter_context(patch('bpdr_input.repair.demo_runtime.available_methods', return_value=METHODS))
            stack.enter_context(patch('bpdr_input.application.demo_examples.examples', return_value=[dict(
                id='feedback_state', label=case['name'], input=path, reference=None,
                source_kind='Previously processed state or explicit synthetic UI fixture.', method=METHODS[3])]))
            app = AppTest.from_file(str(ENTRY), default_timeout=180)
            app.session_state['demo_result'] = dict(directory=str(case['directory']), report=case['report'],
                review=case['review'], method='Saved workflow state', reference=None, known_case=None)
            app.run()
            assert not app.exception, [str(e) for e in app.exception]
            assert not app.error, [e.value for e in app.error]
            result = app.session_state['demo_result']
            assert result['report']['handoff']['status'] == case['report']['handoff']['status']
            if case['name'] == 'Malformed source state':
                assert any('Recording feedback is unavailable' in item.value for item in app.info)
                observations.append(dict(name=case['name'], handled_unavailable=True,
                                         status=result['report']['handoff']['status']))
            else:
                feedback = result['recording_feedback']
                assert not feedback['acceptance_changed'] and not feedback['intended_notes_assessed']
                assert all(0 <= row['start_s'] < row['end_s'] <= feedback['duration_seconds'] for row in feedback['observations'])
                if case['name'] == 'Synthetic silent state':
                    assert len(feedback['observations']) == 1
                    assert feedback['observations'][0]['start_s'] == 0 and feedback['observations'][0]['end_s'] == 1
                if case['role']:
                    package, handoff = export_reviewed(case['directory'], result['report'], result['review'],
                        use_preview=case['role'] == 'preview', confirmed=True, confirmation_kind='automated_test_fixture')
                    assert handoff['accepted_audio']['checksum'] == case['expected_sha256']
                    exported = json.loads((package / 'diagnostics.json').read_text())
                    assert exported['recording_feedback'] == canonical(feedback)
                    assert exported['user_confirmed'] is False
                observations.append(dict(name=case['name'], feedback=feedback,
                    status=result['report']['handoff']['status'], exported_sha256=case['expected_sha256']))
            print('RECORDING_FEEDBACK_UI', mode, case['name'], flush=True)
    result = dict(declaration_sha256=checksum(declaration), observations=observations,
                  no_new_consumer_calls=True, human_judgment=False, production_qualified=False)
    path = ROOT / 'results.json'
    if mode == 'create':
        write(path, result)
    else:
        assert json.loads(path.read_text()) == result
        write(ROOT / 'integrity_receipt.json', dict(results_sha256=checksum(path),
            declaration_sha256=checksum(declaration), ui_states_replayed=4, exports_per_process=2,
            exact_feedback_and_export_hashes=True, human_judgment=False))
    print('RECORDING_FEEDBACK_UI_COMPLETE', mode, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['create', 'verify'], required=True)
    main(parser.parse_args().mode)
