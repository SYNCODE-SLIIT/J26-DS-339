"""Automated real-hum UI workflow, with honest fixture confirmations.

No microphone capture, participant judgment, fresh confirmation or fitting.
"""

import argparse
from contextlib import ExitStack
import io
import json
import os
from pathlib import Path
import platform
import time
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch
from streamlit.testing.v1 import AppTest

from bpdr_input import COMPONENT_ROOT
from bpdr_input.audio.pipeline import checksum
from bpdr_input.application.saved_results import noise_example, paired_snr_gain
from bpdr_input.repair.demo_runtime import METHODS
from bpdr_input.repair import primary_handoff as primary
from bpdr_input.repair.review_export import export_reviewed as actual_export


WS = primary.WORKSPACE
ROOT = WS / 'results/demo_sessions/2026-11-08_real_hum_workflow'
ENTRY = WS / 'source/bpdr_input/application/review_demo_app.py'


def canonical(value):
    return json.loads(json.dumps(value, allow_nan=False))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def button(app, label):
    return next(b for b in app.button if b.label == label)


def assert_ok(app):
    assert not app.exception, [str(e) for e in app.exception]
    assert not app.error, [e.value for e in app.error]


def export_and_consume(app, role):
    next(r for r in app.radio if r.label == 'Version to hand off').set_value(role).run()
    assert_ok(app)
    next(c for c in app.checkbox if c.label.startswith('I listened')).check().run()
    assert_ok(app)
    button(app, 'Create accepted demo package').click().run()
    assert_ok(app)
    exported = app.session_state['demo_export']
    path = Path(exported['directory'])
    diagnostics = json.loads((path / 'diagnostics.json').read_text())
    assert diagnostics['confirmation_kind'] == 'automated_test_fixture'
    assert diagnostics['user_confirmed'] is False
    button(app, 'Try the accepted audio with the melody extractor').click().run()
    assert_ok(app)
    consumed = app.session_state['demo_extraction']
    selected = exported['handoff']['accepted_audio']
    assert consumed['success'] and consumed['consumer_sha256'] == selected['checksum']
    assert checksum(selected['path']) == consumed['consumer_sha256']
    assert consumed['human_review_claimed'] is False
    return dict(package=str(path), status=exported['handoff']['status'],
                accepted_sha256=selected['checksum'], consumer=consumed)


def run_cases(clean, noise):
    audio_payload = io.BytesIO(clean.read_bytes())
    audio_payload.name = 'humtrans_training_phrase.wav'
    records = [dict(id='workflow_original', label='Known training hum', input=clean, reference=None,
                    source_kind='Already opened training-role HumTrans phrase; no guide input.', method=METHODS[3]),
               dict(id='noise', label='Known controlled rain hum', input=noise['input'], reference=noise['reference'],
                    source_kind='The same training phrase with digitally added rain and a declared noise-only prefix.', method=METHODS[2]),
               dict(id='silence', label='Constructed silence', input=None, reference=None,
                    source_kind='Synthetic refusal fixture.', method=METHODS[3])]
    actual_consume = primary.run

    def fixture_export(*args, **kwargs):
        kwargs['confirmation_kind'] = 'automated_test_fixture'
        return actual_export(*args, **kwargs)

    def fixture_consume(*args, **kwargs):
        kwargs['allow_test_fixture'] = True
        return actual_consume(*args, **kwargs)

    cases = []
    with ExitStack() as stack:
        stack.enter_context(patch('bpdr_input.application.demo_examples.examples', return_value=records))
        stack.enter_context(patch('bpdr_input.repair.demo_runtime.available_methods', return_value=METHODS))
        stack.enter_context(patch('streamlit.file_uploader', return_value=audio_payload))
        stack.enter_context(patch('streamlit.audio_input', return_value=audio_payload))
        stack.enter_context(patch('bpdr_input.repair.review_export.export_reviewed', side_effect=fixture_export))
        stack.enter_context(patch('bpdr_input.repair.primary_handoff.run', side_effect=fixture_consume))
        for route in ['Upload audio', 'Record a voice']:
            started = time.perf_counter()
            app = AppTest.from_file(str(ENTRY), default_timeout=180)
            app.run()
            next(r for r in app.radio if r.label == 'Input').set_value(route).run()
            next(s for s in app.selectbox if s.label == 'What did you intend?').select_index(0)
            next(s for s in app.selectbox if s.label == 'Processing method').set_value(METHODS[3])
            button(app, 'Process recording').click().run()
            assert_ok(app)
            result = app.session_state['demo_result']
            report = result['report']
            assert report['handoff']['status'] == 'ACCEPT_UNCHANGED'
            assert result['review']['original']['decision'] == 'PASS_EVIDENCE'
            assert not report.get('candidate')
            assert checksum(report['source']) == checksum(clean)
            accepted = report['handoff']['accepted_audio']
            assert accepted['sample_count'] / accepted['sample_rate_hz'] == 8.
            handoff = export_and_consume(app, 'original')
            cases.append(dict(route=route, source_sha256=checksum(clean), outcome=report['outcome'],
                working_sha256=accepted['checksum'], handoff=handoff,
                seconds=time.perf_counter() - started, real_microphone_capture=False))
            print('REAL_HUM_WORKFLOW_CASE', route, flush=True)
        started = time.perf_counter()
        app = AppTest.from_file(str(ENTRY), default_timeout=180)
        app.run()
        next(s for s in app.selectbox if s.label == 'Recording').set_value(1).run()
        button(app, 'Process recording').click().run()
        assert_ok(app)
        result = app.session_state['demo_result']
        assert result['report']['outcome'] == 'NOISE_PREVIEW_PASSED', result['report']
        candidate = result['report']['noise_candidate']
        assert result['review']['preview']['decision'] == 'PASS_EVIDENCE'
        next(n for n in app.number_input if n.label == 'Protect until (seconds)').set_value(.7)
        button(app, 'Protect this range').click().run()
        assert_ok(app)
        protection = app.session_state['demo_result']['report']['user_protection']
        assert protection['preview_vetoed'] and protection['changed_protected_samples'] > 0
        button(app, 'Undo last review change').click().run()
        assert_ok(app)
        result = app.session_state['demo_result']
        assert result['report']['noise_candidate']['sha256'] == candidate['sha256']
        accepted = result['report']['handoff']['accepted_audio']
        before, sr0 = sf.read(accepted['path'], dtype='float64')
        after, sr1 = sf.read(candidate['path'], dtype='float64')
        reference, sr2 = sf.read(noise['reference'], dtype='float64')
        assert sr0 == sr1 == sr2 and len(before) == len(after) == len(reference)
        gain = paired_snr_gain(reference, before, after)
        assert gain > 0.
        handoff = export_and_consume(app, 'preview')
        assert handoff['status'] == 'ACCEPT_PREPARED'
        assert handoff['accepted_sha256'] == candidate['sha256'] != accepted['checksum']
        cases.append(dict(route='Controlled rain example', source_sha256=checksum(noise['input']),
            working_sha256=accepted['checksum'], candidate_sha256=candidate['sha256'],
            snr_gain_db=gain, preservation_audit=result['report']['noise']['audit'],
            protection_veto=protection, handoff=handoff, seconds=time.perf_counter() - started))
        print('REAL_HUM_WORKFLOW_CASE', 'controlled rain', flush=True)
        next(s for s in app.selectbox if s.label == 'Recording').set_value(2).run()
        button(app, 'Process recording').click().run()
        assert_ok(app)
        refused = app.session_state['demo_result']['report']['handoff']
        assert refused['status'] == 'RERECORD_REQUIRED' and refused['accepted_audio'] is None
        assert not any(b.label == 'Create accepted demo package' for b in app.button)
        assert not any(b.label == 'Try the accepted audio with the melody extractor' for b in app.button)
        cases.append(dict(route='Constructed silence', status=refused['status'],
                          accepted_audio=None, reasons=refused['reason_codes'], feedback=refused['user_message']))
        print('REAL_HUM_WORKFLOW_CASE', 'silence refusal', flush=True)
    return canonical(cases)


def summary(cases):
    # Paths and timing vary; consequential audio and policy results must match.
    result = []
    for case in cases:
        row = {k: v for k, v in case.items() if k not in ('seconds', 'handoff')}
        if 'handoff' in case:
            row['status'] = case['handoff']['status']
            row['accepted_sha256'] = case['handoff']['accepted_sha256']
            row['consumer_sha256'] = case['handoff']['consumer']['consumer_sha256']
        result.append(row)
    return result


def main(mode):
    torch.set_num_threads(2)
    ROOT.mkdir(exist_ok=True)
    clean = WS / 'datasets/humtrans/subset/train/F01_0001_0001_1_D/original.wav'
    noise = noise_example()
    assert sf.info(clean).duration == 8. and sf.info(noise['input']).duration == 8.8
    sources = [Path(__file__), ENTRY,
        WS / 'source/bpdr_input/repair/review_export.py', Path(primary.__file__),
        WS / 'source/bpdr_input/repair/demo_runtime.py', WS / 'source/bpdr_input/audio/pipeline.py',
        WS / 'source/bpdr_input/repair/noise_preview.py', WS / 'source/bpdr_input/repair/noise_policy_v2.py']
    protocol = dict(source_hashes={str(p): checksum(p) for p in sources},
        input_hashes={str(p): checksum(p) for p in [clean, Path(noise['input']), Path(noise['reference'])]},
        primary_freeze_sha256=primary._freeze(),
        hypothesis='Existing reference-free hum input and bounded spectral preview survive UI review/export into the exact selected consumer waveform; silence is refused.',
        data_role='One already opened training-role HumTrans phrase F01_0001_0001_1_D, its old controlled rain derivative, and constructed silence. No new participant or confirmation source.',
        budget='Four processing routes and three real consumer calls per create/verify process. Eight route executions/six consumer calls total; no fitting.',
        mocked='Only incoming upload/audio_input bytes, selectable menu inventory, fixture confirmation kind and fixture permission. Processing, acoustic review, numerical audits, export and consumer are real.',
        criteria='Preserved original; positive paired SNR for fixed rain example with existing preservation audit; sample/time integrity; protection/undo; exact selected/consumer hash; no accepted path on refusal.',
        stopping='One fixed workflow. Fail without retuning or selecting another source.',
        real_microphone_capture=False, human_judgment=False, fresh_confirmation=False)
    declaration = ROOT / 'declaration.json'
    if declaration.exists():
        assert json.loads(declaration.read_text()) == protocol
    else:
        assert mode == 'create'
        write(declaration, protocol)
        snapshots = ROOT / 'source_snapshots'
        snapshots.mkdir(exist_ok=True)
        for source in sources:
            (snapshots / source.name).write_bytes(source.read_bytes())
    cases = run_cases(clean, noise)
    result = dict(declaration_sha256=checksum(declaration), cases=cases, summary=summary(cases),
        hardware=dict(processor=platform.processor(), processor_identifier=os.environ.get('PROCESSOR_IDENTIFIER'),
            logical_cpus=os.cpu_count(), torch_threads=torch.get_num_threads(), concurrent_training_session=36082),
        timing_scope='Concurrent engineering execution, not isolated cold/warm deployment latency.',
        human_judgment=False, real_microphone_capture=False, production_qualified=False)
    path = ROOT / 'results.json'
    if mode == 'create':
        write(path, result)
    else:
        saved = json.loads(path.read_text())
        assert saved['summary'] == result['summary']
        write(ROOT / 'integrity_receipt.json', dict(results_sha256=checksum(path),
            declaration_sha256=checksum(declaration), routes_replayed=4, real_consumer_calls_total=6,
            consequential_results_match=True, human_judgment=False, real_microphone_capture=False))
    print('REAL_HUM_WORKFLOW_COMPLETE', mode, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['create', 'verify'], required=True)
    main(parser.parse_args().mode)
