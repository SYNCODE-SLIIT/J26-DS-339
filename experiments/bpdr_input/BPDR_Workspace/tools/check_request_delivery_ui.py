"""Replay request/permission/selected-WAV summaries on already opened states."""

import argparse
import copy
from contextlib import ExitStack
import json
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from bpdr_input.application import request_delivery
from bpdr_input.audio.pipeline import checksum
from bpdr_input.repair.demo_runtime import METHODS
from bpdr_input.repair.primary_handoff import WORKSPACE
from bpdr_input.repair.review_export import export_reviewed


ROOT = WORKSPACE / 'results/demo_sessions/2026-11-08_request_delivery_ui'
ENTRY = WORKSPACE / 'source/bpdr_input/application/review_demo_app.py'


def canonical(value):
    return json.loads(json.dumps(value, allow_nan=False))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def main(mode):
    ROOT.mkdir(exist_ok=True)
    legacy_path = ROOT.parent / 'ui_recovery_check.json'
    legacy = json.loads(legacy_path.read_text())
    assert legacy['passed'] and legacy['human_musical_judgment'] is False
    prior_path = ROOT.parent / '2026-11-08_real_hum_workflow/results.json'
    prior = json.loads(prior_path.read_text())
    prior_receipt = json.loads(prior_path.with_name('integrity_receipt.json').read_text())
    assert prior_receipt['results_sha256'] == checksum(prior_path)
    folders = [('Legacy pitch preview', Path(legacy['learned']['directory']), ['original', 'preview']),
        ('Training hum', Path(prior['cases'][0]['handoff']['package']).parents[1], ['original']),
        ('Controlled rain', Path(prior['cases'][2]['handoff']['package']).parents[1], ['original', 'preview'])]
    cases = []
    inputs = {str(legacy_path): checksum(legacy_path), str(prior_path): checksum(prior_path)}
    for name, directory, roles in folders:
        report = json.loads((directory / 'model_report.json').read_text())
        review = json.loads((directory / 'usability_review.json').read_text())
        for role in roles:
            cases.append(dict(name=name + ' / ' + role, directory=directory, role=role,
                              report=copy.deepcopy(report), review=copy.deepcopy(review)))
        for filename in ['model_report.json', 'usability_review.json', 'model_predictions.npz', 'features.npz']:
            path = directory / filename
            if path.exists():
                inputs[str(path)] = checksum(path)
        for item in [report['handoff']['accepted_audio'], report.get('noise_candidate') or report.get('candidate')]:
            if item:
                inputs[item['path']] = checksum(item['path'])
    withheld = copy.deepcopy(cases[0])
    withheld['name'] = 'Synthetic withheld state / original'
    withheld['report']['outcome'] = 'ROLLED_BACK'
    withheld['review'].pop('preview', None)
    cases.append(withheld)
    sources = [Path(__file__), ENTRY, Path(request_delivery.__file__),
               WORKSPACE / 'source/bpdr_input/repair/review_export.py']
    protocol = dict(source_hashes={str(p): checksum(p) for p in sources}, input_hashes=inputs,
        hypothesis='Current role changes select exact WAV summaries; requests and permitted previews do not imply delivery.',
        data='Already opened legacy failed-generalization pilot, training HumTrans phrase/rain derivative and one synthetic withheld state. No fresh sources.',
        budget='Six state/role renders and six automated exports per create/verify process; no processing, new model or consumer inference.',
        criteria='Original selection has zero changed samples; preview samples are compared directly; stale metadata refused; all exported hashes and summaries replay exactly.',
        stopping='One fixed UI check; no score-based source selection or gate change.',
        human_judgment=False, research_qualification=False)
    declaration = ROOT / 'declaration.json'
    if declaration.exists():
        assert json.loads(declaration.read_text()) == protocol
    else:
        assert mode == 'create'
        write(declaration, protocol)
        snapshots = ROOT / 'source_snapshots'
        snapshots.mkdir(exist_ok=True)
        for path in sources:
            (snapshots / path.name).write_bytes(path.read_bytes())
    records = []
    for case in cases:
        with ExitStack() as stack:
            stack.enter_context(patch('bpdr_input.repair.demo_runtime.available_methods', return_value=METHODS))
            stack.enter_context(patch('bpdr_input.application.demo_examples.examples', return_value=[dict(
                id='saved_summary', label=case['name'], input=Path(case['report']['source']), reference=None,
                source_kind='Saved numerical state; no new inference or qualification.', method=METHODS[3])]))
            app = AppTest.from_file(str(ENTRY), default_timeout=180)
            app.session_state['demo_result'] = dict(directory=str(case['directory']), report=case['report'],
                review=case['review'], method='Saved numerical state', reference=None, known_case=None)
            app.run()
            next(r for r in app.radio if r.label == 'Version to hand off').set_value(case['role']).run()
            assert not app.exception, [str(e) for e in app.exception]
            assert not app.error, [e.value for e in app.error]
            result = app.session_state['demo_result']
            summary = result['report']['delivery_summary']
            assert summary['selected_role'] == case['role']
            if case['role'] == 'original':
                assert summary['actual_changed_samples'] == 0 and summary['selected_request_max_abs_cents'] == 0.
            else:
                assert summary['actual_changed_samples'] > 0
            if case['name'].startswith('Synthetic withheld'):
                assert summary['permitted_preview_request_max_abs_cents'] == 0.
                assert summary['model_request_max_abs_cents'] > 0.
            if case['name'] == 'Controlled rain / preview':
                assert summary['preview_operation'] == 'whole_clip_cleanup'
                assert summary['permitted_preview_request_max_abs_cents'] is None
            package, handoff = export_reviewed(case['directory'], result['report'], result['review'],
                use_preview=case['role'] == 'preview', confirmed=True, confirmation_kind='automated_test_fixture')
            assert handoff['accepted_audio']['checksum'] == summary['selected_sha256'] == checksum(package / 'accepted.wav')
            diagnostics = json.loads((package / 'diagnostics.json').read_text())
            assert diagnostics['delivery_summary'] == canonical(summary) and diagnostics['user_confirmed'] is False
            records.append(dict(name=case['name'], summary=canonical(summary)))
            print('REQUEST_DELIVERY_UI', mode, case['name'], flush=True)
    result = dict(declaration_sha256=checksum(declaration), records=records,
                  no_new_consumer_inference=True, human_judgment=False, research_qualification=False)
    path = ROOT / 'results.json'
    if mode == 'create':
        write(path, result)
    else:
        assert json.loads(path.read_text()) == result
        write(ROOT / 'integrity_receipt.json', dict(results_sha256=checksum(path),
            declaration_sha256=checksum(declaration), state_role_pairs_replayed=6,
            exact_summaries_and_export_hashes=True, human_judgment=False))
    print('REQUEST_DELIVERY_UI_COMPLETE', mode, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['create', 'verify'], required=True)
    main(parser.parse_args().mode)
