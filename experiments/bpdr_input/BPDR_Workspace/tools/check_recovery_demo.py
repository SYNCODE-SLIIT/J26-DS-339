"""Exercise the real learned-model demo without claiming human listening."""
import json
import os
from pathlib import Path
import sys
import time

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / 'source'))
os.environ['PYTHONPATH'] = str(WORKSPACE / 'source')
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['TEMP'] = os.environ['TMP'] = str(WORKSPACE / 'results/demo_sessions/verification_temp')

from streamlit.testing.v1 import AppTest
from bpdr_input.audio.pipeline import checksum
from bpdr_input.repair.learned_runtime import CARD, METHOD_NAME, available
from bpdr_input.repair.demo_review import export_reviewed


def main():
    assert available(), 'No verified exploratory or adopted model is available'
    started = time.perf_counter()
    card = json.loads(CARD.read_text(encoding='utf-8'))
    app = AppTest.from_file(str(WORKSPACE / 'source/bpdr_input/application/demo_app.py'), default_timeout=120)
    app.run()
    assert not app.exception, [str(e) for e in app.exception]
    assert next(s for s in app.selectbox if s.label == 'Processing method').value == METHOD_NAME
    next(b for b in app.button if b.label == 'Process recording').click().run()
    assert not app.exception, [str(e) for e in app.exception]
    result = app.session_state['demo_result']
    report, directory = result['report'], Path(result['directory'])
    assert result['known_case'] == 'learned_pitch'
    assert report['outcome'] == 'VERIFIED_FIXTURE_CANDIDATE', report
    assert report['transaction']['delivered_wav_audit']['passed']
    assert result['review']['preview']['decision'] == 'PASS_EVIDENCE', result['review']
    values = {m.label: m.value for m in app.metric}
    before = float(values['Input local absolute pitch error'].split()[0])
    after = float(values['Output local absolute pitch error'].split()[0])
    assert before-after > 10, values
    assert next(b for b in app.button if b.label == 'Create accepted demo package').disabled
    # This separate fixture exercises export mechanics without a human-review claim.
    exported, handoff = export_reviewed(directory, report, result['review'], use_preview=True,
        confirmed=True, confirmation_kind='automated_test_fixture')
    assert handoff['status'] == 'ACCEPT_PREPARED'
    assert checksum(exported / 'accepted.wav') == report['candidate']['sha256']
    learned = dict(directory=str(directory), outcome=report['outcome'], input_error_cents=before,
        output_error_cents=after, prepared_fixture=str(exported), candidate_sha256=report['candidate']['sha256'])
    next(s for s in app.selectbox if s.label == 'Recording').select_index(1).run()
    next(b for b in app.button if b.label == 'Process recording').click().run()
    assert not app.exception, [str(e) for e in app.exception]
    original = app.session_state['demo_result']
    assert original['known_case'] == 'learned_original'
    assert original['report']['outcome'] in ('IDENTITY', 'WITHHELD')
    assert 'candidate' not in original['report']
    receipt = dict(passed=True, model_sha256=card['checkpoint_sha256'],
        validation_status=card.get('validation_status'), learned=learned,
        original=dict(directory=original['directory'], outcome=original['report']['outcome']),
        human_musical_judgment=False, automated_export_fixture=True,
        elapsed_seconds=round(time.perf_counter()-started, 3))
    path = WORKSPACE / 'results/demo_sessions/ui_recovery_check.json'
    path.write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    main()
