"""Real consumer check on two explicitly synthetic, automated export fixtures."""

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from bpdr_input.audio.pipeline import checksum
from bpdr_input.repair.review_export import export_reviewed
from bpdr_input.repair import primary_handoff as primary
from bpdr_input.repair.review_protection import protect


ROOT = primary.WORKSPACE / 'results/demo_sessions/2026-11-08_primary_selected_handoff_v3'


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def main():
    ROOT.mkdir(exist_ok=True)
    snapshots = ROOT / 'source_snapshots'
    snapshots.mkdir(exist_ok=True)
    bound_sources = [Path(__file__), Path(primary.__file__),
        Path(primary.__file__).with_name('review_export.py'),
        Path(primary.__file__).with_name('demo_review.py'),
        Path(primary.__file__).with_name('review_protection.py'),
        primary.WORKSPACE / 'source/bpdr_input/application/review_demo_app.py']
    for source in bound_sources:
        target = snapshots / source.name
        if target.exists():
            assert target.read_bytes() == source.read_bytes()
        else:
            target.write_bytes(source.read_bytes())
    declaration = dict(hypothesis='Consumer receives exact selected bytes, including a prepared version distinct from original.',
        data_role='Synthetic engineering fixture only; no singer, intended-note or quality evidence.',
        source_sha256=checksum(__file__), adapter_sha256=checksum(primary.__file__),
        review_export_sha256=checksum(Path(primary.__file__).with_name('review_export.py')),
        historical_export_sha256=checksum(Path(primary.__file__).with_name('demo_review.py')),
        source_snapshots={str(p): checksum(p) for p in sorted(snapshots.glob('*.py'))},
        protection_source_sha256=checksum(Path(primary.__file__).with_name('review_protection.py')),
        freeze_sha256=primary._freeze(), consumer_calls=6, fitting=False,
        criterion='Original/prepared differ; protected-preview veto exports original with provenance; every selected/copied/consumer hash and repeated result matches excluding runtime.',
        stopping_rule='Six calls only. A failure stops the check; no candidate promotion or threshold change.')
    path = ROOT / 'declaration.json'
    if path.exists():
        assert json.loads(path.read_text()) == declaration
    else:
        write(path, declaration)
    rate = 22050
    times = np.arange(5 * rate) / rate
    # A short synthetic melodic signal, never presented as a participant hum.
    hz = np.take([220., 246.94, 261.63, 220., 293.66], np.minimum(times.astype(int), 4))
    phase = np.cumsum(2 * np.pi * hz / rate)
    envelope = np.minimum(1., np.minimum(times % 1, 1 - times % 1) / .02)
    original = ROOT / 'synthetic_original.wav'
    prepared = ROOT / 'synthetic_gain_fixture.wav'
    sf.write(original, .2 * envelope * np.sin(phase), rate, subtype='PCM_24')
    samples, _ = sf.read(original, dtype='float64')
    sf.write(prepared, .6 * samples, rate, subtype='PCM_24')
    assert checksum(original) != checksum(prepared)
    report = dict(outcome='VERIFIED_FIXTURE_CANDIDATE',
        handoff=dict(request_id='synthetic_signal_no_participant', source_id='synthetic_signal_no_participant', status='ACCEPT_UNCHANGED',
            accepted_audio=dict(path=str(original), checksum=checksum(original),
                sample_rate_hz=rate, sample_count=len(samples), channels=1)),
        candidate=dict(path=str(prepared), sha256=checksum(prepared)),
        transaction=dict(post_render_audit=dict(passed=bool(
            np.isfinite(sf.read(prepared)[0]).all() and sf.info(prepared).frames == len(samples)))))
    review = {role: dict(decision='PASS_EVIDENCE', waveform_sha256=checksum(wave),
        scope='Automated synthetic integration fixture; no musical judgment.')
        for role, wave in [('original', original), ('preview', prepared)]}
    rows = []
    protected_report, protected_review = protect(report, review, [[.5, .75]])
    assert protected_report['user_protection']['preview_vetoed']
    cases = [('original', report, review, False, original),
             ('prepared', report, review, True, prepared),
             ('protected_original', protected_report, protected_review, False, original)]
    for selection, selected_report, selected_review, use_preview, selected in cases:
        package, handoff = export_reviewed(ROOT, selected_report, selected_review, use_preview=use_preview,
            confirmed=True, confirmation_kind='automated_test_fixture')
        if selection == 'protected_original':
            diagnostics = json.loads((package / 'diagnostics.json').read_text())
            assert diagnostics['user_protection'] == protected_report['user_protection']
            assert handoff['status'] == 'ACCEPT_UNCHANGED'
        receipts = [primary.run(package, allow_test_fixture=True) for _ in range(2)]
        for receipt in receipts:
            assert receipt['success'], receipt
            assert receipt['selected_sha256'] == receipt['consumer_sha256'] == checksum(selected)
            assert checksum(receipt['consumer_input']) == checksum(selected)
            assert not receipt['human_review_claimed']
        outputs = []
        for receipt in receipts:
            result = json.loads(Path(receipt['artifacts'][0]['path']).read_text())['result']
            result.pop('inference_seconds')
            assert result['source_duration_s'] == 5.
            assert result['preprocessing']['trim_enabled'] is False
            outputs.append(result)
        assert outputs[0] == outputs[1]
        rows.append(dict(status=handoff['status'], selection=selection,
            selected_sha256=checksum(selected), package=str(package),
            consumer_notes=len(outputs[0]['notes']), receipts=receipts))
    assert rows[0]['selected_sha256'] != rows[1]['selected_sha256']
    write(ROOT / 'integrity_receipt.json', dict(declaration_sha256=checksum(path),
        checked=True, consumer_calls=6, exact_selected_audio=True,
        independent_child_repeats_match=True, human_musical_judgment=False,
        research_qualification=False, shared_pp1_integration=False, rows=rows))
    print('PRIMARY_SELECTED_HANDOFF_VERIFIED', len(rows), flush=True)


if __name__ == '__main__':
    main()
