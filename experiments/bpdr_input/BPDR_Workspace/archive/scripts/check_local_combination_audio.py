"""Bounded real-auditor test of two manual synthetic local edits, no fitting."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from bpdr_input import COMPONENT_ROOT
from bpdr_input.audio.analysis import extract
from bpdr_input.audio.pipeline import checksum
from bpdr_input.repair import local_review as local
from bpdr_input.repair import primary_handoff as primary
from bpdr_input.repair.demo_review import _assessment
from bpdr_input.repair.framewise_renderer import render_candidate, RendererPolicy
from bpdr_input.repair.interventions import AuditPolicy
from bpdr_input.repair.repair_controller import plan_fixture, FixturePolicy
from bpdr_input.repair.review_export import export_reviewed


ROOT = local.WORKSPACE / 'results/demo_sessions/2026-11-08_local_combination_audio_v2'


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def canonical(value):
    return json.loads(json.dumps(value, allow_nan=False))


def main(mode):
    torch.set_num_threads(2)
    ROOT.mkdir(exist_ok=True)
    files = [Path(__file__), Path(local.__file__), Path(primary.__file__),
        Path(local.__file__).with_name('review_export.py'),
        Path(local.__file__).with_name('demo_review.py'),
        Path(local.__file__).with_name('framewise_renderer.py'),
        Path(local.__file__).with_name('framewise_audit.py'),
        Path(local.__file__).with_name('learned_runtime.py'),
        Path(local.__file__).parents[1] / 'audio/analysis.py']
    protocol = dict(source_hashes={str(p): checksum(p) for p in files},
        hypothesis='Subsets of an already permitted local preview retain sample support and pass the unchanged final decoded-WAV auditor.',
        data='One generated ten-second 220 Hz harmonic signal. Two manual +/-20-cent requests on [2,3) and [6,7). No trained model, person or intended-note labels.',
        renderer=asdict(RendererPolicy()), permission=asdict(FixturePolicy()), audit=asdict(AuditPolicy()),
        budget='Create and fresh-process verify: parent plus three subset WAV audits per process (eight total). Two repeated real pYIN calls on the combined accepted fixture.',
        criteria='Existing acoustic limits; exact outside/protected samples; exact repeated hashes and audits; selected/exported/consumer hash equality.',
        stopping='One fixed synthetic fixture. Stop on any gate failure; no parameter or source sweep.',
        production_eligible=False, human_musical_judgment=False)
    protocol = canonical(protocol)
    declaration = ROOT / 'declaration.json'
    if declaration.exists():
        assert json.loads(declaration.read_text()) == protocol
    else:
        assert mode == 'create'
        write(declaration, protocol)
        snapshots = ROOT / 'source_snapshots'
        snapshots.mkdir(exist_ok=True)
        for p in files:
            (snapshots / p.name).write_bytes(p.read_bytes())
    rate = 22050
    times_samples = np.arange(rate * 10) / rate
    signal = sum(amplitude * np.sin(2 * np.pi * frequency * times_samples)
                 for frequency, amplitude in [(220., .16), (440., .04), (660., .02)])
    original_path = ROOT / 'synthetic_original.wav'
    if not original_path.exists():
        sf.write(original_path, signal, rate, subtype='PCM_24')
    original, _ = sf.read(original_path, dtype='float32')
    before, analyzer = extract(original, rate, COMPONENT_ROOT / 'checkpoints')
    times = before['times_seconds']
    curve = np.zeros(len(times))
    curve[(times >= 2.) & (times < 3.)] = 20.
    curve[(times >= 6.) & (times < 7.)] = -20.
    reliable = before['voiced_mask'].copy()
    protected = np.zeros(len(times), dtype=bool)
    plan = plan_fixture(original, rate, times, curve, curve, np.zeros(len(times)), reliable, protected)
    assert plan['permitted'], plan
    candidate, _ = render_candidate(original, rate, times, curve, tuple(plan['intervals_seconds']))
    path = ROOT / 'parent_preview.wav'
    if not path.exists():
        sf.write(path, candidate, rate, subtype='PCM_24')
    parent_audit, preview_review = local._audit_and_assess(path, original, rate, times, curve,
        reliable, tuple(plan['intervals_seconds']), 'singing_humming_chant')
    original_review = _assessment(original_path, before, 'singing_humming_chant')
    assert parent_audit['passed']
    report = dict(source=str(original_path), outcome='VERIFIED_FIXTURE_CANDIDATE', permission=plan,
        model_status='manual synthetic requests; no trained model',
        candidate=dict(path=str(path), sha256=checksum(path)),
        transaction=dict(post_render_audit=parent_audit['acoustic_audit'], delivered_wav_audit=parent_audit),
        handoff=dict(request_id='synthetic_local_combination', source_id='synthetic_no_participant', status='ACCEPT_UNCHANGED',
            accepted_audio=dict(path=str(original_path), checksum=checksum(original_path), sample_rate_hz=rate,
                sample_count=len(original), channels=1)))
    review = dict(intent='singing_humming_chant', original=original_review, preview=preview_review)
    predictions = ROOT / 'model_predictions.npz'
    if not predictions.exists():
        np.savez_compressed(predictions, times=times, proposed_correction=curve, reliable=reliable, protected=protected)
    results = []
    consumer_receipts = []
    for indices in [[0], [1], [0, 1]]:
        updated, assessed = local.select_intervals(ROOT, report, review, indices)
        row = dict(selected=indices, sha256=updated['candidate']['sha256'],
            final_decoded_audit=updated['local_selection']['final_decoded_audit'],
            acoustic_evidence=assessed['preview']['decision'])
        results.append(row)
        if mode == 'create' and indices == [0, 1]:
            package, handoff = export_reviewed(ROOT, updated, assessed, use_preview=True,
                confirmed=True, confirmation_kind='automated_test_fixture')
            diagnostics = json.loads((package / 'diagnostics.json').read_text())
            assert diagnostics['local_selection'] == canonical(updated['local_selection'])
            consumer_receipts = [primary.run(package, allow_test_fixture=True) for _ in range(2)]
            assert all(r['success'] and r['consumer_sha256'] == row['sha256'] for r in consumer_receipts)
            row['exported_fixture'] = str(package)
            row['status'] = handoff['status']
        print('LOCAL_COMBINATION_AUDITED', mode, indices, flush=True)
    result_path = ROOT / 'results.json'
    result = canonical(dict(declaration_sha256=checksum(declaration), source_sha256=checksum(original_path),
        analyzer=analyzer, parent_audit=parent_audit, rows=results,
        consumer_receipts=consumer_receipts, production_eligible=False, human_musical_judgment=False))
    if mode == 'create':
        write(result_path, result)
    else:
        saved = json.loads(result_path.read_text())
        assert saved['source_sha256'] == result['source_sha256'] and saved['parent_audit'] == result['parent_audit']
        assert saved['analyzer'] == result['analyzer']
        for previous, actual in zip(saved['rows'], result['rows']):
            for key, value in actual.items():
                assert previous[key] == value, (key, previous, actual)
        write(ROOT / 'integrity_receipt.json', dict(results_sha256=checksum(result_path),
            declaration_sha256=checksum(declaration), subset_combinations_replayed=3,
            parent_audit_replayed=True, exact_hashes_and_acoustic_audits=True,
            human_musical_judgment=False, production_eligible=False))
    print('LOCAL_COMBINATION_COMPLETE', mode, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['create', 'verify'], required=True)
    main(parser.parse_args().mode)
