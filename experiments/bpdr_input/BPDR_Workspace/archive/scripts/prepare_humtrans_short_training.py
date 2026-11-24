"""Training-only short physical interventions on the pre-existing local hum intake.

No development audio, guide MIDI, reserved unseen people or model fit is opened.
Uses the separately calibrated experimental three-tracker protocol, not production.
"""
from collections import Counter
from dataclasses import asdict
from math import gcd
from pathlib import Path
import sys

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import torch

WS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WS / 'source'))
sys.path.insert(0, str(WS / 'tools'))
import calibrate_short_support_audit as calibration
import reconcile_local_humtrans as intake
from acquire_fresh_confirmation import check_lock
from bpdr_input.repair.rule_correction import segment_notes, RulePolicy
from prepare_short_phrase_training import save_analysis, read, write, norm

ROOT = WS / 'results/transformer_usability/2026-11-02_v50_humtrans_short_training'
RECONCILIATION = intake.ROOT / 'reconciliation.json'
sha = calibration.sha


def declare():
    check_lock()
    report = read(RECONCILIATION)
    sources = [r for r in report['records'] if r['split'] == 'train']
    assert len(sources) == 12 and len({r['person_id'] for r in sources}) == 6
    for source in sources:
        assert sha(source['path']) == source['sha256']
    prior = read(calibration.ROOT / 'integrity_receipt.json')
    assert prior['physical_audits_replayed'] == 128 and prior['negative_pitch_controls_rejected'] == 512
    ROOT.mkdir(parents=True, exist_ok=True)
    declaration = norm(dict(version='historical-humtrans-short-training-1',
        source_hashes={str(p.resolve()): sha(p) for p in [Path(__file__), Path(calibration.short.__file__)]},
        reconciliation_sha256=sha(RECONCILIATION), prior_calibration_receipt_sha256=sha(calibration.ROOT / 'integrity_receipt.json'),
        sources=sources, planned_attempts=192, widths_seconds=[.24, .28, .36, .50], cents=[-30., -20., 20., 30.],
        crop='First min(10s,duration), highest-energy native channel, resample22050Hz PCM24; no normalization, padding or outcome-based replacement',
        support='Center of longest input-only reliable region split70c; at least150ms original context each side; no MIDI/support hints in inference',
        renderer=asdict(calibration.RENDER), audit=asdict(calibration.short.POLICY),
        labels='Nominal negative injected cents on >=3 triple-estimator common plateau frames; zero outside support+70ms; >=3 retained C anchors on each side; no interpolation',
        historical_roles_retained=True, prior_training_use=True, fresh_confirmation=False,
        development_audio_accessed=False, unseen_audio_accessed=False, guide_notes_loaded=False,
        research_training_eligible=False, production_eligible=False, new_model_fits=0))
    path = ROOT / 'declaration.json'
    if path.exists():
        assert read(path) == declaration
    else:
        write(path, declaration)
    return declaration, sources


def main():
    torch.set_num_threads(2)
    d, sources = declare()
    all_rows, originals, hashes = [], [], {}
    for index, source in enumerate(sources, 1):
        sid = source['key']
        folder = ROOT / 'training' / sid
        folder.mkdir(parents=True, exist_ok=True)
        result_path = folder / 'source_result.json'
        if result_path.exists():
            result = read(result_path)
            assert result['declaration_sha256'] == sha(ROOT / 'declaration.json')
            for path, expected in result['output_hashes'].items():
                assert sha(path) == expected
            all_rows.extend(result['pairs']); originals.append(result['original']); hashes.update(result['output_hashes'])
            print('HUM_SHORT_SOURCE_REUSED', sid, flush=True)
            continue
        native, sr = sf.read(source['path'], dtype='float32', always_2d=True)
        channel = int(np.argmax(np.mean(native.astype(np.float64) ** 2, axis=0)))
        audio = native[:min(len(native), 10 * sr), channel]
        divisor = gcd(sr, 22050)
        if sr != 22050:
            audio = resample_poly(audio, 22050 // divisor, sr // divisor).astype(np.float32)
        original = dict(source_id=sid, group='HumTrans:' + source['person_id'], source=source,
                        source_channel=channel, crop_offset_seconds=0., duration_seconds=len(audio) / 22050,
                        sample_rate_hz=22050, split='train', research_training_eligible=False)
        paths, rows = [], []
        if len(audio) < 5 * 22050 or not np.isfinite(audio).all() or abs(audio).max() >= 1:
            original['excluded'] = 'INVALID_SOURCE_DURATION_OR_SAMPLES'
        else:
            a_path = folder / 'original_working.wav'
            sf.write(a_path, audio, 22050, subtype='PCM_24')
            audio, _ = sf.read(a_path, dtype='float32')
            a = calibration.recovery.measured(audio)
            times = a['features']['times_seconds']; reliable = a['reliable']
            a_npz = folder / 'original_analysis.npz'; save_analysis(a_npz, a)
            original.update(path_A=str(a_path), analysis_A=str(a_npz))
            paths.extend([a_path, a_npz])
            regions = segment_notes(a['views'][0][:, 64], reliable, RulePolicy(note_split_cents=70.))
            region = max(regions, key=lambda r: r[1] - r[0]) if regions else None
            original['longest_region_frames'] = region
            bf = calibration.coarse_features(audio, 22050, WS / 'checkpoints')
            yb = calibration.fine_hz(audio, 22050, times)
            cb = calibration.short.cross_correlation_pitch(audio, 22050, times)
            before = folder / 'before_measurements.npz'
            np.savez_compressed(before, **bf, yin=yb, cc=cb, reliable=reliable)
            paths.append(before)
        for width in d['widths_seconds']:
            for cents in d['cents']:
                row = dict(source_id=sid, group=original['group'], width_seconds=width, cents=cents,
                           split='train', research_training_eligible=False)
                if 'excluded' in original:
                    rows.append(dict(**row, excluded=original['excluded'])); continue
                if region is None or (region[1] - region[0]) * .02 < width + .30 - 1e-12:
                    rows.append(dict(**row, excluded='NO_SINGLE_RELIABLE_REGION_WITH_150MS_CONTEXT')); continue
                center = float(np.mean(region) * .02)
                support = (center - width / 2, center + width / 2)
                curve = np.where((times >= support[0]) & (times < support[1]), cents, 0.)
                case = folder / f'{width:.2f}_{cents:+g}'; case.mkdir(exist_ok=True)
                output, render_receipt = calibration.render_candidate(audio, 22050, times, curve, [support], calibration.RENDER)
                c_path = case / 'intervention.wav'; sf.write(c_path, output, 22050, subtype='PCM_24')
                decoded, rate = sf.read(c_path, dtype='float32')
                assert rate == 22050 and decoded.shape == audio.shape and np.isfinite(decoded).all() and abs(decoded).max() < 1
                sample_times = np.arange(len(audio)) / 22050
                outside_samples = (sample_times <= support[0]) | (sample_times >= support[1])
                assert np.array_equal(audio[outside_samples], decoded[outside_samples])
                af = calibration.coarse_features(decoded, 22050, WS / 'checkpoints')
                ya = calibration.fine_hz(decoded, 22050, times)
                ca = calibration.short.cross_correlation_pitch(decoded, 22050, times)
                audit = calibration.short.audit_short_curve(bf, af, yb, ya, cb, ca, reliable, [support], curve)
                controls = {name: calibration.short.audit_short_curve(bf, af if name == 'opposite' else bf,
                            yb, ya if name == 'opposite' else yb, cb, ca if name == 'opposite' else cb,
                            reliable, [support], -curve if name == 'opposite' else curve)
                            for name in ('unchanged', 'opposite')}
                assert not any(control['passed'] for control in controls.values())
                audit_path = case / 'audit.json'; write(audit_path, dict(audit=audit, controls=controls))
                measurement_path = case / 'measurements.npz'
                np.savez_compressed(measurement_path, **af, yin=ya, cc=ca, requested=curve)
                paths.extend([c_path, audit_path, measurement_path])
                row.update(path_A=str(a_path), analysis_A=str(a_npz), path_C=str(c_path),
                           support_seconds=support, audit=audit, audit_path=str(audit_path), render_receipt=render_receipt)
                if not audit['passed']:
                    rows.append(dict(**row, excluded='THREE_TRACKER_PHYSICAL_AUDIT_FAILED')); continue
                c = calibration.recovery.measured(decoded)
                common = reliable & c['reliable'] & bf['voiced_mask'] & af['voiced_mask'] & np.isfinite(yb) & np.isfinite(ya) & (yb > 0) & (ya > 0)
                plateau = (times >= support[0] + .09) & (times <= support[1] - .09) & common & np.isfinite(cb) & np.isfinite(ca) & (cb > 0) & (ca > 0)
                zero = ((times < support[0] - .07) | (times > support[1] + .07)) & common
                left = (times >= region[0] * .02) & (times < support[0] - .07) & c['reliable']
                right = (times > support[1] + .07) & (times < region[1] * .02) & c['reliable']
                if plateau.sum() < 3 or min(left.sum(), right.sum()) < 3:
                    rows.append(dict(**row, excluded='INSUFFICIENT_RETAINED_PLATEAU_OR_CONTEXT')); continue
                context_path = case / 'input_context.npz'
                np.savez_compressed(context_path, raw_views=np.stack(c['views']), reliable=c['reliable'], fine=c['fine'], times_seconds=times)
                target_path = case / 'targets_NOT_MODEL_INPUT.npz'
                np.savez_compressed(target_path, target=np.where(plateau, -cents, 0.).astype(np.float32),
                                    mask=plateau | zero, plateau=plateau, known_zero=zero, times=times)
                paths.extend([context_path, target_path])
                rows.append(dict(**row, analysis_C=str(context_path), target_path=str(target_path),
                                 positive_frames=int(plateau.sum()), zero_frames=int(zero.sum())))
        result = dict(declaration_sha256=sha(ROOT / 'declaration.json'), original=original, pairs=rows,
                      output_hashes={str(p): sha(p) for p in paths})
        write(result_path, result)
        all_rows.extend(rows); originals.append(original); hashes.update(result['output_hashes'])
        write(ROOT / 'progress.json', dict(completed_sources=index, planned_sources=12, attempted=len(all_rows), valid=sum('excluded' not in r for r in all_rows)))
        print('HUM_SHORT_SOURCE', index, '/12', sid, 'valid', sum('excluded' not in r for r in all_rows), '/', len(all_rows), flush=True)
    assert len(all_rows) == 192
    write(ROOT / 'train_pairs.json', all_rows); write(ROOT / 'train_originals.json', originals); write(ROOT / 'input_hashes.json', hashes)
    summary = dict(attempted=192, valid=sum('excluded' not in r for r in all_rows),
                   valid_by_group=dict(Counter(r['group'] for r in all_rows if 'excluded' not in r)),
                   exclusions=dict(Counter(r['excluded'] for r in all_rows if 'excluded' in r)),
                   original_attempts=12, valid_originals=sum('excluded' not in o for o in originals),
                   development_audio_accessed=False, unseen_audio_accessed=False, new_model_fits=0,
                   research_training_eligible=False, production_eligible=False)
    write(ROOT / 'summary.json', summary)
    print('HUMTRANS_SHORT_TRAINING_READY', summary, flush=True)


if __name__ == '__main__':
    main()
