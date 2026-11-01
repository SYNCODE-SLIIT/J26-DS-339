"""Frozen FCPE/YIN/PESTO relative-shift measurements; no correction-model change."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import importlib.metadata
import sys
import time
import numpy as np
import soundfile as sf
import torch

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / 'source'))
from bpdr_input.modeling import fcpe_measurement as measurement
from bpdr_input.repair.rule_correction import fine_pitch
from bpdr_input.repair.interventions import envelope
import train_dense_residual_transformer as reference

ROOT = WORKSPACE / 'results/transformer_usability/2026-10-30_v29_fcpe_measurement'
DEPENDENCIES = ROOT.parent / 'dependencies/fcpe_0_0_4'


def declare(tracker):
    ex = reference.ex; old = reference.declare(); ROOT.mkdir(exist_ok=True)
    declaration = ROOT / 'declaration.json'
    if not declaration.exists():
        sources = dict(old['source_hashes'])
        for p in (Path(__file__).resolve(), Path(measurement.__file__), Path(fine_pitch.__globals__['__file__'])):
            sources[str(p)] = ex.sha(p)
        ex.write(declaration, dict(created_at=datetime.now(timezone.utc).isoformat(),
                 role='Training-side acoustic relative-shift diagnostic; not intended pitch, correction-model accuracy or fresh final evaluation',
                 input_hashes=old['input_hashes'], source_hashes=sources,
                 dependency_receipt_sha256=ex.sha(DEPENDENCIES / 'receipt.json'),
                 checkpoint=str(tracker.checkpoint), checkpoint_sha256=ex.sha(tracker.checkpoint),
                 versions={n: importlib.metadata.version(n) for n in ('torch', 'torchaudio', 'numpy', 'scipy')},
                 tracker_config=tracker.model.args_dict, feature_version=measurement.VERSION,
                 protocol='Official bundled FCPE0.0.4 CPU/local_argmax weighted9-bin decoding, fixed0.006 detection threshold. Explicit scipy polyphase resampling to16kHz. Official mel/model methods without wrapper terminal repeat, frequency clamp, UV interpolation or target-length resizing. Native10ms frames centered5ms; nearest existing frame, earlier ties, <=5ms offset, no extrapolation. Frequencies outside50..1500Hz remain missing, not clamped.',
                 comparison='Current YIN fine and PESTO cached coarse tracks versus FCPE. On same108 audited references, compare C-A cents with known injected cents, preserving all200 attempts/92 construction exclusions. Record paired observed/missing frames, conditional errors and within5/10c with missing counted as failure in fixed-reference percentages. Outside-known-error delta evaluated separately. No relabelling excluded cases or auditor replacement.',
                 synthetic='Three harmonic fundamentals110/220/440Hz,4s, same analytic voiced envelope, shifts-30/-20/+20/+30c as global and local pulses1.3..2.7s with80ms ramps; zeros2s. Record all results without tuning decoder, thresholds, timing or signal after outcomes.',
                 provenance_limit='Bundled checkpoint config identifies yx2/train but does not document identities sufficiently to assert pretrained singer-disjointness from VocalSet. Engineering diagnostic only; no independent-research superiority claim.',
                 new_neural_fits=0, demo_activated=False, selection_eligible=False, research_validation_passed=False))
        snapshot = ROOT / 'executed_sources'; snapshot.mkdir(exist_ok=True)
        for path in sources:
            (snapshot / Path(path).name).write_bytes(Path(path).read_bytes())
        for name in ('train_pairs.json', 'anchor_sources.json'):
            (ROOT / name).write_bytes((reference.ROOT / name).read_bytes())
    d = ex.read(declaration)
    for key in ('input_hashes', 'source_hashes'):
        for path, digest in d[key].items():
            assert ex.sha(path) == digest, path
    assert ex.sha(DEPENDENCIES / 'receipt.json') == d['dependency_receipt_sha256']
    assert ex.sha(tracker.checkpoint) == d['checkpoint_sha256']
    for name, version in d['versions'].items():
        assert importlib.metadata.version(name) == version
    return d


def hz_cents(hz):
    hz = np.asarray(hz); out = np.full(hz.shape, np.nan)
    valid = np.isfinite(hz) & (hz > 0)
    out[valid] = 1200. * np.log2(hz[valid] / 440.)
    return out


def statistics(delta, wanted, mask):
    mask = np.asarray(mask, bool); valid = mask & np.isfinite(delta)
    count = int(mask.sum()); observed = int(valid.sum())
    errors = abs(delta[valid] - wanted)
    return dict(reference_frames=count, observed_frames=observed, missing_frames=count - observed,
                observed_fraction=observed / count if count else None,
                conditional_mae_cents=float(errors.mean()) if observed else None,
                conditional_p90_error_cents=float(np.quantile(errors, .9)) if observed else None,
                within5_over_reference_percent=float(100 * np.sum(errors <= 5.) / count) if count else None,
                within10_over_reference_percent=float(100 * np.sum(errors <= 10.) / count) if count else None,
                direction_over_reference_fraction=float(np.sum(delta[valid] * wanted > 0) / count) if count and wanted else None)


def track_file(tracker, path):
    ex = reference.ex; digest = ex.sha(path)
    folder = ROOT / 'tracks'; folder.mkdir(exist_ok=True)
    array, receipt = folder / (digest + '.npz'), folder / (digest + '.json')
    if receipt.exists():
        old = ex.read(receipt)
        assert old['source_sha256'] == digest and old['declaration_sha256'] == ex.sha(ROOT / 'declaration.json')
        assert old['array_sha256'] == ex.sha(array)
        with np.load(array) as z:
            return z['times'].copy(), z['f0'].copy()
    audio, rate = sf.read(path, dtype='float32'); started = time.perf_counter()
    times, f0, metadata = tracker.native(audio, rate)
    elapsed = time.perf_counter() - started
    assert ex.sha(path) == digest
    np.savez_compressed(array, times=times, f0=f0)
    ex.write(receipt, dict(source_sha256=digest, source=str(path), array_sha256=ex.sha(array),
             declaration_sha256=ex.sha(ROOT / 'declaration.json'), inference_seconds=elapsed,
             waveform_seconds=len(audio) / rate, metadata=metadata))
    return times, f0


def synthetic(tracker):
    ex = reference.ex
    if (ROOT / 'synthetic/results.json').exists():
        raise ValueError('Retain existing synthetic results')
    folder = ROOT / 'synthetic'; folder.mkdir(exist_ok=True)
    sr = 22050; times = np.arange(4 * sr) / sr; grid = np.arange(200) * .02
    voiced = envelope(times, (.2, 3.8), .05)
    weights = np.arange(1, 11, dtype=float) ** -1.4

    def signal(fundamental, cents):
        phase = 2 * np.pi * np.cumsum(fundamental * 2 ** (cents / 1200.)) / sr
        wave = sum(w * np.sin(h * phase) for h, w in enumerate(weights, 1))
        return (wave * voiced * (.2 / weights.sum())).astype(np.float32)

    result = []; arrays = {}
    for fundamental in (110., 220., 440.):
        before = signal(fundamental, np.zeros(len(times)))
        native_t, native_f, meta = tracker.native(before, sr)
        a = hz_cents(measurement.align_native(native_t, native_f, grid)[0])
        yin_a = fine_pitch(before, sr, grid)
        for mode in ('global', 'local'):
            pulse = np.ones(len(times)) if mode == 'global' else envelope(times, (1.3, 2.7), .08)
            for cents in (-30., -20., 20., 30.):
                after = signal(fundamental, cents * pulse)
                tt, ff, _ = tracker.native(after, sr)
                c = hz_cents(measurement.align_native(tt, ff, grid)[0])
                yin_c = fine_pitch(after, sr, grid)
                mask = (grid >= .4) & (grid <= 3.6)
                if mode == 'local':
                    mask &= (grid >= 1.48) & (grid <= 2.52)
                key = f'{int(fundamental)}_{mode}_{cents:+g}'
                arrays[key + '_times'], arrays[key + '_f0'] = tt, ff
                result.append(dict(fundamental_hz=fundamental, mode=mode, injected_cents=cents,
                     generated_audio_sha256=hashlib.sha256(after.tobytes()).hexdigest(),
                     fcpe=statistics(c - a, cents, mask), yin=statistics(yin_c - yin_a, cents, mask),
                     native_metadata=meta))
    tt, ff, meta = tracker.native(np.zeros(2 * sr, np.float32), sr)
    arrays['silence_times'], arrays['silence_f0'] = tt, ff
    np.savez_compressed(folder / 'native_tracks.npz', **arrays)
    ex.write(folder / 'results.json', dict(cases=result, silence_detected_frames=int((ff > 0).sum()),
             silence_total_frames=len(ff), native_metadata=meta,
             arrays_sha256=ex.sha(folder / 'native_tracks.npz'),
             declaration_sha256=ex.sha(ROOT / 'declaration.json'),
             diagnostic_only=True, synthetic_not_people=True, research_validation_passed=False))
    print('FCPE_SYNTHETIC_COMPLETE', len(result), 'silence', int((ff > 0).sum()), '/', len(ff), flush=True)


def macro(results):
    assessed = [r for r in results if 'excluded' not in r]
    by_group = {}
    for group in sorted({r['group'] for r in results}):
        rr = [r for r in assessed if r['group'] == group]
        if not rr:
            by_group[group] = dict(reference_cases=0, assessment='unassessed_no_valid_reference')
            continue
        metrics = {}
        for key in ('conditional_mae_cents', 'conditional_p90_error_cents', 'observed_fraction',
                    'within5_over_reference_percent', 'within10_over_reference_percent', 'direction_over_reference_fraction'):
            vv = [r['plateau'][key] for r in rr if r['plateau'][key] is not None]
            metrics[key] = float(np.mean(vv)) if vv else None
        by_group[group] = dict(reference_cases=len(rr), observed_cases=sum(r['plateau']['observed_frames'] > 0 for r in rr), **metrics)
    summary = dict(attempted=len(results), reference_scoreable=len(assessed), excluded=len(results) - len(assessed))
    for key in ('conditional_mae_cents', 'observed_fraction', 'within5_over_reference_percent',
                'within10_over_reference_percent', 'direction_over_reference_fraction'):
        values = [g[key] for g in by_group.values() if g['reference_cases'] and g[key] is not None]
        summary[key] = float(np.mean(values)) if values else None
    summary['singers_with_observed_cases'] = sum(g.get('observed_cases', 0) > 0 for g in by_group.values())
    return dict(summary=summary, by_singer=by_group, cases=results)


def real(tracker):
    ex = reference.ex
    if not (ROOT / 'synthetic/results.json').exists():
        raise ValueError('Complete structural/synthetic checks before real diagnostic')
    if (ROOT / 'real_measurements.json').exists():
        raise ValueError('Retain real measurements')
    rows = ex.read(ROOT / 'train_pairs.json'); pairs = reference.training.held.prepared(rows)
    originals = []
    for source in ex.read(ROOT / 'anchor_sources.json'):
        tt, ff = track_file(tracker, source['working_audio'])
        originals.append(dict(source_sha256=source['source']['sha256'], group=source['source']['singer_id'],
                              native_frames=len(ff), detected_frames=int((ff > 0).sum())))
    results = {name: [] for name in ('yin', 'pesto', 'fcpe')}
    for index, pair in enumerate(pairs):
        if 'excluded' in pair:
            for name in results:
                results[name].append(dict(index=index, group=pair['group'], excluded=pair['excluded']))
            continue
        grid = pair['C']['features']['times_seconds']
        fcpe = {}
        for view in ('A', 'C'):
            tt, ff = track_file(tracker, pair['path_' + view])
            fcpe[view] = hz_cents(measurement.align_native(tt, ff, grid)[0])
        tracks = dict(fcpe=fcpe,
                      yin={v: np.where(pair[v]['reliable'], pair[v]['fine'], np.nan) for v in ('A', 'C')},
                      pesto={v: np.where(pair[v]['features']['voiced_mask'], hz_cents(pair[v]['features']['f0_hz']), np.nan) for v in ('A', 'C')})
        plateau = pair['mask'] & (pair['target'] != 0)
        lo, hi = pair['support_seconds']
        outside = pair['mask'] & (pair['target'] == 0) & ((grid < lo - .1) | (grid > hi + .1))
        for name, views in tracks.items():
            delta = views['C'] - views['A']
            results[name].append(dict(index=index, group=pair['group'], source_sha256=pair['source']['sha256'],
                     injected_cents=float(pair['cents']), plateau=statistics(delta, pair['cents'], plateau),
                     outside=statistics(delta, 0., outside)))
        ex.write(ROOT / 'real_progress.json', dict(completed_attempt_index=index, total_attempts=len(rows)))
    summarized = {name: macro(rr) for name, rr in results.items()}
    ex.write(ROOT / 'real_measurements.json', dict(results=summarized, originals=originals,
             declaration_sha256=ex.sha(ROOT / 'declaration.json'), diagnostic_only=True,
             new_neural_fits=0, selection_eligible=False, model_activated=False, research_validation_passed=False))
    print('ACOUSTIC_MEASUREMENT_COMPLETE', {n: r['summary'] for n, r in summarized.items()}, flush=True)


def main():
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['declare', 'synthetic', 'real'])
    args = parser.parse_args()
    tracker = measurement.FCPEMeasurement(DEPENDENCIES); declare(tracker)
    if args.stage == 'synthetic':
        synthetic(tracker)
    elif args.stage == 'real':
        real(tracker)
    declare(tracker)


if __name__ == '__main__':
    main()
