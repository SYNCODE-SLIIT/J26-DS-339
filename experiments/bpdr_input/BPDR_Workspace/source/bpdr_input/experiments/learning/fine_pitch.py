"""Gate E fine-pitch rebuild and training-only learning check (one declaration, two commands).

Why: frozen PESTO mir-1k_g7 has one-hot activations, so its pitch reads in 33.3-cent steps and cannot
resolve the 10-30-cent shifts this study injects. The 71-feature contract and the model stay as proposed;
only the source of column 64 (performed pitch, cents re A440) and column 67 (pitch change) changes to the
YIN estimator already used in this component (librosa.yin, 50-1500 Hz, frame 2048, hop 441 = feature grid).
PESTO keeps confidence (65) and voicing (66), except that frames where YIN and PESTO disagree by more than
50 cents become unvoiced (octave/estimation-error guard). All other columns are copied byte-for-byte.

`build`   declare, rebuild every view of the 995 training and 304 development records into new archives,
          fit the training-only normalizer with the frozen hierarchical weights, verify.
`check`   one augmentation-only fit (proposal objective L_A + 0.1 L_S) for 150 updates of the frozen
          schedule on XPU; gate on training correction loss versus zero output.
No shared-pipeline file is read for writing; old archives, catalogs and checkpoints stay unchanged.
"""
from bpdr_input import COMPONENT_ROOT
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import uuid
import numpy as np

VERSION = 'fine-pitch-columns-0.1'
PITCH, CONFIDENCE, VOICED, CHANGE, BOUNDARY = 64, 65, 66, 67, 70
DISAGREEMENT_CENTS = 50.
COMPONENT = COMPONENT_ROOT
PREP = COMPONENT/'runs/expanded_experiment_preparation/eeddb03700ad4ddc989dd73b3b113616'
DEV = COMPONENT/'runs/support_development_packaging/779ea64608384daa9014865187c365b4/combined_catalog.json'
ROOT = COMPONENT/'runs/fine_pitch'
CHECK = dict(updates=150, condition='augmentation_only', objective='proposal L_A + 0.1 L_S (losses.objective, response terms off)',
             schedule='first 150 rows of the frozen 300x8 expanded schedule', device='xpu', seed=339, learning_rate=1e-3,
             weight_decay=.01, gradient_clip_norm=1., microbatch=1, effective_batch=8,
             gate=dict(window='updates 126-150', correction_loss_ratio_to_zero_output_max=.90, original_view_mae_cents_max=1.),
             control='saved expanded-comparison augmentation_only arm (same schedule/init/optimizer, quantized pitch) and the 14.38 arms, which stayed at ratio 1.000',
             interpretation='Passing shows the transformer can learn the correction on training data; it is not development or held-out accuracy.')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fine_columns(raw, yin_cents):
    """Return a copy of a 71-column archive with fine pitch/pitch-change, and the number of frames unvoiced by the guard."""
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] != 71 or len(yin_cents) != len(raw):
        raise ValueError('Fine pitch needs a frames x 71 archive and one YIN value per frame')
    out = raw.copy()
    voiced = (raw[:, VOICED] > .5) & (raw[:, PITCH] != 0)
    fine = np.asarray(yin_cents, dtype=np.float64)
    agree = voiced & np.isfinite(fine) & (np.abs(fine-raw[:, PITCH]) <= DISAGREEMENT_CENTS)
    dropped = voiced & ~agree
    out[:, PITCH] = np.where(agree, fine, 0).astype(np.float32)
    out[:, VOICED] = np.where(dropped, 0, raw[:, VOICED]).astype(np.float32)
    change = np.zeros(len(raw))
    both = agree[1:] & agree[:-1]
    change[1:][both] = np.diff(fine)[both]
    out[:, CHANGE] = change.astype(np.float32)
    return out, int(dropped.sum())


def yin_cents(wav, times):
    import soundfile
    from bpdr_input.repair.interventions import independent_pitch
    audio, rate = soundfile.read(wav, dtype='float32')
    if audio.ndim > 1:
        raise ValueError('Expected mono working audio')
    f0 = independent_pitch(audio, rate)
    t = np.arange(len(f0))*441/rate
    idx = np.clip(np.searchsorted(t, times), 0, len(t)-1)
    prev = np.maximum(idx-1, 0)
    idx = np.where(np.abs(t[prev]-times) < np.abs(t[idx]-times), prev, idx)
    if np.max(np.abs(t[idx]-times)) > .011:
        raise ValueError('YIN frames do not align with the feature grid')
    return 1200*np.log2(np.maximum(f0[idx], 1e-6)/440.)


def declare(out):
    d = dict(version=VERSION, declared_utc=datetime.now(timezone.utc).isoformat(), status='declared_before_build',
             reason='PESTO mir-1k_g7 activations are one-hot: pure tones at +20/+29.6/+40.1 cents all read +33.33. Proposal (3.6) fixes the 71 features and says PESTO "starting with" mir-1k_g7; this is a recorded deviation of the pitch source only.',
             estimator='librosa.yin(fmin=50, fmax=1500, frame_length=2048, hop_length=441) via interventions.independent_pitch; tone test: median error <=1.8 cents at 110-880 Hz, +30-cent in-note step read as +30.6',
             columns_changed={'64': 'YIN cents re A440 on frames PESTO calls voiced and YIN agrees within 50 cents, else 0',
                              '66': 'PESTO voicing, set to 0 where YIN/PESTO disagree by >50 cents',
                              '67': 'frame-to-frame change of the new column 64 where both frames are kept, else 0'},
             columns_unchanged='0-63 log-mel, 65 PESTO confidence, 68 energy, 69 onset, 70 timing-only boundary encoding',
             audit_independence='YIN was the independent audit tracker for realized shifts; it now also feeds the model. Existing audits are retained; future audits of model output need a different second tracker.',
             scope=dict(training_records=995, development_records=304,
                        original_controls='88 no-change originals not rebuilt in this step (needed before the matched comparison)'),
             normalizer='fit_streaming_normalizer on the rebuilt 995 training records with the frozen hierarchical weights from sampling_schedule.npz',
             learning_check=CHECK, inputs=dict(training_catalog_sha256=sha(PREP/'training_catalog.json'), development_catalog_sha256=sha(DEV),
                                               sampling_schedule_sha256=sha(PREP/'sampling_schedule.npz'), locked_protocol_sha256=sha(PREP/'locked_protocol.json')),
             source_sha256=sha(__file__), forbidden='No targets, masks or paired originals enter any feature. No shared-pipeline execution; no dev data used in the learning check; no model adoption.')
    (out/'declaration.json').write_text(json.dumps(d, indent=2), encoding='utf-8')
    return d


def rebuild(records, split, out, cache, stats):
    new = []
    for r in records:
        rel = Path(split)/r['dataset']/r['phrase_id']/f"shift_{r['cents']:+g}"/r['corruption']
        folder = out/'inputs'/rel; folder.mkdir(parents=True, exist_ok=True)
        hashes = {}
        for view in 'ABCD':
            src = Path(r['input_directory'])/(view+'_input.npz')
            if sha(src) != r['input_sha256'][view]:
                raise ValueError(f'Frozen input changed: {src}')
            with np.load(src, allow_pickle=False) as z:
                times, raw = z['times_seconds'], z['boundary_features_71']
            wav = r['waveform_paths'][view]
            if wav not in cache:
                if sha(wav) != r['waveform_sha256'][view]:
                    raise ValueError(f'Waveform changed: {wav}')
                cache[wav] = (times, yin_cents(wav, times))
            ctimes, cents = cache[wav]
            if not np.array_equal(ctimes, times):
                raise ValueError('Views sharing a waveform must share the frame grid')
            fixed, dropped = fine_columns(raw, cents)
            stats[split]['frames'] += len(raw); stats[split]['voiced'] += int((raw[:, VOICED] > .5).sum()); stats[split]['dropped'] += dropped
            path = folder/(view+'_input.npz')
            np.savez_compressed(path, times_seconds=times, frame_features_70=fixed[:, :70], boundary_features_71=fixed)
            hashes[view] = sha(path)
        new.append(dict(r, input_directory=str(folder), input_sha256=hashes, quantized_input_directory=r['input_directory'],
                        quantized_input_sha256=r['input_sha256'], pitch_columns=VERSION))
    return new


def realized_shift(records):
    """Median (C - A) of the rebuilt pitch column inside the audited plateau, by injected cents: a resolution check on real audio."""
    by = defaultdict(list)
    for r in records:
        with np.load(Path(r['input_directory'])/'A_input.npz') as z: a = z['boundary_features_71']
        with np.load(Path(r['input_directory'])/'C_input.npz') as z: c = z['boundary_features_71']
        with np.load(r['target_path']) as z: m = z['common_reliable_mask'].astype(bool) & (z['correction_target_cents'] != 0)
        m &= (a[:, VOICED] > .5) & (c[:, VOICED] > .5)
        if m.any():
            by[r['cents']].append(float(np.median(c[m, PITCH]-a[m, PITCH])))
    return {str(k): dict(records=len(v), median_cents=float(np.median(v)), p10=float(np.quantile(v, .1)), p90=float(np.quantile(v, .9)))
            for k, v in sorted(by.items())}


def build():
    from bpdr_input.experiments.learning.comparison_preparation import fit_streaming_normalizer
    out = ROOT/uuid.uuid4().hex; out.mkdir(parents=True)
    declare(out)
    print('FINE_PITCH_RUN:', out, flush=True)
    train = json.loads((PREP/'training_catalog.json').read_text())['records']
    dev = json.loads(DEV.read_text())['records']
    cache, stats = {}, defaultdict(lambda: defaultdict(int))
    new_train = rebuild(train, 'train', out, cache, stats); print('train rebuilt', len(new_train), flush=True)
    new_dev = rebuild(dev, 'development', out, cache, stats); print('development rebuilt', len(new_dev), flush=True)
    (out/'training_catalog.json').write_text(json.dumps(dict(records=new_train), indent=1), encoding='utf-8')
    (out/'development_catalog.json').write_text(json.dumps(dict(records=new_dev), indent=1), encoding='utf-8')
    with np.load(PREP/'sampling_schedule.npz') as z:
        weights = z['weights'].copy()
    norm = fit_streaming_normalizer(new_train, weights)
    np.savez(out/'normalizer.npz', mean=norm['mean'], scale=norm['scale'])
    with np.load(PREP/'candidate_normalizer.npz') as z:
        old_mean, old_scale = z['mean'], z['scale']
    same = [i for i in range(71) if i not in (PITCH, VOICED, CHANGE)]
    evidence = dict(version=VERSION, declaration_sha256=sha(out/'declaration.json'), waveforms_analyzed=len(cache),
                    frames={k: dict(v) for k, v in stats.items()},
                    unchanged_columns_normalizer_identical=bool(np.allclose(norm['mean'][same], old_mean[same], rtol=1e-5, atol=1e-6)
                                                               and np.allclose(norm['scale'][same], old_scale[same], rtol=1e-5, atol=1e-6)),
                    normalizer=dict(old_pitch_mean_scale=[float(old_mean[PITCH]), float(old_scale[PITCH])],
                                    new_pitch_mean_scale=[float(norm['mean'][PITCH]), float(norm['scale'][PITCH])],
                                    old_change_scale=float(old_scale[CHANGE]), new_change_scale=float(norm['scale'][CHANGE])),
                    realized_shift_training=realized_shift(new_train),
                    artifacts=dict(training_catalog=sha(out/'training_catalog.json'), development_catalog=sha(out/'development_catalog.json'),
                                   normalizer=sha(out/'normalizer.npz')))
    (out/'build_evidence.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    (ROOT/'current.json').write_text(json.dumps(dict(path=str(out)), indent=2), encoding='utf-8')
    print(json.dumps({k: evidence[k] for k in ('frames', 'unchanged_columns_normalizer_identical', 'normalizer', 'realized_shift_training')}, indent=1))
    return out


def correction_metrics(pred, batch):
    """Correction-region (C/D on mask & detuning) and original-view (A/B on mask) absolute errors, plus zero-output correction loss."""
    import torch
    mask, det = batch['mask'], batch['detuning']
    P = mask & (det != 0)
    if not P.any():
        return None
    corr = sum(((pred[v]+det).abs()*P).sum()/P.sum() for v in 'CD')/2
    zero = (det.abs()*P).sum()/P.sum()
    orig = sum((pred[v].abs()*mask).sum()/mask.sum() for v in 'AB')/2
    return float(corr), float(zero), float(orig)


def check():
    import torch
    from bpdr_input.modeling.examples import Normalizer, collate, read_pair
    from bpdr_input.experiments.learning.expanded_comparison import to_device
    from bpdr_input.modeling.losses import LossWeights, objective
    from bpdr_input.experiments.learning.matched_comparison import matched_model
    out = Path(json.loads((ROOT/'current.json').read_text())['path'])
    decl = json.loads((out/'declaration.json').read_text())
    ev = json.loads((out/'build_evidence.json').read_text())
    if ev['declaration_sha256'] != sha(out/'declaration.json') or decl['learning_check'] != CHECK:
        raise ValueError('Declaration changed after build')
    for name, key in (('training_catalog.json', 'training_catalog'), ('normalizer.npz', 'normalizer')):
        if sha(out/name) != ev['artifacts'][key]:
            raise ValueError(f'{name} changed after build')
    if (out/'learning_check').exists():
        raise ValueError('Learning check already executed; do not silently repeat it')
    if not torch.xpu.is_available():
        raise ValueError('Declared XPU unavailable; no CPU fallback')
    train = json.loads((out/'training_catalog.json').read_text())['records']
    with np.load(out/'normalizer.npz') as z:
        normalizer = Normalizer(z['mean'], z['scale'])
    with np.load(PREP/'sampling_schedule.npz') as z:
        schedule = z['indices'][:CHECK['updates']].copy()
    protocol = json.loads((PREP/'locked_protocol.json').read_text())
    folder = out/'learning_check'; folder.mkdir()
    torch.set_num_threads(2)
    model = matched_model(protocol, 71)
    torch.save(model.state_dict(), folder/'initial_state.pt'); model.to('xpu')
    optimizer = torch.optim.AdamW(model.parameters(), lr=CHECK['learning_rate'], weight_decay=CHECK['weight_decay'])
    weights = LossWeights(**protocol['loss_weights'])
    cache, history, started = {}, [], time.monotonic()
    for step, indices in enumerate(schedule, 1):
        model.train(); optimizer.zero_grad(); row = defaultdict(float); n = 0
        for i in indices:
            i = int(i)
            if i not in cache:
                cache[i] = read_pair(train[i], 'engineering')
            batch = to_device(collate([cache[i]], normalizer, 71), 'xpu')
            pred = {v: model(x, batch['padding']) for v, x in batch['inputs'].items()}
            loss, parts, _ = objective(pred, batch, CHECK['condition'], weights, response=False)
            (loss/8).backward()
            row['loss'] += float(loss.detach())/8
            m = correction_metrics({k: v.detach() for k, v in pred.items()}, batch)
            if m:
                row['correction'] += m[0]; row['zero_correction'] += m[1]; row['original_view'] += m[2]; n += 1
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), CHECK['gradient_clip_norm']))
        optimizer.step()
        row = dict(update=step, gradient_norm_before_clip=norm, loss=row['loss'], correction=row['correction']/n,
                   zero_correction=row['zero_correction']/n, original_view=row['original_view']/n)
        row['ratio'] = row['correction']/row['zero_correction']
        history.append(row)
        if step % 10 == 0:
            print('CHECK', step, {k: round(v, 4) for k, v in row.items() if k != 'update'}, 'elapsed', round(time.monotonic()-started), flush=True)
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, folder/'final_state.pt')
    window = [h for h in history if 126 <= h['update'] <= 150]
    ratio = float(np.mean([h['correction'] for h in window])/np.mean([h['zero_correction'] for h in window]))
    original = float(np.mean([h['original_view'] for h in window]))
    # Whole-training-set fit of the final state (training data only).
    model.eval(); fit = defaultdict(list)
    with torch.inference_mode():
        for r in train:
            pair = read_pair(r, 'engineering'); batch = to_device(collate([pair], normalizer, 71), 'xpu')
            pred = {v: model(x, batch['padding']) for v, x in batch['inputs'].items()}
            m = correction_metrics(pred, batch)
            if m:
                P = (batch['mask'] & (batch['detuning'] != 0))
                resp = float(((pred['C']-pred['A'])*P).sum()/P.sum())/float(-(batch['detuning']*P).sum()/P.sum())
                fit[r['dataset']].append((m[0], m[1], m[2], resp))
    fit_summary = {ds: dict(records=len(v), correction_mae=float(np.mean([x[0] for x in v])), zero_mae=float(np.mean([x[1] for x in v])),
                            correction_gain=float(1-np.mean([x[0] for x in v])/np.mean([x[1] for x in v])),
                            original_view_mae=float(np.mean([x[2] for x in v])), median_response_gain=float(np.median([x[3] for x in v])))
                   for ds, v in fit.items()}
    result = dict(declaration_sha256=sha(out/'declaration.json'), updates=len(history), elapsed_seconds=round(time.monotonic()-started),
                  gate=dict(correction_loss_ratio=ratio, original_view_mae_cents=original,
                            passes=bool(ratio <= CHECK['gate']['correction_loss_ratio_to_zero_output_max'] and original <= CHECK['gate']['original_view_mae_cents_max'])),
                  training_fit=fit_summary, final_state_sha256=sha(folder/'final_state.pt'), interpreter=sys.executable, torch=str(torch.__version__))
    (folder/'history.json').write_text(json.dumps(history, indent=1), encoding='utf-8')
    (folder/'evidence.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    {'build': build, 'check': check}[sys.argv[1]]()
