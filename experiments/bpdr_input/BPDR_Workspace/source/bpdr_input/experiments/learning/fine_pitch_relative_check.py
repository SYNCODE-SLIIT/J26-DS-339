"""Gate E learning check: phrase-relative fine pitch, squared-error anchor, 300 updates, gated on paired response.

Input change (BPDR-local, model-input adapter only; archives unchanged): column 64 becomes the fine pitch minus the
median fine pitch of that view's voiced frames (cents), 0 when unvoiced. Two declared normalizations of that column:
  relative_standardized   training-fitted mean/scale with the frozen hierarchical weights (the approved arm)
  relative_100_cents      fixed mean 0, scale 100 cents, so a 30-cent shift is 0.3 input units (declared sensitivity arm,
                          because standardizing by the melody's spread would keep the shift at a few hundredths of a unit)
All other columns use the fine-pitch run's training-only normalizer. Objective: augmentation-only with squared-error
L_A (fine_pitch_mse_check.squared_objective), smoothness 0.1. 300 updates = the full frozen 300x8 schedule; same init,
AdamW 1e-3, clip 1, XPU, effective batch 8.

Gate per arm (evaluation mode, all 995 training records, declared before fitting):
  median paired response gain (mean C-A on the correction frames / correction target) >= 0.25, and
  mean original-view (A/B) absolute alteration <= 1 cent.
Also reported: correction gain and response by dataset, for records drawn by the schedule and records never drawn
(the latter share crops/artists with drawn records, so they are not held-out). Training data only; no development data.
"""
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import uuid
import numpy as np
from bpdr_input.experiments.learning.fine_pitch import ROOT, PREP, PITCH, VOICED, sha, correction_metrics

UPDATES = 300
GATE = dict(median_response_gain_min=.25, original_view_mae_cents_max=1.)
ARMS = ('relative_standardized', 'relative_100_cents')


def relative_pitch(raw):
    """Copy of a 71-column archive with column 64 = pitch minus the view's voiced median (cents); unvoiced 0."""
    out = np.array(raw, dtype=np.float32, copy=True)
    voiced = (out[:, VOICED] > .5) & (out[:, PITCH] != 0)
    if voiced.any():
        out[:, PITCH] = np.where(voiced, out[:, PITCH]-np.median(out[voiced, PITCH]), 0)
    return out


def relative_pair(pair):
    return dict(pair, inputs={v: relative_pitch(x) for v, x in pair['inputs'].items()})


def pitch_moments(train, weights):
    from bpdr_input.modeling.examples import read_pair
    mean = second = 0.
    for r, w in zip(train, weights):
        pair = relative_pair(read_pair(r, 'engineering'))
        for v in 'ABCD':
            x = pair['inputs'][v][:, PITCH].astype(np.float64)
            mean += w/4*x.mean(); second += w/4*np.square(x).mean()
    return float(mean), float(np.sqrt(max(second-mean*mean, 0.)))


def run():
    import torch
    from bpdr_input.modeling.examples import Normalizer, collate, read_pair
    from bpdr_input.experiments.learning.expanded_comparison import to_device
    from bpdr_input.experiments.learning.fine_pitch_mse_check import squared_objective
    from bpdr_input.modeling.losses import LossWeights
    from bpdr_input.experiments.learning.matched_comparison import matched_model
    base = Path(json.loads((ROOT/'current.json').read_text())['path'])
    ev = json.loads((base/'build_evidence.json').read_text())
    if sha(base/'training_catalog.json') != ev['artifacts']['training_catalog'] or sha(base/'normalizer.npz') != ev['artifacts']['normalizer']:
        raise ValueError('Fine-pitch artifacts changed')
    if not torch.xpu.is_available():
        raise ValueError('Declared XPU unavailable; no CPU fallback')
    out = base/'relative_check'/uuid.uuid4().hex; out.mkdir(parents=True)
    decl = dict(declared_utc=datetime.now(timezone.utc).isoformat(), version='fine-pitch-relative-check-0.1', fine_pitch_run=str(base),
                arms=ARMS, updates=UPDATES, gate=GATE, objective='augmentation-only, squared-error L_A, smoothness 0.1',
                previous='mse_check: training-loss gate passed (0.895) but evaluation-mode paired response gain ~0.0006',
                source_sha256=sha(__file__))
    (out/'declaration.json').write_text(json.dumps(decl, indent=2), encoding='utf-8')
    print('RELATIVE_CHECK_RUN:', out, flush=True)
    torch.set_num_threads(2)
    train = json.loads((base/'training_catalog.json').read_text())['records']
    with np.load(PREP/'sampling_schedule.npz') as z:
        schedule, weights = z['indices'].copy(), z['weights'].copy()
    if schedule.shape != (UPDATES, 8):
        raise ValueError('Frozen schedule changed')
    with np.load(base/'normalizer.npz') as z:
        mean0, scale0 = z['mean'].copy(), z['scale'].copy()
    rel_mean, rel_scale = pitch_moments(train, weights)
    print('relative pitch moments', rel_mean, rel_scale, flush=True)
    protocol = json.loads((PREP/'locked_protocol.json').read_text())
    lw = LossWeights(**protocol['loss_weights'])
    seen = {int(i) for row in schedule for i in row}
    result = dict(declaration_sha256=sha(out/'declaration.json'), relative_pitch_moments=[rel_mean, rel_scale], arms={})
    for arm in ARMS:
        mean, scale = mean0.copy(), scale0.copy()
        mean[PITCH], scale[PITCH] = (rel_mean, rel_scale) if arm == 'relative_standardized' else (0., 100.)
        normalizer = Normalizer(mean, scale)
        folder = out/arm; folder.mkdir()
        np.savez(folder/'normalizer.npz', mean=mean, scale=scale)
        model = matched_model(protocol, 71).to('xpu')
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=.01)
        cache, history, started = {}, [], time.monotonic()
        for step, indices in enumerate(schedule, 1):
            model.train(); optimizer.zero_grad(); row = defaultdict(float); n = 0
            for i in indices:
                i = int(i)
                if i not in cache:
                    cache[i] = relative_pair(read_pair(train[i], 'engineering'))
                batch = to_device(collate([cache[i]], normalizer, 71), 'xpu')
                pred = {v: model(x, batch['padding']) for v, x in batch['inputs'].items()}
                loss, _ = squared_objective(pred, batch, 'augmentation_only', lw, False)
                (loss/8).backward()
                m = correction_metrics({k: v.detach() for k, v in pred.items()}, batch)
                row['loss'] += float(loss.detach())/8
                if m:
                    row['correction'] += m[0]; row['zero'] += m[1]; row['original'] += m[2]; n += 1
            norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.))
            optimizer.step()
            h = dict(update=step, loss=row['loss'], correction=row['correction']/n, zero=row['zero']/n, original=row['original']/n, gradient_norm_before_clip=norm)
            history.append(h)
            if step % 25 == 0:
                print(arm, step, {k: round(v, 4) for k, v in h.items() if k != 'update'}, 'elapsed', round(time.monotonic()-started), flush=True)
        cache.clear()
        torch.save({k: v.cpu() for k, v in model.state_dict().items()}, folder/'final_state.pt')
        (folder/'history.json').write_text(json.dumps(history, indent=1), encoding='utf-8')
        model.eval(); rows = []
        with torch.inference_mode():
            for i, r in enumerate(train):
                batch = to_device(collate([relative_pair(read_pair(r, 'engineering'))], normalizer, 71), 'xpu')
                pred = {v: model(x, batch['padding']) for v, x in batch['inputs'].items()}
                c, z, o = correction_metrics(pred, batch)
                P = batch['mask'] & (batch['detuning'] != 0)
                gain = float(((pred['C']-pred['A'])*P).sum()/P.sum())/float((-batch['detuning']*P).sum()/P.sum())
                rows.append(dict(index=i, dataset=r['dataset'], cents=r['cents'], seen=i in seen, correction_mae=c, zero_mae=z, original_view_mae=o, response_gain=gain))
        (folder/'training_records.json').write_text(json.dumps(rows, indent=1), encoding='utf-8')
        groups = defaultdict(list)
        for x in rows:
            groups[f"{x['dataset']}|{'drawn' if x['seen'] else 'never_drawn'}"].append(x)
            groups['all'].append(x)
        summary = {k: dict(records=len(v), correction_gain=float(1-np.mean([x['correction_mae'] for x in v])/np.mean([x['zero_mae'] for x in v])),
                           median_response_gain=float(np.median([x['response_gain'] for x in v])),
                           original_view_mae=float(np.mean([x['original_view_mae'] for x in v]))) for k, v in sorted(groups.items())}
        by_shift = {str(c): float(np.median([x['response_gain'] for x in rows if x['cents'] == c])) for c in sorted({x['cents'] for x in rows})}
        a = summary['all']
        result['arms'][arm] = dict(summary=summary, median_response_gain_by_injected_cents=by_shift, final_state_sha256=sha(folder/'final_state.pt'),
                                   passes=bool(a['median_response_gain'] >= GATE['median_response_gain_min'] and a['original_view_mae'] <= GATE['original_view_mae_cents_max']))
        print(arm, json.dumps(result['arms'][arm], indent=1), flush=True)
    (out/'evidence.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


if __name__ == '__main__':
    run()
