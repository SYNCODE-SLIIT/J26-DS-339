"""Gate E learnability diagnostic on the fine-pitch inputs: can the proposal's transformer fit 8 fixed training records?

Arms (same init, objective, optimizer, XPU, 200 updates, the same 8 records every update):
  scale_1   head outputs cents directly (the setting of every earlier fit)
  scale_30  output = 30 x head, with head weight/bias divided by 30 at init so initial predictions are identical.
            Losses stay in cents; 30 is the largest injected magnitude, fixed in advance, not searched.
If scale_30 passes the overfit gate, the same declaration authorises one full-schedule check with scale 30
(150 updates, the unchanged 0.90 gate). Training data only; no development data; no model adoption.
"""
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import uuid
import numpy as np
from bpdr_input.experiments.learning.fine_pitch import ROOT, PREP, CHECK, sha, correction_metrics

SCALE = 30.
OVERFIT = dict(updates=200, records=8, gate=dict(window='updates 176-200', correction_loss_ratio_max=.5, original_view_mae_cents_max=1.,
                                                   median_response_gain_min=.25),
               selection='First Jamendo record in catalog order for each of -30,-20,-10,+10,+20,+30 with corruption "split", each from an artist not yet chosen; '
                         'then the first two Vocadito records with corruption "split" and opposite signs from different singers.')
FULL = dict(CHECK, output_scale=SCALE, authorised_only_if='scale_30 passes the overfit gate')


def select(records):
    chosen, artists = [], set()
    for cents in (-30., -20., -10., 10., 20., 30.):
        for i, r in enumerate(records):
            if r['dataset'] == 'jamendo' and r['cents'] == cents and r['corruption'] == 'split' and r['group_id'] not in artists:
                chosen.append(i); artists.add(r['group_id']); break
    signs = set()
    for i, r in enumerate(records):
        s = np.sign(r['cents'])
        if r['dataset'] == 'vocadito' and r['corruption'] == 'split' and s not in signs and r['group_id'] not in artists:
            chosen.append(i); signs.add(s); artists.add(r['group_id'])
        if len(signs) == 2:
            break
    if len(chosen) != 8:
        raise ValueError('Fixed selection rule did not yield 8 records')
    return chosen


def scaled(model, scale):
    """Wrap the encoder so outputs are scale x head, with identical initial predictions."""
    import torch
    with torch.no_grad():
        model.head.weight /= scale; model.head.bias /= scale
    forward = model.forward

    def scaled_forward(features, padding=None):
        return scale*forward(features, padding)
    model.forward = scaled_forward
    return model


def fit(train, indices_per_update, normalizer, scale, folder, protocol, log):
    import torch
    from bpdr_input.modeling.examples import collate, read_pair
    from bpdr_input.experiments.learning.expanded_comparison import to_device
    from bpdr_input.modeling.losses import LossWeights, objective
    from bpdr_input.experiments.learning.matched_comparison import matched_model
    model = matched_model(protocol, 71)
    init = {k: v.clone() for k, v in model.state_dict().items()}
    if scale != 1:
        model = scaled(model, scale)
    model.to('xpu')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=.01)
    weights = LossWeights(**protocol['loss_weights'])
    cache, history, started = {}, [], time.monotonic()
    for step, indices in enumerate(indices_per_update, 1):
        model.train(); optimizer.zero_grad(); row = defaultdict(float); n = 0
        for i in indices:
            i = int(i)
            if i not in cache:
                cache[i] = read_pair(train[i], 'engineering')
            batch = to_device(collate([cache[i]], normalizer, 71), 'xpu')
            pred = {v: model(x, batch['padding']) for v, x in batch['inputs'].items()}
            loss, _, _ = objective(pred, batch, 'augmentation_only', weights, response=False)
            (loss/len(indices)).backward()
            m = correction_metrics({k: v.detach() for k, v in pred.items()}, batch)
            row['loss'] += float(loss.detach())/len(indices)
            if m:
                row['correction'] += m[0]; row['zero'] += m[1]; row['original'] += m[2]; n += 1
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.))
        optimizer.step()
        h = dict(update=step, loss=row['loss'], correction=row['correction']/n, zero=row['zero']/n, original=row['original']/n, gradient_norm_before_clip=norm)
        history.append(h)
        if step % 10 == 0:
            print(log, step, {k: round(v, 4) for k, v in h.items() if k != 'update'}, 'elapsed', round(time.monotonic()-started), flush=True)
    model.eval(); responses, per = [], []
    with torch.inference_mode():
        for i in sorted({int(i) for row in indices_per_update for i in row}):
            batch = to_device(collate([read_pair(train[i], 'engineering')], normalizer, 71), 'xpu')
            pred = {v: model(x, batch['padding']) for v, x in batch['inputs'].items()}
            P = batch['mask'] & (batch['detuning'] != 0)
            target = float((-batch['detuning']*P).sum()/P.sum())
            gain = float(((pred['C']-pred['A'])*P).sum()/P.sum())/target
            m = correction_metrics(pred, batch)
            responses.append(gain)
            per.append(dict(index=i, pair_id=train[i]['phrase_id']+f"|{train[i]['cents']:+g}", dataset=train[i]['dataset'],
                            correction_mae=m[0], zero_mae=m[1], original_view_mae=m[2], response_gain=gain))
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, folder/'final_state.pt')
    (folder/'history.json').write_text(json.dumps(history, indent=1), encoding='utf-8')
    return history, per, float(np.median(responses)), init


def run():
    import torch
    from bpdr_input.modeling.examples import Normalizer
    base = Path(json.loads((ROOT/'current.json').read_text())['path'])
    ev = json.loads((base/'build_evidence.json').read_text())
    if sha(base/'training_catalog.json') != ev['artifacts']['training_catalog'] or sha(base/'normalizer.npz') != ev['artifacts']['normalizer']:
        raise ValueError('Fine-pitch artifacts changed')
    if not torch.xpu.is_available():
        raise ValueError('Declared XPU unavailable; no CPU fallback')
    train = json.loads((base/'training_catalog.json').read_text())['records']
    chosen = select(train)
    out = base/'overfit'/uuid.uuid4().hex; out.mkdir(parents=True)
    decl = dict(declared_utc=datetime.now(timezone.utc).isoformat(), version='fine-pitch-overfit-0.1', fine_pitch_run=str(base),
                overfit=OVERFIT, selected=[dict(index=i, phrase_id=train[i]['phrase_id'], cents=train[i]['cents'], dataset=train[i]['dataset'],
                                                group_id=train[i]['group_id']) for i in chosen],
                arms={'scale_1': 1., 'scale_30': SCALE}, conditional_full_check=FULL, source_sha256=sha(__file__),
                previous_failure='runs/fine_pitch/66e3619e611241dd80d33a32578b73c6/learning_check: ratio 0.99995 at output scale 1')
    (out/'declaration.json').write_text(json.dumps(decl, indent=2), encoding='utf-8')
    print('OVERFIT_RUN:', out, flush=True)
    torch.set_num_threads(2)
    with np.load(base/'normalizer.npz') as z:
        normalizer = Normalizer(z['mean'], z['scale'])
    protocol = json.loads((PREP/'locked_protocol.json').read_text())
    result = dict(declaration_sha256=sha(out/'declaration.json'), arms={})
    g = OVERFIT['gate']
    inits = {}
    for arm, scale in (('scale_1', 1.), ('scale_30', SCALE)):
        folder = out/arm; folder.mkdir()
        history, per, response, inits[arm] = fit(train, [chosen]*OVERFIT['updates'], normalizer, scale, folder, protocol, arm)
        window = [h for h in history if h['update'] > 175]
        ratio = float(np.mean([h['correction'] for h in window])/np.mean([h['zero'] for h in window]))
        original = float(np.mean([h['original'] for h in window]))
        result['arms'][arm] = dict(correction_loss_ratio=ratio, original_view_mae_cents=original, median_response_gain=response, per_record=per,
                                   passes=bool(ratio <= g['correction_loss_ratio_max'] and original <= g['original_view_mae_cents_max']
                                               and response >= g['median_response_gain_min']))
        print(arm, json.dumps({k: v for k, v in result['arms'][arm].items() if k != 'per_record'}), flush=True)
    result['identical_initial_parameters'] = all(torch.equal(inits['scale_1'][k], inits['scale_30'][k]) for k in inits['scale_1'])
    if result['arms']['scale_30']['passes']:
        result['full_check'] = full_check(train, normalizer, protocol, out)
    (out/'evidence.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'arms'}, indent=2))
    return result


def full_check(train, normalizer, protocol, out):
    folder = out/'full_check_scale_30'; folder.mkdir()
    with np.load(PREP/'sampling_schedule.npz') as z:
        schedule = z['indices'][:FULL['updates']].copy()
    history, per, response, _ = fit(train, list(schedule), normalizer, SCALE, folder, protocol, 'full')
    window = [h for h in history if 126 <= h['update'] <= 150]
    ratio = float(np.mean([h['correction'] for h in window])/np.mean([h['zero'] for h in window]))
    original = float(np.mean([h['original'] for h in window]))
    by = defaultdict(list)
    for p in per:
        by[p['dataset']].append(p)
    summary = {ds: dict(records=len(v), correction_mae=float(np.mean([x['correction_mae'] for x in v])), zero_mae=float(np.mean([x['zero_mae'] for x in v])),
                        original_view_mae=float(np.mean([x['original_view_mae'] for x in v])), median_response_gain=float(np.median([x['response_gain'] for x in v])))
               for ds, v in by.items()}
    gate = FULL['gate']
    return dict(correction_loss_ratio=ratio, original_view_mae_cents=original, seen_records_median_response_gain=response, seen_records_fit=summary,
                passes=bool(ratio <= gate['correction_loss_ratio_to_zero_output_max'] and original <= gate['original_view_mae_cents_max']))


if __name__ == '__main__':
    run()
