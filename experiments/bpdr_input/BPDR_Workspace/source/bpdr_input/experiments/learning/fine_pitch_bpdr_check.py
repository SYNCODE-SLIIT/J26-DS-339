"""Gate E learning check: the proposal's full BPDR objective on fine, phrase-relative pitch.

Objective (proposal 3.6): L = L_A + L_B + L_D + 0.1 L_S, with L_A a masked squared error (fine_pitch_mse_check) and
L_B, L_D, L_S unchanged (mean absolute d_M). Input: fine YIN pitch made phrase-relative and standardized with the
training-fitted moments (fine_pitch_relative_check, arm relative_standardized). Same init, AdamW 1e-3, clip 1,
effective batch 8, XPU, the full frozen 300x8 schedule.

Arms (declared before fitting):
  bpdr_warmup100  proposal schedule: anchor+smoothness for updates 1-100, response terms from update 101 (primary)
  bpdr_from_1     response terms from update 1 (the setting of the only earlier fit that learned; sensitivity arm)
Gate per arm (evaluation mode, all 995 training records): median paired response gain >= 0.25 and mean original-view
alteration <= 1 cent. Training data only; no development data; no model adoption.
"""
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import uuid
import numpy as np
from bpdr_input.experiments.learning.fine_pitch import ROOT, PREP, PITCH, sha, correction_metrics
from bpdr_input.experiments.learning.fine_pitch_relative_check import GATE, relative_pair, pitch_moments

UPDATES = 300
ARMS = dict(bpdr_warmup100=100, bpdr_from_1=0)


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
    out = base/'bpdr_check'/uuid.uuid4().hex; out.mkdir(parents=True)
    decl = dict(declared_utc=datetime.now(timezone.utc).isoformat(), version='fine-pitch-bpdr-check-0.1', fine_pitch_run=str(base),
                arms=ARMS, updates=UPDATES, gate=GATE, objective='BPDR: squared L_A + L_B + L_D + 0.1 L_S (L_B, L_D, L_S mean absolute)',
                input='fine YIN pitch, phrase-relative, standardized with training-fitted moments',
                previous='relative_check: augmentation-only failed the response gate (0.0008 / 0.0002)', source_sha256=sha(__file__))
    (out/'declaration.json').write_text(json.dumps(decl, indent=2), encoding='utf-8')
    print('BPDR_CHECK_RUN:', out, flush=True)
    torch.set_num_threads(2)
    train = json.loads((base/'training_catalog.json').read_text())['records']
    with np.load(PREP/'sampling_schedule.npz') as z:
        schedule, weights = z['indices'].copy(), z['weights'].copy()
    if schedule.shape != (UPDATES, 8):
        raise ValueError('Frozen schedule changed')
    with np.load(base/'normalizer.npz') as z:
        mean, scale = z['mean'].copy(), z['scale'].copy()
    mean[PITCH], scale[PITCH] = pitch_moments(train, weights)
    normalizer = Normalizer(mean, scale); np.savez(out/'normalizer.npz', mean=mean, scale=scale)
    protocol = json.loads((PREP/'locked_protocol.json').read_text())
    lw = LossWeights(**protocol['loss_weights'])
    seen = {int(i) for row in schedule for i in row}
    result = dict(declaration_sha256=sha(out/'declaration.json'), relative_pitch_moments=[float(mean[PITCH]), float(scale[PITCH])], arms={})
    for arm, warmup in ARMS.items():
        folder = out/arm; folder.mkdir()
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
                loss, parts = squared_objective(pred, batch, 'bpdr', lw, step > warmup)
                (loss/8).backward()
                m = correction_metrics({k: v.detach() for k, v in pred.items()}, batch)
                row['loss'] += float(loss.detach())/8
                for k in ('detuning', 'boundary'):
                    if k in parts: row[k] += float(parts[k].detach())/8
                if m:
                    row['correction'] += m[0]; row['zero'] += m[1]; row['original'] += m[2]; n += 1
            norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.))
            optimizer.step()
            h = dict(update=step, loss=row['loss'], counter_detuning=row['detuning'], boundary=row['boundary'], correction=row['correction']/n,
                     zero=row['zero']/n, original=row['original']/n, gradient_norm_before_clip=norm)
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
                bsens = float(((pred['A']-pred['B']).abs()*batch['mask']).sum()/batch['mask'].sum())
                rows.append(dict(index=i, dataset=r['dataset'], cents=r['cents'], seen=i in seen, correction_mae=c, zero_mae=z,
                                 original_view_mae=o, response_gain=gain, boundary_sensitivity=bsens))
        (folder/'training_records.json').write_text(json.dumps(rows, indent=1), encoding='utf-8')
        groups = defaultdict(list)
        for x in rows:
            groups[f"{x['dataset']}|{'drawn' if x['seen'] else 'never_drawn'}"].append(x); groups['all'].append(x)
        summary = {k: dict(records=len(v), correction_gain=float(1-np.mean([x['correction_mae'] for x in v])/np.mean([x['zero_mae'] for x in v])),
                           median_response_gain=float(np.median([x['response_gain'] for x in v])),
                           original_view_mae=float(np.mean([x['original_view_mae'] for x in v])),
                           boundary_sensitivity=float(np.mean([x['boundary_sensitivity'] for x in v]))) for k, v in sorted(groups.items())}
        by_shift = {str(c): float(np.median([x['response_gain'] for x in rows if x['cents'] == c])) for c in sorted({x['cents'] for x in rows})}
        a = summary['all']
        result['arms'][arm] = dict(summary=summary, median_response_gain_by_injected_cents=by_shift, final_state_sha256=sha(folder/'final_state.pt'),
                                   passes=bool(a['median_response_gain'] >= GATE['median_response_gain_min'] and a['original_view_mae'] <= GATE['original_view_mae_cents_max']))
        print(arm, json.dumps(result['arms'][arm], indent=1), flush=True)
    (out/'evidence.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


if __name__ == '__main__':
    run()
