"""Gate E full-schedule learning check with a squared-error supervised anchor on the fine-pitch inputs.

Evidence that motivated it (training data only):
- overfit v1/v2 (declared): with the mean-absolute anchor, 8 fixed records cannot be fitted, even with the answer given
  in the input (ratio 0.9998) or with BPDR response terms.
- scratch harness debug (not of record, scripts copied into this run): same 8 records, real inputs, squared-error anchor:
  ratio 0.273 after 100 updates and 0.076 after 200, original-view error 0.13-0.20 cents.
Reason: with a mean-absolute anchor every zero-target frame pulls with full strength however small its error, so
the ~3% correction frames cannot move the shared network away from zero output.

Proposal fit: section 3.6 defines d_M (mean absolute) for L_B and L_D only; L_A "compares predictions with trustworthy
targets" without fixing a distance. This step changes only L_A to a masked mean squared error in cents^2 (same masks,
same view averaging); smoothness and both response terms keep d_M. Recorded as an implementation decision.

Check (declared before fitting): augmentation-only, 150 updates of the frozen schedule, same init/optimizer/XPU as the
failed check; gate: correction loss over updates 126-150 <= 0.90 of zero output and original-view error <= 1 cent.
Then whole-training-set fit by dataset. Training data only; no development data; no model adoption.
"""
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time
import uuid
import numpy as np
from .fine_pitch import ROOT, PREP, CHECK, sha, correction_metrics

SCRATCH = [Path(p) for p in sys.argv[1:]]


def squared_objective(r, batch, condition, weights, response):
    """Proposal objective with L_A as masked mean squared error; L_S, L_B, L_D unchanged (mean absolute)."""
    import torch
    from .losses import smoothness, boundary_invariance, counter_detuning
    mask, det = batch['mask'], batch['detuning']
    m = mask.to(det.dtype); frames = m.sum(1); used = frames > 0
    parts = {}
    def mse(pred, target):
        return ((((pred-target)**2)*m).sum(1)[used]/frames[used]).mean()
    parts['anchor'] = sum(mse(r[v], torch.zeros_like(det) if v in 'AB' else -det) for v in 'ABCD')/4
    parts['smoothness'] = sum(smoothness(r[v], batch['voiced']['A' if v in 'AB' else 'C'])[0] for v in 'ABCD')/4
    total = parts['anchor']+weights.smoothness*parts['smoothness']
    if condition == 'bpdr' and response:
        parts['boundary'] = boundary_invariance(r, mask)[0]; parts['detuning'] = counter_detuning(r, det, mask)[0]
        total = total+weights.boundary*parts['boundary']+weights.detuning*parts['detuning']
    return total, parts


def run():
    import torch
    from .examples import Normalizer, collate, read_pair
    from .expanded_comparison import to_device
    from .losses import LossWeights
    from .matched_comparison import matched_model
    base = Path(json.loads((ROOT/'current.json').read_text())['path'])
    ev = json.loads((base/'build_evidence.json').read_text())
    if sha(base/'training_catalog.json') != ev['artifacts']['training_catalog'] or sha(base/'normalizer.npz') != ev['artifacts']['normalizer']:
        raise ValueError('Fine-pitch artifacts changed')
    if not torch.xpu.is_available():
        raise ValueError('Declared XPU unavailable; no CPU fallback')
    out = base/'mse_check'/uuid.uuid4().hex; out.mkdir(parents=True)
    for p in SCRATCH:
        shutil.copy2(p, out/('scratch_'+p.name))
    decl = dict(declared_utc=datetime.now(timezone.utc).isoformat(), version='fine-pitch-mse-anchor-check-0.1', fine_pitch_run=str(base),
                change='L_A: masked mean squared error (cents^2); L_S, L_B, L_D unchanged', check=CHECK, source_sha256=sha(__file__),
                scratch_evidence={('scratch_'+p.name): sha(p) for p in SCRATCH})
    (out/'declaration.json').write_text(json.dumps(decl, indent=2), encoding='utf-8')
    print('MSE_CHECK_RUN:', out, flush=True)
    torch.set_num_threads(2)
    train = json.loads((base/'training_catalog.json').read_text())['records']
    with np.load(base/'normalizer.npz') as z:
        normalizer = Normalizer(z['mean'], z['scale'])
    with np.load(PREP/'sampling_schedule.npz') as z:
        schedule = z['indices'][:CHECK['updates']].copy()
    protocol = json.loads((PREP/'locked_protocol.json').read_text())
    model = matched_model(protocol, 71).to('xpu')
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
            loss, _ = squared_objective(pred, batch, CHECK['condition'], weights, False)
            (loss/8).backward()
            m = correction_metrics({k: v.detach() for k, v in pred.items()}, batch)
            row['loss'] += float(loss.detach())/8
            if m:
                row['correction'] += m[0]; row['zero'] += m[1]; row['original'] += m[2]; n += 1
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.))
        optimizer.step()
        h = dict(update=step, loss=row['loss'], correction=row['correction']/n, zero=row['zero']/n, original=row['original']/n, gradient_norm_before_clip=norm)
        history.append(h)
        if step % 10 == 0:
            print('MSE', step, {k: round(v, 4) for k, v in h.items() if k != 'update'}, 'elapsed', round(time.monotonic()-started), flush=True)
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, out/'final_state.pt')
    window = [h for h in history if 126 <= h['update'] <= 150]
    ratio = float(np.mean([h['correction'] for h in window])/np.mean([h['zero'] for h in window]))
    original = float(np.mean([h['original'] for h in window]))
    model.eval(); fit = defaultdict(list); seen = {int(i) for r in schedule for i in r}
    with torch.inference_mode():
        for i, r in enumerate(train):
            batch = to_device(collate([read_pair(r, 'engineering')], normalizer, 71), 'xpu')
            pred = {v: model(x, batch['padding']) for v, x in batch['inputs'].items()}
            m = correction_metrics(pred, batch)
            P = batch['mask'] & (batch['detuning'] != 0)
            gain = float(((pred['C']-pred['A'])*P).sum()/P.sum())/float((-batch['detuning']*P).sum()/P.sum())
            fit[(r['dataset'], i in seen)].append((m[0], m[1], m[2], gain))
    summary = {f'{ds}|{"seen" if s else "unseen_in_150_updates"}': dict(records=len(v), correction_mae=float(np.mean([x[0] for x in v])),
               zero_mae=float(np.mean([x[1] for x in v])), correction_gain=float(1-np.mean([x[0] for x in v])/np.mean([x[1] for x in v])),
               original_view_mae=float(np.mean([x[2] for x in v])), median_response_gain=float(np.median([x[3] for x in v])))
               for (ds, s), v in sorted(fit.items())}
    g = CHECK['gate']
    result = dict(declaration_sha256=sha(out/'declaration.json'), updates=len(history), elapsed_seconds=round(time.monotonic()-started),
                  gate=dict(correction_loss_ratio=ratio, original_view_mae_cents=original,
                            passes=bool(ratio <= g['correction_loss_ratio_to_zero_output_max'] and original <= g['original_view_mae_cents_max'])),
                  training_fit=summary, final_state_sha256=sha(out/'final_state.pt'))
    (out/'history.json').write_text(json.dumps(history, indent=1), encoding='utf-8')
    (out/'evidence.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    run()
