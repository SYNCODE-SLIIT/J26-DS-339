"""Gate E learnability diagnostic v2 on fine-pitch inputs: does the paired BPDR response term provide the learning signal?

v1 (overfit/0537be3d...) showed the augmentation-only objective cannot fit 8 fixed records at output scale 1 or 30.
The only earlier fit that learned (learning_control, 16 October) used the BPDR objective with response terms active
from update 1. Every corpus fit since then started with 100 anchor-only warm-up updates.

Arms (same 8 records chosen by the v1 rule, same init, scale 1, AdamW 1e-3, XPU, 200 updates):
  bpdr_from_1     proposal BPDR objective (L_A + L_B + L_D + 0.1 L_S), response terms from update 1
  bpdr_warmup     the same with the proposal's 100-update anchor-only warm-up
  answer_cue      augmentation-only, column 70 replaced by the private correction/30 (C/D) or 0 (A/B):
                  an optimisation sanity arm that is invalid at runtime and never used outside this diagnostic
Gate per arm: v1 overfit gate (ratio <= 0.5, original-view error <= 1 cent, median response gain >= 0.25).
If a BPDR arm passes, the declaration authorises one full-schedule BPDR check (150 updates, the passing setting,
bpdr_warmup preferred when both pass, unchanged 0.90 gate). Training data only; no model adoption.
"""
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import uuid
import numpy as np
from bpdr_input.experiments.learning.fine_pitch import ROOT, PREP, CHECK, sha, correction_metrics
from bpdr_input.experiments.learning.fine_pitch_overfit import OVERFIT, select

ARMS = dict(bpdr_from_1=dict(condition='bpdr', warmup=0, cue=False),
            bpdr_warmup=dict(condition='bpdr', warmup=100, cue=False),
            answer_cue=dict(condition='augmentation_only', warmup=0, cue=True))


def with_answer_cue(batch):
    """Private diagnostic: replace the boundary column with the correction target / 30 on C/D, 0 on A/B."""
    import torch
    inputs = {}
    for v, x in batch['inputs'].items():
        x = x.clone()
        x[..., 70] = (-batch['detuning']/30.) if v in 'CD' else torch.zeros_like(batch['detuning'])
        inputs[v] = x
    return dict(batch, inputs=inputs)


def fit(train, rows, normalizer, arm, folder, protocol, log):
    import torch
    from bpdr_input.modeling.examples import collate, read_pair
    from bpdr_input.experiments.learning.expanded_comparison import to_device
    from bpdr_input.modeling.losses import LossWeights, objective
    from bpdr_input.experiments.learning.matched_comparison import matched_model
    spec = ARMS[arm]
    model = matched_model(protocol, 71).to('xpu')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=.01)
    weights = LossWeights(**protocol['loss_weights'])
    cache, history, started = {}, [], time.monotonic()

    def prepared(i):
        if i not in cache:
            cache[i] = read_pair(train[i], 'engineering')
        batch = to_device(collate([cache[i]], normalizer, 71), 'xpu')
        return with_answer_cue(batch) if spec['cue'] else batch

    for step, indices in enumerate(rows, 1):
        model.train(); optimizer.zero_grad(); row = defaultdict(float); n = 0
        for i in indices:
            batch = prepared(int(i))
            pred = {v: model(x, batch['padding']) for v, x in batch['inputs'].items()}
            loss, _, _ = objective(pred, batch, spec['condition'], weights, response=step > spec['warmup'])
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
    model.eval(); per = []
    with torch.inference_mode():
        for i in sorted({int(i) for r in rows for i in r}):
            batch = prepared(i)
            pred = {v: model(x, batch['padding']) for v, x in batch['inputs'].items()}
            P = batch['mask'] & (batch['detuning'] != 0)
            target = float((-batch['detuning']*P).sum()/P.sum())
            gain = float(((pred['C']-pred['A'])*P).sum()/P.sum())/target
            m = correction_metrics(pred, batch)
            per.append(dict(index=i, pair_id=train[i]['phrase_id']+f"|{train[i]['cents']:+g}|{train[i]['corruption']}", dataset=train[i]['dataset'],
                            correction_mae=m[0], zero_mae=m[1], original_view_mae=m[2], response_gain=gain))
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, folder/'final_state.pt')
    (folder/'history.json').write_text(json.dumps(history, indent=1), encoding='utf-8')
    (folder/'per_record.json').write_text(json.dumps(per, indent=1), encoding='utf-8')
    return history, per


def gate(history, per, lo, hi, g):
    window = [h for h in history if lo <= h['update'] <= hi]
    ratio = float(np.mean([h['correction'] for h in window])/np.mean([h['zero'] for h in window]))
    original = float(np.mean([h['original'] for h in window]))
    response = float(np.median([p['response_gain'] for p in per]))
    return ratio, original, response


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
    out = base/'overfit_v2'/uuid.uuid4().hex; out.mkdir(parents=True)
    full = dict(CHECK, condition='bpdr', authorised_only_if='a BPDR arm passes the overfit gate; use bpdr_warmup setting if both pass')
    decl = dict(declared_utc=datetime.now(timezone.utc).isoformat(), version='fine-pitch-overfit-0.2', fine_pitch_run=str(base),
                arms=ARMS, overfit=OVERFIT, selected=chosen, conditional_full_check=full, source_sha256=sha(__file__),
                previous='overfit v1: augmentation-only failed at output scale 1 and 30 (ratio 0.9998 / 0.99998)')
    (out/'declaration.json').write_text(json.dumps(decl, indent=2), encoding='utf-8')
    print('OVERFIT_V2_RUN:', out, flush=True)
    torch.set_num_threads(2)
    with np.load(base/'normalizer.npz') as z:
        normalizer = Normalizer(z['mean'], z['scale'])
    protocol = json.loads((PREP/'locked_protocol.json').read_text())
    g = OVERFIT['gate']; result = dict(declaration_sha256=sha(out/'declaration.json'), arms={})
    for arm in ARMS:
        folder = out/arm; folder.mkdir()
        history, per = fit(train, [chosen]*OVERFIT['updates'], normalizer, arm, folder, protocol, arm)
        ratio, original, response = gate(history, per, 176, 200, g)
        result['arms'][arm] = dict(correction_loss_ratio=ratio, original_view_mae_cents=original, median_response_gain=response,
                                   passes=bool(ratio <= g['correction_loss_ratio_max'] and original <= g['original_view_mae_cents_max'] and response >= g['median_response_gain_min']))
        print(arm, json.dumps(result['arms'][arm]), flush=True)
    passing = [a for a in ('bpdr_warmup', 'bpdr_from_1') if result['arms'][a]['passes']]
    if passing:
        arm = passing[0]; folder = out/('full_check_'+arm); folder.mkdir()
        with np.load(PREP/'sampling_schedule.npz') as z:
            schedule = z['indices'][:CHECK['updates']].copy()
        history, per = fit(train, list(schedule), normalizer, arm, folder, protocol, 'full_'+arm)
        ratio, original, response = gate(history, per, 126, 150, None)
        by = defaultdict(list)
        for p in per:
            by[p['dataset']].append(p)
        result['full_check'] = dict(arm=arm, correction_loss_ratio=ratio, original_view_mae_cents=original, seen_records_median_response_gain=response,
                                    seen_records_fit={ds: dict(records=len(v), correction_mae=float(np.mean([x['correction_mae'] for x in v])),
                                                               zero_mae=float(np.mean([x['zero_mae'] for x in v])),
                                                               original_view_mae=float(np.mean([x['original_view_mae'] for x in v])),
                                                               median_response_gain=float(np.median([x['response_gain'] for x in v]))) for ds, v in by.items()},
                                    passes=bool(ratio <= CHECK['gate']['correction_loss_ratio_to_zero_output_max'] and original <= CHECK['gate']['original_view_mae_cents_max']))
        print('full', json.dumps(result['full_check']), flush=True)
    (out/'evidence.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


if __name__ == '__main__':
    run()
