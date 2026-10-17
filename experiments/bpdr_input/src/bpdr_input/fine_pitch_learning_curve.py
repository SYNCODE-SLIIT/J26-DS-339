"""Gate E learning-curve test with held-out singers inside the training partition.

Question: with enough training, does the transformer learn to undo a 10-30-cent in-note detune on voices it has
never heard? Every earlier corpus fit ran only 150-300 updates (2-4 passes per record).

Split (rule fixed before any fit): sort the Jamendo artist-proxy group_ids and the Vocadito singer group_ids
separately; every 4th group in each sorted list (positions 0, 4, 8, ...) is held out. All variants of a held-out
group are held out. Development, PP1 and final-holdout data are not read.

Fit: augmentation-only, squared-error L_A + 0.1 L_S, fine phrase-relative pitch standardized with moments fitted on
the trained groups only (frozen hierarchical weights renormalized), all other columns from the fine-pitch training
normalizer. Same init, AdamW 1e-3, weight decay 0.01, clip 1, effective batch 8, XPU. 2,400 updates; the schedule
draws 8 trained records per update without replacement within an update, with probability proportional to the
renormalized hierarchical weights, numpy seed 339.

Every 200 updates (evaluation mode): median paired response gain, correction gain and mean unshifted (A/B)
alteration on all held-out records and on a fixed seed-339 sample of 150 trained records; held-out results are
also reported per dataset.
Gate (final update, held-out records only): median paired response gain >= 0.25 and unshifted alteration <= 1 cent.
Vocadito held-out results are reported separately and never hidden by Jamendo. No model adoption.
"""
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import uuid
import numpy as np
from .fine_pitch import ROOT, PREP, PITCH, VOICED, sha, correction_metrics
from .fine_pitch_relative_check import relative_pair

UPDATES, EVERY, SAMPLE, SEED = 2400, 200, 150, 339
GATE = dict(heldout_median_response_gain_min=.25, heldout_original_view_mae_cents_max=1.)


def split(records):
    """Return (trained indices, held-out indices, held-out groups) by the fixed every-4th-group rule."""
    held = set()
    for ds in ('jamendo', 'vocadito'):
        groups = sorted({r['group_id'] for r in records if r['dataset'] == ds})
        held |= set(groups[::4])
    trained = [i for i, r in enumerate(records) if r['group_id'] not in held]
    heldout = [i for i, r in enumerate(records) if r['group_id'] in held]
    return trained, heldout, sorted(held)


def schedule(weights, trained, updates=UPDATES, seed=SEED):
    w = np.asarray(weights, dtype=np.float64)[trained]; w = w/w.sum()
    rng = np.random.default_rng(seed)
    return np.stack([np.asarray(trained)[rng.choice(len(trained), size=8, replace=False, p=w)] for _ in range(updates)])


def evaluate(model, records, indices, normalizer, cache):
    import torch
    from .examples import collate, read_pair
    from .expanded_comparison import to_device
    model.eval(); rows = []
    with torch.inference_mode():
        for i in indices:
            pair = cache[i] if i in cache else relative_pair(read_pair(records[i], 'engineering'))
            batch = to_device(collate([pair], normalizer, 71), 'xpu')
            pred = {v: model(x, batch['padding']) for v, x in batch['inputs'].items()}
            c, z, o = correction_metrics(pred, batch)
            P = batch['mask'] & (batch['detuning'] != 0)
            gain = float(((pred['C']-pred['A'])*P).sum()/P.sum())/float((-batch['detuning']*P).sum()/P.sum())
            rows.append(dict(index=int(i), dataset=records[i]['dataset'], cents=records[i]['cents'], correction_mae=c, zero_mae=z, original_view_mae=o, response_gain=gain))
    model.train()
    return rows


def summarize(rows):
    if not rows:
        return None
    return dict(records=len(rows), median_response_gain=float(np.median([r['response_gain'] for r in rows])),
                correction_gain=float(1-np.mean([r['correction_mae'] for r in rows])/np.mean([r['zero_mae'] for r in rows])),
                original_view_mae=float(np.mean([r['original_view_mae'] for r in rows])))


def run():
    import torch
    from .examples import Normalizer, collate, read_pair
    from .expanded_comparison import to_device
    from .fine_pitch_mse_check import squared_objective
    from .losses import LossWeights
    from .matched_comparison import matched_model
    base = Path(json.loads((ROOT/'current.json').read_text())['path'])
    ev = json.loads((base/'build_evidence.json').read_text())
    if sha(base/'training_catalog.json') != ev['artifacts']['training_catalog'] or sha(base/'normalizer.npz') != ev['artifacts']['normalizer']:
        raise ValueError('Fine-pitch artifacts changed')
    if not torch.xpu.is_available():
        raise ValueError('Declared XPU unavailable; no CPU fallback')
    records = json.loads((base/'training_catalog.json').read_text())['records']
    with np.load(PREP/'sampling_schedule.npz') as z:
        weights = z['weights'].copy()
    trained, heldout, held_groups = split(records)
    plan = schedule(weights, trained)
    sample = sorted(np.random.default_rng(SEED).choice(trained, size=SAMPLE, replace=False).tolist())
    out = base/'learning_curve'/uuid.uuid4().hex; out.mkdir(parents=True)
    np.savez(out/'schedule.npz', indices=plan, trained=np.asarray(trained), heldout=np.asarray(heldout), trained_sample=np.asarray(sample))
    count = lambda idx, ds: sum(records[i]['dataset'] == ds for i in idx)
    decl = dict(declared_utc=datetime.now(timezone.utc).isoformat(), version='fine-pitch-learning-curve-0.1', fine_pitch_run=str(base),
                held_out_groups=held_groups, trained_records=dict(jamendo=count(trained, 'jamendo'), vocadito=count(trained, 'vocadito')),
                heldout_records=dict(jamendo=count(heldout, 'jamendo'), vocadito=count(heldout, 'vocadito')),
                updates=UPDATES, evaluate_every=EVERY, trained_sample=SAMPLE, gate=GATE, schedule_sha256=sha(out/'schedule.npz'),
                objective='augmentation-only, squared-error L_A + 0.1 L_S', input='fine YIN pitch, phrase-relative, standardized on trained groups only',
                source_sha256=sha(__file__))
    (out/'declaration.json').write_text(json.dumps(decl, indent=2), encoding='utf-8')
    print('LEARNING_CURVE_RUN:', out, json.dumps({k: decl[k] for k in ('held_out_groups', 'trained_records', 'heldout_records')}), flush=True)
    torch.set_num_threads(2)
    # Relative-pitch moments from trained groups only.
    w = weights[trained]/weights[trained].sum(); m1 = m2 = 0.
    for i, wi in zip(trained, w):
        pair = relative_pair(read_pair(records[i], 'engineering'))
        for v in 'ABCD':
            x = pair['inputs'][v][:, PITCH].astype(np.float64); m1 += wi/4*x.mean(); m2 += wi/4*np.square(x).mean()
    with np.load(base/'normalizer.npz') as z:
        mean, scale = z['mean'].copy(), z['scale'].copy()
    mean[PITCH], scale[PITCH] = m1, np.sqrt(max(m2-m1*m1, 0.))
    np.savez(out/'normalizer.npz', mean=mean, scale=scale); normalizer = Normalizer(mean, scale)
    protocol = json.loads((PREP/'locked_protocol.json').read_text()); lw = LossWeights(**protocol['loss_weights'])
    model = matched_model(protocol, 71).to('xpu')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=.01)
    cache, history, curve, started = {}, [], [], time.monotonic()
    model.train()
    for step, indices in enumerate(plan, 1):
        optimizer.zero_grad(); loss_sum = 0.
        for i in indices:
            i = int(i)
            # Read from disk every time: ~1 GB of cached pairs does not fit safely in free memory.
            batch = to_device(collate([relative_pair(read_pair(records[i], 'engineering'))], normalizer, 71), 'xpu')
            pred = {v: model(x, batch['padding']) for v, x in batch['inputs'].items()}
            loss, _ = squared_objective(pred, batch, 'augmentation_only', lw, False)
            (loss/8).backward(); loss_sum += float(loss.detach())/8
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.))
        optimizer.step()
        history.append(dict(update=step, loss=loss_sum, gradient_norm_before_clip=norm))
        if step % EVERY == 0:
            held_rows = evaluate(model, records, heldout, normalizer, cache)
            train_rows = evaluate(model, records, sample, normalizer, cache)
            point = dict(update=step, elapsed_seconds=round(time.monotonic()-started), trained_sample=summarize(train_rows), heldout=summarize(held_rows),
                         heldout_by_dataset={ds: summarize([r for r in held_rows if r['dataset'] == ds]) for ds in ('jamendo', 'vocadito')})
            curve.append(point)
            torch.save({k: v.cpu() for k, v in model.state_dict().items()}, out/f'state_{step:05d}.pt')
            (out/'curve.json').write_text(json.dumps(curve, indent=1), encoding='utf-8')
            (out/'history.json').write_text(json.dumps(history), encoding='utf-8')
            print('CURVE', json.dumps(point), flush=True)
    final = curve[-1]['heldout']
    result = dict(declaration_sha256=sha(out/'declaration.json'), curve=curve, gate=dict(heldout_final=final,
                  passes=bool(final['median_response_gain'] >= GATE['heldout_median_response_gain_min'] and final['original_view_mae'] <= GATE['heldout_original_view_mae_cents_max'])))
    (out/'evidence.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result['gate'], indent=2))
    return result


if __name__ == '__main__':
    run()
