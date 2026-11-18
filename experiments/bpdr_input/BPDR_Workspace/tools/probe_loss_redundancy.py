"""No-fit check (plan 13 Nov): are the BPDR response terms collinear with the positive anchor on the frozen batch?"""
import json
import numpy as np
import torch
import compare_balanced_short_objectives as study
from audit_objective_gradients import vector
from bpdr_input.modeling.losses import boundary_invariance, counter_detuning

previous = study.previous
torch.set_num_threads(2)
declaration, rows, orig = study.declare()
data = study.records(rows, orig)
train = [r for r in data if r['group'] not in declaration['folds'][0]]
originals = {r['source_id']: r for r in train if r['kind'] == 'original'}
trace0 = np.load(previous.parent.ROOT / 'fold_1/upper_tail_preservation/physical_trace.npy')[0]
study.set_kind(train, 'split')
tokens, padding, batch = previous.training_batch(train, originals, trace0, 'bpdr')
out = {}
for name, ckpt in (('init_gate_open', None), ('bpdr_fold1_checkpoint', study.ROOT / 'fold_1/bpdr/checkpoint.pt')):
    model = previous.short_experiment.new_model(declaration)
    if ckpt is None:
        with torch.no_grad(): model.activity.bias.fill_(2.0)
    else:
        model.load_state_dict(torch.load(ckpt, weights_only=True, map_location='cpu'))
    model.eval()
    o = model(tokens, padding).reshape(len(trace0), 4, 2) / 30.
    r = dict(zip('ABCD', o.unbind(1)))
    target = -batch['detuning']; mask = batch['mask']; pos = mask & (target != 0); zero_c = mask & (target == 0)
    def mean(x, m): return (x * m).sum() / m.sum().clamp_min(1)
    terms = dict(positive_anchor=.5 * (mean((r['C'] - target).abs(), pos) + mean((r['D'] - target).abs(), pos)),
                 zero_anchor=.5 * (mean(r['A'].abs(), mask) + mean(r['B'].abs(), mask)),
                 counter_detuning=counter_detuning(r, batch['detuning'], mask)[0],
                 boundary_invariance=boundary_invariance(r, mask)[0])
    P = tuple(model.parameters()); g = {k: vector(v, P) for k, v in terms.items()}
    cos = lambda a, b: float(torch.dot(g[a], g[b]) / (g[a].norm() * g[b].norm()).clamp_min(1e-30))
    out[name] = dict(values={k: float(v) for k, v in terms.items()}, norms={k: float(v.norm()) for k, v in g.items()},
                     cos_detuning_vs_positive_anchor=cos('counter_detuning', 'positive_anchor'),
                     cos_detuning_vs_zero_anchor=cos('counter_detuning', 'zero_anchor'),
                     cos_boundary_vs_positive_anchor=cos('boundary_invariance', 'positive_anchor'))
root = study.ROOT.parent / '2026-11-13_loss_redundancy'; root.mkdir(exist_ok=True)
(root / 'results.json').write_text(json.dumps(dict(scope='fold-1 frozen batch, no fit', results=out), indent=1))
print(json.dumps(out, indent=1))
