"""Original and rule requested-correction references on the same 304 dev variants.

These are raw correction-curve scores, matching the neural prediction scores.
They are not post-render or matched-coverage musical quality measurements.
"""
from datetime import datetime, timezone
import uuid

import numpy as np

from bpdr_input.experiments.evaluation.transfer_study import ROOT, FINE, read, write, sha, score_predictions, summaries


def run():
    from bpdr_input.modeling.examples import read_pair
    from bpdr_input.repair.rule_correction import decide
    out=ROOT/uuid.uuid4().hex;out.mkdir(parents=True)
    write(out/'declaration.json',dict(version='presentation-reference-curves-1',declared_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=sha(__file__),catalog_sha256=sha(FINE/'development_catalog.json'),optimizer_updates=0,
        policy='Existing rule-chromatic-0.1, no threshold calibration. Fine-pitch archive guard is retained.',
        scope='Raw requested curves only, as with the neural scoring; no delivered-repair or matched-coverage claim.'))
    rows={'original_audio':[],'rule_based':[]}
    for record in read(FINE/'development_catalog.json')['records']:
        p=read_pair(record,'engineering');r={}
        for v in 'AC':
            x=p['inputs'][v];r[v]=decide(x[:,64],x[:,66]>.5)[0]
        r['B']=r['A'];r['D']=r['C']
        zero={v:np.zeros(len(p['mask'])) for v in 'ABCD'}
        meta=dict(dataset=record['dataset'],group=record['singer_id'],phrase_id=record['phrase_id'],cents=record['cents'],corruption=record['corruption'])
        for name,pred in [('original_audio',zero),('rule_based',r)]:
            rows[name].append(dict(meta,**score_predictions(p,pred)))
    result={k:summaries(rr) for k,rr in rows.items()}
    write(out/'rows.json',rows);write(out/'evidence.json',dict(declaration_sha256=sha(out/'declaration.json'),results=result,optimizer_updates=0))
    write(ROOT/'baselines_current.json',dict(path=str(out)))
    print('BASELINES',out,flush=True)
    return out


if __name__=='__main__':
    run()
