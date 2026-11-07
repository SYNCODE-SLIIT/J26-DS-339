"""Distinguish fitting failure from singer-transfer failure without another fit."""
from pathlib import Path
import sys,json
import numpy as np
import torch

WS=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(WS/'source'));sys.path.insert(0,str(WS/'tools'))
import compare_short_phrase_models as study
from probe_short_phrase_renderer import sha


def main():
    torch.set_num_threads(2);d,rows,originals=study.declare();data=study.records(rows,originals)
    root=study.ROOT/'generalization_diagnosis';root.mkdir(exist_ok=True)
    assert (study.ROOT/'integrity_receipt.json').exists();results=[]
    for index,groups in enumerate(d['folds'],1):
        folder=study.ROOT/f'fold_{index}/short_transformer'
        receipt=study.read(folder/'fit_receipt.json');path=folder/'checkpoint.pt'
        assert sha(path)==receipt['checkpoint_sha256']
        model,_=study.short.load(path)
        train=[r for r in data if r['group'] not in groups];test=[r for r in data if r['group'] in groups]
        pred={r['id']:study.predict(model,r,'short_transformer') for r in train}
        np.savez_compressed(root/f'fold_{index}_training_predictions.npz',**pred)
        with np.load(folder/'heldout_predictions.npz') as z:test_pred={k:z[k].copy() for k in z.files}
        train_summary=study.summarize(train,pred);test_summary=study.summarize(test,test_pred)
        results.append(dict(fold=index,checkpoint_sha256=sha(path),training=train_summary,heldout=test_summary,
                            training_prediction_sha256=sha(root/f'fold_{index}_training_predictions.npz')))
        print('SHORT_GENERALIZATION',index,train_summary['positive_singer_macro_mae_cents'],test_summary['positive_singer_macro_mae_cents'],flush=True)
    study.write(root/'results.json',dict(script_sha256=sha(__file__),
        model_integrity_sha256=sha(study.ROOT/'integrity_receipt.json'),folds=results,
        new_fits=0,scope='Known training-source diagnostic; no causal claim that architecture alone causes transfer failure',
        production_eligible=False,research_validation_passed=False))


if __name__=='__main__':main()
