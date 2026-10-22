"""Common-support metrics; masked gaps and window boundaries never form transitions."""
import numpy as np
from .data import IGNORE


def metrics(y,pred):
    y,pred=np.asarray(y),np.asarray(pred)
    valid=y!=IGNORE
    if not valid.any():raise ValueError('No labeled positions')
    cm=np.bincount(3*y[valid]+pred[valid],minlength=9).reshape(3,3)
    support=cm.sum(1);precision=np.diag(cm)/np.maximum(cm.sum(0),1)
    recall=np.diag(cm)/np.maximum(support,1)
    f1=2*precision*recall/np.maximum(precision+recall,1e-12)
    pairs=valid[:,:-1]&valid[:,1:]
    changes=(y[:,:-1]!=y[:,1:])&pairs
    guesses=(pred[:,:-1]!=pred[:,1:])&pairs
    tp=int((changes&guesses).sum());fp=int((~changes&guesses).sum());fn=int((changes&~guesses).sum())
    correct_pairs=(pred[:,:-1]==y[:,:-1])&(pred[:,1:]==y[:,1:])
    true_changes=int(changes.sum());n_pairs=int(pairs.sum())
    return {'accuracy':float(np.trace(cm)/cm.sum()),'macro_f1':float(f1.mean()),
            'balanced_accuracy':float(recall.mean()),'per_class_f1':f1.tolist(),'per_class_recall':recall.tolist(),
            'support':support.tolist(),'confusion':cm.tolist(),'valid_positions':int(valid.sum()),
            'change_f1':2*tp/max(2*tp+fp+fn,1),'change_precision':tp/max(tp+fp,1),
            'change_recall':tp/max(tp+fn,1),'true_changes':true_changes,'valid_pairs':n_pairs,
            'true_change_rate':true_changes/max(n_pairs,1),'predicted_change_rate':int(guesses.sum())/max(n_pairs,1),
            'transition_pair_accuracy':int((changes&correct_pairs).sum())/max(true_changes,1)}


def selection_score(result):
    return .7*result['macro_f1']+.3*result['change_f1']


def bar_oracle(y,bar_index,bars):
    """Optimistic per-bar gold-majority accuracy ceiling on SAME beat support.

    Only an accuracy ceiling; not an F1 ceiling or an inference baseline.
    """
    pred=np.zeros_like(y)
    for i in range(len(y)):
        for bar in range(bars):
            mask=(bar_index[i]==bar)&(y[i]>=0)
            winner=np.bincount(y[i,mask],minlength=3).argmax() if mask.any() else 0
            pred[i,bar_index[i]==bar]=winner
    return metrics(y,pred)


def subgroup_metrics(y,pred,examples):
    rows=[]
    for kind,groups in [('mode',examples['mode']),('constant',examples.function_changes.eq(0).map({True:'no_observed_change',False:'has_observed_changes'}))]:
        for label in sorted(groups.unique()):
            mask=np.asarray(groups==label)
            if (y[mask]>=0).any():rows.append({'group':kind,'label':str(label),'windows':int(mask.sum()),**metrics(y[mask],pred[mask])})
    # Tags overlap: these rows must never be summed.
    from collections import Counter
    for genre,count in Counter(g for gs in examples.genres for g in gs).most_common(12):
        mask=np.array([genre in gs for gs in examples.genres])
        if (y[mask]>=0).any():rows.append({'group':'overlapping_genre','label':genre,'windows':count,**metrics(y[mask],pred[mask])})
    return rows


def artist_bootstrap(y,pred,baseline,examples,repeats=300,seed=2026):
    """Paired 95% percentile intervals, resampling artists not correlated beats."""
    artists=sorted(examples.artist_id.unique());groups=[np.flatnonzero(np.asarray(examples.artist_id==a)) for a in artists]
    rng=np.random.default_rng(seed);scores=[]
    for _ in range(repeats):
        idx=np.concatenate([groups[j] for j in rng.integers(0,len(groups),len(groups))])
        a,b=metrics(y[idx],pred[idx]),metrics(y[idx],baseline[idx])
        scores.append([a['macro_f1']-b['macro_f1'],a['change_f1']-b['change_f1']])
    interval=np.percentile(scores,[2.5,97.5],axis=0)
    return {name:{'lower':float(interval[0,j]),'upper':float(interval[1,j])} for j,name in enumerate(['macro_f1_delta','change_f1_delta'])}
