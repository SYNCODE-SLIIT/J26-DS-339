"""Prospective fixed median/quartile readouts; frozen model and WAV gates."""
import argparse
from datetime import datetime,timezone
from functools import partial
from pathlib import Path
import sys
import torch

WORKSPACE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(WORKSPACE/'source'))
from bpdr_input.experiments.learning import context_improvement as ex,held_voice_training as held
from bpdr_input.modeling import candidate_distribution_projection as readout
from bpdr_input.reporting.transformer_accuracy_report import complete_report
import evaluate_candidate_center_readout as previous
import run_transformer_curve_study as curve

ROOT=WORKSPACE/'results/transformer_usability/2026-10-30_v25_distribution_readout'
MODES=('median','upper_quartile')


def declare():
    ROOT.mkdir(exist_ok=True)
    old=previous.declare()
    if not (ROOT/'declaration.json').exists():
        retained=('folds','default_seed','policy','curve_condition','renderer','input_hashes',
                  'checkpoint_hashes','prediction_hashes','full_refits','full_refit_hashes',
                  'baseline_cv','baseline_cv_sha256','baseline_evidence','baseline_evidence_sha256')
        d={k:old[k] for k in retained}
        sources=dict(old['source_hashes'])
        for p in (Path(__file__).resolve(),Path(readout.__file__)):sources[str(p)]=ex.sha(p)
        d.update(created_at=datetime.now(timezone.utc).isoformat(),model_family='joint_candidate_rank_readout',
                 decoder_version=readout.VERSION,modes=list(MODES),source_hashes=sources,
                 rule='Fixed nearest-rank0.5 or0.75 of median-sign-oriented original learned values inside each INPUT-measured region>=43 frames; stable sorting and NumPy round-to-even rank. Exact-zero sign vote suppresses region. Select an actually observed frame value and its associated activity; retain dense values elsewhere and zero unvoiced. No interpolation, clipping or target routing.',
                 comparison='Two globally fixed readouts of four frozen out-of-fold Transformers; no new fit or calibrated-confidence claim. Actual held-singer/source evaluation and unchanged numerical/encoded-WAV checks.',
                 selection='Raw all-singer prediction gate, then all50 original WAV identities, then minimum singer-macro delivered error. Include original reference and prefer it on ties. No repair-count-only selection or development tuning.',
                 previous_center_result_sha256=ex.sha(previous.ROOT/'locked_selection.json'),
                 new_neural_fits=0,fresh_audio_access=False,demo_activated=False,research_validation_passed=False)
        ex.write(ROOT/'declaration.json',d)
        snapshot=ROOT/'executed_sources';snapshot.mkdir(exist_ok=True)
        for path in sources:(snapshot/Path(path).name).write_bytes(Path(path).read_bytes())
        for name in ('train_pairs.json','anchor_sources.json'):(ROOT/name).write_bytes((previous.ROOT/name).read_bytes())
    d=ex.read(ROOT/'declaration.json')
    for label in ('source_hashes','input_hashes','checkpoint_hashes','prediction_hashes','full_refit_hashes'):
        for path,digest in d[label].items():assert ex.sha(path)==digest,path
    for label in ('baseline_cv','baseline_evidence'):assert ex.sha(d[label])==d[label+'_sha256']
    assert ex.sha(previous.ROOT/'locked_selection.json')==d['previous_center_result_sha256']
    return d


def build(d,rows,pairs,mode):
    oldproject,oldpredict=previous.project,previous.predict_raw
    previous.project=partial(readout.project,mode=mode);previous.predict_raw=partial(readout.predict_raw,mode=mode)
    try:return previous.build(d,rows,pairs)
    finally:previous.project,previous.predict_raw=oldproject,oldpredict


def cv(d,rows,pairs):
    results={}
    for mode in MODES:
        cache,_=build(d,rows,pairs,mode);folder=ROOT/'validation'/mode;folder.mkdir(parents=True,exist_ok=True)
        if (folder/'accuracy.json').exists():raise ValueError('Existing rank-readout result retained')
        ex.write(folder/'evaluation_declaration.json',dict(pairs_manifest=str(ROOT/'train_pairs.json'),
                 role='Training-side singer/source-held-out fixed distribution readout; not fresh',mode=mode))
        old=(ex.predictions,ex.legacy_proposal)
        ex.predictions=lambda model,p,view:cache[p['path_'+view]]
        ex.legacy_proposal=lambda p,view,pp:curve.conditioned_proposal(p,view,pp,curve.POLICY,'median11')
        try:result=ex.evaluate(None,dict(seed=339),pairs,folder,render=False,seed=339)
        finally:ex.predictions,ex.legacy_proposal=old
        result.update(model_family=d['model_family'],decoder_version=readout.VERSION,readout=mode)
        ex.write(folder/'accuracy.json',result);ex.write_report(folder,result,'Fixed candidate rank readout');complete_report(folder)
        results[mode]=result
        print('RANK_READOUT_CV',mode,result['prediction_gate_passed'],result['summary'],flush=True)
    results['matched_baseline']=ex.read(d['baseline_cv'])['results']['matched_baseline']
    ex.write(ROOT/'cross_validation.json',dict(results=results,new_neural_fits=0,frozen_cv_checkpoints=4))


def delivery(d,rows,pairs):
    cvresults=ex.read(ROOT/'cross_validation.json')['results'];evidence={'matched_baseline':ex.read(d['baseline_evidence'])}
    for mode in MODES:
        cache,models=build(d,rows,pairs,mode);folder=ROOT/'arms'/mode;folder.mkdir(parents=True,exist_ok=True)
        if not (folder/'declaration.json').exists():ex.write(folder/'declaration.json',dict(parent=str(ROOT/'declaration.json'),parent_sha256=ex.sha(ROOT/'declaration.json'),readout=mode))
        oldroot,oldpredict=curve.ROOT,curve.predict_raw;curve.ROOT=folder;curve.predict_raw=partial(readout.predict_raw,mode=mode)
        try:evidence[mode]=curve.run_condition(rows,pairs,cache,models,ex.read(ROOT/'anchor_sources.json'),'median11','training',ROOT/'train_pairs.json')
        finally:curve.ROOT,curve.predict_raw=oldroot,oldpredict
    conditions=['matched_baseline']+list(MODES)
    selected=min(conditions,key=lambda mode:(not cvresults[mode]['prediction_gate_passed'],
                 not evidence[mode]['gates']['original_controls_all_identity'],
                 evidence[mode]['summary']['delivered_after_mae_cents'],conditions.index(mode)))
    ex.write(ROOT/'locked_selection.json',dict(arm=selected,results=evidence,
             prediction_gate_passed=cvresults[selected]['prediction_gate_passed'],
             all_originals_identity=evidence[selected]['gates']['original_controls_all_identity'],
             default_seed=339,diagnostic_only=True,demo_activated=False,research_validation_passed=False))
    print('RANK_READOUT_SELECTION',selected,flush=True)


def main():
    torch.set_num_threads(2)
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['cv','delivery']);args=parser.parse_args()
    d=declare();rows=ex.read(ROOT/'train_pairs.json');pairs=held.prepared(rows)
    if args.stage=='cv':cv(d,rows,pairs)
    else:delivery(d,rows,pairs)
    declare()


if __name__=='__main__':main()
