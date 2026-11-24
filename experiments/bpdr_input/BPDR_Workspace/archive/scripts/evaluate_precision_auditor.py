"""Frozen reference Transformer plus precision-only coarse-auditor WAV comparison."""
from pathlib import Path
from datetime import datetime,timezone
import torch
import numpy as np
import diagnose_pesto_precision as diagnostic
import run_transformer_curve_study as curve
from bpdr_input.repair import precision_audit
from verify_frozen_audio_studies import outputs

ROOT=diagnostic.ROOT.parent/'2026-11-01_v33_precision_auditor'


def declare():
    ex=diagnostic.upstream.study.ex;old,parent=diagnostic.declare();ROOT.mkdir(exist_ok=True)
    if not (ROOT/'declaration.json').exists():
        check=ex.read(diagnostic.ROOT/'integrity_receipt.json');assert check['strict_sign_implementation_agreement']
        sources=dict(old['source_and_input_hashes'])
        for p in (Path(__file__).resolve(),Path(precision_audit.__file__),Path(__file__).with_name('verify_precision_auditor.py').resolve(),diagnostic.ROOT/'results.json',diagnostic.ROOT/'integrity_receipt.json'):
            sources[str(p)]=ex.sha(p)
        ex.write(ROOT/'declaration.json',dict(created_at=datetime.now(timezone.utc).isoformat(),source_and_input_hashes=sources,parent_declaration_sha256=ex.sha(diagnostic.ROOT/'declaration.json'),manifest=parent['manifest'],manifest_sha256=ex.sha(parent['manifest']),baseline_folder=parent['arms']['reference'],baseline_evidence_sha256=ex.sha(Path(parent['arms']['reference'])/'evidence.json'),
            intervention='Frozen reference Transformer only. Change coarse audit decoding arithmetic to float64 on the same float32 encoder/distributions and identical ALWA window/weights. Keep all model input features/predictions/scope/permission/supports/uncertainty,50 original controls,median11/Praat rendering and every fine/coarse numerical limit including strict per-frame direction. Recheck encoded PCM24 WAV using the same precision auditor. No new model fit or case-wise decoder routing.',
            control='Reuse authenticated reference23/200 WAV evidence after independent reconstruction of all30 original candidate waveforms/audits and exact old native F0/confidence reproduction on all90 diagnostic tracks. All216 requested curves/permission decisions must match reference. No permission/safety threshold changes.',
            selection='Require unchanged reference raw all-singer gate,all50 original WAV identity,then strictly lower singer-macro delivered remaining error than20.3421417351c. No count-only selection. A winner remains inactive pending full-fit lineage,known development and untouched confirmation; natural-intention review and reference-coverage limits remain unresolved.',
            demo_activated=False,research_validation_passed=False))
    d=ex.read(ROOT/'declaration.json')
    for p,h in d['source_and_input_hashes'].items():assert ex.sha(p)==h,p
    assert ex.sha(d['manifest'])==d['manifest_sha256']
    assert ex.sha(Path(d['baseline_folder'])/'evidence.json')==d['baseline_evidence_sha256']
    return d


def main():
    torch.set_num_threads(2);d=declare();ex=diagnostic.upstream.study.ex
    if (ROOT/'locked_selection.json').exists():raise ValueError('Retain completed precision-auditor study')
    baseline=Path(d['baseline_folder']);auth=outputs(baseline,Path(d['manifest']),ex.read(diagnostic.upstream.study.ROOT/'anchor_sources.json'));assert auth['encoded_verified_wavs']==23
    rows=ex.read(d['manifest']);pairs=diagnostic.upstream.study.training.held.prepared(rows)
    predictions,models=curve.training_cache(rows)
    with np.load(baseline/'predictions.npz') as z:
        for i,row in enumerate(rows):
            if 'excluded' in row:continue
            for view in ('A','C'):np.testing.assert_array_equal(predictions[row['path_'+view]],z[f'{i}_{view}'])
    folder=ROOT/'arm';folder.mkdir(exist_ok=True)
    if not (folder/'declaration.json').exists():ex.write(folder/'declaration.json',dict(parent=str(ROOT/'declaration.json'),parent_sha256=ex.sha(ROOT/'declaration.json')))
    oldroot=curve.ROOT;oldrender=ex.rendered_output;cache={}
    ex.rendered_output=lambda p,v,c,plan,out:precision_audit.rendered_output(p,v,c,plan,out,ex.COMPONENT_ROOT/'checkpoints',cache)
    curve.ROOT=folder
    try:evidence=curve.run_condition(rows,pairs,predictions,models,ex.read(diagnostic.upstream.study.ROOT/'anchor_sources.json'),'median11','training',Path(d['manifest']))
    finally:curve.ROOT=oldroot;ex.rendered_output=oldrender
    original=ex.read(baseline/'evidence.json');raw=ex.read(baseline/'accuracy.json')['prediction_gate_passed']
    chosen=bool(raw and evidence['gates']['original_controls_all_identity'] and evidence['summary']['delivered_after_mae_cents']<original['summary']['delivered_after_mae_cents'])
    ex.write(ROOT/'locked_selection.json',dict(pipeline='precision_auditor' if chosen else 'reference',selected_training_side=chosen,default_seed=339,raw_gate_passed=raw,originals_identity=evidence['gates']['original_controls_all_identity'],demo_activated=False,research_validation_passed=False))
    declare();print('PRECISION_AUDITOR_SELECTION',chosen,'repairs',evidence['gates']['cohort_improved'],'after',evidence['summary']['delivered_after_mae_cents'],flush=True)

if __name__=='__main__':main()
