"""Centered but algebraically unchanged ALWA coarse-auditor training-side comparison."""
from pathlib import Path
from datetime import datetime,timezone
import torch
import numpy as np
import evaluate_precision_auditor as upstream
from bpdr_input.repair import centered_precision_audit as precision_audit
from bpdr_input.modeling import pesto_centered_precision
from verify_frozen_audio_studies import outputs

diagnostic=upstream.diagnostic
curve=upstream.curve
ex=diagnostic.upstream.study.ex
ROOT=upstream.ROOT.parent/'2026-11-01_v35_centered_auditor'


def declare():
    old=upstream.declare();ROOT.mkdir(exist_ok=True)
    if not (ROOT/'declaration.json').exists():
        stability=ROOT.parent/'2026-11-01_pcm24_stability'
        assert ex.read(stability/'integrity_receipt.json')['strict_sign_backend_agreement']
        assert ex.read(upstream.ROOT/'integrity_receipt.json')['selected_training_side']
        sources=dict(old['source_and_input_hashes'])
        paths=[Path(__file__).resolve(),Path(__file__).with_name('verify_centered_auditor.py').resolve(),Path(precision_audit.__file__),Path(pesto_centered_precision.__file__),curve.WORKSPACE/'tests/test_pesto_centered_precision.py',upstream.ROOT/'locked_selection.json',upstream.ROOT/'integrity_receipt.json',stability/'results.json',stability/'integrity_receipt.json']
        for p in paths:sources[str(p)]=ex.sha(p)
        baseline=upstream.ROOT/'arm/training/median11'
        ex.write(ROOT/'declaration.json',dict(created_at=datetime.now(timezone.utc).isoformat(),source_and_input_hashes=sources,parent_declaration_sha256=ex.sha(upstream.ROOT/'declaration.json'),manifest=old['manifest'],manifest_sha256=ex.sha(old['manifest']),baseline_folder=str(baseline),baseline_evidence_sha256=ex.sha(baseline/'evidence.json'),
            intervention='Same float32 frozen PESTO encoder/distributions, argmax, clipped five-bin ALWA window, weights, native grid and voiced masks. Algebraic recentering only: mean_bin=center+sum(weight*(index-center))/sum(weight); Hz=440*2**((center-207+relative)/36). Do not form a large absolute MIDI mean then subtract69. Both sum/division/exponent use float64. No epsilon, rounding-to-zero, discarded frames, peak-only decoding, changed neural weights, input predictions, scope/support, renderer, YIN or acceptance thresholds. Measure actual PCM24 WAV with the same centered recipe.',
            rationale='All11 development candidates/33 audio variants independently reconstructed. Opposing residuals1.35e-12..1.54e-12c and PCM24 flip motivate testing arithmetic conditioning. Centered formula passes90-digit exact binary-weight reference and local-peak-mass invariance tests. Known-development inspection motivated engineering; reused CV is not fresh generalization evidence.',
            control='Reference Transformer only; reuse authenticated V33 precision baseline27/200 and19.824704787234875c. Require all648 raw/requested/permitted arrays,216 permission decisions and50 original raw predictions/plans identical. No new training or per-case decoder routing.',
            selection='Require raw all-scoreable-singer gate,all50 originals unchanged and strictly lower singer-macro delivered remaining error than V33. Count alone cannot select; baseline wins ties. Remain inactive pending independent verification and separate prospective default339 known-development replay before any fresh confirmation.',
            fresh_confirmation_access=False,demo_activated=False,research_validation_passed=False))
    d=ex.read(ROOT/'declaration.json')
    for p,h in d['source_and_input_hashes'].items():assert ex.sha(p)==h,p
    assert d['parent_declaration_sha256']==ex.sha(upstream.ROOT/'declaration.json')
    assert ex.sha(d['manifest'])==d['manifest_sha256']
    assert ex.sha(Path(d['baseline_folder'])/'evidence.json')==d['baseline_evidence_sha256']
    return d


def main():
    torch.set_num_threads(2);d=declare();ex=diagnostic.upstream.study.ex
    if (ROOT/'locked_selection.json').exists():raise ValueError('Retain completed precision-auditor study')
    baseline=Path(d['baseline_folder']);auth=outputs(baseline,Path(d['manifest']),ex.read(diagnostic.upstream.study.ROOT/'anchor_sources.json'));assert auth['encoded_verified_wavs']==27
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
    declare();print('CENTERED_AUDITOR_SELECTION',chosen,'repairs',evidence['gates']['cohort_improved'],'after',evidence['summary']['delivered_after_mae_cents'],flush=True)

if __name__=='__main__':main()
