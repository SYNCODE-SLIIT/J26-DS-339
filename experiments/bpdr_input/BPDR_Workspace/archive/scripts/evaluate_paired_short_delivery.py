"""Locked three-tracker actual-output evaluation including known-zero collateral error."""
from pathlib import Path
from dataclasses import asdict,replace
from collections import Counter
import argparse,sys
import numpy as np
import soundfile as sf
import torch

WS=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(WS/'source'));sys.path.insert(0,str(WS/'tools'))
import train_paired_short_models as study
from probe_short_phrase_renderer import sha
from bpdr_input.repair import short_phrase_policy as policy,experimental_short_audit as audit_module
from bpdr_input.repair.framewise_renderer import RendererPolicy,render_candidate
from bpdr_input.repair.centered_precision_audit import coarse_features
from bpdr_input.repair.rule_correction import fine_hz,fine_pitch
from bpdr_input.repair.repair_controller import transact_fixture,waveform_hash
from bpdr_input.repair.learned_runtime import audit_delivered_wav
from evaluate_short_phrase_delivery import load_input

ROOT=WS/'results/transformer_usability/2026-11-02_v47_paired_delivery'
RENDER=replace(RendererPolicy(),version='paired-short-diagnostic-render-1',ramp_seconds=.02,min_support_seconds=.24)


def declare():
    ROOT.mkdir(exist_ok=True)
    paths=[Path(__file__),Path(policy.__file__),Path(audit_module.__file__)]
    d=study.norm(dict(version='paired-short-three-tracker-delivery-1',source_hashes={str(p.resolve()):sha(p) for p in paths},
        training_declaration_sha256=sha(study.ROOT/'declaration.json'),data_integrity_sha256=sha(study.DATA/'integrity_receipt.json'),
        arms=study.ARMS,policy=asdict(policy.ShortPolicy()),renderer=asdict(RENDER),audit=asdict(audit_module.POLICY),
        gate='Compare both arms on same data/permission/three-tracker audit. Every original control unchanged. Verified beneficial C requires encoded audit, lower fixed positive-plateau error AND no increase>1e-6c on predeclared known-zero frames. Any increase in either counts as reference harm.',
        original_attempts=21,valid_originals=19,physical_attempts=336,scoreable_pairs=115,
        known_zero_scope='Existing observed-reference frames only; not a natural intended-note truth claim.',
        input_only='Source raw acoustics and nominal predictions only. Target, reference A, source ID and known injected support used only after output to score.',
        uncertainty='Boundary-view disagreement, not calibrated confidence. Exact input-token equality is verified before nominal prediction reuse.',
        changed_protocol='Compared with V42,3 native common frames plus CC is a new calibrated hypothesis. Do not compare counts across those differing datasets as an identical-protocol model gain.',
        no_model_promotion=True,production_eligible=False,research_validation_passed=False))
    p=ROOT/'declaration.json'
    if p.exists():assert study.read(p)==d
    else:study.write(p,d)
    return d


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--declare-only',action='store_true');args=parser.parse_args()
    torch.set_num_threads(2);d=declare()
    if args.declare_only:print('PAIRED_SHORT_DELIVERY_LOCKED',flush=True);return
    receipt=study.read(study.ROOT/'integrity_receipt.json');assert receipt['checkpoints_reloaded']==8
    _,rows,originals=study.declare();data=study.records(rows,originals);original_by_id={s['source_id']:s for s in originals}
    cache={};fine_cache={};all_arms={}
    def measured(audio,times):
        key=waveform_hash(audio)
        if key not in cache:cache[key]=(coarse_features(audio,22050,WS/'checkpoints'),fine_hz(audio,22050,times),audit_module.cross_correlation_pitch(audio,22050,times))
        return cache[key]
    def fine(audio,times):
        key=waveform_hash(audio)
        if key not in fine_cache:fine_cache[key]=fine_pitch(audio,22050,times)
        return fine_cache[key]
    for arm in study.ARMS:
        with np.load(study.ROOT/(arm+'_all_predictions.npz')) as z:predictions={k:z[k].copy() for k in z.files}
        reports=[]
        for record in data:
            folder=ROOT/arm/record['id'];folder.mkdir(parents=True,exist_ok=True)
            audio,times,views,reliable=load_input(record,rows,originals)
            for raw in views[1:]:
                x,pad=study.model_module.tokens_from_features(study.model_module.frame_features(raw),np.arange(len(raw)))
                np.testing.assert_array_equal(x,record['short_tokens']);np.testing.assert_array_equal(pad,record['short_pad'])
            proposal=policy.proposal(audio,times,views,np.stack([predictions[record['id']]]*3));curve=proposal['curve'];plan=proposal['plan']
            np.savez_compressed(folder/'proposal_NOT_REFERENCE.npz',curve=curve,protected=proposal['protected'],reliable=proposal['reliable'],predictions=proposal['predictions'])
            def auditor(before,after,sr,intervals):
                assert sr==22050
                bf,yb,cb=measured(before,times);af,ya,ca=measured(after,times)
                return audit_module.audit_short_curve(bf,af,yb,ya,cb,ca,reliable,intervals,curve)
            renderer=lambda a,sr,intervals:render_candidate(a,sr,times,curve,intervals,RENDER)[0]
            output,transaction=transact_fixture(audio,22050,plan,renderer,auditor);verified=None
            if transaction['outcome']=='VERIFIED_FIXTURE_CANDIDATE':
                path=folder/'candidate.wav';sf.write(path,output,22050,subtype='PCM_24')
                try:
                    transaction['encoded_audit']=audit_delivered_wav(path,audio,22050,plan['intervals_seconds'],auditor)
                    output,_=sf.read(path,dtype='float32');verified=str(path)
                except Exception as exc:
                    transaction.update(outcome='ROLLED_BACK_ENCODED',rollback=True,encoded_error=str(exc));output=audio.copy()
            r=dict(id=record['id'],source_id=record['source_id'],group=record['group'],kind=record['kind'],
                plan=plan,candidates=proposal['candidates'],transaction=transaction,verified_output=verified,
                actual_output_waveform_sha256=waveform_hash(output),waveform_unchanged=bool(np.array_equal(audio,output)))
            if record['kind']=='corrupted':
                row=rows[record['pair_index']];original=original_by_id[row['source_id']]
                reference,sr=sf.read(original['path_A'],dtype='float32');assert sr==22050
                with np.load(row['target_path']) as z:positive=z['plateau'];zero=z['known_zero']
                before=fine(audio,times)-fine(reference,times);after=fine(output,times)-fine(reference,times)
                bm=float(np.mean(abs(before[positive])));am=float(np.mean(abs(after[positive])))
                bz=float(np.mean(abs(before[zero])));az=float(np.mean(abs(after[zero])))
                gain=bool(verified and am<bm-1e-6);collateral=bool(verified and az>bz+1e-6)
                harm=bool(verified and (am>bm+1e-6 or collateral))
                r.update(pair_index=record['pair_index'],width_seconds=row['width_seconds'],cents=row['cents'],
                    before_mae_cents=bm,after_mae_cents=am,before_known_zero_mae_cents=bz,after_known_zero_mae_cents=az,
                    target_improvement=gain,known_zero_harm=collateral,verified_beneficial=bool(gain and not collateral),verified_harm=harm)
            study.write(folder/'result.json',r);reports.append(r)
        groups=[]
        for group in sorted({r['group'] for r in rows}):
            scored=[r for r in reports if r['group']==group and r['kind']=='corrupted']
            controls=[r for r in reports if r['group']==group and r['kind']=='original']
            before=float(np.mean([r['before_mae_cents'] for r in scored])) if scored else None
            after=float(np.mean([r['after_mae_cents'] for r in scored])) if scored else None
            groups.append(dict(group=group,attempted=sum(r['group']==group for r in rows),scoreable=len(scored),
                verified_beneficial=sum(r['verified_beneficial'] for r in scored),verified_harms=sum(r['verified_harm'] for r in scored),
                target_improvements=sum(r['target_improvement'] for r in scored),known_zero_harms=sum(r['known_zero_harm'] for r in scored),
                before_mae_cents=before,after_mae_cents=after,error_reduction_fraction=(before-after)/before if before else None,
                original_controls=len(controls),original_controls_unchanged=all(r['waveform_unchanged'] for r in controls)))
        scored=[g for g in groups if g['scoreable']]
        summary=dict(groups=groups,physical_attempts=336,scoreable=115,construction_excluded=221,
            original_attempts=21,original_controls=19,original_controls_unchanged=all(r['waveform_unchanged'] for r in reports if r['kind']=='original'),
            verified_beneficial=sum(g['verified_beneficial'] for g in groups),verified_harms=sum(g['verified_harms'] for g in groups),
            target_improvements=sum(g['target_improvements'] for g in groups),known_zero_harms=sum(g['known_zero_harms'] for g in groups),
            singer_macro_before_mae_cents=float(np.mean([g['before_mae_cents'] for g in scored])),
            singer_macro_after_mae_cents=float(np.mean([g['after_mae_cents'] for g in scored])),
            outcomes=dict(Counter(r['transaction']['outcome'] for r in reports)))
        all_arms[arm]=dict(summary=summary,records=reports);study.write(ROOT/(arm+'_results.json'),all_arms[arm])
        print('PAIRED_DELIVERY_ARM_COMPLETE',arm,summary['verified_beneficial'],summary['verified_harms'],summary['singer_macro_after_mae_cents'],summary['original_controls_unchanged'],flush=True)
    study.write(ROOT/'results.json',dict(declaration_sha256=sha(ROOT/'declaration.json'),model_integrity_sha256=sha(study.ROOT/'integrity_receipt.json'),
        arms=all_arms,production_eligible=False,research_validation_passed=False))


if __name__=='__main__':main()
