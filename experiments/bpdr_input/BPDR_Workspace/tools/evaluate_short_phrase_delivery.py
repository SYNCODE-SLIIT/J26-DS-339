"""Prospectively fixed input-only permission and actual-WAV comparison, no activation."""
from pathlib import Path
from dataclasses import asdict
from collections import Counter
import argparse,json,sys
import numpy as np
import soundfile as sf
import torch

WS=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(WS/'source'));sys.path.insert(0,str(WS/'tools'))
import compare_short_phrase_models as study
from probe_short_phrase_renderer import sha
from bpdr_input.repair import short_phrase_policy as policy
from bpdr_input.repair.framewise_renderer import RendererPolicy,render_candidate
from bpdr_input.repair.interventions import AuditPolicy
from bpdr_input.repair.framewise_audit import audit_curve
from bpdr_input.repair.centered_precision_audit import coarse_features
from bpdr_input.repair.rule_correction import fine_hz,fine_pitch
from bpdr_input.repair.repair_controller import transact_fixture,waveform_hash
from bpdr_input.repair.learned_runtime import audit_delivered_wav

ROOT=WS/'results/transformer_usability/2026-11-02_v42_short_delivery'
RENDER=RendererPolicy(version='short-delivery-diagnostic-render-1',ramp_seconds=.02,min_support_seconds=.24)
AUDIT=AuditPolicy(version='short-delivery-diagnostic-audit-1',ramp_seconds=.02,analysis_guard_seconds=.07)
ARMS=['frozen_joint',*study.ARMS]


def declare():
    ROOT.mkdir(parents=True,exist_ok=True)
    model_declaration=study.ROOT/'declaration.json';assert model_declaration.is_file()
    paths=[Path(__file__),Path(policy.__file__),WS/'tests/test_short_phrase_policy.py']
    d=study.norm(dict(version='short-phrase-delivery-comparison-1',
        source_hashes={str(p.resolve()):sha(p) for p in paths},
        model_declaration_sha256=sha(model_declaration),data_integrity_sha256=sha(study.DATA/'integrity_receipt.json'),
        arms=ARMS,policy=asdict(policy.ShortPolicy()),renderer=asdict(RENDER),auditor=asdict(AUDIT),
        proposal='Input-only signed hysteresis12/6c,median3,>=.24s real support; veto full candidate touching unreliable/energy/pitch-transition guards or user protection. No gap filling/padding. Controller max30c/max25% duration,disagreement/uncertainty5c unchanged.',
        boundary_uncertainty='Each of three raw boundary views must produce identical token matrices; reuse nominal deterministic prediction only after exact input comparison. Zero disagreement is invariance by construction, not calibrated confidence or intended-pitch evidence.',
        scoring='Only after planning/rendering/auditing: same immutable A-reference and target plateau; mean absolute actual fine-pitch error before/after. Verified improvement requires encoded audit plus >1e-6c error reduction. No oracle interval or target in proposal.',
        controls='Every valid original must remain sample-exact identity; invalid source original attempts retained as preprocessing exclusions. Every336 physical attempt retained per arm; labels absent ->unassessed, not zero model accuracy.',
        no_development_or_final_audio_access=True,research_validation_passed=False,production_eligible=False,
        selection='No automatic promotion. Compare all arms on identical data/policy/audits and retain raw-gate failures. Usability requires separate source/duration coverage, preservation and later independent confirmation/review.'))
    p=ROOT/'declaration.json'
    if p.exists():assert study.read(p)==d
    else:study.write(p,d)
    return d


def load_input(record,rows,originals):
    if record['kind']=='original':
        s=next(s for s in originals if s['source_id']==record['source_id'])
        audio_path,analysis_path=s['path_A'],s['analysis_A']
    else:
        r=rows[record['pair_index']];audio_path,analysis_path=r['path_C'],r['analysis_C']
    audio,sr=sf.read(audio_path,dtype='float32');assert sr==22050
    with np.load(analysis_path) as z:views=z['raw_views'].copy();times=z['times_seconds'].copy();reliable=z['reliable'].copy()
    return audio,times,views,reliable


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--declare-only',action='store_true');args=parser.parse_args()
    torch.set_num_threads(2);d=declare()
    if args.declare_only:print('SHORT_DELIVERY_PROTOCOL_LOCKED',flush=True);return
    mreceipt=study.read(study.ROOT/'integrity_receipt.json');assert mreceipt['checkpoints_reloaded']==12
    _,rows,originals=study.declare();records=study.records(rows,originals)
    cache={};by_arm={};original_by_id={s['source_id']:s for s in originals}
    def measured(audio,times):
        key=waveform_hash(audio)
        if key not in cache:cache[key]=(coarse_features(audio,22050,WS/'checkpoints'),fine_hz(audio,22050,times))
        return cache[key]
    for arm in ARMS:
        with np.load(study.ROOT/(arm+'_all_predictions.npz')) as z:pred={k:z[k].copy() for k in z.files}
        reports=[]
        for record in records:
            folder=ROOT/arm/record['id'];folder.mkdir(parents=True,exist_ok=True)
            audio,times,views,reliable=load_input(record,rows,originals)
            # Boundary variants are metadata perturbations, not new acoustic evidence.
            prefix='joint' if arm in ['frozen_joint','joint_retrained'] else 'short'
            for raw in views[1:]:
                if prefix=='joint':x,pad=study.old.joint_tokens(raw,np.arange(len(raw)))
                else:x,pad=study.short.tokens_from_features(study.short.frame_features(raw),np.arange(len(raw)))
                np.testing.assert_array_equal(x,record[prefix+'_tokens']);np.testing.assert_array_equal(pad,record[prefix+'_pad'])
            proposal=policy.proposal(audio,times,views,np.stack([pred[record['id']]]*3))
            curve=proposal['curve'];plan=proposal['plan']
            np.savez_compressed(folder/'proposal_NOT_REFERENCE.npz',curve=curve,protected=proposal['protected'],
                                reliable=proposal['reliable'],predictions=proposal['predictions'])
            def auditor(before,after,sr,intervals):
                assert sr==22050
                bf,yb=measured(before,times);af,ya=measured(after,times)
                return audit_curve(bf,af,yb,ya,reliable,intervals,curve,AUDIT)
            renderer=lambda a,sr,intervals:render_candidate(a,sr,times,curve,intervals,RENDER)[0]
            output,transaction=transact_fixture(audio,22050,plan,renderer,auditor)
            verified_path=None
            if transaction['outcome']=='VERIFIED_FIXTURE_CANDIDATE':
                candidate_path=folder/'candidate.wav';sf.write(candidate_path,output,22050,subtype='PCM_24')
                try:
                    transaction['encoded_audit']=audit_delivered_wav(candidate_path,audio,22050,plan['intervals_seconds'],auditor)
                    output,_=sf.read(candidate_path,dtype='float32');verified_path=str(candidate_path)
                except Exception as exc:
                    transaction.update(outcome='ROLLED_BACK_ENCODED',rollback=True,encoded_error=str(exc));output=audio.copy()
            report=dict(id=record['id'],group=record['group'],kind=record['kind'],source_id=record['source_id'],
                plan=plan,transaction=transaction,candidates=proposal['candidates'],
                waveform_unchanged=bool(np.array_equal(audio,output)),verified_output=verified_path,
                actual_output_waveform_sha256=waveform_hash(output))
            if record['kind']=='corrupted':
                r=rows[record['pair_index']];source=original_by_id[r['source_id']]
                with np.load(source['analysis_A']) as z:reference=z['fine'].copy()
                with np.load(r['analysis_C']) as z:before=z['fine'].copy()
                mask=record['mask']&(record['target']!=0);after=fine_pitch(output,22050,times)
                assert mask.any() and np.isfinite(after[mask]).all()
                before_error=float(np.mean(abs(before[mask]-reference[mask])))
                after_error=float(np.mean(abs(after[mask]-reference[mask])))
                report.update(pair_index=record['pair_index'],width_seconds=r['width_seconds'],cents=r['cents'],
                    before_mae_cents=before_error,after_mae_cents=after_error,
                    verified_improvement=bool(verified_path and after_error<before_error-1e-6),
                    verified_harm=bool(verified_path and after_error>before_error+1e-6))
            study.write(folder/'result.json',report);reports.append(report)
        groups=[]
        for group in sorted({r['group'] for r in rows}):
            attempted=[r for r in rows if r['group']==group]
            scored=[r for r in reports if r['group']==group and r['kind']=='corrupted']
            controls=[r for r in reports if r['group']==group and r['kind']=='original']
            before=float(np.mean([r['before_mae_cents'] for r in scored])) if scored else None
            after=float(np.mean([r['after_mae_cents'] for r in scored])) if scored else None
            groups.append(dict(group=group,attempted=len(attempted),scoreable=len(scored),
                verified_improvements=sum(r['verified_improvement'] for r in scored),
                verified_harms=sum(r['verified_harm'] for r in scored),
                before_mae_cents=before,after_mae_cents=after,
                error_reduction_fraction=(before-after)/before if before else None,
                original_controls=len(controls),originals_unchanged=all(r['waveform_unchanged'] for r in controls)))
        scored=[g for g in groups if g['scoreable']]
        summary=dict(groups=groups,attempted=336,scoreable=44,construction_excluded=292,
            original_attempts=21,original_controls=19,original_preprocessing_excluded=2,
            original_controls_all_identity=all(r['waveform_unchanged'] for r in reports if r['kind']=='original'),
            verified_improvements=sum(r.get('verified_improvement',False) for r in reports),
            verified_harms=sum(r.get('verified_harm',False) for r in reports),
            singer_macro_before_mae_cents=float(np.mean([g['before_mae_cents'] for g in scored])),
            singer_macro_after_mae_cents=float(np.mean([g['after_mae_cents'] for g in scored])),
            outcomes=dict(Counter(r['transaction']['outcome'] for r in reports)))
        by_arm[arm]=dict(summary=summary,records=reports)
        study.write(ROOT/(arm+'_results.json'),by_arm[arm])
        print('SHORT_DELIVERY_ARM',arm,summary,flush=True)
    study.write(ROOT/'results.json',dict(declaration_sha256=sha(ROOT/'declaration.json'),
        model_integrity_sha256=sha(study.ROOT/'integrity_receipt.json'),arms=by_arm,
        production_eligible=False,research_validation_passed=False))
    print('SHORT_DELIVERY_COMPLETE',flush=True)


if __name__=='__main__':main()
