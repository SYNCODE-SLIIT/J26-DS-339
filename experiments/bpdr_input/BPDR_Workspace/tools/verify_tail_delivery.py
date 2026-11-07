"""Replay every real-hum transfer decision, encoded file and positive/known-zero score."""
from pathlib import Path
import sys
import numpy as np
import soundfile as sf
import torch

WS=Path(__file__).resolve().parents[1];sys.path.insert(0,str(WS/'tools'))
import evaluate_tail_delivery as run


def main():
    torch.set_num_threads(2);d=run.declare();study=run.study;result=study.read(run.ROOT/'results.json')
    _,rows,originals=study.declare();data=study.records(rows,originals);by_id={r['id']:r for r in data};sources={r['source_id']:r for r in originals}
    assert result['declaration_sha256']==run.sha(run.ROOT/'declaration.json')
    assert result['model_integrity_sha256']==run.sha(study.ROOT/'integrity_receipt.json')
    for p,h in d['source_hashes'].items():assert run.sha(p)==h,p
    recreated=run.ROOT/'verification_recreated';recreated.mkdir(exist_ok=True)
    cache={};fcache={};plans=0;candidates=0;encoded=0;scores=0;controls=0;arm_receipts=[]
    def measured(audio,times):
        key=run.waveform_hash(audio)
        if key not in cache:cache[key]=(run.coarse_features(audio,22050,WS/'checkpoints'),run.fine_hz(audio,22050,times),run.audit_module.cross_correlation_pitch(audio,22050,times))
        return cache[key]
    def fine(audio,times):
        key=run.waveform_hash(audio)
        if key not in fcache:fcache[key]=run.fine_pitch(audio,22050,times)
        return fcache[key]
    for arm in study.ARMS:
        if arm=='frozen_uniform':
            parent=run.ROOT.parent/'2026-11-04_v66_preservation_delivery'
            proof=study.read(parent/'integrity_receipt.json');assert proof['results_sha256']==run.sha(parent/'results.json')
            assert d['baseline_integrity_sha256']==run.sha(parent/'integrity_receipt.json')
            old_declaration=study.read(parent/'declaration.json')
            for key in ['policy','renderer','audit','data_pairs_sha256','data_originals_sha256']:assert d[key]==old_declaration[key]
            for path,h in old_declaration['source_hashes'].items():
                if path in d['source_hashes']:assert d['source_hashes'][path]==h
            old=study.read(parent/'results.json')['arms']['uniform'];assert result['arms'][arm]==old
            for record in old['records']:
                expected=parent/'uniform'/record['id']/'proposal_NOT_REFERENCE.npz';actual=run.ROOT/arm/record['id']/'proposal_NOT_REFERENCE.npz'
                assert run.sha(expected)==run.sha(actual)
                if record['verified_output']:
                    decoded,rate=sf.read(record['verified_output'],dtype='float32');assert rate==22050 and run.waveform_hash(decoded)==record['actual_output_waveform_sha256']
            with np.load(study.ROOT/'frozen_uniform_all_predictions.npz') as fresh,np.load(run.transfer.previous.ROOT/'uniform_all_predictions.npz') as previous:
                for key in fresh.files:np.testing.assert_array_equal(fresh[key],previous[key])
            arm_receipts.append(dict(arm=arm,inherited_verified_records=len(old['records']),new_renders_replayed=0,parent_integrity_sha256=run.sha(parent/'integrity_receipt.json')))
            continue
        with np.load(study.ROOT/(arm+'_all_predictions.npz')) as z:pred={k:z[k].copy() for k in z.files}
        report=result['arms'][arm];assert len(report['records'])==498 and set(r['id'] for r in report['records'])==set(by_id)
        reproduced=[]
        for saved in report['records']:
            record=by_id[saved['id']];audio,times,views,reliable=run.load_input(record,rows,originals)
            prop=run.policy.proposal(audio,times,views,np.stack([pred[record['id']]]*3));plans+=1
            assert study.norm(prop['plan'])==saved['plan'] and study.norm(prop['candidates'])==saved['candidates']
            with np.load(run.ROOT/arm/record['id']/'proposal_NOT_REFERENCE.npz') as z:
                for k in ['curve','protected','reliable','predictions']:np.testing.assert_array_equal(z[k],prop[k])
            curve=prop['curve'];plan=prop['plan']
            def auditor(before,after,sr,intervals):
                assert sr==22050
                bf,yb,cb=measured(before,times);af,ya,ca=measured(after,times)
                return run.audit_module.audit_short_curve(bf,af,yb,ya,cb,ca,reliable,intervals,curve)
            renderer=lambda a,sr,intervals:run.render_candidate(a,sr,times,curve,intervals,run.RENDER)[0]
            output,transaction=run.transact_fixture(audio,22050,plan,renderer,auditor);verified=None
            if 'post_render_audit' in transaction:candidates+=1
            if transaction['outcome']=='VERIFIED_FIXTURE_CANDIDATE':
                target=recreated/(arm+'_'+record['id']+'.wav');sf.write(target,output,22050,subtype='PCM_24')
                original_path=run.ROOT/arm/record['id']/'candidate.wav';assert run.sha(target)==run.sha(original_path)
                try:
                    transaction['encoded_audit']=run.audit_delivered_wav(target,audio,22050,plan['intervals_seconds'],auditor)
                    output,_=sf.read(target,dtype='float32');verified=str(original_path);encoded+=1
                except Exception as exc:
                    transaction.update(outcome='ROLLED_BACK_ENCODED',rollback=True,encoded_error=str(exc));output=audio.copy()
            assert study.norm(transaction)==saved['transaction']
            assert verified==saved['verified_output'] and run.waveform_hash(output)==saved['actual_output_waveform_sha256']
            unchanged=bool(np.array_equal(output,audio));assert unchanged==saved['waveform_unchanged']
            if record['kind']=='original':controls+=1
            else:
                row=rows[record['pair_index']];reference,sr=sf.read(sources[row['source_id']]['path_A'],dtype='float32');assert sr==22050
                with np.load(row['target_path']) as z:positive=z['plateau'];zero=z['known_zero']
                before=fine(audio,times)-fine(reference,times);after=fine(output,times)-fine(reference,times)
                bm=float(np.mean(abs(before[positive])));am=float(np.mean(abs(after[positive])))
                bz=float(np.mean(abs(before[zero])));az=float(np.mean(abs(after[zero])))
                assert (bm,am,bz,az)==(saved['before_mae_cents'],saved['after_mae_cents'],saved['before_known_zero_mae_cents'],saved['after_known_zero_mae_cents'])
                gain=bool(verified and am<bm-1e-6);collateral=bool(verified and az>bz+1e-6)
                assert gain==saved['target_improvement'] and collateral==saved['known_zero_harm']
                assert bool(gain and not collateral)==saved['verified_beneficial']
                assert bool(verified and (am>bm+1e-6 or collateral))==saved['verified_harm'];scores+=1
            reproduced.append(saved)
        summary=report['summary'];positive=[r for r in reproduced if r['kind']=='corrupted']
        assert summary['physical_attempts']==len(rows)==1152 and summary['scoreable']==len(positive)==445
        assert summary['construction_excluded']==707
        for k in ['verified_beneficial','target_improvements','known_zero_harms','verified_harms']:
            field={'target_improvements':'target_improvement','known_zero_harms':'known_zero_harm','verified_harms':'verified_harm'}.get(k,k)
            assert summary[k]==sum(r[field] for r in positive)
        assert summary['original_controls_unchanged']==all(r['waveform_unchanged'] for r in reproduced if r['kind']=='original')
        b=[];a=[]
        for group in summary['groups']:
            scored=[r for r in positive if r['group']==group['group']]
            assert group['scoreable']==len(scored) and group['attempted']==sum(r['group']==group['group'] for r in rows)
            if scored:
                before=float(np.mean([r['before_mae_cents'] for r in scored]));after=float(np.mean([r['after_mae_cents'] for r in scored]))
                assert (before,after,(before-after)/before)==(group['before_mae_cents'],group['after_mae_cents'],group['error_reduction_fraction'])
                b.append(before);a.append(after)
        assert summary['singer_macro_before_mae_cents']==float(np.mean(b)) and summary['singer_macro_after_mae_cents']==float(np.mean(a))
        arm_receipts.append(dict(arm=arm,verified_beneficial=summary['verified_beneficial'],verified_harms=summary['verified_harms'],original_controls_unchanged=summary['original_controls_unchanged']))
        print('HUMTRANS_DELIVERY_REPLAY_ARM',arm,arm_receipts[-1],flush=True)
    study.write(run.ROOT/'integrity_receipt.json',dict(baseline_verified_records_inherited=498,plans_replayed=plans,candidate_audits_replayed=candidates,
        encoded_audits_replayed=encoded,positive_and_zero_reference_records_reproduced=scores,original_controls_rechecked=controls,
        arms=arm_receipts,results_sha256=run.sha(run.ROOT/'results.json'),verifier_source_sha256=run.sha(__file__),production_eligible=False,research_validation_passed=False))
    print('HUMTRANS_DELIVERY_INDEPENDENT_REPLAY_PASS',plans,candidates,encoded,scores,controls,flush=True)


if __name__=='__main__':main()
