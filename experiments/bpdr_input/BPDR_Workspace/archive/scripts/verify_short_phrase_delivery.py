"""Independent proposal, transaction, encoded-WAV and reference-score replay."""
from pathlib import Path
import sys,json
import numpy as np
import soundfile as sf
import torch

WS=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(WS/'source'));sys.path.insert(0,str(WS/'tools'))
import evaluate_short_phrase_delivery as run
from probe_short_phrase_renderer import sha


def main():
    torch.set_num_threads(2);d=run.declare();study=run.study
    result=study.read(run.ROOT/'results.json');_,rows,originals=study.declare()
    records=study.records(rows,originals);by_id={r['id']:r for r in records}
    source_by_id={s['source_id']:s for s in originals};cache={};fine_cache={}
    assert result['model_integrity_sha256']==sha(study.ROOT/'integrity_receipt.json')
    assert result['declaration_sha256']==sha(run.ROOT/'declaration.json')
    recreated=run.ROOT/'verification_recreated';recreated.mkdir(exist_ok=True)
    plans=0;encoded=0;rendered=0;reference_scores=0;original_checks=0;arm_receipts=[]
    def measure(audio,times):
        key=run.waveform_hash(audio)
        if key not in cache:cache[key]=(run.coarse_features(audio,22050,WS/'checkpoints'),run.fine_hz(audio,22050,times))
        return cache[key]
    def fine(audio,times):
        key=run.waveform_hash(audio)
        if key not in fine_cache:fine_cache[key]=run.fine_pitch(audio,22050,times)
        return fine_cache[key]
    for arm in run.ARMS:
        with np.load(study.ROOT/(arm+'_all_predictions.npz')) as z:pred={k:z[k].copy() for k in z.files}
        report=result['arms'][arm];assert len(report['records'])==63
        assert set(r['id'] for r in report['records'])==set(by_id)
        gains=0;harms=0;identities=0;verified_paths=[]
        for saved in report['records']:
            record=by_id[saved['id']];audio,times,views,reliable=run.load_input(record,rows,originals)
            prop=run.policy.proposal(audio,times,views,np.stack([pred[record['id']]]*3));plans+=1
            assert study.norm(prop['plan'])==saved['plan']
            assert study.norm(prop['candidates'])==saved['candidates']
            with np.load(run.ROOT/arm/record['id']/'proposal_NOT_REFERENCE.npz') as z:
                for k in ['curve','protected','reliable','predictions']:np.testing.assert_array_equal(z[k],prop[k])
            curve=prop['curve'];plan=prop['plan']
            def auditor(before,after,sr,intervals):
                assert sr==22050
                bf,yb=measure(before,times);af,ya=measure(after,times)
                return run.audit_curve(bf,af,yb,ya,reliable,intervals,curve,run.AUDIT)
            renderer=lambda a,sr,intervals:run.render_candidate(a,sr,times,curve,intervals,run.RENDER)[0]
            output,transaction=run.transact_fixture(audio,22050,plan,renderer,auditor)
            path=None
            if 'post_render_audit' in transaction:rendered+=1
            if transaction['outcome']=='VERIFIED_FIXTURE_CANDIDATE':
                target=recreated/(arm+'_'+record['id']+'.wav')
                sf.write(target,output,22050,subtype='PCM_24')
                original_candidate=run.ROOT/arm/record['id']/'candidate.wav'
                assert sha(target)==sha(original_candidate)
                try:
                    transaction['encoded_audit']=run.audit_delivered_wav(target,audio,22050,plan['intervals_seconds'],auditor)
                    output,_=sf.read(target,dtype='float32');path=str(original_candidate)
                    encoded+=1;verified_paths.append(dict(path=path,sha256=sha(target)))
                except Exception as exc:
                    transaction.update(outcome='ROLLED_BACK_ENCODED',rollback=True,encoded_error=str(exc));output=audio.copy()
            assert study.norm(transaction)==saved['transaction'],(arm,record['id'])
            assert path==saved['verified_output']
            assert run.waveform_hash(output)==saved['actual_output_waveform_sha256']
            unchanged=bool(np.array_equal(audio,output));assert unchanged==saved['waveform_unchanged']
            if record['kind']=='original':
                original_checks+=1;identities+=unchanged
            else:
                row=rows[record['pair_index']];source=source_by_id[row['source_id']]
                reference,sr=sf.read(source['path_A'],dtype='float32');assert sr==22050
                q=record['mask']&(record['target']!=0)
                before=float(np.mean(abs(fine(audio,times)[q]-fine(reference,times)[q])))
                after=float(np.mean(abs(fine(output,times)[q]-fine(reference,times)[q])))
                assert before==saved['before_mae_cents'] and after==saved['after_mae_cents']
                gain=bool(path and after<before-1e-6);harm=bool(path and after>before+1e-6)
                assert gain==saved['verified_improvement'] and harm==saved['verified_harm']
                gains+=gain;harms+=harm;reference_scores+=1
        summary=report['summary']
        assert summary['verified_improvements']==gains and summary['verified_harms']==harms
        assert summary['original_controls_all_identity']==(identities==19)
        assert summary['attempted']==len(rows)==336 and summary['scoreable']==44
        assert summary['construction_excluded']==sum('excluded' in r for r in rows)==292
        before=[];after=[]
        for g in summary['groups']:
            cases=[r for r in report['records'] if r['group']==g['group'] and r['kind']=='corrupted']
            assert g['scoreable']==len(cases)
            assert g['attempted']==sum(r['group']==g['group'] for r in rows)
            if cases:
                b=float(np.mean([r['before_mae_cents'] for r in cases]));a=float(np.mean([r['after_mae_cents'] for r in cases]))
                assert g['before_mae_cents']==b and g['after_mae_cents']==a
                assert g['error_reduction_fraction']==(b-a)/b
                before.append(b);after.append(a)
        assert summary['singer_macro_before_mae_cents']==float(np.mean(before))
        assert summary['singer_macro_after_mae_cents']==float(np.mean(after))
        arm_receipts.append(dict(arm=arm,verified_outputs=verified_paths,gains=gains,harms=harms,unchanged_originals=identities))
        print('SHORT_DELIVERY_REPLAY_ARM',arm,gains,harms,identities,flush=True)
    study.write(run.ROOT/'integrity_receipt.json',dict(plans_replayed=plans,
        candidate_audits_replayed=rendered,encoded_audits_replayed=encoded,
        reference_scores_reproduced=reference_scores,original_controls_rechecked=original_checks,
        arms=arm_receipts,results_sha256=sha(run.ROOT/'results.json'),verifier_source_sha256=sha(__file__),
        source_and_input_declarations_intact=True,production_eligible=False,research_validation_passed=False))
    print('SHORT_DELIVERY_INDEPENDENT_REPLAY_PASS',plans,rendered,encoded,reference_scores,original_checks,flush=True)


if __name__=='__main__':main()
