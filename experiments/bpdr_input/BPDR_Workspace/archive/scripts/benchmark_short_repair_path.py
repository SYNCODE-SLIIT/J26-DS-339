"""Time four fixed known V52 cases through the actual experimental repair path.

Post-hoc diagnostic representatives, not fresh accuracy validation or a population
latency benchmark. Input WAVs are already22.05kHz mono. CV fold routing is offline.
"""
from pathlib import Path
import time
import numpy as np
import soundfile as sf
import torch

import evaluate_humtrans_short_delivery as run
from bpdr_input.audio import pipeline
from bpdr_input.experiments.learning import model_recovery as recovery

ROOT=run.WS/'results/transformer_usability/2026-11-04_v62_path_latency'
TRACE=ROOT.parent/'2026-11-04_v61_pipeline_trace'


def declare():
    transfer=run.transfer;d,rows,originals=transfer.declare()
    trace=transfer.read(TRACE/'results.json')
    examples=trace['arms']['added_real_hums']['representative_examples']
    selected={name:examples[name]['id'] for name in ('beneficial','preservation_harm','changed_original','rollback')}
    result=transfer.read(run.ROOT/'results.json');receipt=transfer.read(run.ROOT/'integrity_receipt.json')
    assert receipt['results_sha256']==transfer.sha(run.ROOT/'results.json')
    records={r['id']:r for r in result['arms']['added_real_hums']['records']}
    by_source={s['source_id']:s for s in originals}
    inputs={}
    for name,key in selected.items():
        r=records[key]
        path=rows[r['pair_index']]['path_C'] if r['kind']=='corrupted' else by_source[r['source_id']]['path_A']
        fold=next(i for i,groups in enumerate(d['folds'],1) if r['group'] in groups)
        checkpoint=transfer.ROOT/f'fold_{fold}/checkpoint.pt'
        inputs[name]=dict(id=key,path=path,input_sha256=transfer.sha(path),checkpoint=str(checkpoint),checkpoint_sha256=transfer.sha(checkpoint))
    ROOT.mkdir(parents=True,exist_ok=True)
    declaration=transfer.norm(dict(version='fixed-representative-short-path-latency-1',source_sha256=transfer.sha(__file__),
        trace_sha256=transfer.sha(TRACE/'results.json'),delivery_integrity_sha256=transfer.sha(run.ROOT/'integrity_receipt.json'),
        sources={str(p):transfer.sha(p) for p in [Path(pipeline.__file__),Path(recovery.__file__),Path(run.__file__)]},
        selected=inputs,thread_count=2,repeat_count=1,
        scope='Four known post-hoc diagnostic cases,22.05kHz mono input; Python imports excluded,first-request lazy estimator/model initialization included where incurred. Not a live UI/deployment or unfamiliar-source trial.',
        timing='Technical pipeline plus fresh acoustic context,model,proposal,render/rollback,PCM encoding and independent output audit. Stage timings exclude reference scoring; no reference/support enters decisions.',
        baseline='Outcome and actual delivered PCM must reproduce V52 for each selected case',
        new_model_fits=0,no_model_promotion=True,production_eligible=False,research_validation_passed=False))
    p=ROOT/'declaration.json'
    if p.exists():assert transfer.read(p)==declaration
    else:transfer.write(p,declaration)
    return declaration,records


def main():
    torch.set_num_threads(2);d,records=declare();transfer=run.transfer;models={};reports=[]
    for category,selected in d['selected'].items():
        folder=ROOT/category;folder.mkdir(exist_ok=True)
        started=time.perf_counter();durations={}
        t=time.perf_counter()
        technical,directory=pipeline.process(selected['path'],folder/'technical',run.WS/'checkpoints')
        durations['input_integrity_and_frontend_seconds']=time.perf_counter()-t
        assert technical.accepted_audio is not None
        assert technical.accepted_audio.checksum==selected['input_sha256']
        t=time.perf_counter();audio,sr=sf.read(technical.accepted_audio.path,dtype='float32');assert sr==22050
        analysis=recovery.measured(audio);times=analysis['features']['times_seconds'];views=analysis['views'];reliable=analysis['reliable']
        durations['repair_feature_context_seconds']=time.perf_counter()-t
        t=time.perf_counter();path=selected['checkpoint']
        if path not in models:
            assert transfer.sha(path)==selected['checkpoint_sha256']
            models[path]=transfer.paired.model_module.load(path)[0]
        durations['checkpoint_load_seconds']=time.perf_counter()-t
        t=time.perf_counter();prediction,_=transfer.paired.model_module.predict_raw(models[path],views[0])
        durations['model_prediction_seconds']=time.perf_counter()-t
        t=time.perf_counter();proposal=run.policy.proposal(audio,times,views,np.stack([prediction]*3))
        durations['candidate_and_permission_seconds']=time.perf_counter()-t
        curve=proposal['curve'];plan=proposal['plan'];cache={};components=dict(render_seconds=0.,pitch_audit_seconds=0.)
        def measured(wave):
            key=run.waveform_hash(wave)
            if key not in cache:
                cache[key]=(run.coarse_features(wave,sr,run.WS/'checkpoints'),run.fine_hz(wave,sr,times),run.audit_module.cross_correlation_pitch(wave,sr,times))
            return cache[key]
        def auditor(before,after,rate,intervals):
            t0=time.perf_counter();assert rate==sr
            bf,yb,cb=measured(before);af,ya,ca=measured(after)
            result=run.audit_module.audit_short_curve(bf,af,yb,ya,cb,ca,reliable,intervals,curve)
            components['pitch_audit_seconds']+=time.perf_counter()-t0
            return result
        def renderer(before,rate,intervals):
            t0=time.perf_counter();output=run.render_candidate(before,rate,times,curve,intervals,run.RENDER)[0]
            components['render_seconds']+=time.perf_counter()-t0;return output
        t=time.perf_counter();output,transaction=run.transact_fixture(audio,sr,plan,renderer,auditor)
        durations['render_and_internal_audit_transaction_seconds']=time.perf_counter()-t
        t=time.perf_counter();verified=None
        if transaction['outcome']=='VERIFIED_FIXTURE_CANDIDATE':
            verified=folder/'candidate.wav';sf.write(verified,output,sr,subtype='PCM_24')
            transaction['encoded_audit']=run.audit_delivered_wav(verified,audio,sr,plan['intervals_seconds'],auditor)
            output,rate=sf.read(verified,dtype='float32');assert rate==sr
        durations['encoding_and_decoded_verification_seconds']=time.perf_counter()-t
        elapsed=time.perf_counter()-started
        expected=records[selected['id']]
        assert transaction['outcome']==expected['transaction']['outcome']
        assert transfer.norm(plan)==expected['plan']
        assert run.waveform_hash(output)==expected['actual_output_waveform_sha256']
        if verified is not None:assert transfer.sha(verified)==transfer.sha(expected['verified_output'])
        report=dict(category=category,id=selected['id'],input_sha256=selected['input_sha256'],
            stages=durations,subcomponents_within_transaction_and_encoding=components,total_request_seconds=elapsed,
            outcome=transaction['outcome'],v52_actual_output_reproduced=True,
            waveform_unchanged=bool(np.array_equal(output,audio)),reference_used_for_decisions=False,
            technical_diagnostics=str(directory/'diagnostics.json'),candidate=str(verified) if verified else None)
        transfer.write(folder/'timing.json',report);reports.append(report)
        print('SHORT_PATH_LATENCY_CASE',category,elapsed,transaction['outcome'],flush=True)
    totals=[r['total_request_seconds'] for r in reports]
    transfer.write(ROOT/'results.json',dict(declaration_sha256=transfer.sha(ROOT/'declaration.json'),records=reports,
        sample_median_p95_seconds=np.quantile(totals,[.5,.95]).tolist(),sample_size=4,
        all_four_below_proposed30s_target=all(t<30 for t in totals),
        limitations='Small known-case diagnostic; first/warm cache differences retained. Not populationp95 or live application qualification. Python startup and user review/upload time excluded.',
        all_v52_outputs_reproduced=True,new_model_fits=0,production_eligible=False,research_validation_passed=False))
    print('SHORT_PATH_LATENCY_COMPLETE',totals,flush=True)


if __name__=='__main__':main()
