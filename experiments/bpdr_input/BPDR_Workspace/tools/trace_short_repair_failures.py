"""Trace verified V52 successes/failures using saved predictions and actual PCM.

No new inference, fitting, threshold choice or intended-pitch claim. Label/support
information enters this retrospective diagnostic only, never a model proposal.
"""
from collections import Counter
from pathlib import Path
import numpy as np
import soundfile as sf

import evaluate_humtrans_short_delivery as study

ROOT=study.WS/'results/transformer_usability/2026-11-04_v61_pipeline_trace'


def main():
    transfer=study.transfer;_,pairs,originals=transfer.declare()
    root=study.ROOT;result=transfer.read(root/'results.json');proof=transfer.read(root/'integrity_receipt.json')
    assert proof['results_sha256']==transfer.sha(root/'results.json')
    sources={r['source_id']:r for r in originals};by_arm={}
    for arm,report in result['arms'].items():
        with np.load(transfer.ROOT/(arm+'_all_predictions.npz')) as z:predictions={k:z[k].copy() for k in z.files}
        traces=[];candidate_reasons=Counter();permission_reasons=Counter();audit_reasons=Counter()
        for r in report['records']:
            source=sources[r['source_id']]
            pair=pairs[r['pair_index']] if r['kind']=='corrupted' else None
            input_path=pair['path_C'] if pair else source['path_A']
            analysis_path=pair['analysis_C'] if pair else source['analysis_A']
            audio,sr=sf.read(input_path,dtype='float32');assert sr==22050
            with np.load(analysis_path) as z:views=z['raw_views'].copy();times=z['times_seconds'].copy();reliable=z['reliable'].copy()
            p=predictions[r['id']];assert p.shape==times.shape
            output,rate=sf.read(r['verified_output'],dtype='float32') if r['verified_output'] else (audio.copy(),sr)
            assert rate==sr and output.shape==audio.shape
            tx=r['transaction'];plan=r['plan'];audit=tx.get('post_render_audit')
            candidate_reasons.update(c['reason'] for c in r['candidates'] if c['reason'] is not None)
            permission_reasons.update(plan['reasons'])
            if audit and not audit['passed']:audit_reasons.update(audit['reasons'])
            raw=dict(mean_abs_on_reliable_cents=float(np.mean(abs(p[reliable]))) if reliable.any() else None,
                max_abs_on_reliable_cents=float(np.max(abs(p[reliable]))) if reliable.any() else None)
            localization=None
            if pair:
                with np.load(pair['target_path']) as z:positive=z['plateau'].copy();zero=z['known_zero'].copy();target=z['target'].copy();label_mask=z['mask'].copy()
                raw.update(positive_mae_cents=float(np.mean(abs(p[positive]-target[positive]))),
                    signed_response=float(np.mean(p[positive]*target[positive])/np.mean(target[positive]**2)),
                    max_known_zero_request_cents=float(np.max(abs(p[zero]))) if zero.any() else None)
                start,end=pair['support_seconds'];planned=plan['intervals_seconds']
                length=sum(b-a for a,b in planned)
                overlap=sum(max(0.,min(b,end)-max(a,start)) for a,b in planned)
                sample_times=np.arange(len(audio))/sr;outside=(sample_times<=start)|(sample_times>=end)
                reference,rate=sf.read(source['path_A'],dtype='float32');assert rate==sr
                np.testing.assert_array_equal(audio[outside],reference[outside])
                requested_frames=np.zeros(len(times),bool)
                for a,b in planned:requested_frames|=(times>=a)&(times<b)
                outside_frames=requested_frames&reliable&((times<start)|(times>=end))
                localization=dict(true_injected_support=[start,end],permitted_intervals=planned,
                    fraction_of_plan_inside_injection=overlap/length if length else None,
                    changed_samples_outside_injected_support=int(np.sum(output[outside]!=audio[outside])),
                    unlabelled_outside_frames_requested=int((outside_frames&~label_mask).sum()),
                    labelled_zero_frames_requested=int((outside_frames&label_mask&(target==0)).sum()))
            trace=dict(id=r['id'],source_id=r['source_id'],group=r['group'],kind=r['kind'],
                input_preparation=dict(working_path=input_path,sample_rate_hz=sr,sample_count=len(audio),
                    peak=float(abs(audio).max()),crop_offset_seconds=source['crop_offset_seconds'],
                    working_sha256=transfer.sha(input_path),reference_scope='Controlled recovery to an observed performance; natural intention unknown'),
                features=dict(path=analysis_path,views_shape=list(views.shape),reliable_frames=int(reliable.sum()),
                    used_short_channels=8,boundary_metadata_channels_used=False),
                predictions=raw,candidate_selection=dict(candidates=r['candidates'],curve_identity=plan['identity']),
                permission=dict(permitted=plan['permitted'],reasons=plan['reasons'],intervals_seconds=plan['intervals_seconds']),
                rendering=dict(outcome=tx['outcome'],rollback=tx['rollback'],post_render_passed=audit['passed'] if audit else None,
                    post_render_reasons=audit['reasons'] if audit else []),
                decoded_wav=dict(verified_output=r['verified_output'],encoded_audit_present='encoded_audit' in tx,
                    waveform_unchanged=r['waveform_unchanged'],changed_sample_count=int(np.sum(output!=audio))),
                localization=localization,
                reference_outcome={k:r[k] for k in ['verified_beneficial','verified_harm','known_zero_harm','before_mae_cents','after_mae_cents'] if k in r})
            traces.append(trace)
        examples={}
        categories=dict(beneficial=lambda t:t['reference_outcome'].get('verified_beneficial',False),
            preservation_harm=lambda t:t['reference_outcome'].get('known_zero_harm',False),
            changed_original=lambda t:t['kind']=='original' and not t['decoded_wav']['waveform_unchanged'],
            rollback=lambda t:t['rendering']['rollback'],
            unchanged_corrupted=lambda t:t['kind']=='corrupted' and t['decoded_wav']['waveform_unchanged'],
            wrong_direction_prediction=lambda t:t['predictions'].get('signed_response',0)<0)
        for name,predicate in categories.items():
            matches=sorted((t for t in traces if predicate(t)),key=lambda t:t['id'])
            examples[name]=matches[0] if matches else None
        by_arm[arm]=dict(rows=traces,representative_examples=examples,
            candidate_event_rejection_reasons=dict(candidate_reasons),permission_rejection_reasons=dict(permission_reasons),
            post_render_rejection_reasons=dict(audit_reasons),
            wrong_direction_prediction_cases=sum(t['predictions'].get('signed_response',0)<0 for t in traces),
            harmful_cases_with_unlabelled_edge_requests=sum(t['reference_outcome'].get('known_zero_harm',False) and t['localization']['unlabelled_outside_frames_requested']>0 for t in traces if t['localization']),
            harmful_unlabelled_outside_frame_requests=sum(t['localization']['unlabelled_outside_frames_requested'] for t in traces if t['reference_outcome'].get('known_zero_harm',False)),
            harmful_labelled_zero_frame_requests=sum(t['localization']['labelled_zero_frames_requested'] for t in traces if t['reference_outcome'].get('known_zero_harm',False)),
            verified_preservation_harms_with_actual_outside_injection_sample_changes=sum(t['reference_outcome'].get('known_zero_harm',False) and t['localization']['changed_samples_outside_injected_support']>0 for t in traces if t['localization']))
    ROOT.mkdir(parents=True,exist_ok=True)
    transfer.write(ROOT/'results.json',dict(source_sha256=transfer.sha(__file__),arms=by_arm,
        delivery_results_sha256=transfer.sha(root/'results.json'),delivery_integrity_sha256=transfer.sha(root/'integrity_receipt.json'),
        latency='Historical stage durations unavailable; this saved-artifact trace is not live inference timing',
        scope='Retrospective mechanism diagnosis on verified controlled data, not new validation or independent musical review',
        new_model_fits=0,nontraining_audio_accessed=False,production_eligible=False,research_validation_passed=False))
    print('SHORT_REPAIR_PIPELINE_TRACE_COMPLETE',{a:{'wrong_direction_cases':v['wrong_direction_prediction_cases'],
        'render_rejections':v['post_render_rejection_reasons'],'harms_with_actual_outside_samples':v['verified_preservation_harms_with_actual_outside_injection_sample_changes']}
        for a,v in by_arm.items()},flush=True)


if __name__=='__main__':main()
