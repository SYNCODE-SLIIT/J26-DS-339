"""Seal audio-only predictions before separate construction-truth scoring."""
from bpdr_input import COMPONENT_ROOT, module_source
import json
from pathlib import Path
import uuid
import numpy as np
import soundfile as sf
from bpdr_input.preparation.sources.data_builder import write_json
from bpdr_input.audio.pipeline import checksum
from bpdr_input.experiments.audio.source_estimator import estimate_source

PROTOCOL = 'runs/known_control_diagnostics/bf53253259254eef89f9bb3fe09e24bd/source_estimation_protocol.json'
PROTOCOL_SHA256 = '63e2c38648b3d3dc0e217332e6a3d624979d55de5dc1b9e25887c262d3f110de'
BASELINE = 'runs/framewise_fixtures/1ae10a68d8c94dcc89d18c45719cef6f/evidence.json'
VERSION = 'source-estimation-experiment-0.1'


def distribution(values):
    a=np.asarray(values)
    if not a.size:return None
    if not np.isfinite(a).all():raise ValueError('Nonfinite score; do not silently discard')
    return dict(samples=int(a.size),median=float(np.median(a)),p10=float(np.quantile(a,.1)),
                p90=float(np.quantile(a,.9)),maximum=float(np.max(a)))


def score_prediction(profile, audio, sr, arrays):
    # Construction helpers are imported only by the scorer after predictions seal.
    from bpdr_input.experiments.audio.framewise_fixtures import constructed, signal
    st,f0,amp=constructed(profile,sr)
    if not np.array_equal(audio,signal(f0,amp,sr)):raise ValueError('Source construction does not match scorer')
    voiced=amp>0;n=len(audio)
    predicted=arrays.get('sample_reliable',np.zeros(n,dtype=bool))
    guarded=arrays.get('guarded_support',np.zeros(n,dtype=bool))
    retained=arrays.get('retained_support',np.zeros(n,dtype=bool))
    if any(a.shape!=(n,) or a.dtype!=bool for a in (predicted,guarded,retained)):
        raise ValueError('Scorer support grid mismatch')
    joint=retained&voiced
    result=dict(total_samples=n,true_voiced_samples=int(voiced.sum()),true_silent_samples=int((~voiced).sum()),
        predicted_reliable_samples=int(predicted.sum()),guarded_samples=int(guarded.sum()),retained_samples=int(retained.sum()),
        joint_scored_samples=int(joint.sum()),retained_true_voiced_coverage=float(joint.sum()/voiced.sum()),
        excluded_true_voiced_samples=int((voiced&~retained).sum()),
        predicted_false_support_in_silence=int((predicted&~voiced).sum()),
        guarded_false_support_in_silence=int((guarded&~voiced).sum()),
        retained_false_support_in_silence=int((retained&~voiced).sum()),
        frequency_error_cents=None,absolute_circular_phase_error_radians=None,amplitude_ratio=None,
        meaning='Descriptive scores also retained for refused profiles where defined; no pass threshold or permission')
    out=dict(sample_times=st,true_f0_hz=f0,true_fundamental_amplitude=amp,joint_scored_support=joint)
    if joint.any():
        frequency=arrays['estimated_frequency_hz'];phase=arrays['estimated_sine_phase'];amplitude=arrays['estimated_amplitude']
        true_phase=2*np.pi*np.cumsum(f0)/sr
        phase_error=np.abs(np.angle(np.exp(1j*(phase-true_phase))))
        ratio=np.divide(amplitude,amp,out=np.full(n,np.nan),where=voiced)
        frequency_error=np.full(n,np.nan)
        if np.isfinite(frequency[joint]).all() and np.all(frequency[joint]>0):
            frequency_error[joint]=abs(1200*np.log2(frequency[joint]/f0[joint]))
            result['frequency_error_cents']=distribution(frequency_error[joint])
        else:result['frequency_score_unavailable_reason']='Nonpositive/nonfinite frequency in declared joint support; no sample removal'
        result['absolute_circular_phase_error_radians']=distribution(phase_error[joint])
        result['amplitude_ratio']=distribution(ratio[joint])
        out.update(absolute_frequency_error_cents=frequency_error,absolute_circular_phase_error_radians=phase_error,
                   estimated_amplitude_ratio=ratio,true_sine_phase=true_phase)
    return result,out


def run():
    import torch
    component=COMPONENT_ROOT;protocol_path=component/PROTOCOL
    if checksum(protocol_path)!=PROTOCOL_SHA256:raise ValueError('Prospective source-estimation protocol changed')
    protocol=json.loads(protocol_path.read_text(encoding='utf-8'))
    if protocol['status']!='declared_not_executed' or len(protocol['sources'])!=8:raise ValueError('Wrong source declaration')
    diagnosis=Path(protocol['created_after_diagnosis']);previous=json.loads(diagnosis.read_text(encoding='utf-8'))
    tracked=dict(previous['tracked_inputs']);tracked[str(diagnosis)]=protocol['diagnosis_sha256']
    tracked[str(protocol_path)]=PROTOCOL_SHA256
    for item in previous['artifacts']:tracked[item['path']]=item['sha256']
    for p in (Path(__file__),module_source('source_estimator.py'),component/BASELINE):tracked[str(p)]=checksum(p)
    for item in protocol['sources']:tracked[item['path']]=item['sha256']
    for p,h in tracked.items():
        if checksum(p)!=h:raise ValueError('Frozen source changed: '+p)
    root=component/'runs/source_estimation'/uuid.uuid4().hex;root.mkdir(parents=True)
    snapshots=root/'source_code_at_run';snapshots.mkdir()
    for p in tracked:
        if Path(p).suffix=='.py':(snapshots/Path(p).name).write_bytes(Path(p).read_bytes())
    declaration=dict(date='2026-10-15',gate='Gate F — audio-only source estimation',protocol_sha256=PROTOCOL_SHA256,
        protocol=protocol,implementation_notes=[
            'No constructor, filename, profile, known center, source phase/F0 or correction passed to the two-argument estimator',
            'All eight predictions saved and hashed before any construction-truth scoring',
            '100-ms guard is 2205 samples; clip support [2205,n-2205), transition at j excludes [j-2205,j+2205)',
            'The nonpositive-frequency refusal covers the entire derived array, including edges and unvoiced regions; report counts on/outside retained support',
            'Refused sources stay refused; descriptive joint-support scores may still be recorded where defined, without dropping invalid joint samples',
            'Scoring runs after sealed predictions; no oracle phase alignment, truth-selected filter or post-outcome tuning'],
        new_candidate_wavs=0,adopted=False,training_ready_examples=0,accepted_prepared_packages=0)
    write_json(root/'declaration.json',declaration)
    record=dict(run_status='running',version=VERSION,data_status='exploratory_not_research',declaration=declaration,
        declaration_sha256=checksum(root/'declaration.json'),tracked_inputs=tracked,artifacts=[],sources=[],
        cpu_threads=torch.get_num_threads(),new_candidate_wavs=0,adopted=False,training_ready_examples=0,accepted_prepared_packages=0)
    def save():write_json(root/'evidence.json',record)
    def artifact(p):record['artifacts'].append(dict(path=str(p),sha256=checksum(p)))
    save();print('SOURCE_ESTIMATION_RUN:',root,flush=True)
    try:
        # Pass one: the estimator receives only decoded audio and sr.
        for number,item in enumerate(protocol['sources']):
            print('ESTIMATE SOURCE',number+1,'of 8',flush=True)
            audio,sr=sf.read(item['path'],dtype='float32')
            estimate=estimate_source(audio,sr)
            array_path=root/f'prediction_{number:02d}.npz';np.savez_compressed(array_path,**estimate.pop('arrays'));artifact(array_path)
            record['sources'].append(dict(profile=item['profile'],source_path=item['path'],prediction=estimate,prediction_arrays=str(array_path)))
            save()
        seal=dict(predictions=[dict(profile=c['profile'],prediction=c['prediction'],arrays=c['prediction_arrays'],sha256=checksum(c['prediction_arrays'])) for c in record['sources']],
                  source_truth_scoring_started=False,declaration_sha256=record['declaration_sha256'])
        write_json(root/'predictions_sealed.json',seal);artifact(root/'predictions_sealed.json')
        record['predictions_seal_sha256']=checksum(root/'predictions_sealed.json');save()
        # Pass two: oracle metadata is loaded only into the separate scorer.
        original=json.loads((component/BASELINE).read_text(encoding='utf-8'));profiles=original['profiles']
        for number,c in enumerate(record['sources']):
            print('SCORE SOURCE',number+1,'of 8',flush=True)
            audio,sr=sf.read(c['source_path'],dtype='float32')
            with np.load(c['prediction_arrays']) as a:arrays={k:a[k] for k in a.files}
            profile=next(p for p in profiles if p['name']==c['profile'])
            expected_analyzer=next(p for p in original['cases'] if p['profile']==c['profile'])['source_analyzer']
            if c['prediction']['analyzer']!=expected_analyzer:raise ValueError('Frozen analyzer/checkpoint provenance changed')
            scores,truth=score_prediction(profile,audio,sr,arrays)
            scores['construction_reference_center_hz']=profile['hz']
            scores['estimated_center_hz']=c['prediction'].get('estimated_center_hz')
            if scores['estimated_center_hz'] is not None:
                scores['center_difference_from_profile_reference_cents']=float(1200*np.log2(scores['estimated_center_hz']/profile['hz']))
            score_path=root/f'scores_{number:02d}.npz';np.savez_compressed(score_path,**truth);artifact(score_path)
            c.update(scores=scores,score_arrays=str(score_path));save()
        for sealed,c in zip(seal['predictions'],record['sources']):
            if sealed['prediction']!=c['prediction'] or checksum(c['prediction_arrays'])!=sealed['sha256']:
                raise ValueError('Prediction changed after truth scoring')
        rows=record['sources']
        record['summary']=dict(sources=len(rows),estimated=sum(c['prediction']['status']=='estimated' for c in rows),
            refused=sum(c['prediction']['status']=='refused' for c in rows),predictions_sealed_before_truth=True,
            refusals=[dict(profile=c['profile'],reasons=c['prediction']['reasons']) for c in rows if c['prediction']['status']=='refused'],
            total_true_voiced_samples=sum(c['scores']['true_voiced_samples'] for c in rows),
            total_joint_scored_samples=sum(c['scores']['joint_scored_samples'] for c in rows),
            total_retained_false_support_in_silence=sum(c['scores']['retained_false_support_in_silence'] for c in rows),
            metrics_are_descriptive_not_accuracy_or_adoption=True)
        if any(checksum(p)!=h for p,h in tracked.items()) or checksum(root/'declaration.json')!=record['declaration_sha256']:
            raise ValueError('Frozen experiment source changed')
        record['run_status']='complete';save();print(json.dumps(record['summary'],indent=2),flush=True)
    except BaseException as exc:
        record.update(run_status='failed',failure=dict(type=type(exc).__name__,reason=str(exc)));save();raise
    return root


if __name__=='__main__':run()
