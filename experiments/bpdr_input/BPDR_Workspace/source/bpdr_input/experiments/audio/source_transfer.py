"""Frozen source-only transfer screen; no vocal correction or annotations input."""
from bpdr_input import COMPONENT_ROOT, module_source
import json
from pathlib import Path
import uuid
import numpy as np
import soundfile as sf
from bpdr_input.preparation.sources.data_builder import write_json
from bpdr_input.experiments.audio.harmonic_separation import separate_source
from bpdr_input.audio.pipeline import checksum
from bpdr_input.repair.repair_controller import waveform_hash
from bpdr_input.experiments.audio.source_estimator import estimate_source

PROTOCOL='runs/estimated_harmonic_correction/a2eea865ced64ec19aafe310e9527d61/transfer_screen_protocol.json'
PROTOCOL_SHA256='1236a98112eeda11cff1c9150db96fe73a8497a2d4e1be13ab39c66442ae6906'
ESTIMATION='runs/source_estimation/8c34deae8796487e9e261cd41bff9171/evidence.json'
VERSION='existing-training-source-transfer-0.1'


def coverage_summary(arrays, sample_count):
    """Predicted support / decoded samples; never true-voiced coverage."""
    result=dict(decoded_samples=sample_count,denominator='All decoded crop samples, not human-voiced samples')
    for key in ('sample_reliable','guarded_support','retained_support'):
        mask=arrays.get(key)
        if mask is None:result[key+'_samples']=None;result[key+'_fraction']=None;continue
        if mask.shape!=(sample_count,) or mask.dtype!=bool:raise ValueError('Invalid predicted sample support')
        result[key+'_samples']=int(mask.sum());result[key+'_fraction']=float(mask.mean())
    frequency=arrays.get('estimated_frequency_hz');support=arrays.get('retained_support')
    if frequency is not None:
        if frequency.shape!=(sample_count,):raise ValueError('Invalid frequency sample grid')
        bad=~np.isfinite(frequency)|(frequency<=0)
        result['nonpositive_total']=int(bad.sum())
        result['nonpositive_on_retained']=int((bad&support).sum()) if support is not None else None
        result['nonpositive_outside_retained']=int((bad&~support).sum()) if support is not None else None
    else:result.update(nonpositive_total=None,nonpositive_on_retained=None,nonpositive_outside_retained=None)
    return result


def screen_crop(audio_float32, sample_rate_hz):
    """Only samples/rate reach estimator and fitter; no dataset/notes/group fields."""
    original_hash=waveform_hash(audio_float32)
    prediction=estimate_source(audio_float32,sample_rate_hz)
    arrays=prediction['arrays'];coverage=coverage_summary(arrays,len(audio_float32))
    if prediction['status']=='estimated':
        decomposition=separate_source(audio_float32,sample_rate_hz,prediction)
        components=decomposition.pop('arrays')
    else:
        decomposition=dict(status='WITHHELD_ESTIMATOR_REFUSAL',reasons=prediction['reasons'])
        components=None
    prediction=prediction.copy();prediction.pop('arrays')
    if waveform_hash(audio_float32)!=original_hash:raise ValueError('Source samples changed during screen')
    return dict(estimation=prediction,coverage=coverage,decomposition=decomposition,source_samples_unchanged=True,
                prediction_arrays=arrays,decomposition_arrays=components)


def validate_selection(protocol,manifest):
    expected={r['record_id']:r for r in manifest['records']}
    selected=protocol['sources'];ids=[s['record_id'] for s in selected]
    if len(ids)!=10 or len(set(ids))!=10 or set(ids)!=set(expected):raise ValueError('Transfer selection changed')
    if sum(s['dataset']=='vocadito' for s in selected)!=3 or sum(s['dataset']=='jamendo' for s in selected)!=7:
        raise ValueError('Declared dataset counts changed')
    for s in selected:
        r=expected[s['record_id']]
        if (s['split']!='train' or r['split']!='train' or r['training_eligible']
                or s['dataset']!=r['dataset'] or s['audio_path']!=r['accepted_audio_path']
                or s['sha256']!=r['technical_result']['accepted_audio']['checksum']
                or r['technical_result']['status']!='ACCEPT_UNCHANGED'):
            raise ValueError('Non-training/changed/nontechnical source in transfer selection')


def run():
    import torch
    component=COMPONENT_ROOT;protocol_path=component/PROTOCOL
    if checksum(protocol_path)!=PROTOCOL_SHA256:raise ValueError('Prospective transfer protocol changed')
    protocol=json.loads(protocol_path.read_text(encoding='utf-8'))
    if protocol['status']!='declared_not_executed':raise ValueError('Wrong prospective protocol')
    manifest_path=Path(protocol['preparation_manifest']);manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    validate_selection(protocol,manifest)
    synthetic_path=Path(protocol['created_after_synthetic_run']);prior=json.loads(synthetic_path.read_text(encoding='utf-8'))
    tracked=dict(prior['tracked_inputs']);tracked[str(synthetic_path)]=protocol['synthetic_evidence_sha256']
    for item in prior['artifacts']:tracked[item['path']]=item['sha256']
    tracked[str(protocol_path)]=PROTOCOL_SHA256;tracked[str(manifest_path)]=protocol['preparation_manifest_sha256']
    for name,key in [('source_estimator.py','estimator_sha256'),('harmonic_separation.py','decomposition_sha256')]:
        tracked[str(module_source(name))]=protocol[key]
    tracked[str(Path(__file__))]=checksum(__file__)
    for item in protocol['sources']:tracked[item['audio_path']]=item['sha256']
    for p,h in tracked.items():
        if checksum(p)!=h:raise ValueError('Frozen source/code changed: '+p)
    expected_analyzer=json.loads((component/ESTIMATION).read_text(encoding='utf-8'))['sources'][0]['prediction']['analyzer']
    root=component/'runs/source_transfer'/uuid.uuid4().hex;root.mkdir(parents=True)
    snapshots=root/'source_code_at_run';snapshots.mkdir()
    for p in tracked:
        if Path(p).suffix=='.py':(snapshots/Path(p).name).write_bytes(Path(p).read_bytes())
    declaration=dict(date='2026-10-15',gate='Gate F — fixed source-only Vocadito/Jamendo transfer',
        protocol=protocol,protocol_sha256=PROTOCOL_SHA256,
        implementation_notes=['Reuse frozen estimator and decomposition code; no patch/tuning of global-frequency refusal or support',
            'All ten saved technically accepted unchanged crops, train only; no label/notes/group metadata passed to source-only API',
            'Estimator refusals are preserved; no fitter invoked on refused predictions. Save arrays even for refused crops',
            'Coverage uses total decoded samples; unavailable support is null, not an invented zero measurement',
            'Source-only reconstruction is descriptive and phase uses the full clip; no annotation-based accuracy or musical approval',
            'No output/candidate WAV, pitch correction, new separation/corpus expansion, research learner training, pair or adoption'],
        training_ready_examples=0,accepted_prepared_packages=0,adopted=False)
    write_json(root/'declaration.json',declaration)
    record=dict(run_status='running',version=VERSION,data_status='exploratory_not_research',declaration=declaration,
        declaration_sha256=checksum(root/'declaration.json'),tracked_inputs=tracked,artifacts=[],crops=[],cpu_threads=torch.get_num_threads(),
        new_candidate_wavs=0,adopted=False,training_ready_examples=0,accepted_prepared_packages=0)
    def save():write_json(root/'evidence.json',record)
    def artifact(p):record['artifacts'].append(dict(path=str(p),sha256=checksum(p)))
    save();print('SOURCE_TRANSFER_RUN:',root,flush=True)
    try:
        for number,item in enumerate(protocol['sources']):
            print('CROP:',item['record_id'],flush=True)
            audio,sr=sf.read(item['audio_path'],dtype='float32')
            screened=screen_crop(audio,sr)
            if screened['estimation']['analyzer']!=expected_analyzer:raise ValueError('Analyzer/checkpoint provenance changed')
            pred=screened.pop('prediction_arrays');parts=screened.pop('decomposition_arrays')
            path=root/f'prediction_{number:02d}.npz';np.savez_compressed(path,**pred);artifact(path)
            row=dict(**item,**screened,prediction_arrays=str(path),decomposition_arrays=None,
                     sample_rate_hz=sr,sample_count=len(audio),duration_seconds=len(audio)/sr)
            if parts is not None:
                path=root/f'decomposition_{number:02d}.npz';np.savez_compressed(path,**parts);artifact(path)
                row['decomposition_arrays']=str(path)
            record['crops'].append(row);save()
        by_dataset={}
        for dataset in ('vocadito','jamendo'):
            rows=[r for r in record['crops'] if r['dataset']==dataset]
            available=[r for r in rows if r['coverage']['retained_support_samples'] is not None]
            by_dataset[dataset]=dict(crops=len(rows),groups=len({r['group_id'] for r in rows}),
                group_kind=rows[0]['group_kind'],estimated=sum(r['estimation']['status']=='estimated' for r in rows),
                estimator_refused=sum(r['estimation']['status']=='refused' for r in rows),
                decomposed=sum(r['decomposition']['status']=='decomposed' for r in rows),
                support_computed_crops=len(available),decoded_samples_with_computed_support=sum(r['sample_count'] for r in available),
                retained_predicted_samples=sum(r['coverage']['retained_support_samples'] for r in available))
        record['summary']=dict(crops=len(record['crops']),datasets=by_dataset,
            estimated=sum(r['estimation']['status']=='estimated' for r in record['crops']),
            estimator_refused=sum(r['estimation']['status']=='refused' for r in record['crops']),
            decomposed=sum(r['decomposition']['status']=='decomposed' for r in record['crops']),
            refusal_reasons={reason:sum(reason in r['estimation']['reasons'] for r in record['crops']) for reason in sorted({reason for r in record['crops'] for reason in r['estimation']['reasons']})},
            coverage_meaning='Predicted support only; no true-voiced, musical accuracy or population yield inference',
            all_source_samples_unchanged=all(r['source_samples_unchanged'] for r in record['crops']))
        if any(checksum(p)!=h for p,h in tracked.items()) or checksum(root/'declaration.json')!=record['declaration_sha256']:
            raise ValueError('Frozen inputs/declaration changed')
        record['run_status']='complete';save();print(json.dumps(record['summary'],indent=2),flush=True)
    except BaseException as exc:
        record.update(run_status='failed',failure=dict(type=type(exc).__name__,reason=str(exc)));save();raise
    return root


if __name__=='__main__':run()
