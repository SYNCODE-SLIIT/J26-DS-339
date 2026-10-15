"""Frozen A/B/C phase diagnostic; source-only predictions seal before scoring."""
import json
from pathlib import Path
import uuid
import numpy as np
import scipy
import soundfile as sf
from .pipeline import checksum
from .data_builder import write_json
from .segment_phase import estimate_segment_phase, interval_summary, VERSION as CANDIDATE_VERSION

PROTOCOL='runs/support_fragmentation/fc21ac54b04a4ee8b623f684c374f613/segment_comparison_protocol.json'
PROTOCOL_SHA256='eb4fab68c50dce995810f0319ab1ede1e2e8b63ae7e5e475bcbb2eb1767938f9'


def paired_counts(b,c):
    assert len(b['intervals'])==len(c['intervals'])
    pairs=list(zip(b['intervals'],c['intervals']))
    assert all((x['start_sample'],x['end_sample'])==(y['start_sample'],y['end_sample']) for x,y in pairs)
    return dict(gained_intervals=sum(not x['phase_feasible'] and y['phase_feasible'] for x,y in pairs),
                lost_intervals=sum(x['phase_feasible'] and not y['phase_feasible'] for x,y in pairs),
                both_feasible_intervals=sum(x['phase_feasible'] and y['phase_feasible'] for x,y in pairs),
                neither_feasible_intervals=sum(not x['phase_feasible'] and not y['phase_feasible'] for x,y in pairs))


def run():
    component=Path(__file__).resolve().parents[2];protocol_path=component/PROTOCOL
    if checksum(protocol_path)!=PROTOCOL_SHA256:raise ValueError('Prospective segment protocol changed')
    protocol=json.loads(protocol_path.read_text(encoding='utf-8'))
    assert protocol['status']=='declared_not_executed' and len(protocol['sources'])==18
    assert protocol['estimator_version']==CANDIDATE_VERSION
    diagnosis_path=Path(protocol['created_after_diagnosis']);diagnosis=json.loads(diagnosis_path.read_text(encoding='utf-8'))
    tracked=dict(diagnosis['tracked_inputs']);tracked[str(diagnosis_path)]=protocol['diagnosis_sha256']
    tracked[str(protocol_path)]=PROTOCOL_SHA256
    for s in protocol['sources']:
        tracked[s['audio_path']]=s['audio_sha256'];tracked[s['prediction_arrays']]=s['prediction_sha256']
    for item in protocol['baseline_evidence']:tracked[item['path']]=item['sha256']
    code_names=('segment_phase.py','segment_phase_scoring.py','segment_phase_experiment.py',
                'source_estimator.py','support_fragmentation.py','framewise_fixtures.py')
    code_paths=[Path(__file__).parent/n for n in code_names]
    code_paths += [component/'tests'/n for n in ('test_segment_phase.py','test_segment_phase_scoring.py')]
    for p in code_paths:tracked.setdefault(str(p),checksum(p))
    for p,h in tracked.items():
        if checksum(p)!=h:raise ValueError('Frozen input changed: '+p)
    root=component/'runs/segment_phase'/uuid.uuid4().hex;root.mkdir(parents=True)
    for p in code_paths:
        saved=root/'source_code_at_run'/p.relative_to(component);saved.parent.mkdir(parents=True,exist_ok=True);saved.write_bytes(p.read_bytes())
    declaration=dict(date='2026-10-15',protocol=protocol,protocol_sha256=PROTOCOL_SHA256,
        implementation_version=CANDIDATE_VERSION,code_sha256={str(p):checksum(p) for p in code_paths},
        numpy_version=np.__version__,scipy_version=scipy.__version__,
        scope='Source-only phase feasibility; no renderer, harmonic fit, output WAV or adoption',
        masks='Exact saved masks in all arms; whole frozen intervals never split around failures',
        truth_rule='All 18 prediction records and NPZ hashes seal before importing synthetic scorer or reading construction profiles')
    write_json(root/'declaration.json',declaration)
    record=dict(run_status='running',version='segment-phase-comparison-execution-0.1',data_status='exploratory_not_research',
        declaration=declaration,declaration_sha256=checksum(root/'declaration.json'),tracked_inputs=tracked,
        sources=[],artifacts=[],predictions_sealed_before_truth=False,new_candidate_wavs=0,
        accepted_prepared_packages=0,training_ready_examples=0,adopted=False)
    def save():write_json(root/'evidence.json',record)
    def artifact(path):record['artifacts'].append(dict(path=str(path),sha256=checksum(path)))
    save();print('SEGMENT_PHASE_RUN:',root,flush=True)
    try:
        for number,item in enumerate(protocol['sources']):
            print('PREDICT',number+1,'of 18',flush=True)
            audio,sr=sf.read(item['audio_path'],dtype='float32');original=audio.copy()
            with np.load(item['prediction_arrays']) as archive:frozen={k:archive[k] for k in archive.files}
            originals={k:v.copy() for k,v in frozen.items()}
            c=estimate_segment_phase(audio,sr,frozen);arrays=c.pop('arrays')
            b=interval_summary(frozen['estimated_frequency_hz'],frozen['retained_support'])
            for k in ('sample_reliable','guarded_support','retained_support'):np.testing.assert_array_equal(arrays[k],frozen[k])
            assert np.array_equal(audio,original)
            for k in originals:np.testing.assert_array_equal(originals[k],frozen[k])
            assert c['retained_samples']==b['retained_samples']
            path=root/f'prediction_{number:02d}.npz';np.savez_compressed(path,**arrays);artifact(path)
            record['sources'].append(dict(**item,B=b,C=c,paired=paired_counts(b,c),prediction_path=str(path),
                local_prediction_sha256=checksum(path),source_samples_unchanged=True,frozen_arrays_unchanged=True))
            save()
        seal=dict(status='sealed_before_truth_scoring',declaration_sha256=record['declaration_sha256'],
                  predictions=record['sources'])
        write_json(root/'prediction_seal.json',seal);artifact(root/'prediction_seal.json')
        record['prediction_seal_sha256']=checksum(root/'prediction_seal.json');record['predictions_sealed_before_truth']=True;save()
        # Separate stage: source-only predictions are already complete and hashed.
        from .segment_phase_scoring import score_saved_source
        previous=[json.loads(Path(p['path']).read_text(encoding='utf-8')) for p in protocol['baseline_evidence']]
        recording={r['record_id']:r['estimation'] for r in previous[0]['crops']}
        synthetic={r['profile']:r['prediction'] for r in previous[1]['sources']}
        fixtures_path=component/'runs/framewise_fixtures/1ae10a68d8c94dcc89d18c45719cef6f/evidence.json'
        profiles=json.loads(fixtures_path.read_text(encoding='utf-8'))['profiles']
        for row in record['sources']:
            row['A']=recording[row['source_id']] if row['kind']=='recording' else synthetic[row['source_id']]
            row['synthetic_scores']=None
            if row['kind']=='synthetic':
                audio,sr=sf.read(row['audio_path'],dtype='float32')
                with np.load(row['prediction_arrays']) as a:frozen={k:a[k] for k in a.files}
                with np.load(row['prediction_path']) as a:local={k:a[k] for k in a.files}
                profile=next(p for p in profiles if p['name']==row['source_id'])
                row['synthetic_scores']=score_saved_source(profile,audio,sr,frozen,local)
        sealed=json.loads((root/'prediction_seal.json').read_text(encoding='utf-8'))
        for before,after in zip(sealed['predictions'],record['sources']):
            assert all(after[k]==v for k,v in before.items())
            assert checksum(after['prediction_path'])==before['local_prediction_sha256']
        summary={}
        for name in ('vocadito','jamendo','synthetic'):
            rows=[r for r in record['sources'] if (r['kind']=='synthetic' if name=='synthetic' else r['source_id'].startswith(name+'_'))]
            summary[name]=dict(sources=len(rows),A_refused=sum(r['A']['status']=='refused' for r in rows),
                raw_islands=sum(len(r['C']['raw_islands']) for r in rows),
                short_islands=sum(r['C']['short_islands'] for r in rows),
                retained_samples=sum(r['B']['retained_samples'] for r in rows),
                B={k:sum(r['B'][k] for r in rows) for k in ('retained_intervals','phase_feasible_intervals','phase_feasible_samples','phase_feasible_long_intervals','nonfinite_on_retained','nonpositive_on_retained')},
                C={k:sum(r['C'][k] for r in rows) for k in ('retained_intervals','phase_feasible_intervals','phase_feasible_samples','phase_feasible_long_intervals','nonfinite_on_retained','nonpositive_on_retained')},
                paired={k:sum(r['paired'][k] for r in rows) for k in rows[0]['paired']})
        record['summary']=summary
        for p,h in tracked.items():
            if checksum(p)!=h:raise ValueError('Frozen input changed during run: '+p)
        assert checksum(root/'declaration.json')==record['declaration_sha256']
        record['run_status']='complete';save()
        write_json(component/'runs/segment_phase_current.json',dict(path=str(root)))
        print(json.dumps(summary,indent=2),flush=True)
    except BaseException as exc:
        record.update(run_status='failed',failure=dict(type=type(exc).__name__,reason=str(exc)));save();raise
    return root


if __name__=='__main__':run()
