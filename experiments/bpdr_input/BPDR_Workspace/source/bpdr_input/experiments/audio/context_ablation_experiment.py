"""Execute declared context ablation; freeze predictions before truth scoring."""
from bpdr_input import COMPONENT_ROOT, module_source
import json
from pathlib import Path
import uuid
import numpy as np
import scipy
import soundfile as sf
from bpdr_input.audio.pipeline import checksum
from bpdr_input.preparation.sources.data_builder import write_json
from bpdr_input.experiments.audio.context_ablation import estimate_contexts, VERSION
from bpdr_input.experiments.audio.segment_phase_experiment import paired_counts

PROTOCOL='runs/phase_regression_diagnostics/9a019681b7184066b1a15a5c78d1645e/context_ablation_protocol.json'
PROTOCOL_SHA256='10319621d2777ee4686e9c97fe906dbcc8dee5429400d1dceb90f954edc3ef64'


def run():
    component=COMPONENT_ROOT;protocol_path=component/PROTOCOL
    assert checksum(protocol_path)==PROTOCOL_SHA256
    protocol=json.loads(protocol_path.read_text(encoding='utf-8'))
    assert protocol['status']=='declared_not_executed' and len(protocol['sources'])==18
    diagnosis_path=Path(protocol['diagnosis_path']);diagnosis=json.loads(diagnosis_path.read_text())
    baseline_path=Path(protocol['baseline_evidence']);baseline=json.loads(baseline_path.read_text())
    previous={r['source_id']:r for r in baseline['sources']}
    tracked=dict(diagnosis['tracked_inputs']);tracked[str(diagnosis_path)]=protocol['diagnosis_sha256']
    tracked[str(baseline_path)]=protocol['baseline_evidence_sha256'];tracked[str(protocol_path)]=PROTOCOL_SHA256
    for a in baseline['artifacts']:tracked[a['path']]=a['sha256']
    code=[module_source(n) for n in ('context_ablation.py','context_ablation_scoring.py','context_ablation_experiment.py')]
    code.append(component/'tests/test_context_ablation.py')
    for p in code:tracked[str(p)]=checksum(p)
    for s in protocol['sources']:
        for key,hkey in [('audio_path','audio_sha256'),('frozen_arrays','frozen_sha256'),('local_arrays','local_sha256')]:tracked[s[key]]=s[hkey]
    for p,h in tracked.items():
        if checksum(p)!=h:raise ValueError('Frozen input changed: '+p)
    root=component/'runs/context_ablation'/uuid.uuid4().hex;root.mkdir(parents=True)
    for p in code:
        target=root/'source_code_at_run'/p.relative_to(component);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
    declaration=dict(date='2026-10-15',version=VERSION,protocol=protocol,protocol_sha256=PROTOCOL_SHA256,
        code_sha256={str(p):checksum(p) for p in code},numpy_version=np.__version__,scipy_version=scipy.__version__,
        notes=['No cached filter results; D/E share one full-clip filtered array per nonshort island.',
            'B/C replay is unchanged. Centers are consumed unchanged from saved C metadata; no new amplitude mask.',
            'All 18 new D/E prediction artifacts and metadata seal before importing synthetic scorer or reading construction profiles.',
            'No waveform generation, fitting, adoption or review modification. Full context is offline analysis, not causal streaming.'])
    write_json(root/'declaration.json',declaration)
    record=dict(run_status='running',version='context-ablation-execution-0.1',data_status='exploratory_not_research',
        declaration=declaration,declaration_sha256=checksum(root/'declaration.json'),tracked_inputs=tracked,
        sources=[],artifacts=[],predictions_sealed_before_truth=False,new_candidate_wavs=0,
        training_ready_examples=0,accepted_prepared_packages=0,adopted=False)
    def save():write_json(root/'evidence.json',record)
    def artifact(path):record['artifacts'].append(dict(path=str(path),sha256=checksum(path)))
    save();print('CONTEXT_ABLATION_RUN:',root,flush=True)
    try:
        for number,item in enumerate(protocol['sources']):
            print('PREDICT',number+1,'of 18',flush=True)
            old=previous[item['source_id']];audio,sr=sf.read(item['audio_path'],dtype='float32');original=audio.copy()
            with np.load(item['frozen_arrays']) as a:frozen={k:a[k] for k in a.files}
            saved={k:v.copy() for k,v in frozen.items()}
            result=estimate_contexts(audio,sr,frozen,old['C']['raw_islands']);arrays=result.pop('arrays')
            np.testing.assert_array_equal(original,audio)
            for k in saved:np.testing.assert_array_equal(saved[k],frozen[k])
            for k in ('sample_reliable','guarded_support','retained_support'):np.testing.assert_array_equal(arrays[k],frozen[k])
            path=root/f'prediction_{number:02d}.npz';np.savez_compressed(path,**arrays);artifact(path)
            row=dict(**item,B=old['B'],C=old['C'],D=result['D'],E=result['E'],
                context_path=str(path),context_sha256=checksum(path),status=result['status'],production_eligible=False,
                accepted_audio=None,source_samples_unchanged=True,frozen_arrays_unchanged=True)
            row['contrasts']={name:paired_counts(row[left],row[right]) for name,left,right in [('B_to_D','B','D'),('D_to_E','D','E'),('E_to_C','E','C')]}
            record['sources'].append(row);save()
        write_json(root/'prediction_seal.json',dict(status='sealed_before_truth_scoring',declaration_sha256=record['declaration_sha256'],predictions=record['sources']))
        artifact(root/'prediction_seal.json');record['prediction_seal_sha256']=checksum(root/'prediction_seal.json')
        record['predictions_sealed_before_truth']=True;save()
        from bpdr_input.experiments.audio.context_ablation_scoring import score_saved
        fixtures=component/'runs/framewise_fixtures/1ae10a68d8c94dcc89d18c45719cef6f/evidence.json'
        profiles=json.loads(fixtures.read_text())['profiles']
        for row in record['sources']:
            row['A']=previous[row['source_id']]['A'];row['synthetic_scores']=None
            if row['kind']=='synthetic':
                audio,sr=sf.read(row['audio_path'],dtype='float32')
                with np.load(row['frozen_arrays']) as a:frozen={k:a[k] for k in a.files}
                with np.load(row['local_arrays']) as a:local={k:a[k] for k in a.files}
                with np.load(row['context_path']) as a:new={k:a[k] for k in a.files}
                profile=next(p for p in profiles if p['name']==row['source_id'])
                row['synthetic_scores']=score_saved(profile,audio,sr,frozen,local,new,row['C']['raw_islands'])
        sealed=json.loads((root/'prediction_seal.json').read_text())
        for before,after in zip(sealed['predictions'],record['sources']):
            assert all(after[k]==v for k,v in before.items())
            assert checksum(after['context_path'])==before['context_sha256']
        summary={}
        for name in ('vocadito','jamendo','synthetic'):
            rows=[r for r in record['sources'] if (r['kind']=='synthetic' if name=='synthetic' else r['source_id'].startswith(name+'_'))]
            summary[name]=dict(sources=len(rows),A_refused=sum(r['A']['status']=='refused' for r in rows),
                retained_samples=sum(r['B']['retained_samples'] for r in rows),
                arms={arm:{k:sum(r[arm][k] for r in rows) for k in ('retained_intervals','phase_feasible_intervals','phase_feasible_samples','phase_feasible_long_intervals','nonfinite_on_retained','nonpositive_on_retained')} for arm in 'BCDE'},
                contrasts={contrast:{k:sum(r['contrasts'][contrast][k] for r in rows) for k in rows[0]['contrasts'][contrast]} for contrast in rows[0]['contrasts']})
        record['summary']=summary
        for p,h in tracked.items():
            if checksum(p)!=h:raise ValueError('Frozen inputs changed: '+p)
        assert checksum(root/'declaration.json')==record['declaration_sha256']
        record['run_status']='complete';save();write_json(component/'runs/context_ablation_current.json',dict(path=str(root)))
        print(json.dumps(summary,indent=2),flush=True)
    except BaseException as exc:
        record.update(run_status='failed',failure=dict(type=type(exc).__name__,reason=str(exc)));save();raise
    return root


if __name__=='__main__':run()
