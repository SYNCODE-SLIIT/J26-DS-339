"""Record constructed permission/rollback cases using the unchanged v0.3 shift auditor."""
from bpdr_input import COMPONENT_ROOT, module_source
from dataclasses import asdict
import json
from pathlib import Path
import uuid
import numpy as np
import soundfile as sf
from bpdr_input.audio.analysis import extract
from bpdr_input.repair.interventions import AuditPolicy, audit_shift, independent_pitch, render_local
from bpdr_input.audio.pipeline import checksum
from bpdr_input.preparation.sources.data_builder import write_json
from bpdr_input.repair.repair_controller import FixturePolicy, plan_fixture, transact_fixture, waveform_hash


def run():
    component=COMPONENT_ROOT
    root=component/'runs/controller_fixtures'/uuid.uuid4().hex
    root.mkdir(parents=True)
    sr=22050
    t=np.arange(4*sr)/sr
    # Known constructed +20-cent input; supplied -20-cent proposal, not model inference.
    phase=2*np.pi*np.cumsum(np.full(len(t),440.*2**(20./1200)))/sr
    source=(.12*(np.sin(phase)+.4*np.sin(2*phase)+.2*np.sin(3*phase))).astype(np.float32)
    before,analyzer=extract(source,sr,component/'checkpoints')
    times=before['times_seconds']; yin=independent_pitch(source,sr)
    reliable=before['voiced_mask'] & np.isfinite(yin) & (yin>0)
    reliable[reliable] &= abs(1200*np.log2(yin[reliable]/before['f0_hz'][reliable]))<=AuditPolicy().tracker_agreement_cents
    protected=np.zeros(len(times),dtype=bool)
    proposal=np.where((times>=1.)&(times<1.9),-20.,0.)
    record=dict(run_status='running',scope='engineering_fixture_only',data_status='exploratory_not_research',
        training_ready_examples=0,production_eligible=False,active_pipeline_changed=False,
        fixture_policy=asdict(FixturePolicy()),audit_policy=asdict(AuditPolicy()),analyzer=analyzer,
        source_definition='4-second harmonic tone at known +20 cents relative to 440 Hz; local -20 proposal supplied by fixture',
        code_sha256={},cases=[],artifacts=[],limitations=[
            'No real-user inference, musical approvals or calibrated model uncertainty',
            'Constant-shift laboratory renderer is reused for fixture mechanics; not a production framewise renderer',
            'Permission values are predeclared illustrative fixture bounds, not development-calibrated thresholds',
            'No ACCEPT_PREPARED package, real vocal accuracy claim or team integration'])
    snapshots=root/'source_code_at_run'; snapshots.mkdir()
    for name in ('controller_fixtures','repair_controller','analysis','interventions'):
        path=module_source(name+'.py')
        record['code_sha256'][name]=checksum(path)
        (snapshots/path.name).write_bytes(path.read_bytes())
    source_path=root/'source.wav'; sf.write(source_path,source,sr,subtype='FLOAT')
    record['artifacts'].append(dict(path=str(source_path),sha256=checksum(source_path)))
    write_json(root/'evidence.json',record)
    print('CONTROLLER_FIXTURE_RUN:',root,flush=True)
    def audit(original,candidate,rate,intervals):
        after,_=extract(candidate,rate,component/'checkpoints')
        measured,_,_=audit_shift(before,after,yin,independent_pitch(candidate,rate),reliable,intervals[0],-20.,AuditPolicy())
        return measured
    def normal(x,rate,intervals): return render_local(x,rate,-20.,intervals[0],AuditPolicy())[0]
    def outside(x,rate,intervals): x[0]+=.01; return x
    cases=[('identity',np.zeros(len(times)),proposal,np.ones(len(times)),protected,normal,audit,'IDENTITY'),
           ('bounded_correction',proposal,proposal,np.ones(len(times)),protected,normal,audit,'VERIFIED_FIXTURE_CANDIDATE'),
           ('over_limit',proposal*2,proposal*2,np.ones(len(times)),protected,normal,audit,'WITHHELD'),
           ('disagreement',proposal,-proposal,np.ones(len(times)),protected,normal,audit,'WITHHELD'),
           ('uncertainty',proposal,proposal,np.full(len(times),6.),protected,normal,audit,'WITHHELD'),
           ('protected',proposal,proposal,np.ones(len(times)),(times>=1.)&(times<1.9),normal,audit,'WITHHELD'),
           ('outside_fault',proposal,proposal,np.ones(len(times)),protected,outside,audit,'ROLLED_BACK'),
           ('sample_count_fault',proposal,proposal,np.ones(len(times)),protected,lambda x,*a:x[:-1],audit,'ROLLED_BACK'),
           ('no_pitch_change',proposal,proposal,np.ones(len(times)),protected,lambda x,*a:x.copy(),audit,'ROLLED_BACK')]
    try:
        for name,cents,alternate,uncertainty,protect,renderer,auditor,expected in cases:
            plan=plan_fixture(source,sr,times,cents,alternate,uncertainty,reliable,protect)
            result,diagnostic=transact_fixture(source,sr,plan,renderer,auditor)
            if diagnostic['outcome']!=expected:
                raise ValueError(f'{name}: expected {expected}, observed {diagnostic["outcome"]}: {diagnostic}')
            if expected!='VERIFIED_FIXTURE_CANDIDATE' and not np.array_equal(source,result):
                raise ValueError('Withheld/identity/rollback fixture changed returned source')
            record['cases'].append(dict(name=name,expected_outcome=expected,diagnostics=diagnostic,
                output_sha256=waveform_hash(result),output_equals_source=bool(np.array_equal(result,source)),
                verified=True))
            if name=='bounded_correction':
                path=root/'verified_fixture_candidate.wav'; sf.write(path,result,sr,subtype='FLOAT')
                record['artifacts'].append(dict(path=str(path),sha256=checksum(path)))
            write_json(root/'evidence.json',record)
        for name,digest in record['code_sha256'].items():
            if checksum(module_source(name+'.py'))!=digest: raise ValueError('Source changed during fixture execution')
        record.update(run_status='complete',verified_cases=len(cases),accepted_prepared_packages=0)
        write_json(root/'evidence.json',record)
        print('Verified',len(cases),'constructed cases; no accepted repair package',flush=True)
    except BaseException as exc:
        record.update(run_status='failed',failure=dict(type=type(exc).__name__,reason=str(exc)))
        write_json(root/'evidence.json',record)
        raise
    return root


if __name__=='__main__': run()
