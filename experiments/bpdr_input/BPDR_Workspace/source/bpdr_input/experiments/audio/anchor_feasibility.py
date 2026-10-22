"""Conditional constructed-phase closure budgets; not a renderer or runtime policy."""
from bpdr_input import COMPONENT_ROOT
from dataclasses import asdict
import json
from pathlib import Path
import uuid
import numpy as np
import soundfile as sf
from bpdr_input.preparation.sources.data_builder import write_json
from bpdr_input.experiments.audio.framewise_fixtures import constructed, signal, requested_curve
from bpdr_input.repair.framewise_renderer import RendererPolicy
from bpdr_input.repair.interventions import AuditPolicy, envelope
from bpdr_input.audio.pipeline import checksum

BASELINE='runs/framewise_fixtures/1ae10a68d8c94dcc89d18c45719cef6f/evidence.json'
PREVIOUS='runs/phase_comparison/6bdf0a7f2d3741e6a1a7c1c48dfa39de/evidence.json'
VERSION='constructed-anchor-feasibility-0.1'
DECLARATION=dict(gate='Gate F — source-anchored local resynthesis feasibility',date='2026-10-15',
    selection='All 32 existing constructed cases; all 36 supports counted separately, no excluded outcome',
    model='Known single harmonic-family source with unchanged amplitude and relative frequency integrated on the 22050-Hz sample grid',
    closure='Added cycles sum((f_new-f_source)/sr) over each support must be an integer to recover source fundamental phase modulo 2*pi',
    scope='Conditional model feasibility, not a theorem that all waveform editing is impossible or an actual natural-vocal renderer',
    fixed_interior='Original sample-interpolated supplied cents unchanged outside the two actual 80-ms ramps',
    endpoint='Correction and basis zero at first and last support sample; no duration/rate or outside-waveform operation',
    bounds=['Sign/magnitude preserving: ramp cents between min(0,requested(t)) and max(0,requested(t))',
            'Relaxed diagnostic: ramp cents anywhere in [-30,+30]'],
    bounds_meaning='New explicit feasibility hypotheses, not adopted musical permission or a change to any existing audit',
    necessary_bound='Integrate pointwise min/max possible ramp frequency changes with fixed interior, then inspect contained integers',
    fixed_basis='One sin-squared frequency-offset basis per ramp, multiplied by source Hz; scalar coefficient intervals enforce all pointwise bounds',
    solver='Choose feasible integer nearest nominal tapered cycles, lower integer on ties; use one common interpolation fraction through the two admissible coefficient intervals',
    tuning='Closed-form fixed basis, no parameter search or retuning after outcomes; do not clip coefficients or cents',
    numerical_validation=dict(phase_residual_cycles=1e-10,cents_roundoff=1e-8),
    validation_meaning='Float64 equation/roundoff checks only; not pitch-audit or musical tolerances',
    deliverables='Numerical arrays and records only; no new candidate WAV, extraction, fitting, pairs, adoption or accepted package',
    musical_review='Gate D/J4 deferred by owner; engineering continues',adopted=False,training_ready_examples=0)


def nearest_integer_in_interval(lo,hi,preferred):
    if not all(np.isfinite(v) for v in (lo,hi,preferred)) or lo>hi:raise ValueError('Invalid cycle interval')
    first,last=int(np.ceil(lo)),int(np.floor(hi))
    if first>last:return None
    options={first,last,max(first,min(last,int(np.floor(preferred)))),max(first,min(last,int(np.ceil(preferred))))}
    return min(options,key=lambda k:(abs(k-preferred),k))


def solve_support(f0,requested,sr,ramp_samples,max_cents=30.):
    """Solve one known-source support; input includes both endpoint samples.

    A necessary interval allows pointwise arbitrary ramp choices. The narrower
    fixed-basis interval additionally enforces the declared smooth shape. Failure
    of that basis alone is not proof that every possible bridge is infeasible.
    """
    f0=np.asarray(f0,dtype=np.float64);requested=np.asarray(requested,dtype=np.float64)
    if (f0.ndim!=1 or f0.shape!=requested.shape or not np.isfinite(f0).all()
            or not np.isfinite(requested).all() or np.any(f0<=0) or max_cents<=0
            or abs(requested).max()>max_cents or ramp_samples<2 or len(f0)<=2*ramp_samples):
        raise ValueError('Invalid constructed support/frequency/curve')
    n=len(f0);index=np.arange(n);ramp_seconds=ramp_samples/sr
    t=index/sr;duration=n/sr
    weight=envelope(t,(0.,duration),ramp_seconds);nominal=requested*weight
    nominal[[0,-1]]=0.
    base=f0*np.expm1(np.log(2)*nominal/1200.)
    raw=f0*np.expm1(np.log(2)*requested/1200.)
    entry=index<ramp_samples;exit=index>=n-ramp_samples;ramps=entry|exit
    base[~ramps]=raw[~ramps]
    basis=[]
    for mask in (entry,exit):
        p=np.flatnonzero(mask);b=np.zeros(n)
        b[p]=f0[p]*np.sin(np.pi*np.linspace(0,1,len(p)))**2
        b[p[[0,-1]]]=0.
        basis.append(b)
    base_cycles=float(base.sum()/sr);records={};arrays=dict(source_hz=f0,requested=requested,
        nominal_cents=nominal,nominal_frequency_delta=base,entry_mask=entry,exit_mask=exit)
    for name in ('sign_preserving','relaxed'):
        lower_c=np.minimum(0.,requested) if name=='sign_preserving' else np.full(n,-max_cents)
        upper_c=np.maximum(0.,requested) if name=='sign_preserving' else np.full(n,max_cents)
        lower=f0*np.expm1(np.log(2)*lower_c/1200.);upper=f0*np.expm1(np.log(2)*upper_c/1200.)
        lower[~ramps]=raw[~ramps];upper[~ramps]=raw[~ramps]
        lower[[0,-1]]=0.;upper[[0,-1]]=0.
        necessary_lo=float(lower.sum()/sr);necessary_hi=float(upper.sum()/sr)
        necessary_integer=nearest_integer_in_interval(necessary_lo,necessary_hi,base_cycles)
        ranges=[]
        for b in basis:
            active=b>0;fixed=ramps&~(basis[0]>0)&~(basis[1]>0)
            # Endpoints and ramp/interior junctions are fixed by the base curve.
            if np.any(base[fixed]<lower[fixed]-1e-12) or np.any(base[fixed]>upper[fixed]+1e-12):raise ValueError('Fixed samples violate feasibility bounds')
            lo=float(np.max((lower[active]-base[active])/b[active]));hi=float(np.min((upper[active]-base[active])/b[active]))
            if lo>hi:raise ValueError('Nominal valid curve missing from coefficient interval')
            ranges.append(dict(lower=lo,upper=hi,cycles_per_coefficient=float(b.sum()/sr)))
        basis_lo=base_cycles+sum(r['lower']*r['cycles_per_coefficient'] for r in ranges)
        basis_hi=base_cycles+sum(r['upper']*r['cycles_per_coefficient'] for r in ranges)
        integer=nearest_integer_in_interval(basis_lo,basis_hi,base_cycles)
        result=dict(necessary_cycle_interval=[necessary_lo,necessary_hi],necessary_contains_integer=necessary_integer is not None,
            fixed_basis_cycle_interval=[basis_lo,basis_hi],fixed_basis_contains_integer=integer is not None,
            coefficient_intervals=ranges,selected_integer=integer,phase_closed_solution=False)
        if integer is not None:
            fraction=0. if basis_hi==basis_lo else (integer-basis_lo)/(basis_hi-basis_lo)
            coefficients=[r['lower']+fraction*(r['upper']-r['lower']) for r in ranges]
            changed=base+sum(c*b for c,b in zip(coefficients,basis))
            cents=1200*np.log1p(changed/f0)/np.log(2)
            residual=float(changed.sum()/sr-integer)
            physical=dict(phase_equation=abs(residual)<=1e-10,positive_frequency=bool(np.all(f0+changed>0)),
                bounded=bool(np.all(cents>=lower_c-1e-8)&np.all(cents<=upper_c+1e-8)),
                interior_target=bool(np.allclose(cents[~ramps],requested[~ramps],rtol=0,atol=1e-8)),
                endpoint_zero=bool(changed[0]==0 and changed[-1]==0))
            if not all(physical.values()):raise ValueError('Closed-form phase solution failed validation')
            result.update(phase_closed_solution=True,coefficients=coefficients,phase_residual_cycles=residual,
                minimum_cents=float(cents.min()),maximum_cents=float(cents.max()),checks=physical)
            arrays[name+'_cents']=cents;arrays[name+'_added_cycles']=np.cumsum(changed)/sr
        records[name]=result
    return dict(nominal_added_cycles=base_cycles,**records),arrays


def run():
    component=COMPONENT_ROOT
    prior_path=component/PREVIOUS;prior=json.loads(prior_path.read_text(encoding='utf-8'))
    baseline_path=component/BASELINE;baseline=json.loads(baseline_path.read_text(encoding='utf-8'))
    if any(v['run_status']!='complete' or v['adopted'] for v in (prior,baseline)):raise ValueError('Invalid prior evidence')
    if baseline['renderer_policy']!=asdict(RendererPolicy()) or baseline['base_audit_limits']!=json.loads(json.dumps(asdict(AuditPolicy()))):raise ValueError('Policy changed')
    tracked=dict(prior['tracked_inputs']);tracked[str(prior_path)]=checksum(prior_path)
    for a in prior['artifacts']:tracked[a['path']]=a['sha256']
    own=Path(__file__);tracked[str(own)]=checksum(own)
    for path,digest in tracked.items():
        if checksum(path)!=digest:raise ValueError('Changed input: '+path)
    root=component/'runs/anchor_feasibility'/uuid.uuid4().hex;root.mkdir(parents=True)
    snapshots=root/'source_code_at_run';snapshots.mkdir()
    for path in tracked:
        if Path(path).suffix=='.py':(snapshots/Path(path).name).write_bytes(Path(path).read_bytes())
    write_json(root/'declaration.json',DECLARATION)
    record=dict(run_status='running',version=VERSION,data_status='exploratory_not_research',declaration=DECLARATION,
        declaration_sha256=checksum(root/'declaration.json'),tracked_inputs=tracked,numpy_version=np.__version__,
        cases=[],artifacts=[],adopted=False,accepted_prepared_packages=0,training_ready_examples=0,new_candidate_wavs=0)
    def save():write_json(root/'evidence.json',record)
    save();print('ANCHOR_FEASIBILITY_RUN:',root,flush=True)
    try:
        for profile in baseline['profiles']:
            print('PROFILE:',profile['name'],flush=True)
            st,f0,amplitude=constructed(profile);source=signal(f0,amplitude)
            cases=[c for c in baseline['cases'] if c['profile']==profile['name']]
            loaded,sr=sf.read(cases[0]['source_path'],dtype='float32')
            if sr!=22050 or not np.array_equal(source,loaded):raise ValueError('Known source does not match saved samples')
            frame_times=np.arange(len(source)//441+1)*.02
            for case in cases:
                curve=requested_curve(frame_times,case['intervals'],case['curve']);sample_curve=np.interp(st,frame_times,curve)
                supports=[]
                for number,(start,end) in enumerate(case['intervals']):
                    mask=(st>=start)&(st<end)
                    result,arrays=solve_support(f0[mask],sample_curve[mask],sr,round(.08*sr))
                    output=root/(profile['name']+'_'+case['curve']+'_'+str(number)+'.npz')
                    np.savez_compressed(output,times_seconds=st[mask],**arrays)
                    record['artifacts'].append(dict(path=str(output),sha256=checksum(output)))
                    supports.append(dict(start_seconds=start,end_seconds=end,arrays=str(output),**result))
                record['cases'].append(dict(profile=profile['name'],curve=case['curve'],source_path=case['source_path'],supports=supports,
                    baseline_audit_passed=case['audit']['passed'],no_renderer_run=True))
                save()
        all_supports=[s for c in record['cases'] for s in c['supports']]
        record['summary']=dict(cases=len(record['cases']),supports=len(all_supports),unchanged_baseline_passes=sum(c['baseline_audit_passed'] for c in record['cases']))
        for name in ('sign_preserving','relaxed'):
            records=[s[name] for s in all_supports]
            record['summary'][name]=dict(necessary_integer_supports=sum(r['necessary_contains_integer'] for r in records),
                fixed_basis_closed_supports=sum(r['phase_closed_solution'] for r in records),
                all_supports_closed_cases=sum(all(s[name]['phase_closed_solution'] for s in c['supports']) for c in record['cases']),
                max_phase_residual_cycles=max([abs(r['phase_residual_cycles']) for r in records if r['phase_closed_solution']],default=0))
        if any(checksum(path)!=digest for path,digest in tracked.items()) or checksum(root/'declaration.json')!=record['declaration_sha256']:raise ValueError('Frozen input/declaration changed')
        record['run_status']='complete';save();print(json.dumps(record['summary'],indent=2),flush=True)
    except BaseException as exc:
        record.update(run_status='failed',failure=dict(type=type(exc).__name__,reason=str(exc)));save();raise
    return root


if __name__=='__main__':run()
