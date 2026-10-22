"""Constructed-signal phase/amplitude diagnostic; no new rendered delivery or approval."""
import json
from pathlib import Path
import uuid
import numpy as np
import soundfile as sf
from scipy.signal import butter,hilbert,sosfiltfilt
from .data_builder import write_json
from .framewise_diagnostics import private_resynthesis
from .framewise_fixtures import requested_curve
from .interventions import envelope
from .pipeline import checksum

BOUNDARY='runs/boundary_comparison/e69c9a73c3f54f7c843701f5dadbf522/evidence.json'
BASELINE='runs/framewise_fixtures/1ae10a68d8c94dcc89d18c45719cef6f/evidence.json'
VERSION='constructed-phase-feasibility-0.1'
DECLARATION=dict(gate='Gate F — phase-aware joining feasibility',
    selection='All 32 original constructed cases, both existing local methods; all 72 ramps per method',
    strategy_to_investigate='A future join that controls relative fundamental phase instead of only tapering cents; not implemented here',
    method='Fourth-order Butterworth bandpass at [0.7,1.3] x known constructed center Hz, default odd-padded forward/backward SOS filter, then Hilbert analytic signal',
    band_scope='Known constructed center only; never inferred intended melody or a natural-vocal F0 oracle',
    diagnostic_mix_region='Only ramp samples with waveform blend weight strictly between .1 and .9',
    branch_amplitude_floor_fraction=.1,
    phase_opposition_marker='cos(relative fundamental phase)<0; descriptive only',
    algebraic_cancellation_ratio='abs((1-w)*z_source+w*z_full)/((1-w)*abs(z_source)+w*abs(z_full))',
    measured_amplitude_ratio='abs(z_local)/((1-w)*abs(z_source)+w*abs(z_full)); filtering and time-varying mixing do not commute',
    amplitude_markers=[.5,.8],markers_meaning='Descriptive ratios, not auditory/acceptance thresholds',
    controls=['Equal phase -> ratio 1','Equal-amplitude opposite phase at half blend -> ratio 0',
              'Quarter-cycle phase offset at half blend -> sqrt(.5)','Source/source diagnostic per profile'],
    save_stride_samples=8,summary_resolution='Full sample resolution; saved plot arrays every eight samples',
    musical_review='Deferred by owner; no experienced reviewers available. No reviews fabricated or replaced by trackers',
    output_scope='Numerical diagnostics and source snapshots only; no new candidate waveform, fitting, pair generation or adoption',
    tuning='Fixed filters/masks/markers; no selection or retuning after outcomes',
    adopted=False,accepted_prepared_packages=0,training_ready_examples=0)


def analytic_fundamental(wave, sr, center_hz):
    wave=np.asarray(wave,dtype=np.float64)
    if wave.ndim!=1 or not np.isfinite(wave).all() or not 0<.7*center_hz<1.3*center_hz<sr/2:
        raise ValueError('Invalid known-center diagnostic input')
    sos=butter(4,[.7*center_hz,1.3*center_hz],btype='bandpass',fs=sr,output='sos')
    return hilbert(sosfiltfilt(sos,wave))


def mixing_arrays(source, full, local, weight):
    source,full,local,weight=map(np.asarray,(source,full,local,weight))
    if (source.ndim!=1 or any(v.shape!=source.shape for v in (full,local,weight))
            or any(not np.isfinite(v).all() for v in (source,full,local,weight))
            or np.any((weight<0)|(weight>1))):raise ValueError('Invalid diagnostic branches/weight')
    a,b=abs(source),abs(full);denom=(1-weight)*a+weight*b
    phase=np.angle(full*np.conj(source))
    ratio=np.full(len(source),np.nan);measured=np.full(len(source),np.nan)
    nonzero=denom>0
    ratio[nonzero]=abs((1-weight[nonzero])*source[nonzero]+weight[nonzero]*full[nonzero])/denom[nonzero]
    measured[nonzero]=abs(local[nonzero])/denom[nonzero]
    return dict(phase_gap_radians=phase,cos_phase_gap=np.cos(phase),algebraic_ratio=ratio,
                measured_ratio=measured,source_amplitude=a,full_amplitude=b,local_amplitude=abs(local),blend_weight=weight)


def ramp_metrics(arrays, sample_times, start, end, side, amplitude_floor):
    lo,hi=(start,start+.08) if side=='entry' else (end-.08,end)
    weight=arrays['blend_weight'];scope=(sample_times>=lo)&(sample_times<hi)&(weight>.1)&(weight<.9)
    valid=scope&(arrays['source_amplitude']>amplitude_floor)&(arrays['full_amplitude']>amplitude_floor)
    if not valid.any():return dict(side=side,start_seconds=lo,end_seconds=hi,eligible_samples=int(scope.sum()),valid_samples=0,withheld=True)
    indices=np.flatnonzero(valid);i=int(indices[np.argmin(arrays['measured_ratio'][valid])])
    opposing=arrays['cos_phase_gap'][valid]<0
    return dict(side=side,start_seconds=lo,end_seconds=hi,eligible_samples=int(scope.sum()),valid_samples=int(valid.sum()),withheld=False,
        opposing_phase_fraction=float(np.mean(opposing)),minimum_algebraic_ratio=float(np.min(arrays['algebraic_ratio'][valid])),
        minimum_measured_ratio=float(arrays['measured_ratio'][i]),minimum_at_seconds=float(sample_times[i]),
        phase_gap_at_measured_min_radians=float(arrays['phase_gap_radians'][i]),
        measured_fraction_below_half=float(np.mean(arrays['measured_ratio'][valid]<.5)),
        measured_fraction_below_point8=float(np.mean(arrays['measured_ratio'][valid]<.8)))


def run():
    import scipy
    import parselmouth
    component=Path(__file__).resolve().parents[2]
    boundary_path=component/BOUNDARY;baseline_path=component/BASELINE
    boundary=json.loads(boundary_path.read_text(encoding='utf-8'));baseline=json.loads(baseline_path.read_text(encoding='utf-8'))
    if any(v['run_status']!='complete' or v['adopted'] for v in (boundary,baseline)):raise ValueError('Wrong source evidence')
    tracked=dict(boundary['tracked_inputs']);tracked[str(boundary_path)]=checksum(boundary_path)
    for artifact in boundary['artifacts']:tracked[artifact['path']]=artifact['sha256']
    own=Path(__file__);tracked[str(own)]=checksum(own)
    for path,digest in tracked.items():
        if checksum(path)!=digest:raise ValueError('Changed input: '+path)
    if checksum(parselmouth.__file__)!=boundary['backend_sha256']:raise ValueError('Backend changed')
    root=component/'runs/phase_feasibility'/uuid.uuid4().hex;root.mkdir(parents=True)
    snapshots=root/'source_code_at_run';snapshots.mkdir()
    for path in tracked:
        if Path(path).suffix=='.py':(snapshots/Path(path).name).write_bytes(Path(path).read_bytes())
    write_json(root/'declaration.json',DECLARATION)
    record=dict(run_status='running',version=VERSION,scope='Gate F diagnostics without musical reviewers',
        data_status='exploratory_not_research',declaration=DECLARATION,declaration_sha256=checksum(root/'declaration.json'),
        tracked_inputs=tracked,backend_path=parselmouth.__file__,backend_sha256=boundary['backend_sha256'],
        scipy_version=scipy.__version__,controls=[],cases=[],artifacts=[],adopted=False,accepted_prepared_packages=0,training_ready_examples=0)
    def save():write_json(root/'evidence.json',record)
    save();print('PHASE_FEASIBILITY_RUN:',root,flush=True)
    try:
        # Analytic phasors test the mixing equation without filter/estimator bias.
        for name,offset,expected in [('aligned',0.,1.),('opposed',np.pi,0.),('quarter_cycle',np.pi/2,np.sqrt(.5))]:
            z=np.ones(4,dtype=complex);other=z*np.exp(1j*offset);w=np.full(4,.5)
            a=mixing_arrays(z,other,(1-w)*z+w*other,w)
            if not np.allclose(a['algebraic_ratio'],expected,rtol=0,atol=1e-12):raise ValueError('Algebraic control failed')
            record['controls'].append(dict(name=name,expected_ratio=float(expected),measured_ratio=float(a['algebraic_ratio'][0]),passed=True))
        for profile in baseline['profiles']:
            print('PROFILE:',profile['name'],flush=True)
            cases=[c for c in baseline['cases'] if c['profile']==profile['name']]
            audio,sr=sf.read(cases[0]['source_path'],dtype='float32');st=np.arange(len(audio))/sr
            times=np.arange(len(audio)//441+1)*.02
            original_analytic=analytic_fundamental(audio,sr,profile['hz'])
            voiced=(st<2.5)|(st>=4.5) if profile['kind']=='rest' else np.ones(len(st),dtype=bool)
            floor=DECLARATION['branch_amplitude_floor_fraction']*float(np.median(abs(original_analytic[voiced])))
            if floor<=0:raise ValueError('Unsupported diagnostic source amplitude')
            blend=np.zeros(len(audio))
            for support in cases[0]['intervals']:blend=np.maximum(blend,envelope(st,support,.08))
            control=mixing_arrays(original_analytic,original_analytic,original_analytic,blend)
            valid=abs(original_analytic)>floor
            if not np.allclose(control['measured_ratio'][valid],1,rtol=0,atol=1e-12) or not np.allclose(control['phase_gap_radians'][valid],0,rtol=0,atol=1e-12):
                raise ValueError('Source/source diagnostic failed')
            record['controls'].append(dict(name='source_source_'+profile['name'],passed=True,valid_samples=int(valid.sum()),ratio_max_error=float(np.max(abs(control['measured_ratio'][valid]-1)))))
            for case in cases:
                counterpart=next(c for c in boundary['cases'] if c['profile']==profile['name'] and c['curve']==case['curve'])
                raw=requested_curve(times,case['intervals'],case['curve'])
                with np.load(counterpart['arrays']) as archived:tapered=archived['tapered_requested']
                for method,curve,path,peaks in [('baseline',raw,case['candidate_path'],counterpart['baseline_transition_peaks']),
                        ('boundary_taper',tapered,counterpart['candidate_path'],counterpart['transition_peaks'])]:
                    local,rate=sf.read(path,dtype='float32')
                    if rate!=sr:raise ValueError('Rate changed')
                    full=private_resynthesis(audio,sr,times,curve)
                    if not np.array_equal((audio*(1-blend)+full*blend).astype(np.float32),local):
                        raise ValueError('Native branch does not reproduce saved local waveform')
                    arrays=mixing_arrays(original_analytic,analytic_fundamental(full,sr,profile['hz']),
                        analytic_fundamental(local,sr,profile['hz']),blend)
                    ramps=[ramp_metrics(arrays,st,start,end,side,floor) for start,end in case['intervals'] for side in ('entry','exit')]
                    output=root/(profile['name']+'_'+case['curve']+'_'+method+'.npz')
                    stride=DECLARATION['save_stride_samples']
                    np.savez_compressed(output,sample_times=st[::stride],**{k:v[::stride] for k,v in arrays.items()})
                    record['artifacts'].append(dict(path=str(output),sha256=checksum(output)))
                    record['cases'].append(dict(profile=profile['name'],curve=case['curve'],method=method,
                        source_path=case['source_path'],local_candidate_path=path,local_reproduced_exactly=True,
                        amplitude_floor=floor,ramps=ramps,arrays=str(output),transition_peaks=peaks))
                    save()
        summary={}
        for method in ('baseline','boundary_taper'):
            ramps=[r for c in record['cases'] if c['method']==method for r in c['ramps']]
            supported=[r for r in ramps if not r['withheld']]
            summary[method]=dict(cases=32,ramps=len(ramps),supported_ramps=len(supported),withheld_ramps=len(ramps)-len(supported),
                ramps_with_any_opposing_phase=sum(r['opposing_phase_fraction']>0 for r in supported),
                ramps_measured_below_half=sum(r['minimum_measured_ratio']<.5 for r in supported),
                ramps_measured_below_point8=sum(r['minimum_measured_ratio']<.8 for r in supported),
                minimum_measured_ratio=min(r['minimum_measured_ratio'] for r in supported),
                minimum_algebraic_ratio=min(r['minimum_algebraic_ratio'] for r in supported))
        record['summary']=summary
        if any(checksum(path)!=digest for path,digest in tracked.items()) or checksum(parselmouth.__file__)!=record['backend_sha256'] or checksum(root/'declaration.json')!=record['declaration_sha256']:
            raise ValueError('Frozen inputs/declaration changed')
        record['run_status']='complete';save();print(json.dumps(summary,indent=2),flush=True)
    except BaseException as exc:
        record.update(run_status='failed',failure=dict(type=type(exc).__name__,reason=str(exc)));save();raise
    return root


if __name__=='__main__':run()
