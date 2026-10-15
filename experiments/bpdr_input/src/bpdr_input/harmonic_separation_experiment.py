"""Declared source-only decomposition; seal all fits before construction scoring."""
import json
from pathlib import Path
import uuid
import numpy as np
import soundfile as sf
from .data_builder import write_json
from .pipeline import checksum
from .harmonic_separation import separate_source,relative_rmse

SOURCE='runs/source_estimation/8c34deae8796487e9e261cd41bff9171/evidence.json'
BASELINE='runs/framewise_fixtures/1ae10a68d8c94dcc89d18c45719cef6f/evidence.json'
VERSION='harmonic-separation-experiment-0.1'


def score_parts(profile,audio,sr,arrays):
    from .framewise_fixtures import constructed,signal
    t,f0,amp=constructed(profile,sr)
    if not np.array_equal(signal(f0,amp,sr),audio):raise ValueError('Scorer/source construction mismatch')
    phase=2*np.pi*np.cumsum(f0)/sr
    truth=np.column_stack([amp*ratio*np.sin(k*phase) for k,ratio in ((1,1.),(2,.4),(3,.2))])
    scores=[]
    for k in range(3):
        scores.append(dict(harmonic=k+1,**{name:relative_rmse(arrays['estimated_partials'][:,k],truth[:,k],arrays[mask])
            for name,mask in (('fit','fit_support'),('evaluation','evaluation_support'),('retained','retained_support'))}))
    return scores,dict(sample_times=t,true_partials=truth)


def run():
    component=Path(__file__).resolve().parents[2];source_path=component/SOURCE
    prior=json.loads(source_path.read_text(encoding='utf-8'))
    if prior['run_status']!='complete' or prior['adopted']:raise ValueError('Wrong frozen estimation run')
    tracked=dict(prior['tracked_inputs']);tracked[str(source_path)]=checksum(source_path)
    for item in prior['artifacts']:tracked[item['path']]=item['sha256']
    for p in (Path(__file__),Path(__file__).parent/'harmonic_separation.py',component/BASELINE):tracked[str(p)]=checksum(p)
    for p,h in tracked.items():
        if checksum(p)!=h:raise ValueError('Changed frozen input: '+p)
    selected=[c['profile'] for c in prior['sources'] if c['prediction']['status']=='estimated']
    withheld=[c['profile'] for c in prior['sources'] if c['prediction']['status']!='estimated']
    root=component/'runs/harmonic_separation'/uuid.uuid4().hex;root.mkdir(parents=True)
    snapshots=root/'source_code_at_run';snapshots.mkdir()
    for p in tracked:
        if Path(p).suffix=='.py':(snapshots/Path(p).name).write_bytes(Path(p).read_bytes())
    declaration=dict(date='2026-10-15',gate='Gate F — source-estimated harmonic/residual reconstruction',
        selected=selected,withheld=withheld,selection='Only five non-refused prior estimates; retain all three prior refusals with no new fit',
        harmonics=[1,2,3],method='Six sine/cosine coefficients from fixed estimated sine phase, ordinary numpy.linalg.lstsq rcond=None; no regularization/fallback',
        fit='Retained samples strictly before clip midpoint',evaluation='Retained samples at/after clip midpoint',
        evaluation_limit='Only coefficient-fit samples held out; estimated phase and filter center used the full clip. Not independent prediction or research validation',
        support='Reuse exact frozen retained mask; set harmonic model to zero outside it; preserve outside source in residual',
        residual='float64 source minus supported harmonic model, no denoising or discarded residual',
        reconstruction='Measure raw float64 sum and its float32 cast without forced replacement; separate exact no-edit bypass',
        truth='All source-only coefficients/arrays sealed before scoring against known three-part construction; no truth phase/coefficients input to fitter',
        scope='Known three-harmonic family; descriptive fidelity, no natural-vocal generalization or musical pass threshold',
        stop='Insufficient fit/evaluation support, rank !=6, invalid/nonpositive prediction or nonfinite coefficients: explicit stop; no tuning',
        outputs='Numerical components only; no candidate WAV, pitch change, new source estimate, training, pair or adopted renderer',
        musical_review='Gate D/J4 deferred by owner',adopted=False,training_ready_examples=0,accepted_prepared_packages=0)
    write_json(root/'declaration.json',declaration)
    record=dict(run_status='running',version=VERSION,data_status='exploratory_not_research',declaration=declaration,
        declaration_sha256=checksum(root/'declaration.json'),tracked_inputs=tracked,artifacts=[],sources=[],
        new_candidate_wavs=0,adopted=False,training_ready_examples=0,accepted_prepared_packages=0)
    def save():write_json(root/'evidence.json',record)
    def artifact(p):record['artifacts'].append(dict(path=str(p),sha256=checksum(p)))
    save();print('HARMONIC_SEPARATION_RUN:',root,flush=True)
    try:
        for number,item in enumerate(prior['sources']):
            if item['profile'] in withheld:
                record['sources'].append(dict(profile=item['profile'],status='WITHHELD_PRIOR_REFUSAL',
                    prior_status=item['prediction']['status'],reasons=item['prediction']['reasons'],arrays=None));save();continue
            print('DECOMPOSE:',item['profile'],flush=True)
            audio,sr=sf.read(item['source_path'],dtype='float32')
            with np.load(item['prediction_arrays']) as a:prediction={k:a[k] for k in a.files}
            result=separate_source(audio,sr,dict(status=item['prediction']['status'],arrays=prediction))
            arrays=result.pop('arrays')
            path=root/f'decomposition_{number:02d}.npz';np.savez_compressed(path,**arrays);artifact(path)
            record['sources'].append(dict(profile=item['profile'],source_path=item['source_path'],status=result['status'],
                decomposition=result,arrays=str(path)));save()
        seal=dict(construction_scoring_started=False,declaration_sha256=record['declaration_sha256'],
            decompositions=[dict(profile=c['profile'],decomposition=c['decomposition'],arrays=c['arrays'],sha256=checksum(c['arrays'])) for c in record['sources'] if c['status']=='decomposed'])
        write_json(root/'decompositions_sealed.json',seal);artifact(root/'decompositions_sealed.json')
        record['decompositions_seal_sha256']=checksum(root/'decompositions_sealed.json');save()
        profiles=json.loads((component/BASELINE).read_text(encoding='utf-8'))['profiles']
        for number,c in enumerate(record['sources']):
            if c['status']!='decomposed':continue
            print('SCORE:',c['profile'],flush=True)
            audio,sr=sf.read(c['source_path'],dtype='float32')
            with np.load(c['arrays']) as a:arrays={k:a[k] for k in a.files}
            profile=next(p for p in profiles if p['name']==c['profile'])
            scores,truth=score_parts(profile,audio,sr,arrays)
            path=root/f'construction_scores_{number:02d}.npz';np.savez_compressed(path,**truth);artifact(path)
            c.update(harmonic_fidelity=scores,score_arrays=str(path));save()
        for frozen in seal['decompositions']:
            c=next(c for c in record['sources'] if c['profile']==frozen['profile'])
            if c['decomposition']!=frozen['decomposition'] or checksum(c['arrays'])!=frozen['sha256']:raise ValueError('Decomposition changed after scoring')
        measured=[c for c in record['sources'] if c['status']=='decomposed']
        record['summary']=dict(sources=len(record['sources']),decomposed=len(measured),withheld=len(withheld),
            new_fit_refusals=sum(c['status']=='refused' for c in record['sources']),
            exact_zero_bypasses=sum(c['decomposition']['zero_edit_bypass_exact'] for c in measured),
            exact_raw_float32_recombinations=sum(c['decomposition']['raw_recombination_float32_different_samples']==0 for c in measured),
            raw_float32_different_samples=sum(c['decomposition']['raw_recombination_float32_different_samples'] for c in measured),
            all_outside_exact=all(c['decomposition']['raw_recombination_outside_exact'] for c in measured),
            max_raw_recombination_abs_error=max((c['decomposition']['raw_recombination_max_abs_error'] for c in measured),default=0.),
            per_profile=[dict(profile=c['profile'],coefficient_evaluation_relative_rmse=c['decomposition']['harmonic_source_evaluation']['relative_rmse'],
                partial_evaluation_relative_rmse=[h['evaluation']['relative_rmse'] for h in c['harmonic_fidelity']],
                raw_float32_different_samples=c['decomposition']['raw_recombination_float32_different_samples']) for c in measured])
        if any(checksum(p)!=h for p,h in tracked.items()) or checksum(root/'declaration.json')!=record['declaration_sha256']:
            raise ValueError('Frozen inputs/declaration changed')
        record['run_status']='complete';save();print(json.dumps(record['summary'],indent=2),flush=True)
    except BaseException as exc:
        record.update(run_status='failed',failure=dict(type=type(exc).__name__,reason=str(exc)));save();raise
    return root


if __name__=='__main__':run()
