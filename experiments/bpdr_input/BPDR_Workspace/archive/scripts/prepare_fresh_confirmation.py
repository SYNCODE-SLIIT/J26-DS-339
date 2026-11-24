"""Frozen V6 intervention protocol on the prospectively reserved fresh source cohort."""
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime,timezone
from math import gcd
from pathlib import Path
import json,ast
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
import acquire_fresh_confirmation as acquisition
from bpdr_input.experiments.learning import model_recovery as recovery,context_improvement as ex
import prepare_recoverable_step_pairs as generator

ROOT=acquisition.ROOT
PREP=ROOT/'preparation'


def declare():
    acquisition.check_lock();PREP.mkdir(exist_ok=True)
    manifest=ex.read(ROOT/'raw_source_manifest.json');assert len(manifest['records'])==12 and len({r['singer_id'] for r in manifest['records']})==4
    for row in manifest['records']:
        assert ex.sha(row['path'])==row['sha256'] and not row['training_permitted']
        payload=Path(row['path']).read_bytes();entry=row['archive_entry'];assert len(payload)==entry['decoded_bytes'] and acquisition.zlib.crc32(payload)&0xffffffff==entry['crc32']
    old=ex.read(ROOT.parent/'2026-10-28_v6/data_declaration.json')
    spec=dict(old['spec']);spec['root']=str(PREP/'cases')
    # Whole historical policy files include later proposal-only changes. Verify
    # the unchanged data-generation worker and the actual functions it calls.
    for p,h in old['source_hashes'].items():
        if Path(p).name in ('prepare_step_context_training.py','prepare_recoverable_step_pairs.py'):assert ex.sha(p)==h,p
    snapshot=ROOT.parent/'2026-10-28_v6/executed_training_sources/held_voice_policy.py'
    current=acquisition.WORKSPACE/'source/bpdr_input/repair/held_voice_policy.py'
    expected=next(h for p,h in old['source_hashes'].items() if Path(p).name=='held_voice_policy.py')
    assert ex.sha(snapshot)==expected
    def functions(path):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)}
    historical,qualified=functions(snapshot),functions(current)
    assert all(historical[k]==qualified[k] for k in ('spans','eligibility'))
    assert 'step_context' not in Path(generator.__file__).read_text()
    amendment=ex.read(PREP/'historical_helper_audit.json')
    assert amendment['revised_preparation_source_sha256']==ex.sha(Path(__file__))
    path=PREP/'declaration.json'
    if not path.exists():
        bound={str(p):ex.sha(p) for p in [Path(__file__).resolve(),Path(generator.__file__),Path(recovery.__file__),ROOT/'raw_source_manifest.json',ROOT/'pipeline_lock.json',ROOT/'component_source_lock.json',ROOT/'cohort_reservation.json']}
        ex.write(path,dict(created_at=datetime.now(timezone.utc).isoformat(),source_and_input_hashes=bound,spec=spec,old_v6_data_declaration_sha256=ex.sha(ROOT.parent/'2026-10-28_v6/data_declaration.json'),protocol='Exact prior V6 stable local step construction1.16..1.40s,0.26s flanks,<=70% held region,<=24% recording,trim3,median9 stability<=10c. Source-hash deterministic positions; +/-20/30c Praat physical interventions, existing original label audits unchanged. No targets supplied to runtime model/controller. All12 sources/48 attempts retained including exclusions; never replace a source or choose a different crop from outcomes.',preprocessing='Same source preprocessing: highest-energy native channel,fixed firstmin(20s,duration),no pad/normalization,polyphase22050Hz,PCM24 working copy,min5s,nonfinite/clipped rejection. Original identity later refers to these full working waveforms,not uncropped raw recordings.',role='fresh_controlled_confirmation_only',no_model_inference=True,research_validation_passed=False))
    d=ex.read(path)
    for p,h in d['source_and_input_hashes'].items():assert ex.sha(p)==h,p
    assert d['spec']==spec
    return d,manifest['records']


def prepare_original(task):
    torch.set_num_threads(2);i,row=task
    receipt=PREP/f'original_{i:02d}.json'
    if receipt.exists():
        source=ex.read(receipt);assert source['source']['sha256']==row['sha256']
        if 'working_audio' in source:
            assert ex.sha(source['working_audio'])==source['working_sha256']
            assert ex.sha(Path(source['working_audio']).parent/'original_features.npz')==source['original_features_sha256']
        return source
    assert ex.sha(row['path'])==row['sha256']
    audio,sr=sf.read(row['path'],dtype='float32',always_2d=True)
    selected=int(np.argmax(np.mean(audio.astype(np.float64)**2,axis=0)));wave=audio[:min(len(audio),round(20.*sr)),selected]
    divisor=gcd(sr,22050)
    if sr!=22050:wave=resample_poly(wave,22050//divisor,sr//divisor).astype(np.float32)
    wave=np.asarray(wave,dtype=np.float32)
    source=dict(source=row,crop_offset_seconds=0.,source_channel=selected,original_rate_hz=int(sr),original_sample_count=len(audio),original_channels=audio.shape[1],duration_seconds=len(wave)/22050,role='fresh_confirmation_only')
    if not np.isfinite(wave).all() or len(wave)/22050<5.:
        source['excluded']='Source shorter than declared five-second context or nonfinite';ex.write(receipt,source);return source
    if np.max(abs(wave))>=1:
        source['excluded']='Clipped source; no normalization rescue';ex.write(receipt,source);return source
    folder=PREP/'originals'/f'{i:02d}';folder.mkdir(parents=True,exist_ok=True);path=folder/'original_working.wav'
    sf.write(path,wave,22050,subtype='PCM_24');wave,_=sf.read(path,dtype='float32')
    before=recovery.measured(wave);features=folder/'original_features.npz';np.savez_compressed(features,**before['features'])
    source.update(working_audio=str(path),working_sha256=ex.sha(path),original_features_sha256=ex.sha(features),original_analysis=before['metadata'])
    ex.write(receipt,source);return source


def main():
    torch.set_num_threads(2);d,rows=declare()
    if (PREP/'prepared_manifest.json').exists():raise ValueError('Retain completed fresh preparation')
    sources=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for i,source in enumerate(pool.map(prepare_original,enumerate(rows,1)),1):
            sources.append(source);print('FRESH_ORIGINAL_PREPARED',i,'/',len(rows),source.get('excluded','working waveform ready'),flush=True)
    ex.write(PREP/'original_sources.json',sources);(PREP/'cases').mkdir(exist_ok=True);pairs=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        tasks=[(i,s,d['spec']) for i,s in enumerate(sources,1)]
        for i,record in enumerate(pool.map(generator.worker,tasks),1):
            assert record['source']['source']['sha256']==sources[i-1]['source']['sha256']
            pairs.extend(record['pairs']);ex.write(PREP/'preparation_progress.json',dict(completed_sources=i,total_sources=12,attempted=len(pairs),scoreable=sum('excluded' not in p for p in pairs)))
            print('FRESH_PAIRS_PREPARED',i,'/',12,'valid',sum('excluded' not in p for p in pairs),'/',len(pairs),flush=True)
    assert len(pairs)==48;ex.write(PREP/'pairs.json',pairs)
    hashes=ex.input_hashes(pairs)
    for source in sources:
        if 'working_audio' in source:
            for p in (Path(source['working_audio']),Path(source['working_audio']).parent/'original_features.npz'):hashes[str(p)]=ex.sha(p)
    ex.write(PREP/'input_hashes.json',hashes);declare()
    ex.write(PREP/'prepared_manifest.json',dict(attempted=48,scoreable=sum('excluded' not in p for p in pairs),excluded=sum('excluded' in p for p in pairs),source_performers=4,original_recordings=12,originals_with_working_audio=sum('working_audio' in s for s in sources),pairs_sha256=ex.sha(PREP/'pairs.json'),sources_sha256=ex.sha(PREP/'original_sources.json'),input_hashes_sha256=ex.sha(PREP/'input_hashes.json'),declaration_sha256=ex.sha(PREP/'declaration.json'),model_inference_started=False,selection_eligible=False,research_validation_passed=False))
    print('FRESH_PREPARATION_COMPLETE',48,sum('excluded' not in p for p in pairs),flush=True)

if __name__=='__main__':main()
