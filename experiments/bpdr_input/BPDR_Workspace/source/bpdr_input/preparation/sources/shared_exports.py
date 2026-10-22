"""Component-owned intake for shared exports; never imports/runs shared code."""
import hashlib
import json
from pathlib import Path,PurePosixPath
import re
import stat
import uuid
import zipfile
import numpy as np
import soundfile as sf

VERSION='shared-bpdr-export-intake-0.1'
ARTIFACTS=('vocal.wav','features.npz','boundaries.json','masks.npz')


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def archive_entries(archive):
    """Preflight before any write; skip metadata without modifying the archive."""
    kept=[];skipped=[];seen=set();total=0
    for item in archive.infolist():
        name=item.filename;p=PurePosixPath(name)
        if item.is_dir():continue
        if any(x=='__MACOSX' or x=='.DS_Store' or x.startswith('._') for x in p.parts):
            skipped.append(name);continue
        if ('\\' in name or ':' in name or p.is_absolute() or '..' in p.parts or len(p.parts)<3
                or p.parts[0]!='outputs' or not re.fullmatch(r'track_\d{7,}',p.parts[1])
                or any(x.endswith((' ','.')) for x in p.parts) or stat.S_ISLNK(item.external_attr>>16)):
            raise ValueError('Unsafe/unexpected export path: '+name)
        folded=name.casefold()
        if folded in seen:raise ValueError('Duplicate/case-colliding archive path: '+name)
        seen.add(folded);total+=item.file_size;kept.append(item)
    if not kept or len(kept)>100000 or total>2_000_000_000:raise ValueError('Empty or oversized export archive')
    return kept,skipped


def import_archive(source,component):
    component=Path(component).resolve();source=Path(source).resolve();sha=digest(source)
    base=component/'data/shared_exports';base.mkdir(parents=True,exist_ok=True)
    target=base/sha[:20]
    if target.exists():
        receipt=json.loads((target/'import_receipt.json').read_text())
        if receipt['archive_sha256']!=sha:raise ValueError('Import identity collision')
        for item in receipt['files']:
            if digest(target/item['relative_path'])!=item['sha256']:raise ValueError('Imported file changed')
        return target,receipt
    with zipfile.ZipFile(source) as archive:
        kept,skipped=archive_entries(archive)
        staging=base/(sha[:20]+'.staging-'+uuid.uuid4().hex);staging.mkdir()
        files=[]
        for item in kept:
            path=(staging/item.filename).resolve()
            if not path.is_relative_to(staging.resolve()):raise ValueError('Extraction leaves component staging')
            path.parent.mkdir(parents=True,exist_ok=True)
            raw=archive.read(item);path.write_bytes(raw)
            files.append(dict(relative_path=item.filename,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
        receipt=dict(version=VERSION,source_archive=str(source),archive_sha256=sha,
            original_archive_unchanged=True,files=files,skipped_metadata_files=len(skipped),
            preserved_files=len(files),preserved_bytes=sum(r['bytes'] for r in files))
        if digest(source)!=sha:raise ValueError('Source archive changed during import')
        (staging/'import_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
        staging.rename(target)
    return target,receipt


def load_export(folder):
    """Verified source input, NOT an audited pair or musical/training approval."""
    folder=Path(folder);record=json.loads((folder/'record.json').read_text())
    if record.get('schema_version')!='bpdr-track-v1' or record.get('status')!='exported':raise ValueError('Not an exported BPDR crop')
    if record.get('reference_type')!='model_estimate' or record.get('training_eligible') is not False:
        raise ValueError('Unexpected reference/training claim')
    if set(record.get('artifacts',{}))!=set(ARTIFACTS):raise ValueError('Missing/unexpected BPDR artifacts')
    for name in ARTIFACTS:
        if digest(folder/name)!=record['artifacts'][name]:raise ValueError('BPDR artifact hash mismatch: '+name)
    if folder.parent.name!=record['mtg_track_id'] or record['mtg_track_id']!=f"track_{int(record['track_id']):07d}":
        raise ValueError('Track ID mismatch')
    full=folder.parent/'melody/game_vocal.json'
    if digest(full)!=record['full_game_json_sha256']:raise ValueError('Full-song GAME hash mismatch')
    audio,sr=sf.read(folder/'vocal.wav',dtype='float32')
    if sr!=22050 or audio.ndim!=1 or not np.isfinite(audio).all() or not .25<=len(audio)/sr<=20+1/sr:
        raise ValueError('Invalid decoded crop format/grid')
    if np.max(abs(audio))>=1 or np.sqrt(np.mean(audio.astype(np.float64)**2))<1e-5:raise ValueError('Clipped/silent crop')
    crop=record['crop'];original_sr=crop['original_sample_rate_hz']
    if abs(crop['offset_seconds']-crop['start_sample']/original_sr)>1e-8:raise ValueError('Crop origin mismatch')
    if abs(len(audio)/sr-(crop['end_sample']-crop['start_sample'])/original_sr)>1/sr+1e-9:raise ValueError('Crop duration mapping mismatch')
    with np.load(folder/'features.npz',allow_pickle=False) as z:features={k:z[k] for k in z.files}
    with np.load(folder/'masks.npz',allow_pickle=False) as z:masks={k:z[k] for k in z.files}
    matrix=features['boundary_features_71'];times=features['times_seconds'];n=len(audio)//441+1
    if matrix.shape!=(n,71) or matrix.dtype!=np.float32 or not np.isfinite(matrix).all():raise ValueError('Invalid 71-feature matrix')
    if times.shape!=(n,) or not np.allclose(times,np.arange(n)*.02,rtol=0,atol=1e-12):raise ValueError('Feature time grid mismatch')
    if not np.all((matrix[:,65]>=0)&(matrix[:,65]<=1)) or not np.all((matrix[:,66]==0)|(matrix[:,66]==1)):
        raise ValueError('Confidence/voicing columns invalid')
    f0=features['f0_hz'];voiced=masks['voiced'];reliable=masks['reliable_pitch_screen']
    if voiced.shape!=(n,) or reliable.shape!=(n,) or voiced.dtype!=bool or reliable.dtype!=bool:raise ValueError('Invalid masks')
    if not np.array_equal(matrix[:,66],voiced.astype(np.float32)) or np.any(reliable&~voiced):raise ValueError('Voicing masks disagree')
    if f0.shape!=(n,) or not np.isfinite(f0[voiced]).all() or np.any(f0[voiced]<=0) or not np.isnan(f0[~voiced]).all():
        raise ValueError('Invalid voiced/unvoiced F0 representation')
    if np.any(matrix[~voiced,64]!=0) or not np.allclose(matrix[voiced,64],1200*np.log2(f0[voiced]/440),rtol=0,atol=.002):
        raise ValueError('Masked pitch feature/units mismatch')
    bounds=json.loads((folder/'boundaries.json').read_text())
    if bounds.get('time_origin')!='crop_start' or bounds.get('pitch_field')!='model_midi_pitch' or bounds.get('pitch_shift_semitones')!=0:
        raise ValueError('Boundary origin or pitch convention mismatch')
    gaussian=np.zeros(n,dtype=np.float32);inside=np.zeros(n,dtype=bool);overlap=np.zeros(n,dtype=int);note_hz=np.zeros(n)
    previous=-1.
    for note in bounds['notes']:
        a,b,p=(float(note[k]) for k in ('onset_sec','offset_sec','model_midi_pitch'))
        if not np.isfinite([a,b,p]).all() or not 0<=a<b<=len(audio)/sr+.05 or a<previous or not 0<=p<=127:
            raise ValueError('Invalid crop-local GAME note')
        previous=a;overlap+=((times>=a)&(times<b)).astype(int)
        mid=(times>=a+.1)&(times<=b-.1);inside|=mid;note_hz[mid]=440*2**((p-69)/12)
        for edge in (a,b):gaussian=np.maximum(gaussian,np.exp(-.5*((times-edge)/.04)**2).astype(np.float32))
    if not np.allclose(matrix[:,70],gaussian,rtol=0,atol=1e-6):raise ValueError('Column 70 is not the declared GAME boundary score')
    delta=np.r_[False,voiced[1:]&voiced[:-1]]
    if not np.array_equal(features['pitch_change_valid'],delta):raise ValueError('Adjacent pitch-valid mask mismatch')
    expected_edge=(times<.1)|(times>len(audio)/sr-.1)
    if not np.array_equal(masks['crop_edge_protected'],expected_edge):raise ValueError('Crop edge mask mismatch')
    for key in ('heuristic_boundary_score','crop_edge_protected'):
        if masks[key].shape!=(n,):raise ValueError('Auxiliary mask grid mismatch')
    heuristic=masks['heuristic_boundary_score']
    if not np.isfinite(heuristic).all() or np.any((heuristic<0)|(heuristic>1)):raise ValueError('Invalid heuristic score')
    screen=record['screen']
    if (screen['frame_count']!=n or screen['voiced_frames']!=int(voiced.sum()) or screen['reliable_frames']!=int(reliable.sum())
            or screen['inside_note_frames']!=int(inside.sum()) or abs(screen['voiced_fraction']-float(voiced.mean()))>1e-10):
        raise ValueError('Serialized screen counts disagree with arrays')
    # Source-only arrays have no detuning target, aligned A/B/C/D views or musical approval.
    model_inputs=dict(times_seconds=times,boundary_features_71=matrix,frame_features_70=matrix[:,:70].copy(),
        confidence=matrix[:,65].copy(),voiced_mask=voiced.copy(),f0_hz=f0.copy(),
        game_boundary_score=matrix[:,70].copy(),heuristic_boundary_score=heuristic.copy())
    return dict(record=record,audio=audio,sample_rate_hz=sr,features=features,masks=masks,boundaries=bounds,
        model_inputs=model_inputs,inside_notes=inside,note_hz=note_hz,overlap=overlap,
        boundary_kind='game_crop_note_edges; acoustic heuristic kept separately',
        supervised_targets_present=False,paired_training_ready=False,research_eligible=False)
