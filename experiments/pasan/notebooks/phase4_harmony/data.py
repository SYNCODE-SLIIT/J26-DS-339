"""Conservative weak-label preparation from absolute chords, never old Roman labels."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
import gzip
import json
import platform
import re
import shutil
import tempfile

import numpy as np
import pandas as pd
from phase2_harmony.data import CleanConfig, clean_notes, complete_windows, interval_stats

LABELS = ['TONIC', 'PREDOMINANT', 'DOMINANT']
IGNORE = -100
FEATURE_DIM = 77
FILES = {'melody': 'melody/combined.json', 'chords': 'chords/submission.json',
         'beats': 'beats/beats.json', 'key': 'key.json', 'comparison': 'key_comparison.json'}
NATURAL = dict(C=0, D=2, E=4, F=5, G=7, A=9, B=11)
# Deliberately finite vocabulary: don't guess for altered/omitted thirds or fifths.
QUALITY = {'maj': 'MAJ', 'maj7': 'MAJ', 'maj9': 'MAJ', 'min': 'MIN',
           'min7': 'MIN', 'min9': 'MIN', '7': 'DOM', '9': 'DOM',
           'dim': 'DIM', 'dim7': 'DIM', 'hdim7': 'DIM'}
# Core functions only. iii/vi/III/VI, natural-minor v/bVII, secondary dominants,
# suspensions and modal mixture need context and remain unknown, not tonic.
CORE = {'major': {(0, 'MAJ'): 0, (2, 'MIN'): 1, (5, 'MAJ'): 1,
                  (7, 'MAJ'): 2, (7, 'DOM'): 2, (11, 'DIM'): 2},
        'minor': {(0, 'MIN'): 0, (2, 'DIM'): 1, (5, 'MIN'): 1,
                  (7, 'MAJ'): 2, (7, 'DOM'): 2, (11, 'DIM'): 2}}
MAJOR = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
MINOR = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])


@dataclass(frozen=True)
class Config:
    bars: int = 8
    beat_dominance: float = .80
    bar_dominance: float = .70
    min_function_fraction: float = .50
    min_melody_coverage: float = .50
    min_notes: int = 8
    max_gap_beats: float = 12.
    min_genre_tracks: int = 3
    key_confidence: str = 'high'
    policy: str = 'core'


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def code_hash():
    paths = [Path(__file__), Path(__file__).parent.parent/'phase2_harmony/data.py']
    return sha256(b''.join(p.read_bytes() for p in paths)).hexdigest()


def root_pc(name):
    if not isinstance(name, str):
        return None
    m = re.fullmatch(r'([A-G])([#b]*)', name.replace('♯', '#').replace('♭', 'b'))
    return (NATURAL[m[1]] + m[2].count('#') - m[2].count('b')) % 12 if m else None


def function_id(root, quality, tonic, mode, policy='core'):
    pc = root_pc(root)
    if pc is None or quality not in QUALITY or mode not in CORE:
        return IGNORE
    relative, family = (pc-tonic) % 12, QUALITY[quality]
    if policy == 'substitute' and mode == 'major' and (relative, family) == (9, 'MIN'):
        return 0  # Explicit sensitivity assumption: vi as tonic substitute.
    return CORE[mode].get((relative, family), IGNORE)


def prepare_chords(obj, tonic, mode, policy):
    """Validate all intervals, including N; no sorting/merging across gaps."""
    result, previous = [], -np.inf
    duration = float(obj['duration_sec'])
    for e in obj['events']:
        a,b = float(e['start_sec']),float(e['end_sec'])
        if not np.isfinite([a,b]).all() or a < 0 or b <= a or b > duration+.1 or a < previous-1e-5:
            raise ValueError('invalid_or_overlapping_chords')
        previous = b
        label = e['chord_label']
        if label != 'N':
            m = re.fullmatch(r'([^:]+):([^/]+)(?:/(.+))?', label)
            if not m or root_pc(m[1]) != root_pc(e['root']) or m[2] != e['quality']:
                raise ValueError('chord_label_fields_disagree')
        y = IGNORE if label == 'N' else function_id(e['root'],e['quality'],tonic,mode,policy)
        result.append({'start_sec':a,'end_sec':b,'chord_label':label,'function':y,
                       'reason':'no_chord' if label=='N' else ('mapped' if y>=0 else 'context_or_quality_unknown')})
    return result


def targets(events, bounds, dominance):
    """Sum duration by FUNCTION, with full cell duration as denominator."""
    ys,confidence = [],[]
    for a,b in zip(bounds[:-1],bounds[1:]):
        mass = np.zeros(3)
        for e in events:
            if e['function'] >= 0:
                mass[e['function']] += max(0., min(b,e['end_sec'])-max(a,e['start_sec']))
        best = int(mass.argmax())
        certainty = float(mass[best]/(b-a))
        ys.append(best if certainty >= dominance-1e-8 else IGNORE)
        confidence.append(certainty)
    return np.array(ys,np.int64),np.array(confidence,np.float32)


def clean_key_stable(notes, selected):
    """Recompute the pipeline's 24-key ranking after cleaning, including endings.

    This is an instability screen, not another independent key estimator.
    """
    histogram=np.zeros(12)
    for n in notes:
        histogram[n['pitch']%12] += min(2.,n['offset_sec']-n['onset_sec'])
    if np.count_nonzero(histogram)<3:
        return False
    gaps=[b['onset_sec']-a['offset_sec'] for a,b in zip(notes[:-1],notes[1:])]
    positives=[g for g in gaps if g>0]
    cutoff=max(.6,2*float(np.median(positives))) if positives else .6
    endings=[n['pitch']%12 for n,g in zip(notes,gaps) if g>=cutoff]+[notes[-1]['pitch']%12]
    ranked=[]
    for pc in range(12):
        for mode,profile in [('major',MAJOR),('minor',MINOR)]:
            shifted=np.roll(histogram,-pc)
            score=float(np.corrcoef(shifted,profile)[0,1])+.08*endings.count(pc)/len(endings)
            ranked.append((score,pc,mode))
    best=max(ranked,key=lambda r:r[0])
    return (best[1],best[2]) == (selected['tonic_pc'],selected['mode'])


def features(notes,bounds,tonic,beats):
    """Four ordered quarter-beat slots + onsets + meter, no chord features.

    Keep carry-in sustains, without fabricating an onset at the window edge.
    Relative PC preserves tonic and minor mode is a separate model input.
    """
    count=len(bounds)-1
    x=np.zeros((count,FEATURE_DIM),np.float32)
    selected=[n for n in notes if n['onset_sec']<bounds[-1] and n['offset_sec']>bounds[0]]
    anchor=selected[0]['pitch'] if selected else 0
    for i in range(count):
        a,b=bounds[i:i+2]
        for j in range(4):
            left,right=a+(b-a)*j/4,a+(b-a)*(j+1)/4
            mass=contour=0.;spans=[];onsets=0
            for n in selected:
                overlap=max(0.,min(right,n['offset_sec'])-max(left,n['onset_sec']))
                if overlap:
                    pc=(n['pitch']-tonic)%12
                    x[i,15*j+pc]+=overlap;mass+=overlap
                    contour+=overlap*np.clip((n['pitch']-anchor)/24.,-2,2)
                    spans.append((n['onset_sec'],n['offset_sec']))
                onsets+=left<=n['onset_sec']<right
            x[i,15*j:15*j+12]/=max(mass,right-left)
            x[i,15*j+12:15*j+15]=[interval_stats(spans,left,right)[0],contour/max(mass,1e-8),min(onsets/4.,1)]
        for n in selected:
            if a<=n['onset_sec']<b:
                x[i,60+(n['pitch']-tonic)%12]+=1/4
        x[i,60:72]=np.minimum(x[i,60:72],1)
        beat=beats[i];phase=2*np.pi*(beat['beat_in_bar']-1)/beat['meter_numerator']
        x[i,72:]=[np.sin(phase),np.cos(phase),float(beat['beat_in_bar']==1),beat['meter_numerator']/12.,beat['meter_denominator']/8.]
    return x


def normalized_fingerprint(notes):
    """Exact normalized note sequence with relative times; no chord dependence."""
    if not notes:
        return ''
    a,p=notes[0]['onset_sec'],notes[0]['pitch']
    seq=[(round(n['onset_sec']-a,3),round(n['offset_sec']-n['onset_sec'],3),n['pitch']-p) for n in notes]
    return sha256(json.dumps(seq).encode()).hexdigest()


def build_snapshot(project_root,destination,config=Config()):
    root,destination=Path(project_root),Path(destination)
    if config.policy not in {'core','substitute'} or config.key_confidence not in {'high','medium'}:
        raise ValueError('Unknown labeling policy or confidence threshold')
    if destination.exists():
        meta=verify_snapshot(destination,root)
        if meta['config']!=asdict(config) or meta['builder_sha256']!=code_hash():
            raise ValueError('Changed build policy/code: choose a new snapshot directory')
        return meta
    pipeline=root/'datasets/mtg-jamendo/melody-chord-extraction-pipeline'
    manifest=pipeline/'manifest.csv'
    table=pd.read_csv(manifest,dtype=str).fillna('')
    if not table.mtg_track_id.is_unique:
        raise ValueError('duplicate_manifest_track')
    if table.groupby('artist_id').split_0.nunique().max()!=1:
        raise ValueError('official_split_artist_leakage')
    metadata=table.set_index('mtg_track_id').to_dict('index')
    hashes={'manifest.csv':digest(manifest)};inventory=[];audits=[];windows=[];examples=[];records=[]
    track_fingerprints=defaultdict(list)
    for folder in sorted((pipeline/'outputs').glob('track_*')):
        if not folder.is_dir():continue
        tid=folder.name;inventory.append(tid)
        audit={'track_id':tid,'accepted_track':False,'reason':''};audits.append(audit)
        before_examples,before_records,before_windows=len(examples),len(records),len(windows)
        try:
            obj={}
            for key,rel in FILES.items():
                file=folder/rel;name=f'outputs/{tid}/{rel}'
                hashes[name]=digest(file) if file.exists() else None
                if not file.exists():raise ValueError('missing_'+key)
                obj[key]=read(file)
            info=metadata.get(tid)
            if not info or not info['artist_id'] or info['split_0'] not in {'train','validation','test'}:
                raise ValueError('missing_identity_or_split')
            audit.update(artist_id=info['artist_id'],split=info['split_0'])
            duration=float(obj['melody']['duration_sec'])
            if not np.isfinite(duration) or duration<=0:raise ValueError('invalid_duration')
            for key in ['melody','chords','beats','key','comparison']:
                if obj[key].get('time_origin','full_song_start')!='full_song_start':
                    raise ValueError('different_time_origin')
                if obj[key].get('mtg_track_id')!=tid:raise ValueError('different_track_identity')
                if obj[key].get('source_fingerprint')!=obj['melody'].get('source_fingerprint'):
                    raise ValueError('different_audio_fingerprint')
                if abs(float(obj[key].get('duration_sec',duration))-duration)>.1:
                    raise ValueError('different_durations')
            if (obj['key'].get('melody_json_sha256')!=hashes[f'outputs/{tid}/melody/combined.json'] or
                obj['comparison'].get('melody_key_json_sha256')!=hashes[f'outputs/{tid}/key.json'] or
                obj['comparison'].get('submission_chord_json_sha256')!=hashes[f'outputs/{tid}/chords/submission.json']):
                raise ValueError('stale_key_derivative')
            selected=obj['key']['melody_key'].get('selected') or {}
            other=obj['comparison']['chord_key'].get('selected') or {}
            audit.update(mode=selected.get('mode'),tonic_pc=selected.get('tonic_pc'),
                         melody_key_confidence=selected.get('confidence'),chord_key_confidence=other.get('confidence'))
            confidences={'high'} if config.key_confidence=='high' else {'high','medium'}
            if (selected.get('confidence') not in confidences or other.get('confidence') not in confidences or
                selected.get('mode') not in CORE or selected.get('tonic_pc') not in range(12) or
                (selected.get('tonic_pc'),selected.get('mode'))!=(other.get('tonic_pc'),other.get('mode'))):
                raise ValueError('key_uncertain_or_disagree')
            notes,actions=clean_notes(obj['melody']['notes'],duration,CleanConfig())
            audit['note_actions']=json.dumps(dict(Counter(a['action'] for a in actions)),sort_keys=True)
            if not clean_key_stable(notes,selected):raise ValueError('key_changes_after_cleaning')
            # Overlapping DIFFERENT pitches are ambiguous voice selection: reject, don't choose using chords.
            max_end=-1.
            for n in notes:
                if n['onset_sec']<max_end-1e-4:raise ValueError('polyphonic_melody_overlap')
                max_end=max(max_end,n['offset_sec'])
            beats=obj['beats']['beats'];times=np.array([b['time_sec'] for b in beats],float)
            if len(times)<2 or not np.isfinite(times).all() or np.any(np.diff(times)<=0) or times[0]<0 or times[-1]>duration:
                raise ValueError('invalid_beat_timeline')
            if obj['beats'].get('downbeat_lab_matches_beat_positions') is False:
                raise ValueError('downbeat_mismatch')
            events=prepare_chords(obj['chords'],selected['tonic_pc'],selected['mode'],config.policy)
            genres=sorted(t.removeprefix('genre---') for t in info['tags'].split(';') if t.startswith('genre---'))
            track_fingerprints[normalized_fingerprint(notes)].append(tid)
            audit['accepted_track']=True
            for idx,bounds in complete_windows(beats,CleanConfig(window_bars=config.bars)):
                entries=[beats[i] for i in idx];meter=entries[0]['meter_numerator']
                # Verify true next DOWNBEAT; no extrapolation for the final beat.
                after=beats[idx[-1]+1]
                if after['beat_in_bar']!=1 or after['bar_index']!=entries[-1]['bar_index']+1:
                    continue
                ns=[n for n in notes if n['onset_sec']<bounds[-1] and n['offset_sec']>bounds[0]]
                coverage,_=interval_stats([(n['onset_sec'],n['offset_sec']) for n in ns],bounds[0],bounds[-1])
                spans=[np.interp([n['onset_sec'],n['offset_sec']],bounds,np.arange(len(bounds))) for n in ns]
                _,gap=interval_stats(spans,0,len(idx))
                y,conf=targets(events,bounds,config.beat_dominance)
                bar_bounds=bounds[np.arange(0,len(bounds),meter)]
                by,bc=targets(events,bar_bounds,config.bar_dominance)
                row={'track_id':tid,'artist_id':info['artist_id'],'split':info['split_0'],'mode':selected['mode'],
                     'tonic_pc':selected['tonic_pc'],'start_sec':float(bounds[0]),'end_sec':float(bounds[-1]),
                     'melody_coverage':coverage,'max_gap_beats':gap,'note_count':len(ns),'genres':genres,
                     'function_fraction':float(np.mean(y>=0)),'bar_function_fraction':float(np.mean(by>=0)),
                     'tonic_fraction':float(np.mean(y[y>=0]==0)) if (y>=0).any() else 0.,
                     'function_changes':int(np.sum((y[:-1]>=0)&(y[1:]>=0)&(y[:-1]!=y[1:]))),
                     'n_beats':len(idx),'meter_numerator':meter,'meter_denominator':entries[0]['meter_denominator']}
                reasons=[]
                if coverage<config.min_melody_coverage:reasons.append('sparse_melody')
                if gap>config.max_gap_beats:reasons.append('melody_gap')
                if len(ns)<config.min_notes:reasons.append('few_notes')
                if row['function_fraction']<config.min_function_fraction:reasons.append('few_core_functions')
                row.update(accepted=not reasons,reason=','.join(reasons));windows.append(row)
                if reasons:continue
                x=features(notes,bounds,selected['tonic_pc'],entries)
                row['melody_sha256']=sha256(np.round(x,4).tobytes()).hexdigest()
                examples.append((x,y,conf,by,bc,row))
                records.append({'track_id':tid,'start_sec':row['start_sec'],'key':selected,'split':row['split'],
                                'beat_boundaries_sec':bounds.tolist(),'bar_boundaries_sec':bar_bounds.tolist(),
                                'normalized_notes':[{'onset_sec':max(n['onset_sec'],bounds[0])-bounds[0],
                                    'offset_sec':min(n['offset_sec'],bounds[-1])-bounds[0],
                                    'relative_midi':n['pitch']-selected['tonic_pc'],'source':n['source']} for n in ns],
                                'absolute_chords':[{**e,'start_sec':max(e['start_sec'],bounds[0]),'end_sec':min(e['end_sec'],bounds[-1])}
                                                   for e in events if e['start_sec']<bounds[-1] and e['end_sec']>bounds[0]],
                                'beat_targets':y.tolist(),'bar_targets':by.tolist()})
        except (OSError,ValueError,TypeError,KeyError) as exc:
            # A failed track cannot leave partially admitted windows behind.
            del examples[before_examples:];del records[before_records:];del windows[before_windows:]
            audit.update(accepted_track=False,reason=str(exc))
    # Conservative exact duplicate quarantine across official splits, using inputs only.
    split_lookup={r['track_id']:r.get('split') for r in audits}
    duplicate_tracks={t for group in track_fingerprints.values() if len({split_lookup[t] for t in group})>1 for t in group}
    fingerprints=defaultdict(set)
    for *_,w in examples:fingerprints[w['melody_sha256']].add(w['split'])
    keep=[];kept_records=[]
    for ex,record in zip(examples,records):
        w=ex[-1]
        if w['track_id'] in duplicate_tracks or len(fingerprints[w['melody_sha256']])>1:
            w.update(accepted=False,reason='cross_split_exact_duplicate')
        else:
            w['example_index']=len(keep);keep.append(ex);kept_records.append(record)
    examples=keep
    if not examples:raise ValueError(f'Empty snapshot: {Counter(a["reason"] for a in audits)}')
    accepted=pd.DataFrame([e[-1] for e in examples])
    if set(accepted.split)!={'train','validation','test'}:raise ValueError('empty_split')
    genre_counts=Counter(g for gs in accepted[accepted.split=='train'].drop_duplicates('track_id').genres for g in gs)
    vocab=['<unknown>']+sorted(g for g,c in genre_counts.items() if c>=config.min_genre_tracks)
    n=len(examples);max_beats=max(len(e[1]) for e in examples)
    arrays={'x':np.zeros((n,max_beats,FEATURE_DIM),np.float32),'y_beat':np.full((n,max_beats),IGNORE,np.int64),
            'confidence_beat':np.zeros((n,max_beats),np.float32),'y_bar':np.full((n,config.bars),IGNORE,np.int64),
            'confidence_bar':np.zeros((n,config.bars),np.float32),'lengths':np.zeros(n,np.int64),
            'bar_index':np.full((n,max_beats),-1,np.int64),'mode':np.zeros(n,np.int64),
            'genres':np.zeros((n,len(vocab)),np.float32)}
    for i,(x,y,conf,by,bc,w) in enumerate(examples):
        length=len(y)
        arrays['x'][i,:length]=x;arrays['y_beat'][i,:length]=y;arrays['confidence_beat'][i,:length]=conf
        arrays['y_bar'][i]=by;arrays['confidence_bar'][i]=bc;arrays['lengths'][i]=length
        arrays['mode'][i]=int(w['mode']=='minor');arrays['bar_index'][i,:length]=np.arange(length)//w['meter_numerator']
        ids=[vocab.index(g) for g in w['genres'] if g in vocab];arrays['genres'][i,ids or [0]]=1
    summary={'source_folders':len(inventory),'key_eligible_tracks':sum(a['accepted_track'] for a in audits),
             'accepted_tracks':int(accepted.track_id.nunique()),'windows':n,'splits':accepted.split.value_counts().to_dict(),
             'track_rejections':dict(Counter(a['reason'] for a in audits if a['reason'])),
             'window_rejections':dict(Counter(w['reason'] for w in windows if w['reason']))}
    destination.parent.mkdir(parents=True,exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix=destination.name+'.partial-',dir=destination.parent))
    try:
        np.savez_compressed(staging/'features.npz',**arrays)
        accepted.to_parquet(staging/'examples.parquet',index=False)
        pd.DataFrame(windows).to_parquet(staging/'window_audit.parquet',index=False)
        pd.DataFrame(audits).to_csv(staging/'track_audit.csv',index=False)
        with gzip.open(staging/'records.jsonl.gz','wt') as out:
            for record in kept_records:out.write(json.dumps(record,allow_nan=False)+'\n')
        meta={'schema_version':1,'config':asdict(config),'labels':LABELS,'mapping':{mode:{f'{r}:{q}':LABELS[y] for (r,q),y in rules.items()} for mode,rules in CORE.items()},
              'label_status':'weak labels from automatic chord/key estimates; not human verified',
              'genre_vocab':vocab,'feature_dim':FEATURE_DIM,'summary':summary,'builder_sha256':code_hash(),
              'source_sha256':hashes,'source_inventory':inventory,'python':platform.python_version(),
              'files_sha256':{p.name:digest(p) for p in staging.iterdir()}}
        (staging/'metadata.json').write_text(json.dumps(meta,indent=2,allow_nan=False))
        verify_snapshot(staging)
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging);raise
    return meta


def verify_snapshot(destination,project_root=None):
    destination=Path(destination);meta=read(destination/'metadata.json')
    if meta['schema_version']!=1 or meta['labels']!=LABELS:raise ValueError('incompatible_schema')
    for name,value in meta['files_sha256'].items():
        if digest(destination/name)!=value:raise ValueError('snapshot_hash_mismatch: '+name)
    if project_root is not None:
        pipeline=Path(project_root)/'datasets/mtg-jamendo/melody-chord-extraction-pipeline'
        inventory=sorted(p.name for p in (pipeline/'outputs').glob('track_*') if p.is_dir())
        if inventory!=meta['source_inventory']:raise ValueError('source_inventory_changed')
        for name,value in meta['source_sha256'].items():
            p=pipeline/name
            if (digest(p) if p.exists() else None)!=value:raise ValueError('source_changed: '+name)
    return meta


def load_snapshot(destination):
    destination=Path(destination);meta=verify_snapshot(destination)
    with np.load(destination/'features.npz',allow_pickle=False) as archive:
        arrays={k:archive[k] for k in archive.files}
    examples=pd.read_parquet(destination/'examples.parquet')
    n,t,f=arrays['x'].shape
    if len(examples)!=n or f!=FEATURE_DIM or not np.isfinite(arrays['x']).all():raise ValueError('invalid_features')
    if examples.groupby('artist_id').split.nunique().max()!=1:raise ValueError('artist_leakage')
    if examples.groupby('track_id').split.nunique().max()!=1:raise ValueError('track_leakage')
    for key in ['y_beat','y_bar']:
        y=arrays[key]
        if not ((y==IGNORE)|((y>=0)&(y<3))).all():raise ValueError('invalid_target')
    for i,length in enumerate(arrays['lengths']):
        if not 0<length<=t or (arrays['y_beat'][i,length:]!=IGNORE).any():raise ValueError('invalid_padding')
        if not (arrays['bar_index'][i,:length]>=0).all():raise ValueError('invalid_bar_index')
    return arrays,examples,meta
