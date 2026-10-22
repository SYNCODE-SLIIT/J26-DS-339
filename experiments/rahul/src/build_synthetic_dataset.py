"""Construct a synthetic chord-refinement baseline from existing extraction files.

No extraction, vocabulary fitting, model training, or production writes occur here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

from experiments.rahul import config
from experiments.rahul.src.normalize_chords import parse_chord_label
from experiments.rahul.src.canonicalize_chords import canonicalize_chord, ROOT_TO_PITCH_CLASS

INPUTS = ('basic_chord basic_root basic_quality basic_root_pitch_class basic_chord_canonical '
          'melody_midi melody_pitch_classes melody_durations melody_relative_start '
          'melody_overlap_durations melody_sources genre_tags key mode key_confidence '
          'roman_numeral harmonic_function beat_index bar_number beat_position '
          'previous_basic_chord next_basic_chord event_duration').split()
TARGETS = ('reference_chord reference_chord_canonical reference_root_pitch_class '
           'reference_quality reference_bass_degree reference_overlap_ratio target_valid').split()
DEFAULT_SOURCE = config.REPOSITORY_ROOT / 'datasets/mtg-jamendo/melody-chord-extraction-pipeline/outputs'
BASIC_QUALITY_POLICY = 'reference_quality_families_v3'


def basic_quality_family(detail):
    """Map the isolated quality; never infer a family from root or bass text."""
    if not detail:
        return 'maj' if detail == '' else None
    if detail in ('m7b5', 'co7', 'hdim', 'hdim7'):
        return 'hdim'
    if detail.startswith('sus'):
        return 'sus'
    if detail.startswith('dim'):
        return 'dim'
    if detail == '+' or detail.startswith('aug'):
        return 'aug'
    if detail.startswith('maj') or detail in ('6', 'add9'):
        return 'maj'
    if detail.startswith('min') or detail in ('m', 'm6', 'm7', 'm9', 'm11', 'm13'):
        return 'min'
    if detail in ('dom', '7', '9', '11', '13', '7b9', '7(b9)'):
        return 'dom'
    return None


def overlap(a, b, c, d):
    return max(0.0, min(b, d) - max(a, c))


def simplify(label):
    p = parse_chord_label(label)
    if label.strip() == 'N':
        return dict(basic_chord='N', basic_root=None, basic_quality='N', simplification_status='supported')
    # Parse the quality separately from root/inversion. Preserve the full reference
    # label as supervision while collapsing extensions to the requested families.
    root, detail = p.root, p.quality
    if not p.parsed:
        # Shorthand is an input adapter, not a change to target parsing/validity.
        match = re.fullmatch(r'([A-G](?:#|b)?)([^/\s]*)(?:/[#b]?\d+)?', label.strip())
        root, detail = (match[1], match[2]) if match else (None, None)
    quality = basic_quality_family(detail) if root else None
    if quality is None:
        return dict(basic_chord=None, basic_root=root, basic_quality=None, simplification_status='unsupported')
    return dict(basic_chord=root + {'maj': '', 'min': 'm', 'dom': '7', 'dim': 'dim',
                                  'hdim': 'm7b5', 'aug': 'aug', 'sus': 'sus'}[quality],
                basic_root=root, basic_quality=quality, simplification_status='supported')


def context(basic, key, mode):
    """Conservative diatonic triads; chromatic/modal functions remain unknown."""
    if basic['basic_chord'] == 'N':
        return 'N', 'unknown'
    scales = {
        'major': [(0,'maj','I','tonic'),(2,'min','ii','predominant'),(4,'min','iii','tonic'),
                  (5,'maj','IV','predominant'),(7,'maj','V','dominant'),(9,'min','vi','tonic'),(11,'dim','vii°','dominant')],
        'minor': [(0,'min','i','tonic'),(2,'dim','ii°','predominant'),(3,'maj','III','unknown'),
                  (5,'min','iv','predominant'),(7,'maj','V','dominant'),(8,'maj','VI','unknown'),(11,'dim','vii°','dominant')],
    }
    if key not in ROOT_TO_PITCH_CLASS or basic['basic_root'] not in ROOT_TO_PITCH_CLASS:
        return 'unknown', 'unknown'
    relative = (ROOT_TO_PITCH_CLASS[basic['basic_root']] - ROOT_TO_PITCH_CLASS[key]) % 12
    if basic['basic_quality'] == 'dom':
        return ('V', 'dominant') if relative == 7 and mode in scales else ('unknown', 'unknown')
    triad_quality = 'dim' if basic['basic_quality'] == 'hdim' else basic['basic_quality']
    for pc, quality, roman, function in scales.get(mode, []):
        if (pc, quality) == (relative, triad_quality):
            return roman, function
    return 'unknown', 'unknown'


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def join_roman(chords, romans):
    def identity(r):
        return (str(int(r['track_id'])), float(r['start_sec']), float(r['end_sec']))
    lookup = {identity(r): r for r in romans}
    if len(lookup) != len(romans) or len(chords) != len(romans):
        raise ValueError('Roman correspondence is not one-to-one')
    result = []
    for chord in chords:
        roman = lookup.pop(identity(chord), None)
        if roman is None or chord['chord_label'] != roman['original_chord_label']:
            raise ValueError('Roman interval/label mismatch')
        result.append({**chord, 'source_roman': roman})
    return result


def melody_features(notes, start, end):
    aligned = []
    for n in notes:
        seconds = overlap(start, end, n['onset_sec'], n['offset_sec'])
        if seconds > 0:
            aligned.append({**n, 'original_duration': n['offset_sec']-n['onset_sec'],
                            'overlap_duration': seconds, 'relative_start': n['onset_sec']-start})
    return dict(melody=aligned, melody_midi=[n['midi_pitch'] for n in aligned],
                melody_pitch_classes=[n['midi_pitch_rounded'] % 12 for n in aligned],
                melody_durations=[n['original_duration'] for n in aligned],
                melody_relative_start=[n['relative_start'] for n in aligned],
                melody_overlap_durations=[n['overlap_duration'] for n in aligned],
                melody_sources=[n['source'] for n in aligned])


def align(start, end, chords):
    candidates = [dict(source_chord_event_id=i, chord=c['chord_label'],
                       overlap_duration=overlap(start,end,c['start_sec'],c['end_sec']))
                  for i,c in enumerate(chords) if overlap(start,end,c['start_sec'],c['end_sec']) > 0]
    candidates.sort(key=lambda c: (-c['overlap_duration'], c['source_chord_event_id']))
    best = candidates[0] if candidates else None
    ratio = best['overlap_duration']/(end-start) if best else 0.0
    ambiguous = len(candidates)>1 and candidates[0]['overlap_duration']-candidates[1]['overlap_duration'] <= .05*(end-start)+1e-10
    return best, candidates, ratio, ambiguous


def write_csv(path, rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: json.dumps(v,ensure_ascii=False,allow_nan=False) if isinstance(v,(list,dict)) else v
                          for k,v in row.items()} for row in rows)
    loaded = read_csv(path)
    for original, restored in zip(rows,loaded):
        for k,v in original.items():
            expected = json.dumps(v,ensure_ascii=False,allow_nan=False) if isinstance(v,(list,dict)) else '' if v is None else str(v)
            if restored[k] != expected:
                raise ValueError(f'Export integrity failure: {k}')
    if len(loaded) != len(rows):
        raise ValueError('Export row count mismatch')


def dump(path, obj):
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')


def build(track_id='382', source_root=DEFAULT_SOURCE, output=None, member2=None,
          chord_endpoint_tolerance_sec=.01):
    if not math.isfinite(chord_endpoint_tolerance_sec) or not 0<=chord_endpoint_tolerance_sec<=1:
        raise ValueError('Chord endpoint tolerance must be within [0,1] seconds')
    track_id = str(int(str(track_id).removeprefix('track_')))
    name = f'track_{int(track_id):07d}'
    source = Path(source_root)/name
    output = Path(output) if output else config.OUTPUT_DIR/'synthetic_dataset'/name
    # Confine writes to this component; production folders are never accepted.
    output = output.resolve()
    if not output.is_relative_to((config.OUTPUT_DIR/'synthetic_dataset').resolve()):
        raise ValueError('Output must be inside configured outputs/synthetic_dataset')
    paths = [source/p for p in ('melody/combined.csv','chords/full.csv','roman/full.csv','beats/beats.csv','beats/beats.json','chords/full.json','key.json')]
    genre_path = config.REPOSITORY_ROOT/'datasets/mtg-jamendo/annotations/autotagging_genre.tsv'
    paths.append(genre_path)
    if member2:
        paths.append(Path(member2))
    # key and genre metadata are optional; mandatory extraction tables fail loudly.
    tracked_paths=[p for p in paths if p.exists() or p not in (paths[6],genre_path)]
    hashes = {p.relative_to(config.REPOSITORY_ROOT).as_posix() if p.is_relative_to(config.REPOSITORY_ROOT) else p.name:
              hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked_paths}
    notes,chords,romans,beats = [read_csv(p) for p in paths[:4]]
    if not all((notes,chords,romans,beats)):
        raise ValueError('Required source table is empty')
    checks = {}
    def check(name, condition):
        checks[name] = bool(condition)
        if not condition:
            raise ValueError(f'Validation failed: {name}')
    check('source_track_ids', all(str(int(r['track_id']))==track_id for table in (notes,chords,romans,beats) for r in table))
    for table, fields in ((notes,('onset_sec','offset_sec','midi_pitch','midi_pitch_rounded')),
                          (chords,('start_sec','end_sec')), (beats,('time_sec',))):
        for r in table:
            for field in fields:
                r[field]=float(r[field])
        check('finite_'+fields[0],all(math.isfinite(r[f]) for r in table for f in fields))
        check('chronological_'+fields[0],all(a[fields[0]]<=b[fields[0]] for a,b in zip(table,table[1:])))
    check('valid_note_pitches',all(0<=n['midi_pitch']<=127 and 0<=n['midi_pitch_rounded']<=127 and n['midi_pitch_rounded'].is_integer() for n in notes))
    for n in notes:
        n['midi_pitch_rounded']=int(n['midi_pitch_rounded'])
    timing=json.loads(paths[4].read_text(encoding='utf-8'))
    chord_doc=json.loads(paths[5].read_text(encoding='utf-8'))
    duration=float(timing['duration_sec'])
    check('validated_duration',math.isfinite(duration) and duration>0 and abs(duration-float(chord_doc['duration_sec']))<1e-6)
    check('metadata_track_ids',all(str(int(d['track_id']))==track_id for d in (timing,chord_doc)))
    # LV-Chordia uses centisecond timestamps (see extraction code/chords.py).
    # Default preserves Track 382 policy. Batch builds may explicitly select the
    # source extractor's one-second endpoint policy; no timestamps are changed.
    check('source_intervals',all(0<=n['onset_sec']<n['offset_sec']<=duration+1e-5 for n in notes) and all(0<=c['start_sec']<c['end_sec']<=duration+chord_endpoint_tolerance_sec+1e-6 for c in chords))
    check('nonoverlapping_reference',all(a['end_sec']<=b['start_sec']+1e-8 for a,b in zip(chords,chords[1:])))
    chords=join_roman(chords,romans)
    check('roman_correspondence',True)
    check('source_chord_fields',all((p:=parse_chord_label(c['chord_label'])).root==(c['root'] or None) and p.quality==(c['quality'] or None) and p.bass_degree==(c['bass_degree'] or None) for c in chords))
    keydoc=json.loads(paths[6].read_text(encoding='utf-8')) if paths[6].exists() else {}
    if keydoc:
        check('key_track_id',str(int(keydoc['track_id']))==track_id)
    melody_key=keydoc.get('melody_key') or {}
    key=melody_key.get('selected') or {}
    if not key and melody_key.get('confidence'):
        key={'confidence':melody_key['confidence']}
    genres=[]
    if genre_path.exists():
        with genre_path.open(encoding='utf-8',newline='') as f:
            for row in csv.reader(f,delimiter='\t'):
                if row[0]==name:
                    genres=[v.removeprefix('genre---') for v in row[5:]]
    independent=read_csv(Path(member2)) if member2 else None
    if independent is not None:
        check('member2_track_and_unique_beats',len(independent)==len(beats) and len({r['beat_index'] for r in independent})==len(beats) and all(str(int(r['track_id']))==track_id for r in independent))
        independent={r['beat_index']:r['basic_chord'] for r in independent}
    intermediate=[]
    for i,c in enumerate(chords):
        intermediate.append({**c,'chord_event_id':i,**simplify(c['chord_label']),**melody_features(notes,c['start_sec'],c['end_sec'])})
    events=[]
    for i,b in enumerate(beats):
        start=b['time_sec']; end=beats[i+1]['time_sec'] if i+1<len(beats) else duration
        if not 0<=start<end<=duration:
            raise ValueError(f'Invalid beat interval {i}: {start}, {end}')
        best,candidates,ratio,ambiguous=align(start,end,chords)
        c=chords[best['source_chord_event_id']] if best else None
        label=c['chord_label'] if c else None
        basic=simplify(label) if label else dict(basic_chord=None,basic_root=None,basic_quality=None,simplification_status='missing_reference')
        if independent is not None:
            value=independent[b['beat_index']]
            # Member 2 adapter accepts the documented basic triad vocabulary only.
            import re
            match=re.fullmatch(r'([A-G](?:#|b)?)(m|dim)?',value)
            raw='N' if value=='N' else f"{match[1]}:{ {'m':'min','dim':'dim',None:'maj'}[match[2]]}" if match else None
            if raw is None: raise ValueError(f'Unsupported Member 2 basic chord: {value}')
            basic=simplify(raw)
        canonical,pc=canonicalize_chord(basic['basic_chord']) if basic['basic_chord'] else (None,None)
        refcan,refpc=canonicalize_chord(label) if label else (None,None)
        roman,function=context(basic,key.get('tonic'),key.get('mode'))
        p=parse_chord_label(label) if label else None
        events.append(dict(track_id=track_id,beat_index=int(b['beat_index']),start_sec=start,end_sec=end,event_duration=end-start,
            bar_number=int(b['bar_index']) if b.get('bar_index') else None,beat_position=int(b['beat_in_bar']) if b.get('beat_in_bar') else None,
            **basic,basic_chord_canonical=canonical,basic_root_pitch_class=pc,**melody_features(notes,start,end),
            genre_tags=genres,key=key.get('tonic'),mode=key.get('mode'),key_confidence=key.get('confidence'),
            roman_numeral=roman,harmonic_function=function,reference_chord=label,reference_chord_canonical=refcan,
            reference_root_pitch_class=refpc,reference_quality=p.quality if p else None,reference_bass_degree=p.bass_degree if p else None,
            reference_overlap_ratio=ratio,reference_candidates=candidates,reference_ambiguous=ambiguous,
            target_valid=bool(label is not None and p.parsed and p.normalized and ratio>=.5 and not ambiguous),
            source_chord_event_id=best['source_chord_event_id'] if best else None,
            basic_source='member2' if independent is not None else 'synthetic_reference_simplification'))
    for i,e in enumerate(events):
        e['previous_basic_chord']=events[i-1]['basic_chord'] if i else 'START'
        e['next_basic_chord']=events[i+1]['basic_chord'] if i+1<len(events) else 'END'
    check('one_event_per_unique_source_beat',len(events)==len(beats)==len({e['beat_index'] for e in events}))
    check('positive_chronological_beat_intervals',all(0<=e['start_sec']<e['end_sec']<=duration for e in events) and all(a['end_sec']==b['start_sec'] for a,b in zip(events,events[1:])))
    check('melody_overlap_and_order',all(e['melody']==melody_features(notes,e['start_sec'],e['end_sec'])['melody'] for e in events+intermediate))
    check('reference_policy',all((lambda a: e['reference_candidates']==a[1] and e['reference_overlap_ratio']==a[2] and e['reference_ambiguous']==a[3])(align(e['start_sec'],e['end_sec'],chords)) for e in events))
    check('validity_policy',all(e['target_valid']==bool(e['reference_chord'] is not None and parse_chord_label(e['reference_chord']).parsed and parse_chord_label(e['reference_chord']).normalized and e['reference_overlap_ratio']>=.5 and not e['reference_ambiguous']) for e in events))
    check('basic_root_preserved',all(r['basic_root']==parse_chord_label(r['chord_label']).root for r in intermediate))
    check('input_target_separation',not set(INPUTS)&set(TARGETS) and not any(x.startswith('reference') or x in ('source_roman','family','basic_label','melody') for x in INPUTS))
    check('context_from_basic_only',all((e['roman_numeral'],e['harmonic_function'])==context(e,e['key'],e['mode']) for e in events))
    check('source_chords_preserved', [r['chord_label'] for r in intermediate]==[r['chord_label'] for r in chords])
    check('no_chord_events_preserved',sum(c['chord_label']=='N' for c in chords)==sum(c['chord_label']=='N' for c in intermediate))
    check('silent_beats_preserved',all(bool(e['melody'])==any(n['onset_sec']<e['end_sec'] and n['offset_sec']>e['start_sec'] for n in notes) for e in events))
    check('melody_overlap_values',all(abs(n['overlap_duration']-(min(n['offset_sec'],e['end_sec'])-max(n['onset_sec'],e['start_sec'])))<1e-9 for e in events+intermediate for n in e['melody']))
    stats=dict(original_melody_notes=len(notes),source_chord_events=len(chords),roman_events=len(romans),detected_beats=len(beats),final_events=len(events),
        valid_targets=sum(e['target_valid'] for e in events),invalid_targets=sum(not e['target_valid'] for e in events),
        ambiguous_beats=sum(e['reference_ambiguous'] for e in events),beats_with_melody=sum(bool(e['melody']) for e in events),silent_beats=sum(not e['melody'] for e in events),
        unique_basic_chords=len({e['basic_chord'] for e in events}),unique_reference_chords=len({e['reference_chord'] for e in events}),
        basic_quality_distribution=dict(Counter(e['basic_quality'] for e in events)),reference_quality_distribution=dict(Counter(e['reference_quality'] or 'N' for e in events)),
        source_quality_distribution=dict(Counter(c['quality'] or 'N' for c in chords)),
        unchanged_source_chords=sum(c['chord_label']=='N' or (c['quality'] in ('maj','min','dim') and not c['bass_degree']) for c in chords),
        dominant_seventh_simplifications=sum(c['quality']=='7' for c in chords),inversion_simplifications=sum(bool(c['bass_degree']) for c in chords),
        enrichment_source_events=sum(c['simplification_status']=='supported' and (c['quality']=='7' or bool(c['bass_degree'])) for c in intermediate),
        enrichment_beat_events=sum(e['reference_quality']=='7' or bool(e['reference_bass_degree']) for e in events),
        unsupported_labels=sorted({c['chord_label'] for c in intermediate if c['simplification_status']=='unsupported'}),
        unmapped_roman_labels=sorted({c['chord_label'] for c in chords if c['source_roman']['mapping_status'] not in ('mapped','no_chord')}),
        no_chord_source_events=sum(c['chord_label']=='N' for c in chords),no_chord_beats=sum(e['reference_chord']=='N' for e in events))
    output.mkdir(parents=True,exist_ok=True)
    write_csv(output/f'{name}_chord_events.csv',intermediate)
    write_csv(output/f'{name}_training_events.csv',events)
    check('export_reload_integrity',True)
    after={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in zip(hashes,tracked_paths)}
    check('source_hashes_unchanged',hashes==after)
    descriptions={
        'track_id':'Numeric track identifier; preserve for track-wise splits and sequence boundaries.',
        'beat_index':'Source beat index (zero based).','start_sec':'Beat start in source seconds.',
        'end_sec':'Next beat timestamp, or validated duration for final beat.','event_duration':'end_sec minus start_sec.',
        'bar_number':'Source bar_index, zero based; not renumbered.','beat_position':'Source beat_in_bar, one based.',
        'melody':'Audit-only ordered note objects, including raw/rounded MIDI, original timing, source, velocity and model pitch.',
        'melody_midi':'Ordered original MIDI pitches; may be fractional.','melody_pitch_classes':'Rounded MIDI modulo 12.',
        'melody_durations':'Original full note durations in seconds.','melody_relative_start':'Original onset minus event start; may be negative.',
        'melody_overlap_durations':'Intersection durations within this beat, in seconds.','melody_sources':'Extraction source per overlapping note.',
        'genre_tags':'MTG genre tags; empty if unavailable.','key':'Estimated melody-key tonic.', 'mode':'Estimated melody-key mode.',
        'key_confidence':'Source categorical confidence; not a calibrated probability.',
        'roman_numeral':'Conservative diatonic Roman derived only from basic chord and estimated key; unknown otherwise.',
        'harmonic_function':'Conventional triad function from basic chord/key; chromatic or uncertain functions are unknown.',
        'basic_chord':'Synthetic basic display label (major/minor/diminished/suspended), or Member 2 basic triad. Null for unsupported simplification.',
        'basic_root':'Basic root spelling.','basic_quality':'maj, min, dom, dim, hdim, aug, sus, N, or null. Reference extensions collapse to quality families; detailed references remain unchanged.',
        'basic_root_pitch_class':'Basic root modulo 12; null for N/unknown.','basic_chord_canonical':'Basic chord with sharp root spelling.',
        'previous_basic_chord':'Previous basic label within track, START at boundary.','next_basic_chord':'Next basic label within track, END at boundary.',
        'reference_chord':'Exact recognized source chord label; pseudo-label, not human ground truth.',
        'reference_chord_canonical':'Reference label with sharp root; suffix preserved.',
        'reference_root_pitch_class':'Recognized root modulo 12, null for N.',
        'reference_quality':'Recognized detailed quality, null for N.','reference_bass_degree':'Recognized inversion/bass degree, null if absent.',
        'reference_overlap_ratio':'Selected source event overlap divided by beat duration.',
        'reference_candidates':'All positive-overlap source event IDs, exact labels and overlap seconds; greatest overlap first, source order breaks ties.',
        'reference_ambiguous':'Top two event overlaps differ by at most 5% of beat duration.',
        'target_valid':'Recognized supported target, overlap >= 0.50, and unambiguous. Use as loss mask.',
        'source_chord_event_id':'Selected source chord row index; audit only.',
        'basic_source':'Synthetic-reference or Member 2 provenance.',
        'simplification_status':'supported, unsupported, or missing_reference; unsupported basics remain null.',
    }
    schema=dict(schema_version=1,input_columns=INPUTS,target_columns=TARGETS,
        columns={k:dict(description=descriptions[k],role='input' if k in INPUTS else 'target' if k in TARGETS else 'audit',
                        type='json_array' if isinstance(v,list) else 'boolean' if isinstance(v,bool) else 'number' if isinstance(v,(int,float)) else 'string_or_null') for k,v in events[0].items()},
        serialization='UTF-8 CSV; arrays/objects are JSON; empty scalar cells are null. Boolean cells are True/False.',
        sequence_policy='Sort by track_id and beat_index. Pad/mask melody arrays. Mask invalid targets. Split by track before fitting vocabularies; no beat-random split.')
    dump(output/'dataset_schema.json',schema)
    report=dict(track_id=track_id,basic_quality_policy=BASIC_QUALITY_POLICY,statistics=stats,checks=checks,all_passed=all(checks.values()),source_sha256=hashes,
                duration_sec=duration,leading_unrepresented_seconds=beats[0]['time_sec'],
                chord_endpoint_tolerance_sec=chord_endpoint_tolerance_sec,chord_endpoint_overrun_sec=max(0,chords[-1]['end_sec']-duration),
                policy=dict(min_overlap=.5,ambiguity_margin=.05,ambiguous_targets_invalid=True),
                notebook_comparison='No Track 382 notebook result was found under experiments; available Pasan notebooks contain no Track 382 results. key.json reports 313 notes in the key estimator, not the combined melody CSV population.',
                missing_metadata=[field for field,value in [('key',key.get('tonic')),('mode',key.get('mode')),('genre_tags',genres)] if not value],
                key_metadata_note=f"key estimator note_count={melody_key.get('note_count')}; combined.csv note count={len(notes)}; no melody substitution")
    dump(output/f'{name}_validation_report.json',report)
    summary='''# Synthetic chord-refinement baseline

Reference labels are automatically recognized pseudo-labels. Synthetic basics use distinct maj/min/dom/dim/hdim/aug/sus families, removing extensions and inversions within each family. Dominant 7/9/13/7b9 map to dom; hdim7/m7b5/co7 map to hdim; aug/+ map to aug. Both sus2 and sus4 use generic sus. Detailed reference labels remain unchanged. Unsupported qualities remain null and are reported. This relationship is a baseline limitation and does not establish independent inputs. No Transformer, splits, vocabularies, or fitted transformations are created.

The existing Rahul parser and root canonicalizer are reused. The older builder's largest-overlap rule is retained, but its extraction and independent harmony generation are intentionally not invoked. It has no ambiguity/validity threshold policy, so this builder applies the requested 0.50/0.05 defaults. One row begins at each source beat; no extra intro beat is invented. Final duration is cross-checked between beats.json and chords/full.json. Bar indices preserve the source's zero-based numbering. Crossing melody notes retain original onset (possibly negative relative onset), duration, overlap and extraction source. Empty arrays preserve silence. No-chord targets remain eligible under the same validity policy.

Chord events join Roman rows by numeric track ID and exact interval, requiring matching original labels and one-to-one correspondence. Source Roman fields are audit-only in the intermediate table. Final Roman/function inputs use basic triads and estimated key only. Chromatic or unsupported contexts are unknown. Key confidence preserves its categorical source value. The key estimator may have analyzed a different melody subset; it is not a replacement for combined.csv.

Every final column and the explicit input/target lists are described in dataset_schema.json. Melody audit objects are excluded from inputs. Target labels, qualities, inversions, candidate overlaps and source IDs must not enter model sequences. Previous/next context uses basics within the same track. Null unsupported basics require an unknown token or input mask later. Invalid targets require loss masks. Split by track before learning class vocabularies; preserve boundaries and pad/mask melody arrays. More independently predicted basic chords, manually assessed labels, diverse tracks and key/beat quality evaluation remain necessary before research-quality training.

Run from repository root: `python -m experiments.rahul.src.build_synthetic_dataset --track-id 382`. Repeat `--track-id` for additional tracks. Outputs are confined to Rahul's configured outputs/synthetic_dataset/<track>. An optional `--member2-csv` accepts track_id, beat_index, basic_chord (C, Am, Bdim, N vocabulary), exactly one prediction per source beat; source targets and validity remain unchanged. This adapter supports replacing synthetic basics without changing extraction. All input files are SHA-256 checked before/after. Existing production outputs are untouched.

Statistics below use beat events except explicitly named source-event counts; inversion and seventh source counts can overlap. Meaningful enrichment means a supported seventh or inversion beyond its basic triad, not a claim of musical correctness.

'''
    summary+=report['notebook_comparison']+'\n\n'+report['key_metadata_note']+'\n\n'
    summary+=f"The source final chord ends at {chords[-1]['end_sec']} s versus metadata duration {duration} s. The extractor documents centisecond timestamps; an explicit {chord_endpoint_tolerance_sec:g} s endpoint tolerance is allowed without changing source intervals. Beats end at metadata duration.\n\n"
    summary+='| Statistic | Value |\n|---|---|\n'+''.join(f'| {k} | {json.dumps(v,ensure_ascii=False)} |\n' for k,v in stats.items())
    summary+=f"\nValidation: {len(checks)} checks passed, including CSV reload and source hashes. Source duration: {duration}; intro before first beat: {beats[0]['time_sec']} seconds.\n"
    (output/'dataset_build_summary.md').write_text(summary,encoding='utf-8')
    print(json.dumps(report,indent=2,ensure_ascii=False))
    return events,report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--track-id',action='append')
    parser.add_argument('--source-root',type=Path,default=DEFAULT_SOURCE)
    parser.add_argument('--member2-csv',type=Path)
    args=parser.parse_args()
    if args.member2_csv and len(args.track_id or ['382'])!=1:
        parser.error('--member2-csv requires a single track')
    for track in args.track_id or ['382']:
        build(track,args.source_root,member2=args.member2_csv)


if __name__=='__main__':
    main()
