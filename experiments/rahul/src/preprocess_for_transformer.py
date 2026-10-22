"""Phase 2 numerical preprocessing only: fit, mask, sequence, validate, export."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.rahul import config
from experiments.rahul.src.build_synthetic_dataset import dump
from experiments.rahul.src.create_sequences import split_tracks, validate_splits
from experiments.rahul.src.encode_features import (
    CATEGORICAL, FORBIDDEN_INPUTS, IGNORE_INDEX, USED_INPUTS,
    FeatureEncoder, load_events,
)
from experiments.rahul.src.transformer_dataset import (
    ChordSequenceDataset, SavedSequenceDataset, export_npz, make_dataloader,
)

DEFAULT_INPUT = config.OUTPUT_DIR/'synthetic_dataset'
DEFAULT_OUTPUT = config.COMPONENT_DIR/'outputs/model_preprocessing'


def portable(path):
    path = Path(path).resolve()
    return path.relative_to(config.REPOSITORY_ROOT).as_posix() if path.is_relative_to(config.REPOSITORY_ROOT) else str(path)


def source_identities(track_ids):
    """Use known artist/album and recording path identities, never infer from labels."""
    result = {t:[] for t in track_ids}
    annotations = config.REPOSITORY_ROOT/'datasets/mtg-jamendo/annotations/autotagging_genre.tsv'
    observed = []
    if annotations.exists():
        observed.append(annotations)
        with annotations.open(encoding='utf-8',newline='') as handle:
            for row in csv.reader(handle,delimiter='\t'):
                if row and row[0].startswith('track_'):
                    track = str(int(row[0].removeprefix('track_')))
                    if track in result:
                        result[track] += [f'artist:{row[1]}',f'album:{row[2]}']
    base = config.REPOSITORY_ROOT/'datasets/mtg-jamendo/melody-chord-extraction-pipeline/outputs'
    for track in track_ids:
        key = base/f'track_{int(track):07d}'/'key.json'
        if key.exists():
            observed.append(key)
            data=json.loads(key.read_text(encoding='utf-8'))
            if data.get('source_mp3'):
                result[track].append('recording:'+data['source_mp3'].replace('\\','/'))
    return result,observed


def digest(paths):
    return {portable(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))}


def run(track_ids, input_root=DEFAULT_INPUT, output_dir=DEFAULT_OUTPUT, sequence_length=8,
        stride=None, development_only=False, seed=42, ratios=(.8,.1,.1), max_notes=None,
        truncate_notes=False, batch_size=4, split_manifest=None):
    track_ids = [str(int(str(t).removeprefix('track_'))) for t in track_ids]
    if len(set(track_ids)) != len(track_ids):
        raise ValueError('Duplicate requested track IDs')
    if batch_size < 1:
        raise ValueError('Batch size must be positive')
    output_dir = Path(output_dir).resolve()
    if not output_dir.is_relative_to(DEFAULT_OUTPUT.resolve()):
        raise ValueError('Keep Phase 2 output under experiments/rahul/outputs/model_preprocessing')
    sources = []
    for track in track_ids:
        folder=Path(input_root)/f'track_{int(track):07d}'
        sources.extend(p for p in folder.iterdir() if p.is_file())
    # Record all Phase 1 exports plus the reused preprocessing implementations.
    sources.extend(config.COMPONENT_DIR/'src'/name for name in (
        'build_synthetic_dataset.py','canonicalize_chords.py','normalize_chords.py'))
    identities,metadata_sources = source_identities(track_ids)
    sources.extend(metadata_sources)
    if split_manifest:
        sources.append(Path(split_manifest))
    before = digest(sources)
    tracks,schemas = {},{}
    for track in track_ids:
        file=Path(input_root)/f'track_{int(track):07d}'/f'track_{int(track):07d}_training_events.csv'
        rows,schema=load_events(file)
        if {r['track_id'] for r in rows} != {track}:
            raise ValueError(f'Requested track {track} does not match CSV')
        if len({tuple(sorted(r['genre_tags'])) for r in rows}) != 1:
            raise ValueError('Genre tags must be invariant within a track')
        tracks[track],schemas[track] = rows,schema
    if split_manifest:
        manifest=json.loads(Path(split_manifest).read_text(encoding='utf-8'))
        validate_splits(manifest,track_ids)
        if development_only != manifest.get('development_only',False):
            raise ValueError('Development flag differs from supplied split manifest')
        # Recheck current identities, even if a supplied manifest omitted them.
        groups=split_tracks(track_ids,development_only=True,identities=identities)['identity_groups']
        manifest['identity_groups']=groups
        manifest['identities']=identities
        validate_splits(manifest,track_ids)
    else:
        manifest=split_tracks(track_ids,ratios,seed,identities,development_only)
    encoder=FeatureEncoder.fit(tracks,manifest['splits']['train'],max_notes,truncate_notes)
    statistics = {'by_track':{t:encoder.statistics(rows) for t,rows in tracks.items()},
                  'by_split':{s:encoder.statistics([r for t in ids for r in tracks[t]]) for s,ids in manifest['splits'].items()},
                  'unique_basic_chords':len({r['basic_chord_canonical'] for rows in tracks.values() for r in rows}),
                  'unique_reference_chords':len({r['reference_chord_canonical'] for rows in tracks.values() for r in rows}),
                  'target_musical_classes':len(encoder.targets.tokens)-2,
                  'target_vocabulary_size_including_PAD_UNK':len(encoder.targets.tokens),
                  'genre_dimensions':len(encoder.genres)}
    # Transform every beat before creating output files; fail if held-out notes exceed policy.
    for rows in tracks.values():
        for row in rows:
            encoder.encode_inputs(row['input_snapshot'])
    datasets={split:ChordSequenceDataset({t:tracks[t] for t in ids},encoder,sequence_length,stride)
              for split,ids in manifest['splits'].items() if ids}
    checks={}
    def check(name,value):
        checks[name]=bool(value)
        if not value:
            raise ValueError(f'Phase 2 validation failed: {name}')
    check('input_columns_schema_verified',all(USED_INPUTS <= set(s['input_columns']) for s in schemas.values()))
    check('target_fields_excluded_from_inputs',not USED_INPUTS & FORBIDDEN_INPUTS)
    check('train_only_encoder_fit',set(encoder.training_tracks)==set(manifest['splits']['train']))
    check('no_target_audit_dependency_in_inputs',all(
        all(np.array_equal(value,encoder.encode_inputs({**row['input_snapshot'],
            'reference_chord':'mutated audit label','reference_quality':'mutated','target_valid':False,
            'reference_overlap_ratio':-1})[key]) for key,value in encoder.encode_inputs(row['input_snapshot']).items())
        for rows in tracks.values() for row in rows))
    validate_splits(manifest,track_ids)
    check('track_and_identity_split_isolation',True)
    batches={}
    sequence_manifest=[]
    for split,dataset in datasets.items():
        covered={t:set() for t in manifest['splits'][split]}
        masked_positions=0
        padded_positions=0
        for i in range(len(dataset)):
            item=dataset[i]
            inputs,targets,meta=item['inputs'],item['targets'],item['metadata']
            beat_mask=inputs['beat_attention_mask']
            ids=meta['beat_indices'][beat_mask].tolist()
            if ids != sorted(ids) or len(ids)!=len(set(ids)):
                raise ValueError('Sequence beat ordering failed')
            covered[meta['track_id']].update(ids)
            if torch.any(targets['target_loss_mask'] & ~beat_mask):
                raise ValueError('Padded targets must be masked')
            if not torch.equal(targets['target_chord_ids']==IGNORE_INDEX,~targets['target_loss_mask']):
                raise ValueError('Ignore-index and target masks differ')
            if torch.any(inputs['melody_note_mask'][~beat_mask]):
                raise ValueError('Padded beat contains real notes')
            source_rows=tracks[meta['track_id']][dataset.windows[i]['start_row']:dataset.windows[i]['stop_row']]
            if inputs['melody_note_mask'].sum().item() != sum(min(len(r['melody_midi']),encoder.max_notes) for r in source_rows):
                raise ValueError('Real melody note count changed during padding')
            if any(targets['target_loss_mask'][j] and not row['target_valid'] for j,row in enumerate(source_rows)):
                raise ValueError('Invalid source target became trainable')
            if not torch.all(inputs['melody_attention_mask'].any(dim=-1)):
                raise ValueError('A melody attention row is fully masked')
            if not all(torch.isfinite(v).all().item() for group in ('inputs','targets') for v in item[group].values()):
                raise ValueError('Nonfinite tensor')
            padded_positions+=int((~beat_mask).sum())
            masked_positions+=int((beat_mask & ~targets['target_loss_mask']).sum())
            sequence_manifest.append(dict(split=split,sequence_index=i,track_id=meta['track_id'],
                first_beat_index=ids[0],last_beat_index=ids[-1],real_beats=len(ids),padded_beats=sequence_length-len(ids),
                valid_targets=int(targets['target_loss_mask'].sum())))
        check(f'{split}_all_beats_covered',all(covered[t]=={r['beat_index'] for r in tracks[t]} for t in covered))
        check(f'{split}_chronology_masks_and_finite_tensors',True)
        batch=next(iter(make_dataloader(dataset,batch_size=batch_size,seed=seed)))
        batches[split]={group:{k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in batch[group].items()}
                        for group in ('inputs','targets')}
        statistics['by_split'][split].update(sequences=len(dataset),padded_beat_positions=padded_positions,
                                            masked_real_target_positions=masked_positions)
    output_dir.mkdir(parents=True,exist_ok=True)
    encoder.save(output_dir)
    reloaded=FeatureEncoder.load(output_dir)
    check('encoder_reload_identical',all(np.array_equal(encoder.encode_inputs(r['input_snapshot'])[key],reloaded.encode_inputs(r['input_snapshot'])[key])
        for rows in tracks.values() for r in rows for key in CATEGORICAL))
    for split,dataset in datasets.items():
        path=output_dir/f'{split}_sequences.npz'
        export_npz(path,dataset)
        saved=SavedSequenceDataset(path)
        check(f'{split}_npz_reload_identical',len(saved)==len(dataset) and all(torch.equal(saved[i][g][k],dataset[i][g][k])
              for i in range(len(dataset)) for g in ('inputs','targets') for k in saved[i][g]))
    implementation_sources=[config.COMPONENT_DIR/'src'/name for name in (
        'encode_features.py','create_sequences.py','transformer_dataset.py','preprocess_for_transformer.py')]
    check('phase1_source_hashes_unchanged',before==digest(sources))
    selected_inputs=sorted(USED_INPUTS)
    settings=dict(schema_version=1,development_only=development_only,sequence_length=sequence_length,
        stride=sequence_length if stride is None else stride,seed=manifest['seed'],batch_size=batch_size,
        max_notes_per_beat=encoder.max_notes,truncate_notes=truncate_notes,ignore_index=IGNORE_INDEX,
        selected_input_columns=selected_inputs,target_column='reference_chord_canonical',mask_column='target_valid',
        input_schema_columns={t:s['input_columns'] for t,s in schemas.items()},
        excluded_input_columns={t:sorted(set(s['input_columns'])-USED_INPUTS) for t,s in schemas.items()},
        audit_columns={t:sorted(set(s['columns'])-USED_INPUTS-{'reference_chord_canonical','target_valid'}) for t,s in schemas.items()},
        input_files=[portable(Path(input_root)/f'track_{int(t):07d}'/f'track_{int(t):07d}_training_events.csv') for t in track_ids],
        melody_pitch_policy='MIDI=round(original melody_midi, ties-to-even)+2; PAD=0, UNK=1. Continuous original pitch/127 retained. Supplied PC+2 (0=PAD,1=UNK). No clipping/transposition.',
        timing_policy='[original duration, relative onset, overlap seconds] / event_duration; preserve negatives; no fitted scaling/clipping; malformed/missing/nonfinite numbers fail.',
        silence_policy='Real note mask remains all false. Safe attention mask enables dummy slot 0 with zero data; pooled melody must be gated to zero by melody_silent. Includes padded beats; beat mask remains false there.',
        unknown_genre_policy='Unseen tags activate UNK dimension; missing tags produce all-zero multi-hot vector.',
        truncation_policy='Preserve original first M notes deterministically only with explicit opt-in; otherwise overflow is an error, including held-out data.',
        beat_policy='Numerical source beat index, zero-based bar number and one-based beat position; separate missing-value masks. Relative bar number subtracts first known bar in each window. No positional encoding.',
        implementation_sha256=digest(implementation_sources),
        torch_version=torch.__version__,numpy_version=np.__version__)
    report=dict(all_passed=all(checks.values()),checks=checks,source_sha256=before,batch_tensors=batches,
                evaluation='Development smoke test only; no validation/test evaluation.' if development_only else 'Track-group split preprocessing only; no model evaluation.',
                identity_detection_limit='Known MTG artist/album and extraction source paths are grouped; duplicate audio under different identities cannot be ruled out without audio fingerprints.')
    for name,value in [('preprocessing_config.json',settings),('split_manifest.json',manifest),
                       ('preprocessing_statistics.json',statistics),('preprocessing_validation_report.json',report)]:
        dump(output_dir/name,value)
    with (output_dir/'sequence_manifest.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(sequence_manifest[0]))
        writer.writeheader(); writer.writerows(sequence_manifest)
    summary = f'''# Phase 2 model preprocessing

{report['evaluation']} No Transformer, training, loss calculation or predictive accuracy is implemented.

The existing Phase 1 JSON/CSV reader, canonicalizer, target parser and basic/key context rule are reused. Phase 1 exports and implementations remain unchanged (SHA-256 verified). Synthetic basic chords remain reference-derived pseudo-labels; schema separation does not make this baseline independent. Next-basic context assumes the entire basic sequence is available (offline refinement, not causal streaming).

Categorical inputs use stable sorted training-only vocabularies: PAD=0, UNK=1, N=2, START=3, END=4, unknown=5. Current/previous/next chord maps share the same vocabulary, after root canonicalization. Missing root on N maps to N; other missing categories map to UNK. The literal unknown harmonic context remains distinct from unseen UNK and genuine no-chord N. Target vocabulary contains PAD/UNK/N plus supported valid training labels; PAD/UNK are reserved and never supervised. Detailed labels are encoded only as targets. Unsupported, missing, invalid or unseen targets become -100 and have false loss masks. Genuine valid N targets remain trainable.

{settings['melody_pitch_policy']}

Melody timing channels are duration_beats, relative_start_beats, overlap_fraction. {settings['timing_policy']} {settings['truncation_policy']} Melody extraction sources have training-only IDs. Real MIDI 0 has ID 2, distinct from padding. Fractional MIDI is preserved in the continuous channel even when rounded for categorical encoding. Empty melody lists preserve silence; {settings['silence_policy']} A future attention module must use the safe attention mask, invert it for PyTorch key_padding_mask, and zero silent pooled outputs. The real note mask must not be passed directly as an all-masked attention row.

Genres use track-level multi-hot [B,G]. {settings['unknown_genre_policy']} Key, mode, categorical confidence, Roman numeral and harmonic function use input-side Phase 1 values, validated against the existing basic/key context rule. {settings['beat_policy']} beat_timing_known_mask has [bar-known, beat-position-known]. Other redundant declared inputs (raw basic label/root PC) are recorded but not separately tensorized.

Sequences have length {sequence_length}, stride {settings['stride']}, never cross tracks or splits, and retain invalid-target beats as context. The final window is padded. Overlap is supported with stride <= length; repeated beats/targets are repeated training occurrences and must be deduplicated for later beat-level evaluation. Note masks and beat masks are independent. All masks use True=real/keep, except melody_attention_mask includes the safe dummy slot. No positional encoding is added. Sequence metadata retains exact int64 source beat IDs (-1 for padding) and track ID. Debug raw labels are opt-in metadata only. Individual-beat debugging is supported with individual_beats=True.

Encoders and note capacity are fit exclusively from split_manifest.json train tracks. Transform/load never refits. Split ratios operate on connected recording/artist/album groups; realized track ratios can differ. Seed and assignments are saved. {report['identity_detection_limit']} A one-track run requires --development-only and has no validation/test partition.

Inputs are exclusively the selected_input_columns in preprocessing_config.json. Only reference_chord_canonical is the prediction target; target_valid is the loss eligibility gate. Reference quality, bass, root PC, overlaps, candidates, raw labels, source IDs and raw melody objects remain audit-only and are never tensorized as inputs. The original schema input/target lists, selected inputs and audit columns are saved in the config.

NPZ archives contain numerical arrays and Unicode track IDs only. Load with np.load(path, allow_pickle=False) or SavedSequenceDataset(path). ChordSequenceDataset builds the same tensors directly from validated CSV rows and FeatureEncoder.load(output_dir). make_dataloader uses the custom collator; each batch has separate inputs, targets, metadata dictionaries. Do not pass the targets or metadata dictionaries into a model's feature encoder.

Future model choices include embedding sizes, melody pooling with silence handling, positional encoding, attention architecture and loss weighting. Before research evaluation, obtain independent basic predictions, assess pseudo-label/key/beat quality, and use enough distinct track groups. No vocabulary/statistical parameters may be refit on held-out data.

## Actual statistics

```json
{json.dumps(statistics,indent=2)}
```

## First batch tensors (B is actual batch size, L beats, M training-fitted notes, G genre dimensions)

| Split / group / tensor | Shape | dtype |
|---|---|---|
'''
    for split,groups in batches.items():
        for group,tensors in groups.items():
            for name,info in tensors.items():
                summary+=f"| {split} / {group} / {name} | {info['shape']} | {info['dtype']} |\n"
    summary+=f"\nAll {len(checks)} preprocessing checks passed. Run tests with `.\\.venv\\Scripts\\python.exe -m unittest experiments.rahul.tests.test_model_preprocessing -v`.\n"
    (output_dir/'preprocessing_summary.md').write_text(summary,encoding='utf-8')
    print(json.dumps(dict(output=portable(output_dir),statistics=statistics,batches=batches,checks=checks),indent=2))
    return dict(tracks=tracks,encoder=encoder,datasets=datasets,statistics=statistics,report=report,settings=settings)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--track-id',action='append',required=True)
    parser.add_argument('--input-root',type=Path,default=DEFAULT_INPUT)
    parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--development-only',action='store_true')
    parser.add_argument('--sequence-length',type=int,default=8)
    parser.add_argument('--stride',type=int)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--split-ratios',type=float,nargs=3,default=(.8,.1,.1))
    parser.add_argument('--max-notes',type=int)
    parser.add_argument('--truncate-notes',action='store_true')
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--split-manifest',type=Path)
    args=parser.parse_args()
    run(args.track_id,args.input_root,args.output_dir,args.sequence_length,args.stride,args.development_only,
        args.seed,args.split_ratios,args.max_notes,args.truncate_notes,args.batch_size,args.split_manifest)


if __name__=='__main__':
    main()
