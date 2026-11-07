"""Bind existing Vocadito source groups before new short-phrase preparation."""
from pathlib import Path
from collections import Counter
import json,sys
import soundfile as sf

WS=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(WS/'source'));sys.path.insert(0,str(WS/'tools'))
from bpdr_input.preparation.sources.dataset import singer_splits
from probe_short_phrase_renderer import sha

ROOT=WS/'results/transformer_usability/2026-11-02_v40_short_phrase_data'


def main():
    latest=json.loads((WS.parent/'runs/phase2/latest.json').read_text(encoding='utf-8'))
    manifest_path=Path(latest['path'])/'manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    metadata=WS.parent/'data/vocadito/vocadito_metadata.csv'
    assert sha(metadata)==manifest['metadata_sha256']
    assert len(manifest['sources'])==40
    split=singer_splits([s['singer_id'] for s in manifest['sources']],manifest['policy']['split_seed'])
    assert split==manifest['singer_splits'] and len(split)==29
    sources=[]
    for source in sorted(manifest['sources'],key=lambda s:int(s['source_id'].split('_')[-1])):
        path=Path(source['paths']['audio']);assert sha(path)==source['sha256']['audio']
        assert source['split']==split[source['singer_id']]
        info=sf.info(path)
        assert info.samplerate==source['sample_rate_hz'] and info.frames==source['sample_count']
        sources.append(dict(source_id=source['source_id'],singer_id=source['singer_id'],
            split=source['split'],audio_path=str(path),audio_sha256=sha(path),
            sample_rate_hz=info.samplerate,sample_count=info.frames,channels=info.channels,
            construction_allowed_in_next_training_step=source['split']=='train',
            human_reference_paths=source['paths'],human_reference_hashes=source['sha256'],
            research_training_eligible=False,
            reason='Existing source grouping retained. Controlled-recovery engineering use needs valid physical-intervention labels; independent reference-suitability reviews remain pending.'))
    d=dict(version='short-phrase-source-reservation-1',script_sha256=sha(__file__),
        existing_manifest_path=str(manifest_path),existing_manifest_sha256=sha(manifest_path),
        metadata_sha256=sha(metadata),singer_splits=split,sources=sources,
        split_source_counts=dict(Counter(s['split'] for s in sources)),
        split_singer_counts=dict(Counter(split.values())),
        existing_partitions_preserved=True,source_count=40,singer_count=29,
        no_acoustic_decode_in_this_command=True,no_annotation_arrays_loaded=True,
        historically_inspected_dataset=True,untouched_final_validation_claim=False,
        planned_input='First min(10s, raw duration); min5s; highest-energy channel; resample22050Hz PCM24; no normalization or padding; retain crop mapping.',
        short_duration_targets_seconds=[.24,.36,.50,.70],planned_shifts_cents=[-30.,-20.,20.,30.],
        train_only_planned_attempts=336,model_training_started=False,
        research_validation_passed=False,production_eligible=False,
        license='Creative Commons Attribution 4.0 International',
        license_source='https://mirdata.readthedocs.io/en/1.0.0/_modules/mirdata/datasets/vocadito.html',
        dataset_source='https://zenodo.org/records/5578807',
        attribution='Bittner, Pasalo, Bosch, Meseguer Brocal and Rubinstein (2021), vocadito: A dataset of solo vocals with f0, note, and lyric annotations. DOI10.5281/zenodo.5578807.',
        limitations='Observed performed-pitch references and controlled deviations are not natural intended-note truth. These existing professional annotations do not replace project-specific independent suitability review.')
    ROOT.mkdir(parents=True,exist_ok=True);path=ROOT/'source_reservation.json'
    if path.exists():assert json.loads(path.read_text(encoding='utf-8'))==d
    else:path.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print('SHORT_PHRASE_SOURCES_RESERVED',d['split_source_counts'],d['split_singer_counts'],flush=True)


if __name__=='__main__':main()
