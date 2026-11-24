"""Compare excluded training crops with the existing app's technical frontend.

No renderer/model fitting, threshold change or reinterpretation of old results.
The target's five-second minimum remains separate from technical acceptance.
"""
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np
import soundfile as sf

WS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WS / 'source'))
sys.path.insert(0, str(WS / 'tools'))
from acquire_fresh_confirmation import check_lock
from prepare_short_phrase_training import read, write, norm
from probe_short_phrase_renderer import sha
from bpdr_input.audio import pipeline

ROOT = WS / 'results/transformer_usability/2026-11-02_v53_frontend_parity'
VOC = ROOT.parent / '2026-11-02_v40_short_phrase_data'
HUM = ROOT.parent / '2026-11-02_v50_humtrans_short_training'


def main():
    check_lock()
    ROOT.mkdir(parents=True, exist_ok=True)
    sources = []
    for data, domain in ((VOC, 'vocadito'), (HUM, 'humming')):
        for original in read(data / 'train_originals.json'):
            if 'excluded' not in original:
                continue
            source = original['source']
            path = source['audio_path'] if domain == 'vocadito' else source['path']
            expected = source['audio_sha256'] if domain == 'vocadito' else source['sha256']
            assert sha(path) == expected and source['split'] == 'train'
            sources.append(dict(source_id=original['source_id'], domain=domain, path=path,
                sha256=expected, historical_exclusion=original['excluded']))
    assert len(sources) == 5
    declaration = norm(dict(version='excluded-training-existing-frontend-parity-1',
        source_sha256=sha(__file__), frontend_sha256=sha(pipeline.__file__), policy=asdict(pipeline.Policy()),
        sources=sources, planned_sources=5,
        construction='Same native highest-energy channel and first min(10s,duration) crop; save FLOAT working source, then call the unmodified production process(). Original full-source bytes retained unchanged.',
        target_duration_minimum_seconds=5.,
        boundary='Technical acceptance does not override the five-second task minimum or musical-reference approval. Existing frontend peak protection is not clipping repair.',
        nontraining_audio_accessed=False, model_fitting=False, production_eligible=False))
    dpath = ROOT / 'declaration.json'
    if dpath.exists():
        assert read(dpath) == declaration
    else:
        write(dpath, declaration)
    reports = []
    for source in sources:
        folder = ROOT / source['source_id']; folder.mkdir(exist_ok=True)
        saved = folder / 'result.json'
        if saved.exists():
            row = read(saved)
            assert row['declaration_sha256'] == sha(dpath)
            if row['accepted_working_path']:
                assert sha(row['accepted_working_path']) == row['accepted_working_sha256']
            reports.append(row); continue
        native, sr = sf.read(source['path'], dtype='float32', always_2d=True)
        channel = int(np.argmax(np.mean(native.astype(np.float64)**2, axis=0)))
        crop = native[:min(len(native), 10 * sr), channel]
        crop_path = folder / 'native_training_crop.wav'
        sf.write(crop_path, crop, sr, subtype='FLOAT')
        decoded, _ = sf.read(crop_path, dtype='float32')
        np.testing.assert_array_equal(crop, decoded)
        result, directory = pipeline.process(crop_path, folder / 'canonical_process', WS / 'checkpoints')
        diagnostics = read(directory / 'diagnostics.json')
        accepted = result.accepted_audio
        row = dict(source_id=source['source_id'], domain=source['domain'],
            declaration_sha256=sha(dpath), original_source_sha256=sha(source['path']),
            historical_exclusion=source['historical_exclusion'],
            crop_duration_seconds=len(crop) / sr, native_peak=float(abs(crop).max()),
            native_at_or_above_0999_fraction=float(np.mean(abs(crop) >= .999)),
            canonical_status=result.status.value, canonical_reasons=list(result.reason_codes),
            meets_five_second_target=bool(len(crop) / sr >= 5.),
            canonical_operations=diagnostics['operations'], diagnostics_path=str(directory / 'diagnostics.json'),
            accepted_working_path=accepted.path if accepted else None,
            accepted_working_sha256=accepted.checksum if accepted else None,
            technical_only=True, research_training_eligible=False, new_model_fits=0)
        assert row['original_source_sha256'] == source['sha256']
        write(saved, row); reports.append(row)
        print('FRONTEND_PARITY_SOURCE', source['source_id'], result.status.value, row['meets_five_second_target'], flush=True)
    summary = dict(records=reports, declaration_sha256=sha(dpath), source_sha256=sha(__file__),
        technical_accepts=sum(r['accepted_working_path'] is not None for r in reports),
        technical_and_target_duration_accepts=sum(r['accepted_working_path'] is not None and r['meets_five_second_target'] for r in reports),
        old_results_unchanged=True, new_model_fits=0, production_eligible=False)
    write(ROOT / 'results.json', summary)
    print('FRONTEND_PARITY_COMPLETE', summary['technical_accepts'], summary['technical_and_target_duration_accepts'], flush=True)


if __name__ == '__main__':
    main()
