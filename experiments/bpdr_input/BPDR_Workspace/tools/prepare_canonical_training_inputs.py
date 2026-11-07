"""Apply the unmodified app frontend uniformly to all33 declared training sources.

No intervention, fitting, development access or musical-reference approval occurs.
Old context arrays are reused only when the canonical PCM bytes are identical.
"""
from collections import Counter
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np
import soundfile as sf
import torch

WS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WS / 'source'))
sys.path.insert(0, str(WS / 'tools'))
from acquire_fresh_confirmation import check_lock
from probe_short_phrase_renderer import sha
from prepare_short_phrase_training import read, write, norm, save_analysis
from bpdr_input.audio import pipeline
from bpdr_input.experiments.learning import model_recovery as recovery
from bpdr_input.repair.rule_correction import segment_notes, RulePolicy

ROOT = WS / 'results/transformer_usability/2026-11-02_v54_canonical_training_inputs'
PARENTS = [ROOT.parent / '2026-11-02_v45_native_short_pairs', ROOT.parent / '2026-11-02_v50_humtrans_short_training']


def declare():
    check_lock()
    ROOT.mkdir(parents=True, exist_ok=True)
    sources = []
    inputs = {}
    for parent, domain in zip(PARENTS, ['vocadito', 'humming']):
        for name in ('train_originals.json', 'integrity_receipt.json', 'input_hashes.json'):
            inputs[str(parent / name)] = sha(parent / name)
        for path, expected in read(parent / 'input_hashes.json').items():
            assert sha(path) == expected
        for original in read(parent / 'train_originals.json'):
            source = original['source']
            assert source['split'] == 'train'
            native_path = source['audio_path'] if domain == 'vocadito' else source['path']
            native_hash = source['audio_sha256'] if domain == 'vocadito' else source['sha256']
            assert sha(native_path) == native_hash
            sources.append(dict(source_id=original['source_id'], group=original['group'], domain=domain,
                native_path=native_path, native_sha256=native_hash, prior_original=original))
    assert len(sources) == 33 and len({r['group'] for r in sources}) == 20
    d = norm(dict(version='canonical-app-training-inputs-1', source_sha256=sha(__file__),
        frontend_sha256=sha(pipeline.__file__), context_source_sha256=sha(recovery.__file__),
        parent_hashes=inputs, sources=sources, planned_sources=33, policy=asdict(pipeline.Policy()),
        crop='Same first min(10s,duration) and highest-energy full-native channel as historical construction; preserve samples in FLOAT before calling unmodified process()',
        five_second_target_minimum=True, no_extension_to_short_recordings=True,
        context_reuse='Only when accepted PCM24 file SHA256 equals prior original working file; otherwise remeasure input-only context',
        context_region='Same source-only70c split; no intended-note guide or model outcome',
        reason='Remove training/app frontend mismatch; use existing clipping-fraction, resampling length and peak-protection behavior uniformly',
        nontraining_audio_accessed=False, new_interventions=0, new_model_fits=0,
        research_training_eligible=False, production_eligible=False))
    path = ROOT / 'declaration.json'
    if path.exists():
        assert read(path) == d
    else:
        write(path, d)
    return d


def main():
    torch.set_num_threads(2)
    d = declare()
    originals, all_hashes = [], {}
    for index, source in enumerate(d['sources'], 1):
        sid = source['source_id']
        folder = ROOT / sid; folder.mkdir(exist_ok=True)
        result_path = folder / 'result.json'
        if result_path.exists():
            saved = read(result_path)
            assert saved['declaration_sha256'] == sha(ROOT / 'declaration.json')
            for path, expected in saved['output_hashes'].items():
                assert sha(path) == expected
            originals.append(saved['original']); all_hashes.update(saved['output_hashes'])
            print('CANONICAL_TRAINING_SOURCE_REUSED', sid, flush=True)
            continue
        native, sr = sf.read(source['native_path'], dtype='float32', always_2d=True)
        channel = int(np.argmax(np.mean(native.astype(np.float64)**2, axis=0)))
        crop = native[:min(len(native), 10 * sr), channel]
        crop_path = folder / 'native_training_crop.wav'
        sf.write(crop_path, crop, sr, subtype='FLOAT')
        decoded, rate = sf.read(crop_path, dtype='float32')
        assert rate == sr
        np.testing.assert_array_equal(decoded, crop)
        result, directory = pipeline.process(crop_path, folder / 'canonical_process', WS / 'checkpoints')
        diagnostics_path = directory / 'diagnostics.json'
        diagnostics = read(diagnostics_path)
        original = dict(source_id=sid, group=source['group'], domain=source['domain'], source=source,
            split='train', crop_offset_seconds=0., crop_native_samples=len(crop), source_channel=channel,
            native_duration_seconds=len(crop) / sr, canonical_status=result.status.value,
            canonical_diagnostics=str(diagnostics_path), canonical_operations=diagnostics['operations'],
            sample_rate_hz=22050, research_training_eligible=False)
        paths = [crop_path, diagnostics_path, Path(diagnostics['source']['preserved_path'])]
        accepted = result.accepted_audio
        if accepted is None:
            original.update(excluded='CANONICAL_FRONTEND_REJECTED', reason_codes=list(result.reason_codes))
        else:
            paths.append(Path(accepted.path))
            original.update(path_A=accepted.path, duration_seconds=accepted.duration_seconds,
                            canonical_working_sha256=accepted.checksum)
            if len(crop) / sr < 5.:
                original['excluded'] = 'SOURCE_SHORTER_THAN_FIVE_SECOND_TARGET'
            else:
                old = source['prior_original']
                identical = 'excluded' not in old and sha(old['path_A']) == accepted.checksum
                original['identical_to_historical_working_pcm'] = identical
                if identical:
                    original.update(analysis_A=old['analysis_A'], longest_region_frames=old['longest_region_frames'])
                    paths.append(Path(old['analysis_A']))
                else:
                    audio, rate = sf.read(accepted.path, dtype='float32')
                    assert rate == 22050
                    analysis = recovery.measured(audio)
                    context = folder / 'canonical_repair_analysis.npz'; save_analysis(context, analysis)
                    regions = segment_notes(analysis['views'][0][:,64], analysis['reliable'], RulePolicy(note_split_cents=70.))
                    region = max(regions, key=lambda r:r[1] - r[0]) if regions else None
                    original.update(analysis_A=str(context), longest_region_frames=region)
                    paths.append(context)
                region = original['longest_region_frames']
                original['widths_with_150ms_context'] = [width for width in (.24,.28,.36,.50)
                    if region is not None and (region[1] - region[0]) * .02 >= width + .30 - 1e-12]
        assert sha(source['native_path']) == source['native_sha256']
        saved = dict(declaration_sha256=sha(ROOT / 'declaration.json'), original=original,
            output_hashes={str(path):sha(path) for path in paths})
        write(result_path, saved)
        originals.append(original); all_hashes.update(saved['output_hashes'])
        write(ROOT / 'progress.json', dict(completed_sources=index, planned_sources=33))
        print('CANONICAL_TRAINING_SOURCE', index, '/33', sid,
              original.get('excluded', original.get('identical_to_historical_working_pcm')), flush=True)
    write(ROOT / 'train_originals.json', originals); write(ROOT / 'input_hashes.json', all_hashes)
    valid = [r for r in originals if 'excluded' not in r]
    summary = dict(attempted=33, valid=len(valid), excluded=dict(Counter(r['excluded'] for r in originals if 'excluded' in r)),
        working_pcm_identical_to_prior=sum(r.get('identical_to_historical_working_pcm',False) for r in valid),
        working_pcm_changed_or_new=sum(not r.get('identical_to_historical_working_pcm',False) for r in valid),
        prior_excluded_now_target_ready=[r['source_id'] for r in valid if 'excluded' in r['source']['prior_original']],
        groups_with_short_context=sorted({r['group'] for r in valid if r['widths_with_150ms_context']}),
        new_interventions=0, new_model_fits=0, nontraining_audio_accessed=False,
        research_training_eligible=False, production_eligible=False)
    write(ROOT / 'summary.json', summary)
    print('CANONICAL_TRAINING_INPUTS_READY', summary, flush=True)


if __name__ == '__main__':
    main()
