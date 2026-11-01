"""Bounded actual-WAV renderer comparison on frozen Transformer outputs."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import torch

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE/'source'))
import run_transformer_curve_study as study
from bpdr_input.experiments.learning import context_improvement as ex
from bpdr_input.experiments.learning import held_voice_training as held
from bpdr_input.modeling.joint_transformer import load, predict_raw
from bpdr_input.modeling.correction_accuracy import assert_independent
from bpdr_input.repair import framewise_renderer
from bpdr_input.repair.phase_vocoder_renderer import render_candidate

ROOT = WORKSPACE/'results/transformer_usability/2026-10-28_v10_renderer'
V9 = ROOT.parent/'2026-10-28_v9_curve'
V8 = ROOT.parent/'2026-10-28_v8_joint'


def declare():
    ROOT.mkdir(exist_ok=True)
    if not (ROOT/'declaration.json').exists():
        files = [Path(__file__), Path(study.__file__),
                 WORKSPACE/'source/bpdr_input/repair/phase_vocoder_renderer.py',
                 WORKSPACE/'source/bpdr_input/repair/framewise_renderer.py',
                 WORKSPACE/'source/bpdr_input/repair/framewise_audit.py',
                 WORKSPACE/'source/bpdr_input/repair/curve_conditioning.py']
        ex.write(ROOT/'declaration.json', dict(created_at=datetime.now(timezone.utc).isoformat(),
                 source_model_run=str(V8), conditions=['praat_median11', 'praat_span_median', 'phase_vocoder_span_median'],
                 selection='Training-side original identity first, then singer-macro delivered error. Tie retain current Praat median11. No development-based selection.',
                 comparison='All prediction supports, model weights and audits frozen. Compare two constant-curve renderers plus current median11/Praat reference. No new fits.',
                 baseline_evidence={c: dict(path=str(V9/f'training/{kind}/evidence.json'), sha256=ex.sha(V9/f'training/{kind}/evidence.json'))
                     for c, kind in [('praat_median11', 'median11'), ('praat_span_median', 'span_median')]},
                 fixed_dependencies=ex.read(V9/'declaration.json'),
                 source_hashes={str(p): ex.sha(p) for p in files},
                 renderer_versions=dict(librosa='1.0.0', soxr='1.1.0', parselmouth='0.4.7'),
                 human_quality='Not assessed; phase vocoder does not establish formant preservation or naturalness',
                 fresh_final_audio_access=False, demo_activated=False, research_validation_passed=False))
        snapshot = ROOT/'executed_sources'
        snapshot.mkdir(exist_ok=True)
        for p in files:
            (snapshot/p.name).write_bytes(p.read_bytes())
    d = ex.read(ROOT/'declaration.json')
    for p, h in d['source_hashes'].items():
        assert ex.sha(p) == h
    for name in ('model_checkpoints', 'source_hashes', 'training_input_hashes'):
        for p, h in d['fixed_dependencies'][name].items():
            assert ex.sha(p) == h
    for baseline in d['baseline_evidence'].values():
        assert ex.sha(baseline['path']) == baseline['sha256']
    arm = ROOT/'arms/phase_vocoder'
    arm.mkdir(parents=True, exist_ok=True)
    if not (arm/'declaration.json').exists():
        ex.write(arm/'declaration.json', dict(parent_declaration=str(ROOT/'declaration.json'),
                 parent_sha256=ex.sha(ROOT/'declaration.json'), renderer='phase_vocoder', curve='span_median'))
    return d, arm


def run_stage(stage, arm):
    study.ROOT = arm
    previous = framewise_renderer.render_candidate
    framewise_renderer.render_candidate = render_candidate
    try:
        if stage == 'training':
            manifest = V8/'train_pairs.json'
            rows = ex.read(manifest)
            pairs = held.prepared(rows)
            cache, models = study.training_cache(rows)
            sources = ex.read(V8/'anchor_sources.json')
        else:
            manifest = V8/'development/pairs.json'
            rows = ex.read(manifest)
            pairs = held.prepared(rows)
            model, payload = load(V8/'final/seed_339/checkpoint.pt')
            assert_independent(payload['training_groups'], payload['training_sources'], rows)
            cache = {}
            for p in pairs:
                if 'excluded' not in p:
                    for view in ('A', 'C'):
                        if p['path_'+view] not in cache:
                            cache[p['path_'+view]] = np.stack([predict_raw(model, p[view]['views'][0])[0]]*3)
            source_folder = V8.parent/'2026-10-28_v7_guard/development'
            sources = [ex.read(p)['source'] for p in sorted(source_folder.glob('source_*.json'))]
            models = {s['source']['singer_id']: (model, payload) for s in sources}
        return study.run_condition(rows, pairs, cache, models, sources, 'span_median', stage, manifest)
    finally:
        framewise_renderer.render_candidate = previous


def main():
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['training', 'development'])
    args = parser.parse_args()
    d, arm = declare()
    if args.stage == 'training':
        phase = run_stage('training', arm)
        results = {c: ex.read(row['path']) for c, row in d['baseline_evidence'].items()}
        results['phase_vocoder_span_median'] = phase
        selected = min(d['conditions'], key=lambda c: (not results[c]['gates']['original_controls_all_identity'],
                       results[c]['summary']['delivered_after_mae_cents'], d['conditions'].index(c)))
        ex.write(ROOT/'locked_selection.json', dict(condition=selected, results=results,
                 selection_data='Training-side actual-WAV only', diagnostic_only=True,
                 demo_activated=False, research_validation_passed=False))
        print('RENDERER_SELECTION_LOCKED', selected, flush=True)
    else:
        selection = ex.read(ROOT/'locked_selection.json')
        if selection['condition'] != 'phase_vocoder_span_median':
            print('PHASE_NOT_SELECTED; no development access for an inferior training-side condition', flush=True)
        else:
            run_stage('development', arm)
    declare()


if __name__ == '__main__':
    main()
