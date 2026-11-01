"""Privileged local-context diagnostic; no new runtime features or model selection."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
import train_dense_residual_transformer as study
from bpdr_input.modeling import step_context
from bpdr_input.repair.held_voice_policy import HeldPolicy, proposal, spans

ROOT = study.ROOT.parent / '2026-11-01_step_context_locality'


def evidence_candidates(values):
    return [(int(a), int(b)) for a, b in spans(values[:, 0] != 0)
            if b - a >= 43 and values[a:b, 1].max() >= .75]


def main():
    ex = study.ex; d = study.declare(); ROOT.mkdir(exist_ok=True)
    baseline = study.ROOT.parent / '2026-10-30_known_label_diagnostic/results.json'
    declaration = ROOT / 'declaration.json'
    if not declaration.exists():
        ex.write(declaration, dict(created_at=datetime.now(timezone.utc).isoformat(),
                 source_sha256=ex.sha(Path(__file__)), input_hashes=d['input_hashes'],
                 fitter_sha256=ex.sha(step_context.__file__), policy=d['policy'],
                 known_label_preflight_sha256=ex.sha(baseline),
                 protocol='Compare unchanged whole-recording step fitter with the EXACT SAME fitter on201-frame context centered at the known injected support midpoint, radius100 matching Transformer context. Lift cropped evidence back without extending support; all other input-only scope/controller checks unchanged. Known-label correction predictions as earlier oracle. Inspect paired unchanged-A cue patterns separately, deduplicated source/crop.',
                 limitation='Crop position and predictions use privileged labels. This is a causal context diagnostic, NOT a runtime method, trained model, valid new feature contract or general performance bound. Original cue patterns are not edits. No optimal crop/window search, no changed gain threshold, no rendering.',
                 NOT_MODEL_PERFORMANCE=True, selection_eligible=False, demo_activated=False, research_validation_passed=False))
    locked = ex.read(declaration)
    assert locked['source_sha256'] == ex.sha(Path(__file__))
    assert locked['fitter_sha256'] == ex.sha(step_context.__file__)
    assert locked['known_label_preflight_sha256'] == ex.sha(baseline)
    for path, digest in locked['input_hashes'].items():
        assert ex.sha(path) == digest
    if (ROOT / 'results.json').exists():
        raise ValueError('Keep completed diagnostic')
    pairs = study.training.held.prepared(ex.read(study.ROOT / 'train_pairs.json'))
    cases, originals, counts = [], {}, Counter()
    original_function = step_context.evidence
    with threadpool_limits(limits=2):
        for index, pair in enumerate(pairs):
            if 'excluded' in pair:
                continue
            raw = pair['C']['views'][0]; times = pair['C']['features']['times_seconds']
            center = int(np.argmin(abs(times - np.mean(pair['support_seconds']))))
            left, right = max(0, center - 100), min(len(raw), center + 101)
            localized = np.zeros((len(raw), 3), np.float32)
            localized[left:right] = original_function(raw[left:right])
            values = np.stack([np.where(pair['mask'], pair['target'], 0.)] * 3)
            old = proposal(pair['audio_C'], times, pair['C']['views'], values, HeldPolicy(**locked['policy']))
            def one_crop(input_raw):
                np.testing.assert_array_equal(input_raw, raw)
                return localized
            step_context.evidence = one_crop
            try:
                new = proposal(pair['audio_C'], times, pair['C']['views'], values, HeldPolicy(**locked['policy']))
            finally:
                step_context.evidence = original_function
            assert [(c['start'], c['stop']) for c in old['candidates']] == [(c['start'], c['stop']) for c in new['candidates']]
            for a, b in zip(old['candidates'], new['candidates'], strict=True):
                if a['reason'] != b['reason']:
                    assert {a['reason'], b['reason']} <= {None, 'STEP_SHAPE_EVIDENCE'}
            old_permitted = bool(old['plan']['permitted'] and not old['plan']['identity'])
            new_permitted = bool(new['plan']['permitted'] and not new['plan']['identity'])
            counts['scoreable'] += 1
            counts['old_permitted'] += int(old_permitted); counts['local_permitted'] += int(new_permitted)
            counts['newly_permitted'] += int(new_permitted and not old_permitted)
            counts['lost_permission'] += int(old_permitted and not new_permitted)
            key = (pair['source']['sha256'], left, right)
            if key not in originals:
                before = pair['A']['views'][0]
                local_a = np.zeros((len(before), 3), np.float32)
                local_a[left:right] = original_function(before[left:right])
                originals[key] = dict(source_sha256=key[0], crop_frames=[left, right],
                     whole_recording_cue_candidates=evidence_candidates(original_function(before)),
                     local_cue_candidates=evidence_candidates(local_a))
            cases.append(dict(index=index, group=pair['group'], crop_frames=[left, right],
                 old_permitted=old_permitted, local_permitted=new_permitted,
                 old_candidates=old['candidates'], local_candidates=new['candidates'],
                 old_plan=old['plan'], local_plan=new['plan']))
            if len(cases) % 20 == 0:
                ex.write(ROOT / 'progress.json', dict(completed_cases=len(cases), counts=dict(counts)))
                print('CONTEXT_LOCALITY_PROGRESS', len(cases), dict(counts), flush=True)
    assert counts['old_permitted'] == ex.read(baseline)['counts']['nonidentity_permitted'] == 45
    protected = ex.read(study.ROOT.parent / '2026-10-28_v7_guard/declaration.json')['protected_files']
    for path, digest in protected.items():
        assert ex.sha(path) == digest
    result = dict(counts=dict(counts), attempted=200, scoreable=108, excluded=92,
                  paired_original_crop_queries=len(originals),
                  paired_originals_with_global_cue=sum(bool(r['whole_recording_cue_candidates']) for r in originals.values()),
                  paired_originals_with_local_cue=sum(bool(r['local_cue_candidates']) for r in originals.values()),
                  cases=cases, original_queries=list(originals.values()),
                  declaration_sha256=ex.sha(declaration), NOT_MODEL_PERFORMANCE=True,
                  selection_eligible=False, protected_files_unchanged=len(protected), research_validation_passed=False)
    ex.write(ROOT / 'results.json', result)
    print('CONTEXT_LOCALITY_DIAGNOSTIC', {k: v for k, v in result.items() if k not in ('cases', 'original_queries')}, flush=True)


if __name__ == '__main__':
    main()
