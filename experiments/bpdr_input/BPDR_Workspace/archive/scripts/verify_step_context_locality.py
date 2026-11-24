"""Replay the privileged context diagnostic; never approve it as runtime input."""
import numpy as np
from threadpoolctl import threadpool_limits
import diagnose_step_context_locality as study
from bpdr_input.repair.repair_controller import decision_hash


def main():
    ex = study.study.ex; study.study.declare()
    d = ex.read(study.ROOT / 'declaration.json'); result = ex.read(study.ROOT / 'results.json')
    assert ex.sha(study.__file__) == d['source_sha256']
    assert ex.sha(study.step_context.__file__) == d['fitter_sha256']
    assert result['declaration_sha256'] == ex.sha(study.ROOT / 'declaration.json')
    assert result['NOT_MODEL_PERFORMANCE'] and not result['selection_eligible']
    for path, digest in d['input_hashes'].items():
        assert ex.sha(path) == digest
    pairs = study.study.training.held.prepared(ex.read(study.study.ROOT / 'train_pairs.json'))
    stored = {row['index']: row for row in result['cases']}; counts = study.Counter()
    original = study.step_context.evidence; originals = {}
    with threadpool_limits(limits=2):
        for index, pair in enumerate(pairs):
            if 'excluded' in pair:
                assert index not in stored
                continue
            row = stored[index]; raw = pair['C']['views'][0]; times = pair['C']['features']['times_seconds']
            center = int(np.argmin(abs(times - np.mean(pair['support_seconds']))))
            lo, hi = max(0, center - 100), min(len(raw), center + 101)
            assert row['crop_frames'] == [lo, hi]
            local = np.zeros((len(raw), 3), np.float32); local[lo:hi] = original(raw[lo:hi])
            values = np.stack([np.where(pair['mask'], pair['target'], 0.)] * 3)
            old = study.proposal(pair['audio_C'], times, pair['C']['views'], values, study.HeldPolicy(**d['policy']))
            study.step_context.evidence = lambda incoming: local
            try:
                new = study.proposal(pair['audio_C'], times, pair['C']['views'], values, study.HeldPolicy(**d['policy']))
            finally:
                study.step_context.evidence = original
            for kind, detail in (('old', old), ('local', new)):
                assert detail['plan']['decision_sha256'] == decision_hash(detail['plan'])
                assert detail['plan']['decision_sha256'] == row[kind + '_plan']['decision_sha256']
                assert detail['candidates'] == row[kind + '_candidates']
            a = bool(old['plan']['permitted'] and not old['plan']['identity'])
            b = bool(new['plan']['permitted'] and not new['plan']['identity'])
            counts.update(scoreable=1, old_permitted=int(a), local_permitted=int(b),
                          newly_permitted=int(b and not a), lost_permission=int(a and not b))
            key = pair['source']['sha256'], lo, hi
            if key not in originals:
                before = pair['A']['views'][0]; crop = np.zeros((len(before), 3), np.float32)
                crop[lo:hi] = original(before[lo:hi])
                originals[key] = dict(source_sha256=key[0], crop_frames=[lo, hi],
                     whole_recording_cue_candidates=study.evidence_candidates(original(before)),
                     local_cue_candidates=study.evidence_candidates(crop))
            if counts['scoreable'] % 30 == 0:
                print('LOCALITY_CASES_REPLAYED', counts['scoreable'], flush=True)
    assert dict(counts) == result['counts']
    assert len(originals) == result['paired_original_crop_queries'] == 97
    for row in result['original_queries']:
        expected = originals[(row['source_sha256'], *row['crop_frames'])]
        for key in ('whole_recording_cue_candidates', 'local_cue_candidates'):
            assert [list(span) for span in expected[key]] == row[key]
    protected = ex.read(study.study.ROOT.parent / '2026-10-28_v7_guard/declaration.json')['protected_files']
    for path, digest in protected.items():
        assert ex.sha(path) == digest
    assert not (study.study.ROOT.parent / 'active_model.json').exists()
    receipt = dict(cases_replayed=108, paired_original_crop_queries_verified=97,
                   old_permission_count=45, local_permission_count=48, changed_case_count=3,
                   fitter_and_controller_unchanged=True, privileged_geometry_NOT_RUNTIME=True,
                   results_sha256=ex.sha(study.ROOT / 'results.json'), verifier_sha256=ex.sha(__file__),
                   protected_files_unchanged=len(protected), selection_eligible=False, research_validation_passed=False)
    ex.write(study.ROOT / 'integrity_receipt.json', receipt)
    print('LOCALITY_DIAGNOSTIC_VERIFIED', receipt, flush=True)


if __name__ == '__main__':
    main()
