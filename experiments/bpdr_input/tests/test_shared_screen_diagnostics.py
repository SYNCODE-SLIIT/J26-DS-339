import unittest
from unittest.mock import patch
import numpy as np
from bpdr_input.shared_screen_diagnostics import decompose, diagnose_training_crop, missing_training_tracks


def fixture():
    inside = np.array([False, True, True, True, True, True, True, True])
    overlap = np.ones(8, dtype=int);overlap[1] = 2
    voiced = np.ones(8, dtype=bool);voiced[2] = False
    expected = np.array([False]*6+[True, True])
    note = np.full(8, 220.);note[5] = 110.
    loaded = dict(features=dict(f0_hz=np.full(8, 220.)), inside_notes=inside, overlap=overlap,
                  note_hz=note, model_inputs=dict(voiced_mask=voiced),
                  masks=dict(reliable_pitch_screen=expected))
    yin = np.full(8, 220.);yin[3] = np.nan;yin[4] = 330.
    return loaded, yin


class ScreenDiagnosticsTests(unittest.TestCase):
    def test_first_failure_accounting_and_contiguous_support(self):
        loaded, yin = fixture()
        result, arrays = decompose(loaded, yin)
        self.assertEqual(list(result['counts'].values()), [7, 6, 5, 4, 3, 2])
        self.assertEqual(list(result['first_failure_losses'].values()), [1]*5)
        self.assertEqual(result['longest_reliable_run_frames'], 2)
        self.assertEqual(result['longest_reliable_endpoint_span_seconds'], .02)
        self.assertTrue(np.array_equal(arrays['note_agreement'], loaded['masks']['reliable_pitch_screen']))

    def test_mask_mismatch_fails_closed(self):
        loaded, yin = fixture();loaded['masks']['reliable_pitch_screen'][0] = True
        with self.assertRaisesRegex(ValueError, 'does not replay'):
            decompose(loaded, yin)

    def test_misaligned_tracker_fails_closed(self):
        loaded, yin = fixture()
        with self.assertRaisesRegex(ValueError, 'grid mismatch'):
            decompose(loaded, yin[:-1])

    def test_holdout_guard_runs_before_file_access(self):
        with patch('bpdr_input.shared_screen_diagnostics.load_export') as loader:
            with self.assertRaisesRegex(ValueError, 'training partition'):
                diagnose_training_crop(dict(split='final_holdout'))
            loader.assert_not_called()

    def test_nonfinite_pitch_never_becomes_reliable(self):
        loaded, yin = fixture();loaded['features']['f0_hz'][6] = np.inf
        loaded['masks']['reliable_pitch_screen'][6] = False
        result, _ = decompose(loaded, yin)
        self.assertEqual(result['counts']['note_agreement'], 1)
        self.assertEqual(result['first_failure_losses']['finite_trackers'], 2)

    def test_priority_excludes_holdout_ineligible_existing_and_flagged_tracks(self):
        def row(n, artist='a', **kwargs):
            return dict(track_id=str(n), mtg_track_id=f'track_{n:07d}', artist_id=artist,
                        **(dict(split='train',policy_export_eligible=True,bpdr_record_present=False,issues=[])|kwargs))
        evidence = dict(tracks=[row(10),row(9),row(8,split='final_holdout'),row(7,policy_export_eligible=False),
                               row(6,bpdr_record_present=True),row(5,issues=['bad']),row(11,artist='b')])
        selected, first = missing_training_tracks(evidence)
        self.assertEqual([r['track_id'] for r in selected], ['9','10','11'])
        self.assertEqual(first, {'track_0000009','track_0000011'})


if __name__ == '__main__':
    unittest.main()
