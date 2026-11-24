import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from bpdr_input.comparison_preparation import (balanced_weights, checked_training_records,
    comparison_protocol, fit_streaming_normalizer)


def row(dataset='v', group='g', source='s', phrase='p', cents=10., corruption='split'):
    return dict(dataset=dataset, group_id=group, source_id=source, phrase_id=phrase,
                cents=cents, corruption=corruption, split='train', audit_passed=True,
                exploratory=True, training_eligible=False)


class ComparisonPreparationTests(unittest.TestCase):
    def test_dataset_group_source_phrase_shift_corruption_balance(self):
        records = [row(), row(corruption='merge'), row(cents=20), row(phrase='q'),
                   row(source='t', phrase='r'), row(group='h', source='u', phrase='z'),
                   row(dataset='j', group='artist', source='track', phrase='track')]
        weights = balanced_weights(records)
        np.testing.assert_allclose(weights, [1/64,1/64,1/32,1/16,1/8,1/4,1/2])
        self.assertAlmostEqual(weights.sum(), 1.)

    def test_reject_heldout_before_normalizer_file_io(self):
        r = row();r.update(split='development', input_directory='does-not-exist')
        with self.assertRaisesRegex(ValueError, 'frozen-training'):
            fit_streaming_normalizer([r], [1.])

    def test_reject_research_claim_and_unaudited_source(self):
        for key, value in [('training_eligible',True), ('audit_passed',False), ('exploratory',False)]:
            r = row();r[key] = value
            with self.assertRaises(ValueError):checked_training_records([r])

    def test_duplicate_identity_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate'):balanced_weights([row(), row()])

    def test_weighted_normalizer_does_not_weight_long_phrases_more(self):
        with tempfile.TemporaryDirectory() as temp:
            records = [row(phrase='short'), row(phrase='long')]
            for r, length, value in zip(records, [2,100], [2.,6.]):
                p = Path(temp)/r['phrase_id'];p.mkdir();r['input_directory'] = str(p)
                x = np.full((length,71), value, dtype=np.float32)
                x[:,0] = 9.
                for view in 'ABCD':np.savez_compressed(p/(view+'_input.npz'), boundary_features_71=x)
            result = fit_streaming_normalizer(records, [.5,.5])
            np.testing.assert_allclose(result['mean'][1:], 4.)
            np.testing.assert_allclose(result['scale'][1:], 2.)
            self.assertEqual(result['scale'][0],1.)
            self.assertEqual(result['real_view_frames'],408)
            # All three conditions share these moments; frame-only truncates exactly.
            self.assertEqual(len(result['mean'][:70]),70)

    def test_invalid_normalizer_weights_refused_before_file_io(self):
        for weights in [[0.],[-1.],[float('nan')],[.5],[]]:
            with self.assertRaisesRegex(ValueError,'weights'):fit_streaming_normalizer([row()],weights)

    def test_sampling_schedule_reproducible_across_conditions(self):
        weights = balanced_weights([row(),row(dataset='j')])
        schedules = [np.random.default_rng(339).choice(2,(300,8),p=weights) for _ in range(3)]
        for schedule in schedules[1:]:np.testing.assert_array_equal(schedule,schedules[0])

    def test_locked_protocol_keeps_development_and_holdout_roles_explicit(self):
        p = comparison_protocol()
        self.assertEqual(p['conditions'],dict(augmentation_only=71,frame_only=70,bpdr=71))
        self.assertEqual(p['checkpoint_selection']['partition'],'development')
        self.assertIn('Stop selection',p['checkpoint_selection']['no_development_pairs'])
        self.assertFalse(p['development_processing_executed'])
        self.assertEqual(p['training_updates_executed'],0)


if __name__ == '__main__':unittest.main()
