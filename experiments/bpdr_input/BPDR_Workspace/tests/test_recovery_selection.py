"""Selection reports must retain exclusions and cannot pass on zero edits."""
import unittest
from unittest.mock import patch

import numpy as np

from bpdr_input.experiments.learning.model_recovery import score


class RecoverySelectionTests(unittest.TestCase):
    def pair(self, group):
        times = np.arange(101)*.02
        mask = np.ones(len(times), dtype=bool)
        target = np.zeros(len(times)); target[30:70] = -20.
        analysis = dict(views=[np.zeros((len(times),71))]*3, reliable=mask,
                        protected=np.zeros(len(times),bool), features=dict(times_seconds=times))
        return dict(group=group,source=dict(singer_id=group),source_id=group,
            A=analysis,C=analysis,audio_A=np.zeros(44100,np.float32),audio_C=np.zeros(44100,np.float32),
            mask=mask,target=target,path_A='unused_A',path_C='unused_C',target_path='unused_targets',
            cents=20.,support_seconds=(.6,1.4))

    def zero_proposal(self, views, *_):
        size=len(views[0])
        return dict(requested_curve=np.zeros(size),operating_predictions=np.zeros((3,size)),
                    raw_predictions=np.zeros((3,size)),disagreement=np.zeros(size),reliable=np.ones(size,bool))

    def test_group_counts_do_not_collide_with_aggregate_counts(self):
        pairs=[self.pair('singer_a'),self.pair('singer_b')]
        pairs.append(dict(excluded='AUDIT_FAILED',source=dict(singer_id='singer_a'),cents=-20.))
        with patch('bpdr_input.experiments.learning.model_recovery.propose',side_effect=self.zero_proposal):
            summary,rows=score({},dict(gain=1.,deadband_cents=3.),pairs)
        self.assertEqual(summary['attempted'],3)
        self.assertEqual(summary['audited'],2)
        self.assertEqual(summary['intervention_audit_exclusions'],1)
        self.assertEqual(summary['by_group']['singer_a']['attempted'],2)
        self.assertEqual(len(rows),3)
        self.assertFalse(summary['passes'])
        self.assertEqual(summary['permitted_correction_gain'],0.)


if __name__ == '__main__':
    unittest.main()
