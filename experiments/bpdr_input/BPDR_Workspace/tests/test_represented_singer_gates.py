import unittest
from bpdr_input.reporting.represented_singer_gates import assess


class RepresentedSingerGateTests(unittest.TestCase):
    def fixture(self):
        rows=[];controls=[]
        for group in ('female1','male1'):
            for cents in (-30.,-20.,20.,30.):
                rows.append(dict(group=group,cents=cents,delivered_before_mae_cents=25.,delivered_after_mae_cents=0.,delivered_verified_improvement=True))
            controls.append(dict(group=group,raw_original_abs_cents=.1,delivered_identity=True))
        rows.extend(dict(group='female2',cents=c,excluded='NO_VALID_REFERENCE') for c in (-30.,-20.,20.,30.))
        controls.append(dict(group='female2',raw_original_abs_cents=.1,delivered_identity=True))
        return dict(rows=rows),controls

    def test_unscoreable_singer_is_unassessed_and_kept_in_cohort_denominator(self):
        result,controls=self.fixture();gate=assess(result,controls)
        self.assertTrue(gate['passed']);self.assertIsNone(gate['by_singer']['female2']['passed'])
        self.assertEqual(gate['unassessed_singers'],['female2']);self.assertEqual(gate['cohort_attempted'],12)
        self.assertAlmostEqual(gate['cohort_all_attempt_coverage'],8/12)

    def test_failed_original_from_unassessed_singer_still_blocks_all_output_adoption(self):
        result,controls=self.fixture();controls[-1]['delivered_identity']=False
        self.assertFalse(assess(result,controls)['passed'])

    def test_construction_failures_within_represented_singer_stay_in_coverage(self):
        result,controls=self.fixture()
        result['rows'].extend(dict(group='male1',cents=20.,excluded='FAILED_INTERVENTION') for _ in range(20))
        gate=assess(result,controls)
        self.assertFalse(gate['passed']);self.assertAlmostEqual(gate['by_singer']['male1']['all_attempt_coverage'],4/24)

    def test_one_scoreable_category_is_insufficient(self):
        result,controls=self.fixture();result['rows']=[r for r in result['rows'] if r['group']!='male1']
        gate=assess(result,controls)
        self.assertFalse(gate['passed']);self.assertFalse(gate['sufficient_scoreable_representation'])


if __name__=='__main__':unittest.main()
