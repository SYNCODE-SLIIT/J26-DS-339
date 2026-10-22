import unittest
import numpy as np
from bpdr_input.phase_regression_diagnostics import diagnose_saved,runs


def fixture(n=22050):
    support=np.zeros(n,dtype=bool);support[2205:n-2205]=True
    frozen=dict(sample_reliable=np.ones(n,dtype=bool),guarded_support=support.copy(),retained_support=support.copy(),
                estimated_frequency_hz=np.full(n,200.),estimated_amplitude=np.ones(n))
    local={k:frozen[k].copy() for k in ('sample_reliable','guarded_support','retained_support')}
    local.update(local_frequency_hz=np.full(n,200.),local_amplitude=np.ones(n))
    islands=[dict(start_sample=0,end_sample=n,status='computed',local_center_hz=200.)]
    return frozen,local,islands


class PhaseRegressionDiagnosticsTests(unittest.TestCase):
    def test_low_amplitude_failure_is_counted_without_removing_support(self):
        a,c,i=fixture();c['local_frequency_hz'][7000:7002]=-2;c['local_amplitude'][7000:7002]=.01
        r=diagnose_saved(a,c,i,200.)
        self.assertEqual(r['local_bad_on_retained'],2)
        self.assertEqual(r['raw_islands'][0]['bad_runs'],[dict(start_sample=7000,end_sample=7002,samples=2)])
        self.assertEqual(r['retained_samples'],17640)
        self.assertAlmostEqual(r['raw_islands'][0]['local_bad_amplitude_relative_to_guarded_median']['maximum'],.01)
        self.assertIsNone(r['synthetic_error_by_edge_distance'])

    def test_edge_bins_partition_retained_samples_and_keep_common_domain(self):
        a,c,i=fixture();c['local_frequency_hz'][8000]=np.nan
        r=diagnose_saved(a,c,i,200.,np.full(22050,200.),np.ones(22050,dtype=bool))['synthetic_error_by_edge_distance']
        self.assertEqual(sum(b['retained_samples'] for b in r['bins']),17640)
        self.assertEqual(sum(b['common_valid_voiced_samples'] for b in r['bins']),17639)
        self.assertEqual(r['invalid_C_on_retained'],1)
        self.assertEqual(r['bins'][0]['retained_samples'],0)

    def test_center_comparison_and_inputs_are_immutable(self):
        a,c,i=fixture();before={k:v.copy() for k,v in c.items()};r=diagnose_saved(a,c,i,200.)
        self.assertTrue(r['raw_islands'][0]['center_exactly_equals_global'])
        self.assertEqual(r['raw_islands'][0]['center_difference_cents'],0.)
        for k in c:np.testing.assert_array_equal(c[k],before[k])

    def test_changed_mask_or_incomplete_truth_is_rejected(self):
        a,c,i=fixture();c['retained_support'][0]=True
        with self.assertRaises(ValueError):diagnose_saved(a,c,i,200.)
        a,c,i=fixture()
        with self.assertRaises(ValueError):diagnose_saved(a,c,i,200.,np.ones(22050))

    def test_bad_runs_preserve_half_open_endpoints(self):
        self.assertEqual(runs(np.array([1,0,1,1],dtype=bool)),[
            dict(start_sample=0,end_sample=1,samples=1),dict(start_sample=2,end_sample=4,samples=2)])
