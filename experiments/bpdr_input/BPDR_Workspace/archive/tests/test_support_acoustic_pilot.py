from copy import deepcopy
import unittest
import numpy as np
from bpdr_input.support_acoustic_pilot import pilot_sources,audited_candidate,EXACT_CHECKS


class SupportAcousticPilotTests(unittest.TestCase):
    def setUp(self):
        self.t=np.arange(0,4,.02);self.mask=np.ones(len(self.t),bool)
        self.before=dict(times_seconds=self.t,f0_hz=np.full(len(self.t),220.),voiced_mask=self.mask.copy())
        self.after={k:v.copy() for k,v in self.before.items()}
        p=(self.t>=1+.08+.10)&(self.t<=2-.08-.10)
        self.after['f0_hz'][p]*=2**(30/1200)
        self.exact={k:True for k in EXACT_CHECKS}
        def row(pid,source,singer):return dict(phrase_id=pid,source_id=source,singer_id=singer,split='development',
            evidence_level='cached_final_reliable_mask_exact_replay',training_eligible=False,duration_seconds=4,
            candidate=dict(support_seconds=[1,2],plateau_reference_frames=10,outside_reference_frames=20))
        self.e=dict(phrases=[row('vocadito_2_p00','vocadito_2','S2'),row('vocadito_2_p01','vocadito_2','S2'),row('vocadito_5_p00','vocadito_5','S5')])

    def call(self,exact=None):
        return audited_candidate(self.before,self.after,self.before['f0_hz'],self.after['f0_hz'],self.mask,(1,2),30.,self.exact if exact is None else exact)

    def test_selection_preserves_named_sources_and_independence(self):
        before=deepcopy(self.e);r=pilot_sources(self.e)
        self.assertEqual([x['phrase_id'] for x in r],['vocadito_2_p00','vocadito_5_p00']);self.assertEqual(before,self.e)

    def test_no_substitute_when_named_source_missing(self):
        self.e['phrases'][0]['candidate']=None
        with self.assertRaisesRegex(ValueError,'no substitute'):pilot_sources(self.e)

    def test_train_or_reserved_sources_cannot_enter(self):
        for split in ['train','pp1_evaluation','final_holdout']:
            e=deepcopy(self.e);e['phrases'][0]['split']=split
            with self.assertRaises(ValueError):pilot_sources(e)

    def test_annotation_only_geometry_not_enough(self):
        self.e['phrases'][0]['evidence_level']='annotation_only_not_final_acoustic_eligibility'
        with self.assertRaises(ValueError):pilot_sources(self.e)

    def test_passed_target_sign_and_unlabelled_regions(self):
        m=self.mask.copy();m[(self.t>1.4)&(self.t<1.6)]=False;self.mask=m
        audit,anchors,target=self.call()
        self.assertTrue(audit['passed']);self.assertTrue(np.all(target[target!=0]==-30))
        self.assertFalse(np.any(anchors&~m));self.assertTrue(np.all(target[~anchors]==0))

    def test_each_physical_failure_clears_all_anchors(self):
        for key in EXACT_CHECKS:
            checks=self.exact.copy();checks[key]=False
            audit,anchors,target=self.call(checks)
            self.assertTrue(audit['core_pitch_audit_pass']);self.assertFalse(audit['passed'])
            self.assertFalse(anchors.any());self.assertFalse(target.any());self.assertIn(key.upper(),audit['failure_reasons'])

    def test_missing_or_non_boolean_physical_checks_refused(self):
        checks=self.exact.copy();checks.pop('identity_pass')
        with self.assertRaises(ValueError):self.call(checks)
        checks=self.exact.copy();checks['identity_pass']=1
        with self.assertRaises(ValueError):self.call(checks)

    def test_acoustic_failure_retained_without_targets(self):
        self.after['f0_hz']=self.before['f0_hz'].copy()
        audit,anchors,target=self.call();self.assertFalse(audit['passed'])
        self.assertIn('FINE_DIRECTION',audit['failure_reasons']);self.assertFalse(anchors.any());self.assertFalse(target.any())

    def test_undeclared_shift_refused(self):
        with self.assertRaises(ValueError):audited_candidate(self.before,self.after,self.before['f0_hz'],self.after['f0_hz'],self.mask,(1,2),20.,self.exact)
