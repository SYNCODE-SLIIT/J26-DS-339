"""Meaningful permission failures, exact preservation and disposable callback buffers."""
import unittest
import numpy as np
from bpdr_input.repair_controller import plan_fixture, transact_fixture


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.sr = 22050
        self.audio = np.full(4*self.sr, .1, dtype=np.float32)
        self.times = np.arange(len(self.audio)//441+1)*.02
        self.cents = np.where((self.times>=1.) & (self.times<1.9), -20., 0.)
        self.reliable = np.ones(len(self.times),dtype=bool)
        self.protected = np.zeros(len(self.times),dtype=bool)

    def plan(self, cents=None, alternate=None, uncertainty=None, reliable=None, protected=None):
        c = self.cents if cents is None else cents
        return plan_fixture(self.audio,self.sr,self.times,c,c if alternate is None else alternate,
            np.ones(len(c)) if uncertainty is None else uncertainty,
            self.reliable if reliable is None else reliable,self.protected if protected is None else protected)

    def test_identity_does_not_call_renderer_or_auditor(self):
        plan = self.plan(np.zeros(len(self.times)))
        def forbidden(*args): raise AssertionError('Identity called an edit stage')
        output,result = transact_fixture(self.audio,self.sr,plan,forbidden,forbidden)
        self.assertTrue(np.array_equal(output,self.audio))
        self.assertEqual(result['outcome'],'IDENTITY')
        self.assertIsNone(result['accepted_audio'])

    def test_limits_withhold_whole_proposal_without_clipping(self):
        for c,reason in [(self.cents*2,'CORRECTION_LIMIT'),
                         (np.where(self.times<2.,-20.,0.),'AFFECTED_FRACTION_LIMIT'),
                         (np.where((self.times>=1.)&(self.times<1.2),-20.,0.),'INSUFFICIENT_SUPPORT')]:
            plan = self.plan(c)
            self.assertFalse(plan['permitted'])
            self.assertIn(reason,plan['reasons'])
            self.assertEqual(plan['intervals_seconds'],[])

    def test_disagreement_uncertainty_unvoiced_and_protected_are_independent_gates(self):
        unreliable = self.reliable.copy(); unreliable[60]=False
        protected = self.protected.copy(); protected[60]=True
        for kwargs,reason in [({'alternate':-self.cents},'BOUNDARY_DISAGREEMENT'),
                              ({'uncertainty':np.full(len(self.times),6.)},'UNCERTAINTY_LIMIT'),
                              ({'reliable':unreliable},'UNRELIABLE_VOICING'),
                              ({'protected':protected},'PROTECTED_REGION')]:
            self.assertIn(reason,self.plan(**kwargs)['reasons'])

    def test_missing_uncertainty_never_grants_an_edit(self):
        plan = plan_fixture(self.audio,self.sr,self.times,self.cents,self.cents,None,self.reliable,self.protected)
        self.assertFalse(plan['permitted'])

    def test_changed_source_cannot_reuse_a_plan(self):
        plan = self.plan()
        changed = self.audio.copy(); changed[0]=.2
        with self.assertRaises(ValueError):
            transact_fixture(changed,self.sr,plan,None,None)

    def test_modified_permission_intervals_cannot_reuse_a_decision(self):
        plan = self.plan()
        plan['intervals_seconds'] = [(0.,4.)]
        with self.assertRaises(ValueError):
            transact_fixture(self.audio,self.sr,plan,None,None)

    def test_bad_render_count_nonfinite_and_outside_change_roll_back(self):
        plan = self.plan()
        def outside(x,*args): x[0]=.2; return x
        def nonfinite(x,*args): x[30000]=np.nan; return x
        for renderer in (lambda x,*a:x[:-1], outside, nonfinite):
            output,result=transact_fixture(self.audio,self.sr,plan,renderer,lambda *a:{'passed':True})
            self.assertEqual(result['outcome'],'ROLLED_BACK')
            self.assertTrue(np.array_equal(output,self.audio))
            self.assertIsNone(result['accepted_audio'])

    def test_failed_post_audit_discards_candidate_and_preserves_input(self):
        plan=self.plan()
        def renderer(x,*args): x[30000]=.2; return x
        original=self.audio.copy()
        output,result=transact_fixture(self.audio,self.sr,plan,renderer,lambda *a:{'passed':False})
        self.assertTrue(np.array_equal(output,original))
        self.assertTrue(np.array_equal(self.audio,original))
        self.assertEqual(result['outcome'],'ROLLED_BACK')

    def test_callbacks_cannot_mutate_preserved_source_or_returned_candidate(self):
        def renderer(x,*args): x[30000]=.2; return x
        def auditor(before,after,*args): before[:]=0.; after[:]=0.; return {'passed':True,'scope':'test_stub_only'}
        output,result=transact_fixture(self.audio,self.sr,self.plan(),renderer,auditor)
        self.assertEqual(result['outcome'],'VERIFIED_FIXTURE_CANDIDATE')
        self.assertEqual(output[30000],np.float32(.2))
        self.assertTrue(np.all(self.audio==np.float32(.1)))
        self.assertFalse(result['production_eligible'])
        self.assertIsNone(result['accepted_audio'])


if __name__=='__main__': unittest.main()
