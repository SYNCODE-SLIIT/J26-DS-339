from copy import deepcopy
import unittest
from bpdr_input.support_development_audit import completion_inventory,completion_summary


class DevelopmentSupportAuditTests(unittest.TestCase):
    def setUp(self):
        def row(pid,sid,singer,candidate=True):
            return dict(phrase_id=pid,source_id=sid,singer_id=singer,split='development',training_eligible=False,
                evidence_level='cached_final_reliable_mask_exact_replay',duration_seconds=5,
                candidate=dict(support_seconds=[1.,2.],plateau_reference_frames=10,outside_reference_frames=30) if candidate else None)
        rows=[row('vocadito_2_p00','vocadito_2','S2'),row('vocadito_5_p00','vocadito_5','S5')]
        rows += [row('other_'+str(k),'s'+str(k),'singer'+str(k),k<10) for k in range(12)]
        self.g=dict(phrases=rows);self.p=dict(status='complete',new_pairs=0,new_optimizer_updates=0,research_ready_examples=0,
            sources=[dict(phrase_id=r['phrase_id'],support_seconds=[1.,2.],identity_pass=True) for r in rows[:2]],
            attempts=[dict(phrase_id=r['phrase_id'],cents=c,status='measured',passed=c==-30,audit_path='fixed') for r in rows[:2] for c in [-30.,30.]])

    def terminal(self):
        dev,_,a=completion_inventory(self.g,self.p)
        for r in a:
            if r['origin']=='new_completion':r.update(status='measured',passed=False)
        return dev,a

    def test_inventory_preserves_all_twenty_eight_keys_and_failed_pilot(self):
        before=deepcopy(self.p);dev,new,a=completion_inventory(self.g,self.p)
        self.assertEqual((len(dev),len(new),len(a)),(14,10,28));self.assertEqual(before,self.p)
        self.assertEqual(sum(r['origin']=='verified_pilot_reuse' for r in a),4)
        self.assertEqual(sum(r['origin']=='geometry_refusal' for r in a),4)
        self.assertEqual(sum(r['origin']=='verified_pilot_reuse' and not r['passed'] for r in a),2)

    def test_changed_pilot_support_refused(self):
        self.p['sources'][0]['support_seconds']=[1,3]
        with self.assertRaises(ValueError):completion_inventory(self.g,self.p)

    def test_missing_or_duplicate_pilot_attempt_refused(self):
        self.p['attempts'][-1]=deepcopy(self.p['attempts'][0])
        with self.assertRaises(ValueError):completion_inventory(self.g,self.p)

    def test_missing_development_or_annotation_only_refused(self):
        self.g['phrases'][2]['split']='train'
        with self.assertRaises(ValueError):completion_inventory(self.g,self.p)
        self.g['phrases'][2]['split']='development';self.g['phrases'][2]['evidence_level']='annotation_only'
        with self.assertRaises(ValueError):completion_inventory(self.g,self.p)

    def test_summary_keeps_measured_and_refused_denominators_separate(self):
        dev,a=self.terminal();s=completion_summary(a,dev)
        self.assertEqual((s['prospective_keys'],s['measured_shifts'],s['geometry_refused_shifts']),(28,24,4))
        self.assertEqual((s['reused_measured_shifts'],s['newly_measured_shifts'],s['passed_shifts']),(4,20,2))
        self.assertEqual(s['phrases_with_both_directions_pass'],0)

    def test_missing_or_unfinished_outcome_refused(self):
        dev,a=self.terminal()
        with self.assertRaises(ValueError):completion_summary(a[:-1],dev)
        a[0]['status']='not_started'
        with self.assertRaises(ValueError):completion_summary(a,dev)

    def test_refusal_cannot_be_promoted(self):
        dev,a=self.terminal();a[-1]['passed']=True
        with self.assertRaises(ValueError):completion_summary(a,dev)
