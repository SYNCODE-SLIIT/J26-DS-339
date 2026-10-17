from copy import deepcopy
import unittest
from bpdr_input.training_support_audit import audit_inventory,summarize


class TrainingSupportAuditTests(unittest.TestCase):
    def setUp(self):
        rows=[]
        for i in range(30):
            rows.append(dict(phrase_id=f'p{i:02d}',source_id=f'r{i}',singer_id=f's{i}',split='train',status='prepared',
                training_eligible=False,geometry_available=i<23,multi_interior_candidate=dict(support_seconds=[1,2]) if i<23 else None,
                arrays_directory=f'cache{i}',selected_cache=f'diag{i}'))
        self.e=dict(status='complete',rows=rows);self.m=dict(phrases=deepcopy(rows))
        self.plan=dict(status='declared_not_executed',partition='train',shifts_cents=[-30.,30.],cases=[
            dict(**{k:r[k] for k in ['phrase_id','source_id','singer_id','arrays_directory','selected_cache']},support_seconds=[1,2],cents=c)
            for r in rows[:23] for c in [-30.,30.]])

    def test_full_denominator_and_geometry_refusals_retained_without_mutation(self):
        before=deepcopy(self.e);selected,a=audit_inventory(self.e,self.plan,self.m)
        self.assertEqual(len(selected),23);self.assertEqual(len(a),60)
        self.assertEqual(sum(x['status']=='not_attempted_geometry_refused' for x in a),14)
        self.assertEqual(before,self.e)

    def test_no_missing_duplicate_or_reordered_case(self):
        for cases in [self.plan['cases'][:-1],self.plan['cases']+[self.plan['cases'][0]],list(reversed(self.plan['cases']))]:
            with self.assertRaises(ValueError):audit_inventory(self.e,dict(self.plan,cases=cases),self.m)

    def test_reserved_partition_or_changed_source_refused(self):
        for key,val in [('split','development'),('source_id','other'),('training_eligible',True)]:
            e=deepcopy(self.e);e['rows'][0][key]=val
            with self.assertRaises(ValueError):audit_inventory(e,self.plan,self.m)

    def test_support_cannot_be_reselected_after_outcomes(self):
        p=deepcopy(self.plan);p['cases'][0]['support_seconds']=[1,3]
        with self.assertRaises(ValueError):audit_inventory(self.e,p,self.m)

    def terminal(self):
        _,a=audit_inventory(self.e,self.plan,self.m)
        for r in a:
            if r['status']=='not_started':r['status']='measured'
        return a

    def test_direction_and_group_counts_use_cases_not_boundary_variants(self):
        a=self.terminal();a[0]['passed']=a[1]['passed']=a[2]['passed']=True
        s=summarize(a,self.e['rows'])
        self.assertEqual((s['passed_shifts'],s['negative_passes'],s['positive_passes']),(3,2,1))
        self.assertEqual((s['phrases_with_any_pass'],s['phrases_with_both_directions_pass']),(2,1))

    def test_unfinished_missing_or_falsely_promoted_refusal_rejected(self):
        _,a=audit_inventory(self.e,self.plan,self.m)
        with self.assertRaises(ValueError):summarize(a,self.e['rows'])
        a=self.terminal()
        with self.assertRaises(ValueError):summarize(a[:-1],self.e['rows'])
        a[-1]['passed']=True
        with self.assertRaises(ValueError):summarize(a,self.e['rows'])

    def test_execution_and_identity_refusals_do_not_disappear(self):
        a=self.terminal();a[0]['status']='execution_refused';a[1]['status']='not_attempted_identity_or_source_refused'
        s=summarize(a,self.e['rows'])
        self.assertEqual((s['execution_refused_keys'],s['identity_or_source_refused_keys'],s['measured_shifts']),(1,1,44))
