from copy import deepcopy
import unittest
from bpdr_input.training_support_packaging import select_cases,candidate_metadata


class TrainingSupportPackagingTests(unittest.TestCase):
    def setUp(self):
        self.phrases=[dict(phrase_id=f'p{i}',source_id=f'r{i}',singer_id=f's{i}',split='train') for i in range(30)]
        attempts=[dict(phrase_id=p['phrase_id'],cents=c,status='measured',passed=i<14) for i,p in enumerate(self.phrases) for c in [-30.,30.]]
        self.audit=dict(status='complete',new_pairs=0,new_optimizer_updates=0,research_ready_examples=0,
            sources=[dict(p,identity_pass=True) for p in self.phrases],attempts=attempts)
        self.manifest=dict(phrases=deepcopy(self.phrases)+[dict(phrase_id='dev',source_id='reserved',singer_id='reserved',split='development')])

    def test_passed_selection_retains_every_excluded_key_without_mutation(self):
        before=deepcopy(self.audit);selected,excluded=select_cases(self.audit,self.manifest,[dict(split='train')])
        self.assertEqual((len(selected),len(excluded)),(28,32));self.assertEqual(before,self.audit)

    def test_missing_or_duplicate_keys_refused(self):
        for attempts in [self.audit['attempts'][:-1],self.audit['attempts']+[self.audit['attempts'][0]]]:
            with self.assertRaises(ValueError):select_cases(dict(self.audit,attempts=attempts),self.manifest,[])

    def test_blocked_or_identity_failed_case_cannot_be_promoted(self):
        a=deepcopy(self.audit);a['attempts'][0]['status']='execution_refused'
        with self.assertRaises(ValueError):select_cases(a,self.manifest,[])
        a=deepcopy(self.audit);a['sources'][0]['identity_pass']=False
        with self.assertRaises(ValueError):select_cases(a,self.manifest,[])

    def test_development_singer_and_recording_leakage_refused(self):
        for key in ['source_id','singer_id']:
            m=deepcopy(self.manifest);m['phrases'][-1][key]=self.phrases[0][key]
            with self.assertRaises(ValueError):select_cases(self.audit,m,[])

    def test_unfinished_or_research_promoted_audit_refused(self):
        with self.assertRaises(ValueError):select_cases(dict(self.audit,research_ready_examples=1),self.manifest,[])
        a=deepcopy(self.audit);a['attempts'][-1]['status']='started'
        with self.assertRaises(ValueError):select_cases(a,self.manifest,[])

    def test_legacy_partition_drift_refused(self):
        with self.assertRaises(ValueError):select_cases(self.audit,self.manifest,[dict(split='development')])

    def test_candidate_metadata_does_not_invent_adoption_or_review(self):
        r=dict(dataset='vocadito',split='train',audit_passed=True,exploratory=True,training_eligible=False,
            review_status='pending',data_status='exploratory_not_research',reference_type='human_annotation',
            group_kind='annotated_singer_id',group_id='vocadito:s1',singer_id='s1',boundary_encoding='gaussian_max_sigma_0.04_seconds',
            support_protocol='multi-interior-geometry-0.1',corruption='split',adoption_status='candidate_not_adopted')
        candidate_metadata(r)
        for change in [dict(training_eligible=True),dict(review_status='approved'),dict(adoption_status='adopted'),dict(split='development')]:
            with self.assertRaises(ValueError):candidate_metadata(dict(r,**change))
