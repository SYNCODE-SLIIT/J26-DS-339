from copy import deepcopy
import unittest
import numpy as np
from bpdr_input.support_development_packaging import packaging_selection,adapt_private_targets,development_metadata


class SupportDevelopmentPackagingTests(unittest.TestCase):
    def setUp(self):
        phrases=[dict(phrase_id='p'+str(k),source_id='dev'+str(k),singer_id='S'+str(k),split='development') for k in range(14)]
        phrases.append(dict(phrase_id='train',source_id='train',singer_id='T',split='train'))
        self.manifest=dict(phrases=phrases)
        self.audit=dict(status='complete',new_pairs=0,new_optimizer_updates=0,research_ready_examples=0,
            attempts=[dict(phrase_id='p'+str(k),cents=c,status='measured' if k<12 else 'not_attempted_geometry_refused',
                passed=(2*k+j)<7) for k in range(14) for j,c in enumerate([-30.,30.])])
        self.training=[dict(dataset='vocadito',source_id='train',singer_id='T',split='train')]
        self.raw=dict(times_seconds=np.arange(30)*.02,anchor_mask=np.r_[np.ones(20,bool),np.zeros(10,bool)],
            correction_target_cents=np.r_[np.full(10,30.),np.zeros(20)])

    def test_selects_only_passes_and_retains_failed_and_refused_keys(self):
        saved=deepcopy(self.audit);selected,excluded=packaging_selection(self.audit,self.manifest,self.training)
        self.assertEqual((len(selected),len(excluded)),(7,21));self.assertEqual(saved,self.audit)
        self.assertEqual(sum(a['status']=='not_attempted_geometry_refused' for a in excluded),4)

    def test_missing_or_duplicated_audit_key_refused(self):
        self.audit['attempts'][-1]=deepcopy(self.audit['attempts'][0])
        with self.assertRaises(ValueError):packaging_selection(self.audit,self.manifest,self.training)

    def test_training_singer_overlap_refused_before_file_access(self):
        self.training[0]['singer_id']='S0'
        with self.assertRaisesRegex(ValueError,'overlaps'):packaging_selection(self.audit,self.manifest,self.training)

    def test_unattempted_key_cannot_be_promoted(self):
        self.audit['attempts'][-1]['passed']=True
        with self.assertRaises(ValueError):packaging_selection(self.audit,self.manifest,self.training)

    def test_target_adapter_keeps_natural_unlabelled_frames_and_does_not_mutate(self):
        old={k:v.copy() for k,v in self.raw.items()};r=adapt_private_targets(self.raw,-30.)
        self.assertEqual(int(r['detuned_plateau_mask'].sum()),10);self.assertEqual(int(r['zero_anchor_mask'].sum()),10)
        self.assertFalse(r['common_reliable_mask'][20:].any())
        for k in old:np.testing.assert_array_equal(old[k],self.raw[k])
        r['correction_target_cents'][0]=0;self.assertEqual(self.raw['correction_target_cents'][0],30.)

    def test_wrong_sign_or_target_in_gap_refused(self):
        with self.assertRaises(ValueError):adapt_private_targets(self.raw,30.)
        self.raw['correction_target_cents'][25]=30
        with self.assertRaises(ValueError):adapt_private_targets(self.raw,-30.)

    def test_failed_or_short_target_mask_refused(self):
        self.raw['anchor_mask'][:]=False;self.raw['correction_target_cents'][:]=0
        with self.assertRaises(ValueError):adapt_private_targets(self.raw,-30.)

    def test_metadata_cannot_promote_partition_review_or_reference_claim(self):
        r=dict(dataset='vocadito',split='development',training_eligible=False,audit_passed=True,exploratory=True,review_status='pending',
            data_status='exploratory_not_research',reference_type='human_annotation',singer_id='S1',group_id='vocadito:S1',
            group_kind='annotated_singer_id',boundary_encoding='gaussian_max_sigma_0.04_seconds',support_protocol='multi-interior-geometry-0.1',corruption='split')
        development_metadata(r)
        for key,value in [('split','train'),('training_eligible',True),('review_status','approved'),('reference_type','model_estimate'),('corruption','unknown')]:
            wrong=dict(r);wrong[key]=value
            with self.assertRaises(ValueError):development_metadata(wrong)
