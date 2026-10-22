import itertools
import unittest
import numpy as np
from bpdr_input.interventions import AuditPolicy, select_support
from bpdr_input.support_geometry import select_multi_interior, geometry_masks, note_pair_ids


class SupportGeometryTests(unittest.TestCase):
    def setUp(self):
        self.t = np.arange(0, 4, .02)
        self.mask = ((self.t >= .4) & (self.t <= .6)) | ((self.t >= 1) & (self.t <= 1.2)) | ((self.t >= 2.5) & (self.t <= 2.9))
        self.ids = np.where(self.t < .8, 1, np.where(self.t < 2, 2, 3))

    def test_bridges_trusted_interiors_without_promoting_gaps(self):
        with self.assertRaises(ValueError): select_support(self.t, self.mask)
        x = select_multi_interior(self.t, self.mask, self.ids, 4, 'development')
        p, o = geometry_masks(self.t, self.mask, x['support_seconds'])
        self.assertGreaterEqual(p.sum(), 10); self.assertGreaterEqual(o.sum(), 10)
        self.assertFalse(np.any((p | o) & ~self.mask))
        self.assertGreaterEqual(len(np.unique(self.ids[p])), 2)

    def test_matches_exhaustive_boolean_oracle(self):
        t = np.arange(0, 4, .08); mask = (np.arange(len(t)) % 3 != 0)
        ids = np.arange(len(t))//5+1; policy = AuditPolicy()
        answer = select_multi_interior(t, mask, ids, 4, 'train', policy)
        choices = []
        for a,b in itertools.combinations(t[(t >= .1) & (t <= 3.9)], 2):
            if b-a < .3+2*(.08+.1): continue
            p,o = geometry_masks(t,mask,(a,b),policy)
            if p.sum() >= 10 and o.sum() >= 10 and len(np.unique(ids[p])) >= 2:
                choices.append(((round(float(b-a),12),-int(min(p.sum(),o.sum())),float(a),float(b)),[a,b]))
        self.assertEqual(answer['support_seconds'],min(choices)[1])

    def test_reserved_partitions_refused(self):
        for split in ['pp1_evaluation','final_holdout','unknown']:
            with self.assertRaisesRegex(ValueError,'Reserved'): select_multi_interior(self.t,self.mask,self.ids,4,split)

    def test_does_not_mutate_inputs_and_is_deterministic(self):
        saved = [a.copy() for a in (self.t,self.mask,self.ids)]
        a = select_multi_interior(self.t,self.mask,self.ids,4,'train')
        self.assertEqual(a,select_multi_interior(self.t,self.mask,self.ids,4,'train'))
        for before,after in zip(saved,(self.t,self.mask,self.ids)): np.testing.assert_array_equal(before,after)

    def test_outside_anchors_required(self):
        t=np.arange(0,.9,.02);mask=np.ones(len(t),bool);ids=np.arange(len(t))//5+1
        self.assertIsNone(select_multi_interior(t,mask,ids,.9,'train'))

    def test_one_note_fragmentation_does_not_count_as_two_notes(self):
        self.assertIsNone(select_multi_interior(self.t,self.mask,np.ones(len(self.t),int),4,'train'))

    def test_empty_reliable_mask_is_refusal(self):
        self.assertIsNone(select_multi_interior(self.t,np.zeros(len(self.t),bool),self.ids,4,'train'))

    def test_invalid_grid_or_mask_refused(self):
        for t,m,ids in [(self.t[::-1],self.mask,self.ids),(self.t,self.mask.astype(int),self.ids),
                         (self.t,self.mask,np.zeros(len(self.t),int))]:
            with self.assertRaises(ValueError): select_multi_interior(t,m,ids,4,'train')

    def test_note_identity_excludes_partial_edges_and_overlap(self):
        note=lambda a,d,partial=False:dict(start_seconds=a,duration_seconds=d,partial=partial)
        notes={'A1':[note(0,1),note(1,1,True)],'A2':[note(0,.8),note(.7,.3)]}
        t=np.array([0,.2,.75,.95,1.3]);x=note_pair_ids(t,notes,.1)
        self.assertEqual(x.tolist(),[0,4,0,0,0])
