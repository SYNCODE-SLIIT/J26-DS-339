import unittest
from bpdr_input.review_bundle import candidates


class ReviewBundleTests(unittest.TestCase):
    def test_withheld_material_is_not_queued_and_approval_is_not_created(self):
        row=dict(record_id='a',screen={'status':'review_candidate'},split='train',training_eligible=False,accepted_audio_path='a.wav')
        withheld=dict(row,record_id='b',screen={'status':'withheld'})
        self.assertEqual(candidates({'records':[row,withheld]}),[row])
        self.assertFalse(row['training_eligible'])

    def test_held_out_or_inconsistent_eligibility_cannot_enter_bundle(self):
        for change in ({'split':'final_holdout'},{'training_eligible':True}):
            row=dict(record_id='a',screen={'status':'review_candidate'},split='train',training_eligible=False,accepted_audio_path='a.wav')
            row.update(change)
            with self.assertRaises(ValueError): candidates({'records':[row]})
