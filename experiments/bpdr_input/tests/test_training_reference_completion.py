import copy
import unittest
from bpdr_input.training_reference_completion import completion_phrases, merge_completion


class TrainingReferenceCompletionTests(unittest.TestCase):
    def setUp(self):
        self.phrases=[dict(phrase_id=f'p{i}',source_id=f'r{i}',singer_id=f's{i}',split='train') for i in range(2)]
        self.rows=[dict(self.phrases[0],status='prepared',training_eligible=False,geometry_available=False,reliable_frames=0),
            dict(self.phrases[1],status='unavailable',training_eligible=False,geometry_available=None,reliable_frames=None)]
        self.manifest=dict(phrases=self.phrases+[dict(phrase_id='reserved',split='final_holdout')])

    def test_only_missing_training_selected_and_no_mutation(self):
        before=copy.deepcopy(self.rows)
        self.assertEqual(completion_phrases(self.manifest,self.rows),[self.phrases[1]])
        self.assertEqual(before,self.rows)

    def test_missing_duplicate_or_foreign_rows_refused(self):
        for rows in [self.rows[:1],self.rows+self.rows[:1],self.rows+[dict(self.rows[0],phrase_id='reserved')]]:
            with self.assertRaises(ValueError):completion_phrases(self.manifest,rows)

    def test_partition_or_singer_drift_refused(self):
        for key,value in [('split','development'),('singer_id','changed')]:
            rows=copy.deepcopy(self.rows);rows[1][key]=value
            with self.assertRaises(ValueError):completion_phrases(self.manifest,rows)

    def test_success_preserves_previous_refusal_and_complete_inventory(self):
        addition=dict(self.rows[1],status='prepared',geometry_available=True,reliable_frames=40)
        result=merge_completion(self.rows,[addition])
        self.assertEqual(result[0],self.rows[0]);self.assertEqual(result[1],addition)
        with self.assertRaises(ValueError):merge_completion(self.rows,[])
        with self.assertRaises(ValueError):merge_completion(self.rows,[addition,addition])

    def test_technical_refusal_is_retained_without_inventing_masks_or_approval(self):
        refusal=dict(self.rows[1],status='technical_refusal',reason_codes=['SILENT_RECORDING'])
        self.assertEqual(merge_completion(self.rows,[refusal])[1],refusal)
        for change in [dict(training_eligible=True),dict(reliable_frames=0),dict(geometry_available=False)]:
            with self.assertRaises(ValueError):merge_completion(self.rows,[dict(refusal,**change)])

    def test_already_prepared_rows_and_changed_sources_cannot_be_replaced(self):
        addition=dict(self.rows[1],status='prepared',geometry_available=False,reliable_frames=0)
        with self.assertRaises(ValueError):merge_completion(self.rows,[addition,self.rows[0]])
        with self.assertRaises(ValueError):merge_completion(self.rows,[dict(addition,source_id='other')])
