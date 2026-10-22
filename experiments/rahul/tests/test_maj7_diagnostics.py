"""Read-only diagnosis: exact ranks, binary quality metrics and real note cues."""
import ast
from pathlib import Path
import unittest

import nbformat
import torch

from experiments.rahul.src.maj7_diagnostics import (
    chord_parts, binary_quality_metrics, diagnose_logits, melody_features,
)


class Maj7DiagnosticTests(unittest.TestCase):
    def test_quality_parsing_does_not_collapse_extensions_or_inversions(self):
        self.assertEqual(chord_parts('G#:maj7/3'),('G#','maj7'))
        self.assertEqual(chord_parts('G:maj'),('G','maj'))
        self.assertEqual(chord_parts('N'),(None,None))

    def test_exact_rank_and_triad_competition_use_full_softmax(self):
        labels=['PAD','UNK','N','C:maj','C:maj7','D:maj7','C:maj7/3']
        logits=torch.tensor([-3.,-3.,-3.,2.,1.,0.,0.5])
        result=diagnose_logits(logits,4,labels)
        self.assertEqual(result['exact_rank'],2)
        self.assertFalse(result['top1_correct'])
        self.assertTrue(result['top3_correct'])
        self.assertTrue(result['top5_correct'])
        self.assertTrue(result['predicted_corresponding_plain_triad'])
        self.assertAlmostEqual(result['reference_probability'],float(logits.softmax(-1)[4]))
        self.assertAlmostEqual(result['same_root_maj7_probability_mass'],float(logits.softmax(-1)[[4,6]].sum()))
        logits[5]=3
        result=diagnose_logits(logits,4,labels)
        self.assertEqual(result['predicted_label'],'D:maj7')
        self.assertFalse(result['top1_correct'])
        self.assertFalse(result['predicted_same_root_major_any_inversion'])

    def test_rank_ties_keep_reserved_logits_and_break_by_id(self):
        labels=['PAD','UNK','N','C:maj','C:maj7']
        result=diagnose_logits(torch.zeros(5),4,labels)
        self.assertEqual(result['predicted_label'],'PAD')
        self.assertEqual(result['exact_rank'],5)
        self.assertAlmostEqual(result['reference_probability'],0.2)

    def test_binary_quality_metrics_handle_no_positive_predictions(self):
        result=binary_quality_metrics(0,0,926)
        self.assertIsNone(result['precision'])
        self.assertEqual(result['recall'],0)
        self.assertEqual(result['f1'],0)
        result=binary_quality_metrics(2,1,2)
        self.assertEqual(result['precision'],2/3)
        self.assertEqual(result['recall'],1/2)
        self.assertEqual(result['f1'],4/7)

    def test_melody_masks_and_sequence_cue(self):
        inputs={'melody_pitch_ids':torch.tensor([[[73,62],[67,999]]]),
            'melody_note_mask':torch.tensor([[[True,True],[True,False]]]),
            'beat_attention_mask':torch.tensor([[True,True]]),
            'melody_silent':torch.tensor([[False,False]]),
            'melody_timing':torch.zeros(1,2,2,3)}
        first=melody_features(inputs,0,0,0)
        self.assertEqual(first['rounded_midi'],[71,60])
        self.assertTrue(first['major_seventh_pc_in_beat'])
        second=melody_features(inputs,0,1,0)
        self.assertEqual(second['rounded_midi'],[65])
        self.assertFalse(second['major_seventh_pc_in_beat'])
        self.assertTrue(second['major_seventh_pc_in_sequence'])
        inputs['beat_attention_mask'][0,0]=False
        self.assertFalse(melody_features(inputs,0,1,0)['major_seventh_pc_in_sequence'])

    def test_notebook_contains_no_training_operations(self):
        path=Path(__file__).resolve().parents[1]/'notebooks/08_maj7_failure_analysis.ipynb'
        notebook=nbformat.read(path,as_version=4)
        for cell in notebook.cells:
            if cell.cell_type!='code': continue
            tree=ast.parse(cell.source)
            for node in ast.walk(tree):
                if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
                    self.assertNotIn(node.func.attr,('backward','step','zero_grad'))


if __name__=='__main__': unittest.main()
