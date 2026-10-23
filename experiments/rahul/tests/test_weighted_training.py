"""Weighted loss, training-only masking and controlled evaluation definitions."""
import ast
from pathlib import Path
import unittest

import nbformat
import torch
from torch.nn import functional as F

from experiments.rahul.src.weighted_training import (
    supervised_mask, count_training_targets, sqrt_inverse_weights,
    weighted_masked_cross_entropy, metrics_from_confusion, quality_metrics,
)


class WeightedTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.logits=torch.randn(1,5,5,requires_grad=True)
        self.targets={'target_chord_ids':torch.tensor([[2,3,4,-100,1]]),
                      'target_loss_mask':torch.tensor([[True,True,True,False,True]])}
        self.beats=torch.tensor([[True,True,False,False,True]])

    def test_training_counts_use_both_masks_and_keep_n(self):
        batch={'inputs':{'beat_attention_mask':self.beats},'targets':self.targets}
        counts=count_training_targets([batch],5)
        self.assertEqual(counts.tolist(),[0,0,1,1,0])

    def test_formula_cap_absent_and_full_size(self):
        counts=torch.tensor([0,0,1000,4,1])
        actual=sqrt_inverse_weights(counts)
        expected=torch.tensor([0,0,(1005/(3*1000))**0.5,5,5])
        torch.testing.assert_close(actual,expected)
        self.assertEqual(actual.shape,counts.shape)
        with self.assertRaises(ValueError): sqrt_inverse_weights(torch.zeros(5,dtype=torch.int64))

    def test_weighted_ce_equals_pytorch_and_ignores_inactive_reserved(self):
        weights=torch.tensor([0.,0.,0.5,3.,5.])
        actual=weighted_masked_cross_entropy(self.logits,self.targets,self.beats,weights)
        expected=F.cross_entropy(self.logits[0,:2],self.targets['target_chord_ids'][0,:2],weight=weights)
        torch.testing.assert_close(actual,expected)
        changed=self.logits.detach().clone().requires_grad_()
        with torch.no_grad(): changed[0,2:]=1000
        torch.testing.assert_close(actual,weighted_masked_cross_entropy(changed,self.targets,self.beats,weights))
        actual.backward()
        self.assertTrue(torch.isfinite(self.logits.grad).all())
        self.assertTrue((self.logits.grad[0,2:]==0).all())

    def test_empty_and_zero_weight_batches_are_finite_zero(self):
        for weights,mask in [(torch.zeros(5),self.beats),(torch.ones(5),torch.zeros_like(self.beats))]:
            loss=weighted_masked_cross_entropy(self.logits,self.targets,mask,weights)
            self.assertEqual(float(loss.detach()),0)
            loss.backward()
            self.assertTrue(torch.isfinite(self.logits.grad).all())

    def test_reject_invalid_active_targets_and_weights(self):
        self.targets['target_chord_ids'][0,0]=5
        with self.assertRaises(ValueError): supervised_mask(self.targets,self.beats,5)
        with self.assertRaises(ValueError):
            weighted_masked_cross_entropy(self.logits,self.targets,self.beats,torch.ones(4))

    def test_metrics_macro_policy_zero_support_and_exact_quality(self):
        labels=['PAD','UNK','N','C:maj','C:maj7','D:min7']
        counts=torch.zeros(6,6,dtype=torch.int64)
        counts[2,2]=2
        counts[3,3]=3
        counts[3,4]=1
        counts[4,3]=2
        summary,rows=metrics_from_confusion(counts,labels)
        self.assertEqual(summary['valid_targets'],8)
        self.assertEqual(summary['accuracy'],5/8)
        self.assertEqual(summary['macro_classes'],3)
        self.assertFalse(rows[5]['included_in_macro'])
        self.assertIsNone(rows[5]['recall'])
        self.assertEqual(rows[4]['f1'],0)
        self.assertAlmostEqual(summary['macro_f1'],(1+2/3+0)/3)
        self.assertAlmostEqual(summary['weighted_f1'],(2*1+4*(2/3)+2*0)/8)
        qualities={r['reference_quality']:r for r in quality_metrics(counts,labels)}
        self.assertEqual(qualities['maj']['exact_chord_accuracy'],3/4)
        self.assertEqual(qualities['maj7']['exact_chord_accuracy'],0)
        self.assertIsNone(qualities['6']['exact_chord_accuracy'])

    def test_notebook_has_separate_outputs_same_settings_and_unweighted_selection(self):
        p=Path(__file__).resolve().parents[1]/'notebooks/07_train_transformer_weighted.ipynb'
        notebook=nbformat.read(p,as_version=4)
        for cell in notebook.cells:
            if cell.cell_type=='code': ast.parse(cell.source)
        settings=next(c.source for c in notebook.cells if 'weighted_settings' in c.metadata.get('tags',[]))
        loop=next(c.source for c in notebook.cells if 'weighted_training_loop' in c.metadata.get('tags',[]))
        self.assertIn("baseline_config['model']",''.join(c.source for c in notebook.cells))
        self.assertIn('weighted_transformer',settings)
        self.assertIn('best_weighted.pt',loop)
        self.assertIn("scheduler.step(validation_metrics['loss'])",loop)
        self.assertIn("validation_metrics['loss']<best_loss",loop.replace(' ',''))


if __name__=='__main__': unittest.main()
