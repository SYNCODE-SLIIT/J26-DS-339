"""Quality mapping, shared initialization, two-head masks/gradients and reload."""
from pathlib import Path
import tempfile
import unittest

import torch
from torch.nn import functional as F

from experiments.rahul.src.inspect_transformer import DEFAULT_DATASET, mixed_length_examples
from experiments.rahul.src.model import DatasetSpec, ModelConfig, ChordRefinementTransformer
from experiments.rahul.src.model.multitask_transformer import (
    QualityMapping, MultiTaskChordTransformer, multitask_loss,
    save_multitask_checkpoint, load_multitask_checkpoint,
)
from experiments.rahul.src.transformer_dataset import collate_sequences


class MultiTaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.spec=DatasetSpec.from_folder(DEFAULT_DATASET)
        cls.batch=collate_sequences(mixed_length_examples(DEFAULT_DATASET))

    def settings(self):
        return ModelConfig(d_model=32,d_ff=64,chord_dim=8,melody_dim=8,
                           harmonic_dim=8,genre_temporal_dim=8,dropout=0,use_absolute_timing=True)

    def test_mapping_inversions_no_chord_and_specials(self):
        labels=['PAD','UNK','N','G:maj7','C:maj7/3','A:min7/b7','D:7','unknown']
        mapping=QualityMapping.from_chords(labels)
        self.assertEqual(mapping,QualityMapping.from_chords(labels))
        self.assertEqual(mapping.chord_to_quality[0],-100)
        self.assertEqual(mapping.chord_to_quality[1],-100)
        self.assertEqual(mapping.chord_to_quality[-1],-100)
        self.assertEqual(mapping.labels[mapping.chord_to_quality[2]],'N')
        self.assertEqual(mapping.chord_to_quality[3],mapping.chord_to_quality[4])
        self.assertEqual(mapping.labels[mapping.chord_to_quality[5]],'min7')

    def test_shared_initialization_and_rng_match_fresh_baseline(self):
        torch.manual_seed(42)
        original=ChordRefinementTransformer(self.spec,self.settings()).eval()
        original_rng=torch.get_rng_state().clone()
        torch.manual_seed(42)
        model=MultiTaskChordTransformer(self.spec,self.settings()).eval()
        self.assertTrue(torch.equal(original_rng,torch.get_rng_state()))
        for name,value in original.state_dict().items():
            torch.testing.assert_close(value,model.state_dict()[name],rtol=0,atol=0)
        output=model(self.batch['inputs'])
        torch.testing.assert_close(original(self.batch['inputs']),output['chord_logits'],rtol=0,atol=0)

    def test_head_shapes_padding_and_variable_length(self):
        model=MultiTaskChordTransformer(self.spec,self.settings()).eval()
        output=model(self.batch['inputs'],True)
        self.assertEqual(output['chord_logits'].shape,(3,20,self.spec.num_target_classes))
        self.assertEqual(output['quality_logits'].shape,(3,20,len(model.quality_mapping.labels)))
        self.assertEqual(output['details']['encoded'].shape,(3,20,32))
        padding=~self.batch['inputs']['beat_attention_mask']
        for key in ('chord_logits','quality_logits'):
            self.assertTrue((output[key][padding]==0).all())
            self.assertTrue(torch.isfinite(output[key]).all())

    def test_both_heads_and_shared_parameters_receive_finite_gradients(self):
        model=MultiTaskChordTransformer(self.spec,self.settings())
        output=model(self.batch['inputs'])
        loss=multitask_loss(output,self.batch['targets'],self.batch['inputs']['beat_attention_mask'],model.chord_to_quality)
        torch.testing.assert_close(loss['total_loss'],loss['chord_loss']+0.5*loss['quality_loss'])
        loss['total_loss'].backward()
        for name,p in model.named_parameters():
            self.assertIsNotNone(p.grad,name)
            self.assertTrue(torch.isfinite(p.grad).all(),name)
        self.assertGreater(float(model.classifier.weight.grad.norm()),0)
        self.assertGreater(float(model.quality_classifier.weight.grad.norm()),0)
        self.assertGreater(float(model.layers[0].attention.query.weight.grad.norm()),0)

    def test_losses_ignore_masks_reserved_and_unmapped_auxiliary_only(self):
        output={'chord_logits':torch.randn(1,6,8,requires_grad=True),
                'quality_logits':torch.randn(1,6,3,requires_grad=True)}
        targets={'target_chord_ids':torch.tensor([[3,4,-100,0,1,7]]),
                 'target_loss_mask':torch.tensor([[True,True,False,True,True,True]])}
        beat_mask=torch.ones(1,6,dtype=torch.bool)
        mapping=torch.tensor([-100,-100,0,1,2,1,1,-100])
        loss=multitask_loss(output,targets,beat_mask,mapping)
        torch.testing.assert_close(loss['chord_loss'],F.cross_entropy(output['chord_logits'][0,[0,1,5]],torch.tensor([3,4,7])))
        torch.testing.assert_close(loss['quality_loss'],F.cross_entropy(output['quality_logits'][0,:2],torch.tensor([1,2])))
        self.assertEqual(int(loss['chord_mask'].sum()),3)
        self.assertEqual(int(loss['quality_mask'].sum()),2)
        loss['total_loss'].backward()
        self.assertTrue((output['quality_logits'].grad[0,2:]==0).all())

    def test_empty_masks_safe_differentiable_and_lambda_configurable(self):
        output={'chord_logits':torch.randn(1,2,5,requires_grad=True),
                'quality_logits':torch.randn(1,2,2,requires_grad=True)}
        targets={'target_chord_ids':torch.full((1,2),-100),'target_loss_mask':torch.zeros(1,2,dtype=torch.bool)}
        loss=multitask_loss(output,targets,torch.ones(1,2,dtype=torch.bool),[-100,-100,0,1,1],lambda_quality=0.7)
        self.assertEqual(float(loss['total_loss'].detach()),0)
        loss['total_loss'].backward()
        self.assertTrue((output['chord_logits'].grad==0).all())
        self.assertTrue((output['quality_logits'].grad==0).all())
        with self.assertRaises(ValueError):
            multitask_loss(output,targets,torch.ones(1,2,dtype=torch.bool),[-100,-100,0,1,1],lambda_quality=-1)

    def test_no_reference_fields_read_by_forward(self):
        model=MultiTaskChordTransformer(self.spec,self.settings()).eval()
        inputs=dict(self.batch['inputs'])
        expected=model(inputs)
        inputs['reference_chord']='must not be read'
        inputs['quality_targets']='must not be read'
        actual=model(inputs)
        for key in ('chord_logits','quality_logits'):
            torch.testing.assert_close(expected[key],actual[key],rtol=0,atol=0)

    def test_checkpoint_roundtrip_and_prediction_utility(self):
        model=MultiTaskChordTransformer(self.spec,self.settings()).eval()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'multi.pt'
            save_multitask_checkpoint(path,model)
            restored=load_multitask_checkpoint(path,DEFAULT_DATASET)
            for key in ('chord_logits','quality_logits'):
                torch.testing.assert_close(model(self.batch['inputs'])[key],restored(self.batch['inputs'])[key],rtol=0,atol=0)
            self.assertEqual(len(restored.predict_topk(self.batch['inputs'])[0][0]),3)
            self.assertEqual(restored.predict_topk(self.batch['inputs'])[0][-1],[])


if __name__=='__main__': unittest.main()
