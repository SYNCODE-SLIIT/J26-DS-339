import unittest
import numpy as np
import torch
from bpdr_input.modeling.step_context import evidence
from bpdr_input.modeling.step_transformer import StepTransformer,StepConfig,predict_raw,windows


class StepTransformerTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)

    def test_measured_step_direction_and_original_drift_control(self):
        raw=np.zeros((500,71),np.float32);raw[:,65:67]=1.;raw[:,64]=np.arange(500)*.01
        self.assertEqual(float(abs(evidence(raw)).max()),0.)
        raw[220:280,64]+=20.;cue=evidence(raw)
        self.assertAlmostEqual(float(cue[250,0]*30),-20.,delta=2.)
        self.assertGreater(cue[250,1],.9)
        self.assertFalse(cue.flags.writeable)

    def test_source_translation_and_boundary_metadata_cannot_change_cues(self):
        raw=np.zeros((500,71),np.float32);raw[:,65:67]=1.;raw[220:280,64]=20.
        changed=raw.copy();changed[:,64]+=700.;changed[:,70]=1.
        np.testing.assert_allclose(evidence(raw),evidence(changed),atol=1e-5)

    def test_transformer_has_attention_and_retains_exact_center(self):
        model=StepTransformer(StepConfig(hidden=16,layers=1,feed_forward=32))
        self.assertTrue(any(isinstance(layer,torch.nn.MultiheadAttention) for layer in model.modules()))
        features=np.arange(15,dtype=np.float32).reshape(5,3)
        x,pad=windows(features,[0,4],model.config)
        np.testing.assert_array_equal(x[:,25],features[[0,4]])
        self.assertTrue(pad[0,0]);self.assertTrue(pad[1,-1])


if __name__=='__main__':unittest.main()
