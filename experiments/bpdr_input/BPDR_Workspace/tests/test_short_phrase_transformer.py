import unittest
import numpy as np
import torch
from bpdr_input.modeling.short_phrase_transformer import frame_features,tokens_from_features,OFFSETS,ShortPhraseTransformer,predict_raw


class ShortPhraseContractTest(unittest.TestCase):
    def raw(self):
        r=np.zeros((120,71),np.float32);r[:,64]=np.linspace(-70,70,120);r[:,65]=.9
        r[:,66]=1;r[:,68]=-20;return r

    def test_provisional_boundaries_and_nonacoustic_channels_cannot_change_tokens(self):
        r=self.raw();other=r.copy();other[:,:64]=123;other[:,69:]=42
        x,p=tokens_from_features(frame_features(r),[0,60,119])
        y,q=tokens_from_features(frame_features(other),[0,60,119])
        np.testing.assert_array_equal(x,y);np.testing.assert_array_equal(p,q)

    def test_global_transposition_and_gain_are_not_intended_pitch_targets(self):
        r=self.raw();other=r.copy();other[:,64]+=200;other[:,68]+=20
        x,_=tokens_from_features(frame_features(r),[60]);y,_=tokens_from_features(frame_features(other),[60])
        np.testing.assert_allclose(x,y,atol=2e-6,rtol=0)

    def test_center_and_padding_use_actual_timing(self):
        x,p=tokens_from_features(frame_features(self.raw()),[0,60,119])
        center=np.flatnonzero(OFFSETS==0)[0]
        self.assertEqual(int(center),40);self.assertEqual(x.shape,(3,81,8))
        self.assertEqual(int(p[0].sum()),40);self.assertFalse(p[0,center])
        np.testing.assert_array_equal(x[:,center,0],0.)

    def test_unvoiced_is_finite_identity(self):
        torch.set_num_threads(2);model=ShortPhraseTransformer()
        r=np.zeros((35,71),np.float32)
        self.assertTrue(np.isfinite(frame_features(r)).all())
        p,a=predict_raw(model,r)
        np.testing.assert_array_equal(p,0.);np.testing.assert_array_equal(a,0.)

    def test_invalid_shapes_and_values_are_rejected(self):
        with self.assertRaises(ValueError):frame_features(np.zeros((0,71)))
        r=self.raw();r[0,0]=np.nan
        with self.assertRaises(ValueError):frame_features(r)
        with self.assertRaises(ValueError):tokens_from_features(frame_features(self.raw()),[-1])


if __name__=='__main__':unittest.main()
