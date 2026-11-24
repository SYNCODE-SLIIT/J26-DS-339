from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from bpdr_input.modeling.joint_transformer import JointConfig,JointTransformer,predict_raw as dense_predict
from bpdr_input.modeling.candidate_transformer import CandidateTransformer,predict_raw,save,load
from bpdr_input.modeling.step_candidates import measured_candidates,candidate_context


class CandidateTransformerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.parent=JointTransformer(JointConfig(hidden=16,layers=1,feed_forward=32)).eval()
        with torch.no_grad():self.parent.regression.bias.fill_(.2)
        self.raw=np.zeros((350,71),np.float32);self.raw[:,65:67]=1.

    def test_dense_fallback_exact_and_parent_frozen(self):
        model=CandidateTransformer(self.parent);model.train()
        self.assertFalse(model.parent.training)
        np.testing.assert_array_equal(predict_raw(model,self.raw)[0],dense_predict(self.parent,self.raw)[0])
        x,pad,pool=candidate_context(self.raw,(100,160),model.config)
        hidden=model.embed(torch.from_numpy(x[None]),torch.from_numpy(pad[None]),torch.from_numpy(pool[None]))
        loss=(model(hidden)-20.).square().mean();loss.backward()
        self.assertTrue(all(p.grad is None for p in model.parent.parameters()))
        self.assertIsNotNone(model.regression.weight.grad)

    def test_override_only_measured_regions_and_reload(self):
        self.raw[100:170,64]=20.;self.raw[1:,67]=np.diff(self.raw[:,64])
        model=CandidateTransformer(self.parent)
        with torch.no_grad():model.regression.bias.fill_(.7)
        before=dense_predict(self.parent,self.raw)[0];after=predict_raw(model,self.raw)[0]
        covered=np.zeros(len(after),bool)
        for a,b in measured_candidates(self.raw):
            covered[a:b]=True;self.assertTrue(np.all(after[a:b]==after[a]))
        self.assertTrue(covered.any());np.testing.assert_array_equal(before[~covered],after[~covered])
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model.pt';save(path,model,dict(seed=339))
            restored,_=load(path);np.testing.assert_array_equal(after,predict_raw(restored,self.raw)[0])


if __name__=='__main__':unittest.main()
