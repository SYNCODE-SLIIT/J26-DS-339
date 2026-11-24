from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from bpdr_input.modeling.joint_transformer import JointConfig,JointTransformer
from bpdr_input.modeling.candidate_transformer import CandidateTransformer,predict_raw
from bpdr_input.modeling.candidate_evidence_transformer import EvidenceCandidateTransformer,save,load
from bpdr_input.modeling.step_candidates import candidate_context,measured_candidates


class CandidateEvidenceTests(unittest.TestCase):
    def test_initial_function_preserved_new_cues_learn_and_parent_stays_frozen(self):
        torch.set_num_threads(2)
        parent=JointTransformer(JointConfig(hidden=16,layers=1,feed_forward=32)).eval()
        with torch.no_grad():parent.regression.bias.fill_(.2)
        base=CandidateTransformer(parent);new=EvidenceCandidateTransformer(parent)
        raw=np.zeros((350,71),np.float32);raw[:,65:67]=1.;raw[100:170,64]=20.;raw[1:,67]=np.diff(raw[:,64])
        np.testing.assert_allclose(predict_raw(base,raw)[0],predict_raw(new,raw)[0],atol=1e-5)
        x,pad,pool=candidate_context(raw,measured_candidates(raw)[0],new.config)
        out=new(new.embed(torch.from_numpy(x[None]),torch.from_numpy(pad[None]),torch.from_numpy(pool[None])))
        (out-20.).square().mean().backward()
        self.assertGreater(float(abs(new.regression.weight.grad[:,-3:]).sum()),0.)
        self.assertTrue(all(p.grad is None for p in parent.parameters()))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model.pt';save(path,new,dict(seed=339))
            restored,_=load(path);np.testing.assert_array_equal(predict_raw(new,raw)[0],predict_raw(restored,raw)[0])


if __name__=='__main__':unittest.main()
