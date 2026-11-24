import io
import unittest
from unittest.mock import Mock
from unittest.mock import patch
import copy
from pathlib import Path
import zipfile
import numpy as np
import torch
from bpdr_input.real_learning_v2 import relative,hierarchical_weights,balanced_objective
from bpdr_input.humming_intake import RangeReader,select


class LearningV2Tests(unittest.TestCase):
    def test_a4_centers_using_explicit_voicing(self):
        x=np.zeros((4,71),np.float32);x[:3,66]=1.;x[:3,64]=[0.,100.,100.]
        y=relative(x)
        np.testing.assert_array_equal(y[:,64],[-100.,0.,0.,0.])
        np.testing.assert_array_equal(y[:,66],x[:,66])

    def test_repeated_variants_do_not_outweigh_another_singer(self):
        records=[dict(dataset='v',singer_id='a',phrase_id='x') for _ in range(10)]
        records+=[dict(dataset='v',singer_id='b',phrase_id='y')]
        w=hierarchical_weights(records)
        self.assertAlmostEqual(w[:10].sum(),.5);self.assertAlmostEqual(w[10],.5)

    def test_correction_weight_does_not_vanish_with_more_zero_targets(self):
        def loss(n):
            det=torch.zeros((1,n));det[0,0]=30.
            batch=dict(mask=torch.ones((1,n),dtype=torch.bool),detuning=det,
                       voiced={v:torch.zeros((1,n),dtype=torch.bool) for v in 'AC'})
            pred={v:torch.zeros((1,n)) for v in 'ABCD'}
            return float(balanced_objective(pred,batch,'augmentation_only'))
        self.assertAlmostEqual(loss(10),.25);self.assertAlmostEqual(loss(1000),.25)

    def test_normalizer_and_output_gain_remap_preserves_pretrained_function(self):
        from bpdr_input.real_learning_v2 import initialise
        from bpdr_input.model import CorrectionEncoder,ModelConfig
        from bpdr_input.examples import Normalizer
        torch.set_num_threads(2);torch.manual_seed(17)
        old=Normalizer(np.linspace(-10,10,71),np.linspace(2,3,71))
        new=Normalizer(old.mean+.5,old.scale*1.3)
        original=CorrectionEncoder(ModelConfig(hidden=8,layers=1,heads=2,feed_forward=16,dropout=0.)).eval()
        changed=copy.deepcopy(original)
        x=np.random.default_rng(8).normal(size=(1,21,71)).astype(np.float32)
        with patch('bpdr_input.bpdr_showcase.load_model',return_value=(changed,old,71)), \
             patch('bpdr_input.presentation_study.frozen_runs',return_value=[(339,Path('unused'))]), \
             patch('torch.load',return_value={}):
            model,_,_=initialise('bpdr',new)
        with torch.inference_mode():
            np.testing.assert_allclose(30*model(torch.from_numpy(new(x))).numpy(),
                                       original(torch.from_numpy(old(x))).numpy(),atol=2e-5)


class IntakeTests(unittest.TestCase):
    def test_separate_people_and_compositions_and_reserved_people(self):
        keys=[f'{sex}{p:02d}_{p+offset:04d}_0001_1' for sex,offset in [('F',0),('M',100)] for p in range(1,6)]
        # Two distinct compositions for each participant.
        keys+=[k.replace('_0001_1','_0002_1').replace(k.split('_')[1],f'{int(k.split("_")[1])+1000:04d}') for k in list(keys)]
        records,roles=select(keys)
        self.assertEqual(len(records),16)
        self.assertEqual(len(set(r['composition_id'] for r in records)),16)
        train={r['person_id'] for r in records if r['split']=='train'}
        dev={r['person_id'] for r in records if r['split']=='development'}
        self.assertFalse(train & dev);self.assertEqual(list(roles.values()).count('reserved_unseen'),2)

    def test_ranges_can_read_zip_without_full_archive_transfer(self):
        b=io.BytesIO()
        with zipfile.ZipFile(b,'w') as z:z.writestr('one.wav',b'abc')
        data=b.getvalue()
        class Response:
            status_code=206
            def __init__(self,a,b):self.headers={'Content-Range':f'bytes {a}-{b}/{len(data)}'};self.raw=io.BytesIO(data[a:b+1])
            def __enter__(self):return self
            def __exit__(self,*args):pass
        s=Mock()
        def get(url,headers,**kwargs):
            a,b=map(int,headers['Range'][6:].split('-'));return Response(a,b)
        s.get.side_effect=get
        reader=RangeReader(s,'https://example.invalid/test.zip')
        with zipfile.ZipFile(reader) as z:self.assertEqual(z.read('one.wav'),b'abc')
        response=Response(0,0);response.status_code=200;s.get.side_effect=None;s.get.return_value=response
        with self.assertRaisesRegex(ValueError,'full download refused'):RangeReader(s,'https://example.invalid/test.zip')


if __name__=='__main__':unittest.main()
