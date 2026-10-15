import unittest
from unittest.mock import patch
import numpy as np
from bpdr_input.source_transfer import coverage_summary,screen_crop,validate_selection


class SourceTransferTests(unittest.TestCase):
    def test_uncomputed_support_is_unavailable_not_an_invented_zero(self):
        r=coverage_summary({},100)
        self.assertIsNone(r['retained_support_samples']);self.assertIsNone(r['nonpositive_total'])

    def test_predicted_coverage_and_bad_frequency_inside_support_are_distinct(self):
        a=dict(sample_reliable=np.array([1,1,0,0],dtype=bool),guarded_support=np.array([0,1,0,0],dtype=bool),
            retained_support=np.array([0,1,0,0],dtype=bool),estimated_frequency_hz=np.array([-1.,-2.,220.,220.]))
        r=coverage_summary(a,4)
        self.assertEqual(r['retained_support_fraction'],.25);self.assertEqual(r['nonpositive_on_retained'],1)
        self.assertEqual(r['nonpositive_outside_retained'],1)

    def test_refused_prediction_does_not_invoke_fitter_or_lose_arrays(self):
        audio=np.ones(10,dtype=np.float32);a=dict(frame_reliable=np.zeros(1,dtype=bool))
        prediction=dict(status='refused',reasons=['NO_RELIABLE_SOURCE_FRAMES'],arrays=a)
        with patch('bpdr_input.source_transfer.estimate_source',return_value=prediction) as estimator,patch('bpdr_input.source_transfer.separate_source') as fitter:
            r=screen_crop(audio,22050)
        fitter.assert_not_called();self.assertEqual(len(estimator.call_args.args),2)
        self.assertIs(r['prediction_arrays'],a);self.assertIsNone(r['decomposition_arrays'])
        self.assertEqual(r['decomposition']['status'],'WITHHELD_ESTIMATOR_REFUSAL')

    def test_estimated_prediction_reaches_fitter_without_dataset_metadata(self):
        audio=np.ones(10,dtype=np.float32);a=dict(retained_support=np.ones(10,dtype=bool))
        prediction=dict(status='estimated',reasons=[],arrays=a);parts=dict(test=np.ones(10))
        with patch('bpdr_input.source_transfer.estimate_source',return_value=prediction),patch('bpdr_input.source_transfer.separate_source',return_value=dict(status='decomposed',arrays=parts)) as fitter:
            r=screen_crop(audio,22050)
        self.assertEqual(len(fitter.call_args.args),3);self.assertIs(fitter.call_args.args[2],prediction)
        self.assertEqual(r['coverage']['retained_support_samples'],10);self.assertIs(r['decomposition_arrays'],parts)

    def test_selection_requires_every_declared_training_record(self):
        records=[];sources=[]
        for i in range(10):
            dataset='vocadito' if i<3 else 'jamendo';record_id=str(i)
            records.append(dict(record_id=record_id,split='train',training_eligible=False,dataset=dataset,accepted_audio_path=record_id,
                technical_result=dict(status='ACCEPT_UNCHANGED',accepted_audio=dict(checksum=record_id))))
            sources.append(dict(record_id=record_id,split='train',dataset=dataset,audio_path=record_id,sha256=record_id))
        validate_selection(dict(sources=sources),dict(records=records))
        sources[0]['split']='test'
        with self.assertRaises(ValueError):validate_selection(dict(sources=sources),dict(records=records))
        sources[0]['split']='train'
        with self.assertRaises(ValueError):validate_selection(dict(sources=sources[:-1]),dict(records=records))
