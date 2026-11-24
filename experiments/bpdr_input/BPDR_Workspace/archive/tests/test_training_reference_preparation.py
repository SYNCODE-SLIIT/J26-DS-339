import copy
from pathlib import Path
import tempfile
import unittest
import numpy as np
import soundfile as sf
from bpdr_input.pipeline import checksum
from bpdr_input.training_reference_preparation import authenticate_cache, decoded_cache_identity, prepare_masks


class TrainingReferencePreparationTests(unittest.TestCase):
    def setUp(self):
        self.times=np.arange(0,4,.02)
        self.phrase=dict(split='train',duration_seconds=4,review_status='pending')
        self.before=dict(times_seconds=self.times,voiced_mask=np.ones(200,bool),
            confidence=np.ones(200),f0_hz=np.full(200,440.))
        self.f0=np.column_stack([self.times,np.full(200,440.)])
        note=lambda a:dict(start_seconds=a,duration_seconds=.8,pitch_hz=440.,partial=False)
        self.notes={n:[note(a) for a in [0.,1.,2.,3.]] for n in ['A1','A2']}

    def test_reserved_or_development_partition_blocked_before_io(self):
        for split in ['development','pp1_evaluation','final_holdout']:
            with self.assertRaisesRegex(ValueError,'training'):
                authenticate_cache(dict(split=split),{})

    def test_human_gaps_unvoicing_and_tracker_disagreement_never_become_targets(self):
        before=copy.deepcopy(self.before);before['confidence'][60:70]=.7
        before['voiced_mask'][60:70]=False;before['f0_hz'][60:70]=np.nan
        yin=np.full(200,440.);yin[110:120]=880.
        arrays,row=prepare_masks(self.phrase,before,yin,self.f0,self.notes)
        self.assertFalse(arrays['reliable_mask'][60:70].any())
        self.assertFalse(arrays['reliable_mask'][110:120].any())
        self.assertFalse(np.any(arrays['reliable_mask'] & ~arrays['annotation_mask']))
        self.assertFalse(np.any((arrays['candidate_plateau_mask']|arrays['candidate_outside_mask']) & ~arrays['reliable_mask']))
        self.assertFalse(row['training_eligible']);self.assertIsNone(row['acoustic_shift_audit_passed'])

    def test_empty_final_mask_is_explicit_geometry_refusal(self):
        arrays,row=prepare_masks(self.phrase,self.before,np.zeros(200),self.f0,self.notes)
        self.assertEqual(row['reliable_frames'],0);self.assertFalse(row['geometry_available'])
        self.assertIsNone(row['multi_interior_candidate']);self.assertIsNotNone(row['old_continuous_refusal'])

    def test_misaligned_yin_or_changed_voicing_rule_refused(self):
        with self.assertRaises(ValueError):prepare_masks(self.phrase,self.before,np.ones(199),self.f0,self.notes)
        b=copy.deepcopy(self.before);b['confidence'][0]=.5
        with self.assertRaises(ValueError):prepare_masks(self.phrase,b,np.ones(200),self.f0,self.notes)

    def test_no_mask_input_mutation(self):
        before=copy.deepcopy(self.before);f0=self.f0.copy();yin=np.full(200,440.)
        prepare_masks(self.phrase,self.before,yin,self.f0,self.notes)
        for k in before:np.testing.assert_array_equal(before[k],self.before[k])
        np.testing.assert_array_equal(self.f0,f0);np.testing.assert_array_equal(yin,np.full(200,440.))

    def test_audio_identity_rejects_encoding_or_rate_differences(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'wave.wav';audio=np.full(2000,.12345678,dtype=np.float32)
            sf.write(path,audio,22050,subtype='PCM_24')
            with self.assertRaisesRegex(ValueError,'exactly'):decoded_cache_identity((audio,22050),path)
            decoded,sr=sf.read(path,dtype='float32')
            self.assertTrue(decoded_cache_identity((decoded,sr),path))

    def test_source_and_accepted_checksums_are_both_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'wave.wav';sf.write(path,np.full(22050,.1),22050,subtype='PCM_24')
            h=checksum(path);p=dict(self.phrase,audio_path=str(path),sha256=h,duration_seconds=1)
            d=dict(source=dict(path=str(path),preserved_path=str(path),sha256=h),result=dict(
                request_id='cache',source_id='crop',status='ACCEPT_UNCHANGED',reason_codes=[],policy_version='phase1-technical-0.1',
                accepted_audio=dict(path=str(path),checksum=h,sample_rate_hz=22050,sample_count=22050,channels=1)))
            self.assertEqual(authenticate_cache(p,d)[2],22050)
            bad=copy.deepcopy(d);bad['source']['sha256']='0'*64
            with self.assertRaisesRegex(ValueError,'different crop'):authenticate_cache(p,bad)
            bad=copy.deepcopy(d);bad['result']['accepted_audio']['checksum']='0'*64
            with self.assertRaises(ValueError):authenticate_cache(p,bad)
