import unittest
from unittest.mock import patch
from pathlib import Path
from bpdr_input.shared_batch_intake import validate_record_metadata


class BatchMetadataTests(unittest.TestCase):
    def setUp(self):
        self.track=dict(mtg_track_id='track_0000001',track_id='1',artist_id='artist_a',license=dict(kind='by',version='4.0'))
        self.record=dict(schema_version='bpdr-track-v1',status='withheld',reference_type='model_estimate',training_eligible=False,
            mtg_track_id='track_0000001',track_id='1',artist_id_metadata_only='artist_a',license=dict(kind='by',version='4.0'),
            full_game_json_sha256='digest',artifacts={},reasons=['NO_REST_ALIGNED_CROP_WITH_FIVE_SUNG_SECONDS'])
        self.digest=patch('bpdr_input.shared_batch_intake.digest',return_value='digest');self.digest.start();self.addCleanup(self.digest.stop)
        self.exists=patch.object(Path,'exists',return_value=False);self.exists.start();self.addCleanup(self.exists.stop)

    def validate(self):
        return validate_record_metadata(self.record,self.track,Path('not-a-real-batch/bpdr'))

    def test_expected_withholding_is_valid_evidence_without_audio(self):
        self.assertEqual(self.validate(),'withheld')

    def test_unknown_status_is_rejected(self):
        self.record['status']='approved'
        with self.assertRaisesRegex(ValueError,'status'):self.validate()

    def test_serialized_training_approval_is_rejected(self):
        self.record['training_eligible']=True
        with self.assertRaisesRegex(ValueError,'training claim'):self.validate()

    def test_wrong_artist_or_license_is_rejected(self):
        self.record['artist_id_metadata_only']='different'
        with self.assertRaisesRegex(ValueError,'Artist'):self.validate()
        self.record['artist_id_metadata_only']='artist_a';self.record['license']['kind']='by-nd'
        with self.assertRaisesRegex(ValueError,'License'):self.validate()

    def test_withholding_without_reason_or_with_claimed_artifacts_is_rejected(self):
        self.record['reasons']=[]
        with self.assertRaisesRegex(ValueError,'reasons'):self.validate()
        self.record['reasons']=['a'];self.record['artifacts']={'vocal.wav':'hash'}
        with self.assertRaisesRegex(ValueError,'artifacts'):self.validate()

    def test_stale_audio_under_withheld_record_is_rejected(self):
        with patch.object(Path,'exists',return_value=True):
            with self.assertRaisesRegex(ValueError,'contradictory'):self.validate()


if __name__=='__main__':unittest.main()
