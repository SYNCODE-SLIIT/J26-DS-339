import io
import unittest
import zipfile
import json
from pathlib import Path
import tempfile
import numpy as np
import soundfile as sf
from bpdr_input.shared_exports import archive_entries,load_export,digest


def archive(names):
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w') as z:
        for name in names:z.writestr(name,'{}')
    buffer.seek(0)
    return zipfile.ZipFile(buffer)


class SharedExportTests(unittest.TestCase):
    def test_metadata_is_skipped_without_losing_real_record(self):
        with archive(['__MACOSX/._outputs','outputs/.DS_Store','outputs/track_0000001/bpdr/record.json']) as z:
            good,skipped=archive_entries(z)
        self.assertEqual(len(good),1);self.assertEqual(len(skipped),2)

    def test_path_escape_absolute_backslash_and_drive_are_rejected(self):
        for name in ['outputs/track_0000001/../../bad','/outputs/track_0000001/x','C:/x']:
            with archive([name]) as z:
                with self.assertRaises(ValueError):archive_entries(z)
        # ZipInfo's constructor normalizes backslashes on Windows; inject the
        # raw header name to exercise our cross-platform preflight explicitly.
        with archive(['outputs/track_0000001/x']) as z:
            z.infolist()[0].filename='outputs\\track_0000001\\x'
            with self.assertRaises(ValueError):archive_entries(z)

    def test_case_colliding_or_duplicate_destinations_are_rejected(self):
        with archive(['outputs/track_0000001/key.json','outputs/track_0000001/KEY.json']) as z:
            with self.assertRaises(ValueError):archive_entries(z)

    def test_empty_metadata_only_archive_is_not_a_dataset(self):
        with archive(['__MACOSX/._outputs']) as z:
            with self.assertRaises(ValueError):archive_entries(z)


class SharedExportLoaderTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.folder=Path(self.temp.name)/'track_0000001/bpdr';self.folder.mkdir(parents=True)
        full=self.folder.parent/'melody/game_vocal.json';full.parent.mkdir();full.write_text('{}')
        st=np.arange(22050)/22050;sf.write(self.folder/'vocal.wav',.1*np.sin(2*np.pi*220*st),22050,subtype='PCM_24')
        times=np.arange(51)*.02;matrix=np.zeros((51,71),dtype=np.float32)
        matrix[:,64]=-1200;matrix[:,65]=.9;matrix[:,66]=1
        for edge in (.2,.8):matrix[:,70]=np.maximum(matrix[:,70],np.exp(-.5*((times-edge)/.04)**2).astype(np.float32))
        self.features=dict(boundary_features_71=matrix,times_seconds=times,f0_hz=np.full(51,220.,dtype=np.float32),
            pitch_change_valid=np.r_[False,np.ones(50,dtype=bool)])
        inside=(times>=.2+.1)&(times<=.8-.1)
        np.savez_compressed(self.folder/'features.npz',**self.features)
        np.savez_compressed(self.folder/'masks.npz',voiced=np.ones(51,dtype=bool),reliable_pitch_screen=inside,
            heuristic_boundary_score=np.zeros(51,dtype=np.float32),crop_edge_protected=(times<.1)|(times>.9))
        (self.folder/'boundaries.json').write_text(json.dumps(dict(time_origin='crop_start',pitch_field='model_midi_pitch',
            pitch_shift_semitones=0,notes=[dict(onset_sec=.2,offset_sec=.8,model_midi_pitch=57.)])))
        self.record=dict(schema_version='bpdr-track-v1',status='exported',reference_type='model_estimate',training_eligible=False,
            mtg_track_id='track_0000001',track_id='1',full_game_json_sha256=digest(full),
            crop=dict(start_sample=11025,end_sample=55125,original_sample_rate_hz=44100,offset_seconds=.25),
            screen=dict(frame_count=51,voiced_frames=51,reliable_frames=int(inside.sum()),inside_note_frames=int(inside.sum()),voiced_fraction=1.),
            artifacts={name:digest(self.folder/name) for name in ('vocal.wav','features.npz','masks.npz','boundaries.json')})
        self.save_record()

    def save_record(self):
        (self.folder/'record.json').write_text(json.dumps(self.record))

    def save_features(self):
        np.savez_compressed(self.folder/'features.npz',**self.features)
        self.record['artifacts']['features.npz']=digest(self.folder/'features.npz');self.save_record()

    def test_valid_source_preserves_distinct_boundaries_and_is_not_a_training_pair(self):
        r=load_export(self.folder)
        self.assertFalse(r['paired_training_ready']);self.assertFalse(r['research_eligible'])
        self.assertEqual(r['model_inputs']['frame_features_70'].shape,(51,70))
        self.assertFalse(np.array_equal(r['model_inputs']['game_boundary_score'],r['model_inputs']['heuristic_boundary_score']))

    def test_corrupted_artifact_is_rejected(self):
        with (self.folder/'features.npz').open('ab') as f:f.write(b'changed')
        with self.assertRaises(ValueError):load_export(self.folder)

    def test_rehashed_wrong_boundary_semantics_are_rejected(self):
        self.features['boundary_features_71'][:,70]=0;self.save_features()
        with self.assertRaisesRegex(ValueError,'GAME boundary'):load_export(self.folder)

    def test_invalid_f0_voicing_representation_is_rejected(self):
        self.features['f0_hz'][0]=np.nan;self.save_features()
        with self.assertRaisesRegex(ValueError,'F0 representation'):load_export(self.folder)

    def test_serialized_training_approval_is_not_trusted(self):
        self.record['training_eligible']=True;self.save_record()
        with self.assertRaisesRegex(ValueError,'training claim'):load_export(self.folder)
