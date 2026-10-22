import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from bpdr_input.shared_pair_audit import game_boundary_track,training_candidates,write_views,validate_engineering_pair
from bpdr_input.shared_exports import digest
from bpdr_input.examples import load_pairs


class SelectionAndBoundaryTests(unittest.TestCase):
    def row(self,**kwargs):
        return dict(mtg_track_id='track_1',audio_sha256='a',split='train',integrity='verified_export',
                    engineering_input_candidate=True,candidate_for_pair_audit=True,**kwargs)

    def test_nontraining_selection_is_rejected_before_file_io(self):
        row=self.row();row['split']='final_holdout'
        with self.assertRaisesRegex(ValueError,'frozen-training'):training_candidates(dict(candidates=[row]))

    def test_duplicate_audio_and_empty_selection_are_rejected(self):
        a=self.row();b=self.row();b['mtg_track_id']='track_2'
        with self.assertRaisesRegex(ValueError,'Duplicate'):training_candidates(dict(candidates=[a,b]))
        with self.assertRaises(ValueError):training_candidates(dict(candidates=[]))

    def test_gaussian_preserves_duplicate_edges_and_edges_after_last_frame(self):
        times=np.arange(100)*.02
        result=game_boundary_track(times,[.2,.2,2.001])
        expected=np.maximum(np.exp(-.5*((times-.2)/.04)**2),np.exp(-.5*((times-2.001)/.04)**2)).astype(np.float32)
        self.assertTrue(np.array_equal(result,expected))
        self.assertGreater(result[-1],.8)


class PairIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);n=60;self.times=np.arange(n)*.02
        before=np.zeros((n,70),dtype=np.float32);before[:,66]=1
        after=before.copy();after[:,64]=20
        nominal=game_boundary_track(self.times,[.1,.9]);corrupt=game_boundary_track(self.times,[.1,.5,.9])
        self.inputs=self.root/'inputs';write_views(self.inputs,self.times,before,after,nominal,corrupt)
        plateau=np.zeros(n,dtype=bool);plateau[20:30]=True
        common=np.ones(n,dtype=bool);common[:3]=False
        self.targets=dict(times_seconds=self.times,anchor_mask=common,common_reliable_mask=common,
            detuned_plateau_mask=plateau,zero_anchor_mask=common&~plateau,correction_target_cents=np.where(plateau,-20.,0.))
        self.target_path=self.root/'targets.npz';np.savez_compressed(self.target_path,**self.targets)
        for name in ('manifest.json','policy.json'): (self.root/name).write_text('{}')
        (self.root/'audit.json').write_text(json.dumps(dict(passed=True,requested_cents=20.,policy={})))
        (self.root/'original.wav').write_bytes(b'fixture original');(self.root/'changed.wav').write_bytes(b'fixture shifted')
        self.record=dict(phrase_id='fixture',source_id='fixture',singer_id='artist_proxy',split='train',cents=20.,corruption='split',
            training_eligible=False,exploratory=True,review_status='pending',reference_type='model_estimate',
            data_status='exploratory_not_research',audit_passed=True,input_directory=str(self.inputs),
            input_sha256={v:digest(self.inputs/(v+'_input.npz')) for v in 'ABCD'},
            waveform_paths={v:str(self.root/('original.wav' if v in 'AB' else 'changed.wav')) for v in 'ABCD'},
            waveform_sha256={v:digest(self.root/('original.wav' if v in 'AB' else 'changed.wav')) for v in 'ABCD'})
        for path,key,filename in [('manifest_path','manifest_sha256','manifest.json'),('target_path','targets_sha256','targets.npz'),
                                  ('audit_path','audit_sha256','audit.json'),('audit_policy_path','audit_policy_sha256','policy.json')]:
            self.record[path]=str(self.root/filename);self.record[key]=digest(self.root/filename)

    def save_targets(self):
        np.savez_compressed(self.target_path,**self.targets);self.record['targets_sha256']=digest(self.target_path)

    def test_four_views_load_in_engineering_but_research_is_refused(self):
        self.assertTrue(validate_engineering_pair(self.record))
        (self.root/'summary.json').write_text(json.dumps(dict(examples=[self.record])))
        pairs=load_pairs(self.root,'engineering');self.assertEqual(len(pairs),1)
        self.assertEqual(pairs[0]['detuning'][20],20.)
        with self.assertRaisesRegex(ValueError,'reviewed'):load_pairs(self.root,'research')

    def test_modified_input_hash_is_rejected(self):
        with (self.inputs/'A_input.npz').open('ab') as f:f.write(b'changed')
        with self.assertRaisesRegex(ValueError,'Input changed'):validate_engineering_pair(self.record)

    def test_rehashed_target_outside_audit_mask_is_rejected(self):
        self.targets['correction_target_cents'][0]=-20.;self.save_targets()
        with self.assertRaisesRegex(ValueError,'outside audit mask'):validate_engineering_pair(self.record)

    def test_rehashed_reference_field_in_input_is_rejected(self):
        path=self.inputs/'A_input.npz'
        with np.load(path) as z:values={k:z[k] for k in z.files}
        values['reference_pitch_hz']=np.ones(60)*220;np.savez_compressed(path,**values)
        self.record['input_sha256']['A']=digest(path)
        with self.assertRaisesRegex(ValueError,'leaked'):validate_engineering_pair(self.record)

    def test_serialized_research_or_nontraining_claim_is_rejected(self):
        self.record['training_eligible']=True
        with self.assertRaisesRegex(ValueError,'exploratory'):validate_engineering_pair(self.record)
        self.record['training_eligible']=False;self.record['split']='development'
        with self.assertRaises(ValueError):validate_engineering_pair(self.record)

    def test_rehashed_wrong_audit_shift_is_rejected(self):
        p=Path(self.record['audit_path']);p.write_text(json.dumps(dict(passed=True,requested_cents=-20.,policy={})))
        self.record['audit_sha256']=digest(p)
        with self.assertRaisesRegex(ValueError,'requested shift'):validate_engineering_pair(self.record)


if __name__=='__main__':unittest.main()
