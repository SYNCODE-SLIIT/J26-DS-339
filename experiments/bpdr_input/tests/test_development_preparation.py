import copy
import json
from pathlib import Path
import unittest
import test_shared_pair_audit as fixtures

from bpdr_input.development_preparation import development_selection, validate_development_record
from bpdr_input.examples import load_pairs


class DevelopmentSelectionTests(unittest.TestCase):
    def setUp(self):
        self.v=dict(phrases=[dict(phrase_id='v',source_id='v',singer_id='S2',split='development'),
                             dict(phrase_id='reserved',source_id='reserved',singer_id='S3',split='final_holdout')])
        self.j=dict(tracks=[dict(mtg_track_id='j',policy_export_eligible=True)],
                    records=[dict(mtg_track_id='j',artist_id='2',split='development',integrity='verified_export',screen_status='review_candidate')])
        self.training=[dict(group_id='vocadito:S1',dataset='vocadito',source_id='train')]

    def test_only_declared_development_candidates_selected(self):
        rows=development_selection(self.v,self.j,self.training)
        self.assertEqual([r['phrase_id'] for r in rows],['v','j'])
        self.assertEqual(rows[1]['group_kind'],'artist_proxy_singer_unknown')
        self.assertEqual(rows[0]['reference_type'],'human_annotation')

    def test_upstream_withheld_is_excluded_without_audio_access(self):
        self.j['records'][0]['screen_status']='withheld'
        self.assertEqual(len(development_selection(self.v,self.j,self.training)),1)

    def test_training_group_overlap_refused(self):
        self.training[0]['group_id']='vocadito:S2'
        with self.assertRaisesRegex(ValueError,'overlaps'):development_selection(self.v,self.j,self.training)

    def test_policy_ineligible_candidate_refused(self):
        self.j['tracks'][0]['policy_export_eligible']=False
        with self.assertRaisesRegex(ValueError,'license/policy'):development_selection(self.v,self.j,self.training)

    def test_duplicate_source_refused(self):
        self.v['phrases'].append(copy.deepcopy(self.v['phrases'][0]))
        with self.assertRaisesRegex(ValueError,'Duplicate'):development_selection(self.v,self.j,self.training)


class DevelopmentRecordTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.PairIntegrityTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.record=self.fixture.record
        self.record['split']='development'

    def test_development_record_valid_but_training_loader_refuses(self):
        self.assertTrue(validate_development_record(self.record))
        p=Path(self.record['input_directory']).parent
        (p/'summary.json').write_text(json.dumps(dict(examples=[self.record])))
        with self.assertRaisesRegex(ValueError,'No usable'):load_pairs(p,'engineering')
        with self.assertRaisesRegex(ValueError,'reviewed'):load_pairs(p,'research')

    def test_training_or_approval_claim_refused_before_io(self):
        self.record['split']='train'
        with self.assertRaisesRegex(ValueError,'development-only'):validate_development_record(self.record)
        self.record.update(split='development',training_eligible=True)
        with self.assertRaisesRegex(ValueError,'development-only'):validate_development_record(self.record)

    def test_changed_development_view_refused(self):
        with (Path(self.record['input_directory'])/'A_input.npz').open('ab') as f:f.write(b'changed')
        with self.assertRaisesRegex(ValueError,'inputs'):validate_development_record(self.record)


if __name__=='__main__':unittest.main()
