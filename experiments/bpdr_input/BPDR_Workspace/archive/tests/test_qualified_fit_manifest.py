import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from evaluate_response_transformer import fitting_rows
from bpdr_input.experiments.learning import context_improvement as ex


class QualifiedFitManifestTests(unittest.TestCase):
    def test_selected_augmented_manifest_is_used_and_hash_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'augmented_pairs.json'
            row=dict(source=dict(sha256='known',singer_id='female1'),group='female1')
            path.write_text(json.dumps([row,dict(row,excluded='construction failed')]))
            declaration=dict(fit_manifests={'audio_augmented':str(path)},
                             fit_manifest_hashes={'audio_augmented':ex.sha(path)},
                             approved_training_sources={'known':'female1'})
            rows,digest=fitting_rows(root,declaration,'audio_augmented')
            self.assertEqual(rows,[row]);self.assertEqual(digest,ex.sha(path))
            path.write_text('[]')
            with self.assertRaises(ValueError):fitting_rows(root,declaration,'audio_augmented')

    def test_holdout_or_wrong_singer_cannot_enter_even_with_updated_manifest_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'fit.json'
            declaration=dict(fit_manifests={'audio_augmented':str(path)},
                             fit_manifest_hashes={},approved_training_sources={'known':'female1'})
            for row in (dict(source=dict(sha256='holdout',singer_id='female5'),group='female5'),
                        dict(source=dict(sha256='known',singer_id='female1'),group='female5')):
                path.write_text(json.dumps([row]));declaration['fit_manifest_hashes']['audio_augmented']=ex.sha(path)
                with self.assertRaises(ValueError):fitting_rows(root,declaration,'audio_augmented')

    def test_external_manifest_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'model';root.mkdir();path=Path(tmp)/'fit.json';path.write_text('[]')
            declaration=dict(fit_manifests={'audio_augmented':str(path)},
                             fit_manifest_hashes={'audio_augmented':ex.sha(path)})
            with self.assertRaises(ValueError):fitting_rows(root,declaration,'audio_augmented')


if __name__=='__main__':unittest.main()
