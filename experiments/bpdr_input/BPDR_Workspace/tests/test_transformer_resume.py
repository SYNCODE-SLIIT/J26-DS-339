"""Technical resume must preserve identity, checkpoint hashes and UUID behavior."""
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
import uuid

WORKSPACE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(WORKSPACE/'tools'))
from resume_transformer_experiment import ResumeRoot, evaluation_complete, fit_complete


class ResumeTests(unittest.TestCase):
    def test_run_routing_and_latest_pointer_are_separate(self):
        root=WORKSPACE/'results/transformer_improvement'
        run=root/('a'*32)
        routing=ResumeRoot(run,root)
        factory=uuid.uuid4
        self.assertEqual(routing/('b'*32),run)
        self.assertEqual(routing/'latest_experiment.json',root/'latest_experiment.json')
        self.assertIs(uuid.uuid4,factory)
        with self.assertRaises(ValueError):
            routing/'unexpected-output'

    def test_cached_fit_requires_matching_receipt(self):
        with TemporaryDirectory(dir=WORKSPACE/'results/transformer_improvement') as folder:
            p=Path(folder)
            self.assertFalse(fit_complete(p,'decl'))
            (p/'checkpoint.pt').write_bytes(b'fixture')
            digest=hashlib.sha256(b'fixture').hexdigest()
            (p/'fit_receipt.json').write_text(json.dumps(dict(declaration_sha256='decl',checkpoint_sha256=digest)))
            self.assertTrue(fit_complete(p,'decl'))
            (p/'checkpoint.pt').write_bytes(b'changed')
            with self.assertRaises(ValueError):
                fit_complete(p,'decl')

    def test_cached_evaluation_requires_prediction_hash(self):
        with TemporaryDirectory(dir=WORKSPACE/'results/transformer_improvement') as folder:
            p=Path(folder)
            self.assertFalse(evaluation_complete(p))
            (p/'predictions.npz').write_bytes(b'fixture')
            (p/'accuracy.json').write_text(json.dumps(dict(predictions_sha256=hashlib.sha256(b'fixture').hexdigest())))
            self.assertTrue(evaluation_complete(p))
            (p/'predictions.npz').write_bytes(b'changed')
            with self.assertRaises(ValueError):
                evaluation_complete(p)


if __name__=='__main__':
    unittest.main()
