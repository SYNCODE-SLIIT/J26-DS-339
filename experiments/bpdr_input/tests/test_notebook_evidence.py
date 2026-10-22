"""Presentation evidence must fail closed on unsafe paths or altered artifacts."""
import json
from pathlib import Path
import tempfile
import unittest

from bpdr_input.notebook_evidence import component_path, load_index, verified_hash
from bpdr_input.pipeline import checksum


class NotebookEvidenceTests(unittest.TestCase):
    def test_existing_file_outside_component_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            parent = Path(folder)
            component = parent/'bpdr'
            component.mkdir()
            (parent/'teammate.json').write_text('{}')
            with self.assertRaises(ValueError):
                component_path(component, '../teammate.json')

    def test_altered_artifact_is_not_presented_as_verified(self):
        with tempfile.TemporaryDirectory() as folder:
            artifact = Path(folder)/'result.json'
            artifact.write_text('{"passing": 5}')
            digest = checksum(artifact)
            artifact.write_text('{"passing": 18}')
            with self.assertRaises(ValueError):
                verified_hash(folder, artifact, digest)

    def test_missing_evidence_stops_instead_of_creating_a_run(self):
        with tempfile.TemporaryDirectory() as folder:
            notebook_dir = Path(folder)/'notebooks'
            notebook_dir.mkdir()
            (notebook_dir/'one.ipynb').write_text('{}')
            (notebook_dir/'evidence_index.json').write_text(json.dumps(dict(
                maintained_notebooks=['one.ipynb'], evidence={'calibration':'missing.json'})))
            with self.assertRaises(FileNotFoundError):
                load_index(folder)
            self.assertFalse((Path(folder)/'missing.json').exists())


if __name__ == '__main__':
    unittest.main()
