"""Frozen sample validation prevents silent changes to the five-artist batch."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from bpdr_input import jamendo_batch
from bpdr_input.pipeline import checksum


class FrozenBatchTests(unittest.TestCase):
    def test_same_artist_or_missing_license_cannot_enter_batch(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root/'selection.json'
            rows = [dict(track_id=str(i), artist_id='artist_'+str(i), license={'kind': 'by', 'version': '4.0'}) for i in range(5)]
            path.write_text(json.dumps({'tracks': rows}))
            with patch.object(jamendo_batch, 'eligible', return_value=rows):
                self.assertEqual(jamendo_batch.frozen_selection(root, path), rows)
                changed = [dict(row) for row in rows]
                changed[1]['artist_id'] = changed[0]['artist_id']
                path.write_text(json.dumps({'tracks': changed}))
                with self.assertRaisesRegex(ValueError, 'five distinct'):
                    jamendo_batch.frozen_selection(root, path)
                path.write_text(json.dumps({'tracks': rows}))
                current = [dict(row) for row in rows]
                current[0]['license'] = {'kind': 'by-nd', 'version': '4.0'}
                with patch.object(jamendo_batch, 'eligible', return_value=current):
                    with self.assertRaisesRegex(ValueError, 'training artists/licenses'):
                        jamendo_batch.frozen_selection(root, path)

    def test_changed_sample_cannot_resume_existing_batch(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            resume = root/'runs/jamendo_sample/existing'
            resume.mkdir(parents=True)
            before = json.dumps({'track_ids': ['old']})
            (resume/'batch.json').write_text(before)
            rows = [dict(track_id=str(i), artist_id='artist_'+str(i)) for i in range(5)]
            with patch.object(jamendo_batch, 'frozen_selection', return_value=rows):
                with self.assertRaisesRegex(ValueError, 'Cannot change the frozen sample'):
                    jamendo_batch.run(root, root/'selection.json', resume)
            self.assertEqual((resume/'batch.json').read_text(), before)

    def test_same_ids_with_changed_selection_metadata_cannot_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root/'selection.json'
            rows = [dict(track_id=str(i), artist_id='artist_'+str(i)) for i in range(5)]
            path.write_text(json.dumps({'tracks': rows}))
            resume = root/'runs/jamendo_sample/existing'
            resume.mkdir(parents=True)
            before = json.dumps({'track_ids': [r['track_id'] for r in rows], 'selection_sha256': checksum(path)})
            (resume/'batch.json').write_text(before)
            path.write_text(json.dumps({'tracks': rows, 'changed_metadata': True}))
            with patch.object(jamendo_batch, 'frozen_selection', return_value=rows):
                with self.assertRaisesRegex(ValueError, 'selection metadata changed'):
                    jamendo_batch.run(root, path, resume)
            self.assertEqual((resume/'batch.json').read_text(), before)


if __name__ == '__main__':
    unittest.main()
