"""Independent-source counts and incomplete-run safeguards for the sample report."""
import json
from pathlib import Path
import tempfile
import unittest
from bpdr_input.jamendo_yield import counts, run


class YieldTests(unittest.TestCase):
    def test_windows_do_not_inflate_independent_artist_count(self):
        rows = [dict(source_id='one', group_id='artist_A'), dict(source_id='one', group_id='artist_A'),
                dict(source_id='two', group_id='artist_B')]
        self.assertEqual(counts(rows), {'crops': 3, 'songs': 2, 'artists': 2})
        self.assertEqual(counts([]), {'crops': 0, 'songs': 0, 'artists': 0})

    def test_incomplete_batch_cannot_be_reported_as_measured_yield(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)/'runs/jamendo_sample/incomplete'
            root.mkdir(parents=True)
            (root/'batch.json').write_text(json.dumps({'run_status': 'running', 'track_ids': list('12345')}))
            with self.assertRaisesRegex(ValueError, 'completed frozen five-track'):
                run(root)
            self.assertFalse((root.parents[2]/'runs/jamendo_yield').exists())


if __name__ == '__main__':
    unittest.main()
