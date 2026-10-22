"""Numerical/semantic checks for the independent comparison, without model training."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from experiments.rahul.src.comprehensive_evaluation import (
    chord_tones, compatible_duration, confusion, label_info, scores, save_table,
    read_json, track_bootstrap, clean,
)


class ComprehensiveEvaluationTests(unittest.TestCase):
    def test_project_parser_roots_and_qualities(self):
        self.assertEqual(label_info('G#:maj7/3'), (8, 'maj7', '3'))
        self.assertEqual(label_info('Ab:min7/b7'), (8, 'min7', 'b7'))
        self.assertEqual(label_info('N'), (None, 'N', None))
        self.assertIsNone(chord_tones('UNK'))

    def test_chord_tones_preserve_extensions_and_bass(self):
        self.assertEqual(chord_tones('C:maj'), {0, 4, 7})
        self.assertEqual(chord_tones('C:maj7'), {0, 4, 7, 11})
        self.assertEqual(chord_tones('C:maj/7'), {0, 4, 7, 11})
        self.assertEqual(chord_tones('C:dim7'), {0, 3, 6, 9})
        self.assertEqual(chord_tones('C:hdim7'), {0, 3, 6, 10})
        self.assertEqual(chord_tones('A:min7/b7'), {9, 0, 4, 7})
        self.assertIsNone(chord_tones('C:unsupported'))
        self.assertIsNone(chord_tones('N'))

    def test_overlap_duration_weighting_and_silence(self):
        self.assertAlmostEqual(compatible_duration({0, 4, 7}, [0, 2], [.75, .25]), .75)
        self.assertIsNone(compatible_duration({0}, [], []))
        self.assertIsNone(compatible_duration(None, [0], [1.]))

    def test_class_metrics_match_sklearn_with_reserved_predictions(self):
        y = np.array([2, 2, 2, 3, 3, 4]); p = np.array([2, 3, 0, 3, 4, 4])
        labels = ['PAD', 'UNK', 'N', 'C:maj', 'G:maj7']
        result, _ = scores(confusion(y, p, len(labels)), labels)
        self.assertAlmostEqual(result['accuracy'], accuracy_score(y, p))
        for average in ('macro', 'weighted'):
            precision, recall, f1, _ = precision_recall_fscore_support(y, p, labels=[2, 3, 4], average=average, zero_division=0)
            self.assertAlmostEqual(result[average + '_precision'], precision)
            self.assertAlmostEqual(result[average + '_recall'], recall)
            self.assertAlmostEqual(result[average + '_f1'], f1)
        self.assertEqual(result['micro_f1'], result['accuracy'])

    def test_quality_precision_is_not_exact_chord_accuracy(self):
        y = np.array([2, 3]); p = np.array([3, 2])
        summary, _ = scores(confusion(y, p, 4), ['PAD', 'UNK', 'C:maj7', 'G:maj7'])
        quality, _ = scores(confusion([0, 0], [0, 0], 1), ['maj7'], reserved=())
        self.assertEqual(summary['accuracy'], 0)
        self.assertEqual(quality['accuracy'], 1)

    def test_csv_json_preserve_numeric_and_missing_values(self):
        with tempfile.TemporaryDirectory() as folder:
            rows = [{'value': 1 / 3, 'unavailable': None, 'count': 5, 'top5': [{'label': 'C:maj', 'p': .2}]}]
            save_table(folder, 'roundtrip', rows)
            self.assertEqual(read_json(Path(folder) / 'roundtrip.json'), rows)
            self.assertIn('0.3333333333333333', (Path(folder) / 'roundtrip.csv').read_text())

    def test_nonfinite_exports_rejected(self):
        for value in (float('nan'), float('inf'), -float('inf')):
            with self.assertRaises(ValueError): clean({'score': value})

    def test_paired_track_bootstrap_is_seeded_and_constant_difference_zero(self):
        y = np.array([2, 2, 3, 3, 2, 3]); ti = np.array([0, 0, 1, 1, 2, 2])
        with tempfile.TemporaryDirectory() as folder:
            study = SimpleNamespace(track_ids=['a', 'b', 'c'], track_index=ti, y=y, C=4,
                tables=Path(folder), output=Path(folder),
                predictions={n: {'predicted_ids': y.copy()} for n in ('baseline', 'weighted', 'multitask')},
                results={n: {'metrics': {'accuracy': 1., 'macro_f1': 1.}} for n in ('baseline', 'weighted', 'multitask')})
            first = track_bootstrap(study, repetitions=100, seed=17)
            second = track_bootstrap(study, repetitions=100, seed=17)
            self.assertTrue(first.equals(second))
            diffs = first[first.comparison.str.contains(' minus ')]
            self.assertTrue((diffs.ci_low == 0).all() and (diffs.ci_high == 0).all())
            acc = first[(first.metric == 'accuracy') & ~first.comparison.str.contains(' minus ')]
            self.assertTrue((acc.ci_low == 1).all() and (acc.ci_high == 1).all())


if __name__ == '__main__': unittest.main()
