"""Fit/save/reload binding and active-demo failures must not fall back silently."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from bpdr_input.experiments.learning import notebook_training as training
from bpdr_input.modeling.recovery_model import load_locked_diagnostic
from bpdr_input.repair import learned_runtime


class NotebookTrainingTests(unittest.TestCase):
    def test_invalid_arrays_cannot_be_recorded_as_training(self):
        x = np.zeros((12, 31)); y = np.zeros(12); w = np.ones(12) / 12
        training.validate_arrays(x, y, w)
        for xx, yy, ww in [(x[:, :30], y, w), (x, y[:-1], w), (x, y, -w),
                           (x, y, np.ones(12)), (x * np.nan, y, w)]:
            with self.subTest(shape=xx.shape), self.assertRaises(ValueError):
                training.validate_arrays(xx, yy, ww)

    def test_actual_fit_save_reload_keeps_predictions_and_detects_changed_file(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp); folder = root / 'run'; folder.mkdir()
            with patch.object(training, 'ROOT', root):
                training.write(folder / 'declaration.json', dict(operating_policy=dict(gain=1., deadband_cents=3.)))
                training.write(folder / 'locked_selection.json', dict(operating_policy=dict(gain=1., deadband_cents=3.)))
                training.write(folder / 'input_hashes.json', {})
                training.write(folder / 'train_pairs.json', [dict(source=dict(sha256='source', singer_id='train_singer'))])
                x = np.arange(12 * 31, dtype=float).reshape(12, 31)
                y = np.arange(12, dtype=float); w = np.ones(12) / 12
                estimator = ExtraTreesRegressor(n_estimators=3, random_state=339).fit(x, y, sample_weight=w)
                training.record_fit(folder, estimator, x, y, w, .01)
                checkpoint = folder / 'selected_model.joblib'
                joblib.dump(dict(model=dict(kind='extra_trees', estimator=estimator),
                                 payload=training.checkpoint_payload(folder)), checkpoint)
                training.record_checkpoint(folder)
                reloaded, _ = load_locked_diagnostic(checkpoint)
                np.testing.assert_array_equal(estimator.predict(x), reloaded['estimator'].predict(x))
                checkpoint.write_bytes(checkpoint.read_bytes() + b'changed')
                with self.assertRaisesRegex(ValueError, 'checkpoint changed'):
                    load_locked_diagnostic(checkpoint)

    def test_fitting_receipt_cannot_be_rebound_to_another_declaration(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp); folder = root / 'run'; folder.mkdir()
            with patch.object(training, 'ROOT', root):
                training.write(folder / 'fit_receipt.json', dict(declaration_sha256='wrong'))
                training.write(folder / 'declaration.json', dict(version='changed'))
                with self.assertRaisesRegex(ValueError, 'declaration changed'):
                    training.record_checkpoint(folder)

    def test_absent_active_pointer_retains_historical_model(self):
        with TemporaryDirectory() as tmp:
            with patch.object(learned_runtime, 'ACTIVE_DEMO', Path(tmp) / 'absent.json'):
                self.assertEqual(learned_runtime.active_card_path(), learned_runtime.CARD)

    def test_active_pointer_outside_training_root_fails_closed(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp); pointer = root / 'pointer.json'
            training.write(pointer, dict(card=str(root / 'foreign/demo_model.json')))
            with patch.object(learned_runtime, 'WORKSPACE', root), patch.object(learned_runtime, 'ACTIVE_DEMO', pointer):
                with self.assertRaisesRegex(ValueError, 'leaves its training-run'):
                    learned_runtime.active_card_path()

    def test_changed_active_card_cannot_silently_use_historical_weights(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp); pointer = root / 'pointer.json'
            card = root / 'results/model_recovery/notebook_training/run/demo_model.json'
            card.parent.mkdir(parents=True); training.write(card, dict(changed=True))
            training.write(pointer, dict(card=str(card), card_sha256='wrong'))
            with patch.object(learned_runtime, 'WORKSPACE', root), patch.object(learned_runtime, 'ACTIVE_DEMO', pointer):
                with self.assertRaisesRegex(ValueError, 'card changed'):
                    learned_runtime.load_model()

    def test_verified_card_supplies_the_metrics_required_by_the_app(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp); folder = root / 'run'; folder.mkdir()
            (folder / 'selected_model.joblib').write_bytes(b'fixture checkpoint identity')
            training.write(folder / 'fit_receipt.json', dict(fit_completed=True))
            training.write(folder / 'locked_selection.json', dict(operating_policy=dict(gain=1., deadband_cents=3.)))
            originals = [dict(input='original'+str(i), preserved=True) for i in range(4)]
            shifts = [dict(input='shift'+str(i), reference='original0', target_path='target',
                           singer='singer', input_mae_cents=30., output_mae_cents=10.,
                           delivered_preview=True, verified_improvement=True) for i in range(4)]
            shifts += [dict(excluded='NO_LABELS') for _ in range(12)]
            verification = dict(originals=originals, shifts=shifts,
                checkpoint_sha256=training.checksum(folder / 'selected_model.joblib'),
                by_singer=dict(singer=dict(audited=4), unscorable_singer=dict(audited=0)))
            training.write(folder / 'rendered_verification.json', verification)
            with patch.object(training, 'ROOT', root):
                card = training.build_demo_card(folder, verification)
            required = {'model_type', 'error_reduction_fraction', 'audited_shifts', 'heldout_singers',
                        'verified_improvements', 'originals_preserved', 'originals_tested', 'attempted_shifts'}
            self.assertTrue(required <= card['metrics'].keys())
            self.assertFalse(card['research_gates_passed'])


if __name__ == '__main__':
    unittest.main()
