"""Unit tests for AI Music Evaluator and BPDR Pitch-Repair Transformer integration."""
import unittest
import numpy as np
from pathlib import Path

WS = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(WS / 'source'))

from bpdr_input.studio import evaluator, evidence as E, repair_transformer, span_repair


class EvaluatorAndRepairTests(unittest.TestCase):
    def setUp(self):
        # Generate 2 seconds of synthetic in-tune vocal tone at 440 Hz (A4)
        sr = E.SR
        t = np.arange(int(2.0 * sr)) / sr
        self.ref_audio = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
        
        # Corrupted singing: flat by 40 cents (440 * 2^(-40/1200) = ~429.9 Hz)
        f_flat = 440.0 * (2.0 ** (-40.0 / 1200.0))
        self.flat_audio = (0.5 * np.sin(2 * np.pi * f_flat * t)).astype(np.float32)

    def test_evaluator_detects_flat_intonation(self):
        res = evaluator.evaluate_singing_against_reference(
            user_audio_or_path=self.flat_audio,
            ref_audio_or_path_or_notes=self.ref_audio,
            sr=E.SR,
        )
        sc = res['scorecard']
        self.assertIn('overall_accuracy_score', sc)
        self.assertIn('pitch_score', sc)
        self.assertIn('grade', sc)
        self.assertTrue(0.0 <= sc['overall_accuracy_score'] <= 100.0)
        
        # Verify flat error is recorded
        errors = res['errors']
        self.assertGreater(len(errors), 0, "Evaluator should detect flat intonation")
        flat_errs = [e for e in errors if 'Flat' in e['error_type']]
        self.assertGreater(len(flat_errs), 0, "Evaluator should categorize error as Flat Intonation")
        self.assertLess(flat_errs[0]['cents_error'], -15.0)

    def test_evaluator_solo_mode(self):
        res = evaluator.evaluate_solo_singing(self.ref_audio, sr=E.SR)
        self.assertIn('scorecard', res)
        self.assertIn('overall_accuracy_score', res['scorecard'])

    def test_repair_transformer_architecture(self):
        net = repair_transformer.BPDRRepairTransformer(nin=8)
        dummy_x = np.zeros((2, 100, 8), dtype=np.float32)
        dummy_x[:, :, 3] = 1.0  # voiced
        import torch
        pred = net(torch.tensor(dummy_x))
        self.assertEqual(pred.shape, (2, 100))
        # Unvoiced frames should be 0
        dummy_unvoiced = np.zeros((1, 50, 8), dtype=np.float32)
        pred_unv = net(torch.tensor(dummy_unvoiced))
        self.assertTrue(torch.all(pred_unv == 0.0))

    def test_repair_singing_do_no_harm(self):
        rep = span_repair.repair_singing_with_transformer(
            user_audio_or_path=self.flat_audio,
            ref_audio_or_notes=self.ref_audio,
            sr=E.SR,
            max_shift_cents=80.0,
        )
        self.assertIn(rep['repaired_status'], ('ACCEPT_PREPARED', 'ACCEPT_UNCHANGED'))
        self.assertEqual(len(rep['repaired_audio']), len(self.flat_audio))
        self.assertIn('cents_mae_before', rep)
        self.assertIn('cents_mae_after', rep)

    def test_multiview_bpdr_controller(self):
        t = np.arange(0.0, 1.6, 0.02)
        c5_sharp30 = 523.2511 * (2.0 ** (30.0 / 1200.0))
        f0 = np.where(t < 0.80, 440.0, c5_sharp30)
        trk = {"t": t, "f0": f0, "voiced": np.ones(len(t), bool), "conf": np.full(len(t), 0.95, dtype=np.float32)}
        mv = repair_transformer.predict_multiview_bpdr(trk, sr=E.SR)
        self.assertIn("r_nominal_cents", mv)
        self.assertIn("r_split_cents", mv)
        self.assertIn("r_merge_cents", mv)
        self.assertIn("mean_boundary_sensitivity_cents", mv)
        self.assertEqual(len(mv["r_nominal_cents"]), len(t))


if __name__ == '__main__':
    unittest.main()
