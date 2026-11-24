from pathlib import Path
import tempfile
import unittest
import numpy as np
import soundfile
import torch
from bpdr_input import fine_pitch as fp


def archive(pitch, voiced, seed=0):
    raw = np.random.default_rng(seed).normal(size=(len(pitch), 71)).astype(np.float32)
    raw[:, fp.PITCH] = np.where(voiced, pitch, 0); raw[:, fp.VOICED] = voiced
    return raw


class FinePitchColumnTests(unittest.TestCase):
    def test_only_pitch_voicing_and_change_columns_change(self):
        n = 50; voiced = np.ones(n, bool); voiced[10:15] = False
        pesto = np.full(n, 33.33); raw = archive(pesto, voiced)
        out, dropped = fp.fine_columns(raw, np.full(n, 29.6))
        keep = [i for i in range(71) if i not in (fp.PITCH, fp.VOICED, fp.CHANGE)]
        np.testing.assert_array_equal(out[:, keep], raw[:, keep])
        self.assertEqual(dropped, 0)
        np.testing.assert_allclose(out[voiced, fp.PITCH], 29.6, atol=1e-4)
        self.assertTrue((out[~voiced, fp.PITCH] == 0).all())

    def test_disagreement_becomes_unvoiced_and_change_is_local(self):
        n = 20; voiced = np.ones(n, bool)
        fine = np.zeros(n); fine[5] = 1200.  # octave error
        fine[12:] = 30.                       # a real 30-cent step
        out, dropped = fp.fine_columns(archive(np.zeros(n)+.001, voiced), fine)
        self.assertEqual(dropped, 1); self.assertEqual(out[5, fp.VOICED], 0); self.assertEqual(out[5, fp.PITCH], 0)
        self.assertEqual(out[5, fp.CHANGE], 0); self.assertEqual(out[6, fp.CHANGE], 0)
        self.assertAlmostEqual(float(out[12, fp.CHANGE]), 30., places=4)
        self.assertEqual(float(np.abs(out[13:, fp.CHANGE]).max()), 0.)

    def test_rejects_wrong_shapes(self):
        with self.assertRaises(ValueError):
            fp.fine_columns(np.zeros((5, 70)), np.zeros(5))
        with self.assertRaises(ValueError):
            fp.fine_columns(np.zeros((5, 71)), np.zeros(4))

    def test_yin_reads_a_detuned_tone_on_the_feature_grid(self):
        sr = 22050; t = np.arange(sr*2)/sr; f = 220*2**(20/1200)
        x = (.3*np.sin(2*np.pi*f*t)+.1*np.sin(4*np.pi*f*t)).astype(np.float32)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'tone.wav'; soundfile.write(p, x, sr)
            times = np.arange(1+len(x)//441)*441/sr
            cents = fp.yin_cents(str(p), times)
        self.assertLess(abs(np.median(cents[10:-10])-(-1200+20)), 3)


class CorrectionMetricTests(unittest.TestCase):
    def test_perfect_and_zero_predictions(self):
        det = torch.tensor([[0., 30., 30., 0.]]); mask = torch.ones(1, 4, dtype=torch.bool)
        batch = dict(mask=mask, detuning=det)
        zero = {v: torch.zeros(1, 4) for v in 'ABCD'}
        perfect = dict(zero, C=-det, D=-det)
        self.assertEqual(fp.correction_metrics(zero, batch), (30., 30., 0.))
        self.assertEqual(fp.correction_metrics(perfect, batch)[0], 0.)
        self.assertIsNone(fp.correction_metrics(zero, dict(mask=mask, detuning=torch.zeros(1, 4))))



class LearnabilityHelperTests(unittest.TestCase):
    def batch(self):
        det = torch.tensor([[0., -30., -30., 0.]]); mask = torch.ones(1, 4, dtype=torch.bool)
        return dict(mask=mask, detuning=det, padding=torch.zeros(1, 4, dtype=torch.bool), voiced={v: mask.clone() for v in 'AC'},
                    inputs={v: torch.zeros(1, 4, 71) for v in 'ABCD'})

    def test_squared_anchor_is_zero_for_perfect_predictions_and_mse_otherwise(self):
        from bpdr_input.fine_pitch_mse_check import squared_objective
        from bpdr_input.losses import LossWeights
        b = self.batch(); zero = {v: torch.zeros(1, 4) for v in 'ABCD'}
        perfect = dict(zero, C=-b['detuning'], D=-b['detuning'])
        self.assertEqual(float(squared_objective(perfect, b, 'augmentation_only', LossWeights(), False)[1]['anchor']), 0.)
        # zero output: C/D each have mean squared error (0+900+900+0)/4 = 450; averaged over four views -> 225
        self.assertAlmostEqual(float(squared_objective(zero, b, 'augmentation_only', LossWeights(), False)[1]['anchor']), 225., places=4)

    def test_answer_cue_only_touches_column_70(self):
        from bpdr_input.fine_pitch_overfit_v2 import with_answer_cue
        b = self.batch(); cued = with_answer_cue(b)
        self.assertTrue(torch.equal(cued['inputs']['C'][0, :, 70], torch.tensor([0., 1., 1., 0.])))
        self.assertTrue(torch.equal(cued['inputs']['A'][0, :, 70], torch.zeros(4)))
        self.assertTrue(torch.equal(cued['inputs']['C'][..., :70], b['inputs']['C'][..., :70]))
        self.assertTrue(torch.equal(b['inputs']['C'][0, :, 70], torch.zeros(4)))

    def test_fixed_selection_rule_is_deterministic(self):
        from bpdr_input.fine_pitch_overfit import select
        recs = [dict(dataset='jamendo', cents=c, corruption='split', group_id=f'j{k}') for k, c in enumerate((-30., -20., -10., 10., 20., 30.))]
        recs += [dict(dataset='vocadito', cents=30., corruption='split', group_id='v1'), dict(dataset='vocadito', cents=-30., corruption='split', group_id='v2')]
        self.assertEqual(select(recs), list(range(8)))


class RelativePitchTests(unittest.TestCase):
    def test_relative_pitch_centres_voiced_frames_only(self):
        from bpdr_input.fine_pitch_relative_check import relative_pitch
        voiced = np.array([1, 1, 1, 0, 1], bool); pitch = np.array([-900., -870., -800., 0., -900.])
        raw = archive(pitch, voiced); out = relative_pitch(raw)
        np.testing.assert_allclose(out[:, fp.PITCH], [-15., 15., 85., 0., -15.], atol=1e-4)  # voiced median -885
        keep = [i for i in range(71) if i != fp.PITCH]
        np.testing.assert_array_equal(out[:, keep], raw[:, keep])
        self.assertEqual(float(raw[0, fp.PITCH]), -900.)

    def test_relative_pitch_is_invariant_to_global_transposition(self):
        from bpdr_input.fine_pitch_relative_check import relative_pitch
        voiced = np.ones(6, bool); pitch = np.array([-1000., -990., -800., -790., -500., -970.])
        np.testing.assert_allclose(relative_pitch(archive(pitch, voiced))[:, fp.PITCH], relative_pitch(archive(pitch+137., voiced))[:, fp.PITCH], atol=1e-3)


class BpdrObjectiveTests(unittest.TestCase):
    def test_response_terms_join_only_for_bpdr_after_warmup(self):
        from bpdr_input.fine_pitch_mse_check import squared_objective
        from bpdr_input.losses import LossWeights
        det = torch.tensor([[0., -30., -30., 0.]]); mask = torch.ones(1, 4, dtype=torch.bool)
        b = dict(mask=mask, detuning=det, voiced={v: mask.clone() for v in 'AC'})
        zero = {v: torch.zeros(1, 4) for v in 'ABCD'}
        total, parts = squared_objective(zero, b, 'bpdr', LossWeights(), True)
        self.assertAlmostEqual(float(parts['detuning']), 15., places=4)  # mean |0-30| over 2 of 4 frames
        self.assertEqual(float(parts['boundary']), 0.)
        self.assertAlmostEqual(float(total), 225.+15., places=3)
        self.assertNotIn('detuning', squared_objective(zero, b, 'bpdr', LossWeights(), False)[1])
        self.assertNotIn('detuning', squared_objective(zero, b, 'augmentation_only', LossWeights(), True)[1])


class LearningCurveSplitTests(unittest.TestCase):
    def records(self):
        return ([dict(dataset='jamendo', group_id=f'jamendo:a{k}') for k in range(9) for _ in range(3)]
                + [dict(dataset='vocadito', group_id=f'vocadito:S{k}') for k in range(5) for _ in range(2)])

    def test_every_fourth_sorted_group_is_held_out_with_all_its_variants(self):
        from bpdr_input.fine_pitch_learning_curve import split
        recs = self.records(); trained, heldout, held = split(recs)
        self.assertEqual(held, ['jamendo:a0', 'jamendo:a4', 'jamendo:a8', 'vocadito:S0', 'vocadito:S4'])
        self.assertFalse({recs[i]['group_id'] for i in trained} & set(held))
        self.assertEqual(len(trained)+len(heldout), len(recs)); self.assertEqual(len(heldout), 3*3+2*2)

    def test_schedule_draws_only_trained_records_and_is_reproducible(self):
        from bpdr_input.fine_pitch_learning_curve import split, schedule
        recs = self.records(); trained, heldout, _ = split(recs)
        w = np.ones(len(recs))/len(recs)
        a, b = schedule(w, trained, updates=50), schedule(w, trained, updates=50)
        np.testing.assert_array_equal(a, b); self.assertTrue(set(a.ravel()) <= set(trained))
        self.assertTrue(all(len(set(row)) == 8 for row in a))


if __name__ == '__main__':
    unittest.main()
