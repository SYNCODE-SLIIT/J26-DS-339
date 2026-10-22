import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from bpdr_input.examples import EXPLORATORY, Normalizer, collate, load_pairs
from bpdr_input.losses import LossWeights, boundary_invariance, counter_detuning, masked_mean_abs, objective, smoothness
from bpdr_input.metrics import evaluate
from bpdr_input.model import CorrectionEncoder, ModelConfig
from bpdr_input.training import predict, reload, train


def curves(frames=10, cents=20.):
    """Constructed curves: a detuned plateau on frames 2-5 inside a reliable mask on frames 1-8."""
    detuning = torch.zeros(1, frames)
    detuning[0, 2:6] = cents
    mask = torch.zeros(1, frames, dtype=torch.bool)
    mask[0, 1:9] = True
    voiced = torch.ones(1, frames, dtype=torch.bool)
    return dict(mask=mask, detuning=detuning, voiced={'A': voiced, 'C': voiced}, pair_ids=['p'], singer_ids=['S1'])


def perfect(batch):
    zero = torch.zeros_like(batch['detuning'])
    return {'A': zero, 'B': zero.clone(), 'C': -batch['detuning'], 'D': -batch['detuning'].clone()}


def write_run(root, exploratory=True, split='train', audit_passed=True, frames=60, shifts=(20., -20.)):
    """A pair collection on disk in the Phase 2 layout, small enough to fit in seconds."""
    rng = np.random.default_rng(7)
    times = np.arange(frames)*.02
    records = []
    for cents in shifts:
        directory = Path(root)/f'shift_{cents:+g}'/'split'
        directory.mkdir(parents=True)
        original = rng.normal(size=(frames, 70)).astype(np.float32)
        original[:, 66] = 1.
        detuned = original.copy()
        detuned[:, 64] += np.where((times >= .3) & (times < .8), cents/100, 0).astype(np.float32)
        tracks = dict(nominal=np.zeros(frames, dtype=np.float32), corrupt=np.zeros(frames, dtype=np.float32))
        tracks['nominal'][10], tracks['corrupt'][[10, 30]] = 1., 1.
        for view, acoustics, track in (('A', original, 'nominal'), ('B', original, 'corrupt'), ('C', detuned, 'nominal'), ('D', detuned, 'corrupt')):
            np.savez_compressed(directory/(view+'_input.npz'), times_seconds=times, frame_features_70=acoustics,
                                boundary_features_71=np.column_stack([acoustics, tracks[track]]).astype(np.float32))
        anchors = (times >= .1) & (times < 1.)
        target = np.where((times >= .3) & (times < .8), -cents, 0.)
        np.savez_compressed(directory.parent/'targets_NOT_MODEL_INPUT.npz', anchor_mask=anchors, common_reliable_mask=anchors,
                            detuned_plateau_mask=anchors & (target != 0), zero_anchor_mask=anchors & (target == 0),
                            correction_target_cents=target, times_seconds=times)
        records.append(dict(phrase_id='fixture_p00', source_id='fixture', singer_id='S1', split=split, cents=cents, corruption='split',
                            audit_passed=audit_passed, exploratory=exploratory, training_eligible=False, review_status='pending',
                            input_directory=str(directory), target_path=str(directory.parent/'targets_NOT_MODEL_INPUT.npz'),
                            waveform_sha256={'A': 'a', 'B': 'a', 'C': 'c', 'D': 'c'}, input_sha256={view: view for view in 'ABCD'}))
    (Path(root)/'summary.json').write_text(json.dumps(dict(examples=records)), encoding='utf-8')
    return Path(root)


class LossTests(unittest.TestCase):
    def test_masked_difference_value_and_empty_mask_is_excluded_and_counted(self):
        a = torch.tensor([[1., 2., 3.], [5., 5., 5.]])
        mask = torch.tensor([[True, True, False], [False, False, False]])
        value, skipped = masked_mean_abs(a, torch.zeros_like(a), mask)
        self.assertAlmostEqual(float(value), 1.5)
        self.assertEqual(skipped, 1)
        value, skipped = masked_mean_abs(a, torch.zeros_like(a), torch.zeros_like(mask))
        self.assertEqual((float(value), skipped), (0., 2))

    def test_counter_detuning_requires_the_opposite_sign(self):
        batch = curves()
        r = perfect(batch)
        self.assertEqual(float(counter_detuning(r, batch['detuning'], batch['mask'])[0]), 0.)
        wrong = dict(r, C=batch['detuning'], D=batch['detuning'])
        # Four plateau frames out of eight masked frames are each 40 cents from the target.
        self.assertAlmostEqual(float(counter_detuning(wrong, batch['detuning'], batch['mask'])[0]), 20.)

    def test_flat_zero_output_is_invariant_but_fails_the_anchor_and_detuning_terms(self):
        batch = curves()
        zero = {view: torch.zeros_like(batch['detuning']) for view in 'ABCD'}
        self.assertEqual(float(boundary_invariance(zero, batch['mask'])[0]), 0.)
        self.assertAlmostEqual(float(counter_detuning(zero, batch['detuning'], batch['mask'])[0]), 10.)
        _, parts, _ = objective(zero, batch, 'bpdr')
        self.assertAlmostEqual(float(parts['anchor']), 5.)
        total, parts, _ = objective(perfect(batch), batch, 'bpdr', LossWeights(smoothness=0.))
        self.assertEqual(float(total), 0.)

    def test_boundary_invariance_measures_only_the_boundary_pair_difference(self):
        batch = curves()
        r = perfect(batch)
        r['B'] = r['B']+3.
        self.assertAlmostEqual(float(boundary_invariance(r, batch['mask'])[0]), 1.5)

    def test_smoothness_ignores_changes_across_unvoiced_frames(self):
        curve = torch.tensor([[0., 0., 10., 10.]])
        voiced = torch.ones(1, 4, dtype=torch.bool)
        self.assertEqual(float(smoothness(torch.full((1, 4), 7.), voiced)[0]), 0.)
        self.assertAlmostEqual(float(smoothness(curve, voiced)[0]), 10/3, places=5)
        voiced[0, 2] = False
        self.assertEqual(float(smoothness(curve, voiced)[0]), 0.)

    def test_response_terms_belong_to_bpdr_after_warmup_only(self):
        batch = curves()
        r = perfect(batch)
        self.assertEqual(set(objective(r, batch, 'augmentation_only')[1]), {'anchor', 'smoothness'})
        self.assertEqual(set(objective(r, batch, 'frame_only')[1]), {'anchor', 'smoothness'})
        self.assertEqual(set(objective(r, batch, 'bpdr', response=False)[1]), {'anchor', 'smoothness'})
        self.assertEqual(set(objective(r, batch, 'bpdr')[1]), {'anchor', 'smoothness', 'boundary', 'detuning'})
        with self.assertRaises(ValueError):
            objective(r, batch, 'other')

    def test_gradients_reach_masked_frames_only(self):
        batch = curves()
        r = {view: torch.zeros(1, 10, requires_grad=True) for view in 'ABCD'}
        total, _, _ = objective(r, batch, 'bpdr', LossWeights(smoothness=0.))
        total.backward()
        gradient = r['C'].grad[0]
        self.assertTrue((gradient[2:6] != 0).all())
        self.assertTrue((gradient[~batch['mask'][0]] == 0).all())


class ModelTests(unittest.TestCase):
    def test_shape_parameter_counts_and_feature_check(self):
        torch.manual_seed(0)
        boundary, frame = CorrectionEncoder(ModelConfig(features=71)), CorrectionEncoder(ModelConfig(features=70))
        self.assertEqual(boundary(torch.zeros(2, 50, 71)).shape, (2, 50))
        # The two models differ only by the weights of the one extra input column.
        self.assertEqual(boundary.parameter_count()-frame.parameter_count(), 128)
        self.assertEqual(boundary.describe()['layers'], 4)
        with self.assertRaises(ValueError):
            frame(torch.zeros(1, 5, 71))

    def test_padding_does_not_change_real_frames(self):
        torch.manual_seed(0)
        model = CorrectionEncoder().eval()
        real = torch.randn(1, 30, 71)
        padded = torch.cat([real, torch.randn(1, 12, 71)*50], dim=1)
        padding = torch.arange(42).unsqueeze(0) >= 30
        with torch.inference_mode():
            alone, together = model(real), model(padded, padding)
        self.assertTrue(torch.allclose(alone[0], together[0, :30], atol=1e-4))
        self.assertTrue((together[0, 30:] == 0).all())

    def test_frame_only_model_cannot_react_to_boundary_metadata(self):
        torch.manual_seed(0)
        acoustics = torch.randn(1, 40, 70)
        nominal, corrupt = torch.zeros(1, 40, 1), torch.zeros(1, 40, 1)
        corrupt[0, 20] = 1.
        frame, boundary = CorrectionEncoder(ModelConfig(features=70)).eval(), CorrectionEncoder(ModelConfig(features=71)).eval()
        with torch.inference_mode():
            self.assertTrue(torch.equal(frame(acoustics), frame(acoustics.clone())))
            a, b = boundary(torch.cat([acoustics, nominal], -1)), boundary(torch.cat([acoustics, corrupt], -1))
        self.assertGreater(float((a-b).abs().max()), 0.)


class MetricTests(unittest.TestCase):
    def test_zero_output_fails_repair_and_perfect_output_passes(self):
        batch = curves()
        zero = {view: torch.zeros_like(batch['detuning']) for view in 'ABCD'}
        flat = evaluate(zero, batch)['summary']
        self.assertEqual(flat['repair_error_cents'], 20.)
        self.assertEqual(flat['boundary_sensitivity_cents'], 0.)
        self.assertEqual(flat['edit_coverage'], 0.)
        exact = evaluate(perfect(batch), batch)['summary']
        self.assertEqual((exact['repair_error_cents'], exact['outside_alteration_cents'], exact['unchanged_alteration_cents']), (0., 0., 0.))
        self.assertEqual(exact['edit_coverage'], .5)
        self.assertEqual((exact['pairs'], exact['singers']), (1, 1))

    def test_pair_without_a_detuned_mask_is_excluded_and_counted(self):
        batch = curves()
        batch['detuning'] = torch.zeros_like(batch['detuning'])
        result = evaluate(perfect(batch), batch)
        self.assertEqual((result['summary']['pairs'], result['summary']['excluded_pairs']), (0, 1))
        self.assertIsNone(result['summary']['repair_error_cents'])


class LoaderAndTrainingTests(unittest.TestCase):
    def test_research_mode_refuses_exploratory_pairs(self):
        with tempfile.TemporaryDirectory() as folder:
            run = write_run(folder)
            with self.assertRaises(ValueError):
                load_pairs(run, 'research')
            with self.assertRaises(ValueError):
                load_pairs(run, 'anything')

    def test_engineering_mode_takes_audited_training_pairs_only(self):
        with tempfile.TemporaryDirectory() as folder:
            pairs = load_pairs(write_run(folder), 'engineering')
            self.assertEqual(len(pairs), 2)
            self.assertTrue(all(pair['data_status'] == EXPLORATORY for pair in pairs))
        for options in (dict(split='development'), dict(audit_passed=False)):
            with tempfile.TemporaryDirectory() as folder:
                with self.assertRaises(ValueError):
                    load_pairs(write_run(folder, **options), 'engineering')

    def test_batches_hold_normalized_inputs_and_no_target(self):
        with tempfile.TemporaryDirectory() as folder:
            pairs = load_pairs(write_run(folder), 'engineering')
            normalizer = Normalizer.fit(pairs)
            batch = collate(pairs, normalizer, 71)
            self.assertEqual(batch['inputs']['A'].shape, (2, 60, 71))
            self.assertEqual(collate(pairs, normalizer, 70)['inputs']['B'].shape, (2, 60, 70))
            expected = normalizer(pairs[0]['inputs']['C'])
            self.assertTrue(np.allclose(batch['inputs']['C'][0].numpy(), expected))
            # A and B share acoustics; only the boundary column may differ.
            self.assertTrue(torch.equal(batch['inputs']['A'][..., :70], batch['inputs']['B'][..., :70]))
            self.assertFalse(torch.equal(batch['inputs']['A'][..., 70], batch['inputs']['B'][..., 70]))
            self.assertEqual(float(batch['detuning'][0].max()), 20.)
            self.assertEqual(float(batch['detuning'][1].min()), -20.)
            self.assertTrue(np.isfinite(normalizer.scale).all() and (normalizer.scale > 0).all())

    def test_tiny_set_is_fitted_saved_stamped_and_reloaded(self):
        with tempfile.TemporaryDirectory() as folder:
            run = write_run(Path(folder)/'pairs')
            directory, record = train(run, 'bpdr', 'engineering', Path(folder)/'runs', updates=120, warmup=40, seed=1)
            self.assertEqual(directory.parent.name, 'engineering')
            self.assertEqual(record['data_status'], EXPLORATORY)
            self.assertEqual(json.loads((directory/'run_record.json').read_text())['data_status'], EXPLORATORY)
            self.assertEqual(record['model']['features'], 71)
            self.assertLess(record['training_fit']['repair_error_cents'], 20.)
            self.assertLess(record['last_step']['anchor'], record['first_step']['anchor'])
            # The stamp travels with every saved file, including the checkpoint a later evaluation would reload.
            for name in ('history.json', 'training_fit.json'):
                self.assertEqual(json.loads((directory/name).read_text())['data_status'], EXPLORATORY)
            self.assertEqual(torch.load(directory/'checkpoint.pt', weights_only=True)['pairs_summary_sha256'], record['pairs_summary_sha256'])
            self.assertTrue(record['numpy_version'] and record['python_version'] and record['interpreter'])
            model, normalizer, condition, status = reload(directory)
            self.assertEqual((condition, status), ('bpdr', EXPLORATORY))
            pairs = load_pairs(run, 'engineering')
            batch = collate(pairs, normalizer, 71)
            with torch.inference_mode():
                again = evaluate(predict(model, batch), batch)['summary']
            for key in ('repair_error_cents', 'boundary_sensitivity_cents', 'unchanged_alteration_cents', 'outside_alteration_cents'):
                self.assertAlmostEqual(again[key], record['training_fit'][key], places=4)
            _, control = train(run, 'frame_only', 'engineering', Path(folder)/'runs', updates=3, warmup=1, seed=1)
            self.assertEqual(control['model']['features'], 70)
            self.assertEqual(control['training_fit']['boundary_sensitivity_cents'], 0.)
            with self.assertRaises(ValueError):
                train(run, 'bpdr', 'research', Path(folder)/'runs', updates=1)


if __name__ == '__main__':
    unittest.main()
