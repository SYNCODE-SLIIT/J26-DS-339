"""Real v6 integration and architectural invariants, without full training."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from experiments.rahul.src.inspect_transformer import DEFAULT_DATASET, mixed_length_examples
from experiments.rahul.src.model import DatasetSpec, ModelConfig, ChordRefinementTransformer, masked_cross_entropy
from experiments.rahul.src.model.attention import MultiHeadSelfAttention
from experiments.rahul.src.model.checkpoint import save_checkpoint, load_checkpoint
from experiments.rahul.src.model.positional_encoding import SinusoidalPositionalEncoding
from experiments.rahul.src.transformer_dataset import SavedSequenceDataset, make_dataloader, collate_sequences


class TransformerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.spec = DatasetSpec.from_folder(DEFAULT_DATASET)
        cls.examples = mixed_length_examples(DEFAULT_DATASET)
        cls.batch = collate_sequences(cls.examples)

    def setUp(self):
        torch.manual_seed(42)
        self.config = ModelConfig(d_model=32, num_heads=4, d_ff=64, chord_dim=8,
                                  melody_dim=8, harmonic_dim=8, genre_temporal_dim=8, dropout=0)
        self.model = ChordRefinementTransformer(self.spec, self.config).eval()
        self.inputs = {k: v.clone() for k, v in self.batch['inputs'].items()}
        self.targets = {k: v.clone() for k, v in self.batch['targets'].items()}

    def test_real_loader_contract_and_dtypes(self):
        batch = next(iter(make_dataloader(SavedSequenceDataset(DEFAULT_DATASET/'train_sequences.json'))))
        self.assertEqual(batch['inputs']['basic_chord_ids'].dtype, torch.int64)
        self.assertEqual(batch['inputs']['melody_timing'].dtype, torch.float32)
        self.assertEqual(batch['inputs']['beat_attention_mask'].dtype, torch.bool)
        self.assertEqual(batch['inputs']['melody_timing'].shape[-2:], (self.spec.note_capacity, 3))
        self.assertEqual(batch['inputs']['genre_features'].shape[-1], self.spec.genre_dim)
        self.assertFalse(any(k.startswith('inputs__') for k in batch['inputs']))

    def test_variable_length_batches(self):
        for length, example in zip((12, 16, 20), self.examples):
            batch = collate_sequences([example])
            self.assertEqual(self.model(batch['inputs']).shape, (1, length, self.spec.num_target_classes))
        self.assertEqual(self.inputs['beat_attention_mask'].sum(1).tolist(), [12, 16, 20])
        self.assertEqual(self.model(self.inputs).shape, (3, 20, self.spec.num_target_classes))

    def test_feature_shapes(self):
        logits, details = self.model(self.inputs, True)
        for features in details['features'].values():
            self.assertEqual(features.shape, (3, 20, 8))
        self.assertEqual(details['fused'].shape, (3, 20, 32))
        self.assertEqual(logits.shape, (3, 20, self.spec.num_target_classes))

    def test_masked_notes_do_not_affect_pool(self):
        encoder = self.model.encoders['melody']
        expected = encoder(self.inputs)
        unused = ~self.inputs['melody_note_mask']
        self.inputs['melody_pitch_ids'][unused] = 99999
        self.inputs['melody_timing'][unused] = float('nan')
        torch.testing.assert_close(expected, encoder(self.inputs), rtol=0, atol=0)

    def test_silent_beats_ignore_dummy_attention_slot(self):
        self.inputs['melody_note_mask'].zero_()
        self.inputs['melody_silent'].fill_(True)
        self.inputs['melody_attention_mask'].zero_()
        self.inputs['melody_attention_mask'][..., 0] = True
        output = self.model.encoders['melody'](self.inputs)
        self.assertTrue(torch.isfinite(output).all())
        expected = self.model.encoders['melody'].silence.weight[1].expand_as(output)
        torch.testing.assert_close(output, expected)

    def test_multi_hot_genre(self):
        self.inputs['genre_features'].zero_()
        self.inputs['genre_features'][:, :3] = 1
        encoder = self.model.encoders['genre_temporal']
        genre = encoder.genre(self.inputs['genre_features'])
        expected = encoder.genre.bias + encoder.genre.weight[:, :3].sum(1)
        torch.testing.assert_close(genre, expected.expand_as(genre))
        self.assertTrue(torch.isfinite(encoder(self.inputs)).all())

    def test_qkv_head_dimensions(self):
        features = self.model.fuse_features(self.model.encode_features(self.inputs), self.inputs['beat_attention_mask'])
        attention = self.model.layers[0].attention
        for tensor in attention.project_qkv(features):
            self.assertEqual(tensor.shape, (3, 4, 20, 8))

    def test_manual_attention_matches_equation(self):
        attention = MultiHeadSelfAttention(8, 2, 0).eval()
        x = torch.randn(1, 3, 8)
        mask = torch.tensor([[True, True, False]])
        output, weights = attention(x, mask, True)
        q, k, v = attention.project_qkv(x.masked_fill(~mask.unsqueeze(-1), 0))
        scores = q @ k.transpose(-2, -1) / (4**0.5)
        expected = scores.masked_fill(~mask[:, None, None, :], -torch.inf).softmax(-1)
        expected = expected.masked_fill(~mask[:, None, :, None], 0)
        torch.testing.assert_close(weights, expected)
        combined = (expected @ v).transpose(1, 2).reshape(1, 3, 8)
        torch.testing.assert_close(output, attention.output(combined).masked_fill(~mask.unsqueeze(-1), 0))

    def test_padding_cannot_influence_real_outputs(self):
        expected = self.model(self.inputs)
        padding = ~self.inputs['beat_attention_mask']
        for key, value in self.inputs.items():
            if key in ('genre_features', 'beat_attention_mask'):
                continue
            if value.dtype == torch.float32:
                value[padding] = 1234.5
            elif value.dtype == torch.int64:
                value[padding] = 1
        actual, details = self.model(self.inputs, True)
        torch.testing.assert_close(expected[~padding], actual[~padding], rtol=0, atol=0)
        self.assertTrue((actual[padding] == 0).all())
        for weights in details['attention']:
            self.assertTrue((weights.masked_select(padding[:, None, None, :]) == 0).all())

    def test_padded_batch_matches_single_sequence(self):
        mixed = self.model(self.inputs)[0, :12]
        single = self.model(collate_sequences([self.examples[0]])['inputs'])[0]
        torch.testing.assert_close(mixed, single, rtol=1e-5, atol=1e-6)

    def test_attention_is_bidirectional(self):
        attention = self.model.layers[0].attention
        x = torch.randn(1, 4, 32)
        mask = torch.ones(1, 4, dtype=torch.bool)
        before, weights = attention(x, mask, True)
        self.assertGreater(float(weights[0, 0, 0, 3].detach()), 0)
        self.assertGreater(float(weights[0, 0, 3, 0].detach()), 0)
        x[:, -1] += torch.randn(32)
        self.assertFalse(torch.allclose(before[:, 0], attention(x, mask)[:, 0]))

    def test_masked_loss_matches_manual_selection(self):
        logits = self.model(self.inputs)
        valid = self.targets['target_loss_mask'] & self.inputs['beat_attention_mask']
        expected = F.cross_entropy(logits[valid], self.targets['target_chord_ids'][valid])
        actual = masked_cross_entropy(logits, self.targets, self.inputs['beat_attention_mask'])
        torch.testing.assert_close(actual, expected)
        altered = logits.clone()
        altered[~valid] = 999
        torch.testing.assert_close(actual, masked_cross_entropy(altered, self.targets, self.inputs['beat_attention_mask']))

    def test_empty_supervision_is_differentiable_zero(self):
        logits = self.model(self.inputs)
        self.targets['target_loss_mask'].zero_()
        loss = masked_cross_entropy(logits, self.targets, self.inputs['beat_attention_mask'])
        self.assertEqual(float(loss.detach()), 0)
        loss.backward()
        self.assertTrue(torch.isfinite(self.model.classifier.weight.grad).all())

    def test_invalid_active_target_fails(self):
        self.targets['target_loss_mask'][0, 0] = True
        self.targets['target_chord_ids'][0, 0] = -100
        with self.assertRaises(ValueError):
            masked_cross_entropy(self.model(self.inputs), self.targets, self.inputs['beat_attention_mask'])

    def test_gradient_flow_and_optimizer_update(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        before = {k: p.clone().detach() for k, p in self.model.named_parameters()}
        loss = masked_cross_entropy(self.model(self.inputs), self.targets, self.inputs['beat_attention_mask'])
        loss.backward()
        for name, parameter in self.model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        for name, module in self.model.encoders.items():
            self.assertTrue(any(p.grad.abs().sum() > 0 for p in module.parameters()), name)
        optimizer.step()
        self.assertTrue(any(not torch.equal(before[k], p) for k, p in self.model.named_parameters()))

    def test_finite_logits_loss_attention_and_all_padding(self):
        logits, details = self.model(self.inputs, True)
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.isfinite(masked_cross_entropy(logits, self.targets, self.inputs['beat_attention_mask'])))
        self.assertTrue(all(torch.isfinite(a).all() for a in details['attention']))
        self.inputs['beat_attention_mask'].zero_()
        logits, details = self.model(self.inputs, True)
        self.assertTrue((logits == 0).all())
        self.assertTrue(all(torch.isfinite(a).all() and (a == 0).all() for a in details['attention']))

    def test_vocabulary_ranges(self):
        for key, size in self.spec.vocabulary_sizes.items():
            self.assertTrue(((self.inputs[key] >= 0) & (self.inputs[key] < size)).all(), key)
        pitches = self.inputs['melody_pitch_ids']
        self.assertTrue(((pitches >= 0) & (pitches < self.spec.pitch_vocab_size)).all())
        valid = self.targets['target_loss_mask']
        self.assertTrue(((self.targets['target_chord_ids'][valid] >= 2) &
                         (self.targets['target_chord_ids'][valid] < self.spec.num_target_classes)).all())

    def test_no_target_or_audit_features_used(self):
        expected = self.model(self.inputs)
        self.inputs['reference_chord'] = 'must never be read'
        self.inputs['target_chord_ids'] = torch.full((3, 20), 99999)
        self.inputs['source_roman'] = 'must never be read'
        torch.testing.assert_close(expected, self.model(self.inputs), rtol=0, atol=0)
        self.assertFalse(any('reference' in key or 'target' in key for key in self.batch['inputs']))

    def test_harmonic_features_come_from_basic_key_rule(self):
        from experiments.rahul.src.build_synthetic_dataset import context, simplify
        first = simplify('F:maj7')
        second = simplify('F:maj/3')
        self.assertEqual(context(first, 'C', 'major'), ('IV', 'predominant'))
        self.assertEqual(context(first, 'C', 'major'), context(second, 'C', 'major'))
        # Loading a Phase-1 file recomputes context(basic,key,mode) and rejects mismatches.
        from experiments.rahul.src.encode_features import load_events
        import json
        cfg = json.loads((DEFAULT_DATASET/'preprocessing_config.json').read_text())
        from experiments.rahul import config
        root = Path(cfg['phase1_output_root'])
        if not root.is_absolute():
            root = config.REPOSITORY_ROOT/root
        source = next(root.rglob('*_training_events.csv'))
        rows, _ = load_events(source)
        self.assertGreater(len(rows), 0)

    def test_deterministic_initialization(self):
        torch.manual_seed(17)
        first = ChordRefinementTransformer(self.spec, self.config).eval()
        torch.manual_seed(17)
        second = ChordRefinementTransformer(self.spec, self.config).eval()
        torch.testing.assert_close(first(self.inputs), second(self.inputs), rtol=0, atol=0)

    def test_checkpoint_round_trip(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'model.pt'
            save_checkpoint(path, self.model)
            restored = load_checkpoint(path, DEFAULT_DATASET)
            torch.testing.assert_close(self.model(self.inputs), restored(self.inputs), rtol=0, atol=0)
            saved = torch.load(path, weights_only=True)
            saved['spec']['vocabulary_hashes'] = {}
            torch.save(saved, path)
            with self.assertRaises(ValueError):
                load_checkpoint(path, DEFAULT_DATASET)

    def test_all_experiments_and_optional_features(self):
        for name in ('baseline', 'A', 'B', 'C', 'full'):
            settings = ModelConfig.experiment(name, d_model=32, d_ff=64, dropout=0)
            model = ChordRefinementTransformer(self.spec, settings).eval()
            self.assertTrue(torch.isfinite(model(self.inputs)).all(), name)
        model = ChordRefinementTransformer(self.spec, replace(self.config, use_root_quality=True,
            use_neighbor_chords=True, use_pitch_class=True, use_continuous_pitch=True))
        self.assertTrue(torch.isfinite(model(self.inputs)).all())

    def test_melody_source_is_not_a_model_input(self):
        expected = self.model(self.inputs)
        self.inputs.pop('melody_source_ids', None)
        torch.testing.assert_close(expected, self.model(self.inputs), rtol=0, atol=0)
        self.inputs['melody_source_ids'] = 'audit only; must not be read'
        torch.testing.assert_close(expected, self.model(self.inputs), rtol=0, atol=0)
        self.assertNotIn('melody_source_ids', self.spec.vocabulary_sizes)

    def test_unknown_timing_does_not_read_placeholder_values(self):
        encoder = self.model.encoders['genre_temporal']
        self.inputs['beat_timing_known_mask'].zero_()
        expected = encoder(self.inputs)
        self.inputs['beat_position'].fill_(999)
        self.inputs['relative_bar_number'].fill_(999)
        torch.testing.assert_close(expected, encoder(self.inputs), rtol=0, atol=0)

    def test_explicit_sinusoidal_positions_and_odd_dimensions(self):
        encoder = SinusoidalPositionalEncoding()
        actual = encoder(torch.zeros(2, 25, 7))
        self.assertEqual(actual.shape, (2, 25, 7))
        self.assertAlmostEqual(float(actual[0, 0, 0]), 0)
        self.assertAlmostEqual(float(actual[0, 0, 1]), 1)
        self.assertAlmostEqual(float(actual[0, 1, 0]), float(np.sin(1)), places=6)

    def test_top_three_prediction_decoding_and_padding(self):
        self.model.train()
        predictions = self.model.predict_topk(self.inputs)
        self.assertTrue(self.model.training)
        self.assertEqual(len(predictions[0][0]), 3)
        self.assertEqual(predictions[0][-1], [])
        for candidate in predictions[0][0]:
            self.assertEqual(candidate['label'], self.spec.target_labels[candidate['id']])
            self.assertNotIn(candidate['label'], ('PAD', 'UNK'))
            self.assertTrue(0 <= candidate['probability'] <= 1)

    def test_invalid_configuration(self):
        with self.assertRaises(ValueError):
            ModelConfig(d_model=31)
        with self.assertRaises(ValueError):
            ModelConfig(dropout=1)


if __name__ == '__main__':
    unittest.main()
