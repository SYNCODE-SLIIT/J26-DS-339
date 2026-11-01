import unittest
import torch

from bpdr_input.experiments.learning.temporal_transformer_training import adjacent_loss


class TemporalSupervisionTests(unittest.TestCase):
    def test_unknown_ramp_label_cannot_create_a_smoothing_target(self):
        output = torch.tensor([[0., 100., 0.]], requires_grad=True)
        targets = torch.zeros_like(output)
        known = torch.tensor([[True, False, True]])
        loss = adjacent_loss(output, targets, known)
        self.assertEqual(float(loss), 0.)
        loss.backward()
        self.assertTrue(torch.equal(output.grad, torch.zeros_like(output)))

    def test_correct_genuine_labelled_change_has_zero_temporal_loss(self):
        target = torch.tensor([[0., 10., 20.]])
        self.assertEqual(float(adjacent_loss(target.clone(), target, torch.ones_like(target, dtype=torch.bool))), 0.)

    def test_known_plateau_jitter_produces_a_corrective_gradient(self):
        output = torch.tensor([[20., 23., 20.]], requires_grad=True)
        loss = adjacent_loss(output, torch.ones_like(output)*20., torch.ones_like(output, dtype=torch.bool))
        self.assertGreater(float(loss), 0.)
        loss.backward()
        self.assertGreater(float(output.grad[0, 1]), 0.)
        self.assertLess(float(output.grad[0, 0]), 0.)


if __name__ == '__main__':
    unittest.main()
