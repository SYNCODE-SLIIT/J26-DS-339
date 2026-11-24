import json
from pathlib import Path
import unittest
import numpy as np
import torch

from bpdr_input.modeling.local_context_pilot import FRAME_FEATURES, LocalContextMLP, context_features, load_candidate, load_checkpoint, predict_raw
from bpdr_input.experiments.learning.small_real_pilot import ROOT, select_subset


def raw_example():
    raw = np.zeros((151, 71), np.float32)
    raw[:, 64] = 7.+3*np.sin(np.arange(151)*.6)
    raw[:, 65:67] = .95, 1.
    raw[:, 68] = -30.
    raw[:, 70] = np.exp(-((np.arange(151)-70)/4)**2)
    return raw


class LocalContextPilotTests(unittest.TestCase):
    def test_boundary_only_changes_cannot_modify_acoustic_context(self):
        a = raw_example()
        b = a.copy()
        b[:, 70] = np.roll(a[:, 70], 4)
        fa, fb = context_features(a), context_features(b)
        np.testing.assert_array_equal(fa[:, :-1], fb[:, :-1])
        self.assertFalse(np.array_equal(fa[:, -1], fb[:, -1]))

    def test_transposition_keeps_relative_context_and_voiced_a4(self):
        a = raw_example()
        b = a.copy()
        b[:, 64] += 300.
        np.testing.assert_allclose(context_features(a), context_features(b), atol=2e-5)
        a[:, 64] = 0.
        features = context_features(a)
        self.assertTrue(np.all(features[:, 5] == 1.))
        self.assertEqual(features.shape[1], FRAME_FEATURES+1)

    def test_unvoiced_pitch_placeholders_cannot_become_context(self):
        a = raw_example()
        a[30:40, 66] = 0
        b = a.copy()
        b[30:40, 64] = 9000.
        np.testing.assert_array_equal(context_features(a), context_features(b))

    def test_inference_interface_needs_no_target_or_original(self):
        model = LocalContextMLP()
        raw = raw_example()
        raw[10:15, 66] = 0
        output = predict_raw(model, raw)
        self.assertEqual(output.shape, (151,))
        self.assertTrue(np.isfinite(output).all())
        np.testing.assert_array_equal(output[10:15], np.zeros(5))

    def test_training_subset_does_not_depend_on_scores_or_catalog_order(self):
        rows = []
        for dataset in ("vocadito", "jamendo"):
            for sign in (-1, 1):
                for corruption in ("split", "jitter"):
                    rows.append(dict(dataset=dataset,singer_id="s1",cents=sign*20.,corruption=corruption,
                                     phrase_id="p1",input_directory=dataset+str(sign)+corruption))
        self.assertEqual(select_subset(rows, 1), select_subset(list(reversed(rows)), 1))

    def test_failed_development_gates_cannot_be_loaded_for_demo(self):
        current = ROOT / "current.json"
        if not current.exists():
            self.skipTest("No bounded pilot has been executed")
        directory = Path(json.loads(current.read_text())["path"])
        evidence = json.loads((directory/"evidence.json").read_text())
        for result in evidence["results"].values():
            if not result["eligible_for_demo"]:
                with self.assertRaisesRegex(ValueError, "did not pass"):
                    load_candidate(result["checkpoint"])

    def test_saved_adapter_reproduces_independently_scored_single_view(self):
        current = ROOT / "current.json"
        if not current.exists():
            self.skipTest("No bounded pilot has been executed")
        directory = Path(json.loads(current.read_text())["path"])
        record = json.loads((directory/"selected_records.json").read_text())["development"][0]
        with np.load(Path(record["input_directory"])/"C_input.npz") as z:
            raw = z["boundary_features_71"]
        for condition in ("augmentation_only", "frame_only", "bpdr"):
            model, payload = load_checkpoint(directory/(condition+".pt"))
            with np.load(directory/(condition+"_development_predictions.npz")) as z:
                np.testing.assert_allclose(predict_raw(model, raw), z["0_C"], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
