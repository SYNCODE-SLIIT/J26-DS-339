"""Protect abstention and expressive-input behavior of the provisional demo gate."""
import json
import unittest

import numpy as np

from bpdr_input.audio.contracts import Status
from bpdr_input.audio.usability import assess, UsabilityPolicy


def phrase(pitch=None, count=100, voiced=None):
    pitch = np.full(count, 220.) if pitch is None else np.asarray(pitch, dtype=float)
    count = len(pitch)
    voiced = np.ones(count, dtype=bool) if voiced is None else np.asarray(voiced, dtype=bool)
    return dict(times_seconds=np.arange(count)*.02, f0_hz=pitch,
                confidence=np.where(voiced, .95, .1), voiced_mask=voiced,
                log_energy_db=np.full(count, -20.))


def check(features, secondary=None, intent="singing_humming_chant", technical=Status.ACCEPT_UNCHANGED, **kwargs):
    if secondary is None and features is not None:
        secondary = np.asarray(features["f0_hz"]).copy()
    return assess(features, technical, intent, independent_f0_hz=secondary, **kwargs)


class ProvisionalUsabilityTests(unittest.TestCase):
    def test_technical_failure_is_separate_and_not_a_singing_verdict(self):
        result = check(None, technical=Status.TECHNICAL_FAILURE)
        self.assertEqual(result["decision"], "NOT_ASSESSED")
        self.assertEqual(result["next_step"], "FOLLOW_TECHNICAL_FEEDBACK")
        self.assertFalse(result["handoff_approval"])

    def test_silence_or_no_reliable_pitch_requests_clearer_take(self):
        f = phrase(np.full(100, np.nan), voiced=np.zeros(100, dtype=bool))
        result = check(f)
        self.assertEqual(result["decision"], "NEED_CLEARER_INPUT")
        self.assertIn("NO_RELIABLE_VOICED_EVIDENCE", result["reason_codes"])

    def test_stable_single_note_is_allowed_without_pitch_variation(self):
        result = check(phrase(count=15), duration_seconds=.30)
        self.assertEqual(result["decision"], "PASS_EVIDENCE")
        self.assertEqual(result["observations"]["reliable_pitch_p90_minus_p10_cents"], 0.)
        self.assertFalse(result["musical_correctness_assessed"])
        self.assertFalse(result["intended_note_identified"])

    def test_expressive_slide_does_not_require_stable_plateau_or_scale(self):
        f = phrase(220*2**(np.linspace(0, 1.5, 100)))
        result = check(f, secondary=f["f0_hz"]*2**(10/1200))
        self.assertEqual(result["decision"], "PASS_EVIDENCE")
        self.assertGreater(result["observations"]["reliable_pitch_p90_minus_p10_cents"], 1000)

    def test_long_pauses_and_repeated_notes_are_allowed(self):
        voiced = np.zeros(500, dtype=bool)
        voiced[20:35] = True
        voiced[450:465] = True
        f = phrase(np.where(voiced, 220., np.nan), voiced=voiced)
        result = check(f)
        self.assertEqual(result["decision"], "PASS_EVIDENCE")
        self.assertLess(result["observations"]["reliable_fraction_of_all_frames"], .1)
        self.assertGreater(result["observations"]["longest_internal_unvoiced_gap_seconds"], 8)
        self.assertEqual(result["observations"]["agreeing_fraction_of_primary_voiced"], 1.)

    def test_declared_reading_does_not_become_melody_because_pitch_is_trackable(self):
        result = check(phrase(), intent="reading_spoken_text")
        self.assertEqual(result["decision"], "NEED_CLEARER_INPUT")
        self.assertEqual(result["next_step"], "RECORD_INTENDED_TUNE")
        self.assertIn("NO_INTENDED_TUNE_DECLARED", result["reason_codes"])

    def test_unknown_intent_abstains_instead_of_classifying_pitch_as_singing(self):
        for intent in ("unsure", "unrecognized-declaration"):
            result = check(phrase(), intent=intent)
            self.assertEqual(result["decision"], "NEED_INTENT_CONFIRMATION")
            self.assertFalse(result["handoff_approval"])

    def test_missing_independent_measurement_is_not_a_pass(self):
        result = assess(phrase(), Status.ACCEPT_UNCHANGED, "singing_humming_chant")
        self.assertEqual(result["decision"], "NOT_ASSESSED")
        self.assertIn("INDEPENDENT_TRACKER_UNAVAILABLE", result["reason_codes"])

    def test_disagreeing_tracks_and_isolated_spikes_do_not_pass(self):
        f = phrase()
        secondary = f["f0_hz"]*2
        secondary[::10] = f["f0_hz"][::10]
        result = check(f, secondary=secondary)
        self.assertEqual(result["decision"], "NEED_CLEARER_INPUT")
        self.assertIn("PITCH_SUPPORT_TOO_FRAGMENTED", result["reason_codes"])
        self.assertIn("TRACKERS_DISAGREE_ON_MUCH_OF_VOICED_INPUT", result["reason_codes"])
        self.assertAlmostEqual(result["observations"]["agreeing_fraction_of_primary_voiced"], .1)

    def test_invalid_and_out_of_grid_measurements_abstain(self):
        modifications = [
            ("times_seconds", np.arange(100)*-.02),
            ("times_seconds", np.r_[np.arange(99)*.02, np.inf]),
            ("f0_hz", np.full(100, -1.)),
            ("confidence", np.full(100, np.nan)),
            ("confidence", np.full(100, 1.2)),
            ("voiced_mask", np.full(100, 2)),
        ]
        for key, values in modifications:
            with self.subTest(key=key):
                f=phrase(); f[key]=values
                result=check(f)
                self.assertEqual(result["decision"], "NOT_ASSESSED")
                self.assertIn("ANALYSIS_INVALID", result["reason_codes"])
        self.assertEqual(check(phrase(), duration_seconds=.5)["decision"], "NOT_ASSESSED")
        self.assertEqual(check(phrase(), secondary=np.ones(5))["decision"], "NOT_ASSESSED")

    def test_measurements_are_unmodified_and_result_serializes_without_nan(self):
        f=phrase(); original={k:v.copy() for k,v in f.items()}
        result=check(f)
        for key in f:
            np.testing.assert_array_equal(f[key],original[key])
        json.dumps(result,allow_nan=False)
        self.assertTrue(result["provisional"])
        self.assertIn("unvalidated",result["policy"]["scope"])

    def test_bad_policy_is_not_silently_used(self):
        with self.assertRaises(ValueError):
            UsabilityPolicy(minimum_reliable_seconds=-1)
        with self.assertRaises(ValueError):
            UsabilityPolicy(minimum_confidence=1.2)


if __name__ == "__main__":
    unittest.main()
