"""Regression coverage for a controller plan that would fail at waveform end."""

from dataclasses import replace
import unittest

import numpy as np

from bpdr_input.repair.repair_controller import FixturePolicy, transact_fixture
from bpdr_input.repair.sample_backed_permission import plan_sample_backed_fixture


class SampleBackedPermissionTests(unittest.TestCase):
    def plan(self, sample_count, include_endpoint=True):
        audio = np.zeros(sample_count, dtype=np.float32)
        times = np.arange(sample_count // 441 + 1) * .02
        correction = np.where(times >= 11.72, 15., 0.)
        if not include_endpoint:
            correction[-1] = 0.
        reliable = np.ones(len(times), dtype=bool)
        protected = np.zeros_like(reliable)
        plan = plan_sample_backed_fixture(
            audio, 22050, times, correction, correction, np.zeros_like(correction),
            reliable, protected, replace(FixturePolicy(), min_support_seconds=.24)
        )
        return audio, plan

    def test_sample_free_endpoint_withholds_without_rendering(self):
        audio, plan = self.plan(12 * 22050)
        self.assertEqual(plan['reasons'], ['REQUEST_WITHOUT_WAVEFORM_SAMPLE'])
        self.assertEqual(plan['intervals_seconds'], [])

        def forbidden(*args):
            self.fail('A withheld proposal must not render or audit audio')

        output, status = transact_fixture(audio, 22050, plan, forbidden, forbidden)
        self.assertEqual(status['outcome'], 'WITHHELD')
        np.testing.assert_array_equal(output, audio)

    def test_last_frame_with_a_real_sample_is_allowed(self):
        _, plan = self.plan(12 * 22050 + 1)
        self.assertTrue(plan['permitted'])
        self.assertEqual(plan['reasons'], [])

    def test_zero_request_at_sample_free_endpoint_is_allowed(self):
        _, plan = self.plan(12 * 22050, include_endpoint=False)
        self.assertTrue(plan['permitted'])
        self.assertEqual(plan['intervals_seconds'], [(11.72, 12.)])
