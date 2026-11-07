"""Experimental permission guard for requests at a sample-free endpoint."""

from dataclasses import replace

import numpy as np

from bpdr_input.repair.repair_controller import FixturePolicy, decision_hash, plan_fixture


def plan_sample_backed_fixture(audio, sr, times, corrections, alternate, uncertainty,
                               reliable, protected, policy=FixturePolicy()):
    """Withhold the whole request if a nonzero grid frame has no audio sample."""
    policy = replace(policy, version=f'{policy.version}-sample-backed')
    plan = plan_fixture(audio, sr, times, corrections, alternate, uncertainty,
                        reliable, protected, policy)
    # The validated 20 ms grid is exactly 441 samples per frame at 22,050 Hz.
    unsupported = np.arange(len(times)) * 441 >= len(audio)
    if np.any(unsupported & (np.asarray(corrections) != 0)):
        plan['reasons'].append('REQUEST_WITHOUT_WAVEFORM_SAMPLE')
        plan.update(permitted=False, intervals_seconds=[])
        plan['decision_sha256'] = decision_hash(plan)
    return plan
