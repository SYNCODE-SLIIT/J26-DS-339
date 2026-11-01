"""Withholds/exclusions must not disappear from delivered success rates."""
import unittest
import json
from pathlib import Path
import tempfile
from bpdr_input.reporting.transformer_accuracy_report import evidence_counts,complete_report


class TransformerReportTests(unittest.TestCase):
    def test_all_attempt_rate_keeps_construction_exclusions(self):
        rows=[dict(delivered_verified_improvement=True),dict(delivered_verified_improvement=False),
              dict(excluded='AUDIT_FAILED')]
        result=evidence_counts(dict(rows=rows,rendered_waveforms_evaluated=True,prediction_gate_passed=True))
        self.assertAlmostEqual(result['verified_improvement_percent_all_attempts'],100/3)
        self.assertEqual(result['verified_improvement_percent_scoreable'],50.)
        self.assertEqual(result['unique_encoded_verified_wavs'],0)
        self.assertFalse(result['demo_activated'])

    def test_not_rendered_is_unknown_not_success_or_failure(self):
        result=evidence_counts(dict(rows=[{}],rendered_waveforms_evaluated=False,prediction_gate_passed=False))
        self.assertIsNone(result['delivered_improved_case_evaluations'])
        self.assertIsNone(result['verified_improvement_percent_all_attempts'])

    def test_repeated_original_audits_are_not_new_wavs(self):
        trace=dict(permission=dict(source_sha256='source'),render_attempted=True,
            transaction=dict(outcome='VERIFIED_FIXTURE_CANDIDATE',delivered_wav_audit=dict(passed=True,sha256='wav')))
        rows=[dict(trace={'A':trace},delivered_verified_improvement=False)]*3
        result=evidence_counts(dict(rows=rows,rendered_waveforms_evaluated=True,prediction_gate_passed=False))
        self.assertEqual(result['unique_render_attempts'],1)
        self.assertEqual(result['unique_encoded_verified_wavs'],1)

    def test_component_local_training_manifest_counts_exclusions(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)
            (folder/'declaration.json').write_text(json.dumps(dict(version='held-voice')))
            (folder/'train_pairs.json').write_text(json.dumps([dict(source=dict(sha256='a')),dict(source=dict(sha256='b'),excluded='NO_VOICE')]))
            (folder/'accuracy.json').write_text(json.dumps(dict(rows=[{},dict(excluded='NO_VOICE')],prediction_gate_passed=False,rendered_waveforms_evaluated=False)))
            summary=complete_report(folder)
            self.assertEqual(summary['manifest_attempted_interventions'],2)
            self.assertEqual(summary['manifest_scoreable_interventions'],1)
            self.assertEqual(summary['manifest_source_recordings'],2)


if __name__=='__main__':
    unittest.main()
