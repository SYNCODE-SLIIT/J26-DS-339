import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from bpdr_input.repair import held_voice_runtime as runtime
from bpdr_input.repair.demo_runtime import available_methods
from bpdr_input.audio.pipeline import checksum


class HeldRuntimeTests(unittest.TestCase):
    def test_ineligible_card_is_rejected_before_inference(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'card.json';p.write_text(json.dumps(dict(eligible_for_scoped_preview=False,requires_user_review=True)))
            with self.assertRaisesRegex(ValueError,'not passed'):runtime.load_model(p)

    def test_review_cannot_be_omitted(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'card.json';p.write_text(json.dumps(dict(eligible_for_scoped_preview=True,requires_user_review=False)))
            with self.assertRaisesRegex(ValueError,'not passed'):runtime.load_model(p)

    def test_engineering_card_cannot_assert_research_completion(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'card.json';p.write_text(json.dumps(dict(eligible_for_scoped_preview=True,requires_user_review=True,research_validation_passed=True)))
            with self.assertRaisesRegex(ValueError,'research validation'):runtime.load_model(p)

    def test_unverified_transformer_hidden_from_app_methods(self):
        with patch.object(runtime,'available',return_value=False):
            self.assertNotIn(runtime.METHOD_NAME,available_methods())

    def test_changed_verification_receipt_blocks_checkpoint_loading(self):
        with tempfile.TemporaryDirectory(dir=runtime.WORKSPACE/'results/transformer_usability') as d:
            folder=Path(d);checkpoint=folder/'model.pt';checkpoint.write_bytes(b'fixture')
            evidence=folder/'evidence.json';evidence.write_text(json.dumps(dict(gates=dict(passed=True),checkpoint_sha256=checksum(checkpoint),policy={})))
            receipt=dict(path=str(evidence),sha256=checksum(evidence))
            card=dict(eligible_for_scoped_preview=True,requires_user_review=True,research_validation_passed=False,
                checkpoint=str(checkpoint),checkpoint_sha256=checksum(checkpoint),policy={},verification=dict(development=receipt,final=receipt),runtime_source_hashes={})
            p=folder/'card.json';p.write_text(json.dumps(card));evidence.write_text('{}')
            with patch.object(runtime,'load') as loader:
                with self.assertRaisesRegex(ValueError,'evidence changed'):runtime.load_model(p)
                loader.assert_not_called()


if __name__=='__main__':unittest.main()
