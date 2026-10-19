"""Evidence integrity checks: a changed asset must lose its verified status."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import soundfile as sf

from bpdr_input.reporting.panel_demo import (
    DEFAULT_OUTPUT, SCHEMA, WORKSPACE, checksum, load_verified_manifest,
    paired_audio_metrics, within,
)


class PanelDemoTests(unittest.TestCase):
    def _write_manifest(self, root, assets):
        path = root / "manifest.json"
        data = dict(schema_version=SCHEMA, integrity=dict(assets=assets, sources=[]))
        path.write_text(json.dumps(data), encoding="utf-8")
        path.with_suffix(".sha256").write_text(checksum(path)+"  manifest.json\n", encoding="utf-8")
        return path

    def test_changed_asset_cannot_retain_verified_label(self):
        with TemporaryDirectory(dir=WORKSPACE / "tests") as folder:
            root = Path(folder)
            asset = root / "audio.wav"
            asset.write_bytes(b"saved-output")
            path = self._write_manifest(root, [dict(path="audio.wav", sha256=checksum(asset))])
            load_verified_manifest(path)
            asset.write_bytes(b"different-output")
            with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
                load_verified_manifest(path)

    def test_manifest_edits_require_a_new_receipt(self):
        with TemporaryDirectory(dir=WORKSPACE / "tests") as folder:
            path = self._write_manifest(Path(folder), [])
            path.write_text(path.read_text()+" ", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
                load_verified_manifest(path)

    def test_asset_paths_cannot_escape_the_package(self):
        with TemporaryDirectory(dir=WORKSPACE / "tests") as folder:
            path = self._write_manifest(Path(folder), [dict(path="../unrelated.wav", sha256="unused")])
            with self.assertRaisesRegex(ValueError, "must remain inside"):
                load_verified_manifest(path)
        with self.assertRaisesRegex(ValueError, "must remain inside"):
            within(WORKSPACE, WORKSPACE.parent / "outside")

    def test_different_sample_grids_cannot_claim_paired_improvement(self):
        with TemporaryDirectory(dir=WORKSPACE / "tests") as folder:
            a, b = Path(folder)/"a.wav", Path(folder)/"b.wav"
            sf.write(a, np.zeros(8), 22050)
            sf.write(b, np.zeros(8), 44100)
            with self.assertRaisesRegex(ValueError, "sample grids differ"):
                paired_audio_metrics(a, b)

    def test_packaged_claims_preserve_experimental_boundaries(self):
        if not (DEFAULT_OUTPUT / "manifest.json").exists():
            self.skipTest("Build the panel package first")
        data = load_verified_manifest()
        cases = {case["id"]: case for case in data["cases"]}
        for case_id in ("controlled_pitch", "real_humming_noise"):
            self.assertEqual(cases[case_id]["handoff_status"], "ACCEPT_UNCHANGED")
            self.assertFalse(cases[case_id]["metrics"]["prepared_handoff_enabled"])
        self.assertEqual(cases["silent_rerecord"]["handoff_status"], "RERECORD_REQUIRED")
        self.assertNotIn("output", cases["silent_rerecord"])
        self.assertTrue(cases["real_pitch_rollback"]["metrics"]["rollback"])
        self.assertFalse(data["downstream_check"]["shared_pipeline_execution"])
        self.assertEqual(data["downstream_check"]["scores"]["noisy"]["onset_pitch_f1"],
                         data["downstream_check"]["scores"]["prepared_preview"]["onset_pitch_f1"])


if __name__ == "__main__":
    unittest.main()
