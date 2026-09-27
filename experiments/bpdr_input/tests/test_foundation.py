import csv
import json
from pathlib import Path
import tempfile
import unittest
import wave

from bpdr_input.contracts import AcceptedAudio, ProcessingResult, Status, require_accepted_audio
from bpdr_input.inventory import audit_vocadito


class ContractTests(unittest.TestCase):
    def audio(self):
        return AcceptedAudio("accepted.wav", "a" * 64, 22050, 44100)

    def test_non_accepted_statuses_block_consumer(self):
        for status in (Status.RERECORD_REQUIRED, Status.TECHNICAL_FAILURE):
            result = ProcessingResult("request", "source", status, reason_codes=("TEST_REASON",))
            self.assertIsNone(result.to_dict()["accepted_audio"])
            with self.assertRaises(ValueError):
                require_accepted_audio(result)
            with self.assertRaises(ValueError):
                ProcessingResult("request", "source", status, self.audio())

    def test_accepted_statuses_require_audio(self):
        for status in (Status.ACCEPT_UNCHANGED, Status.ACCEPT_PREPARED):
            with self.assertRaises(ValueError):
                ProcessingResult("request", "source", status)
            result = ProcessingResult("request", "source", status, self.audio())
            self.assertEqual(require_accepted_audio(result).duration_seconds, 2)
            encoded = json.loads(json.dumps(result.to_dict()))
            self.assertEqual(encoded["status"], status.value)
            self.assertEqual(encoded["accepted_audio"]["duration_seconds"], 2)

    def test_invalid_metadata_is_rejected(self):
        with self.assertRaises(ValueError):
            AcceptedAudio("audio.wav", "not-a-checksum", 22050, 1)
        with self.assertRaises(ValueError):
            AcceptedAudio("audio.wav", "a" * 64, 0, 1)
        with self.assertRaises(ValueError):
            AcceptedAudio("audio.wav", "a" * 64, 22050, 1, 2)
        with self.assertRaises(TypeError):
            ProcessingResult("request", "source", "ACCEPT_UNCHANGED", self.audio())


class InventoryTests(unittest.TestCase):
    def make_dataset(self, root):
        (root / "Audio").mkdir()
        with wave.open(str(root / "Audio" / "vocadito_1.wav"), "wb") as stream:
            stream.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
            stream.writeframes(b"\x00\x00" * 8000)
        with (root / "vocadito_metadata.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["track_id", "singer_id", "language"])
            writer.writerow(["1", "S1", "none"])
        for folder, suffix in [("F0", "_f0.csv"), ("Notes", "_notesA1.csv"), ("Notes", "_notesA2.csv")]:
            directory = root / "Annotations" / folder
            directory.mkdir(parents=True, exist_ok=True)
            (directory / ("vocadito_1" + suffix)).write_text("0,0\n", encoding="utf-8")

    def test_inventory_reports_missing_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_dataset(root)
            report = audit_vocadito(root)
            self.assertEqual(report["summary"]["recordings_with_errors"], 0)
            self.assertEqual(report["recordings"][0]["duration_seconds"], 1)
            (root / "Annotations" / "Notes" / "vocadito_1_notesA2.csv").unlink()
            report = audit_vocadito(root)
            self.assertEqual(report["summary"]["recordings_with_errors"], 1)
            self.assertIn("MISSING_NOTES_A2", report["recordings"][0]["errors"])

    def test_inventory_reports_truncated_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_dataset(root)
            audio = root / "Audio" / "vocadito_1.wav"
            audio.write_bytes(audio.read_bytes()[:-20])
            report = audit_vocadito(root)
            self.assertIn("TRUNCATED_AUDIO_DATA", report["recordings"][0]["errors"])


if __name__ == "__main__":
    unittest.main()
