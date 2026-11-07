"""Acquisition guards using synthetic metadata; no real source/network access."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))
import acquire_humtrans_pilot as acquire
import test_humtrans_metadata_audit as metadata_tests


class AcquisitionGuards(unittest.TestCase):
    def prepare(self, root):
        raw = json.dumps(metadata_tests.HumTransMetadataTests().fixture()).encode()
        saved = acquire.metadata.audit(json.loads(raw))
        saved.update(dataset_revision=acquire.metadata.REVISION,
                     metadata_sha256=hashlib.sha256(raw).hexdigest(),
                     auditor_sha256=hashlib.sha256(Path(acquire.metadata.__file__).read_bytes()).hexdigest())
        (root / 'train_valid_test_keys.json').write_bytes(raw)
        (root / 'audit_and_reservation.json').write_text(json.dumps(saved))
        (root / 'README.upstream.md').write_text('license: cc-by-nc-4.0\nSynthetic test fixture only\n')
        return saved

    def test_missing_reservation_stops_before_network(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(acquire, 'ROOT', Path(folder)), patch.object(acquire, 'RangeArchive') as network:
                with self.assertRaises(FileNotFoundError):
                    acquire.main()
                network.assert_not_called()

    def test_replayed_reservation_excludes_confirmation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.prepare(root)
            with patch.object(acquire, 'ROOT', root):
                _, _, selected = acquire.reservation()
            self.assertEqual(len(selected), 56)
            self.assertFalse({row['person'] for _, row in selected} & {'F05', 'M05'})
            self.assertEqual({role for role, _ in selected}, {'train', 'development'})

    def test_tampered_reservation_stops_before_network(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            saved = self.prepare(root)
            saved['reservation']['train'][0]['person'] = 'F05'
            (root / 'audit_and_reservation.json').write_text(json.dumps(saved))
            with patch.object(acquire, 'ROOT', root), patch.object(acquire, 'RangeArchive') as network:
                with self.assertRaisesRegex(ValueError, 'replay differs'):
                    acquire.main()
                network.assert_not_called()


if __name__ == '__main__':
    unittest.main()
