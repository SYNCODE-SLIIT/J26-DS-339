"""Synthetic contract checks only; these do not validate/acquire HumTrans."""
import importlib.util
import itertools
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('humtrans_auditor', Path(__file__).resolve().parents[1] / 'tools/audit_humtrans_metadata.py')
audit_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_module)


class HumTransMetadataTests(unittest.TestCase):
    def fixture(self):
        result = {'TRAIN': [], 'VALID': [], 'TEST': []}
        for person in [f'{p}0{i}' for p in ('F', 'M') for i in range(1, 6)]:
            for music in range(1, 501):
                for index, split in enumerate(result):
                    result[split].append(f'{person}_{music:04}_0001_{1 if index < 2 else 2}' + ('_D' if index == 1 else ''))
        return result

    def test_reservation_excludes_people_music_and_repeat_leakage(self):
        result = audit_module.audit(self.fixture())
        selected = result['reservation']
        self.assertEqual({k: len(v) for k, v in selected.items()}, {'train': 48, 'development': 8, 'confirmation': 8})
        for left, right in itertools.combinations(selected, 2):
            for field in ('person', 'music'):
                self.assertFalse({r[field] for r in selected[left]} & {r[field] for r in selected[right]})
        self.assertEqual(result['official_overlap']['TRAIN/VALID']['repeated_segment_families'], 5000)
        self.assertEqual(result, audit_module.audit(self.fixture()))

    def test_duplicate_and_unknown_ids_refused(self):
        data = self.fixture()
        data['TEST'].append(data['TRAIN'][0])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            audit_module.audit(data)
        data = self.fixture()
        data['TRAIN'][0] = '../unknown.wav'
        with self.assertRaisesRegex(ValueError, 'Unrecognized'):
            audit_module.audit(data)

    def test_changed_official_schema_refused(self):
        with self.assertRaisesRegex(ValueError, 'schema'):
            audit_module.audit({'train': []})

    def test_inventory_mode_never_reserves_or_relabels_existing_people(self):
        result = audit_module.audit(self.fixture(), reserve=False)
        self.assertNotIn('reservation', result)
        self.assertEqual(set(result), {'official_counts', 'official_people', 'official_overlap'})


if __name__ == '__main__':
    unittest.main()
