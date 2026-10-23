import importlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import bpdr_input


class WorkspaceLayoutTests(unittest.TestCase):
    def test_flat_source_root_contains_only_package_initialization(self):
        self.assertEqual([p.name for p in bpdr_input.SOURCE_ROOT.glob('*.py')], ['__init__.py'])
        mapping=json.loads((bpdr_input.SOURCE_ROOT/'_module_map.json').read_text())
        for name,path in mapping.items():
            self.assertTrue((bpdr_input.SOURCE_ROOT/path).is_file(),name)

    def test_renamed_study_imports_share_the_same_module_and_state(self):
        for old,new in [('presentation_study','transfer_study'),('presentation_statistics','paired_statistics'),
                        ('presentation_baselines','baseline_evaluation')]:
            self.assertIs(importlib.import_module('bpdr_input.'+old),importlib.import_module('bpdr_input.'+new))

    def test_source_hashing_finds_modules_across_responsibility_folders(self):
        from bpdr_input.data_builder import code_hashes
        from bpdr_input.laboratory import source_code_sha256
        self.assertTrue(code_hashes())
        self.assertTrue(source_code_sha256())
        self.assertEqual(bpdr_input.module_source('pipeline.py').parent.name,'audio')
        self.assertEqual(bpdr_input.module_source('model.py').parent.name,'modeling')
        self.assertEqual(bpdr_input.module_source('presentation_study.py').name,'transfer_study.py')

    def test_legacy_asset_paths_reach_the_single_physical_dataset(self):
        root=bpdr_input.COMPONENT_ROOT
        self.assertTrue((root/'data').samefile(root/'BPDR_Workspace/datasets'))
        self.assertTrue((root/'runs').samefile(root/'research_archive/evidence'))

    def test_current_runtime_uses_verified_workspace_weights(self):
        from bpdr_input.model_runtime import default_checkpoint
        self.assertEqual(default_checkpoint(),bpdr_input.COMPONENT_ROOT/'BPDR_Workspace/selected_model/bpdr.pt')

    def test_current_imports_are_real_subpackages_with_flat_compatibility(self):
        from bpdr_input.modeling.model import CorrectionEncoder
        from bpdr_input.model import CorrectionEncoder as LegacyEncoder
        self.assertIs(CorrectionEncoder,LegacyEncoder)
        self.assertEqual(CorrectionEncoder.__module__,'bpdr_input.modeling.model')

    def test_changed_workspace_weights_are_rejected(self):
        from bpdr_input.model_runtime import default_checkpoint
        with TemporaryDirectory(dir=bpdr_input.COMPONENT_ROOT/'runs') as temp:
            root=Path(temp);folder=root/'BPDR_Workspace/selected_model';folder.mkdir(parents=True)
            (folder/'bpdr.pt').write_bytes(b'changed')
            (folder/'model.json').write_text(json.dumps(dict(checkpoint='bpdr.pt',sha256='0'*64)))
            with patch('bpdr_input.model_runtime.COMPONENT',root):
                with self.assertRaisesRegex(ValueError,'Working checkpoint checksum mismatch'):
                    default_checkpoint()


if __name__=='__main__':unittest.main()
