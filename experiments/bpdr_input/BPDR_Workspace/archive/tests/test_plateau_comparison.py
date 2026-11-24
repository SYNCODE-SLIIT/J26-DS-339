import ast
import copy
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from bpdr_input import expanded_comparison,plateau_comparison,plateau_objective
from bpdr_input.expanded_experiment_preparation import locked_protocol as previous_protocol


class PlateauAdapterTests(unittest.TestCase):
    def test_engine_wires_new_objective_and_keeps_step_mechanics_exact(self):
        self.assertIs(plateau_comparison.objective,plateau_objective.objective)
        self.assertIsNot(plateau_comparison.objective,expanded_comparison.objective)
        original=ast.dump(ast.parse(inspect.getsource(expanded_comparison.accumulated_step)))
        prepared=ast.dump(ast.parse(inspect.getsource(plateau_comparison.accumulated_step)))
        self.assertEqual(prepared,original)
        for helper in ['assess','same_vectors','to_device','matched_model','select_checkpoint']:
            self.assertIs(getattr(plateau_comparison,helper),getattr(expanded_comparison,helper))

    def test_sealed_protocol_refuses_changed_objective_or_matched_settings(self):
        with tempfile.TemporaryDirectory() as temporary:
            reference=previous_protocol();reference['fit']['weight_decay']=.01
            path=Path(temporary)/'locked_protocol.json';path.write_text(json.dumps(reference),encoding='utf-8')
            p=plateau_objective.locked_protocol(reference,temporary);schedule=np.zeros((300,8),np.int64)
            plateau_comparison.validate_protocol(p,schedule,995)
            for key in ['objective','fit','checkpoint_selection']:
                changed=copy.deepcopy(p);changed[key]['unexpected_change']=True
                with self.assertRaises(ValueError):plateau_comparison.validate_protocol(changed,schedule,995)
            reference['fit']['updates']=301;path.write_text(json.dumps(reference),encoding='utf-8')
            with self.assertRaises(ValueError):plateau_comparison.validate_protocol(p,schedule,995)

    def test_recorded_run_is_never_silently_repeated(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);(root/'runs').mkdir();(root/'runs/plateau_comparison_current.json').write_text('{}')
            with patch.object(plateau_comparison,'matched_model') as model,patch.object(plateau_comparison.torch.optim,'AdamW') as optimizer:
                with self.assertRaises(ValueError):plateau_comparison.run(root)
                model.assert_not_called();optimizer.assert_not_called()
