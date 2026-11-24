"""Resume unchanged fits with an explicit numerical duplicate-view check.

Only the frame-only within-batch equality assertion changes (four float32
ULPs). Inputs are still exact; saved prediction replay remains bitwise.
"""

import argparse
import ast
import hashlib
import inspect
from pathlib import Path

import numpy as np
import torch

import compare_balanced_short_objectives as study


ROOT = study.ROOT / 'numerical_resume'


def main(mode):
    torch.set_num_threads(2)
    study.previous.preflight()
    declaration, rows, originals = study.declare()
    diagnostic = study.ROOT.parent / '2026-11-08_frame_duplicate_diagnosis/results.json'
    measured = study.read(diagnostic)
    assert measured['duplicate_tokens_exact'] and measured['repeats_exact']
    assert measured['maximum_float32_ulps'] == 2.
    ROOT.mkdir(exist_ok=True)
    source = inspect.getsource(study.run)
    tree = ast.parse(source)

    class ReplaceDuplicateAssertion(ast.NodeTransformer):
        count = 0

        def visit_Call(self, node):
            expected = "np.testing.assert_array_equal(value[0], value[1])"
            if ast.dump(node) == ast.dump(ast.parse(expected, mode='eval').body):
                self.count += 1
                return ast.copy_location(ast.Call(func=ast.Name(id='check_duplicate', ctx=ast.Load()),
                                                  args=node.args, keywords=[]), node)
            return self.generic_visit(node)

    transformer = ReplaceDuplicateAssertion()
    tree = ast.fix_missing_locations(transformer.visit(tree))
    assert transformer.count == 1
    transformed = ast.unparse(tree)
    protocol = dict(wrapper_sha256=study.sha(Path(__file__)), parent_declaration_sha256=study.sha(study.ROOT / 'declaration.json'),
        frozen_runner_sha256=study.sha(Path(study.__file__)), diagnostic_sha256=study.sha(diagnostic),
        transformed_run_sha256=hashlib.sha256(transformed.encode()).hexdigest(),
        changed_check='Frame-only duplicate-view output at most four float32 ULPs; exact input equality required.',
        unchanged='All fitting, traces, initialization, budgets, raw predictions, scientific gates and exact checkpoint/prediction replay.',
        fits='Reuse all verified existing fit receipts; finish only the missing fits from the original twelve.',
        production_eligible=False)
    path = ROOT / 'declaration.json'
    if path.exists():
        previous_protocol = study.read(path)
        assert all(previous_protocol[k] == v for k, v in protocol.items())
        for filename, digest in previous_protocol['initial_fit_artifacts'].items():
            assert study.sha(Path(filename)) == digest
    else:
        fits = sorted(study.ROOT.glob('fold_*/*/fit_receipt.json'))
        assert len(fits) == 5
        protocol['initial_fit_artifacts'] = {str(p): study.sha(p) for r in fits for p in [r, r.parent / 'checkpoint.pt']}
        study.write(path, protocol)
        (ROOT / 'frozen_run.py.txt').write_text(source, encoding='utf-8')
        (ROOT / 'numerical_run.py.txt').write_text(transformed, encoding='utf-8')
        (ROOT / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    residuals = []

    def check_duplicate(a, b):
        assert a.dtype == b.dtype == np.float32
        np.testing.assert_array_max_ulp(a, b, maxulp=4)

    original_predict = study.previous.predict

    def guarded_predict(model, record, condition):
        if condition == 'frame_only':
            for start in range(0, len(record['raw']), 128):
                centers = np.arange(start, min(start + 128, len(record['raw'])))
                views, _ = study.previous.select_tokens(record, centers, condition)
                np.testing.assert_array_equal(views[0], views[1])
                assert np.count_nonzero(views[..., 8:]) == 0
        values = original_predict(model, record, condition)
        if condition == 'frame_only':
            check_duplicate(values[0], values[1])
            if not np.array_equal(values[0], values[1]):
                residuals.append(dict(record_id=record['id'], maximum_cents=float(np.max(abs(values[0] - values[1])))))
        return values

    # Preserve existing files and demand exact equality before reusing a name.
    original_savez = np.savez_compressed

    def preserve_predictions(filename, **arrays):
        if Path(filename).exists():
            with np.load(filename) as saved:
                assert set(saved.files) == set(arrays)
                for key, value in arrays.items():
                    np.testing.assert_array_equal(value, saved[key])
        else:
            original_savez(filename, **arrays)

    namespace = dict(study.__dict__, check_duplicate=check_duplicate)
    exec(compile(tree, str(ROOT / 'numerical_run.py.txt'), 'exec'), namespace)
    study.previous.predict = guarded_predict
    np.savez_compressed = preserve_predictions
    try:
        namespace['run'](declaration, study.records(rows, originals), mode)
    finally:
        study.previous.predict = original_predict
        np.savez_compressed = original_savez
        study.write(ROOT / f'{mode}_numerical_observations.json', dict(
            declaration_sha256=study.sha(path), residuals=residuals,
            prediction_values_modified=False, max_allowed_float32_ulps=4))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['fit', 'verify'], required=True)
    main(parser.parse_args().mode)
