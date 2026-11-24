"""Replay fixed BPDR Transformer predictions through the existing controller.

Post-fit engineering audit only: no fitting, threshold changes, model selection,
waveform rendering or demo activation. Reuses the old comparison's fixed gain=1
and deadband=3 cents. This is not the demo's tuned ExtraTrees operating policy.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(WORKSPACE/'source'))
from train_context_transformer import attach_context, neural_predictor, read, sha, write
from compare_recovery_models import aggregate, score_case
from bpdr_input.experiments.learning import model_recovery as recovery
from bpdr_input.modeling.context_transformer import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path)
    args = parser.parse_args()
    run = args.run or Path(read(WORKSPACE/'results/transformer_recovery/latest_experiment.json')['path'])
    evidence = read(run/'evidence.json')
    model,_ = load_checkpoint(run/'bpdr/checkpoint.pt')
    assert sha(run/'bpdr/checkpoint.pt') == read(run/'locked_endpoints.json')['fits']['bpdr']['checkpoint_sha256']
    data = Path(read(run/'declaration.json')['data_run'])
    torch.set_num_threads(2)
    pairs = attach_context(recovery.cached_pairs(read(data/'calibration_pairs.json')))
    predictor = neural_predictor(model)
    cache, arrays, rows = {}, {}, []
    for index,pair in enumerate(pairs):
        if 'excluded' in pair:
            rows.append(dict(singer=pair['source']['singer_id'],cents=pair['cents'],excluded=pair['excluded']))
            continue
        predictions = {}
        for view in ('A','C'):
            key = pair['path_'+view]
            if key not in cache:
                cache[key] = predictor(pair,view)
            predictions[view] = cache[key]
            arrays[f'{index}_{view}'] = cache[key]
        row,curves = score_case(pair,predictions)
        rows.append(row)
        for view in ('A','C'):
            arrays[f'{index}_permitted_{view}'] = curves[view]
    summary,groups = aggregate(rows)
    np.testing.assert_allclose(summary['raw_mae_cents'],evidence['results']['bpdr']['summary']['raw_mae_cents'],atol=1e-6)
    np.savez_compressed(run/'bpdr_permission_predictions.npz',**arrays)
    protected = read(run/'declaration.json')['protected_artifacts']
    assert all(sha(p)==digest for p,digest in protected.items())
    result = dict(protocol='Fixed historical-comparison controller, gain 1 / deadband 3c; post-fit diagnostic',
        checkpoint_sha256=sha(run/'bpdr/checkpoint.pt'),source_sha256=sha(__file__),
        existing_score_case_sha256=sha(WORKSPACE/'tools/compare_recovery_models.py'),
        summary=summary,by_singer=groups,rows=rows,
        predictions_sha256=sha(run/'bpdr_permission_predictions.npz'),
        protected_artifacts_unchanged=True,waveform_rendered=False,demo_activated=False)
    write(run/'permission_audit.json',result)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
