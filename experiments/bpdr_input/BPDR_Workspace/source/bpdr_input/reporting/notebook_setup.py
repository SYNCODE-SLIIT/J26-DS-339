"""Shared notebook setup and readable displays; no fitting or downloads."""

def prepare(WORKSPACE):
    import os, sys, json, hashlib, zlib
    from pathlib import Path
    from collections import Counter
    from dataclasses import asdict
    from importlib.metadata import version
    COMPONENT = WORKSPACE.parent
    sys.dont_write_bytecode = True
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import soundfile as sf
    import torch
    import nbformat
    from IPython.display import Markdown, HTML, Audio
    torch.set_num_threads(2)
    plt.rcParams.update({'figure.dpi': 115, 'axes.spines.top': False, 'axes.spines.right': False})
    COLORS = {'input':'#9b645e', 'output':'#357985', 'neutral':'#6a7280'}

    def read_json(path):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f'Required local evidence is missing: {path}. Restore it; this notebook does not download or retrain.')
        return json.loads(path.read_text(encoding='utf-8'))

    def sha(path):
        with Path(path).open('rb') as stream:
            return hashlib.file_digest(stream, 'sha256').hexdigest()


    DATASET = WORKSPACE/'datasets/vocalset_pilot'
    MANIFEST = read_json(DATASET/'manifest.json')
    DATA_DECLARATION = read_json(DATASET/'declaration.json')
    RUN = Path(read_json(WORKSPACE/'results/model_recovery/current.json')['path']).resolve()
    assert RUN.is_relative_to((WORKSPACE/'results/model_recovery').resolve())
    EVIDENCE = read_json(RUN/'evidence.json')
    DECLARATION = read_json(RUN/'declaration.json')
    LOCK = read_json(RUN/'locked_selection.json')
    CARD = read_json(WORKSPACE/'results/model_recovery/demo_model.json')
    CHECKPOINT = Path(EVIDENCE['checkpoint'])
    assert sha(CHECKPOINT) == EVIDENCE['checkpoint_sha256'] == CARD['checkpoint_sha256']
    assert sha(RUN/'declaration.json') == EVIDENCE['declaration_sha256']
    assert sha(DATASET/'manifest.json') == DECLARATION['manifest_sha256']

    return locals()


def load_catalogs(RUN, MANIFEST):
    import pandas as pd
    from bpdr_input.reporting.research_support import read as read_json
    from bpdr_input.reporting.research_support import fixed_metrics,paired_reduction,historical_assets
    from bpdr_input.repair.rule_correction import fine_pitch
    from bpdr_input.modeling.local_context_pilot import context_features
    from bpdr_input.repair.model_runtime import runtime_views
    SPLITS=['train','calibration','pilot_evaluation']
    SOURCE_RECORDS={s:read_json(RUN/(s+'_sources.json')) for s in SPLITS}
    PAIRS={s:read_json(RUN/(s+'_pairs.json')) for s in SPLITS}
    SOURCES=pd.DataFrame([dict(singer=r['singer_id'],split=r['split'],sha256=r['sha256']) for r in MANIFEST['records']])
    groups={s:set(SOURCES.loc[SOURCES.split==s,'singer']) for s in SPLITS}
    FEATURE_NAMES=['phrase_relative_pitch','chromatic_residual','residual_sine','residual_cosine','pitch_confidence','voiced','pitch_delta','relative_energy','onset']
    for width in (11,31,61,121):
        FEATURE_NAMES += [f'{prefix}_{width}' for prefix in ['pitch_minus_median','residual_minus_median','residual_mean','residual_spread','voiced_coverage']]
    FEATURE_NAMES += ['endpoint_contour_residual','absolute_pitch_delta']
    assert len(FEATURE_NAMES)==31
    HISTORY_PATHS=historical_assets()
    return locals()


def show(value):
    """Display small, high-contrast tables in light or dark notebook themes."""
    import pandas as pd
    from IPython.display import display
    if not isinstance(value, (pd.DataFrame, pd.Series)):
        return display(value)
    if isinstance(value, pd.Series):
        value = value.to_frame()
    styles = [
        {'selector': 'th', 'props': [('background-color', '#e6efee'), ('color', '#172c31'), ('text-align', 'left'), ('padding', '7px 10px')]},
        {'selector': 'td', 'props': [('background-color', '#ffffff'), ('color', '#172c31'), ('padding', '7px 10px'), ('border-bottom', '1px solid #d6dfdd')]},
        {'selector': 'table', 'props': [('border-collapse', 'collapse'), ('font-size', '14px')]},
    ]
    styled = value.style.format(precision=4, na_rep='—').set_table_styles(styles)
    markup = '<div style="overflow-x:auto;background:#fff;color:#172c31;padding:8px;border-radius:4px">' + styled.to_html() + '</div>'
    display({'text/html': markup, 'text/plain': value.to_string(max_rows=40, max_cols=12)}, raw=True)


def replay_comparison(COMPARISON_RUN, comparison_pairs, model_names, COMPARISON):
    import numpy as np
    from compare_recovery_models import score_case as rescore_frozen_case, aggregate as aggregate_comparison
    comparison_rows = []
    comparison_by_singer = []
    with np.load(COMPARISON_RUN / 'predictions.npz') as frozen:
        for name, label in model_names.items():
            case_rows = []
            for index, pair in enumerate(comparison_pairs):
                if 'excluded' in pair:
                    case_rows.append(dict(singer=pair['source']['singer_id'], cents=pair['cents'], excluded=pair['excluded']))
                    continue
                predictions = {v: frozen[f'{name}__{index}__{v}'] for v in ('A', 'C')}
                assert all((p.shape == (3, len(pair['target'])) and np.isfinite(p).all() for p in predictions.values()))
                row, curves = rescore_frozen_case(pair, predictions)
                for v in ('A', 'C'):
                    np.testing.assert_allclose(curves[v], frozen[f'{name}__{index}__permitted_{v}'])
                case_rows.append(row)
            summary, by_singer = aggregate_comparison(case_rows)
            for metric, value in summary.items():
                np.testing.assert_allclose(value, COMPARISON['results'][name]['summary'][metric], atol=1e-08)
            assert (summary['attempted'], summary['audited'], summary['singers'], summary['original_recordings']) == (16, 10, 2, 3)
            comparison_rows.append(dict(model=label, raw_MAE_cents=summary['raw_mae_cents'], accuracy_within_5c_percent=summary['raw_within_5c_percent'], accuracy_within_10c_percent=summary['raw_within_10c_percent'], original_raw_proposal_cents=summary['original_raw_abs_mean_cents'], permitted_MAE_cents=summary['permitted_mae_cents'], permitted_accuracy_within_10c_percent=summary['permitted_within_10c_percent'], original_edited_frames_percent=summary['original_edited_frames_percent'], permitted_coverage_all_16_attempts_percent=summary['coverage_all_attempts_percent']))
            for singer, result in by_singer.items():
                comparison_by_singer.append(dict(model=label, singer=singer, **result))
    return comparison_rows, comparison_by_singer
