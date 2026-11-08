"""Freeze recovered DeepFilterNet runtime and replay the fixed three-case subset.

Run in the historical isolated dfenv, never the project environment. No fit,
candidate selection, alignment shift or replacement of historical outputs.
"""

import argparse
import configparser
import hashlib
from importlib.metadata import version
from importlib import import_module
import inspect
import json
import os
from pathlib import Path
import shutil
import sys


WS = Path(__file__).resolve().parents[1]
BASE = WS / 'results/transformer_usability'
ROOT = BASE / '2026-11-08_neural_runtime_replay'
CACHE = Path(r'C:\Users\lnipu\AppData\Local\DeepFilterNet\DeepFilterNet\Cache\DeepFilterNet3')
SCRATCH = Path(r'C:\Users\lnipu\AppData\Local\Temp\claude\65a1bab97930dc17\e0dbd492-cfe4-48d6-ab42-178c470a8086\scratchpad')
FILES = ('1_clean_0.wav', '1_10_10.wav', '1_10_0.wav')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def declare():
    import df
    enhancer = import_module('df.enhance')
    import libdf
    import torch
    import torchaudio
    versions = {name: version(name) for name in
                ('deepfilternet', 'DeepFilterLib', 'torch', 'torchaudio', 'numpy', 'soundfile')}
    assert versions['deepfilternet'] == versions['DeepFilterLib'] == '0.5.6'
    assert versions['torch'].split('+')[0] == versions['torchaudio'].split('+')[0] == '2.2.2'
    config = configparser.ConfigParser()
    config.read(CACHE / 'config.ini')
    sr, fft, hop = [config.getint('df', field) for field in ('sr', 'fft_size', 'hop_size')]
    assert (sr, fft, hop) == (48000, 960, 480)
    signature = inspect.signature(enhancer.enhance)
    assert signature.parameters['pad'].default is True
    assert signature.parameters['atten_lim_db'].default is None
    source = Path(enhancer.__file__).read_text(encoding='utf-8')
    assert 'd = n_fft - hop' in source and 'audio[:, d : orig_len + d]' in source
    wrapper = SCRATCH / 'df_enhance.py'
    text = wrapper.read_text(encoding='utf-8')
    assert 'y=enhance(model,state,t)' in text and 'model,state,_=init_df()' in text
    assert 'torchaudio.functional.resample' in text
    source_paths = [Path(__file__), wrapper, Path(libdf.__file__), Path(inspect.getfile(torchaudio.functional.resample))]
    source_paths.extend(sorted(Path(df.__file__).parent.rglob('*.py')))
    inputs = {str(BASE / '2026-11-05_real_noise_neural' / folder / name):
              sha(BASE / '2026-11-05_real_noise_neural' / folder / name)
              for name in FILES for folder in ('mix', 'enh')}
    record = dict(source_hashes={str(p): sha(p) for p in source_paths},
        cached_model_hashes={str(p): sha(p) for p in
            (CACHE / 'config.ini', CACHE / 'checkpoints/model_120.ckpt.best')},
        recovered_log_sha256=sha(CACHE / 'enhance.log'), input_hashes=inputs,
        versions=versions, interpreter=str(Path(sys.executable).resolve()),
        threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
        thread_environment={k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS')},
        configuration=dict(sample_rate_hz=sr, fft_size=fft, hop_size=hop,
            pad_default=True, stft_compensation_samples=fft-hop,
            stft_compensation_ms=1000*(fft-hop)/sr, post_filter=False,
            attenuation_limit_db=None, df_lookahead=config.getint('df', 'df_lookahead'),
            convolution_lookahead=config.getint('deepfilternet', 'conv_lookahead')),
        cases=list(FILES), hypothesis='Recovered cached weights/configuration and default compensated inference reproduce the historical fixed subset.',
        budget='Three enhancement calls in create, three in fresh-process verify; no fit, parameter sweep, rescore or reserved data.',
        criteria='Report exact historical FLOAT waveform equality separately from exact new-run replay. Retain all differences; no model promotion.',
        limitation='Artifacts were recovered after the historical run, not prospectively pinned then. The log and wrapper are provenance evidence, not a complete original execution receipt. STFT compensation is not a universal perceptual-delay guarantee.',
        production_eligible=False)
    ROOT.mkdir(exist_ok=True)
    path = ROOT / 'declaration.json'
    if path.exists():
        assert read(path) == record
    else:
        write(path, record)
        snapshots = ROOT / 'source_snapshots'
        snapshots.mkdir(exist_ok=True)
        for index, item in enumerate(source_paths):
            if item.suffix == '.py':
                (snapshots / f'{index:03d}_{item.name}').write_bytes(item.read_bytes())
    # Isolated copy avoids changes to the user's enhancer cache/log/config.
    model_dir = ROOT / 'model'
    for relative in ('config.ini', 'checkpoints/model_120.ckpt.best'):
        target = model_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(CACHE / relative, target)
        assert sha(target) == sha(CACHE / relative)
    return record, enhancer, model_dir


def main(mode):
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio
    declaration, enhancer, model_dir = declare()
    if mode == 'declare':
        print('NEURAL_RUNTIME_DECLARED', declaration['configuration'], flush=True)
        return
    result_path = ROOT / 'results.json'
    if mode == 'create':
        assert not result_path.exists(), 'Results exist; use --mode verify.'
    # Same defaults as recovered wrapper; only model location/log destination differ.
    model, state, _ = enhancer.init_df(model_base_dir=str(model_dir), log_file=None)
    assert state.sr() == 48000 and state.fft_size() == 960 and state.hop_size() == 480
    rows = []
    target_dir = ROOT / ('first_pass' if mode == 'create' else 'independent_replay')
    target_dir.mkdir(exist_ok=True)
    for name in FILES:
        mixed_path = BASE / '2026-11-05_real_noise_neural/mix' / name
        historical_path = BASE / '2026-11-05_real_noise_neural/enh' / name
        mixed, rate = sf.read(mixed_path, dtype='float32')
        historical, historical_rate = sf.read(historical_path, dtype='float32')
        tensor = torch.from_numpy(mixed)[None]
        resampled = torchaudio.functional.resample(tensor, rate, state.sr())
        enhanced = enhancer.enhance(model, state, resampled)
        assert enhanced.shape == resampled.shape
        delivered = torchaudio.functional.resample(enhanced, state.sr(), rate)[0].numpy()
        assert rate == historical_rate == 22050 and delivered.shape == mixed.shape == historical.shape
        assert np.isfinite(delivered).all()
        output = target_dir / name
        assert not output.exists(), 'Do not overwrite a replay; inspect the prior attempt.'
        sf.write(output, delivered, rate, subtype='FLOAT')
        decoded, decoded_rate = sf.read(output, dtype='float32')
        assert decoded_rate == rate and np.array_equal(decoded, delivered)
        if mode == 'verify':
            first = ROOT / 'first_pass' / name
            first_wave, _ = sf.read(first, dtype='float32')
            np.testing.assert_array_equal(decoded, first_wave)
            assert sha(output) == sha(first)
        rows.append(dict(file=name, samples=len(delivered), sample_rate_hz=rate,
            input_sha256=sha(mixed_path), historical_sha256=sha(historical_path),
            output_sha256=sha(output), historical_samples_exact=bool(np.array_equal(decoded, historical)),
            historical_file_exact=sha(output) == sha(historical_path),
            changed_samples_vs_history=int(np.count_nonzero(decoded != historical)),
            maximum_absolute_difference_vs_history=float(np.max(abs(decoded-historical)))))
        print('NEURAL_RUNTIME_CASE', mode, name, rows[-1]['historical_samples_exact'], flush=True)
    result = dict(declaration_sha256=sha(ROOT / 'declaration.json'), records=rows,
        historical_exact_cases=sum(r['historical_samples_exact'] for r in rows),
        stft_compensation_configuration_verified=True, production_eligible=False)
    if mode == 'create':
        write(result_path, result)
    else:
        assert read(result_path) == result
        write(ROOT / 'integrity_receipt.json', dict(results_sha256=sha(result_path),
            cases_replayed=3, exact_new_run_waveform_replay=True,
            historical_exact_cases=result['historical_exact_cases'], production_eligible=False))
    # Source, model and historical input bindings must remain intact.
    for field in ('source_hashes', 'cached_model_hashes', 'input_hashes'):
        assert all(sha(path) == digest for path, digest in declaration[field].items())
    print('NEURAL_RUNTIME_COMPLETE', mode, result['historical_exact_cases'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('declare', 'create', 'verify'), required=True)
    main(parser.parse_args().mode)
