"""Phase 1 integrity, unchanged-audio and acoustic-analysis entry point."""
import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import shutil
import uuid
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd
from .contracts import AcceptedAudio, ProcessingResult, Status, require_accepted_audio
from .analysis import extract


@dataclass(frozen=True)
class Policy:
    version: str = 'phase1-technical-0.1'
    sample_rate_hz: int = 22050
    max_seconds: float = 20.
    min_seconds: float = .25
    silence_rms: float = 1e-5
    max_clipped_fraction: float = .01


def checksum(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def validate_package(result):
    """Downstream must call this before consuming a locally produced package."""
    audio = require_accepted_audio(result)
    path = Path(audio.path)
    if checksum(path) != audio.checksum:
        raise ValueError('Accepted audio checksum mismatch')
    x, sr = sf.read(path, dtype='float32', always_2d=True)
    if sr != audio.sample_rate_hz or x.shape != (audio.sample_count, audio.channels):
        raise ValueError('Accepted audio metadata mismatch')
    if not np.isfinite(x).all() or float(abs(x).max()) > 1:
        raise ValueError('Accepted audio has invalid sample values')
    return audio


def _plot(directory, audio, sr, f):
    # A private canvas saves files without changing an interactive notebook's backend.
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=(12, 8))
    FigureCanvasAgg(fig)
    axes = fig.subplots(3, 1, sharex=True)
    axes[0].plot(np.arange(len(audio))/sr, audio, linewidth=.5)
    axes[0].set_ylabel('Amplitude')
    axes[1].plot(f['times_seconds'], f['f0_hz'], linewidth=.8)
    axes[1].set_ylabel('Measured F0 (Hz)')
    axes[2].plot(f['times_seconds'], f['provisional_boundary_score'])
    axes[2].set_ylabel('Provisional boundary')
    axes[2].set_xlabel('Time from original start (seconds)')
    fig.suptitle('Phase 1: unchanged pitch; measured acoustics, no intended-note inference')
    fig.tight_layout()
    fig.savefig(directory/'analysis.png', dpi=130)
    fig.clear()


def process(source, output_root, cache_dir, policy=Policy(), pitch_provider=None):
    source = Path(source).resolve()
    directory = Path(output_root).resolve()/uuid.uuid4().hex
    directory.mkdir(parents=True, exist_ok=False)
    diagnostics = dict(schema_version='0.2', policy=asdict(policy), operations=[], edit_intervals_seconds=[],
                       limitations=['Technical acceptance only; musical usability is unvalidated',
                                    'No denoising, trimming, pitch correction or BPDR inference'])
    accepted_path = directory/'accepted.wav'
    source_id = source.name

    def finish(status, reasons=(), message='', accepted=None):
        result = ProcessingResult(directory.name, source_id, status, accepted,
                                  tuple(reasons), message, policy.version)
        diagnostics['result'] = result.to_dict()
        (directory/'diagnostics.json').write_text(json.dumps(diagnostics, indent=2), encoding='utf-8')
        if accepted is None and accepted_path.exists():
            accepted_path.unlink()
        return result, directory

    try:
        if source.suffix.lower() not in {'.wav', '.flac'}:
            return finish(Status.TECHNICAL_FAILURE, ['UNSUPPORTED_FORMAT'], 'Upload a WAV or FLAC recording.')
        # Preserve exactly the bytes subsequently decoded; source is never modified.
        preserved = directory/('original'+source.suffix.lower())
        shutil.copyfile(source, preserved)
        diagnostics['source'] = dict(path=str(source), preserved_path=str(preserved),
                                     sha256=checksum(preserved))
        info = sf.info(preserved)
        diagnostics['input'] = dict(sample_rate_hz=info.samplerate, sample_count=info.frames,
                                    channels=info.channels, duration_seconds=info.duration)
        if info.frames == 0:
            return finish(Status.RERECORD_REQUIRED, ['EMPTY_AUDIO'], 'Record a short solo singing or humming phrase.')
        if info.duration > policy.max_seconds or info.duration < policy.min_seconds:
            return finish(Status.RERECORD_REQUIRED, ['DURATION_OUT_OF_SCOPE'], 'Provide a complete phrase between 0.25 and 20 seconds; no audio was cropped.')
        if info.channels not in (1, 2):
            return finish(Status.RERECORD_REQUIRED, ['CHANNELS_OUT_OF_SCOPE'], 'Provide a mono or stereo solo recording.')
        x, sr = sf.read(preserved, dtype='float32', always_2d=True)
        if len(x) != info.frames:
            raise ValueError('Decoded sample count does not match the file header')
        if not np.isfinite(x).all():
            return finish(Status.TECHNICAL_FAILURE, ['NONFINITE_SAMPLES'], 'The audio contains invalid sample values. Export it again.')
        rms = np.sqrt(np.mean(x.astype(np.float64)**2, axis=0))
        clipped = np.mean(abs(x) >= .999, axis=0)
        diagnostics['observations'] = dict(channel_rms=rms.tolist(), channel_clipped_fraction=clipped.tolist())
        if x.shape[1] == 1:
            mono = x[:, 0]
        else:
            # Choose the strongest channel instead of unsafe phase cancellation.
            selected = int(np.argmax(rms))
            mono = x[:, selected]
            diagnostics['operations'].append(dict(operation='select_channel', channel_index=selected,
                                                  reason='highest RMS; avoid stereo cancellation'))
        selected_clipped = float(np.mean(abs(mono) >= .999))
        if float(np.sqrt(np.mean(mono.astype(np.float64)**2))) < policy.silence_rms:
            return finish(Status.RERECORD_REQUIRED, ['SILENT_RECORDING'], 'No audible signal was detected. Record again closer to the microphone.')
        if selected_clipped > policy.max_clipped_fraction or abs(mono).max() > 1:
            return finish(Status.RERECORD_REQUIRED, ['EXCESSIVE_CLIPPING'], 'The selected channel is overloaded. Record again with lower input volume.')
        target = policy.sample_rate_hz
        divisor = gcd(sr, target)
        working = resample_poly(mono, target//divisor, sr//divisor) if sr != target else mono.copy()
        expected = round(len(mono)*target/sr)
        working = working[:expected]
        if len(working) < expected:
            working = np.pad(working, (0, expected-len(working)))
        diagnostics['operations'].append(dict(operation='working_copy', input_rate_hz=sr,
                                              output_rate_hz=target, sample_count=expected,
                                              timeline_origin_seconds=0, trimmed=False,
                                              duration_error_seconds=expected/target-len(mono)/sr))
        peak = float(abs(working).max())
        if peak > 1:
            gain = .999/peak
            working *= gain
            diagnostics['operations'].append(dict(operation='peak_protection_gain', gain=gain,
                                                  reason='resampling overshoot; no pitch/timing change'))
        options = {} if pitch_provider is None else {'pitch_provider': pitch_provider}
        features, analysis_metadata = extract(working, target, cache_dir, **options)
        diagnostics['analysis'] = analysis_metadata
        diagnostics['analysis']['voiced_fraction'] = float(features['voiced_mask'].mean())
        np.savez_compressed(directory/'features.npz', **features)
        _plot(directory, working, target, features)
        sf.write(accepted_path, working, target, subtype='PCM_24')
        diagnostics['operations'].append(dict(operation='encode_working_copy', format='WAV PCM_24'))
        accepted = AcceptedAudio(str(accepted_path), checksum(accepted_path), target, expected)
        result = ProcessingResult(directory.name, source_id, Status.ACCEPT_UNCHANGED, accepted,
                                  policy_version=policy.version)
        validate_package(result)
        # Recheck original preservation after all analysis/output operations.
        if checksum(preserved) != diagnostics['source']['sha256']:
            raise ValueError('Preserved original changed')
        (directory/'report.html').write_text('''<!doctype html><meta charset="utf-8"><title>BPDR Phase 1</title>
<style>body{font:18px system-ui;max-width:1100px;margin:40px auto}img{width:100%}</style>
<h1>Phase 1: original retained, pitch unchanged</h1>
<p>Technical acceptance only. Measured F0 is performed pitch, not intended notes.
Boundary scores are provisional. No BPDR repair model is running.</p>
<h2>Original recording</h2><audio controls src="'''+preserved.name+'''"></audio>
<h2>Accepted working copy</h2><audio controls src="accepted.wav"></audio>
<img src="analysis.png" alt="Aligned amplitude, performed F0 and provisional boundary score">
<p><a href="diagnostics.json">Diagnostics and operation trace</a></p>''', encoding='utf-8')
        return finish(Status.ACCEPT_UNCHANGED, message='Recording passed the provisional technical checks. Pitch and timing were preserved.', accepted=accepted)
    except Exception as exc:
        diagnostics['technical_error'] = dict(type=type(exc).__name__, detail=str(exc))
        return finish(Status.TECHNICAL_FAILURE, ['PROCESSING_FAILED'], 'Processing failed. See diagnostics before retrying.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    component = Path(__file__).resolve().parents[2]
    parser.add_argument('--output-root', type=Path, default=component/'runs')
    parser.add_argument('--cache-dir', type=Path, default=component/'checkpoints')
    args = parser.parse_args()
    result, directory = process(args.source, args.output_root, args.cache_dir)
    print(json.dumps(dict(result=result.to_dict(), run_directory=str(directory)), indent=2))
    return 0 if result.accepted_audio else 1


if __name__ == '__main__':
    raise SystemExit(main())
