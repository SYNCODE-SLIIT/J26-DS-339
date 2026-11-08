"""Opt-in reviewed-WAV connection to the frozen owning pYIN consumer.

This supplies audio to the existing component; it implements no transcription.
"""

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

from bpdr_input import COMPONENT_ROOT
from bpdr_input.audio.contracts import AcceptedAudio, ProcessingResult, Status
from bpdr_input.audio.pipeline import checksum, validate_package


WORKSPACE = (COMPONENT_ROOT / 'BPDR_Workspace').resolve()
FREEZE = WORKSPACE / 'results/transformer_usability/2026-11-07_primary_extractor_freeze/manifest.json'


def _write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def _freeze():
    manifest = json.loads(FREEZE.read_text(encoding='utf-8'))
    for path, digest in manifest['source_hashes'].items():
        if checksum(path) != digest:
            raise ValueError('Frozen consumer source changed: ' + path)
    for package, expected in manifest['versions'].items():
        if version(package) != expected:
            raise ValueError('Frozen consumer dependency changed: ' + package)
    return checksum(FREEZE)


def run(handoff_directory, timeout_seconds=120, *, allow_test_fixture=False):
    directory = Path(handoff_directory).resolve()
    if not directory.is_relative_to(WORKSPACE):
        raise ValueError('Reviewed handoff must remain inside BPDR_Workspace.')
    manifest_path = directory / 'handoff.json'
    manifest_hash = checksum(manifest_path)
    data = json.loads(manifest_path.read_text(encoding='utf-8'))
    accepted = data.get('accepted_audio')
    if data.get('status') not in ('ACCEPT_UNCHANGED', 'ACCEPT_PREPARED') or not accepted:
        raise ValueError('Only an accepted package can reach the consumer.')
    audio = AcceptedAudio(**{key: accepted[key] for key in
        ['path', 'checksum', 'sample_rate_hz', 'sample_count', 'channels']})
    if not Path(audio.path).resolve().is_relative_to(directory):
        raise ValueError('Selected audio leaves its reviewed package.')
    validate_package(ProcessingResult(data['request_id'], data['source_id'], Status(data['status']), audio))
    diagnostics_path = directory / 'diagnostics.json'
    diagnostics_hash = checksum(diagnostics_path)
    diagnostics = json.loads(diagnostics_path.read_text(encoding='utf-8'))
    fixture = diagnostics.get('confirmation_kind') == 'automated_test_fixture'
    if fixture:
        if not allow_test_fixture or diagnostics.get('user_confirmed') is not False:
            raise ValueError('An automated fixture requires explicit fixture mode.')
    elif diagnostics.get('confirmation_kind') != 'user' or diagnostics.get('user_confirmed') is not True:
        raise ValueError('The selected waveform requires explicit user confirmation.')
    role = 'preview' if data['status'] == 'ACCEPT_PREPARED' else 'original'
    if diagnostics.get('selection') != role or diagnostics.get('usability', {}).get('waveform_sha256') != audio.checksum:
        raise ValueError('Review does not belong to the selected waveform.')
    freeze_hash = _freeze()
    attempt = directory / 'primary_consumer_attempts' / uuid.uuid4().hex
    attempt.mkdir(parents=True)
    target = attempt / 'accepted.wav'
    shutil.copyfile(audio.path, target)
    if checksum(target) != audio.checksum:
        raise ValueError('Consumer copy differs from selected waveform.')
    output = attempt / 'consumer_output.json'
    command = [sys.executable, '-X', 'utf8', '-B', '-m', 'bpdr_input.repair.primary_handoff',
               '--consume', str(target), '--output', str(output), '--expected', audio.checksum,
               '--freeze-sha256', freeze_hash]
    environment = dict(os.environ, PYTHONPATH=str(WORKSPACE / 'source'),
                       OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1')
    started = time.perf_counter()
    receipt = dict(success=False, consumer='frozen_shared_pyin',
        selected_status=data['status'], selected_sha256=audio.checksum,
        handoff_sha256=manifest_hash, review_sha256=diagnostics_hash,
        consumer_freeze_sha256=freeze_hash, consumer_input=str(target),
        confirmation_kind=diagnostics['confirmation_kind'], human_review_claimed=not fixture,
        scope='Exact selected-WAV delivery only; no model or musical-accuracy qualification.',
        attempt_directory=str(attempt))
    try:
        process = subprocess.run(command, cwd=WORKSPACE, env=environment,
            timeout=timeout_seconds, capture_output=True, text=True, encoding='utf-8', errors='replace')
        (attempt / 'stdout.txt').write_text(process.stdout, encoding='utf-8')
        (attempt / 'stderr.txt').write_text(process.stderr, encoding='utf-8')
        receipt['returncode'] = process.returncode
        if process.returncode == 0 and output.is_file():
            result = json.loads(output.read_text(encoding='utf-8'))
            if result['input_sha256'] != audio.checksum or result['consumer_freeze_sha256'] != freeze_hash:
                raise ValueError('Consumer receipt does not match selected audio or frozen consumer.')
            if checksum(target) != audio.checksum or checksum(audio.path) != audio.checksum:
                raise ValueError('Selected audio changed during consumption.')
            if checksum(manifest_path) != manifest_hash or checksum(diagnostics_path) != diagnostics_hash:
                raise ValueError('Selection or review changed during consumption.')
            receipt.update(success=True, consumer_sha256=result['input_sha256'],
                           artifacts=[dict(path=str(output), sha256=checksum(output))])
        else:
            receipt['reason'] = 'CONSUMER_FAILED'
    except subprocess.TimeoutExpired:
        receipt.update(reason='CONSUMER_TIMEOUT', timeout_seconds=timeout_seconds)
    except (OSError, ValueError, KeyError) as exc:
        receipt.update(reason='CONSUMER_VERIFICATION_FAILED', detail=str(exc))
    receipt['elapsed_seconds'] = time.perf_counter() - started
    _write(attempt / 'receipt.json', receipt)
    _write(directory / 'primary_consumer_result.json', receipt)
    return receipt


def consume(path, output, expected, freeze_hash):
    if checksum(path) != expected or _freeze() != freeze_hash:
        raise ValueError('Consumer input or configuration mismatch.')
    sys.path.insert(0, str(WORKSPACE.parents[2] / 'src'))
    from melody_understanding.pipeline import analyze_vocal
    result, _ = analyze_vocal(path, trim=False)
    if checksum(path) != expected or _freeze() != freeze_hash:
        raise ValueError('Consumer input or configuration changed while running.')
    if result['preprocessing']['trim_enabled'] is not False:
        raise ValueError('Consumer unexpectedly trimmed selected audio.')
    _write(output, dict(input_sha256=expected, consumer_freeze_sha256=freeze_hash,
                        result=result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--consume', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected', required=True)
    parser.add_argument('--freeze-sha256', required=True)
    args = parser.parse_args()
    consume(args.consume, args.output, args.expected, args.freeze_sha256)
