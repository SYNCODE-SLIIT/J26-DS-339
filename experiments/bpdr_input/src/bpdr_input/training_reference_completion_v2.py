"""Exact source-sample provenance for newly prepared original training audio."""
from fractions import Fraction
import soundfile as sf
from .contracts import AcceptedAudio,ProcessingResult,Status
from .pipeline import checksum,validate_package
from .training_reference_preparation import require_train


def expected_accepted_count(phrase, source_info, target_rate=22050):
    require_train(phrase)
    frames=phrase['end_sample']-phrase['start_sample']
    rate=phrase['original_sample_rate_hz']
    if frames<=0 or source_info.frames!=frames or source_info.samplerate!=rate:
        raise ValueError('Original crop sample provenance differs from manifest')
    if abs(phrase['duration_seconds']-frames/rate)>1e-12:
        raise ValueError('Manifest duration differs from original sample provenance')
    # Round once from exact counts; duration_seconds * target can cross a half tie.
    return round(Fraction(frames*target_rate,rate))


def authenticate_original(phrase, diagnostics):
    require_train(phrase)
    source=diagnostics['source']
    if source['sha256']!=phrase['sha256']:
        raise ValueError('Cache belongs to a different crop')
    for path in [phrase['audio_path'],source['path'],source['preserved_path']]:
        if checksum(path)!=phrase['sha256']:raise ValueError('Original source checksum changed')
    raw=diagnostics['result'];a=raw.get('accepted_audio')
    package=None if a is None else AcceptedAudio(**{k:a[k] for k in ['path','checksum','sample_rate_hz','sample_count','channels']})
    result=ProcessingResult(raw['request_id'],raw['source_id'],Status(raw['status']),package,
        tuple(raw['reason_codes']),policy_version=raw['policy_version'])
    accepted=validate_package(result)
    expected=expected_accepted_count(phrase,sf.info(phrase['audio_path']))
    if accepted.sample_rate_hz!=22050 or accepted.sample_count!=expected:
        raise ValueError('Accepted timeline differs from exact original sample provenance')
    audio,sr=sf.read(accepted.path,dtype='float32')
    return accepted,audio,sr
