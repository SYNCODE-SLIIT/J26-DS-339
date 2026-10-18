"""Optional, declared-noise-reference engineering preview (proposal p. 26).

No automatic speech/noise classifier, musical approval, or prepared handoff.
The caller must identify an interval containing room noise only, without voice,
speech or breaths. Screening can reject an interval; it cannot certify it.
"""
from bpdr_input import COMPONENT_ROOT
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter

COMPONENT = COMPONENT_ROOT


@dataclass(frozen=True)
class NoisePolicy:
    version: str = 'declared-noise-preview-1'
    min_reference_seconds: float = .5
    max_attenuation_db: float = 6.
    min_flatness: float = .25
    max_periodicity: float = .5
    max_reference_level_range_db: float = 6.
    min_reliable_frames: int = 25
    max_voiced_loss_fraction: float = .02
    max_median_pitch_change_cents: float = 2.
    max_p95_pitch_change_cents: float = 5.
    max_p95_voice_attenuation_db: float = 1.5
    min_reference_attenuation_db: float = 1.


def screen_reference(audio, sr, interval, policy=NoisePolicy()):
    """Require caller declaration plus broadband/stationary reference checks."""
    from scipy.signal import correlate
    if interval is None:
        return dict(eligible=False, reasons=['NO_DECLARED_NOISE_ONLY_INTERVAL'])
    start, end = map(float, interval)
    if not np.isfinite([start, end]).all() or not 0 <= start < end <= len(audio)/sr:
        return dict(eligible=False, reasons=['INVALID_REFERENCE_INTERVAL'])
    if end-start < policy.min_reference_seconds:
        return dict(eligible=False, reasons=['REFERENCE_TOO_SHORT'])
    ref = audio[round(start*sr):round(end*sr)].astype(np.float64)
    blocks = ref[:len(ref)//512*512].reshape(-1, 512)
    rms = np.sqrt(np.mean(blocks**2, axis=1))
    spectrum = np.abs(np.fft.rfft(blocks*np.hanning(512), axis=1))**2
    psd = np.mean(spectrum, axis=0)[1:]
    flatness = float(np.exp(np.mean(np.log(psd+1e-20)))/(np.mean(psd)+1e-20))
    # Local autocorrelation catches humming in the declared reference even when
    # a single whole-window correlation would cancel due to pitch changes.
    periodicity = []
    for offset in range(0, len(ref)-2048+1, 1024):
        part = ref[offset:offset+2048]; part = part-part.mean()
        ac = correlate(part, part, mode='full', method='fft')[2047:]
        periodicity.append(float(np.max(ac[max(1, int(sr/1000)):int(sr/65)])/(ac[0]+1e-20)))
    periodicity = max(periodicity, default=1.)
    level_range = float(20*np.log10((np.percentile(rms,90)+1e-12)/(np.percentile(rms,10)+1e-12)))
    reasons = []
    if float(np.sqrt(np.mean(ref**2))) < 1e-5: reasons.append('REFERENCE_IS_SILENCE')
    if flatness < policy.min_flatness: reasons.append('REFERENCE_NOT_BROADBAND')
    if periodicity > policy.max_periodicity: reasons.append('REFERENCE_MAY_CONTAIN_VOICE_OR_TONAL_NOISE')
    if level_range > policy.max_reference_level_range_db: reasons.append('REFERENCE_NOT_STATIONARY')
    return dict(eligible=not reasons, reasons=reasons, interval_seconds=[start,end],
                flatness=flatness, periodicity=periodicity, level_range_db=level_range,
                limitation='Caller declaration required; these screens do not prove absence of speech, breaths or voice')


def attenuate(audio, sr, interval, policy=NoisePolicy()):
    """Bounded smoothed magnitude mask; original STFT phase and exact length."""
    import librosa
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all():
        raise ValueError('Expected finite mono audio')
    screen = screen_reference(audio, sr, interval, policy)
    if not screen['eligible']:
        return audio.copy(), dict(outcome='WITHHELD_UNCHANGED', reference=screen)
    n_fft, hop = 1024, 256
    z = librosa.stft(audio, n_fft=n_fft, hop_length=hop)
    times = np.arange(z.shape[1])*hop/sr
    a,b = interval
    inside = (times >= a+n_fft/(2*sr)) & (times <= b-n_fft/(2*sr))
    if inside.sum() < 4:
        return audio.copy(), dict(outcome='WITHHELD_UNCHANGED', reference=screen, reasons=['INSUFFICIENT_REFERENCE_FRAMES'])
    power = np.abs(z)**2
    noise = np.mean(power[:,inside], axis=1, keepdims=True)
    gain = np.sqrt(np.maximum(1.-noise/(power+1e-20), 0.))
    floor = 10**(-policy.max_attenuation_db/20)
    gain = np.clip(uniform_filter(gain, size=(3,5), mode='nearest'), floor, 1.)
    candidate = librosa.istft(z*gain, hop_length=hop, length=len(audio))
    return candidate, dict(outcome='UNVERIFIED_NOISE_CANDIDATE', reference=screen,
        policy=asdict(policy), algorithm='Smoothed spectral subtraction magnitude mask; original phase',
        fft_samples=n_fft, hop_samples=hop, smoothing_bins_frames=[3,5],
        minimum_gain=float(gain.min()), maximum_gain=float(gain.max()),
        trimmed=False, time_warped=False, sample_count=len(candidate))


def audit(audio, candidate, sr, before, after, interval, policy=NoisePolicy()):
    """Preservation checks against the input, not unknown intended musical notes."""
    import librosa
    from bpdr_input.repair.rule_correction import fine_pitch, reliable_frames
    times = before['times_seconds']
    f0 = fine_pitch(audio, sr, times); f1 = fine_pitch(candidate, sr, times)
    reliable = reliable_frames(before, f0)
    retained = reliable & reliable_frames(after, f1)
    change = np.abs(f1[retained]-f0[retained])
    loss = float((reliable & ~retained).sum()/max(1, reliable.sum()))
    rms0=librosa.feature.rms(y=audio,frame_length=1024,hop_length=round(sr*.02))[0]
    rms1=librosa.feature.rms(y=candidate,frame_length=1024,hop_length=round(sr*.02))[0]
    attenuation=20*np.log10((rms0+1e-12)/(rms1+1e-12))
    a,b=interval; region=slice(round(a*sr),round(b*sr))
    noise_drop=float(10*np.log10((np.mean(audio[region]**2)+1e-20)/(np.mean(candidate[region]**2)+1e-20)))
    med=float(np.median(change)) if len(change) else None
    p95=float(np.percentile(change,95)) if len(change) else None
    voice_drop=float(np.percentile(attenuation[reliable],95)) if reliable.any() else None
    reasons=[]
    if len(audio)!=len(candidate) or not np.isfinite(candidate).all() or np.max(np.abs(candidate))>1.: reasons.append('AUDIO_INTEGRITY')
    if reliable.sum()<policy.min_reliable_frames or retained.sum()<policy.min_reliable_frames: reasons.append('INSUFFICIENT_RELIABLE_VOICE_FOR_AUDIT')
    if loss>policy.max_voiced_loss_fraction: reasons.append('VOICED_FRAMES_LOST')
    if med is None or med>policy.max_median_pitch_change_cents: reasons.append('MEDIAN_PITCH_CHANGED')
    if p95 is None or p95>policy.max_p95_pitch_change_cents: reasons.append('PITCH_TAIL_CHANGED')
    if voice_drop is None or voice_drop>policy.max_p95_voice_attenuation_db: reasons.append('VOICE_ATTENUATED')
    if noise_drop<policy.min_reference_attenuation_db: reasons.append('INSUFFICIENT_NOISE_ATTENUATION')
    return dict(passes=not reasons, reasons=reasons, reliable_frames=int(reliable.sum()),
        voiced_loss_fraction=loss, median_pitch_change_cents=med, p95_pitch_change_cents=p95,
        p95_voice_attenuation_db=voice_drop, reference_attenuation_db=noise_drop,
        limitation='Numerical preservation only; transient consonants, breaths, perceived quality and intended pitch are not certified')


def run(source, interval, render_preview=False, output_root=None, noise_policy=None):
    """Keep original handoff; re-extract and infer on a separate passed preview."""
    import soundfile as sf
    import torch
    from bpdr_input.audio.pipeline import process, checksum
    from bpdr_input.audio.analysis import extract
    from bpdr_input.repair.model_runtime import run as infer
    torch.set_num_threads(2)
    policy=noise_policy or NoisePolicy()
    result, directory=process(Path(source), output_root or COMPONENT/'runs/noise_preview', COMPONENT/'checkpoints')
    report=dict(version='noise-model-preview-1', source=str(source), handoff=result.to_dict(),
        outcome='NOT_RUN_TECHNICAL_STOP', prepared_handoff_enabled=False,
        noise_policy='Caller-declared noise-only interval, screened; optional preview, original handoff retained')
    if result.accepted_audio:
        try:
            audio,sr=sf.read(result.accepted_audio.path,dtype='float64')
            candidate, receipt=attenuate(audio,sr,interval,policy)
            report['noise']=receipt
            report['outcome']='NOISE_WITHHELD_UNCHANGED'
            if receipt['outcome']=='UNVERIFIED_NOISE_CANDIDATE':
                path=directory/'unverified_noise_candidate.wav'
                sf.write(path,candidate,sr,subtype='PCM_24')
                # Audit the encoded waveform that will actually be delivered.
                candidate,_=sf.read(path,dtype='float64')
                with np.load(directory/'features.npz') as z: before={k:z[k] for k in z.files}
                after,metadata=extract(candidate,sr,COMPONENT/'checkpoints')
                check=audit(audio,candidate,sr,before,after,interval,policy)
                report['noise']['audit']=check
                if check['passes']:
                    final=directory/'experimental_noise_candidate.wav';path.rename(final)
                    np.savez_compressed(directory/'noise_candidate_features.npz',**after)
                    report['noise_candidate']=dict(path=str(final),sha256=checksum(final),features_recalculated=True)
                    model_dir,model_report=infer(final,output_root=directory/'model',render_preview=render_preview)
                    report.update(outcome='NOISE_PREVIEW_PASSED',model_outcome=model_report['outcome'],
                        model_report_path=str(model_dir/'model_report.json'), model_input=str(final))
                    if model_report.get('candidate'):report['candidate']=model_report['candidate']
                    report['model_preview_handoff_is_internal_only']=True
                else:
                    report['noise']['outcome']='ROLLED_BACK_UNCHANGED'
        except Exception as exc:
            report.update(outcome='NOISE_PREVIEW_FAILED_UNCHANGED',error_type=type(exc).__name__,reason=str(exc))
    # The component handoff always points to its preserved original working copy.
    (directory/'model_report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    return directory,report


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('--noise-start',type=float)
    parser.add_argument('--noise-end',type=float)
    parser.add_argument('--render-preview',action='store_true')
    args=parser.parse_args()
    if (args.noise_start is None)!=(args.noise_end is None):
        parser.error('Declare both noise interval endpoints, or neither')
    interval=None if args.noise_start is None else (args.noise_start,args.noise_end)
    directory,report=run(args.source,interval,args.render_preview)
    print(json.dumps(dict(directory=str(directory),outcome=report['outcome'],handoff=report['handoff']['status']),indent=2))
