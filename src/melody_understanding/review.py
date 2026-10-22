"""Independent pitch comparison and explicit review flags."""
import numpy as np
import parselmouth


# Aligning Independent Pitch Frames without Filling Gaps
def map_pitch_frames(source_times, source_hz, target_times, max_distance):
    """Nearest-frame mapping with no interpolation through unvoiced gaps."""
    source_times=np.asarray(source_times)
    if len(source_times)==0:
        raise ValueError('Pitch estimator returned no frames')
    # Compare the neighboring source frames and select the nearest timestamp.
    right=np.searchsorted(source_times,target_times).clip(0,len(source_times)-1)
    left=(right-1).clip(0,len(source_times)-1)
    index=np.where(np.abs(source_times[left]-target_times)<=np.abs(source_times[right]-target_times),left,right)
    # Reject distant frames and preserve zero-frequency frames as unvoiced.
    supported=np.abs(source_times[index]-target_times)<=max_distance+1e-9
    values=np.where(supported & (source_hz[index]>0),source_hz[index],np.nan)
    return values,supported


# Estimating an Independent Pitch Track with Praat
def praat_pitch(audio, target_times):
    """Run Praat autocorrelation using the same pitch search range as pYIN."""
    # Use the baseline hop duration and a C1-to-C7 pitch search range.
    step=256/audio.sample_rate
    pitch=parselmouth.Sound(audio.samples,sampling_frequency=audio.sample_rate).to_pitch_ac(
        time_step=step,pitch_floor=32.70319566257483,pitch_ceiling=2093.004522404789,
        silence_threshold=.03,voicing_threshold=.45,very_accurate=False)
    # Restore the source offset before mapping onto the requested pitch grid.
    native_times=pitch.xs()+audio.offset_s
    hz,supported=map_pitch_frames(native_times,pitch.selected_array['frequency'],target_times,step/2)
    return {'times_s':np.asarray(target_times),'f0_hz':hz,'voiced':np.isfinite(hz)},supported


# Comparing Estimators and Flagging Reference Discrepancies
def compare_pitches(reference, first, second, supported):
    """Describe estimator agreement and signed deviations from held MIDI notes."""
    # Place the held MIDI pitches on the first estimator’s frame grid.
    times=first['times_s'];target=np.full(len(times),np.nan)
    for start,end,pitch in reference:target[(times>=start)&(times<end)]=pitch
    # Compare only supported frames with a reference and voiced estimates from both methods.
    both=supported & first['voiced'] & second['voiced'] & np.isfinite(first['f0_hz']) & np.isfinite(target)
    first_midi=69+12*np.log2(first['f0_hz'][both]/440)
    second_midi=69+12*np.log2(second['f0_hz'][both]/440)
    if not both.any():
        return {'common_voiced_frames':0,'review_flag':'insufficient_common_voicing'}
    # Measure agreement within half a semitone and signed median reference errors.
    agreement=float(np.mean(np.abs(first_midi-second_midi)<.5))
    error_first=float(np.median(first_midi-target[both]))
    error_second=float(np.median(second_midi-target[both]))
    # Flag shared octave-sized errors for review; agreement alone does not verify labels.
    octave=round(error_first/12)
    shared_octave=(agreement>=.8 and octave!=0 and abs(error_first-12*octave)<.75 and abs(error_second-12*octave)<.75)
    flag='shared_octave_discrepancy' if shared_octave else ('estimator_disagreement' if agreement<.8 else 'inspect_timing_and_pitch')
    return {'common_voiced_frames':int(both.sum()),'estimator_agreement_50c':agreement,
        'pyin_median_error_semitones':error_first,'praat_median_error_semitones':error_second,
        'review_flag':flag}
