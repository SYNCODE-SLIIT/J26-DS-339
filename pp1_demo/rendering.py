"""Instrument previews, not generated vocal voices or recovered stems."""
import numpy as np
import pretty_midi
import mir_eval.chord

SR=22050

def synth(events,duration):
    audio=np.zeros(round(duration*SR),np.float32)
    midi=pretty_midi.PrettyMIDI(); instrument=pretty_midi.Instrument(0)
    for start,end,pitches in events:
        a,b=max(0,round(start*SR)),min(len(audio),round(end*SR))
        if b<=a: continue
        time=np.arange(b-a)/SR
        envelope=np.minimum(1,time/.015)*np.minimum(1,(b-a-np.arange(b-a))/SR/.04)
        for pitch in pitches:
            freq=440*2**((int(pitch)-69)/12)
            audio[a:b]+=(.12/max(1,len(pitches))*(np.sin(2*np.pi*freq*time)+.2*np.sin(4*np.pi*freq*time))*envelope).astype(np.float32)
            instrument.notes.append(pretty_midi.Note(80,int(pitch),a/SR,b/SR))
    midi.instruments.append(instrument)
    return audio,midi

def melody(notes,duration):
    return synth([(n['onset_s'],n['onset_s']+n['duration_s'],[n['pitch_midi']]) for n in notes],duration)

def chords(rows,duration):
    events=[];unsupported=[]
    for c in rows:
        if c['label']=='N':continue
        try:
            root,semitones,bass=mir_eval.chord.encode(c['label'])
            pitches=[48+(root+int(x))%12 for x in np.flatnonzero(semitones)]
            if bass>=0: pitches=[36+(root+int(bass))%12]+pitches
            if not pitches:raise ValueError('No chord tones')
            events.append((c['start_s'],c['end_s'],pitches))
        except Exception:
            unsupported.append(c['label'])
    audio,midi=synth(events,duration)
    return audio,midi,sorted(set(unsupported))

def mix(original,accompaniment):
    n=min(len(original),len(accompaniment))
    result=.65*original[:n]+.55*accompaniment[:n]
    peak=float(np.max(np.abs(result)))
    return result*min(1,.95/max(peak,1e-8))
