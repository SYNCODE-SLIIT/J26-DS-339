import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
import numpy as np
import mido
from bpdr_input.humming_experiment import midi_notes,alignment


class ReferenceTests(unittest.TestCase):
    def test_tempo_changes_and_velocity_zero_are_resolved_in_seconds(self):
        with TemporaryDirectory() as d:
            path=Path(d)/'reference.mid';m=mido.MidiFile(ticks_per_beat=480);track=mido.MidiTrack();m.tracks.append(track)
            track.extend([mido.MetaMessage('set_tempo',tempo=500000,time=0),
                mido.Message('note_on',note=69,velocity=100,time=0),
                mido.Message('note_on',note=69,velocity=0,time=480),
                mido.MetaMessage('set_tempo',tempo=1000000,time=0),
                mido.Message('note_on',note=72,velocity=100,time=0),
                mido.Message('note_off',note=72,velocity=0,time=480)])
            m.save(path)
            self.assertEqual(midi_notes(path),[(0.,.5,69),(.5,1.5,72)])

    def test_alignment_reports_octave_mismatch_without_rewriting_reference(self):
        notes=[(0.,1.,69)];times=np.arange(0,1,.02)
        result=alignment(notes,times,np.full(len(times),-1200.),np.ones(len(times),bool))
        self.assertEqual(result['nominal']['within_50_cents'],0.)
        self.assertEqual(result['best_diagnostic_only']['octave_semitones'],-12)
        self.assertEqual(notes,[(0.,1.,69)])


if __name__=='__main__':unittest.main()
