"""Check tempo changes and reject timelines that cannot be merged."""
from pathlib import Path
import tempfile
import sys
import unittest
import mido
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from melody_understanding.annotations import read_midi_notes, overlap_count

class AnnotationTests(unittest.TestCase):
    def test_tempo_change_and_zero_velocity_note_off(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'tempo.mid'
            midi = mido.MidiFile(ticks_per_beat=480)
            tempo = mido.MidiTrack([mido.MetaMessage('set_tempo', tempo=500000),
                mido.MetaMessage('set_tempo', tempo=1000000, time=480)])
            track = mido.MidiTrack([mido.Message('note_on', note=69, velocity=90),
                mido.Message('note_on', note=69, velocity=0, time=960)])
            midi.tracks.extend([tempo, track]); midi.save(path)
            notes, anomalies = read_midi_notes(path)
            self.assertEqual(notes, [(0.0, 1.5, 69)])
            self.assertFalse(any(anomalies.values()))

    def test_overlap_and_independent_tracks(self):
        self.assertEqual(overlap_count([(0, 1, 60), (1, 2, 62)]), 0)
        self.assertEqual(overlap_count([(0, 1, 60), (.5, 2, 62)]), 1)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'type2.mid'
            midi = mido.MidiFile(type=2)
            midi.tracks.append(mido.MidiTrack())
            midi.save(path)
            with self.assertRaises(ValueError): read_midi_notes(path)

    def test_explicit_fast_tempo_at_tick_zero(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'fast.mid'
            midi = mido.MidiFile(ticks_per_beat=480)
            midi.tracks.append(mido.MidiTrack([
                mido.MetaMessage('set_tempo', tempo=250000),
                mido.Message('note_on', note=60, velocity=80),
                mido.Message('note_off', note=60, time=960)]))
            midi.save(path)
            notes, _ = read_midi_notes(path)
            self.assertEqual(notes, [(0.0, .5, 60)])
