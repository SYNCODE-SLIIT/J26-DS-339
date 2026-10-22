"""Behavioral tests for the isolated synthetic baseline (stdlib unittest)."""
import contextlib
import io
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from experiments.rahul.src import build_synthetic_dataset as ds


class SyntheticDatasetTests(unittest.TestCase):
    def test_simplification(self):
        for label,expected in [('C:maj','C'),('A:min','Am'),('B:dim','Bdim'),('G:7','G7'),
                               ('F:maj/3','F'),('G:maj/5','G'),('G:min/b3','Gm'),('N','N')]:
            with self.subTest(label=label):
                self.assertEqual(ds.simplify(label)['basic_chord'],expected)
        for label in ('C:mystery','garbage'):
            self.assertIsNone(ds.simplify(label)['basic_chord'])
            self.assertEqual(ds.simplify(label)['simplification_status'],'unsupported')

    def test_extended_quality_families_preserve_root_and_drop_inversion(self):
        for detail, quality, suffix in [('min7','min','m'),('min9','min','m'),
                ('maj7','maj',''),('maj9','maj',''),('dim7','dim','dim'),
                ('hdim7','hdim','m7b5'),('sus2','sus','sus'),('sus4','sus','sus')]:
            with self.subTest(detail=detail):
                result=ds.simplify(f'Bb:{detail}/b3')
                self.assertEqual(result, dict(basic_chord='Bb'+suffix, basic_root='Bb',
                    basic_quality=quality, simplification_status='supported'))
        # Suspensions are not assigned a major/minor diatonic triad function.
        self.assertEqual(ds.context(ds.simplify('C:sus4'),'C','major'),('unknown','unknown'))

    def test_requested_quality_spellings_and_shorthand(self):
        groups = {'maj': ('maj','6','maj7','maj9','add9'),
                  'min': ('min','min7','min9','min11','m','m7','m9','m11'),
                  'dom': ('7','9','13','7b9','7(b9)'),
                  'dim': ('dim','dim7'), 'hdim': ('m7b5','co7','hdim7'),
                  'aug': ('aug','+'), 'sus': ('sus2','sus4')}
        for family, spellings in groups.items():
            for detail in spellings:
                for label in (f'F#:{detail}/b3', f'F#{detail}/b3'):
                    with self.subTest(label=label):
                        result=ds.simplify(label)
                        self.assertEqual(result['basic_quality'],family)
                        self.assertEqual(result['basic_root'],'F#')
        self.assertEqual(ds.simplify('C')['basic_quality'],'maj')
        self.assertEqual(ds.simplify(' N ')['basic_quality'],'N')
        self.assertEqual(ds.context(ds.simplify('D:9'),'C','major'),('unknown','unknown'))

    def test_overlap(self):
        self.assertEqual(ds.overlap(0,1,1,2),0)
        self.assertAlmostEqual(ds.overlap(0,2,1,3),1)
        self.assertEqual(ds.overlap(0,3,1,2),1)

    def test_roman_correspondence(self):
        chord=dict(track_id='382',start_sec=0,end_sec=1,chord_label='C:maj')
        roman=dict(track_id='382',start_sec='0.0',end_sec='1',original_chord_label='C:maj')
        self.assertEqual(ds.join_roman([chord],[roman])[0]['source_roman'],roman)
        for bad in ({**roman,'original_chord_label':'C:7'},{**roman,'end_sec':2}):
            with self.assertRaises(ValueError): ds.join_roman([chord],[bad])
        with self.assertRaises(ValueError): ds.join_roman([chord],[roman,roman])

    def test_crossing_melody_and_silence(self):
        notes=[dict(onset_sec=0,offset_sec=2,midi_pitch=61.1,midi_pitch_rounded=61,source='a'),
               dict(onset_sec=1.5,offset_sec=3,midi_pitch=64,midi_pitch_rounded=64,source='b')]
        result=ds.melody_features(notes,1,2)
        self.assertEqual(result['melody_pitch_classes'],[1,4])
        self.assertEqual(result['melody_overlap_durations'],[1,.5])
        self.assertEqual(result['melody_durations'],[2,1.5])
        self.assertEqual(result['melody_relative_start'],[-1,.5])
        self.assertEqual(ds.melody_features(notes,3,4)['melody'],[])

    def test_alignment_tie_and_threshold(self):
        chords=[dict(start_sec=0,end_sec=.5,chord_label='C:maj'),dict(start_sec=.5,end_sec=1,chord_label='G:7')]
        best,candidates,ratio,ambiguous=ds.align(0,1,chords)
        self.assertEqual(best['source_chord_event_id'],0)
        self.assertEqual(ratio,.5)
        self.assertTrue(ambiguous)
        self.assertEqual(len(candidates),2)
        chords[0]['end_sec']=.526
        chords[1]['start_sec']=.526
        self.assertFalse(ds.align(0,1,chords)[3])
        self.assertEqual(ds.align(2,3,chords),(None,[],0.0,False))

    def test_basic_context_and_input_separation(self):
        self.assertEqual(ds.context(ds.simplify('G:7'),'C','major'),('V','dominant'))
        self.assertEqual(ds.context(ds.simplify('Bb:maj'),'C','major'),('unknown','unknown'))
        self.assertFalse(set(ds.INPUTS)&set(ds.TARGETS))
        for forbidden in ('reference_quality','reference_bass_degree','reference_chord','family','basic_label','source_roman'):
            self.assertNotIn(forbidden,ds.INPUTS)

    def test_export_reload(self):
        rows=[dict(track_id='382',melody=[dict(pitch=60)],empty=[],valid=False,missing=None)]
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'events.csv'
            ds.write_csv(path,rows)
            result=ds.read_csv(path)[0]
            self.assertEqual(json.loads(result['melody']),rows[0]['melody'])
            self.assertEqual(json.loads(result['empty']),[])
            self.assertEqual(result['valid'],'False')

    def test_track382_integration(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(ds.config,'OUTPUT_DIR',Path(temporary)), contextlib.redirect_stdout(io.StringIO()):
            rows,report=ds.build('382', output=Path(temporary)/'synthetic_dataset/track_0000382')
        self.assertTrue(report['all_passed'])
        self.assertEqual(len(rows),428)
        self.assertEqual(sum(r['target_valid'] for r in rows),425)
        self.assertEqual(sum(r['reference_ambiguous'] for r in rows),3)
        self.assertEqual(sum(not r['melody'] for r in rows),57)
        self.assertEqual(sum(r['reference_chord']=='N' for r in rows),10)
        self.assertEqual(rows[-1]['end_sec'],211.04)
        self.assertEqual(rows[0]['start_sec'],.01)
        self.assertEqual(rows[0]['bar_number'],0)
        for r in rows:
            self.assertEqual(r['target_valid'],r['reference_overlap_ratio']>=.5 and not r['reference_ambiguous'])
        # Independent predictions may change basics/context without changing targets.
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'member2.csv'
            ds.write_csv(path,[dict(track_id='382',beat_index=r['beat_index'],basic_chord='Am') for r in rows])
            with patch.object(ds.config,'OUTPUT_DIR',Path(temp)), contextlib.redirect_stdout(io.StringIO()):
                independent,_=ds.build('382',member2=path)
            self.assertTrue(all(r['basic_chord']=='Am' and r['roman_numeral']=='vi' for r in independent))
            self.assertEqual([r['reference_chord'] for r in rows],[r['reference_chord'] for r in independent])


if __name__=='__main__':
    unittest.main()
