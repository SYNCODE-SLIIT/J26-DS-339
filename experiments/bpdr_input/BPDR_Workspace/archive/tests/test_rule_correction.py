import unittest
import numpy as np
from bpdr_input import rule_correction as rc

SR = 22050


def phrase(note_cents, seconds=1.2, vibrato=0., offset=0.):
    """Synthetic hum: held notes (cents re A440) on the 20-ms grid plus matching audio."""
    per = int(seconds/.02)
    cents = np.concatenate([np.full(per, c, float) for c in note_cents])+offset
    t = np.arange(len(cents))*.02
    cents = cents+vibrato*np.sin(2*np.pi*5.5*t)
    return cents


def synth(cents):
    n = (len(cents)-1)*441
    track = np.interp(np.arange(n)/SR, np.arange(len(cents))*.02, 440*2**(cents/1200))
    ph = 2*np.pi*np.cumsum(track)/SR
    return (.25*np.sin(ph)+.12*np.sin(2*ph)+.06*np.sin(3*ph)).astype(np.float32)


class DecisionTests(unittest.TestCase):
    def test_detuned_held_note_is_corrected_toward_the_singers_grid(self):
        cents = phrase([-900, -700, -500+20, -400, -200])
        corr, unc, offset, notes = rc.decide(cents, np.ones(len(cents), bool))
        self.assertAlmostEqual(offset, 4., delta=1.5)  # circular mean includes the detuned note
        acts = [n['action'] for n in notes]
        self.assertEqual(acts.count('CORRECT'), 1)
        fixed = next(n for n in notes if n['action'] == 'CORRECT')
        self.assertAlmostEqual(fixed['correction_cents'], -16., delta=2.)
        self.assertTrue((corr[(corr != 0)] < 0).all())

    def test_whole_phrase_transposition_produces_no_edit(self):
        for shift in (0., 37., -23.):
            corr, _, _, notes = rc.decide(phrase([-900, -700, -500, -400], offset=shift), np.ones(4*60, bool))
            self.assertFalse(corr.any(), shift)
            self.assertTrue(all(n['reason'] == 'WITHIN_TOLERANCE' for n in notes))

    def test_vibrato_is_withheld_not_flattened(self):
        cents = phrase([-900, -700+20, -500], vibrato=40.)
        corr, _, _, notes = rc.decide(cents, np.ones(len(cents), bool))
        self.assertFalse(corr.any())
        self.assertTrue(any(n['reason'] == 'UNSTABLE_NOTE_VIBRATO_OR_SLIDE' for n in notes))

    def test_large_and_short_deviations_are_withheld_not_clipped(self):
        cents = np.concatenate([phrase([-900, -700]), phrase([-500+45]), phrase([-400], seconds=.5)])
        corr, _, _, notes = rc.decide(cents, np.ones(len(cents), bool))
        self.assertFalse(corr.any())
        self.assertIn('NOTE_TOO_SHORT', [n['reason'] for n in notes])

    def test_unreliable_frames_split_notes(self):
        cents = phrase([-900]); mask = np.ones(len(cents), bool); mask[30] = False
        self.assertEqual(rc.segment_notes(cents, mask), [(0, 30), (31, 60)])


class FullChainTests(unittest.TestCase):
    def test_render_audit_and_rollback_on_a_synthetic_hum(self):
        cents = phrase([-900, -700, -500+20, -400, -200], seconds=1.4)
        audio = synth(cents); times = np.arange(len(audio)//441+1)*.02

        def analyze(a):
            hz = rc.fine_hz(a, SR, times)
            return dict(times_seconds=times, f0_hz=hz, voiced_mask=np.ones(len(times), bool))
        features = analyze(audio)
        out, report = rc.run(audio, SR, features, analyze)
        self.assertEqual(len(out), len(audio))
        # A clean synthetic hum with one 16-cent-sharp held note must pass the whole chain.
        self.assertEqual(report['outcome'], 'VERIFIED_FIXTURE_CANDIDATE', report['transaction'].get('reasons'))
        if True:
            note = next(n for n in report['notes'] if n['action'] == 'CORRECT')
            a, b = int(note['start_seconds']/.02)+10, int(note['end_seconds']/.02)-10
            after = rc.fine_pitch(out, SR, times)
            self.assertLess(abs(np.median(after[a:b]-cents[a:b])-note['correction_cents']), 5)
            outside = np.ones(len(audio), bool); outside[int(note['start_seconds']*SR):int(note['end_seconds']*SR)] = False
            np.testing.assert_array_equal(out[outside], audio[outside])

    def test_failed_audit_rolls_back_to_the_unchanged_input(self):
        cents = phrase([-900, -700, -500+20, -400, -200], seconds=1.4)
        audio = synth(cents); times = np.arange(len(audio)//441+1)*.02
        def analyze(a):
            return dict(times_seconds=times, f0_hz=rc.fine_hz(a, SR, times), voiced_mask=np.ones(len(times), bool))
        features = analyze(audio)
        # A re-measurement that sees no voicing makes the post-render audit fail.
        out, report = rc.run(audio, SR, features, lambda a: dict(analyze(a), voiced_mask=np.zeros(len(times), bool)))
        self.assertEqual(report['outcome'], 'ROLLED_BACK')
        np.testing.assert_array_equal(out, audio)


if __name__ == '__main__':
    unittest.main()
