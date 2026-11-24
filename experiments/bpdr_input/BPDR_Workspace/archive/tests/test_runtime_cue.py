import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import numpy as np
import soundfile
from bpdr_input import cue_features as cf
from bpdr_input import reference_cue_ceiling as rc
from bpdr_input import runtime_cue_diagnosis as rd


def raw_track(pitch, voiced=None, seed=0):
    rng = np.random.default_rng(seed)
    n = len(pitch)
    raw = rng.normal(size=(n, 71)).astype(np.float32)
    voiced = np.ones(n, bool) if voiced is None else voiced
    raw[:, 64] = np.where(voiced, pitch, 0)
    raw[:, 66] = voiced.astype(np.float32)
    return raw


def melody(offset=0., n=400):
    notes = np.repeat([0, 200, 400, 500, 700, 500, 400, 200], n//8).astype(np.float64)
    return notes-900+offset


class CueFeatureTests(unittest.TestCase):
    def test_shape_finite_and_unvoiced_frames_are_zero(self):
        voiced = np.ones(400, bool); voiced[50:80] = False
        x = cf.cue_features(raw_track(melody(), voiced))
        self.assertEqual(x.shape, (400, cf.WIDTH)); self.assertTrue(np.isfinite(x).all())
        self.assertTrue((x[50:80] == 0).all())

    def test_global_tuning_and_whole_semitones_are_invisible(self):
        base = cf.cue_features(raw_track(melody()))
        for shift in (30., -17., 100., 1200.):
            shifted = cf.cue_features(raw_track(melody()+shift))
            # Steps are absolute pitch differences and unchanged; grid features are offset-relative.
            np.testing.assert_allclose(shifted, base, atol=1e-5)

    def test_local_shift_moves_deviation_with_correct_sign(self):
        p = melody(); q = p.copy(); q[110:140] += 30
        a, b = cf.cue_features(raw_track(p)), cf.cue_features(raw_track(q))
        dev7 = cf.NAMES.index('dev_w7')
        self.assertGreater(b[125, dev7]-a[125, dev7], .4)
        q[110:140] -= 60
        c = cf.cue_features(raw_track(q))
        self.assertLess(c[125, dev7]-a[125, dev7], -.4)

    def test_only_pitch_and_voicing_columns_are_read(self):
        p = melody()
        np.testing.assert_array_equal(cf.cue_features(raw_track(p, seed=1)), cf.cue_features(raw_track(p, seed=2)))


class DiagnosisSummaryTests(unittest.TestCase):
    def records_and_rows(self):
        records = [dict(ds='jamendo'), dict(ds='vocadito')]
        Y = np.array([0., 0., 30., 30., 0., 0., -20., 0.])
        R = np.array([0, 0, 0, 0, 1, 1, 1, 1]); V = np.array([0, 0, 1, 1, 0, 0, 1, 2])
        return records, Y, R, V

    def test_perfect_prediction_has_unit_gain_and_zero_output_has_none(self):
        records, Y, R, V = self.records_and_rows()
        V = V.copy(); V[2] = 2; Y = Y.copy(); Y[2] = 0  # give record 0 an outside region too
        perfect = rd.summarize(records, Y, Y.copy(), R, V)
        zero = rd.summarize(records, Y, np.zeros_like(Y), R, V)
        self.assertAlmostEqual(perfect['all']['correction_gain'], 1.); self.assertAlmostEqual(perfect['all']['objective_gain'], 1.)
        self.assertAlmostEqual(zero['all']['correction_gain'], 0.); self.assertAlmostEqual(zero['all']['objective_gain'], 0.)

    def test_base_scalars_exclude_log_mel(self):
        raw = raw_track(melody())
        self.assertEqual(rd.base_scalars(raw).shape, (400, 6))


class ReferenceCeilingTests(unittest.TestCase):
    def test_explicit_cue_matches_runtime_cue_with_own_offset(self):
        raw = raw_track(melody(23.)); p = raw[:, 64].astype(np.float64); v = (raw[:, 66] > .5) & (p != 0)
        np.testing.assert_allclose(rc.cue(p, v, rc.own_offset(p, v)), cf.cue_features(raw), atol=1e-5)

    def test_own_offset_recovers_known_tuning(self):
        p = melody(17.); v = np.ones(len(p), bool)
        self.assertAlmostEqual(rc.own_offset(p, v), 17., places=6)
        self.assertEqual(rc.own_offset(p, np.zeros(len(p), bool)), 0.)

    def test_gains_and_bootstrap_are_consistent(self):
        rows = [dict(original=(0., 0.), correction=(15., 30.), outside=(0., 0.)) for _ in range(4)]
        self.assertEqual(rc.gains(rows, range(4)), (.5, .5))
        records = [dict(grp=g) for g in 'abcd']
        b = rc.bootstrap(records, dict(ref0=rows, ref1=rows), n=50)
        self.assertEqual(b['gain_intervals']['ref0']['correction'], [.5, .5])
        self.assertEqual(b['difference_from_ref0_intervals']['ref1']['objective'], [0., 0.])

    def test_yin_alignment_on_known_tone(self):
        sr = 22050; t = np.arange(sr*2)/sr
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'tone.wav'; soundfile.write(path, (.3*np.sin(2*np.pi*220*t)).astype(np.float32), sr)
            cents = rc.yin_on_frames(str(path), np.linspace(.2, 1.8, 50))
        self.assertTrue(np.all(np.abs(cents+1200) < 5))

    def test_run_refuses_a_changed_declaration(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d)/'declaration.json').write_text(json.dumps(dict(changed=True)))
            with mock.patch.object(rc, 'OUT', Path(d)):
                with self.assertRaisesRegex(ValueError, 'Declaration changed'):
                    rc.run()


if __name__ == '__main__':
    unittest.main()
