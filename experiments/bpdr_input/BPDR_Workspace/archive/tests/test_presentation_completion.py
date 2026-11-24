from bpdr_input import COMPONENT_ROOT
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from bpdr_input.presentation_study import score_predictions, summaries


class TransferScoringTests(unittest.TestCase):
    def pair(self):
        return dict(mask=np.ones(6, bool), detuning=np.array([0, 20, 20, 20, 0, 0], np.float32))

    def test_constant_bias_cannot_pass_as_counter_detuning(self):
        r = {v: np.full(6, -20., np.float32) for v in 'ABCD'}
        s = score_predictions(self.pair(), r)
        self.assertEqual(s['correction_mae'], 0.)
        self.assertEqual(s['response_gain'], 0.)
        self.assertEqual(s['unshifted_alteration'], 20.)

    def test_true_compensation_and_outside_preservation(self):
        p = self.pair(); r = {v: np.zeros(6, np.float32) for v in 'ABCD'}
        r['C'] = r['D'] = -p['detuning']
        s = score_predictions(p, r)
        self.assertEqual(s['response_gain'], 1.)
        self.assertEqual(s['correction_mae'], 0.)
        self.assertEqual(s['outside_alteration'], 0.)

    def test_bootstrap_unit_is_group_not_augmented_frame(self):
        base = dict(zero_mae=20., response_gain=.5, boundary_sensitivity=0., unshifted_alteration=0., outside_alteration=0.,
                    dataset='vocadito', corruption='split', cents=20.)
        rows = [dict(base, correction_mae=0., group='one') for _ in range(100)] + [dict(base, correction_mae=20., group='two')]
        s = summaries(rows)
        self.assertEqual(s['group_uncertainty']['vocadito']['groups'], 2)
        self.assertEqual(s['group_uncertainty']['vocadito']['group_equal_mean'][0], 10.)


class AudioOnlyRuntimeTests(unittest.TestCase):
    def test_a4_zero_cents_is_voiced_not_a_missing_value(self):
        from bpdr_input.model_runtime import runtime_views
        raw=np.zeros((100,71),np.float32);raw[:,66]=1.
        views,reliable,_,_=runtime_views(dict(times_seconds=np.arange(100)*.02,
            boundary_features_71=raw,provisional_boundary_score=np.zeros(100)),np.zeros(100))
        self.assertTrue(reliable.all())
        self.assertTrue((views[0][:,66]==1).all())
        self.assertTrue((views[0][:,64]==0).all())

    def test_release_checkpoint_is_verified_before_use(self):
        import json
        from bpdr_input.model_runtime import default_checkpoint
        root=COMPONENT_ROOT/'runs'
        with TemporaryDirectory(dir=root) as temp:
            p=Path(temp);release=p/'release';release.mkdir();(release/'model.pt').write_bytes(b'changed')
            (release/'model_card.json').write_text(json.dumps(dict(checkpoint='model.pt',checkpoint_sha256='0'*64)))
            pointer=p/'runs/presentation_release';pointer.mkdir(parents=True)
            (pointer/'current.json').write_text(json.dumps(dict(path=str(release))))
            with patch('bpdr_input.model_runtime.COMPONENT',p):
                with self.assertRaisesRegex(ValueError,'checksum mismatch'):default_checkpoint()

    def test_challenge_note_is_never_the_longest_across_seeds(self):
        from bpdr_input.generalization_probe import notes_for_voice
        from bpdr_input.controlled_study import voice
        for seed in range(100):
            rng=np.random.default_rng(seed);notes,index=notes_for_voice(voice(rng),rng)
            self.assertLess(notes[index][1]-notes[index][0],max(b-a for a,b,_ in notes))
            self.assertLess(notes[-1][1],6.)

    def test_boundary_views_do_not_change_acoustics_or_unvoiced_pitch(self):
        from bpdr_input.model_runtime import runtime_views
        times = np.arange(100)*.02
        raw = np.zeros((100,71), np.float32); raw[:,64] = -900.; raw[:,66] = 1.; raw[:4,66] = 0.
        boundary = np.zeros(100); boundary[[15,55]] = 1.
        views, reliable, protected, metadata = runtime_views(dict(times_seconds=times,boundary_features_71=raw,provisional_boundary_score=boundary),np.full(100,-899.))
        np.testing.assert_array_equal(views[0][:,:70],views[1][:,:70])
        np.testing.assert_array_equal(views[0][:,:70],views[2][:,:70])
        self.assertFalse(reliable[:4].any());self.assertTrue(protected[15])
        self.assertTrue(np.all(views[0][:4,64] == 0.))

    def test_technical_rejection_never_loads_a_checkpoint(self):
        import soundfile as sf
        from bpdr_input.model_runtime import run
        root=COMPONENT_ROOT/'runs'
        with TemporaryDirectory(dir=root) as temp:
            p=Path(temp);sf.write(p/'silence.wav',np.zeros(22050),22050)
            with patch('bpdr_input.bpdr_showcase.load_model', side_effect=AssertionError('must not load model')):
                folder, report = run(p/'silence.wav',p/'outputs')
            self.assertEqual(report['outcome'],'NOT_RUN_TECHNICAL_STOP')
            self.assertIsNone(report['handoff']['accepted_audio'])
            self.assertTrue((folder/'original.wav').exists())


if __name__ == '__main__':
    unittest.main()
