import unittest
import numpy as np
from diagnose_render_disagreement import direction_details

class RenderDisagreementTests(unittest.TestCase):
    def test_tiny_opposing_shift_is_retained(self):
        times=np.arange(101)*.02;f=np.full(101,220.)
        before={'times_seconds':times,'f0_hz':f,'voiced_mask':np.ones(101,bool)}
        after={'times_seconds':times,'f0_hz':f*2**(20/1200),'voiced_mask':np.ones(101,bool)}
        after['f0_hz'][50]=220.*2**(-.0001/1200)
        checked,opposing,fine,coarse=direction_details(before,after,f,f*2**(20/1200),np.ones(101,bool),[(0,2)],np.full(101,20.))
        self.assertEqual(list(np.flatnonzero(opposing)),[50]);self.assertTrue(checked[50]);self.assertGreater(fine[50],0);self.assertLess(coarse[50],0)

    def test_unreliable_and_guard_frames_are_context_only(self):
        times=np.arange(101)*.02;f=np.full(101,220.);voiced=np.ones(101,bool)
        before={'times_seconds':times,'f0_hz':f,'voiced_mask':voiced}
        after={'times_seconds':times,'f0_hz':f*.99,'voiced_mask':voiced}
        reliable=np.ones(101,bool);reliable[50]=False
        checked,opposing,_,_=direction_details(before,after,f,f*1.01,reliable,[(0,2)],np.full(101,20.))
        self.assertFalse(checked[0]);self.assertFalse(checked[50]);self.assertTrue(checked[30]);self.assertTrue(opposing[30]);self.assertFalse(opposing[50])

if __name__=='__main__':unittest.main()
