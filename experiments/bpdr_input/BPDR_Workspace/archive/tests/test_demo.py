from pathlib import Path
import tempfile
import unittest
import numpy as np
import soundfile as sf
from bpdr_input import demo


class DemoStatusTests(unittest.TestCase):
    def run_case(self, name, write):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d)/name; write(src)
            case = demo.run_one(src, Path(d)/'out')
            self.assertTrue((Path(case['run_directory'])/'demo_report.html').exists())
            return case

    def test_silent_recording_asks_for_another_take_without_audio(self):
        case = self.run_case('silent.wav', lambda p: sf.write(p, np.zeros(22050, np.float32), 22050))
        self.assertEqual(case['hand_off']['status'], 'RERECORD_REQUIRED')
        self.assertIsNone(case['hand_off']['accepted_audio'])
        self.assertNotIn('rule_outcome', case)

    def test_unsupported_file_is_a_technical_failure(self):
        case = self.run_case('take.mp3', lambda p: p.write_bytes(b'not audio'))
        self.assertEqual(case['hand_off']['status'], 'TECHNICAL_FAILURE')
        self.assertEqual(list(case['hand_off']['reason_codes']), ['UNSUPPORTED_FORMAT'])

    def test_constructed_hum_controls(self):
        sharp = demo._hum([-900, -700, -500], sharp_index=1, sharp_cents=20.)
        self.assertEqual(sharp.dtype, np.float32); self.assertLess(np.abs(sharp).max(), 1)


if __name__ == '__main__':
    unittest.main()
