"""Streamlit AppTest verification for Vocal Rescue Studio (source/bpdr_input/application/studio_app.py)."""
from pathlib import Path
import shutil
import sys
import unittest

WS = Path(__file__).resolve().parents[1]
SRC = WS / "source"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from streamlit.testing.v1 import AppTest


APP_PATH = str(WS / "source" / "bpdr_input" / "application" / "studio_app.py")


class StudioUITests(unittest.TestCase):
    def test_prepare_boundary_hints_confirmation_harmony_and_evidence_tab(self):
        at = AppTest.from_file(APP_PATH, default_timeout=90)
        at.run()
        self.assertEqual(len(at.exception), 0, list(at.exception))
        self.assertEqual(len(at.tabs), 2)

        # Research evidence tab: verify evidence summary & frame-control table render cleanly
        evidence_tab = at.tabs[1]
        self.assertGreaterEqual(len(at.table), 1)
        self.assertTrue(
            any("Does preparation help the real extractor?" in s.value for s in evidence_tab.subheader)
        )

        out_folder = None
        try:
            # Select an example recording with reference annotations (index 0: rain 0 dB mix) and Prepare
            at.selectbox[0].select_index(0).run()
            prepare_btn = next(b for b in at.button if b.label == "Prepare recording")
            prepare_btn.click()
            at.run()
            self.assertEqual(len(at.exception), 0, list(at.exception))

            if "out_folder" in at.session_state:
                out_folder = Path(at.session_state["out_folder"])

            # Verify ACCEPT_* banner
            self.assertTrue(
                any("**ACCEPT_" in s.value for s in at.success),
                [s.value for s in at.success],
            )

            # Verify boundary hints summary expander (_boundary_hints_block)
            hint_expanders = [e for e in at.expander if e.label.startswith("Note boundary hints (")]
            self.assertEqual(len(hint_expanders), 1)
            metric_labels = {m.label for m in at.metric}
            self.assertTrue({"Detected boundaries", "Threshold", "Grid step"}.issubset(metric_labels))

            # Hear-back confirmation: Yes, this is my melody
            yes_btn = next(b for b in at.button if b.label == "Yes, this is my melody")
            yes_btn.click()
            at.run()
            self.assertEqual(len(at.exception), 0, list(at.exception))
            self.assertTrue(
                any("Melody verified" in s.value for s in at.success),
                [s.value for s in at.success],
            )

            # Hear-back confirmation: No, some notes are wrong -> dialog -> submit feedback
            no_btn = next(b for b in at.button if b.label == "No, some notes are wrong")
            no_btn.click()
            at.run()
            self.assertEqual(len(at.exception), 0, list(at.exception))
            at.text_input(key="conf_spans").set_value("1.0-2.0")
            at.text_input(key="conf_cmt").set_value("missed transition")
            submit_btn = next(b for b in at.button if b.label == "Submit feedback")
            submit_btn.click()
            at.run()
            self.assertEqual(len(at.exception), 0, list(at.exception))
            self.assertTrue(
                any("Melody flagged" in w.value for w in at.warning),
                [w.value for w in at.warning],
            )

            # Harmonize with the team's models
            harm_btn = next(b for b in at.button if b.label == "Harmonize with the team's models")
            harm_btn.click()
            at.run()
            self.assertEqual(len(at.exception), 0, list(at.exception))
            self.assertEqual(len(at.error), 0, [e.value for e in at.error])
            self.assertTrue(
                any("Chord labels from Pasan's planner refined by Rahul's model." in c.value for c in at.caption),
                [c.value for c in at.caption],
            )
        finally:
            if out_folder is not None and out_folder.exists():
                shutil.rmtree(out_folder, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
