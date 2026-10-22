"""Notebook startup must use the component kernel and the physical workspace."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from bpdr_input import COMPONENT_ROOT
from bpdr_input.reporting import notebooks


class NotebookRunnerTests(unittest.TestCase):
    def test_kernel_has_explicit_source_path_and_interpreter(self):
        with tempfile.TemporaryDirectory(dir=COMPONENT_ROOT/'runs') as folder:
            root=Path(folder).resolve()
            with patch.dict(os.environ):
                name=notebooks.configure_kernel(root)
                spec=json.loads((root/'.jupyter/kernels'/name/'kernel.json').read_text())
                self.assertEqual(spec['argv'][0],sys.executable)
                self.assertEqual(spec['env']['PYTHONPATH'],str(root/'BPDR_Workspace/source'))
                self.assertEqual(spec['env']['BPDR_COMPONENT_ROOT'],str(root))
                self.assertEqual(os.environ['JUPYTER_RUNTIME_DIR'],str(root/'.jupyter/runtime'))

    def test_launch_opens_browser_and_declared_first_notebook(self):
        with patch.object(notebooks,'configure_kernel'), patch.object(notebooks,'load_index',
                return_value=({'maintained_notebooks':['first.ipynb']},{})), \
                patch.object(sys,'argv',['notebooks','launch']), \
                patch.object(notebooks.subprocess,'call',return_value=0) as call:
            self.assertEqual(notebooks.main(),0)
            command=call.call_args.args[0]
            self.assertNotIn('--no-browser',command)
            self.assertIn('--ServerApp.default_url=/lab/tree/first.ipynb',command)
            self.assertIn('--ServerApp.root_dir='+str((COMPONENT_ROOT/'notebooks').resolve()),command)

    def test_manual_launch_preserves_requested_port(self):
        with patch.object(notebooks,'configure_kernel'), patch.object(notebooks,'load_index',
                return_value=({'maintained_notebooks':['first.ipynb']},{})), \
                patch.object(sys,'argv',['notebooks','launch','--no-browser','--port','8897']), \
                patch.object(notebooks.subprocess,'call',return_value=0) as call:
            notebooks.main()
            command=call.call_args.args[0]
            self.assertIn('--no-browser',command)
            self.assertIn('--ServerApp.port=8897',command)


if __name__=='__main__':
    unittest.main()
