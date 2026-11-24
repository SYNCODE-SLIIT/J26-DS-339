"""Smoke test: the Component 1 web service starts as a real process and answers /api/health."""
import json
import os
import socket
import subprocess
import sys
import time
import unittest
import urllib.request
from pathlib import Path

WS = Path(__file__).resolve().parents[1]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class LaunchSmokeTests(unittest.TestCase):
    def test_service_starts_and_reports_health(self):
        port = _free_port()
        env = dict(os.environ, PYTHONPATH=str(WS / "source"), PYTHONDONTWRITEBYTECODE="1")
        proc = subprocess.Popen(
            [sys.executable, "-X", "utf8", "-B", "-m", "bpdr_input.web.app", "--host", "127.0.0.1", "--port", str(port)],
            cwd=WS, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        self.addCleanup(lambda: (proc.terminate(), proc.wait(timeout=15)))
        deadline = time.time() + 60
        body = None
        while time.time() < deadline and body is None:
            if proc.poll() is not None:
                self.fail(f"service exited early with code {proc.returncode}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=2) as r:
                    body = json.load(r)
            except Exception:
                time.sleep(0.5)
        self.assertIsNotNone(body, "service did not become healthy in 60 s")
        self.assertEqual(body["status"], "ok")
        for page in ("/", "/research", "/static/app.js"):
            with urllib.request.urlopen(f"http://127.0.0.1:{port}{page}", timeout=5) as r:
                self.assertEqual(r.status, 200, page)

    def test_launcher_scripts_parse(self):
        ps = "powershell"
        for name in ("Run-Component1.ps1", "Run-Component1-Panel.ps1", "Run-Studio.ps1"):
            cmd = (f"$e=$null;[void][System.Management.Automation.Language.Parser]::ParseFile("
                   f"'{WS / name}',[ref]$null,[ref]$e); if($e.Count){{$e|%{{$_.Message}};exit 1}}")
            try:
                r = subprocess.run([ps, "-NoProfile", "-Command", cmd], capture_output=True, text=True, timeout=60)
            except FileNotFoundError:
                self.skipTest("powershell not available")
            self.assertEqual(r.returncode, 0, f"{name}: {r.stdout}{r.stderr}")


if __name__ == "__main__":
    unittest.main()
