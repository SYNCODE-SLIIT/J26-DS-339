import hashlib
import io
import json
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
from starlette.testclient import TestClient

from bpdr_input.studio import evidence as E, question_engine as qe
from bpdr_input.web.app import app
from bpdr_input.web import sessions


def _wav_bytes(x: np.ndarray, sr: int = E.SR) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, x.astype(np.float32), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _sung_phrase(freqs=(220.0, 246.94, 261.63, 293.66), each_s=0.6) -> np.ndarray:
    parts = []
    for f in freqs:
        t = np.arange(int(each_s * E.SR)) / E.SR
        parts.append(0.35 * (np.sin(2 * np.pi * f * t) + 0.25 * np.sin(4 * np.pi * f * t)))
    return np.concatenate(parts).astype(np.float32)


class QuestionEngineTests(unittest.TestCase):
    def test_synthetic_candidates_and_all_strategies(self):
        t = np.arange(100) * 0.02
        # Two notes: A4 (440 Hz) and C5 + 30c sharp (532.4 Hz)
        c5_30 = 523.2511 * (2.0 ** (30.0 / 1200.0))
        f0 = np.where(t < 1.0, 440.0, c5_30)
        trk = {"t": t, "f0": f0, "voiced": np.ones(100, bool), "conf": np.full(100, 0.92)}
        probs = np.zeros(100)
        probs[50] = 0.55  # uncertain boundary peak
        b_hints = {"times_s": t.tolist(), "probabilities": probs.tolist(), "peaks_s": [1.0]}
        span_diag = {
            "tuning_center_cents": 0.0,
            "spans": [
                {"span_id": 1, "start_s": 0.0, "end_s": 0.98, "duration_s": 0.98, "pitch_midi": 69, "note_name": "A4", "vibrato_std_cents": 5.0, "mean_confidence": 0.92, "tuning_adjusted_deviation_cents": 0.0, "review_repair_candidate": False, "recommended_shift_cents": 0.0},
                {"span_id": 2, "start_s": 1.0, "end_s": 1.98, "duration_s": 0.98, "pitch_midi": 72, "note_name": "C5", "vibrato_std_cents": 6.0, "mean_confidence": 0.92, "tuning_adjusted_deviation_cents": 30.0, "review_repair_candidate": True, "recommended_shift_cents": -30.0},
            ],
        }
        cands = qe.extract_ambiguity_candidates(trk, boundary_hints=b_hints, span_diagnostics=span_diag, duration_s=2.0)
        cands = qe.calibrate_candidate_probabilities(cands)
        factors = {c["factor"] for c in cands}
        self.assertIn("boundary", factors)
        self.assertIn("pitch", factors)

        self.assertEqual(qe.select_questions(cands, k=3, strategy="none"), [])
        for strat in ("random", "lowest_confidence", "eig"):
            sel = qe.select_questions(cands, k=3, max_k=5, strategy=strat)
            self.assertGreaterEqual(len(sel), 1)
            self.assertLessEqual(len(sel), 3)
        # Dedicated pitch budget preserves pitch question under eig
        eig_sel = qe.select_questions(cands, k=2, max_k=5, strategy="eig", pitch_budget=1)
        self.assertTrue(any(q["factor"] == "pitch" for q in eig_sel))

        # Skip leaves span unresolved
        pitch_q = next(q for q in eig_sel if q["factor"] == "pitch")
        res = qe.apply_user_answers(cands, span_diag, [{"qid": pitch_q["qid"], "choice": "skip"}], duration_s=2.0)
        s2_row = next(r for r in res["intent_map"] if r["span_id"] == 2)
        self.assertEqual(s2_row["state"], "unresolved")
        self.assertEqual(res["centred_span_ids"], [])


class WebServiceContractTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.created_sids = []

    def tearDown(self):
        for sid in self.created_sids:
            try:
                sessions.delete_session(sid)
            except Exception:
                pass

    def _new_session(self) -> str:
        r = self.client.post("/api/sessions", json={"consent": True})
        self.assertEqual(r.status_code, 201)
        sid = r.json()["session_id"]
        self.assertRegex(sid, r"^[0-9a-f]{32}$")
        self.created_sids.append(sid)
        return sid

    def test_pages_and_research_claims_guard(self):
        r_home = self.client.get("/")
        self.assertEqual(r_home.status_code, 200)
        self.assertIn("Active Intent Confirmation", r_home.text)

        r_res = self.client.get("/research")
        self.assertEqual(r_res.status_code, 200)

        r_sum = self.client.get("/api/research/summary")
        self.assertEqual(r_sum.status_code, 200)
        summary = r_sum.json()
        self.assertIn("claims_ledger", summary)
        may_text = json.dumps(summary["claims_ledger"]["may_claim"])
        self.assertNotIn("+110%", may_text)
        self.assertNotIn("0% harmful distortion", may_text)

    def test_room_check_and_four_statuses(self):
        # 1. ACCEPT_UNCHANGED + room check + clarification + handoff + SHA-256 + DELETE
        sid = self._new_session()
        r_room = self.client.post(f"/api/sessions/{sid}/room", json={"preset": "quiet"})
        self.assertEqual(r_room.status_code, 200)
        self.assertEqual(r_room.json()["verdict"], "good")

        wav = _wav_bytes(_sung_phrase())
        r_up = self.client.post(f"/api/sessions/{sid}/audio", files={"file": ("take.wav", wav, "audio/wav")})
        self.assertEqual(r_up.status_code, 200)
        self.assertFalse(r_up.json()["technical_failure"])

        r_an = self.client.post(f"/api/sessions/{sid}/analyze", json={"intent": "sung", "strategy": "eig", "k": 3})
        self.assertEqual(r_an.status_code, 200)
        status = r_an.json()["status"]
        self.assertIn(status, ("ACCEPT_UNCHANGED", "ACCEPT_PREPARED"))

        r_qs = self.client.get(f"/api/sessions/{sid}/questions")
        self.assertEqual(r_qs.status_code, 200)

        r_ans = self.client.post(f"/api/sessions/{sid}/answers", json={"confirmed_as_heard": True, "answers": []})
        self.assertEqual(r_ans.status_code, 200)

        r_hf = self.client.get(f"/api/sessions/{sid}/handoff")
        self.assertEqual(r_hf.status_code, 200)
        hf = r_hf.json()
        self.assertEqual(hf["schema"], "bpdr_studio_handoff/0.2")
        self.assertIsNotNone(hf["accepted_audio"])
        self.assertEqual(
            hashlib.sha256(Path(hf["accepted_audio"]).read_bytes()).hexdigest(),
            hf["accepted_sha256"],
        )
        self.assertEqual(hf["intent_map"]["status_tag"], "proposal")
        self.assertEqual(hf["trust_curve"]["status_tag"], "proposal")

        r_del = self.client.delete(f"/api/sessions/{sid}")
        self.assertEqual(r_del.status_code, 200)
        self.assertTrue(r_del.json()["deleted"])

        # 2. RERECORD_REQUIRED (clipped audio -> accepted_audio is None)
        sid_clip = self._new_session()
        clipped = np.clip(_sung_phrase() * 40.0, -1.0, 1.0)
        self.client.post(f"/api/sessions/{sid_clip}/audio", files={"file": ("clip.wav", _wav_bytes(clipped), "audio/wav")})
        r_clip_an = self.client.post(f"/api/sessions/{sid_clip}/analyze", json={"intent": "sung"})
        self.assertEqual(r_clip_an.json()["status"], "RERECORD_REQUIRED")
        r_clip_hf = self.client.get(f"/api/sessions/{sid_clip}/handoff")
        self.assertIsNone(r_clip_hf.json()["accepted_audio"])

        # 3. TECHNICAL_FAILURE (silence -> accepted_audio is None)
        sid_sil = self._new_session()
        silence = np.zeros(E.SR, dtype=np.float32)
        self.client.post(f"/api/sessions/{sid_sil}/audio", files={"file": ("sil.wav", _wav_bytes(silence), "audio/wav")})
        r_sil_an = self.client.post(f"/api/sessions/{sid_sil}/analyze", json={})
        self.assertEqual(r_sil_an.json()["status"], "TECHNICAL_FAILURE")
        r_sil_hf = self.client.get(f"/api/sessions/{sid_sil}/handoff")
        self.assertIsNone(r_sil_hf.json()["accepted_audio"])

    def test_trimmed_trailing_silence_does_not_crash_render_notes(self):
        sid = self._new_session()
        # Phrase followed by 1.5s of trailing silence so auto_trim shortens dur below untrimmed note onset
        x = np.concatenate([_sung_phrase(), np.zeros(int(1.5 * E.SR), dtype=np.float32)])
        r_up = self.client.post(f"/api/sessions/{sid}/audio", files={"file": ("trail.wav", _wav_bytes(x), "audio/wav")})
        self.assertEqual(r_up.status_code, 200)
        r_an = self.client.post(f"/api/sessions/{sid}/analyze", json={"intent": "sung", "auto_trim": True})
        self.assertEqual(r_an.status_code, 200)
        self.assertIn(r_an.json()["status"], ("ACCEPT_UNCHANGED", "ACCEPT_PREPARED"))


if __name__ == "__main__":
    unittest.main()
