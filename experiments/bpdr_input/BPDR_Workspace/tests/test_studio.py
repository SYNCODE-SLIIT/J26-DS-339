import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from bpdr_input.studio import boundary, chords, clean, confirmation, evidence as E, pipeline, room, speech, synth, trim

SR = E.SR


def tone_phrase(freqs=(220.0, 247.0, 262.0, 294.0), each=0.6, noise=0.0, seed=1):
    rng = np.random.default_rng(seed)
    parts = []
    for f in freqs:
        t = np.arange(int(each * SR)) / SR
        parts.append(0.4 * (np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t)))
    x = np.concatenate(parts).astype(np.float32)
    return (x + noise * rng.standard_normal(len(x))).astype(np.float32)


class StudioTests(unittest.TestCase):
    def write(self, x, name="a.wav"):
        d = Path(tempfile.mkdtemp()); p = d / name; sf.write(p, x, SR, subtype="PCM_24"); return p, d

    def test_clipped_spans_found(self):
        x = np.zeros(SR, dtype=np.float32); x[1000:1200] = 1.0
        spans = E.clipped_spans(x)
        self.assertEqual(len(spans), 1); self.assertAlmostEqual(spans[0][0], 1000 / SR, places=4)

    def test_handoff_contract_extractor_reads_the_accepted_file(self):
        """The file the shared extractor opens must be byte-identical to the accepted WAV the studio reports."""
        import hashlib
        from unittest import mock
        from bpdr_input.studio import consumer
        p, d = self.write(tone_phrase())
        seen = []

        def spy(path):
            seen.append((str(path), hashlib.sha256(Path(path).read_bytes()).hexdigest()))
            return []

        with mock.patch.object(consumer, "shared_extractor_notes", side_effect=spy):
            r = pipeline.prepare(p, d / "o")
        self.assertIn(r["status"], ("ACCEPT_UNCHANGED", "ACCEPT_PREPARED"))
        self.assertGreaterEqual(len(seen), 2)
        path, digest = seen[-1]                      # the call made on the accepted audio
        self.assertEqual(Path(path), Path(r["accepted_audio"]))
        self.assertEqual(digest, r["accepted_sha256"])
        self.assertEqual(hashlib.sha256(Path(r["accepted_audio"]).read_bytes()).hexdigest(), r["accepted_sha256"])

    def test_retention_deletes_only_expired_session_folders(self):
        import os, time
        from bpdr_input.studio import retention
        root = Path(tempfile.mkdtemp())
        old, fresh, other = root / ("a" * 32), root / ("b" * 32), root / "keep_me"
        for f in (old, fresh, other):
            f.mkdir(); (f / "x.wav").write_bytes(b"1")
        past = time.time() - 48 * 3600
        for f in (old, other):
            os.utime(f, (past, past))
        self.assertEqual(retention.purge_expired(root, 24), 1)
        self.assertFalse(old.exists()); self.assertTrue(fresh.exists()); self.assertTrue(other.exists())
        outside = Path(tempfile.mkdtemp())
        self.assertEqual(retention.delete_sessions(root, [outside, fresh, None]), 1)
        self.assertTrue(outside.exists()); self.assertFalse(fresh.exists())

    def test_room_check_labels_synthetic_rooms(self):
        rng = np.random.default_rng(3); n = 3 * SR; t = np.arange(n) / SR
        quiet = (1e-4 * rng.standard_normal(n)).astype(np.float32)
        hum = (0.004 * np.sin(2 * np.pi * 50 * t) + 0.002 * np.sin(2 * np.pi * 100 * t) + 1e-4 * rng.standard_normal(n)).astype(np.float32)
        fan = (0.03 * rng.standard_normal(n)).astype(np.float32)
        knock = quiet.copy(); knock[SR:SR + 800] += 0.3 * rng.standard_normal(800).astype(np.float32)
        loud = (0.3 * rng.standard_normal(n)).astype(np.float32)
        self.assertEqual(room.room_check(quiet)["verdict"], "good")
        h = room.room_check(hum); self.assertEqual(h["verdict"], "noisy"); self.assertEqual(h["hum_hz"], 50.0)
        self.assertEqual(room.room_check(fan)["verdict"], "noisy")
        self.assertFalse(room.room_check(knock)["usable_reference"])
        self.assertEqual(room.room_check(loud)["verdict"], "too_loud")
        self.assertEqual(room.room_check(quiet[:SR // 2])["verdict"], "unusable")

    def test_room_noise_profile_is_used_and_recorded(self):
        rng = np.random.default_rng(5)
        noise = (0.02 * rng.standard_normal(2 * SR)).astype(np.float32)
        x = tone_phrase() + (0.02 * rng.standard_normal(len(tone_phrase()))).astype(np.float32)
        p, d = self.write(x)
        r1 = pipeline.prepare(p, d / "a", with_pyin=False)
        r2 = pipeline.prepare(p, d / "b", with_pyin=False, room_noise=noise)
        self.assertEqual(r1["noise_profile_policy"], "quietest_frames_of_recording")
        self.assertEqual(r2["noise_profile_policy"], "room_check_voice_free_clip")
        side = __import__("json").loads(Path(r2["sidecar"]).read_text(encoding="utf-8")) if r2.get("sidecar") else {}
        self.assertEqual(side.get("noise_profile_policy"), "room_check_voice_free_clip")

    def test_clipping_detected_in_48k_stereo_before_resampling(self):
        import librosa
        base = tone_phrase()
        up = librosa.resample(base, orig_sr=SR, target_sr=48000)
        hot = np.clip(up * 40, -1, 1).astype(np.float32)
        d = Path(tempfile.mkdtemp()); p = d / "s.wav"
        sf.write(p, np.c_[hot, hot * 0.9], 48000, subtype="PCM_24")
        r = pipeline.prepare(p, d / "o")
        self.assertEqual(r["status"], "RERECORD_REQUIRED"); self.assertTrue(r["guidance"])

    def test_notes_from_track_merges_runs_and_keeps_rests(self):
        t = np.arange(40) * 0.01
        f0 = np.r_[np.full(15, 440.0), np.full(5, np.nan), np.full(20, 493.88)]
        trk = {"t": t, "f0": f0, "voiced": np.isfinite(f0)}
        notes = E.notes_from_track(trk, 0.4)
        self.assertEqual([n["pitch_midi"] for n in notes], [69, 71])
        self.assertGreater(notes[1]["onset_s"], notes[0]["onset_s"] + notes[0]["duration_s"] - 1e-9)

    def test_spectral_gate_keeps_length_and_is_finite(self):
        x = tone_phrase(noise=0.05); y = clean.spectral_gate(x)
        self.assertEqual(len(y), len(x)); self.assertTrue(np.isfinite(y).all()); self.assertLessEqual(float(np.abs(y).max()), 0.951)

    def test_chooser_rejects_pitch_shifted_candidate(self):
        n = 100
        base = {"voiced": np.ones(n, bool), "conf": np.full(n, 0.8), "f0": np.full(n, 220.0)}
        shifted = {"voiced": np.ones(n, bool), "conf": np.full(n, 0.99), "f0": np.full(n, 220.0 * 2 ** (60 / 1200))}
        use, ev = clean.choose(base, shifted)
        self.assertFalse(use); self.assertGreater(ev["median_pitch_shift_cents"], 25)

    def test_silence_is_technical_failure_without_accepted_audio(self):
        p, d = self.write(np.zeros(SR, dtype=np.float32))
        r = pipeline.prepare(p, d / "o")
        self.assertEqual(r["status"], "TECHNICAL_FAILURE"); self.assertIsNone(r["accepted_audio"])

    def test_clipping_gets_located_rerecord_guidance(self):
        p, d = self.write(np.clip(tone_phrase() * 40, -1, 1).astype(np.float32))
        r = pipeline.prepare(p, d / "o")
        self.assertEqual(r["status"], "RERECORD_REQUIRED"); self.assertIsNone(r["accepted_audio"])
        self.assertTrue(any(" s:" in g for g in r["guidance"]))

    def test_accepted_package_is_consistent(self):
        p, d = self.write(tone_phrase())
        r = pipeline.prepare(p, d / "o", intent="sung", with_pyin=False)
        self.assertTrue(r["status"].startswith("ACCEPT"), r)
        out, rate = sf.read(r["accepted_audio"]); self.assertEqual(rate, SR)
        self.assertEqual(len(out), len(E.load_mono(p)[0]))               # sample count preserved
        self.assertEqual(r["accepted_sha256"], pipeline.sha256(r["accepted_audio"]))
        self.assertIn("ownership_note", __import__("json").loads(Path(r["sidecar"]).read_text()))

    def test_speech_to_melody_stays_in_scale_and_in_bounds(self):
        rng = np.random.default_rng(3)
        t, v, f0, parts = [], [], [], []
        hop = 0.01; n = 0
        trk_f0 = []
        for k in range(12):                                            # 12 syllables of ~0.2 s, gliding pitches
            base = 120 + 25 * rng.standard_normal()
            trk_f0 += list(np.linspace(base, base * 1.1, 20)) + [np.nan] * 5
        trk = {"t": np.arange(len(trk_f0)) * hop, "f0": np.array(trk_f0), "voiced": np.isfinite(trk_f0), "conf": np.ones(len(trk_f0))}
        env = np.concatenate([np.r_[np.hanning(20), np.zeros(5)] for _ in range(12)])
        x = (np.repeat(env, int(hop * SR)) * 0.5 * np.sin(np.arange(len(env) * int(hop * SR)) * 0.05)).astype(np.float32)
        m = speech.speech_to_melody(x, trk, tonic_pc=0, scale="major", bpm=100)
        allowed = set(speech.SCALES["major"])
        self.assertTrue(all(n["pitch_midi"] % 12 in allowed for n in m["notes"]))
        self.assertEqual(m["notes"][-1]["pitch_midi"] % 12, 0)
        self.assertTrue(all(n["duration_s"] > 0 for n in m["notes"]))

    def test_chord_labels_voice_to_midi(self):
        self.assertEqual(chords.label_to_midi("C:maj"), [48, 52, 55])
        self.assertEqual(chords.label_to_midi("A:min7"), [57, 60, 64, 67])

    def test_render_notes_length(self):
        y = synth.render_notes([{"pitch_midi": 60, "onset_s": 0.1, "duration_s": 0.3}], 1.0)
        self.assertGreaterEqual(len(y), SR); self.assertGreater(float(np.abs(y).max()), 0.05)

    def test_silence_trimming_removes_leading_and_trailing_silence(self):
        phrase = tone_phrase()
        silence_lead = np.zeros(int(2.0 * SR), dtype=np.float32)
        silence_trail = np.zeros(int(1.5 * SR), dtype=np.float32)
        x = np.concatenate([silence_lead, phrase, silence_trail])
        trimmed_x, meta = trim.trim_silence(x, sr=SR)
        self.assertTrue(meta["was_trimmed"])
        self.assertGreaterEqual(meta["trimmed_leading_s"], 1.8)
        self.assertGreaterEqual(meta["trimmed_trailing_s"], 1.3)
        self.assertLess(len(trimmed_x), len(x))
        # Ensure phrase content is intact
        self.assertGreater(len(trimmed_x), len(phrase) - int(0.5 * SR))

    def test_phrase_selection_extracts_best_20s_from_long_recording(self):
        # 28 seconds total: 2s silence, 24s tones (4 x 6s), 2s silence
        lead = np.zeros(int(2.0 * SR), dtype=np.float32)
        tones = tone_phrase(each=6.0)  # 4 x 6.0 = 24s active singing/tones
        trail = np.zeros(int(2.0 * SR), dtype=np.float32)
        long_x = np.concatenate([lead, tones, trail])
        selected_x, info = trim.select_best_phrase(long_x, sr=SR, max_duration_s=20.0)
        self.assertTrue(info["phrase_selected"])
        self.assertLessEqual(info["duration_s"], 20.0)
        self.assertGreaterEqual(info["start_s"], 1.5)
        self.assertGreater(float(np.abs(selected_x).max()), 0.1)

    def test_pipeline_integrates_phrase_trimming_and_records_metadata(self):
        phrase = tone_phrase()
        lead = np.zeros(int(2.5 * SR), dtype=np.float32)
        x = np.concatenate([lead, phrase])
        p, d = self.write(x)
        r = pipeline.prepare(p, d / "o", with_pyin=False)
        self.assertTrue(r["status"].startswith("ACCEPT"))
        self.assertIsNotNone(r.get("phrase_selection"))
        self.assertTrue(r["phrase_selection"].get("trimmed_silence"))
        self.assertGreaterEqual(r["phrase_selection"].get("trimmed_leading_s", 0), 2.0)
        # Check sidecar
        sidecar = __import__("json").loads(Path(r["sidecar"]).read_text(encoding="utf-8"))
        self.assertIn("phrase_selection", sidecar)
        self.assertIn("offset", sidecar["time_reference"])

    def test_hear_back_confirmation_loop(self):
        # Positive confirmation
        conf = confirmation.evaluate_confirmation(True, notes_version="after_prepared_pesto")
        self.assertEqual(conf["status"], "CONFIRMED")
        self.assertEqual(conf["action"], "PROCEED_TO_DOWNSTREAM")
        self.assertTrue(conf["confirmed"])

        # Negative confirmation with flagged spans
        dis = confirmation.evaluate_confirmation(False, flagged_spans=[(1.2, 3.4)], user_comment="Note pitch was off")
        self.assertEqual(dis["status"], "USER_DISAGREED")
        self.assertEqual(dis["action"], "RERECORD_SPANS")
        self.assertFalse(dis["confirmed"])
        self.assertTrue(any("1.2" in g and "3.4" in g for g in dis["guidance"]))

        # Integration with session recording and sidecar update
        p, d = self.write(tone_phrase())
        r = pipeline.prepare(p, d / "o", with_pyin=False)
        confirmation.record_confirmation(d / "o", confirmed=True, notes_version="after_prepared_pesto")
        saved = confirmation.load_confirmation(d / "o")
        self.assertIsNotNone(saved)
        self.assertTrue(saved["confirmed"])
        sidecar = __import__("json").loads(Path(r["sidecar"]).read_text(encoding="utf-8"))
        self.assertIn("user_confirmation", sidecar)
        self.assertTrue(sidecar["user_confirmation"]["confirmed"])

    def test_boundary_model_loader_and_architecture(self):
        model = boundary.load_boundary_model()
        self.assertIsInstance(model, boundary.FastTNet)
        self.assertEqual(model.nin, 5)
        self.assertFalse(model.training)
        self.assertAlmostEqual(model.default_threshold, 0.8)
        self.assertAlmostEqual(model.eval_logit_bias, 0.9)
        if boundary.V2_CHECKPOINT.exists():
            v2_model = boundary.load_boundary_model(boundary.V2_CHECKPOINT)
            self.assertIsInstance(v2_model, boundary.FastTNetV2)
            self.assertEqual(v2_model.nin, 5)
            self.assertFalse(v2_model.training)
            self.assertAlmostEqual(v2_model.default_threshold, 0.65)
            self.assertAlmostEqual(v2_model.eval_logit_offset, 0.0)
        # Non-existent checkpoint should raise FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            boundary.load_boundary_model("non_existent_boundary_model.pt")

    def test_boundary_feature_extraction_shapes_and_values(self):
        t = np.arange(80) * 0.02
        f0 = np.full(80, 440.0)
        f0[30:40] = np.nan
        trk = {"t": t, "f0": f0, "voiced": np.isfinite(f0), "conf": np.full(80, 0.9)}
        x = tone_phrase()

        # Full features (nin=5)
        feats5 = boundary.extract_boundary_features(trk, x=x, sr=SR, nin=5)
        self.assertEqual(feats5.shape, (80, 5))
        self.assertTrue(np.isfinite(feats5).all())
        self.assertEqual(feats5.dtype, np.float32)

        # Lean features (nin=3)
        feats3 = boundary.extract_boundary_features(trk, x=None, sr=SR, nin=3)
        self.assertEqual(feats3.shape, (80, 3))
        self.assertTrue(np.isfinite(feats3).all())

        # Features with boundary edges (nin=6)
        feats6 = boundary.extract_boundary_features(trk, x=x, sr=SR, boundary_edges=[0.4, 0.8], nin=6)
        self.assertEqual(feats6.shape, (80, 6))
        self.assertTrue(np.isfinite(feats6).all())

    def test_boundary_inference_and_hints_extraction(self):
        model = boundary.load_boundary_model()
        t = np.arange(100) * 0.02
        f0 = np.full(100, 330.0)
        trk = {"t": t, "f0": f0, "voiced": np.ones(100, bool), "conf": np.full(100, 0.95)}
        x = tone_phrase()

        probs = boundary.predict_boundary_probabilities(model, trk, x=x, sr=SR)
        self.assertEqual(len(probs), 100)
        self.assertTrue((probs >= 0.0).all() and (probs <= 1.0).all())

        hints = boundary.extract_boundary_hints(trk, x=x, sr=SR, model=model, threshold=0.4)
        for key in ("times_s", "probabilities", "peaks_s", "peak_indices", "threshold", "model", "checkpoint"):
            self.assertIn(key, hints)
        self.assertEqual(len(hints["times_s"]), 100)
        self.assertEqual(len(hints["probabilities"]), 100)
        self.assertEqual(hints["threshold"], 0.4)

    def test_boundary_hints_integration_in_pipeline_and_sidecar(self):
        p, d = self.write(tone_phrase())
        r = pipeline.prepare(p, d / "o", with_pyin=False, with_boundary_hints=True)
        self.assertTrue(r["status"].startswith("ACCEPT"))
        self.assertIn("boundary_hints", r)
        hints = r["boundary_hints"]
        self.assertIsNotNone(hints)
        self.assertIn("probabilities", hints)
        self.assertGreater(len(hints["probabilities"]), 0)

        # Check sidecar content
        sidecar = __import__("json").loads(Path(r["sidecar"]).read_text(encoding="utf-8"))
        self.assertIn("boundary_hints", sidecar)
        self.assertEqual(sidecar["boundary_hints"]["probabilities"], hints["probabilities"])

        # Check write_boundary_hints_to_sidecar updater
        dummy_hints = {"test_marker": 123}
        boundary.write_boundary_hints_to_sidecar(r["sidecar"], dummy_hints)
        updated_sidecar = __import__("json").loads(Path(r["sidecar"]).read_text(encoding="utf-8"))
        self.assertEqual(updated_sidecar["boundary_hints"], dummy_hints)

    def test_boundary_20ms_grid_resampling_and_100hz_vs_50hz_parity(self):
        x = tone_phrase()
        # Construct identical step pitch contour at 50 Hz (20 ms) and 100 Hz (10 ms), with an unvoiced gap
        def make_f0(t_arr):
            f0 = np.select(
                [t_arr < 0.6, t_arr < 1.2, t_arr < 1.8],
                [220.0, 247.0, 262.0],
                default=294.0,
            ).astype(np.float32)
            f0[(t_arr >= 0.88) & (t_arr <= 0.96)] = np.nan
            return f0

        t_50 = np.arange(120) * 0.02
        f0_50 = make_f0(t_50)
        trk_50 = {"t": t_50, "f0": f0_50, "voiced": np.isfinite(f0_50), "conf": np.full(120, 0.9, dtype=np.float32)}

        t_100 = np.arange(240) * 0.01
        f0_100 = make_f0(t_100)
        trk_100 = {"t": t_100, "f0": f0_100, "voiced": np.isfinite(f0_100), "conf": np.full(240, 0.9, dtype=np.float32)}

        feats_50 = boundary.extract_boundary_features(trk_50, x=x, sr=SR, nin=5)
        feats_100 = boundary.extract_boundary_features(trk_100, x=x, sr=SR, nin=5)
        self.assertEqual(feats_50.shape, (120, 5))
        self.assertEqual(feats_100.shape, (120, 5))
        np.testing.assert_allclose(feats_100, feats_50, atol=1e-6)

        model = boundary.load_boundary_model()
        probs_50 = boundary.predict_boundary_probabilities(model, trk_50, x=x, sr=SR)
        probs_100 = boundary.predict_boundary_probabilities(model, trk_100, x=x, sr=SR)
        self.assertEqual(len(probs_100), 120)
        np.testing.assert_allclose(probs_100, probs_50, atol=1e-6)

        hints_50 = boundary.extract_boundary_hints(trk_50, x=x, sr=SR, model=model)
        hints_100 = boundary.extract_boundary_hints(trk_100, x=x, sr=SR, model=model)
        self.assertEqual(hints_100["times_s"], hints_50["times_s"])
        self.assertEqual(hints_100["peaks_s"], hints_50["peaks_s"])
        self.assertEqual(hints_100["peak_indices"], hints_50["peak_indices"])
        self.assertEqual(hints_100["frame_hop_s"], 0.02)

        # Also verify live 100 Hz PESTO track is resampled onto canonical 20 ms grid
        pesto_100 = E.pesto_track(x)
        self.assertGreater(len(pesto_100["t"]), 200)
        hints_pesto = boundary.extract_boundary_hints(pesto_100, x=x, sr=SR, model=model)
        self.assertEqual(len(hints_pesto["probabilities"]), 120)
        self.assertAlmostEqual(hints_pesto["times_s"][1] - hints_pesto["times_s"][0], 0.02, places=4)

    def test_boundary_feature_parity_against_analysis(self):
        from bpdr_input.audio import analysis

        x = tone_phrase()
        def stub_pitch(audio, sr, cache_dir):
            t = np.arange(240) * 0.01
            f0 = np.select([t < 0.6, t < 1.2, t < 1.8], [220.0, 247.0, 262.0], default=294.0)
            conf = np.full(len(t), 0.95)
            return t, f0, conf

        cache_dir = Path(tempfile.mkdtemp())
        feat_dict, _ = analysis.extract(x, SR, cache_dir, pitch_provider=stub_pitch)
        trk = {
            "t": feat_dict["times_seconds"],
            "f0": feat_dict["f0_hz"],
            "voiced": feat_dict["voiced_mask"],
            "conf": feat_dict["confidence"],
        }
        feats = boundary.extract_boundary_features(trk, x=x, sr=SR, nin=5)
        self.assertEqual(feats.shape, (len(feat_dict["times_seconds"]), 5))

        e_ref = feat_dict["log_energy_db"]
        e_ref_zs = (e_ref - e_ref.mean()) / (e_ref.std() + 1e-6)
        np.testing.assert_allclose(feats[:, 3], e_ref_zs, atol=1e-5)
        np.testing.assert_allclose(feats[:, 4], feat_dict["onset_score"], atol=1e-5)
        np.testing.assert_allclose(feats[:, 1], feat_dict["voiced_mask"].astype(np.float32), atol=1e-6)
        np.testing.assert_allclose(feats[:, 2], feat_dict["confidence"], atol=1e-6)

    def test_boundary_threshold_calibration_and_metadata_loading(self):
        import json
        import torch

        v1_model = boundary.load_boundary_model()
        self.assertIsInstance(v1_model, boundary.FastTNet)
        self.assertAlmostEqual(v1_model.default_threshold, 0.8)
        self.assertAlmostEqual(v1_model.eval_logit_bias, 0.9)

        t = np.arange(80) * 0.02
        trk = {"t": t, "f0": np.full(80, 330.0), "voiced": np.ones(80, bool), "conf": np.full(80, 0.95)}
        v1_hints = boundary.extract_boundary_hints(trk, x=tone_phrase(), sr=SR, model=v1_model)
        self.assertAlmostEqual(v1_hints["threshold"], 0.8)

        # Verify v2 standalone and BPDR checkpoints + metadata JSON loading
        if boundary.V2_CHECKPOINT.exists():
            v2_real = boundary.load_boundary_model(boundary.V2_CHECKPOINT)
            self.assertIsInstance(v2_real, boundary.FastTNetV2)
            self.assertEqual(v2_real.nin, 5)
            self.assertAlmostEqual(v2_real.default_threshold, 0.65)
            self.assertAlmostEqual(v2_real.eval_logit_offset, 0.0)
            v2_real_hints = boundary.extract_boundary_hints(trk, x=tone_phrase(), sr=SR, model=v2_real)
            self.assertAlmostEqual(v2_real_hints["threshold"], 0.65)
            self.assertEqual(v2_real_hints["model"], "FastTNetV2")

        bpdr_v2_ckpt = boundary.V2_CHECKPOINT.with_name("boundary_transformer_v2_bpdr.pt")
        if bpdr_v2_ckpt.exists():
            bpdr_model = boundary.load_boundary_model(bpdr_v2_ckpt)
            self.assertIsInstance(bpdr_model, boundary.FastTNetV2)
            self.assertEqual(bpdr_model.nin, 6)
            self.assertAlmostEqual(bpdr_model.default_threshold, 0.60)
            bpdr_hints = boundary.extract_boundary_hints(
                trk, x=tone_phrase(), sr=SR, model=bpdr_model, boundary_edges=[0.6, 1.2]
            )
            self.assertAlmostEqual(bpdr_hints["threshold"], 0.60)

        # Verify v3 standalone and BPDR checkpoints + metadata JSON loading (DEFAULT remains V1)
        self.assertEqual(boundary.DEFAULT_CHECKPOINT, boundary.V1_CHECKPOINT)
        if boundary.V3_CHECKPOINT.exists():
            v3_model = boundary.load_boundary_model(boundary.V3_CHECKPOINT)
            self.assertIsInstance(v3_model, boundary.FastTNet)
            self.assertEqual(v3_model.nin, 5)
            self.assertAlmostEqual(v3_model.default_threshold, 0.65)
            self.assertAlmostEqual(v3_model.eval_logit_bias, 0.0)
            v3_hints = boundary.extract_boundary_hints(trk, x=tone_phrase(), sr=SR, model=v3_model)
            self.assertAlmostEqual(v3_hints["threshold"], 0.65)
            self.assertEqual(v3_hints["model"], "transformer_v3")

        bpdr_v3_ckpt = boundary.V3_CHECKPOINT.with_name("boundary_transformer_v3_bpdr.pt")
        if bpdr_v3_ckpt.exists():
            bpdr_v3 = boundary.load_boundary_model(bpdr_v3_ckpt)
            self.assertIsInstance(bpdr_v3, boundary.FastTNet)
            self.assertEqual(bpdr_v3.nin, 6)
            self.assertAlmostEqual(bpdr_v3.default_threshold, 0.60)
            self.assertAlmostEqual(bpdr_v3.eval_logit_bias, 0.0)
            bpdr_v3_hints = boundary.extract_boundary_hints(
                trk, x=tone_phrase(), sr=SR, model=bpdr_v3, boundary_edges=[0.6, 1.2]
            )
            self.assertAlmostEqual(bpdr_v3_hints["threshold"], 0.60)

        # Verify metadata sidecar loading for custom calibrated v1-style checkpoints
        d = Path(tempfile.mkdtemp())
        ckpt_path = d / "boundary_custom_test.pt"
        meta_path = d / "boundary_custom_test.json"
        torch.save(v1_model.state_dict(), ckpt_path)
        meta_path.write_text(
            json.dumps({"model": "boundary_transformer_custom", "calibrated_threshold": 0.45, "eval_logit_bias": 0.0}),
            encoding="utf-8",
        )
        custom_model = boundary.load_boundary_model(ckpt_path)
        self.assertAlmostEqual(custom_model.default_threshold, 0.45)
        self.assertAlmostEqual(custom_model.eval_logit_bias, 0.0)
        self.assertEqual(custom_model.model_name, "boundary_transformer_custom")
        custom_hints = boundary.extract_boundary_hints(trk, x=tone_phrase(), sr=SR, model=custom_model)
        self.assertAlmostEqual(custom_hints["threshold"], 0.45)
        self.assertEqual(custom_hints["model"], "boundary_transformer_custom")

    def test_studio_app_boundary_markers(self):
        from bpdr_input.application import studio_app
        from bpdr_input.studio import plots

        notes = [{"pitch_midi": 60, "onset_s": 0.1, "duration_s": 0.4}, {"pitch_midi": 62, "onset_s": 0.6, "duration_s": 0.4}]
        fig = plots.piano_rolls([("Test", notes, "after")], 1.2)
        studio_app._annotate_boundary_markers(fig, {"peaks_s": [0.1, 0.6]})
        self.assertGreaterEqual(len(fig.axes[0].lines), 2)

    def test_provisional_boundary_edges_parity_against_analysis(self):
        from scipy.signal import find_peaks
        from bpdr_input.audio import analysis

        x = tone_phrase()

        def stub_pitch(audio, sr, cache_dir):
            t = np.arange(240) * 0.01
            f0 = np.select([t < 0.6, t < 1.2, t < 1.8], [220.0, 247.0, 262.0], default=294.0)
            conf = np.full(len(t), 0.95)
            # Include an unvoiced gap to verify delta-masking and transition parity
            gap = (t >= 0.88) & (t <= 0.96)
            f0[gap] = 0.0
            conf[gap] = 0.2
            return t, f0, conf

        cache_dir = Path(tempfile.mkdtemp())
        feat_dict, _ = analysis.extract(x, SR, cache_dir, pitch_provider=stub_pitch)
        trk_20 = {
            "t": feat_dict["times_seconds"],
            "f0": feat_dict["f0_hz"],
            "voiced": feat_dict["voiced_mask"],
            "conf": feat_dict["confidence"],
        }
        edges, t_grid, score = boundary.provisional_boundary_edges(trk_20, x=x, sr=SR, return_score=True)
        raw_70 = feat_dict["boundary_features_71"][:, 70]
        np.testing.assert_allclose(score, raw_70, atol=1e-6)
        self.assertEqual(float(np.max(np.abs(score - raw_70))), 0.0)

        pk_ref, _ = find_peaks(raw_70, height=0.5, distance=5)
        np.testing.assert_allclose(edges, feat_dict["times_seconds"][pk_ref], atol=1e-6)

        # Also verify 100 Hz input produces consistent provisional edges
        t_100, f0_100, conf_100 = stub_pitch(x, SR, cache_dir)
        trk_100 = {
            "t": t_100,
            "f0": np.where(conf_100 >= 0.8, f0_100, np.nan),
            "voiced": conf_100 >= 0.8,
            "conf": conf_100,
        }
        edges_100 = boundary.provisional_boundary_edges(trk_100, x=x, sr=SR)
        # trk_100 covers 0..2.39s (120 20ms frames) vs analysis 0..2.40s (121 frames); compare overlapping range
        np.testing.assert_allclose(edges_100, edges[edges <= t_100[-1] + 1e-6], atol=1e-6)

    def test_bpdr_nin6_automatic_provisional_edges_and_pipeline(self):
        x = tone_phrase()
        t = np.arange(120) * 0.02
        f0 = np.select([t < 0.6, t < 1.2, t < 1.8], [220.0, 247.0, 262.0], default=294.0)
        trk = {"t": t, "f0": f0, "voiced": np.ones(120, bool), "conf": np.full(120, 0.95)}

        # nin=5 should record boundary_edges_source == "none"
        v1_hints = boundary.extract_boundary_hints(trk, x=x, sr=SR, checkpoint_path=boundary.V1_CHECKPOINT)
        self.assertEqual(v1_hints["boundary_edges_source"], "none")

        if boundary.V2_BPDR_CHECKPOINT.exists():
            auto_hints = boundary.extract_boundary_hints(
                trk, x=x, sr=SR, checkpoint_path=boundary.V2_BPDR_CHECKPOINT
            )
            self.assertEqual(auto_hints["boundary_edges_source"], "provisional_20ms")
            self.assertAlmostEqual(auto_hints["threshold"], 0.60)

            edges = boundary.provisional_boundary_edges(trk, x=x, sr=SR)
            manual_hints = boundary.extract_boundary_hints(
                trk, x=x, sr=SR, checkpoint_path=boundary.V2_BPDR_CHECKPOINT, boundary_edges=edges
            )
            self.assertEqual(manual_hints["boundary_edges_source"], "provided")
            self.assertEqual(auto_hints["probabilities"], manual_hints["probabilities"])
            self.assertEqual(auto_hints["peaks_s"], manual_hints["peaks_s"])

            # End-to-end pipeline.prepare with nin=6 V2_BPDR_CHECKPOINT
            p, d = self.write(x)
            r = pipeline.prepare(
                p, d / "o_bpdr", with_pyin=False, with_boundary_hints=True, boundary_checkpoint=boundary.V2_BPDR_CHECKPOINT
            )
            self.assertTrue(r["status"].startswith("ACCEPT"), r)
            self.assertIn("boundary_hints", r)
            self.assertEqual(r["boundary_hints"]["boundary_edges_source"], "provisional_20ms")
            self.assertAlmostEqual(r["boundary_hints"]["threshold"], 0.60)
            sidecar = __import__("json").loads(Path(r["sidecar"]).read_text(encoding="utf-8"))
            self.assertEqual(sidecar["boundary_hints"]["probabilities"], r["boundary_hints"]["probabilities"])


if __name__ == "__main__":
    unittest.main()
