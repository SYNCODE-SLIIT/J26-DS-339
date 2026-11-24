"""Starlette ASGI web application for Component 1 (Active Intent Confirmation + BPDR).

Serves the static HTML/CSS/ES-module UI (`web/`) and wraps `studio.pipeline`,
`studio.room`, `studio.trim`, `studio.question_engine`, `studio.span_repair`,
`studio.confirmation`, and `studio.retention`.
"""
import argparse
import hashlib
import json
import mimetypes
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import soundfile as sf
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response, StreamingResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles
import uvicorn

from ..studio import (
    boundary,
    confirmation,
    evidence as E,
    pipeline,
    question_engine as qe,
    retention,
    room,
    span_repair,
    speech,
    synth,
    trim,
)
from . import sessions

WORKSPACE = Path(__file__).resolve().parents[3]
WEB_DIR = WORKSPACE / "web"
EXAMPLES_DIR = WORKSPACE / "results" / "studio" / "examples"
RESULTS_DIR = WORKSPACE / "results"

CLAIMS_LEDGER = {
    "may_claim": [
        {
            "claim": "Do-no-harm acceptance and located re-record guidance",
            "evidence": "0 of 96 sung or hummed inputs falsely rejected; clipping/silence/speech feedback with exact time ranges.",
            "limit": "Room-check thresholds (-55/-40 dB) are preliminary and untested on real rooms.",
        },
        {
            "claim": "+0.145 note F1 over the PESTO note track on development data, mostly from the note-formation rule",
            "evidence": "studio_v1 0.631 [0.585, 0.676] vs pesto_notes_from_track 0.486 [0.431, 0.546]. Ablation: segment-median rule with heuristic extractor edges alone reaches 0.602 (+0.115); the learned boundary model adds +0.029 [-0.015, +0.077] on clean windows (CI includes 0).",
            "limit": "39 Vocadito training-role windows (14 singers); model selected on the same windows. About 80% of the gain is the note rule plus heuristic edges, not the learned model.",
        },
        {
            "claim": "Learned boundary model is more robust than the extractor's edges under timing jitter",
            "evidence": "Note F1 under boundary jitter: studio_v1 minus extractor edges +0.050 [+0.018, +0.084]. On clean windows the difference is not distinguishable from zero.",
            "limit": "Trained on HumTrans humming, checked on Vocadito singing; development evidence only. Do not claim an onset-F1 win on clean audio.",
        },
        {
            "claim": "Reference-free blind repair of small pitch errors is information-limited (honest negative result)",
            "evidence": "Blind detection AUROC 0.62-0.64; 4 model families tie around ~10% verified repairs; R2 weight-matched control (12/12 healthy fits) fails delivery gate in all arms (advance_to_delivery = false).",
            "limit": "Original BPDR repair hypothesis reported unchanged as an honest negative.",
        },
        {
            "claim": "Active Intent Confirmation (AIC) development simulation",
            "evidence": "Evaluated on 39 Vocadito windows (A1 oracle & A2 noisy user) and 498 v64 development records with leave-one-out selection, paired CIs, recall ceilings and a measured harm rate (aic_dev_evaluation/0.2). Expected-information-gain selection is NOT shown to beat lowest-confidence selection.",
            "limit": "Simulated users only; human pilot requires ethics approval. Downstream pipeline impact requires teammate adoption of sidecar/intent_map.",
        },
    ],
    "do_not_claim": [
        "That the Transformer repairs pitch.",
        "That BPDR beat the controls.",
        "That R1 works (untested / NO-GO on span-median shift).",
        "That the frame-length gain holds beyond 21 development files.",
        "That the component rescues any input.",
        "\"+110%\" (confounds frame-length effect 4096->2048 with learned model gain).",
        "\"0% harmful distortion, guaranteed\" (the earlier 0.0 was a constant; the v0.2 rate is measured against a simulated noisy listener only).",
        "That EIG question selection beats the simple lowest-confidence rule.",
        "That trust_curve is calibrated (its values are hand-set heuristics).",
        "That BPDR's custom loss terms caused the boundary gain (frame_only achieves equivalent accuracy).",
        "That AIC improves the live pipeline before Vishwa/Pasan adopt confirmed_boundary and trust_curve.",
        "Any human-user benefit before an ethics-approved human pilot.",
    ],
}


def _json_err(msg: str, status_code: int = 400) -> JSONResponse:
    return JSONResponse({"error": msg}, status_code=status_code)


def _sha256_file(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _waveform_envelope(x: np.ndarray, n_bins: int = 220) -> List[Dict[str, float]]:
    if len(x) == 0:
        return []
    n_bins = min(n_bins, len(x))
    edges = np.linspace(0, len(x), n_bins + 1, dtype=int)
    env = []
    for i in range(n_bins):
        seg = x[edges[i] : edges[i + 1]]
        if len(seg) == 0:
            env.append({"min": 0.0, "max": 0.0})
        else:
            env.append({"min": round(float(np.min(seg)), 4), "max": round(float(np.max(seg)), 4)})
    return env


def _file_range_response(request: Request, file_path: Path) -> Response:
    if not file_path.is_file():
        return _json_err("File not found", 404)
    total = file_path.stat().st_size
    ctype = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    range_header = request.headers.get("range")
    if range_header:
        m = re.fullmatch(r"bytes=(\d+)-(\d*)", range_header.strip())
        if not m:
            return _json_err("Unsupported byte range", 416)
        start = int(m.group(1))
        end = min(int(m.group(2)) if m.group(2) else total - 1, total - 1)
        if start > end or start >= total:
            return _json_err("Range outside file", 416)
        length = end - start + 1

        def _iter_range():
            with file_path.open("rb") as f:
                f.seek(start)
                rem = length
                while rem > 0:
                    chunk = f.read(min(65536, rem))
                    if not chunk:
                        break
                    rem -= len(chunk)
                    yield chunk

        headers = {
            "Content-Range": f"bytes {start}-{end}/{total}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(length),
            "Cache-Control": "no-store",
        }
        return StreamingResponse(_iter_range(), status_code=206, media_type=ctype, headers=headers)

    return FileResponse(file_path, media_type=ctype, headers={"Accept-Ranges": "bytes", "Cache-Control": "no-store"})


async def index_page(request: Request) -> Response:
    return FileResponse(WEB_DIR / "index.html")


async def research_page(request: Request) -> Response:
    return FileResponse(WEB_DIR / "research.html")


async def health(request: Request) -> JSONResponse:
    return JSONResponse({
        "status": "ok",
        "application": "bpdr_component1_aic",
        "boundary_checkpoint": str(boundary.DEFAULT_CHECKPOINT.name),
        "schema_handoff": "bpdr_studio_handoff/0.2",
    })


async def list_examples(request: Request) -> JSONResponse:
    descriptions = {
        "clean_hum.wav": "Clean sung/hummed phrase (expected: ACCEPT)",
        "hum_rain_0db.wav": "Hummed phrase with 0 dB rain noise (expected: ACCEPT / CLEAN)",
        "hum_ticks_10db.wav": "Hummed phrase with 10 dB background ticks",
        "hum_vacuum_0db.wav": "Hummed phrase with 0 dB fan/vacuum noise",
        "clipped.wav": "Intentionally clipped recording (expected: RERECORD_REQUIRED)",
        "speech.wav": "Spoken sentence (expected: RERECORD_REQUIRED + speech advisory)",
        "speech_noise.wav": "Noisy speech sample",
        "silence.wav": "Silent file (expected: TECHNICAL_FAILURE)",
    }
    items = []
    if EXAMPLES_DIR.is_dir():
        for p in sorted(EXAMPLES_DIR.glob("*.wav")):
            items.append({
                "name": p.name,
                "label": descriptions.get(p.name, p.stem.replace("_", " ").title()),
            })
    voc1 = WORKSPACE / "datasets" / "vocadito" / "Audio" / "vocadito_1.wav"
    if voc1.is_file():
        items.append({"name": "vocadito_1.wav", "label": "Vocadito real singing sample (vocadito_1)"})
    return JSONResponse({"examples": items})


async def api_create_session(request: Request) -> JSONResponse:
    try:
        body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
    except Exception:
        body = {}
    consent = bool(body.get("consent", True))
    if not consent:
        return _json_err("Privacy consent must be accepted before creating a session.", 400)
    data = sessions.create_session(consent=consent)
    return JSONResponse(data, status_code=201)


async def api_room_check(request: Request) -> JSONResponse:
    sid = request.path_params["session_id"]
    try:
        sess = sessions.load_session(sid)
    except FileNotFoundError:
        return _json_err("Session not found", 404)
    except ValueError as exc:
        return _json_err(str(exc), 400)

    sdir = sessions.session_path(sid)
    ctype = request.headers.get("content-type", "")
    room_wav = sdir / "room_check_raw.wav"

    if "multipart/form-data" in ctype:
        form = await request.form()
        up = form.get("file")
        if up is None:
            return _json_err("Missing 'file' upload for room check", 400)
        room_wav.write_bytes(await up.read())
        await up.close()
        try:
            x, _ = E.load_mono(room_wav, max_s=6.0)
        except ValueError:
            # Extremely quiet silence is valid for room check; read directly without silence rejection
            data, sr_in = sf.read(room_wav, dtype="float32", always_2d=True)
            x = data.mean(axis=1)
    else:
        body = await request.json()
        preset = body.get("preset")
        if preset == "skip":
            sess["room_check"] = {"verdict": "skipped", "usable_reference": False, "tip": "Room check skipped."}
            sessions.save_session(sid, sess)
            return JSONResponse(sess["room_check"])
        rng = np.random.default_rng(7)
        n = int(2.5 * E.SR)
        if preset == "noisy":
            x = (0.03 * rng.standard_normal(n)).astype(np.float32)
        elif preset == "too_loud":
            x = (0.35 * rng.standard_normal(n)).astype(np.float32)
        else:
            x = (1e-4 * rng.standard_normal(n)).astype(np.float32)
        sf.write(room_wav, x, E.SR, subtype="PCM_16")

    res = room.room_check(x, sr=E.SR)
    if res.get("usable_reference"):
        sf.write(sdir / "room_noise.wav", x, E.SR, subtype="PCM_16")
        res["noise_profile_saved"] = True
    else:
        res["noise_profile_saved"] = False
    sess["room_check"] = res
    sessions.save_session(sid, sess)
    return JSONResponse(res)


async def api_upload_audio(request: Request) -> JSONResponse:
    sid = request.path_params["session_id"]
    try:
        sess = sessions.load_session(sid)
    except FileNotFoundError:
        return _json_err("Session not found", 404)
    except ValueError as exc:
        return _json_err(str(exc), 400)

    sdir = sessions.session_path(sid)
    raw_path = sdir / "raw_input.wav"
    ctype = request.headers.get("content-type", "")

    if "multipart/form-data" in ctype:
        form = await request.form()
        up = form.get("file")
        if up is None:
            return _json_err("Missing audio file", 400)
        raw_path.write_bytes(await up.read())
        await up.close()
        source_label = getattr(up, "filename", "recording.wav") or "recording.wav"
    else:
        body = await request.json()
        ex_name = str(body.get("example_name", ""))
        cand = EXAMPLES_DIR / ex_name
        if not cand.is_file() and ex_name == "vocadito_1.wav":
            cand = WORKSPACE / "datasets" / "vocadito" / "Audio" / "vocadito_1.wav"
        if not cand.is_file():
            return _json_err(f"Unknown example audio: {ex_name}", 404)
        shutil.copy2(cand, raw_path)
        source_label = ex_name

    try:
        x_full, meta = E.load_mono(raw_path, max_s=None)
    except Exception as exc:
        sess["upload_error"] = str(exc)
        sess["source_label"] = source_label
        sess["status"] = "TECHNICAL_FAILURE"
        sess["verdict"] = {
            "status": "TECHNICAL_FAILURE",
            "reasons": [str(exc)],
            "guidance": ["Check microphone or choose a valid non-silent WAV file."],
            "accepted_audio": None,
        }
        sessions.save_session(sid, sess)
        return JSONResponse({
            "source_label": source_label,
            "technical_failure": True,
            "error": str(exc),
            "input_wav_url": f"/api/sessions/{sid}/files/raw_input.wav",
        })

    _, phrase_info = trim.select_best_phrase(x_full, sr=E.SR, max_duration_s=E.MAX_SECONDS)
    needs_trim = bool(phrase_info.get("phrase_selected") or phrase_info.get("trimmed_silence"))
    sess["source_label"] = source_label
    sess["upload_meta"] = meta
    sess["phrase_preview"] = phrase_info
    sess["upload_error"] = None
    sess["status"] = "AUDIO_READY"
    sessions.save_session(sid, sess)

    return JSONResponse({
        "source_label": source_label,
        "technical_failure": False,
        "duration_s": round(float(meta["source_duration_s"]), 2),
        "clipped_fraction": round(float(meta["clipped_fraction"]), 4),
        "clipped_spans_s": [[round(a, 2), round(b, 2)] for a, b in meta["clipped_spans_s"][:5]],
        "phrase_selection": phrase_info,
        "needs_trim_confirmation": needs_trim,
        "input_wav_url": f"/api/sessions/{sid}/files/raw_input.wav",
    })


async def api_analyze(request: Request) -> JSONResponse:
    sid = request.path_params["session_id"]
    try:
        sess = sessions.load_session(sid)
    except FileNotFoundError:
        return _json_err("Session not found", 404)
    except ValueError as exc:
        return _json_err(str(exc), 400)

    sdir = sessions.session_path(sid)
    raw_path = sdir / "raw_input.wav"
    if not raw_path.is_file():
        return _json_err("Upload or record audio before analyzing.", 400)

    try:
        body = await request.json()
    except Exception:
        body = {}

    intent = str(body.get("intent", "unsure"))
    strategy = str(body.get("strategy", "eig"))
    k = min(qe.MAX_K, max(0, int(body.get("k", qe.DEFAULT_K))))
    auto_trim = bool(body.get("auto_trim", True))
    time_range = body.get("time_range")

    if not sessions.acquire_job_lock(timeout_s=0.1):
        return _json_err("Another analysis job is currently running on this machine. Please wait a moment.", 409)

    job_id = uuid.uuid4().hex[:12]
    t_start = time.perf_counter()
    try:
        room_noise = None
        room_noise_path = sdir / "room_noise.wav"
        if room_noise_path.is_file():
            rn, _ = sf.read(room_noise_path, dtype="float32", always_2d=True)
            room_noise = rn.mean(axis=1)

        out_dir = sdir / "prepared"
        res = pipeline.prepare(
            raw_path,
            out_dir,
            intent=intent,
            with_pyin=True,
            room_noise=room_noise,
            auto_trim=auto_trim,
            time_range=time_range,
            with_boundary_hints=True,
        )

        status = res["status"]
        verdict: Dict[str, Any] = {
            "status": status,
            "reasons": res.get("reasons", []),
            "guidance": res.get("guidance", []),
            "advisories": res.get("advisories", []),
            "quality": res.get("quality"),
            "phrase_selection": res.get("phrase_selection"),
            "speech_detected": bool(res.get("speech_detected", False)),
            "accepted_audio": None,
            "accepted_sha256": None,
        }

        if status in ("ACCEPT_UNCHANGED", "ACCEPT_PREPARED"):
            prep_wav = Path(res["accepted_audio"])
            sidecar_data = json.loads(Path(res["sidecar"]).read_text(encoding="utf-8"))
            x_prep, _ = sf.read(prep_wav, dtype="float32", always_2d=True)
            x_mono = x_prep.mean(axis=1)
            dur = float(res["duration_s"])

            t_s = np.asarray(sidecar_data["t_s"], dtype=float)
            f0_hz = np.asarray([np.nan if v is None else float(v) for v in sidecar_data["f0_hz"]], dtype=float)
            conf = np.asarray(sidecar_data["confidence"], dtype=float)
            voiced = np.asarray(sidecar_data["voiced"], dtype=bool)
            trk = {"t": t_s, "f0": f0_hz, "voiced": voiced, "conf": conf}

            b_hints = sidecar_data.get("boundary_hints")
            span_diag = res.get("span_diagnostics") or (
                b_hints.get("span_diagnostics") if b_hints else boundary.analyze_boundary_spans(trk, {"peaks_s": []}, end_s=dur)
            )

            cands = qe.extract_ambiguity_candidates(
                trk,
                x=x_mono,
                sr=E.SR,
                boundary_hints=b_hints,
                span_diagnostics=span_diag,
                duration_s=dur,
            )
            cands = qe.calibrate_candidate_probabilities(cands)
            selected_qs = qe.select_questions(cands, k=k, max_k=qe.MAX_K, strategy=strategy)
            selected_qs = qe.render_question_audio_clips(selected_qs, x_mono, out_dir, sr=E.SR)
            for q in selected_qs:
                clips = q["clips"]
                q["clips"]["orig_url"] = f"/api/sessions/{sid}/files/prepared/{clips['orig_file']}"
                q["clips"]["opt_a_url"] = f"/api/sessions/{sid}/files/prepared/{clips['opt_a_file']}"
                q["clips"]["opt_b_url"] = f"/api/sessions/{sid}/files/prepared/{clips['opt_b_file']}"

            intent_map, trust_curve = qe.compute_intent_and_trust(
                span_diagnostics=span_diag,
                candidates=cands,
                answers_by_qid={},
                duration_s=dur,
            )

            # Copy initial prepared.wav -> accepted.wav (updated if user confirms repairs in Screen 7)
            accepted_wav = out_dir / "accepted.wav"
            shutil.copy2(prep_wav, accepted_wav)

            verdict.update({
                "duration_s": round(dur, 3),
                "accepted_audio": str(accepted_wav),
                "accepted_sha256": _sha256_file(accepted_wav),
                "waveform": _waveform_envelope(x_mono),
                "pitch_track": {
                    "t_s": sidecar_data["t_s"],
                    "f0_hz": sidecar_data["f0_hz"],
                    "voiced": sidecar_data["voiced"],
                    "confidence": sidecar_data["confidence"],
                },
                "boundary_hints": {
                    "peaks_s": b_hints.get("peaks_s", []) if b_hints else [],
                    "threshold": b_hints.get("threshold") if b_hints else None,
                    "model": b_hints.get("model") if b_hints else None,
                },
                "span_diagnostics": span_diag,
                "bpdr_multiview": res.get("bpdr_multiview") or sidecar_data.get("bpdr_multiview"),
                "audio_urls": {
                    "original": f"/api/sessions/{sid}/files/prepared/original_mono.wav",
                    "heard_before": f"/api/sessions/{sid}/files/prepared/heard_before.wav",
                    "heard_after": f"/api/sessions/{sid}/files/prepared/heard_after.wav",
                    "prepared": f"/api/sessions/{sid}/files/prepared/accepted.wav",
                },
            })
            sess["aic"] = {
                "strategy": strategy,
                "k": k,
                "policy_hash": qe.policy_hash(),
                "all_candidates": cands,
                "questions": selected_qs,
                "intent_map": intent_map,
                "trust_curve": trust_curve,
                "user_answers": {"strategy": strategy, "policy_hash": qe.policy_hash(), "answers": []},
                "centred_span_ids": [],
                "repairs": None,
            }

        elapsed_s = round(time.perf_counter() - t_start, 3)
        verdict["processing_elapsed_s"] = elapsed_s
        sess["status"] = status
        sess["verdict"] = verdict
        sess["job"] = {"job_id": job_id, "state": "complete", "step": "done", "progress": 100, "elapsed_s": elapsed_s}
        sessions.save_session(sid, sess)
        return JSONResponse({"job_id": job_id, "status": status, "elapsed_s": elapsed_s, "verdict": verdict})
    except Exception as exc:
        return _json_err(f"Analysis failed: {type(exc).__name__}: {exc}", 500)
    finally:
        sessions.release_job_lock()


async def api_get_verdict(request: Request) -> JSONResponse:
    sid = request.path_params["session_id"]
    try:
        sess = sessions.load_session(sid)
    except FileNotFoundError:
        return _json_err("Session not found", 404)
    except ValueError as exc:
        return _json_err(str(exc), 400)
    if "verdict" not in sess:
        return _json_err("Analysis has not been run yet", 400)
    return JSONResponse(sess["verdict"])


async def api_get_questions(request: Request) -> JSONResponse:
    sid = request.path_params["session_id"]
    try:
        sess = sessions.load_session(sid)
    except FileNotFoundError:
        return _json_err("Session not found", 404)
    except ValueError as exc:
        return _json_err(str(exc), 400)

    aic = sess.get("aic")
    if not aic:
        return _json_err("No clarification questions available for this session status", 400)

    strategy = request.query_params.get("strategy", aic.get("strategy", "eig"))
    k = min(qe.MAX_K, max(0, int(request.query_params.get("k", aic.get("k", qe.DEFAULT_K)))))
    if strategy != aic.get("strategy") or k != aic.get("k"):
        selected = qe.select_questions(aic["all_candidates"], k=k, max_k=qe.MAX_K, strategy=strategy)
        out_dir = sessions.session_path(sid) / "prepared"
        x_prep, _ = sf.read(out_dir / "prepared.wav", dtype="float32", always_2d=True)
        selected = qe.render_question_audio_clips(selected, x_prep.mean(axis=1), out_dir, sr=E.SR)
        for q in selected:
            clips = q["clips"]
            q["clips"]["orig_url"] = f"/api/sessions/{sid}/files/prepared/{clips['orig_file']}"
            q["clips"]["opt_a_url"] = f"/api/sessions/{sid}/files/prepared/{clips['opt_a_file']}"
            q["clips"]["opt_b_url"] = f"/api/sessions/{sid}/files/prepared/{clips['opt_b_file']}"
        aic["strategy"] = strategy
        aic["k"] = k
        aic["questions"] = selected
        sessions.save_session(sid, sess)

    return JSONResponse({
        "strategy": aic["strategy"],
        "k": aic["k"],
        "policy_hash": aic["policy_hash"],
        "questions": aic["questions"],
        "intent_map": aic["intent_map"],
        "trust_curve": aic["trust_curve"],
    })


async def api_submit_answers(request: Request) -> JSONResponse:
    sid = request.path_params["session_id"]
    try:
        sess = sessions.load_session(sid)
    except FileNotFoundError:
        return _json_err("Session not found", 404)
    except ValueError as exc:
        return _json_err(str(exc), 400)

    aic = sess.get("aic")
    verdict = sess.get("verdict")
    if not aic or not verdict:
        return _json_err("Session is not ready for clarification answers", 400)

    body = await request.json()
    confirmed_as_heard = bool(body.get("confirmed_as_heard", False))
    answers = body.get("answers", [])
    out_dir = sessions.session_path(sid) / "prepared"

    if confirmed_as_heard:
        confirmation.record_confirmation(out_dir, confirmed=True, notes_version="after_fasttnet_boundary_hints")
        # Mark all spans confirmed as sung/heard
        for row in aic["intent_map"]:
            if row["state"] == "unresolved":
                row["state"] = "confirmed_as_sung"
                row["span_trust"] = 0.92
        aic["trust_curve"]["trust"] = [round(max(0.88, float(v)), 3) for v in aic["trust_curve"]["trust"]]
        aic["trust_curve"]["mean_trust"] = round(float(np.mean(aic["trust_curve"]["trust"])), 3)
        aic["user_answers"] = {
            "strategy": aic["strategy"],
            "policy_hash": aic["policy_hash"],
            "confirmed_as_heard": True,
            "answers": [],
        }
        aic["centred_span_ids"] = []
        sessions.save_session(sid, sess)
        return JSONResponse({
            "intent_map": aic["intent_map"],
            "trust_curve": aic["trust_curve"],
            "user_answers": aic["user_answers"],
            "centred_span_ids": [],
            "needs_repair_review": False,
        })

    applied = qe.apply_user_answers(
        candidates=aic["all_candidates"],
        span_diagnostics=verdict["span_diagnostics"],
        answers=answers,
        duration_s=float(verdict["duration_s"]),
        strategy=aic["strategy"],
    )
    aic["intent_map"] = applied["intent_map"]
    aic["trust_curve"] = applied["trust_curve"]
    aic["user_answers"] = applied["user_answers"]
    aic["centred_span_ids"] = applied["centred_span_ids"]
    aic["confirmed_boundaries"] = applied["confirmed_boundaries"]

    confirmation.record_confirmation(
        out_dir,
        confirmed=not applied["needs_repair_review"],
        notes_version="after_fasttnet_boundary_hints",
        user_comment="Clarified via Active Intent Confirmation",
    )
    sessions.save_session(sid, sess)

    return JSONResponse({
        "intent_map": aic["intent_map"],
        "trust_curve": aic["trust_curve"],
        "user_answers": aic["user_answers"],
        "centred_span_ids": aic["centred_span_ids"],
        "needs_repair_review": applied["needs_repair_review"],
    })


async def api_repair(request: Request) -> JSONResponse:
    sid = request.path_params["session_id"]
    try:
        sess = sessions.load_session(sid)
    except FileNotFoundError:
        return _json_err("Session not found", 404)
    except ValueError as exc:
        return _json_err(str(exc), 400)

    aic = sess.get("aic")
    verdict = sess.get("verdict")
    if not aic or not verdict:
        return _json_err("Session is not ready for repair", 400)

    try:
        body = await request.json()
    except Exception:
        body = {}

    approved_ids = body.get("approved_span_ids")
    if approved_ids is None:
        approved_ids = aic.get("centred_span_ids", [])
    approved_ids = [int(i) for i in approved_ids]

    out_dir = sessions.session_path(sid) / "prepared"
    prep_wav = out_dir / "prepared.wav"
    accepted_wav = out_dir / "accepted.wav"
    x_prep, _ = sf.read(prep_wav, dtype="float32", always_2d=True)
    x_mono = x_prep.mean(axis=1)

    pt = verdict["pitch_track"]
    trk_before = {
        "t": np.asarray(pt["t_s"], dtype=float),
        "f0": np.asarray([np.nan if v is None else float(v) for v in pt["f0_hz"]], dtype=float),
        "voiced": np.asarray(pt["voiced"], dtype=bool),
        "conf": np.asarray(pt["confidence"], dtype=float),
    }

    rep = span_repair.verify_and_apply_span_repairs(
        x=x_mono,
        trk_before=trk_before,
        span_diagnostics=verdict["span_diagnostics"],
        approved_span_ids=approved_ids,
        sr=E.SR,
        max_shift_cents=35.0,
    )

    if rep["applied_count"] > 0:
        sf.write(accepted_wav, rep["audio"], E.SR, subtype="PCM_24")
        sess["status"] = "ACCEPT_PREPARED"
        verdict["status"] = "ACCEPT_PREPARED"
    else:
        shutil.copy2(prep_wav, accepted_wav)

    sha_val = _sha256_file(accepted_wav)
    verdict["accepted_audio"] = str(accepted_wav)
    verdict["accepted_sha256"] = sha_val

    # Save short per-span A/B clips for Screen 7 review
    edits_out = []
    for ed in rep.get("edits", []):
        sid_num = ed["span_id"]
        t0 = max(0.0, float(ed["start_s"]) - 0.30)
        t1 = min(len(x_mono) / E.SR, float(ed["end_s"]) + 0.30)
        i0, i1 = int(t0 * E.SR), int(t1 * E.SR)
        b_name = f"span_{sid_num}_before.wav"
        a_name = f"span_{sid_num}_after.wav"
        sf.write(out_dir / b_name, E.normalize(x_mono[i0:i1]), E.SR, subtype="PCM_16")
        sf.write(out_dir / a_name, E.normalize(rep["audio"][i0:i1]), E.SR, subtype="PCM_16")
        edits_out.append({
            **ed,
            "before_url": f"/api/sessions/{sid}/files/prepared/{b_name}",
            "after_url": f"/api/sessions/{sid}/files/prepared/{a_name}",
        })

    repair_summary = {
        "repair_status": rep["status"],
        "applied_count": rep["applied_count"],
        "rolled_back_count": rep["rolled_back_count"],
        "edits": edits_out,
        "accepted_sha256": sha_val,
    }
    aic["repairs"] = repair_summary
    sessions.save_session(sid, sess)
    return JSONResponse(repair_summary)


async def api_handoff(request: Request) -> JSONResponse:
    sid = request.path_params["session_id"]
    try:
        sess = sessions.load_session(sid)
    except FileNotFoundError:
        return _json_err("Session not found", 404)
    except ValueError as exc:
        return _json_err(str(exc), 400)

    verdict = sess.get("verdict")
    if not verdict:
        return _json_err("Analysis has not been run yet", 400)

    status = verdict["status"]
    if status in ("RERECORD_REQUIRED", "TECHNICAL_FAILURE"):
        return JSONResponse({
            "schema": "bpdr_studio_handoff/0.2",
            "session_id": sid,
            "status": status,
            "accepted_audio": None,
            "accepted_sha256": None,
            "reasons": verdict.get("reasons", []),
            "guidance": verdict.get("guidance", []),
            "sidecar": None,
            "intent_map": {"status_tag": "proposal", "spans": []},
            "trust_curve": {"status_tag": "proposal", "times_s": [], "trust": []},
            "user_answers": None,
            "processing_elapsed_s": verdict.get("processing_elapsed_s"),
        })

    out_dir = sessions.session_path(sid) / "prepared"
    accepted_wav = out_dir / "accepted.wav"
    sidecar_path = out_dir / "sidecar.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    aic = sess.get("aic", {})

    # Update sidecar with v0.2 AIC fields (marked proposal where teammate adoption is needed)
    sidecar["schema"] = "bpdr_studio_handoff/0.2"
    sidecar["intent_map"] = {
        "status_tag": "proposal",
        "adoption_note": "Vishwa adopts confirmed_boundary as a segmentation constraint.",
        "spans": aic.get("intent_map", []),
    }
    sidecar["trust_curve"] = aic.get("trust_curve", {"status_tag": "proposal", "times_s": [], "trust": []})
    sidecar["user_answers"] = aic.get("user_answers")
    sidecar["repairs"] = aic.get("repairs")
    sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")

    sha_val = _sha256_file(accepted_wav)
    return JSONResponse({
        "schema": "bpdr_studio_handoff/0.2",
        "session_id": sid,
        "status": status,
        "accepted_audio": str(accepted_wav),
        "accepted_audio_url": f"/api/sessions/{sid}/files/prepared/accepted.wav",
        "accepted_sha256": sha_val,
        "sidecar_url": f"/api/sessions/{sid}/files/prepared/sidecar.json",
        "processing_elapsed_s": verdict.get("processing_elapsed_s"),
        "intent_map": sidecar["intent_map"],
        "trust_curve": sidecar["trust_curve"],
        "user_answers": sidecar["user_answers"],
        "repairs": aic.get("repairs"),
        "downstream_contracts": [
            {"owner": "Vishwa (Melody Understanding)", "consumes": "accepted.wav + sidecar.json + confirmed_boundary", "status": "proposal (needs Vishwa adoption for sidecar/constraints)"},
            {"owner": "Pasan (Chord Planning)", "consumes": "notes + beat grid + optional trust_curve", "status": "proposal (needs Pasan decision on trust_curve)"},
            {"owner": "Rahul (Chord Refinement)", "consumes": "melody + planned chords", "status": "unchanged"},
        ],
        "sidecar": sidecar,
    })


async def api_speech_preview(request: Request) -> JSONResponse:
    sid = request.path_params["session_id"]
    try:
        sess = sessions.load_session(sid)
    except FileNotFoundError:
        return _json_err("Session not found", 404)
    except ValueError as exc:
        return _json_err(str(exc), 400)

    sdir = sessions.session_path(sid)
    raw_path = sdir / "raw_input.wav"
    if not raw_path.is_file():
        return _json_err("Audio not uploaded", 400)

    x, _ = E.load_mono(raw_path, max_s=E.MAX_SECONDS)
    trk = E.pesto_track(E.normalize(x))
    m = speech.speech_to_melody(x, trk, tonic_pc=0, scale="major", bpm=100)
    wav = synth.render_notes(m["notes"], len(x) / E.SR, sr=E.SR)
    out_wav = sdir / "speech_melody_preview.wav"
    sf.write(out_wav, wav, E.SR, subtype="PCM_16")
    return JSONResponse({
        "outside_research_claim": True,
        "note": "Creative speech-to-melody preview only (explicitly outside the Component 1 research claim).",
        "notes": m["notes"],
        "wav_url": f"/api/sessions/{sid}/files/speech_melody_preview.wav",
    })


async def api_delete_session(request: Request) -> JSONResponse:
    sid = request.path_params["session_id"]
    try:
        removed = sessions.delete_session(sid)
    except ValueError as exc:
        return _json_err(str(exc), 400)
    return JSONResponse({"deleted": removed > 0, "removed_folders": removed, "session_id": sid})


async def api_session_file(request: Request) -> Response:
    sid = request.path_params["session_id"]
    rel = request.path_params["filepath"]
    try:
        sdir = sessions.session_path(sid).resolve()
    except ValueError as exc:
        return _json_err(str(exc), 400)
    target = (sdir / rel).resolve()
    if not target.is_relative_to(sdir):
        return _json_err("Forbidden path", 403)
    return _file_range_response(request, target)


async def api_research_summary(request: Request) -> JSONResponse:
    files = {
        "bpdr_4view_benchmark": RESULTS_DIR / "studio" / "r1" / "bpdr_repair_benchmark.json",
        "aic_dev": RESULTS_DIR / "studio" / "aic" / "aic_dev_results.json",
        "aic_declaration": RESULTS_DIR / "studio" / "aic" / "declaration.json",
        "vocadito_benchmark": RESULTS_DIR / "studio" / "r1" / "studio_vocadito_benchmark.json",
        "vocadito_benchmark_noisy": RESULTS_DIR / "studio" / "r1" / "studio_vocadito_benchmark_noisy.json",
        "frame_control": RESULTS_DIR / "studio" / "evidence" / "frame_control.json",
        "r2_control": RESULTS_DIR / "transformer_usability" / "2026-11-17_r2_weight_matched_control" / "results.json",
        "v5_ladder": RESULTS_DIR / "studio" / "r1" / "v5" / "ladder_results.json",
    }

    provenance = []
    loaded: Dict[str, Any] = {}
    for key, p in files.items():
        if p.is_file():
            loaded[key] = json.loads(p.read_text(encoding="utf-8"))
            provenance.append({
                "key": key,
                "path": str(p.relative_to(WORKSPACE)).replace("\\", "/"),
                "sha256": _sha256_file(p),
            })

    # Extract concise R2 summary directly from saved results.json
    r2_summary = None
    if "r2_control" in loaded:
        r2 = loaded["r2_control"]
        r2_summary = {
            "fit_health": r2.get("fit_health"),
            "advance_to_delivery": r2.get("advance_to_delivery"),
            "production_eligible": r2.get("production_eligible"),
            "declaration_sha256": r2.get("declaration_sha256"),
            "arms": {},
        }
        for arm_name, arm_data in r2.get("arms", {}).items():
            split_raw = arm_data.get("split", {}).get("raw", {}).get("groups", [])
            pos_vals = [g["positive_mae_cents"] for g in split_raw if g.get("positive_mae_cents") is not None]
            orig_vals = [g["original_mae_cents"] for g in split_raw if g.get("original_mae_cents") is not None]
            r2_summary["arms"][arm_name] = {
                "raw_positive_mae_cents": round(float(np.mean(pos_vals)), 3) if pos_vals else None,
                "raw_original_false_corr_cents": round(float(np.mean(orig_vals)), 3) if orig_vals else None,
            }

    voc_summary = loaded.get("vocadito_benchmark", {}).get("summary", {})

    return JSONResponse({
        "bpdr_4view_benchmark": loaded.get("bpdr_4view_benchmark"),
        "aic_dev": loaded.get("aic_dev"),
        "vocadito_benchmark_summary": voc_summary,
        "r2_control_summary": r2_summary,
        "claims_ledger": CLAIMS_LEDGER,
        "provenance": provenance,
    })


routes = [
    Route("/", index_page),
    Route("/research", research_page),
    Route("/api/health", health),
    Route("/api/examples", list_examples),
    Route("/api/sessions", api_create_session, methods=["POST"]),
    Route("/api/sessions/{session_id}/room", api_room_check, methods=["POST"]),
    Route("/api/sessions/{session_id}/audio", api_upload_audio, methods=["POST"]),
    Route("/api/sessions/{session_id}/analyze", api_analyze, methods=["POST"]),
    Route("/api/sessions/{session_id}/verdict", api_get_verdict),
    Route("/api/sessions/{session_id}/questions", api_get_questions),
    Route("/api/sessions/{session_id}/answers", api_submit_answers, methods=["POST"]),
    Route("/api/sessions/{session_id}/repair", api_repair, methods=["POST"]),
    Route("/api/sessions/{session_id}/handoff", api_handoff),
    Route("/api/sessions/{session_id}/speech-preview", api_speech_preview, methods=["POST"]),
    Route("/api/sessions/{session_id}", api_delete_session, methods=["DELETE"]),
    Route("/api/sessions/{session_id}/files/{filepath:path}", api_session_file),
    Route("/api/research/summary", api_research_summary),
    Mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static"),
]

app = Starlette(routes=routes)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Component 1 Web Service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8505)
    args = parser.parse_args()
    WEB_DIR.mkdir(parents=True, exist_ok=True)
    retention.purge_expired(sessions.SESSIONS_DIR, max_age_hours=retention.RETENTION_HOURS)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
