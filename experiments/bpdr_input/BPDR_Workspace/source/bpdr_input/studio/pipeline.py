"""Vocal Rescue Studio pipeline: validate -> evidence -> candidates -> do-no-harm choice -> hand-off.

Public statuses follow the proposal: ACCEPT_UNCHANGED, ACCEPT_PREPARED, RERECORD_REQUIRED,
TECHNICAL_FAILURE. Rejected statuses never carry an accepted-audio path. Notes are NOT owned
here: the note lists below exist only to show what the shared extractor would do with the audio.
"""
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import soundfile as sf

from . import boundary, clean, confirmation, consumer, evidence as E, speech, synth, trim

SCHEMA = "bpdr_studio_handoff/0.1"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _unvoiced_spans(trk, min_s=1.5):
    t, v = trk["t"], trk["voiced"]
    spans, i = [], 0
    while i < len(v):
        if not v[i]:
            j = i
            while j < len(v) and not v[j]:
                j += 1
            if t[min(j, len(t) - 1)] - t[i] >= min_s:
                spans.append((float(t[i]), float(t[min(j, len(t) - 1)])))
            i = j
        else:
            i += 1
    return spans


def _tuning_center_cents(trk):
    v = trk["voiced"] & (trk["conf"] > 0.5)
    if v.sum() < 20:
        return None
    res = 100 * ((12 * np.log2(trk["f0"][v] / 440.0)) % 1.0)
    ang = 2 * np.pi * np.where(res >= 50, res - 100, res) / 100
    return float(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean()) * 100 / (2 * np.pi))


def prepare(path, outdir, intent="unsure", with_pyin=True, clip_limit=0.02, min_voiced=0.10, room_noise=None, auto_trim=True, time_range=None, with_boundary_hints=False, boundary_checkpoint=None):
    """Run the studio on one recording. Returns a JSON-serialisable result dict; files go to outdir."""
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    result = {"schema": SCHEMA, "source": str(path), "source_sha256": sha256(path), "status": None,
              "reasons": [], "guidance": [], "accepted_audio": None, "advisories": [], "phrase_selection": None}
    try:
        if time_range is not None or auto_trim:
            x_full, meta = E.load_mono(path, max_s=None)
            if time_range is not None:
                t0, t1 = float(time_range[0]), float(time_range[1])
                s0, s1 = max(0, int(t0 * E.SR)), min(len(x_full), int(t1 * E.SR))
                x = x_full[s0:s1]
                result["phrase_selection"] = {
                    "original_duration_s": meta["source_duration_s"],
                    "start_s": t0, "end_s": t1, "duration_s": len(x) / E.SR,
                    "phrase_selected": True, "trimmed_silence": False, "user_specified": True
                }
                result["advisories"].append(f"Custom phrase selection analysed ({t0:.1f}–{t1:.1f} s).")
            else:
                x, phrase_info = trim.select_best_phrase(x_full, sr=E.SR, max_duration_s=E.MAX_SECONDS)
                result["phrase_selection"] = phrase_info
                if phrase_info.get("phrase_selected"):
                    result["advisories"].append(
                        f"Recording was {phrase_info['original_duration_s']:.1f} s (>20 s limit). "
                        f"Selected best {phrase_info['duration_s']:.1f} s phrase ({phrase_info['start_s']:.1f}–{phrase_info['end_s']:.1f} s)."
                    )
                elif phrase_info.get("trimmed_silence"):
                    result["advisories"].append(
                        f"Trimmed {phrase_info['trimmed_leading_s']:.1f} s leading and {phrase_info['trimmed_trailing_s']:.1f} s trailing silence."
                    )
        else:
            x, meta = E.load_mono(path, max_s=E.MAX_SECONDS)
            if meta["truncated_to_max"]:
                result["advisories"].append("Only the first 20 s were analysed (supported maximum).")
    except Exception as exc:                                     # decoding/silence problems are technical, not musical
        result.update(status="TECHNICAL_FAILURE", reasons=[str(exc)])
        return result
    result["input"] = meta
    if len(x) / E.SR < E.MIN_SECONDS:
        result.update(status="RERECORD_REQUIRED", reasons=["Recording is shorter than 0.25 s"],
                      guidance=["Record at least a few seconds of one voice."])
        return result

    xn = E.normalize(x)                                           # PESTO voicing is level-sensitive (measured): evidence is taken at a standard level
    trk_o = E.pesto_track(xn)                                     # frozen pretrained pitch evidence
    q = E.quality_report(x, trk_o)
    q["tuning_center_cents"] = _tuning_center_cents(trk_o)
    q["speech"] = speech.speech_likeness(trk_o)
    q["clipped_spans_s"], q["clipped_fraction"] = meta["clipped_spans_s"], meta["clipped_fraction"]   # from decoded samples
    q["normalization_gain"] = E.normalization_gain(x)
    result["quality"] = q

    if q["clipped_fraction"] > clip_limit:
        result.update(status="RERECORD_REQUIRED",
                      reasons=[f"{100*q['clipped_fraction']:.1f}% of samples are clipped; clipping is not repaired"],
                      guidance=[f"Clipping at {a:.1f}-{b:.1f} s: move back from the microphone or lower input gain, then record again."
                                for a, b in q["clipped_spans_s"][:5]])
        return result
    if q["speech"]["looks_like_speech"] and intent != "sung":
        result.update(status="RERECORD_REQUIRED",
                      reasons=["Sounds like speech: pitch rarely holds steady, so there is no melody to prepare."],
                      guidance=["Hum or sing the idea with held notes, or try the optional speech-to-melody mode."])
        result["advisories"].append("speech_to_melody_available")
        result["speech_detected"] = True
        return result

    cleaned = clean.spectral_gate(x, noise=room_noise)
    result["noise_profile_policy"] = "room_check_voice_free_clip" if room_noise is not None else "quietest_frames_of_recording"
    trk_c = E.pesto_track(cleaned)
    use_clean, selector = clean.choose(trk_o, trk_c)
    prepared = cleaned if use_clean else xn
    trk_p = trk_c if use_clean else trk_o
    result["selector"] = selector
    result["cleaned_candidate_used"] = use_clean

    if trk_p["voiced"].mean() < min_voiced:
        gaps = _unvoiced_spans(trk_p)
        result.update(status="RERECORD_REQUIRED",
                      reasons=[f"Only {100*trk_p['voiced'].mean():.0f}% of the recording has reliable pitch evidence"],
                      guidance=[f"No reliable pitch at {a:.1f}-{b:.1f} s (noise, silence or unvoiced sound). Sing or hum louder and closer."
                                for a, b in gaps[:5]] or ["Sing or hum louder, closer to the microphone, in a quieter place."])
        return result

    dur = len(x) / E.SR
    accepted = out / "prepared.wav"
    sf.write(accepted, prepared, E.SR, subtype="PCM_24")
    time_ref = "seconds from recording start (unchanged by preparation)"
    if result.get("phrase_selection") and result["phrase_selection"].get("start_s", 0) > 0:
        time_ref = f"seconds from recording start (offset +{result['phrase_selection']['start_s']:.2f} s relative to original file)"
    sidecar = {"schema": SCHEMA, "time_reference": time_ref,
               "sample_rate": E.SR, "duration_s": dur, "tracker": "pesto " + E.PESTO_MODEL,
               "voicing_threshold": trk_p["threshold"], "frame_hop_s": float(np.median(np.diff(trk_p["t"]))),
               "t_s": np.round(trk_p["t"], 4).tolist(),
               "f0_hz": [None if not np.isfinite(f) else round(float(f), 3) for f in trk_p["f0"]],
               "confidence": np.round(trk_p["conf"], 3).tolist(), "voiced": trk_p["voiced"].astype(int).tolist(),
               "quality": q, "noise_profile_policy": result["noise_profile_policy"], "tuning_center_cents": q["tuning_center_cents"], "normalization_gain": q["normalization_gain"],
               "phrase_selection": result.get("phrase_selection"),
               "ownership_note": "Pitch evidence only. Notes, key, tempo and chords belong to the downstream components."}
    if with_boundary_hints:
        try:
            b_hints = boundary.extract_boundary_hints(trk_p, x=prepared, sr=E.SR, checkpoint_path=boundary_checkpoint)
            span_diag = boundary.analyze_boundary_spans(
                trk_p, b_hints, end_s=dur, tuning_center_cents=q["tuning_center_cents"], sr=E.SR
            )
            b_hints["span_diagnostics"] = span_diag
            sidecar["boundary_hints"] = b_hints
            result["boundary_hints"] = b_hints
            result["span_diagnostics"] = span_diag
            try:
                from . import repair_transformer
                mv_bpdr = repair_transformer.predict_multiview_bpdr(
                    trk_p, x=prepared, sr=E.SR, boundary_hints=b_hints, span_diagnostics=span_diag
                )
                sidecar["bpdr_multiview"] = mv_bpdr
                result["bpdr_multiview"] = mv_bpdr
            except Exception as mv_exc:
                result["advisories"].append(f"BPDR multi-view analysis skipped: {mv_exc}")
        except Exception as exc:
            result["advisories"].append(f"Boundary hints could not be extracted: {exc}")
    (out / "sidecar.json").write_text(json.dumps(sidecar), encoding="utf-8")

    notes = {"after_prepared_pesto": E.notes_from_track(trk_p, dur),              # PROPOSED: needs the extractor to use the sidecar
             "evidence_only_original_pesto": E.notes_from_track(trk_o, dur)}
    if result.get("boundary_hints"):
        notes["after_fasttnet_boundary_hints"] = boundary.notes_from_track_with_boundary_hints(
            trk_p, result["boundary_hints"], end_s=dur, sr=E.SR
        )
    if with_pyin:
        before = consumer.shared_extractor_notes(path)                            # what the shared extractor does today
        if before is None:
            before = E.notes_from_track(E.pyin_track(x), dur)
        notes["before_current_extractor_pyin"] = before
        notes["shared_extractor_on_prepared"] = consumer.shared_extractor_notes(accepted)
    after_key = "after_fasttnet_boundary_hints" if "after_fasttnet_boundary_hints" in notes else "after_prepared_pesto"
    for key, wavname in (
        ("before_current_extractor_pyin", "heard_before.wav"),
        (after_key, "heard_after.wav"),
        ("after_fasttnet_boundary_hints", "heard_after_fasttnet.wav"),
    ):
        if key in notes:
            sf.write(out / wavname, synth.render_notes(notes[key], dur), E.SR, subtype="PCM_16")
    sf.write(out / "original_mono.wav", xn, E.SR, subtype="PCM_24")
    sf.write(out / "cleaned_preview_listening_only.wav", cleaned, E.SR, subtype="PCM_24")   # NOT handed on unless chosen

    user_conf = confirmation.load_confirmation(out)
    result.update(status="ACCEPT_PREPARED" if use_clean else "ACCEPT_UNCHANGED", accepted_audio=str(accepted),
                  accepted_sha256=sha256(accepted), sidecar=str(out / "sidecar.json"), notes=notes, duration_s=dur,
                  user_confirmation=user_conf,
                  reasons=["Cleaned candidate chosen: pitch evidence improved and pitch/voicing were preserved."] if use_clean
                  else ["Original kept: no candidate beat it without risk to pitch or voicing (format conversion only)."])
    return result


def evaluate_and_repair_vocal(
    user_audio_path: str,
    ref_audio_or_notes: Optional[Any] = None,
    outdir: Optional[str] = None,
    max_shift_cents: float = 120.0,
    boundary_checkpoint: Optional[str] = None,
    repair_checkpoint: Optional[str] = None,
) -> Dict[str, Any]:
    """Complete paired/solo vocal evaluation and transformer-guided minimal repair pipeline.
    
    1. Evaluates user singing (amateur tool/karaoke) against reference song or solo cues.
    2. Runs BPDRRepairTransformer with boundary conditioning to fix flat/sharp intonation & drift.
    3. Verifies repaired audio post-render with PESTO tracking (Do-No-Harm contract).
    4. Produces accepted audio WAV + comprehensive provenance sidecar for downstream components.
    """
    from . import evaluator, span_repair

    user_x, meta = E.load_mono(user_audio_path, max_s=E.MAX_SECONDS)
    dur = len(user_x) / E.SR
    
    # Run transformer repair pipeline
    repair_res = span_repair.repair_singing_with_transformer(
        user_audio_or_path=user_x,
        ref_audio_or_notes=ref_audio_or_notes,
        sr=E.SR,
        checkpoint_path=repair_checkpoint,
        max_shift_cents=max_shift_cents,
        boundary_checkpoint=boundary_checkpoint,
    )

    repaired_wav = repair_res['repaired_audio']
    status = repair_res['repaired_status']
    
    # Save files if outdir provided
    accepted_path = None
    sidecar_path = None
    if outdir is not None:
        out_p = Path(outdir)
        out_p.mkdir(parents=True, exist_ok=True)
        accepted_path = str(out_p / "prepared.wav")
        sf.write(accepted_path, repaired_wav, E.SR, subtype="PCM_24")
        
        # Save sidecar
        trk_final = E.pesto_track(E.normalize(repaired_wav))
        sidecar = {
            "schema": SCHEMA,
            "pipeline_mode": "paired_evaluator_transformer_repair" if ref_audio_or_notes is not None else "solo_evaluator_repair",
            "source_user_audio": str(user_audio_path),
            "status": status,
            "duration_s": dur,
            "scorecard": repair_res['eval_after']['scorecard'],
            "applied_repairs_count": repair_res['applied_count'],
            "rolled_back_count": repair_res['rolled_back_count'],
            "verified_edits": repair_res['verified_edits'],
            "cents_mae_before": repair_res['cents_mae_before'],
            "cents_mae_after": repair_res['cents_mae_after'],
            "summary": repair_res['eval_after']['summary'],
            "ownership_note": "Audio validation, pitch repair, and provenance owned by Component 1 (BPDR). Downstream owns note transcription, tempo, and harmony."
        }
        sidecar_path = str(out_p / "sidecar.json")
        with open(sidecar_path, 'w', encoding='utf-8') as f:
            json.dump(sidecar, f, indent=2)

    return {
        "status": status,
        "accepted_audio": accepted_path,
        "sidecar": sidecar_path,
        "repair_result": repair_res,
        "scorecard": repair_res['eval_after']['scorecard'],
        "errors": repair_res['eval_before']['errors'],
        "duration_s": dur,
    }

