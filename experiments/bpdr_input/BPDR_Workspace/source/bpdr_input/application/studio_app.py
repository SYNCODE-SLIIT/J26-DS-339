"""Vocal Rescue Studio: the panel demonstration (separate from the frozen review demo)."""
import json
import uuid
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import streamlit as st

from bpdr_input.studio import chords, confirmation, evidence as E, examples, pipeline, plots, retention, room, scoring, speech, synth, trim

WORKSPACE = Path(__file__).resolve().parents[3]
SESSIONS = WORKSPACE / "results/studio/sessions"
EVIDENCE = WORKSPACE / "results/studio/evidence/pesto_vs_pyin.json"
FRAME_CONTROL = WORKSPACE / "results/studio/evidence/frame_control.json"
CLEANUP_DEV = WORKSPACE / "results/transformer_usability/2026-11-08_primary_cleanup_development/results.json"
KEYS = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
GENRES = ["pop", "jazz", "soul", "blues", "rock", "folk"]   # all present in the chord refiner's vocabulary

BANNER = {"ACCEPT_UNCHANGED": ("success", "Accepted unchanged: this recording is already usable."),
          "ACCEPT_PREPARED": ("success", "Accepted prepared: a cleaned version was chosen and passed the safety checks."),
          "RERECORD_REQUIRED": ("warning", "Please record again: this one cannot be prepared safely."),
          "TECHNICAL_FAILURE": ("error", "Technical failure: the file could not be processed.")}


def _audio(path):
    st.audio(str(path))


def _score_block(res, ref):
    if not ref:
        return
    try:
        notes = res["notes"]
        rows = [("Shared extractor, original audio (today)", notes.get("before_current_extractor_pyin")),
                ("Shared extractor on the prepared audio (real consumer)", notes.get("shared_extractor_on_prepared")),
                ("If the extractor used BPDR's pitch evidence (proposed)", notes["after_prepared_pesto"])]
        base = scoring.note_f1(ref, rows[0][1] or [], max_s=8)
        cols = st.columns(3)
        for col, (name, ns) in zip(cols, rows):
            f = scoring.note_f1(ref, ns or [], max_s=8)
            col.metric(name, f"F1 {f:.2f}", None if name == rows[0][0] else f"{f - base:+.2f} vs today", delta_color="normal")
        st.caption("Note F1 against two human annotators (onset 50 ms, pitch 50 cents). One recording is an illustration, "
                   "not proof: the 21-file study is in the Research evidence tab. Noise here is a controlled digital mix.")
    except Exception as exc:
        st.caption("Reference scoring unavailable: " + str(exc)[:120])


def _harmony_block(res, key_prefix):
    st.subheader("Hear what the next components make of it")
    st.caption("Demo bridge: the notes shown come from the same nearest-semitone rule as the shared extractor; the real notes, key "
               "and chords belong to the melody and harmony components. Key is chosen here, not guessed.")
    c1, c2, c3 = st.columns(3)
    key = c1.selectbox("Key (major)", KEYS, key=key_prefix + "k")
    bpm = c2.slider("Tempo (BPM)", 60, 140, 100, key=key_prefix + "b")
    genre = c3.selectbox("Genre", GENRES, key=key_prefix + "g")
    if st.button("Harmonize with the team's models", key=key_prefix + "go"):
        try:
            notes = res["notes"]["after_prepared_pesto"]
            if len(notes) < 3:
                raise ValueError("Too few notes to harmonize.")
            # re-time the extracted notes onto the chosen tempo grid is NOT done: harmony follows source time.
            h = chords.harmonize(notes, bpm, KEYS.index(key), genre)
            dur = min(h["duration_s"], 24.0)
            mel = synth.render_notes([n for n in h["tiled_notes"] if n["onset_s"] < dur], dur)
            ch = synth.render_chords([c for c in h["chords"] if c["start_s"] < dur], dur)
            out = Path(res["accepted_audio"]).parent / "harmony_demo.wav"
            sf.write(out, synth.mix(mel, ch), E.SR, subtype="PCM_16")
            st.session_state[key_prefix + "h"] = (h, str(out))
        except Exception as exc:
            st.error("Harmony stage: " + str(exc))
    if key_prefix + "h" in st.session_state:
        h, wav = st.session_state[key_prefix + "h"]
        st.pyplot(plots.chord_strip(h["chords"], h["duration_s"], max_s=24.0))
        _audio(wav)
        st.caption(h["note"] + " Chord labels from Pasan's planner refined by Rahul's model.")


def _speech_block(res, key_prefix="sp_"):
    st.divider()
    st.subheader("Optional creative mode: compose a melody from speech")
    st.warning("Not repair. This composes a new tune from the intonation and rhythm of speech. It does not recover an intended "
               "melody and sits outside the BPDR research claim.")
    c1, c2, c3 = st.columns(3)
    key = c1.selectbox("Key", KEYS, key=key_prefix + "k")
    scale = c2.selectbox("Scale", list(speech.SCALES), key=key_prefix + "s")
    bpm = c3.slider("Tempo (BPM)", 60, 140, 100, key=key_prefix + "b")
    if st.button("Compose melody from my speech", type="primary", key=key_prefix + "go"):
        try:
            x, _ = E.load_mono(res["source"])
            trk = E.pesto_track(x)
            m = speech.speech_to_melody(x, trk, KEYS.index(key), scale, float(bpm))
            dur = m["notes"][-1]["onset_s"] + m["notes"][-1]["duration_s"] + 0.5
            h = chords.harmonize(m["notes"], float(bpm), KEYS.index(key), "pop")
            show = min(h["duration_s"], 24.0)
            full = synth.mix(synth.render_notes([n for n in h["tiled_notes"] if n["onset_s"] < show], show),
                             synth.render_chords([c for c in h["chords"] if c["start_s"] < show], show))
            folder = SESSIONS / uuid.uuid4().hex; folder.mkdir(parents=True)
            sf.write(folder / "melody.wav", synth.render_notes(m["notes"], dur), E.SR, subtype="PCM_16")
            sf.write(folder / "melody_chords.wav", full, E.SR, subtype="PCM_16")
            st.session_state[key_prefix + "r"] = dict(m=m, dur=dur, h=h, folder=str(folder), trk=trk)
        except Exception as exc:
            st.error("Could not compose: " + str(exc))
    if key_prefix + "r" in st.session_state:
        r = st.session_state[key_prefix + "r"]
        st.caption(r["m"]["label"] + f" Found {r['m']['syllables_found']} syllable-like events.")
        st.pyplot(plots.piano_rolls([("Melody composed from the speech", r["m"]["notes"], "speech", r["trk"])], r["dur"]))
        a, b = st.columns(2)
        with a: st.write("Melody only"); _audio(Path(r["folder"]) / "melody.wav")
        with b: st.write("Melody + chords (Pasan → Rahul)"); _audio(Path(r["folder"]) / "melody_chords.wav")
        st.pyplot(plots.chord_strip(r["h"]["chords"], r["h"]["duration_s"], max_s=24.0))
        st.caption(r["h"]["note"])


def _confirmation_block(res, key_prefix="conf_"):
    out = Path(res["accepted_audio"]).parent
    st.divider()
    st.subheader("Hear-back verification: Is this your melody?")
    st.caption("Verification loop: confirming whether the synthesized playback matches your intended vocal melody. "
               "If you disagree, flag the section so located guidance can help you get a cleaner take.")

    saved_conf = confirmation.load_confirmation(out)
    if saved_conf:
        if saved_conf.get("confirmed"):
            st.success("✅ **Melody verified:** You confirmed this playback matches your intended melody.")
        else:
            st.warning("⚠️ **Melody flagged:** You reported a discrepancy with the extracted melody.")
            for g in saved_conf.get("guidance", []):
                st.write("↳ " + g)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Yes, this is my melody", type="primary", key=key_prefix + "yes"):
            conf = confirmation.record_confirmation(out, confirmed=True, notes_version="after_prepared_pesto")
            st.session_state[key_prefix + "status"] = conf
            st.rerun()
    with c2:
        if st.button("No, some notes are wrong", key=key_prefix + "no"):
            st.session_state[key_prefix + "open_dialog"] = True

    if st.session_state.get(key_prefix + "open_dialog"):
        with st.form(key=key_prefix + "form"):
            st.write("**Where did the melody sound incorrect?**")
            spans_text = st.text_input("Time range in seconds (e.g. 2.5-4.0, or leave blank for whole take)", key=key_prefix + "spans")
            comment = st.text_input("Optional comment (e.g. octave error, missed transition)", key=key_prefix + "cmt")
            if st.form_submit_button("Submit feedback"):
                flagged = []
                if spans_text.strip():
                    for part in spans_text.split(","):
                        if "-" in part:
                            try:
                                a, b = part.split("-")
                                flagged.append((float(a.strip()), float(b.strip())))
                            except Exception:
                                pass
                conf = confirmation.record_confirmation(out, confirmed=False, flagged_spans=flagged, user_comment=comment)
                st.session_state[key_prefix + "open_dialog"] = False
                st.session_state[key_prefix + "status"] = conf
                st.rerun()


def _annotate_boundary_markers(fig, hints):
    """Overlay boundary hint peak markers onto piano-roll axes without altering plots.py."""
    peaks = (hints or {}).get("peaks_s") or []
    if not peaks:
        return fig
    for ax in fig.axes:
        for i, peak_s in enumerate(peaks):
            ax.axvline(
                float(peak_s),
                color="#6d28d9",
                linestyle="--",
                linewidth=0.9,
                alpha=0.65,
                label="Boundary hint" if i == 0 else None,
            )
    if fig.axes:
        fig.axes[-1].legend(loc="upper right", fontsize=8, frameon=False)
    return fig


def _boundary_hints_block(res):
    """Render a compact summary of canonical 20 ms boundary hints if available."""
    hints = res.get("boundary_hints")
    if not hints:
        return
    peaks = hints.get("peaks_s") or []
    with st.expander(f"Note boundary hints ({len(peaks)} detected, advisory only)"):
        c1, c2, c3 = st.columns(3)
        c1.metric("Detected boundaries", len(peaks))
        c2.metric("Threshold", f"{hints.get('threshold', 0.8):.2f}")
        c3.metric("Grid step", f"{1000 * hints.get('frame_hop_s', 0.02):.0f} ms")
        if peaks:
            shown = ", ".join(f"{p:.2f}s" for p in peaks[:24])
            suffix = " …" if len(peaks) > 24 else ""
            st.caption(f"Peak times ({hints.get('model', 'boundary_transformer')}): {shown}{suffix}")
        else:
            st.caption(f"No peaks above threshold {hints.get('threshold', 0.8):.2f} ({hints.get('model', 'boundary_transformer')}).")


def run_tab():
    mode = st.radio("Input", ["Example recordings", "Upload audio", "Record with microphone"], horizontal=True)
    source, ref, note = None, None, None
    if mode == "Example recordings":
        items = examples.build()
        pick = st.selectbox("Recording", range(len(items)), format_func=lambda i: items[i]["label"])
        source, ref, note = items[pick]["path"], items[pick]["reference"], items[pick]["note"]
        st.caption(note)
        _audio(source)
    elif mode == "Upload audio":
        up = st.file_uploader("Upload a vocal recording (WAV / FLAC)", type=["wav", "flac"])
        if up:
            folder = SESSIONS / uuid.uuid4().hex; folder.mkdir(parents=True)
            source = folder / ("input" + Path(up.name).suffix.lower()); source.write_bytes(up.getvalue()); _audio(source)
            st.session_state["input_folder"] = str(folder)
    else:
        rec = st.audio_input("Record up to 20 seconds: hum, sing, or even read something aloud")
        if rec:
            folder = SESSIONS / uuid.uuid4().hex; folder.mkdir(parents=True)
            source = folder / "input.wav"; source.write_bytes(rec.getvalue())
            st.session_state["input_folder"] = str(folder)
    if mode != "Example recordings":
        st.caption(f"Privacy: your recording is stored only on this computer and is deleted automatically after "
                   f"{retention.RETENTION_HOURS} hours. You can delete it sooner with the button below the result.")
    room_noise = None
    if mode != "Example recordings":
        with st.expander("Optional room check: record 2 to 3 seconds of silence first"):
            clip = st.audio_input("Stay silent in the same room, same microphone", key="room_clip")
            if clip:
                rf = SESSIONS / uuid.uuid4().hex; rf.mkdir(parents=True); st.session_state["room_folder"] = str(rf)
                (rf / "room.wav").write_bytes(clip.getvalue())
                try:
                    xr, _ = E.load_mono(rf / "room.wav")
                    rc = room.room_check(xr)
                    st.write(f"**Room: {rc['verdict'].replace('_', ' ')}.** {rc['tip']}")
                    if rc.get("usable_reference"):
                        room_noise = xr
                        st.caption("This silent clip will be used as the noise reference for cleaning instead of guessing from the quietest moments.")
                except Exception as exc:
                    st.caption("Could not read the room clip: " + str(exc)[:120])
    sung = st.checkbox("I sang or chanted this on purpose (do not treat it as speech)", value=False)
    if st.button("Prepare recording", type="primary", disabled=source is None):
        for k in [k for k in st.session_state if k.startswith(("hm_", "sp_"))]:
            del st.session_state[k]
        with st.spinner("Measuring pitch evidence, testing a cleaned candidate, checking for harm…"):
            folder = SESSIONS / uuid.uuid4().hex
            st.session_state["out_folder"] = str(folder)
            st.session_state["res"] = dict(r=pipeline.prepare(source, folder, intent="sung" if sung else "unsure", room_noise=room_noise, with_boundary_hints=True), ref=ref)
    if "res" not in st.session_state:
        st.info("Pick a recording and press Prepare. Try the rain, silence, clipped and speech examples.")
        return
    res, ref = st.session_state["res"]["r"], st.session_state["res"]["ref"]
    kind, text = BANNER[res["status"]]
    getattr(st, kind)(f"**{res['status']}**: {text}")
    for r in res["reasons"]:
        st.write("• " + r)
    for g in res.get("guidance", []):
        st.write("↳ " + g)
    for a in res.get("advisories", []):
        if a != "speech_to_melody_available": st.caption(a)
    ps = res.get("phrase_selection")
    if ps:
        if ps.get("phrase_selected"):
            st.info(f"**Phrase selection:** recording was {ps['original_duration_s']:.1f} s (>20 s limit). "
                    f"Selected best {ps['duration_s']:.1f} s phrase ({ps['start_s']:.1f}–{ps['end_s']:.1f} s).")
        elif ps.get("trimmed_silence"):
            st.caption(f"Trimmed {ps['trimmed_leading_s']:.1f} s leading and {ps['trimmed_trailing_s']:.1f} s trailing silence.")
    q = res.get("quality")
    if q:
        c = st.columns(5)
        c[0].metric("Voiced evidence", f"{100*q['voiced_ratio']:.0f}%")
        c[1].metric("Pitch confidence", f"{q['mean_confidence_voiced']:.2f}")
        c[2].metric("Noise margin", f"{q['snr_estimate_db']:.0f} dB")
        c[3].metric("Clipped", f"{100*q['clipped_fraction']:.1f}%")
        c[4].metric("Held-pitch share", f"{q['speech']['plateau_fraction']:.2f}",
                    help="Sung/hummed notes hold a steady pitch; speech glides. Preliminary threshold 0.22.")
    if st.session_state.get("input_folder") and st.button("Delete my recording and results now"):
        retention.delete_sessions(SESSIONS, [st.session_state.get("input_folder"), st.session_state.get("out_folder"), st.session_state.get("room_folder")]
                                  + [v.get("folder") for k, v in st.session_state.items() if k.endswith("r") and isinstance(v, dict)])
        for k in [k for k in st.session_state if k.startswith(("hm_", "sp_")) or k in ("res", "input_folder", "out_folder", "room_folder")]:
            del st.session_state[k]
        st.success("Deleted. Nothing from this recording is kept."); st.rerun()
    if res["status"].startswith("ACCEPT"):
        out = Path(res["accepted_audio"]).parent
        a, b = st.columns(2)
        with a: st.write("Original"); _audio(res["source"])
        with b: st.write("Prepared (what the next component receives)"); _audio(res["accepted_audio"])
        st.subheader("What the melody extractor gets from it")
        dur = res["duration_s"]
        rows = []
        if res["notes"].get("before_current_extractor_pyin") is not None:
            rows.append(("Today: current extractor (pYIN) on the original", res["notes"]["before_current_extractor_pyin"], "before"))
        if res["notes"].get("shared_extractor_on_prepared") is not None:
            rows.append(("Shared extractor on the prepared audio (what it receives now)", res["notes"]["shared_extractor_on_prepared"], "before"))
        rows.append(("Illustration: finer pitch evidence (PESTO); a shorter pYIN frame gives most of the same gain", res["notes"]["after_prepared_pesto"], "after"))
        fig = plots.piano_rolls(rows, dur)
        _annotate_boundary_markers(fig, res.get("boundary_hints"))
        st.pyplot(fig)
        _boundary_hints_block(res)
        h1, h2 = st.columns(2)
        with h1: st.write("Hear what the shared extractor gets today"); _audio(out / "heard_before.wav")
        with h2: st.write("Hear what it would get from BPDR's pitch evidence (proposed)"); _audio(out / "heard_after.wav")
        _confirmation_block(res)
        with st.expander("Listening-only cleaned preview (not handed on)"):
            st.caption("The cleaned candidate was " + ("chosen." if res["cleaned_candidate_used"] else "not chosen: it did not improve pitch evidence without risk."))
            _audio(out / "cleaned_preview_listening_only.wav")
        _score_block(res, ref)
        with st.expander("Hand-off package (what Vishwa's component receives)"):
            st.json({k: res[k] for k in ("status", "accepted_audio", "accepted_sha256", "sidecar", "cleaned_candidate_used", "selector")})
            st.download_button("Download pitch-evidence sidecar (JSON)", Path(res["sidecar"]).read_bytes(), "sidecar.json")
            st.caption("The sidecar carries pitch evidence only. Notes, key, tempo and chords stay with their owners.")
        _harmony_block(res, "hm_")
    elif res.get("speech_detected"):
        _speech_block(res)


def evidence_tab():
    st.subheader("Does preparation help the real extractor? Measured on development data")
    if EVIDENCE.exists():
        d = json.loads(EVIDENCE.read_text(encoding="utf-8"))
        st.caption(d["scope"])
        for cond, s in d["summary"].items():
            st.markdown(f"**{cond}**: mean note F1 " + ", ".join(f"{k} {v:.3f}" for k, v in s["mean_f1"].items())
                        + f". Cleaned candidate chosen in {s['cleaned_chosen']}/{s['files']} files.")
            for arm in ("pesto_unnormalized", "pesto_original", "pesto_cleaned", "pesto_studio"):
                g = s[arm + "_minus_pyin"]
                st.write(f"  {arm} minus pyin: {g['gain']:+.3f}  (95% [{g['bootstrap95'][0]:+.3f}, {g['bootstrap95'][1]:+.3f}], "
                         f"better/worse {g['better']}/{g['worse']})")
    else:
        st.info("Run `tools/studio_evidence.py` to generate the evidence table.")
    st.divider()
    st.subheader("Control: how much of that gap is the tracker, and how much the analysis frame?")
    if FRAME_CONTROL.exists():
        fc = json.loads(FRAME_CONTROL.read_text(encoding="utf-8"))
        st.caption(fc["scope"] + " Onset+pitch note F1 (50 ms / 50 cents).")
        arms = ["pyin_4096_C1_today", "pyin_4096_C2", "pyin_2048_C2", "pyin_1024_C2", "pesto_normalised"]
        st.table({"Arm": arms, **{c: [round(fc["summary"][c]["mean"][a]["onset_pitch_f1_50ms"], 3) for a in arms] for c in fc["summary"]}})
        g = fc["summary"]["clean"]["vs_today"]["pyin_2048_C2|onset_pitch_f1_50ms"]
        st.markdown(f"A 2048-sample frame (shared extractor uses 4096, about 186 ms) recovers **{g['gain']:+.3f}** F1 on clean audio "
                    f"(95% [{g['bootstrap95'][0]:+.3f}, {g['bootstrap95'][1]:+.3f}], singer-level), about 88% of the gap to PESTO. "
                    "So most of the earlier PESTO gain is a frame-length effect. This is a development-data recommendation for the "
                    "extractor's owner, not yet confirmed on fresh recordings.")
    else:
        st.info("Run `tools/studio_frame_control.py` to generate the frame-size control.")
    st.divider()
    st.subheader("What the earlier experiments already showed")
    st.markdown(
        "- **Noise removal alone** (DeepFilterNet) did not improve note F1 on the shared extractor (singer-macro change about 0 to -0.01, "
        "intervals span zero); a perfect original-or-cleaned chooser would add at most about 0.03 (project records, read 10 Nov).\n"
        "- **Short-note pitch repair** was net harmful (22 helped, 89 harmed of 445), so the Studio keeps repair review-only.\n"
        "- **BPDR three-arm comparison** (full BPDR vs augmentation-only vs frame-only): replayed, did not pass the delivery gate; "
        "first-pass errors 21.4 / 23.2 / 24.9 cents against 24.8 for no correction. Most control fits never left the zero output "
        "(9 of 12; 6 of 11 after a matched fix), so this ordering is NOT a fair comparison and no arm is claimed better.\n"
        "- **Research gap** (checked 10 Nov): BERT-APC tests shifted boundaries only; a 2026 reference-free preprint assumes the note sequence is correct. "
        "Neither tests false splits or merges. This is a gap statement, not a priority claim.")
    st.caption("Honest summary: preparation adds little by itself. The largest measured effect is the extractor's analysis frame, "
               "which belongs to the extractor's owner. The Studio's role is restraint, located feedback and a faithful hand-off.")


def main():
    st.set_page_config(page_title="Vocal Rescue Studio", page_icon="🎙", layout="wide")
    st.title("Vocal Rescue Studio")
    st.write("Give it a noisy hum, a clipped recording or even someone reading aloud. It measures, protects, prepares and hands over "
             "the audio with evidence, and shows what that changes for the next component.")
    retention.purge_expired(SESSIONS)
    t1, t2 = st.tabs(["Prepare a recording", "Research evidence"])
    with t1: run_tab()
    with t2: evidence_tab()


if __name__ == "__main__":
    main()
