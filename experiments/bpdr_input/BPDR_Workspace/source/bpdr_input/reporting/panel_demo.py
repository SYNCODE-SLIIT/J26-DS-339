"""Build an offline panel demonstration from saved evidence, without fitting models.

Writes only inside BPDR_Workspace. Existing runs remain untouched. The lightweight
YIN measurement repeats the saved pitch example's scoring, not model inference.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import html
import json
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

import numpy as np
import soundfile as sf

from bpdr_input import COMPONENT_ROOT
from bpdr_input.application.saved_results import checksum, noise_example, pitch_example

WORKSPACE = COMPONENT_ROOT / "BPDR_Workspace"
DEFAULT_OUTPUT = WORKSPACE / "results/panel_demo"
SCHEMA = "bpdr-panel-demo-1"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def within(root, path):
    root, path = Path(root).resolve(), Path(path).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Path must remain inside {root}: {path}")
    return path


def paired_audio_metrics(before, after):
    a, sr_a = sf.read(before, dtype="float64")
    b, sr_b = sf.read(after, dtype="float64")
    if sr_a != sr_b or a.shape != b.shape:
        raise ValueError("Input/output sample grids differ")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Non-finite audio cannot be scored")
    diff = b - a
    return dict(sample_rate_hz=sr_a, sample_count=len(a),
                same_sample_grid=True, exact_sample_identity=bool(np.array_equal(a, b)),
                maximum_absolute_sample_change=float(np.max(np.abs(diff))),
                rms_sample_change=float(np.sqrt(np.mean(diff**2))))


def verify_file(path, expected):
    if checksum(path) != expected:
        raise ValueError(f"Checksum mismatch: {path}")


def load_verified_manifest(path=None, verify_sources=False):
    """Reject missing/tampered assets and paths outside the demo package.

    Source verification is optional so the copied HTML/audio package can travel
    without the historical evidence tree. Source hashes remain in the manifest.
    """
    path = Path(path or DEFAULT_OUTPUT / "manifest.json")
    digest_file = path.with_suffix(".sha256")
    if not digest_file.exists():
        raise ValueError("Manifest checksum receipt is missing")
    verify_file(path, digest_file.read_text(encoding="utf-8").split()[0])
    data = read_json(path)
    if data.get("schema_version") != SCHEMA:
        raise ValueError("Unsupported panel-demo schema")
    for asset in data["integrity"]["assets"]:
        target = within(path.parent, path.parent / asset["path"])
        verify_file(target, asset["sha256"])
    if verify_sources:
        for source in data["integrity"]["sources"]:
            verify_file(source["path"], source["sha256"])
    return data


class Package:
    def __init__(self, out):
        self.out = within(WORKSPACE, out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.assets = {}
        self.sources = {}

    def record_asset(self, path):
        path = within(self.out, path)
        rel = path.relative_to(self.out).as_posix()
        record = dict(path=rel, sha256=checksum(path), bytes=path.stat().st_size)
        self.assets[rel] = record
        return record

    def evidence(self, source, label):
        source = Path(source)
        item = dict(path=str(source), sha256=checksum(source))
        self.sources[str(source)] = item
        target = self.out / "evidence" / f"{label}{source.suffix}"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        copied = self.record_asset(target)
        return dict(**item, copy=copied["path"])

    def audio(self, source, label, expected=None):
        source = Path(source)
        if expected:
            verify_file(source, expected)
        self.sources[str(source)] = dict(path=str(source), sha256=checksum(source))
        target = self.out / "audio" / f"{label}{source.suffix}"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        info = sf.info(target)
        return dict(**self.record_asset(target), sample_rate_hz=info.samplerate,
                    sample_count=info.frames, channels=info.channels,
                    duration_seconds=info.duration)

    def plot(self, fig, name):
        target = self.out / "plots" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=140, facecolor="#faf8f2", bbox_inches="tight")
        self.record_asset(target)
        return target.relative_to(self.out).as_posix()


def figure(rows=1, height=3.7):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=(11, height), layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(rows, 1, squeeze=False)[:, 0]
    for ax in axes:
        ax.set_facecolor("#faf8f2")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.16)
    return fig, axes


def _runtime_records():
    path = COMPONENT_ROOT / "runs/presentation_runtime_checks_mixed.json"
    return path, {r["label"]: Path(r["path"]) for r in read_json(path)}


def build_pitch(pack, runtime):
    from bpdr_input.repair.rule_correction import fine_pitch
    example = pitch_example()  # validates saved candidate, source and checkpoint
    folder = runtime["mixed_synthetic_detuned_audio_only"]
    report = read_json(folder / "model_report.json")
    pair_path = example["input"].parent / "views.npz"
    with np.load(pair_path) as z:
        times = z["times"].copy()
        mask = z["mask"].astype(bool) & (z["correction_target_cents"] != 0)
        target = z["correction_target_cents"].copy()
    curves = []
    for path in (example["reference"], example["input"], example["output"]):
        audio, sr = sf.read(path, dtype="float32")
        curves.append(fine_pitch(audio, sr, times))
    ref, before, after = curves
    m_before = float(np.median((before-ref)[mask]))
    m_after = float(np.median((after-ref)[mask]))
    if not np.isclose(m_before, example["before_cents"], atol=.01) or not np.isclose(m_after, example["after_cents"], atol=.01):
        raise ValueError("Recomputed YIN residual disagrees with the selected saved measurement")
    metrics = paired_audio_metrics(example["input"], example["output"])
    before_audio, rate = sf.read(example["input"], dtype="float64")
    after_audio, _ = sf.read(example["output"], dtype="float64")
    sample_times = np.arange(len(before_audio)) / rate
    changed_support = np.zeros(len(before_audio), dtype=bool)
    for left, right in report["permission"]["intervals_seconds"]:
        changed_support |= (sample_times >= left) & (sample_times < right)
    outside_delta = float(np.max(np.abs(after_audio[~changed_support]-before_audio[~changed_support])))
    if outside_delta > 2**-23:
        raise ValueError("Saved pitch output changed samples outside the permitted interval")
    metrics.update(before_median_cents=m_before, after_median_cents=m_after,
                   absolute_median_error_reduction_cents=abs(m_before)-abs(m_after),
                   scoring_frames=int(mask.sum()), measurement="Recomputed YIN; known injected support; baseline used only for scoring",
                   historical_elapsed_seconds=report["elapsed_seconds"],
                   permitted_intervals_seconds=report["permission"]["intervals_seconds"],
                   maximum_outside_interval_sample_change=outside_delta,
                   outside_interval_preserved_within_pcm24=True,
                   prepared_handoff_enabled=False)
    fig, (ax,) = figure()
    ax.plot(times, before-ref, color="#b94a35", label="Input minus constructed original")
    ax.plot(times, after-ref, color="#087e83", label="Output preview minus constructed original")
    ax.axhline(0, color="#2b3938", lw=.8)
    left, right = times[mask].min(), times[mask].max()
    ax.axvspan(left, right, color="#c7b072", alpha=.16, label="Known injected plateau")
    ax.set(xlabel="Time (seconds)", ylabel="Residual pitch (cents)", ylim=(-32, 16),
           title="Controlled pitch repair: approximately −20 cents becomes +1.48 cents")
    ax.legend(fontsize=8, loc="lower right")
    case = dict(id="controlled_pitch", title="A small known pitch error is reduced", source_kind=example["kind"],
                outcome=report["outcome"], handoff_status=report["handoff"]["status"],
                input=pack.audio(example["input"], "pitch_input"),
                output=pack.audio(example["output"], "pitch_output", report["candidate"]["sha256"]),
                reference=pack.audio(example["reference"], "pitch_reference"),
                metrics=metrics, plot=pack.plot(fig, "pitch_residual.png"),
                summary=f"Median residual on {mask.sum()} injected plateau frames: {m_before:.2f} → {m_after:+.2f} cents. The preview is separate from accepted audio.",
                limitations=[example["scope"], "The constructed original is a scoring reference, not an inference input.",
                             "Numerical rendering checks do not certify musical intention or listening quality."],
                source_evidence=[pack.evidence(folder/"model_report.json", "pitch_model_report"),
                                 pack.evidence(COMPONENT_ROOT/"runs/presentation_selected_example_measurement.json", "pitch_saved_measurement"),
                                 pack.evidence(pair_path, "pitch_four_view_arrays"),
                                 pack.evidence(folder/"model_predictions.npz", "pitch_runtime_predictions")])
    pack.sources[str(Path(report["model_path"]))] = dict(path=report["model_path"], sha256=report["model_sha256"])
    target_file = pack.out / "evidence/pitch_remeasurement.npz"
    np.savez_compressed(target_file, times=times, mask=mask, reference_cents=ref,
                        input_cents=before, output_cents=after, target_cents=target)
    pack.record_asset(target_file)
    case["measurement_arrays"] = "evidence/pitch_remeasurement.npz"
    return case


def build_noise(pack):
    example = noise_example()  # recomputes paired SNR from saved audio
    saved_path = COMPONENT_ROOT / "runs/humming_noise_live_check.json"
    saved = read_json(saved_path)
    report = saved["report"]
    metrics = paired_audio_metrics(example["input"], example["output"])
    metrics.update(snr_gain_db=example["snr_gain_db"],
                   reference_attenuation_db=example["reference_attenuation_db"],
                   median_pitch_change_cents=example["median_pitch_change_cents"],
                   p95_pitch_change_cents=report["noise"]["audit"]["p95_pitch_change_cents"],
                   voiced_loss_fraction=report["noise"]["audit"]["voiced_loss_fraction"],
                   measurement="SNR gain recomputed against the same pre-noise recording; pitch preservation from saved audit",
                   prepared_handoff_enabled=False)
    fig, axes = figure(2, 5.2)
    waves = [sf.read(p, dtype="float64")[0] for p in (example["reference"], example["input"], example["output"])]
    sr = metrics["sample_rate_hz"]
    from scipy.signal import welch
    for wave, label, color in zip((waves[1]-waves[0], waves[2]-waves[0]), ("Added-noise residual before", "Residual after preview"), ("#b94a35", "#087e83")):
        hz, density = welch(wave, fs=sr, nperseg=1024)
        axes[0].plot(hz, 10*np.log10(density+1e-20), label=label, color=color, lw=1)
    axes[0].set(xlabel="Frequency (Hz)", ylabel="Residual density (dB/Hz)", xlim=(0, 8000),
                title=f"Real humming + recorded rain: paired SNR improves {example['snr_gain_db']:.2f} dB")
    axes[0].legend(fontsize=8)
    sample_time = np.arange(len(waves[1])) / sr
    prefix = sample_time < .7
    for wave, label, color in zip(waves[1:], ("Input rain-only prefix", "Prepared preview prefix"), ("#b94a35", "#087e83")):
        axes[1].plot(sample_time[prefix][::5], wave[prefix][::5], lw=.55, label=label, color=color)
    axes[1].set(xlabel="Time (seconds)", ylabel="Amplitude", title="Caller-declared 0–0.7 second noise-only reference")
    axes[1].legend(fontsize=8)
    return dict(id="real_humming_noise", title="Real humming with known added rain", source_kind=example["kind"],
                outcome=report["outcome"], handoff_status=report["handoff"]["status"],
                input=pack.audio(example["input"], "noise_input"),
                output=pack.audio(example["output"], "noise_output", report["noise_candidate"]["sha256"]),
                reference=pack.audio(example["reference"], "noise_reference"),
                metrics=metrics, plot=pack.plot(fig, "noise_difference.png"),
                summary=f"Paired SNR gain {example['snr_gain_db']:.2f} dB; saved median pitch movement {example['median_pitch_change_cents']:.3f} cents. This is signal cleaning, not learned pitch repair.",
                limitations=[example["scope"], "Needs a caller-declared noise-only prefix; no automatic voice/noise decision.",
                             "Numerical checks do not certify consonants, perceived quality or user intention."],
                source_evidence=[pack.evidence(saved_path, "noise_saved_live_check"),
                                 pack.evidence(Path(saved["path"])/"model_report.json", "noise_model_report")])


def build_identity(pack, runtime, key, case_id, title, kind):
    folder = runtime[key]
    report_path = folder / "model_report.json"
    report = read_json(report_path)
    source = Path(report["source"])
    accepted = report["handoff"]["accepted_audio"]
    metrics = paired_audio_metrics(source, accepted["path"])
    # Input and its own working copy may differ by at most PCM24 quantisation.
    if metrics["maximum_absolute_sample_change"] > 2**-23:
        raise ValueError("Identity case changed beyond PCM24 quantisation")
    metrics.update(mean_absolute_prediction_cents=report["mean_abs_prediction_cents"],
                   historical_elapsed_seconds=report["elapsed_seconds"], prepared_handoff_enabled=False)
    return dict(id=case_id, title=title, source_kind=kind,
                outcome=report["outcome"], handoff_status=report["handoff"]["status"],
                input=pack.audio(source, case_id+"_input"),
                output=pack.audio(accepted["path"], case_id+"_accepted", accepted["checksum"]),
                metrics=metrics,
                summary="The model proposes no permitted edit; the accepted working copy preserves pitch and timing.",
                limitations=["Technical acceptance does not certify a correct note, musical intent or sufficient melody evidence."],
                source_evidence=[pack.evidence(report_path, case_id+"_model_report")])


def build_negative_cases(pack):
    summary_path = COMPONENT_ROOT / "runs/demo/844c8fc7a70549dda1d6812787f9a633/results.json"
    saved = read_json(summary_path)
    cases = []
    for filename, case_id in (("06_silent.wav", "silent_rerecord"), ("09_unsupported.mp3", "unsupported_format"), ("01_jamendo_detuned.wav", "real_pitch_rollback")):
        row = next(r for r in saved["cases"] if r["file"] == filename)
        folder = Path(row["run_directory"])
        handoff_file = folder / "handoff.json"
        handoff = read_json(handoff_file)
        source = COMPONENT_ROOT / saved["inputs"] / filename
        case = dict(id=case_id, title={"silent_rerecord":"Silence requests another recording",
                    "unsupported_format":"An unsupported format receives clear feedback",
                    "real_pitch_rollback":"Real-source pitch preview fails its audit"}[case_id],
                    source_kind=row["kind"]+": "+row["detail"],
                    outcome=row["rule_outcome"] or row["phase1_status"],
                    handoff_status=handoff["status"], metrics={},
                    summary=handoff["user_message"], limitations=[],
                    source_evidence=[pack.evidence(handoff_file, case_id+"_handoff")])
        if source.suffix == ".wav":
            case["input"] = pack.audio(source, case_id+"_input")
        else:
            case["source_evidence"].append(pack.evidence(source, case_id+"_input"))
        if case_id == "real_pitch_rollback":
            rule_file = folder / "rule_report.json"
            rule = read_json(rule_file)
            transaction = rule["transaction"]
            case["source_evidence"].append(pack.evidence(rule_file, case_id+"_rule_report"))
            accepted = handoff["accepted_audio"]
            case["output"] = pack.audio(accepted["path"], case_id+"_accepted", accepted["checksum"])
            case["metrics"] = paired_audio_metrics(source, accepted["path"])
            case["metrics"].update(rollback=transaction["rollback"], audit_reasons=transaction["reasons"], method="Non-learning rule baseline; not the selected Transformer")
            case["summary"] = "The real Jamendo crop has a known injected shift. The rule candidate failed its render audit and was rolled back; the accepted audio remains unchanged."
            case["limitations"] = ["This is a recorded failure, not a successful pitch repair or natural pitch-error diagnosis.",
                                     "Natural real-voice pitch repair and musical-intention assessment remain unvalidated."]
        elif case_id == "silent_rerecord":
            wave, _ = sf.read(source)
            case["metrics"] = dict(peak_amplitude=float(np.max(np.abs(wave))), accepted_audio=None)
            case["limitations"] = ["Demonstrates a technical silence check; no validated speech-versus-melody gate is claimed."]
        else:
            case["metrics"] = dict(accepted_audio=None)
        cases.append(case)
    pack.evidence(summary_path, "negative_cases_summary")
    return cases


def build_four_view(pack, runtime):
    pitch_folder = runtime["mixed_synthetic_detuned_audio_only"]
    original_folder = runtime["mixed_synthetic_original_audio_only"]
    record_path = Path(read_json(pitch_folder/"model_report.json")["source"]).parent / "views.npz"
    with np.load(record_path) as z:
        a, b, c, d, times, mask, target = (z[k].copy() for k in ("A", "B", "C", "D", "times", "mask", "correction_target_cents"))
    for x, y in ((a, b), (c, d)):
        if not np.array_equal(x[:, :70], y[:, :70]):
            raise ValueError("Boundary-only paired views changed acoustic features")
    # Separate audio-only runtime calls used three inferred boundary alternatives.
    # These are not falsely labelled as saved predictions on the archive false split.
    with np.load(original_folder/"model_predictions.npz") as z:
        pred_a = z["predictions"].copy()
        if not np.array_equal(z["times"], times):
            raise ValueError("Original runtime prediction grid differs from the paired grid")
    with np.load(pitch_folder/"model_predictions.npz") as z:
        pred_c = z["predictions"].copy()
        if not np.array_equal(z["times"], times):
            raise ValueError("Detuned runtime prediction grid differs from the paired grid")
    support = mask.astype(bool) & (target != 0)
    metrics = dict(acoustic_features_equal_A_B=True, acoustic_features_equal_C_D=True,
                   boundary_channel_max_change_A_B=float(np.max(np.abs(a[:, 70]-b[:, 70]))),
                   boundary_channel_max_change_C_D=float(np.max(np.abs(c[:, 70]-d[:, 70]))),
                   runtime_original_boundary_mean_difference_cents=float(np.mean(np.abs(pred_a[0]-pred_a[1])[mask])),
                   runtime_detuned_boundary_mean_difference_cents=float(np.mean(np.abs(pred_c[0]-pred_c[1])[mask])),
                   runtime_median_response_on_injected_support_cents=float(np.median((pred_c[0]-pred_a[0])[support])),
                   ideal_response_cents=float(np.median(target[support])),
                   runtime_median_response_gain=float(np.median(((pred_c[0]-pred_a[0])/np.where(target != 0, target, 1))[support])))
    fig, axes = figure(3, 7.8)
    axes[0].plot(times, a[:, 70], label="A / C nominal boundary metadata", color="#3b6287")
    axes[0].plot(times, b[:, 70], label="B / D false-split metadata", color="#b94a35", ls="--")
    axes[0].set(ylabel="Boundary feature", title="Archived four-view construction: only boundary metadata changes within A/B and C/D")
    axes[0].legend(fontsize=8)
    for pred, name, color, style in ((pred_a[0], "Original, nominal", "#384644", "-"),
                                    (pred_a[1], "Original, +40 ms jitter", "#777e7b", "--"),
                                    (pred_c[0], "Detuned, nominal", "#087e83", "-"),
                                    (pred_c[1], "Detuned, +40 ms jitter", "#b94a35", "--")):
        axes[1].plot(times, np.where(mask, pred, np.nan), label=name, color=color, ls=style)
    axes[1].plot(times, np.where(support, target, np.nan), color="#c7aa66", lw=3, alpha=.6, label="Known ideal response")
    axes[1].set(ylabel="Correction (cents)", title="Separate saved audio-only runtime predictions: stable timing perturbation, changed acoustic response")
    axes[1].legend(fontsize=7.5, ncol=2)
    axes[2].plot(times, np.where(mask, pred_c[0]-pred_a[0], np.nan), color="#087e83", label="Detuned minus original prediction")
    axes[2].plot(times, target, color="#c7aa66", ls="--", label="Known counter-detuning target")
    axes[2].set(xlabel="Time (seconds)", ylabel="Response (cents)", title="Low boundary sensitivity alone is insufficient: a useful model must also respond to detuning")
    axes[2].legend(fontsize=8)
    return dict(title="The research gap: ignore a boundary mistake, respond to a pitch change",
                definitions=dict(A="Original audio + nominal boundaries", B="Same original + corrupted boundaries",
                                 C="Locally detuned audio + nominal boundaries", D="Same detuned audio + corrupted boundaries"),
                metrics=metrics, plot=pack.plot(fig, "four_view_response.png"),
                prediction_scope="Archive arrays demonstrate a false split. Saved audio-only runtime predictions use nominal / +40 ms jitter, not the archive false-split inputs.",
                limitations=["One selected generated phrase; no population-level BPDR advantage or real-singer generalization claim."],
                source_evidence=[pack.evidence(original_folder/"model_predictions.npz", "original_runtime_predictions")])


def build_downstream(pack):
    folder = Path(read_json(COMPONENT_ROOT/"runs/completion_handoff_check/current.json")["path"])
    declaration_path, evidence_path = folder/"declaration.json", folder/"evidence.json"
    declaration, evidence = read_json(declaration_path), read_json(evidence_path)
    for source in declaration["sources"].values():
        verify_file(source["path"], source["sha256"])
        pack.sources[source["path"]] = source
    copied = []
    for label in ("baseline", "baseline_repeat", "noisy", "prepared_preview"):
        copied.append(pack.evidence(folder/f"notes/{label}.csv", "game_"+label))
    return dict(title="Saved connection diagnostic with fixed GAME melody extraction",
                scores=evidence["scores"], measurement="Saved fixed-GAME agreement scores, with note CSVs copied and hashed; no new GAME inference",
                summary="18 baseline notes; onset+pitch F1 against baseline 0.944 for both noisy and cleaned audio. This check found no downstream F1 improvement.",
                preview_not_delivered_downstream=True, shared_pipeline_execution=False,
                interpretation=declaration["interpretation"],
                limitations=[evidence["limitation"], "Agreement with the extractor's pre-noise output is consistency, not human-annotated melody accuracy.",
                             "The cleaned candidate was measured separately; the production handoff remains original audio."],
                source_evidence=[pack.evidence(declaration_path, "game_declaration"), pack.evidence(evidence_path, "game_evidence")]+copied)


def render_html(manifest):
    e = html.escape
    cards = []
    for case in manifest["cases"]:
        players = []
        for key, title in (("input", "Input"), ("output", "Output preview" if case["id"] in ("controlled_pitch", "real_humming_noise") else "Accepted working copy"), ("reference", "Scoring reference")):
            if key in case:
                audio = case[key]
                players.append(f'<div><b>{title}</b><audio controls preload="none" src="{e(audio["path"])}"></audio><small>{audio["duration_seconds"]:.2f} s · {audio["sample_rate_hz"]:,} Hz</small></div>')
        metrics = case["metrics"]
        if case["id"] == "controlled_pitch":
            headline = f'{metrics["before_median_cents"]:.2f} → {metrics["after_median_cents"]:+.2f} cents'
        elif case["id"] == "real_humming_noise":
            headline = f'+{metrics["snr_gain_db"]:.2f} dB paired SNR'
        else:
            headline = case["outcome"].replace("_", " ")
        plot = f'<img src="{e(case["plot"])}" alt="Measured input and output difference">' if case.get("plot") else ""
        caveats = " ".join(case["limitations"])
        links = " · ".join(f'<a href="{e(item["copy"])}">{e(Path(item["path"]).name)}</a>' for item in case["source_evidence"])
        cards.append(f'<section id="{e(case["id"])}"><div class="kicker">{e(case["source_kind"])}</div><h2>{e(case["title"])}</h2><div class="result">{e(headline)}</div><p>{e(case["summary"])}</p><div class="players">{"".join(players)}</div>{plot}<p class="limits">{e(caveats)}</p><details><summary>Evidence and machine-readable measurements</summary><p>Handoff: <b>{e(case["handoff_status"])}</b></p><pre>{e(json.dumps(metrics, indent=2))}</pre><p>{links}</p></details></section>')
    research = manifest["four_view"]
    rows = "".join(f'<tr><td>{e(name)}</td><td>{score["estimate_notes"]}</td><td>{score["onset_pitch_f1"]:.3f}</td><td>{score["median_abs_pitch_difference_cents"]:.2f}</td></tr>' for name, score in manifest["downstream_check"]["scores"].items())
    definitions = "".join(f'<li><b>{v}</b> {e(text)}</li>' for v, text in research["definitions"].items())
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>BPDR · Evidence and vocal demo</title>
<style>:root{{--ink:#233c3a;--teal:#087e83;--paper:#faf8f2;--rust:#a7422c}}*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.6 system-ui,sans-serif}}main{{max-width:1120px;margin:auto;padding:38px 24px 80px}}h1,h2{{font-family:Georgia,serif;line-height:1.13;letter-spacing:-.025em}}h1{{font-size:clamp(42px,7vw,76px);max-width:900px;margin:16px 0}}h2{{font-size:31px;margin:14px 0}}.kicker{{text-transform:uppercase;letter-spacing:.13em;font-size:12px;font-weight:700;color:var(--rust)}}.intro{{font-size:21px;max-width:870px}}.notice{{padding:20px;border-left:4px solid var(--rust);background:#efe9dc}}nav{{display:flex;gap:10px;flex-wrap:wrap;margin:28px 0}}a{{color:#075d65}}nav a{{border:1px solid #b7c5be;border-radius:30px;padding:6px 15px;text-decoration:none}}section{{border-top:1px solid #b6c3ba;padding:32px 0;margin-top:28px;scroll-margin-top:20px}}.result{{font-size:30px;color:var(--teal);font-weight:700}}.players{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:20px;margin:22px 0}}audio{{display:block;width:100%;margin:10px 0}}small,.limits{{color:#586a60}}.limits{{font-size:14px}}img{{max-width:100%;height:auto;margin:14px 0}}details{{border:1px solid #c3cbbf;padding:12px}}summary{{cursor:pointer;font-weight:650}}pre{{overflow:auto;background:#eeeadd;padding:16px;font-size:12px}}table{{border-collapse:collapse;width:100%}}th,td{{padding:10px;border-bottom:1px solid #c3cbbf;text-align:left}}footer{{font-size:13px;margin-top:35px;color:#586a60}}@media print{{audio,nav{{display:none}}section{{break-inside:avoid}}}}</style></head><body><main>
<div class="kicker">PP1 / Boundary-Pitch Disentangled Repair</div><h1>Preserve the idea.<br>Show every change.</h1>
<p class="intro">A vocal-input research prototype: process a recoverable phrase, compare a bounded candidate with its input, and keep the original when an edit is uncertain.</p>
<p class="notice"><b>What this demo establishes:</b> a measured pitch-repair example on a generated held-out phrase, conservative noise cleaning on a real recording with known added rain, and recorded preservation / rollback / rerecord outcomes. Natural real-voice pitch repair, musical intention, and a BPDR advantage remain unproven. Candidate previews are separate; downstream audio stays unchanged.</p>
<nav><a href="#controlled_pitch">Pitch difference</a><a href="#real_humming_noise">Real humming</a><a href="#research">Research gap</a><a href="#downstream">Next-component evidence</a><a href="#real_pitch_rollback">Failure evidence</a></nav>
{"".join(cards[:2])}
<section id="research"><div class="kicker">Research contribution</div><h2>{e(research["title"])}</h2><p>The same audio should receive similar repair when boundary metadata changes. A known acoustic detuning should receive a compensating response. Predicting zero everywhere can look stable while failing the second requirement.</p><ol>{definitions}</ol><img src="{e(research["plot"])}" alt="Boundary construction and acoustic response"><p class="limits">{e(research["prediction_scope"])}</p><details><summary>Selected-case measurements</summary><pre>{e(json.dumps(research["metrics"],indent=2))}</pre></details></section>
{"".join(cards[2:])}
<section id="downstream"><div class="kicker">Next-component evidence</div><h2>Measure usefulness after processing</h2><p>{e(manifest["downstream_check"]["summary"])}</p><table><thead><tr><th>Fixed GAME input</th><th>Notes</th><th>Onset + pitch F1</th><th>Median pitch difference (cents)</th></tr></thead><tbody>{rows}</tbody></table><p class="limits">{e(" ".join(manifest["downstream_check"]["limitations"]))}</p></section>
<section><div class="kicker">What remains</div><h2>From a processing preview to a useful vocal</h2><p>Technical validity alone does not establish a recoverable melody. A validated melody-usability gate, natural-singer repair, user intention, listening quality and improved extraction need separate evidence. Ordinary reading is not automatically a failed song; the system should ask for a sung or hummed phrase when melodic information cannot be established.</p><p><a href="manifest.json">Full manifest</a> · <a href="manifest.sha256">Manifest checksum receipt</a> · <a href="README.md">Run and verification instructions</a></p></section>
<footer>Built {e(manifest["created_at"])}. Audio, plots and evidence copies are local. No model training or new GAME inference is performed by this demo. SHA-256 records detect later file changes; they are not external research approval.</footer></main></body></html>'''


def build(output_dir=None):
    pack = Package(output_dir or DEFAULT_OUTPUT)
    runtime_path, runtime = _runtime_records()
    pack.evidence(runtime_path, "selected_runtime_index")
    cases = [build_pitch(pack, runtime), build_noise(pack),
             build_identity(pack, runtime, "mixed_synthetic_original_audio_only", "original_keep", "An acceptable original remains unchanged", "Generated held-out original; no injected shift"),
             build_identity(pack, runtime, "mixed_vocadito_real_original", "real_original_keep", "A real vocal remains unchanged", "Real Vocadito original vocadito_2_p00; development example")]
    cases += build_negative_cases(pack)
    four_view = build_four_view(pack, runtime)
    downstream = build_downstream(pack)
    manifest = dict(schema_version=SCHEMA, created_at=datetime.now(ZoneInfo("Asia/Colombo")).isoformat(),
                    scope="Saved-run demonstration with current audio metrics; no training, dataset rebuild or policy adoption",
                    limitations=["No validated musical-usability or singer-intention classifier.",
                                 "No evidence that natural real-voice pitch repair generalizes.",
                                 "The selected example is not population accuracy or evidence of a BPDR advantage.",
                                 "Prepared preview adoption is disabled; accepted audio preserves original pitch and timing."],
                    cases=cases, four_view=four_view, downstream_check=downstream,
                    integrity=dict(algorithm="SHA-256", assets=[], sources=[]))
    readme = pack.out / "README.md"
    readme.write_text("""# BPDR panel demo

Open `index.html` in a browser. All media and plots use local relative paths.
The folder can be copied for offline playback; historical source paths in the
manifest describe provenance and need not exist for playback.

Rebuild from BPDR_Workspace with `tools/Build-Panel-Demo.ps1`, or run the component
Python with `-m bpdr_input.reporting.panel_demo`. This reads saved runs, repeats
the selected YIN residual measurement and audio metrics, and writes only here.
It does not fit a model, run GAME, adopt a candidate, or clear rehearsal/presentation
subfolders. External source evidence is preserved in the research archive.

Verify with `tools/Build-Panel-Demo.ps1 -VerifyOnly`. The loader checks the manifest
receipt and every packaged asset. `-VerifySources` additionally checks historical
sources and the saved checkpoint. SHA-256 checks detect drift, not research validity.

The first two outputs are experimental candidates. In both saved runs the actual
accepted handoff remains unchanged. Real pitch rollback uses the separate rule
baseline and is labelled accordingly. Fixed GAME results show a saved consistency
diagnostic with no observed onset+pitch F1 improvement, not full team integration.
""", encoding="utf-8")
    pack.record_asset(readme)
    page = pack.out / "index.html"
    page.write_text(render_html(manifest), encoding="utf-8")
    pack.record_asset(page)
    manifest["integrity"]["assets"] = list(pack.assets.values())
    manifest["integrity"]["sources"] = list(pack.sources.values())
    # Assert source immutability across the build before writing its receipt.
    for source in pack.sources.values():
        verify_file(source["path"], source["sha256"])
    target = pack.out / "manifest.json"
    write_json(target, manifest)
    target.with_suffix(".sha256").write_text(checksum(target)+"  manifest.json\n", encoding="utf-8")
    load_verified_manifest(target, verify_sources=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--verify-sources", action="store_true")
    args = parser.parse_args()
    out = args.output or DEFAULT_OUTPUT
    manifest = load_verified_manifest(out/"manifest.json", args.verify_sources) if args.verify_only else build(out)
    print(json.dumps(dict(status="verified", manifest=str(out/"manifest.json"), cases=len(manifest["cases"]),
                          assets=len(manifest["integrity"]["assets"]), sources=len(manifest["integrity"]["sources"])), indent=2))


if __name__ == "__main__":
    main()
