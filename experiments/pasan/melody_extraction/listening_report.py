"""Build a local HTML page for listening across completed pipelines."""

from __future__ import annotations

import html
import json
import os
from pathlib import Path
from urllib.parse import quote


def _media_uri(path: str | Path, report_dir: Path) -> str:
    relative = os.path.relpath(Path(path).expanduser().resolve(), report_dir.resolve())
    return quote(Path(relative).as_posix())


def _audio(path: str | Path | None, report_dir: Path) -> str:
    if not path or not Path(path).exists():
        return "<span class='missing'>not available</span>"
    uri = _media_uri(path, report_dir)
    return f'<audio controls preload="none" src="{uri}"></audio>'


def build_listening_report(
    output_dir: Path,
    input_dir: Path,
    pipelines: list[str] | None = None,
) -> Path:
    """Create an audio-player matrix for every completed song/pipeline."""
    output_dir = output_dir.expanduser().resolve()
    input_dir = input_dir.expanduser().resolve()
    selected = set(pipelines or [])
    grouped: dict[str, list[dict]] = {}

    for metadata_path in sorted(output_dir.glob("*/**/metadata.json")):
        metadata = json.loads(metadata_path.read_text())
        pipeline = metadata.get("pipeline")
        if selected and pipeline not in selected:
            continue
        grouped.setdefault(metadata["audio_path"], []).append(metadata)

    sections: list[str] = []
    for audio_path_text, results in grouped.items():
        audio_path = Path(audio_path_text)
        try:
            song = audio_path.resolve().relative_to(input_dir).with_suffix("")
        except ValueError:
            song = audio_path.with_suffix("").name
        stem_candidates = sorted((output_dir / "_stems").glob(f"*/{song}/vocals.wav"))
        vocal_stem = stem_candidates[0] if stem_candidates else None

        rows: list[str] = []
        for result in sorted(results, key=lambda value: value["pipeline"]):
            artifacts = result.get("artifacts") or {}
            midi = artifacts.get("midi")
            midi_link = (
                f'<a href="{_media_uri(midi, output_dir)}">MIDI</a>'
                if midi and Path(midi).exists()
                else "—"
            )
            rows.append(
                "<tr>"
                f"<td><code>{html.escape(result['pipeline'])}</code></td>"
                f"<td>{result.get('note_count', '—')}</td>"
                f"<td>{_audio(artifacts.get('melody_audio'), output_dir)}</td>"
                f"<td>{_audio(artifacts.get('melody_overlay'), output_dir)}</td>"
                f"<td>{midi_link}</td>"
                "</tr>"
            )

        sections.append(
            f"<section><h2>{html.escape(str(song))}</h2>"
            "<div class='sources'>"
            f"<div><strong>Original</strong>{_audio(audio_path, output_dir)}</div>"
            f"<div><strong>Separated vocal</strong>{_audio(vocal_stem, output_dir)}</div>"
            "</div>"
            "<table><thead><tr><th>Pipeline</th><th>Notes</th>"
            "<th>Extracted melody</th><th>Overlay on original</th><th>Data</th>"
            f"</tr></thead><tbody>{''.join(rows)}</tbody></table></section>"
        )

    report_path = output_dir / "listening_comparison.html"
    report_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Melody extraction listening comparison</title>"
        "<style>"
        "body{font:15px system-ui;margin:2rem;background:#f7f7f8;color:#202124}"
        "h1{margin-bottom:.35rem}section{background:white;padding:1.25rem;margin:1.5rem 0;"
        "border-radius:12px;box-shadow:0 1px 4px #0002;overflow:auto}"
        ".sources{display:flex;gap:2rem;flex-wrap:wrap;margin-bottom:1rem}"
        ".sources div{display:grid;gap:.35rem}table{border-collapse:collapse;width:100%}"
        "th,td{text-align:left;padding:.55rem;border-bottom:1px solid #ddd;white-space:nowrap}"
        "audio{width:250px;height:36px;vertical-align:middle}.missing{color:#777}"
        "code{font-size:.9em}</style></head><body>"
        "<h1>Melody extraction listening comparison</h1>"
        "<p>Standalone previews are synthesized tones. Overlays mix those tones over the source audio.</p>"
        + "".join(sections)
        + "</body></html>\n"
    )
    return report_path
