"""Save SheetSage2's full-song beat and downbeat grid."""

from __future__ import annotations

import csv
import io
import json
import math
import statistics
from pathlib import Path

from common import atomic_text, fingerprint, track_dir


MODEL_REVISION = "32c7c7473e7b59e1ed36bd8a8fefbd11ae8d468a"
CSV_COLUMNS = ("track_id", "beat_index", "time_sec", "beat_in_bar", "bar_index",
               "meter_numerator", "meter_denominator", "is_downbeat")


def paths(track: dict[str, str]) -> dict[str, Path]:
    directory = track_dir(track) / "beats"
    return {suffix: directory / f"beats.{suffix}" for suffix in ("json", "csv")}


def read_beat_lab(path: Path, duration_sec: float) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"SheetSage2 did not write {path.name}")
    beats = []
    bar_index = 0
    previous_time = -1.0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 4:
            raise ValueError(f"{path.name}:{line_number} expected time, beat, meter numerator/denominator")
        time_sec, beat_in_bar, numerator, denominator = (float(fields[0]), int(fields[1]),
                                                         int(fields[2]), int(fields[3]))
        if (not math.isfinite(time_sec) or time_sec < 0 or time_sec > duration_sec + 1
                or time_sec <= previous_time or numerator < 1 or denominator < 1
                or not 1 <= beat_in_bar <= numerator):
            raise ValueError(f"{path.name}:{line_number} invalid beat position or meter")
        if beats and beat_in_bar == 1:
            bar_index += 1
        beats.append({"beat_index": len(beats), "time_sec": round(time_sec, 5),
                      "beat_in_bar": beat_in_bar, "bar_index": bar_index,
                      "meter_numerator": numerator, "meter_denominator": denominator,
                      "is_downbeat": beat_in_bar == 1})
        previous_time = time_sec
    if len(beats) < 2:
        raise ValueError(f"{path.name} contains fewer than two beats")
    return beats


def read_downbeat_lab(path: Path, duration_sec: float) -> list[float]:
    if not path.is_file():
        raise FileNotFoundError(f"SheetSage2 did not write {path.name}")
    times = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 1:
            raise ValueError(f"{path.name}:{line_number} expected one downbeat time")
        time_sec = float(fields[0])
        if (not math.isfinite(time_sec) or time_sec < 0 or time_sec > duration_sec + 1
                or times and time_sec <= times[-1]):
            raise ValueError(f"{path.name}:{line_number} invalid downbeat time")
        times.append(round(time_sec, 5))
    return times


def complete(track: dict[str, str]) -> dict | None:
    files = paths(track)
    if not all(path.is_file() and path.stat().st_size for path in files.values()):
        return None
    try:
        data = json.loads(files["json"].read_text(encoding="utf-8"))
        if (data.get("schema_version") != 1 or data.get("track_id") != track["track_id"]
                or data.get("mtg_track_id") != track["mtg_track_id"]
                or data.get("source_mp3") != track["source_mp3"]
                or data.get("source_fingerprint") != fingerprint(track)
                or data.get("method") != "SheetSage2 beat.lab and downbeat.lab"
                or data.get("model_revision") != MODEL_REVISION):
            return None
        beats = data["beats"]
        duration = float(data["duration_sec"])
        if not math.isfinite(duration) or duration <= 0 or len(beats) < 2:
            return None
        for index, beat in enumerate(beats):
            if (beat["beat_index"] != index
                    or not 0 <= float(beat["time_sec"]) <= duration + 1
                    or not 1 <= beat["beat_in_bar"] <= beat["meter_numerator"]
                    or beat["meter_denominator"] < 1
                    or beat["is_downbeat"] != (beat["beat_in_bar"] == 1)):
                return None
        if any(float(right["time_sec"]) <= float(left["time_sec"])
               for left, right in zip(beats, beats[1:])):
            return None
        with files["csv"].open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != CSV_COLUMNS:
                return None
            rows = list(reader)
        if len(rows) != len(beats):
            return None
        for beat, row in zip(beats, rows):
            if (row["track_id"] != track["track_id"]
                    or any(row[column] != str(beat[column]) for column in CSV_COLUMNS[1:])):
                return None
        return data
    except (OSError, ValueError, TypeError, KeyError):
        return None


def save(track: dict[str, str], duration_sec: float,
         beat_lab: Path, downbeat_lab: Path) -> dict:
    beats = read_beat_lab(beat_lab, duration_sec)
    downbeats = read_downbeat_lab(downbeat_lab, duration_sec)
    intervals = [right["time_sec"] - left["time_sec"]
                 for left, right in zip(beats, beats[1:])
                 if 0.2 <= right["time_sec"] - left["time_sec"] <= 2.0]
    period = round(float(statistics.median(intervals)), 5) if intervals else None
    beat_ones = [beat["time_sec"] for beat in beats if beat["is_downbeat"]]
    aligned = (len(beat_ones) == len(downbeats)
               and all(abs(a - b) <= 0.05 for a, b in zip(beat_ones, downbeats)))
    data = {"schema_version": 1, "track_id": track["track_id"],
            "mtg_track_id": track["mtg_track_id"], "source_mp3": track["source_mp3"],
            "source_fingerprint": fingerprint(track), "method": "SheetSage2 beat.lab and downbeat.lab",
            "model_revision": MODEL_REVISION, "time_origin": "full_song_start",
            "duration_sec": duration_sec, "median_beat_period_sec": period,
            "median_beat_rate_bpm": round(60 / period, 3) if period else None,
            "downbeat_lab_times_sec": downbeats, "downbeat_lab_matches_beat_positions": aligned,
            "beats": beats}
    files = paths(track)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS)
    writer.writeheader()
    for beat in beats:
        writer.writerow({"track_id": track["track_id"], **beat})
    atomic_text(files["csv"], buffer.getvalue())
    atomic_text(files["json"], json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    if complete(track) is None:
        raise ValueError(f"Written beat files failed validation for {track['track_id']}")
    return data
