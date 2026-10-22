"""License and voice-tag eligibility for per-song Jamendo BPDR extraction."""

from __future__ import annotations

import re
from pathlib import Path


ALLOWED_LICENSES = frozenset({"by", "by-sa", "by-nc", "by-nc-sa"})
ELIGIBILITY_VERSION = "jamendo-bpdr-eligible-v1"


def licenses(path: Path) -> dict[str, dict[str, str]]:
    found: dict[str, dict[str, str]] = {}
    current = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        match = re.fullmatch(r"\d+/(\d+)\.mp3", line)
        if match:
            current = str(int(match.group(1)))
        elif current:
            match = re.search(r"https?://creativecommons\.org/licenses/([a-z-]+)/([0-9.]+)/?", line)
            if match:
                found[current] = {"kind": match.group(1), "version": match.group(2),
                                  "url": match.group(0)}
    return found


def eligible(tracks: list[dict[str, str]], license_path: Path) -> dict[str, dict]:
    """All local voice-tagged songs permitting derivatives; no artist sampling/splitting."""
    license_map = licenses(license_path)
    result = {}
    for row in tracks:
        license_info = license_map.get(row["track_id"])
        if (license_info and license_info["kind"] in ALLOWED_LICENSES
                and "instrument---voice" in row["tags"].split(";")):
            result[row["track_id"]] = license_info
    return result
