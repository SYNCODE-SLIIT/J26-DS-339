"""Portable file provenance for result JSONs: workspace-relative path plus SHA-256."""
from __future__ import annotations

import hashlib
from pathlib import Path


def relpath(path, root) -> str:
    """POSIX path relative to `root`; falls back to the absolute path if outside it."""
    p, r = Path(path).resolve(), Path(root).resolve()
    try:
        return p.relative_to(r).as_posix()
    except ValueError:
        return p.as_posix()


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
