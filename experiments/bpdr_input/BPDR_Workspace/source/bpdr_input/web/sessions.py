"""Session store and single-job concurrency guard for Component 1 web service.

All session directories use 32 lowercase hex characters as direct children of
`results/studio/sessions/` so `bpdr_input.studio.retention` manages 24 h expiry
and immediate user deletion safely.
"""
import json
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from ..studio import retention

WORKSPACE = Path(__file__).resolve().parents[3]
SESSIONS_DIR = WORKSPACE / "results" / "studio" / "sessions"
_HEX32 = re.compile(r"^[0-9a-f]{32}$")
_JOB_LOCK = threading.Lock()


def is_valid_session_id(session_id: str) -> bool:
    return bool(isinstance(session_id, str) and _HEX32.match(session_id))


def session_path(session_id: str) -> Path:
    if not is_valid_session_id(session_id):
        raise ValueError("Invalid session ID format")
    return SESSIONS_DIR / session_id


def create_session(consent: bool = True) -> Dict[str, Any]:
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    purged = retention.purge_expired(SESSIONS_DIR, max_age_hours=retention.RETENTION_HOURS)
    sid = uuid.uuid4().hex
    sdir = SESSIONS_DIR / sid
    sdir.mkdir(parents=True, exist_ok=True)
    data = {
        "session_id": sid,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "consent": bool(consent),
        "retention_hours": retention.RETENTION_HOURS,
        "status": "CREATED",
        "job": {"state": "idle", "step": "waiting", "progress": 0, "elapsed_s": None},
    }
    save_session(sid, data)
    return {**data, "purged_expired": purged}


def load_session(session_id: str) -> Dict[str, Any]:
    sdir = session_path(session_id)
    meta_file = sdir / "session.json"
    if not sdir.is_dir() or not meta_file.is_file():
        raise FileNotFoundError(f"Session not found: {session_id}")
    return json.loads(meta_file.read_text(encoding="utf-8"))


def save_session(session_id: str, data: Dict[str, Any]) -> None:
    sdir = session_path(session_id)
    sdir.mkdir(parents=True, exist_ok=True)
    tmp = sdir / "session.tmp"
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(sdir / "session.json")


def delete_session(session_id: str) -> int:
    sdir = session_path(session_id)
    if not sdir.exists():
        return 0
    return retention.delete_sessions(SESSIONS_DIR, [sdir])


def acquire_job_lock(timeout_s: float = 0.05) -> bool:
    return _JOB_LOCK.acquire(timeout=timeout_s)


def release_job_lock() -> None:
    if _JOB_LOCK.locked():
        _JOB_LOCK.release()
