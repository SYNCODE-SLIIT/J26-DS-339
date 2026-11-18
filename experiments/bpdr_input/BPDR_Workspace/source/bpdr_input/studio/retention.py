"""Session-audio retention: recordings are kept only for a short, stated time, and can be deleted on demand.

Only direct children of the sessions folder whose names are 32 hex characters (the names the Studio
creates) are ever removed; anything else is left alone, so a wrong path can never delete project files.
"""
import re
import shutil
import time
from pathlib import Path

RETENTION_HOURS = 24
_NAME = re.compile(r"^[0-9a-f]{32}$")


def _is_session(sessions, folder):
    sessions, folder = Path(sessions).resolve(), Path(folder).resolve()
    return folder.parent == sessions and bool(_NAME.match(folder.name)) and folder.is_dir() and not folder.is_symlink()


def purge_expired(sessions, max_age_hours=RETENTION_HOURS, now=None):
    """Delete session folders not modified for max_age_hours. Returns the number removed."""
    sessions = Path(sessions)
    if not sessions.is_dir():
        return 0
    now = time.time() if now is None else now
    removed = 0
    for folder in sessions.iterdir():
        if _is_session(sessions, folder) and now - folder.stat().st_mtime > max_age_hours * 3600:
            shutil.rmtree(folder, ignore_errors=True); removed += 1
    return removed


def delete_sessions(sessions, folders):
    """Delete the named session folders now (the user's 'delete my recording'). Returns the number removed."""
    removed = 0
    for folder in {str(Path(f).resolve()) for f in folders if f}:
        if _is_session(sessions, folder):
            shutil.rmtree(folder, ignore_errors=True); removed += 1
    return removed
