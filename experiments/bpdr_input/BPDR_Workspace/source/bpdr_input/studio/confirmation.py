"""Hear-back confirmation loop for Vocal Rescue Studio.

In line with Tier 1 build-plan F1:
- Plays back what the extractor heard from the accepted WAV (synthesised audio).
- Asks the user: "Is this your melody?"
- If confirmed: logs user acceptance in session metadata and sidecar.
- If rejected: records where the user disagrees (flagged spans), providing located
  re-record guidance rather than guessing an ungrounded pitch correction.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


def evaluate_confirmation(
    confirmed: bool,
    notes: Optional[List[Dict[str, Any]]] = None,
    flagged_spans: Optional[List[Tuple[float, float]]] = None,
    notes_version: str = "after_prepared_pesto",
    user_comment: str = ""
) -> Dict[str, Any]:
    """Generate structured confirmation result and located guidance."""
    if confirmed:
        return {
            "status": "CONFIRMED",
            "confirmed": True,
            "notes_version": notes_version,
            "action": "PROCEED_TO_DOWNSTREAM",
            "message": "User verified playback matches intended vocal melody.",
            "flagged_spans": [],
            "guidance": [],
            "user_comment": user_comment
        }

    # User disagreed with playback
    guidance = []
    if flagged_spans and len(flagged_spans) > 0:
        for a, b in flagged_spans:
            guidance.append(
                f"Flagged span at {a:.1f}–{b:.1f} s: hum or sing this section louder with steadier note holds, or re-record."
            )
    else:
        # General guidance if no specific spans were marked
        guidance.append(
            "Melody heard does not match intended melody: try re-recording closer to the mic, keeping note pitches steady."
        )

    return {
        "status": "USER_DISAGREED",
        "confirmed": False,
        "notes_version": notes_version,
        "action": "RERECORD_SPANS",
        "message": "User flagged discrepancies between intended melody and extracted notes.",
        "flagged_spans": flagged_spans or [],
        "guidance": guidance,
        "user_comment": user_comment
    }


def record_confirmation(
    session_dir: Path,
    confirmed: bool,
    notes: Optional[List[Dict[str, Any]]] = None,
    flagged_spans: Optional[List[Tuple[float, float]]] = None,
    notes_version: str = "after_prepared_pesto",
    user_comment: str = ""
) -> Dict[str, Any]:
    """Save user confirmation to session_dir/user_confirmation.json and update sidecar if present."""
    session_path = Path(session_dir)
    res = evaluate_confirmation(
        confirmed=confirmed,
        notes=notes,
        flagged_spans=flagged_spans,
        notes_version=notes_version,
        user_comment=user_comment
    )

    conf_file = session_path / "user_confirmation.json"
    conf_file.write_text(json.dumps(res, indent=2), encoding="utf-8")

    # Update sidecar.json if it exists
    sidecar_file = session_path / "sidecar.json"
    if sidecar_file.exists():
        try:
            sidecar = json.loads(sidecar_file.read_text(encoding="utf-8"))
            sidecar["user_confirmation"] = {
                "confirmed": res["confirmed"],
                "status": res["status"],
                "notes_version": res["notes_version"],
                "action": res["action"]
            }
            sidecar_file.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
        except Exception:
            pass

    return res


def load_confirmation(session_dir: Path) -> Optional[Dict[str, Any]]:
    """Load existing user_confirmation.json if present."""
    conf_file = Path(session_dir) / "user_confirmation.json"
    if not conf_file.exists():
        return None
    try:
        return json.loads(conf_file.read_text(encoding="utf-8"))
    except Exception:
        return None
