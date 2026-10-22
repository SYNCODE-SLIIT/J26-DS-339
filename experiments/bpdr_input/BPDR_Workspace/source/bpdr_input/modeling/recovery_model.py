"""Scoped learned pitch-preview model; never an intended-note classifier.

The callable feature interface reads only the current input. The operating
policy is selected on training-side calibration singers and stored unchanged.
"""
import json
from pathlib import Path

import joblib
import numpy as np

from bpdr_input.application.saved_results import checksum
from bpdr_input.modeling.local_context_pilot import context_features
from bpdr_input.repair.rule_correction import segment_notes
from bpdr_input.reporting.bpdr_showcase import correction_from_prediction

FEATURE_VERSION = "recovery-single-input-context-1"


def scope_mask(raw):
    """Eligibility of a long voiced region, derived only from this recording."""
    raw = np.asarray(raw)
    voiced = (raw[:, 66] > .5) & (raw[:, 65] >= .8)
    mask = np.zeros(len(raw), dtype=bool)
    regions = []
    for start, stop in segment_notes(raw[:, 64], voiced):
        if (stop-start)*.02 < 1.10:
            continue
        y = raw[start:stop, 64]
        x = np.arange(len(y))
        slope = np.polyfit(x, y, 1)
        spread = float(np.std(y-np.polyval(slope, x)))
        if spread <= 20.:
            mask[start:stop] = True
            regions.append(dict(start_seconds=start*.02, end_seconds=stop*.02,
                                detrended_pitch_spread_cents=spread))
    return mask, dict(version="own-audio-long-voiced-context-1", regions=regions,
                      voiced_frames=int(voiced.sum()), eligible_frames=int(mask.sum()),
                      scope="Long held-note previews only; no musical correctness or intention decision")


def predict_raw(model, raw71):
    """Uncalibrated physical cents; no reference, target or intervention interval."""
    features = context_features(raw71, include_boundary=False)
    if model["kind"] == "old_mlp":
        from bpdr_input.modeling.local_context_pilot import predict_raw as old_predict
        return old_predict(model["estimator"], raw71)
    output = np.asarray(model["estimator"].predict(features), dtype=np.float64)
    if output.shape != (len(raw71),) or not np.isfinite(output).all():
        raise ValueError("Invalid learned recovery prediction")
    return output


def propose(raw_views, model, payload):
    """Exactly the same calibrated proposal used for calibration and live input.

    This creates a request only. Existing permission/render/rollback still apply.
    Disagreement is numerical sensitivity, not a probability of being correct.
    """
    views = list(raw_views)
    if len(views) != 3 or len({len(x) for x in views}) != 1:
        raise ValueError("Expected three aligned single-input runtime boundary views")
    predictions = np.stack([predict_raw(model, raw) for raw in views])
    allowed, scope = scope_mask(views[0])
    voiced = views[0][:, 66] > .5
    operating = predictions*float(payload["operating_policy"]["gain"])
    operating[:, ~allowed] = 0.
    disagreement = np.max(np.abs(operating[1:]-operating[0]), axis=0)
    curve = correction_from_prediction(operating[0], operating[1], voiced,
        threshold=float(payload["operating_policy"]["deadband_cents"]))
    return dict(requested_curve=curve, operating_predictions=operating,
                raw_predictions=predictions, disagreement=disagreement,
                reliable=voiced, scope_metadata=scope)


def save_candidate(path, estimator, kind, policy, metadata):
    payload = dict(feature_version=FEATURE_VERSION, kind=kind, operating_policy=policy,
                   **metadata)
    joblib.dump(dict(model=dict(kind=kind, estimator=estimator), payload=payload), path)
    return payload


def load_candidate(path, require_rendered=True):
    """Load only the locked selected checkpoint with independently checked gates."""
    path = Path(path)
    evidence = json.loads((path.parent/"evidence.json").read_text(encoding="utf-8"))
    if checksum(path) != evidence["checkpoint_sha256"]:
        raise ValueError("Learned recovery checkpoint checksum mismatch")
    if (not evidence["outer_proposal_gate_passes"] or
            not evidence["selection"].get("calibration_gate_passes",False) or
            evidence["outer"].get("groups") != 2):
        raise ValueError("The selected recovery model failed its independent singer gates")
    if require_rendered and not evidence.get("rendered_demo_gate_passes", False):
        raise ValueError("Independent-singer rendered verification remains incomplete")
    stored = joblib.load(path)
    if stored["payload"]["feature_version"] != FEATURE_VERSION:
        raise ValueError("Unknown learned recovery feature interface")
    declaration = path.parent / "declaration.json"
    if checksum(declaration) != stored["payload"]["declaration_sha256"]:
        raise ValueError("Learned recovery declaration changed")
    if require_rendered:
        verification = Path(evidence['rendered_verification'])
        if checksum(verification) != evidence['rendered_verification_sha256']:
            raise ValueError('Rendered recovery verification changed')
        measured = json.loads(verification.read_text(encoding='utf-8'))
        if (not measured.get('rendered_demo_gate_passes') or measured['checkpoint_sha256'] != checksum(path)
                or measured['declaration_sha256'] != checksum(declaration)):
            raise ValueError('Rendered recovery verification does not match this model')
    return stored["model"], stored["payload"]


def load_locked_diagnostic(path):
    """Integrity-checked failed candidate for explicitly labelled experiments.

    This never asserts that calibration, coverage or research gates passed.
    It is used by the all-case rendered diagnostic, and by a separately hashed
    exploratory preview card only after actual listening artifacts are verified.
    """
    path = Path(path)
    evidence = json.loads((path.parent/'evidence.json').read_text(encoding='utf-8'))
    if checksum(path) != evidence['checkpoint_sha256']:
        raise ValueError('Diagnostic checkpoint changed')
    stored = joblib.load(path)
    payload = stored['payload']
    if (payload['feature_version'] != FEATURE_VERSION or
            checksum(path.parent/'declaration.json') != payload['declaration_sha256'] or
            checksum(path.parent/'locked_selection.json') != payload['selection_sha256']):
        raise ValueError('Diagnostic model declaration, selection or feature version changed')
    return stored['model'], payload
