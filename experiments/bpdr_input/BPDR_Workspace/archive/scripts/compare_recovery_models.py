"""Supplementary same-example model comparison; never changes demo selection.

Reconstruct the unsaved HGB estimator once from its exact declared training
arrays/settings, verify its old calibration scores, and freeze predictions for
the notebook. Historical neural checkpoints are evaluated without fine-tuning.
"""
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import uuid

import joblib
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "source"))
from bpdr_input.experiments.learning import model_recovery as experiment
from bpdr_input.modeling.local_context_pilot import load_checkpoint
from bpdr_input.modeling.recovery_model import predict_raw, scope_mask
from bpdr_input.reporting.bpdr_showcase import load_model, correction_from_prediction
from bpdr_input.repair.repair_controller import plan_fixture


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def aggregate(rows):
    valid = [r for r in rows if "excluded" not in r]
    keys = [k for k in valid[0] if k.startswith(("raw_", "permitted_", "original_", "baseline_"))]
    groups = {}
    for group in sorted({r["singer"] for r in rows}):
        rr = [r for r in valid if r["singer"] == group]
        if not rr:
            groups[group] = dict(audited=0, attempted=sum(r["singer"] == group for r in rows))
            continue
        groups[group] = {k: float(np.mean([r[k] for r in rr])) for k in keys}
        groups[group].update(audited=len(rr), attempted=sum(r["singer"] == group for r in rows))
        groups[group]["coverage_all_attempts_percent"] = 100 * sum(r["permitted_nonidentity"] for r in rr) / groups[group]["attempted"]
    represented = [g for g in groups.values() if g["audited"]]
    summary = {k: float(np.mean([g[k] for g in represented])) for k in keys}
    summary.update(coverage_all_attempts_percent=float(np.mean([g["coverage_all_attempts_percent"] for g in represented])),
                   attempted=len(rows), audited=len(valid), singers=len(represented),
                   original_recordings=len({r["path_A"] for r in valid}),
                   raw_error_reduction_percent=100 * float(np.mean([1-g["raw_mae_cents"]/g["baseline_mae_cents"] for g in represented])),
                   permitted_error_reduction_percent=100 * float(np.mean([1-g["permitted_mae_cents"]/g["baseline_mae_cents"] for g in represented])))
    return summary, groups


def score_case(pair, predictions):
    curves, details = {}, {}
    for view in ("A", "C"):
        analysis = pair[view]
        raw = predictions[view]
        allowed, _ = scope_mask(analysis["views"][0])
        operating = raw.copy()
        operating[:, ~allowed] = 0
        reliable = analysis["views"][0][:, 66] > .5
        disagreement = np.max(np.abs(operating[1:]-operating[0]), axis=0)
        requested = correction_from_prediction(operating[0], operating[1], reliable, threshold=3.)
        plan = plan_fixture(pair["audio_"+view], 22050, analysis["features"]["times_seconds"],
                            requested, operating[1], disagreement, reliable, analysis["protected"])
        curves[view] = requested if plan["permitted"] else np.zeros(len(requested))
        details[view] = dict(permitted=bool(plan["permitted"]), identity=bool(plan["identity"]), reasons=plan["reasons"])
    target, mask = pair["target"], pair["mask"]
    plateau = mask & (target != 0)
    assert plateau.any()
    raw_error = np.abs(predictions["C"][0][plateau]-target[plateau])
    permitted_error = np.abs(curves["C"][plateau]-target[plateau])
    original = pair["A"]["reliable"]
    row = dict(singer=pair["group"], cents=pair["cents"], path_A=pair["path_A"], path_C=pair["path_C"],
               target_path=pair["target_path"], scored_frames=int(plateau.sum()),
               baseline_mae_cents=float(np.abs(target[plateau]).mean()),
               raw_mae_cents=float(raw_error.mean()), permitted_mae_cents=float(permitted_error.mean()),
               raw_within_5c_percent=100*float(np.mean(raw_error <= 5.)),
               raw_within_10c_percent=100*float(np.mean(raw_error <= 10.)),
               permitted_within_10c_percent=100*float(np.mean(permitted_error <= 10.)),
               original_raw_abs_mean_cents=float(np.abs(predictions["A"][0][original]).mean()),
               original_permitted_abs_mean_cents=float(np.abs(curves["A"][original]).mean()),
               original_edited_frames_percent=100*float(np.mean(curves["A"][original] != 0)),
               permitted_nonidentity=bool(np.any(curves["C"])), controller=details)
    return row, curves


def main():
    torch.set_num_threads(2)
    run = Path(read(WORKSPACE/"results/model_recovery/current.json")["path"])
    evidence = read(run/"evidence.json")
    declaration = read(run/"declaration.json")
    calibration = read(run/"calibration_all.json")
    extra_path = Path(evidence["checkpoint"])
    assert sha(extra_path) == evidence["checkpoint_sha256"]
    small = Path(read(WORKSPACE/"results/small_real_pilot/current.json")["path"])
    old_record = read(small/"evidence.json")["results"]["bpdr"]
    old_path = Path(old_record["checkpoint"])
    assert sha(old_path) == old_record["checkpoint_sha256"]
    transformer_card = read(WORKSPACE/"selected_model/model.json")
    transformer_path = WORKSPACE/"selected_model"/transformer_card["checkpoint"]
    assert sha(transformer_path) == transformer_card["sha256"]
    protected = [run/"evidence.json", run/"locked_selection.json", WORKSPACE/"results/model_recovery/demo_model.json",
                 extra_path, old_path, transformer_path]
    unchanged = {str(p): sha(p) for p in protected}
    out = WORKSPACE/"results/model_comparison"/uuid.uuid4().hex
    out.mkdir(parents=True)
    protocol = dict(version="frozen-model-common-calibration-1", created_at=datetime.now(timezone.utc).isoformat(),
        recovery_run=str(run), recovery_declaration_sha256=sha(run/"declaration.json"),
        split="calibration only: female5/male5; no pilot/final-test access",
        purpose="Post-hoc descriptive checkpoint comparison; does not select, adopt or tune models",
        shared_policy=dict(gain=1., deadband_cents=3.), tolerances_cents=[5.,10.],
        metric="Per-case proportion of fixed audited shifted-plateau frames within tolerance; average cases within singer then singers equally",
        comparison_stage="Raw predicted signed cents; common held-note scope/controller proposals separately, no waveform rendering",
        hgb_reconstruction="One deterministic repetition of original fit, training arrays/settings/weights unchanged; old HGB estimator was not saved",
        hgb_settings=declaration["models"]["hist_gradient_boosting"],
        training_differences="Tree models fit revised VocalSet training singers; MLP and Transformer retain historical training and their own input features/normalizers; not matched architecture training",
        feature_adapter="All read same saved input features via runtime-fine-v2; Transformer uses stored normalizer, MLP its 32 features, trees 31; no targets/reference/support enter predictor",
        exclusions="All 16 attempts retained; six without audited labels are unscorable, not successes",
        source_sha256=sha(__file__), protected_artifacts=unchanged)
    write(out/"declaration.json", protocol)
    pairs = experiment.cached_pairs(read(run/"calibration_pairs.json"))
    train = experiment.cached_pairs(read(run/"train_pairs.json"))
    x, y, weights = experiment.training_arrays(train)
    assert x.shape == (16842,31) and np.isclose(weights.sum(),1.)
    hgb = HistGradientBoostingRegressor(**protocol["hgb_settings"])
    started = time.perf_counter()
    with threadpool_limits(limits=2):
        hgb.fit(x, y, sample_weight=weights)
    fit_seconds = time.perf_counter()-started
    joblib.dump(hgb, out/"reconstructed_hgb.joblib")
    extra = joblib.load(extra_path)["model"]
    old, _ = load_checkpoint(old_path)
    transformer, normalizer, width = load_model(transformer_path)
    models = {"extra_trees": extra, "hist_gradient_boosting": dict(kind="hist_gradient_boosting",estimator=hgb),
              "old_mlp": dict(kind="old_mlp",estimator=old)}
    names = ["unchanged_baseline", "historical_transformer", "old_mlp", "hist_gradient_boosting", "extra_trees"]
    outputs, results = {}, {}
    for name in names:
        rows = []
        for index, pair in enumerate(pairs):
            if "excluded" in pair:
                rows.append(dict(singer=pair["source"]["singer_id"],cents=pair["cents"],excluded=pair["excluded"]))
                continue
            predictions = {}
            for view in ("A", "C"):
                views = pair[view]["views"]
                if name == "unchanged_baseline":
                    pred = np.zeros((3,len(views[0])))
                elif name == "historical_transformer":
                    features = np.stack([normalizer(v[:,:width]) for v in views])
                    with torch.inference_mode():
                        pred = transformer(torch.from_numpy(features)).numpy().astype(np.float64)
                else:
                    with threadpool_limits(limits=2):
                        pred = np.stack([predict_raw(models[name],v) for v in views])
                assert pred.shape == (3,len(views[0])) and np.isfinite(pred).all()
                predictions[view] = pred
                outputs[f"{name}__{index}__{view}"] = pred
            row, curves = score_case(pair, predictions)
            rows.append(row)
            for view in ("A", "C"):
                outputs[f"{name}__{index}__permitted_{view}"] = curves[view]
        summary, by_singer = aggregate(rows)
        results[name] = dict(summary=summary, by_singer=by_singer, rows=rows)
        if name in models:
            original = next(r["summary"] for r in calibration if r["model"]==name and r["operating_policy"]==protocol["shared_policy"])
            for new_key, old_key in [("raw_mae_cents","raw_correction_mae"),
                                     ("permitted_mae_cents","permitted_correction_mae"),
                                     ("original_raw_abs_mean_cents","raw_original_abs_mean"),
                                     ("original_permitted_abs_mean_cents","permitted_original_abs_mean")]:
                np.testing.assert_allclose(summary[new_key],original[old_key],atol=1e-6,rtol=1e-7)
        print(name, json.dumps(summary), flush=True)
    np.savez_compressed(out/"predictions.npz", **outputs)
    inputs = set()
    for split in ("train", "calibration"):
        inputs.add(run/(split+"_pairs.json"))
        for p in read(run/(split+"_pairs.json")):
            if "excluded" not in p:
                folder = Path(p["path_A"]).parent
                inputs.update([Path(p["path_A"]),Path(p["path_C"]),Path(p["target_path"]),
                               folder/"original_features.npz",folder/f"features_{p['cents']:+g}.npz"])
    assert all(sha(p)==digest for p,digest in unchanged.items())
    evidence_out = dict(protocol=protocol, declaration_sha256=sha(out/"declaration.json"),
        results=results, input_hashes={str(p):sha(p) for p in sorted(inputs)},
        checkpoint_hashes={"extra_trees":sha(extra_path),"old_mlp":sha(old_path),"historical_transformer":sha(transformer_path),
                           "reconstructed_hgb":sha(out/"reconstructed_hgb.joblib")},
        predictions_sha256=sha(out/"predictions.npz"), reconstruction_fit_seconds=fit_seconds,
        reconstructed_original_metrics_match=True, original_artifacts_unchanged=True,
        limitations=["Post-hoc calibration comparison, not independent test accuracy",
                     "Historical neural models were not fitted on matched VocalSet training data",
                     "Within-tolerance controlled-correction accuracy is not natural singing correctness",
                     "No rendering/listening or new model adoption in this comparison"])
    write(out/"evidence.json",evidence_out)
    write(out.parent/"current.json",dict(path=str(out),evidence_sha256=sha(out/"evidence.json")))
    print("Comparison saved:",out,flush=True)


if __name__ == "__main__":
    main()
