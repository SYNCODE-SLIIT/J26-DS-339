"""A new, prospectively declared VocalSet long-tone engineering pilot.

Tree families and a frozen old-MLP comparator are selected only on calibration
singers. Pilot-evaluation singers are scored once after the operating point is
locked. Old failed research gates and historical evidence are never rewritten.
"""
from collections import defaultdict
from datetime import datetime
from math import gcd
from pathlib import Path
import hashlib
import json
import time
import uuid
from zoneinfo import ZoneInfo

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from bpdr_input import COMPONENT_ROOT
from bpdr_input.application.saved_results import checksum
from bpdr_input.audio.analysis import extract
from bpdr_input.modeling.local_context_pilot import context_features, load_checkpoint
from bpdr_input.modeling.recovery_model import predict_raw, propose, save_candidate
from bpdr_input.repair.interventions import render_local, audit_shift
from bpdr_input.repair.rule_correction import fine_pitch, fine_hz, segment_notes
from bpdr_input.repair.model_runtime import runtime_views
from bpdr_input.repair.repair_controller import plan_fixture

WORKSPACE = COMPONENT_ROOT / "BPDR_Workspace"
ROOT = WORKSPACE / "results/model_recovery"
DATASET = WORKSPACE / "datasets/vocalset_pilot"
SEED = 339
SHIFTS = (-30., -20., 20., 30.)
POLICIES = [dict(gain=g, deadband_cents=d) for g in (.5, .75, 1.) for d in (3., 5., 8.)]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def source_records(manifest):
    rows = manifest.get("records", manifest.get("files", []))
    if isinstance(rows, dict):
        rows = list(rows.values())
    if len(rows) != 24:
        raise ValueError("The prospectively selected 24 VocalSet files are not all available")
    return rows


def measured(audio):
    features = extract(audio, 22050, COMPONENT_ROOT/"checkpoints")[0]
    fine = fine_pitch(audio, 22050, features["times_seconds"])
    views, reliable, protected, metadata = runtime_views(features, fine)
    return dict(features=features, views=views, reliable=reliable, protected=protected,
                fine=fine, metadata=metadata)


def from_cached(path,feature_path,cache):
    if path not in cache:
        audio,sr=sf.read(path,dtype="float32")
        if sr != 22050:
            raise ValueError("Cached working waveform rate changed")
        with np.load(feature_path) as z:
            features={k:z[k] for k in z.files}
        fine=fine_pitch(audio,sr,features["times_seconds"])
        views,reliable,protected,metadata=runtime_views(features,fine)
        cache[path]=(audio,dict(features=features,views=views,reliable=reliable,protected=protected,fine=fine,metadata=metadata))
    return cache[path]


def cached_pairs(rows):
    result=[];cache={}
    for row in rows:
        if "excluded" in row:
            result.append(row);continue
        p=dict(row)
        folder=Path(p["path_A"]).parent
        p["audio_A"],p["A"]=from_cached(p["path_A"],folder/"original_features.npz",cache)
        p["audio_C"],p["C"]=from_cached(p["path_C"],folder/f"features_{p['cents']:+g}.npz",cache)
        with np.load(p["target_path"]) as z:
            p["target"]=z["target"].astype(np.float32);p["mask"]=z["mask"].astype(bool)
        result.append(p)
    return result


def prepare_source(row, out, index):
    original = Path(row.get("path", row.get("local_path", "")))
    if not original.is_absolute():
        original = DATASET/original
    if checksum(original) != row["sha256"]:
        raise ValueError("Downloaded VocalSet original changed")
    audio, sr = sf.read(original, dtype="float32", always_2d=True)
    selected = int(np.argmax(np.mean(audio.astype(np.float64)**2, axis=0)))
    wave = audio[:, selected]
    length = min(len(wave), round(20.*sr))
    start = 0  # All selected original files already fit the component's 20s scope.
    wave = wave[start:start+length]
    divisor = gcd(sr, 22050)
    if sr != 22050:
        wave = resample_poly(wave, 22050//divisor, sr//divisor).astype(np.float32)
    wave = np.asarray(wave, dtype=np.float32)
    if not np.isfinite(wave).all() or len(wave)/22050 < 5.:
        reason="Source shorter than declared five-second context"
        return dict(source=row, excluded=reason), [dict(excluded=reason,source=row,cents=shift,path_A=str(original)) for shift in SHIFTS]
    if np.max(np.abs(wave)) >= 1:
        reason="Clipped source; no normalization rescue"
        return dict(source=row, excluded=reason), [dict(excluded=reason,source=row,cents=shift,path_A=str(original)) for shift in SHIFTS]
    folder = out/"prepared"/f"{index:02d}"
    folder.mkdir(parents=True)
    baseline = folder/"original_working.wav"
    sf.write(baseline, wave, 22050, subtype="PCM_24")
    wave, _ = sf.read(baseline, dtype="float32")
    before = measured(wave)
    times = before["features"]["times_seconds"]
    np.savez_compressed(folder/"original_features.npz", **before["features"])
    source = dict(source=row, working_audio=str(baseline), working_sha256=checksum(baseline),
                  crop_offset_seconds=start/sr, source_channel=selected,
                  duration_seconds=len(wave)/22050, original_analysis=before["metadata"])
    eligible = []
    for a,b in segment_notes(before["views"][0][:,64],before["reliable"]):
        y = before["views"][0][a:b,64]
        x = np.arange(len(y))
        spread = float(np.std(y-np.polyval(np.polyfit(x,y,1),x))) if len(y)>2 else float("inf")
        if (b-a)*.02 >= 1.36-1e-9 and spread <= 20.:
            eligible.append((a,b,spread))
    if not eligible:
        source["excluded"] = "NO_OWN_AUDIO_STABLE_RUN_WITH_CONTEXT_FOR_096_SECOND_INTERVENTION"
        return source,[dict(excluded=source["excluded"],source=row,cents=shift,path_A=str(baseline)) for shift in SHIFTS]
    a,b,spread = max(eligible,key=lambda v:(v[1]-v[0],-v[0]))
    source["intervention_design_run"] = dict(start_seconds=a*.02,end_seconds=b*.02,detrended_spread_cents=spread,
        use="Intervention creation only; never a model/proposal input")
    pairs = []
    for shift in SHIFTS:
        # Time and width depend on an identity hash, not known pitch/quality.
        seed = int(hashlib.sha256((row["sha256"]+str(shift)).encode()).hexdigest()[:12], 16)
        rng = np.random.default_rng(seed)
        max_width = min(1.12,(b-a)*.02-.40)
        width = float(rng.uniform(.96,max(.96,max_width)))
        earliest = a*.02+.20
        latest = b*.02-.20-width
        left = float(rng.uniform(earliest,max(earliest,latest)))
        support = (left, left+width)
        candidate, _ = render_local(wave, 22050, shift, support)
        path = folder/f"detuned_{shift:+g}.wav"
        sf.write(path, candidate, 22050, subtype="PCM_24")
        candidate, _ = sf.read(path, dtype="float32")
        after = measured(candidate)
        audit, mask, target = audit_shift(before["features"], after["features"],
            fine_hz(wave, 22050, times), fine_hz(candidate, 22050, times),
            before["reliable"] & after["reliable"], support, shift)
        # Failed intervention creation is retained and cannot supply labels.
        write(folder/f"audit_{shift:+g}.json", audit)
        if not audit["passed"]:
            pairs.append(dict(excluded="CONTROLLED_INTERVENTION_AUDIT_FAILED", source=row,
                              cents=shift, support_seconds=support, audit=audit, path=str(path)))
            continue
        target_path = folder/f"targets_{shift:+g}_NOT_MODEL_INPUT.npz"
        np.savez_compressed(target_path, mask=mask, target=target, times=times)
        np.savez_compressed(folder/f"features_{shift:+g}.npz", **after["features"])
        pairs.append(dict(source=row, cents=shift, support_seconds=support,
            A=before, C=after, audio_A=wave, audio_C=candidate, target=target.astype(np.float32), mask=mask,
            path_A=str(baseline), path_C=str(path), target_path=str(target_path),
            group=row["singer_id"], source_id=original.name, audit=audit))
    return source, pairs


def serialise_pair(pair):
    return {k:v for k,v in pair.items() if k not in ("A", "C", "audio_A", "audio_C", "target", "mask")}


def training_arrays(pairs):
    xx, yy, ww = [], [], []
    group_counts = defaultdict(int)
    for p in pairs:
        if "excluded" not in p:
            group_counts[p["group"]] += 1
    for p in pairs:
        if "excluded" in p:
            continue
        m, target = p["mask"], p["target"]
        plateau, outside = m & (target != 0), m & (target == 0)
        weight = 1./len(group_counts)/group_counts[p["group"]]
        for view, wanted, region, share in (("A", np.zeros(len(m)), m, .5),
                                            ("C", target, plateau, .35),
                                            ("C", target, outside, .15)):
            if not region.any():
                continue
            xx.append(context_features(p[view]["views"][0], include_boundary=False)[region])
            yy.append(wanted[region])
            ww.append(np.full(int(region.sum()), weight*share/region.sum()))
    if not xx:
        raise ValueError("No audited original-source training pairs")
    return np.concatenate(xx), np.concatenate(yy), np.concatenate(ww)


def score(model, policy, pairs):
    rows = []
    for p in pairs:
        if "excluded" in p:
            rows.append(dict(group=p["source"]["singer_id"], excluded=p["excluded"], cents=p["cents"]))
            continue
        curves, raw = {}, {}
        details = {}
        for v in ("A", "C"):
            analysis = p[v]
            proposal = propose(analysis["views"], model, dict(operating_policy=policy))
            pred = proposal["operating_predictions"]
            curve = proposal["requested_curve"]
            plan = plan_fixture(p["audio_"+v], 22050, analysis["features"]["times_seconds"], curve,
                pred[1], proposal["disagreement"], proposal["reliable"], analysis["protected"])
            curves[v] = curve if plan["permitted"] else np.zeros(len(curve))
            raw[v] = proposal["raw_predictions"][0]
            details[v] = dict(outcome="IDENTITY" if plan["identity"] else ("PERMITTED_PREVIEW" if plan["permitted"] else "WITHHELD"),
                              reasons=plan["reasons"], intervals_seconds=plan["intervals_seconds"],
                              affected_fraction=plan["affected_fraction"])
        m, target = p["mask"], p["target"]
        plateau = m & (target != 0)
        outside = m & (target == 0)
        reliable_a = p["A"]["reliable"]
        rows.append(dict(group=p["group"], source_id=p["source_id"], cents=p["cents"], path_A=p["path_A"], path_C=p["path_C"],
            target_path=p["target_path"], support_seconds=p["support_seconds"],
            zero_mae=float(np.abs(target[plateau]).mean()),
            permitted_correction_mae=float(np.abs(curves["C"][plateau]-target[plateau]).mean()),
            permitted_response_gain=float((curves["C"][plateau]-curves["A"][plateau]).mean()/target[plateau].mean()),
            raw_correction_mae=float(np.abs(raw["C"][plateau]-target[plateau]).mean()),
            raw_original_abs_mean=float(np.abs(raw["A"][reliable_a]).mean()),
            permitted_original_abs_mean=float(np.abs(curves["A"][reliable_a]).mean()),
            original_edit_frame_fraction=float(np.mean(curves["A"][reliable_a] != 0)),
            permitted_outside_abs_mean=float(np.abs(curves["C"][outside]).mean()),
            permitted_detuned_nonidentity=bool(np.any(curves["C"])),
            controller=details))
    valid = [r for r in rows if "excluded" not in r]
    by_group = {}
    for group in sorted({r["group"] for r in valid}):
        rr = [r for r in valid if r["group"] == group]
        keys = ("zero_mae", "permitted_correction_mae", "permitted_response_gain", "raw_correction_mae", "raw_original_abs_mean",
                "permitted_original_abs_mean", "original_edit_frame_fraction", "permitted_outside_abs_mean", "permitted_detuned_nonidentity")
        result = {k: float(np.mean([r[k] for r in rr])) for k in keys}
        attempted = sum(r["group"] == group for r in rows)
        result["all_attempt_permitted_coverage"] = float(sum(r["permitted_detuned_nonidentity"] for r in rr)/attempted)
        result["attempted"] = attempted
        result["audited"] = len(rr)
        result["permitted_correction_gain"] = 1.-result["permitted_correction_mae"]/result["zero_mae"]
        by_group[group] = result
    if not by_group:
        return dict(passes=False, attempted=len(rows), audited=0, groups=0, by_group={}), rows
    means = {k: float(np.mean([g[k] for g in by_group.values()])) for k in next(iter(by_group.values())) if k not in ("attempted","audited")}
    # These are NEW scoped demo criteria, not modifications to old research gates.
    passed = len(by_group) == 2 and all(g["permitted_correction_gain"] > 0 and g["permitted_response_gain"] >= .1 and
                 g["raw_original_abs_mean"] <= 3. and g["permitted_original_abs_mean"] <= 1. and
                 g["original_edit_frame_fraction"] <= .05 and g["all_attempt_permitted_coverage"] >= .25
                 for g in by_group.values())
    return dict(**means, passes=passed, attempted=len(rows), audited=len(valid),
                intervention_audit_exclusions=len(rows)-len(valid), groups=len(by_group), by_group=by_group,
                interpretation="Permitted curves before rendering; not delivered waveform verification"), rows


def run(resume_from=None):
    import torch
    torch.set_num_threads(2)
    manifest_path = DATASET/"manifest.json"
    manifest = read(manifest_path)
    rows = source_records(manifest)
    out = Path(resume_from).resolve() if resume_from else ROOT/uuid.uuid4().hex
    if not out.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("Recovery output must stay within its workspace folder")
    if not resume_from:
        out.mkdir(parents=True)
    files = [Path(__file__), WORKSPACE/"source/bpdr_input/modeling/recovery_model.py",
             WORKSPACE/"source/bpdr_input/modeling/local_context_pilot.py", manifest_path]
    declaration = dict(version="vocalset-longtone-recovery-2", created_at=datetime.now(ZoneInfo("Asia/Colombo")).isoformat(),
        dataset="VocalSet official preselected straight long tones, vowels a/e, CC-BY4.0",
        groups="Singer IDs1..4fit(8people),5calibration(2people),6pilot_evaluation(2people)",
        feature_interface="31 single-input local fine-pitch context features, runtime-equivalent voicing/confidence; no references,target masks,support intervals or absolute time",
        preprocessing="Retain full originals (all<20s); strongest channel; resample22050; reject<5s or clipped without rescue",
        interventions="Known +/-20,+/-30c local shifts: choose longest OWN-ORIGINAL reliable stable run>=1.36s,spread<=20c; hash-varied.96–1.12s support with>=.20s measuredflanks; no arbitrarysilentgapplacement; librosarenderer;YIN+PESTOaudit; allfailures/exclusionsretained",
        models={"extra_trees": dict(n_estimators=96,max_depth=8,min_samples_leaf=12,random_state=SEED,n_jobs=2),
                "hist_gradient_boosting": dict(max_iter=120,max_leaf_nodes=15,max_depth=5,learning_rate=.08,l2_regularization=1.,min_samples_leaf=30,early_stopping=False,random_state=SEED)},
        comparator="Frozen previous local-context BPDR-loss MLP checkpoint; diagnostic only, not relabelled as successful",
        training_weights="Group-equal; original .50,detuned plateau .35,detuned outside .15; no calibration/evaluation fitting",
        operating_points=POLICIES, scope="Own-audio voiced runs>=1.10s,detrended spread<=20c; existing3c-or-stricter deadband/largestspan and unchanged .86s/25%/30c/5c controller",
        selection="Fit2fixed treefamilies only; score all9operatingpoints plus frozenMLP on calibration2singers; choose highest group-equal permitted gain among practical-gate passers, otherwise best gain diagnosticonly; lock before evaluation source preparation",
        practical_gates="BOTH2cal/eval singers must be represented;eachsinger:positive permitted errorgain,response>=.1,raw originalmean<=3c,permitted originalmean<=1c,original editedframes<=5%,ALL-ATTEMPTdetuned permittedcoverage>=25%; then actualrender improvement for EACH singer and unchanged originalanchors",
        caps="2fixed treefits;120HGB iterations or96trees; max120s perfit; no hyperparameter retries or outertuning",
        old_research_gate_preserved=True, final_reserved_data_access=False, natural_intended_pitch_labels=False,
        supersedes_preparation_only="14a10a583e40497ea2514fd3570a7f79 retained unchanged: randomwholecrop interventions hit silence; longvoiced TRAIN/CAL context diagnosis motivated this protocol correction",
        reused_development_pilot=True,
        source_hashes={str(p):checksum(p) for p in files}, manifest_sha256=checksum(manifest_path),
        limitations=["Task-specific controlled detuning on sustained human vowels, not natural errors or free melody correctness",
                     "Two unseen-in-fit singers reused from prior preparation-failed pilot; development validation, not untouched finaltest,population estimates orBPDR superiority",
                     "Known original and injected support are used only for supervision/audits/scoring, never inference"])
    if resume_from:
        prospective=read(out/"declaration.json")
        if prospective["manifest_sha256"] != checksum(manifest_path) or (out/"locked_selection.json").exists():
            raise ValueError("Technical resume requires unchanged inputs and no prior locked selection")
        write(out/"technical_resume.json",dict(reason="Correct duplicate attempted/audited summary keys; prior two fits completed but calibration summary raised TypeError before any endpoint selection or outer preparation",
            original_declaration_sha256=checksum(out/"declaration.json"),new_source_sha256=checksum(__file__),
            protocol_unchanged=True,hyperparameters_unchanged=True,heavy_extraction_repeated=False,
            model_fits_repeated="Same two deterministic fits, because failed process closed before serializing them"))
        declaration=prospective
        (out/"technical_resume_model_recovery.py").write_bytes(Path(__file__).read_bytes())
    else:
        write(out/"declaration.json", declaration)
    print("RECOVERY_DECLARED", out, flush=True)
    split_rows = defaultdict(list)
    for row in rows:
        split_rows[row["split"]].append(row)
    print("SPLITS", {k:len(v) for k,v in split_rows.items()}, flush=True)
    pairs = {}
    source_summaries = {}
    # Evaluation source acoustics are deliberately not opened until lock below.
    for split in ("train", "calibration"):
        if resume_from:
            pairs[split]=cached_pairs(read(out/(split+"_pairs.json")))
            source_summaries[split]=read(out/(split+"_sources.json"))
            print("RESUMED_PREPARED",split,len(pairs[split]),flush=True)
            continue
        pairs[split], source_summaries[split] = [], []
        for i, row in enumerate(sorted(split_rows[split], key=lambda r: (r["singer_id"],r["vowel"]))):
            source, generated = prepare_source(row, out/split, i)
            source_summaries[split].append(source)
            pairs[split].extend(generated)
            print("PREPARED", split, i+1, "audited", sum("excluded" not in p for p in generated), flush=True)
        write(out/(split+"_sources.json"), source_summaries[split])
        write(out/(split+"_pairs.json"), [serialise_pair(p) for p in pairs[split]])
    x,y,w = training_arrays(pairs["train"])
    models = {}
    fit_receipts = {}
    with threadpool_limits(limits=2):
        for name, params in declaration["models"].items():
            estimator = ExtraTreesRegressor(**params) if name == "extra_trees" else HistGradientBoostingRegressor(**params)
            started = time.monotonic()
            estimator.fit(x,y,sample_weight=w)
            elapsed = time.monotonic()-started
            models[name] = dict(kind=name,estimator=estimator)
            fit_receipts[name] = dict(seconds=elapsed,training_rows=len(y),completed_within_cap=elapsed<=120.)
            print("FIT",name,round(elapsed,2),len(y),flush=True)
    previous = Path(read(WORKSPACE/"results/small_real_pilot/current.json")["path"])/"bpdr.pt"
    old, _ = load_checkpoint(previous)
    models["old_mlp"] = dict(kind="old_mlp",estimator=old)
    calibration = []
    for name, model in models.items():
        for policy in POLICIES:
            summary, scored = score(model, policy, pairs["calibration"])
            calibration.append(dict(model=name,operating_policy=policy,summary=summary))
        print("CALIBRATED",name,flush=True)
    write(out/"calibration_all.json",calibration)
    passing = [r for r in calibration if r["summary"]["passes"] and fit_receipts.get(r["model"],{}).get("completed_within_cap",True)]
    options = passing or calibration
    chosen = max(options,key=lambda r:(r["summary"].get("permitted_correction_gain",-1),-r["summary"].get("raw_original_abs_mean",100),r["model"]))
    selection = dict(**chosen,calibration_gate_passes=bool(passing),locked_before_outer_preparation=True)
    write(out/"locked_selection.json",selection)
    print("LOCKED",json.dumps(selection),flush=True)
    checkpoint = out/"selected_model.joblib"
    selected = models[chosen["model"]]
    save_candidate(checkpoint,selected["estimator"],selected["kind"],chosen["operating_policy"],
        dict(declaration_sha256=checksum(out/"declaration.json"),scope="VocalSet straight sustained-vowel controlled-detuning demonstration",
             requires_user_review=True,model_type=chosen["model"],selection_sha256=checksum(out/"locked_selection.json")))
    split = "pilot_evaluation"
    pairs[split],source_summaries[split] = [],[]
    for i,row in enumerate(sorted(split_rows[split],key=lambda r:(r["singer_id"],r["vowel"]))):
        source,generated = prepare_source(row,out/split,i)
        source_summaries[split].append(source);pairs[split].extend(generated)
        print("PREPARED",split,i+1,"audited",sum("excluded" not in p for p in generated),flush=True)
    write(out/(split+"_sources.json"),source_summaries[split])
    write(out/(split+"_pairs.json"),[serialise_pair(p) for p in pairs[split]])
    outer, outer_rows = score(selected,chosen["operating_policy"],pairs[split])
    write(out/"outer_rows.json",outer_rows)
    write(out/"evidence.json",dict(declaration_sha256=checksum(out/"declaration.json"),selection=selection,
        fit_receipts=fit_receipts,outer=outer,outer_proposal_gate_passes=bool(passing) and outer["passes"],
        rendered_demo_gate_passes=False,checkpoint_sha256=checksum(checkpoint),checkpoint=str(checkpoint),
        actual_render_verification="Pending independent singer outputs via unchanged renderer/audit; no delivered-success claim",
        model_adopted=False,old_research_gate_preserved=True))
    write(ROOT/"current.json",dict(path=str(out)))
    print("RECOVERY_COMPLETE",out,json.dumps(outer),flush=True)
    return out


if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume",type=Path)
    run(parser.parse_args().resume)
