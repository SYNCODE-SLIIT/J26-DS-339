"""One prospectively declared, capped real-source controlled-detuning pilot.

No final or reserved test is opened; no extraction, full-dataset fit, retry,
development tuning, default checkpoint mutation, or delivery-policy change.
"""
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import time
import uuid
from zoneinfo import ZoneInfo

import numpy as np
import torch

from bpdr_input import COMPONENT_ROOT
from bpdr_input.application.saved_results import checksum
from bpdr_input.modeling.local_context_pilot import (
    FEATURE_VERSION, FRAME_FEATURES, LocalConfig, LocalContextMLP, context_features,
)
from bpdr_input.modeling.losses import boundary_invariance, counter_detuning, smoothness
from bpdr_input.experiments.evaluation.transfer_study import score_predictions

WORKSPACE = COMPONENT_ROOT / "BPDR_Workspace"
ROOT = WORKSPACE / "results/small_real_pilot"
CONDITIONS = ("augmentation_only", "frame_only", "bpdr")
SEED = 339
UPDATES = 180
MAX_SECONDS_PER_ARM = 90.


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def select_subset(records, groups_per_dataset, development=False):
    """Selection uses identity/sign/corruption, never model scores or feature quality."""
    selected = []
    for dataset in ("vocadito", "jamendo"):
        grouped = defaultdict(list)
        for record in records:
            if record["dataset"] == dataset:
                grouped[record["singer_id"]].append(record)
        used = 0
        for group in sorted(grouped):
            rows = grouped[group]
            signs = {int(np.sign(r["cents"])) for r in rows}
            if not development and signs != {-1, 1}:
                continue
            chosen = []
            for sign in (-1, 1):
                for corruption in (("split",) if development else ("split", "jitter")):
                    candidates = [r for r in rows if np.sign(r["cents"]) == sign and r["corruption"] == corruption]
                    if candidates:
                        chosen.append(min(candidates, key=lambda r: (abs(abs(r["cents"])-20.), r["phrase_id"], r["input_directory"])))
            if development and len(chosen) == 1:
                candidates = [r for r in rows if r not in chosen]
                if candidates:
                    chosen.append(min(candidates, key=lambda r: (abs(abs(r["cents"])-20.), r["phrase_id"], r["corruption"], r["input_directory"])))
            if len(chosen) != (2 if development else 4):
                continue
            selected.extend(chosen)
            used += 1
            if used == groups_per_dataset:
                break
        if used != groups_per_dataset:
            raise ValueError(f"Insufficient deterministic {dataset} source groups")
    return selected


def load_pair(record, frozen, cache):
    inputs = {}
    raw_inputs = {}
    for view in "ABCD":
        path = Path(record["input_directory"]) / (view+"_input.npz")
        if checksum(path) != record["input_sha256"][view]:
            raise ValueError("Cached fine-pitch input changed: "+str(path))
        frozen[str(path)] = checksum(path)
        with np.load(path) as z:
            raw = z["boundary_features_71"].astype(np.float32)
        raw_inputs[view] = raw
        # Cache exact input archives; there is never cross-view/reference context.
        key = frozen[str(path)]
        if key not in cache:
            cache[key] = context_features(raw)
        inputs[view] = cache[key]
    target_path = Path(record["target_path"])
    if checksum(target_path) != record["targets_sha256"]:
        raise ValueError("Controlled-intervention target changed")
    frozen[str(target_path)] = checksum(target_path)
    with np.load(target_path) as z:
        mask = z["common_reliable_mask"].astype(bool)
        detuning = -z["correction_target_cents"].astype(np.float32)
    mask &= (raw_inputs["A"][:, 66] > .5) & (raw_inputs["C"][:, 66] > .5)
    if not (mask & (detuning != 0)).any() or not (mask & (detuning == 0)).any():
        raise ValueError("Selected record has no correction/preservation support")
    for view in "ABCD":
        waveform = Path(record["waveform_paths"][view])
        if str(waveform) not in frozen:
            if checksum(waveform) != record["waveform_sha256"][view]:
                raise ValueError("Original/corrupted source waveform changed")
            frozen[str(waveform)] = checksum(waveform)
    return dict(inputs=inputs, voiced={view: raw_inputs[view][:, 66] > .5 for view in "ABCD"},
                mask=mask, detuning=detuning, record=record)


def batch(pairs, indices, width):
    chosen = [pairs[i] for i in indices]
    length = max(len(p["mask"]) for p in chosen)
    def pad(x):
        return np.concatenate([x, np.zeros((length-len(x), *x.shape[1:]), dtype=x.dtype)])
    return dict(inputs={v: torch.from_numpy(np.stack([pad(p["inputs"][v][:, :width]) for p in chosen])) for v in "ABCD"},
                voiced={v: torch.from_numpy(np.stack([pad(p["voiced"][v]) for p in chosen])) for v in "ABCD"},
                mask=torch.from_numpy(np.stack([pad(p["mask"]) for p in chosen])),
                detuning=torch.from_numpy(np.stack([pad(p["detuning"]) for p in chosen])))


def objective(pred, b, condition):
    # Existing physical-cent objectives, balanced exposure to the short injected
    # plateau and unchanged frames. Original preservation receives fixed 4x
    # emphasis, declared before fitting; no development-derived thresholds.
    mask, det = b["mask"], b["detuning"]
    plateau, outside = mask & (det != 0), mask & (det == 0)
    def mse(value, target, region):
        count = region.sum(1)
        used = count > 0
        return ((((value-target)/30.)**2*region).sum(1)[used]/count[used]).mean()
    zero = torch.zeros_like(det)
    anchor = sum(mse(pred[v], zero, mask) for v in "AB")
    anchor += sum(.5*mse(pred[v], -det, plateau)+.5*mse(pred[v], zero, outside) for v in "CD")/4
    scaled = {v: value/30. for v, value in pred.items()}
    smooth = sum(smoothness(scaled[v], b["voiced"][v])[0] for v in "ABCD")/4
    total = anchor+.1*smooth
    if condition == "bpdr":
        total += boundary_invariance(scaled, mask)[0]+counter_detuning(scaled, det/30., mask)[0]
    return total


def summarise(rows):
    def values(rr):
        keys = ("correction_mae", "zero_mae", "boundary_sensitivity", "unshifted_alteration", "outside_alteration")
        # Independent source groups receive equal weight, not every correlated variant.
        grouped = defaultdict(list)
        for r in rr:
            grouped[r["group"]].append(r)
        means = [{k: float(np.mean([r[k] for r in g])) for k in keys} for g in grouped.values()]
        result = {k: float(np.mean([r[k] for r in means])) for k in keys}
        result.update(records=len(rr), independent_groups=len(grouped),
                      original_recordings=len({r["phrase_id"] for r in rr}),
                      median_response_gain=float(np.median([np.median([r["response_gain"] for r in g]) for g in grouped.values()])))
        result["correction_gain"] = 1.-result["correction_mae"]/result["zero_mae"]
        result["passes"] = result["median_response_gain"] >= .25 and result["unshifted_alteration"] <= 1. and result["correction_gain"] > 0
        return result
    return dict(all=values(rows), by_dataset={ds: values([r for r in rows if r["dataset"] == ds]) for ds in sorted({r["dataset"] for r in rows})})


def evaluate(model, pairs, width):
    model.eval()
    rows = []
    arrays = {}
    with torch.inference_mode():
        for i, pair in enumerate(pairs):
            b = batch(pairs, [i], width)
            pred = {v: model(x, b["voiced"][v])[0].numpy() for v, x in b["inputs"].items()}
            record = pair["record"]
            rows.append(dict(score_predictions(pair, pred), dataset=record["dataset"], group=record["singer_id"],
                             phrase_id=record["phrase_id"], cents=record["cents"], corruption=record["corruption"]))
            for v in "ABCD":
                arrays[str(i)+"_"+v] = pred[v]
    return summarise(rows), rows, arrays


def run():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    base = Path(read(COMPONENT_ROOT / "runs/real_learning_v2/current.json")["path"])
    source_catalogs = {s: base/(s+"_catalog.json") for s in ("training", "development")}
    all_records = {s: read(p)["records"] for s, p in source_catalogs.items()}
    records = {"training": select_subset(all_records["training"], 4),
               "development": select_subset(all_records["development"], 3, development=True)}
    train_groups = {(r["dataset"], r["singer_id"]) for r in records["training"]}
    dev_groups = {(r["dataset"], r["singer_id"]) for r in records["development"]}
    if train_groups & dev_groups:
        raise ValueError("Source-group leakage")
    if {r["waveform_sha256"]["A"] for r in records["training"]} & {r["waveform_sha256"]["A"] for r in records["development"]}:
        raise ValueError("Source-audio leakage")
    out = ROOT / uuid.uuid4().hex
    out.mkdir(parents=True)
    write(out/"selected_records.json", records)
    frozen = {str(p): checksum(p) for p in source_catalogs.values()}
    feature_source = Path(__file__).resolve().parents[2]/"modeling/local_context_pilot.py"
    frozen[str(feature_source)] = checksum(feature_source)
    frozen[str(Path(__file__))] = checksum(__file__)
    # Fully prospectively declare architecture, losses, caps and acceptance gates
    # before reading targets into tensors or performing any optimizer update.
    declaration = dict(version="small-real-context-pilot-1", created_at=datetime.now(ZoneInfo("Asia/Colombo")).isoformat(),
        conditions=CONDITIONS, seed=SEED, updates=UPDATES, device="cpu", seconds_per_arm_cap=MAX_SECONDS_PER_ARM,
        model="LocalContextMLP: 32 hidden,16 hidden,SiLU,physical output gain30; zero-initialized final head; not a Transformer",
        feature_version=FEATURE_VERSION, features="31 fixed-scale single-input acoustic/context features plus boundary; frame-only omits boundary",
        context="Own circular chromatic residual, confidence, explicit voicing, energy/onset, .22/.62/1.22/2.42s rolling contrasts and same-input same-pitch-class endpoint contrast",
        forbidden_inputs=["original reference for the detuned input", "intervention support", "targets", "group identity", "annotation melody"],
        data_selection="32train: first4lexical groups per dataset with both signs, one nearest20c split+jitter eachsign;12development:first3groups per dataset, one sign-balanced split pair where possible;otherwise one extra available variant",
        selected_record_counts={s: len(r) for s, r in records.items()},
        selected_groups={s: sorted({r["dataset"]+":"+r["singer_id"] for r in rr}) for s, rr in records.items()},
        optimizer="AdamW lr.003,decay.01,clip1;batch8; identicalseed/schedule; no development normalization or early stopping",
        objective="Squared anchor/900: A+B unit weights, C/D half correction+half outside each /4; smoothness .1; BPDR adds L_B+L_D on cents/30 from first update",
        gate="Each dataset: group-equal original alteration<=1c, group-median response>=.25, positive correction gain over zero; endpoint only; no gate loosening",
        wall_clock_rule="Stop each arm at180updates or90s, retain incomplete endpoint and mark ineligible; no retries or longer runs",
        feature_normalization="Fixed physical scales; no fitted normalizer", final_test_audio_accessed=False,
        natural_error_truth=False, research_ready_examples=0, model_adopted=False,
        limitations=["Exploratory pending-review original recordings with known added detuning; not natural intended-note truth",
                     "Jamendo artists are singer proxies; dataset separation is not guaranteed across unknown identities",
                     "Development subset previously used by historical experiments; not an untouched confirmatory test",
                     "Original performance is a preservation anchor, not certified correct singing"],
        frozen=frozen, selected_records_sha256=checksum(out/"selected_records.json"))
    write(out/"declaration.json", declaration)
    print("DECLARED", out, json.dumps(declaration["selected_record_counts"]), flush=True)
    cache = {}
    pairs = {s: [load_pair(r, frozen, cache) for r in rr] for s, rr in records.items()}
    write(out/"input_hashes.json", frozen)
    (out/"executed_small_real_pilot.py").write_bytes(Path(__file__).read_bytes())
    (out/"executed_local_context_pilot.py").write_bytes(feature_source.read_bytes())
    rng = np.random.default_rng(SEED)
    schedule = [rng.choice(len(pairs["training"]), 8, replace=False).tolist() for _ in range(UPDATES)]
    write(out/"schedule.json", schedule)
    results = {}
    for condition in CONDITIONS:
        torch.manual_seed(SEED)
        width = FRAME_FEATURES if condition == "frame_only" else FRAME_FEATURES+1
        model = LocalContextMLP(LocalConfig(features=width))
        optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.01)
        history = []
        started = time.monotonic()
        stopped = False
        for step, indices in enumerate(schedule, 1):
            if time.monotonic()-started >= MAX_SECONDS_PER_ARM:
                stopped = True
                break
            model.train()
            b = batch(pairs["training"], indices, width)
            pred = {v: model(x, b["voiced"][v]) for v, x in b["inputs"].items()}
            loss = objective(pred, b, condition)
            if not torch.isfinite(loss):
                stopped = True
                break
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            history.append(dict(update=step, loss=float(loss.detach())))
            if step % 30 == 0:
                print(condition, step, "loss", round(history[-1]["loss"], 5), "seconds", round(time.monotonic()-started, 1), flush=True)
        elapsed = time.monotonic()-started
        endpoint = {}
        for split in ("training", "development"):
            summary, rows, arrays = evaluate(model, pairs[split], width)
            write(out/(condition+"_"+split+"_rows.json"), rows)
            np.savez_compressed(out/(condition+"_"+split+"_predictions.npz"), **arrays)
            endpoint[split] = summary
        complete = not stopped and len(history) == UPDATES
        eligible = complete and all(r["passes"] for r in endpoint["development"]["by_dataset"].values())
        checkpoint = out / (condition+".pt")
        payload = dict(state=model.state_dict(), config=asdict(model.config), feature_version=FEATURE_VERSION,
                       model_type="LocalContextMLP", condition=condition, data_status="exploratory_controlled_real_sources",
                       eligible_for_demo=eligible, declaration_sha256=checksum(out/"declaration.json"))
        torch.save(payload, checkpoint)
        write(out/(condition+"_history.json"), history)
        results[condition] = dict(**endpoint, completed_updates=len(history), completed_budget=complete,
                                  training_seconds=elapsed, model=model.describe(), checkpoint=str(checkpoint),
                                  checkpoint_sha256=checksum(checkpoint), eligible_for_demo=eligible)
        print("ENDPOINT", condition, json.dumps(results[condition]["development"]["by_dataset"]), flush=True)
    unchanged = all(checksum(p) == digest for p, digest in frozen.items())
    if not unchanged:
        raise ValueError("Source input changed during pilot")
    report = dict(declaration_sha256=checksum(out/"declaration.json"), input_hashes_sha256=checksum(out/"input_hashes.json"),
                  results=results, source_files_unchanged=True, final_test_audio_accessed=False,
                  model_adopted=False, eligible_conditions=[c for c in CONDITIONS if results[c]["eligible_for_demo"]],
                  inference_interface="load_checkpoint(path)->model,payload; predict_raw(model, fine_pitch_runtime_71)->physical_cents_curve",
                  interpretation="Known-intervention response and original preservation only; no natural-intention truth or claimed BPDR advantage")
    write(out/"evidence.json", report)
    write(ROOT/"current.json", dict(path=str(out)))
    print("PILOT_COMPLETE", out, "eligible", report["eligible_conditions"], flush=True)
    return out, report


if __name__ == "__main__":
    run()
