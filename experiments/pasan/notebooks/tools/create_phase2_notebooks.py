"""Author the two reproducible experiment notebooks (execution is separate)."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]

def make_notebook(name, cells):
    nb=nbf.v4.new_notebook()
    nb.cells=[nbf.v4.new_markdown_cell(s) if kind=='md' else nbf.v4.new_code_cell(s) for kind,s in cells]
    nb.metadata={'kernelspec':{'display_name':'Python (vocal-harmonization)','language':'python','name':'python3'},
                 'language_info':{'name':'python','version':'3.12'}}
    nbf.validate(nb)
    nbf.write(nb,ROOT/name)

setup='''from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

PROJECT = next(p for p in [Path.cwd(), *Path.cwd().parents]
               if (p / "datasets/mtg-jamendo/melody-chord-extraction-pipeline").is_dir())
NOTEBOOKS = PROJECT / "experiments/pasan/notebooks"
sys.path.insert(0, str(NOTEBOOKS))
SNAPSHOT = NOTEBOOKS / "phase2_dataset_v2"
plt.rcParams.update({"figure.figsize": (10, 4), "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .18, "axes.axisbelow": True})
BLUE, GOLD, PINK = "#2864A0", "#B17A12", "#AB5178"
pd.set_option("display.max_columns", 14)
pd.set_option("display.max_rows", 16)'''

make_notebook('03_phase2_clean_dataset.ipynb',[
('md', '''# Phase 2: auditable, transposition-invariant training data

## tl;dr
Run top to bottom. This notebook audits **every extracted track folder**, then saves a new, immutable `phase2_dataset_v2` snapshot. Your existing audit, development notebook, v1 snapshot and checkpoints remain intact. **Executed result:** 1,372 folders audited; 4,446 accepted windows from 373 songs. The executed summary below reports split counts and exclusions.

The unit of supervision is **one chord per detected beat in a non-overlapping eight-bar melody window**. The requested product is a progression; the next notebook converts beat predictions into Roman-numeral segments with durations.

The extraction pipeline supplies **pseudo-labels**, not manually verified harmonic ground truth. Cleaning can establish timing and representation consistency; it cannot prove that an extracted melody note or chord is musically correct.'''),
('md', '''## Context & Methods

Your `02_phase2_dataset_audit` already checked inventory, key agreement, melody coverage, meters and chord alignment. Your `phase2_transformer_development` tested beat queries, temporal cross-attention, weighted cross entropy and melody sensitivity. We retain their strongest ideas: complete bars, eight-bar windows, duration-dominant targets, official artist-separated splits and explicit masks.

Changes here: reproducible paths and hashes; conservative note edits; **train-only multi-label genre vocabulary**; fixed root × quality output space; local, ordered melody features; unknown/ambiguous chords remain masked; no forced key snapping or gap filling.

### Key Assumptions

- A tonic and major/minor mode are available at inference. Normalized C-major/A-minor MIDI is also accepted. The model itself never sees the original key or absolute register.
- Transposition invariance does not mean discarding rhythm or contour. They often distinguish melodies with the same pitch classes. Four ordered slots per beat preserve coarse within-beat order; very fast ornaments are compressed.
- Roman roots use the pipeline's **chromatic semitones relative to tonic**, with major-scale degree names in both modes. Thus A minor's C major chord is `bIII:MAJ`. Chord quality is explicit (`i:MIN`, `V:DOM7`); inversions/extensions outside the submission vocabulary are not modeled.
- The strict first cohort requires melody/chord key agreement and high confidence from both. This uses chord evidence for **dataset selection only**, never model inputs or key selection. It can favor easy, tonic-heavy tracks; scores apply to this selected cohort.
- Unsupported/incomplete meters, uncertain keys, long gaps and insufficient labels are excluded with reasons. Genuine rests, leaps and chromatic tones remain. Source mixtures (vocal plus instrumental fallback) are measured, not hidden.
- Octave-error candidates are flagged but **not repaired by default**. Distinguishing a real leap from an error requires listening or stronger evidence.'''),
('md','### 1. Setup and explicit cleaning policy'),
('code',setup),
('code','''from dataclasses import asdict
from phase2_harmony.data import (CleanConfig, build_snapshot, load_snapshot, verify_snapshot,
                                LABELS, FAMILIES, FILES, melody_features, file_hash)
CONFIG = CleanConfig()
display(pd.Series(asdict(CONFIG), name="value").to_frame())
print("Source:", PROJECT / "datasets/mtg-jamendo/melody-chord-extraction-pipeline/outputs")
print("Snapshot:", SNAPSHOT)
# Change SNAPSHOT to a new version if changing policy or source files.
metadata = build_snapshot(PROJECT, SNAPSHOT, CONFIG)
arrays, examples, metadata = load_snapshot(SNAPSHOT)
track_audit = pd.read_csv(SNAPSHOT / "track_audit.csv")
window_audit = pd.read_parquet(SNAPSHOT / "window_audit.parquet")
display(pd.Series(metadata["summary"], name="observed").to_frame())'''),
('md','''## Data
### 2. Inventory, key reliability and exclusions
`track_audit.csv` covers all folders, including missing and unreadable inputs. Structural/timing checks run on every complete track before key filtering. `window_audit.parquet` covers windows from the strict key cohort; windows are accepted or carry one or more reasons. Source freshness checks validate the melody→key→Roman/chord comparison chain.'''),
('code','''display(track_audit.groupby(["status", "reason"], dropna=False).size().rename("tracks").reset_index())
display(pd.crosstab(track_audit.melody_confidence, track_audit.agreement))
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
track_audit.agreement.fillna("missing").value_counts().plot.barh(ax=axes[0], color=BLUE)
axes[0].set(title="Key agreement: all extracted folders", xlabel="Tracks")
reasons = window_audit.loc[~window_audit.accepted, "rejection"].str.split(",").explode().value_counts()
reasons.sort_values().plot.barh(ax=axes[1], color=GOLD)
axes[1].set(title="Window rejection reasons (can overlap)", xlabel="Windows failing each rule")
plt.tight_layout(); plt.show()
print("Complete strict-key candidates:", (track_audit.status == "candidate").sum())
print("Window retention:", f"{window_audit.accepted.mean():.1%}")'''),
('md','### 3. Cleaning edits, pitch leaps, gaps and extraction sources'),
('code','''actions = pd.read_json(SNAPSHOT / "note_actions.jsonl", lines=True) if (SNAPSHOT / "note_actions.jsonl").stat().st_size else pd.DataFrame()
if len(actions):
    display(actions.action.value_counts().rename("events").to_frame())
    display(actions.head(8))
else:
    print("No event edits or octave candidates under this policy.")
display(track_audit[["raw_notes", "cleaned_notes", "overlapping_notes", "large_leaps", "octave_candidates", "vocal_fraction"]].describe().round(3))
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for accepted, color, label in [(False, GOLD, "Rejected"), (True, BLUE, "Accepted")]:
    part = window_audit[window_audit.accepted == accepted]
    axes[0].hist(part.coverage, bins=np.linspace(0,1,21), histtype="step", linewidth=2, color=color, label=label)
    axes[1].hist(part.gap_beats.clip(upper=32), bins=np.arange(0,34,2), histtype="step", linewidth=2, color=color)
axes[0].set(title="Melody coverage by window", xlabel="Union voiced fraction", ylabel="Windows")
axes[0].legend(frameon=False)
axes[1].set(title="Longest melody rest / missing span", xlabel="Detected beats (capped at 32)", ylabel="Windows")
track_audit.vocal_fraction.dropna().plot.hist(bins=20, ax=axes[2], color=BLUE)
axes[2].set(title="Vocal share of cleaned note events", xlabel="Fraction from game_vocal", ylabel="Tracks")
plt.tight_layout(); plt.show()'''),
('md','''### 4. Splits and train-only genre coverage
Official `split_0` is retained (the manifest calls the middle split `validation`). Each artist must occur in exactly one split. Exact normalized **melody + chord window** duplicates across splits are quarantined; this does not establish the absence of alternate recordings, covers, or near-duplicate audio. Rare/unseen genres map to an explicit unknown token; multiple known genres are all retained.'''),
('code','''assert examples.groupby("artist_id").split.nunique().max() == 1
assert examples.groupby("track_id").split.nunique().max() == 1
assert not examples.duplicated(["track_id", "start_sec", "end_sec"]).any()
assert examples.groupby("content_hash").split.nunique().max() == 1
display(examples.groupby("split").agg(windows=("track_id","size"), songs=("track_id","nunique"), artists=("artist_id","nunique")))
print("Genre vocabulary (training songs only, minimum three songs):", metadata["genre_vocab"])
for split in ["train","validation","test"]:
    ix = examples.split.to_numpy() == split
    print(split, "unknown-only genre windows:", f"{arrays['genres'][ix,0].mean():.1%}")
train_tracks = examples[examples.split == "train"].drop_duplicates("track_id")
from collections import Counter
genre_counts = Counter(g for gs in train_tracks.genres for g in gs)
fig, ax = plt.subplots(figsize=(10, 5))
pd.Series(genre_counts).nlargest(12).sort_values().plot.barh(ax=ax, color=BLUE)
ax.set(title="Genres in training songs (multi-label counts overlap)", xlabel="Songs")
plt.tight_layout(); plt.show()'''),
('md','''## Results
### 5. Target imbalance and vocabulary coverage
Fixed output space: 12 chromatic roots × nine supported qualities = 108 Roman-numeral classes. Rare qualities are preserved. The factorized model can share root/quality evidence even when a particular combination is absent from training. `N`, unsupported chords and ambiguous beat boundaries are ignored in the loss; they are never relabeled as tonic.

The target duration threshold is measured against the **entire beat**, including unannotated/no-chord spans. We do not snap chord boundaries to beats. Root/quality accuracy in the next notebook supplements exact chord accuracy.'''),
('code','''counts = {}
for split in ["train", "validation", "test"]:
    y = arrays["y"][examples.split.to_numpy() == split]
    counts[split] = np.bincount(y[y>=0], minlength=len(LABELS))
    real = arrays["lengths"][examples.split.to_numpy() == split].sum()
    print(split, "valid labels:", int((y>=0).sum()), "/", real,
          "observed classes:", int((counts[split]>0).sum()))
top = np.argsort(counts["train"])[-12:]
fig, axes = plt.subplots(1,2,figsize=(13,5))
for split,color in [("train",BLUE),("validation",GOLD)]:
    values = counts[split] / counts[split].sum()
    axes[0].plot(values[top], np.arange(len(top)), "o", label=split, color=color)
axes[0].set_yticks(np.arange(len(top)), [LABELS[i] for i in top])
axes[0].set(title="Most frequent training chords", xlabel="Fraction of valid beats")
axes[0].legend(frameon=False)
for split,color in [("train",BLUE),("validation",GOLD),("test",PINK)]:
    subset=examples[examples.split==split]
    axes[1].scatter([split]*len(subset), subset.valid_target_fraction, alpha=.07, color=color, s=12)
axes[1].set(title="Supervision retained per accepted window", ylabel="Valid chord fraction", ylim=(0,1.03))
plt.tight_layout(); plt.show()
unseen = (counts["train"] == 0)
for split in ["validation","test"]:
    print(split, "labels with unseen root-quality combination:", int(counts[split][unseen].sum()),
          f"({counts[split][unseen].sum()/counts[split].sum():.2%} of valid beats)")'''),
('md','### 6. Inspect local order and one aligned example'),
('code','''sample_index = int(examples[examples.split == "train"].index[0])
w = examples.iloc[sample_index]
x, y = arrays["x"][sample_index], arrays["y"][sample_index]
n = int(arrays["lengths"][sample_index])
# Reassemble the four chronological pitch-class slots per beat.
chromagram = np.concatenate([x[:n, 15*j:15*j+12][:,None,:] for j in range(4)], axis=1).reshape(n*4,12).T
fig, ax = plt.subplots(figsize=(12,3.5))
im=ax.imshow(chromagram, origin="lower", aspect="auto", extent=(0,n, -.5,11.5), cmap="Blues", vmin=0,vmax=1)
ax.set(title=f"{w.track_id}: local melody evidence, tonic-relative pitch classes", xlabel="Detected beat", ylabel="Semitones from tonic", yticks=range(12))
fig.colorbar(im,ax=ax,label="Relative voiced occupancy")
plt.tight_layout(); plt.show()
display(pd.DataFrame({"beat":np.arange(n),"reference":[LABELS[v] if v>=0 else "MASKED" for v in y[:n]]}).head(16))
# Invariance is checked over a tonic wrap and multiple octaves.
notes = [{"onset_sec":i*.5,"offset_sec":(i+1)*.5,"pitch":p} for i,p in enumerate([71,73,78,70])]
base = melody_features(notes,[0,.5,1,1.5,2],11)
for shift in [-24,-11,1,7,12,25]:
    shifted=[{**note,"pitch":note["pitch"]+shift} for note in notes]
    np.testing.assert_allclose(base,melody_features(shifted,[0,.5,1,1.5,2],(11+shift)%12))
print("Transposition and global octave invariance: passed")'''),
('md','### 7. Snapshot verification and portable outputs'),
('code','''verify_snapshot(SNAPSHOT, verify_raw=True, project_root=PROJECT)
assert arrays["x"].shape[-1] == 77
assert np.isfinite(arrays["x"]).all()
assert arrays["confidence"][arrays["y"]>=0].min() >= CONFIG.min_chord_dominance - 1e-6
display(pd.DataFrame([{"file":p.name,"MB":p.stat().st_size/1e6} for p in sorted(SNAPSHOT.iterdir())]).round(3))
print("Verified source hashes:", len(metadata["raw_sha256"]))
print("Verified snapshot hashes:", len(metadata["files_sha256"]))'''),
('md','''## Takeaways
The saved snapshot includes clean feature arrays (`features.npz`), per-window metadata and boundaries (`examples.parquet`), full window/track audits, cleaned note events, individual edit/flag records and hashes. It is reloadable without the raw audio. Rerunning with unchanged inputs verifies and reuses it; changed inputs or policy require a new version.

This is a conservative structural cleanup, **not a claim that every off-note or erroneous chord has been corrected**. Listen to flagged octave candidates, low-vocal-share tracks and high-overlap tracks before enabling repairs. Do not fill multi-beat gaps or force notes to the diatonic scale. The next notebook determines whether a model learns melody-dependent harmony beyond the mode-only baseline.

Useful follow-up data experiments: compare `strict_key=False` in a separate snapshot; compare vocal-only windows against the current mixed melody; manually annotate a small held-out listening set. Keep the current test split frozen while choosing these experiments from validation evidence.''')])

make_notebook('04_phase2_harmony_transformer.ipynb',[
('md','''# Phase 2: melody + genre + mode → Roman-numeral progression

## tl;dr
This is an executable experiment ladder, using `03_phase2_clean_dataset`'s saved snapshot. It runs a tiny-window memorization probe, two fixed priors, a local melody MLP, a small transformer, and a balanced transformer with root/quality/change auxiliary tasks. A second stage tests two smaller harmonic transformers with multi-scale chroma and chord-tone evidence after inspecting the first ladder. It selects on **validation only**, optionally tunes transition decoding there, freezes a portable inference bundle, and evaluates the selected system on the test split.

**Executed result:** the selected transformer reaches 51.52% validation accuracy and 44.64% test accuracy, below the mode-only priors (53.87% and 45.45%). It improves minority/chord-change metrics, but tonic overprediction remains. The notebook records this limitation rather than claiming the task is solved.

All reported results below come from executed cells. The old notebook's saved temporal experiment achieved ~53.57% validation accuracy and macro-F1 ~0.065, with persistent tonic collapse. Our filtered cohort/features differ slightly, so that number is context, not an exact controlled comparison. The new baselines use exactly the same new examples as the new transformers.'''),
('md','''## Context & Methods

Your earlier note encoder → beat-query cross-attention → beat encoder is a sensible starting point, but its saved attention was close to uniform and pitch shuffling barely affected predictions. This version gives **every beat direct local melody evidence**: four ordered pitch/coverage/contour/onset slots, plus onset chroma and meter. A short convolution precedes two independently initialized self-attention layers. There is no previous ground-truth chord input and no teacher-forcing mismatch.

### Key Assumptions

- Full melody windows are available: attention is bidirectional, suitable for offline harmonization. This is not a streaming causal model.
- Inputs are key- and global-octave invariant but retain rhythm, chromatic pitches and relative contour. Genre is multi-label, fitted only from training-song tags; unseen genres use `<unknown>`.
- Output is one Roman chord per beat, then run-length encoding produces chord order **with durations**. It cannot recover sub-beat chord changes with this target resolution.
- All learned models use the same train-only song-balanced sampler. Capped square-root class weights are an experiment, not assumed to help. The structured version also learns root, quality and change targets; its change head is auxiliary, while decoding uses train-only transition counts.
- Selection score, declared before training: `0.5 × macro-F1 + 0.3 × accuracy + 0.2 × exact-beat change-F1`. No hyperparameter uses the test scores. Root/quality scores, non-tonic accuracy, song-average accuracy, segment edit similarity and input ablations diagnose different failure modes.
- Macro-F1 averages target-supported classes. Change-F1 compares adjacent **valid** beat pairs at exact boundaries. Segment edit similarity collapses repeats inside each contiguous valid region and averages those regions; it is not a standard audio chord-recognition score.
- These are extracted reference chords, not the only valid harmonization of a melody. Automated exact-label accuracy cannot replace listening.

Reference APIs: [PyTorch TransformerEncoderLayer](https://docs.pytorch.org/docs/2.12/generated/torch.nn.TransformerEncoderLayer.html) and [CrossEntropyLoss](https://docs.pytorch.org/docs/2.12/generated/torch.nn.CrossEntropyLoss.html). Actual installed versions are recorded below and in the exported bundle.'''),
('md','### 1. Setup and reproducible controls'),
('code',setup),
('code','''import torch, platform
from dataclasses import asdict
from phase2_harmony.data import load_snapshot, LABELS, FAMILIES, file_hash
from phase2_harmony.model import (ModelConfig, HarmonyModel, seed_everything, device_available,
    make_loader, overfit_probe, train_experiment, baseline_predictions, metrics,
    predict, evaluate, score, transition_log_probs, viterbi, harmonize, load_bundle)
arrays, examples, metadata = load_snapshot(SNAPSHOT)
DEVICE = device_available()
SEED, EPOCHS = 42, 20
REFINEMENT_EPOCHS = 30
RUNS = NOTEBOOKS / "phase2_runs_v2"
RUNS.mkdir(exist_ok=True)
seed_everything(SEED)
print({"python":platform.python_version(), "torch":torch.__version__, "numpy":np.__version__,
       "device":DEVICE, "epochs":EPOCHS, "snapshot":str(SNAPSHOT)})
print("Checkpoint resume: identical configurations reuse/continue prior runs; use a new RUNS for a fresh experiment.")
val_loader = make_loader(arrays, examples, "validation")
train_indices = np.flatnonzero(examples.split.to_numpy() == "train")
val_indices = np.flatnonzero(examples.split.to_numpy() == "validation")'''),
('md','''## Data
### 2. Check the contract and memorize a tiny batch
This probe uses **four training windows only**. Its purpose is to expose alignment, masking, optimization or capacity bugs before judging generalization. A good memorization score does not measure model quality.'''),
('code','''probe = overfit_probe(arrays, examples, len(metadata["genre_vocab"]), device=DEVICE)
display(pd.Series(probe).to_frame("probe"))
assert probe["training_accuracy"] > .95, "Resolve memorization failure before full training."
assert examples.groupby("artist_id").split.nunique().max() == 1
assert arrays["x"].shape[-1] == 77
print("Original tonic/key are absent from model inputs. Mode and normalized melody are retained.")'''),
('md','### 3. Fixed baselines on the same validation population'),
('code','''idx, prior_predictions = baseline_predictions(arrays, examples, "validation")
prior_results = {}
for name, pred in prior_predictions.items():
    prior_results[name] = metrics(arrays["y"][idx], pred, arrays["lengths"][idx],
                                 arrays["mode"][idx], examples.iloc[idx].track_id.to_numpy())
display(pd.DataFrame(prior_results).T[["accuracy","macro_f1","change_f1","non_tonic_accuracy","predicted_tonic_fraction"]].round(4))'''),
('md','''## Results
### 4. Train a controlled experiment ladder
Start with a small local model; add sequence context; then add imbalance compensation and auxiliary tasks.

**Refinement after the first ladder:** initial models still hovered near the mode prior, although shifting melody pitch sharply reduced their validation accuracy. A training-only check found more melody mass on the reference chord tones than on arbitrary chord templates. The two `harmonic_*` runs therefore add chroma context at 1/3/7/15 beats, a learned linear skip and a soft chord-tone template skip, with one 64-wide attention layer and stronger dropout. Non-chord melody tones remain allowed. This refinement used validation evidence only; the test split was still closed. Each checkpoint is selected using the same validation score and patience. Training is deterministic on this CPU run; accelerator kernels can have small numerical differences.

Inspect the implementation in `phase2_harmony/model.py`; `HarmonyModel`, `loss_function`, `metrics`, and `train_experiment` are separate functions so each experiment can be changed independently. No hidden training service is involved.'''),
('code','''EXPERIMENTS = [
    ("local_mlp", ModelConfig(architecture="mlp"), False, False, EPOCHS),
    ("transformer_ce", ModelConfig(), False, False, EPOCHS),
    ("transformer_balanced", ModelConfig(), True, True, EPOCHS),
    ("harmonic_transformer", ModelConfig(width=64,layers=1,dropout=.25,architecture="harmonic_transformer"), False, True, REFINEMENT_EPOCHS),
    ("harmonic_balanced", ModelConfig(width=64,layers=1,dropout=.25,architecture="harmonic_transformer"), True, True, REFINEMENT_EPOCHS),
]
models, histories, checkpoints = {}, [], {}
for name, config, balanced, structured, epochs in EXPERIMENTS:
    model, history, checkpoint = train_experiment(
        name, arrays, examples, len(metadata["genre_vocab"]), RUNS,
        epochs=epochs, balanced=balanced, structured=structured, config=config,
        seed=SEED, device=DEVICE, dataset_fingerprint=file_hash(SNAPSHOT / "metadata.json"))
    models[name], checkpoints[name] = model, checkpoint
    histories.append(history)
history = pd.concat(histories, ignore_index=True)
comparison = pd.DataFrame([{ "model":name, "best_epoch":checkpoints[name]["epoch"],
    "train_accuracy":evaluate(model, make_loader(arrays,examples,"train"),arrays,examples,DEVICE)["accuracy"],
    **evaluate(model, val_loader, arrays, examples, DEVICE)} for name,model in models.items()])
comparison["selection_score"] = comparison.apply(lambda row: score(row), axis=1)
display(comparison[["model","best_epoch","train_accuracy","accuracy","macro_f1","change_f1","non_tonic_accuracy","song_macro_accuracy","selection_score"]].round(4))
comparison.to_csv(RUNS / "validation_comparison.csv", index=False)'''),
('md','### 5. Learning curves and class collapse checks'),
('code','''fig, axes = plt.subplots(1,3,figsize=(14,4))
for (name, group), color in zip(history.groupby("experiment",sort=False), [BLUE,GOLD,PINK,"#657B3A","#765A82"]):
    for ax, metric, title in zip(axes,["accuracy","macro_f1","change_f1"],
                                ["Validation beat accuracy","Validation supported-class macro-F1","Validation chord-change F1"]):
        ax.plot(group.epoch,group[metric],label=name,color=color)
        ax.set(title=title,xlabel="Epoch",ylim=(0,1))
for ax, upper in zip(axes,[.65,.10,.25]):
    ax.set_ylim(0,upper)
axes[0].axhline(prior_results["mode_only"]["accuracy"],color="#444444",ls="--",label="mode-only prior")
axes[0].legend(frameon=False,fontsize=9)
plt.tight_layout(); plt.show()
display(comparison[["model","actual_tonic_fraction","predicted_tonic_fraction","root_accuracy","quality_accuracy","segment_edit_similarity","top3_accuracy"]].round(4))
winner_name = comparison.sort_values(["selection_score","model"],ascending=[False,True]).iloc[0]["model"]
winner = models[winner_name]
print("Frozen architecture/checkpoint winner:", winner_name, "epoch", checkpoints[winner_name]["epoch"])'''),
('md','''### 6. Does the chosen transformer use the melody, genre and mode?
These are held-out validation perturbations on exactly the same windows. Zeroing melody removes pitch, local rhythm and contour; reversing melody changes local order while retaining beat/meter positions; wrong-pitch shifts chroma by five semitones while keeping the supplied tonic fixed. They are diagnostic distribution shifts, not separate training experiments. A small genre effect can mean weak use, noisy tags or limited power; it does not prove genres are irrelevant.'''),
('code','''ablation_rows=[]
for condition in [None,"no_melody","reverse_melody","wrong_pitch","no_genre","wrong_mode"]:
    result=evaluate(winner,val_loader,arrays,examples,DEVICE,ablation=condition)
    ablation_rows.append({"condition":condition or "original",**result})
ablations=pd.DataFrame(ablation_rows)
ablations["accuracy_drop"] = ablations.iloc[0].accuracy - ablations.accuracy
ablations["macro_f1_drop"] = ablations.iloc[0].macro_f1 - ablations.macro_f1
display(ablations[["condition","accuracy","macro_f1","change_f1","accuracy_drop","macro_f1_drop"]].round(4))
ablations.to_csv(RUNS / "validation_ablations.csv",index=False)
fig, ax=plt.subplots(figsize=(10,4))
ax.barh(ablations.condition,ablations.accuracy,color=BLUE)
ax.set(title=f"Input dependence: {winner_name}",xlabel="Validation accuracy",xlim=(0,1))
plt.tight_layout();plt.show()'''),
('md','''### 7. Decode progression continuity using training transitions
Viterbi decoding combines model logits with a train-only, mode-conditioned chord transition prior. Its weight is chosen from the declared small validation grid, including **zero**. No reference masks or reference chord boundaries enter decoding. Smoothing can improve continuity while destroying real changes, so raw and decoded metrics are both retained. If all learned systems fail to beat the mode-only validation baseline, the notebook records that rather than claiming success.'''),
('code','''transition = transition_log_probs(arrays, examples)
raw_pred, val_logits, val_idx = predict(winner,val_loader,DEVICE)
decode_rows=[]
for alpha in [0., .15, .4]:
    pred=viterbi(val_logits,arrays["lengths"][val_idx],arrays["mode"][val_idx],transition,alpha)
    result=metrics(arrays["y"][val_idx],pred,arrays["lengths"][val_idx],arrays["mode"][val_idx],
                   examples.iloc[val_idx].track_id.to_numpy())
    decode_rows.append({"alpha":alpha,"selection_score":score(result),**result})
decoding=pd.DataFrame(decode_rows)
ALPHA=float(decoding.sort_values(["selection_score","alpha"],ascending=[False,True]).iloc[0].alpha)
display(decoding[["alpha","accuracy","macro_f1","change_f1","segment_edit_similarity","selection_score"]].round(4))
print("Frozen decoder weight:",ALPHA)
final_val_pred=viterbi(val_logits,arrays["lengths"][val_idx],arrays["mode"][val_idx],transition,ALPHA)
decoding.to_csv(RUNS / "validation_decoder.csv",index=False)'''),
('md','### 8. Which chords are learned?'),
('code','''valid=arrays["y"][val_idx]>=0
yt,yp=arrays["y"][val_idx][valid],final_val_pred[valid]
support=np.bincount(yt,minlength=len(LABELS)); predicted=np.bincount(yp,minlength=len(LABELS))
correct=np.bincount(yt[yt==yp],minlength=len(LABELS))
per_class=pd.DataFrame({"roman":LABELS,"support":support,"predicted":predicted,
                        "recall":correct/np.maximum(support,1),
                        "f1":2*correct/np.maximum(support+predicted,1)})
display(per_class[per_class.support>0].sort_values("support",ascending=False).head(16).round(3))
per_class.to_csv(RUNS / "validation_per_class.csv",index=False)
top=np.argsort(support)[-12:];pos=np.arange(len(top))
fig,ax=plt.subplots(figsize=(11,5))
ax.barh(pos-.18,support[top]/support.sum(),height=.35,color=BLUE,label="Reference")
ax.barh(pos+.18,predicted[top]/predicted.sum(),height=.35,color=GOLD,label="Prediction")
ax.set_yticks(pos,[LABELS[i] for i in top]);ax.set(title="Validation class distribution: selected decoder",xlabel="Fraction of valid chord beats")
ax.legend(frameon=False);plt.tight_layout();plt.show()'''),
('md','''### 8b. Repeat the frozen architecture across training seeds
These repeats diagnose validation stability after architecture selection. They do not replace the seed-42 checkpoint or choose a seed using the test split. The saved table includes the selection seed for comparison.'''),
('code','''seed_results=[{"seed":SEED,**evaluate(winner,val_loader,arrays,examples,DEVICE)}]
for repeat_seed in [7,123]:
    repeated,_,state=train_experiment(
        f"{winner_name}_seed{repeat_seed}",arrays,examples,len(metadata["genre_vocab"]),RUNS,
        epochs=checkpoints[winner_name]["params"]["epochs"],
        balanced=checkpoints[winner_name]["params"]["balanced"],
        structured=checkpoints[winner_name]["params"]["structured"],config=winner.config,
        seed=repeat_seed,device=DEVICE,dataset_fingerprint=file_hash(SNAPSHOT / "metadata.json"))
    seed_results.append({"seed":repeat_seed,**evaluate(repeated,val_loader,arrays,examples,DEVICE)})
seed_table=pd.DataFrame(seed_results)
display(seed_table[["seed","accuracy","macro_f1","change_f1","non_tonic_accuracy"]].round(4))
seed_table.to_csv(RUNS / "validation_seed_stability.csv",index=False)'''),
('md','''### 9. Freeze the portable inference bundle, then evaluate test once
Only the selected architecture/checkpoint and validation-chosen decoder are tested here, alongside fixed baselines. Test metrics are recorded for reporting, not used to change the configuration. Future model changes should be judged on validation or a new development split; repeated adaptation to these test numbers would invalidate the holdout.'''),
('code','''bundle=RUNS / "selected_bundle"
bundle.mkdir(exist_ok=True)
torch.save({"state_dict":{k:v.detach().cpu() for k,v in winner.state_dict().items()},
            "model_config":asdict(winner.config)},bundle / "model.pt")
np.save(bundle / "transition.npy",transition,allow_pickle=False)
bundle_meta={"winner":winner_name,"epoch":checkpoints[winner_name]["epoch"],
             "alpha":ALPHA,"genre_vocab":metadata["genre_vocab"],"labels":LABELS,
             "clean_config":metadata["clean_config"],"feature_schema":"phase2_harmony.data.melody_features v1, 77 features",
             "dataset_metadata_sha256":file_hash(SNAPSHOT / "metadata.json"),
             "torch":torch.__version__,"numpy":np.__version__,"python":platform.python_version(),
             "model_sha256":file_hash(bundle / "model.pt"),"transition_sha256":file_hash(bundle / "transition.npy"),
             "implementation_sha256":{p.name:file_hash(p) for p in (NOTEBOOKS / "phase2_harmony").glob("*.py")}}
import shutil
(bundle / "phase2_harmony").mkdir(exist_ok=True)
for source in (NOTEBOOKS / "phase2_harmony").glob("*.py"):
    shutil.copy2(source,bundle / "phase2_harmony" / source.name)
(bundle / "metadata.json").write_text(json.dumps(bundle_meta,indent=2))
test_loader=make_loader(arrays,examples,"test")
test_pred,test_logits,test_idx=predict(winner,test_loader,DEVICE)
test_decoded=viterbi(test_logits,arrays["lengths"][test_idx],arrays["mode"][test_idx],transition,ALPHA)
test_results={}
for name,pred in [("selected_raw",test_pred),("selected_decoded",test_decoded),
                  *baseline_predictions(arrays,examples,"test")[1].items()]:
    test_results[name]=metrics(arrays["y"][test_idx],pred,arrays["lengths"][test_idx],arrays["mode"][test_idx],
                               examples.iloc[test_idx].track_id.to_numpy())
test_table=pd.DataFrame(test_results).T
display(test_table[["accuracy","macro_f1","change_f1","non_tonic_accuracy","song_macro_accuracy","segment_edit_similarity"]].round(4))
test_table.to_csv(RUNS / "test_results.csv")
print("Portable bundle:",bundle)'''),
('md','''### 9b. Uncertainty and mode slices for the fixed test result
A paired artist bootstrap resamples held-out artists (2,000 draws) and recomputes the beat-weighted accuracy difference from the mode-only prior. It captures artist sampling variability conditional on these trained weights and pseudo-labels; it does not measure annotation uncertainty. Test slices below are descriptive and do not alter model selection.'''),
('code','''reference=arrays["y"][test_idx];valid=reference>=0
mode_prior=baseline_predictions(arrays,examples,"test")[1]["mode_only"]
artist_rows=pd.DataFrame({"artist":examples.iloc[test_idx].artist_id.to_numpy(),
    "correct_model":((test_decoded==reference)&valid).sum(1),
    "correct_prior":((mode_prior==reference)&valid).sum(1),"beats":valid.sum(1)})
artist_counts=artist_rows.groupby("artist")[["correct_model","correct_prior","beats"]].sum().to_numpy()
rng=np.random.default_rng(2026)
resampled=artist_counts[rng.integers(len(artist_counts),size=(2000,len(artist_counts)))].sum(1)
delta=(resampled[:,0]-resampled[:,1])/resampled[:,2]
ci=np.quantile(delta,[.025,.975])
print(f"Model minus mode-only accuracy: {(test_results['selected_decoded']['accuracy']-test_results['mode_only']['accuracy'])*100:+.2f} pp")
print(f"95% paired artist-bootstrap interval: [{ci[0]*100:+.2f}, {ci[1]*100:+.2f}] pp; {len(artist_counts)} held-out artists")
slices=[]
for mode_name,mode_id in [("major",0),("minor",1)]:
    select=arrays["mode"][test_idx]==mode_id
    slices.append({"mode":mode_name,"songs":examples.iloc[test_idx[select]].track_id.nunique(),
        **metrics(reference[select],test_decoded[select],arrays["lengths"][test_idx][select],
                  arrays["mode"][test_idx][select],examples.iloc[test_idx[select]].track_id.to_numpy())})
display(pd.DataFrame(slices)[["mode","songs","accuracy","macro_f1","change_f1","non_tonic_accuracy"]].round(4))
(RUNS / "test_accuracy_uncertainty.json").write_text(json.dumps({"artist_count":len(artist_counts),"draws":2000,"difference_95ci":ci.tolist()}))'''),
('md','''### 10. Honest sample output and whole-song inference
Load the saved weights in a fresh model and pass **only melody, mode, genre and beat/meter timing**. The example below rebuilds inputs from saved cleaned notes; it does not pass any reference chord to `harmonize`. An eight-bar validation window is used so its alignment can be inspected. For a longer song, inference averages overlapping 48-beat windows, then decodes the full song.

For normalized MIDI, set `normalized=True`: C major uses tonic pitch class 0, A minor uses 9. For original-key MIDI, supply the actual `tonic_pc`. MIDI numbers carry melody contour; only invariant relative features reach the model. If you start with scale degrees rather than MIDI, preserve accidental and octave/contour information when converting them to the reference scale. Meter defaults to 4/4 only when omitted; supply actual beat records for other meters.'''),
('code','''loaded,bundle_meta,loaded_transition=load_bundle(bundle)
# First validation example is deterministic; no selection by prediction quality.
sample_id=int(val_idx[0]);sample=examples.iloc[sample_id]
clean_row=next(json.loads(line) for line in (SNAPSHOT / "cleaned_notes.jsonl").open()
               if json.loads(line)["track_id"] == sample.track_id)
# Recover only the normalization tonic from raw key metadata, never chord labels.
raw_key=json.loads((PROJECT / "datasets/mtg-jamendo/melody-chord-extraction-pipeline/outputs" /
                    sample.track_id / "key.json").read_text())["melody_key"]["selected"]
notes=[{"onset_sec":n["onset_sec"],"offset_sec":n["offset_sec"],
        "midi_pitch_rounded":n["pitch"]} for n in clean_row["notes"]]
# Reuse metrical positions saved as the last five input features.
n=int(arrays["lengths"][sample_id]);bounds=sample.beat_boundaries_sec
num=int(round(arrays["x"][sample_id,0,75]*12));den=int(round(arrays["x"][sample_id,0,76]*8))
beat_records=[{"beat_in_bar":i%num+1,"meter_numerator":num,"meter_denominator":den} for i in range(n)]
result=harmonize(loaded,notes,bounds,sample["mode"],list(sample.genres),bundle_meta["genre_vocab"],
                 tonic_pc=int(raw_key["tonic_pc"]),beats=beat_records,transition=loaded_transition,alpha=ALPHA)
np.testing.assert_allclose(result["logits"], val_logits[0,:n],rtol=1e-4,atol=1e-4)
display(pd.DataFrame(result["progression"]))
print("Chord order:"," → ".join(event["roman"] for event in result["progression"]))
reference_tonic=0 if sample["mode"]=="major" else 9
shift=reference_tonic-int(raw_key["tonic_pc"])
example_input={"notes":[{**note,"midi_pitch_rounded":note["midi_pitch_rounded"]+shift}
                        for note in notes if note["onset_sec"] < bounds[-1] and note["offset_sec"] > bounds[0]],
               "beat_boundaries":bounds.tolist(),"mode":sample["mode"],
               "genres":list(sample.genres),"normalized":True,"beats":beat_records}
(bundle / "example_input.json").write_text(json.dumps(example_input,indent=2))
normalized_result=harmonize(loaded,example_input["notes"],bounds,sample["mode"],list(sample.genres),
    bundle_meta["genre_vocab"],normalized=True,beats=beat_records,transition=loaded_transition,alpha=ALPHA)
np.testing.assert_allclose(result["logits"],normalized_result["logits"],atol=1e-5)
print("Original-key and C-major/A-minor normalized inference agree.")
display(pd.DataFrame({"beat":np.arange(n),
    "reference":[LABELS[v] if v>=0 else "MASKED" for v in arrays["y"][sample_id,:n]],
    "prediction":result["beat_chords"]}).head(20))'''),
('md','### 11. Record the observed result and decision'),
('code','''chosen=decoding.loc[decoding.alpha==ALPHA].iloc[0]
base=prior_results["mode_only"]
melody_drop=float(ablations.loc[ablations.condition=="no_melody","accuracy_drop"].iloc[0])
summary={"winner":winner_name,"epoch":checkpoints[winner_name]["epoch"],"alpha":ALPHA,
         "validation_accuracy":float(chosen.accuracy),"validation_macro_f1":float(chosen.macro_f1),
         "validation_change_f1":float(chosen.change_f1),"mode_only_accuracy":base["accuracy"],
         "melody_ablation_accuracy_drop":melody_drop,"test":test_results,
         "limitation":"architecture selection at one seed; separate validation seed repeats; strict key-selected pseudo-label cohort; listening validation remains necessary"}
(RUNS / "summary.json").write_text(json.dumps(summary,indent=2))
display(Markdown(f"**Selected:** `{winner_name}`, epoch {summary['epoch']}, decoder weight {ALPHA}. "
    f"Validation accuracy **{chosen.accuracy:.1%}**, macro-F1 **{chosen.macro_f1:.3f}**, "
    f"change-F1 **{chosen.change_f1:.3f}**. Mode-only accuracy: **{base['accuracy']:.1%}**. "
    f"Removing melody changes accuracy by **{melody_drop*100:+.1f} percentage points** (original minus ablated). When removing melody falls back to tonic, macro-F1/change-F1 and the wrong-pitch ablation are more informative than accuracy alone."))
if chosen.accuracy <= base["accuracy"]:
    print("Accuracy does not beat the mode prior; assess minority/change improvements and inspect labels before increasing capacity.")
if melody_drop < .03:
    print("No-melody accuracy alone is inconclusive: inspect its macro-F1/change-F1 losses and the wrong-pitch effect before judging melody dependence.")'''),
('md','''## Takeaways
Read the executed comparison and ablations before choosing the next experiment. A higher accuracy with tonic collapse is not a useful progression model; a balanced model can improve minority recall while worsening overall accuracy. Keep raw/decoded scores separate.

This run establishes a tested small-model baseline and inference path. Architecture selection used seed 42; the fixed winner is also repeated at seeds 7 and 123 below for validation stability, without replacing the selected checkpoint. This is not an exhaustive architecture search. Choose future changes using validation, not the reported test values. Larger attention stacks are justified only if small-model overfitting works and the learning curves support more capacity.

If improvements plateau: manually review a stratified sample of tonic runs, key disagreements, chord transitions and extraction-source mixtures; compare vocal-only inputs; run a separately versioned relaxed-key sensitivity experiment; consider a two-beat target grid or joint boundary/duration model if exact chord timing is essential. Chromatic tones, short ornaments and genuine octave leaps should not be erased to manufacture a cleaner-looking dataset.

**Rerun:** select this repository's `.venv` Python kernel, execute notebook 03, then notebook 04. Set a new `RUNS` for changed experiments. Supporting code is in `phase2_harmony/`, and invariant tests are in `tests/test_phase2_harmony.py`.''')])
print('Created notebooks 03 and 04')
