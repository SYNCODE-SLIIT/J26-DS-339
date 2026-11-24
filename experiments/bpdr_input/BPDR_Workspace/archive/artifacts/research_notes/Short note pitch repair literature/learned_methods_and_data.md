# Learned approaches and training-data strategies for small pitch-error detection/correction in singing and humming

Scope note: about 15 searches/fetches. Several full PDFs (Deep Autotuner journal version, HumTrans) were not machine-readable, so details come from abstracts, HTML versions, and search summaries. Everything not linked is in Gaps.

## Q1. Published learned singing pitch-correction / intonation-assessment systems: inputs, data sizes, results

### Takeaway
The published systems that actually do correction are all trained on large corpora (about 4.7k karaoke performances, about 510 h) and all use audio-derived features (spectrograms of vocal plus accompaniment, or mel plus pitch), not F0 alone. None is small-data. Intonation-assessment papers I found are weak (macro-F1 0.47 on a low-resource task).

### Cited Findings
- Deep Autotuner (Wager, Tzanetakis, Wang, Kim; ICASSP 2020) predicts note-wise pitch shifts from the relationship between vocal and accompaniment spectrograms, using gated recurrent units on convolutional layers. It was trained on 4,702 amateur karaoke performances "selected for good intonation". It treats pitch as continuous and learns to preserve intentional deviation. The abstract gives no numbers, only "promising performance". — [arXiv 2002.05511](https://arxiv.org/abs/2002.05511)
- The journal-style companion describes a convolutional-GRU model that works score-free, for both vocal and accompaniment. It notes extension to unsupervised autotuning. — [arXiv 1902.00956](https://arxiv.org/pdf/1902.00956) (found via search summary only; full text unreadable)
- BERT-APC (Kim, Na, Choi, Kim; arXiv Nov 2025, venue unverified) is reference-free. A stationary pitch predictor takes a vocal pitch sequence (in semitones) concatenated with an 80-bin mel spectrogram. It is explicitly not F0-only. A context-aware note pitch predictor, built on a repurposed symbolic music language model, infers the intended pitch sequence. — [arXiv 2511.20006](https://arxiv.org/html/2511.20006v1)
- BERT-APC data: 12,287 samples (509.67 h). Splits: train 9,828 (406.43 h), val 1,229 (44.45 h), test 1,230 (58.79 h). Samples were ranked by detuning. The stationary pitch predictor is trained on the in-tune lowest 10%, the note pitch predictor on the middle 80%, and the detuner on the highest-error 10%. — [arXiv 2511.20006](https://arxiv.org/html/2511.20006v1)
- BERT-APC results: listening-test pitch-accuracy MOS 4.32 ± 0.15, versus AutoTune 3.22 ± 0.18 and Melodyne 3.08 ± 0.18 (34 subjects). Expression preservation is comparable (3.80 vs 3.85 for Melodyne). The reported margin over ROSVOT on highly detuned raw pitch accuracy is 10.49 points, per the aitopics summary; I did not verify it in the paper body. — [arXiv 2511.20006](https://arxiv.org/html/2511.20006v1); [aitopics summary](https://aitopics.org/tag/pitch%20correction)
- BERT-APC RPA ablation (moderate / highly detuned): full 94.95 / 89.24; no augmentation 94.94 / 87.41; no context note-pitch predictor (plain rounding) 94.12 / 71.25. So context mattered most on large errors, and augmentation mattered mainly on the heavily detuned set. — [arXiv 2511.20006](https://arxiv.org/html/2511.20006v1)
- BERT-APC stationary-pitch metric: Perceptual Tolerance Rate counts a note as correct within -10 to +15 cents, plus MAE in cents, evaluated on 6,578 manually annotated notes. The paper does not report inter-annotator agreement. — [arXiv 2511.20006](https://arxiv.org/html/2511.20006v1)
- Diff-Pitcher (JHU, 2024) is a generative neural network that corrects off-key singing while preserving timbre. I only saw a press page; no inputs or numbers were verified. — [JHU Hub](https://hub.jhu.edu/2024/02/12/ai-auto-tune-diff-pitcher)
- Kurdish-maqam vocal error detection (arXiv Feb 2026, venue unverified): a two-headed CNN-BiLSTM with attention on log-mel spectrograms detects pitch, rhythm, and modal-drift errors. Validation macro-F1 was 0.468. On a 50-song evaluation, recall was 39.4% and precision 25.8%. Modal-drift F1 was 0.133. This is a useful reminder that learned error detection on small, label-scarce data is weak. — [arXiv 2602.20744](https://arxiv.org/html/2602.20744)
- The general frame-wise cents-error / RMSD summary of intonation, and the split between reference-based and reference-independent assessment, come from an aggregator overview only (secondary). — [emergentmind overview](https://api.emergentmind.com/topics/automatic-singing-assessment)

### Inferences
- A ~21k-parameter model on a few thousand frames from ~17 singers is two to three orders of magnitude below the training data of every published correction system. Their numbers do not transfer as expected performance; only their design choices do.
- Notable design pattern: note-level (not frame-level) correction with the frame-level residual retained. This is how BERT-APC keeps naturalness, and it fits a short-note regime where only a per-note offset is estimated.
- Both Deep Autotuner and BERT-APC define "correct" through data selection (in-tune subsets), not human error labels. This is a weak-label strategy that a small project could copy: treat a good-intonation subset as "reference" behaviour.

### Gaps
- Deep Autotuner quantitative results, ablations (accompaniment on/off), and data-creation details: PDF not readable; not verified.
- Peer-reviewed venues for BERT-APC and the Kurdish paper were not checked.
- Diff-Pitcher paper (arXiv/venue, inputs, results) not retrieved.
- No published "singing voice intonation assessment" paper with an F0-only vs audio-features ablation was found.

## Q2. F0-only versus audio-derived features (mel, SSL, CREPE/PESTO embeddings)

### Takeaway
I found no paper that directly ablates F0-only against audio features for singing pitch-error detection. Evidence is indirect: the strongest systems feed both pitch and mel (BERT-APC, ROSVOT), and PESTO shows a tiny model on CQT can estimate pitch well.

### Cited Findings
- ROSVOT (robust singing voice transcription) feeds mel spectrograms (24 kHz, window 512, hop 128; only the first 30 of 80 bins as a bottleneck) plus log-F0 from RMVPE quantized into 256 classes. Optional word boundaries help: removing word-boundary conditioning drops clean COnPOff F from 77.4 to 70.2. — [arXiv 2405.09940](https://arxiv.org/html/2405.09940v1)
- ROSVOT's noise-robustness training (MUSAN at 6-20 dB, Gaussian noise on log-F0, std 0 to 0.04) leaves clean F at 76.4 but raises noisy-input F from 70.1 to 77.0. — [arXiv 2405.09940](https://arxiv.org/html/2405.09940v1)
- BERT-APC explicitly concatenates pitch (semitones) with an 80-bin mel in its stationary-pitch encoder (see Q1). The paper's ablations I could see do not remove the mel branch. — [arXiv 2511.20006](https://arxiv.org/html/2511.20006v1)
- PESTO: a Siamese network with fewer than 30k parameters, taking constant-Q transform input, trained without labels on a small unlabeled dataset. It evaluates on singing voice and instruments, surpasses self-supervised baselines, and narrows the gap to supervised methods. ISMIR 2023 Best Paper. A later extension (arXiv 2508.01488) swaps CQT for VQT and reports a drastic improvement. — [arXiv 2309.02265](https://arxiv.org/abs/2309.02265); [extension](https://arxiv.org/html/2508.01488v2)
- Dynamic HumTrans uses a CQT with harmonic stacking as input, argued to beat STFT/mel for MIR because its geometry matches Western pitch. — [arXiv 2410.05455](https://arxiv.org/html/2410.05455v1)
- Deep Autotuner uses vocal plus accompaniment spectrograms, so harmony context is part of its signal. — [arXiv 2002.05511](https://arxiv.org/abs/2002.05511)

### Inferences
- An F0 trace holds the deviation itself but not voicing quality, vibrato versus drift, vowel/consonant context, or breath/onset transients. These are exactly what separate a short pitch error from an unstable onset on a short note. This reasoning is mine; no cited ablation supports it directly.
- PESTO shows that a ~30k-parameter model on a log-frequency (CQT) frontend is feasible. The ~21k-parameter model is in the same size class, so a CQT/VQT-style input and transposition-equivariant pretraining are the most transferable ideas.
- HuBERT/wav2vec/MERT embeddings are large (tens to hundreds of millions of parameters). Used as frozen features they add a large input dimension, which is likely to overfit with a few thousand frames from ~17 singers. Unverified here.

### Gaps
- No source found on SSL (HuBERT, wav2vec 2.0, MERT) features for singing pitch-error or intonation detection.
- No source found on CREPE embeddings used for error detection.
- No source found with an explicit F0-only vs mel ablation for intonation.

## Q3. Low-data training strategies: detune augmentation, self-supervision, weak labels, reference conditioning, calibration/abstention

### Takeaway
Synthetic detuning helps but its distribution matters: BERT-APC replaces uniform random detuning with a learned detuner fitted to real detuning patterns. PESTO provides a label-free pitch objective that works with little data. I found no published work on conformal or reject-option methods for pitch editing.

### Cited Findings
- BERT-APC's learnable detuner is a GRU that autoregressively predicts note-wise pitch errors from note pitches and durations. It is trained on the highest-error 10% of data and added to the model input, with probability annealed 0 to 0.4. Baselines were uniform random error in (-0.5, +0.5) and (-1, +1) semitones. Its error distribution looks "closely similar" to real highly detuned data while random detuning differs "substantially". This is a visual comparison only, with no divergence metric. The ablation removed augmentation entirely and did not compare learnable against random detuning. — [arXiv 2511.20006](https://arxiv.org/html/2511.20006v1)
- BERT-APC's own gain from augmentation is small on moderate detune (94.94 → 94.95) and about 1.8 RPA points on highly detuned (87.41 → 89.24). — [arXiv 2511.20006](https://arxiv.org/html/2511.20006v1)
- PESTO's pretext task: compare two pitch-shifted copies of the same audio and learn a transposition-equivariant mapping with a class-based objective and Toeplitz layers; no pitch labels are needed. — [arXiv 2309.02265](https://arxiv.org/abs/2309.02265)
- ROSVOT does not use pitch shift or detune augmentation. It uses noise mixing, Gaussian noise on log-F0, and Gaussian-smoothed (about 80 ms) boundary labels. Its pseudo-labels for singing synthesis reach about 91% of the pitch accuracy of manual annotation (per its introduction). Cross-lingual/out-of-domain drop is large (English SVS model RPA 45.2 versus 96.1 for ground truth on a TONAS-related test). — [arXiv 2405.09940](https://arxiv.org/html/2405.09940v1)
- Dynamic HumTrans reports that HumTrans onset/offset labels are misaligned because subjects hummed along to the melody (a "self-labeling" issue). The authors kept only the 52% (train) and 57% (test) of samples where their heuristic note count matched the dataset's. This is a label-noise cleaning strategy via agreement. — [arXiv 2410.05455](https://arxiv.org/html/2410.05455v1)
- Deep Autotuner: the training set was selected for good intonation, and the model learns to correct incorrect intonation while preserving intentional variation. The mechanism for creating the incorrect examples is not stated on the abstract page. — [arXiv 2002.05511](https://arxiv.org/abs/2002.05511)
- Reference-conditioned vs reference-free: BERT-APC and Deep Autotuner are reference-free (musical context or accompaniment inferred); commercial tools snap to a user-defined score or the nearest equal-tempered pitch (contrast drawn in the Deep Autotuner abstract). — [arXiv 2002.05511](https://arxiv.org/abs/2002.05511)
- Confidence/rejection analogue: RMVPE outputs a voicing confidence and zeroes output below a 0.5 threshold. This is an uncalibrated cutoff, not conformal. — [arXiv 2306.15412](https://arxiv.org/pdf/2306.15412) (via search summary)
- Search for conformal prediction, selective prediction, or abstention applied to pitch estimation, audio editing, or singing returned nothing relevant. — see the search listing at [arXiv 2309.02265](https://arxiv.org/pdf/2309.02265) (no hit; negative result)

### Inferences
- Detune augmentation risk: uniform ±1 semitone random detune is far wider and flatter than real small errors (tens of cents), so a model trained on it will see an easier, differently shaped problem. Matching the empirical error distribution (per-note, duration-dependent, correlated with neighbours) is the published remedy. Because real labeled errors are scarce, one could fit the detune distribution to the few real examples and hold real singers out of any augmentation fit.
- The BERT-APC ablation suggests augmentation is a modest gain, not a rescue, when real data is large. With a few thousand frames the benefit could be larger or the shift risk larger; neither is tested in the literature I found.
- PESTO-style equivariance pretraining of the frontend uses only unlabeled monophonic singing and fits a ~30k-parameter budget; this is the strongest published fit to the project constraint.
- For edit safety, a simple selective rule (act only when predicted correction magnitude and a confidence score exceed thresholds, with thresholds picked on held-out singers) is the practical option. Conformal calibration is a reasonable candidate but I found no precedent to cite.
- Contrastive or anomaly-detection framing (train on in-tune only, flag deviation) is consistent with BERT-APC and Deep Autotuner using in-tune subsets, but I found no paper that frames it as anomaly detection explicitly.

### Gaps
- Contrastive/anomaly-detection framing for singing intonation: no source found.
- Weak labels from listening tests used to train (rather than only evaluate) a model: no source found. BERT-APC uses listening tests for evaluation only.
- Conformal prediction / reject option for pitch editing: no source found.
- Key-or-scale snapping / target-melody conditioning as a learned approach: only commercial-tool contrast found.

## Q4. Humming / query-by-humming / transcription systems: short notes and pitch uncertainty

### Takeaway
Transcription systems handle short notes mostly through post-processing thresholds (minimum note length, boundary merging, DP/HMM smoothing) rather than learning short-note pitch explicitly. Accurate onsets and cents-level pitch for short notes remains weak.

### Cited Findings
- HumTrans (Liu, Li, Li, Shan; Tencent; arXiv 2309.09623, 2023): 500 compositions, 1000 segments, about 56.22 hours, released on Hugging Face with baselines. — [arXiv 2309.09623](https://arxiv.org/pdf/2309.09623) (via search summary; PDF unreadable, so singer count and note statistics not verified)
- Dynamic HumTrans (Gupta et al., arXiv 2410.05455, 2024): CNN on harmonic-stacked CQT predicts one note class per frame, including a dummy silence/boundary class (index 89). A Viterbi-style DP over the affinity matrix forces notes to start and end via the dummy class. Final cleanup enforces a minimum note length, but the actual value is not stated. Training batches use random segments of 5-10 notes. Parameter count not reported. — [arXiv 2410.05455](https://arxiv.org/html/2410.05455v1)
- Dynamic HumTrans results on the cleaned test set (note+onset F1, 50 ms onset tolerance, offsets ignored): octave-invariant 0.673 (ours), 0.604 (MIR-ST500 model), 0.564 (VOCANO), 0.434 (Basic Pitch). Octave-aware: 0.651, 0.361, 0.341, 0.268. — [arXiv 2410.05455](https://arxiv.org/html/2410.05455v1). Note: the extraction reported a "±1 cent" pitch tolerance, which looks implausible (probably a semitone or 50 cents); treat as unverified.
- ROSVOT predicts boundaries with focal loss because positive boundaries are rare (about 2.42 per second), uses a 16x downsample (about 85 ms step, the best setting), and post-processes to remove boundaries that are too close. It does not explicitly discuss very short notes. The threshold mu (default 0.8) trades off granularity. — [arXiv 2405.09940](https://arxiv.org/html/2405.09940v1)
- MusicYOLO (ICASSP 2022) casts onset/offset detection as object detection (YOLOX) and labels pitch by a spectrum peak search. It reports onset F1 94.16 and offset F1 91.35 on ISMIR2014. Sung notes are noted as harder because articulated words cause unstable pitch and blurry boundaries. — [ICASSP 2022 listing](https://rc.signalprocessingsociety.org/conferences/icassp-2022/spsicassp22vid0707)
- Basic Pitch: a lightweight instrument-agnostic model (Spotify, ICASSP 2022) for multi-pitch, note, and pitch-bend estimation. A third-party MLX port lists a default minimum_note_length of 127.7 ms. This suggests that shorter notes are dropped by default; unverified against the official repo. — [Basic Pitch paper](https://arxiv.org/abs/2106.00497); [MLX port](https://huggingface.co/schism-audio/basic-pitch-mlx)
- Omnizart is a general toolbox covering vocal transcription among others; I found no documentation of its vocal short-note behaviour. — [arXiv 2106.00497](https://arxiv.org/abs/2106.00497)
- pYIN/Tony: pYIN uses HMM pitch tracking choosing at most one pitch per frame, and has a note-tracking algorithm on top of the pYIN pitch track, used in Tony. Reference papers: Mauch and Dixon, ICASSP 2014; Mauch et al., TENOR 2015. Minimum-duration and smoothing parameters were not retrieved. — [pYIN project page](https://code.soundsoftware.ac.uk/projects/pyin)

### Inferences
- A model whose job is to repair pitch inside a short note can reuse the HumTrans lesson: dataset onset/offset labels may be shifted by tens of ms, so any per-note supervision should come from a cleaned or self-consistent segmentation, and metrics should tolerate boundary jitter.
- A 5-10 note random-window training scheme (Dynamic HumTrans) is a cheap way to create many contexts from small data.
- Frame hop matters: ROSVOT's best temporal step (about 85 ms) is itself comparable to a short note. Notes near or under that duration are effectively sub-resolution for those models.

### Gaps
- MIR-QBSH-specific handling of short notes and uncertainty: not researched.
- HumTrans singer count and note-duration histogram: PDF unreadable; not verified.
- Official Basic Pitch / Omnizart / Tony minimum-note values: not verified from primary documentation.
- Dynamic HumTrans pitch tolerance (stated as "±1 cent") is likely an extraction error.

## Cross-cutting: what transfers to a ~21k-parameter model, a few thousand frames, ~17 singers
- Transfers: note-level correction with retained frame-level residual (BERT-APC); a PESTO-style equivariant, label-free frontend (about 30k parameters); in-tune subset as reference distribution; distribution-matched synthetic detuning; held-out-singer evaluation with cents-level tolerance (for example -10/+15 cents) and listening-test or human spot checks; DP/HMM smoothing with a minimum-length rule.
- Does not transfer directly: large-corpus accompaniment-conditioned models, symbolic music language models, SSL embeddings of 100M+ parameters, and any reported MOS or F1 values.
- Missing from the literature I found (so flag as open): F0-only versus audio-feature ablations for intonation, conformal or reject-option editing, and short-note-specific benchmarks.
