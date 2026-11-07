# Evaluation designs for demonstrating "usable-input improvement" from vocal audio restoration

Note: findings come from search-result snippets/abstracts, not full-text reads, unless stated. Items from my own background knowledge with no fetched source are in Gaps/Inferences and marked UNVERIFIED.

## 1. How synthetic corruptions of clean vocal data are built (noise, reverb, clipping, band-limit, codec, level, music)

### Takeaway
Published recipes are mostly speech-centric (DNS Challenge: public noise + RIR corpora, random SNR, a fraction of clips reverberated). Singing work (ROSVOT) uses simple MUSAN noise mixing; no standard singing/humming robustness benchmark with a full corruption suite was found, so BPDR must define and publish its own grid and report per-corruption results.

### Cited Findings
- ROSVOT (ACL 2024) trains by randomly mixing input waveforms with MUSAN noise and reports gains with noisy inputs (relative pitch-transcription improvement of 17% clean / 23% noisy over the prior best). Paper's noise evaluation seems additive-noise only (reverb not confirmed) — [Robust Singing Voice Transcription Serves Synthesis, ACL 2024 / arXiv 2405.09940](https://arxiv.org/html/2405.09940v2)
- R2-SVC (arXiv Oct 2025, singing voice conversion, not transcription) simulates reverberation, echo, and separation artifacts plus f0 perturbation for robustness — [arXiv 2510.20677](https://arxiv.org/abs/2510.20677v1)
- ICASSP 2021 DNS Challenge: >118,000 real and synthetic RIRs supplied with T60, C50 and isReal metadata per RIR; clean speech 760.53 h — [ICASSP 2021 DNS Challenge, arXiv 2009.06122](https://arxiv.org/pdf/2009.06122)
- A secondary paper trained on DNS data used SNR -5 to 25 dB, ~30% of utterances reverberated, T60 0.3-1.3 s (the authors' choices, NOT the official recipe) — [arXiv 2102.04198](https://arxiv.org/pdf/2102.04198)
- Pitch trackers under noise: CREPE paper reports CREPE beats pYIN at SNR < 10 dB, except brown noise — [CREPE, ICASSP 2018, arXiv 1802.06182](https://arxiv.org/pdf/1802.06182)
- On MIR-1K with music accompaniment, pYIN RPA falls 90.47% clean -> 74.10% at 10 dB -> 14.70% at 0 dB; CREPE 96.41% clean, 84.22% at 0 dB — [HarmoF0, arXiv 2205.01019](https://arxiv.org/pdf/2205.01019)
- Candidate clean bases: vocadito (40 monophonic excerpts, 7 languages, f0 + two note annotation sets; Bittner et al., ISMIR 2021 / arXiv 2110.05580) — [arXiv](https://arxiv.org/pdf/2110.05580); MIR-ST500 (500 pop songs, >160k notes; Wang & Jang, ICASSP 2021; access by request to author) — [NTU record](https://scholars.lib.ntu.edu.tw/entities/publication/060bc457-9f07-4ca5-9e68-071571055f80/full), [repo](https://github.com/york135/singing_transcription_ICASSP2021); HumTrans (1,000 segments, ~56 h, 10 hummers, 44.1 kHz web-recorded, CC BY-NC 4.0, train/valid/test split provided; arXiv Sep 2023) — [arXiv 2309.09623](https://arxiv.org/pdf/2309.09623), [HF](https://huggingface.co/datasets/dadinghh2/HumTrans/blob/main/README.md)
- HumTrans onset/offset ground truth has reported problems; Dynamic HumTrans provides corrected annotations — [arXiv 2410.05455](https://arxiv.org/pdf/2410.05455)

### Inferences
- Recommended grid (design choice, not a cited standard): per-corruption severity ladders (SNR e.g. 20/10/5/0/-5 dB with MUSAN/DEMAND-style noise and music; T60 buckets from real RIRs with RT60 metadata; hard clipping at several thresholds; low-pass 4/8 kHz; codec at low bitrates; gain -30 to -50 dBFS; vocal+accompaniment) plus a small "compound" set, with noise/RIR sources split so test noises and rooms are unseen by any trained component.
- HumTrans is mostly clean studio-ish web recordings, so it is a clean base, not a real-noise test. Because of label issues, use corrected labels or restrict to metrics robust to onset error.
- Frame-level f0 bases (vocadito, MIR-1K) are better for pitch-level checks; note-level bases are weaker (see section 2).

### Gaps
- Official DNS synthesis script values (SNR distribution, reverb probability, clipping) not confirmed; DNS Challenge repo scripts should be read directly.
- Could not source MUSAN (Snyder et al. 2015 arXiv), DEMAND (Thiemann 2013), OpenAIR, BUT ReverbDB (Szoke et al., 2019) via this search; they are cited from background knowledge (UNVERIFIED here, 2015/2013 are also outside the 2018-2026 window except BUT ReverbDB).
- No singing-specific noisy-reverberant robustness benchmark found.
- Whether HumTrans recordings contain background noise was not found.

## 2. Downstream metrics, usability threshold, paired statistics, refusal clips

### Takeaway
Use mir_eval note-level F1 (50 ms onset, 50 cents, offset optional) plus frame RPA/RCA, with a paired per-clip delta (restored minus raw) and clustered bootstrap over singers. Define "usable" as an a priori threshold on the downstream metric relative to the clean-input score, and note annotation noise caps achievable F1.

### Cited Findings
- mir_eval defaults: onset tolerance 0.05 s, pitch tolerance 50 cents, offset tolerance max(0.2 x ref duration, 0.05 s); bipartite matching; precision_recall_f1_overlap and onset_precision_recall_f1 available — [mir_eval transcription docs](https://mir-eval.readthedocs.io/stable/api/transcription.html)
- Vocadito two-annotator note agreement is low: F ~0.64 with offsets, ~0.74 ignoring offsets (table headers unclear in snippet); disagreement mainly from short notes/ornament splitting — [vocadito, arXiv 2110.05580](https://arxiv.org/pdf/2110.05580)
- Some papers argue for wider onset tolerances (100 ms / 500 ms) because of loose annotations (instrument AMT, 2026 preprint) — [arXiv 2606.11903](https://arxiv.org/pdf/2606.11903)
- Bisani & Ney (ICASSP 2004): sentence-level bootstrap gives overly narrow intervals under dependence; use all utterances of a speaker as one resampling unit — [BootLog page](https://www-i6.informatik.rwth-aachen.de/~bisani/bootlog.html)
- Blockwise bootstrap for dependent ASR data (Interspeech 2020) — [arXiv 1912.09508](https://arxiv.org/pdf/1912.09508); block structure is the open issue — [arXiv 2209.05281](https://arxiv.org/pdf/2209.05281)
- With few clusters (30 contributors) cluster-level intervals are correspondingly wide — [arXiv 2609.14542](https://arxiv.org/pdf/2609.14542)

### Inferences
- Usability definitions (pre-register): (a) clip is "usable" if note F1 >= X% of its clean-input F1 or >= absolute floor; (b) report threshold sweep rather than one cutoff to avoid threshold tuning; (c) use the vocadito inter-annotator F (~0.64) as a ceiling-style reference so improvements near that ceiling are not over-read.
- Primary statistic: per-singer mean of paired deltas, bootstrap over singers (cluster), plus Wilcoxon signed-rank on singer means; report the fraction of singers/clips improved, harmed, and unchanged (harm rate is as important as mean gain).
- Refused clips: report both (i) all-clip intention-to-treat result with refused clips scored using the raw input (or as failures) and (ii) accepted-only result; never report accepted-only alone.
- Keep f0 and note extractors frozen and not trained on the restorer's output or test singers; evaluate several extractors (e.g., pYIN, CREPE, a note transcriber) to show effect is not extractor-specific.

### Gaps
- No published singer-held-out paired-bootstrap protocol specific to singing restoration found.
- No standard "usability threshold" for vocal pitch/note extraction in the literature found; thresholds above are proposals.

## 3. Evaluating refusal / gating quality

### Takeaway
Treat the gate as a selective predictor: risk-coverage curve and AURC, but supplement with cost-weighted analysis and operating-point reporting, since AURC alone is criticized.

### Cited Findings
- Selective prediction framework (El-Yaniv & Wiener 2010); risk-coverage curve plots selective risk vs coverage; empirical risk = loss on accepted / coverage — [SelectiveNet, ICML 2019, arXiv 1901.09192](https://arxiv.org/pdf/1901.09192)
- AURC/E-AURC defined by Geifman et al. 2018 (ICLR; original paper not retrieved, second-hand) — [arXiv 2206.02152](https://arxiv.org/pdf/2206.02152)
- Critique: AURC poorly translates working-point risk into an aggregate score; AUGRC proposed (average risk of undetected failures) — [Overcoming Common Flaws in the Evaluation of Selective Classification Systems, arXiv 2407.01032](https://arxiv.org/html/2407.01032v2)
- Rejection learning assumes constant rejection cost c; reject when confidence < 1-c — [Mozannar & Sontag, ICML 2020, arXiv 2006.01862](https://arxiv.org/pdf/2006.01862)
- Cost-sensitive deferral with asymmetric error types — [Alves et al., arXiv 2403.06906](https://arxiv.org/html/2403.06906v3)
- Learning to reject under general metrics — Narasimhan et al., ICLR 2024 (via [snippet source](https://object.cloud.sdsc.edu/v1/AUTH_da4962d3368042ac8337e2dfdd3e7bf3/ml-papers-txt/ICLR/2024/learning_to_reject_meets_longtail_learning__00d79510.txt))

### Inferences
- Define the "positive" for the gate as: restoration would yield usable output. Report a 3x outcome matrix per clip: correctly repaired, wrongly repaired (restoration harms downstream metric), correctly refused (raw unusable and/or restoration would harm), wrongly refused (restoration would have helped). Then compute expected cost = c_harm*P(wrong repair) + c_miss*P(wrong refusal) over a cost ratio sweep (e.g., harm cost 2x-10x miss cost) rather than one chosen ratio.
- Include oracle gate and always-repair / never-repair baselines on the same plot; gate must beat always-repair at matched coverage.
- Report coverage at fixed usable-precision (e.g., coverage when accepted-set harm rate <= 5%) with bootstrap-over-singers CIs; select gate threshold on a validation set of different singers.
- Corrupted-beyond-recovery and non-vocal clips should be explicit "should refuse" strata evaluated separately.

### Gaps
- Original Geifman 2018 AURC paper not directly retrieved.
- No music/singing-specific gating evaluation literature found.

## 4. Avoiding over-claiming; listening tests

### Takeaway
Main traps: oracle references, extractor-own pseudo-labels, tuning on test, synthetic-only evidence, and enhancement that sounds better but degrades the downstream task. Use blinded paired listening only as a secondary perceptual check with clear caveats about small panels.

### Cited Findings
- Speech enhancement can hurt ASR because of artifacts: artifact component is the main cause of degradation (Iwamoto et al., Interspeech 2022, [arXiv 2201.06685](https://arxiv.org/abs/2201.06685)); journal follow-up finds residual noise has limited effect whereas artifacts are particularly harmful, and mixing a scaled copy of the observation back ("observation adding") improves ASR on simulated and real data ([arXiv 2404.14860](https://www.arxiv.org/pdf/2404.14860))
- Generalization gap: Clarity-challenge teams did well on simulated but failed on real data (as described in the CHiME-7 UDASE context); mismatch in corpus/noise/room degrades enhancement — [arXiv 2309.06183](https://arxiv.org/pdf/2309.06183)
- CHiME-7 UDASE: limited correlation between subjective ratings and several supervised non-intrusive metrics; systems reduced noise but increased distortion — [HAL 04430786](https://hal-hprints.archives-ouvertes.fr/INRIA/hal-04430786v1)
- Generative enhancement hallucination tied to SNR (2026, seen only via aggregator; verify at arXiv 2606.02913) — [Emergent Mind](https://www.emergentmind.com/papers/2606.02913)
- Separation metrics for vocals: SDR remained best-performing for vocal estimates and embedding metrics did not correlate for vocals in a large listener study (arXiv 2507.06917) — [link](https://awesomepapers.io/speech-audio/papers/2507.06917); intrusive embedding metrics correlated better in a Surrey singing-separation study, with a caveat about generative models breaking reference-based metrics — [Surrey](https://openresearch.surrey.ac.uk/esploro/outputs/conferencePaper/Towards-Reliable-Objective-Evaluation-Metrics-for/991013165202346)
- SI-SDR can negatively correlate with perceived noisiness when references are noisy — [arXiv 2508.14623](https://arxiv.org/pdf/2508.14623v1)
- MUSHRA (ITU-R BS.1534): paired design allows paired tests/repeated-measures ANOVA; if scores non-Gaussian use non-parametric stats — [AudioLabs tutorial](https://www.audiolabs-erlangen.de/content/resources/aesCodingTutorial/mushra.html)
- Resampling study in TTS MUSHRA: ~30+ listeners and ~30 utterances needed for Spearman >0.95 agreement with full panel; more listeners helps more than more utterances at small sizes — [Rethinking MUSHRA, arXiv 2411.12719](https://arxiv.org/html/2411.12719v3)

### Inferences
- Pitfall checklist: (1) Oracle references: if "clean" reference used for gating threshold/alignment or note labels, label results as oracle-aided. (2) Pseudo-labels from the same extractor used for scoring are circular; score against human annotations (vocadito, MIR-ST500) or the clean-input extractor output only as a labeled secondary proxy, and show both. (3) Freeze all thresholds/hyperparameters on a dev split with different singers, noises and RIRs; touch test once. (4) Always report harm rate and "observation adding"/dry-wet-mix control: if blending the raw signal back matches the restorer, the gain is not from restoration. (5) Include a trivial baseline (spectral gating/high-pass/normalization) and a classical DSP baseline. (6) Show that perceptual metrics (e.g., DNSMOS-style) and pitch metrics can disagree; never use perceptual gain as evidence for downstream usability.
- Listening test with few participants: use blinded paired A/B (preference + "which is more in tune/ same melody") on a pre-registered clip subset, randomized order, hidden reference, anchors for MUSHRA; report participant-level bootstrap CIs and agreement (e.g., Krippendorff alpha / Fleiss kappa -- not sourced), explicitly call it exploratory if < ~20 listeners given the ~30-listener rule of thumb from the TTS study.
- Reviewer agreement for gating: have 2+ reviewers label "usable/not usable" on a stratified sample, report Cohen's kappa with CI, and compare gate vs reviewers.

### Gaps
- No MUSHRA or listening study specific to restoring singing for pitch extraction found.
- Specific minimum listener counts for singing restoration not found.

## 5. Public real-world noisy singing/humming data for a non-synthetic check

### Takeaway
No ideal public set (noisy, phone-recorded, with human pitch/note labels) was found; best partial options are vocadito (mixed devices, small) and Smule-derived DAMP data (large, real, unlabeled). A non-synthetic check can therefore rely on label-free proxies and a small newly annotated subset.

### Cited Findings
- vocadito: many recordings from cell phone or computer microphones; 40 clips with f0/note labels, CC licensed, in mirdata; noisiness unclear — [arXiv 2110.05580](https://arxiv.org/pdf/2110.05580)
- Smule "Intonation" data (4,702 vocal tracks) used in Deep Autotuner, unprocessed real recordings with artifacts, assumed in tune rather than pitch-annotated — [arXiv 2002.05511](https://arxiv.org/pdf/2002.05511)
- DAMP-S-AG (Smule karaoke mobile recordings), Zenodo 3593895; pitch annotation not mentioned — [Zenodo](https://zenodo.org/record/3593895)

### Inferences
- Non-synthetic check options: (a) vocadito subset recorded on phones, with note-F1; (b) DAMP/Smule recordings scored by label-free consistency proxies (e.g., agreement between multiple extractors, pitch stability against known karaoke backing/reference melody if available) — must be labeled proxy; (c) a small in-house set annotated by 2+ musicians with kappa reported; (d) real-vs-synthetic gap table: same metrics on real set vs matched-SNR synthetic.

### Gaps
- Other candidate datasets (e.g., in-the-wild humming sets) not searched/verified; none confirmed to carry both noise and human annotations. DAMP pitch labels unconfirmed.
