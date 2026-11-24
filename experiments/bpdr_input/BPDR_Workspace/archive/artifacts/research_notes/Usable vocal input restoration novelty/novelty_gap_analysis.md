# Novelty gap analysis: usable-vocal-input restoration for humming/singing-to-melody-to-chords

Method note: about 19 web searches/fetches on 2026-11-05 using a search tool that returns summarised snippets. Only one page (arXiv 2602.20967) was fetched. Everything else is abstract/snippet level and should be treated as unverified until the full text is read. Absence of a hit means "not found in these searches", not "does not exist". Queries used are listed under each question.

## Q1. Closed-loop, downstream-aware input restoration for singing/humming transcription or QbH (and ASR analogues)

### Takeaway
Downstream-aware enhancement is well established in ASR (joint training, ASR-loss fine-tuning, ASR-guided fusion), but in the singing/humming transcription and QbH literature I found only noise-robust training, noise-robust pitch extraction, and one thesis-level learned denoising front end. I found no work that uses the note extractor's output as a per-clip accept/reject criterion on candidate repairs. Evidence that enhancement can hurt a downstream recognizer is strong in ASR, which supports the "verify every repair" design.

### Cited Findings
- Multi-task SE+ASR training alternates an SE step and an ASR cross-entropy step to update the enhancer, because enhancers add artifacts that hurt ASR (Eskimez et al., 2021) — [arXiv 2106.02896](https://arxiv.org/pdf/2106.02896)
- Downstream-task-aware unified separation uses a task prompt (ASR-dedicated prompt plus an artifact-reducing regularised loss); IWAENC 2026, MERL — [arXiv 2609.11092](https://arxiv.org/pdf/2609.11092), [MERL TR2026-126](https://shadow.merl.com/publications/docs/TR2026-126.pdf)
- Task-aware warping factors in mask-based enhancement let the enhancement degree be set at test time per downstream task — [PolyU](https://research.polyu.edu.hk/en/publications/task-aware-warping-factors-in-mask-based-speech-enhancement/)
- Artifacts, not residual noise, drive ASR degradation; fine-tuning SE with an ASR-level objective lowered WER even when SDR/STOI/PESQ did not improve; enhanced input could be worse than raw noisy input — [arXiv 2311.11599](https://arxiv.org/pdf/2311.11599)
- A Dec 2025 technical report (not peer-reviewed) found raw noisy audio beat MetricGAN+-enhanced audio in all 40 configurations over four modern ASR systems — [arXiv 2512.17562](https://arxiv.org/abs/2512.17562v1)
- Training-free intelligibility-guided observation addition (Interspeech 2026) fuses noisy and enhanced speech with weights derived from intelligibility estimates from the backend ASR. SE and ASR are not modified. The page does not call it closed-loop or iterative — [arXiv 2602.20967](https://arxiv.org/abs/2602.20967). This is the nearest ASR analogue to "use the downstream model as the success signal".
- ROSVOT (ACL 2024) gets robust singing-voice transcription from noise-augmented training (MUSAN), not from a separate restoration front end — [arXiv 2405.09940](https://arxiv.org/html/2405.09940v2)
- Audio-visual singing transcription improves noise robustness using lip movements — [arXiv 2304.12082](https://arxiv.org/pdf/2304.12082)
- QbH noise work: an NTU thesis proposes an autoencoder waveform denoiser ahead of pitch/onset stages plus noise-robust pitch extraction, tested on 17 noise types from -5 to 30 dB. This is a front end for QbH, not closed-loop — [NTU TDR](https://tdr.lib.ntu.edu.tw/handle/123456789/83095). Also Kim et al. noise-robust pitch estimator and Total Variation contour smoothing — [arXiv 2302.04577](https://ar5iv.labs.arxiv.org/html/2302.04577)
- Agentic audio enhancement exists as an emerging area: StrixAE (enhancement under coupled distortions) [arXiv 2609.03414](https://arxiv.org/pdf/2609.03414); AudioFab (tool-selecting audio agent, iterative correction) [arXiv 2512.24645](https://arxiv.org/html/2512.24645v1); SpeechQM-Agent (data-quality agent with replanning on failed steps, ACL Anthology 2026). The search snippets did not confirm that any of them verifies a repair against a downstream task.
- Humming2Music (IJCAI 2023) is the closest full pipeline: humming transcription, melody, chord, accompaniment. Its transcription step fine-tunes some dissonant notes. No input-usability gate was mentioned in the snippet — [IJCAI 2023](https://ijcai.org/proceedings/2023/840)
- Source separation as preprocessing for singing tasks is discussed in a conversion context, with the over-suppression caveat (RobustSVC, via the ROSVOT search results). Not verified at source.

### Inferences
- Principle of "downstream-aware, verified enhancement" is NOT novel (ASR has joint training and ASR-guided fusion). The defensible niche is the application: a per-clip, accept-only-if-the-note-extractor-improves loop, for sung/hummed input, over several repair operators with bounds.
- The ASR evidence that enhancement often hurts is the strongest published justification for the closed-loop verifier. Cite it as motivation, not as the project's finding.
- No transfer of ASR task-aware SE to singing transcription was found. That absence is a motivation, not a guaranteed priority claim.

### Gaps
- Queries used: "task-aware speech enhancement downstream ASR joint training"; "singing voice transcription noisy recordings robustness enhancement preprocessing"; "query by humming noise robustness denoising front-end"; "agentic audio restoration closed-loop verification..."; "speech enhancement hurts ASR artifacts...". Not searched: Google Scholar directly, ISMIR/ICASSP proceedings full-text, TISMIR, DAFx, JAES. 
- Not checked: whether StrixAE/AudioFab have explicit verification loops (full text unread). Not checked: any ISMIR paper on denoising before vocal melody extraction or Demucs-then-transcribe studies for voice (one search returned only a student project with no AMT impact numbers).
- The NTU thesis details came from a search summary only.

## Q2. Usability gating, reject options, recording-quality feedback, time-specific re-record guidance

### Takeaway
Pre-capture or post-capture quality gates with "try again" feedback are standard engineering practice (voice biometrics enrollment, recording apps) but are documented mostly in patents and vendor docs. Selective prediction/abstention exists in ASR and in music audio-language models, but I found none for note extraction/AMT, and none that ties a gate to singing/humming, nor one that uses the downstream extractor's own failure to localise a re-record request.

### Cited Findings
- ASR rejection-threshold adaptation from input quality (SNR, loudness, duration) — patents [US8000962B2](https://patents.google.com/patent/US8000962), [US8190430B2](https://patents.google.com/patent/US8190430B2/en)
- Voice biometrics enrollment: samples rejected for too loud/soft, low SNR, wrong speaking rate, with feedback and a retry loop — USPTO patents e.g. [9917833](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/9917833), [7340042](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/7340042); vendor practice [Talkdesk](https://support.talkdesk.com/hc/en-us/articles/31254755299099-Troubleshooting-Voice-Biometric-Failures)
- Manuscript narration patent: system may prompt the narrator to re-record a particular section when defect level meets a severity threshold, and re-prompt if the new take is still defective — [US10922484](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/10922484). This is prior art (patent) for section-specific re-record prompts, in scripted narration, not singing.
- Real-time impairment feedback while recording (wind, background noise) — [US10665250](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/10665250); sound quality prediction and recording interface — [US11138989B2](https://patents.google.com/patent/US11138989)
- Karaoke app Quanmin K Song reports real-time monitoring of clipping, volume, background noise (engineering blog, not peer-reviewed) — [besthub.dev](https://besthub.dev/articles/technical-guidelines-for-high-quality-mobile-recording-and-audio-processing-in-quanmin-k-song-6577ff62cc92)
- Singing mistake detection for pedagogy gives targeted feedback, but is about performance mistakes, not recording quality — [arXiv 2602.06917](https://arxiv.org/html/2602.06917)
- Non-intrusive speech quality estimators exist (P.563, Quality-Net) — [Quality-Net](https://astro.paperswithcode.com/paper/quality-net-an-end-to-end-non-intrusive); search found no paper linking them to an ASR accept/reject decision evaluated against WER (search-summary statement).
- Abstention: ASR token-level abstention and RAS metric [arXiv 2604.24278](https://arxiv.org/pdf/2604.24278); pseudo-ensemble abstention for music audio-language models, described as the first UQ/abstention study for them [arXiv 2609.04362](https://arxiv.org/pdf/2609.04362)

### Inferences
- A generic "quality gate plus re-record" is prior art (patents, products). Do not claim the gate itself.
- Possibly distinctive: gating by what the note extractor can and cannot recover after bounded repair, and emitting time-localised re-record spans for sung/hummed input. Patent prior art on section-level re-record prompts for narration means the "time-specific" idea alone is weak; it needs the extractor-grounded justification.
- Selective prediction for note-level music extraction appears open (one search only).

### Gaps
- Queries: "speech quality estimation reject option ASR input quality gating non-intrusive"; "enrollment audio quality check voice biometrics reject poor recording re-record feedback"; "automatic music transcription selective prediction abstention confidence rejection"; "singing assessment recording quality detection noise clipping feedback user app"; "where to re-record time-localized feedback...". 
- Not found, not searched deeply: academic singing-assessment papers (e.g., Smule/SingStar-style) with input validity checks; MIR confidence-estimation for melody extraction (e.g., voicing confidence, "melody extraction reliability"); Google/other patents on humming-app gating. Patents were only skimmed through search snippets.

## Q3. Verified, bounded, reversible edits with provenance in MIR pipelines

### Takeaway
Provenance technology exists (C2PA manifests, Merkle chunk provenance, ZK edit proofs, forensic "preserve original, hash copies, log transforms"), but all found work targets authenticity/deepfake/forensics, not MIR input preparation. I found nothing applying it to MIR or tying edit bounds to a downstream-task verifier.

### Cited Findings
- C2PA: signed manifest appended with each edit, ingredients carry own provenance; meant to cover audio. Soft-binding critiques exist — [Fotoware overview](https://www.fotoware.com/blog/what-is-c2pa), [W3C minutes](https://www.w3.org/2024/08/20-me-irc), [HackerNoon critique](https://hackernoon.com/probably-provenance-you-cant-just-slap-a-sticker-on-it). Secondary sources; audio-specific encoding not confirmed. Check c2pa.org spec.
- Forensic audio practice: preserve original, hash working copies, document every transform, compare processed to original — [Hawkeye Forensic](https://hawkeyeforensic.com/?p=4615)
- Cryptographic proofs that edits were permitted without revealing the original: IACR ePrint 2026/1308 — [link](https://eprint.iacr.org/2026/1308)
- MerkleSpeech chunk-level provenance, noting C2PA bindings break under re-encoding — [arXiv 2602.10166](https://arxiv.org/html/2602.10166) (the mapping of this ID to MerkleSpeech is from a search summary; verify)
- Signing embedded in the audio signal — [arXiv 2605.07241](https://arxiv.org/html/2605.07241v1)

### Inferences
- "Preserve original plus hash plus log" is standard forensic hygiene, not novel. Reversibility via keeping the original is trivial. Claim only the specific combination (bounded edit budget, per-edit extractor verification, provenance record consumed by the pipeline), and only if it is actually implemented.
- No evidence MIR pipelines publish such records; this is an absence in a thin search.

### Gaps
- Queries: "C2PA audio provenance edit history tamper-evident manifest"; "audio edit provenance reversible edits cryptographic log speech enhancement original preserved forensic". Not searched: DAFx/AES on non-destructive audio editing, MIR reproducibility/provenance (e.g., ISMIR "provenance" papers), W3C PROV in audio workflows, ASVspoof-style "processed audio" labelling.
- C2PA spec version (2.0 vs 2.1) unresolved.

## Q4. Pitch repair prior art and the project's Boundary-Pitch Disentangled Repair (BPDR)

### Takeaway
Score-free, note-level and context-aware pitch correction is already published (Deep Autotuner 2020, BERT-APC 2025), including a learned note segmenter and a statement that context-aware correction depends on segmentation. Equivariance (PESTO) is published for pitch estimation. Whether "boundary-invariance plus local counter-detuning with supervised preservation anchors" is new could not be determined from these searches; the individual ingredients are not new.

### Cited Findings
- Deep Autotuner (Wager, Tzanetakis, Wang, Kim; ICASSP 2020): score-free notewise pitch-shift prediction from vocal and accompaniment spectrograms, CNN+GRU, trained on 4,702 karaoke performances, preserves intentional pitch variation — [arXiv 2002.05511](https://arxiv.org/pdf/2002.05511)
- BERT-APC (Kim, Na, Choi, Kim; arXiv Nov 2025, preprint): stationary pitch predictor, then context-aware predictor using a repurposed music language model; learned note segmenter with boundary probabilities and greedy NMS because vibrato/glides blur boundaries; learnable detuning augmentation; note-level correction preserving intentional deviations; reported +10.49 pts raw pitch accuracy over ROSVOT on highly detuned samples and MOS 4.32 vs AutoTune 3.22 and Melodyne 3.08 (authors' numbers, self-described first reference-free APC with a symbolic music LM) — [arXiv 2511.20006](https://arxiv.org/pdf/2511.20006)
- PESTO (Riou et al., ISMIR 2023; TISMIR extension 2025): self-supervised transposition-equivariant pitch estimation via pitch-shifted Siamese pairs, under 30k parameters — [arXiv 2309.02265](https://arxiv.org/pdf/2309.02265), [TISMIR ext. arXiv 2508.01488](https://arxiv.org/html/2508.01488v2). The "6868 times fewer parameters" figure in the snippet looks like an error; do not quote.
- Commercial tools (AutoTune, Melodyne) snap to a score or nearest equal-tempered pitch (as described in the Deep Autotuner abstract).

### Inferences
- The project's cited statement that context-aware correction depends on note segmentation is supported by BERT-APC's design (learned segmenter). That supports the problem framing but also means boundary-sensitivity is known.
- BPDR positioning: a defensible, narrow claim is the training objective combination (boundary-invariance, local counter-detuning, supervised preservation anchors) evaluated in the context of downstream note-extraction fidelity, if experiments show it. A claim of "first note-boundary-aware pitch repair" would be contradicted by BERT-APC.
- Mean-teacher consistency, SPICE and equivariance are prior; BPDR cannot claim them. I did not search mean-teacher or SPICE specifically; cite originals (Tarvainen & Valpola 2017; Gfeller et al. SPICE 2020) from memory only after verification.

### Gaps
- Queries: "Deep Autotuner pitch correction singing voice neural network"; "BERT-APC automatic pitch correction singing"; "PESTO pitch estimation self-supervised equivariance Riou ISMIR"; "pitch correction singing note segmentation context-aware intonation correction without score". 
- Not read: BERT-APC full text (check whether it uses boundary-invariance or counter-detuning-like losses; this is the highest-risk overlap). Not searched: SPICE, mean teacher, other pitch-correction papers 2021-2026, "disentangle pitch and boundary" phrasing, DAFx pitch-correction work.

## Q5. Which combinations look unpublished, and what NOT to claim

### Takeaway
Individually, every ingredient has prior art. The combination (extractor-verified bounded local repairs, with refusal and time-specific re-record guidance and preserved provenance, specifically for sung/hummed input feeding a melody-to-chord pipeline) did not surface in about 19 searches, which is weak evidence of novelty, not proof.

### Cited Findings (what is already published, see above)
- ASR-guided enhancement/fusion: [2602.20967](https://arxiv.org/abs/2602.20967), [2106.02896](https://arxiv.org/pdf/2106.02896), [2311.11599](https://arxiv.org/pdf/2311.11599)
- Quality-gate and re-record flows: patents cited in Q2
- Score-free, context-aware pitch correction with note segmentation: [2002.05511](https://arxiv.org/pdf/2002.05511), [2511.20006](https://arxiv.org/pdf/2511.20006)
- Edit provenance and forensic logging: Q3 sources
- Humming-to-music pipelines: [IJCAI 2023](https://ijcai.org/proceedings/2023/840)

### Inferences
Candidate novelty claims, ordered by defensibility (all conditional on the project's own experiments, and phrased as "to our knowledge, not found in a limited search"):
1. Using a note extractor's output as the accept/reject criterion for bounded repairs on vocal (hummed/sung) input, with refusal when no repair passes. Nearest prior: ASR-guided observation addition (no refusal, no singing).
2. Refusal with time-localised re-record guidance derived from extractor-level unreliability on vocal input. Nearest prior: patent for section-level re-record in narration; so claim the grounding, not the idea.
3. Evidence for or against whether speech-style enhancement helps singing/humming note extraction (even a negative result is publishable, given ASR evidence that it often hurts).
4. BPDR objective combination, only if ablations show it matters, and only against BERT-APC and Deep Autotuner as baselines or at least discussion.

Claims NOT to make:
- "First downstream-aware / task-aware enhancement" or "first closed-loop enhancement" (ASR prior art).
- "First pitch-correction method aware of note boundaries/context" (BERT-APC, Deep Autotuner).
- "First usability gate or re-record prompt for audio input" (patents, enrollment systems, narration QA).
- "Novel provenance/reversibility mechanism" (C2PA, forensic practice; reversibility from retaining the original is trivial).
- "First equivariance/consistency approach to pitch" (PESTO, mean teacher).
- Any worldwide priority claim, consistent with the proposal. Also avoid "first" for the full pipeline until a proper search of ISMIR, ICASSP, TISMIR, DAFx, JAES and Google Scholar is done.
- Do not present BERT-APC's results (preprint, self-reported) as established.

### Gaps
- Not searched: Google Scholar and proceedings full-text; patents specific to humming apps; works on "recording-condition-aware" melody extraction; self-supervised reliability estimation for pitch trackers (e.g., CREPE/pYIN confidence-based gating); MIR "input validation" literature; 2026 papers after the first-pass indexes (several 2026 arXiv IDs in this note, 2609.xxxxx, 2604.xxxxx, 2602.xxxxx, were only seen via search snippets and should be opened before citing).
- Recommended follow-ups: read BERT-APC loss section; read StrixAE and AudioFab for verification loops; query "pYIN confidence reliability gating humming", "melody extraction quality estimation", "vocal quality assessment for transcription"; check c2pa.org for audio.
