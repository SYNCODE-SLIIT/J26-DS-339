# Restoration methods for short (5-15 s) user-recorded hums/singing: CPU-only Windows feasibility, and effect on pitch/note extraction

Method note: ~19 tool calls (WebSearch + WebFetch). Many search results were tool-generated summaries, not page text. Items marked [fetched] were read from the repo/page by WebFetch; [snippet] were only seen in a search summary; [memory] are my background knowledge, NOT verified this session, and carry no link. Model sizes/RTF were mostly NOT stated on README pages - see Gaps.

## 1. Denoising / enhancement: license, size, CPU speed, Windows pip

### Takeaway
Practical CPU-friendly, permissively licensed candidates are DeepFilterNet (MIT/Apache-2.0, pip, Windows supported, single-thread RTF 0.19 reported on a laptop i5), RNNoise (BSD-3, tiny, 48 kHz, speech-trained), noisereduce (MIT, classical spectral gating) and the heavier generative options (ResembleEnhance MIT, Voicefixer MIT, SGMSE+ MIT). Demucs-denoiser is CC-BY-NC 4.0 and Apollo is CC BY-SA 4.0 (and is a lossy-codec repair model, not a denoiser). Miipher has no confirmed open release. Almost none of these were designed or evaluated on singing.

### Cited Findings
- DeepFilterNet: dual-licensed MIT OR Apache-2.0; supports Linux/macOS/Windows (training Linux-only); install torch+torchaudio first, then `pip install deepfilternet`; building from source needs Rust/maturin. Versions DF (ICASSP 2022), DF2 (IWAENC 2022), DF3 (Interspeech 2023). [fetched] — [Rikorose/DeepFilterNet](https://github.com/Rikorose/DeepFilterNet). README gives no model size/RTF; repo shows no releases listed in the capture.
- DeepFilterNet3 paper reports single-thread RTF 0.19 on a notebook Intel i5-8250U [snippet, secondary summary of the paper] — [arXiv 2305.08227](https://arxiv.org/pdf/2305.08227.pdf). Verify in paper before citing.
- Demucs "denoiser" (Facebook Research, Real-Time Speech Enhancement in the Waveform Domain, Interspeech 2020 [memory]): repo licensed CC-BY-NC 4.0 (non-commercial); pretrained dns48, dns64, master64; README RTF on quad-core 2 GHz i5: H=48 1 thread 0.8, 4 threads 0.6; H=64 1 thread 1.2, 4 threads 1.0; `pip install denoiser`; README recommends Python 3.7; live use officially Mac/Linux only [fetched] — [facebookresearch/denoiser](https://github.com/facebookresearch/denoiser). Weights license not separately stated.
- RNNoise (Xiph): BSD-3-Clause; 48 kHz mono 16-bit; a smaller "little" model exists; model files fetched by autogen.sh; build is shell/autotools (no Windows notes); no pip wrapper mentioned on the page [fetched] — [xiph/rnnoise](https://github.com/xiph/rnnoise). Hybrid DSP+small RNN, designed for speech (design description from [memory]).
- noisereduce: MIT; spectral gating, stationary (threshold from noise clip or the signal) and non-stationary (time-smoothed threshold) modes; `pip install noisereduce`; optional PyTorch backend, `n_jobs`, `chunk_size` [fetched] — [timsainb/noisereduce](https://github.com/timsainb/noisereduce). No RTF in README (pure NumPy/FFT is cheap for 5-15 s clips [inference]).
- ResembleEnhance: MIT; `pip install resemble-enhance`; two stages (denoiser, then enhancer that "restores distortions and extends bandwidth"); `--denoise_only` flag; README silent on CPU, size, Windows [fetched] — [resemble-ai/resemble-enhance](https://github.com/resemble-ai/resemble-enhance). Hugging Face card also MIT [snippet]. A third-party MLX port lists a 41 MB denoiser UNet and 390 MB autoencoder file, and is CC-BY-NC-4.0 [snippet; port, not upstream] — [HF multimodalart port](https://huggingface.co/multimodalart/resemble-enhance-mlx). Repo had only 13 commits at capture (low activity).
- Voicefixer: MIT; `pip install voicefixer`; CPU works (`cuda=False`, CPU docker); 44.1 kHz vocoder, accepts 2-44.1 kHz input; Mode 2 may help "seriously degraded real speech"; 40 open issues; size not stated [fetched] — [haoheliu/voicefixer](https://github.com/haoheliu/voicefixer). Targets speech, general speech restoration (denoise, dereverb, declip, bandwidth extension) [memory - the paper "VoiceFixer: A Unified Framework..." is 2021, check venue].
- SGMSE+: MIT; Python 3.11 tested; checkpoints for VB-DMD, WSJ0-CHiME3, 48 kHz EARS-WHAM (denoise) and EARS-Reverb, WSJ0-REVERB (dereverb, use `--N 50 --snr 0.33`), plus a "Singing-ReverbFX" SGMSE+/Schrödinger-bridge checkpoint for singing reverb; no size/speed/CPU info on page [fetched] — [sp-uhh/sgmse](https://github.com/sp-uhh/sgmse). Diffusion needs many reverse steps, so CPU latency likely high [inference].
- Apollo (Li et al., ICASSP 2025, arXiv 2409.08514): CC BY-SA 4.0; repairs lossy MP3-compressed music (bandwidth/codec restoration), 44.1 kHz; device=auto falls back to CPU, chunked inference; no CPU timings in text [fetched] — [JusperLee/Apollo](https://github.com/JusperLee/Apollo). Not a noise suppressor.
- Miipher (Google, WASPAA 2023) and Miipher-2 (2025, arXiv 2505.04457): project page claims RTF 0.0078 (on accelerators, not CPU); no license for Google weights found; only unofficial reimplementations (Atotti/miipher-2, Open-Miipher-2 with no weights) [snippet] — [Miipher-2 page](https://google.github.io/df-conformer/miipher2), [arXiv](https://arxiv.org/pdf/2505.04457v1). Treat as not usable.
- UVR / MDX-Net vocal separation: UVR GUI code is MIT per mirrors; Kim_Vocal_2 weights have no stated license upstream; the PyPI `audio-separator` license could not be confirmed [snippet] — [HF Kim_Vocal_2 GGUF](https://huggingface.co/thepatch/mdx-net-kim-vocal-2-GGUF). Designed for music-with-accompaniment separation, not room-noise denoising [inference].
- NVSR (neural vocoder bandwidth extension, Liu et al.): not researched; no source retrieved.

### Inferences
- For a CPU-only Windows tool with clear licensing, shortlist = noisereduce (classical) + DeepFilterNet (MIT/Apache). Demucs-denoiser (NC) and Apollo (SA, wrong task) are poor fits; Miipher is unavailable.
- Heavier generative models (ResembleEnhance, Voicefixer, SGMSE+) are runnable on CPU for 5-15 s clips, but RTF is unmeasured here; a local benchmark is required before claiming "plausible".

### Gaps
- Official model sizes and CPU RTF for DF3, Voicefixer, ResembleEnhance, SGMSE+, RNNoise were not on README pages; need a local benchmark or paper tables.
- DeepFilterNet3 RTF figure, PyPI latest version/Python-version constraints, and maintenance recency unverified.
- NVSR, Voicefixer paper venue, and Demucs-denoiser paper details were not looked up.
- Whether Windows `pip install deepfilternet` actually works on current Python (wheels vs Rust build) is untested.

## 2. Declipping, dereverberation, normalization, bandwidth extension: classical vs verifiable

### Takeaway
Declipping has a mature, classical, mathematically "consistent" literature (C-IHT, SPADE, weighted l1) with a public benchmark/survey (Záviška et al., JSTSP 2021); these methods enforce that unclipped samples are preserved and clipped samples stay beyond the clip level, which makes them verifiable. Dereverberation has classical WPE (nara_wpe, MIT) but it is designed for multichannel/long-reverb speech; neural dereverb (SGMSE+ variants, Voicefixer, ResembleEnhance) is not verifiable by construction. Level normalization is trivially reversible and verifiable.

### Cited Findings
- Consistent IHT for declipping (Kitić, Bertin, Gribonval; arXiv 1303.1023, ~2013): sparsity-regularized, keeps reconstruction consistent with clipping constraints [snippet] — [arXiv](https://arxiv.org/pdf/1303.1023).
- A-SPADE/S-SPADE: "Sparsity and cosparsity for audio declipping: a flexible non-convex approach" (Kitić et al., 2015); proves bounded termination of SPADE algorithms [snippet] — [arXiv 1506.01830](https://arxiv.org/pdf/1506.01830).
- Survey: Záviška, Rajmic, Ozerov, Rencker, "A Survey and an Extensive Evaluation of Popular Audio Declipping Methods", IEEE JSTSP vol.15 no.1, 2021; evaluates with SDR and perceptual metrics; repository of evaluated methods accompanies it [snippet] — [arXiv 2007.07663](https://arxiv.org/abs/2007.07663v2). Which method ranked best was NOT retrieved.
- Psychoacoustically weighted l1 declipping (Záviška et al., arXiv 1905.00628): claims it can compete or outperform the state of the art in SDR and PEMO-Q ODG at low complexity [fetched abstract] — [arXiv](https://arxiv.org/abs/1905.00628).
- nara_wpe: MIT; Weighted Prediction Error blind dereverberation (Yoshioka & Nakatani 2012 formulation); offline, block-online, recursive variants; NumPy and TensorFlow; tested Py3.7-3.10; `pip install nara_wpe` [fetched] — [fgnt/nara_wpe](https://github.com/fgnt/nara_wpe). Single-channel support not stated on the page.
- Neural dereverb options with singing relevance: SGMSE+ repo ships "Singing-ReverbFX" checkpoints (SGMSE+ and Schrödinger bridge) [fetched] — [sp-uhh/sgmse](https://github.com/sp-uhh/sgmse). Associated paper not retrieved.
- Bandwidth extension: ResembleEnhance enhancer and Voicefixer claim it (above); Apollo targets codec band-limited music. No evidence retrieved on effect on pitch.

### Inferences
- Pitch trackers mostly use content below ~1-2 kHz-ish for F0 but harmonics help; bandwidth extension adds hallucinated content that cannot help F0 and is the lowest-value, highest-risk step [inference, no source].
- Declipping is the only restoration class here with hard, checkable consistency constraints.

### Gaps
- No timings for SPADE/C-IHT on 5-15 s clips (SPADE is known to be iterative and slow in MATLAB/Python implementations [memory]); survey numbers not read.
- No source on single-channel WPE efficacy on hums, or on loudness-normalization effects on pitch trackers (note: trackers like CREPE normalize input internally [memory]).
- Pip availability of a maintained Python SPADE/C-IHT implementation not checked.

## 3. Does enhancement help or hurt pitch tracking / singing transcription?

### Takeaway
Direct evidence is thin: no retrieved study measures a pitch tracker before/after neural enhancement on singing. Speech-side evidence says enhancement helps clean-trained trackers on noisy input, but enhancement distortion limits gains, and generative enhancers are documented to hallucinate and alter acoustic properties like pitch. Singing-transcription work suggests noise-aware training beats separate front-end denoising.

### Cited Findings
- Zhang, Wang, Wang, "Densely-connected Convolutional Recurrent Network for F0 Estimation in Noisy Speech", Interspeech 2022 pp.401-405: extracting pitch from enhanced speech with a clean-trained model "functions reasonably well", but distortion from the enhancement model limits performance; jointly trained enhancement+pitch module performs better; complex STFT input beats magnitude [snippet/search summary; PDF fetch unreadable] — [paper PDF](https://pnlwang.github.io/papers/ZWW.interspeech22.pdf), [ISCA](https://www.isca-archive.org/interspeech_2022).
- An IEEE-indexed older study found only a subspace enhancement approach improved F0 detection and voiced/unvoiced segmentation among tested enhancers (speech) [snippet; details unverified] — [infona record](https://www.infona.pl/resource/bwmeta1.element.ieee-art-000007010129).
- CREPE (Kim et al., ICASSP 2018) vs pYIN under additive noise: CREPE most accurate at nearly all SNRs for pub, white noise, and all but highest pink-noise level; brown noise is the exception (pYIN barely affected); CREPE better whenever SNR < 10 dB [snippet] — [arXiv 1802.06182](https://arxiv.org/pdf/1802.06182). Implication: a robust neural tracker may not need enhancement.
- 2025 preprint SwiftF0 claims robustness at 10 dB SNR with far fewer resources than CREPE [snippet] — [arXiv 2508.18440](https://arxiv.org/pdf/2508.18440v1).
- Generative SE artifacts: ArtiFree lists phoneme insertions/substitutions, breathing/hiss, robotic tones, high-frequency loss, worst at low SNR [snippet] — [arXiv 2509.19495](https://arxiv.org/pdf/2509.19495). StoRM reports SGMSE+ gives worse ASR WER than predictive baselines, attributed to vocalizing artifacts and phonetic confusions [snippet] — [arXiv 2212.11851](https://arxiv.org/pdf/2212.11851). PASE splits hallucination into linguistic vs acoustic (pitch/timbre) errors [snippet; no F0 metric seen] — [arXiv 2511.13300](https://arxiv.org/pdf/2511.13300v1). A 2026 SE paradigm comparison (SGMSE+, GAN, flow matching) measures hallucination by WER/phoneme similarity [snippet] — [arXiv 2606.02913](https://arxiv.org/html/2606.02913).
- Singing: ROSVOT (ACL 2024, arXiv 2405.09940) gets robust transcription via noise-augmented training rather than a denoiser; noise mixing even helped clean input [snippet] — [arXiv](https://arxiv.org/html/2405.09940v1). RobustSVC (arXiv 2409.06237) states separation-based preprocessing leaves artifacts or over-suppresses, and that pitch is not noise-robust [snippet] — [arXiv](https://arxiv.org/html/2409.06237v1).

### Inferences
- The best-supported plan: treat enhancement as an A/B hypothesis, not an assumption. Run a pitch tracker (CREPE/pYIN/RMVPE etc.) on raw vs restored audio against a ground-truth or against raw-on-clean-segments and keep restoration only if it does not worsen F0/note metrics.
- Hums are low-harmonic, breathy, with glides; speech-trained denoisers may treat sustained voiced tones with vibrato as noise or fricative-like - plausible risk but NOT shown by any source I found.
- A claim like "neural enhancement distorts F0" is supported only indirectly (hallucination literature, Interspeech 2022 distortion remark).

### Gaps
- No paper found measuring GPE/FPE/RPA/note-F1 before and after DeepFilterNet, Demucs-denoiser, Voicefixer, ResembleEnhance, or SGMSE+ on singing or humming.
- No hum-specific (query-by-humming) enhancement study retrieved.
- Interspeech 2022 PDF could not be parsed; quantitative numbers not read.

## 4. Verifying an edit as safe

### Takeaway
No standard published "edit safety" protocol for pitch-targeted restoration was found; the verification ideas below are engineering practice combining consistency constraints with re-measurement. Classical declipping offers built-in checks; for neural enhancement only empirical before/after checks are possible.

### Cited Findings
- Consistency: C-IHT/SPADE explicitly constrain reconstruction to match unclipped samples and respect clipping bounds, giving an automatic check [snippet] — [arXiv 1303.1023](https://arxiv.org/pdf/1303.1023), [arXiv 1506.01830](https://arxiv.org/pdf/1506.01830).
- Metrics used in declipping evaluation: SDR and PEMO-Q ODG [fetched/snippet] — [arXiv 1905.00628](https://arxiv.org/abs/1905.00628), [JSTSP survey page](https://signalprocessingsociety.org/publications-resources/ieee-journal-selected-topics-signal-processing/2021/01/survey-and-extensive-evaluation-popular-audio-declipping-methods).
- Pitch error metrics for singing/voice: F0 RMSE (often in semitones after DTW), F0 correlation, voiced/unvoiced error rate; gross pitch error and fine pitch error for trackers [snippet] — [arXiv 2209.10446](https://arxiv.org/pdf/2209.10446), [arXiv 2106.15205](https://arxiv.org/pdf/2106.15205), [arXiv 1807.00752](https://arxiv.org/pdf/1807.00752).
- Hallucination detection via embedding variance (ArtiFree) and phonological priors (PASE) exist for speech generative SE [snippet] — [arXiv 2509.19495](https://arxiv.org/html/2509.19495v1), [arXiv 2511.13300](https://arxiv.org/pdf/2511.13300v1). Phoneme-based, not pitch-based.

### Inferences
Proposed verification stack (my design, not from a source):
1. Region-limited edits: apply restoration only where defects are detected (clipped samples, noise-only gaps); require bit-exact (or near) identity outside edit mask and report max sample difference.
2. Dry/wet: store original; output = dry + mask*(wet-dry) with crossfades; keep operation reversible and log parameters.
3. Re-measure F0 on dry and wet with two independent trackers (e.g., pYIN + CREPE); flag if median absolute F0 change in cents outside defect regions exceeds a threshold (e.g., a few cents), if voicing flips, or if note segmentation changes; compare in log-spectral difference outside corrupted regions and check harmonic peak positions/ratios.
4. For clipping: assert unclipped samples unchanged and restored clipped samples exceed clip level.
5. Reject (fall back to dry) on any failed check.

### Gaps
- No literature found on cents-level tolerances for "safe" intonation edits or on listening-test protocols for hums.
- No retrieved paper on dry/wet reversible edit verification; this section is partly design inference.
