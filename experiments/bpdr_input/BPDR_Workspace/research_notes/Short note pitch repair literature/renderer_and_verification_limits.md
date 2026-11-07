# Limits of pitch-shift renderers and pitch trackers on very short voiced segments (0.24-0.5 s), and verification of local pitch edits

Scope note: about 16 tool calls. Several PDFs (CREPE, RMVPE, penn) came back as unreadable binary, so their numbers come from search snippets or from the PESTO/FCPE tables that re-report them. I found NO primary source that studies renderers or trackers specifically at 0.24-0.5 s, and none that evaluates humming at the cent level. Those are the main gaps. Items marked [UNVERIFIED] are from memory or secondhand.

## 1. How PSOLA, WORLD, Praat, Rubber Band, phase vocoder and neural methods behave on short segments

### Takeaway
I found no primary source that quantifies minimum useful segment length for any renderer. What the sources do establish is the mechanism of failure. PSOLA depends on accurate pitch marks. Phase-vocoder-type tools are tuned for long material. Classical shifting produces formant shifts and "robotic" coloration that grows with shift size. Short spans inherit all of this with less context to hide it.

### Cited Findings
- TD-PSOLA places marks at signal peaks in voiced regions and overlap-adds Hanning-windowed segments between marks. Marks centred on excitation peaks whose phase is not synchronous cause phase distortion — [USPTO 7822599 (patent text, secondary)](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/7822599)
- Colotte and Laprie (EUSIPCO 2002), "Higher precision pitch marking for TD-PSOLA": marks are chosen among local extrema of the waveform. Errors at rapid formant transitions are removed, but some errors cannot be eliminated, and the remaining ones are visible as mark spacing far from the pitch period — [EUSIPCO 2002 paper318](https://www.eurasip.org/Proceedings/Eusipco/2002/articles/paper318.pdf); [HAL record](https://hal.archives-ouvertes.fr/inria-00107610)
- A thesis (found via search, not opened) reports that using 3 candidate marks per search region cut the pitch-mark error rate and gave smoother results — search snippet only, [core.ac.uk](https://core.ac.uk/download/50420394.pdf) [UNVERIFIED detail]
- A patent on steady-sound synthesis notes that repeating pitch-period segments are audible as transitions and are especially bad for hybrid sounds with noisy and periodic components. That concerns long duration stretching, not short-span shifting — [USPTO 7558727](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/7558727)
- Traditional time-domain methods such as TD-PSOLA introduce artifacts when the pitch factor is too high and depend heavily on accurate pitch markers. Classical methods give formant shift and robotic colouration that worsen at larger transpositions — [Liu and Akama, arXiv 2601.10345 (Jan 2026, preprint)](https://arxiv.org/abs/2601.10345)
- The same paper builds training pairs with WORLD (forward then inverse shift, e.g. +12/-12 semitones), meaning WORLD round trips are themselves degrading. It compares against TD-PSOLA, WORLD, SiFiGAN and Diff-Pitcher (comparison list is from a secondary review, not the paper) — [arXiv 2601.10345](https://arxiv.org/abs/2601.10345); [review](https://www.themoonlight.io/tw/review/self-supervised-restoration-of-singing-voice-degraded-by-pitch-shifting-using-shallow-diffusion)
- A 2025 automatic pitch-correction paper (BERT-APC) applies TD-PSOLA through Praat-Parselmouth as a simple, training-free renderer — [arXiv 2511.20006](https://arxiv.org/pdf/2511.20006)
- WORLD (Morise, Yokomori, Ozawa, IEICE Trans. Inf. Syst. E99-D(7), 2016) is the DIO / StoneMask / CheapTrick / D4C pipeline. DIO is from IEICE 2010 (Japanese), CheapTrick from Speech Communication 2015, D4C from Speech Communication 84:57-65 (2016). Typical workflow is DIO, then optional StoneMask refinement, then CheapTrick and D4C, then synthesis. If many F0 elements are zero, the allowed range should be widened — [WORLD publications page](https://www.isc.meiji.ac.jp/~mmorise/world/english/publications.html); [go-world docs / programming guide](https://pkg.go.dev/github.com/r9y9/go-world); [repo](https://github.com/mmorise/World)
- Praat Manipulation: overlap-add resynthesis uses the original sound, pulses (placed where the sound has much energy, following the pitch contour and voicing decisions), a pitch tier and a duration tier. Typical analysis step is 0.010 s, floor 75 Hz, ceiling 600 Hz. The manual page gives no pitch-shift limits or quality statements — [Praat Manipulation manual](https://www.fon.hum.uva.nl/praat/manual/Manipulation.html)
- Rubber Band: R3 (fine) engine "almost always produces better results than R2 but with significantly higher CPU load". `-F` tries to keep the formant envelope unchanged. Transient and window options are R2-only. Usage text says nothing about short inputs — [Rubber Band usage.txt](https://breakfastquay.com/rubberband/usage.txt)
- Rubber Band v3.0.0 added independent formant shifting with R3. v3.1 added an R3 "short window" mode that lowers CPU and delay (not aimed at short clips). v4.0 added a simpler pitch-shift-only API — [Rubber Band CHANGELOG v3.0.0](https://github.com/breakfastquay/rubberband/blob/v3.0.0/CHANGELOG)
- Neural vocoder with F0 control: PeriodGrad found its pitch-shifted outputs scored lowest, suggesting the model also uses pitch information embedded in the mel spectrogram, which does not change when F0 is shifted — [arXiv 2402.14692](https://arxiv.org/pdf/2402.14692)
- Neurodyne (arXiv 2505.15368, May 2025) does neural pitch manipulation with a cycle-consistency GAN and evaluates global key shifts — [secondary review](https://www.themoonlight.io/review/neurodyne-neural-pitch-manipulation-with-representation-learning-and-cycle-consistency-gan) (not opened directly)

### Inferences
- A 0.24-0.5 s span at 150-400 Hz contains only about 36-200 pitch periods. A handful of wrong or phase-misaligned marks at the edges is a large fraction of the span. This is my arithmetic, not a sourced result.
- Methods that need a wide analysis window (Rubber Band R3, phase vocoder) cannot be fed a hard-cut 0.3 s clip without edge distortion. Rendering with surrounding context and then cutting is safer. This is inference.
- Because the shift is a local edit, the boundaries (where the shift ramps to zero) matter as much as the interior.

### Gaps
- No published minimum useful segment length, ramp length, or crossfade length for PSOLA, WORLD, Praat or Rubber Band.
- No official documentation statement on short-input behavior for Rubber Band or WORLD.
- Phase-vocoder literature (Laroche and Dolson, Roebel on transients and vertical phase coherence) was not retrieved; I cannot cite specifics.
- Formant-handling comparison across methods (cepstral/LPC envelope preservation numbers) not found.
- Praat's manual on PSOLA limits (factor range) was not found on the page I read.

## 2. Pitch-tracker accuracy at the 1-10 cent level; humming; short voiced runs

### Takeaway
Published evaluations mostly use a 50-cent tolerance (RPA) and clean singing or synthesized stems. CREPE reports high accuracy at 10 cents on synthetic data, but I found nothing at 1-10 cents on real singing, nothing on humming, and nothing on runs shorter than 0.5 s. Output resolution is also limited by bin width (about 20 cents CREPE/FCPE, about 33 cents PESTO) before any interpolation.

### Cited Findings
- PESTO (Riou et al., TISMIR 8(1):334-352, 2025; arXiv 2508.01488): RPA is the share of voiced frames with error under 50 cents. Error is 12·|log2(f̂/f)| semitones. All metrics are computed on voiced frames only, and the voicing detection method is not described. Annotation hops: MIR-1K 20 ms, MDB-stem-synth 2.904 ms, PTDB 10 ms. VQT uses 3 bins per semitone, so about 33 cents per bin. Max kernel is 8192 samples at 48 kHz (about 171 ms). Monophonic only — [PESTO arXiv v2 HTML](https://arxiv.org/html/2508.01488v2)
- PESTO Table 2 (RPA / RCA / mean err semitones / median err semitones): on MIR-1K, CREPE 97.5/98.0/0.23/0.07; PESTO (trained on all) 95.6/96.7/0.49/0.14; PENN 90.6/92.4/1.27/0.11. On MDB-stem-synth, CREPE 97.3, PESTO 97.1, PENN 99.6. On PTDB, CREPE 87.1 (mean 1.24 st), PESTO 88.5, PENN 95.1. Median errors of 0.02-0.14 semitones (2-14 cents) show that even typical frames are not at 1-cent accuracy — [PESTO arXiv v2 HTML](https://arxiv.org/html/2508.01488v2)
- PESTO notes: CREPE was evaluated with Viterbi smoothing off. PTDB annotations are offset about 5 ms relative to MIR-1K and MDB-stem-synth, and not accounting for it hurts metrics. At 0 dB vocal-to-background SNR best PESTO RPA is 82.5 versus 85.8 for CREPE. Latency is stated as under 10 ms in the abstract and under 5 ms in the conclusion (inconsistent) — [PESTO arXiv v2 HTML](https://arxiv.org/html/2508.01488v2)
- CREPE (Kim, Salamon, Li, Bello, ICASSP 2018; arXiv 1802.06182): beats pYIN and SWIPE on synthesized benchmarks, over 90% RPA even at a strict 10-cent threshold. RWC-synth RPA at 50 cents: SWIPE 96.99%, pYIN 99.19%, CREPE 99.98% (snippets of the paper) — [CREPE on alphaXiv](https://alphaxiv.org/abs/1802.06182); [DeepAI copy](https://deepai.org/publication/crepe-a-convolutional-representation-for-pitch-estimation). The PDF was unreadable, so the following are [UNVERIFIED, from memory]: 1024-sample window at 16 kHz (64 ms), 360 output bins of 20 cents, 10 ms hop.
- RMVPE (Wei et al., 2023; arXiv 2306.15412), figures via a secondary summary of its tables: MDB-stem-synth RPA pYIN about 65.8%, CREPE about 97.5%, RMVPE about 97.1%. MIR-1K pYIN about 74.7%, CREPE about 95.7%, RMVPE about 97.3% [UNVERIFIED, search-summary only]. Note this pYIN figure conflicts with the 99.19% on RWC-synth above; the datasets and pYIN settings differ and the RMVPE number was not checked against the paper — [RMVPE arXiv](https://arxiv.org/pdf/2306.15412)
- FCPE (arXiv 2509.15140): 360 bins at 20-cent spacing (C1 to B7), local weighted average over ±4 bins around the peak. Output is 0 Hz (unvoiced) when peak probability is under 0.05. Voicing accuracy is not separately evaluated; hop, window and cent tolerance are not stated. RPA clean: MIR-1K FCPE 96.79, RMVPE 97.77, CREPE 97.90, PESTO 98.47. Vocadito FCPE 95.80, RMVPE 95.83, CREPE 97.35, PESTO 95.50. TONAS FCPE 96.03, RMVPE 95.64, CREPE 95.20, PESTO 91.30. RTF on RTX 4090: FCPE 0.0062. The authors note RMVPE, CREPE and PESTO were trained on MIR-1K so its results may be inflated — [FCPE arXiv HTML](https://arxiv.org/html/2509.15140v1)
- penn (Morrison et al., "Cross-domain Neural Pitch and Periodicity Estimation", arXiv 2301.12258, 2023, v3 Aug 2024, no venue listed): introduces an entropy-based periodicity and voiced/unvoiced decision from neural pitch posteriors. Reports cents error on voiced frames, RPA and periodicity F1, but I could not read the numbers — [abs page](https://arxiv.org/abs/2301.12258)
- pYIN (Mauch and Dixon, ICASSP 2014, pp. 659-663): YIN with multiple candidates from a prior over the threshold, then HMM/Viterbi smoothing; evaluated with a 256-sample (5.8 ms) step. Frame-level voicing is probabilistic — [core.ac.uk PDF](https://core.ac.uk/download/30696897.pdf); [librosa pyin doc](https://librosa.org/doc-playground/0.8.1/generated/librosa.pyin.html)
- Praat: Pitch (raw cross-correlation) is recommended in the manual for experiments and where short time windows are needed (e.g. voice analysis). Default floor 75 Hz. If time step is 0, it is 0.25/floor (3.33 ms at 75 Hz), always 4 pitch values per window length — [Praat raw cc manual](https://www.fon.hum.uva.nl/praat/manual/Sound__To_Pitch__raw_cross-correlation____.html) (snippet via search); [pitch method chooser](https://www.fon.hum.uva.nl/praat/manual/how_to_choose_a_pitch_analysis_method.html). Praat autocorrelation defaults: voicing threshold 0.45, silence threshold 0.03, octave cost 0.01/octave, octave-jump cost 0.35, voiced/unvoiced cost 0.14, 15 candidates, "very accurate" off — [Praat To Pitch (ac) manual](https://www.fon.hum.uva.nl/praat/manual/Sound__To_Pitch__ac____.html)
- Humming: no study found. A snippet states even pYIN can be noisy on rapidly fluctuating pitch curves, but it is not a humming evaluation — [CREPE-related search result]; a 2021 thesis says CREPE and SPICE beat pYIN on monophonic F0 — [Stillings 2021](https://zenodo.org/record/5554724/files/2021-Logan-Stillings.pdf)

### Inferences
- Output grids of 20-33 cents mean 1-10 cent claims depend on interpolation or weighted-average decoding (FCPE, and CREPE's weighted average [UNVERIFIED]). Their errors are not independent of the true F0 fractional position. This is inference.
- Window length bounds time resolution: a 64 ms CREPE window (UNVERIFIED) or a 3-period Praat window at a 75 Hz floor (40 ms; formula from memory, UNVERIFIED) is a sizeable fraction of a 240 ms span, so edges are measured with context from outside the span.
- Published median errors (2-14 cents) imply that a verification threshold below about 10-15 cents is at or under tracker noise for many models.

### Gaps
- No cent-level (1-10 cent) evaluation on real singing or humming found; no humming dataset benchmark.
- No evaluation of accuracy as a function of voiced-run length or of voicing-decision error on 0.24-0.5 s runs.
- RMVPE, CREPE and penn PDFs unreadable; hop and window details unverified.
- PESTO paper's final 16k characters (references/appendix) were not read.

## 3. How published systems verify an edit; failure modes of re-measured verification

### Takeaway
Published pitch-correction work verifies mainly by objective error against note targets (RMSE, correlation, cents) plus listening tests. I found no paper that formally analyses re-measure verification, shared tracker bias or smoothing. Those failure modes below are inferred from tracker documentation and evaluation caveats.

### Cited Findings
- Sinsy-related evaluation uses RMSE and CORR of corrected pitch versus note pitch, plus a MOS test of naturalness and in-tuneness — [arXiv 2108.02776](https://arxiv.org/pdf/2108.02776) (as characterised by search summary)
- Deep Autotuner (Wagner et al., arXiv 2002.05511): reports error in cents on synthetic data, then a listening test on real performances comparing original, model output and a nearest-semitone baseline whose shifts are capped at 50 cents, with a detuned "quiz" item to screen out inattentive raters — [arXiv 2002.05511](https://arxiv.org/pdf/2002.05511)
- Biswas et al. 2020: trained singers typically deviate about 15-20 cents, non-musicians often over 50 cents — [Sci Rep s41598-020-73932-1](https://link.springer.com/10.1038/s41598-020-73932-1)
- Larrouy-Maestri et al.: listeners can accept intervals 20-25 cents off as in tune — [ORBi manuscript](https://orbi.uliege.be/bitstream/2268/160709/1/MP_Manuscript_operatic_voices.pdf)
- Evaluation caveats that bear on re-measurement: annotation offset of about 5 ms hurts metrics (PESTO); RPA tolerates octave errors only in RCA; voiced-frame-only metrics hide voicing errors; CREPE benchmarked without Viterbi — [PESTO](https://arxiv.org/html/2508.01488v2)

### Inferences
- Shared tracker bias: if the same tracker picks the correction target and re-measures it, systematic bias cancels and the edit passes even if perceptually off. Using a different tracker family (e.g. autocorrelation plus a neural one) and checking disagreement is the standard hedge, but I found no citation for it.
- A pitch-marking or voicing mistake in the renderer can create an octave or sub-harmonic error that the tracker then reproduces. Viterbi/HMM tracker smoothing (pYIN, Praat path cost) can hide short local deviations on short spans.
- A re-measured result within ±50 cents of target says little; sub-10-cent verification sits at the tracker noise floor (Section 2).

### Gaps
- No paper on shared-tracker bias, tracker disagreement thresholds, or smoothing as a failure mode of re-measure verification.
- No cent-level listening-test thresholds specific to short spans.

## 4. Why exact-reference correction might be withheld or fail audit on short spans; design changes reported to help

### Takeaway
I found no source about audit-withholding specifically. Plausible causes follow from Sections 1-3 (tracker context and resolution, renderer edge artifacts, voicing decisions). Reported mitigations in the literature are mostly general: better pitch marks, restoration or neural re-synthesis, and smoother parameter curves.

### Cited Findings
- Better pitch-mark selection (multi-candidate search, local-extrema selection with error removal) reduces PSOLA errors but not completely — [EUSIPCO 2002](https://www.eurasip.org/Proceedings/Eusipco/2002/articles/paper318.pdf)
- Restoration after classical shifting with a mel-space diffusion model conditioned on f0, volume and content features is proposed to cut formant and robotic artifacts — [arXiv 2601.10345](https://arxiv.org/abs/2601.10345)
- Rubber Band R3 (finer engine) and formant preservation / independent formant shift options are the library's route to better quality — [usage.txt](https://breakfastquay.com/rubberband/usage.txt); [CHANGELOG](https://github.com/breakfastquay/rubberband/blob/v3.0.0/CHANGELOG)
- Praat Manipulation and Rubber Band pitchmap options accept time-varying pitch targets (pitch tier; `--pitchmap`/`--freqmap`, which implies realtime mode and inexact duration), allowing a smooth ramp curve rather than a step — [Praat manual](https://www.fon.hum.uva.nl/praat/manual/Manipulation.html); [usage.txt](https://breakfastquay.com/rubberband/usage.txt)

### Inferences
- Likely reasons a correction fails audit: (a) tracker window and bin resolution make the re-measured median differ by more than the tolerance; (b) the edit edges are measured in frames overlapping unedited audio; (c) vibrato makes the mean-shift target ambiguous; (d) voicing flips at onset/offset drop frames; (e) renderer pitch-mark phase error. All inferred.
- Likely helpful design changes: render with context on both sides and trim; use ramps spanning multiple pitch periods and tracker windows; keep the shift curve smooth and shift a vibrato-aware residual instead of flattening the contour; audit on the interior frames with edge guard bands equal to half the tracker window; use two trackers and require agreement. These are design suggestions, not reported results.

### Gaps
- No paper on ramp or crossfade length in ms for local pitch edits; no reported vibrato-aware local shifting study found; no source on withheld corrections.
- Not retrieved: Laroche and Dolson, Roebel, Moulines and Charpentier (PSOLA original, 1990), Boersma (1993) Praat pitch, de Cheveigne and Kawahara YIN (JASA 2002), DAFx tutorials. Recommend a follow-up pass on these.
