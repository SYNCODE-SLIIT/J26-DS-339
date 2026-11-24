# Usability prediction (no-reference quality) and transcriber/pitch-tracker robustness for sung/hummed audio

## Q1. No-reference quality predictors: what they validate on, and do they predict pitch/transcription success?

### Takeaway
Speech-trained NR predictors (DNSMOS, NISQA, UTMOS, TorchAudio-SQUIM) are validated against human MOS or intrusive metrics (PESQ/STOI/SI-SDR) on speech, not against F0 or note-transcription accuracy; on singing they transfer poorly (SingMOS-Pro). The only directly relevant "predict transcription from recording quality" work found is Schwabe et al. 2022 (piano, noise/disturbance/reverb estimators), seen only at abstract level. No work found that predicts hum/singing pitch-tracking or note-transcription reliability from voicing coverage, SNR or pitch confidence.

### Cited Findings
- TorchAudio-SQUIM (ICASSP 2023): reference-less estimators of STOI, WB-PESQ, SI-SDR (plus a NORESQA-MOS MOS model). Trained and tested on DNS Challenge 2020 data (noise SNR -15 to 25 dB, plus GCRN-enhanced mixtures; 5 s clips). Test PCC/SRCC: STOI 0.950/0.950, WB-PESQ 0.958/0.963, SI-SDR 0.985/0.985 (with multi-task learning). The paper reports no reverberation, no out-of-domain test, no music/singing discussion, and no MOS numbers (MOS model taken from prior work). — [Kumar et al. 2023, arXiv 2304.01448 (ar5iv)](https://ar5iv.labs.arxiv.org/html/2304.01448)
- A third-party aggregator lists Squim at ~0.74 on NISQA test and ~0.71 on VoiceMOS 1 (statistic undefined; snippet only, unverified). — [search snippet via paperswithcode mirror](https://cs.paperswithcode.com/paper/torchaudio-squim-reference-less-speech)
- SingMOS (Tang et al., 2024, arXiv 2406.10911): open singing MOS dataset, v1 3,421 Chinese/Japanese clips, 4.25 h, 16 kHz; ground truth plus 21 SVS, 11 SVC and 6 vocoder systems; baseline SSL predictor tested with extra pitch information. Served as Track 2 of the VoiceMOS Challenge 2024 (ASRU 2024). Data are synthesized/converted singing, not user-recorded hums. — [SingMOS](https://arxiv.org/html/2406.10911v2)
- SingMOS-Pro (arXiv 2510.01812, 2025): 7,981 clips (3,425 SVS, 1,307 SVC, 2,671 SVR, 578 ground truth), >=5 raters/clip, 4,155 clips with lyrics and melody sub-scores. Benchmark weighted-average (Utt LCC / Utt SRCC / Sys LCC / Sys SRCC), not fine-tuned on SingMOS-Pro: DNSMOS 0.39/0.33/0.60/0.41; UTMOS 0.26/0.36/0.36/0.54; SingMOS(preview) 0.57/0.53/0.78/0.69; SHEET-ssqa 0.56/0.50/0.79/0.69. Fine-tuned wav2vec2: 0.58/0.50/0.86/0.77; with pitch-histogram feature 0.58/0.51/0.88/0.79. Authors: speech MOS models "perform poorly on the singing task due to the substantial domain gap"; the singing-trained SingMOS drops on out-of-domain test2/3. — [SingMOS-Pro v4](https://arxiv.org/html/2510.01812v4)
- A VoiceMOS 2024 Track 2 team argued speech-MOS methods transfer inappropriately to singing (snippet only). — [VoiceMOS 2024 review](https://www.themoonlight.io/review/the-voicemos-challenge-2024-beyond-speech-quality-prediction); overview paper arXiv 2411.11123
- Out-of-domain generalization of SSQA models is weak in general (MOS-Bench; VMC'23): DNSMOS good on DNS3 but poor elsewhere; NISQA/UTMOS better but fail on OOD synthetic speech (snippet-level). — [MOS-Bench, arXiv 2411.03715](https://arxiv.org/pdf/2411.03715)
- Schwabe, Hoffmann, Murgul, Heizmann (2022, ICSM / Springer LNCS, DOI 10.1007/978-3-031-22061-6_24), "Estimation of Music Recording Quality to Predict Automatic Music Transcription Performance": piano recordings; three neural nets estimate background noise, sound disturbances, reverberation; they report transcription performance (Onsets and Frames) can be forecast from these quality parameters. Read only from abstract/record snippets; no effect sizes verified. — [KIT repository](https://publikationen.bibliothek.kit.edu/1000153948)
- HumTrans (Liu et al., 2023, arXiv 2309.09623): 10 music-proficient students, each hummed each segment twice ("Da"), web recorder, 44.1 kHz, 500 compositions / 1000 segments, ~56.22 h, 14,614 files (4.9-29.9 s). A search summary says recordings were screened by DNSMOS with a 2.8 cutoff, in a quiet room with headphones, with re-recording of poor takes. I could not verify this from the paper (PDF not readable by my tools); treat as UNVERIFIED. This would be a precedent for using an NR MOS as an acceptance gate on hums. — [HumTrans](https://arxiv.org/abs/2309.09623v2); [search summary](https://arxiv.org/pdf/2309.09623)
- ITU-T TS 26.940 (via search snippet): speech-oriented metrics (DNSMOS, NISQA, UTMOS) vs general-audio metrics (ViSQOL, PEAQ) for music; unverified. — [iTecSpec](https://www.itecspec.com/3gpp/26.940/s/9.1.4.1)

### Inferences
- No evidence that any off-the-shelf NR MOS predictor tracks F0/transcription success on hums; a usability predictor for this task likely has to be trained against measured transcription outcome (novel gap).
- SingMOS-Pro's gain from adding a pitch histogram is small (sys SRCC 0.77 to 0.79), so pitch features are not a proven route to quality prediction either.
- HumTrans' reported DNSMOS gate (if verified) suggests dataset curation already uses speech MOS as a proxy, with no validation against transcription.

### Gaps
- Original DNSMOS (P.835), NISQA, UTMOS papers' own validation numbers not retrieved this session.
- No paper found relating voicing coverage/pitch confidence/SNR to transcription accuracy for singing; no vocal-specific "transcribability" predictor found. Search was limited (about 15 tool calls).
- Schwabe 2022 full text and numbers not read. HumTrans DNSMOS threshold not verified.

## Q2. Quantified degradation of pitch trackers and transcribers

### Takeaway
Additive noise at >=10 dB costs little for modern neural trackers, but accuracy falls sharply at <=0 dB, with non-stationary pub/babble noise the worst; CREPE/RMVPE beat pYIN at low SNR. Background music at 0 dB hurts even CREPE/PESTO. I found no published RT60, clipping, low-level or codec/band-limiting numbers for these trackers (only an old QBH codec study), and very little for Basic Pitch, FCPE, HumTrans-style systems.

### Cited Findings
- RMVPE (Wei et al., Interspeech 2023, arXiv 2306.15412) on MIR-1K vocals with Audio Degradation Toolbox noise; sweep 40-0 dB, text states performance of all models "decreases sharply" below 10 dB. At 0 dB, RPA/RCA (%): pYIN white 33.94/34.21, pink 57.83/58.30, brown 79.36/79.80, pub 40.80/48.64 (avg 52.98/55.24); CREPE white 92.15, pink 89.01, brown 91.47, pub 61.22 RPA (avg 81.16/87.67); RMVPE-Poly 96.47, 93.87, 97.78, 86.26 (avg 93.60/94.37); RMVPE-Vocal avg 92.51/93.51; JDC avg 74.92; HarmoF0 avg 77.07. Loss from 10 to 0 dB: CREPE 3.96 (white), 15.34 (pink), 4.48 (brown), 29.67 (pub); RMVPE 1.08, 3.48, 0.08, 10.07. Authors' own comparison (not independent); per-SNR values only in a figure. — [RMVPE ar5iv](https://ar5iv.labs.arxiv.org/html/2306.15412)
- CREPE original (Kim et al., ICASSP 2018): evaluated with pub/white/pink/brown noise; CREPE better than pYIN at SNR < 10 dB and lower variance; pYIN better under brown (low-frequency) noise at high SNR. Snippet-level. — [CREPE](https://arxiv.org/pdf/1802.06182)
- Independent multi-tracker test (Koguchi and Koriyama, 2026, arXiv 2602.01727; mixed speech/MIR-1K/MDB-stem-synth, NOISEX92 + QUT-NOISE, 48 kHz): RPA50 (%) clean/30/20/10/0 dB: pYIN 72.30/71.74/65.39/61.85/26.27; CREPE 75.84/64.69/64.38/62.31/50.65; FCNF0++ 68.91/70.32/68.80/56.52/22.86; Harvest 76.49/67.60/66.75/57.54/25.51; Praat 53.01/31.95/31.65/28.16/13.66; REAPER (also the speech reference) 80.56/80.24/78.01/68.85/37.45. Note ground truth from REAPER for speech; reverb not tested; RCA not reported; data are mostly speech. V/UV recall at 0 dB: CREPE 64.86, pYIN 32.12. — [arXiv 2602.01727 HTML](https://arxiv.org/html/2602.01727)
- PESTO (Cardoso/Serra et al., journal version 2025, arXiv 2508.01488) — only robustness test is background music on MIR-1K. RPA (%) clean/20/10/0 dB: PESTO trained clean-only 97.2/93.7/81.6/46.8; PESTO with augmentation 98.3/98.0/95.9/79.2 (best, beta~U(0,1/2)) or 98.1/97.8/95.6/82.5; CREPE 97.5/97.1/95.3/85.8 (from SPICE paper); SPICE 91.4/91.2/90.0/81.6; PENN 90.6/81.0/51.1/20.9. No reverb, compression or noise tests; no pYIN/RMVPE/FCPE comparison. — [PESTO journal](https://arxiv.org/html/2508.01488v2)
- ROSVOT (Li et al., ACL 2024, arXiv 2405.09940): singing note transcription, noisy test = MUSAN (babble/music/noise/speech) at SNR 6-20 dB, p=0.8; datasets M4Singer + in-house (studio-quality). Clean to noisy: TONY COn 67.5 to 49.2, COnPOff 43.9 to 28.4, RPA 73.9 to 45.2; VOCANO COnPOff 50.2 to 43.4, RPA 76.6 to 59.8; MusicYOLO COnPOff 58.9 to 51.5, RPA 81.6 to 78.9; Yong et al. 2023 COnPOff 65.8 to 62.1; ROSVOT full COnPOff 77.4 to 77.0, RPA 87.6 to 87.4. ROSVOT without noise augmentation: COnPOff 76.4 to 70.1. Abstract claim: 23% relative improvement on noisy inputs. Appendix inconsistency for one ablation row noted by extractor. RCA not reported. — [ROSVOT](https://arxiv.org/html/2405.09940v2)
- Query-by-singing/humming: Molina et al., ISMIR 2014, "The importance of F0 tracking in QBSH": MIR-QBSH with added pub-style noise and smartphone-style distortion, several F0 trackers x 3 matchers (189 combinations); F0 accuracy correlates with QBSH performance but does not fully capture suitability; matchers differ in tolerance to F0 errors. Per-condition numbers not retrieved (PDF unreadable). — [UMA record](https://riuma.uma.es/entities/publication/8030ed20-0a10-4165-82fd-74066da5b10e/full)
- Older QBH work (snippet-level, 2000s-2010s): noise-robust mobile QBH (ICME 2007) ~85% top-10 in noise vs ~91% clean; Kim et al. noise-robust pitch estimator +11.7% average accuracy at 0 dB over G.729 pitch; GSM codec robustness claimed for an ear-model front end; NTU thesis autoencoder denoising for humming across 17 noises, -5 to 30 dB, notes noise creates false onsets. — [ECNU](https://pure.ecnu.edu.cn/en/publications/a-noise-robust-content-based-music-retrieval-system-for-mobile-de/); [NTU](https://tdr.lib.ntu.edu.tw/handle/123456789/83095)

### Inferences
- Rule of thumb from RMVPE/Koguchi/PESTO: 20 dB SNR is near-harmless for neural trackers; 10 dB moderate (pYIN -10 RPA points, CREPE/PESTO small); 0 dB severe (pYIN 26-41% RPA; CREPE 50-86% depending on noise type/data). Noise type matters more than SNR alone (pub/babble and music are worst; brown is benign).
- Note-level metrics (COnPOff) fall much more than frame RPA under noise for non-robust models (TONY: RPA -29 pts, COnPOff -15 pts; onset localization is the fragile part), which supports gating on onset-related reliability, not only pitch.
- Training with augmentation is the main reason robust models (ROSVOT, RMVPE) are flat; clean-only models (PESTO beta=0, PENN) collapse.
- All of these noise levels are synthetic; none corresponds to measured phone-room conditions.

### Gaps
- Reverberation (RT60), clipping, low recording level, band-limiting/codec (MP3/Opus/AMR) effects on pYIN/CREPE/PESTO/RMVPE/FCPE/Basic Pitch/ROSVOT: no published numbers found.
- FCPE: no primary paper found. Basic Pitch (Bittner et al., ICASSP 2022): no robustness results found. HumTrans baseline transcription results and any HumTrans noise stress tests not retrieved.
- Molina 2014 numeric results not retrieved.

## Q3. Dominant failure modes for user-recorded hums on phones/laptops

### Takeaway
I found no study that ranks corruptions for real phone/laptop humming. The evidence is indirect: QBH papers treat background noise and phone/codec distortion as the main environmental problems, and note noise corrupts onsets; but the dominant error source in QBH may be singer error (pitch, tempo), not signal corruption.

### Cited Findings
- HumTrans recording was done by musically proficient students in a controlled web-recording setup with 44.1 kHz audio (verified), not in-the-wild phone conditions; a search summary claims quiet room/headphones/DNSMOS gate (unverified). Consequently it provides little evidence on real-world corruption. — [HumTrans](https://arxiv.org/abs/2309.09623v2)
- Molina et al. 2014 choose pub noise and smartphone distortion as the representative QBSH corruptions (they simulate them on MIR-QBSH); conclusion that F0 accuracy only partially explains retrieval. — [UMA record](https://riuma.uma.es/entities/publication/8030ed20-0a10-4165-82fd-74066da5b10e/full)
- Kim et al. and ICME 2007 papers state QbSH performance is "considerably degraded" by background noise (snippet-level). — [ECNU](https://pure.ecnu.edu.cn/en/publications/a-noise-robust-content-based-music-retrieval-system-for-mobile-de/)
- NTU thesis: noise creates incorrect onset positions in QBH (snippet-level). — [NTU](https://tdr.lib.ntu.edu.tw/handle/123456789/83095)
- MIR-QBSH collection conditions not found (a review calls it relatively small). — search result only

### Inferences
- Noise (non-stationary) and onset corruption are the best-supported dominant failure modes; clipping, low level and reverb are untested in the literature found, so a study measuring them on hums would be novel.

### Gaps
- No survey of real user hum recording quality (device, room, level distributions). MIR-QBSH original paper and recording details not found. Missing: any analysis of failures per corruption type for in-the-wild hums.
