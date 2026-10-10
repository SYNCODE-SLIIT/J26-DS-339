# Genre-Aware AI System for Automatic Vocal Melody Harmonization

## 1. Overview of the Full System

The proposed research develops an intelligent music harmonization system that converts a raw vocal melody into a complete, genre-appropriate chord progression.

A user provides:

- A raw vocal recording
- A preferred musical genre, such as Pop, Jazz, Gospel, or Lo-fi
- An optional chord-refinement level: Low, Medium, or High

The system produces:

- A timed chord progression aligned with the vocal melody
- Chords that reflect the selected genre
- Refined chord extensions and transitions
- An explainable validation report identifying strong harmonies, stylistic exceptions, and possible harmonic problems

Instead of asking one model to perform the entire task, the system separates harmonization into four connected components:

1. Harmony-oriented melody understanding
2. Genre-conditioned functional harmonic planning
3. Controllable sequence-aware chord refinement
4. Explainable post-generation harmony validation

This modular design makes the system easier to train, evaluate, control, interpret, and improve.

---

## 2. Problem Being Addressed

A vocal melody contains notes, rhythm, phrases, tension, stable points, and implied harmonic movement. However, it does not directly specify which chords should accompany it.

The same melody can also be harmonized differently depending on genre. For example:

- A Pop arrangement may use a simple progression such as `I–V–vi–IV`.
- A Jazz arrangement may use seventh chords, ninth chords, secondary dominants, and passing harmony.
- Gospel may use richer dominant movement, suspensions, and expressive resolutions.
- Lo-fi may use relaxed major-seventh, minor-seventh, and added-note chords.

Existing automatic harmonization approaches commonly face several problems:

- They may treat melody transcription and harmonization as separate, unrelated tasks.
- They may predict exact chord names too early.
- They may generate harmonically correct but stylistically generic progressions.
- They may use an excessively large chord vocabulary.
- They may ignore long-range progression context.
- They may enrich individual chords without considering neighbouring chords.
- Rule-based systems may reject valid Jazz or Gospel dissonances.
- Neural systems may generate outputs without explaining why they are acceptable.
- A single average quality score can hide one serious harmonic problem.

The proposed system addresses these limitations through a structured pipeline that separates melody understanding, functional planning, chord realization, and validation.

---

## 3. Main Research Goal

The main research goal is:

> To develop and evaluate a modular, genre-aware artificial intelligence system that transforms raw vocal melodies into harmonically coherent, controllable, stylistically appropriate, and explainable chord accompaniments.

The system should not only generate chords. It should also:

- Understand the musical behaviour of the melody
- Plan harmony at a functional level
- Adapt the harmony to a selected genre
- Refine basic chords without destroying the original progression
- Validate the generated harmony using both music theory and learned musical context
- Explain potential defects and valid stylistic exceptions

---

## 4. Main Research Question

> How can a modular artificial intelligence system convert a raw vocal melody into a genre-appropriate, musically coherent, controllable, and explainable chord accompaniment while preserving harmonic structure and distinguishing valid stylistic exceptions from genuine harmonic defects?

### Supporting research questions

1. Do harmony-oriented melody features improve chord generation compared with using only pitch and note duration?

2. Does a key-independent functional representation, such as Roman numerals, improve cross-key learning and reduce chord-vocabulary complexity?

3. Can genre and mode conditioning produce recognisably different but musically appropriate harmonic progressions?

4. Can a sequence-aware refinement model enrich basic chords while preserving the underlying harmonic structure?

5. Can a context-trajectory validation method distinguish a legitimate genre-specific dissonance from an actual harmonization error?

6. Does the complete modular pipeline perform better than a single end-to-end chord prediction model?

---

## 5. Main Research Idea

The central idea is to treat harmonization as a hierarchy of musical decisions.

The system first asks:

> What is happening in the melody?

It then asks:

> What harmonic functions should accompany the melody?

After that:

> How should those functions be realised as detailed chords for the selected genre?

Finally:

> Is the completed harmony musically supported, or does it contain a genuine problem?

This produces the following pipeline:

**Raw vocal audio + selected genre**

→ **Melody and rhythmic understanding**

→ **Functional Roman-numeral progression**

→ **Detailed genre-aware chord realization**

→ **Independent harmony validation**

→ **Final timed chord progression and validation report**

The research novelty is therefore not simply the use of a Transformer or a BiLSTM. These are established models. The principal contribution is the way musical responsibilities are separated and connected, combined with controllable refinement and context-sensitive validation.

---

# 6. Component 1: Harmony-Oriented Melody Understanding

## Objective

The first component converts a raw vocal recording into a structured musical representation suitable for harmonization.

Traditional melody transcription normally stops after detecting notes. However, chord generation requires more than a sequence of pitches. It must understand how those notes behave within the key, rhythm, and phrase.

## Input

- Raw monophonic vocal recording, such as a `.wav` file

## Processing

The component performs:

- Audio preprocessing and noise reduction
- Vocal pitch detection
- Note onset and offset detection
- Note-duration estimation
- Tempo and beat estimation
- Key and mode detection
- Phrase-boundary detection
- Scale-degree calculation
- Strong-beat and weak-beat classification
- Harmonic-stability estimation
- Cadence-likelihood estimation

For example, the note G in the key of C major is represented not only as G, but also as:

- Scale degree 5
- A potentially stable chord tone
- A note occurring on a particular beat
- A note appearing at a particular position within the phrase

## Output

A sequence of melody events containing:

- Pitch
- Start time
- Duration
- Beat position
- Bar position
- Key and mode
- Scale degree
- Rhythmic strength
- Harmonic stability
- Phrase position
- Cadence likelihood

## Problem solved

This component closes the gap between audio transcription and harmony generation. It produces a melody representation designed specifically for chord selection rather than only creating MIDI notes.

## Research contribution

The component investigates whether harmony-oriented melody features produce better chord progressions than notes-only transcription.

Its contribution can be evaluated through an ablation study:

- Model using only pitch and duration
- Model using pitch, rhythm, and key
- Model using the complete harmony-oriented feature set

---

# 7. Component 2: Genre- and Mode-Conditioned Functional Harmonic Planning

## Objective

The second component produces the structural harmonic plan of the song.

Instead of immediately predicting exact chord names such as `Cmaj9`, `Dm7/F`, or `G13`, it generates a key-independent functional progression using Roman numerals.

Example:

`I:MAJ – V:DOM – vi:MIN – IV:MAJ`

In C major, this can later become:

`C – G – Am – F`

In G major, the same functional plan becomes:

`G – D – Em – C`

## Input

- Structured melody events from Component 1
- Detected key and mode
- Beat, bar, and phrase information
- Selected genre

## Model

A Transformer-based sequence model is proposed because harmony depends on both local melody notes and longer musical phrases.

The model considers:

- Melody context
- Previous harmonic functions
- Key mode
- Genre
- Beat and bar positions
- Phrase boundaries

The melody is encoded as a sequence, while the harmonic plan is generated autoregressively:

`P(H | M, g, q) = ∏ P(hₜ | h₁…hₜ₋₁, M, g, q)`

Where:

- `H` is the harmonic-function sequence
- `M` is the melody representation
- `g` is the genre
- `q` is the mode

Because a melody can support several correct harmonizations, the model may generate multiple candidate progressions rather than claiming that only one answer is correct.

## Output

- Roman-numeral chord progression
- Coarse chord quality
- Harmonic timing
- Multiple progression candidates, where appropriate

## Problem solved

Predicting detailed chord symbols directly creates an extremely large output vocabulary and repeats the same harmonic relationships in every key.

Functional representation:

- Reduces vocabulary size
- Supports transposition
- Improves cross-key learning
- Separates harmonic structure from chord decoration
- Makes generated progressions easier to interpret

## Research contribution

The study compares:

- Absolute chord names against Roman-numeral functions
- Genre-conditioned generation against unconditioned generation
- Functional planning followed by refinement against direct detailed-chord prediction

---

# 8. Component 3: Controllable Sequence-Aware Chord Refinement

## Objective

The third component converts the functional progression into detailed chord symbols appropriate for the melody and selected genre.

A progression such as:

`C – Am – F – G`

may be harmonically correct but too plain. Depending on the context, it could be refined into:

`Cmaj7 → C6 | Am9 | Fadd9 | Gsus4 → G7`

The underlying structure remains:

`I → vi → IV → V`

## Input

- Melody events from Component 1
- Roman-numeral progression from Component 2
- Key and mode
- Base chord names
- Chord timings
- Selected genre
- User-selected refinement level

## Model

A bidirectional LSTM examines the complete progression so that each chord is refined using both previous and following musical context.

For every chord position, the model predicts one of three actions:

- Keep the basic chord
- Produce one enriched chord
- Produce two related chord realizations within the same harmonic segment

Possible chord refinements include:

- Major seventh
- Minor seventh
- Dominant seventh
- Major sixth
- Minor sixth
- Added ninth
- Major ninth
- Minor ninth
- Dominant ninth
- Suspended second
- Suspended fourth

Relative pitch-class profiles are calculated for:

- The complete chord segment
- The first half of the segment
- The second half of the segment

This allows the model to decide whether one chord should be maintained or whether two related realizations would better follow the melody.

A deterministic music-theory mask blocks invalid or structurally destructive outputs.

## Output

- Exact chord symbols
- Chord extensions
- Optional chord transitions
- Start and end times
- Preserved functional progression

## Problem solved

This component prevents detailed chord generation from overwhelming the structural planning process. It allows the system to create richer harmony without changing the intended Roman-numeral skeleton.

## Research contribution

The component introduces:

- Sequence-aware rather than isolated chord refinement
- User-controllable refinement intensity
- Structure-preserving enrichment
- Melody-aware extension selection
- Music-theory constraints that guarantee valid outputs

---

# 9. Component 4: Explainable Post-Generation Harmony Validation

## Objective

The fourth component independently evaluates the completed harmony.

It does not generate replacement chords. Its purpose is quality control: identifying where the output is supported, stylistically unusual, uncertain, or likely incorrect.

## Input

- Synchronized melody
- Refined chord progression
- Actual chord voicings
- Chord timing
- Tempo and beats
- Key and mode
- Phrase boundaries
- Selected genre

## Validation dimensions

The component examines seven evidence areas:

1. Melody–chord compatibility
2. Voice leading
3. Cadential behaviour
4. Genre authenticity
5. Harmonic structure and complexity
6. Chord confidence
7. Overall musical realism

## Hybrid validation approach

Music-theory rules provide interpretable evidence, such as:

- Whether strong melody notes belong to the chord
- Whether dissonances resolve
- Whether chord transitions are excessively abrupt
- Whether cadences behave as expected

A shared Transformer provides contextual evidence, including:

- Genre suitability
- Longer progression context
- Learned exception patterns
- Chord confidence
- Overall sequence realism

## Harmonic exception trajectory

A central contribution is evaluating every detected harmonic issue through four expanding levels of context:

1. Current chord
2. Chord transition
3. Bar or loop
4. Complete phrase

A melody note may initially appear incompatible with the current chord. However, it might be:

- A passing note
- A suspension
- An anticipation
- A delayed resolution
- A genre-specific tension
- Part of a larger phrase-level movement

The validator records the first context level that provides a stable musical explanation. This is called the **rescue depth**.

It also records:

- Whether the explanation remains valid at wider context levels
- Whether theory and learned evidence conflict
- Whether the system has enough evidence to make a reliable judgement

## Output

Each event receives one of the following decisions:

- Supported
- Supported with a stylistic warning
- Review recommended
- Likely harmonic defect
- Insufficient evidence

The report includes:

- Location of the issue
- Melody and chord involved
- Music-theory explanation
- Genre-based evidence
- Rescue depth
- Confidence
- Suggested review action

## Problem solved

A simple rule system may incorrectly reject expressive Jazz, Gospel, or contemporary harmony. A neural model may accept unusual outputs without explanation. An average score may allow high scores in other areas to hide one severe defect.

The proposed validator is non-compensatory: an unresolved critical problem cannot disappear merely because the other quality scores are high.

---

# 10. How the Four Components Work Together

The components are connected but have clearly separated responsibilities.

| Stage | Main question | Primary output |
|---|---|---|
| Component 1 | What is happening in the vocal melody? | Structured harmony-oriented melody events |
| Component 2 | What harmonic functions should accompany it? | Roman-numeral progression |
| Component 3 | How should those functions be realised for this genre? | Detailed timed chord symbols |
| Component 4 | Is the completed result supported or problematic? | Explainable validation report |

This separation also makes errors easier to locate. For example:

- Incorrect notes indicate a Component 1 problem.
- Weak progression structure indicates a Component 2 problem.
- Inappropriate extensions indicate a Component 3 problem.
- Incorrect acceptance or rejection indicates a Component 4 problem.

---

# 11. Overall Research Contribution

The expected overall contributions are:

1. An end-to-end pipeline from raw vocal audio to validated chord accompaniment.

2. A harmony-oriented melody representation that goes beyond ordinary transcription.

3. A functional Roman-numeral planning stage that separates harmonic structure from detailed realization.

4. Genre- and mode-conditioned harmony generation.

5. A controllable sequence-aware chord-refinement method.

6. Music-theory constraints that preserve harmonic structure.

7. An explainable validation framework based on context expansion, rescue depth, persistence, conflict detection, and abstention.

8. A modular evaluation method that determines which part of the system contributes to musical quality.

---

# 12. Evaluation Strategy

The full system should be evaluated through both objective measurements and human listening tests.

## Component-level evaluation

### Component 1

- Pitch accuracy
- Onset and duration accuracy
- Tempo and beat accuracy
- Key-detection accuracy
- Phrase-boundary F1 score
- Downstream harmonization improvement

### Component 2

- Melody–chord compatibility
- Functional progression coherence
- Cadence accuracy
- Genre differentiation
- Candidate diversity
- Cross-key generalisation

### Component 3

- Refinement classification accuracy
- Macro-F1 for rare chord classes
- Valid-output rate
- Structural preservation
- Melody–extension compatibility
- Perceived richness and controllability

### Component 4

- Defect-localisation accuracy
- False rejection of valid stylistic exceptions
- False acceptance of genuine defects
- Confidence calibration
- Abstention quality
- Improvement gained from wider context

## Full-system evaluation

Listeners can compare:

- Basic chord generation
- Genre-conditioned functional generation
- Functional generation with refinement
- Complete system with validation

Participants may rate:

- Melody–chord fit
- Harmonic coherence
- Genre suitability
- Musical richness
- Naturalness
- Overall preference

Dataset splitting should occur at song or artist level to prevent different sections of the same song from appearing in both training and testing data.

---

# 13. Research Foundation and Related Papers

The project builds upon several important research directions.

Vaswani et al.’s *Attention Is All You Need* introduced the Transformer architecture. Its self-attention mechanism supports the modelling of long-range relationships and forms the architectural basis of the functional planning and contextual validation models. :codex-file-citation{path="/Users/praise/.codex/.chatgpt-projects/g-p-6a59cbe3a0488191b5f91ec64e37e0f7/sources/Attention Is All You Need.pdf" purpose="source"}

Yeh et al.’s work on automatic melody harmonization with triad chords compares template-based, probabilistic, evolutionary, and recurrent neural approaches. Its large melody–chord dataset, objective metrics, and human study demonstrate that harmonization should be evaluated through several musical dimensions rather than simple chord accuracy alone. :codex-file-citation{path="/Users/praise/.codex/.chatgpt-projects/g-p-6a59cbe3a0488191b5f91ec64e37e0f7/sources/Automatic_Melody_Harmonization_with_Triad_Chords_A.pdf" purpose="source"}

Rhyu et al.’s *Translating Melody to Chord* treats melody harmonization as a sequence-translation task and explores structured, flexible, and diverse Transformer-based harmonization. This supports the use of a melody-conditioned Transformer in Component 2. :codex-file-citation{path="/Users/praise/.codex/.chatgpt-projects/g-p-6a59cbe3a0488191b5f91ec64e37e0f7/sources/Translating_Melody_to_Chord_Structured_and_Flexible_Harmonization_of_Melody_With_Transformer.pdf" purpose="source"}

*AutoHarmonizer* uses melody information, musical metadata, and flexible harmonic rhythm to produce a large variety of chord types. It demonstrates the importance of key, timing, and harmonic density, while its large chord vocabulary also helps motivate separating functional planning from detailed chord realization. :codex-file-citation{path="/Users/praise/.codex/.chatgpt-projects/g-p-6a59cbe3a0488191b5f91ec64e37e0f7/sources/AutoHarmonizer.pdf" purpose="source"}

Ji et al.’s *RL-Chord* combines a convolutional LSTM with reinforcement learning and theory-informed rewards. It supports the idea that melody context, chord-sequence context, and explicit musical knowledge can be combined instead of relying entirely on unconstrained prediction. :codex-file-citation{path="/Users/praise/.codex/.chatgpt-projects/g-p-6a59cbe3a0488191b5f91ec64e37e0f7/sources/RL-Chord_CLSTM-Based_Melody_Harmonization_Using_Deep_Reinforcement_Learning.pdf" purpose="source"}

Huang and Yang’s work on emotion-driven melody harmonization uses functional Roman-numeral representation, explicit key modelling, and conditional Transformer generation. It directly supports the project’s use of key-independent harmonic functions and external conditioning information. :codex-file-citation{path="/Users/praise/.codex/.chatgpt-projects/g-p-6a59cbe3a0488191b5f91ec64e37e0f7/sources/EMOTION-DRIVEN MELODY HARMONIZATION.pdf" purpose="source"}

The curriculum-masking study *Pay (Cross) Attention to the Melody* shows that a harmonization model may over-rely on previous chord information and underuse the melody. This is important for Component 2 because it motivates explicit testing of whether the generated harmony is genuinely conditioned on the vocal melody. :codex-file-citation{path="/Users/praise/.codex/.chatgpt-projects/g-p-6a59cbe3a0488191b5f91ec64e37e0f7/sources/Pay (Cross) Attention to the Melody- Curriculum Masking for Single-Encoder Melodic Harmonization.pdf" purpose="source"}

The study of Pop and Jazz dataset-mixing ratios demonstrates that genre adaptation can improve one musical style while damaging previously learned styles if the training data is not balanced carefully. This supports genre-aware fine-tuning, rehearsal data, and genre-specific evaluation. :codex-file-citation{path="/Users/praise/.codex/.chatgpt-projects/g-p-6a59cbe3a0488191b5f91ec64e37e0f7/sources/Empirical Study of Pop and Jazz Mix Ratios for Genre-Adaptive Chord Generation.pdf" purpose="source"}

---

# 14. Research Gap

Previous studies have investigated melody harmonization, Transformer chord generation, harmonic-rhythm control, emotion conditioning, reinforcement learning, and genre adaptation.

However, the proposed research addresses the combined gap of producing a system that:

- Begins directly from raw vocal audio
- Extracts features specifically intended for harmonization
- Plans harmony using key-independent musical functions
- Separates harmonic planning from detailed chord enrichment
- Offers user-controlled refinement
- Preserves the original harmonic structure
- Validates the final progression independently
- Explains whether unusual harmony is a valid stylistic exception or a genuine defect

The complete four-stage combination is therefore the primary system-level research contribution.

---

# 15. Scope and Limitations

The research focuses on symbolic harmony accompanying a monophonic vocal melody.

It does not attempt to evaluate:

- Vocal performance quality
- Audio mixing quality
- Instrument timbre
- Emotional quality in every musical culture
- The universal artistic value of a composition

The system’s decisions will also depend on:

- Accuracy of melody transcription
- Availability of genre-labelled data
- Balance between genres
- Accuracy of chord and melody annotations
- Representation of uncommon chords
- Subjectivity in human musical judgement

The validation report should therefore be presented as decision support rather than an absolute judgement of musical quality.

---

# 16. Concise Explanation for a Presentation or Viva

Our research develops a genre-aware AI system that converts a raw vocal melody into a complete and validated chord accompaniment. The system first extracts notes, rhythm, key, phrase, and harmonic features from the vocal recording. A Transformer then creates a functional Roman-numeral chord progression based on the melody, genre, and mode. A sequence-aware BiLSTM refines the basic chords using extensions, suspensions, and controlled chord transitions while preserving the original harmonic structure. Finally, a hybrid rule-based and Transformer-based validator examines the result at chord, transition, bar, and phrase levels. This allows the system to distinguish valid stylistic tensions from genuine harmonic errors and provide an explainable validation report.
