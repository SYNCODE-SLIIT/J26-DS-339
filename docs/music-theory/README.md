# Music Theory Knowledge Base for Melody Harmonization

This knowledge base covers the music-theory concepts needed to understand and design melody harmonization systems. It intentionally focuses on theory rather than machine learning.

## 1. Foundations of Music

Before studying harmony, learn its basic building blocks.

### Sound, pitch, and notes

Sound is produced by vibration. **Pitch** is the perceptual quality that lets us hear a sound as relatively high or low, and a **note** is a named musical pitch.

For example:

```text
A4 = 440 Hz
```

- Higher frequency generally produces a higher perceived pitch.
- Lower frequency generally produces a lower perceived pitch.

### Pitch classes

In twelve-tone equal temperament, the octave is divided into 12 pitch classes:

```text
C  C#/Db  D  D#/Eb  E  F  F#/Gb  G  G#/Ab  A  A#/Bb  B
```

Enharmonic spellings such as `C#` and `Db` sound at the same pitch in equal temperament, but their names communicate different harmonic meanings.

After B, the letter-name pattern repeats:

```text
C3 -> C4 -> C5
```

`C` is the pitch-class name; the number identifies the octave register.

### Intervals

An **interval** is the distance between two notes.

Examples:

```text
C -> D  = major second (whole step)
C -> E  = major third
C -> G  = perfect fifth
```

Intervals are foundational to:

- scales;
- chords;
- melodies;
- harmony.

### Semitones and whole tones

The **semitone** (or half step) is the smallest standard interval in twelve-tone equal temperament.

```text
C -> C# = one semitone
```

A **whole tone** (or whole step) contains two semitones.

```text
C -> D = two semitones
```

### Octaves

An **octave** spans from one note to the next note with the same letter name:

```text
C4 -> C5
```

Ascending one octave doubles the frequency; descending one octave halves it.

## 2. Scales and Tonality

### Scales

A **scale** is an ordered collection of pitches.

For example, the C-major scale is:

```text
C D E F G A B C
```

### Keys

A **key** organizes music around a tonal center and a major or minor collection.

In C major, C feels like “home,” and the music tends to gravitate toward it.

### Tonal center

The **tonal center**, or **tonic**, is the pitch that sounds like the primary point of rest. In C major, the tonal center is C.

### Major-scale formula

The major-scale step pattern is:

```text
Whole - Whole - Half - Whole - Whole - Whole - Half
```

Applied to C:

```text
C - D - E - F - G - A - B - C
```

### Minor scales

Minor-key music commonly draws from three related scale forms.

#### Natural minor

```text
A B C D E F G A
```

#### Harmonic minor

Raise the seventh scale degree of natural minor:

```text
A B C D E F G# A
```

The raised seventh creates a leading tone and supports a stronger dominant-to-tonic resolution.

#### Melodic minor

In traditional classical practice, raise the sixth and seventh scale degrees when ascending and usually restore them when descending:

```text
Ascending:  A B C D E F# G# A
Descending: A G F E D C B A
```

In jazz theory, “melodic minor” commonly means the ascending form used in both directions.

## 3. Modes

**Modes** are scale collections defined by distinctive interval patterns. The seven diatonic modes are:

```text
Ionian  Dorian  Phrygian  Lydian  Mixolydian  Aeolian  Locrian
```

The notes of C major can illustrate all seven modes when a different note is heard as the tonal center:

```text
C to C = C Ionian
D to D = D Dorian
E to E = E Phrygian
F to F = F Lydian
G to G = G Mixolydian
A to A = A Aeolian
B to B = B Locrian
```

Simply starting on a different note does not necessarily establish a mode; the harmony and melodic emphasis must make that note sound central.

### Dorian

Dorian has a minor third and a raised sixth relative to natural minor. It is common in jazz, funk, folk, and modal music.

### Mixolydian

Mixolydian has a major third and a lowered seventh relative to major. It is common in rock, blues, folk, and modal jazz, and it supports a major `bVII` chord.

## 4. Chords

### Triads

A **triad** is a three-note chord built by stacking thirds:

```text
Root + third + fifth
```

For example:

```text
C major = C E G
```

### Chord quality

Chord quality is determined by the intervals above the root.

#### Major triad

```text
C E G
```

Major third plus perfect fifth.

#### Minor triad

```text
C Eb G
```

Minor third plus perfect fifth.

#### Diminished triad

```text
B D F
```

Minor third plus diminished fifth. It is strongly unstable in tonal contexts.

#### Augmented triad

```text
C E G#
```

Major third plus augmented fifth. Its symmetry creates ambiguity and tension.

Words such as “happy” or “dark” can describe common associations, but musical effect always depends on register, voicing, rhythm, timbre, and context.

### Seventh chords

A **seventh chord** adds another stacked third above a triad.

#### Major seventh

```text
Cmaj7 = C E G B
```

#### Minor seventh

```text
Cm7 = C Eb G Bb
```

#### Dominant seventh

```text
C7 = C E G Bb
```

A dominant seventh contains a major triad and a minor seventh. In functional harmony, it strongly tends to resolve a perfect fifth downward, as in `G7 -> C`.

#### Half-diminished seventh

```text
Cm7b5 = Cø7 = C Eb Gb Bb
```

#### Fully diminished seventh

```text
Cdim7 = C°7 = C Eb Gb Bbb
```

The spelling `Bbb` is theoretically important: it is a diminished seventh above C, even though it sounds like A in equal temperament.

## 5. Diatonic Harmony

**Diatonic** harmony uses notes belonging to the current scale or key.

The C-major collection is:

```text
C D E F G A B
```

Build a triad on each scale degree by stacking alternate scale notes:

| Degree | Notes | Chord | Common function |
| --- | --- | --- | --- |
| `I` | C E G | C major | Tonic |
| `ii` | D F A | D minor | Predominant |
| `iii` | E G B | E minor | Context-dependent; often tonic prolongation |
| `IV` | F A C | F major | Predominant |
| `V` | G B D | G major | Dominant |
| `vi` | A C E | A minor | Tonic substitute or prolongation |
| `vii°` | B D F | B diminished | Dominant |

The major-key diatonic-triad pattern is:

```text
Major - minor - minor - Major - Major - minor - diminished
```

## 6. Roman-Numeral Analysis

Roman numerals describe chord roots and qualities relative to a key.

In C major:

```text
C    = I
Dm   = ii
Em   = iii
F    = IV
G    = V
Am   = vi
Bdim = vii°
```

- Uppercase numerals represent major triads: `I`, `IV`, `V`.
- Lowercase numerals represent minor triads: `ii`, `iii`, `vi`.
- `°` means diminished.
- `+` means augmented.
- Accidentals show altered roots, as in `bVII` or `#iv°`.

Chord members, extensions, and inversions can be included:

```text
Imaj7  V7  ii9  I6  V4/3
```

Figured-bass numbers such as `6`, `6/4`, `6/5`, `4/3`, and `4/2` indicate inversions in classical Roman-numeral analysis.

## 7. Harmonic Function

**Functional harmony** describes how a chord behaves within a tonal progression. The three broad functional families are:

```text
Tonic (T) -> Predominant (PD) -> Dominant (D) -> Tonic (T)
```

Not every piece follows this complete cycle, and a chord’s function can change with context.

### Tonic function

Tonic function conveys stability, arrival, or “home.”

Common tonic-family chords in major include:

```text
I  vi  iii
```

`I` is the clearest tonic. The functions of `vi` and especially `iii` depend heavily on context.

### Predominant function

Predominant function moves away from tonic and prepares dominant.

Common predominant chords include:

```text
ii  IV
```

### Dominant function

Dominant function creates directed tension that tends to resolve to tonic.

Common dominant chords include:

```text
V  V7  vii°  vii°7
```

### Basic harmonic motion

A common tonal pattern is:

```text
I -> IV -> V -> I
T    PD   D    T
```

## 8. Cadences

A **cadence** is a harmonic and melodic gesture that closes or articulates a phrase.

### Perfect authentic cadence (PAC)

```text
V or V7 -> I
```

For a conventional perfect authentic cadence, both chords are in root position and scale degree 1 is in the soprano on the final tonic. It is the strongest standard tonal closure.

### Imperfect authentic cadence (IAC)

```text
V or V7 -> I
```

The cadence is imperfect when it lacks one or more PAC conditions—for example, a chord is inverted or the soprano ends on scale degree 3 or 5.

### Half cadence (HC)

A half cadence ends on `V` and sounds open or unfinished.

### Plagal cadence

```text
IV -> I
```

This is often called the “Amen” cadence.

### Deceptive cadence

```text
V -> vi
```

The expected resolution to `I` is redirected, most commonly to `vi` in major.

## 9. Melody and Harmony

A melody note does not determine a single chord.

For example, the melody note G can belong to:

```text
C major  = fifth
E minor  = third
G major  = root
A minor 7 = seventh
```

Relative to a chord, a melody note may be:

- the root;
- the third;
- the fifth;
- the seventh;
- an extension;
- a non-chord tone.

When selecting harmony, consider metrical position, phrase direction, surrounding notes, harmonic rhythm, bass motion, style, and the note’s tendency to resolve.

### Non-chord tones

**Non-chord tones** are notes that are not members of the current chord. They may decorate, connect, delay, or intensify the harmony.

#### Passing tone

A passing tone fills the step between two chord tones:

```text
C - D - E
```

If C and E belong to the harmony, D may be a passing tone.

#### Neighbor tone

A neighbor tone moves by step away from a stable note and returns:

```text
C - D - C
```

#### Suspension

A suspension holds or repeats a note from the previous harmony, creating a dissonance that normally resolves by step downward. A full suspension has preparation, suspension, and resolution phases.

#### Appoggiatura

An appoggiatura is approached by leap, receives emphasis, and usually resolves by step.

#### Other common non-chord tones

- anticipation;
- escape tone;
- retardation;
- pedal point;
- cambiata.

> **Important:** Not every melody note needs a new chord. Effective harmony supports the phrase rather than mechanically matching every note.

## 10. Chord Progressions

A **chord progression** is a sequence of chords understood through harmonic motion and context.

### Common-practice tonal progression

```text
I - IV - V - I
```

### Common pop loop

```text
I - V - vi - IV
```

In C major:

```text
C - G - Am - F
```

### Jazz ii–V–I

```text
ii7 - V7 - Imaj7
```

In C major:

```text
Dm7 - G7 - Cmaj7
```

### Basic blues framework

Blues harmony is commonly organized around `I`, `IV`, and `V`, often as dominant-seventh chords within a 12-bar form. Its pitch language does not always behave like conventional major-key functional harmony.

## 11. Chord Extensions and Color

“Color” is an informal term for notes, voicings, and alterations that shape a chord’s character without necessarily changing its basic function.

Starting with C major:

```text
C      = C E G
Cmaj7  = C E G B
Cmaj9  = C E G B D
C6     = C E G A
C6/9   = C E G A D
```

Depending on context, all of these may express tonic function in C major.

### Function versus color

```text
Roman numeral: I
Chord symbol:  Cmaj9
Function:      tonic
Color:         major seventh and ninth
```

Extensions can affect voice leading and perceived tension, so “color” and function are related even when they are analytically distinct.

## 12. Chromatic Harmony

**Chromatic harmony** uses pitches or chords outside the prevailing diatonic collection.

The notes of C major are:

```text
C D E F G A B
```

An E-major chord contains:

```text
E G# B
```

Because G# is outside C major, E major is chromatic in that key.

Chromatic harmony can create:

- directed tension;
- stronger motion;
- expressive color;
- surprise;
- temporary or lasting changes of tonal center.

## 13. Secondary Dominants

A **secondary dominant** is a major triad or dominant-seventh chord that functions as the dominant of a diatonic chord other than the tonic. It temporarily makes its target sound tonic-like.

### Example: targeting vi

In C major, the `vi` chord is A minor. Its dominant is E major or E7:

```text
E7 -> Am
V7/vi -> vi
```

`V7/vi` is read “five-seven of six.” The G# in E7 is chromatic to C major and acts as the leading tone to A.

### Example: targeting ii

In C major:

```text
A7 -> Dm
V7/ii -> ii
```

The C# in A7 is chromatic to C major and leads upward to D.

A convincing secondary dominant normally contains an alteration that creates a leading tone or other dominant tendency toward its target.

## 14. Tonicization and Modulation

### Tonicization

**Tonicization** briefly emphasizes a non-tonic chord without establishing a lasting new key.

```text
E7 -> Am
V7/vi -> vi
```

Here, A minor may feel temporarily tonicized while C major remains the governing key.

### Modulation

**Modulation** establishes a new key strongly enough for it to become the new tonal framework.

```text
C major -> G major
```

The boundary between extended tonicization and modulation depends on duration, cadential confirmation, thematic emphasis, and analytical context.

## 15. Borrowed Chords and Modal Interchange

**Modal interchange** borrows material from a parallel mode—that is, a mode with the same tonic.

Compare C major and C natural minor:

```text
C major: C D E  F G A  B
C minor: C D Eb F G Ab Bb
```

A common borrowed-chord progression is:

```text
C - Bb - F - C
I - bVII - IV - I
```

The Bb-major chord is borrowed from C minor or another parallel C mode containing `b7`.

Common chords borrowed from the parallel minor include:

```text
iv  bVI  bVII  ii°
```

Modal interchange is common in rock, pop, jazz, and film music.

## 16. Jazz Harmony

Jazz harmony makes extensive use of:

- seventh chords;
- upper extensions;
- substitutions;
- altered dominants;
- chromatic approach harmony;
- modal harmony.

### The ii–V–I progression

In C major:

```text
Dm7 -> G7 -> Cmaj7
ii7 -> V7 -> Imaj7
```

This progression creates strong root motion and efficient guide-tone resolution.

### Chord-scale theory

Chord-scale theory relates a chord to one or more pitch collections available for improvisation or composition.

Basic diatonic examples in C major include:

```text
Cmaj7 -> C Ionian
Dm7   -> D Dorian
G7    -> G Mixolydian
```

These are starting points, not automatic rules. Harmonic function, melody, alterations, style, and voice leading determine which notes are effective.

### Tritone substitution

A dominant-seventh chord may be replaced by a dominant-seventh chord whose root is a tritone away.

```text
Original:   G7  -> C
Substitute: Db7 -> C
```

`G7` contains the tritone B–F. `Db7` contains Cb–F, which is enharmonically the same tritone in equal temperament. The substitute also gives chromatic bass motion from Db to C.

## 17. Gospel Harmony

Gospel harmony often features:

- extended and altered dominants;
- chromatic bass movement;
- passing chords;
- secondary dominants and dominant chains;
- rich voicings;
- call-and-response phrasing.

Common frameworks include:

```text
I - vi - ii - V
```

and a descending cycle-of-fifths pattern often called “7–3–6–2–5–1”:

```text
vii -> iii -> vi -> ii -> V -> I
```

The exact chord qualities and chromatic alterations vary by style and context.

## 18. Voice Leading

**Voice leading** is the way each individual musical line moves from one note to the next as the harmony changes.

Effective voice leading often uses:

- common tones;
- stepwise movement;
- contrary or oblique motion;
- clear treatment of tendency tones;
- singable, independent lines.

For example:

```text
C major: C E G
G major: B D G
```

- G remains as a common tone.
- E moves down to D.
- C moves down to B.

In common-practice part writing, typical guidelines include:

- avoid parallel perfect fifths and octaves between independent voices;
- resolve the leading tone upward to the tonic when its tendency is active;
- resolve a chordal seventh downward by step in standard dominant-seventh treatment;
- avoid voice crossing and excessive spacing when writing in a chorale texture.

These are style-specific principles, not universal laws for all music.

## 19. Advanced Topics to Study Next

### Harmony and analysis

- functional harmony;
- chromatic harmony;
- modal harmony;
- voice leading;
- counterpoint;
- form and phrase structure;
- Schenkerian analysis;
- Neo-Riemannian theory;
- set theory and post-tonal analysis.

### Chord vocabulary

- seventh, ninth, eleventh, and thirteenth chords;
- suspended and added-note chords;
- altered dominants;
- slash chords and inversions;
- polychords and upper structures;
- quartal and quintal harmony.

### Progressions and schemas

- circle-of-fifths progressions;
- ii–V–I progressions;
- turnarounds;
- 12-bar blues;
- pop harmonic schemas and loops;
- jazz standards;
- gospel progressions;
- sequences and harmonic rhythm.

### Melody

- motives;
- phrases and periods;
- contour;
- melodic cadences;
- tendency tones;
- melodic tension and release;
- ornamentation;
- chord-tone targeting.

### Rhythm and meter

- beat, subdivision, and meter;
- syncopation;
- rhythmic motives;
- harmonic rhythm;
- phrase rhythm;
- groove and microtiming.

### Genre-focused harmony

#### Classical

- functional harmony;
- counterpoint;
- cadences;
- form;
- modulation.

#### Jazz

- extensions and alterations;
- substitutions;
- modal harmony;
- guide tones;
- reharmonization.

#### Pop

- harmonic loops;
- modal interchange;
- repetitive schemas;
- harmonic ambiguity;
- production-informed harmony.

#### Gospel

- chromatic movement;
- dominant chains;
- passing harmony;
- voicing and voice exchange;
- rhythmic placement.

## 20. Core Mental Model

A useful conceptual hierarchy is:

```text
Sound
  -> Pitch and rhythm
    -> Intervals
      -> Scales and modes
        -> Keys and tonal centers
          -> Chords and voicings
            -> Harmonic functions
              -> Progressions and cadences
                -> Form, style, and expression
```

For melody harmonization, a complementary decision path is:

```text
Melody note and phrase context
  -> Candidate chord membership
    -> Harmonic function
      -> Bass motion and voice leading
        -> Chord quality and extensions
          -> Voicing, register, rhythm, and style
```

The central idea is that a chord is more than a collection of notes. In context, it has:

1. an identity;
2. a harmonic function;
3. a relationship to the melody and bass;
4. a position within a phrase and key;
5. a characteristic color and voicing;
6. a tendency to remain, depart, or resolve.

Those layers of context explain why the same melody can support several valid harmonizations.
