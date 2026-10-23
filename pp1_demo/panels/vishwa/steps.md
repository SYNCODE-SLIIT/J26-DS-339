### From voice to structured melody

1. **Read the recording:** preserve the original file and time origin.
2. **Prepare audio:** average channels to mono, resample to 22,050 Hz and normalize the peak to 0.95. No trimming or noise removal is applied.
3. **Measure sound:** calculate 80 Mel bands using a 4,096-sample window and 256-sample hop.
4. **Estimate pitch:** pYIN estimates fundamental frequency and voiced frames. Silence is masked rather than treated as a note.
5. **Build notes:** group contiguous voiced frames at the nearest semitone; retain events lasting at least 60 ms.
6. **Send melody downstream:** note pitch, onset and duration enter Pasan's adapter. The operator supplies a major reference key and beat grid; these are not detected by this component.
7. **Compare learned pilots:** CNN and CNN–Transformer predict rest or pitch class. They do not determine octave and are not used to manufacture complete note events.

**Music terms:** pitch is how high or low a sound is; onset is when a note begins; duration is how long it lasts; pitch class groups octave-equivalent notes such as C3 and C4.
