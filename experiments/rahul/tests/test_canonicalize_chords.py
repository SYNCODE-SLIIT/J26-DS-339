from experiments.rahul.src.canonicalize_chords import canonicalize_chord


def test_flat_roots_become_sharps_without_changing_suffixes():
    assert canonicalize_chord("Abm") == ("G#m", 8)
    assert canonicalize_chord("Bbm7") == ("A#m7", 10)
    assert canonicalize_chord("Ebm7") == ("D#m7", 3)
    assert canonicalize_chord("Eb7") == ("D#7", 3)
    assert canonicalize_chord("Ab7") == ("G#7", 8)
    assert canonicalize_chord("Eb/5") == ("D#/5", 3)
    assert canonicalize_chord("Ab/3") == ("G#/3", 8)


def test_quality_and_all_supported_bass_suffixes_are_opaque():
    for label in ("Gm/2", "Gm/3", "Gm/5", "Gm/b3", "Am/b7"):
        canonical, _ = canonicalize_chord(label)
        assert canonical == label


def test_n_has_no_pitch_class():
    assert canonicalize_chord("N") == ("N", None)
