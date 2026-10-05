from pathlib import Path

from chord_refinement.src.normalize_chords import normalize_chord_label, parse_chord_label
from chord_refinement.src.utils import extract_track_id
from chord_refinement.src.extract_reference_chords import quality_category


def test_known_chords_are_normalized() -> None:
    assert normalize_chord_label("C:maj") == "C"
    assert normalize_chord_label("A:min7") == "Am7"
    assert normalize_chord_label("G:7") == "G7"
    assert normalize_chord_label("D:sus4") == "Dsus4"
    assert normalize_chord_label("N") == "N"


def test_unknown_chord_is_preserved() -> None:
    assert normalize_chord_label("C:mystery(1,2)") == "C:mystery(1,2)"


def test_inversion_structure_is_preserved() -> None:
    parsed = parse_chord_label("F:maj/3")
    assert parsed.root == "F"
    assert parsed.quality == "maj"
    assert parsed.bass_degree == "3"
    assert parsed.reference_chord == "F/3"


def test_track_id_from_low_quality_filename() -> None:
    assert extract_track_id(Path("683800.low.mp3")) == "683800"


def test_requested_quality_categories_use_raw_labels() -> None:
    assert quality_category("C#:dim") == "diminished"
    assert quality_category("C:maj7") == "major7"
    assert quality_category("Bb:min7") == "minor7"
    assert quality_category("F:7") == "dominant7"
    assert quality_category("A:sus4") == "sus4"
    assert quality_category("C:unrecognized") == "unknown/other"

