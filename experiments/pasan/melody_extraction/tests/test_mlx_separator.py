from __future__ import annotations

import numpy as np
import soundfile as sf

from experiments.pasan.melody_extraction.mlx_separator import _stereo_input


def test_stereo_input_expands_mono_without_touching_source(tmp_path) -> None:
    source = tmp_path / "mono.wav"
    sf.write(source, np.zeros(800, dtype=np.float32), 8_000)

    with _stereo_input(source) as converted:
        assert converted != source
        assert sf.info(converted).channels == 2
        assert source.exists()

    assert not converted.exists()
    assert sf.info(source).channels == 1

