"""A window/patch that starts within one crossfade of the buffer end
must fade over the audio that exists, not crash (sa3-small sessions
died at the loop end with a (n,2) vs (1200,1) broadcast error)."""

from types import SimpleNamespace

import numpy as np

from acestep.streaming.pipeline_runner import PipelineRunner


def test_patch_wrapped_near_buffer_end_does_not_crash():
    current = np.ones((48000, 2), dtype=np.float32)
    written = {}
    fake = SimpleNamespace(
        audio_eng=SimpleNamespace(
            patch_window=lambda pcm, start: written.update(pcm=pcm, start=start)),
        on_audio_ready=lambda pcm, s, e: None,
    )
    backend = SimpleNamespace(
        render_window=lambda s: SimpleNamespace(
            pcm=np.zeros((9600, 2), dtype=np.float32)))
    start = current.shape[0] - 769  # less than the 1200-sample fade remains
    PipelineRunner._patch_wrapped(fake, backend, current, 0.0, start, 9600)
    pcm = written["pcm"]
    assert written["start"] == start
    # fade-in over the 769 live samples: starts at the old audio
    assert np.isclose(pcm[0, 0], 1.0)
    assert np.isclose(pcm[768, 0], 0.0)
