from types import SimpleNamespace

import numpy as np
import pytest
import torch

from acestep.engine.stream import SlotRequest
from acestep.nodes.types import Latent
from acestep.streaming.audio_edit import (
    AudioEditError,
    LiveAudioEdit,
    EditRegion,
    composite_window,
    parse_live_audio_edit,
    regenerate_mask,
    sa3_inpaint_bundle,
)
from acestep.streaming.ace_backend import ACEStepBackend


def test_extension_is_a_tail_region_on_a_preallocated_canvas():
    edit = parse_live_audio_edit(
        [{"start_s": 10, "end_s": 15}],
        enabled=True,
        source_mode="waveform",
        strength=1,
        canvas_duration_s=15,
        source_duration_s=10,
    )
    assert edit.regions == (EditRegion(10, 15),)

    with pytest.raises(AudioEditError, match="preserved"):
        parse_live_audio_edit(
            [{"start_s": 11, "end_s": 15}],
            enabled=True,
            source_mode="waveform",
            strength=1,
            canvas_duration_s=15,
            source_duration_s=10,
        )


def test_ace_mask_uses_absolute_regions_for_a_walk_window():
    edit = LiveAudioEdit(True, (EditRegion(61, 62),), "waveform", 0.5)
    mask = regenerate_mask(edit, total_frames=50, rate_hz=25, offset_s=60)
    assert torch.count_nonzero(mask == 0.5).item() == 25
    assert torch.count_nonzero(mask).item() == 25


def test_window_compositor_restores_only_preserved_samples():
    source = torch.ones((2, 100))
    generated = np.zeros((40, 2), dtype=np.float32)
    edit = LiveAudioEdit(True, (EditRegion(0.04, 0.06),), "waveform", 1)
    out = composite_window(
        generated,
        start_sample=30,
        source=source,
        edit=edit,
        sample_rate=1000,
        crossfade_s=0,
    )
    np.testing.assert_array_equal(out[:10], 1)
    np.testing.assert_array_equal(out[10:30], 0)
    np.testing.assert_array_equal(out[30:], 1)


def test_sa3_bundle_uses_preserve_polarity_and_masked_source():
    source = torch.ones((1, 20, 256))
    base = {"local_add_cond": torch.zeros((1, 257, 20))}
    edit = LiveAudioEdit(
        True,
        (EditRegion(5 / (44100 / 4096), 10 / (44100 / 4096)),),
        "waveform",
        1,
    )
    bundle = sa3_inpaint_bundle(base, source, edit)
    mask = bundle["local_add_cond"][:, :1]
    assert torch.all(mask[..., :5] == 1)
    assert torch.all(mask[..., 5:10] == 0)
    assert torch.all(mask[..., 10:] == 1)
    assert torch.equal(bundle["local_add_cond"][:, 1:], source.movedim(1, 2) * mask)


def test_ace_generate_attaches_edit_to_the_normal_stream_tick():
    edit = LiveAudioEdit(True, (EditRegion(0, 0.4),), "waveform", 1)

    class FakeStream:
        def __init__(self):
            self.pipeline = SimpleNamespace(last_finished_request=None)
            self.kwargs = None

        def tick(self, **kwargs):
            self.kwargs = kwargs
            self.pipeline.last_finished_request = SlotRequest(
                audio_edit=kwargs["audio_edit"],
            )
            return Latent(tensor=torch.zeros((1, 25, 64)))

    backend = object.__new__(ACEStepBackend)
    backend.stream = FakeStream()
    backend._walk_active = False
    backend._walk_chunk_start_s = 0
    backend._current_shift = 3
    backend._emerged_audio_edit = None
    source = Latent(tensor=torch.zeros((1, 25, 64)))
    prep = {
        "raw": {},
        "source_lat": None,
        "live_src_lat": source,
        "audio_edit": edit,
        "denoise": 1,
        "seed": 1,
        "x0_tgt": source,
        "x0_target_curve": None,
        "initial_noise_curve": None,
        "tick_kwargs": {},
    }
    backend._generate(prep)
    sent = backend.stream.kwargs
    assert sent["audio_edit"] is edit
    assert sent["source_latent"].mask is not None
    assert backend._emerged_audio_edit is edit
