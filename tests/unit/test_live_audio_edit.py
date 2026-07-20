from types import SimpleNamespace
import threading

import numpy as np
import pytest
import torch

from acestep.engine.stream import SlotRequest
from acestep.nodes.diffusion_nodes import StreamDenoise
from acestep.nodes.types import Latent
from acestep.streaming.audio_edit import (
    AudioEditError,
    LiveAudioEdit,
    EditRegion,
    composite_window,
    constrain_audio_edit,
    parse_live_audio_edit,
    regenerate_mask,
    sa3_inpaint_bundle,
)
from acestep.streaming.ace_backend import ACEStepBackend
from acestep.streaming.audio_engine import AudioEngine
from acestep.streaming.session import StreamingSession


def test_stream_denoise_registers_hidden_audio_edit_snapshot():
    param = next(
        p for p in StreamDenoise.get_definition().params
        if p.name == "audio_edit"
    )
    assert param.type == "any"
    assert param.default is None
    assert param.hidden is True


def test_explicit_right_extension_requires_only_the_added_tail():
    edit = parse_live_audio_edit(
        [{"start_s": 10, "end_s": 15}],
        enabled=True,
        source_mode="waveform",
        strength=1,
        canvas_duration_s=15,
        source_duration_s=10,
        right_extension_s=5,
    )
    assert edit.regions == (EditRegion(10, 15),)

    with pytest.raises(AudioEditError, match="explicit waveform extension"):
        parse_live_audio_edit(
            [{"start_s": 11, "end_s": 15}],
            enabled=True,
            source_mode="waveform",
            strength=1,
            canvas_duration_s=15,
            source_duration_s=10,
            right_extension_s=5,
        )


def test_ordinary_canvas_padding_does_not_require_tail_coverage():
    edit = parse_live_audio_edit(
        [{"start_s": 2, "end_s": 3}],
        enabled=True,
        source_mode="waveform",
        strength=1,
        canvas_duration_s=60,
        source_duration_s=10,
    )
    assert edit.regions == (EditRegion(2, 3),)


def test_empty_waveform_edit_preserves_the_entire_source():
    edit = parse_live_audio_edit(
        [],
        enabled=True,
        source_mode="waveform",
        strength=1,
        canvas_duration_s=1,
        source_duration_s=1,
    )
    assert edit.enabled is True
    assert edit.regions == ()

    source = np.linspace(-1, 1, 200, dtype=np.float32).reshape(100, 2)
    generated = np.full((100, 2), 0.75, dtype=np.float32)
    out = composite_window(
        generated,
        start_sample=0,
        source=source,
        edit=edit,
        sample_rate=100,
        crossfade_s=0,
    )
    np.testing.assert_array_equal(out, source)


def test_empty_current_edit_rejects_stale_unmasked_generation():
    current = LiveAudioEdit(True, (), "waveform", 1)
    stale = LiveAudioEdit(False)
    effective = constrain_audio_edit(current, stale)
    assert effective.enabled is True
    assert effective.regions == ()

    source = np.arange(200, dtype=np.float32).reshape(100, 2)
    generated = np.full((100, 2), -123, dtype=np.float32)
    np.testing.assert_array_equal(
        composite_window(
            generated,
            start_sample=0,
            source=source,
            edit=effective,
            sample_rate=100,
            crossfade_s=0,
        ),
        source,
    )


def test_changed_regions_only_accept_current_and_emerged_intersection():
    current = LiveAudioEdit(True, (EditRegion(2, 5),), "waveform", 1)
    emerged = LiveAudioEdit(True, (EditRegion(0, 3),), "waveform", 1)
    effective = constrain_audio_edit(current, emerged)
    assert effective.regions == (EditRegion(2, 3),)


def test_session_reasserts_empty_mask_after_runner_crossfade():
    source = np.arange(200, dtype=np.float32).reshape(100, 2)
    generated = np.full((100, 2), -123, dtype=np.float32)
    empty_edit = LiveAudioEdit(True, (), "waveform", 1)
    published = []

    streaming = object.__new__(StreamingSession)
    streaming.backend = SimpleNamespace(
        finalize_audio_edit_window=lambda pcm, start: composite_window(
            pcm,
            start_sample=start,
            source=source,
            edit=empty_edit,
            sample_rate=100,
            crossfade_s=0,
        ),
    )
    streaming._audio_edit_delivery_lock = threading.RLock()
    streaming.audio_eng = AudioEngine(generated, 100)
    streaming.state = SimpleNamespace(params={}, n_channels=2)
    streaming.bus = SimpleNamespace(publish=published.append)

    streaming._on_audio_ready(generated.copy(), 0, 100)

    np.testing.assert_array_equal(streaming.audio_eng.current, source)
    np.testing.assert_array_equal(published[0].audio, source)


def test_explicit_right_extension_excludes_backend_padding_after_request():
    edit = parse_live_audio_edit(
        [{"start_s": 10, "end_s": 15}],
        enabled=True,
        source_mode="waveform",
        strength=1,
        canvas_duration_s=16,
        source_duration_s=10,
        right_extension_s=5,
    )
    assert edit.regions == (EditRegion(10, 15),)


def test_explicit_left_extension_requires_only_the_added_tail():
    edit = parse_live_audio_edit(
        [
            {"start_s": 0, "end_s": 5},
            {"start_s": 8, "end_s": 9},
        ],
        enabled=True,
        source_mode="waveform",
        strength=1,
        canvas_duration_s=15,
        source_duration_s=10,
        left_extension_s=5,
    )
    assert edit.regions[0] == EditRegion(0, 5)

    with pytest.raises(AudioEditError, match="explicit waveform extension"):
        parse_live_audio_edit(
            [{"start_s": 1, "end_s": 5}],
            enabled=True,
            source_mode="waveform",
            strength=1,
            canvas_duration_s=15,
            source_duration_s=10,
            left_extension_s=5,
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


def test_window_compositor_regenerates_left_extension_and_preserves_source():
    source = np.concatenate([
        np.zeros((20, 2), dtype=np.float32),
        np.ones((80, 2), dtype=np.float32),
    ])
    generated = np.full((100, 2), 0.25, dtype=np.float32)
    edit = LiveAudioEdit(True, (EditRegion(0, 0.02),), "waveform", 1)
    out = composite_window(
        generated,
        start_sample=0,
        source=source,
        edit=edit,
        sample_rate=1000,
        crossfade_s=0,
    )
    np.testing.assert_array_equal(out[:20], 0.25)
    np.testing.assert_array_equal(out[20:], 1)


def test_window_compositor_regenerates_right_extension_and_preserves_source():
    source = np.concatenate([
        np.ones((80, 2), dtype=np.float32),
        np.zeros((20, 2), dtype=np.float32),
    ])
    generated = np.full((100, 2), 0.25, dtype=np.float32)
    edit = LiveAudioEdit(True, (EditRegion(0.08, 0.1),), "waveform", 1)
    out = composite_window(
        generated,
        start_sample=0,
        source=source,
        edit=edit,
        sample_rate=1000,
        crossfade_s=0,
    )
    np.testing.assert_array_equal(out[:80], 1)
    np.testing.assert_array_equal(out[80:], 0.25)


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
