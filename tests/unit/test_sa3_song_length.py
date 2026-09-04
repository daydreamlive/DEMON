"""Song-length conditioning for SA3 (the "every loop ends in an outro"
fix). CPU-only: the conditioning helper runs against a stub model and a
stub of the two ``stable_audio_3.data.utils`` functions it imports."""

from __future__ import annotations

import math
import sys
import types

import pytest
import torch

from acestep.engine import sa3_context as ctx_mod
from acestep.engine import sa3_stream_helpers as helpers

SR = 44100
DS = 4096
ALIGN = DS * 2  # chunk_size 32 // stride 16 -> 2 latents


# ---- env resolution -------------------------------------------------------


def test_song_seconds_setting_defaults_and_overrides():
    assert ctx_mod.song_seconds_setting({}) == ctx_mod.DEFAULT_SONG_SECONDS
    assert ctx_mod.song_seconds_setting({ctx_mod.SONG_SECONDS_ENV: ""}) == (
        ctx_mod.DEFAULT_SONG_SECONDS
    )
    # 0 (or negative) disables the label -> legacy semantics.
    assert ctx_mod.song_seconds_setting({ctx_mod.SONG_SECONDS_ENV: "0"}) is None
    assert ctx_mod.song_seconds_setting({ctx_mod.SONG_SECONDS_ENV: "-3"}) is None
    assert ctx_mod.song_seconds_setting({ctx_mod.SONG_SECONDS_ENV: "240"}) == 240.0
    # Clamped to the conditioner's max_val.
    assert ctx_mod.song_seconds_setting({ctx_mod.SONG_SECONDS_ENV: "9000"}) == (
        ctx_mod.SONG_SECONDS_MAX
    )
    with pytest.raises(ValueError):
        ctx_mod.song_seconds_setting({ctx_mod.SONG_SECONDS_ENV: "long"})


def test_song_schedule_setting():
    assert ctx_mod.song_schedule_from_window({}) is False
    assert ctx_mod.song_schedule_from_window(
        {ctx_mod.SONG_SCHEDULE_ENV: "window"}) is True
    assert ctx_mod.song_schedule_from_window(
        {ctx_mod.SONG_SCHEDULE_ENV: "Song"}) is False
    with pytest.raises(ValueError):
        ctx_mod.song_schedule_from_window({ctx_mod.SONG_SCHEDULE_ENV: "x"})


def test_outro_pad_setting():
    # Wrapped-loop headroom under a song label; upstream's 6 s silent
    # outro pad otherwise.
    assert ctx_mod.outro_pad_setting(180.0, {}) == ctx_mod.DEFAULT_LOOP_WRAP_S
    assert ctx_mod.outro_pad_setting(180.0, {ctx_mod.OUTRO_PAD_ENV: "0"}) == 0.0
    assert ctx_mod.outro_pad_setting(None, {}) == ctx_mod.LEGACY_OUTRO_PAD_S
    assert ctx_mod.outro_pad_setting(180.0, {ctx_mod.OUTRO_PAD_ENV: "2.5"}) == 2.5
    assert ctx_mod.outro_pad_setting(None, {ctx_mod.OUTRO_PAD_ENV: "-1"}) == 0.0
    with pytest.raises(ValueError):
        ctx_mod.outro_pad_setting(None, {ctx_mod.OUTRO_PAD_ENV: "six"})


def test_label_seconds_for():
    assert ctx_mod.label_seconds_for(30.0, 180.0) == 180.0
    assert ctx_mod.label_seconds_for(30.0, None) == 30.0
    # A loop longer than the label is still "the whole song".
    assert ctx_mod.label_seconds_for(200.0, 180.0) == 200.0
    assert ctx_mod.label_seconds_for(30, 30.0) == 30.0


# ---- prepare_sa3_conditioning against a stub model --------------------------


class _Param:
    dtype = torch.float32


class _Pretransform:
    downsampling_ratio = DS


class _Inner:
    def parameters(self):
        yield _Param()


class _Model:
    sample_rate = SR
    io_channels = 256
    sampling_dist_shift = None
    pretransform = _Pretransform()
    model = _Inner()

    def conditioner(self, conditioning, device):
        # Record what the conditioner was fed; return a token per entry.
        self.fed = conditioning
        return {"prompt": torch.zeros(1, 4, 8)}

    def get_conditioning_inputs(self, tensors):
        return {"cross_attn_cond": tensors["prompt"]}


class _Sam:
    """The slice of ``StableAudioModel`` the helper touches."""

    device = "cpu"
    model_config = {"model": {"pretransform": {"config": {"encoder": {
        "config": {"chunk_size": 32, "strides": [16]}}}}}}

    def __init__(self):
        self.model = _Model()

    @staticmethod
    def _build_conditioning_dicts(prompt, negative_prompt, duration, batch_size):
        return [{"prompt": prompt, "seconds_total": duration}] * batch_size, None

    def _adapt_sample_size(self, conditioning, sample_size, duration_padding_sec):
        # Upstream's arithmetic: (seconds + pad) * sr, rounded up to the
        # chunk alignment, capped at sample_size.
        max_seconds = max(c["seconds_total"] for c in conditioning)
        target = int((max_seconds + duration_padding_sec) * SR)
        target = ((target + ALIGN - 1) // ALIGN) * ALIGN
        return min(target, sample_size)


@pytest.fixture
def stub_vendor(monkeypatch):
    """``stable_audio_3.data.utils`` with the two pure functions the
    helper imports (mirrors of the vendored implementations)."""
    def compute_effective_seq_len_from_conditioning(
        conditioning, sample_rate, downsampling_ratio=1, device="cpu",
    ):
        lens = [
            math.ceil(int(c["seconds_total"] * sample_rate) / downsampling_ratio)
            for c in conditioning
        ]
        return torch.tensor(lens, dtype=torch.float32)

    def create_padding_mask_from_lengths(valid_lengths, total_seq_len):
        positions = torch.arange(total_seq_len).unsqueeze(0)
        return positions < valid_lengths.unsqueeze(1)

    utils = types.ModuleType("stable_audio_3.data.utils")
    utils.compute_effective_seq_len_from_conditioning = (
        compute_effective_seq_len_from_conditioning
    )
    utils.create_padding_mask_from_lengths = create_padding_mask_from_lengths
    pkg = types.ModuleType("stable_audio_3")
    data = types.ModuleType("stable_audio_3.data")
    pkg.data = data
    data.utils = utils
    for name, mod in (("stable_audio_3", pkg), ("stable_audio_3.data", data),
                      ("stable_audio_3.data.utils", utils)):
        monkeypatch.setitem(sys.modules, name, mod)


def _frames(seconds):
    return math.ceil(int(seconds * SR) / DS)


def test_legacy_label_equals_render_duration(stub_vendor):
    sam = _Sam()
    cond = helpers.prepare_sa3_conditioning(
        sam, prompt="p", duration=30.0, steps=8, duration_padding_sec=6.0,
    )
    assert sam.model.fed[0]["seconds_total"] == 30.0
    assert cond.seconds_total == 30.0
    # Window = 30 + 6 s pad; the mask marks the padded tail invalid only
    # past the 6 s headroom (here: nothing past it, so all valid).
    assert cond.audio_sample_size == sam._adapt_sample_size(
        [{"seconds_total": 30.0}], helpers.SA3_DEFAULT_SAMPLE_SIZE, 6.0,
    )
    assert cond.latent_frames == cond.audio_sample_size // DS
    assert cond.sched_args["effective_seq_len"].item() == _frames(30.0)


def test_song_label_keeps_the_window_and_relabels(stub_vendor):
    sam = _Sam()
    legacy = helpers.prepare_sa3_conditioning(
        sam, prompt="p", duration=30.0, steps=8, duration_padding_sec=0.0,
    )
    song = helpers.prepare_sa3_conditioning(
        sam, prompt="p", duration=30.0, steps=8, duration_padding_sec=0.0,
        song_seconds_total=180.0,
    )
    # The model is told 180 s ...
    assert sam.model.fed[0]["seconds_total"] == 180.0
    assert song.seconds_total == 180.0
    # ... but the render window is still the 30 s loop.
    assert song.audio_sample_size == legacy.audio_sample_size
    assert song.latent_frames == legacy.latent_frames
    # A slice of a longer song is fully valid to attention.
    assert bool(song.cond_bundle["padding_mask"].all())
    # Training-consistent default: the dist-shift schedule follows the label.
    assert song.sched_args["effective_seq_len"].item() == _frames(180.0)
    assert song.sched_args["fallback_seq_len"] == song.latent_frames


def test_song_label_not_longer_than_render_is_legacy(stub_vendor):
    sam = _Sam()
    cond = helpers.prepare_sa3_conditioning(
        sam, prompt="p", duration=60.0, steps=8, song_seconds_total=45.0,
    )
    assert sam.model.fed[0]["seconds_total"] == 60.0
    assert cond.seconds_total == 60.0


def test_schedule_from_window_decouples_the_shift(stub_vendor):
    sam = _Sam()
    cond = helpers.prepare_sa3_conditioning(
        sam, prompt="p", duration=30.0, steps=8, duration_padding_sec=0.0,
        song_seconds_total=180.0, schedule_from_window=True,
    )
    assert sam.model.fed[0]["seconds_total"] == 180.0
    assert cond.sched_args["effective_seq_len"].item() == _frames(30.0)
    assert bool(cond.cond_bundle["padding_mask"].all())


# ---- SA3Context seam (no model load) --------------------------------------


def _bare_context(song_seconds, outro_pad_s):
    c = ctx_mod.SA3Context.__new__(ctx_mod.SA3Context)
    c.model_id = "medium"
    c.sam = _Sam()
    c.downsampling_ratio = DS
    c.sample_rate = SR
    c.song_seconds = song_seconds
    c.schedule_from_window = False
    c.outro_pad_s = outro_pad_s
    c._helpers = helpers
    return c


def test_context_label_and_window_frames():
    c = _bare_context(180.0, 0.0)
    assert c.cond_seconds_total(30.0) == 180.0
    assert c.cond_seconds_total(300.0) == 300.0
    # 60 s, no pad: 2646000 samples -> aligned to 8192 -> 646 latents.
    assert c.window_latent_frames(60.0) == 646
    legacy = _bare_context(None, 6.0)
    assert legacy.cond_seconds_total(30.0) == 30.0
    assert legacy.window_latent_frames(60.0) == 712  # 66 s padded window


def test_clamp_duration_for_trt_uses_the_window_arithmetic(monkeypatch):
    from acestep.engine import sa3_trt

    monkeypatch.setattr(sa3_trt, "max_dit_engine_latents", lambda mid: 646)
    # Eager never clamps.
    assert _bare_context(180.0, 0.0).clamp_duration_for_trt(
        90.0, backend="eager") == 90.0
    # Under the song label a 60 s loop fits the 646-latent engine exactly
    # (the #336 follow-up: no more 54 s clamp).
    assert _bare_context(180.0, 0.0).clamp_duration_for_trt(
        60.0, backend="tensorrt") == 60.0
    # Legacy pad still costs the tail: 60 + 6 s doesn't fit, cap lands on
    # the largest 0.1 s step whose padded window does.
    cap = _bare_context(None, 6.0).clamp_duration_for_trt(
        60.0, backend="tensorrt")
    assert cap < 60.0
    legacy = _bare_context(None, 6.0)
    assert legacy.window_latent_frames(cap) <= 646
    assert legacy.window_latent_frames(round(cap + 0.1, 1)) > 646


def test_tile_loop_wraps_the_source_to_the_window():
    c = _bare_context(180.0, 3.0)
    sr = 48000
    loop = torch.arange(2 * sr, dtype=torch.float32).reshape(2, sr)  # 1 s, 2 ch
    window = 3 * SR  # 3 s of model-rate samples
    out_sr, tiled = c.tile_loop((sr, loop), window)
    assert out_sr == sr
    target = math.ceil(window * sr / SR) + 1
    assert tiled.shape == (2, target)
    # The second lap starts where the first ended: sample N == sample 0.
    assert torch.equal(tiled[:, sr], loop[:, 0])
    assert torch.equal(tiled[:, :sr], loop)
    # numpy in, tensor out; already-covering sources are untouched.
    _, tiled_np = c.tile_loop((sr, loop.numpy()), window)
    assert tiled_np.shape == (2, target)
    long = torch.zeros(2, 10 * sr)
    assert c.tile_loop((sr, long), window)[1] is long
    # Legacy (no label) never tiles: encode_source passes the input through.
    seen = {}
    legacy = _bare_context(None, 6.0)
    legacy._helpers = types.SimpleNamespace(
        encode_sa3_source=lambda sam, ai, n: seen.setdefault("ai", ai))
    legacy.encode_source((sr, loop), window)
    assert seen["ai"][1] is loop
