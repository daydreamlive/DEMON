"""SAME-L engine selection: one finder (narrowest covering profile) and the
window codec's one decode entry over fake engines and fake decoders.

CPU-only: engine directories are empty files under a temp models dir; the
decoders record the latent shape they are handed.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from acestep.engine import sa3_context, sa3_stream_helpers, sa3_trt

DS = 4096
SR = 44100


def _engine(root: Path, lo: int, opt: int, hi: int) -> Path:
    name = f"same_l_decode_window_{sa3_trt.same_l_plugin_build_tag()}_t{lo}_{opt}_{hi}"
    path = root / "sa3" / "trt_engines" / name / f"{name}.trt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"engine")
    return path


@pytest.fixture
def models(monkeypatch, tmp_path):
    monkeypatch.setenv("ACESTEP_MODELS_DIR", str(tmp_path))
    monkeypatch.delenv("SA3_SWA_AOT", raising=False)
    return tmp_path


@pytest.fixture(params=["sorted", "reversed"])
def listing_order(request, monkeypatch):
    """Run under both directory listing orders (the old window finder
    returned whichever matching directory the listing gave first)."""
    real = Path.iterdir

    def iterdir(self):
        entries = sorted(real(self), key=lambda p: p.name)
        return iter(entries if request.param == "sorted" else entries[::-1])

    monkeypatch.setattr(Path, "iterdir", iterdir)
    return request.param


# --- finder ---------------------------------------------------------------


def test_narrowest_covering_engine_wins_in_either_listing_order(models, listing_order):
    window = _engine(models, 32, 56, 96)
    full = _engine(models, 32, 1292, 3230)

    assert sa3_trt.find_same_l_window_engine() == (window, 32, 96)
    assert sa3_trt.find_same_l_engine(50, 50) == (window, 32, 96)
    assert sa3_trt.find_same_l_engine(97, 1292) == (full, 32, 3230)
    assert sa3_trt.find_same_l_engine(1292, 1292) == (full, 32, 3230)
    assert sa3_trt.find_same_l_engine(90, 90) == (window, 32, 96)
    assert [e[0] for e in sa3_trt.find_same_l_engines()] == [window, full]


def test_a_range_no_engine_covers_returns_none(models, listing_order):
    _engine(models, 32, 56, 96)
    _engine(models, 32, 1292, 3230)

    assert sa3_trt.find_same_l_engine(16, 50) is None  # under every min
    assert sa3_trt.find_same_l_engine(1000, 4000) is None  # over every max
    assert sa3_trt.find_same_l_engine(3231, 3231) is None


def test_window_finder_does_not_fall_back_to_a_wide_engine(models):
    full = _engine(models, 32, 1292, 3230)
    # a wide engine covers the window range too: it serves it when it is
    # the only engine built
    assert sa3_trt.find_same_l_window_engine() == (full, 32, 3230)
    window = _engine(models, 32, 56, 96)
    assert sa3_trt.find_same_l_window_engine() == (window, 32, 96)


def test_no_engines_dir(models):
    assert sa3_trt.find_same_l_engines() == []
    assert sa3_trt.find_same_l_engine(32, 96) is None


def test_full_class_constant_is_gone():
    assert not hasattr(sa3_trt, "SAME_L_FULL_MIN_MAX_T")


# --- codec ----------------------------------------------------------------


class _FakeDecoder:
    def __init__(self, path):
        self.engine_path = Path(path)
        self.calls: list[tuple] = []

    def decode(self, latent_1ct):
        self.calls.append(tuple(latent_1ct.shape))
        t = int(latent_1ct.shape[-1])
        # sample i of the slice = i, so a crop is checkable by value
        return torch.arange(t * DS, dtype=torch.float32).repeat(2, 1)


@pytest.fixture
def codec_factory(models, monkeypatch):
    built: dict[str, _FakeDecoder] = {}
    monkeypatch.setattr(sa3_context, "_SAME_L_DECODERS", {})
    monkeypatch.setattr(
        sa3_trt, "SameLWindowTRTDecoder",
        lambda path: built.setdefault(str(path), _FakeDecoder(path)),
    )
    eager: list[tuple] = []

    def decode_window_eager(sam, latent_bct, *, target_start_sample, target_num_samples, **kw):
        eager.append(("window", tuple(latent_bct.shape), target_start_sample, target_num_samples))
        return SimpleNamespace(audio_ct=torch.zeros(2, target_num_samples))

    def decode_full_eager(sam, latent_bct):
        eager.append(("full", tuple(latent_bct.shape)))
        return torch.zeros(latent_bct.shape[0], 2, latent_bct.shape[-1] * DS)

    helpers = SimpleNamespace(
        resolve_sa3_decode_window=sa3_stream_helpers.resolve_sa3_decode_window,
        sa3_decode_rng=sa3_stream_helpers.sa3_decode_rng,
        decode_sa3_latent_window=decode_window_eager,
        decode_sa3_latent=decode_full_eager,
    )
    ctx = SimpleNamespace(
        _helpers=helpers,
        sam=SimpleNamespace(model=SimpleNamespace(pretransform=SimpleNamespace(scale=1.0))),
        sample_rate=SR,
        downsampling_ratio=DS,
    )

    def make(use_trt=True):
        return sa3_context.SA3SAMEWindowCodec(ctx, use_trt=use_trt)

    return SimpleNamespace(make=make, built=built, eager=eager)


def _lat(t: int) -> torch.Tensor:
    return torch.zeros(1, 256, t)


def test_codec_routes_windows_to_the_window_engine_and_songs_to_the_wide_one(
    models, codec_factory, listing_order,
):
    window = _engine(models, 32, 56, 96)
    full = _engine(models, 32, 1292, 3230)
    codec = codec_factory.make()
    # the window engine is loaded at construction, the wide one is not
    assert list(codec_factory.built) == [str(window)]

    out = codec.decode_window(_lat(1292), 600 * DS, SR)
    assert out.shape == (2, SR)
    assert codec_factory.built[str(window)].calls[-1][-1] <= 96

    song = codec.decode_full(_lat(1292))
    assert song.shape == (2, 1292 * DS)
    assert codec_factory.built[str(full)].calls == [(1, 256, 1292)]
    assert not codec_factory.eager


def test_decode_full_is_decode_window_over_the_whole_song(models, codec_factory):
    _engine(models, 32, 1292, 3230)
    codec = codec_factory.make()
    lat = _lat(400)
    a = codec.decode_full(lat)
    b = codec.decode_window(lat, 0, 400 * DS)
    assert torch.equal(a, b)
    # whole slice, no crop: sample i is i
    assert torch.equal(a[0, :5], torch.arange(5, dtype=torch.float32))


def test_decode_full_batch_loops_the_engine(models, codec_factory):
    full = _engine(models, 32, 1292, 3230)
    codec = codec_factory.make()
    out = codec.decode_full_batch(torch.zeros(3, 256, 500))
    assert out.shape == (3, 2, 500 * DS)
    assert codec_factory.built[str(full)].calls == [(1, 256, 500)] * 3


def test_slice_no_engine_covers_falls_to_eager(models, codec_factory):
    _engine(models, 32, 56, 96)
    codec = codec_factory.make()
    # a whole 1292-frame song: wider than the only engine
    codec.decode_window(_lat(1292), 0, 1292 * DS)
    assert codec_factory.eager[-1][0] == "window"
    codec.decode_full(_lat(1292), decode_seed=7)
    assert codec_factory.eager[-1] == ("full", (1, 256, 1292))
    # a latent shorter than the engine minimum
    codec.decode_window(_lat(20), 0, 20 * DS)
    assert codec_factory.eager[-1][:2] == ("window", (1, 256, 20))


def test_edge_window_grows_to_the_engine_minimum(models, codec_factory):
    window = _engine(models, 40, 56, 96)
    codec = codec_factory.make()
    # first second of a 1292-frame song: the resolved slice is short of 40
    out = codec.decode_window(_lat(1292), 0, SR)
    assert codec_factory.built[str(window)].calls == [(1, 256, 40)]
    assert torch.equal(out[0, :3], torch.arange(3, dtype=torch.float32))


def test_decoders_are_shared_process_wide(models, codec_factory):
    _engine(models, 32, 56, 96)
    full = _engine(models, 32, 1292, 3230)
    a, b = codec_factory.make(), codec_factory.make()
    a.decode_full(_lat(1292))
    b.decode_full(_lat(1292))
    assert len(codec_factory.built) == 2
    assert codec_factory.built[str(full)].calls == [(1, 256, 1292)] * 2
    assert set(sa3_context._SAME_L_DECODERS) == set(codec_factory.built)


def test_eager_codec_never_loads_an_engine(models, codec_factory):
    _engine(models, 32, 56, 96)
    codec = codec_factory.make(use_trt=False)
    assert not codec.uses_trt
    codec.decode_window(_lat(200), 0, SR)
    assert not codec_factory.built
    assert codec_factory.eager[-1][0] == "window"
