"""A companion of the full-decode kind decodes on the SAME-L window engine
the main model already holds: the codec choice, the engine list, and the
eager full decode a slice outside the engine takes.

CPU-only: engine directories are empty files under a temp models dir; the
decoders record the latent shape they are handed.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from acestep.engine import sa3_context, sa3_stream_helpers, sa3_trt
from acestep.engine.sa3_placement import CardPart

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


# --- make_codec -------------------------------------------------------------


@pytest.fixture
def chosen(monkeypatch):
    """Replace the three codec classes with recorders of which one
    ``make_codec`` built."""
    for name in ("SA3SAMECodec", "SA3SAMEWindowCodec", "SA3CompanionWindowCodec"):
        monkeypatch.setattr(
            sa3_context, name,
            lambda ctx, *, use_trt=True, _n=name: (_n, use_trt),
        )


def _make(model_id="small-sfx", companion=True, backend="tensorrt"):
    ctx = SimpleNamespace(model_id=model_id, companion=companion)
    return sa3_context.SA3Context.make_codec(ctx, backend=backend)


def test_a_companion_with_the_window_engine_built_takes_it(models, chosen):
    _engine(models, 32, 56, 96)
    assert _make()[0] == "SA3CompanionWindowCodec"


@pytest.mark.parametrize("flip, expected", [
    ({"companion": False}, ("SA3SAMECodec", True)),
    ({"backend": "eager"}, ("SA3SAMECodec", False)),
    ({"model_id": "medium"}, ("SA3SAMEWindowCodec", True)),
])
def test_each_condition_flipped_gives_todays_codec(models, chosen, flip, expected):
    _engine(models, 32, 56, 96)
    assert _make(**flip) == expected


def test_no_window_engine_built_gives_todays_codec(models, chosen):
    assert _make() == ("SA3SAMECodec", True)


# --- the codec --------------------------------------------------------------


class _FakeDecoder:
    def __init__(self, path):
        self.engine_path = Path(path)
        self.calls: list[tuple] = []

    def decode(self, latent_1ct):
        self.calls.append(tuple(latent_1ct.shape))
        t = int(latent_1ct.shape[-1])
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
    noise: list[bool] = []

    def decode_window_eager(sam, latent_bct, **kw):
        eager.append(("window", tuple(latent_bct.shape)))
        return SimpleNamespace(audio_ct=torch.zeros(2, kw["target_num_samples"]))

    def decode_full_eager(sam, latent_bct):
        eager.append(("full", tuple(latent_bct.shape)))
        t = int(latent_bct.shape[-1])
        # sample i of the decode = -i, so a crop is checkable by value
        return -torch.arange(t * DS, dtype=torch.float32).repeat(latent_bct.shape[0], 2, 1)

    @contextmanager
    def decode_noise_mode(sam, *, enabled):
        noise.append(enabled)
        yield

    helpers = SimpleNamespace(
        resolve_sa3_decode_window=sa3_stream_helpers.resolve_sa3_decode_window,
        sa3_decode_rng=sa3_stream_helpers.sa3_decode_rng,
        sa3_decode_noise_mode=decode_noise_mode,
        decode_sa3_latent_window=decode_window_eager,
        decode_sa3_latent=decode_full_eager,
    )
    # a stub decoder on the CPU that really parks and loads
    part = CardPart("decoder", torch.nn.Linear(1, 1), "cpu")
    ctx = SimpleNamespace(
        _helpers=helpers,
        sam=SimpleNamespace(model=SimpleNamespace(pretransform=SimpleNamespace(scale=1.0))),
        sample_rate=SR,
        downsampling_ratio=DS,
        torch_decoder=part,
    )

    def make():
        return sa3_context.SA3CompanionWindowCodec(ctx)

    return SimpleNamespace(make=make, built=built, eager=eager, noise=noise, part=part)


def _lat(t: int) -> torch.Tensor:
    return torch.zeros(1, 256, t)


def test_the_engine_list_never_holds_a_wide_engine(models, codec_factory):
    window = _engine(models, 32, 56, 96)
    _engine(models, 32, 1292, 3230)
    codec = codec_factory.make()
    assert codec._engines == [(window, 32, 96)]
    assert list(codec_factory.built) == [str(window)]


def test_a_slice_the_window_engine_holds_goes_to_it_decoder_parked(models, codec_factory):
    window = _engine(models, 32, 56, 96)
    codec = codec_factory.make()
    out = codec.decode_window(_lat(200), 100 * DS, SR)
    assert out.shape == (2, SR)
    assert codec_factory.built[str(window)].calls[-1][-1] <= 96
    assert not codec_factory.eager
    assert not codec._holding
    assert codec_factory.part.parked and codec_factory.part.count == 0


def test_a_latent_under_the_minimum_is_a_full_decode_cropped(models, codec_factory):
    _engine(models, 32, 56, 96)
    codec = codec_factory.make()
    lat = _lat(20)
    a = codec.decode_window(lat, 0, SR)
    b = codec.decode_window(lat, SR, SR)
    c = codec.decode_window(lat, 20 * DS - 100, 300)
    assert codec_factory.eager == [("full", (1, 256, 20))] * 3
    assert codec_factory.noise == [False] * 3
    assert codec._holding and codec_factory.part.count == 1
    assert not codec_factory.part.parked
    assert torch.equal(a[0, :3], -torch.arange(3, dtype=torch.float32))
    assert torch.equal(b[1, :3], -torch.arange(SR, SR + 3, dtype=torch.float32))
    # past the end: zero-padded to num
    assert c.shape == (2, 300)
    assert torch.equal(c[0, :100], -torch.arange(20 * DS - 100, 20 * DS, dtype=torch.float32))
    assert torch.equal(c[:, 100:], torch.zeros(2, 200))
    assert not codec_factory.built[next(iter(codec_factory.built))].calls


def test_decode_full_on_a_long_latent_is_eager_never_a_wide_engine(models, codec_factory):
    window = _engine(models, 32, 56, 96)
    full = _engine(models, 32, 1292, 3230)
    codec = codec_factory.make()
    song = codec.decode_full(_lat(600), decode_seed=7)
    assert song.shape == (2, 600 * DS)
    assert codec_factory.eager == [("full", (1, 256, 600))]
    assert str(full) not in codec_factory.built
    assert not codec_factory.built[str(window)].calls
    # T inside the window engine's profile goes to the engine
    codec.decode_full(_lat(64))
    assert codec_factory.built[str(window)].calls == [(1, 256, 64)]
