"""sa3-small TensorRT engines: builder naming + runtime discovery.

CPU-only: engine files are empty placeholders in a temp models dir; the
discovery functions only look at directory names and file presence.
"""

from __future__ import annotations

import argparse

import pytest
import torch

from acestep.engine import sa3_trt
from acestep.engine.trt import sa3_build


def _touch_engine(base, name: str) -> None:
    d = base / name
    d.mkdir(parents=True)
    (d / f"{name}.trt").write_bytes(b"")


@pytest.fixture
def engines_dir(tmp_path, monkeypatch):
    base = tmp_path / "trt_engines"
    base.mkdir()
    monkeypatch.setattr(sa3_trt, "trt_engines_dir", lambda: base)
    return base


def test_small_has_its_own_engine_prefix():
    assert sa3_trt.DIT_ENGINE_PREFIX["small-music"] == "sa3_sm_dit"
    assert sa3_trt.DIT_ENGINE_PREFIX["medium"] == "sa3_m_dit"


def test_small_build_config_names_and_onnx():
    cfg = sa3_build.SA3SmallDiTBuildConfig(1, 646, 646)
    assert cfg.engine_name() == "sa3_sm_dit_l1_646_646"
    assert cfg.onnx_files == ["onnx/sa3-sm-music/dit_fp16.onnx"]
    # The medium config's identity (and so its engines' metadata) is untouched.
    med = sa3_build.SA3DiTBuildConfig(1, 646, 646)
    assert med.engine_name() == "sa3_m_dit_l1_646_646"
    assert med.onnx_files == list(sa3_build.DIT_ONNX_FILES)


def test_small_matrix_is_dit_only_and_covers_the_120s_window():
    args = argparse.Namespace(
        model="small-music", duration=None, same_l_only=False,
        dit_only=True, fp8=False, refit=False,
    )
    names = [name for _, name in sa3_build._matrix_jobs(args)]
    assert names == [
        "sa3_sm_dit_l1_324_324",
        "sa3_sm_dit_l1_646_646",
        "sa3_sm_dit_l1_1292_1292",
    ]


def test_medium_matrix_unchanged():
    args = argparse.Namespace(
        model="medium", duration=None, same_l_only=False,
        dit_only=False, fp8=False, refit=False,
    )
    names = [name for _, name in sa3_build._matrix_jobs(args)]
    assert names[:2] == ["sa3_m_dit_l1_324_324", "sa3_m_dit_l1_646_646"]
    assert names[2].startswith("same_l_decode_window_")


def test_discovery_keeps_models_apart(engines_dir):
    _touch_engine(engines_dir, "sa3_m_dit_l1_646_646")
    _touch_engine(engines_dir, "sa3_sm_dit_l1_324_324")
    _touch_engine(engines_dir, "sa3_sm_dit_l1_646_646")
    _touch_engine(engines_dir, "sa3_sm_dit_l1_1292_1292")

    # Smallest covering small engine wins.
    assert sa3_trt.find_dit_engine("small-music", 292).parent.name == "sa3_sm_dit_l1_324_324"
    assert sa3_trt.find_dit_engine("small-music", 614).parent.name == "sa3_sm_dit_l1_646_646"
    assert sa3_trt.find_dit_engine("small-music", 1002).parent.name == "sa3_sm_dit_l1_1292_1292"
    assert sa3_trt.find_dit_engine("small-music", 1293) is None
    # Medium never picks up a small engine (and vice versa).
    assert sa3_trt.find_dit_engine("medium", 614).parent.name == "sa3_m_dit_l1_646_646"
    assert sa3_trt.find_dit_engine("medium", 700) is None
    # The duration clamp sees the full small range, medium's own.
    assert sa3_trt.max_dit_engine_latents("small-music") == 1292
    assert sa3_trt.max_dit_engine_latents("medium") == 646


def test_no_small_engines_means_eager(engines_dir):
    _touch_engine(engines_dir, "sa3_m_dit_l1_646_646")
    assert sa3_trt.find_dit_engine("small-music", 614) is None
    assert sa3_trt.max_dit_engine_latents("small-music") is None


def test_preflight_hints_the_small_build_command_when_engines_are_missing(engines_dir):
    from acestep.streaming import preflight as pf

    req = pf.PreflightRequest("small-music")
    hint = pf._sa3_engine_hint(req)
    assert hint is not None and "--model small-music --all" in hint
    # eager boot: no hint; engines present: no hint
    assert pf._sa3_engine_hint(pf.PreflightRequest(
        "small-music", decoder_accel="eager", vae_accel="eager")) is None
    _touch_engine(engines_dir, "sa3_sm_dit_l1_646_646")
    # DiT present, SAME-S decoder still missing: the hint names the decoder.
    hint = pf._sa3_engine_hint(req)
    assert hint is not None and "SAME-S decoder" in hint and "DiT" not in hint
    assert pf._sa3_engine_hint(pf.PreflightRequest("small-music", vae_accel="eager")) is None
    _touch_engine(engines_dir, _same_s_name())
    assert pf._sa3_engine_hint(req) is None


def _same_s_name(lo=32, opt=646, hi=1292) -> str:
    return sa3_build.SameSDecodeBuildConfig(lo, opt, hi).engine_name()


def test_same_s_decode_engine_name_is_versioned():
    name = _same_s_name()
    tag = sa3_trt.same_s_decode_build_tag()
    w = sa3_trt.SAME_S_UPSTREAM_DECODER_SHA256[:12]
    assert name == f"same_s_decode_{tag}_w{w}_t32_646_1292"
    assert tag.startswith("fp16mixed_") and "tail32" in tag
    assert sa3_build.SameSDecodeBuildConfig(32, 646, 1292).onnx_files == [
        "onnx/same-s/dec_bf16.onnx"
    ]


def test_small_matrix_includes_same_s_decoder():
    args = argparse.Namespace(
        model="small-music", duration=None, same_l_only=False,
        dit_only=False, fp8=False, refit=False,
    )
    names = [name for _, name in sa3_build._matrix_jobs(args)]
    assert names[-1] == _same_s_name()
    assert not any(n.startswith("same_l_") for n in names)


def test_same_s_discovery_keys_on_recipe_and_codec_weights(engines_dir):
    sha = sa3_trt.SAME_S_UPSTREAM_DECODER_SHA256
    assert sa3_trt.find_same_s_decode_engine(sha) is None
    w = sha[:12]
    _touch_engine(engines_dir, f"same_s_decode_fp16mixed_attention_tail32_v000000000000_w{w}_t32_646_1292")
    assert sa3_trt.find_same_s_decode_engine(sha) is None  # stale recipe tag
    _touch_engine(engines_dir, _same_s_name())
    path, lo, hi = sa3_trt.find_same_s_decode_engine(sha)
    assert path.parent.name == _same_s_name() and (lo, hi) == (32, 1292)
    # A different codec (e.g. SAME-L in medium-base) or an unknown one: eager.
    assert sa3_trt.find_same_s_decode_engine("985a6741a3bd" + "0" * 52) is None
    assert sa3_trt.find_same_s_decode_engine(None) is None


def test_codec_weights_hash_reads_only_decoder_tensors(tmp_path):
    from safetensors.torch import save_file

    def ckpt(name, dec_val, other_val):
        d = tmp_path / name
        d.mkdir()
        save_file({
            "pretransform.model.decoder.layers.0.weight": torch.full((2, 2), dec_val),
            "pretransform.model.bottleneck.running_std": torch.ones(1),
            "pretransform.model.encoder.layers.0.weight": torch.full((2,), other_val),
            "model.model.x": torch.full((3,), other_val),
        }, str(d / "model.safetensors"))
        return d

    a = sa3_trt.same_decoder_weights_sha256(ckpt("a", 1.0, 1.0))
    b = sa3_trt.same_decoder_weights_sha256(ckpt("b", 1.0, 2.0))  # same codec, other DiT/encoder
    c = sa3_trt.same_decoder_weights_sha256(ckpt("c", 3.0, 1.0))  # different decoder
    assert a == b and a != c and len(a) == 64
    assert sa3_trt.same_decoder_weights_sha256(tmp_path / "missing") is None


class _FakeTRT:
    def __init__(self):
        self.calls = []

    def decode(self, latent):
        self.calls.append(tuple(latent.shape))
        return torch.zeros(2, latent.shape[-1] * 4096)


def _codec_with_fake_trt(monkeypatch):
    from acestep.engine import sa3_context

    codec = sa3_context.SA3SAMECodec.__new__(sa3_context.SA3SAMECodec)
    codec._scale = 1.0
    codec._trt = _FakeTRT()
    codec._min_t, codec._max_t = 32, 1292
    eager_calls = []

    class _Helpers:
        @staticmethod
        def sa3_decode_rng(seed, device=None):
            import contextlib
            return contextlib.nullcontext()

        @staticmethod
        def decode_sa3_latent(sam, latent):
            eager_calls.append(tuple(latent.shape))
            return torch.ones(1, 2, latent.shape[-1] * 4096)

    codec._helpers = _Helpers()
    codec._context = type("Ctx", (), {"sam": None})()
    return codec, eager_calls


def test_codec_routes_even_in_profile_latents_to_trt(monkeypatch):
    codec, eager_calls = _codec_with_fake_trt(monkeypatch)
    out = codec.decode_full(torch.zeros(1, 256, 614))
    assert codec._trt.calls == [(1, 256, 614)] and not eager_calls
    assert out.shape == (2, 614 * 4096)
    # Odd length (different eager chunk phase) and out-of-profile -> eager.
    codec.decode_full(torch.zeros(1, 256, 581))
    codec.decode_full(torch.zeros(1, 256, 1294))
    codec.decode_full(torch.zeros(1, 256, 30))
    assert eager_calls == [(1, 256, 581), (1, 256, 1294), (1, 256, 30)]
    assert len(codec._trt.calls) == 1


def test_make_codec_passes_the_backend_to_full_decode_codecs(monkeypatch):
    from acestep.engine import sa3_context

    seen = []

    class _Codec:
        def __init__(self, ctx, *, use_trt=False):
            seen.append(use_trt)

    monkeypatch.setattr(sa3_context, "SA3SAMECodec", _Codec)
    ctx = sa3_context.SA3Context.__new__(sa3_context.SA3Context)
    ctx.model_id = "small-music"
    ctx.make_codec(backend="tensorrt")
    ctx.make_codec(backend="eager")
    assert seen == [True, False]
