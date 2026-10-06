"""sa3-small TensorRT engines: builder naming + runtime discovery.

CPU-only: engine files are empty placeholders in a temp models dir; the
discovery functions only look at directory names and file presence.
"""

from __future__ import annotations

import argparse

import pytest

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
    assert pf._sa3_engine_hint(pf.PreflightRequest("small-music", decoder_accel="eager")) is None
    _touch_engine(engines_dir, "sa3_sm_dit_l1_646_646")
    assert pf._sa3_engine_hint(req) is None
