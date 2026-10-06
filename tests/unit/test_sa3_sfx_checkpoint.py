"""``--checkpoint sa3-sfx``: the Stable Audio 3 small-sfx checkpoint.

SFX shares small-music's architecture and SAME-S codec, so it is only an
alias, a catalog location, a preflight remedy and its own DiT engine
prefix (a TensorRT engine bakes in its weights)."""

from __future__ import annotations

from pathlib import Path

from acestep.engine import sa3_helpers, sa3_trt
from acestep.engine.trt.sa3_build import (
    DIT_MODELS,
    SA3DiTBuildConfig,
    SA3SfxDiTBuildConfig,
)
from acestep.streaming.families import FAMILIES, resolve_checkpoint


def _engine(root: Path, name: str) -> Path:
    path = root / "sa3" / "trt_engines" / name / f"{name}.trt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"engine")
    return path


def test_sa3_sfx_alias_resolves_to_small_sfx():
    assert resolve_checkpoint("sa3-sfx") == ("sa3", "small-sfx")


def test_sa3_sfx_alias_is_not_a_family_name():
    assert "sa3-sfx" not in FAMILIES


def test_sfx_checkpoint_dir_matches_hf_layout(monkeypatch, tmp_path):
    monkeypatch.setenv("ACESTEP_MODELS_DIR", str(tmp_path))
    assert sa3_helpers.sa3_checkpoint_dir("small-sfx") == (
        tmp_path / "sa3" / "checkpoints" / "stable-audio-3-small-sfx"
    )


def test_missing_sfx_checkpoint_preflight_names_the_gated_download(
    monkeypatch, tmp_path,
):
    monkeypatch.setenv("ACESTEP_MODELS_DIR", str(tmp_path))
    ok, msg = sa3_helpers.sa3_checkpoint_status("small-sfx")
    assert not ok
    assert "huggingface.co/stabilityai/stable-audio-3-small-sfx" in msg
    assert "gated" in msg
    assert "huggingface-cli download stabilityai/stable-audio-3-small-sfx" in msg


def test_sfx_engines_serve_only_sfx(monkeypatch, tmp_path):
    monkeypatch.setenv("ACESTEP_MODELS_DIR", str(tmp_path))
    sfx = _engine(tmp_path, "sa3_sfx_dit_l1_646_646")
    _engine(tmp_path, "sa3_m_dit_l1_646_646")
    assert sa3_trt.find_dit_engine("small-sfx", 600) == sfx
    assert sa3_trt.find_dit_engine("medium", 600).parent.name == "sa3_m_dit_l1_646_646"
    # small-music shares the architecture but not the weights.
    assert sa3_trt.find_dit_engine("small-music", 600) is None
    assert sa3_trt.max_dit_engine_latents("small-sfx") == 646


def test_sfx_smallest_covering_engine_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("ACESTEP_MODELS_DIR", str(tmp_path))
    _engine(tmp_path, "sa3_sfx_dit_l1_646_646")
    short = _engine(tmp_path, "sa3_sfx_dit_l1_173_173")
    assert sa3_trt.find_dit_engine("small-sfx", 150) == short


def test_sfx_build_config_names_and_keeps_medium_identity():
    cfg = SA3SfxDiTBuildConfig(1, 646, 646)
    assert cfg.engine_name() == "sa3_sfx_dit_l1_646_646"
    assert cfg.onnx_files == ["onnx/sa3-sm-sfx/dit_fp16.onnx"]
    # The engine name must be what discovery matches for small-sfx.
    assert cfg.engine_name().startswith(sa3_trt.DIT_ENGINE_PREFIX["small-sfx"] + "_l")
    # Medium's hashed config identity is unchanged by the subclass.
    assert SA3DiTBuildConfig(1, 646, 646).engine_name() == "sa3_m_dit_l1_646_646"
    assert DIT_MODELS["small-sfx"][1] == "sa3_sfx_dit"
