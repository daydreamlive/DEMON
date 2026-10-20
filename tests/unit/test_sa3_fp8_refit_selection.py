"""find_dit_engine: fp8 refit engines for LoRA sessions behind the
DEMON_SA3_FP8_REFIT opt-in (fake engine trees, CPU)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import acestep.engine.sa3_trt as sa3_trt  # noqa: E402

FP16 = "sa3_m_dit_l1_324_324"
FP8 = "sa3_m_dit_fp8_l1_324_324"
REFIT = "sa3_m_dit_refit_l1_646_646"
FP8_REFIT_S = "sa3_m_dit_fp8_refit_l1_324_324"
FP8_REFIT_L = "sa3_m_dit_fp8_refit_l1_646_646"


def _tree(root: Path, *names: str) -> Path:
    for n in names:
        d = root / n
        d.mkdir(parents=True)
        (d / f"{n}.trt").write_bytes(b"x")
    return root


@pytest.fixture
def engines(tmp_path, monkeypatch):
    monkeypatch.setattr(sa3_trt, "trt_engines_dir", lambda: tmp_path)
    monkeypatch.delenv(sa3_trt.FP8_REFIT_ENV, raising=False)
    return tmp_path


def _pick(*a, **k):
    p = sa3_trt.find_dit_engine(*a, **k)
    return None if p is None else p.parent.name


@pytest.mark.parametrize("val,on", [
    ("1", True), ("true", True), ("ON", True), ("yes", True),
    ("0", False), ("", False), ("false", False), ("2", False),
])
def test_env_parsing(monkeypatch, val, on):
    monkeypatch.setenv(sa3_trt.FP8_REFIT_ENV, val)
    assert sa3_trt.fp8_refit_enabled() is on


def test_default_ignores_fp8_refit_for_lora(engines):
    _tree(engines, FP16, FP8, REFIT, FP8_REFIT_S, FP8_REFIT_L)
    # LoRA: fp16mixed refit engine wins, exactly as on main.
    assert _pick("medium", 300, want_refittable=True) == REFIT
    assert _pick("medium", 600, want_refittable=True) == REFIT
    # Non-LoRA: fp8 still preferred; refit engines never picked.
    assert _pick("medium", 300) == FP8
    assert _pick("medium", 600) is None


def test_default_without_refit_engine_keeps_fp16mixed(engines):
    _tree(engines, FP16, FP8, FP8_REFIT_S)
    assert _pick("medium", 300, want_refittable=True) == FP16


def test_opt_in_prefers_smallest_covering_fp8_refit(engines, monkeypatch):
    _tree(engines, FP16, FP8, REFIT, FP8_REFIT_S, FP8_REFIT_L)
    monkeypatch.setenv(sa3_trt.FP8_REFIT_ENV, "1")
    assert _pick("medium", 300, want_refittable=True) == FP8_REFIT_S
    assert _pick("medium", 600, want_refittable=True) == FP8_REFIT_L
    # Non-LoRA sessions are untouched by the opt-in.
    assert _pick("medium", 300) == FP8


def test_opt_in_falls_back_when_no_fp8_refit_covers(engines, monkeypatch):
    _tree(engines, FP16, REFIT, FP8_REFIT_S)
    monkeypatch.setenv(sa3_trt.FP8_REFIT_ENV, "1")
    assert _pick("medium", 600, want_refittable=True) == REFIT
    assert _pick("medium", 700, want_refittable=True) is None


def test_opt_in_only_for_medium_prefix(engines, monkeypatch):
    _tree(engines, FP8_REFIT_S)
    monkeypatch.setenv(sa3_trt.FP8_REFIT_ENV, "1")
    assert _pick("small-music", 300, want_refittable=True) is None


def test_max_latents_counts_fp8_refit_only_when_opted_in(engines, monkeypatch):
    _tree(engines, FP16, FP8_REFIT_L)
    assert sa3_trt.max_dit_engine_latents("medium") == 324
    monkeypatch.setenv(sa3_trt.FP8_REFIT_ENV, "1")
    assert sa3_trt.max_dit_engine_latents("medium") == 646


def test_selection_reason_logged(engines, monkeypatch):
    from loguru import logger

    _tree(engines, REFIT, FP8_REFIT_L)
    lines: list = []
    hid = logger.add(lambda m: lines.append(str(m)), level="INFO")
    try:
        _pick("medium", 600, want_refittable=True)
        monkeypatch.setenv(sa3_trt.FP8_REFIT_ENV, "1")
        _pick("medium", 600, want_refittable=True)
    finally:
        logger.remove(hid)
    text = "".join(lines)
    assert "sa3_dit_fp8_refit_skipped" in text and "reason=opt_in_off" in text
    assert "reason=refit_covering" in text
    assert f"engine={FP8_REFIT_L}" in text and "reason=fp8_refit_opt_in" in text


def test_fp8_refit_path_classifier():
    p = Path(FP8_REFIT_S) / f"{FP8_REFIT_S}.trt"
    assert sa3_trt.is_fp8_refit_engine_path(p)
    assert sa3_trt.is_refittable_engine_path(p)
    assert not sa3_trt.is_fp8_refit_engine_path(Path(REFIT) / f"{REFIT}.trt")
