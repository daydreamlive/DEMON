"""SAME-L engine selection: one finder (narrowest covering profile).

CPU-only: engine directories are empty files under a temp models dir.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from acestep.engine import sa3_trt


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
    assert sa3_trt.find_same_l_full_engine(1292) == (full, 32, 3230)
    assert sa3_trt.find_same_l_full_engine(90) == (window, 32, 96)
    assert [e[0] for e in sa3_trt.find_same_l_engines()] == [window, full]


def test_a_range_no_engine_covers_returns_none(models, listing_order):
    _engine(models, 32, 56, 96)
    _engine(models, 32, 1292, 3230)

    assert sa3_trt.find_same_l_engine(16, 50) is None  # under every min
    assert sa3_trt.find_same_l_engine(1000, 4000) is None  # over every max
    assert sa3_trt.find_same_l_full_engine(3231) is None


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
