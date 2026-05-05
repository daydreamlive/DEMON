"""Smoke test: helpers + clamp + auto-select logic for the windowed
VAE decode wiring. Avoids loading the model so it runs fast and
without GPU contention."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_helpers():
    from acestep.paths import (
        WINDOWED_VAE_DECODE_NAME,
        WINDOWED_DREAMVAE_DECODE_NAME,
        WINDOWED_VAE_PROFILE_FRAMES,
        WINDOWED_VAE_WINDOW_RANGE_S,
        windowed_vae_decode_engine_name,
        windowed_vae_decode_engine_path,
        available_windowed_vae_decode_engine,
        looks_like_dreamvae_engine,
    )

    assert WINDOWED_VAE_DECODE_NAME == "vae_decode_fp16_3to30s"
    assert WINDOWED_DREAMVAE_DECODE_NAME == "dreamvae_decode_fp16_3to30s"
    assert WINDOWED_VAE_PROFILE_FRAMES == (75, 125, 750)
    assert WINDOWED_VAE_WINDOW_RANGE_S == (5.0, 30.0)

    assert windowed_vae_decode_engine_name() == "vae_decode_fp16_3to30s"
    assert windowed_vae_decode_engine_name(dreamvae=True) == "dreamvae_decode_fp16_3to30s"

    std_path = windowed_vae_decode_engine_path()
    dv_path = windowed_vae_decode_engine_path(dreamvae=True)
    assert std_path.name.endswith("vae_decode_fp16_3to30s.engine")
    assert dv_path.name.endswith("dreamvae_decode_fp16_3to30s.engine")

    # Both are built on disk per the previous build steps.
    assert available_windowed_vae_decode_engine() is not None, \
        f"std windowed engine missing at {std_path}"
    assert available_windowed_vae_decode_engine(dreamvae=True) is not None, \
        f"dreamvae windowed engine missing at {dv_path}"

    # Engine-variant detection by basename. trt_engine_path always
    # produces .../<name>/<name>.engine so checking the filename is
    # equivalent to checking the engine variant.
    assert looks_like_dreamvae_engine("/x/y/dreamvae_decode_fp16_60s/dreamvae_decode_fp16_60s.engine")
    assert looks_like_dreamvae_engine("dreamvae_decode_fp16_3to30s.engine")
    assert not looks_like_dreamvae_engine("vae_decode_fp16_60s.engine")
    assert not looks_like_dreamvae_engine("vae_decode_fp16_3to30s.engine")

    print("[ok] helpers")


def test_clamp_window_range():
    """The clamp applied in StreamVAEDecode.execute and Session.__init__."""
    from acestep.paths import WINDOWED_VAE_WINDOW_RANGE_S
    lo, hi = WINDOWED_VAE_WINDOW_RANGE_S

    def clamp(v: float) -> float:
        if v <= 0:
            return v   # disable sentinel
        return max(lo, min(hi, v))

    assert clamp(-1.0) == -1.0     # disable preserved
    assert clamp(0.0) == 0.0       # disable preserved
    assert clamp(0.1) == 5.0       # tiny -> floor
    assert clamp(3.0) == 5.0       # below floor
    assert clamp(5.0) == 5.0
    assert clamp(15.0) == 15.0
    assert clamp(30.0) == 30.0
    assert clamp(60.0) == 30.0     # above ceiling
    print("[ok] clamp_window_range")


def test_session_imports_clean():
    """Importing Session must not fail; the new auto-select branch
    introduces a couple of imports inside the constructor that we want
    to make sure exist."""
    from acestep.engine.session import Session  # noqa: F401
    from acestep.nodes.vae_nodes import StreamVAEDecode  # noqa: F401
    print("[ok] imports")


def test_autoselect_logic_simulation():
    """Simulate what Session.__init__ does with a fake trt_engines dict
    so we don't have to load weights. Mirrors the real branch in
    session.py exactly."""
    from acestep.paths import (
        WINDOWED_VAE_WINDOW_RANGE_S,
        available_windowed_vae_decode_engine,
        looks_like_dreamvae_engine,
    )

    def simulate(trt_engines: dict, vae_backend: str, vae_window: float) -> dict:
        # Clamp.
        if vae_window > 0:
            lo, hi = WINDOWED_VAE_WINDOW_RANGE_S
            vae_window = max(lo, min(hi, vae_window))
        # Auto-select.
        if vae_window > 0 and vae_backend == "tensorrt" and "vae_decode" in trt_engines:
            current = trt_engines["vae_decode"]
            is_dv = looks_like_dreamvae_engine(current)
            windowed = available_windowed_vae_decode_engine(dreamvae=is_dv)
            if windowed is not None and str(windowed) != current:
                trt_engines = dict(trt_engines)
                trt_engines["vae_decode"] = str(windowed)
        return trt_engines, vae_window

    # Case 1: standard 240s + window=15  ->  windowed standard engine.
    out, w = simulate({"vae_decode": "/x/vae_decode_fp16_240s.engine"},
                      "tensorrt", 15.0)
    assert "vae_decode_fp16_3to30s" in out["vae_decode"]
    assert w == 15.0

    # Case 2: dreamvae 240s + window=8  ->  windowed dreamvae engine.
    out, w = simulate({"vae_decode": "/x/dreamvae_decode_fp16_240s.engine"},
                      "tensorrt", 8.0)
    assert "dreamvae_decode_fp16_3to30s" in out["vae_decode"]
    assert w == 8.0

    # Case 3: window=0 -> no swap.
    src = {"vae_decode": "/x/vae_decode_fp16_240s.engine"}
    out, w = simulate(src, "tensorrt", 0.0)
    assert out["vae_decode"] == src["vae_decode"]
    assert w == 0.0

    # Case 4: window=-1 -> no swap (disable sentinel preserved).
    out, w = simulate({"vae_decode": "/x/vae_decode_fp16_240s.engine"},
                      "tensorrt", -1.0)
    assert "fp16_240s" in out["vae_decode"]
    assert w == -1.0

    # Case 5: vae_backend != tensorrt -> no swap.
    src = {"vae_decode": "/x/vae_decode_fp16_240s.engine"}
    out, w = simulate(src, "eager", 10.0)
    assert out["vae_decode"] == src["vae_decode"]

    # Case 6: window above max -> clamp to 30, swap applied.
    out, w = simulate({"vae_decode": "/x/vae_decode_fp16_240s.engine"},
                      "tensorrt", 999.0)
    assert w == 30.0
    assert "3to30s" in out["vae_decode"]

    print("[ok] autoselect_logic_simulation")


def test_find_best_vae_engine_accepts_dreamvae():
    """Pre-existing bug: lookup must match both vae_decode_* and
    dreamvae_decode_* for the same 'vae_decode' component, otherwise
    sessions with fast_vae or windowed dreamvae crash at runtime when
    they fall back to PyTorch and the VAE weights were skipped."""
    from acestep.nodes import vae_nodes

    saved = dict(vae_nodes._trt_vae_cache)
    try:
        vae_nodes._trt_vae_cache.clear()
        vae_nodes._trt_vae_cache["/x/dreamvae_decode_fp16_3to30s/dreamvae_decode_fp16_3to30s.engine"] = object()
        vae_nodes._trt_vae_cache["/x/vae_encode_fp16_60s/vae_encode_fp16_60s.engine"] = object()

        decode_hit = vae_nodes._find_best_vae_engine("vae_decode")
        assert decode_hit is not None and "dreamvae_decode" in decode_hit, \
            f"vae_decode lookup failed to match dreamvae path: {decode_hit!r}"

        encode_hit = vae_nodes._find_best_vae_engine("vae_encode")
        assert encode_hit is not None and "vae_encode" in encode_hit
        # Must not return a dreamvae_decode entry under encode lookup.
        assert "dreamvae_decode" not in encode_hit

        # Also verify the standard (non-dreamvae) decode path still matches.
        vae_nodes._trt_vae_cache.clear()
        vae_nodes._trt_vae_cache["/x/vae_decode_fp16_3to30s/vae_decode_fp16_3to30s.engine"] = object()
        std_hit = vae_nodes._find_best_vae_engine("vae_decode")
        assert std_hit is not None and "vae_decode_fp16_3to30s" in std_hit
    finally:
        vae_nodes._trt_vae_cache.clear()
        vae_nodes._trt_vae_cache.update(saved)
    print("[ok] find_best_vae_engine_accepts_dreamvae")


def test_streamvae_param_spec():
    """Param spec in StreamVAEDecode reflects the new bounds."""
    from acestep.nodes.vae_nodes import StreamVAEDecode

    defn = StreamVAEDecode.get_definition()
    params = {p.name: p for p in defn.params}
    win = params["vae_window"]
    assert win.min == 0.0
    assert win.max == 30.0
    assert win.default == 5.0
    print("[ok] streamvae_param_spec")


if __name__ == "__main__":
    test_helpers()
    test_clamp_window_range()
    test_session_imports_clean()
    test_autoselect_logic_simulation()
    test_find_best_vae_engine_accepts_dreamvae()
    test_streamvae_param_spec()
    print("\nAll smoke tests passed.")
