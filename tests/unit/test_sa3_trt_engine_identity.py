from pathlib import Path

from acestep.engine import sa3_trt
from acestep.engine.trt.sa3_build import SameLWindowBuildConfig


def _engine(root: Path, name: str) -> Path:
    path = root / "sa3" / "trt_engines" / name / f"{name}.trt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"engine")
    return path


def test_same_l_engine_name_carries_plugin_build_identity(monkeypatch):
    monkeypatch.delenv("SA3_SWA_AOT", raising=False)
    config = SameLWindowBuildConfig(32, 56, 96)

    assert config.plugin_build_tag == sa3_trt.same_l_plugin_build_tag()
    assert config.engine_name() == (
        f"same_l_decode_window_{config.plugin_build_tag}_t32_56_96"
    )


def test_same_l_discovery_ignores_legacy_engine(monkeypatch, tmp_path):
    monkeypatch.setenv("ACESTEP_MODELS_DIR", str(tmp_path))
    monkeypatch.delenv("SA3_SWA_AOT", raising=False)
    _engine(tmp_path, "same_l_decode_window_t32_56_96")

    assert sa3_trt.find_same_l_window_engine() is None


def test_same_l_discovery_selects_current_plugin_engine(monkeypatch, tmp_path):
    monkeypatch.setenv("ACESTEP_MODELS_DIR", str(tmp_path))
    monkeypatch.delenv("SA3_SWA_AOT", raising=False)
    tag = sa3_trt.same_l_plugin_build_tag()
    expected = _engine(tmp_path, f"same_l_decode_window_{tag}_t32_56_96")

    assert sa3_trt.find_same_l_window_engine() == (expected, 32, 96)


def test_same_l_aot_backend_is_part_of_identity(monkeypatch):
    monkeypatch.setenv("SA3_SWA_PLUGIN", "aot")
    monkeypatch.setenv("SA3_SWA_AOT", "mma")
    mma = sa3_trt.same_l_plugin_build_tag()
    monkeypatch.setenv("SA3_SWA_AOT", "ptx")
    ptx = sa3_trt.same_l_plugin_build_tag()

    assert mma != ptx
    assert mma.startswith("aot_mma_v")
    assert ptx.startswith("aot_ptx_v")


def test_same_l_plugin_kind_is_part_of_identity(monkeypatch):
    monkeypatch.setenv("SA3_SWA_PLUGIN", "aot")
    aot = sa3_trt.same_l_plugin_build_tag()
    monkeypatch.setenv("SA3_SWA_PLUGIN", "jit")
    jit = sa3_trt.same_l_plugin_build_tag()

    assert aot != jit
    assert jit.startswith("jit_jit_v")


def test_window_and_full_same_l_engines_coexist(monkeypatch, tmp_path):
    """The ring's window engine (t..96) and the one-shot full-song engine
    (t..3230) live in the same directory; each finder picks its own,
    whatever the directory listing order."""
    monkeypatch.setenv("ACESTEP_MODELS_DIR", str(tmp_path))
    monkeypatch.delenv("SA3_SWA_AOT", raising=False)
    tag = sa3_trt.same_l_plugin_build_tag()
    full = _engine(tmp_path, f"same_l_decode_window_{tag}_t32_1292_3230")
    window = _engine(tmp_path, f"same_l_decode_window_{tag}_t32_56_96")
    _engine(tmp_path, "same_l_decode_window_othertag_t32_1292_9000")      # another plugin build

    assert sa3_trt.find_same_l_window_engine() == (window, 32, 96)
    assert sa3_trt.find_same_l_engine(1292, 1292) == (full, 32, 3230)
    assert [e[0] for e in sa3_trt.find_same_l_engines()] == [window, full]


def test_full_same_l_engine_absent_alone_window(monkeypatch, tmp_path):
    monkeypatch.setenv("ACESTEP_MODELS_DIR", str(tmp_path))
    monkeypatch.delenv("SA3_SWA_AOT", raising=False)
    tag = sa3_trt.same_l_plugin_build_tag()
    _engine(tmp_path, f"same_l_decode_window_{tag}_t32_56_96")
    assert sa3_trt.find_same_l_engine(1292, 1292) is None
    tmp2 = tmp_path / "only_full"
    monkeypatch.setenv("ACESTEP_MODELS_DIR", str(tmp2))
    full = _engine(tmp2, f"same_l_decode_window_{tag}_t32_1292_3230")
    # selection is by coverage: a lone wide engine covers the window range too
    assert sa3_trt.find_same_l_window_engine() == (full, 32, 3230)
    assert sa3_trt.find_same_l_engine(1292, 1292) == (full, 32, 3230)


def test_full_decode_routes_by_frames_one_latent_per_call(monkeypatch):
    """decode_full / decode_full_batch go through decode_window on the
    engine covering the frame count (items one at a time, the latent scaled
    like any window); eager when no engine covers it."""
    import contextlib
    from types import SimpleNamespace

    import torch

    from acestep.engine.sa3_context import SA3SAMEWindowCodec

    calls = {"trt": [], "eager": []}

    class FakeTRT:
        def decode(self, lat):
            calls["trt"].append(tuple(lat.shape))
            return lat[0, :2].repeat_interleave(4, dim=-1) * 10.0     # [2, 4T], out of [-1, 1]

    def eager(_sam, lat):
        calls["eager"].append(tuple(lat.shape))
        return torch.zeros(lat.shape[0], 2, lat.shape[-1] * 4)

    import threading

    from acestep.engine import sa3_context, sa3_stream_helpers

    codec = SA3SAMEWindowCodec.__new__(SA3SAMEWindowCodec)
    codec._context = SimpleNamespace(sam=None, sample_rate=4, downsampling_ratio=4)
    codec._helpers = SimpleNamespace(
        sa3_decode_rng=lambda *_a, **_k: contextlib.nullcontext(), decode_sa3_latent=eager,
        resolve_sa3_decode_window=sa3_stream_helpers.resolve_sa3_decode_window,
    )
    codec._scale = 0.5
    codec._engines = [("x.trt", 32, 3230)]
    monkeypatch.setattr(
        sa3_context, "_SAME_L_DECODERS", {"x.trt": (FakeTRT(), threading.Lock())})

    lat = torch.full((5, 256, 40), 0.1)
    out = codec.decode_full_batch(lat, decode_seed=3)
    assert calls["trt"] == [(1, 256, 40)] * 5 and calls["eager"] == []
    assert out.shape == (5, 2, 160)
    assert torch.allclose(out, torch.full_like(out, 0.5))           # 0.1 x scale 0.5 x 10, clamped
    one = codec.decode_full(lat[:1])
    assert one.shape == (2, 160) and len(calls["trt"]) == 6
    codec.decode_full_batch(torch.zeros(2, 256, 20))                 # under the profile min: eager
    codec.decode_full(torch.zeros(1, 256, 4000))                     # over the max: eager
    assert calls["eager"] == [(2, 256, 20), (1, 256, 4000)] and len(calls["trt"]) == 6
    codec._engines = []
    codec.decode_full_batch(lat)
    assert calls["eager"][-1] == (5, 256, 40)
