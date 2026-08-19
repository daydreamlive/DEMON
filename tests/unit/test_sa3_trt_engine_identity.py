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
