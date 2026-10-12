"""CPU tests for sa3_build.build_engine_from_onnx: the real
_build_strongly_typed_engine runs against a fake ``tensorrt`` module (no
GPU); sidecar metadata, skip/rebuild, timing cache and profile checks."""

from __future__ import annotations

import json
import sys
import types

import pytest

pytest.importorskip("onnx")
torch = pytest.importorskip("torch")

from acestep.engine.trt import sa3_build as B  # noqa: E402
from acestep.engine.trt._engine_metadata import metadata_path  # noqa: E402
from test_sa3_export import export_tiny  # noqa: E402

ENV = {"packages": {"tensorrt": "10.16.1.11"},
       "active_gpu": {"compute_capability": "12.0", "name": "fake"}}
PROFILES = {"x": ((1, 32, 140), (1, 32, 140), (1, 32, 140)), "t": ((1,), (1,), (1,)),
            "cond": ((1, 6, 128),) * 3}


class _Rec:
    def __init__(self):
        self.calls = []


def make_fake_trt(rec: _Rec, *, fail_parse=False):
    trt = types.ModuleType("tensorrt")

    class Logger:
        WARNING = 2

        def __init__(self, level):
            pass

    class NetworkDefinitionCreationFlag:
        STRONGLY_TYPED = 5
        PREFER_AOT_PYTHON_PLUGINS = 6
        PREFER_JIT_PYTHON_PLUGINS = 7

    class MemoryPoolType:
        WORKSPACE = 0

    class BuilderFlag:
        REFIT = 1

    class Cache:
        def __init__(self, blob):
            self.blob = blob

        def serialize(self):
            return b"cache:" + bytes(self.blob)

    class Config:
        def set_memory_pool_limit(self, pool, size):
            rec.calls.append(("workspace", size))

        def set_flag(self, flag):
            rec.calls.append(("flag", flag))

        def add_optimization_profile(self, profile):
            rec.calls.append(("profile", dict(profile.shapes)))
            return 0

        def create_timing_cache(self, blob):
            rec.calls.append(("cache_in", bytes(blob)))
            return Cache(blob)

        def set_timing_cache(self, cache, ignore_mismatch):
            self.cache = cache

        def get_timing_cache(self):
            return self.cache

    class Profile:
        def __init__(self):
            self.shapes = {}

        def set_shape(self, name, lo, opt, hi):
            self.shapes[name] = (tuple(lo), tuple(opt), tuple(hi))

    class Builder:
        def __init__(self, logger):
            pass

        def create_network(self, flags):
            rec.calls.append(("network_flags", flags))
            return object()

        def create_builder_config(self):
            return Config()

        def create_optimization_profile(self):
            return Profile()

        def build_serialized_network(self, network, config):
            rec.calls.append(("build",))
            return b"ENGINE"

    class OnnxParser:
        num_errors = 1

        def __init__(self, network, logger):
            pass

        def parse_from_file(self, path):
            rec.calls.append(("parse", path))
            return not fail_parse

        def get_error(self, i):
            return "fake parse error"

    for obj in (Logger, NetworkDefinitionCreationFlag, MemoryPoolType, BuilderFlag, Builder, OnnxParser):
        setattr(trt, obj.__name__, obj)
    return trt


@pytest.fixture(scope="module")
def tiny_onnx(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("build")
    _, rep = export_tiny(tmp)
    return tmp, rep.onnx_path


@pytest.fixture
def fake_trt(monkeypatch):
    rec = _Rec()
    monkeypatch.setitem(sys.modules, "tensorrt", make_fake_trt(rec))
    return rec


def _build(tmp, onnx_path, **kw):
    args = dict(profiles=PROFILES, recipe="demon_fp8_graft", extra_metadata={"arm": "ours_fp8"},
                timing_cache=None, env=ENV)
    args.update(kw)
    return B.build_engine_from_onnx(onnx_path, tmp / "eng" / "sa3_m_dit_ours_fp8_l140_140_140", **args)


def test_build_writes_engine_and_sidecar(tiny_onnx, fake_trt, tmp_path):
    tmp, onnx_path = tiny_onnx
    scales = tmp_path / "s.json"
    scales.write_text(json.dumps({"a": 1.0}))
    path = _build(tmp_path, onnx_path, act_scales_path=scales)
    assert path.name == "sa3_m_dit_ours_fp8_l140_140_140.trt" and path.read_bytes() == b"ENGINE"
    meta = json.loads(metadata_path(path).read_text())
    cfg = meta["config"]
    assert cfg["recipe"] == "demon_fp8_graft" and cfg["strongly_typed"] is True
    assert cfg["profiles"]["x"] == [[1, 32, 140]] * 3
    assert cfg["extra"] == {"arm": "ours_fp8"}
    assert len(cfg["onnx_full_sha256"]) == 64 and len(cfg["act_scales_sha256"]) == 64
    assert meta["tensorrt_version"] == "10.16.1.11" and meta["component"] == "sa3_onnx_engine"
    calls = dict((c[0], c[1:]) for c in fake_trt.calls)
    assert calls["network_flags"] == (1 << 5,)                     # STRONGLY_TYPED only
    assert calls["workspace"] == (16 << 30,)
    assert calls["profile"][0]["x"] == ((1, 32, 140),) * 3
    assert "flag" not in calls                                    # no precision flags


def test_skip_then_rebuild_on_change(tiny_onnx, fake_trt, tmp_path):
    _, onnx_path = tiny_onnx
    _build(tmp_path, onnx_path)
    n = sum(c[0] == "build" for c in fake_trt.calls)
    _build(tmp_path, onnx_path)
    assert sum(c[0] == "build" for c in fake_trt.calls) == n          # skipped
    _build(tmp_path, onnx_path, recipe="demon_fp16mixed_export")
    assert sum(c[0] == "build" for c in fake_trt.calls) == n + 1      # recipe change rebuilds
    _build(tmp_path, onnx_path, recipe="demon_fp16mixed_export", force_rebuild=True)
    assert sum(c[0] == "build" for c in fake_trt.calls) == n + 2


def test_timing_cache_roundtrip_and_weak_typing(tiny_onnx, fake_trt, tmp_path):
    _, onnx_path = tiny_onnx
    cache = tmp_path / "tc" / "sa3.cache"
    _build(tmp_path, onnx_path, timing_cache=cache, strongly_typed=False)
    assert cache.read_bytes() == b"cache:"
    assert ("network_flags", 0) in fake_trt.calls
    _build(tmp_path, onnx_path, timing_cache=cache, force_rebuild=True)
    assert ("cache_in", b"cache:") in fake_trt.calls


def test_profile_validation_and_parse_failure(tiny_onnx, tmp_path, monkeypatch):
    _, onnx_path = tiny_onnx
    with pytest.raises(ValueError):
        _build(tmp_path, onnx_path, profiles={"x": ((1, 32, 200), (1, 32, 140), (1, 32, 140))})
    with pytest.raises(FileNotFoundError):
        _build(tmp_path, tmp_path / "missing.onnx")
    monkeypatch.setitem(sys.modules, "tensorrt", make_fake_trt(_Rec(), fail_parse=True))
    with pytest.raises(RuntimeError, match="parse failed"):
        _build(tmp_path, onnx_path)


def test_fp8_cli_build(tiny_onnx, fake_trt, tmp_path, monkeypatch):
    from acestep.engine.trt import sa3_fp8 as Q

    _, onnx_path = tiny_onnx
    monkeypatch.setattr(B, "_preflight", lambda device: ENV)
    rc = Q.main(["build", "--onnx", str(onnx_path), "--engine-dir", str(tmp_path / "e" / "eng_l140"),
                 "--profile", "x=1x32x140,t=1,cond=1x6x128", "--recipe", "demon_fp16mixed_export"])
    assert rc == 0 and (tmp_path / "e" / "eng_l140" / "eng_l140.trt").is_file()


def test_existing_dit_build_path_unchanged(fake_trt, tmp_path, monkeypatch):
    """The --fp8 / --dit path still builds STRONGLY_TYPED with its six-input
    profile through the shared helper (no new kwargs needed)."""
    onnx_file = tmp_path / "dit_fp8.onnx"
    onnx_file.write_bytes(b"proto")
    cfg = B.SA3DiTFp8BuildConfig(1, 140, 140)
    label, path, _, status = B._build_dit_engine(output_dir=str(tmp_path), config=cfg, env=ENV,
                                                 precision_label="fp8", local_onnx=str(onnx_file))
    assert status == "OK" and ("network_flags", 1 << 5) in fake_trt.calls
    prof = next(c[1] for c in fake_trt.calls if c[0] == "profile")
    assert set(prof) == {"x", "t", "t5_hidden", "t5_mask", "seconds_total", "local_add_cond"}
