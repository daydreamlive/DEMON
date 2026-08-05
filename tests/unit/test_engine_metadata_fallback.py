"""Engine freshness when the source ONNX can't be fetched.

The SA3 builder fetches its ONNX on every run — including runs with
nothing to build — because the freshness check hashes the graph. That
made Stability's 2026-08-02 `dit_fp16mixed` -> `dit_fp16` rename fatal
to a bake whose three engines were all already current (bake-warm #87).

The recovery is a hash-less `expected_metadata()` plus
`metadata_matches(..., ignore=("onnx_sha256",))`: keep an engine that is
current in every respect the sidecar can attest to without the graph.
These tests pin that it stays *narrow* — a TRT bump or a config change
must still demand the ONNX, and a hash-less expectation must never
satisfy the ordinary comparison.

Leaf module (loguru only): no GPU, no torch, no tensorrt.
"""

import json
from dataclasses import dataclass

import pytest

from acestep.engine.trt._engine_metadata import (
    expected_metadata,
    metadata_matches,
    metadata_path,
    write_metadata,
)


@dataclass
class _Config:
    min_latents: int = 1
    opt_latents: int = 646
    max_latents: int = 646
    workspace_gb: float = 16.0


def _env(*, trt: str = "10.13.2.6", cc: str = "12.0") -> dict:
    return {
        "packages": {"tensorrt": trt},
        "active_gpu": {"compute_capability": cc, "name": "NVIDIA GeForce RTX 5090"},
    }


def _built_engine(tmp_path, *, onnx_bytes: bytes = b"graph", env=None):
    """An engine + sidecar as a real build would leave them behind."""
    onnx = tmp_path / "dit_fp16mixed.onnx"
    onnx.write_bytes(onnx_bytes)
    engine = tmp_path / "engine.trt"
    engine.write_bytes(b"plan")
    env = env or _env()
    expected = expected_metadata(
        component="sa3_m_dit", onnx_path=onnx, config=_Config(), env=env,
    )
    write_metadata(engine_path=engine, expected=expected, env=env)
    return engine, onnx


def test_hashless_expectation_omits_the_onnx_keys():
    meta = expected_metadata(
        component="sa3_m_dit", onnx_path=None, config=_Config(), env=_env(),
    )
    assert "onnx_sha256" not in meta
    assert "onnx_path" not in meta
    assert meta["component"] == "sa3_m_dit"
    assert meta["tensorrt_version"] == "10.13.2.6"


def test_current_engine_survives_a_vanished_onnx(tmp_path):
    engine, onnx = _built_engine(tmp_path)
    onnx.unlink()  # upstream renamed it out from under us

    hashless = expected_metadata(
        component="sa3_m_dit", onnx_path=None, config=_Config(), env=_env(),
    )
    matches, reason = metadata_matches(
        engine, hashless, ignore=("onnx_sha256",),
    )
    assert matches, reason


@pytest.mark.parametrize(
    "kwargs, config, key",
    [
        ({"trt": "10.14.0.1"}, _Config(), "tensorrt_version"),
        ({"cc": "9.0"}, _Config(), "gpu_compute_capability"),
        ({}, _Config(max_latents=324), "config"),
    ],
)
def test_stale_engine_still_demands_the_onnx(tmp_path, kwargs, config, key):
    """Ignoring the graph hash must not wave through a stale engine."""
    engine, _ = _built_engine(tmp_path)

    hashless = expected_metadata(
        component="sa3_m_dit", onnx_path=None, config=config, env=_env(**kwargs),
    )
    matches, reason = metadata_matches(
        engine, hashless, ignore=("onnx_sha256",),
    )
    assert not matches
    assert key in reason


def test_hashless_expectation_fails_the_ordinary_comparison(tmp_path):
    """Forgetting `ignore` must not silently pass — it must not match."""
    engine, _ = _built_engine(tmp_path)

    hashless = expected_metadata(
        component="sa3_m_dit", onnx_path=None, config=_Config(), env=_env(),
    )
    matches, reason = metadata_matches(engine, hashless)
    assert not matches
    assert "onnx_sha256" in reason


def test_a_re_export_still_invalidates_the_engine(tmp_path):
    """The hash check itself is unchanged: new graph bytes -> rebuild."""
    engine, onnx = _built_engine(tmp_path, onnx_bytes=b"graph")
    onnx.write_bytes(b"graph v2")

    expected = expected_metadata(
        component="sa3_m_dit", onnx_path=onnx, config=_Config(), env=_env(),
    )
    matches, reason = metadata_matches(engine, expected)
    assert not matches
    assert "onnx_sha256" in reason


def test_sidecar_without_a_hash_is_never_written(tmp_path):
    """A hash-less sidecar could never be invalidated by a re-export."""
    engine = tmp_path / "engine.trt"
    engine.write_bytes(b"plan")
    hashless = expected_metadata(
        component="sa3_m_dit", onnx_path=None, config=_Config(), env=_env(),
    )

    with pytest.raises(ValueError, match="onnx_sha256"):
        write_metadata(engine_path=engine, expected=hashless, env=_env())
    assert not metadata_path(engine).exists()


def test_written_sidecar_records_the_hash_and_build_env(tmp_path):
    engine, onnx = _built_engine(tmp_path)
    payload = json.loads(metadata_path(engine).read_text(encoding="utf-8"))

    assert payload["onnx_sha256"]
    assert payload["onnx_path"].endswith("dit_fp16mixed.onnx")
    assert payload["environment"]["packages"]["tensorrt"] == "10.13.2.6"
    assert payload["built_at"]
