"""Stale (pre-spectral-steering) decoder ONNX detection and recovery.

Spectral steering added a 'steering' graph input to the decoder ONNX;
export.py's engine build hard-fails on files exported before it. The
build's ONNX resolver therefore treats such files as stale: it probes
the hub's graph proto first (cheap, etag-cached) and either replaces
the local file from a fresh hub artifact or exits with the
--export-locally recovery when the hub artifact is itself stale -
without re-downloading the multi-GB weight siblings on every retry.

The detection helper is tested in-process against onnx_hub (a leaf
module). The resolver flow runs in a subprocess because importing
acestep.engine.trt.build patches importlib.util.find_spec at module
level, which must not leak into the pytest process.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent.parent


def _write_decoder_onnx(path, *, with_steering: bool) -> None:
    import onnx
    from onnx import TensorProto, helper

    names = ["hidden_states", "timestep", "encoder_hidden_states",
             "context_latents"]
    if with_steering:
        names.append("steering")
    inputs = [
        helper.make_tensor_value_info(n, TensorProto.FLOAT, [1, 2, 3])
        for n in names
    ]
    out = helper.make_tensor_value_info("sample", TensorProto.FLOAT, [1, 2, 3])
    node = helper.make_node("Identity", [names[0]], ["sample"])
    graph = helper.make_graph([node], "decoder", inputs, [out])
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    onnx.save(helper.make_model(graph), str(path))


class TestDetectionHelper:
    def test_pre_steering_onnx_is_stale(self, tmp_path):
        from acestep.engine.trt.onnx_hub import decoder_onnx_has_steering

        p = tmp_path / "decoder_refit.onnx"
        _write_decoder_onnx(p, with_steering=False)
        assert decoder_onnx_has_steering(p) is False

    def test_current_onnx_is_fresh(self, tmp_path):
        from acestep.engine.trt.onnx_hub import decoder_onnx_has_steering

        p = tmp_path / "decoder_refit.onnx"
        _write_decoder_onnx(p, with_steering=True)
        assert decoder_onnx_has_steering(p) is True

    def test_unreadable_onnx_is_stale(self, tmp_path):
        from acestep.engine.trt.onnx_hub import decoder_onnx_has_steering

        p = tmp_path / "decoder_refit.onnx"
        p.write_bytes(b"not an onnx file")
        assert decoder_onnx_has_steering(p) is False


# Driver executed in a subprocess: monkeypatches onnx_hub's probe/fetch
# (no network), then runs the real _ensure_onnx resolution for
# decoder_refit and reports what happened as a RESULT: JSON line.
_DRIVER = r"""
import json, os, sys

scenario, root = sys.argv[1], sys.argv[2]

import onnx
from onnx import TensorProto, helper

def write_decoder_onnx(path, with_steering):
    names = ["hidden_states", "timestep", "encoder_hidden_states",
             "context_latents"]
    if with_steering:
        names.append("steering")
    inputs = [helper.make_tensor_value_info(n, TensorProto.FLOAT, [1, 2, 3])
              for n in names]
    out = helper.make_tensor_value_info("sample", TensorProto.FLOAT, [1, 2, 3])
    node = helper.make_node("Identity", [names[0]], ["sample"])
    graph = helper.make_graph([node], "decoder", inputs, [out])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    onnx.save(helper.make_model(graph), path)

trt_root = os.path.join(root, "trt_engines")
onnx_dir = os.path.join(trt_root, "_onnx_acestep-v15-turbo")
local = os.path.join(onnx_dir, "decoder_refit", "decoder_refit.onnx")
hub_probe = os.path.join(root, "hub_probe.onnx")

write_decoder_onnx(local, with_steering=(scenario == "fresh_local"))
write_decoder_onnx(hub_probe, with_steering=(scenario != "hub_stale"))

calls = {"probe": 0, "fetch": 0}

from acestep.engine.trt import onnx_hub

def fake_probe(component, *, checkpoint=None):
    calls["probe"] += 1
    return hub_probe

def fake_fetch(component, *, local_root, checkpoint=None,
               force_download=False):
    calls["fetch"] += 1
    calls["fetch_forced"] = force_download
    write_decoder_onnx(local, with_steering=True)
    return local

onnx_hub.probe_onnx_main_file = fake_probe
onnx_hub.fetch_onnx = fake_fetch

from acestep.engine.trt.build import _ensure_onnx

exit_code = 0
try:
    _ensure_onnx(
        onnx_dir=onnx_dir,
        project_root=os.getcwd(),
        checkpoint="acestep-v15-turbo",
        device="cpu",
        need_vae=False,
        need_decoder_std=False,
        need_decoder_refit=True,
        decoder_precision="fp16_mixed",
        skip_onnx=(scenario == "stale_skip_onnx"),
    )
except SystemExit as exc:
    exit_code = int(exc.code or 0)

print("RESULT:" + json.dumps({
    "calls": calls,
    "exit": exit_code,
    "local_fresh": onnx_hub.decoder_onnx_has_steering(local),
}))
"""


def _run_scenario(tmp_path, scenario: str) -> dict:
    driver = tmp_path / "driver.py"
    driver.write_text(_DRIVER, encoding="utf-8")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_REPO_ROOT)
    env["PYTHONUTF8"] = "1"
    result = subprocess.run(
        [sys.executable, str(driver), scenario, str(tmp_path)],
        cwd=str(_REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    lines = [l for l in result.stdout.splitlines() if l.startswith("RESULT:")]
    assert lines, (
        f"driver produced no RESULT (rc={result.returncode}):\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    out = json.loads(lines[-1][len("RESULT:"):])
    out["stderr"] = result.stderr
    return out


class TestResolverFlow:
    def test_fresh_local_file_is_reused_without_any_hub_traffic(self, tmp_path):
        r = _run_scenario(tmp_path, "fresh_local")
        assert r["exit"] == 0
        assert r["calls"]["probe"] == 0
        assert r["calls"]["fetch"] == 0
        assert r["local_fresh"] is True

    def test_stale_local_file_is_replaced_from_a_fresh_hub(self, tmp_path):
        r = _run_scenario(tmp_path, "stale_local")
        assert r["exit"] == 0
        assert r["calls"]["probe"] == 1
        assert r["calls"]["fetch"] == 1
        # The whole point of the probe-first flow: the multi-GB fetch is
        # a plain etag-aware download, never a cache-busting forced one.
        assert r["calls"]["fetch_forced"] is False
        assert r["local_fresh"] is True

    def test_stale_hub_exits_before_any_fetch(self, tmp_path):
        r = _run_scenario(tmp_path, "hub_stale")
        assert r["exit"] == 1
        assert r["calls"]["probe"] == 1
        # No multi-GB download or local rewrite when the hub artifact is
        # known-stale; retries cost only the probe's metadata requests.
        assert r["calls"]["fetch"] == 0
        assert r["local_fresh"] is False
        assert "--export-locally" in r["stderr"]

    def test_stale_with_skip_onnx_fails_fast_and_names_staleness(self, tmp_path):
        r = _run_scenario(tmp_path, "stale_skip_onnx")
        assert r["exit"] == 1
        assert r["calls"]["probe"] == 0
        assert r["calls"]["fetch"] == 0
        assert "Stale ONNX file" in r["stderr"]
