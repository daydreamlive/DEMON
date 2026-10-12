"""SA3 fp8 stock-trunk proof + calibration scripts: CPU tests with fake DiTs."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "sa3"))

import sa3_fp8_stock_calib as calib  # noqa: E402
import sa3_fp8_stock_proof as proof  # noqa: E402

SCHED = [1.0, 0.9, 0.75, 0.6, 0.45, 0.3, 0.18, 0.08, 0.0]


class FakeDiT(torch.nn.Module):
    """v = W x + t, with an optional per-call perturbation."""

    def __init__(self, c=4, noise=0.0, seed=0, nan_at=None):
        super().__init__()
        g = torch.Generator().manual_seed(123)
        self.w = torch.randn(c, c, generator=g) * 0.3
        self.noise, self.nan_at, self.calls = noise, nan_at, 0
        self.g = torch.Generator().manual_seed(seed)

    def step(self, x, t, bundle):
        self.calls += 1
        v = torch.einsum("ij,bjl->bil", self.w, x) + t + bundle["bias"]
        if self.noise:
            v = v + self.noise * torch.randn(v.shape, generator=self.g)
        if self.nan_at is not None and self.calls == self.nan_at:
            v = v.clone()
            v[0, 0, 0] = float("nan")
        return v


def _cases(n=2, L=16):
    out = []
    for i in range(n):
        g = torch.Generator().manual_seed(i)
        out.append({"id": f"c{i}", "x_T": torch.randn(1, 4, L, generator=g),
                    "sched": SCHED, "bundle": {"bias": torch.full((1, 4, 1), 0.1 * i)}})
    return out


def _arm(noise=0.0, nan_at=None, ms=None, cases=None):
    eager = FakeDiT()
    eng = FakeDiT(noise=noise, seed=5, nan_at=nan_at)
    clock = None
    if ms is not None:
        state = {"t": 0.0, "n": 0}

        def clock():
            state["n"] += 1
            if state["n"] % 2 == 0:
                state["t"] += ms / 1e3
            return state["t"]
    return proof.measure_arm(eager.step, eng.step, cases or _cases(), clock=clock)


def test_identical_engine_is_cos_one():
    m = _arm()
    assert m["worst_step_cos"] == pytest.approx(1.0, abs=1e-12)
    assert m["compounded_cos_min"] == pytest.approx(1.0, abs=1e-12)
    assert m["nonfinite"] == 0
    assert m["n_cases"] == 2
    assert len(m["worst_by_step"]) == len(SCHED) - 1


def test_noisy_engine_lowers_cos_and_compounds():
    m = _arm(noise=0.05)
    assert 0.9 < m["worst_step_cos"] < 1.0
    assert m["mean_step_cos"] >= m["worst_step_cos"]
    assert m["compounded_cos_min"] < 1.0


def test_warmups_run_before_measurement():
    eng = FakeDiT()
    eager = FakeDiT()
    proof.measure_arm(eager.step, eng.step, _cases(1), warmups=2)
    steps = len(SCHED) - 1
    # 2 warm-ups + shadow (steps) + own rollout (steps)
    assert eng.calls == 2 * steps + steps + steps


def test_euler_rollout_reaches_x0_estimate():
    eager = FakeDiT()
    x_T = torch.randn(1, 4, 8)
    x, cos, ms, nf = proof.euler_rollout(eager.step, x_T, SCHED, {"bias": torch.zeros(1)})
    assert cos == [] and ms == [] and nf == 0
    # Single-step euler from t=1 to 0 is x_T - v.
    x1, *_ = proof.euler_rollout(eager.step, x_T, [1.0, 0.0], {"bias": torch.zeros(1)})
    assert torch.allclose(x1, x_T - eager.step(x_T, 1.0, {"bias": torch.zeros(1)}))


def test_nonfinite_in_warmup_is_counted():
    m = _arm(nan_at=5)
    assert m["warmup_nonfinite"] >= 1 and m["nonfinite"] >= m["warmup_nonfinite"]


def test_nonfinite_engine_fails_arm():
    m = _arm(nan_at=20)   # past the 16 warm-up calls: inside the shadow pass
    assert m["nonfinite"] >= 1
    assert math.isnan(m["worst_step_cos"])
    arms = {"hf_fp8": _arm(), "ours_fp16mixed": _arm(), "ours_fp8": m}
    v = proof.judge_arms(arms)
    assert v["gate"] == "FAIL"
    assert not v["tiers"]["ours_fp8"]["pass"]


def test_ms_per_step_from_clock():
    m = _arm(ms=3.0)
    assert m["ms_step_median"] == pytest.approx(3.0)


def _fake_metrics(worst, comp, ms, nonfinite=0):
    return {"worst_step_cos": worst, "mean_step_cos": worst, "worst_by_step": [worst] * 8,
            "compounded_cos_min": comp, "compounded_cos_mean": comp,
            "ms_step_median": ms, "nonfinite": nonfinite}


def test_judge_all_pass():
    v = proof.judge_arms({
        "hf_fp8": _fake_metrics(0.95, 0.97, 10.0),
        "ours_fp16mixed": _fake_metrics(0.99985, 0.9999, 15.0),
        "ours_fp8": _fake_metrics(0.945, 0.965, 10.4),
    })
    assert v["gate"] == "PASS", v["checks"]
    assert v["tiers"]["ours_fp16mixed"]["tier"] == "fp16mixed"
    assert v["tiers"]["ours_fp8"]["tier"] == "fp8"


@pytest.mark.parametrize("arm,metrics,failing", [
    ("ours_fp16mixed", _fake_metrics(0.9997, 0.9999, 15.0), "ours_fp16mixed worst step"),
    ("ours_fp8", _fake_metrics(0.93, 0.965, 10.0), "ours_fp8 worst step vs hf_fp8"),
    ("ours_fp8", _fake_metrics(0.95, 0.95, 10.0), "ours_fp8 compounded vs hf_fp8"),
    ("ours_fp8", _fake_metrics(0.95, 0.97, 10.6), "ours_fp8 ms/step vs hf_fp8"),
])
def test_judge_each_bar(arm, metrics, failing):
    arms = {"hf_fp8": _fake_metrics(0.95, 0.97, 10.0),
            "ours_fp16mixed": _fake_metrics(0.99985, 0.9999, 15.0),
            "ours_fp8": _fake_metrics(0.95, 0.97, 10.0)}
    arms[arm] = metrics
    v = proof.judge_arms(arms)
    assert v["gate"] == "FAIL"
    failed = {c["check"] for c in v["checks"] if not c["pass"]}
    assert failing in failed


def test_fp8_floor_independent_of_control():
    v = proof.judge_arms({"hf_fp8": _fake_metrics(0.85, 0.9, 10.0),
                          "ours_fp16mixed": _fake_metrics(0.9999, 0.9999, 15.0),
                          "ours_fp8": _fake_metrics(0.88, 0.9, 10.0)})
    failed = {c["check"] for c in v["checks"] if not c["pass"]}
    assert "ours_fp8 worst step" in failed


def test_missing_arm_is_incomplete():
    v = proof.judge_arms({"hf_fp8": _fake_metrics(0.95, 0.97, 10.0)})
    assert v["gate"] == "incomplete"
    assert set(v["missing_arms"]) == {"ours_fp16mixed", "ours_fp8"}


def test_markdown_table_has_every_arm_and_gate():
    arms = {"hf_fp8": _arm(), "ours_fp16mixed": _arm(), "ours_fp8": _arm(noise=0.01)}
    for name, m in arms.items():
        m["engine"] = f"eng_{name}"
    report = {"latent_frames": 16, "n_cases": 2, "arms": arms,
              "verdict": proof.judge_arms(arms), "eager_ms_step_median": 1.0}
    md = proof.markdown_table(report)
    for name in arms:
        assert f"| {name} | eng_{name} |" in md
    assert "GATE:" in md
    json.dumps(report, default=str)


def test_parse_arms_and_resolve_engine(tmp_path):
    arms = proof.parse_arms([f"hf_fp8={tmp_path}", "ours_fp8 = x"])
    assert set(arms) == {"hf_fp8", "ours_fp8"}
    with pytest.raises(ValueError):
        proof.parse_arms(["bogus=x"])
    with pytest.raises(ValueError):
        proof.parse_arms(["hf_fp8=a", "hf_fp8=b"])
    d = tmp_path / "sa3_m_dit_fp8_l1_140_140"
    d.mkdir()
    (d / "sa3_m_dit_fp8_l1_140_140.trt").write_bytes(b"x")
    assert proof.resolve_engine(d).name == "sa3_m_dit_fp8_l1_140_140.trt"
    e = tmp_path / "other"
    e.mkdir()
    (e / "engine.trt").write_bytes(b"x")
    assert proof.resolve_engine(e).name == "engine.trt"
    with pytest.raises(FileNotFoundError):
        proof.resolve_engine(tmp_path / "missing")


# ---------------------------------------------------------------------------
# calibration script
# ---------------------------------------------------------------------------

class TinyDiT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = torch.nn.Linear(4, 128)
        self.mid = torch.nn.Linear(128, 128)
        self.out = torch.nn.Linear(128, 4)
        self.calls = 0

    def forward(self, x, t, *, bias):
        self.calls += 1
        h = self.inp(x.transpose(1, 2)) + t
        return self.out(torch.relu(self.mid(h))).transpose(1, 2) + bias


def _fake_capture(module, run, *, min_features=128):
    amax, hooks = {}, []
    for name, mod in module.named_modules():
        if isinstance(mod, torch.nn.Linear) and min(mod.in_features, mod.out_features) >= min_features:
            def hook(m, inp, _out, name=name):
                amax[name] = max(amax.get(name, 0.0), float(inp[0].abs().amax()))
            hooks.append(mod.register_forward_hook(hook))
    try:
        run()
    finally:
        for h in hooks:
            h.remove()
    return amax


def test_calibration_run_drives_every_job_through_capture(tmp_path):
    dit = TinyDiT()
    jobs = [{"name": f"j{i}", "bundle": {"bias": torch.zeros(1)}, "sched": SCHED,
             "latent_shape": (1, 4, 10), "seed": i} for i in range(15)]
    logs = []
    run = calib.make_calibration_run(dit, jobs, device="cpu", dtype=torch.float32,
                                     log=logs.append)
    scales = _fake_capture(dit, run)
    assert dit.calls == 15 * (len(SCHED) - 1)
    assert set(scales) == {"mid"}          # only min(in,out) >= 128
    assert len(logs) == 15
    out, meta = calib.write_scales(scales, tmp_path / "calib" / "stock_act_scales.json",
                                   {"module_root": calib.MODULE_ROOT})
    data = json.loads(out.read_text())
    assert data == {"mid": pytest.approx(scales["mid"])}
    m = json.loads(meta.read_text())
    assert m["n_modules"] == 1 and len(m["scales_sha256"]) == 64
    assert meta.name == "stock_act_scales.meta.json"


def test_pingpong_rollout_is_seeded():
    dit = TinyDiT()
    a = calib.pingpong_rollout(dit, {"bias": torch.zeros(1)}, SCHED,
                               latent_shape=(1, 4, 6), seed=3, device="cpu", dtype=torch.float32)
    b = calib.pingpong_rollout(dit, {"bias": torch.zeros(1)}, SCHED,
                               latent_shape=(1, 4, 6), seed=3, device="cpu", dtype=torch.float32)
    assert torch.equal(a, b)


def test_write_scales_rejects_empty_and_nonfinite(tmp_path):
    with pytest.raises(ValueError):
        calib.write_scales({}, tmp_path / "s.json", {})
    with pytest.raises(ValueError):
        calib.write_scales({"a": float("nan")}, tmp_path / "s.json", {})


def test_calib_exits_cleanly_without_d1(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "acestep.engine.trt.sa3_fp8":
            raise ImportError("not yet")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(SystemExit) as exc:
        calib._load_capture()
    assert "d1_capture" in str(exc.value)
