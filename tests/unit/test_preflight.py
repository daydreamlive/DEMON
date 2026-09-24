"""Family preflights return verdicts; the server only prints them.

Each family's check is exercised with its collaborators monkeypatched, so
this runs on CPU with no checkpoints and no engines present.
"""

import pytest

from acestep.streaming import preflight as pf
from acestep.streaming.preflight import PreflightRequest, PreflightResult


# ---- ACE-Step ------------------------------------------------------------------

def test_acestep_reports_missing_checkpoints(monkeypatch):
    import acestep.model_downloader as md

    monkeypatch.setattr(md, "ensure_main_model", lambda: (False, "no weights here"))
    res = pf.acestep_preflight(PreflightRequest("acestep-v15-turbo"))
    assert not res.ok
    assert res.title == "Model checkpoints unavailable"
    assert "no weights here" in res.lines[0]
    assert any("demon-setup" in l for l in res.lines)


def test_acestep_checks_the_dit_variant_when_not_turbo(monkeypatch):
    import acestep.model_downloader as md

    seen = {}
    monkeypatch.setattr(md, "ensure_main_model", lambda: (True, ""))
    monkeypatch.setattr(md, "ensure_dit_model", lambda ck: seen.setdefault("ck", ck) and (False, "xl missing"))
    res = pf.acestep_preflight(PreflightRequest("acestep-v15-xl-turbo", "eager", "eager"))
    assert seen["ck"] == "acestep-v15-xl-turbo"
    assert not res.ok and "xl missing" in res.lines[0]


def test_acestep_skips_engines_when_nothing_runs_tensorrt(monkeypatch):
    import acestep.model_downloader as md
    import acestep.paths as paths

    monkeypatch.setattr(md, "ensure_main_model", lambda: (True, ""))
    monkeypatch.setattr(paths, "available_trt_engines", lambda **kw: pytest.fail("must not check engines"))
    assert pf.acestep_preflight(PreflightRequest("acestep-v15-turbo", "eager", "compile")).ok


def test_acestep_reports_unbuilt_engines_with_the_build_command(monkeypatch):
    import acestep.model_downloader as md
    import acestep.paths as paths

    monkeypatch.setattr(md, "ensure_main_model", lambda: (True, ""))
    seen = {}

    def _raise(**kw):
        seen.update(kw)
        # The real error: it derives its message and build command from the
        # checkpoint's profile table, so the banner carries a runnable fix.
        raise paths.EngineNotBuiltError(
            60.0, ("decoder",), {60.0: ["decoder_fp16_60s.engine"]},
            checkpoint="acestep-v15-turbo",
        )

    monkeypatch.setattr(paths, "available_trt_engines", _raise)
    res = pf.acestep_preflight(PreflightRequest("acestep-v15-turbo", "tensorrt", "eager"))
    assert not res.ok and res.title == "TensorRT engines not built"
    assert seen["needs"] == ("decoder",) and seen["checkpoint"] == "acestep-v15-turbo"
    assert any("fix (manual)" in l for l in res.lines), res.lines
    assert any("--accel compile" in l for l in res.lines)


def test_acestep_vae_only_tensorrt_uses_the_turbo_profile(monkeypatch):
    import acestep.model_downloader as md
    import acestep.paths as paths

    monkeypatch.setattr(md, "ensure_main_model", lambda: (True, ""))
    monkeypatch.setattr(md, "ensure_dit_model", lambda ck: (True, ""))
    seen = {}
    monkeypatch.setattr(paths, "available_trt_engines", lambda **kw: seen.update(kw))
    res = pf.acestep_preflight(PreflightRequest("acestep-v15-xl-turbo", "compile", "tensorrt"))
    assert res.ok
    assert seen["needs"] == ("vae_encode", "vae_decode")
    # A VAE-only TRT setup must not demand XL decoder engines.
    assert seen["checkpoint"] == "acestep-v15-turbo"


# ---- Stable Audio 3 -----------------------------------------------------------

def test_sa3_reports_a_missing_catalog_checkpoint(monkeypatch):
    import acestep.engine.sa3_helpers as h

    monkeypatch.setattr(h, "sa3_checkpoint_status", lambda mid: (False, f"{mid}: download it"))
    res = pf.sa3_preflight(PreflightRequest("medium"))
    assert not res.ok and res.title == "SA3 model unavailable"
    assert res.lines == ("medium: download it",)


def test_sa3_validates_the_operator_directory_when_given(monkeypatch):
    import acestep.engine.sa3_helpers as h

    monkeypatch.setattr(h, "sa3_checkpoint_status", lambda mid: pytest.fail("catalog path must not run"))
    seen = {}
    monkeypatch.setattr(h, "sa3_custom_checkpoint_status", lambda d: seen.setdefault("dir", d) and (True, "ready"))
    assert pf.sa3_preflight(PreflightRequest("medium", checkpoint_dir="/tmp/ckpt")).ok
    assert seen["dir"] == "/tmp/ckpt"


# ---- the server printer ------------------------------------------------------

def test_server_prints_the_banner_and_exits_on_a_failed_verdict(capsys):
    from demos.realtime_motion_graph_web import server
    from acestep.streaming.families import FamilySpec

    spec = FamilySpec(
        name="demo", display_name="Demo", make_backend=lambda ss: None,
        knob_universe=lambda: [],
        preflight=lambda req: PreflightResult.failed("Demo unavailable", "first line", "", "fix: do the thing"),
    )
    with pytest.raises(SystemExit) as exc:
        server._run_family_preflight(spec, PreflightRequest("x"))
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "Demo unavailable" in out and "fix: do the thing" in out


def test_server_passes_on_ok_and_tolerates_a_family_without_preflight():
    from demos.realtime_motion_graph_web import server
    from acestep.streaming.families import FamilySpec

    ok_spec = FamilySpec(
        name="demo", display_name="Demo", make_backend=lambda ss: None,
        knob_universe=lambda: [], preflight=lambda req: PreflightResult.passed(),
    )
    server._run_family_preflight(ok_spec, PreflightRequest("x"))
    bare = FamilySpec(
        name="bare", display_name="Bare", make_backend=lambda ss: None, knob_universe=lambda: [],
    )
    server._run_family_preflight(bare, PreflightRequest("x"))
