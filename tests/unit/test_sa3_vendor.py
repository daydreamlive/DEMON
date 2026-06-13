from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from acestep.engine import sa3_helpers


def test_default_vendor_dir_lives_under_models_dir(monkeypatch, tmp_path):
    monkeypatch.delenv(sa3_helpers.SA3_VENDOR_ENV, raising=False)
    monkeypatch.setenv("ACESTEP_MODELS_DIR", str(tmp_path))

    assert sa3_helpers.sa3_vendor_dir() == (
        tmp_path / "sa3" / "vendor" / "stable-audio-3"
    )


def test_ensure_vendor_clones_missing_tree_at_pinned_commit(
    monkeypatch, tmp_path,
):
    monkeypatch.delenv(sa3_helpers.SA3_VENDOR_ENV, raising=False)
    monkeypatch.setenv("ACESTEP_MODELS_DIR", str(tmp_path))
    calls: list[tuple[tuple[str, ...], Path | None]] = []

    def fake_git(args: list[str], cwd: Path | None = None) -> str:
        calls.append((tuple(args), cwd))
        return ""

    monkeypatch.setattr(sa3_helpers, "_git", fake_git)
    monkeypatch.setattr(sa3_helpers, "_apply_patches", lambda vendor: [])

    vendor = sa3_helpers.ensure_sa3_vendor()

    assert vendor == tmp_path / "sa3" / "vendor" / "stable-audio-3"
    assert calls == [
        (
            (
                "clone",
                "--no-checkout",
                sa3_helpers.SA3_VENDOR_URL,
                str(vendor),
            ),
            None,
        ),
        (
            (
                "-c",
                "advice.detachedHead=false",
                "checkout",
                sa3_helpers.SA3_VENDOR_SHA,
            ),
            vendor,
        ),
    ]


def test_ensure_vendor_updates_clean_existing_tree(monkeypatch, tmp_path):
    monkeypatch.delenv(sa3_helpers.SA3_VENDOR_ENV, raising=False)
    monkeypatch.setenv("ACESTEP_MODELS_DIR", str(tmp_path))
    vendor = sa3_helpers.sa3_vendor_dir()
    (vendor / ".git").mkdir(parents=True)
    old_sha = "0" * 40
    calls: list[tuple[tuple[str, ...], Path | None]] = []

    def fake_git(args: list[str], cwd: Path | None = None) -> str:
        calls.append((tuple(args), cwd))
        if args == ["status", "--porcelain"]:
            return ""
        if args == ["rev-parse", "HEAD"]:
            return old_sha
        return ""

    monkeypatch.setattr(sa3_helpers, "_git", fake_git)
    monkeypatch.setattr(sa3_helpers, "_apply_patches", lambda vendor: [])

    assert sa3_helpers.ensure_sa3_vendor() == vendor
    assert (("fetch", "origin"), vendor) in calls
    assert (
        (
            "-c",
            "advice.detachedHead=false",
            "checkout",
            sa3_helpers.SA3_VENDOR_SHA,
        ),
        vendor,
    ) in calls


def test_ensure_vendor_refuses_dirty_wrong_commit(monkeypatch, tmp_path):
    monkeypatch.delenv(sa3_helpers.SA3_VENDOR_ENV, raising=False)
    monkeypatch.setenv("ACESTEP_MODELS_DIR", str(tmp_path))
    vendor = sa3_helpers.sa3_vendor_dir()
    (vendor / ".git").mkdir(parents=True)

    def fake_git(args: list[str], cwd: Path | None = None) -> str:
        if args == ["status", "--porcelain"]:
            return " M stable_audio_3/model.py"
        if args == ["rev-parse", "HEAD"]:
            return "1" * 40
        return ""

    monkeypatch.setattr(sa3_helpers, "_git", fake_git)

    with pytest.raises(RuntimeError, match="Leaving it untouched"):
        sa3_helpers.ensure_sa3_vendor()


def test_require_vendor_error_points_to_setup(monkeypatch, tmp_path):
    monkeypatch.delenv(sa3_helpers.SA3_VENDOR_ENV, raising=False)
    monkeypatch.setenv("ACESTEP_MODELS_DIR", str(tmp_path))

    def fail_ensure():
        raise RuntimeError("network unavailable")

    monkeypatch.setattr(sa3_helpers, "ensure_sa3_vendor", fail_ensure)

    with pytest.raises(ImportError) as exc:
        sa3_helpers.require_sa3_vendor()

    msg = str(exc.value)
    assert "uv run demon-setup" in msg
    assert "vendor_sa3.py" not in msg
    assert "notes/SA3" not in msg
