"""Import and vendor management for the SA3 integration.

The production SA3 path reuses the validated spike helpers in
``scripts/sa3`` and the upstream ``stable_audio_3`` package source. The
source checkout is a managed install artifact: normal setup fetches the
pinned repository commit under ``ACESTEP_MODELS_DIR``. ``DEMON_SA3_SRC``
remains as a developer-only override for experiments with a local checkout.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from acestep import paths

_REPO_ROOT = Path(__file__).resolve().parents[2]

# DEMON tracks this fork branch until the SA3 TensorRT/FP8 producer work
# merges upstream. The hash is the reproducibility boundary for installs.
SA3_VENDOR_URL = "https://github.com/ryanontheinside/stable-audio-3"
SA3_VENDOR_SHA = "03992120a3e296562271f209f4dd61dcfb55afff"
SA3_VENDOR_ENV = "DEMON_SA3_SRC"
SA3_VENDOR_DIRNAME = "stable-audio-3"


def sa3_scripts_dir() -> Path:
    return _REPO_ROOT / "scripts" / "sa3"


def sa3_vendor_root() -> Path:
    """Managed root for SA3 source checkouts."""
    return paths.models_dir() / "sa3" / "vendor"


def sa3_vendor_dir() -> Path:
    override = os.environ.get(SA3_VENDOR_ENV)
    if override:
        return Path(override).expanduser()
    return sa3_vendor_root() / SA3_VENDOR_DIRNAME


def _git(args: list[str], cwd: Path | None = None) -> str:
    out = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


def _patch_files() -> list[Path]:
    patch_dir = sa3_scripts_dir() / "vendor_patches"
    return sorted(patch_dir.glob("*.patch")) if patch_dir.is_dir() else []


def _patch_state(vendor: Path, patch: Path) -> str:
    applied = subprocess.run(
        ["git", "apply", "--reverse", "--check", str(patch)],
        cwd=vendor,
        capture_output=True,
        text=True,
    )
    if applied.returncode == 0:
        return "applied"
    fresh = subprocess.run(
        ["git", "apply", "--check", str(patch)],
        cwd=vendor,
        capture_output=True,
        text=True,
    )
    return "appliable" if fresh.returncode == 0 else "conflict"


def _apply_patches(vendor: Path) -> list[str]:
    messages: list[str] = []
    for patch in _patch_files():
        state = _patch_state(vendor, patch)
        if state == "applied":
            messages.append(f"patch already applied: {patch.name}")
        elif state == "appliable":
            _git(["apply", str(patch)], cwd=vendor)
            messages.append(f"applied patch: {patch.name}")
        else:
            raise RuntimeError(
                f"SA3 vendor patch does not apply cleanly: {patch.name}"
            )
    return messages


def ensure_sa3_vendor(
    *,
    check_only: bool = False,
    fetch: bool = True,
    url: str = SA3_VENDOR_URL,
) -> Path:
    """Ensure the pinned SA3 source tree exists and is importable.

    The normal path is fully managed: clone the configured repository under
    ``models_dir()/sa3/vendor`` and check out :data:`SA3_VENDOR_SHA`.
    Re-running is idempotent. Existing dirty trees are left untouched and
    reported as an error instead of being overwritten.
    """
    vendor = sa3_vendor_dir()
    if not (vendor / ".git").is_dir():
        if check_only:
            raise FileNotFoundError(f"SA3 vendor source is missing at {vendor}")
        vendor.parent.mkdir(parents=True, exist_ok=True)
        _git(["clone", "--no-checkout", url, str(vendor)])
        _git(
            ["-c", "advice.detachedHead=false", "checkout", SA3_VENDOR_SHA],
            cwd=vendor,
        )
        _apply_patches(vendor)
        return vendor

    dirty = _git(["status", "--porcelain"], cwd=vendor)
    head = _git(["rev-parse", "HEAD"], cwd=vendor)
    if head == SA3_VENDOR_SHA:
        return vendor

    if dirty:
        raise RuntimeError(
            f"SA3 vendor source at {vendor} has local changes and is at "
            f"{head[:12]}, not pinned {SA3_VENDOR_SHA[:12]}. Leaving it "
            "untouched."
        )
    if check_only:
        raise RuntimeError(
            f"SA3 vendor source at {vendor} is at {head[:12]}, expected "
            f"{SA3_VENDOR_SHA[:12]}"
        )
    if fetch:
        _git(["fetch", "origin"], cwd=vendor)
    _git(
        ["-c", "advice.detachedHead=false", "checkout", SA3_VENDOR_SHA],
        cwd=vendor,
    )
    _apply_patches(vendor)
    return vendor


def ensure_sa3_paths() -> None:
    """Make the SA3 spike helpers and vendor source importable.

    ``scripts/sa3`` goes at the front because those files are DEMON's own
    uniquely-named spike modules. The vendor repo root is appended: it must
    be on ``sys.path`` for ``import stable_audio_3``, but its top-level
    ``scripts`` and ``tests`` directories must not shadow DEMON modules.
    """
    scripts = str(sa3_scripts_dir())
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    vendor = str(sa3_vendor_dir())
    if vendor not in sys.path:
        sys.path.append(vendor)


def require_sa3_vendor() -> Path:
    """Return an importable SA3 vendor tree, fetching it when needed."""
    try:
        vendor = ensure_sa3_vendor()
    except (OSError, subprocess.CalledProcessError, RuntimeError) as exc:
        raise ImportError(
            "SA3 vendor source could not be prepared automatically. "
            f"Expected {SA3_VENDOR_URL} at commit {SA3_VENDOR_SHA}. "
            f"Run `uv run demon-setup` to retry. Original error: {exc}"
        ) from exc
    if not (vendor / "stable_audio_3").is_dir():
        raise ImportError(
            f"SA3 vendor source at {vendor} does not contain "
            "`stable_audio_3`. Run `uv run demon-setup` to re-fetch "
            f"{SA3_VENDOR_URL} at {SA3_VENDOR_SHA}."
        )
    ensure_sa3_paths()
    return vendor


def import_stream_helpers():
    """Return the SA3 streaming helper module from ``scripts/sa3``."""
    ensure_sa3_paths()
    import sa3_stream_pipeline  # noqa: PLC0415

    return sa3_stream_pipeline


def import_loader_helpers():
    """Return the SA3 local model loader helper from ``scripts/sa3``."""
    ensure_sa3_paths()
    import sa3_reference_generate  # noqa: PLC0415

    return sa3_reference_generate
