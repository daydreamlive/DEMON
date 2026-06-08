"""Fetch the vendored ``stable-audio-3`` source tree at a pinned commit.

DEMON's SA3 integration imports the upstream ``stable_audio_3`` package
and reuses parts of its ``optimized/tensorRT`` tooling. The tree lives
UNTRACKED at ``notes/SA3/stable-audio-3`` (override with
``DEMON_SA3_SRC``); this script makes that arrangement reproducible on a
fresh checkout (dev box, pod) instead of depending on a hand-made clone.

Behavior:
* absent  -> clone ``--no-checkout``, check out :data:`PINNED_SHA`
  (detached; uses ``git -c advice.detachedHead=false checkout`` inside
  the fresh clone only), then apply the DEMON-side patches in
  ``vendor_patches/`` (small carries not in the pinned rev, e.g. the
  SAME-L plugin signature fix the decoder runtime imports).
* present -> fetch, report where the local tree stands relative to the
  pin and whether each DEMON-side patch is applied, and TOUCH NOTHING
  (local work in the vendor tree is sacred).

Run:
    .venv/Scripts/python.exe scripts/sa3/vendor_sa3.py
    .venv/Scripts/python.exe scripts/sa3/vendor_sa3.py --check   # no network clone
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = next(
    p for p in (_HERE, *_HERE.parents) if (p / "pyproject.toml").exists()
)
while str(_REPO_ROOT) in sys.path:
    sys.path.remove(str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT))

from acestep.engine.sa3_helpers import sa3_vendor_dir  # noqa: E402

# DEMON tracks the fork branch that carries the FP8 producer recipe and the
# `--precision fp8` wiring until that work merges upstream. Override with
# --url (e.g. point at Stability-AI once merged).
UPSTREAM_URL = "https://github.com/ryanontheinside/stable-audio-3"
# feat/dit-fp8 tip (draft PR -> Stability-AI/stable-audio-3). Carries the
# fp16mixed ONNX recipe + optimized/tensorRT consumer scripts + the FP8
# producer (build_dit_fp8.py / make_calib.py). DEMON-side carries that are
# NOT in this rev live in vendor_patches/ and are applied on a fresh clone.
PINNED_SHA = "eb8a2fd"

# DEMON-side patches applied on top of the pinned rev on a FRESH clone only
# (an existing tree is sacred — there we only report whether each is applied).
PATCHES_DIR = _HERE / "vendor_patches"


def _git(args: list[str], cwd: Path | None = None) -> str:
    out = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


def _patch_files() -> list[Path]:
    return sorted(PATCHES_DIR.glob("*.patch")) if PATCHES_DIR.is_dir() else []


def _patch_state(vendor: Path, patch: Path) -> str:
    """One of 'applied' / 'appliable' / 'conflict' for ``patch`` against the
    vendor working tree (probed with ``git apply --check``, no mutation)."""
    applied = subprocess.run(
        ["git", "apply", "--reverse", "--check", str(patch)],
        cwd=vendor, capture_output=True, text=True,
    )
    if applied.returncode == 0:
        return "applied"
    fresh = subprocess.run(
        ["git", "apply", "--check", str(patch)],
        cwd=vendor, capture_output=True, text=True,
    )
    return "appliable" if fresh.returncode == 0 else "conflict"


def _apply_patches(vendor: Path) -> None:
    """Apply every ``vendor_patches/*.patch`` onto a FRESH clone."""
    for patch in _patch_files():
        state = _patch_state(vendor, patch)
        if state == "applied":
            print(f"[vendor] patch already applied: {patch.name}")
        elif state == "appliable":
            _git(["apply", str(patch)], cwd=vendor)
            print(f"[vendor] applied patch: {patch.name}")
        else:
            print(f"[vendor] WARNING: patch does not apply cleanly "
                  f"(pin moved?): {patch.name}")


def _report_patches(vendor: Path) -> None:
    """Report patch state on an existing tree without mutating it."""
    for patch in _patch_files():
        state = _patch_state(vendor, patch)
        note = "" if state == "applied" else (
            "  (apply it or re-clone — DEMON's SAME-L decode imports this)"
        )
        print(f"[vendor] patch {patch.name}: {state}{note}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true",
                    help="report state only; never clone or fetch")
    ap.add_argument("--url", default=UPSTREAM_URL,
                    help="clone URL (e.g. your fork)")
    args = ap.parse_args()

    vendor = sa3_vendor_dir()
    if not (vendor / ".git").is_dir():
        if args.check:
            print(f"[vendor] MISSING: {vendor}")
            print(f"[vendor] run without --check to clone {args.url}"
                  f" @ {PINNED_SHA}")
            return 1
        print(f"[vendor] cloning {args.url} -> {vendor}")
        vendor.parent.mkdir(parents=True, exist_ok=True)
        _git(["clone", "--no-checkout", args.url, str(vendor)])
        _git(["-c", "advice.detachedHead=false", "checkout", PINNED_SHA],
             cwd=vendor)
        print(f"[vendor] checked out {PINNED_SHA} (detached)")
        _apply_patches(vendor)
        return 0

    # Existing tree: report, never mutate.
    head = _git(["rev-parse", "--short", "HEAD"], cwd=vendor)
    dirty = _git(["status", "--porcelain"], cwd=vendor)
    if not args.check:
        try:
            _git(["fetch", "origin"], cwd=vendor)
        except subprocess.CalledProcessError as e:
            print(f"[vendor] fetch failed (offline?): {e.stderr.strip()}")
    print(f"[vendor] present: {vendor}")
    print(f"[vendor] HEAD={head} pin={PINNED_SHA}"
          f" {'(MATCHES)' if head.startswith(PINNED_SHA) or PINNED_SHA.startswith(head) else '(DIFFERS — fine if intentional)'}")
    if dirty:
        print(f"[vendor] local changes present ({len(dirty.splitlines())} files)"
              " — left untouched")
    _report_patches(vendor)
    return 0


if __name__ == "__main__":
    sys.exit(main())
