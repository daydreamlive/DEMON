"""Fetch DEMON's managed Stable Audio 3 source checkout.

This is a developer/debugging wrapper around the same vendoring function
called by ``uv run demon-setup``. Users should not need to run it directly.
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

from acestep.engine.sa3_helpers import (  # noqa: E402
    SA3_VENDOR_SHA,
    SA3_VENDOR_URL,
    ensure_sa3_vendor,
    sa3_vendor_dir,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="report state only")
    ap.add_argument("--url", default=SA3_VENDOR_URL, help="clone URL")
    args = ap.parse_args()

    vendor = sa3_vendor_dir()
    print(f"[vendor] target: {vendor}")
    print(f"[vendor] source: {args.url} @ {SA3_VENDOR_SHA}")
    try:
        ensure_sa3_vendor(check_only=args.check, url=args.url)
    except FileNotFoundError as exc:
        print(f"[vendor] MISSING: {exc}")
        return 1
    except subprocess.CalledProcessError as exc:
        err = (exc.stderr or exc.stdout or str(exc)).strip()
        print(f"[vendor] git failed: {err}")
        return exc.returncode or 1
    except Exception as exc:
        print(f"[vendor] ERROR: {exc}")
        return 1
    print("[vendor] ready")
    return 0


if __name__ == "__main__":
    sys.exit(main())
