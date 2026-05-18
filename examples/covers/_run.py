#!/usr/bin/env python3
"""One-shot runner: applies the legacy TextEncode shim, then exec's the
named example workflow as ``__main__``.

Usage::

    uv run python -u examples/covers/_run.py initial_noise_curve
    uv run python -u examples/covers/_run.py x0_target_blend
    uv run python -u examples/covers/_run.py velocity_scaling
    uv run python -u examples/covers/_run.py guidance_curve
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

# Apply the shim FIRST so the workflow's top-level import resolves.
from examples.covers import _textencode_shim  # noqa: F401  (side-effect import)


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: _run.py <workflow_stem>", file=sys.stderr)
        sys.exit(2)
    stem = sys.argv[1]
    if stem.endswith(".py"):
        stem = stem[:-3]
    target = Path(__file__).resolve().parent / f"{stem}.py"
    if not target.is_file():
        raise SystemExit(f"no such workflow: {target}")
    print(f"[_run] executing {target}", flush=True)
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
