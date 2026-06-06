"""Thin forwarder: the SAME-L window decoder TRT build lives in the package now.

Equivalent to:
    python -m acestep.engine.trt.sa3_build --same-l-window [latent flags]

Kept so spike-era invocations keep working; see
acestep/engine/trt/sa3_build.py for the canonical recipe and the
metadata-gated --all matrix.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Unconditionally force the repo to the FRONT of sys.path: a sibling
# ACE-Step checkout can shadow ``acestep`` otherwise.
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from acestep.engine.trt.sa3_build import main  # noqa: E402

if __name__ == "__main__":
    sys.argv.insert(1, "--same-l-window")
    raise SystemExit(main())
