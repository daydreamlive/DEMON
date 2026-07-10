"""Local ``.env`` fallback for demo entrypoints.

Deployment environments inject process variables directly. This helper only
fills missing values from the repo-root ``.env`` so explicit environment wins.
"""

from __future__ import annotations

import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]


def load_repo_env_defaults(path: Path | None = None) -> int:
    """Load missing environment variables from ``path`` or repo ``.env``."""
    env_path = path or (ROOT_DIR / ".env")
    if not env_path.is_file():
        return 0

    loaded = 0
    for raw in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        os.environ[key] = value.strip().strip('"').strip("'")
        loaded += 1
    return loaded
