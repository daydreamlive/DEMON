"""Local provenance: the Level-1 session action log.

The lightweight cut of the DEMON provenance architecture (see the
demon-provenance design doc, "Level 1: Log Actions"): record what
happened in a session — actions, parameter changes, prompts,
input-source hashes, models — locally and to the cloud action log.
Deliberately NO cryptographic machinery here (no slice hashes, no
receipts, no C2PA); that is the follow-up tier.

- :mod:`~acestep.provenance.session_log` — local JSONL session logs
  using the same event schema as the cloud action log, tapped off the
  streaming event bus.
- :mod:`~acestep.provenance.ledger_client` — thin batched event POST
  client, a complete no-op until ``DEMON_LEDGER_URL`` is set.

Everything here degrades to a no-op (one logged warning, never an
exception on a write path).
"""

from __future__ import annotations

import os
from pathlib import Path

_ENV_PROVENANCE_DIR = "ACESTEP_PROVENANCE_DIR"
_DEFAULT_PROVENANCE_DIR = os.path.join(
    os.path.expanduser("~"), ".daydream-scope", "provenance",
)


def provenance_dir() -> Path:
    """Root directory for local provenance state (session logs).

    Resolution order mirrors :func:`acestep.paths.models_dir`:
        1. ACESTEP_PROVENANCE_DIR environment variable
        2. ~/.daydream-scope/provenance

    Deliberately NOT memoized: tests point it at a tmp dir per case,
    and it is read only on session start, never per tick.
    """
    return Path(os.environ.get(_ENV_PROVENANCE_DIR, _DEFAULT_PROVENANCE_DIR))


def session_logs_dir() -> Path:
    """Directory holding per-session JSONL event logs."""
    return provenance_dir() / "sessions"
