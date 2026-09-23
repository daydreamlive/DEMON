"""Local provenance: session logs, C2PA manifests, local signing keys.

Phase-1 scope of the DEMON provenance architecture (see the
demon-provenance spec, 02-architecture.md §2/§3/§5/§7):

- :mod:`~acestep.provenance.session_log` — local JSONL session logs
  using the same event schema as the (future) cloud ledger, tapped off
  the streaming event bus.
- :mod:`~acestep.provenance.keys` — self-signed local ES256 signing
  material, claim generator "DEMON (local, self-signed)".
- :mod:`~acestep.provenance.manifest` — C2PA manifest construction +
  embedding on the track-asset WAV writers via ``c2pa-python``.
- :mod:`~acestep.provenance.ledger_client` — thin event/slice-hash
  POST client, a complete no-op until ``DEMON_LEDGER_URL`` is set.
- :mod:`~acestep.provenance.receipts` — client-side verification of
  the ledger's signed receipts (spec 06 §2.4: key-cert chain, JCS +
  chain-head recomputation, Ed25519 framing checks).

Everything here degrades to a no-op (one logged warning, never an
exception on a write path) when the optional ``provenance`` extra
(``c2pa-python`` + ``cryptography``) is not installed.
"""

from __future__ import annotations

import os
from pathlib import Path

_ENV_PROVENANCE_DIR = "ACESTEP_PROVENANCE_DIR"
_DEFAULT_PROVENANCE_DIR = os.path.join(
    os.path.expanduser("~"), ".daydream-scope", "provenance",
)


def provenance_dir() -> Path:
    """Root directory for local provenance state (keys, session logs).

    Resolution order mirrors :func:`acestep.paths.models_dir`:
        1. ACESTEP_PROVENANCE_DIR environment variable
        2. ~/.daydream-scope/provenance

    Deliberately NOT memoized: tests point it at a tmp dir per case,
    and it is read only on session start / file sign, never per tick.
    """
    return Path(os.environ.get(_ENV_PROVENANCE_DIR, _DEFAULT_PROVENANCE_DIR))


def session_logs_dir() -> Path:
    """Directory holding per-session JSONL event logs."""
    return provenance_dir() / "sessions"


def keys_dir() -> Path:
    """Directory holding the local self-signed signing material."""
    return provenance_dir() / "keys"
