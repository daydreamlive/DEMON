"""Import bridge to the SA3 spike helpers (``scripts/sa3``) and the
vendored ``stable_audio_3`` source.

The validated SA3 implementation lives on the spike branch's files,
ported additively into ``scripts/sa3/`` (loader, conditioning capture,
schedule builder, SAME windowed decode with chunk-phase alignment).
Production SA3 modules under ``acestep/`` reuse them through this one
controlled bootstrap instead of duplicating the code — the spike files
stay the single implementation until the Phase-3 serving-layer cleanup
promotes them into a real package.

``sa3_stream_pipeline`` itself imports only torch at module level, so
the helpers load without the vendored ``stable_audio_3`` source; the
vendor tree is required only by the functions that touch the model
(loading, conditioning, schedule build, SAME decode). Its canonical
location is the untracked ``notes/SA3/stable-audio-3`` tree;
``DEMON_SA3_SRC`` overrides for worktrees that don't carry it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def sa3_scripts_dir() -> Path:
    return _REPO_ROOT / "scripts" / "sa3"


def sa3_vendor_dir() -> Path:
    return Path(
        os.environ.get(
            "DEMON_SA3_SRC",
            _REPO_ROOT / "notes" / "SA3" / "stable-audio-3",
        )
    )


def ensure_sa3_paths() -> None:
    """Make the spike helpers and the vendored ``stable_audio_3`` importable.

    ``scripts/sa3`` (DEMON's own, uniquely-named spike modules) goes at the
    FRONT. The vendor repo ROOT is APPENDED, not prepended: it must be on
    ``sys.path`` for ``import stable_audio_3``, but its sibling top-level
    entries (``scripts/``, ``tests/``, ``run_gradio.py``) would otherwise
    shadow DEMON's own same-named packages. Appending keeps DEMON's modules
    winning while the vendor-only ``stable_audio_3`` still resolves.
    Idempotent; vendor absence is not an error here — vendor-needing call
    sites fail with their own ImportError."""
    scripts = str(sa3_scripts_dir())
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    vendor = str(sa3_vendor_dir())
    if vendor not in sys.path:
        sys.path.append(vendor)


def require_sa3_vendor() -> Path:
    """Fail loudly and actionably when the vendored ``stable_audio_3``
    source is absent. Called by vendor-needing entry points (SA3Context
    construction) so the operator sees the remedy instead of a deep
    ``ModuleNotFoundError: No module named 'stable_audio_3'`` from
    inside the spike loader. Returns the vendor dir on success."""
    vendor = sa3_vendor_dir()
    if not (vendor / "stable_audio_3").is_dir():
        raise ImportError(
            f"vendored stable_audio_3 source not found at {vendor}; its "
            "canonical location is the untracked notes/SA3/stable-audio-3 "
            "tree. Fetch it with `python scripts/sa3/vendor_sa3.py` (clones "
            "the upstream repo at the pinned commit), or set DEMON_SA3_SRC "
            "to a checkout that has it (e.g. "
            "DEMON_SA3_SRC=C:\\_dev\\projects\\DEMON\\notes\\SA3\\stable-audio-3)"
        )
    return vendor


def import_stream_helpers():
    """The spike's streaming helpers (``scripts/sa3/sa3_stream_pipeline``):
    cond-bundle stacking, decode-window resolution, SAME windowed decode,
    source encode. No vendor tree required to import."""
    ensure_sa3_paths()
    import sa3_stream_pipeline  # noqa: PLC0415

    return sa3_stream_pipeline


def import_loader_helpers():
    """The spike's model loader (``scripts/sa3/sa3_reference_generate``):
    local checkpoint dir resolution + ``load_local_model`` with the
    bundled-t5gemma patch. Importing is light; CALLING the loader needs
    the vendored ``stable_audio_3`` source."""
    ensure_sa3_paths()
    import sa3_reference_generate  # noqa: PLC0415

    return sa3_reference_generate
