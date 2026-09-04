"""MiniMax-Music3 checkpoint discovery and layout.

The weights live outside the repo like every other model here, under
``<models dir>/minimax/`` (``ACESTEP_MODELS_DIR`` overrides), or in the
ordinary Hugging Face cache when the operator already has one. The
upstream repo ships TWO complete copies of the model — a diffusers
layout and an sglang-omni native layout — totalling ~57 GB. We use the
diffusers layout only; ``MINIMAX_DIFFUSERS_COMPONENTS`` is what a
complete install must contain, and the status helper reports which
piece is missing rather than failing on the first absent file.

Unlike SA3 there is no vendored source tree to pin: the renderer is
reimplemented in :mod:`acestep.engine.minimax_dit` precisely so nothing
here depends on a ``diffusers`` version the rest of the repo cannot
carry.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

MINIMAX_HF_REPO = "MiniMaxAI/MiniMax-Music3"

# Only what inference needs from the diffusers layout. The sglang-omni
# duplicate (flowmatching_vae.pth, dav.pth, qwen_7B/) is ~28 GB of the
# same weights in another arrangement; do not fetch it.
MINIMAX_DIFFUSERS_COMPONENTS = (
    "transformer",
    "vocoder",
    "condition_encoder",
    "scheduler",
)

# The autoregressive stage. Separated because it is 17 GB and is needed
# only when CAPTURING a composition, never inside a tick.
MINIMAX_AR_COMPONENTS = ("language_model", "tokenizer")


def models_root() -> Path:
    override = os.environ.get("ACESTEP_MODELS_DIR")
    if override:
        return Path(override)
    return Path.home() / ".daydream-scope" / "models" / "demon"


def minimax_root() -> Path:
    return models_root() / "minimax"


def minimax_checkpoint_dir() -> Path:
    """Where a local (non-HF-cache) install lives."""
    return minimax_root() / "checkpoints" / "MiniMax-Music3"


def minimax_capture_dir() -> Path:
    """Captured AR conditioning bundles.

    A capture is the reusable artifact of a composition: running the
    8.58B LM once and keeping its fused per-frame hidden states. Cheap
    to store, expensive to recompute, so they get a home on disk.
    """
    return minimax_root() / "captures"


def resolve_model_dir(explicit: str | os.PathLike | None = None) -> Path:
    """Locate the diffusers-layout checkpoint.

    Order: an explicit path, then ``DEMON_MINIMAX_DIR``, then the
    in-tree models directory, then the HF cache. Raises with the exact
    fetch command rather than a bare FileNotFoundError.
    """
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    env = os.environ.get("DEMON_MINIMAX_DIR")
    if env:
        candidates.append(Path(env))
    candidates.append(minimax_checkpoint_dir())

    for cand in candidates:
        if (cand / "transformer").is_dir():
            return cand

    try:
        from huggingface_hub import snapshot_download

        cached = snapshot_download(
            MINIMAX_HF_REPO,
            allow_patterns=[f"{c}/*" for c in MINIMAX_DIFFUSERS_COMPONENTS]
            + [f"{c}/*" for c in MINIMAX_AR_COMPONENTS]
            + ["*.json", "LICENSE"],
            local_files_only=True,
        )
        return Path(cached)
    except Exception:
        pass

    raise FileNotFoundError(
        "MiniMax-Music3 checkpoint not found. Fetch it with:\n"
        f"  huggingface-cli download {MINIMAX_HF_REPO} "
        f"--include {' '.join(c + '/*' for c in MINIMAX_DIFFUSERS_COMPONENTS)} "
        f"{' '.join(c + '/*' for c in MINIMAX_AR_COMPONENTS)} '*.json'\n"
        f"or set DEMON_MINIMAX_DIR to an existing checkout. Searched: "
        + ", ".join(str(c) for c in candidates)
    )


def minimax_checkpoint_status(
    explicit: str | os.PathLike | None = None,
) -> Tuple[bool, str]:
    """``(ok, message)`` for a boot preflight. Never raises, never
    downloads, never imports torch."""
    try:
        root = resolve_model_dir(explicit)
    except FileNotFoundError as exc:
        return False, str(exc)

    missing = [
        c for c in MINIMAX_DIFFUSERS_COMPONENTS if not (root / c).is_dir()
    ]
    if missing:
        return False, (
            f"MiniMax-Music3 at {root} is missing: {', '.join(missing)}. "
            "Re-run the download; the renderer cannot start without them."
        )
    ar_missing = [c for c in MINIMAX_AR_COMPONENTS if not (root / c).is_dir()]
    if ar_missing:
        return True, (
            f"MiniMax-Music3 renderer ready at {root}, but the "
            f"autoregressive stage is absent ({', '.join(ar_missing)}). "
            "Streaming from a saved conditioning capture will work; "
            "set_prompt will not."
        )
    return True, f"MiniMax-Music3 ready at {root}"
