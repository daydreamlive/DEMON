"""Tier-2 SA3 model internals: the extension-reachable surface.

DEMON's plugin API (:mod:`acestep.plugins`) is Tier 1 — stable, versioned,
and sufficient for registration and lifecycle. It is NOT sufficient for a
model extension that has to participate in the transformer itself: such an
extension must construct modules against the vendored ``stable_audio_3``
architecture, read the loaded model's object graph, and attach itself to
the trunk. There is no way to abstract that away, because the shapes it
depends on belong to a vendored third-party package rather than to DEMON.

This module is the response to that reality. It does not remove the
coupling; it gives the coupling ONE address. Extensions traverse the
loaded model through the accessors here instead of hard-coding attribute
chains like ``sam.model.model.model.transformer.layers``, so a vendored
SA3 bump breaks one public, tested file with a clear message rather than
failing somewhere inside an out-of-tree package — where the failure mode
is not an ImportError but a silently wrong model.

Contract for extension authors:

* Import from here, never from ``stable_audio_3`` or the private helpers
  directly.
* Assert :data:`API_VERSION` at install time and fail closed on mismatch.
* Treat :data:`VENDOR_SHA` as the architecture revision you were built
  against; a change means re-validating, not just re-pinning.

Bump :data:`API_VERSION` whenever anything exported here changes shape.
The structural canary in ``tests/unit/test_sa3_internals.py`` fails in
public CI when the vendored layout drifts out from under these accessors.
"""

from __future__ import annotations

from typing import Any, Sequence

from acestep.engine.sa3_helpers import (
    SA3_VENDOR_SHA,
    ensure_sa3_paths,
    skip_param_init,
)

#: Shape version for everything this module exports. Extensions assert it.
API_VERSION = 1

#: The vendored stable_audio_3 revision these accessors were written for.
VENDOR_SHA = SA3_VENDOR_SHA

#: Attribute chains this module encapsulates, for diagnostics and for the
#: canary test. Documented rather than repeated at each call site.
TRUNK_WRAPPER_PATH = "sam.model.model"
TRUNK_MODULE_PATH = "sam.model.model.model"
TRUNK_BLOCKS_PATH = "sam.model.model.model.transformer.layers"

__all__ = [
    "API_VERSION",
    "VENDOR_SHA",
    "TRUNK_WRAPPER_PATH",
    "TRUNK_MODULE_PATH",
    "TRUNK_BLOCKS_PATH",
    "LayoutError",
    "ensure_sa3_paths",
    "skip_param_init",
    "dit_class",
    "dit_architecture_config",
    "trunk_wrapper",
    "trunk_module",
    "trunk_blocks",
    "replace_trunk",
    "check_layout",
]


class LayoutError(RuntimeError):
    """The loaded SA3 model does not match the expected vendored layout.

    Raised instead of an AttributeError from halfway down an attribute
    chain, so the message names the accessor, the expected path, and the
    vendor revision the accessor was written for.
    """


def _walk(root: Any, path: str, accessor: str):
    """Traverse a dotted path, reporting the exact hop that failed."""
    node = root
    walked = path.split(".")[0]
    for attr in path.split(".")[1:]:
        if not hasattr(node, attr):
            raise LayoutError(
                f"{accessor}: expected {path!r} on the loaded SA3 model, but "
                f"{walked!r} has no attribute {attr!r}. The vendored "
                f"stable_audio_3 layout changed (accessors written for "
                f"{VENDOR_SHA}); update acestep.engine.sa3_internals."
            )
        node = getattr(node, attr)
        walked = f"{walked}.{attr}"
    return node


def dit_class():
    """The vendored diffusion-transformer class.

    Extensions that build a parallel/derived branch subclass this so their
    module matches the trunk architecture exactly.
    """
    ensure_sa3_paths()
    from stable_audio_3.models.dit import DiffusionTransformer

    return DiffusionTransformer


def dit_architecture_config(model_config: dict) -> tuple[str, dict]:
    """``(diffusion_objective, dit_kwargs)`` from a parsed model config.

    The vendored ``model_config.json`` nests the transformer's constructor
    arguments; this normalizes that lookup so extensions do not re-derive
    the config layout. The returned kwargs are a copy and safe to mutate
    (e.g. to build a shallower branch).
    """
    try:
        diffusion = model_config["model"]["diffusion"]
        dit_kwargs = dict(diffusion["config"])
    except (KeyError, TypeError) as exc:
        raise LayoutError(
            "dit_architecture_config: model_config is missing "
            "model.diffusion.config; not an SA3 model config?"
        ) from exc
    objective = str(diffusion.get("diffusion_objective", "v"))
    return objective, dit_kwargs


def trunk_wrapper(sam):
    """The per-step callable wrapping the DiT (``dit(x_bct, t, **cond)``).

    This is the object an extension REPLACES to interpose itself; see
    :func:`replace_trunk`.
    """
    return _walk(sam, TRUNK_WRAPPER_PATH, "trunk_wrapper")


def trunk_module(sam):
    """The diffusion transformer itself, inside the wrapper."""
    return _walk(sam, TRUNK_MODULE_PATH, "trunk_module")


def trunk_blocks(sam) -> Sequence:
    """The trunk's ordered transformer blocks.

    Indexable and len()-able. Extensions that inject per-block residuals
    attach hooks to these; they must remove every hook they attach when
    their runtime closes, because the model object is process-cached and
    shared across sessions.
    """
    blocks = _walk(sam, TRUNK_BLOCKS_PATH, "trunk_blocks")
    try:
        len(blocks)
    except TypeError as exc:
        raise LayoutError(
            f"trunk_blocks: {TRUNK_BLOCKS_PATH} is not a sized sequence "
            f"(got {type(blocks).__name__}); accessors written for "
            f"{VENDOR_SHA}."
        ) from exc
    return blocks


def replace_trunk(sam, wrapper) -> Any:
    """Install ``wrapper`` as the per-step callable; return the old one.

    The caller owns restoration: keep the returned object and pass it back
    through this function on close. Nothing here is reference-counted, and
    the loaded model is shared, so an extension that fails to restore
    leaves every later session running its graph.
    """
    previous = trunk_wrapper(sam)
    sam.model.model = wrapper
    return previous


def check_layout(sam) -> None:
    """Validate the loaded model against every accessor above.

    Extensions call this once at install, before building anything, so a
    vendor drift surfaces as a clear LayoutError at boot rather than as a
    wrong-shaped tensor mid-generation.
    """
    trunk_wrapper(sam)
    trunk_module(sam)
    blocks = trunk_blocks(sam)
    if len(blocks) < 1:
        raise LayoutError(
            f"check_layout: {TRUNK_BLOCKS_PATH} is empty; the loaded model "
            "has no transformer blocks to extend."
        )
