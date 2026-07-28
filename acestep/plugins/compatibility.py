"""Plugin/DEMON compatibility checks.

Two axes, deliberately separated:

* **Plugin API version** — the real gate. A plugin declares the API shape
  it was built against and DEMON serves exactly one; a mismatch is fatal.
* **``requires_demon``** — advisory only, for now, and honestly labelled
  as such. DEMON's distribution version tracks the ACE-Step *model*
  generation rather than an API contract, so enforcing a specifier
  against it would look like a guarantee while promising nothing. It is
  recorded in logs and diagnostics until the two are decoupled.

Extensions that reach into model internals have a third axis
(``acestep.engine.sa3_internals.API_VERSION``) which they assert
themselves at install time.
"""

from __future__ import annotations

from acestep.engine.obs import logger
from acestep.plugins.api import PLUGIN_API_VERSION
from acestep.plugins.errors import PluginIncompatibleError


def demon_version() -> str:
    """The installed distribution version, or ``"0+unknown"``.

    Not importable as ``acestep.__version__``; read from distribution
    metadata so a source checkout without an install still boots.
    """
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover - stdlib since 3.8
        return "0+unknown"
    try:
        return version("demon")
    except PackageNotFoundError:
        return "0+unknown"


def check_plugin_api(declared, plugin_id: str) -> int:
    """Validate a plugin's declared API version. Fatal on mismatch."""
    if not isinstance(declared, int) or isinstance(declared, bool):
        raise PluginIncompatibleError(
            f"plugin {plugin_id!r} declares PLUGIN_API={declared!r}; "
            "expected an integer"
        )
    if declared != PLUGIN_API_VERSION:
        raise PluginIncompatibleError(
            f"plugin {plugin_id!r} was built against plugin API {declared} "
            f"but this DEMON serves plugin API {PLUGIN_API_VERSION}. "
            "Rebuild or reinstall the plugin against this DEMON version."
        )
    return declared


def note_requires_demon(requires: str | None, plugin_id: str) -> None:
    """Record a plugin's DEMON constraint. Advisory — see module docstring."""
    if not requires:
        return
    logger.info(
        "plugin_requires_demon plugin_id={} requires={} running={} "
        "enforced=false",
        plugin_id, requires, demon_version(),
    )
