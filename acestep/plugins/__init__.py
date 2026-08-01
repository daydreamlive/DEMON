"""DEMON plugin API (Tier 1).

The supported surface for out-of-tree extensions. Everything exported
here is versioned by :data:`PLUGIN_API_VERSION` and changes only with a
version bump. Anything else under ``acestep.*`` is internal and may move
without notice.

Import tiers, for extension authors:

* **Tier 1** — this module. Registration, configuration, lifecycle.
  Sufficient for any extension that does not have to participate in the
  model's internal computation.
* **Tier 2** — ``acestep.engine.sa3_internals``. The documented, tested
  subset of vendored SA3 model internals an extension may traverse when
  it genuinely must build modules against the trunk architecture. Assert
  its ``API_VERSION`` at install and fail closed on mismatch.
* **Tier 0** — everything else. Importing it is a bug in your plugin.

A plugin is an installed distribution advertising a ``demon.plugins``
entry point::

    [project.entry-points."demon.plugins"]
    my_extension = "my_extension.plugin:register"

whose module defines ``PLUGIN_API`` and ``register(registry)``. See
:mod:`acestep.plugins.discovery`.

Plugins are trusted, in-process Python. This is an extensibility
boundary, not a security sandbox: plugins load only from operator-
controlled startup configuration, never at the request of a client.
"""

from __future__ import annotations

from acestep.plugins.api import (
    PLUGIN_API_VERSION,
    LoadedPlugin,
    PluginManifest,
    PluginRegistry,
    validate_extension_config,
)
from acestep.plugins.errors import (
    ExtensionConfigError,
    PluginError,
    PluginIncompatibleError,
    PluginLoadError,
    PluginNotFoundError,
    PluginRegistrationError,
)
from acestep.plugins.model_extensions import (
    CONFIG_FIELD_TYPES,
    EXTENSION_KNOB_TYPES,
    BackendVerdict,
    ConfigField,
    ExtensionContext,
    ModelDescriptor,
    ModelExtension,
    ModelExtensionRuntime,
    ModelExtensionSpec,
    Rejection,
    SourceView,
)
# Re-exported, not re-declared: extension controls ARE core knobs, and
# routing them through a parallel type would be the second schema this
# design exists to avoid. It lives here so declaring a control never
# requires an import from outside the Tier-1 surface.
from acestep.streaming.knobs import KnobSpec

__all__ = [
    "PLUGIN_API_VERSION",
    "CONFIG_FIELD_TYPES",
    "EXTENSION_KNOB_TYPES",
    "BackendVerdict",
    "ConfigField",
    "ExtensionContext",
    "KnobSpec",
    "LoadedPlugin",
    "ModelDescriptor",
    "ModelExtension",
    "ModelExtensionRuntime",
    "ModelExtensionSpec",
    "PluginManifest",
    "PluginRegistry",
    "Rejection",
    "SourceView",
    "PluginError",
    "PluginIncompatibleError",
    "PluginLoadError",
    "PluginNotFoundError",
    "PluginRegistrationError",
    "ExtensionConfigError",
    "validate_extension_config",
]
