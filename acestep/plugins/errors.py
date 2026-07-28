"""Plugin failure taxonomy.

Every one of these is fatal for an EXPLICITLY selected plugin. That is
the central safety property of the plugin system: a model extension the
operator asked for is part of the model, so dropping it silently would
produce output that is plausible and wrong. Auto-discovered plugins that
nobody selected may be isolated and reported instead.
"""

from __future__ import annotations


class PluginError(Exception):
    """Base for every plugin-system failure."""


class PluginLoadError(PluginError):
    """A plugin could not be imported or its entry callable failed."""


class PluginIncompatibleError(PluginError):
    """A plugin declares an API version this DEMON build does not serve."""


class PluginRegistrationError(PluginError):
    """A plugin registered a capability that failed validation."""


class PluginNotFoundError(PluginError):
    """An explicitly selected plugin or capability was not discovered."""


class ExtensionConfigError(PluginError):
    """Operator configuration did not satisfy an extension's schema."""
