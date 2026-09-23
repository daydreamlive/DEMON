"""Startup selection and installation of a model extension.

Selection is startup-only and operator-controlled. That is both a safety
property (a browser client can never name a module, checkpoint, or config
path) and a mechanical necessity: the per-session knob manifest is sent
once on the ``ready`` frame, so an extension's controls have to exist
before any session is accepted for a client to ever see them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from acestep.engine.obs import logger
from acestep.plugins.api import (
    LoadedPlugin,
    PluginManifest,
    validate_extension_config,
)
from acestep.plugins.discovery import discover_plugins
from acestep.plugins.errors import (
    ExtensionConfigError,
    PluginNotFoundError,
)
from acestep.plugins.model_extensions import (
    ExtensionContext,
    ModelDescriptor,
    ModelExtensionRuntime,
    ModelExtensionSpec,
    Rejection,
)


@dataclass
class SelectedExtension:
    """A configured extension, constructed but not yet installed."""

    qualified_id: str
    manifest: PluginManifest
    spec: ModelExtensionSpec
    extension: Any
    config: Mapping[str, Any]

    @property
    def knob_specs(self) -> tuple:
        return tuple(self.spec.knob_specs or ())

    def public_status(self) -> dict:
        """Safe-to-publish identity. Never the configuration values."""
        return {
            "id": self.qualified_id,
            "plugin": self.manifest.id,
            "version": self.manifest.version,
        }

    # ---- lifecycle -------------------------------------------------------

    def validate_base_model(self, descriptor: ModelDescriptor) -> None:
        """Give the extension a veto over the base model. Raises on refusal."""
        hook = getattr(self.extension, "validate_base_model", None)
        if hook is None:
            return
        verdict = hook(descriptor)
        if isinstance(verdict, Rejection):
            raise ExtensionConfigError(
                f"{self.qualified_id} refused the selected base model: "
                f"{verdict.reason}"
            )

    def config_fingerprint(self) -> str:
        hook = getattr(self.extension, "config_fingerprint", None)
        return str(hook() or "") if hook is not None else ""

    def cache_key(self) -> tuple:
        """Identity folded into the model cache key by the caller."""
        return (
            self.qualified_id, self.manifest.version, self.config_fingerprint(),
        )

    def install(self, model: Any, model_config: Mapping[str, Any]):
        """Install onto a freshly loaded model, unwinding a partial failure.

        If ``install`` raises after it has already mutated the model, the
        partially-built runtime still gets ``close()`` before the error
        propagates. A half-hooked model must never be cached or served:
        the operator explicitly asked for this extension, so the stock
        trunk is not an acceptable fallback.

        The runtime must subclass :class:`ModelExtensionRuntime`. That is
        checked HERE, at boot, because the alternative is discovering a
        missing hook at a tick boundary as an ``AttributeError`` in the
        streaming runner. The base class defines a total, neutral default
        for every hook, so subclassing costs an extension nothing.
        """
        runtime = None
        try:
            runtime = self.extension.install(model, model_config)
            if runtime is None:
                raise ExtensionConfigError(
                    f"{self.qualified_id}: install() returned None; it must "
                    "return a runtime"
                )
            if not isinstance(runtime, ModelExtensionRuntime):
                raise ExtensionConfigError(
                    f"{self.qualified_id}: install() returned "
                    f"{type(runtime).__name__}, which does not subclass "
                    "ModelExtensionRuntime. Subclass it so every lifecycle "
                    "hook has a defined default."
                )
            return runtime
        except Exception:
            if runtime is not None:
                _close_quietly(runtime, self.qualified_id)
            else:
                # install() may have mutated the model before raising, so
                # give the extension itself a chance to unwind.
                _close_quietly(self.extension, self.qualified_id)
            raise


def _close_quietly(target, qualified_id: str) -> None:
    try:
        target.close()
    except Exception as exc:  # noqa: BLE001 - unwinding must not mask the cause
        logger.error(
            "extension_unwind_failed id={} error={}", qualified_id, exc,
        )


def load_extension_config(path, qualified_id: str) -> dict:
    """Read the operator's local JSON config for one extension.

    The file is keyed by qualified extension id so one file can configure
    several extensions and so a config can never be applied to the wrong
    one by accident.
    """
    if path is None:
        return {}
    config_path = Path(path).expanduser().resolve()
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ExtensionConfigError(
            f"model extension config not found: {config_path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ExtensionConfigError(
            f"model extension config {config_path} is not valid JSON: {exc}"
        ) from exc

    if not isinstance(raw, dict):
        raise ExtensionConfigError(
            f"model extension config {config_path} must be a JSON object "
            f"keyed by extension id (e.g. {{{qualified_id!r}: {{...}}}})"
        )
    if qualified_id not in raw:
        raise ExtensionConfigError(
            f"model extension config {config_path} has no entry for "
            f"{qualified_id!r} (found: {', '.join(sorted(raw)) or 'nothing'})"
        )
    section = raw[qualified_id]
    if not isinstance(section, dict):
        raise ExtensionConfigError(
            f"model extension config {config_path}: entry {qualified_id!r} "
            "must be a JSON object"
        )
    return section


def split_qualified_id(qualified_id: str) -> tuple[str, str]:
    plugin_id, sep, capability = str(qualified_id).partition(".")
    if not sep or not plugin_id or not capability:
        raise PluginNotFoundError(
            f"invalid model extension id {qualified_id!r}: expected "
            "<plugin-id>.<extension-name>"
        )
    return plugin_id, capability


def select_model_extension(
    qualified_id: str | None,
    *,
    family: str,
    config_path=None,
    plugins: Mapping[str, LoadedPlugin] | None = None,
) -> SelectedExtension | None:
    """Resolve, configure, and construct the selected model extension.

    Returns ``None`` when nothing is selected — the stock path, which must
    behave exactly as if the plugin system did not exist. Every failure
    after a selection is made is fatal.
    """
    if not qualified_id:
        return None

    plugin_id, capability = split_qualified_id(qualified_id)
    if plugins is None:
        plugins = discover_plugins(required=[plugin_id])

    plugin = plugins.get(plugin_id)
    if plugin is None:
        raise PluginNotFoundError(
            f"plugin {plugin_id!r} is not installed (selected via "
            f"{qualified_id!r})"
        )
    spec = plugin.model_extensions.get(capability)
    if spec is None:
        available = ", ".join(plugin.capability_ids) or "none"
        raise PluginNotFoundError(
            f"plugin {plugin_id!r} provides no model extension "
            f"{capability!r} (available: {available})"
        )
    if spec.family != family:
        raise ExtensionConfigError(
            f"{qualified_id} extends the {spec.family!r} backend family, but "
            f"this server is running {family!r}"
        )

    raw_config = load_extension_config(config_path, qualified_id)
    config = validate_extension_config(qualified_id, spec.config_schema, raw_config)

    context = ExtensionContext(
        qualified_id=qualified_id,
        plugin_id=plugin_id,
        plugin_version=plugin.manifest.version,
        config=config,
    )
    extension = spec.create(context)
    if extension is None:
        raise ExtensionConfigError(
            f"{qualified_id}: create() returned None"
        )

    logger.info(
        "model_extension_selected id={} plugin_version={} family={} "
        "knobs={} config_keys={}",
        qualified_id, plugin.manifest.version, spec.family,
        [s.name for s in spec.knob_specs or ()], sorted(config),
    )
    return SelectedExtension(
        qualified_id=qualified_id,
        manifest=plugin.manifest,
        spec=spec,
        extension=extension,
        config=config,
    )
