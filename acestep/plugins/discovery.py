"""Plugin discovery through installed entry points.

One source, on purpose. A private extension is installed into DEMON's
environment (editable or from a private index), which means the standard
``demon.plugins`` entry-point group already answers "what is installed"
without DEMON inventing a search path, mutating ``sys.path``, resolving
root precedence, or creating directories as a side effect of looking.

A distribution advertises itself as::

    [project.entry-points."demon.plugins"]
    my_extension = "my_extension.plugin:register"

The referenced module must define, at module scope:

* ``PLUGIN_API`` — the integer plugin API version it was built against.
* ``register(registry)`` — called with a :class:`PluginRegistry` scoped to
  this plugin. Optionally ``REQUIRES_DEMON`` and ``DESCRIPTION``.

Compatibility is checked after importing the module but BEFORE calling
``register``, so an incompatible plugin never gets to run registration
code against an API it does not understand.

Discovery order is deterministic (sorted by entry-point name) and a
capability id resolves to exactly one provider; a collision is an error,
not a last-writer-wins race.
"""

from __future__ import annotations

from typing import Iterable

from acestep.engine.obs import logger
from acestep.plugins.api import LoadedPlugin, PluginManifest, PluginRegistry
from acestep.plugins.compatibility import check_plugin_api, note_requires_demon
from acestep.plugins.errors import PluginError, PluginLoadError

ENTRY_POINT_GROUP = "demon.plugins"


def _iter_entry_points(group: str):
    from importlib.metadata import entry_points

    try:
        found = entry_points(group=group)
    except TypeError:  # pragma: no cover - Python < 3.10 shape
        found = entry_points().get(group, [])
    return sorted(found, key=lambda ep: ep.name)


def _manifest_for(entry_point, module) -> PluginManifest:
    plugin_id = entry_point.name
    declared = getattr(module, "PLUGIN_API", None)
    if declared is None:
        raise PluginLoadError(
            f"plugin {plugin_id!r} ({entry_point.value}) does not define "
            "PLUGIN_API at module scope"
        )
    check_plugin_api(declared, plugin_id)

    version = "0+unknown"
    dist = getattr(entry_point, "dist", None)
    if dist is not None and getattr(dist, "version", None):
        version = str(dist.version)

    return PluginManifest(
        id=plugin_id,
        version=version,
        plugin_api=int(declared),
        requires_demon=getattr(module, "REQUIRES_DEMON", None),
        description=str(getattr(module, "DESCRIPTION", "") or ""),
        source="entry_point",
    )


def _import_entry_module(entry_point):
    """Import the entry point's MODULE, not its target attribute.

    ``EntryPoint.load()`` resolves all the way to the callable, which
    would give us no way to read the module-scope compatibility
    declaration before deciding whether to run the plugin's code.
    """
    import importlib

    return importlib.import_module(entry_point.module)


def _load_one(entry_point, seen: set[str]) -> LoadedPlugin:
    plugin_id = entry_point.name
    try:
        module = _import_entry_module(entry_point)
    except Exception as exc:  # noqa: BLE001 - any import failure is a load failure
        raise PluginLoadError(
            f"plugin {plugin_id!r} ({entry_point.value}) failed to import: {exc}"
        ) from exc

    # Compatibility is checked against the imported module BEFORE the
    # entry callable is resolved or run, so an incompatible plugin never
    # executes registration code against an API it does not understand.
    manifest = _manifest_for(entry_point, module)
    note_requires_demon(manifest.requires_demon, plugin_id)

    register = getattr(module, entry_point.attr or "register", None)
    if not callable(register):
        raise PluginLoadError(
            f"plugin {plugin_id!r} ({entry_point.value}) has no callable "
            f"{entry_point.attr or 'register'}(registry)"
        )

    registry = PluginRegistry(manifest, seen=seen)
    try:
        register(registry)
    except PluginError:
        raise
    except Exception as exc:  # noqa: BLE001 - plugin code is arbitrary
        raise PluginLoadError(
            f"plugin {plugin_id!r} register() failed: {exc}"
        ) from exc

    plugin = LoadedPlugin(
        manifest=manifest, model_extensions=registry.model_extensions,
    )
    logger.info(
        "plugin_loaded plugin_id={} version={} plugin_api={} capabilities={}",
        manifest.id, manifest.version, manifest.plugin_api,
        plugin.capability_ids,
    )
    return plugin


def discover_plugins(
    *, required: Iterable[str] = (), group: str = ENTRY_POINT_GROUP,
) -> dict[str, LoadedPlugin]:
    """Load every installed plugin. Returns ``{plugin_id: LoadedPlugin}``.

    A plugin named in ``required`` (because the operator selected it, or a
    capability of it) fails the process on any error. A plugin nobody asked
    for is isolated: logged and skipped, so one broken optional plugin
    cannot take the server down.

    This asymmetry is the whole safety story. Silently dropping a model
    extension the operator explicitly selected would produce output that
    looks fine and is not the model they asked for.
    """
    required = set(required)
    plugins: dict[str, LoadedPlugin] = {}
    seen: set[str] = set()

    for entry_point in _iter_entry_points(group):
        plugin_id = entry_point.name
        if plugin_id in plugins:
            raise PluginLoadError(
                f"duplicate plugin id {plugin_id!r}: a plugin id must resolve "
                "to exactly one provider"
            )
        try:
            plugins[plugin_id] = _load_one(entry_point, seen)
        except PluginError as exc:
            if plugin_id in required:
                raise
            logger.warning(
                "plugin_skipped plugin_id={} reason={}", plugin_id, exc,
            )

    missing = sorted(required - set(plugins))
    if missing:
        from acestep.plugins.errors import PluginNotFoundError

        raise PluginNotFoundError(
            f"selected plugin(s) not installed: {', '.join(missing)}. "
            f"Install the distribution providing the {group!r} entry point."
        )
    return plugins
