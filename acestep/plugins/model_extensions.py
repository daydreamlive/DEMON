"""Generic model-extension contracts.

A *model extension* is a persistent, startup-selected participant in a
loaded model: it may inspect the base checkpoint before it loads, wrap or
augment the model after it loads, decorate conditioning, contribute
controls, and report telemetry. It is not a job or a recipe — it is part
of the model for the lifetime of the process.

Nothing here names any concrete extension. The core treats an extension's
configuration, controls, and metrics as opaque, and never branches on its
type.

Lifecycle, in order:

1. ``spec.create(ExtensionContext)`` -> a :class:`ModelExtension`.
   Configuration is already validated against ``spec.config_schema``.
2. ``extension.validate_base_model(descriptor)`` -> ``None`` to accept, or
   a :class:`Rejection` to abort boot. The extension may VETO a base model
   but never substitutes one: which weights load is the operator's call.
3. ``extension.config_fingerprint()`` -> a stable, non-secret string that
   is folded into the model cache key, so two differently configured
   graphs can never share a cached context.
4. ``extension.install(model, model_config)`` -> a
   :class:`ModelExtensionRuntime`. The core loads the model; the extension
   only ever receives one that loaded successfully.
5. Per-session: ``supports_dit_backend`` / ``supports_codec_backend``,
   ``decorate_conditioning``, ``apply_controls``, ``status`` / ``metrics``.
6. ``runtime.close()`` at teardown — idempotent, and responsible for
   removing anything the runtime attached to the shared model.

Installation is wrapped by the core: if ``install`` raises after it has
already mutated the model, ``close()`` is still called on whatever was
produced, and boot aborts. A half-hooked model must never serve traffic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

#: Config field types an extension may declare. Deliberately small: these
#: values come from an operator-authored JSON file, and every one of them
#: has an unambiguous coercion.
#:
#: ``path`` is a coercion type, not a filesystem assertion: it validates
#: as a string and nothing is stat'd. Only the extension knows whether a
#: configured path is one it READS (must already exist) or one it WRITES
#: (must not have to), so existence, readability, and format checks
#: belong in the extension's ``create`` callable — which runs at
#: selection, before any model work at all. See :class:`ModelExtension`.
CONFIG_FIELD_TYPES = ("str", "int", "float", "bool", "path")

#: Knob value types an extension may contribute. Restricted to numeric on
#: purpose: the browser control panel renders float/int knobs generically
#: from manifest bounds, but binds enum/bool knobs by name and renders
#: unknown ones disabled. An extension enum knob would therefore appear in
#: the manifest and be unusable. Revisit when the panel grows a generic
#: binding.
EXTENSION_KNOB_TYPES = ("float", "int")


@dataclass(frozen=True)
class ConfigField:
    """One operator-supplied configuration value."""

    type: str = "str"
    required: bool = True
    default: Any = None
    description: str = ""

    def __post_init__(self):
        if self.type not in CONFIG_FIELD_TYPES:
            raise ValueError(
                f"config field type {self.type!r} is not one of "
                f"{CONFIG_FIELD_TYPES}"
            )


@dataclass(frozen=True)
class Rejection:
    """An extension's refusal to run against a base model."""

    reason: str


@dataclass(frozen=True)
class BackendVerdict:
    """Whether an extension can run under a given acceleration backend.

    ``reason`` is operator-facing and must not leak private architecture
    detail; it is printed at boot when a downgrade happens.
    """

    ok: bool
    reason: str = ""


@dataclass(frozen=True)
class ModelDescriptor:
    """What the core is about to load, offered for veto and fingerprinting."""

    family: str
    model_id: str
    checkpoint_dir: str
    model_config: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExtensionContext:
    """What an extension receives at construction time."""

    qualified_id: str
    plugin_id: str
    plugin_version: str
    config: Mapping[str, Any]


class ModelExtensionRuntime:
    """An installed extension. Subclass and override what you need.

    Every method has a safe default, so an extension implements only the
    hooks it actually uses. The core calls these through the module-level
    helpers below, which tolerate a runtime that is not a subclass.
    """

    def supports_dit_backend(self, backend: str) -> BackendVerdict:
        return BackendVerdict(True)

    def supports_codec_backend(self, backend: str) -> BackendVerdict:
        return BackendVerdict(True)

    def decorate_conditioning(
        self, bundle: Mapping[str, Any], source_latent_bct: Any = None,
    ) -> Mapping[str, Any]:
        """Return conditioning for the extension's own use.

        MUST return a fresh mapping rather than mutating ``bundle``: the
        bundle passed in may already be referenced by in-flight requests,
        which have to keep seeing the conditioning they were submitted
        with. The core always hands over a mapping it just built, and
        never reads back through the one it gave you.

        MUST be idempotent in its own keys. On a source swap the core
        re-decorates the CURRENT conditioning, so ``bundle`` will already
        carry the keys you added last time. Assign your keys; never append
        to or accumulate on a value you find already present, or a second
        swap will compound the first.

        Called on the COMMAND thread (a prompt or source swap), not the
        streaming runner. See ``apply_controls`` for what that means if
        the two share state.
        """
        return bundle

    def apply_controls(self, values: Mapping[str, float]) -> None:
        """Apply this extension's namespaced knob values.

        Called once per tick on the STREAMING RUNNER, unlocked — the
        same thread that then runs your model code, so a value written
        here is read by that tick with no synchronization needed. Do not
        block.

        ``decorate_conditioning``, by contrast, is called on the COMMAND
        thread. If this method and that one touch the same state, that
        is the one race the core does not serialize for you, and you own
        it. Keeping controls to plain scalar rebinds avoids the question
        entirely.
        """

    def status(self) -> Mapping[str, Any]:
        """Operator-visible state. Must not contain paths or secrets."""
        return {}

    def metrics(self) -> Mapping[str, Any]:
        """Operator-only telemetry.

        Not called on the per-generation path — the core polls this on
        demand or on a slow cadence, so it may be more expensive than a
        hot-path accessor.
        """
        return {}

    def close(self) -> None:
        """Release everything, idempotently.

        The loaded model is process-cached and shared across sessions, so
        anything attached to it (wrappers, hooks, buffers) must be removed
        here. Being called twice, or after a failed install, is normal.
        """


class ModelExtension:
    """A configured, not-yet-installed extension. Subclass to implement.

    ``spec.create(context)`` runs this class's constructor at SELECTION
    time — before the model cache key is computed and long before any
    weights load. That is the right place to validate configuration the
    core cannot: that a configured input path exists and is readable,
    that a numeric value is in a range only this extension knows. Raise
    :class:`~acestep.plugins.errors.ExtensionConfigError` there and the
    operator learns immediately, rather than after a multi-second load.
    """

    def validate_base_model(self, descriptor: ModelDescriptor):
        """``None`` to accept, or a :class:`Rejection` to abort boot.

        For facts about the BASE MODEL only — its objective, geometry,
        or size. Configuration this extension owns is validated in the
        constructor, which runs earlier.
        """
        return None

    def config_fingerprint(self) -> str:
        """Stable, non-secret identity of this extension's configuration.

        Folded into the model cache key. Include the CONTENT identity of
        any file you load (size and mtime, or a hash) — not just its path.
        A file that a producer rewrites in place has a stable path and
        different contents, and a path-only fingerprint would serve the
        previously loaded model forever.
        """
        return ""

    def install(self, model: Any, model_config: Mapping[str, Any]) -> ModelExtensionRuntime:
        """Attach to the freshly loaded model and return the runtime."""
        return ModelExtensionRuntime()

    def close(self) -> None:
        """Unwind an ``install()`` that raised part-way. Idempotent.

        ``install`` can fail after it has already mutated the shared,
        process-cached model. When it does, the core has no runtime to
        close, so it calls this instead — the extension is the only
        thing that knows what it had attached. A no-op is correct for an
        extension whose ``install`` cannot fail dirty.
        """


@dataclass(frozen=True)
class ModelExtensionSpec:
    """Registration-time declaration of a model extension.

    ``knob_specs`` are plain :class:`~acestep.streaming.knobs.KnobSpec`
    values, so extension controls flow through exactly the same catalog
    projection and coercion as core knobs. Their names must be namespaced
    (``plugin_<plugin_id>_<control>``); the registry enforces that, along
    with shadowing, type and bounds checks, at registration time so a bad
    declaration fails at boot rather than per connection.
    """

    name: str
    family: str
    create: Callable[[ExtensionContext], ModelExtension]
    config_schema: Mapping[str, ConfigField] = field(default_factory=dict)
    knob_specs: tuple = ()
    description: str = ""


def decorate_conditioning(runtime, bundle, source_latent_bct=None):
    """Call the runtime's conditioning hook, normalizing a ``None`` return.

    The one contract detail a subclass can still get wrong: returning
    nothing from ``decorate_conditioning`` reads as "I made no changes",
    not as "the conditioning is now None". Every other hook has a total
    default on the base class, so the core calls those directly.
    """
    decorated = runtime.decorate_conditioning(bundle, source_latent_bct)
    return bundle if decorated is None else decorated
