"""Plugin manifest, registry, and capability validation.

The registry is where a plugin's declarations become part of DEMON, so it
is where they are checked. Everything validated here fails at BOOT, for
the whole process, rather than per connection — a control that shadows a
core knob or escapes coercion is a silent-wrongness bug, and the cheapest
place to make it loud is registration.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping

from acestep.plugins.errors import PluginRegistrationError
from acestep.plugins.model_extensions import (
    EXTENSION_KNOB_TYPES,
    ConfigField,
    ModelExtensionSpec,
)

#: Shape version of the plugin API. A plugin declares the version it was
#: built against; DEMON serves exactly one.
PLUGIN_API_VERSION = 1

_PLUGIN_ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_CAPABILITY_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,47}$")


def _knob_name_pattern(plugin_id: str) -> re.Pattern:
    return re.compile(rf"^plugin_{re.escape(plugin_id)}_[a-z0-9_]+$")


@dataclass(frozen=True)
class PluginManifest:
    """Identity and compatibility declaration for one plugin."""

    id: str
    version: str
    plugin_api: int
    requires_demon: str | None = None
    description: str = ""
    source: str = "entry_point"


@dataclass
class LoadedPlugin:
    """A plugin whose entry callable ran successfully."""

    manifest: PluginManifest
    model_extensions: dict = field(default_factory=dict)

    @property
    def capability_ids(self) -> list[str]:
        return sorted(self.model_extensions)


class PluginRegistry:
    """What a plugin's ``register()`` callable is handed.

    Scoped to one plugin: every capability it adds is namespaced under
    that plugin's id automatically, and it cannot see or disturb another
    plugin's registrations. ``seen`` catches a plugin registering the
    same capability name twice — the reachable duplicate, since two
    different plugins cannot collide once the id is qualified, and
    ``discover_plugins`` already rejects a duplicate plugin id.
    """

    #: Exposed so a plugin can assert compatibility from inside register().
    api_version = PLUGIN_API_VERSION

    def __init__(self, manifest: PluginManifest, seen: set[str] | None = None):
        self.manifest = manifest
        self._seen = seen if seen is not None else set()
        self._model_extensions: dict[str, ModelExtensionSpec] = {}

    @property
    def plugin_id(self) -> str:
        return self.manifest.id

    @property
    def model_extensions(self) -> dict[str, ModelExtensionSpec]:
        return dict(self._model_extensions)

    def qualified_id(self, capability_name: str) -> str:
        return f"{self.manifest.id}.{capability_name}"

    def add_model_extension(self, spec: ModelExtensionSpec) -> str:
        """Validate and register a model extension. Returns its qualified id."""
        plugin_id = self.manifest.id
        self._check_capability_name(spec.name)
        qualified = self.qualified_id(spec.name)

        if qualified in self._seen:
            raise PluginRegistrationError(
                f"duplicate capability id {qualified!r}: a capability id must "
                "resolve to exactly one provider"
            )
        if not callable(spec.create):
            raise PluginRegistrationError(
                f"{qualified}: create must be callable"
            )
        self._check_family(qualified, spec.family)
        self._check_config_schema(qualified, spec.config_schema)
        self._check_knob_specs(qualified, plugin_id, spec.knob_specs)

        self._model_extensions[spec.name] = spec
        self._seen.add(qualified)
        return qualified

    # ---- validation ------------------------------------------------------

    def _check_capability_name(self, name: str) -> None:
        if not isinstance(name, str) or not _CAPABILITY_NAME_RE.match(name):
            raise PluginRegistrationError(
                f"invalid capability name {name!r}: expected "
                f"{_CAPABILITY_NAME_RE.pattern}"
            )

    def _check_family(self, qualified: str, family: str) -> None:
        from acestep.streaming.families import FAMILIES

        if family not in FAMILIES:
            raise PluginRegistrationError(
                f"{qualified}: unknown backend family {family!r} "
                f"(known: {sorted(FAMILIES)})"
            )

    def _check_config_schema(
        self, qualified: str, schema: Mapping[str, ConfigField],
    ) -> None:
        for key, spec in (schema or {}).items():
            if not isinstance(key, str) or not key:
                raise PluginRegistrationError(
                    f"{qualified}: config field names must be non-empty strings"
                )
            if not isinstance(spec, ConfigField):
                raise PluginRegistrationError(
                    f"{qualified}: config field {key!r} must be a ConfigField, "
                    f"got {type(spec).__name__}"
                )
            # An optional field's default bypasses _coerce_config_value
            # entirely, so a default of the wrong type would reach the
            # extension unconverted — the one way a declared schema can
            # still hand over a value it does not describe.
            if not spec.required and spec.default is not None:
                try:
                    _coerce_config_value(qualified, key, spec, spec.default)
                except Exception as exc:
                    raise PluginRegistrationError(
                        f"{qualified}: config field {key!r} declares type "
                        f"{spec.type!r} but its default {spec.default!r} is "
                        f"not a valid {spec.type} ({exc})"
                    ) from exc

    def _check_knob_specs(
        self, qualified: str, plugin_id: str, specs,
    ) -> None:
        pattern = _knob_name_pattern(plugin_id)
        seen: set[str] = set()

        for spec in specs or ():
            name = getattr(spec, "name", None)
            if not isinstance(name, str):
                raise PluginRegistrationError(
                    f"{qualified}: knob specs must be KnobSpec instances"
                )
            if name in seen:
                raise PluginRegistrationError(
                    f"{qualified}: duplicate knob {name!r}"
                )
            seen.add(name)

            if not pattern.match(name):
                # This is the whole shadowing defense, and it is
                # deliberately the only one: no core or family knob is
                # named plugin_*, so a name matching this pattern cannot
                # collide with one. A blocklist of existing core names
                # would be strictly weaker — it goes stale as knobs are
                # added, and it cannot enumerate the dynamically named
                # ones (lora_str_<id>, man_alpha_<N>) at all.
                raise PluginRegistrationError(
                    f"{qualified}: knob {name!r} must be namespaced as "
                    f"plugin_{plugin_id}_<control>. The namespace is what "
                    "keeps plugin controls from shadowing a core knob and "
                    "from colliding across plugins; a shadowed control is "
                    "silently swallowed by whichever spec map wins and "
                    "stops being coerced."
                )

            knob_type = getattr(spec, "type", "float")
            if knob_type not in EXTENSION_KNOB_TYPES:
                raise PluginRegistrationError(
                    f"{qualified}: knob {name!r} has type {knob_type!r}; "
                    f"extension knobs must be one of {EXTENSION_KNOB_TYPES} "
                    "(the browser panel has no generic binding for the rest)"
                )
            self._check_knob_bounds(qualified, spec)

    def _check_knob_bounds(self, qualified: str, spec) -> None:
        name = spec.name
        low = getattr(spec, "min_val", None)
        # Absent min floors at 0 — the same convention coerce_knob_values
        # applies, so this check and the runtime clamp agree.
        low = 0.0 if low is None else float(low)
        try:
            high = float(getattr(spec, "max_val", 1.0))
        except (TypeError, ValueError) as exc:
            raise PluginRegistrationError(
                f"{qualified}: knob {name!r} has a non-numeric max_val "
                f"{getattr(spec, 'max_val', None)!r}; an extension knob must "
                "declare a finite upper bound to be clampable"
            ) from exc
        if high < low:
            raise PluginRegistrationError(
                f"{qualified}: knob {name!r} has max_val {high} below "
                f"min_val {low}"
            )
        try:
            default = float(getattr(spec, "default", 0.0))
        except (TypeError, ValueError) as exc:
            raise PluginRegistrationError(
                f"{qualified}: knob {name!r} default is not numeric"
            ) from exc
        if not (low <= default <= high):
            # Coercion clamps to bounds, so an out-of-range default would
            # silently become a different value than the one declared.
            raise PluginRegistrationError(
                f"{qualified}: knob {name!r} default {default} is outside "
                f"its own bounds [{low}, {high}]"
            )


def validate_extension_config(
    qualified_id: str,
    schema: Mapping[str, ConfigField],
    raw: Mapping[str, Any] | None,
) -> dict:
    """Coerce operator configuration against an extension's schema.

    Unknown keys are rejected rather than ignored: a typo in an operator's
    config file must not silently mean "default".
    """
    from acestep.plugins.errors import ExtensionConfigError

    schema = schema or {}
    raw = dict(raw or {})

    unknown = sorted(set(raw) - set(schema))
    if unknown:
        raise ExtensionConfigError(
            f"{qualified_id}: unknown configuration key(s): "
            f"{', '.join(unknown)}"
        )

    out: dict[str, Any] = {}
    for key, field_spec in schema.items():
        if key not in raw:
            if field_spec.required:
                raise ExtensionConfigError(
                    f"{qualified_id}: missing required configuration key "
                    f"{key!r} ({field_spec.description or field_spec.type})"
                )
            out[key] = field_spec.default
            continue
        out[key] = _coerce_config_value(qualified_id, key, field_spec, raw[key])
    return out


def _coerce_config_value(qualified_id: str, key: str, field_spec, value):
    from acestep.plugins.errors import ExtensionConfigError

    kind = field_spec.type
    try:
        if kind == "bool":
            if not isinstance(value, bool):
                raise TypeError("expected a JSON boolean")
            return value
        if kind == "int":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("expected a number")
            if isinstance(value, float) and not value.is_integer():
                raise TypeError("expected an integer")
            return int(value)
        if kind == "float":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("expected a number")
            return float(value)
        if kind in ("str", "path"):
            if not isinstance(value, str):
                raise TypeError("expected a string")
            return value
    except TypeError as exc:
        raise ExtensionConfigError(
            f"{qualified_id}: configuration key {key!r} has the wrong type "
            f"({exc}; got {type(value).__name__})"
        ) from exc
    raise ExtensionConfigError(
        f"{qualified_id}: configuration key {key!r} has unsupported type "
        f"{kind!r}"
    )
