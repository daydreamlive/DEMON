"""Typed init-time configuration for :class:`StreamingSession`.

Field names mirror the wire config keys (see
``demos/realtime_motion_graph_web/web/types/protocol.ts``
``SessionConfig``) so the adapter's :meth:`SessionConfig.from_dict`
parse is a simple known-key filter.

Transport-agnostic: no torch, no acestep imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields


@dataclass
class SessionConfig:
    """Session-init configuration. All fields are wire-side optional;
    sensible defaults match the demo's prior ``config.get(...)`` defaults
    so the parse is loss-free for the common case."""

    sde: bool = False
    lora: bool = False
    vae_window: float = 3.0
    crop: float = 0.0
    depth: int = 4
    steps: int = 8
    prompt: str = "instrumental music"
    # Defaulting to None lets ``StreamingSession.create`` use ``prompt``
    # when the wire didn't send a B variant. ``"" `` would force an
    # always-different encode pass — preserve the legacy behavior.
    prompt_b: str | None = None
    fast_vae: bool = False
    walk_window: bool = False
    walk_window_s: float = 60.0
    fixture_name: str | None = None
    use_server_fixture: bool = False
    stem_source_mode: str | None = None
    enabled_loras: list = field(default_factory=list)
    lora_strengths: dict = field(default_factory=dict)
    lora_paths: list = field(default_factory=list)
    client_id: str | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "SessionConfig":
        """Parse a raw config dict (as the WS adapter received it from
        the browser) into a typed :class:`SessionConfig`.

        Tolerates and ignores unknown keys (forward-compat with future
        front-ends). Translates the legacy ``lora_path`` singular form
        into ``lora_paths``. Coerces ``lora_strengths`` keys to ``str``
        and values to ``float`` (the wire sometimes ships strength as
        an int).
        """
        d = dict(data)

        # Legacy single-path form ``{"lora_path": "..."}``. Don't clobber
        # an explicit ``lora_paths`` if both are present.
        if "lora_path" in d and "lora_paths" not in d:
            lp = d.pop("lora_path")
            d["lora_paths"] = [lp] if lp else []

        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in d.items() if k in known}

        # Coerce strength dict keys/values.
        ls = kwargs.get("lora_strengths") or {}
        kwargs["lora_strengths"] = {
            str(k): float(v) for k, v in ls.items()
        }

        # Coerce enabled_loras to a list of str.
        if "enabled_loras" in kwargs:
            kwargs["enabled_loras"] = list(kwargs["enabled_loras"] or [])

        # Coerce lora_paths to list of str.
        if "lora_paths" in kwargs:
            kwargs["lora_paths"] = list(kwargs["lora_paths"] or [])

        return cls(**kwargs)
