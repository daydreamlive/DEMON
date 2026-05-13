"""Bridge ACEStep nodes to Scope's node system.

Creates a Scope BaseNode subclass for each ACEStep node, translating
between the two node APIs. ACEStep nodes use execute(**kwargs) with
keyword arguments; Scope nodes use execute(inputs, **kwargs) with a
dict of inputs.
"""

import inspect
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, ClassVar

from scope.core.nodes.base import BaseNode as ScopeBaseNode
from scope.core.nodes.base import NodeDefinition as ScopeNodeDefinition
from scope.core.nodes.base import NodeParam as ScopeNodeParam
from scope.core.nodes.base import NodePort as ScopeNodePort

_log = logging.getLogger(__name__)

# Set ACESTEP_BRIDGE_TRACE=1 to see per-tick diagnostics (param deltas,
# tick rate, first-latent latency) at INFO level. Off by default because
# a continuous node ticks at ~100Hz and that would flood the logs.
_TRACE = os.environ.get("ACESTEP_BRIDGE_TRACE", "").lower() in ("1", "true", "yes")


def _get_playhead_seconds() -> float | None:
    """Read the primary audio sink's playhead from scope, in seconds.

    Mirrors the realtime demo's ``audio_eng.position / SAMPLE_RATE``.
    Returns None when no audio track is registered yet (e.g. during
    graph warmup or on non-audio sessions), in which case consumers
    should disable their playhead-driven skip gate for that tick.
    """
    try:
        from scope.server.audio_track import get_current_playhead_seconds
    except ImportError:
        return None
    return get_current_playhead_seconds()


# Drop kwargs the installed scope's NodeParam doesn't understand. Lets
# the plugin ship param features (``convertible_to_input``) ahead of
# scope versions that expose them without a hard import-time break.
try:
    _SCOPE_PARAM_FIELDS = set(inspect.signature(ScopeNodeParam).parameters.keys())
except (TypeError, ValueError):
    _SCOPE_PARAM_FIELDS = None


def _param(**kwargs) -> ScopeNodeParam:
    # Every widget is wire-connectable by default so graph authors can
    # drive any param from upstream (curves, number sources, etc.)
    # without the node definition having to opt in per-field. Callers
    # who genuinely want a non-convertible widget can still pass
    # ``convertible_to_input=False`` explicitly.
    kwargs.setdefault("convertible_to_input", True)
    if _SCOPE_PARAM_FIELDS is not None:
        kwargs = {k: v for k, v in kwargs.items() if k in _SCOPE_PARAM_FIELDS}
    return ScopeNodeParam(**kwargs)


# Distinct "never seen" sentinel so the trace can distinguish "param set
# to None by the workflow" from "param not yet observed on this node".
_SENTINEL = object()


# ---------------------------------------------------------------------------
# ACEStep param → Scope param translation
# ---------------------------------------------------------------------------

# ACEStep NodeParam.type values that don't map to a Scope widget. Used to
# skip hidden / internal / ambient-only params when building the Scope
# node definition — the BridgedNode will still pass them through to
# execute() when present, they just don't show up in the UI.
_NON_WIDGET_TYPES = frozenset({"any"})


def _ace_param_to_scope(ace_param) -> ScopeNodeParam | None:
    """Translate an ACEStep ``NodeParam`` to Scope's ``NodeParam``.

    Returns ``None`` for hidden params or params whose type doesn't map
    to a widget — those are valid kwargs but should not appear on the
    node card. The BridgedNode still receives them through ``kwargs``
    when the host sets them; they just aren't user-editable widgets.
    """
    if ace_param.hidden or ace_param.type in _NON_WIDGET_TYPES:
        return None

    # Scope's "integer" and "number" both map to the number widget type;
    # min/max/step live inside the ``ui`` bag.
    param_type = ace_param.type
    if param_type == "integer":
        scope_type = "number"
    elif param_type in ("string", "number", "boolean", "select"):
        scope_type = param_type
    else:
        # Unknown type — treat as string so the node remains usable rather
        # than raising at bridge-load time.
        scope_type = "string"

    ui: dict[str, Any] = {}
    if ace_param.min is not None:
        ui["min"] = ace_param.min
    if ace_param.max is not None:
        ui["max"] = ace_param.max
    if ace_param.step is not None:
        ui["step"] = ace_param.step
    if ace_param.options is not None:
        ui["options"] = list(ace_param.options)

    scope_kwargs: dict[str, Any] = dict(
        name=ace_param.name,
        param_type=scope_type,
        default=ace_param.default,
        description=ace_param.description,
    )
    if ui:
        scope_kwargs["ui"] = ui
    return _param(**scope_kwargs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_acestep_importable():
    """Add ACEStep root to sys.path if not already importable."""
    acestep_root = str(Path(__file__).resolve().parent.parent)
    if acestep_root not in sys.path:
        sys.path.insert(0, acestep_root)


def _coerce_audio_inputs(call_kwargs: dict) -> None:
    """Convert (tensor, sample_rate) tuples to ACEStep Audio objects.

    Scope's built-in audio nodes emit plain tuples to avoid a hard
    dependency on ACEStep. At the plugin boundary we wrap them in the
    Audio dataclass that ACEStep's nodes expect.
    """
    from acestep.nodes import Audio

    for key, value in list(call_kwargs.items()):
        if isinstance(value, tuple) and len(value) == 2:
            tensor, sample_rate = value
            if hasattr(tensor, "shape") and isinstance(sample_rate, int):
                call_kwargs[key] = Audio(
                    waveform=tensor, sample_rate=sample_rate
                )


def _load_acestep_nodes():
    """Import all ACEStep node modules and return the registry."""
    _ensure_acestep_importable()
    from acestep.nodes import (  # noqa: F401
        audio_nodes,
        channel_nodes,
        cond_nodes,
        curve_nodes,
        diffusion_nodes,
        lora_nodes,
        mask_nodes,
        model_nodes,
        semantic_nodes,
        vae_nodes,
    )
    from acestep.nodes.base import NodeRegistry as ACEStepRegistry

    return ACEStepRegistry


# ---------------------------------------------------------------------------
# Bridge factory
# ---------------------------------------------------------------------------


def _make_scope_node_class(acestep_cls):
    """Create a Scope BaseNode subclass wrapping an ACEStep node class."""
    ace_defn = acestep_cls.get_definition()

    def _map_type(t):
        return t.lower()

    scope_inputs = [
        ScopeNodePort(
            name=p.name,
            port_type=_map_type(p.type),
            required=p.required,
            description=p.description,
        )
        for p in ace_defn.inputs
    ]
    scope_outputs = [
        ScopeNodePort(
            name=p.name,
            port_type=_map_type(p.type),
            description=p.description,
        )
        for p in ace_defn.outputs
    ]
    scope_params = [
        sp
        for sp in (_ace_param_to_scope(p) for p in ace_defn.params)
        if sp is not None
    ]

    # ACEStep NodeDefinition has no ``continuous`` field, so nodes that
    # should be re-executed every tick (independent of fresh inputs) opt
    # in via a class attribute ``_scope_continuous``. StreamDenoise uses
    # this to self-clock its ring buffer: upstream runs once, fills the
    # latch cache, then StreamDenoise ticks on its own worker loop
    # without waiting for upstream to re-emit.
    continuous = bool(getattr(acestep_cls, "_scope_continuous", False))

    scope_defn = ScopeNodeDefinition(
        node_type_id=ace_defn.node_type_id,
        display_name=ace_defn.display_name,
        category=ace_defn.category,
        description=ace_defn.description,
        inputs=scope_inputs,
        outputs=scope_outputs,
        params=scope_params,
        continuous=continuous,
    )

    required_input_ports = frozenset(p.name for p in ace_defn.inputs if p.required)
    # Scalar scope params are cheap to log as deltas. Latent/tensor ports
    # aren't — restrict to numeric/string/bool so a trace never touches a
    # CUDA tensor.
    _PARAM_KEYS_FOR_DELTA = frozenset(p.name for p in scope_params)

    class BridgedNode(ScopeBaseNode):
        node_type_id: ClassVar[str] = ace_defn.node_type_id
        _ace_cls = acestep_cls
        _scope_defn = scope_defn
        _required_input_ports: ClassVar[frozenset[str]] = required_input_ports
        _param_keys_for_delta: ClassVar[frozenset[str]] = _PARAM_KEYS_FOR_DELTA

        def __init__(self, node_id, config=None):
            super().__init__(node_id, config)
            self._ace_instance = self._ace_cls()
            # Scope's NodeProcessor latches inputs across ticks for
            # non-continuous nodes but not for continuous ones — its
            # continuous branch only reads from the live queue. Upstream
            # one-shot handles (LoadModel → model/vae/clip) therefore
            # vanish from ``inputs`` after the first tick, breaking
            # StreamDenoise / StreamVAEDecode. We mirror scope's latch
            # here so every tick sees the last known value for every
            # port, with fresh arrivals overriding the cache.
            self._latched_inputs: dict[str, Any] = {}
            # Trace state: used only when ACESTEP_BRIDGE_TRACE=1.
            self._last_param_snapshot: dict[str, Any] = {}
            self._tick_count = 0
            self._emit_count = 0
            self._first_real_call_ts: float | None = None
            self._last_report_ts: float = time.monotonic()

        @classmethod
        def get_definition(cls) -> ScopeNodeDefinition:
            return cls._scope_defn

        def shutdown(self) -> None:
            # Scope's PipelineManager caches the BridgedNode instance across
            # Stop/Run cycles, so ``_latched_inputs`` survives session
            # boundaries by default. For ACEStep that means the previous
            # session's ModelHandle/VAE/CLIP — and the ~30 GB ModelContext
            # they reference — stay pinned until something overwrites them.
            # When the new session's LoadModel.execute() then allocates a
            # fresh ModelContext, both copies coexist on the GPU until the
            # downstream latches get rebound, which on a 32 GB card OOMs
            # before that swap happens.
            #
            # Releasing the latch references here lets Python collect the
            # old context before the new session starts; we also drop the
            # internal pipeline reference on StreamDenoise (its
            # ``_pipeline`` likewise holds the model) and force a
            # ``torch.cuda.empty_cache`` so the freed memory is visible to
            # the next allocation.
            self._latched_inputs.clear()
            pipeline = getattr(self._ace_instance, "_pipeline", None)
            if pipeline is not None:
                self._ace_instance._pipeline = None
            try:
                import gc

                import torch

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        def execute(self, inputs: dict[str, Any], **kwargs) -> dict[str, Any]:
            if inputs:
                self._latched_inputs.update(inputs)

            # Skip until every required upstream port has arrived at
            # least once. Returning {} lets scope's ``_process_once``
            # retry next tick without flipping ``_has_executed``.
            missing = self._required_input_ports - self._latched_inputs.keys()
            if missing:
                if _TRACE:
                    self._tick_count += 1
                return {}

            # Trace param deltas BEFORE calling ACE.execute so we can
            # see exactly what the denoiser is about to denoise with.
            if _TRACE:
                changed = {}
                for k in self._param_keys_for_delta:
                    v = kwargs.get(k)
                    if self._last_param_snapshot.get(k, _SENTINEL) != v:
                        changed[k] = v
                if changed and self._last_param_snapshot:
                    _log.info(
                        "[bridge][%s:%s] param delta: %s",
                        self.node_type_id, self.node_id, changed,
                    )
                self._last_param_snapshot = {
                    k: kwargs.get(k) for k in self._param_keys_for_delta
                }

            call_kwargs = dict(kwargs)
            call_kwargs.update(self._latched_inputs)
            # Inject the sink playhead on every call so any ACE node that
            # opts in (e.g. StreamVAEDecode's skip gate) can read it. Nodes
            # that don't care ignore the extra kwarg harmlessly — every
            # ACE node uses `**kwargs` + `kwargs.get(...)`.
            playhead = _get_playhead_seconds()
            if playhead is not None:
                call_kwargs.setdefault("playhead_seconds", playhead)
            _coerce_audio_inputs(call_kwargs)

            if _TRACE and self._first_real_call_ts is None:
                self._first_real_call_ts = time.monotonic()
                _log.info(
                    "[bridge][%s:%s] first real execute; inputs_latched=%s",
                    self.node_type_id, self.node_id,
                    sorted(self._latched_inputs.keys()),
                )

            out = self._ace_instance.execute(**call_kwargs) or {}

            if _TRACE:
                self._tick_count += 1
                emitted = any(v is not None for v in out.values()) if out else False
                if emitted:
                    self._emit_count += 1
                now = time.monotonic()
                if now - self._last_report_ts >= 2.0:
                    _log.info(
                        "[bridge][%s:%s] ticks=%d emits=%d in %.2fs (%.1f tick/s, %.1f emit/s)",
                        self.node_type_id, self.node_id,
                        self._tick_count, self._emit_count,
                        now - self._last_report_ts,
                        self._tick_count / (now - self._last_report_ts),
                        self._emit_count / (now - self._last_report_ts),
                    )
                    self._tick_count = 0
                    self._emit_count = 0
                    self._last_report_ts = now

            return out

    BridgedNode.__name__ = f"Bridged_{ace_defn.node_type_id.replace('.', '_')}"
    BridgedNode.__qualname__ = BridgedNode.__name__

    return BridgedNode


def get_all_bridged_nodes():
    """Return a list of Scope-compatible node classes for all ACEStep nodes."""
    registry = _load_acestep_nodes()
    bridged = []
    for node_type_id in registry.list_node_types():
        ace_cls = registry.get(node_type_id)
        if ace_cls is not None:
            bridged.append(_make_scope_node_class(ace_cls))
    return bridged
