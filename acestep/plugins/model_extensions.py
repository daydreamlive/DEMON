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
   ``stream_layout``, ``decorate_conditioning``, ``apply_controls``,
   ``status`` / ``metrics``.
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
class SourceView:
    """The session's current source, in every form the core already holds.

    ``latent`` is the encoded anchor in the model's own native layout —
    for the sa3 family, ``[1, C, T]``. It is what most conditioning
    extensions want, and it is what DEMON's own denoising path uses.
    Under a :class:`StreamLayout` it is the STACKED anchor, ``T =
    n_streams * L``.

    ``waveform`` and ``sample_rate`` are the audio that latent was encoded
    FROM, carried because the core already holds it. An extension whose
    condition derives from the audio itself rather than from its latent
    cannot recover the audio here without a decode round-trip that costs
    real time and cannot return what the encoder discarded. Handing over a
    tensor the core is keeping anyway is strictly cheaper than making
    every such extension pay for that.

    Every field is optional and any of them may be ``None``: a session
    created without a source has no latent, and a source re-anchored from
    a latent DEMON did not encode itself has no waveform. Check before you
    use; do not assume the shape of a session you did not create.

    Treat the tensors as READ-ONLY. They are the live session anchor
    itself, not copies, and the streaming runner reads them concurrently.
    """

    latent: Any = None
    waveform: Any = None
    sample_rate: int | None = None


@dataclass(frozen=True)
class StreamLayout:
    """Several equal-length latent streams stacked on the token axis.

    Returned by :meth:`ModelExtensionRuntime.stream_layout` when an
    installed extension's model denoises ``n_streams`` parallel streams as
    one sequence: the ring then carries ``[1, n_streams * L, C]`` latents
    (engine layout; ``[1, C, n_streams * L]`` SA3-native), where ``L`` is
    one stream's latent length. Stream ``i`` occupies tokens
    ``[i * L, (i + 1) * L)``. Every slot, the source anchor, feedback and
    ``x0_target`` operate on the stacked tensor unchanged; only the core
    sites that size, encode, decode or accelerate a latent consult this.

    ``n_streams``: number of stacked streams, ``>= 1``.

    ``tokens_per_stream``: pins ``L``. The session's render window is then
    exactly ``L`` latent frames per stream, whatever the requested
    duration (a longer request is clamped to it). A callable
    ``(duration_s) -> L`` derives ``L`` from each session's requested
    duration with the layout's own rule (the window is then ``L`` frames,
    the duration clamped onto it). ``None`` derives ``L`` from the
    requested duration the stock way. Read it through :meth:`stream_tokens`.

    ``make_dit``: ``(context, latent_frames, seconds_total, backend) ->
    dit`` where ``latent_frames`` is the STACKED length. ``context`` is
    the session's view of the model context; its ``session_params`` is
    the session's ``plugin_*`` config keys (empty when none were sent).
    Return either a
    torch callable ``dit(x_bct, t, **cond)`` or an object with
    ``trt_batch1 = True`` and ``step_bundle(x_1ct, t, bundle)`` (see
    ``acestep.engine.sa3_trt.SA3TRTDit``); return ``None`` to run the
    model's own eager DiT. Absent, the eager DiT runs: the stock engine
    lookup is never consulted under a layout, because stock engines are
    built for one stream. An accelerated engine for a layout is loaded by
    path (``acestep.engine.sa3_trt.open_dit_engine_at``), never placed
    under the stock engine names.

    ``make_noise``: ``(request) -> [1, n_streams * L, C]`` initial noise
    for one slot (``request.seed`` is the slot seed). Absent, the
    pipeline's stock single-generator draw at the stacked shape is used.

    ``make_renoise``: ``(request, step) -> [1, n_streams * L, C] | None``,
    the re-noise draw a stochastic (pingpong / SDE) sampler mixes in after
    step ``step`` of the slot built from ``request``, in place of the
    slot's own generator draw. Return ``None`` to take the stock draw.
    Lets per-stream noise streams continue past the initial draw. Absent,
    the stock draw is used at every step.

    ``on_slot_done``: ``(request, latent) -> None``, called once for every
    slot whose denoise completes, with the finished latent ``[1, T, C]``
    in the ring layout, before it is delivered; lets a layout observe
    finished slots (for example to hold a stream at its latest render).
    Absent, nothing is called.

    ``encode_source``: ``(context, audio_input, audio_sample_size) ->
    [1, C, n_streams * L]`` stacked source anchor, where ``audio_input``
    is the ``(sample_rate, waveform)`` pair and ``audio_sample_size`` is
    ONE stream's window in model-rate samples. Absent, the source is
    encoded once and repeated into every stream.

    ``mix``: ``(streams) -> audio`` where ``streams`` is a float32 numpy
    array ``[n_streams, N, channels]`` of per-stream decoded audio at the
    delivery rate and the return is ``[N, channels]``. Called on every
    render, so a gain the extension reads from its own knobs (delivered
    through ``apply_controls``) lands at the next render without
    regenerating. Absent, the streams are summed. The result is clamped
    to ``[-1, 1]`` after mixing.

    ``render_stream``: ``(index, latent) -> audio | None`` where ``latent``
    is stream ``index``'s slice ``[1, C, L]`` (SA3-native) of the latent
    being rendered and ``audio`` is that stream's audio for the WHOLE
    window, a float32 numpy array ``[N, channels]`` at the delivery rate
    (the length a full decode of the slice resamples to). Consulted for
    every stream on every render before the stream is decoded: return
    audio to have the core use it in place of its own decode (a full
    render the extension made once, for example of a stream it holds
    fixed, so every window is cut from one decode instead of re-decoded
    with window seams), or ``None`` to have the stream decoded as usual.
    Windowed renders slice the returned audio at the window. Called on
    the render thread, so keep it a lookup; do the decode elsewhere or
    once. Absent, every stream is decoded.

    Each stream is decoded separately: a decoder's receptive field would
    otherwise bleed across stream boundaries.
    """

    n_streams: int
    tokens_per_stream: int | None = None
    make_dit: Callable | None = None
    make_noise: Callable | None = None
    encode_source: Callable | None = None
    mix: Callable | None = None
    make_renoise: Callable | None = None
    on_slot_done: Callable | None = None
    render_stream: Callable | None = None

    def __post_init__(self):
        if int(self.n_streams) < 1:
            raise ValueError(f"n_streams must be >= 1, got {self.n_streams}")
        tps = self.tokens_per_stream
        if tps is not None and not callable(tps) and int(tps) < 1:
            raise ValueError(f"tokens_per_stream must be >= 1, got {tps}")

    def stream_tokens(self, duration_s: float) -> "int | None":
        """``L`` for a session of ``duration_s``: the pinned count, the
        callable's answer, or ``None`` (derive it the stock way)."""
        tps = self.tokens_per_stream
        if tps is None:
            return None
        n = int(tps(float(duration_s)) if callable(tps) else tps)
        if n < 1:
            raise ValueError(f"tokens_per_stream must be >= 1, got {n}")
        return n


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

    def stream_layout(self) -> "StreamLayout | None":
        """The stacked-stream geometry this model runs, or ``None``.

        ``None`` (the default) is the stock single-stream path, and the
        core then behaves exactly as if this hook did not exist. Called
        once per session at construction, after ``install``; return the
        same layout for the lifetime of the runtime. See
        :class:`StreamLayout`.
        """
        return None

    def decorate_conditioning(
        self, bundle: Mapping[str, Any], source: "SourceView | None" = None,
    ) -> Mapping[str, Any]:
        """Return conditioning for the extension's own use.

        ``source`` is a :class:`SourceView` over the session's current
        anchor, or ``None`` when the session has no source at all. Its
        fields are individually optional — read
        :class:`SourceView` before depending on any of them.

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


def stream_layout_of(runtime) -> "StreamLayout | None":
    """The runtime's :class:`StreamLayout`, or ``None``.

    Tolerates ``runtime=None`` and a runtime that is not a subclass (no
    ``stream_layout`` attribute), so stock sessions never reach plugin
    code. A layout of one stream with no pinned length and no callables
    is the stock path and is normalized to ``None``.
    """
    if runtime is None:
        return None
    hook = getattr(runtime, "stream_layout", None)
    if hook is None:
        return None
    layout = hook()
    if layout is None:
        return None
    if not isinstance(layout, StreamLayout):
        raise TypeError(
            f"stream_layout() must return a StreamLayout or None, got "
            f"{type(layout).__name__}"
        )
    if (
        int(layout.n_streams) == 1
        and layout.tokens_per_stream is None
        and layout.make_dit is None
        and layout.make_noise is None
        and layout.encode_source is None
        and layout.mix is None
    ):
        return None
    return layout


def decorate_conditioning(runtime, bundle, source=None):
    """Call the runtime's conditioning hook, normalizing a ``None`` return.

    The one contract detail a subclass can still get wrong: returning
    nothing from ``decorate_conditioning`` reads as "I made no changes",
    not as "the conditioning is now None". Every other hook has a total
    default on the base class, so the core calls those directly.
    """
    decorated = runtime.decorate_conditioning(bundle, source)
    return bundle if decorated is None else decorated
