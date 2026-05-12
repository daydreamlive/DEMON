"""Diffusion nodes: configuration and generation.

``StreamDenoise`` is the single surface that owns a
:class:`~acestep.engine.stream.StreamPipeline`. Everything else —
``Generate`` (one-shot), ``Session.generate``, ``Session.stream`` —
drives the pipeline through this node. Grep invariant:
``rg 'StreamPipeline\\(' acestep/`` yields exactly one hit inside
``StreamDenoise._ensure_pipeline``.
"""

from __future__ import annotations

import torch
from typing import Any, ClassVar, Optional

from acestep.engine.conditions import PreparedCondition
from acestep.engine.diffusion import DiffusionConfig, DiffusionEngine
from acestep.engine.stream import SlotCondition, SlotRequest, StreamPipeline

from .base import BaseNode, NodeDefinition, NodeParam, NodePort, NodeRegistry
from .types import (
    Conditioning,
    Config,
    Curve,
    DCW,
    Latent,
    Mask,
    ModelHandle,
    Modulation,
    Solver,
)


def _curve_tensor(value) -> Optional[torch.Tensor]:
    """Extract a tensor from a Curve/Mask/Latent wrapper or pass through."""
    if value is None:
        return None
    return value.tensor if hasattr(value, "tensor") else value


@NodeRegistry.register
class DiffusionConfigNode(BaseNode):
    """Create a diffusion loop configuration.

    Node parameters:
        steps: Number of diffusion steps (default 8 for turbo).
        shift: Timestep shift (default 3.0 for turbo).
        seed: Random seed.
        denoise: Denoising strength 0.0-1.0 (1.0 = full generation).
        use_cache: Enable KV caching (default False).
        noise_on_cpu: Generate noise on CPU for ComfyUI parity (default True).

    Solver choice (``ode`` vs ``sde``) is not a config field — wire a
    ``Solver`` node (``OdeSolver`` / ``SdeSolver``) into the denoiser
    instead. Solver-specific curves travel with the solver.
    """

    node_type_id: ClassVar[str] = "acestep.DiffusionConfig"

    @classmethod
    def get_definition(cls) -> NodeDefinition:
        return NodeDefinition(
            node_type_id=cls.node_type_id,
            display_name="Diffusion Config",
            category="diffusion",
            description="Configure the diffusion sampling loop.",
            inputs=(),
            outputs=(
                NodePort(name="config", type="CONFIG"),
            ),
            params=(
                NodeParam(
                    name="steps", type="integer", default=8,
                    description="Diffusion steps",
                    min=1, max=50, step=1,
                ),
                NodeParam(
                    name="shift", type="number", default=3.0,
                    description="Timestep shift",
                    min=0.0, max=10.0, step=0.05,
                ),
                NodeParam(
                    name="seed", type="integer", default=42,
                    description="Seed",
                    min=0, max=2**31 - 1, step=1,
                ),
                NodeParam(
                    name="denoise", type="number", default=1.0,
                    description="Denoise strength (0 = preserve source, 1 = full gen)",
                    min=0.0, max=1.0, step=0.01,
                ),
                NodeParam(
                    name="x0_target_gate", type="number", default=0.0,
                    description="Curve gate for x0 target (0 = off)",
                    min=0.0, max=1.0, step=0.01,
                ),
                NodeParam(
                    name="use_cache", type="boolean", default=False,
                    description="Enable KV cache",
                    hidden=True,
                ),
                NodeParam(
                    name="noise_on_cpu", type="boolean", default=True,
                    description="Generate noise on CPU (ComfyUI parity)",
                    hidden=True,
                ),
            ),
        )

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        config = DiffusionConfig(
            infer_steps=kwargs.get("steps", 8),
            shift=kwargs.get("shift", 3.0),
            seed=kwargs.get("seed", None),
            use_cache=kwargs.get("use_cache", False),
            noise_on_cpu=kwargs.get("noise_on_cpu", True),
            denoise=kwargs.get("denoise", 1.0),
            x0_target_gate=kwargs.get("x0_target_gate", 0.0),
        )
        return {"config": Config(config=config)}


@NodeRegistry.register
class OdeSolver(BaseNode):
    """Deterministic Euler ODE solver.

    Output feeds the denoiser's ``solver`` input. The optional
    ``ode_noise_curve`` input, when wired, adds scaled Gaussian noise
    after each integration step — a small stochastic kick that can help
    escape local minima during fast-schedule (low-steps) denoising.
    Absent, the solver runs as pure deterministic Euler.
    """

    node_type_id: ClassVar[str] = "acestep.OdeSolver"

    @classmethod
    def get_definition(cls) -> NodeDefinition:
        return NodeDefinition(
            node_type_id=cls.node_type_id,
            display_name="ODE Solver",
            category="diffusion",
            description="Euler ODE solver with optional per-step noise injection.",
            inputs=(
                NodePort(name="ode_noise_curve", type="CURVE", required=False),
            ),
            outputs=(
                NodePort(name="solver", type="SOLVER"),
            ),
        )

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        curve = kwargs.get("ode_noise_curve")
        tensor = curve.tensor if curve is not None else None
        return {"solver": Solver(method="ode", noise_curve=tensor)}


@NodeRegistry.register
class SdeSolver(BaseNode):
    """Stochastic SDE solver with optional per-frame denoise blending.

    Output feeds the denoiser's ``solver`` input. The optional
    ``sde_denoise_curve`` input, when wired, controls per-frame blending
    between pure re-noise (1.0, strong denoising) and source blending
    (0.0, preserves source). The curve only takes effect when the
    denoiser also has a ``source_latent`` wired. Absent the curve, the
    solver falls back to ``model.renoise`` for a schedule-driven global
    re-noise.
    """

    node_type_id: ClassVar[str] = "acestep.SdeSolver"

    @classmethod
    def get_definition(cls) -> NodeDefinition:
        return NodeDefinition(
            node_type_id=cls.node_type_id,
            display_name="SDE Solver",
            category="diffusion",
            description="Stochastic SDE solver with optional per-frame denoise curve.",
            inputs=(
                NodePort(name="sde_denoise_curve", type="CURVE", required=False),
            ),
            outputs=(
                NodePort(name="solver", type="SOLVER"),
            ),
        )

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        curve = kwargs.get("sde_denoise_curve")
        tensor = curve.tensor if curve is not None else None
        return {"solver": Solver(method="sde", noise_curve=tensor)}


@NodeRegistry.register
class DCWConfig(BaseNode):
    """Configure DCW (Differential Correction in Wavelet domain).

    Output feeds the optional ``dcw`` input on ``StreamDenoise`` /
    ``Generate``. Most graphs leave this unwired and inherit the
    upstream-v0.1.7 defaults (enabled=True, mode='double', scaler=0.05,
    high_scaler=0.02, wavelet='haar') from the consumer's hidden
    NodeParams. Wire this node to override.

    DCW is a sampler-side post-step correction applied in the wavelet
    domain after each integration step; see ``acestep.engine.dcw`` for
    the math.
    """

    node_type_id: ClassVar[str] = "acestep.DCWConfig"

    @classmethod
    def get_definition(cls) -> NodeDefinition:
        return NodeDefinition(
            node_type_id=cls.node_type_id,
            display_name="DCW",
            category="diffusion",
            description="Differential Correction in Wavelet domain (post-step sampler correction).",
            inputs=(),
            outputs=(
                NodePort(name="dcw", type="DCW"),
            ),
            params=(
                NodeParam(
                    name="enabled", type="boolean", default=True,
                    description="Enable DCW",
                ),
                NodeParam(
                    name="mode", type="string", default="double",
                    description="DCW mode: low / high / double / pix",
                ),
                NodeParam(
                    name="scaler", type="number", default=0.05,
                    description="Low-band scaler",
                    min=0.0, max=1.0, step=0.005,
                ),
                NodeParam(
                    name="high_scaler", type="number", default=0.02,
                    description="High-band scaler (double mode only)",
                    min=0.0, max=1.0, step=0.005,
                ),
                NodeParam(
                    name="wavelet", type="string", default="haar",
                    description="Wavelet basis: haar / db4 / sym8 / ...",
                ),
            ),
        )

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "dcw": DCW(
                enabled=bool(kwargs.get("enabled", True)),
                mode=str(kwargs.get("mode", "double")),
                scaler=float(kwargs.get("scaler", 0.05)),
                high_scaler=float(kwargs.get("high_scaler", 0.02)),
                wavelet=str(kwargs.get("wavelet", "haar")),
            ),
        }


@NodeRegistry.register
class ModulationNode(BaseNode):
    """Collect every optional denoiser modulation into one bundle.

    Emits ``modulation`` consumed by ``StreamDenoise`` / ``Generate``.
    Users opt in to the modulation surface by adding this node — the
    denoiser stays minimal otherwise. Every input is optional; wire
    only what you need.
    """

    node_type_id: ClassVar[str] = "acestep.Modulation"

    @classmethod
    def get_definition(cls) -> NodeDefinition:
        return NodeDefinition(
            node_type_id=cls.node_type_id,
            display_name="Modulation",
            category="diffusion",
            description="Bundle optional denoiser inputs (curves, masks, CFG, x0 target).",
            inputs=(
                NodePort(name="velocity_scale", type="CURVE", required=False),
                NodePort(name="initial_noise_curve", type="CURVE", required=False),
                NodePort(name="chunk_mask", type="MASK", required=False),
                NodePort(name="x0_target", type="LATENT", required=False),
                NodePort(name="x0_target_curve", type="CURVE", required=False),
                NodePort(name="guidance_curve", type="CURVE", required=False),
            ),
            outputs=(
                NodePort(name="modulation", type="MODULATION"),
            ),
            params=(
                NodeParam(
                    name="x0_target_strength", type="number", default=0.0,
                    description="Scalar blend toward x0 target (second half of schedule). Requires x0_target.",
                    min=0.0, max=1.0, step=0.01,
                ),
                NodeParam(
                    name="x0_target_gate", type="number", default=0.0,
                    description="Curve-path gate: fraction of schedule before x0_target_curve activates.",
                    min=0.0, max=1.0, step=0.01,
                ),
            ),
        )

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "modulation": Modulation(
                velocity_scale=_curve_tensor(kwargs.get("velocity_scale")),
                initial_noise_curve=_curve_tensor(kwargs.get("initial_noise_curve")),
                chunk_mask=_curve_tensor(kwargs.get("chunk_mask")),
                x0_target=kwargs.get("x0_target"),
                x0_target_curve=_curve_tensor(kwargs.get("x0_target_curve")),
                x0_target_strength=float(kwargs.get("x0_target_strength", 0.0)),
                x0_target_gate=float(kwargs.get("x0_target_gate", 0.0)),
                guidance_curve=_curve_tensor(kwargs.get("guidance_curve")),
            ),
        }


def _build_context_latents(
    context_latent: Optional[Latent],
    chunk_mask: Optional[Mask],
    handler,
    device,
    dtype,
    duration: float,
) -> torch.Tensor:
    """Build the [B, T, D*2] context_latents tensor for the decoder.

    Matches the pre-refactor Generate node semantics: cat(ctx_lat, chunk_mask).
    Defaults ctx_lat to silence and chunk_mask to all-ones.
    """
    if context_latent is not None:
        ctx_lat = context_latent.tensor.to(device=device, dtype=dtype)
    else:
        handler._ensure_silence_latent_on_device()
        T = int(duration * 25)
        ctx_lat = handler.silence_latent[:, :T, :].clone().to(
            device=device, dtype=dtype
        )

    T = ctx_lat.shape[1]
    D = ctx_lat.shape[2]

    if chunk_mask is not None:
        cm = chunk_mask.tensor.to(device=device, dtype=dtype)
        if cm.ndim == 1:
            cm = cm.unsqueeze(0).unsqueeze(-1).expand(1, T, D)
        elif cm.ndim == 2:
            cm = cm.unsqueeze(-1).expand(-1, T, D)
    else:
        cm = torch.ones(1, T, D, device=device, dtype=dtype)

    return torch.cat([ctx_lat, cm], dim=-1)


def _curve_to_tensor(curve, device, dtype) -> Optional[torch.Tensor]:
    if curve is None:
        return None
    if hasattr(curve, "tensor"):
        return curve.tensor.to(device=device, dtype=dtype)
    return curve.to(device=device, dtype=dtype)


def _build_slot_request(
    *,
    positive: Conditioning,
    negative: Optional[Conditioning],
    context_latents: torch.Tensor,
    source_latent: Optional[Latent],
    rcfg_mode: Optional[str] = None,
    cfg_rescale=None,
    seed,
    denoise: float,
    velocity_scale,
    sde_denoise_curve,
    ode_noise_curve,
    initial_noise_curve,
    x0_target,
    x0_target_curve,
    x0_target_strength: float,
    x0_target_gate: float,
    guidance_curve,
    device,
    dtype,
) -> SlotRequest:
    pos_entries = positive.to_entries()
    primary = pos_entries[0]
    extra_conditions = [
        SlotCondition(
            encoder_hidden_states=e.encoder_hidden_states,
            encoder_attention_mask=e.encoder_attention_mask,
            temporal_weight=e.temporal_weight,
            step_range=e.step_range,
        )
        for e in pos_entries[1:]
    ]

    # Guidance is engaged when either standard CFG (negative provided)
    # or RCFG-self (virtual uncond from initial noise) is requested.
    # neg_conditions stays empty for the "self" branch.
    neg_conditions: list[SlotCondition] = []
    guidance_curve_t = None
    if guidance_curve is not None and (
        negative is not None or rcfg_mode == "self"
    ):
        if negative is not None:
            for e in negative.to_entries():
                neg_conditions.append(SlotCondition(
                    encoder_hidden_states=e.encoder_hidden_states,
                    encoder_attention_mask=e.encoder_attention_mask,
                    temporal_weight=e.temporal_weight,
                    step_range=e.step_range,
                ))
        guidance_curve_t = _curve_to_tensor(guidance_curve, device, dtype)

    src_tensor = None
    latent_mask = None
    if source_latent is not None:
        src_tensor = source_latent.tensor
        latent_mask = source_latent.mask

    x0_tensor = None
    if x0_target is not None:
        x0_tensor = x0_target.tensor.to(device=device, dtype=dtype)

    return SlotRequest(
        encoder_hidden_states=primary.encoder_hidden_states,
        encoder_attention_mask=primary.encoder_attention_mask,
        context_latents=context_latents,
        seed=seed,
        source_latents=src_tensor,
        denoise=denoise,
        sde_denoise_curve=_curve_to_tensor(sde_denoise_curve, device, dtype),
        velocity_scale=_curve_to_tensor(velocity_scale, device, dtype),
        ode_noise_curve=_curve_to_tensor(ode_noise_curve, device, dtype),
        x0_target=x0_tensor,
        x0_target_strength=x0_target_strength,
        x0_target_curve=_curve_to_tensor(x0_target_curve, device, dtype),
        x0_target_gate=x0_target_gate,
        initial_noise_curve=_curve_to_tensor(initial_noise_curve, device, dtype),
        latent_mask=latent_mask,
        extra_conditions=extra_conditions,
        primary_temporal_weight=primary.temporal_weight,
        primary_step_range=primary.step_range,
        neg_conditions=neg_conditions,
        guidance_curve=guidance_curve_t,
        rcfg_mode=rcfg_mode,
        cfg_rescale_curve=_curve_to_tensor(cfg_rescale, device, dtype),
    )


@NodeRegistry.register
class StreamDenoise(BaseNode):
    """Streaming diffusion denoiser. Owns a ``StreamPipeline`` internally.

    One ``execute()`` call submits a new request and advances the ring
    buffer by one tick. With ``pipeline_depth=steps`` (the default), the
    buffer is StreamDiffusion-shaped: after warmup every tick emits a
    finished latent. With ``pipeline_depth=1`` plus ``drain=True`` the
    node behaves as a synchronous one-shot generator — this is how the
    ``Generate`` node (below) reuses the same pipeline.

    Rebuild-on-change (new pipeline built):
        steps, method, pipeline_depth, noise_on_cpu, use_cache, or a
        new model handle.
    Hot-updatable (no rebuild):
        shift (schedule cache cleared), denoise, seed, all curves,
        x0_target_strength, x0_target_gate. Channel guidance reads
        live from the handler every tick.
    """

    node_type_id: ClassVar[str] = "acestep.StreamDenoise"
    # Tell the Scope bridge to mark this node ``continuous=True``. The
    # ring-buffer self-clocks: every call submits a fresh slot and
    # advances the pipeline by one tick, so the node must re-execute
    # each frame even when upstream inputs haven't changed.
    _scope_continuous: ClassVar[bool] = True

    def __init__(self):
        self._pipeline: Optional[StreamPipeline] = None
        self._engine: Optional[DiffusionEngine] = None
        self._last_handle_id: Optional[int] = None
        self._shape_key: Optional[tuple] = None

    @classmethod
    def get_definition(cls) -> NodeDefinition:
        return NodeDefinition(
            node_type_id=cls.node_type_id,
            display_name="Stream Denoise",
            category="diffusion",
            description="Streaming diffusion denoiser with ring buffer.",
            inputs=(
                NodePort(name="model", type="MODEL"),
                NodePort(
                    name="solver", type="SOLVER",
                    description="Integration solver. Wire OdeSolver or SdeSolver.",
                ),
                NodePort(name="positive", type="CONDITIONING"),
                NodePort(
                    name="negative", type="CONDITIONING", required=False,
                    description="Negative conditioning. Pair with modulation.guidance_curve to enable CFG.",
                ),
                NodePort(
                    name="context_latent", type="LATENT", required=False,
                    description="Structural context (silence, semantic hints, etc.).",
                ),
                NodePort(
                    name="source_latent", type="LATENT", required=False,
                    description="Source audio latent for partial denoise / inpainting.",
                ),
                NodePort(
                    name="modulation", type="MODULATION", required=False,
                    description="Optional bundle of curves, mask, x0 target, guidance_curve.",
                ),
                NodePort(
                    name="dcw", type="DCW", required=False,
                    description="Optional DCW bundle from DCWConfig; overrides hidden DCW NodeParams when wired.",
                ),
            ),
            outputs=(
                NodePort(
                    name="latent", type="LATENT",
                    description="Finished latent when a slot completes this tick; None otherwise.",
                ),
            ),
            params=(
                NodeParam(
                    name="steps", type="integer", default=8,
                    description="Diffusion steps",
                    min=1, max=50, step=1,
                ),
                NodeParam(
                    name="shift", type="number", default=3.0,
                    description="Timestep shift",
                    min=0.0, max=10.0, step=0.05,
                ),
                NodeParam(
                    name="denoise", type="number", default=0.75,
                    description="Denoise strength",
                    min=0.0, max=1.0, step=0.01,
                ),
                NodeParam(
                    name="seed", type="integer", default=42,
                    description="Seed",
                    min=0, max=2**31 - 1, step=1,
                ),
                NodeParam(
                    name="pipeline_depth", type="integer", default=8,
                    description="Ring-buffer depth",
                    min=1, max=16, step=1,
                ),
                NodeParam(
                    name="drain", type="boolean", default=False,
                    description="Synchronous drain (one-shot)",
                ),
                NodeParam(
                    name="duration", type="number", default=60.0,
                    description="Latent duration (s) for silence-context fallback",
                    min=1.0, max=600.0, step=1.0,
                ),
                NodeParam(
                    name="use_cache", type="boolean", default=False,
                    description="Enable KV cache",
                    hidden=True,
                ),
                NodeParam(
                    name="noise_on_cpu", type="boolean", default=True,
                    description="Generate noise on CPU",
                    hidden=True,
                ),
                NodeParam(
                    name="dcw_enabled", type="boolean", default=True,
                    description="DCW (wavelet-domain post-step correction)",
                    hidden=True,
                ),
                NodeParam(
                    name="dcw_mode", type="string", default="double",
                    description="DCW mode: low / high / double / pix",
                    hidden=True,
                ),
                NodeParam(
                    name="dcw_scaler", type="number", default=0.05,
                    description="DCW low-band scaler",
                    hidden=True,
                ),
                NodeParam(
                    name="dcw_high_scaler", type="number", default=0.02,
                    description="DCW high-band scaler (double mode)",
                    hidden=True,
                ),
                NodeParam(
                    name="dcw_wavelet", type="string", default="haar",
                    description="DCW wavelet basis (haar / db4 / sym8 / ...)",
                    hidden=True,
                ),
                NodeParam(
                    name="dcw_advanced", type="any", default=None,
                    description=(
                        "DCW advanced research-surface config "
                        "(DCWAdvanced instance). None = upstream defaults."
                    ),
                    hidden=True,
                ),
                NodeParam(
                    name="rcfg_mode", type="string", default=None,
                    description=(
                        "RCFG mode: None/'full' (standard CFG, neg "
                        "forward every step), 'initialize' (neg forward "
                        "once per slot then cached), 'self' (no neg "
                        "forward; virtual v_uncond = initial noise)."
                    ),
                    hidden=True,
                ),
                NodeParam(
                    name="cfg_rescale", type="any", default=None,
                    description=(
                        "Per-frame mix toward vt_pos's magnitude after APG. "
                        "0 / None disables; 1 fully snaps norm to vt_pos. "
                        "Scalar or per-frame curve. Fixes high-CFG saturation."
                    ),
                    hidden=True,
                ),
            ),
        )

    def _ensure_pipeline(
        self, handler, shape_key: tuple,
    ) -> StreamPipeline:
        """Build or reuse the pipeline. Rebuilds on shape-key change.

        Shape-key changes that affect ring-buffer shape or timestep
        schedule semantics force a fresh pipeline. Hot-path parameters
        (denoise, curves, seed, shift) bypass this entirely — they ride
        on the per-tick ``SlotRequest``.
        """
        handle_id = id(handler)
        if (
            self._pipeline is not None
            and self._last_handle_id == handle_id
            and self._shape_key == shape_key
        ):
            return self._pipeline

        if handler._diffusion_engine is None:
            handler._diffusion_engine = DiffusionEngine(
                handler.model, compile_loops=handler._compile_decoder,
            )
        self._engine = handler._diffusion_engine

        steps, depth, noise_on_cpu, use_cache = shape_key
        config = DiffusionConfig(
            infer_steps=steps,
            shift=3.0,  # updated hot below
            noise_on_cpu=noise_on_cpu,
            use_cache=use_cache,
        )
        self._pipeline = StreamPipeline(
            self._engine, config,
            pipeline_depth=depth,
        )
        self._last_handle_id = handle_id
        self._shape_key = shape_key
        return self._pipeline

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        model: ModelHandle = kwargs["model"]
        positive: Conditioning = kwargs["positive"]
        context_latent: Optional[Latent] = kwargs.get("context_latent")
        source_latent: Optional[Latent] = kwargs.get("source_latent")

        modulation: Modulation = kwargs.get("modulation") or Modulation()
        negative: Optional[Conditioning] = kwargs.get("negative")
        # chunk_mask in the bundle is already a raw tensor; wrap as Mask
        # for the downstream context-latent builder which expects the
        # wrapper (or None).
        chunk_mask = (
            Mask(tensor=modulation.chunk_mask)
            if modulation.chunk_mask is not None
            else None
        )

        solver: Solver = kwargs["solver"]
        steps = int(kwargs.get("steps", 8))
        shift = float(kwargs.get("shift", 3.0))
        denoise = float(kwargs.get("denoise", 1.0))
        seed = kwargs.get("seed")
        depth_arg = kwargs.get("pipeline_depth")
        depth = int(depth_arg) if depth_arg is not None else steps
        noise_on_cpu = bool(kwargs.get("noise_on_cpu", True))
        use_cache = bool(kwargs.get("use_cache", False))
        # x0_target_strength / x0_target_gate live on the Modulation
        # bundle now (structurally coupled to x0_target). Read them off
        # the bundle; fall back to 0.0 when no bundle was wired.
        x0_target_strength = modulation.x0_target_strength
        x0_target_gate = modulation.x0_target_gate
        drain = bool(kwargs.get("drain", False))
        duration = float(kwargs.get("duration", 60.0))

        handler = model.handler
        device = handler.device
        dtype = handler.dtype

        # ``method`` is hot-updatable — it only selects which compiled step
        # helper runs inside the pipeline, no shape dependency, so the
        # pipeline doesn't need a rebuild when the solver wire changes.
        shape_key = (steps, depth, noise_on_cpu, use_cache)
        pipe = self._ensure_pipeline(handler, shape_key)

        # Seed device/dtype so set_channel_guidance picks the correct
        # device on the first call (before any submit populates _device).
        if pipe._device is None:
            pipe._device = torch.device(device)
            pipe._dtype = dtype

        # Hot-update shift (clear schedule cache) and method.
        if pipe.config.shift != shift:
            pipe.config.shift = shift
            pipe._schedule_cache.clear()
        pipe.config.infer_method = solver.method

        # Channel guidance reads live from the handler so ChannelGuidance /
        # RemoveChannelGuidance nodes take effect on the next tick without
        # any rebuild.
        pipe.set_channel_guidance(getattr(handler, "_channel_guidance", None))

        # DCW (Differential Correction in Wavelet domain) — hot-update
        # per tick so toggling/tuning takes effect immediately for all
        # in-flight slots without rebuilding the pipeline.
        #
        # Resolution order:
        #   1. Wired ``dcw`` bundle (DCWConfig output) — graph-builder path.
        #   2. Direct ``dcw_*`` kwargs — Session.stream / demo callers.
        #   3. Hidden NodeParam defaults via the ``kwargs.get`` fallbacks.
        # Tiers 2 and 3 share the same code path; the runtime fills
        # NodeParam defaults into kwargs before this method runs, so
        # explicit kwargs naturally take precedence.
        dcw_bundle: Optional[DCW] = kwargs.get("dcw")
        if dcw_bundle is not None:
            pipe.set_dcw(
                enabled=dcw_bundle.enabled,
                mode=dcw_bundle.mode,
                scaler=dcw_bundle.scaler,
                high_scaler=dcw_bundle.high_scaler,
                wavelet=dcw_bundle.wavelet,
                advanced=dcw_bundle.advanced,
            )
        else:
            pipe.set_dcw(
                enabled=bool(kwargs.get("dcw_enabled", True)),
                mode=str(kwargs.get("dcw_mode", "double")),
                scaler=float(kwargs.get("dcw_scaler", 0.05)),
                high_scaler=float(kwargs.get("dcw_high_scaler", 0.02)),
                wavelet=str(kwargs.get("dcw_wavelet", "haar")),
                advanced=kwargs.get("dcw_advanced"),
            )

        context_latents = _build_context_latents(
            context_latent, chunk_mask, handler, device, dtype, duration,
        )

        # Short-circuit: zero-step or zero-denoise returns source latents
        # (or prepared noise) directly, without submitting to the pipeline.
        # Parity with the pre-refactor DiffusionEngine.generate() fast-exit.
        #
        # Scoped to drain-mode on purpose: in streaming mode the caller
        # expects the returned latent to come from a ring-buffer slot
        # completion (which preserves submit/tick latency contracts).
        # Returning immediately here would emit latents out-of-band and
        # break the streaming timing model. Zero-denoise in streaming
        # mode is pathological but legal — it just spends one slot on
        # a trivial schedule (`t=[0, 0]`, one tick to mark complete)
        # and returns the source latent via the normal path.
        if drain:
            t_schedule = self._engine._build_timestep_schedule(
                DiffusionConfig(
                    infer_steps=steps, shift=shift, denoise=denoise,
                ),
                torch.device(device),
                dtype,
            ).cpu()
            if len(t_schedule) - 1 <= 0 or denoise <= 0.0:
                if source_latent is not None:
                    out = source_latent.tensor
                elif noise_on_cpu:
                    primary = positive.to_entries()[0]
                    ref_cond = PreparedCondition(
                        encoder_hidden_states=primary.encoder_hidden_states,
                        encoder_attention_mask=primary.encoder_attention_mask,
                        context_latents=context_latents,
                    )
                    out = self._engine._prepare_noise_cpu(ref_cond, seed)
                else:
                    out = handler.model.prepare_noise(context_latents, seed)
                return {"latent": Latent(tensor=out)}

        # Route the solver's noise_curve to the solver-appropriate
        # request field. The other field stays None — SlotRequest keeps
        # both (ode_noise_curve, sde_denoise_curve) and the pipeline
        # ignores whichever doesn't apply to the active solver.
        # All other curve/latent/cfg inputs come out of the Modulation
        # bundle so StreamDenoise's surface stays minimal by default.
        request = _build_slot_request(
            positive=positive,
            negative=negative,
            context_latents=context_latents,
            source_latent=source_latent,
            seed=seed,
            denoise=denoise,
            velocity_scale=modulation.velocity_scale,
            sde_denoise_curve=(
                solver.noise_curve if solver.method == "sde" else None
            ),
            ode_noise_curve=(
                solver.noise_curve if solver.method == "ode" else None
            ),
            initial_noise_curve=modulation.initial_noise_curve,
            x0_target=modulation.x0_target,
            x0_target_curve=modulation.x0_target_curve,
            x0_target_strength=x0_target_strength,
            x0_target_gate=x0_target_gate,
            guidance_curve=modulation.guidance_curve,
            rcfg_mode=kwargs.get("rcfg_mode"),
            cfg_rescale=kwargs.get("cfg_rescale"),
            device=device,
            dtype=dtype,
        )

        with handler._load_model_context("model"):
            pipe.submit(request)
            if drain:
                # Synchronous drain — used by Generate for one-shot parity.
                result: Optional[torch.Tensor] = None
                infer_steps = len(t_schedule) - 1
                for _ in range(infer_steps + 2):
                    out = pipe.tick()
                    if out is not None:
                        result = out
                        break
                if result is None:
                    raise RuntimeError(
                        f"StreamDenoise drain produced no latent "
                        f"after {infer_steps + 2} ticks."
                    )
                return {"latent": Latent(tensor=result)}

            out = pipe.tick()

        return {
            "latent": Latent(tensor=out) if out is not None else None,
        }

    # ------------------------------------------------------------------
    # Streaming utility surface — used by Session.stream() callers.
    # Not node-graph API, but part of this node's public contract so the
    # realtime demo and tests can introspect the underlying pipeline
    # without reaching around the node.
    # ------------------------------------------------------------------

    @property
    def pipeline(self) -> Optional[StreamPipeline]:
        return self._pipeline

    @property
    def active_slots(self) -> int:
        return self._pipeline.active_slots if self._pipeline is not None else 0

    @property
    def tick_ms(self) -> float:
        return self._pipeline._last_tick_ms if self._pipeline is not None else 0.0

    def stats(self) -> dict:
        return self._pipeline.stats() if self._pipeline is not None else {}


@NodeRegistry.register
class Generate(BaseNode):
    """One-shot diffusion generation. Thin wrapper over ``StreamDenoise``.

    Runs a transient ``StreamDenoise`` with ``pipeline_depth=1`` and
    ``drain=True``, producing the finished latent synchronously.
    Positive conditioning is required. Negative conditioning + a
    ``guidance_curve`` enables CFG (APG). Source latent is required when
    ``denoise < 1.0`` (or via ``source_latent.mask`` for inpainting).
    """

    node_type_id: ClassVar[str] = "acestep.Generate"

    @classmethod
    def get_definition(cls) -> NodeDefinition:
        return NodeDefinition(
            node_type_id=cls.node_type_id,
            display_name="Generate",
            category="diffusion",
            description="Run the ACE-Step diffusion loop (one-shot drain).",
            inputs=(
                NodePort(name="model", type="MODEL"),
                NodePort(name="config", type="CONFIG"),
                NodePort(name="solver", type="SOLVER"),
                NodePort(name="positive", type="CONDITIONING"),
                NodePort(name="negative", type="CONDITIONING", required=False),
                NodePort(name="context_latent", type="LATENT", required=False),
                NodePort(name="source_latent", type="LATENT", required=False),
                NodePort(name="modulation", type="MODULATION", required=False),
                NodePort(
                    name="dcw", type="DCW", required=False,
                    description="Optional DCW bundle from DCWConfig; overrides upstream-default DCW settings.",
                ),
            ),
            outputs=(
                NodePort(name="latent", type="LATENT"),
            ),
            params=(
                NodeParam(
                    name="duration", type="number", default=60.0,
                    description="Latent duration (s)",
                    min=1.0, max=600.0, step=1.0,
                ),
            ),
        )

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        config_payload: Config = kwargs["config"]
        cfg = config_payload.config

        denoiser = StreamDenoise()
        return denoiser.execute(
            model=kwargs["model"],
            solver=kwargs["solver"],
            positive=kwargs["positive"],
            negative=kwargs.get("negative"),
            context_latent=kwargs.get("context_latent"),
            source_latent=kwargs.get("source_latent"),
            modulation=kwargs.get("modulation"),
            dcw=kwargs.get("dcw"),
            steps=cfg.infer_steps,
            shift=cfg.shift,
            seed=cfg.seed,
            denoise=cfg.denoise,
            noise_on_cpu=cfg.noise_on_cpu,
            use_cache=cfg.use_cache,
            pipeline_depth=1,
            drain=True,
            duration=kwargs.get("duration", 60.0),
        )
