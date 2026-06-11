"""StreamDiffusion-style pipeline for interactive ACE-Step generation.

Maintains a ring buffer of in-flight generations at different denoising
stages. Each tick(), one batched forward pass advances all slots. After
warmup, every tick produces a finished generation.

Supports per-slot denoise and source_latents for cover workflows where
the user adjusts the denoise knob in real time. When a TRT engine is
loaded on the DiffusionEngine, tick() routes through TensorRT
automatically.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Dict, NamedTuple, Optional, List, Tuple, TYPE_CHECKING

from loguru import logger
import torch

from .diffusion import DiffusionConfig, DiffusionEngine
from .model_adapter import ACEAdapter, ModelAdapter
from . import ode_steps
from .dcw import DCWAdvanced, DCWCorrector

if TYPE_CHECKING:
    from .masking import LatentNoiseMask


class _SteeringApply(NamedTuple):
    """One pre-resolved activation-steering shift bound to a layer."""
    vector: torch.Tensor   # 1-D [hidden_dim]
    scale: float           # alpha * magnitude
    step: int              # gate: only rows at this denoise step receive it


@dataclass
class SlotCondition:
    """One conditioning entry for multi-condition per-frame blending.

    A ``SlotRequest`` always has a primary condition (its
    ``encoder_hidden_states`` / ``encoder_attention_mask`` /
    ``primary_temporal_weight`` / ``primary_step_range``) and may carry
    additional conditions in ``extra_conditions``. At each step, the
    decoder runs once per active condition and the velocities are
    blended per-frame by ``temporal_weight`` (see
    ``ode_steps.blend_velocities``).
    """
    encoder_hidden_states: torch.Tensor    # [1, L, D]
    encoder_attention_mask: torch.Tensor   # [1, L]
    temporal_weight: Optional[torch.Tensor] = None   # [T], [1,T], or [1,T,1]
    step_range: Optional[Tuple[float, float]] = None  # (start_frac, end_frac)

    def is_active_at_step(self, step_idx: int, total_steps: int) -> bool:
        if self.step_range is None:
            return True
        if total_steps <= 0:
            return True
        progress = step_idx / total_steps
        return self.step_range[0] <= progress < self.step_range[1]


@dataclass
class SlotRequest:
    """A generation request to be fed into the pipeline.

    Holds the conditioning tensors and noise seed. All requests in a
    pipeline must share the same sequence length T (duration).

    Optional fields near the bottom (``x0_target_curve``,
    ``x0_target_gate``, ``initial_noise_curve``, ``latent_mask``,
    ``extra_conditions``, ``primary_temporal_weight``,
    ``primary_step_range``) were added so ``StreamPipeline`` can serve
    as the single diffusion primitive for both streaming and one-shot
    generation (Phase 1 of the diffusion-primitive unification).

    The three ACE-shaped conditioning tensors are Optional since the
    ModelAdapter seam: non-ACE families carry their conditioning in the
    opaque ``aux_cond`` bundle instead and declare T via
    ``latent_frames`` (see :mod:`acestep.engine.model_adapter`). ACE
    requests keep populating all three exactly as before.
    """
    encoder_hidden_states: Optional[torch.Tensor] = None  # [1, L, D]
    encoder_attention_mask: Optional[torch.Tensor] = None  # [1, L]
    context_latents: Optional[torch.Tensor] = None         # [1, T, D_ctx]
    # Either a single int (same noise for every row in the request's batch)
    # or a list of ints (one seed per row, matching the old
    # _prepare_noise_cpu contract used by DiffusionEngine.generate()).
    seed: Optional["int | List[int]"] = None
    source_latents: Optional[torch.Tensor] = None  # [1, T, D] for cover
    denoise: float = 1.0  # per-request denoise strength
    sde_denoise_curve: Optional[torch.Tensor] = None  # [1, T, 1] per-frame denoise
    velocity_scale: Optional[torch.Tensor] = None     # [1, T, 1] per-frame velocity scaling
    ode_noise_curve: Optional[torch.Tensor] = None    # [1, T, 1] per-step noise injection
    x0_target: Optional[torch.Tensor] = None         # [1, T, D] target latent for blending
    # Blend strength toward x0_target. Scalar (uniform across the timeline)
    # or per-frame curve; both flow through normalize_curve at the read
    # site so the engine sees a uniform [B, T, 1] tensor. Hot-mutable via
    # set_shared_curve("x0_target_strength", value).
    x0_target_strength: "float | torch.Tensor" = 0.0
    # --- New in Phase 1: absorb one-shot generate() features ---
    x0_target_curve: Optional[torch.Tensor] = None   # per-frame blend curve [T], [1,T], or [1,T,1]
    x0_target_gate: float = 0.0                       # gate-start fraction (matches DiffusionConfig default)
    initial_noise_curve: Optional[torch.Tensor] = None  # per-frame noise/source init mix
    latent_mask: Optional["LatentNoiseMask"] = None     # inpainting (two-sided x0 blend)
    extra_conditions: List[SlotCondition] = field(default_factory=list)
    primary_temporal_weight: Optional[torch.Tensor] = None
    primary_step_range: Optional[Tuple[float, float]] = None
    # --- CFG (Phase 2) ---
    # Flat list of negative conditions for classifier-free guidance. When
    # set together with ``guidance_curve``, each step runs a second forward
    # pass with the negative conditions and APG
    # (:func:`ode_steps.apg_forward`) blends the two velocities per frame.
    # Empty list disables CFG.
    neg_conditions: List[SlotCondition] = field(default_factory=list)
    guidance_curve: Optional[torch.Tensor] = None  # [T], [1,T], or [1,T,1]
    # APG momentum coefficient. Scalar (Python number) or per-frame curve;
    # both flow through normalize_curve at the apg_forward boundary so the
    # MomentumBuffer update sees a uniform tensor. Hot-mutable via
    # set_shared_curve("apg_momentum", value) on the pipeline.
    apg_momentum: "float | torch.Tensor" = -0.75
    # --- RCFG (Residual CFG, after StreamDiffusion §3.2) ---
    # Cuts the per-step uncond forward pass that standard CFG requires.
    # Modes:
    #   None / "full" : standard two-pass CFG. Runs a negative forward
    #                   every step (existing behavior).
    #   "initialize"  : run the uncond pass once at step 0 per slot, cache
    #                   the resulting velocity, reuse it as the negative
    #                   for all remaining steps of that slot. One extra
    #                   forward per slot, not per step.
    #   "self"        : skip the uncond forward entirely; approximate
    #                   ``v_uncond`` with the slot's initial noise tensor.
    #                   In flow matching ``v = noise - x0``, so with the
    #                   prior x0_uncond ~ 0 we have v_uncond ~ noise.
    #                   Zero extra forwards.
    # ``guidance_curve`` is still required (sets the APG scale). For
    # "self" mode ``neg_conditions`` is unused; for "initialize" the
    # initial uncond pass uses them just like full CFG.
    rcfg_mode: Optional[str] = None
    # ``cfg_rescale_curve`` blends the APG output's per-frame norm back
    # toward ``vt_pos``'s norm (Lin et al. "Common Diffusion Noise
    # Schedules and Sample Steps are Flawed"). ``None`` disables;
    # otherwise a scalar or ``[1, T, 1]`` curve in [0, 1] where 0 keeps
    # raw APG output and 1 fully snaps magnitude back to ``vt_pos``.
    cfg_rescale_curve: "Optional[float | torch.Tensor]" = None
    # --- Tier 2 (ModelAdapter seam) ---
    # Opaque per-request conditioning bundle for non-ACE families. The
    # pipeline never inspects it; only the family's adapter does (one
    # bundle per request — single-condition v1, per the canonical SA3
    # plan §5). ACE requests leave it None.
    aux_cond: Optional[dict] = None
    # Latent frame count T for families without ``context_latents``
    # (the historical T source). Ignored when ``context_latents`` is
    # present — the adapter's ``request_frames`` decides.
    latent_frames: Optional[int] = None

    def all_conditions(self) -> List[SlotCondition]:
        """Return primary + extra conditions as a single ordered list."""
        primary = SlotCondition(
            encoder_hidden_states=self.encoder_hidden_states,
            encoder_attention_mask=self.encoder_attention_mask,
            temporal_weight=self.primary_temporal_weight,
            step_range=self.primary_step_range,
        )
        if not self.extra_conditions:
            return [primary]
        return [primary] + list(self.extra_conditions)

    @property
    def has_cfg(self) -> bool:
        """True when this request wants APG guidance applied each step.

        Three families satisfy this:
        - Standard CFG: ``neg_conditions`` + ``guidance_curve``.
        - RCFG-initialize: ``neg_conditions`` + ``guidance_curve`` +
          ``rcfg_mode == 'initialize'`` (same inputs as standard, but
          the uncond pass only runs at step 0).
        - RCFG-self: ``guidance_curve`` + ``rcfg_mode == 'self'`` (no
          ``neg_conditions`` needed — uncond is the slot's initial noise).
        """
        if self.guidance_curve is None:
            return False
        if self.rcfg_mode == "self":
            return True
        return bool(self.neg_conditions)

    def needs_neg_forward(self, step_idx: int) -> bool:
        """True when this step requires running the uncond forward pass.

        - "self": never (virtual negative).
        - "initialize": only at step 0; subsequent steps reuse the
          slot's cached velocity.
        - None / "full": every step.
        """
        if not self.has_cfg:
            return False
        if self.rcfg_mode == "self":
            return False
        if self.rcfg_mode == "initialize":
            return step_idx == 0
        return True


@dataclass
class _Slot:
    """Internal state for one pipeline slot."""
    request: SlotRequest
    xt: torch.Tensor          # [1, T, D] current noisy latent
    t_schedule: torch.Tensor  # per-slot timestep schedule (on CPU)
    step_idx: int = 0         # which denoising step we're on (0-indexed)
    # APG momentum accumulator, one per slot with CFG. None for slots
    # without CFG (cheaper than allocating an unused buffer).
    momentum_buffer: Optional[ode_steps.MomentumBuffer] = None
    # RCFG state. ``initial_noise`` is captured at slot init and used as
    # the virtual ``v_uncond`` for ``rcfg_mode == 'self'``. ``vt_neg_cached``
    # holds the cached uncond velocity for ``rcfg_mode == 'initialize'`` —
    # populated on step 0, reused on every subsequent step of this slot.
    initial_noise: Optional[torch.Tensor] = None
    vt_neg_cached: Optional[torch.Tensor] = None


class StreamPipeline:
    """StreamDiffusion-style batched denoising pipeline.

    Pipeline depth = number of denoising steps. After warmup (depth
    ticks), every tick() returns a finished latent.

    Each slot carries its own timestep schedule derived from its
    denoise value, so the user can change denoise between submissions
    and each in-flight generation uses the schedule it was born with.

    When the DiffusionEngine has a TRT engine loaded, tick() uses
    TensorRT for the batched forward pass automatically.

    This is a low-level primitive. The ``StreamDenoise`` node in
    ``acestep.nodes.diffusion_nodes`` is the canonical way to drive
    it — there should be exactly one call site constructing this
    class anywhere in ``acestep/``.
    """

    def __init__(
        self,
        engine: Optional[DiffusionEngine],
        config: DiffusionConfig,
        pipeline_depth: Optional[int] = None,
        adapter: Optional[ModelAdapter] = None,
    ):
        """``adapter`` selects the model family behind the Tier-2 seam
        (:mod:`acestep.engine.model_adapter`). ``None`` (every existing
        call site) builds the default :class:`ACEAdapter` over
        ``engine``. ``engine`` may be ``None`` only when a non-default
        adapter owns its model end-to-end (e.g. SA3); the TRT dispatch
        state below is ACE machinery and stays engine-bound."""
        self.engine = engine
        self.decoder = engine.decoder if engine is not None else None
        self.model = engine.model if engine is not None else None
        self.config = config
        self.adapter: ModelAdapter = (
            adapter if adapter is not None else ACEAdapter(self)
        )
        if engine is None and adapter is None:
            raise ValueError(
                "StreamPipeline needs a DiffusionEngine for the default "
                "ACE adapter; pass adapter= for engine-less families"
            )

        # Decouple ring buffer depth from denoising step count.
        # Default: depth = infer_steps (classic StreamDiffusion).
        self._depth: int = pipeline_depth if pipeline_depth is not None else config.infer_steps

        # Pipeline state
        self._slots: List[Optional[_Slot]] = [None] * self._depth
        self._queue: List[SlotRequest] = []

        # Cached device/dtype (set on first submit)
        self._device: Optional[torch.device] = None
        self._dtype: Optional[torch.dtype] = None

        # Schedule cache: denoise -> cpu tensor
        self._schedule_cache: dict[float, torch.Tensor] = {}

        # TRT state (mirrors DiffusionEngine pattern). Snapshotted from
        # the engine here, refreshed on profile swaps via the
        # engine-swap listener registered below. Engine-less pipelines
        # (non-ACE adapters) carry the null snapshot — their adapter
        # never dispatches TRT through the pipeline.
        self._trt_ctx = engine._trt_ctx if engine is not None else None
        self._trt_stream = engine._trt_stream if engine is not None else None
        self._trt_engine = engine._trt_engine if engine is not None else None
        self._trt_io_dtype = getattr(engine, '_trt_io_dtype', torch.float32)
        self._trt_input_dtypes = getattr(engine, "_trt_input_dtypes", {}) or {}
        self._trt_output_dtype = getattr(engine, "_trt_output_dtype", self._trt_io_dtype)
        # Steering shape (constants per engine); snapshotted so the
        # per-tick buffer fill doesn't re-query TRT.
        self._steering_num_layers = getattr(engine, "_steering_num_layers", 0)
        self._steering_hidden_size = getattr(engine, "_steering_hidden_size", 0)

        # Currently-bound TRT I/O buffers (set by _ensure_trt_bufs to one
        # entry of _trt_bufs_cache). _trt_forward reads these directly.
        self._trt_bufs: Optional[dict] = None
        self._trt_out_buf: Optional[torch.Tensor] = None
        # LRU cache of (B, eff_T, max_L) -> {bufs..., "_out_buf": tensor}.
        # CFG/RCFG passes alternate between pos and neg encoder lengths
        # (e.g. L=83 pos, L=66 empty-prompt neg), so a single-shape cache
        # thrashes on every forward. 4 entries comfortably covers
        # {pos, neg} × {two T values} during T transitions.
        self._trt_bufs_cache: "OrderedDict[tuple, dict]" = OrderedDict()
        self._trt_bufs_cache_max = 4

        # Re-pick up the snapshot after each profile swap. The new
        # engine has different I/O profile bounds and its execution
        # context owns different tensor addresses, so the previous
        # ``_trt_bufs`` cache is invalidated and rebuilt on the next
        # forward pass via :meth:`_ensure_trt_bufs`.
        if hasattr(engine, "add_engine_swap_listener"):
            engine.add_engine_swap_listener(self._on_engine_swapped)

        # Shared mutable curves: when a name is present, the corresponding
        # per-slot field on every in-flight SlotRequest is overridden for
        # that slot's next tick. Bypasses the ring-buffer drain, giving
        # 1-tick latency. Invariant: every value is a normalized
        # ``[1, T, 1]`` tensor (scalar-or-per-frame multiplier broadcast
        # against ``[B, T, *]`` operands). Floats auto-lift to ``[1, 1, 1]``
        # at the setter so callers can pass scalars without thinking
        # about shape.
        self._shared_curves: dict[str, torch.Tensor] = {}

        # Channel guidance: a ``[1, T, 64]`` per-channel gain applied to
        # ``xt`` before each forward pass. Lives in its own field rather
        # than ``_shared_curves`` because its shape (per-channel) breaks
        # the dict's per-frame invariant. Updated via
        # :meth:`set_channel_guidance` / :meth:`set_channel_gain_tensor`,
        # pre-cast to the pipeline's device/dtype so the hot-path
        # ``.to(...)`` is a no-op.
        self._channel_gain: Optional[torch.Tensor] = None

        # Activation steering. Per-DiT-layer additive shift on the
        # post-block residual, gated per-row by denoise step.
        # ``_current_step_per_row`` is populated by _tick_complex_pt
        # around each forward; empty means "skip injection" so a
        # forward issued outside the rendezvous can't fire steering.
        self._steering_by_layer: Dict[int, List[_SteeringApply]] = {}
        self._steering_hooks_installed: bool = False
        self._current_step_per_row: List[int] = []

        # Sentinel tensors for the "always-on multiply" idiom in the step
        # helpers. Built lazily once the first slot's device/dtype is known.
        # ``_ones_3d`` stands in for absent ``velocity_scale`` (vt * 1 = vt).
        # ``_zeros_3d`` stands in for absent ``ode_noise_curve`` (noise * 0 = 0).
        # The sentinels keep the compiled ODE/SDE step graphs branch-free.
        self._ones_3d: Optional[torch.Tensor] = None
        self._zeros_3d: Optional[torch.Tensor] = None

        # Compiled per-step helper cache. Populated lazily by
        # ``_get_compiled`` on first use so we don't pay the
        # ``torch.compile`` warmup on pipelines that never run their
        # PyTorch path (TRT-only streams). Gated on engine.compile_loops —
        # when the engine was constructed with ``compile_loops=False``,
        # primitives run eagerly (still branch-free, just not compiled).
        self._compile_loops: bool = (
            getattr(engine, "_compile_loops", True) if engine is not None
            else False  # engine-less families opt into compile later
        )
        self._compiled_cache: dict[Callable, Callable] = {}

        # DCW (Differential Correction in Wavelet domain) — post-step
        # sampler correction. Always constructed; ``is_active`` short-
        # circuits in the hot path when disabled. Hot-updatable via
        # :meth:`set_dcw` without rebuilding the pipeline.
        self._dcw_corrector: DCWCorrector = DCWCorrector(
            enabled=config.dcw_enabled,
            mode=config.dcw_mode,
            scaler=config.dcw_scaler,
            high_scaler=config.dcw_high_scaler,
            wavelet=config.dcw_wavelet,
            advanced=config.dcw_advanced,
        )

        # Stats
        self.ticks: int = 0
        self._last_tick_ms: float = 0.0

    @property
    def depth(self) -> int:
        return self._depth

    @property
    def active_slots(self) -> int:
        return sum(1 for s in self._slots if s is not None)

    @property
    def is_warmed_up(self) -> bool:
        """True when all slots are occupied (steady state)."""
        return all(s is not None for s in self._slots)

    @property
    def has_trt(self) -> bool:
        return self._trt_engine is not None

    def _on_engine_swapped(self) -> None:
        """Re-snapshot TRT refs and drop the stale buffer cache.

        Wired up in __init__ via ``engine.add_engine_swap_listener``.
        Fires on the runner thread because the profile manager runs the
        swap inside the streaming pipeline's ``before_tick`` rendezvous,
        so no concurrent ``tick()`` can be reading these fields.

        ``_trt_bufs`` is invalidated rather than reallocated: the next
        forward pass calls :meth:`_ensure_trt_bufs` with the live
        ``(B, T, max_L)`` shape, which binds against the new engine's
        profile and only allocates if the shape actually differs from
        the now-discarded one.
        """
        engine = self.engine
        self._trt_engine = engine._trt_engine
        self._trt_ctx = engine._trt_ctx
        self._trt_stream = engine._trt_stream
        self._trt_io_dtype = getattr(engine, "_trt_io_dtype", torch.float32)
        self._trt_input_dtypes = getattr(engine, "_trt_input_dtypes", {})
        self._trt_output_dtype = getattr(
            engine, "_trt_output_dtype", self._trt_io_dtype
        )
        self._steering_num_layers = getattr(engine, "_steering_num_layers", 0)
        self._steering_hidden_size = getattr(engine, "_steering_hidden_size", 0)
        self._trt_bufs = None
        self._trt_out_buf = None
        self._trt_bufs_cache.clear()
        # Hooks live on the old decoder.layers; the new decoder needs a
        # fresh install on the next non-empty set_steering call.
        self._steering_hooks_installed = False

    def submit(self, request: SlotRequest) -> None:
        """Enqueue a generation request.

        The queue is capped at ``_depth`` items.  When the caller submits
        faster than the pipeline consumes (always the case when
        depth < infer_steps), the oldest queued request is dropped so
        that fresh parameters reach the ring buffer promptly instead of
        sitting behind an ever-growing backlog of stale requests.
        """
        if len(self._queue) >= self._depth:
            self._queue.pop(0)
        self._queue.append(request)

    def _get_schedule(self, denoise: float) -> torch.Tensor:
        """Get (cached) timestep schedule for a given denoise value.

        Schedule construction is family knowledge (ACE flow-matching
        ``shift`` warp vs SA3 LogSNR warp), so it lives on the adapter;
        the cache stays here.
        """
        if denoise not in self._schedule_cache:
            self._schedule_cache[denoise] = self.adapter.build_schedule(
                self.config, denoise, self._device, self._dtype
            ).cpu()
        return self._schedule_cache[denoise]

    def _ensure_device(self, device: torch.device, dtype: torch.dtype):
        if self._device is None:
            self._device = device
            self._dtype = dtype

    def _make_noise(self, request: SlotRequest) -> torch.Tensor:
        """Generate initial noise for a request.

        ``request.seed`` accepts the same shapes as the old
        ``DiffusionEngine._prepare_noise_cpu``:
            - ``None``                 → fresh RNG state per call
            - ``int``                  → scalar manual_seed, one randn call
            - ``List[int]``            → per-row seeding, one row per seed
                                         (rows with seed < 0 reuse the
                                         current RNG state; matches upstream
                                         behavior for "don't reseed")

        Layout matches ComfyUI's RandomNoise node when noise_on_cpu is set:
        generate in [B,D,T] on CPU, then transpose to [B,T,D] and move to
        the pipeline's device/dtype.

        T and the channel count are family knowledge (Tier-2 seam): the
        historical ``context_latents.shape[-1] // 2`` is the ACE
        ``src ++ chunk_mask`` convention, now ``adapter.latent_channels``.
        """
        T = self.adapter.request_frames(request)
        D = self.adapter.latent_channels
        seed = request.seed

        cpu = self.config.noise_on_cpu
        gen_device = "cpu" if cpu else self._device
        gen_dtype = torch.float32 if cpu else self._dtype

        if isinstance(seed, list):
            rows = []
            for s in seed:
                if s is not None and s >= 0:
                    torch.manual_seed(int(s))
                if cpu:
                    rows.append(torch.randn(1, D, T, device=gen_device, dtype=gen_dtype))
                else:
                    rows.append(torch.randn(1, T, D, device=gen_device, dtype=gen_dtype))
            stacked = torch.cat(rows, dim=0)
            if cpu:
                stacked = stacked.movedim(-1, -2)
            return stacked.to(device=self._device, dtype=self._dtype)

        if seed is not None:
            torch.manual_seed(int(seed))

        if cpu:
            noise_bdt = torch.randn(1, D, T, device="cpu", dtype=torch.float32)
            return noise_bdt.movedim(-1, -2).to(
                device=self._device, dtype=self._dtype
            )
        return torch.randn(
            1, T, D, device=self._device, dtype=self._dtype
        )

    def _init_slot(self, request: SlotRequest) -> _Slot:
        """Create a new slot from a request, initialized at step 0."""
        self._ensure_device(*self.adapter.request_device_dtype(request))

        t_schedule = self._get_schedule(request.denoise)
        noise = self._make_noise(request)

        t_start = t_schedule[0].item()

        # Resolve the "clean" latent for partial-denoise init. Old generate()
        # fell back to latent_mask.original_latents when no explicit
        # source_latents was provided (inpainting flows rely on this).
        src_clean = request.source_latents
        if src_clean is None and request.latent_mask is not None:
            src_clean = request.latent_mask.original_latents

        if request.initial_noise_curve is not None and src_clean is not None:
            # Per-frame noise/source mixing for the initial state. Ignores
            # the denoise sigma in favor of the explicit per-frame curve —
            # 1.0 = pure noise, 0.0 = pure source.
            curve = ode_steps.normalize_curve(request.initial_noise_curve).to(
                device=self._device, dtype=self._dtype,
            )
            xt = curve * noise + (1.0 - curve) * src_clean
        elif src_clean is not None and request.denoise < 1.0:
            xt = t_start * noise + (1.0 - t_start) * src_clean
        else:
            xt = noise.clone()

        momentum_buffer = (
            ode_steps.MomentumBuffer() if request.has_cfg else None
        )

        # RCFG-self uses the slot's initial noise tensor as the virtual
        # ``v_uncond``. Captured once at slot init; lives on the slot for
        # the rest of its schedule. Only allocated for ``rcfg_mode ==
        # "self"`` — other modes never read this field.
        initial_noise = (
            noise.clone() if request.rcfg_mode == "self" else None
        )

        return _Slot(
            request=request, xt=xt,
            t_schedule=t_schedule, step_idx=0,
            momentum_buffer=momentum_buffer,
            initial_noise=initial_noise,
        )

    # ------------------------------------------------------------------
    # Sentinel tensors + compiled step helpers (PyTorch backend)
    # ------------------------------------------------------------------

    def _ensure_sentinels(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Lazy-build broadcast-safe ones/zeros sentinels.

        Constructed once the pipeline's device/dtype are known (set by the
        first ``submit``/``generate`` call) and reused for the lifetime of
        the pipeline. Shape is ``[1, 1, 1]`` so they broadcast cleanly
        against ``xt`` and ``vt`` (``[1, T, D]``) without per-step allocs.
        """
        if self._ones_3d is None:
            self._ones_3d = torch.ones(
                1, 1, 1, device=self._device, dtype=self._dtype,
            )
            self._zeros_3d = torch.zeros(
                1, 1, 1, device=self._device, dtype=self._dtype,
            )
        return self._ones_3d, self._zeros_3d

    # Compiled wrappers over the pure step primitives in ``ode_steps``.
    # The primitives themselves are branch-free and reference no ``self``,
    # so ``torch.compile`` can trace each into a single fused graph. The
    # sentinel-tensor idiom (``ones_3d`` for absent velocity_scale,
    # ``zeros_3d`` for absent ode_noise_curve) keeps the ODE graph flat
    # without ``is None`` branches — the multiply is a byte-identical
    # no-op but lets the compiler specialize one straight-line kernel.

    def _get_compiled(self, fn: Callable) -> Callable:
        """Return a (possibly compiled) wrapper around a pure step primitive.

        Compilation is lazy — we only pay the inductor cost on PT
        pipelines that actually exercise the primitive. ``dynamic=True``
        lets the graph accept varying T / dtype without re-tracing on
        every shape change. Results are memoized per primitive on
        ``self._compiled_cache``.
        """
        cache = self._compiled_cache
        cached = cache.get(fn)
        if cached is not None:
            return cached
        compiled = fn
        if self._compile_loops:
            try:
                compiled = torch.compile(fn, backend="inductor", dynamic=True)
            except Exception as e:  # pragma: no cover - fallback path
                logger.warning(
                    "torch.compile({}) failed ({}); falling back to eager",
                    fn.__name__, e,
                )
        cache[fn] = compiled
        return compiled

    # ------------------------------------------------------------------
    # Per-slot curve resolution (reads shared overrides first, then slot)
    # ------------------------------------------------------------------

    def _resolve_slot_curves(
        self,
        slot: "_Slot",
        vt: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """Resolve per-slot curve tensors, preferring shared overrides.

        Returns ``(vs, sdc, onc)``:

        - ``vs`` is the velocity_scale curve broadcast-ready against
          ``vt``; returns the ``ones_3d`` sentinel when absent so the
          compiled step helpers can ``vt * vs`` branch-free.
        - ``sdc`` is the normalized sde_denoise_curve or ``None``.
          Callers gate different integration paths on its presence +
          ``source_latents``, so the sentinel idiom doesn't apply.
        - ``onc`` is the normalized ode_noise_curve broadcast-ready
          against ``slot.xt``; returns the ``zeros_3d`` sentinel when
          absent so the post-step ``randn * onc`` injection is a
          branch-free no-op.

        Byte-equivalent to the inline curve-resolution blocks that
        previously lived in both tick paths.
        """
        ones_3d, zeros_3d = self._ensure_sentinels()

        eff_vs = self._eff_shared(slot, "velocity_scale")
        eff_sdc = self._eff_shared(slot, "sde_denoise_curve")
        eff_onc = self._eff_shared(slot, "ode_noise_curve")

        vs = (
            eff_vs.to(device=vt.device, dtype=vt.dtype)
            if eff_vs is not None else ones_3d
        )
        sdc = (
            eff_sdc.to(device=slot.xt.device, dtype=slot.xt.dtype)
            if eff_sdc is not None else None
        )
        onc = (
            eff_onc.to(device=slot.xt.device, dtype=slot.xt.dtype)
            if eff_onc is not None else zeros_3d
        )
        return vs, sdc, onc

    # The batched decoder forward lives on the family's ModelAdapter
    # (``self.adapter.batched_forward``) — moved there verbatim from
    # the old ``_decoder_forward`` so conditioning batching is family
    # knowledge (Tier-2 seam). The TRT dispatch machinery below stays
    # pipeline-owned: it is ACE engine state, reached only via the
    # ACEAdapter (relocating it is Phase-4 acceleration-contract work).

    def _trt_forward(
        self,
        xt_batch: torch.Tensor,
        timestep_list: List[float],
        enc_batch: torch.Tensor,
        ctx_batch: torch.Tensor,
    ) -> torch.Tensor:
        """Run one batched TRT forward pass on pre-built tensors.

        Uses the shape-keyed buffer cache via :meth:`_ensure_trt_bufs`
        so repeated ticks with the same ``(B, T, max_L)`` shape reuse
        allocations.

        The active ring-buffer rows are submitted as one TensorRT
        execution. If ``B`` exceeds the loaded engine profile, TensorRT
        rejects the shape instead of silently falling back to sequential
        per-row dispatch.
        """
        B, T, _ = xt_batch.shape
        if len(timestep_list) != B:
            raise RuntimeError(
                "TRT stream batch mismatch: "
                f"hidden_states batch={B}, timesteps={len(timestep_list)}"
            )
        if enc_batch.shape[0] != B or ctx_batch.shape[0] != B:
            raise RuntimeError(
                "TRT stream batch mismatch: "
                f"hidden_states={tuple(xt_batch.shape)}, "
                f"encoder={tuple(enc_batch.shape)}, "
                f"context={tuple(ctx_batch.shape)}"
            )

        max_L = enc_batch.shape[1]

        self._ensure_trt_bufs(B, T, max_L)
        bufs = self._trt_bufs
        pad = T % 2 == 1

        # hidden_states: cast to engine I/O dtype; pad odd T with zeros.
        xt_io = xt_batch.to(bufs["hidden_states"].dtype)
        if pad:
            bufs["hidden_states"][:, :T, :].copy_(xt_io)
            bufs["hidden_states"][:, T:, :].zero_()
        else:
            bufs["hidden_states"].copy_(xt_io)

        # timestep: one scalar per row.
        for i, t in enumerate(timestep_list):
            bufs["timestep"][i] = t

        # encoder_hidden_states: already padded to max_L + catted by
        # the caller. The engine has no ``encoder_attention_mask``
        # input; padding is handled by zero-value convention.
        bufs["encoder_hidden_states"].copy_(
            enc_batch.to(bufs["encoder_hidden_states"].dtype)
        )

        # context_latents: pad odd T with zeros.
        ctx_io = ctx_batch.to(bufs["context_latents"].dtype)
        if pad:
            bufs["context_latents"][:, :T, :].copy_(ctx_io)
            bufs["context_latents"][:, T:, :].zero_()
        else:
            bufs["context_latents"].copy_(ctx_io)

        # Steering: absent on non-spectral engines.
        if "steering" in bufs:
            self._fill_trt_steering_buffer(bufs["steering"], B)

        # Rebind and execute.
        ctx = self._trt_ctx
        for name, buf in bufs.items():
            if name.startswith("_"):
                continue
            if not ctx.set_tensor_address(name, buf.data_ptr()):
                raise RuntimeError(f"TRT decoder rejected input address for {name}")
        if not ctx.set_tensor_address("velocity", self._trt_out_buf.data_ptr()):
            raise RuntimeError("TRT decoder rejected output address for velocity")

        if not ctx.execute_async_v3(self._trt_stream.ptr):
            raise RuntimeError("TRT decoder execution failed")
        self._trt_stream.synchronize()

        out = self._trt_out_buf
        if pad:
            return out[:, :T, :].to(self._dtype)
        return out.to(self._dtype)

    def _steering_row_mask(self, target_step: int, B: int) -> Optional[List[int]]:
        """Rows whose slot is at ``target_step``, or None to skip.

        Returns None when no per-row step mapping exists or it
        disagrees with ``B`` — the eager hook and TRT buffer fill both
        skip injection in that case rather than firing blindly.
        """
        row_steps = self._current_step_per_row
        if not row_steps or len(row_steps) != B:
            return None
        mask = [i for i, s in enumerate(row_steps) if s == target_step]
        return mask or None

    def _fill_trt_steering_buffer(self, buf: torch.Tensor, B: int) -> None:
        """Populate the TRT steering buffer for one forward.

        ``buf`` is ``[B, num_layers, hidden_size]``; zeroed first so
        previous-tick content doesn't leak. Rows with no matching shift
        stay zero, which the engine adds as a no-op per layer.
        """
        buf.zero_()
        if not self._steering_by_layer:
            return
        for layer_idx, applies in self._steering_by_layer.items():
            if layer_idx < 0 or layer_idx >= self._steering_num_layers:
                continue
            for apply in applies:
                mask_rows = self._steering_row_mask(apply.step, B)
                if mask_rows is None:
                    continue
                v = apply.vector.to(device=buf.device, dtype=buf.dtype)
                buf[mask_rows, layer_idx, :] += apply.scale * v

    # ------------------------------------------------------------------
    # TRT buffer management
    # ------------------------------------------------------------------

    def _ensure_trt_bufs(self, B: int, T: int, max_L: int):
        """Bind TRT I/O buffers for ``(B, T, max_L)`` via the LRU cache.

        Reuses an existing cache entry when the shape has been seen
        recently; allocates a new entry (evicting the oldest if the
        cache is full) otherwise. The TRT execution context still has
        to be re-bound to whichever entry we use, because addresses
        change as we swap entries — but allocations only happen on a
        true miss. Uses the engine's native I/O dtype.
        """
        eff_T = T + 1 if T % 2 == 1 else T
        key = (B, eff_T, max_L)
        ctx = self._trt_ctx

        cached = self._trt_bufs_cache.get(key)
        if cached is not None:
            self._trt_bufs_cache.move_to_end(key)
            for name, buf in cached.items():
                if name.startswith("_"):
                    continue
                ctx.set_input_shape(name, tuple(buf.shape))
                ctx.set_tensor_address(name, buf.data_ptr())
            ctx.set_tensor_address("velocity", cached["_out_buf"].data_ptr())
            self._trt_bufs = cached
            self._trt_out_buf = cached["_out_buf"]
            return

        device = self._device
        io_dtype = self._trt_io_dtype
        in_dtypes = self._trt_input_dtypes
        bufs = {
            "hidden_states": torch.empty(
                B, eff_T, 64,
                dtype=in_dtypes.get("hidden_states", io_dtype),
                device=device,
            ),
            "timestep": torch.empty(
                B,
                dtype=in_dtypes.get("timestep", torch.float32),
                device=device,
            ),
            "encoder_hidden_states": torch.empty(
                B, max_L, 2048,
                dtype=in_dtypes.get("encoder_hidden_states", io_dtype),
                device=device,
            ),
            "context_latents": torch.empty(
                B, eff_T, 128,
                dtype=in_dtypes.get("context_latents", io_dtype),
                device=device,
            ),
        }
        if self._steering_num_layers > 0:
            # Zeroed so a tick with no active configs is a true no-op
            # (engine still adds zeros per layer); repopulated each
            # forward by _trt_forward.
            bufs["steering"] = torch.zeros(
                B, self._steering_num_layers, self._steering_hidden_size,
                dtype=in_dtypes.get("steering", io_dtype),
                device=device,
            )

        for name, buf in bufs.items():
            if not ctx.set_input_shape(name, tuple(buf.shape)):
                raise RuntimeError(
                    f"TRT decoder rejected input shape for {name}: "
                    f"{tuple(buf.shape)}"
                )
            if not ctx.set_tensor_address(name, buf.data_ptr()):
                raise RuntimeError(f"TRT decoder rejected input address for {name}")

        missing = ctx.infer_shapes()
        if missing:
            raise RuntimeError(
                f"TRT decoder shapes are insufficiently specified: {missing}"
            )

        out_shape = tuple(ctx.get_tensor_shape("velocity"))
        if any(d < 0 for d in out_shape):
            raise RuntimeError(
                f"TRT output shape unresolved: {out_shape}. "
                f"B={B}, eff_T={eff_T}, L={max_L}"
            )
        out_buf = torch.empty(out_shape, dtype=self._trt_output_dtype, device=device)
        if not ctx.set_tensor_address("velocity", out_buf.data_ptr()):
            raise RuntimeError("TRT decoder rejected output address for velocity")

        bufs["_key"] = key
        bufs["_eff_T"] = eff_T
        bufs["_T"] = T
        bufs["_out_buf"] = out_buf
        self._trt_bufs_cache[key] = bufs
        while len(self._trt_bufs_cache) > self._trt_bufs_cache_max:
            self._trt_bufs_cache.popitem(last=False)
        self._trt_bufs = bufs
        self._trt_out_buf = out_buf

        logger.debug(
            "Stream TRT bufs allocated: B={} eff_T={} L={}", B, eff_T, max_L
        )

    # ------------------------------------------------------------------
    # Main tick
    # ------------------------------------------------------------------

    @torch.no_grad()
    def tick(self) -> Optional[torch.Tensor]:
        """Run one batched forward pass, advancing all active slots.

        Returns:
            Finished latent [1, T, D] if a slot completed, else None.

        Dispatches to :meth:`_tick_pt`, the single unified per-tick
        path. The PT path handles every feature combination — mask
        pre/post blending, multi-condition temporal blending, CFG
        (APG), per-frame curves (velocity_scale, sde_denoise,
        ode_noise), x0_target blending (scalar or per-frame curve),
        and both ODE and SDE solvers — by composing pure step
        primitives from :mod:`ode_steps`. The TRT backend is selected
        inside the family adapter's forward
        (:meth:`acestep.engine.model_adapter.ACEAdapter.batched_forward`).
        """
        tick_start = time.time()

        # T-coherence: a source swap to a different-length audio leaves
        # in-flight slots holding xt of the old T while fresh submits
        # carry the new T. _tick_pt cats xt across slots in dim 0, which
        # requires the time dim to match — drop stale slots and stale
        # queued requests so the next batch is uniform. The "target" T
        # is the most recently submitted request's; older queued ones
        # from before the swap are filtered out alongside the slots.
        if self._queue:
            target_T = self.adapter.request_frames(self._queue[-1])
            if any(
                self.adapter.request_frames(r) != target_T for r in self._queue
            ):
                self._queue = [
                    r for r in self._queue
                    if self.adapter.request_frames(r) == target_T
                ]
            for i, slot in enumerate(self._slots):
                if slot is not None and slot.xt.shape[1] != target_T:
                    self._slots[i] = None

        # Check for finished slot (slot at final step of its schedule)
        finished = None
        for i, slot in enumerate(self._slots):
            if slot is not None and slot.step_idx >= len(slot.t_schedule) - 1:
                finished = slot.xt
                self._slots[i] = None
                break

        # Fill empty slots from queue
        for i, slot in enumerate(self._slots):
            if slot is None and self._queue:
                req = self._queue.pop(0)
                self._slots[i] = self._init_slot(req)

        # Collect active slots (exclude completed)
        active = [
            (i, s) for i, s in enumerate(self._slots)
            if s is not None and s.step_idx < len(s.t_schedule) - 1
        ]
        if not active:
            self._last_tick_ms = (time.time() - tick_start) * 1000
            self.ticks += 1
            return finished

        indices, slots = zip(*active)
        self._tick_pt(slots, indices)

        self._last_tick_ms = (time.time() - tick_start) * 1000
        self.ticks += 1

        return finished

    # ------------------------------------------------------------------
    # Unified tick — all features compose via pure ode_steps bricks
    # ------------------------------------------------------------------

    def _active_conditions(self, slot) -> List[SlotCondition]:
        """Return positive conditions active at this slot's current step.

        Falls back to the primary condition if no condition's
        ``step_range`` is satisfied, so the decoder always has at least
        one forward pass per slot.
        """
        total_steps = len(slot.t_schedule) - 1
        conds = slot.request.all_conditions()
        active = [c for c in conds if c.is_active_at_step(slot.step_idx, total_steps)]
        return active if active else [conds[0]]

    def _active_neg_conditions(self, slot) -> List[SlotCondition]:
        """Return negative conditions active at this slot's current step.

        Falls back to ``neg_conditions[0]`` when none match — mirrors the
        pre-refactor behavior of ``negative_condition_set``. Returns an
        empty list when CFG is not enabled on the slot.
        """
        if not slot.request.has_cfg:
            return []
        total_steps = len(slot.t_schedule) - 1
        negs = slot.request.neg_conditions
        active = [c for c in negs if c.is_active_at_step(slot.step_idx, total_steps)]
        return active if active else [negs[0]]

    def _tick_pt(self, slots, indices) -> None:
        """Unified PyTorch tick — handles every feature combination.

        For each slot:

        1. ``latent_mask`` pre-blend on xt (integration base) and
           channel-gain on the decoder input.
        2. One forward pass per active (slot, positive cond) pair,
           concatenated into a single batched call. When any slot has
           CFG, a second forward pass runs for the negative conds.
        3. Per-slot velocity assembly: multi-condition temporal
           blending, then APG if CFG is active.
        4. Per-slot integration:
           - **Fast path** (no blends, no SDE, mid-schedule step):
             direct :func:`ode_steps.step_ode_euler` — byte-identical
             to the pre-refactor ``_step_simple_ode`` helper.
           - **x0-blend path** (any of: latent_mask, multi-cond,
             x0_target, CFG, SDE, final step): compute ``x0_pred``,
             apply mask-post / x0-target blends, then dispatch to
             :func:`ode_steps.step_ode_euler` (via a synthesized
             ``v_blended``), :func:`ode_steps.step_sde_curve`, or
             :func:`ode_steps.step_sde_renoise`.
        """
        # --- Per-slot preprocessing ---
        # Two tensors per slot:
        #   xt_mask      : mask_pre_blend applied (or raw xt if no mask).
        #                   This is the integration base for ODE/SDE math;
        #                   channel gain intentionally NOT applied so it
        #                   does not compound across steps.
        #   xt_decoder   : xt_mask * channel_gain (or xt_mask if no gain).
        #                   This is what the decoder sees; channel gain is
        #                   model-input scaling, not state.
        xt_mask_list: List[torch.Tensor] = []
        xt_decoder_list: List[torch.Tensor] = []
        for slot in slots:
            xt = slot.xt
            total_steps = len(slot.t_schedule) - 1
            if slot.request.latent_mask is not None:
                t_curr_scalar = slot.t_schedule[slot.step_idx].item()
                xt = ode_steps.mask_pre_blend(
                    xt, t_curr_scalar, slot.request.latent_mask,
                    slot.step_idx, total_steps,
                )
            xt_mask_list.append(xt)
            # ``_channel_gain`` is pre-converted to device/dtype by the
            # setters, so no per-tick ``.to(...)`` cast is needed.
            if self._channel_gain is not None:
                xt_decoder_list.append(xt * self._channel_gain)
            else:
                xt_decoder_list.append(xt)

        # --- Active pos/neg conditions per slot ---
        # ``neg_conds_per_slot[si]`` is non-empty only when slot ``si``
        # needs an actual uncond forward pass *this step*. RCFG modes
        # gate that: "self" never does a forward (virtual negative),
        # "initialize" only on step 0 (cached thereafter), full CFG
        # every step. Slots that have CFG but skip the forward this
        # step still get APG applied — they read ``vt_neg`` from the
        # slot's ``vt_neg_cached`` (initialize) or ``initial_noise``
        # (self) further down.
        pos_conds_per_slot: List[List[SlotCondition]] = []
        neg_conds_per_slot: List[List[SlotCondition]] = []
        for slot in slots:
            pos_conds_per_slot.append(self._active_conditions(slot))
            if slot.request.needs_neg_forward(slot.step_idx):
                neg_conds_per_slot.append(self._active_neg_conditions(slot))
            else:
                neg_conds_per_slot.append([])

        # --- Forward pass helper: batch N (slot_idx, cond) pairs in one call ---
        # Used independently for the positive pass and the negative pass so
        # the CFG path mirrors the pre-refactor two-forward-pass behavior.
        def _forward_pairs(
            pair_slot_idx: List[int], pair_cond: List[SlotCondition],
        ) -> torch.Tensor:
            xt_b = torch.cat(
                [xt_decoder_list[si] for si in pair_slot_idx], dim=0,
            )
            return self.adapter.batched_forward(
                xt_batch=xt_b,
                timestep_list=[
                    slots[si].t_schedule[slots[si].step_idx].item()
                    for si in pair_slot_idx
                ],
                enc_list=[c.encoder_hidden_states for c in pair_cond],
                mask_list=[c.encoder_attention_mask for c in pair_cond],
                ctx_list=[
                    slots[si].request.context_latents for si in pair_slot_idx
                ],
                aux_list=[
                    slots[si].request.aux_cond for si in pair_slot_idx
                ],
            )

        # --- Positive pass: one call across all slots' pos conditions ---
        pos_pair_si: List[int] = []
        pos_pair_cond: List[SlotCondition] = []
        for si in range(len(slots)):
            for c in pos_conds_per_slot[si]:
                pos_pair_si.append(si)
                pos_pair_cond.append(c)
        # Per-row step mapping for the steering hook; try/finally so a
        # stale list never leaks into a forward outside this rendezvous.
        self._current_step_per_row = [slots[si].step_idx for si in pos_pair_si]
        try:
            vt_pos_all = _forward_pairs(pos_pair_si, pos_pair_cond)
        finally:
            self._current_step_per_row = []

        # --- Negative pass (CFG only): skipped when no slot has CFG. ---
        neg_pair_si: List[int] = []
        neg_pair_cond: List[SlotCondition] = []
        for si in range(len(slots)):
            for c in neg_conds_per_slot[si]:
                neg_pair_si.append(si)
                neg_pair_cond.append(c)
        if neg_pair_si:
            self._current_step_per_row = [slots[si].step_idx for si in neg_pair_si]
            try:
                vt_neg_all = _forward_pairs(neg_pair_si, neg_pair_cond)
            finally:
                self._current_step_per_row = []
        else:
            vt_neg_all = None

        # --- Per-slot: blend pos, blend neg (if CFG), APG-combine ---
        vt_per_slot: List[torch.Tensor] = [None] * len(slots)  # type: ignore[list-item]
        pos_p = 0
        neg_p = 0
        for si, slot in enumerate(slots):
            pos = pos_conds_per_slot[si]
            neg = neg_conds_per_slot[si]

            vt_pos_block = vt_pos_all[pos_p:pos_p + len(pos)]
            pos_p += len(pos)
            if len(pos) == 1:
                vt_pos = vt_pos_block
            else:
                vt_pos = ode_steps.blend_velocities(
                    [(vt_pos_block[i:i + 1], pos[i]) for i in range(len(pos))],
                    self._device, self._dtype,
                )

            # Resolve ``vt_neg`` for this slot. Four cases:
            #   1. No CFG: vt_neg=None, fall through to vt_pos.
            #   2. RCFG-self: virtual negative is the slot's initial
            #      noise tensor. No forward consumed.
            #   3. Neg forward ran this step: consume the next ``len(neg)``
            #      rows from ``vt_neg_all`` (and cache for "initialize").
            #   4. RCFG-initialize after step 0: read from
            #      ``slot.vt_neg_cached`` populated on step 0.
            vt_neg = None
            if slot.request.has_cfg:
                if slot.request.rcfg_mode == "self":
                    vt_neg = slot.initial_noise
                elif neg and vt_neg_all is not None:
                    vt_neg_block = vt_neg_all[neg_p:neg_p + len(neg)]
                    neg_p += len(neg)
                    if len(neg) == 1:
                        vt_neg = vt_neg_block
                    else:
                        vt_neg = ode_steps.blend_velocities(
                            [(vt_neg_block[i:i + 1], neg[i]) for i in range(len(neg))],
                            self._device, self._dtype,
                        )
                    if slot.request.rcfg_mode == "initialize":
                        slot.vt_neg_cached = vt_neg.detach()
                elif slot.vt_neg_cached is not None:
                    vt_neg = slot.vt_neg_cached

            if vt_neg is not None:
                gc = self._eff_shared(slot, "guidance_curve").to(
                    device=vt_pos.device, dtype=vt_pos.dtype,
                )
                mom = self._eff_shared(slot, "apg_momentum")
                v_guided = ode_steps.apg_forward(
                    vt_pos, vt_neg,
                    guidance_scale=gc,
                    momentum_buffer=slot.momentum_buffer,
                    momentum=mom if mom is not None else -0.75,
                )
                rescale = self._eff_shared(slot, "cfg_rescale_curve")
                if rescale is not None:
                    v_guided = ode_steps.cfg_rescale(
                        v_guided, vt_pos, rescale,
                    )
                vt_per_slot[si] = v_guided
            else:
                vt_per_slot[si] = vt_pos

        # --- Per-slot integration ---
        step_ode = self._get_compiled(ode_steps.step_ode_euler)
        step_sde_curve = self._get_compiled(ode_steps.step_sde_curve)
        step_sde_renoise = self._get_compiled(ode_steps.step_sde_renoise)

        for si, slot in enumerate(slots):
            t_curr = slot.t_schedule[slot.step_idx].item()
            t_next = slot.t_schedule[slot.step_idx + 1].item()
            total_steps = len(slot.t_schedule) - 1

            vt = vt_per_slot[si]
            vs, sdc, onc = self._resolve_slot_curves(slot, vt)
            xt = xt_mask_list[si]

            req = slot.request
            sde_curve_active = (
                sdc is not None and req.source_latents is not None
            )
            # SDE activates from an explicit curve+source (per-frame
            # re-noise blending) OR ``config.infer_method='sde'`` (bare
            # re-noise, matching upstream's stock SDE behavior).
            use_sde = sde_curve_active or self.config.infer_method == "sde"

            # ``x0_target_strength`` path: blend toward a target latent
            # at scalar (or per-frame curve) strength, gated to the
            # refinement half.  Preserving the historical "strength==0
            # falls through to the fast path" behavior — checks the
            # effective (shared override or slot field) strength via a
            # tensor.any() sync, which costs one host-device fence per
            # slot per step but lets the gate stay tensor-safe.
            eff_strength = self._eff_shared(slot, "x0_target_strength")
            strength_active = (
                eff_strength is not None
                and bool(eff_strength.abs().any().item())
            )
            scalar_x0_target = (
                req.x0_target is not None
                and strength_active
                and req.x0_target_curve is None
                and slot.step_idx >= total_steps // 2
                and t_curr > 0
            )

            has_x0_target_curve = (
                req.x0_target is not None and req.x0_target_curve is not None
            )

            # Slots that had to go through the pre-refactor "complex"
            # tick must keep taking the x0-blend path here, even when
            # no blend actually fires — otherwise the fast-path Euler
            # skips the ``v_blended = (xt - x0_pred)/t_curr`` roundtrip
            # that the old ``integrate_step`` performed, changing bf16
            # output at the ~0.25 level on multi-cond / CFG workflows.
            has_complex_features = (
                req.latent_mask is not None
                or req.extra_conditions
                or req.primary_temporal_weight is not None
                or req.primary_step_range is not None
                or has_x0_target_curve
                or req.has_cfg
            )

            # Any blend that needs x0_pred pushes us out of the fast
            # direct-ODE path. SDE (curve or bare) also needs x0_pred.
            needs_x0 = (
                has_complex_features
                or scalar_x0_target
                or use_sde
                or t_next <= 0
            )

            if not needs_x0:
                # Fast path: direct Euler with raw (pos-blended, APG'd)
                # velocity. Byte-identical to the pre-refactor
                # ``_step_simple_ode``. When ``onc`` is the zeros sentinel
                # the post-step noise injection is a no-op.
                xt_new = step_ode(xt, vt, t_curr, t_next, vs, onc)
                slot.xt = self._maybe_dcw(xt, vt, xt_new, t_curr)
                slot.step_idx += 1
                continue

            # x0-blend path: compute x0_pred, apply blends, then integrate.
            vt_scaled = vt * vs
            x0_pred = ode_steps.x0_from_vel(xt, vt_scaled, t_curr)

            if req.latent_mask is not None:
                x0_pred = ode_steps.mask_post_blend_x0(
                    x0_pred, req.latent_mask, slot.step_idx, total_steps,
                )

            if has_x0_target_curve:
                # Gated x0_target_curve blend (smooth ramp-in over the
                # refinement portion controlled by x0_target_gate).
                step_progress = slot.step_idx / max(total_steps - 1, 1)
                gate_start = req.x0_target_gate
                blend_gate = (
                    max(0.0, step_progress - gate_start)
                    / max(1.0 - gate_start, 1e-6)
                )
                if blend_gate > 0:
                    curve = ode_steps.normalize_curve(req.x0_target_curve).to(
                        device=xt.device, dtype=xt.dtype,
                    )
                    x0_pred = ode_steps.blend_x0_target(
                        x0_pred, req.x0_target, curve * blend_gate,
                    )
            elif scalar_x0_target:
                alpha = eff_strength.to(device=x0_pred.device, dtype=x0_pred.dtype)
                x0_pred = (1.0 - alpha) * x0_pred + alpha * req.x0_target

            if t_next <= 0:
                # Final step. ODE and bare-SDE return the clean x0
                # directly. SDE-with-curve keeps the per-frame source
                # blend even at t=0 (noise contribution is zero but the
                # sdc mix still applies), matching the pre-refactor
                # ``_step_simple_sde_curve`` behavior on the last step.
                if sde_curve_active:
                    xt_new = step_sde_curve(
                        xt, x0_pred, t_next, sdc, req.source_latents,
                    )
                else:
                    xt_new = x0_pred
                slot.xt = self._maybe_dcw(xt, vt, xt_new, t_curr)
                slot.step_idx += 1
                continue

            if sde_curve_active:
                xt_new = step_sde_curve(
                    xt, x0_pred, t_next, sdc, req.source_latents,
                )
            elif use_sde:
                # Bare SDE (no curve). Use the mask's fixed noise when a
                # latent_mask is active so inpainting semantics are
                # preserved; otherwise draw fresh noise.
                noise = (
                    req.latent_mask.ensure_noise(xt.device, xt.dtype)
                    if req.latent_mask is not None
                    else torch.randn_like(xt)
                )
                xt_new = step_sde_renoise(xt, x0_pred, t_next, noise)
            else:
                # ODE with blended x0: synthesize v_blended and reuse
                # step_ode_euler. Passing the ones sentinel for vs keeps
                # the kernel byte-identical to the fast path.
                v_blended = (xt - x0_pred) / t_curr
                ones_3d, _ = self._ensure_sentinels()
                xt_new = step_ode(
                    xt, v_blended, t_curr, t_next, ones_3d, onc,
                )

            slot.xt = self._maybe_dcw(xt, vt, xt_new, t_curr)
            slot.step_idx += 1

    def set_depth(self, depth: int) -> None:
        """Resize the ring buffer. Active slots drain naturally.

        Args:
            depth: New number of concurrent slots. Clamped to [1, 8],
                matching the canonical decoder engine batch profile.
        """
        depth = max(1, min(depth, 8))
        if depth == self._depth:
            return

        old = self._slots
        if depth < self._depth:
            # Shrink: keep the first `depth` slots, let extras drain
            # by moving active excess slots to the queue... actually
            # we just truncate. Active slots beyond the new depth
            # are lost (their in-progress work is discarded).
            self._slots = old[:depth]
        else:
            # Grow: extend with empty slots
            self._slots = old + [None] * (depth - self._depth)

        self._depth = depth
        logger.info("Pipeline depth changed to {}", depth)

    def flush(self) -> List[torch.Tensor]:
        """Drain the pipeline: keep ticking until all slots complete."""
        results = []
        max_iters = self._depth * 2
        for _ in range(max_iters):
            result = self.tick()
            if result is not None:
                results.append(result)
            if self.active_slots == 0 and not self._queue:
                break
        return results

    # ------------------------------------------------------------------
    # Shared mutable curves (bypass ring-buffer drain)
    # ------------------------------------------------------------------

    def set_shared_curve(
        self,
        name: str,
        value: "float | int | torch.Tensor | None",
    ) -> None:
        """Set (or clear) a shared per-step override.

        ``name`` is the SlotRequest field name to override (e.g.
        ``"sde_denoise_curve"``, ``"velocity_scale"``,
        ``"ode_noise_curve"``).  When set, ALL in-flight slots use this
        value on the very next tick, regardless of pipeline depth.

        ``value`` can be a scalar or a per-frame tensor; both flow
        through :func:`ode_steps.normalize_curve` so the storage form is
        always ``[B, T, 1]`` and downstream consumers do not need to
        type-discriminate. Pass ``None`` to revert that name to per-slot
        behavior.
        """
        if value is None:
            self._shared_curves.pop(name, None)
            return
        self._shared_curves[name] = ode_steps.normalize_curve(value)

    def _eff_shared(self, slot: "_Slot", name: str):
        """Return shared override for ``name`` if set, else slot's field.

        Output is always either ``None`` or a normalized ``[B, T, 1]``
        tensor — the shared override is canonicalized at the setter, and
        any ``SlotRequest`` field is normalized here so callers never
        need to ``isinstance``-check.
        """
        v = self._shared_curves.get(name)
        if v is None:
            v = getattr(slot.request, name, None)
            if v is None:
                return None
        return ode_steps.normalize_curve(v)

    def set_dcw(
        self,
        *,
        enabled: bool,
        mode: Optional[str] = None,
        scaler: "Optional[float | torch.Tensor]" = None,
        high_scaler: "Optional[float | torch.Tensor]" = None,
        wavelet: Optional[str] = None,
        advanced: Optional[DCWAdvanced] = None,
    ) -> None:
        """Replace the DCW corrector. Takes effect on the next tick.

        Hot-updatable for all in-flight slots, toggling DCW does not
        rebuild the pipeline or invalidate the compiled step graphs
        (DCW runs outside the compiled region as a per-slot post-step
        transform). ``scaler`` and ``high_scaler`` may each be a Python
        scalar or a per-frame curve in latent layout (``[T]`` /
        ``[1, T]`` / ``[1, T, 1]``); curves are resampled to the
        wavelet band's downsampled length inside the DCW kernel.
        """
        cur = self._dcw_corrector
        self._dcw_corrector = DCWCorrector(
            enabled=enabled,
            mode=mode if mode is not None else cur.mode,
            scaler=scaler if scaler is not None else cur.scaler,
            high_scaler=high_scaler if high_scaler is not None else cur.high_scaler,
            wavelet=wavelet if wavelet is not None else cur.wavelet,
            advanced=advanced if advanced is not None else cur.advanced,
        )

    def _maybe_dcw(
        self,
        xt_pre: torch.Tensor,
        vt: torch.Tensor,
        xt_post: torch.Tensor,
        t_curr: float,
    ) -> torch.Tensor:
        """Apply DCW correction if active, else return ``xt_post`` unchanged.

        ``vt`` must be the post-APG, pre-velocity-scale model velocity
        (i.e. ``vt_per_slot[si]``) so the synthesized
        ``denoised = xt_pre - vt * t_curr`` matches the model's actual
        x0 prediction. Using the velocity-scaled or x0-blended form
        would mis-calibrate the DCW correction.
        """
        if not self._dcw_corrector.is_active:
            return xt_post
        denoised = xt_pre - vt * t_curr
        return self._dcw_corrector.apply(xt_post, denoised, t_curr)

    def set_channel_guidance(self, configs) -> None:
        """Set channel guidance configs for input scaling during denoising.

        Builds a gain tensor from the configs and caches it on
        ``self._channel_gain``.  Takes effect on the very next tick for
        ALL in-flight slots.  Pass an empty list or None to disable.

        Args:
            configs: List of ChannelGuidanceEntry, or None to clear.
        """
        if not configs:
            self._channel_gain = None
            return

        from acestep.nodes.channel_nodes import build_channel_gain
        self.set_channel_gain_tensor(build_channel_gain(
            configs, self._device or torch.device("cuda"), self._dtype or torch.float16,
        ))

    def set_channel_gain_tensor(self, gain: Optional[torch.Tensor]) -> None:
        """Cache a pre-built channel-gain tensor on the pipeline.

        Converts to the pipeline's device/dtype once so the tick hot
        path can multiply without re-casting every call.
        """
        if gain is None:
            self._channel_gain = None
            return
        dev = self._device or torch.device("cuda")
        dt = self._dtype or torch.float16
        self._channel_gain = gain.to(device=dev, dtype=dt)

    def set_steering(self, configs: List[Dict[str, object]]) -> None:
        """Set activation-steering configs.

        Each config dict has:
          - ``layer``: int, index into ``decoder.layers``
          - ``step``: int, the denoise step at which to fire
          - ``vector``: 1-D unit-norm tensor [hidden_dim]
          - ``magnitude``: float, paired mean-diff scale
          - ``alpha``: float, knob value; effective shift is
            ``alpha * magnitude * vector``

        Multiple configs may share a layer (additions sum, modulo the
        per-row step gate). Zero-alpha entries drop. Pass ``[]`` to
        clear. Eager: forward hooks on ``decoder.layers`` (installed
        lazily). TRT: per-tick buffer fill in ``_trt_forward``.
        """
        by_layer: Dict[int, List[_SteeringApply]] = {}
        for c in configs:
            alpha = float(c.get("alpha", 0.0))
            if alpha == 0.0:
                continue
            mag = float(c.get("magnitude", 1.0))
            li = int(c["layer"])
            by_layer.setdefault(li, []).append(_SteeringApply(
                vector=c["vector"],  # type: ignore[arg-type]
                scale=alpha * mag,
                step=int(c["step"]),
            ))
        self._steering_by_layer = by_layer
        # Eager-path hooks only matter when the PyTorch decoder runs.
        # With TRT active the decoder may be a stub.
        if (
            by_layer
            and not self._steering_hooks_installed
            and self._trt_engine is None
        ):
            self._install_steering_hooks()
            self._steering_hooks_installed = True

    def _install_steering_hooks(self) -> None:
        """Attach one forward hook per DiT layer.

        Reads ``_steering_by_layer`` + ``_current_step_per_row`` at call
        time. Idempotent via ``_steering_hooks_installed`` (cleared on
        engine swap so a new eager decoder gets fresh hooks).
        """
        layers = self.decoder.layers

        def make_hook(layer_idx: int):
            def _hook(_module, _inputs, output):
                applies = self._steering_by_layer.get(layer_idx)
                if not applies:
                    return output
                hs = output[0] if isinstance(output, tuple) else output
                B = hs.shape[0]
                changed = False
                for apply in applies:
                    mask_rows = self._steering_row_mask(apply.step, B)
                    if mask_rows is None:
                        continue
                    v = apply.vector.to(device=hs.device, dtype=hs.dtype)
                    hs[mask_rows] = hs[mask_rows] + apply.scale * v.view(1, 1, -1)
                    changed = True
                if not changed:
                    return output
                if isinstance(output, tuple):
                    return (hs,) + output[1:]
                return hs
            return _hook

        for li in range(len(layers)):
            layers[li].register_forward_hook(make_hook(li))

    def stats(self) -> dict:
        return {
            "ticks": self.ticks,
            "active_slots": self.active_slots,
            "queue_depth": len(self._queue),
            "last_tick_ms": round(self._last_tick_ms, 2),
            "is_warmed_up": self.is_warmed_up,
            "backend": "trt" if self._trt_engine is not None else "pytorch",
        }

    def close(self) -> None:
        """Release per-pipeline GPU buffers + caches.

        Called by :meth:`StreamHandle.close` on session teardown. Drops:

        - The shape-keyed TRT I/O buffers (``_trt_bufs``,
          ``_trt_out_buf``). On a turbo profile this is ~80 MB but the
          buffers reference the engine's input bindings; clearing them
          breaks the cycle that would otherwise pin the engine.
        - In-flight slot tensors (``_slots``) and the queued requests'
          backing tensors (``_queue``) — each slot's ``xt`` is a
          ``[1, T, 64]`` bf16 latent.
        - The schedule cache, compiled ODE/SDE step graphs, sentinel
          tensors, channel-gain tensor, and shared-curve dict.
        - The TRT engine/context refs we captured from ``DiffusionEngine``.
          The engine itself is owned (and freed) by ``DiffusionEngine.close``;
          we only drop our copy of the refs so this pipeline doesn't pin
          the engine after the session is gone.

        Idempotent: subsequent calls are no-ops.
        """
        self._slots = []
        self._queue = []
        self._trt_bufs = None
        self._trt_out_buf = None
        self._trt_bufs_cache.clear()
        self._trt_ctx = None
        self._trt_engine = None
        self._trt_stream = None
        self._channel_gain = None
        self._ones_3d = None
        self._zeros_3d = None
        self._schedule_cache.clear()
        self._compiled_cache.clear()
        self._shared_curves.clear()
        # DCW corrector holds wavelet basis tensors on GPU; drop it.
        self._dcw_corrector = None
        # Detach references to the engine + decoder so DiffusionEngine.close
        # is the sole owner that decides when those objects go. The
        # adapter ref is dropped too (ACEAdapter back-references this
        # pipeline; breaking the cycle here keeps teardown deterministic).
        self.engine = None
        self.decoder = None
        self.model = None
        self.adapter = None
