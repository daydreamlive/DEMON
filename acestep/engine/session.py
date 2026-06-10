"""Persistent session for ACE-Step generation.

Loads the model once and keeps handler, compiled decoder, and TRT engines
alive across multiple generation calls. Provides convenience methods that
delegate to the node system, so intermediate results (latents, hints,
conditioning) can be held by the caller and reused without recomputation.

Typical usage (cover iteration with different seeds):

    session = Session(decoder_backend="compile", vae_backend="compile")
    source = session.prepare_source(audio)
    cond = session.encode_text(
        tags="deathstep", instruction=TASK_INSTRUCTIONS["cover"],
        refer_latent=source.latent, bpm=136, duration=60.0, key="G# minor",
    )
    for seed in [1528, 9999, 42]:
        output = session.generate(
            conditioning=cond, context_latent=source.context_latent,
            source_latent=source.latent, seed=seed,
        )
        save_audio(session.decode(output), f"out_{seed}.wav")

When Daydream Scope integrates, its session management replaces this.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

from loguru import logger

from acestep.constants import TASK_INSTRUCTIONS
from acestep.nodes.types import (
    Audio,
    CLIPHandle,
    Conditioning,
    Curve,
    Latent,
    Mask,
    ModelHandle,
    Modulation,
    Solver,
    VAEHandle,
)


_MODULATION_KEYS = (
    "velocity_scale",
    "initial_noise_curve",
    "chunk_mask",
    "x0_target",
    "x0_target_curve",
    "x0_target_strength",
    "x0_target_gate",
    "guidance_curve",
)


def _extract_modulation(kwargs: dict) -> Optional[Modulation]:
    """Pop legacy modulation kwargs from ``kwargs`` into a ``Modulation``.

    Returns ``None`` when nothing modulation-related was passed — lets
    callers skip wiring when they don't need any optional inputs. Accepts
    either wrapper types (Curve/Mask/Latent, read via ``.tensor``) or
    raw tensors.
    """
    if not any(k in kwargs for k in _MODULATION_KEYS):
        return None

    def _curve(v):
        if v is None:
            return None
        return v.tensor if hasattr(v, "tensor") else v

    return Modulation(
        velocity_scale=_curve(kwargs.pop("velocity_scale", None)),
        initial_noise_curve=_curve(kwargs.pop("initial_noise_curve", None)),
        chunk_mask=_curve(kwargs.pop("chunk_mask", None)),
        x0_target=kwargs.pop("x0_target", None),
        x0_target_curve=_curve(kwargs.pop("x0_target_curve", None)),
        x0_target_strength=float(kwargs.pop("x0_target_strength", 0.0)),
        x0_target_gate=float(kwargs.pop("x0_target_gate", 0.0)),
        guidance_curve=_curve(kwargs.pop("guidance_curve", None)),
    )


@dataclass
class PreparedSource:
    """Cached results from preparing a source audio.

    ``latent`` is the raw VAE encoding; ``context_latent`` is the
    semantic-extracted version used as structural guidance for the
    denoiser. Both are LATENTs and share the same ``[B, T, D]`` shape.
    """
    latent: Latent
    context_latent: Latent


def _build_solver(
    method: str,
    *,
    ode_noise_curve: Any = None,
    sde_denoise_curve: Any = None,
) -> Solver:
    """Promote legacy ``method`` + curve kwargs into a ``Solver`` value.

    Used by ``Session.generate`` and ``StreamHandle.tick`` so callers
    don't have to construct Solver manually. Accepts either Curve
    wrappers (with ``.tensor`` attribute) or raw tensors.

    A solver-specific curve overrides ``method`` when set — this
    preserves the pre-refactor implicit behavior where passing
    ``sde_denoise_curve=...`` alone was enough to switch a stream into
    SDE mode for that tick. Node-level callers (StreamDenoise /
    Generate) don't get this magic; they take the ``Solver`` directly
    and the curve lives inside it.
    """
    if sde_denoise_curve is not None:
        method = "sde"
    elif ode_noise_curve is not None:
        method = "ode"
    curve = ode_noise_curve if method == "ode" else sde_denoise_curve
    tensor = curve.tensor if hasattr(curve, "tensor") else curve
    return Solver(method=method, noise_curve=tensor)


class Session:
    """Persistent GPU state for ACE-Step generation.

    Loads the model, VAE, text encoder, and TRT engines once. Exposes
    node handle types and convenience methods that wrap node execution.

    Intermediate results are returned to the caller; the caller controls
    what gets reused between generations by holding references.

    When ``trt_engines`` is provided, decoder and/or VAE PyTorch weights
    are never loaded to GPU (or at all, for the VAE). The TRT engines
    are loaded via polygraphy and wired into the DiffusionEngine and
    VAE node cache directly.

    Example::

        from acestep.paths import default_trt_engines
        s = Session(
            decoder_backend="tensorrt",
            vae_backend="tensorrt",
            trt_engines=default_trt_engines(),
        )
    """

    def __init__(
        self,
        *,
        project_root: Optional[str] = None,
        config_path: str = "acestep-v15-turbo",
        device: str = "cuda",
        decoder_backend: str = "eager",
        vae_backend: str = "eager",
        use_flash_attention: bool = True,
        offload_to_cpu: bool = False,
        offload_dit_to_cpu: bool = False,
        offload_text_encoder: bool = False,
        quantization: Optional[str] = None,
        trt_engines: Optional[dict[str, str]] = None,
        vae_window: float = 0.0,
        vae_overlap: float = 0.4,
    ):
        """Persistent ACE-Step session.

        Args:
            decoder_backend: Runtime for the DiT decoder. One of:
                - "eager": pure PyTorch, no compile, no TRT (default)
                - "compile": torch.compile (decoder + diffusion inner loops)
                - "tensorrt": load TRT engine. Requires trt_engines["decoder"].
            vae_backend: Runtime for the VAE encode/decode. One of:
                - "eager": pure PyTorch (default)
                - "compile": torch.compile
                - "tensorrt": load TRT engines. Requires both
                  trt_engines["vae_encode"] and trt_engines["vae_decode"].
            trt_engines: Engine paths, used iff a *_backend is "tensorrt".
                Keys: "decoder", "vae_encode", "vae_decode". Use
                acestep.paths.default_trt_engines() for the canonical paths.
            offload_to_cpu: Build every model on CPU and hop weights to
                the GPU per call. NOTE: without ``offload_dit_to_cpu``
                the DiT becomes GPU-resident on its FIRST use and stays
                (see ``ModelContext._load_model_context``); set both for
                a session whose steady-state VRAM must stay ~0.
            offload_dit_to_cpu: Move the DiT back to CPU after each use
                instead of letting it go resident on first touch.
            offload_text_encoder: Override text encoder placement policy.
                Defaults to ``False`` so prompt edits do not pay CPU/GPU
                transfer cost. Set ``True`` for lower steady VRAM usage.
        """
        from acestep.engine.model_context import ModelContext
        from acestep.engine.runtime_init import (
            apply_trt_backends,
            backends_to_model_context_flags,
            validate_backends,
        )

        self.model = None  # type: ignore[assignment]
        self.clip = None  # type: ignore[assignment]
        self.vae = None  # type: ignore[assignment]
        self._trt_vae_engine_paths = ()

        trt_engines = validate_backends(
            decoder_backend=decoder_backend,
            vae_backend=vae_backend,
            trt_engines=trt_engines,
        )

        if project_root is None:
            from acestep.paths import checkpoints_dir
            project_root = str(checkpoints_dir())

        # Kick off disk-bound preloads in background threads while
        # ModelContext loads the DiT shards. Both are pure I/O (no CUDA,
        # no torch ops that contend with the main thread) and release
        # the GIL during file reads.
        #
        # - trt_decoder_bytes_future: pre-reads the TRT engine file (~1–2 GB)
        #   so engine_from_bytes on the main thread can skip the disk read.
        # - lora_discovery_future: pre-walks the LoRA roots (rglob +
        #   .resolve() per file is several seconds on Windows for ~37 LoRAs
        #   across the primary library + ACESTEP_EXTRA_LORA_DIRS).
        from concurrent.futures import ThreadPoolExecutor
        _preload_pool = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="boot-preload",
        )
        trt_decoder_bytes_future = None
        if decoder_backend == "tensorrt":
            from polygraphy.backend.common import bytes_from_path
            trt_decoder_bytes_future = _preload_pool.submit(
                bytes_from_path, str(trt_engines["decoder"]),
            )

        def _discover_loras_safely():
            try:
                from acestep.paths import discover_all_loras
                return discover_all_loras()
            except Exception:
                return None
        lora_discovery_future = _preload_pool.submit(_discover_loras_safely)
        # Pool dies after both futures complete; no need to keep the handle.
        _preload_pool.shutdown(wait=False)

        ctx_flags = backends_to_model_context_flags(
            decoder_backend=decoder_backend, vae_backend=vae_backend
        )

        try:
            ctx = ModelContext(
                project_root=project_root,
                config_path=config_path,
                device=device,
                use_flash_attention=use_flash_attention,
                offload_to_cpu=offload_to_cpu,
                offload_dit_to_cpu=offload_dit_to_cpu,
                offload_text_encoder=offload_text_encoder,
                quantization=quantization,
                **ctx_flags,
            )

            self.model = ModelHandle(handler=ctx)
            self.clip = CLIPHandle(handler=ctx)
            self.vae = VAEHandle(handler=ctx)

            # Windowed VAE decode config (seconds; 0 = full decode).
            # Hard-clamp positive windows to the engine-supported range so
            # the auto-selected windowed engine is never asked to decode a
            # chunk longer than its profile max. <= 0 is the disable
            # sentinel and stays as-is.
            if vae_window > 0:
                from acestep.paths import WINDOWED_VAE_WINDOW_RANGE_S
                lo, hi = WINDOWED_VAE_WINDOW_RANGE_S
                vae_window = max(lo, min(hi, vae_window))
            self._vae_window = vae_window
            self._vae_overlap = vae_overlap

            # Auto-select the windowed VAE decode engine when windowing is
            # active. Saves ~7.7 GB of TRT workspace at context-creation
            # time vs the canonical 240s engine (see
            # scripts/benchmarks/bench_vae_decode_profiles.py). Falls back
            # silently to whatever the caller passed in when the windowed
            # engine isn't built yet.
            if vae_window > 0 and vae_backend == "tensorrt" and "vae_decode" in trt_engines:
                from acestep.paths import (
                    available_windowed_vae_decode_engine,
                    looks_like_dreamvae_engine,
                    windowed_vae_decode_engine_name,
                )
                from loguru import logger as _log

                current = trt_engines["vae_decode"]
                is_dreamvae = looks_like_dreamvae_engine(current)
                windowed = available_windowed_vae_decode_engine(dreamvae=is_dreamvae)
                if windowed is not None and str(windowed) != current:
                    _log.info(
                        "vae_window={:.1f}s active: swapping vae_decode engine "
                        "{} -> {}", vae_window,
                        Path(current).stem, windowed.stem,
                    )
                    # Don't mutate the caller's dict.
                    trt_engines = dict(trt_engines)
                    trt_engines["vae_decode"] = str(windowed)
                elif windowed is None:
                    _log.warning(
                        "vae_window={:.1f}s active but windowed engine not "
                        "built ({}); using {} (will reserve more VRAM than "
                        "necessary). Build with: python -m "
                        "acestep.engine.trt.build --all{}",
                        vae_window,
                        windowed_vae_decode_engine_name(dreamvae=is_dreamvae),
                        Path(current).stem,
                        " --with-dreamvae" if is_dreamvae else "",
                    )

            self._trt_vae_engine_paths = tuple(
                str(trt_engines[key])
                for key in ("vae_encode", "vae_decode")
                if vae_backend == "tensorrt" and key in trt_engines
            )

            apply_trt_backends(
                ctx,
                decoder_backend=decoder_backend,
                vae_backend=vae_backend,
                trt_engines=trt_engines,
                device=device,
                trt_decoder_bytes_future=trt_decoder_bytes_future,
                lora_discovery_future=lora_discovery_future,
            )
        except Exception:
            logger.warning(
                "session_init_failed_cleanup decoder={} vae={}",
                decoder_backend, vae_backend,
            )
            self.close()
            raise

    @property
    def handler(self):
        return self.model.handler

    # ------------------------------------------------------------------
    # Source preparation
    # ------------------------------------------------------------------

    def encode_audio(self, audio: Audio) -> Latent:
        """VAE encode audio waveform to latent."""
        from acestep.nodes.vae_nodes import VAEEncodeAudio

        return VAEEncodeAudio().execute(vae=self.vae, audio=audio)["latent"]

    def extract_hints(self, latent: Latent) -> Latent:
        """Extract semantic structural hints from a latent.

        Returns a LATENT carrying the detokenized hint tensor. Usable
        anywhere a latent is accepted (e.g. ``context_latent`` on
        Generate / StreamDenoise).
        """
        from acestep.nodes.semantic_nodes import SemanticExtract

        return SemanticExtract().execute(
            model=self.model, latent=latent
        )["latent"]

    def prepare_source(self, audio: Audio) -> PreparedSource:
        """VAE encode + semantic extract in one call.

        Returns a PreparedSource holding the raw latent and the
        semantic-extracted context latent. The caller holds this and
        reuses it across generations.
        """
        latent = self.encode_audio(audio)
        context_latent = self.extract_hints(latent)

        return PreparedSource(latent=latent, context_latent=context_latent)

    # ------------------------------------------------------------------
    # Text encoding
    # ------------------------------------------------------------------

    def encode_text(
        self,
        *,
        tags: str = "",
        lyrics: str = "",
        instruction: Optional[str] = None,
        refer_latent: Optional[Latent] = None,
        bpm: int = 120,
        duration: float = 60.0,
        key: str = "C major",
        time_signature: str = "4",
        language: str = "en",
    ) -> Conditioning:
        """Encode text prompt into cross-attention conditioning.

        Composes ``EncodeText`` + ``EncodeConditioning`` internally.
        Callers who want to blend ``refer_latent`` with silence should
        do that upstream with ``Session.blend_latents`` (or the
        ``LatentBlend`` node) before passing it here; this method does
        no mixing itself.
        """
        from acestep.nodes.cond_nodes import EncodeText, EncodeConditioning

        if instruction is None:
            instruction = TASK_INSTRUCTIONS["text2music"]

        text_embed = EncodeText().execute(
            clip=self.clip,
            tags=tags,
            lyrics=lyrics,
            instruction=instruction,
            bpm=bpm,
            duration=duration,
            key=key,
            time_signature=time_signature,
            language=language,
        )["text_embed"]

        return EncodeConditioning().execute(
            model=self.model,
            text_embed=text_embed,
            timbre_ref=refer_latent,
        )["conditioning"]

    def null_conditioning(self, conditioning: Conditioning) -> Conditioning:
        """Build null (unconditional) conditioning for CFG.

        Uses the model's learned ``null_condition_emb`` parameter,
        expanded to match the shape of the positive conditioning's
        encoder_hidden_states. This matches the upstream CFG
        implementation in ``generate_audio()``.

        Args:
            conditioning: The positive conditioning whose shape to match.

        Returns:
            Conditioning with encoder_hidden_states replaced by the
            learned null embedding, and the same attention mask.
        """
        import torch

        entries = conditioning.to_entries()
        if not entries:
            return conditioning

        entry = entries[0]
        enc_hs = entry.encoder_hidden_states

        # Access the learned null embedding from the model
        null_emb = self.handler.model.null_condition_emb  # [1, 1, 2048]
        null_enc = null_emb.expand_as(enc_hs).to(
            device=enc_hs.device, dtype=enc_hs.dtype
        )

        return Conditioning(
            encoder_hidden_states=null_enc,
            encoder_attention_mask=entry.encoder_attention_mask,
        )

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        *,
        conditioning: Conditioning,
        context_latent: Optional[Latent] = None,
        chunk_mask: Optional[Mask] = None,
        source_latent: Optional[Latent] = None,
        seed: Optional[int] = None,
        denoise: float = 1.0,
        steps: int = 8,
        shift: float = 3.0,
        method: str = "ode",
        **kwargs: Any,
    ) -> Latent:
        """Run the diffusion loop. Always executes (never cached).

        ``method``, ``ode_noise_curve``, and ``sde_denoise_curve`` are
        promoted to a ``Solver`` internally; callers who already hold a
        ``Solver`` can pass it via ``solver=`` and those three are
        ignored.
        """
        from acestep.nodes.diffusion_nodes import DiffusionConfigNode, Generate

        config = DiffusionConfigNode().execute(
            steps=steps, shift=shift, seed=seed, denoise=denoise,
        )["config"]

        solver = kwargs.pop("solver", None)
        if solver is None:
            solver = _build_solver(
                method,
                ode_noise_curve=kwargs.pop("ode_noise_curve", None),
                sde_denoise_curve=kwargs.pop("sde_denoise_curve", None),
            )

        modulation = kwargs.pop("modulation", None)
        if modulation is None:
            if chunk_mask is not None:
                kwargs["chunk_mask"] = chunk_mask
                chunk_mask = None
            modulation = _extract_modulation(kwargs)

        return Generate().execute(
            model=self.model,
            config=config,
            solver=solver,
            positive=conditioning,
            context_latent=context_latent,
            source_latent=source_latent,
            modulation=modulation,
            **kwargs,
        )["latent"]

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def decode(self, latent: Latent, t_start: float = 0.0, cyclic: bool = False) -> Audio:
        """VAE decode latent to audio waveform.

        Always routes through :class:`StreamVAEDecode`, which collapses
        the windowed / full-decode branch internally (``vae_window <= 0``
        falls through to a plain :class:`VAEDecodeAudio` call). This
        keeps a single decode codepath instead of diverging here and in
        ``StreamHandle.decode``.

        Args:
            latent: Latent to decode.
            t_start: When ``vae_window`` > 0, the start time (seconds)
                of the window to decode. Ignored when ``vae_window``
                is 0.
            cyclic: When ``vae_window`` > 0 and the audio is meant to
                loop, pull the boundary context from the opposite end
                of the latent so frames near 0 and T-1 don't lose
                receptive-field context. Off by default — only enable
                for callers whose audio is genuinely cyclic.
        """
        from acestep.nodes.vae_nodes import StreamVAEDecode

        return StreamVAEDecode().execute(
            vae=self.vae,
            latent=latent,
            vae_window=self._vae_window,
            vae_overlap=self._vae_overlap,
            t_start=t_start,
            cyclic=cyclic,
        )["audio"]

    # ------------------------------------------------------------------
    # Audio analysis
    # ------------------------------------------------------------------

    @staticmethod
    def audio_info(audio: Audio) -> dict:
        """Detect BPM, key, and duration from audio."""
        from acestep.nodes.audio_nodes import AudioInfo

        return AudioInfo().execute(audio=audio)

    # ------------------------------------------------------------------
    # Latent / LoRA utilities
    # ------------------------------------------------------------------

    def empty_latent(self, duration: float = 60.0) -> Latent:
        """Create a silence latent of a given duration."""
        from acestep.nodes.vae_nodes import EmptyLatent

        return EmptyLatent().execute(
            model=self.model, duration=duration,
        )["latent"]

    @staticmethod
    def blend_latents(
        a: Latent, b: Latent, alpha: float = 0.5,
    ) -> Latent:
        """Blend two latents. 0.0 = all A, 1.0 = all B."""
        from acestep.nodes.vae_nodes import LatentBlend

        return LatentBlend().execute(
            latent_a=a, latent_b=b, alpha=alpha,
        )["latent"]

    def apply_lora(self, path: str, scale: float = 1.0) -> str:
        """Load and apply a LoRA. Stackable (call multiple times).

        Returns the LoRA id (filename stem) for selective removal.
        """
        engine = self.model.handler._diffusion_engine
        lora_id = engine.apply_lora(path, strength=scale)
        if not hasattr(self, '_lora_stack'):
            self._lora_stack = []
        self._lora_stack.append(lora_id)
        return lora_id

    def remove_loras(self) -> None:
        """Remove all applied LoRAs in reverse order."""
        engine = self.model.handler._diffusion_engine
        if hasattr(self, '_lora_stack'):
            while self._lora_stack:
                engine.remove_lora(self._lora_stack.pop())

    def remove_last_lora(self) -> None:
        """Remove the most recently applied LoRA."""
        engine = self.model.handler._diffusion_engine
        if hasattr(self, '_lora_stack') and self._lora_stack:
            engine.remove_lora(self._lora_stack.pop())

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def stream(
        self,
        *,
        source: PreparedSource,
        conditioning: Conditioning,
        steps: int = 8,
        shift: float = 3.0,
        method: str = "ode",
        pipeline_depth: Optional[int] = None,
        solver: Optional[Solver] = None,
        # DCW (post-step wavelet-domain correction; default on, matching
        # upstream v0.1.7 — see acestep.engine.dcw).
        dcw_enabled: bool = True,
        dcw_mode: str = "double",
        dcw_scaler: float = 0.05,
        dcw_high_scaler: float = 0.02,
        dcw_wavelet: str = "haar",
        dcw_advanced: Optional["DCWAdvanced"] = None,
    ) -> "StreamHandle":
        """Build a streaming graph handle for interactive generation.

        Returns a :class:`StreamHandle` carrying the persistent
        ``StreamDenoise`` node (which owns the underlying
        ``StreamPipeline``), a ``StreamVAEDecode`` node, the shared
        ``ModelHandle`` / ``VAEHandle``, and the initial conditioning /
        context. Callers drive the graph by invoking
        ``handle.stream_node.execute(**kwargs)`` per tick; the returned
        dict has a ``latent`` key that is either a finished ``Latent``
        (this tick) or ``None`` (mid-flight).

        Default widget params (``steps``, ``shift``, ``method``,
        ``pipeline_depth``) are baked into the
        returned handle as ``base_kwargs`` for convenience; per-tick
        callers override them as needed. Swapping ``conditioning`` or
        ``context_latent`` on the handle between ticks rewires the next
        request with zero rebuild.
        """
        from acestep.nodes.diffusion_nodes import StreamDenoise
        from acestep.nodes.vae_nodes import StreamVAEDecode

        stream_node = StreamDenoise()
        decoder_node = StreamVAEDecode()

        depth = pipeline_depth if pipeline_depth is not None else steps
        if solver is None:
            solver = Solver(method=method, noise_curve=None)

        return StreamHandle(
            session=self,
            stream_node=stream_node,
            decoder_node=decoder_node,
            model=self.model,
            vae=self.vae,
            source=source,
            conditioning=conditioning,
            context_latent=source.context_latent,
            base_kwargs={
                "steps": steps,
                "shift": shift,
                "solver": solver,
                "pipeline_depth": depth,
                "dcw_enabled": dcw_enabled,
                "dcw_mode": dcw_mode,
                "dcw_scaler": dcw_scaler,
                "dcw_high_scaler": dcw_high_scaler,
                "dcw_wavelet": dcw_wavelet,
                "dcw_advanced": dcw_advanced,
            },
        )

    def close(self) -> None:
        """Release all GPU + CPU state owned by this session.

        Tears down the model context (which chains DiffusionEngine →
        LoRAManagerBase → TRT execution context destruction), then runs
        ``gc.collect()`` and ``torch.cuda.empty_cache()`` so PyTorch
        returns its caching-allocator pages to CUDA. TRT contexts free
        directly through their finalizers and don't go through torch.

        Outstanding ``StreamHandle`` instances should be ``close()``'d
        BEFORE this; the handle's pipeline references the engine, and
        closing the engine while a handle still holds it works (the
        handle's refs are simply dangling) but defeats the cleanup
        ordering documented on each component's close.

        Idempotent: subsequent calls are no-ops.
        """
        import gc
        import torch

        ctx = getattr(self.model, "handler", None) if self.model is not None else None
        if ctx is not None:
            try:
                ctx.close()
            except Exception as e:
                logger.warning("ModelContext.close raised: {}", e)
        # Drop the handle wrappers so nothing else can dereference the
        # now-closed context through them.
        self.model = None  # type: ignore[assignment]
        self.clip = None  # type: ignore[assignment]
        self.vae = None  # type: ignore[assignment]

        trt_vae_paths = getattr(self, "_trt_vae_engine_paths", ())
        if trt_vae_paths:
            from acestep.nodes.vae_nodes import _evict_trt_vae

            for engine_path in trt_vae_paths:
                try:
                    _evict_trt_vae(engine_path)
                except Exception as e:
                    logger.warning(
                        "TRT VAE cache eviction raised for {}: {}", engine_path, e
                    )
            self._trt_vae_engine_paths = ()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@dataclass
class StreamHandle:
    """Pre-wired streaming graph returned by :meth:`Session.stream`.

    This is a dumb container. All state lives in the underlying
    ``stream_node`` (pipeline, ring buffer) and on the session
    (handler). Callers mutate ``conditioning`` / ``context_latent``
    directly between ticks to swap prompts or blend semantic hints.
    """

    session: Session
    stream_node: Any  # StreamDenoise (avoid circular import)
    decoder_node: Any  # StreamVAEDecode
    model: ModelHandle
    vae: VAEHandle
    source: PreparedSource
    conditioning: Conditioning
    context_latent: Latent
    base_kwargs: dict

    def tick(self, **kwargs: Any) -> Optional[Latent]:
        """Convenience: fire one submit+tick with sensible defaults.

        Merges ``base_kwargs`` with caller kwargs and fills in the
        canonical handle-carried inputs (``model``, ``positive``,
        ``context_latent``, ``source_latent``). Equivalent to calling
        ``stream_node.execute(**merged)`` and returning the ``latent``
        output.

        Per-tick ``method``, ``ode_noise_curve``, or ``sde_denoise_curve``
        kwargs are promoted to a fresh ``Solver`` that overrides the
        handle's base solver for this tick only.
        """
        if any(
            k in kwargs
            for k in ("method", "ode_noise_curve", "sde_denoise_curve")
        ):
            kwargs["solver"] = _build_solver(
                kwargs.pop("method", self.base_kwargs["solver"].method),
                ode_noise_curve=kwargs.pop("ode_noise_curve", None),
                sde_denoise_curve=kwargs.pop("sde_denoise_curve", None),
            )

        if "modulation" not in kwargs:
            mod = _extract_modulation(kwargs)
            if mod is not None:
                kwargs["modulation"] = mod

        merged = {
            **self.base_kwargs,
            "model": self.model,
            "positive": self.conditioning,
            "context_latent": self.context_latent,
            "source_latent": self.source.latent,
            **kwargs,
        }
        return self.stream_node.execute(**merged)["latent"]

    def decode(
        self, latent: Latent, *, t_start: float = 0.0,
    ) -> Audio:
        """Decode a finished latent through the handle's decoder node."""
        return self.decoder_node.execute(
            vae=self.vae,
            latent=latent,
            vae_window=self.session._vae_window,
            vae_overlap=self.session._vae_overlap,
            t_start=t_start,
        )["audio"]

    def stage_source_frames(
        self,
        canvas,
        f0: int,
        f1: int,
        *,
        sample: bool = True,
        engine_path: str | None = None,
    ) -> tuple["torch.Tensor", int, int]:
        """Re-encode source-latent frames ``[f0, f1)`` from the canvas.

        The engine half of real-time input ("play into the model"): the
        source latent is the diffusion anchor and :meth:`tick` re-reads
        ``self.source.latent.tensor`` every call, so re-encoded frames
        committed in place propagate to the next submission with no
        rebuild. This method only STAGES — it returns the encoded block
        without touching the live latent, so the caller can run the GPU
        encode outside its state lock and commit under it
        (:meth:`commit_source_frames`), keeping the edit atomic with
        respect to ticks.

        ``canvas`` is the session's :class:`~acestep.engine.canvas.
        SourceCanvas` (the audio mirror); encode windows read their
        receptive-field context from it, so the staged span is extended
        by the margin on each side and the EXTENDED span ``(e0, e1)`` is
        returned with the block — committing it refreshes the frames
        whose receptive field overlapped the edit, leaving no stale
        border.

        Engine selection is internal: the fixed 5 s windowed encode
        engine when built, else the resident ranged ``vae_encode``
        engine; when the session has no TRT VAE (or the canvas is
        shorter than one window) the whole canvas is re-encoded through
        :meth:`Session.encode_audio` (eager fallback), so short sources
        and eager sessions work instead of erroring.

        ``sample=True`` draws ``mean + std*randn`` like the original
        source encode (a committed span carries the same sampling
        statistics as its surroundings); ``sample=False`` returns the
        deterministic mean, for parity checks where sampling noise
        (rms ~0.022) would mask the signal.

        Returns ``(staged, e0, e1)`` with ``staged`` shaped
        ``[1, e1-e0, D]`` in the source latent's layout/dtype.
        """
        from acestep.paths import (
            WINDOWED_VAE_ENCODE_PROFILE_SAMPLES,
            WINDOWED_VAE_ENCODE_MARGIN_FRAMES,
            available_windowed_vae_encode_engine,
        )
        from acestep.nodes.vae_nodes import (
            _trt_vae_encode_window, _find_best_vae_engine,
        )
        from acestep.engine.canvas import SAMPLES_PER_FRAME

        src = self.source.latent.tensor  # [1, T, D]
        n_frames = int(src.shape[1])
        dtype = src.dtype
        device = src.device
        if canvas.frames != n_frames:
            raise ValueError(
                f"canvas/latent geometry mismatch: canvas covers "
                f"{canvas.frames} frames, source latent has {n_frames} "
                f"(stale canvas after a swap?)"
            )
        if not (0 <= f0 < f1 <= n_frames):
            raise ValueError(
                f"stage span [{f0},{f1}) out of bounds for {n_frames} frames"
            )

        win_frames = WINDOWED_VAE_ENCODE_PROFILE_SAMPLES[1] // SAMPLES_PER_FRAME
        margin = WINDOWED_VAE_ENCODE_MARGIN_FRAMES
        # Extend by the receptive-field margin: frames bordering the
        # edit saw the changed audio inside their receptive field, so
        # re-encode them too rather than leave a stale ring.
        e0 = max(0, f0 - margin)
        e1 = min(n_frames, f1 + margin)

        if engine_path is None:
            win_engine = available_windowed_vae_encode_engine()
            if win_engine is not None:
                engine_path = str(win_engine)
            else:
                engine_path = _find_best_vae_engine("vae_encode")
                if engine_path is not None:
                    logger.warning(
                        "write_audio_no_windowed_engine falling back to "
                        "ranged vae_encode={} (build vae_encode_fp16_5s_fixed "
                        "with python -m acestep.engine.trt.build --all to "
                        "reserve less VRAM)", engine_path,
                    )

        if engine_path is None or canvas.frames < win_frames:
            # Whole-canvas fallback: no TRT VAE resident (eager session)
            # or the canvas is shorter than one fixed window. One full
            # encode through the node path (which routes TRT/eager
            # itself); slice out the span we need.
            lat = self.session.encode_audio(
                Audio(waveform=canvas.wf, sample_rate=48000),
            )
            staged = lat.tensor[:, e0:e1, :].to(device=device, dtype=dtype)
            return staged, e0, e1

        block = canvas.stage_frames(
            e0, e1,
            lambda seg: _trt_vae_encode_window(
                seg, engine_path, device, sample=sample,
            ),
            win_frames=win_frames,
            margin_frames=margin,
        )
        return block.transpose(1, 2).to(device=device, dtype=dtype), e0, e1

    def commit_source_frames(
        self, staged: "torch.Tensor", f0: int, f1: int,
    ) -> None:
        """Write a staged block into the live source latent in place.

        One tensor copy, so an edit is all-or-nothing with respect to a
        concurrent tick reading the latent (unlike encode-and-write-per-
        window, which a tick could observe half-applied). Call under the
        session's state lock, after re-validating the source epoch.
        """
        src = self.source.latent.tensor
        if staged.shape[1] != f1 - f0:
            raise ValueError(
                f"staged block covers {staged.shape[1]} frames; "
                f"span [{f0},{f1}) needs {f1 - f0}"
            )
        src[:, f0:f1, :] = staged

    @property
    def pipeline(self):
        """Raw pipeline (for stats, shared curves, etc.)."""
        return self.stream_node.pipeline

    def stats(self) -> dict:
        return self.stream_node.stats()

    def close(self) -> None:
        """Release the underlying ``StreamPipeline`` and detach refs.

        After ``close()``, ``tick()`` and ``decode()`` will fail. Callers
        must not reuse the handle. Idempotent.
        """
        node = getattr(self, "stream_node", None)
        if node is not None:
            pipeline = getattr(node, "_pipeline", None)
            if pipeline is not None:
                try:
                    pipeline.close()
                except Exception as e:
                    logger.warning("StreamPipeline.close raised: {}", e)
                node._pipeline = None
            node._engine = None
        # Detach top-level refs so the session-level close is the
        # single ground-truth owner of model/vae handles.
        self.stream_node = None
        self.decoder_node = None
        self.model = None  # type: ignore[assignment]
        self.vae = None  # type: ignore[assignment]
        self.source = None  # type: ignore[assignment]
        self.conditioning = None  # type: ignore[assignment]
        self.context_latent = None  # type: ignore[assignment]
        self.base_kwargs = None  # type: ignore[assignment]
