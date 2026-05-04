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
from typing import Any, List, Optional

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
        quantization: Optional[str] = None,
        trt_engines: Optional[dict[str, str]] = None,
        vae_window: float = 0.0,
        vae_overlap: float = 0.5,
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
        """
        from acestep.engine.model_context import ModelContext
        from acestep.engine.runtime_init import (
            apply_trt_backends,
            backends_to_model_context_flags,
            validate_backends,
        )

        trt_engines = validate_backends(
            decoder_backend=decoder_backend,
            vae_backend=vae_backend,
            trt_engines=trt_engines,
        )

        if project_root is None:
            from acestep.paths import checkpoints_dir
            project_root = str(checkpoints_dir())

        ctx_flags = backends_to_model_context_flags(
            decoder_backend=decoder_backend, vae_backend=vae_backend
        )

        ctx = ModelContext(
            project_root=project_root,
            config_path=config_path,
            device=device,
            use_flash_attention=use_flash_attention,
            offload_to_cpu=offload_to_cpu,
            quantization=quantization,
            **ctx_flags,
        )

        self.model = ModelHandle(handler=ctx)
        self.clip = CLIPHandle(handler=ctx)
        self.vae = VAEHandle(handler=ctx)

        # Windowed VAE decode config (seconds; 0 = full decode)
        self._vae_window = vae_window
        self._vae_overlap = vae_overlap

        apply_trt_backends(
            ctx,
            decoder_backend=decoder_backend,
            vae_backend=vae_backend,
            trt_engines=trt_engines,
            device=device,
        )

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
        noise_sharing: float = 0.0,
        pipeline_depth: Optional[int] = None,
        solver: Optional[Solver] = None,
        # DCW (post-step wavelet-domain correction; default on, matching
        # upstream v0.1.7 — see acestep.engine.dcw).
        dcw_enabled: bool = True,
        dcw_mode: str = "double",
        dcw_scaler: float = 0.05,
        dcw_high_scaler: float = 0.02,
        dcw_wavelet: str = "haar",
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
        ``pipeline_depth``, ``noise_sharing``) are baked into the
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
                "noise_sharing": noise_sharing,
                "dcw_enabled": dcw_enabled,
                "dcw_mode": dcw_mode,
                "dcw_scaler": dcw_scaler,
                "dcw_high_scaler": dcw_high_scaler,
                "dcw_wavelet": dcw_wavelet,
            },
        )


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

    @property
    def pipeline(self):
        """Raw pipeline (for noise_sharing mutation, stats, etc.)."""
        return self.stream_node.pipeline

    def stats(self) -> dict:
        return self.stream_node.stats()
