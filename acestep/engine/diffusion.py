from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Union

from loguru import logger
import torch

from .conditions import PreparedCondition
from .dcw import DCWAdvanced


@dataclass
class DiffusionConfig:
    """Configuration for the DiffusionEngine loop.

    Attributes:
        infer_steps: Number of diffusion steps.
        infer_method: Solver type, "ode" (Euler) or "sde" (stochastic).
        shift: Timestep shift for flow matching. ACEStep turbo uses 3.0.
        seed: Random seed for noise generation.
        use_cache: Enable KV caching for cross-attention. Disabled by
            default to match ComfyUI behavior (no KV caching). Enable for
            faster inference when exact ComfyUI parity is not required.
        noise_on_cpu: Generate noise on CPU in [B,D,T] layout then
            transpose to [B,T,D], matching ComfyUI's RandomNoise node.
            When False, uses the HF model's prepare_noise (GPU, [B,T,D]).
        timesteps: Explicit timestep schedule (overrides infer_steps/shift).
        denoise: Denoising strength in [0, 1]. Controls how much of the
            source audio to preserve vs regenerate:
              1.0 = generate from pure noise (full generation)
              0.5 = start halfway through the noise schedule (style transfer)
              0.0 = no denoising (passthrough)
            When < 1.0, source_latents must be provided to generate().
            The schedule is computed as int(steps/denoise) full steps, then
            the last (steps+1) entries are used (matching ComfyUI behavior).
    """

    infer_steps: int = 8
    infer_method: str = "ode"
    shift: float = 3.0
    seed: Optional[Union[int, List[int]]] = None
    use_cache: bool = False
    noise_on_cpu: bool = True
    timesteps: Optional[List[float]] = None
    denoise: float = 1.0
    x0_target_gate: float = 0.0
    # DCW (Differential Correction in Wavelet domain) — sampler-side
    # post-step correction ported from upstream v0.1.7. See
    # ``acestep.engine.dcw`` for the math. On by default, matching
    # upstream v0.1.7.
    dcw_enabled: bool = True
    dcw_mode: str = "double"
    dcw_scaler: float = 0.05
    dcw_high_scaler: float = 0.02
    dcw_wavelet: str = "haar"
    # Opt-in advanced surface (mult_blend / mag_phase / soft_thresh).
    # ``None`` keeps the byte-identical upstream-v0.1.7 fast path.
    dcw_advanced: Optional[DCWAdvanced] = None


class DiffusionEngine:
    """TRT/decoder state holder + schedule builder for ACE-Step.

    Phase 3 removed ``DiffusionEngine.generate()`` — all generation
    (one-shot and streaming) now goes through the
    :class:`~acestep.nodes.diffusion_nodes.StreamDenoise` node, which
    owns the sole ``StreamPipeline`` construction site in the codebase.
    The engine remains for TRT engine/LoRA-refit management and for
    sharing the timestep-schedule helper with ``StreamPipeline``.
    """

    def __init__(
        self,
        model,
        trt_engine_path=None,
        compile_loops: bool = True,
        *,
        pending_decoder=None,
        trt_decoder_bytes_future=None,
        lora_discovery_future=None,
    ):
        """
        Args:
            model: AceStepConditionGenerationModel instance.
            trt_engine_path: Optional path to a TRT decoder engine file.
                When provided, the engine is loaded via polygraphy and
                all decoder calls are routed through TensorRT. All engine
                modulations (temporal blending, velocity scaling, noise
                masks, etc.) continue to work because they operate on the
                velocity output, not inside the decoder.
            compile_loops: When True (default), the ODE/SDE inner loops
                are wrapped with ``torch.compile`` on first use. Set to
                False to skip the compile and run the loops eagerly,
                avoiding the autotune warmup.
            pending_decoder: Real decoder module retained on CPU when
                ``ModelContext`` ran with ``skip_decoder=True``. Hands
                live base weights to ``TRTLoRAManager`` so it doesn't
                re-read the checkpoint shards. Dropped after the manager
                is built.
            trt_decoder_bytes_future: Optional Future[bytes] that has been
                pre-reading the TRT engine file from disk in the background
                during ModelContext init. When provided, load_trt_engine
                skips its own bytes_from_path call.
            lora_discovery_future: Optional Future[list[Path]] that has been
                walking the LoRA roots in the background. When provided,
                register_library uses the pre-discovered file list instead
                of re-walking the filesystem.
        """
        self.model = model
        self.decoder = model.decoder
        self._pending_decoder = pending_decoder
        self._trt_decoder_bytes_future = trt_decoder_bytes_future
        self._lora_discovery_future = lora_discovery_future
        self._compile_loops = compile_loops

        # TRT state (owned directly, no wrapper class).
        # Uses polygraphy engine loading and polygraphy CUDA stream to
        # avoid Blackwell multi-engine kernel slowdown.
        self._trt_engine = None
        self._trt_ctx = None
        self._trt_stream = None
        self._trt_buf_cache: dict[tuple, dict] = {}

        # Engine-swap listeners. StreamPipeline (and any other consumer
        # that snapshots ``self._trt_*`` at construction time) registers
        # a callback here so it can re-read the live refs and invalidate
        # any cached I/O bindings after the profile manager swaps the
        # underlying engine.
        self._engine_swap_listeners: list = []

        # Dynamic LoRA manager. TRT path constructs TRTLoRAManager inside
        # load_trt_engine (after the engine is up and refit support is
        # confirmed). Eager path constructs EagerLoRAManager up-front so
        # the LoRA library is usable from tick 0 with no prerequisites.
        self._lora_manager = None

        if trt_engine_path is not None:
            self.load_trt_engine(trt_engine_path)
        else:
            self._init_eager_lora_manager()

    # ------------------------------------------------------------------
    # TRT engine management
    # ------------------------------------------------------------------

    def load_trt_engine(self, engine_path):
        """Load a TRT decoder engine via polygraphy.

        Uses polygraphy.backend.trt.engine_from_bytes (not
        trt.Runtime().deserialize_cuda_engine) to avoid process-global
        TRT state corruption on Blackwell GPUs with multiple engines.
        Shares the process-wide polygraphy CUDA stream with VAE engines.
        """
        from pathlib import Path
        from polygraphy.backend.common import bytes_from_path
        from polygraphy.backend.trt import engine_from_bytes
        from acestep.nodes.vae_nodes import _get_trt_stream

        engine_path = Path(engine_path)
        if not engine_path.exists():
            raise FileNotFoundError(f"TRT engine not found: {engine_path}")

        logger.info("Loading TRT decoder engine from {} ...", engine_path)
        # Prefer the pre-read bytes from the background preloader (started
        # at Session.__init__) — by now the file is already in RAM and the
        # disk read overlapped DiT shard loading. Fall back to a fresh
        # bytes_from_path on direct DiffusionEngine callers (tests, etc.)
        # or if the preload errored out.
        engine_bytes = None
        preload_future = self._trt_decoder_bytes_future
        if preload_future is not None:
            try:
                engine_bytes = preload_future.result()
            except Exception as exc:
                logger.warning(
                    "TRT bytes preload failed ({}); re-reading from disk.", exc,
                )
            self._trt_decoder_bytes_future = None
        if engine_bytes is None:
            engine_bytes = bytes_from_path(str(engine_path))
        self._trt_engine = engine_from_bytes(engine_bytes)
        # Drop the local bytes reference so the GB-scale buffer can be
        # freed; engine_from_bytes has consumed it.
        del engine_bytes
        self._trt_ctx = self._trt_engine.create_execution_context()
        self._trt_stream = _get_trt_stream()
        self._trt_buf_cache = {}

        # Detect per-tensor I/O dtypes from the engine. Strongly-typed
        # builds may use fp16/bf16/fp32 on different inputs, and TensorRT
        # interprets buffers by the serialized tensor dtype.
        import tensorrt as trt
        _trt_dtype_map = {
            trt.float32: torch.float32,
            trt.float16: torch.float16,
        }
        if hasattr(trt, "bfloat16"):
            _trt_dtype_map[trt.bfloat16] = torch.bfloat16

        engine_input_names = {
            self._trt_engine.get_tensor_name(i)
            for i in range(self._trt_engine.num_io_tensors)
        }
        # ``steering`` is present on spectral-prefixed engines only;
        # older builds get _steering_num_layers == 0 and the streaming
        # pipeline skips the buffer entirely.
        has_steering = "steering" in engine_input_names
        base_inputs = (
            "hidden_states", "timestep", "encoder_hidden_states",
            "context_latents",
        )
        input_names = base_inputs + (("steering",) if has_steering else ())
        self._trt_input_dtypes = {
            name: _trt_dtype_map.get(self._trt_engine.get_tensor_dtype(name), torch.float32)
            for name in input_names
        }
        if has_steering:
            steer_shape = tuple(self._trt_engine.get_tensor_shape("steering"))
            if len(steer_shape) != 3:
                raise RuntimeError(f"Unexpected steering input rank: {steer_shape}")
            self._steering_num_layers = int(steer_shape[1])
            self._steering_hidden_size = int(steer_shape[2])
        else:
            self._steering_num_layers = 0
            self._steering_hidden_size = 0
            logger.warning(
                "TRT decoder engine has no 'steering' input — activation-"
                "steering knobs will no-op on this session. Rebuild with "
                "`python -m acestep.engine.trt.build --all --force-rebuild` "
                "to enable."
            )
        self._trt_output_dtype = _trt_dtype_map.get(
            self._trt_engine.get_tensor_dtype("velocity"), torch.float32
        )
        self._trt_io_dtype = self._trt_input_dtypes["hidden_states"]
        logger.info(
            "TRT decoder engine ready (input_dtypes={}, output_dtype={}, "
            "steering L={}, D={})",
            self._trt_input_dtypes,
            self._trt_output_dtype,
            self._steering_num_layers,
            self._steering_hidden_size,
        )

        # Try to initialize LoRA refit manager (requires REFIT-enabled engine)
        self._lora_manager = None
        try:
            from acestep.engine.trt.lora_refit import TRTLoRAManager

            # Find checkpoint for base weights (needed when decoder is
            # discarded in TRT mode). Pass the checkpoint DIR; the
            # lora_refit module handles both single-file
            # (``model.safetensors``) and HF sharded
            # (``model.safetensors.index.json`` + shards) layouts. The 2B
            # turbo decoder fits in a single shard; the XL decoder is
            # split across 4. Passing only the single-file path used to
            # silently leave 0/1184 weights mapped on XL because the
            # candidate didn't exist.
            ckpt_path = None
            cfg = getattr(self.model, "config", None)
            if cfg is not None:
                import os
                ckpt_dir = getattr(cfg, "_name_or_path", "")
                if ckpt_dir and os.path.isdir(ckpt_dir):
                    ckpt_path = ckpt_dir

            # Prefer the stashed real decoder (CPU tensors retained by
            # ModelContext when skip_decoder=True) over the empty stub
            # sitting on self.decoder. Falls back to the checkpoint
            # disk read only when neither is available.
            lora_decoder = self._pending_decoder if self._pending_decoder is not None else self.decoder
            self._lora_manager = TRTLoRAManager(
                engine=self._trt_engine,
                decoder=lora_decoder,
                device=torch.device("cuda"),
                trt_weight_prefix="decoder.",
                checkpoint_path=ckpt_path,
                engine_path=str(engine_path),
            )
            # Drop the temporary reference — the manager has pinned its
            # own copies of the base weights. Holding it would pin ~6 GB
            # of CPU RAM for the life of the engine.
            self._pending_decoder = None
            # Pre-register the on-disk library (MODELS_DIR/loras).  This
            # is the catalog backing the "infinite library" workflow:
            # register every .safetensors as REGISTERED (zero RAM cost),
            # callers materialize on demand via enable_lora.
            try:
                # Consume the pre-discovered file list if Session preloaded
                # it in a background thread. Falls back to a fresh scan
                # when there's no future (direct DiffusionEngine callers)
                # or when the preload errored out.
                discovered = None
                lora_future = self._lora_discovery_future
                if lora_future is not None:
                    try:
                        discovered = lora_future.result()
                    except Exception as exc:
                        logger.warning(
                            "LoRA discovery preload failed ({}); re-scanning.",
                            exc,
                        )
                    self._lora_discovery_future = None
                if discovered is not None:
                    # register_library(directory=...) would re-walk one root.
                    # We already have the union across primary + extra dirs,
                    # so iterate register_lora directly and emit the same
                    # summary line for log parity.
                    ids: list[str] = []
                    for p in discovered:
                        try:
                            ids.append(self._lora_manager.register_lora(str(p)))
                        except Exception as exc:
                            logger.warning("Failed to register {}: {}", p, exc)
                    if discovered:
                        logger.info(
                            "Registered library: {} LoRAs across all "
                            "configured root(s)", len(ids),
                        )
                else:
                    self._lora_manager.register_library()
            except Exception as e:
                logger.warning("Failed to scan LoRA library: {}", e)
        except RuntimeError as e:
            # Engine not built with REFIT, or TRT version too old
            logger.info("TRT LoRA refit not available: {}", e)
        except Exception as e:
            logger.warning("Failed to init TRT LoRA manager: {}", e)

        # Notify subscribers AFTER LoRA wiring is up so a listener that
        # eagerly probes the new engine sees a fully-initialized state.
        # First-time loads (from __init__) hit this with an empty list,
        # which is a cheap no-op.
        self._fire_engine_swap_listeners()

    def add_engine_swap_listener(self, callback) -> None:
        """Register a zero-arg callback fired after every successful
        ``load_trt_engine`` (including the implicit one from __init__).

        Used by ``StreamPipeline`` to re-read the engine's ``_trt_ctx``
        / ``_trt_engine`` / ``_trt_io_dtype`` and drop its shape-keyed
        buffer cache so the next forward pass binds against the new
        engine's profile.
        """
        if callback not in self._engine_swap_listeners:
            self._engine_swap_listeners.append(callback)

    def remove_engine_swap_listener(self, callback) -> None:
        """Detach a listener previously added via
        :meth:`add_engine_swap_listener`. Idempotent."""
        try:
            self._engine_swap_listeners.remove(callback)
        except ValueError:
            pass

    def _fire_engine_swap_listeners(self) -> None:
        """Invoke every registered listener; isolate exceptions so a
        broken listener can't prevent others from running or leave the
        engine in a half-swapped state."""
        for cb in list(self._engine_swap_listeners):
            try:
                cb()
            except Exception as e:
                logger.warning("Engine swap listener raised: {}", e)

    def unload_trt_engine(self) -> None:
        """Drop the active TRT decoder engine and free its GPU workspace.

        Pair with :meth:`load_trt_engine` to swap profiles in-place: the
        TRT engine + execution context together pin several GB of
        workspace, so we MUST release them before loading the next
        engine or VRAM doubles up. The shared polygraphy stream
        survives — it's process-global and reused by every TRT engine.

        Active LoRAs are NOT preserved here: the new engine gets a fresh
        ``TRTLoRAManager`` from ``load_trt_engine``. Callers who need
        continuity should snapshot ENABLED entries before this call and
        re-enable them after the new engine is up (see
        ``acestep.engine.trt.profile_manager.TRTProfileManager``).
        """
        # Drop refit hooks first so a stray callback can't fire on the
        # disposed engine.
        self._lora_manager = None
        # Per-shape buffers reference TRT-owned addresses; clear before
        # the context goes.
        self._trt_buf_cache = {}
        self._trt_ctx = None
        self._trt_engine = None
        torch.cuda.empty_cache()
        logger.info("Unloaded TRT decoder engine")

    # ------------------------------------------------------------------
    # LoRA management (backend-agnostic)
    #
    # Delegates to the active manager (TRTLoRAManager when a TRT engine
    # is loaded with REFIT support, EagerLoRAManager otherwise). All
    # public methods here are stable across backends; the manager
    # subclass differs only in *where* the weight writeback lands.
    # ------------------------------------------------------------------

    def _init_eager_lora_manager(self) -> None:
        """Construct an EagerLoRAManager around the live decoder.

        Skipped silently when the decoder has no parameters (e.g. the
        skip_decoder TRT path before load_trt_engine has run, or a
        pure-VAE/text-encoder ModelContext).
        """
        from acestep.engine.lora import EagerLoRAManager

        try:
            self._lora_manager = EagerLoRAManager(decoder=self.decoder)
        except (RuntimeError, ValueError) as e:
            logger.info("Eager LoRA manager not available: {}", e)
            self._lora_manager = None
            return

        try:
            self._lora_manager.register_library()
        except Exception as e:
            logger.warning("Failed to scan LoRA library: {}", e)

    def apply_lora(self, lora_path: str, strength: float = 1.0) -> str:
        """Register + enable a LoRA in one call. Returns the LoRA id.

        Raises:
            RuntimeError: If no LoRA backend is available.
        """
        self._require_lora_manager()
        return self._lora_manager.apply_lora(lora_path, strength)

    def remove_lora(self, lora_id=-1) -> bool:
        """Remove a LoRA (default: most recently registered)."""
        if self._lora_manager is None:
            return False
        return self._lora_manager.remove_lora(lora_id)

    def set_lora_strength(self, lora_id: str, strength: float) -> None:
        """Adjust the strength of an ENABLED LoRA."""
        self._require_lora_manager()
        self._lora_manager.set_lora_strength(lora_id, strength)

    def remove_all_loras(self) -> None:
        """Remove all LoRAs and restore the decoder to base weights."""
        if self._lora_manager is not None:
            self._lora_manager.remove_all()

    def register_lora(self, lora_path: str, name: str | None = None) -> str:
        """Add a LoRA to the catalog without materializing deltas."""
        self._require_lora_manager()
        return self._lora_manager.register_lora(lora_path, name=name)

    def enable_lora(
        self, lora_id: str, strength: float | None = None,
    ) -> None:
        """Promote a registered LoRA to ENABLED (materialize + refit).

        ``strength``, when provided, sets the entry's strength BEFORE the
        refit so the first decode window already sees the LoRA at its
        target strength — avoids the "first ~5s sounds like the LoRA is
        missing" glitch caused by enabling at strength 0 and waiting for
        the next per-tick set_strength call to ramp it up.
        """
        self._require_lora_manager()
        self._lora_manager.enable_lora(lora_id, strength=strength)

    def disable_lora(self, lora_id: str) -> None:
        """Drop a LoRA's deltas. Strength is preserved on the entry."""
        self._require_lora_manager()
        self._lora_manager.disable_lora(lora_id)

    def prewarm_lora(self, lora_id: str):
        """Kick off background delta materialization. Returns a Future."""
        self._require_lora_manager()
        return self._lora_manager.prewarm_lora(lora_id)

    def list_loras(self):
        """Return descriptors for every entry. Empty list when no manager."""
        if self._lora_manager is None:
            return []
        return self._lora_manager.list_loras()

    def get_lora(self, lora_id: str):
        """Return the current descriptor for ``lora_id``."""
        self._require_lora_manager()
        return self._lora_manager.get_lora(lora_id)

    @property
    def lora_available(self) -> bool:
        """True if a LoRA backend is initialized for this engine."""
        return self._lora_manager is not None

    def _require_lora_manager(self) -> None:
        if self._lora_manager is None:
            raise RuntimeError(
                "LoRA manager not available. For the TRT path, rebuild "
                "the decoder engine with refit=True. For the PyTorch "
                "path, the decoder must have parameters loaded."
            )

    def close(self) -> None:
        """Release per-engine TRT + LoRA state.

        Called by :meth:`acestep.engine.model_context.ModelContext.close`
        as part of the :meth:`acestep.engine.session.Session.close` chain.
        Drops:

        - the TRT execution context (the dominant non-PyTorch GPU buffer:
          activation/workspace memory, ~1–2 GB for a turbo decoder profile)
        - the deserialized engine itself (~1.5–3 GB GPU)
        - the per-shape input/output buffer cache (``_trt_buf_cache``)
        - the LoRA manager (CPU base/refit buffers + GPU mirror on the
          eager backend)

        These are *not* freed by Python GC reliably — the polygraphy /
        pycuda objects keep CUDA references alive until their finalizers
        run, which can be arbitrarily delayed under reference cycles.
        Explicit ``del`` here forces the destructor chain immediately.

        Idempotent: subsequent calls are no-ops.
        """
        # Drop the buffer cache first (each entry holds a torch.Tensor on
        # CUDA); it indirectly references the engine via the context.
        self._trt_buf_cache.clear()
        # LoRA manager next. It holds a Refitter, which holds an engine
        # reference. Closing it before the engine clears that ref.
        if self._lora_manager is not None:
            try:
                self._lora_manager.close()
            except Exception as e:
                logger.warning("LoRAManagerBase.close raised: {}", e)
            self._lora_manager = None
        # Now the TRT context + engine. ``del`` on the instance attribute
        # is what triggers the polygraphy/pycuda finalizer that frees the
        # CUDA workspace. Rebinding to None works only when no other
        # name holds the previous value; ``del`` removes the binding
        # unconditionally.
        for attr in ("_trt_ctx", "_trt_engine"):
            if hasattr(self, attr):
                delattr(self, attr)
        # Re-establish the attributes as None so the rest of the API
        # (load_trt_engine, etc.) keeps working if the engine is reused.
        self._trt_ctx = None
        self._trt_engine = None
        # The CUDA stream is process-shared (see vae_nodes._get_trt_stream)
        # so we just drop our reference, not destroy the stream.
        self._trt_stream = None
        # Drop the decoder reference too — it pins the DiT graph that
        # ModelContext is about to release.
        self.decoder = None
        self.model = None

    def _trt_decoder_step(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        context_latents: torch.Tensor,
        steering: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run one decoder step through TRT with pre-allocated buffers.

        Handles odd-T padding. Caches buffers by shape for reuse across
        steps. Calls execute_async_v3 on the shared polygraphy stream.

        ``steering`` is optional ``[B, num_layers, hidden_size]``;
        when omitted the cached zero buffer makes the per-layer adds a no-op.
        """
        orig_T = hidden_states.shape[1]
        pad = orig_T % 2 == 1
        eff_T = orig_T + 1 if pad else orig_T

        key = (
            (hidden_states.shape[0], eff_T, 64),
            tuple(timestep.shape),
            tuple(encoder_hidden_states.shape),
            (context_latents.shape[0], eff_T, 128),
        )

        if key not in self._trt_buf_cache:
            ctx = self._trt_ctx
            dev = hidden_states.device
            hs_shape, ts_shape, enc_shape, cl_shape = key
            in_dtypes = self._trt_input_dtypes
            B = hs_shape[0]

            bufs = {
                "hidden_states": torch.empty(hs_shape, dtype=in_dtypes["hidden_states"], device=dev),
                "timestep": torch.empty(ts_shape, dtype=in_dtypes["timestep"], device=dev),
                "encoder_hidden_states": torch.empty(enc_shape, dtype=in_dtypes["encoder_hidden_states"], device=dev),
                "context_latents": torch.empty(cl_shape, dtype=in_dtypes["context_latents"], device=dev),
            }
            if self._steering_num_layers > 0:
                bufs["steering"] = torch.zeros(
                    B, self._steering_num_layers, self._steering_hidden_size,
                    dtype=in_dtypes["steering"], device=dev,
                )
            for name, buf in bufs.items():
                if not ctx.set_input_shape(name, tuple(buf.shape)):
                    raise RuntimeError(f"TRT decoder rejected input shape for {name}: {tuple(buf.shape)}")
                if not ctx.set_tensor_address(name, buf.data_ptr()):
                    raise RuntimeError(f"TRT decoder rejected input address for {name}")

            missing = ctx.infer_shapes()
            if missing:
                raise RuntimeError(f"TRT decoder shapes are insufficiently specified: {missing}")

            out_shape = tuple(ctx.get_tensor_shape("velocity"))
            out_buf = torch.empty(out_shape, dtype=self._trt_output_dtype, device=dev)
            if not ctx.set_tensor_address("velocity", out_buf.data_ptr()):
                raise RuntimeError("TRT decoder rejected output address for velocity")

            self._trt_buf_cache[key] = {"bufs": bufs, "output": out_buf}
            logger.info(
                "Allocated TRT buffers for shapes: hs={} enc={}",
                list(hs_shape), list(enc_shape),
            )

        entry = self._trt_buf_cache[key]
        bufs = entry["bufs"]

        if pad:
            bufs["hidden_states"][:, :orig_T, :].copy_(hidden_states)
            bufs["hidden_states"][:, orig_T:, :].zero_()
            bufs["context_latents"][:, :orig_T, :].copy_(context_latents)
            bufs["context_latents"][:, orig_T:, :].zero_()
        else:
            bufs["hidden_states"].copy_(hidden_states)
            bufs["context_latents"].copy_(context_latents)
        bufs["timestep"].copy_(timestep)
        bufs["encoder_hidden_states"].copy_(encoder_hidden_states)
        if "steering" in bufs:
            if steering is not None:
                bufs["steering"].copy_(steering)
            else:
                bufs["steering"].zero_()

        ctx = self._trt_ctx
        for name, buf in bufs.items():
            if not ctx.set_tensor_address(name, buf.data_ptr()):
                raise RuntimeError(f"TRT decoder rejected input address for {name}")
        if not ctx.set_tensor_address("velocity", entry["output"].data_ptr()):
            raise RuntimeError("TRT decoder rejected output address for velocity")

        if not ctx.execute_async_v3(self._trt_stream.ptr):
            raise RuntimeError("TRT decoder execution failed")
        self._trt_stream.synchronize()

        output = entry["output"]
        return output[:, :orig_T, :] if pad else output

    # ------------------------------------------------------------------
    # Noise generation
    # ------------------------------------------------------------------

    def _prepare_noise_cpu(
        self, ref_cond: PreparedCondition, seed: Optional[Union[int, List[int]]]
    ) -> torch.Tensor:
        """Generate noise on CPU in [B,D,T] layout, then transpose to [B,T,D].

        This matches ComfyUI's RandomNoise node which generates noise on CPU
        with torch.manual_seed, producing different values than GPU generation.
        """
        bsz = ref_cond.batch_size
        T = ref_cond.seq_len
        D = ref_cond.context_latents.shape[-1] // 2  # context_latents is [B,T,D*2]
        device = ref_cond.device
        dtype = ref_cond.dtype

        if seed is not None and not isinstance(seed, list):
            torch.manual_seed(int(seed))
            noise_bdt = torch.randn(bsz, D, T, device="cpu", dtype=torch.float32)
        elif isinstance(seed, list):
            noise_list = []
            for s in seed:
                if s is not None and s >= 0:
                    torch.manual_seed(int(s))
                noise_list.append(torch.randn(1, D, T, device="cpu", dtype=torch.float32))
            noise_bdt = torch.cat(noise_list, dim=0)
        else:
            noise_bdt = torch.randn(bsz, D, T, device="cpu", dtype=torch.float32)

        return noise_bdt.movedim(-1, -2).to(device=device, dtype=dtype)

    # ------------------------------------------------------------------
    # Timestep schedule
    # ------------------------------------------------------------------

    # Treat any positive denoise below this as "no denoise". The last
    # ``steps + 1`` entries of the truncated schedule for ``denoise <
    # _DENOISE_MIN`` are indistinguishable from zero at the engine's
    # working precision (bf16 / fp16 / fp32) — the schedule starts at
    # ``denoise`` and descends to 0, so anything in this range is
    # numerical noise. The cap also guards against a UI tween briefly
    # writing a near-zero positive ``denoise`` mid-glide (e.g. the
    # "hear source first" gate animating the ribbon down to 0): without
    # it, ``int(steps / denoise)`` blows up into a 512-PiB ``linspace``
    # request and OOMs on the first ``_init_slot`` after the swap.
    _DENOISE_MIN: float = 1e-6

    def _build_timestep_schedule(
        self, config: DiffusionConfig, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Build the timestep schedule, respecting denoise truncation.

        When ``denoise < 1.0`` the schedule is the last ``steps + 1``
        entries of ``linspace(1.0, 0.0, int(steps/denoise) + 1)``. We
        compute those entries directly as
        ``linspace(steps/full_steps, 0.0, steps + 1)`` instead of
        materializing the full descending range and slicing — the
        intermediate is unbounded in ``denoise`` and a near-zero positive
        ``denoise`` (subnormal-ish floats from a UI tween) used to
        request a ~512-PiB GPU allocation here. The direct form matches
        ComfyUI's ``BasicScheduler.set_steps()`` for all valid inputs
        (``shift`` is elementwise, so applying it after the slice is
        equivalent to applying it before).
        """
        if config.timesteps is not None:
            return torch.tensor(config.timesteps, device=device, dtype=dtype)

        steps = config.infer_steps
        denoise = config.denoise

        if denoise <= self._DENOISE_MIN:
            # No (meaningful) denoising: single-entry schedule (t=0 -> t=0)
            return torch.zeros(2, device=device, dtype=dtype)

        if denoise >= 1.0:
            t_start = 1.0
        else:
            full_steps = int(steps / denoise)
            # full_steps is at most steps / _DENOISE_MIN; well-bounded.
            t_start = steps / full_steps  # ≈ denoise, exact slice endpoint

        t_schedule = torch.linspace(
            t_start, 0.0, steps + 1, device=device, dtype=dtype
        )
        if config.shift != 1.0:
            t_schedule = (
                config.shift * t_schedule
                / (1 + (config.shift - 1) * t_schedule)
            )

        return t_schedule
