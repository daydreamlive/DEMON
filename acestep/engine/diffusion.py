from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Union

from loguru import logger
import torch

from .conditions import PreparedCondition


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


class DiffusionEngine:
    """TRT/decoder state holder + schedule builder for ACE-Step.

    Phase 3 removed ``DiffusionEngine.generate()`` — all generation
    (one-shot and streaming) now goes through the
    :class:`~acestep.nodes.diffusion_nodes.StreamDenoise` node, which
    owns the sole ``StreamPipeline`` construction site in the codebase.
    The engine remains for TRT engine/LoRA-refit management and for
    sharing the timestep-schedule helper with ``StreamPipeline``.
    """

    def __init__(self, model, trt_engine_path=None, compile_loops: bool = True):
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
        """
        self.model = model
        self.decoder = model.decoder
        self._compile_loops = compile_loops

        # TRT state (owned directly, no wrapper class).
        # Uses polygraphy engine loading and polygraphy CUDA stream to
        # avoid Blackwell multi-engine kernel slowdown.
        self._trt_engine = None
        self._trt_ctx = None
        self._trt_stream = None
        self._trt_buf_cache: dict[tuple, dict] = {}

        # Dynamic LoRA via TRT weight refitting (initialized in load_trt_engine
        # when the engine supports REFIT).
        self._lora_manager = None

        if trt_engine_path is not None:
            self.load_trt_engine(trt_engine_path)

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

        logger.info("Loading TRT decoder engine from %s ...", engine_path)
        self._trt_engine = engine_from_bytes(bytes_from_path(str(engine_path)))
        self._trt_ctx = self._trt_engine.create_execution_context()
        self._trt_stream = _get_trt_stream()
        self._trt_buf_cache = {}

        # Detect I/O dtypes from engine (fp16 for mixed-precision, fp32 for legacy)
        import tensorrt as trt
        _trt_dtype_map = {
            trt.float32: torch.float32,
            trt.float16: torch.float16,
            trt.bfloat16: torch.bfloat16,
        }
        hs_trt_dtype = self._trt_engine.get_tensor_dtype("hidden_states")
        self._trt_io_dtype = _trt_dtype_map.get(hs_trt_dtype, torch.float32)
        # Per-input dtype map. timestep may be fp32 or bf16 depending on
        # the export recipe (older bf16_mixed code put time_embed in fp32,
        # current bf16_mixed code leaves it in bf16). Reading the engine's
        # declared dtype avoids silent corruption when the buffer dtype
        # mismatches what TRT expects under STRONGLY_TYPED.
        self._trt_input_dtypes = {
            n: _trt_dtype_map.get(self._trt_engine.get_tensor_dtype(n), torch.float32)
            for n in ("hidden_states", "timestep", "encoder_hidden_states", "context_latents")
        }
        logger.info("TRT decoder engine ready (io_dtype=%s)", self._trt_io_dtype)

        # Try to initialize LoRA refit manager (requires REFIT-enabled engine)
        self._lora_manager = None
        try:
            from acestep.engine.trt.lora_refit import TRTLoRAManager

            # Find checkpoint for base weights (needed when decoder is
            # discarded in TRT mode)
            ckpt_path = None
            cfg = getattr(self.model, "config", None)
            if cfg is not None:
                import os
                candidate = os.path.join(
                    getattr(cfg, "_name_or_path", ""), "model.safetensors"
                )
                if os.path.exists(candidate):
                    ckpt_path = candidate

            self._lora_manager = TRTLoRAManager(
                engine=self._trt_engine,
                decoder=self.decoder,
                device=torch.device("cuda"),
                trt_weight_prefix="decoder.",
                checkpoint_path=ckpt_path,
            )
            # Pre-register the on-disk library (MODELS_DIR/loras).  This
            # is the catalog backing the "infinite library" workflow:
            # register every .safetensors as REGISTERED (zero RAM cost),
            # callers materialize on demand via enable_trt_lora.
            try:
                self._lora_manager.register_library()
            except Exception as e:
                logger.warning("Failed to scan LoRA library: %s", e)
        except RuntimeError as e:
            # Engine not built with REFIT, or TRT version too old
            logger.info("TRT LoRA refit not available: %s", e)
        except Exception as e:
            logger.warning("Failed to init TRT LoRA manager: %s", e)

    # ------------------------------------------------------------------
    # TRT LoRA management (delegates to TRTLoRAManager)
    # ------------------------------------------------------------------

    def apply_trt_lora(self, lora_path: str, strength: float = 1.0) -> int:
        """Apply a LoRA to the TRT engine via weight refitting.

        Args:
            lora_path: Path to .safetensors LoRA file.
            strength: LoRA strength (0.0 = no effect, 1.0 = full).

        Returns:
            LoRA ID for later removal or strength adjustment.

        Raises:
            RuntimeError: If TRT engine doesn't support refit.
        """
        if self._lora_manager is None:
            raise RuntimeError(
                "TRT LoRA refit not available. Rebuild the decoder engine "
                "with refit=True (OnnxExportConfig.for_refit=True + "
                "TRTBuildConfig.refit=True)."
            )
        return self._lora_manager.apply_lora(lora_path, strength)

    def remove_trt_lora(self, lora_id: int = -1) -> bool:
        """Remove a LoRA from the TRT engine. Default: most recent."""
        if self._lora_manager is None:
            return False
        return self._lora_manager.remove_lora(lora_id)

    def set_trt_lora_strength(self, lora_id: int, strength: float) -> None:
        """Adjust strength of an active TRT LoRA."""
        if self._lora_manager is None:
            raise RuntimeError("TRT LoRA refit not available")
        self._lora_manager.set_lora_strength(lora_id, strength)

    def remove_all_trt_loras(self) -> None:
        """Remove all LoRAs and restore the TRT engine to base weights."""
        if self._lora_manager is not None:
            self._lora_manager.remove_all()

    # ------------------------------------------------------------------
    # Lifecycle API: register / enable / disable / prewarm
    # ------------------------------------------------------------------

    def register_trt_lora(self, lora_path: str, name: str | None = None) -> str:
        """Add a LoRA to the catalog without materializing deltas.

        Returns the (filename-stem) id used for subsequent enable/disable
        calls.  Idempotent on path.
        """
        if self._lora_manager is None:
            raise RuntimeError("TRT LoRA refit not available")
        return self._lora_manager.register_lora(lora_path, name=name)

    def enable_trt_lora(
        self, lora_id: str, strength: float | None = None,
    ) -> None:
        """Promote a registered LoRA to ENABLED (materialize + refit).

        ``strength``, when provided, sets the entry's strength BEFORE the
        refit so the streaming pipeline's first decode window already
        sees the LoRA at its target strength — avoids the "first ~5s
        sounds like the LoRA is missing" glitch caused by enabling at
        strength 0 and waiting for the next per-tick set_strength call
        to ramp it up.

        Synchronous; if a prewarm is in flight, blocks on it.  Refit only
        fires when the resulting strength is non-zero.
        """
        if self._lora_manager is None:
            raise RuntimeError("TRT LoRA refit not available")
        self._lora_manager.enable_lora(lora_id, strength=strength)

    def disable_trt_lora(self, lora_id: str) -> None:
        """Drop a LoRA's deltas from CPU RAM.  Strength is preserved."""
        if self._lora_manager is None:
            raise RuntimeError("TRT LoRA refit not available")
        self._lora_manager.disable_lora(lora_id)

    def prewarm_trt_lora(self, lora_id: str):
        """Kick off background materialization.  Returns a Future.

        Use this when the catalog is large but only some LoRAs will
        actually be enabled — call ``prewarm_trt_lora`` for the
        likely-enabled subset at session start so the eventual
        ``enable_trt_lora`` call is fast.
        """
        if self._lora_manager is None:
            raise RuntimeError("TRT LoRA refit not available")
        return self._lora_manager.prewarm_lora(lora_id)

    def list_trt_loras(self):
        """Return a list of :class:`LoRADescriptor` for every entry in
        the catalog.  Empty list if the manager is unavailable."""
        if self._lora_manager is None:
            return []
        return self._lora_manager.list_loras()

    def get_trt_lora(self, lora_id: str):
        """Return the current :class:`LoRADescriptor` for ``lora_id``."""
        if self._lora_manager is None:
            raise RuntimeError("TRT LoRA refit not available")
        return self._lora_manager.get_lora(lora_id)

    @property
    def trt_lora_available(self) -> bool:
        """True if the TRT engine supports dynamic LoRA via refit."""
        return self._lora_manager is not None

    def _trt_decoder_step(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        context_latents: torch.Tensor,
    ) -> torch.Tensor:
        """Run one decoder step through TRT with pre-allocated buffers.

        Handles odd-T padding. Caches buffers by shape for reuse across
        steps. Calls execute_async_v3 on the shared polygraphy stream.
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
            io_dtype = self._trt_io_dtype

            in_dt = self._trt_input_dtypes
            bufs = {
                "hidden_states": torch.empty(hs_shape, dtype=in_dt["hidden_states"], device=dev),
                "timestep": torch.empty(ts_shape, dtype=in_dt["timestep"], device=dev),
                "encoder_hidden_states": torch.empty(enc_shape, dtype=in_dt["encoder_hidden_states"], device=dev),
                "context_latents": torch.empty(cl_shape, dtype=in_dt["context_latents"], device=dev),
            }
            for name, buf in bufs.items():
                ctx.set_input_shape(name, tuple(buf.shape))
                ctx.set_tensor_address(name, buf.data_ptr())

            out_shape = tuple(ctx.get_tensor_shape("velocity"))
            out_buf = torch.empty(out_shape, dtype=io_dtype, device=dev)
            ctx.set_tensor_address("velocity", out_buf.data_ptr())

            self._trt_buf_cache[key] = {"bufs": bufs, "output": out_buf}
            logger.info(
                "Allocated TRT buffers for shapes: hs=%s enc=%s",
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

        ctx = self._trt_ctx
        for name, buf in bufs.items():
            ctx.set_tensor_address(name, buf.data_ptr())
        ctx.set_tensor_address("velocity", entry["output"].data_ptr())

        ctx.execute_async_v3(self._trt_stream.ptr)
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

    def _build_timestep_schedule(
        self, config: DiffusionConfig, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Build the timestep schedule, respecting denoise truncation.

        When denoise < 1.0, computes int(steps/denoise) full steps with
        shift applied, then takes the last (steps+1) entries. This matches
        ComfyUI's BasicScheduler.set_steps() behavior.
        """
        if config.timesteps is not None:
            return torch.tensor(config.timesteps, device=device, dtype=dtype)

        steps = config.infer_steps
        denoise = config.denoise

        if denoise <= 0.0:
            # No denoising: single-entry schedule (t=0 -> t=0)
            return torch.zeros(2, device=device, dtype=dtype)
        elif denoise >= 1.0:
            full_steps = steps
        else:
            # Compute extended schedule, then truncate
            full_steps = int(steps / denoise)

        t_schedule = torch.linspace(
            1.0, 0.0, full_steps + 1, device=device, dtype=dtype
        )
        if config.shift != 1.0:
            t_schedule = (
                config.shift * t_schedule
                / (1 + (config.shift - 1) * t_schedule)
            )

        if denoise < 1.0 and denoise > 0.0:
            # Take the last (steps+1) entries
            t_schedule = t_schedule[-(steps + 1):]

        return t_schedule

