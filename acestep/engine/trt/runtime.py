"""TensorRT runtime for the ACE-Step decoder.

Provides TRTDecoder, a drop-in replacement for model.decoder that runs
inference through a pre-built TensorRT engine.  Designed to slot directly
into DiffusionEngine._trt_decoder_step().

Buffer management:
  - Pre-allocated fp32 buffers per shape (zero per-call allocations)
  - Inputs copied via .copy_() into pinned buffers each step
  - Output is a view of the internal buffer (no clone)
  - Dedicated non-default CUDA stream with wait_stream sync
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from loguru import logger
import torch

_TRT_TO_TORCH = None

def _get_trt_to_torch_map():
    global _TRT_TO_TORCH
    if _TRT_TO_TORCH is None:
        import tensorrt as trt
        _TRT_TO_TORCH = {
            trt.float32: torch.float32,
            trt.float16: torch.float16,
            trt.int32: torch.int32,
            trt.int8: torch.int8,
            trt.bool: torch.bool,
        }
        if hasattr(trt, "bfloat16"):
            _TRT_TO_TORCH[trt.bfloat16] = torch.bfloat16
    return _TRT_TO_TORCH


class TRTDecoder:
    """TensorRT decoder with pre-allocated buffers.

    Usage::

        trt_dec = TRTDecoder("decoder_fp16.engine")
        velocity = trt_dec(
            hidden_states=xt,          # [B, T, 64]
            timestep=t_tensor,         # [B]
            encoder_hidden_states=enc, # [B, L_enc, 2048]
            context_latents=ctx,       # [B, T, 128]
        )
    """

    INPUT_NAMES = (
        "hidden_states", "timestep", "encoder_hidden_states",
        "context_latents", "steering",
    )
    OUTPUT_NAME = "velocity"

    def __init__(
        self,
        engine_path: Union[str, Path],
        device: Union[str, torch.device] = "cuda",
    ):
        import tensorrt as trt
        from polygraphy.backend.common import bytes_from_path
        from polygraphy.backend.trt import engine_from_bytes

        self._trt = trt
        engine_path = Path(engine_path)
        if not engine_path.exists():
            raise FileNotFoundError(f"TRT engine not found: {engine_path}")

        self.device = torch.device(device)

        logger.info("Loading TRT engine from {} ...", engine_path)
        self.engine = engine_from_bytes(bytes_from_path(str(engine_path)))

        self.context = self.engine.create_execution_context()

        # Shared polygraphy stream for all TRT engines.
        from acestep.nodes.vae_nodes import _get_trt_stream
        self._stream = _get_trt_stream()

        # I/O dtypes from engine: each tensor has its own declared dtype.
        # The engine's "hidden_states" / "encoder_hidden_states" / "context_latents"
        # may be fp16 or bf16 (mixed precision builds) or fp32 (legacy). We MUST
        # match the engine's declared dtype when allocating input buffers — TRT
        # reads the buffer as raw bytes interpreted with the engine's declared
        # dtype, so a fp32 buffer fed to a fp16 input produces garbage.
        dtype_map = _get_trt_to_torch_map()
        self._input_dtypes = {
            name: dtype_map.get(self.engine.get_tensor_dtype(name), torch.float32)
            for name in self.INPUT_NAMES
        }
        out_trt_dtype = self.engine.get_tensor_dtype(self.OUTPUT_NAME)
        self._output_dtype = dtype_map.get(out_trt_dtype, torch.float32)

        # Static L, D; B is dynamic per call.
        steer_shape = tuple(self.engine.get_tensor_shape("steering"))
        if len(steer_shape) != 3:
            raise RuntimeError(f"Unexpected steering input rank: {steer_shape}")
        self._steer_L = int(steer_shape[1])
        self._steer_D = int(steer_shape[2])

        # Per-shape buffer cache: shape_key -> {bufs, output}
        self._buf_cache: dict[tuple, dict] = {}

        logger.info("TRT decoder ready (output_dtype={})", self._output_dtype)

    def _get_bufs(self, hs_shape, ts_shape, enc_shape, cl_shape):
        """Get or allocate buffers for these shapes, using each tensor's
        engine-declared dtype.

        WARNING: this used to allocate ALL inputs as fp32 unconditionally,
        which silently produces garbage output for fp16/bf16 engines (TRT
        reads the fp32 bytes through the declared narrower dtype). Always
        use ``self._input_dtypes[name]``.
        """
        key = (hs_shape, ts_shape, enc_shape, cl_shape)
        if key in self._buf_cache:
            return self._buf_cache[key]

        ctx = self.context
        dev = self.device

        in_dt = self._input_dtypes
        B = hs_shape[0]
        bufs = {
            "hidden_states": torch.empty(hs_shape, dtype=in_dt["hidden_states"], device=dev),
            "timestep": torch.empty(ts_shape, dtype=in_dt["timestep"], device=dev),
            "encoder_hidden_states": torch.empty(enc_shape, dtype=in_dt["encoder_hidden_states"], device=dev),
            "context_latents": torch.empty(cl_shape, dtype=in_dt["context_latents"], device=dev),
            "steering": torch.zeros(
                B, self._steer_L, self._steer_D,
                dtype=in_dt["steering"], device=dev,
            ),
        }

        for name, buf in bufs.items():
            if not ctx.set_input_shape(name, tuple(buf.shape)):
                raise RuntimeError(f"TRT decoder rejected input shape for {name}: {tuple(buf.shape)}")
            if not ctx.set_tensor_address(name, buf.data_ptr()):
                raise RuntimeError(f"TRT decoder rejected input address for {name}")

        missing = ctx.infer_shapes()
        if missing:
            raise RuntimeError(f"TRT decoder shapes are insufficiently specified: {missing}")

        out_shape = tuple(ctx.get_tensor_shape(self.OUTPUT_NAME))
        out_buf = torch.empty(out_shape, dtype=self._output_dtype, device=dev)
        if not ctx.set_tensor_address(self.OUTPUT_NAME, out_buf.data_ptr()):
            raise RuntimeError(f"TRT decoder rejected output address for {self.OUTPUT_NAME}")

        entry = {"bufs": bufs, "output": out_buf}
        self._buf_cache[key] = entry
        logger.info("Allocated TRT buffers for shapes: hs={} enc={}", list(hs_shape), list(enc_shape))
        return entry

    def __call__(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        context_latents: torch.Tensor,
        steering: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run one decoder step through TensorRT.

        Accepts any dtype; inputs are copied into pre-allocated fp32 buffers.
        Returns a view of the internal output buffer (caller must not hold
        references across calls with different shapes).

        ``steering`` is optional ``[B, num_layers, hidden_size]``; when
        omitted the per-layer adds are no-ops.
        """
        orig_T = hidden_states.shape[1]
        pad = orig_T % 2 == 1
        eff_T = orig_T + 1 if pad else orig_T

        entry = self._get_bufs(
            (hidden_states.shape[0], eff_T, 64),
            tuple(timestep.shape),
            tuple(encoder_hidden_states.shape),
            (context_latents.shape[0], eff_T, 128),
        )
        bufs = entry["bufs"]

        # Copy into pre-allocated fp32 buffers (handles any dtype)
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
        if steering is not None:
            bufs["steering"].copy_(steering)
        else:
            bufs["steering"].zero_()

        # Bind addresses
        ctx = self.context
        for name, buf in bufs.items():
            if not ctx.set_tensor_address(name, buf.data_ptr()):
                raise RuntimeError(f"TRT decoder rejected input address for {name}")
        if not ctx.set_tensor_address(self.OUTPUT_NAME, entry["output"].data_ptr()):
            raise RuntimeError(f"TRT decoder rejected output address for {self.OUTPUT_NAME}")

        # Execute on shared polygraphy stream.
        stream = self._stream
        if not ctx.execute_async_v3(stream.ptr):
            raise RuntimeError("TRT decoder execution failed")
        stream.synchronize()

        output = entry["output"]
        return output[:, :orig_T, :] if pad else output

    def benchmark(
        self,
        seq_len: int = 750,
        enc_len: int = 200,
        batch_size: int = 1,
        warmup: int = 5,
        iterations: int = 20,
    ) -> dict:
        import time
        B, T, L = batch_size, seq_len, enc_len
        hs = torch.randn(B, T, 64, device=self.device, dtype=torch.float32)
        ts = torch.full((B,), 0.5, device=self.device, dtype=torch.float32)
        enc = torch.randn(B, L, 2048, device=self.device, dtype=torch.float32)
        ctx = torch.randn(B, T, 128, device=self.device, dtype=torch.float32)

        for _ in range(warmup):
            self(hs, ts, enc, ctx)

        torch.cuda.synchronize()
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            self(hs, ts, enc, ctx)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)

        results = {
            "mean_ms": sum(times) / len(times),
            "min_ms": min(times),
            "max_ms": max(times),
            "steps_per_sec": 1000.0 / (sum(times) / len(times)),
            "seq_len": T, "enc_len": L, "batch_size": B,
        }
        logger.info("TRT benchmark (T={}, L={}, B={}):", T, L, B)
        logger.info(
            "  mean={:.1f}ms  min={:.1f}ms  max={:.1f}ms  ({:.1f} steps/sec)",
            results["mean_ms"], results["min_ms"], results["max_ms"],
            results["steps_per_sec"],
        )
        return results
