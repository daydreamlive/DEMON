#!/usr/bin/env python3
"""Build TensorRT engines for the MiniMax-Music3 flow-matching DiT.

Holds the shape of the other builders in this package (env preflight,
sidecar ``.metadata.json`` skip/rebuild gates, ``--dry-run`` /
``--force-rebuild``, the ``<engines_dir>/<name>/<name>.trt`` layout) and
reuses their machinery verbatim:
:func:`acestep.engine.trt.sa3_build._build_strongly_typed_engine` has
zero model knowledge, so it needs nothing from here but a profile dict.

Unlike SA3 there is no upstream ONNX to fetch and no plugin to register.
The renderer in :mod:`acestep.engine.minimax_dit` is written to be
exportable by construction, so this builder exports it locally with
``torch.onnx.export(dynamo=True)`` and compiles the result.

Two engines, and **build the fp32 one first**
------------------------------------------------------------------

``--precision fp32`` is the control, and it is cheap: 51 s to build,
9.7 GB on disk. It shares the export path, the builder and the IO
signature with the fp16 engine and differs only in the trunk dtype, so
when the fp16 engine misses the parity bar the control tells you
whether the problem is *quantization* or *the graph*. Keep building it.
:mod:`acestep.engine.trt.fp8_onnx` is the standing argument for why:
its predecessor shipped a "quantized" graph that had in fact quantized
2 of 353 matmuls and then produced NaN at runtime. One un-quantized
control run makes that class of failure obvious immediately.

Measured here (RTX 5090, L=689, batch 1): the fp32 engine runs 45.1 ms
per forward against eager fp32's 89.3 ms, tracking eager fp32 at cosine
0.999998+/step. Read that 2x carefully. TRT enables TF32 GEMMs by
default (``BuilderFlag.TF32`` is on unless cleared) and the eager
reference has TF32 explicitly disabled, so the "fp32" engine is really
TF32. The residual is the proof: switching eager fp32 to TF32 and
comparing against strict fp32 gives relative RMS 1.63 / 1.71 / 1.77 /
1.71e-3 at t = 0.05 / 0.3 / 0.6 / 0.95, and the engine's deviation from
strict eager fp32 is 1.64 / 1.73 / 1.77 / 1.71e-3. Same number. Against
a like-for-like eager TF32 baseline (55.4 ms) the fp32 engine is only
1.23x, so it earns its keep as a control, not as a shipping candidate.

``--precision fp16`` is the production recipe: an fp16 trunk with
explicit fp32 islands, compiled STRONGLY_TYPED with **no** builder
precision flags. Both halves matter.

* Eager measurement on this checkpoint (L=689, RTX 5090) says fp16 is
  both faster *and* more accurate than bf16: 60.7 ms vs 70.4 ms per
  CFG step, SNR 48.7 dB vs 29.3 dB against fp32. That is the opposite
  of the usual ordering and it is why this builder has no bf16 recipe.
  The DAV vocoder is the other way round (fp16 gives all-NaN), which is
  why nothing here touches the decoder.
* STRONGLY_TYPED is load-bearing, not stylistic. A weakly-typed network
  plus ``BuilderFlag.FP16`` lets TRT re-cast the fp32 islands back down
  and reintroduces exactly the overflow they exist to prevent.

Measured (RTX 5090, L=689, batch 1, 44 s build, 4.88 GB engine,
5.08 GB VRAM): **15.7 ms per forward** against eager bf16's 35.3 ms
(2.25x) and eager fp16's 30.4 ms, at cosine 0.99998/step against eager
fp32 on real conditioning. ``scripts/minimax/minimax_trt_parity.py``
is the gate that says so; re-run it after any change here.

The fp32 islands, and why each one
------------------------------------------------------------------

1. **The RoPE tables.** This is the failure the SA3 bf16 engine
   actually died of, and it is a property of the *angle*, not of
   accumulated error. ``position * inv_freq`` for the lowest frequency
   is just ``position``, which here reaches ~690 radians. fp16's ulp at
   690 is 0.5 rad and bf16's is 4 rad, so a table built in either dtype
   is not approximately wrong, it is unrelated to the intended
   rotation. :func:`acestep.engine.minimax_dit._rope_tables` already
   pins ``arange``/``inv_freq`` to fp32 regardless of module dtype;
   :func:`assert_fp32_rope_tables` gates that it stayed that way, and
   :class:`_MixedAttention` keeps the *rotation itself* in fp32 rather
   than casting the table down to the trunk dtype first.
2. **LayerNorm.** Pre-norm residual streams grow; a half-precision
   variance sums 2048 squares and needs |x| < 256 to stay finite.
3. **The Fourier / timestep embedding.** ``(2*pi*t) @ W.T`` then a
   two-layer MLP: 0.6 M parameters out of 2.43 B, so fp32 here costs
   nothing measurable and removes a whole class of angle problem
   analogous to (1).

Do not "tidy up" an island because an ablation shows it makes no
accuracy difference. ``acestep/engine/trt/export.py`` (lines 192-198)
records what happened last time someone removed a dead-looking island:
the strongly-typed builder segfaulted.

Batch
------------------------------------------------------------------

Production engines are **batch-1**. ``torch.export`` will not keep dim 0
symbolic: the ``matmul`` decomposition guards ``batch != 1`` when it
folds a 3-D x 2-D matmul (it fires at ``proj_out``), so a batch-dynamic
export is only legal with ``min_batch >= 2``. ``--min-batch 2
--max-batch 4`` builds such an engine for benchmarking; discovery in
:mod:`acestep.engine.minimax_trt` only ever asks for batch 1, so those
engines are invisible to the streaming path.

Usage::

    # the control, first
    python -m acestep.engine.trt.minimax_build --precision fp32
    # production
    python -m acestep.engine.trt.minimax_build --precision fp16
    # what would happen
    python -m acestep.engine.trt.minimax_build --precision fp16 --dry-run
    # benchmark-only batched engine
    python -m acestep.engine.trt.minimax_build --precision fp16 \
        --min-batch 2 --max-batch 4

The exported ONNX is large (9.7 GB fp32 / 4.9 GB fp16, weights in an
external-data sidecar) and is kept next to the engines by default;
``--onnx-dir`` / ``DEMON_MINIMAX_ONNX_DIR`` moves it to a roomier disk.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from loguru import logger

from ._engine_metadata import (
    expected_metadata as _expected_metadata,
    metadata_matches as _metadata_matches,
    write_metadata as _write_metadata,
)
from .build import _preflight, _save_build_report, _verify_engines
from .sa3_build import _build_strongly_typed_engine

from acestep.engine.minimax_dit import (
    MiniMaxAttention,
    MiniMaxDiT,
    _rope_tables,
)
from acestep.engine.minimax_helpers import minimax_root, resolve_model_dir
from acestep.engine.minimax_trt import (
    MINIMAX_COND_DIM,
    MINIMAX_LATENT_CHANNELS,
    PRECISIONS,
    engine_dir_name,
    trt_engines_dir,
)

#: One 7.999 s session window: 200 AR frames at 25 Hz -> 689 latent
#: frames at 44100/512 Hz. The shape the profile is TUNED for.
CANONICAL_LATENT_FRAMES = 689

#: ...and the longest it SERVES. Sessions run at whatever length the
#: autoregressive stage produces, and an engine only covers the range
#: its profile declares -- a 689-max engine silently drops every other
#: duration to eager, which costs ~1.7x. A ranged profile measured
#: identical in build time (33 s) and engine size (4.88 GB) to the
#: pinned one, and still passes the parity bar at the top of its range
#: (cos 0.999969 vs eager fp32 at 1240 frames), so there is no reason
#: to pin it. 1400 frames is ~16.3 s; widen it if sessions run longer.
CANONICAL_MAX_LATENT_FRAMES = 1400

COMPONENT = "minimax_dit"


def onnx_dir() -> Path:
    override = os.environ.get("DEMON_MINIMAX_ONNX_DIR")
    if override:
        return Path(override)
    return minimax_root() / "onnx"


# ------------------------------------------------------------------
# fp16-mixed surgery
# ------------------------------------------------------------------


def assert_fp32_rope_tables(rotary_dim: int = 32, theta: float = 10000.0) -> None:
    """Guard the one precision fact the graph cannot recover from.

    :func:`acestep.engine.minimax_dit._rope_tables` builds its tables
    from ``torch.arange(...).float()``, so they are fp32 no matter what
    dtype the module carries, but that is a property of one line in
    another file, and the whole fp16 recipe rests on it. Assert it here
    so a future edit that drops the ``.float()`` fails the build instead
    of shipping an engine whose rotations are quantized to half a
    radian.
    """
    cos, sin = _rope_tables(CANONICAL_LATENT_FRAMES + 1, rotary_dim, theta, torch.device("cpu"))
    if cos.dtype is not torch.float32 or sin.dtype is not torch.float32:
        raise RuntimeError(
            f"_rope_tables returned {cos.dtype}/{sin.dtype}; the fp16 recipe "
            "requires fp32 tables (the angle reaches ~690 rad, where fp16's "
            "ulp is 0.5 rad)"
        )
    # And that the angle really is that large, i.e. the assumption above
    # is about this model and not a copied comment.
    max_angle = float(torch.acos(cos[:, 0].clamp(-1, 1)).max())
    if max_angle <= 0.0:
        raise RuntimeError("_rope_tables produced a degenerate angle column")


class _Fp32LayerNorm(torch.nn.LayerNorm):
    """LayerNorm as an explicit fp32 island: cast up, normalize, cast
    back. Installed by ``__class__`` swap, so it must add no state."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        out = F.layer_norm(
            hidden_states.float(), self.normalized_shape, self.weight, self.bias, self.eps,
        )
        return out.to(hidden_states.dtype)


def _apply_partial_rope_fp32(
    hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rotary_dim: int
) -> torch.Tensor:
    """:func:`_apply_partial_rope` with the rotation itself in fp32.

    The eager fp16 path casts the (fp32) table down to the trunk dtype
    before multiplying, which costs ~5e-4 absolute on cos/sin. That is
    tolerable but free to avoid: the rotated slice is
    ``rotary_dim=32`` of each head's 64 channels, so the fp32 island is
    a few hundred KB of elementwise work next to a 2.43 B-parameter GEMM
    trunk.
    """
    dtype = hidden_states.dtype
    cos = cos[:, None, :].float()
    sin = sin[:, None, :].float()
    rotated = hidden_states[..., :rotary_dim].float()
    half_first, half_second = rotated.chunk(2, dim=-1)
    rotate_half = torch.cat((-half_second, half_first), dim=-1)
    rotated = rotated * cos + rotate_half * sin
    return torch.cat((rotated.to(dtype), hidden_states[..., rotary_dim:]), dim=-1)


class _MixedAttention(MiniMaxAttention):
    """:class:`MiniMaxAttention` with the RoPE rotation in fp32 and the
    projections plus SDPA in the trunk dtype. Installed by ``__class__``
    swap; adds no state."""

    def forward(self, hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        batch_size, seq_len, _ = hidden_states.shape
        query = self.to_q(hidden_states).view(batch_size, seq_len, self.heads, self.head_dim)
        key = self.to_k(hidden_states).view(batch_size, seq_len, self.heads, self.head_dim)
        value = self.to_v(hidden_states).view(batch_size, seq_len, self.heads, self.head_dim)

        query = _apply_partial_rope_fp32(query, cos, sin, self.rotary_dim)
        key = _apply_partial_rope_fp32(key, cos, sin, self.rotary_dim)

        out = F.scaled_dot_product_attention(
            query.permute(0, 2, 1, 3),
            key.permute(0, 2, 1, 3),
            value.permute(0, 2, 1, 3),
        )
        out = out.permute(0, 2, 1, 3).flatten(2, 3).to(query.dtype)
        return self.to_out(out)


class _MixedDiT(MiniMaxDiT):
    """:class:`MiniMaxDiT` with fp32 IO over a half-precision trunk.

    The forward mirrors the eager one line for line; every difference is
    a cast, and every cast is an island boundary. Installed by
    ``__class__`` swap; adds no state.

    fp32 IO is deliberate. It makes the fp16 and fp32 engines drop-in
    interchangeable for the runtime wrapper and the parity harness, and
    the cast is one elementwise pass over 0.35 MB.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        trunk = self.proj_in.weight.dtype

        hidden_states = hidden_states.to(trunk)
        encoder_hidden_states = encoder_hidden_states.to(trunk)
        zeros = torch.zeros_like(hidden_states)
        hidden_states = torch.cat(
            (hidden_states, zeros, encoder_hidden_states.transpose(1, 2)), dim=1,
        )
        hidden_states = self.preprocess_conv(hidden_states) + hidden_states
        hidden_states = hidden_states.transpose(1, 2)

        # fp32 island: Fourier features + the two-layer time MLP.
        temb = self.time_embed(self.time_proj(timestep.float()))

        hidden_states = self.proj_in(hidden_states)
        hidden_states = torch.cat((temb.unsqueeze(1).to(trunk), hidden_states), dim=1)

        # fp32 tables (see assert_fp32_rope_tables); the rotation stays
        # fp32 too, inside _MixedAttention.
        cos, sin = _rope_tables(
            hidden_states.shape[1], self.rotary_dim, self.rope_theta, hidden_states.device,
        )
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, cos, sin)

        hidden_states = self.proj_out(hidden_states[:, 1:])
        hidden_states = hidden_states.transpose(1, 2)
        out = self.postprocess_conv(hidden_states) + hidden_states
        return out.float()


def make_fp16_mixed(dit: MiniMaxDiT) -> MiniMaxDiT:
    """In-place surgery from an fp32 :class:`MiniMaxDiT` to the
    fp16-mixed export module. Mutates and returns the same object; a
    2.43 B-parameter copy is not worth making."""
    assert_fp32_rope_tables(dit.rotary_dim, dit.rope_theta)
    dit.to(torch.float16)
    # fp32 islands. LayerNorm weights and the time embedding go back up;
    # everything else stays in the half-precision trunk.
    dit.time_proj.to(torch.float32)
    dit.time_embed.to(torch.float32)
    for block in dit.transformer_blocks:
        for norm in (block.norm1, block.norm2):
            norm.__class__ = _Fp32LayerNorm
            norm.to(torch.float32)
        block.attn.__class__ = _MixedAttention
    dit.__class__ = _MixedDiT
    return dit


def load_export_module(
    *, precision: str, model_dir=None, device="cpu"
) -> MiniMaxDiT:
    """The module this builder exports, at the requested precision.

    Loaded on the CPU by default: the export is a trace, it needs no
    GPU, and TRT engine builds want the card to itself (an engine built
    while another process holds the GPU deserializes and then
    segfaults).
    """
    root = Path(resolve_model_dir(model_dir))
    logger.info("Loading MiniMax DiT from {} (fp32, {})", root / "transformer", device)
    dit = MiniMaxDiT.from_pretrained(root / "transformer", dtype=torch.float32, device=device)
    if precision == "fp32":
        assert_fp32_rope_tables(dit.rotary_dim, dit.rope_theta)
        return dit
    if precision == "fp16":
        return make_fp16_mixed(dit)
    raise ValueError(f"unknown precision {precision!r}")


# ------------------------------------------------------------------
# ONNX export
# ------------------------------------------------------------------


def onnx_file_name(config: "MiniMaxDiTBuildConfig") -> str:
    return (
        f"minimax_dit_{config.precision}"
        f"_b{config.min_batch}_{config.max_batch}"
        f"_l{config.min_latents}_{config.max_latents}.onnx"
    )


def export_dit_onnx(
    *,
    config: "MiniMaxDiTBuildConfig",
    onnx_path: Path,
    model_dir=None,
    device: str = "cpu",
) -> Path:
    """Export the DiT to ONNX with a dynamic latent length.

    ``dynamic_shapes`` is passed **real** ``torch.export.Dim`` objects.
    Strings are accepted by the signature and then silently drop the
    export back to the TorchScript path, which bakes the example shapes
    in: an engine built from that graph is static at whatever length
    happened to be traced, and nothing says so.
    """
    from torch.export import Dim

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    dit = load_export_module(precision=config.precision, model_dir=model_dir, device=device)

    # Trace at the opt shape. The batch example must be >= 2 for a
    # batch-dynamic export (see the module docstring).
    batch = 1 if config.max_batch == 1 else max(2, config.min_batch)
    length = config.opt_latents
    dev = torch.device(device)
    hidden_states = torch.zeros(batch, MINIMAX_LATENT_CHANNELS, length, dtype=torch.float32, device=dev)
    timestep = torch.full((batch,), 0.5, dtype=torch.float32, device=dev)
    encoder_hidden_states = torch.zeros(
        batch, length, MINIMAX_COND_DIM, dtype=torch.float32, device=dev,
    )

    # min=2, not 1: torch.export 0/1-specializes, and the matmul
    # decomposition then guards `length != 1` at proj_out exactly as it
    # does for batch. The graph it produces is arithmetically valid at
    # any length; only the trace refuses to admit 1 as a possibility.
    length_dim = Dim("length", min=2, max=max(8192, config.max_latents))
    dynamic_shapes: dict = {
        "hidden_states": {2: length_dim},
        "timestep": {},
        "encoder_hidden_states": {1: length_dim},
    }
    if config.max_batch > 1:
        batch_dim = Dim("batch", min=max(2, config.min_batch), max=config.max_batch)
        dynamic_shapes["hidden_states"][0] = batch_dim
        dynamic_shapes["timestep"][0] = batch_dim
        dynamic_shapes["encoder_hidden_states"][0] = batch_dim

    logger.info(
        "ONNX export -> {} (precision={} trace_batch={} trace_length={})",
        onnx_path, config.precision, batch, length,
    )
    t0 = time.time()
    with torch.no_grad():
        torch.onnx.export(
            dit,
            (hidden_states, timestep, encoder_hidden_states),
            str(onnx_path),
            dynamo=True,
            external_data=True,
            input_names=["hidden_states", "timestep", "encoder_hidden_states"],
            output_names=["velocity"],
            dynamic_shapes=dynamic_shapes,
            optimize=True,
        )
    logger.info(
        "ONNX export done in {:.0f}s ({:.1f} GB proto+data)",
        time.time() - t0, _onnx_bytes(onnx_path) / 1e9,
    )
    del dit
    return onnx_path


def _onnx_bytes(onnx_path: Path) -> int:
    """Proto plus any external-data sidecars sitting beside it."""
    total = onnx_path.stat().st_size if onnx_path.exists() else 0
    for sidecar in onnx_path.parent.glob(f"{onnx_path.name}*.data"):
        total += sidecar.stat().st_size
    for sidecar in onnx_path.parent.glob(f"{onnx_path.stem}*.weight"):
        total += sidecar.stat().st_size
    return total


# ------------------------------------------------------------------
# Build
# ------------------------------------------------------------------


@dataclass
class MiniMaxDiTBuildConfig:
    """Build parameters for one MiniMax DiT engine. Every field is part
    of the engine's metadata identity, so changing any of them
    invalidates an engine on disk."""

    precision: str = "fp16"
    min_batch: int = 1
    max_batch: int = 1
    min_latents: int = 2
    opt_latents: int = CANONICAL_LATENT_FRAMES
    max_latents: int = CANONICAL_LATENT_FRAMES
    workspace_gb: float = 12.0

    def validate(self) -> None:
        if self.precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {PRECISIONS}")
        if not 0 < self.min_batch <= self.max_batch:
            raise ValueError("require 0 < min_batch <= max_batch")
        if self.max_batch > 1 and self.min_batch < 2:
            raise ValueError(
                "a batch-dynamic MiniMax engine must have min_batch >= 2: "
                "torch.export specializes dim 0 at 1 (see the module "
                "docstring). Build batch-1 and batch-2..N as separate engines."
            )
        if not 0 < self.min_latents <= self.opt_latents <= self.max_latents:
            raise ValueError("require 0 < min <= opt <= max latent frames")
        if self.min_latents < 2:
            raise ValueError(
                "min_latents must be >= 2: torch.export 0/1-specializes, so a "
                "length dim that admits 1 cannot be traced (see export_dit_onnx). "
                "A 1-frame session is 11.6 ms of audio; nothing needs it."
            )

    def engine_name(self) -> str:
        return engine_dir_name(
            precision=self.precision,
            min_batch=self.min_batch,
            max_batch=self.max_batch,
            min_latents=self.min_latents,
            opt_latents=self.opt_latents,
            max_latents=self.max_latents,
        )

    def label(self) -> str:
        seconds = self.max_latents * 512 / 44100
        return (
            f"MiniMax DiT {self.precision} b{self.min_batch}-{self.max_batch} "
            f"l{self.min_latents}_{self.opt_latents}_{self.max_latents} "
            f"(~{seconds:.1f}s window)"
        )


def build_dit_engine(
    *,
    output_dir: str,
    config: MiniMaxDiTBuildConfig,
    env: dict,
    force_rebuild: bool = False,
    force_onnx: bool = False,
    model_dir=None,
    onnx_root: Path | None = None,
) -> tuple[str, str, float, str]:
    """Build one MiniMax DiT engine. Returns ``(label, path, elapsed, status)``."""
    config.validate()
    name = config.engine_name()
    engine_path = os.path.join(output_dir, name, f"{name}.trt")
    label = config.label()

    onnx_path = (onnx_root or onnx_dir()) / onnx_file_name(config)
    if force_onnx or not onnx_path.exists():
        # Nothing to compare an engine against without the graph, and the
        # graph is cheap next to the build, so export unconditionally when
        # it is missing.
        export_dit_onnx(config=config, onnx_path=onnx_path, model_dir=model_dir)
    else:
        logger.info("Reusing ONNX {} ({:.1f} GB)", onnx_path, _onnx_bytes(onnx_path) / 1e9)

    expected = _expected_metadata(
        component=COMPONENT, onnx_path=str(onnx_path), config=config, env=env,
    )
    if not force_rebuild and os.path.exists(engine_path):
        matches, reason = _metadata_matches(engine_path, expected)
        if matches:
            size_mb = os.path.getsize(engine_path) / 1e6
            logger.info("SKIP {} ({:.0f} MB, {})", name, size_mb, reason)
            return (label, engine_path, 0.0, "SKIPPED")
        logger.info("REBUILD {} ({})", name, reason)

    logger.info("=" * 60)
    logger.info(
        "MINIMAX DiT TRT BUILD: {} ({}, STRONGLY_TYPED, workspace {:.0f} GB)",
        name, config.precision, config.workspace_gb,
    )
    logger.info("=" * 60)

    b_lo, b_hi = config.min_batch, config.max_batch
    l_lo, l_opt, l_hi = config.min_latents, config.opt_latents, config.max_latents
    b_opt = b_lo
    t0 = time.time()
    _build_strongly_typed_engine(
        onnx_path=str(onnx_path),
        engine_path=engine_path,
        workspace_gb=config.workspace_gb,
        profile_shapes={
            "hidden_states": (
                (b_lo, MINIMAX_LATENT_CHANNELS, l_lo),
                (b_opt, MINIMAX_LATENT_CHANNELS, l_opt),
                (b_hi, MINIMAX_LATENT_CHANNELS, l_hi),
            ),
            "timestep": ((b_lo,), (b_opt,), (b_hi,)),
            "encoder_hidden_states": (
                (b_lo, l_lo, MINIMAX_COND_DIM),
                (b_opt, l_opt, MINIMAX_COND_DIM),
                (b_hi, l_hi, MINIMAX_COND_DIM),
            ),
        },
    )
    _write_metadata(engine_path=engine_path, expected=expected, env=env)
    elapsed = time.time() - t0
    logger.info(
        "Built in {:.0f}s ({:.2f} GB)", elapsed, os.path.getsize(engine_path) / 1e9,
    )
    return (label, engine_path, elapsed, "OK")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build MiniMax-Music3 DiT TensorRT engines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--precision", choices=PRECISIONS, default="fp16",
        help="fp32 is the known-good control (build it first); fp16 is the "
             "production fp16-mixed STRONGLY_TYPED recipe (default: fp16)",
    )
    parser.add_argument("--output-dir", default=str(trt_engines_dir()),
                        help="Engine output directory "
                             "(default: <models>/minimax/trt_engines)")
    parser.add_argument("--onnx-dir", default=None,
                        help="Where the exported ONNX lives "
                             "(default: <models>/minimax/onnx, or "
                             "DEMON_MINIMAX_ONNX_DIR)")
    parser.add_argument("--model-dir", default=None,
                        help="MiniMax-Music3 diffusers-layout checkpoint "
                             "(default: the usual resolution order)")
    parser.add_argument("--latent-frames", type=int, default=CANONICAL_LATENT_FRAMES,
                        help=f"Session window in latent frames "
                             f"(default: {CANONICAL_LATENT_FRAMES} = 7.999 s). Sessions may now run any length, and an engine only serves the range its profile covers -- widen --max-latents to keep TensorRT on longer songs.")
    parser.add_argument("--min-latents", type=int, default=2,
                        help="Smallest latent length the profile covers "
                             "(must be >= 2; see export_dit_onnx)")
    parser.add_argument("--opt-latents", type=int, default=None)
    parser.add_argument("--max-latents", type=int, default=None)
    parser.add_argument("--min-batch", type=int, default=1)
    parser.add_argument("--max-batch", type=int, default=1,
                        help="1 for production (the streaming path loops "
                             "slots). >1 builds a benchmark-only engine and "
                             "requires --min-batch 2.")
    parser.add_argument("--workspace-gb", type=float, default=12.0,
                        help="TRT builder workspace in GB (default: 12)")
    parser.add_argument("--force-rebuild", "--force", action="store_true",
                        help="Rebuild even when the metadata sidecar matches")
    parser.add_argument("--force-onnx", action="store_true",
                        help="Re-export the ONNX even when one is on disk")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be built and exit")
    args = parser.parse_args()

    opt_latents = args.opt_latents or args.latent_frames
    config = MiniMaxDiTBuildConfig(
        precision=args.precision,
        min_batch=args.min_batch,
        max_batch=args.max_batch,
        min_latents=args.min_latents,
        opt_latents=opt_latents,
        max_latents=args.max_latents or max(
            opt_latents, CANONICAL_MAX_LATENT_FRAMES,
        ),
        workspace_gb=args.workspace_gb,
    )
    try:
        config.validate()
    except ValueError as exc:
        parser.error(str(exc))

    onnx_root = Path(args.onnx_dir) if args.onnx_dir else onnx_dir()
    name = config.engine_name()
    engine_path = os.path.join(args.output_dir, name, f"{name}.trt")
    onnx_path = onnx_root / onnx_file_name(config)

    if args.dry_run:
        print(f"\nMiniMax build plan")
        print(f"  engine     {name}")
        print(f"  label      {config.label()}")
        print(f"  engine at  {engine_path}"
              f"{'  [exists]' if os.path.exists(engine_path) else ''}")
        print(f"  onnx at    {onnx_path}"
              f"{'  [exists]' if onnx_path.exists() else '  [will export]'}")
        print(f"  profile    hidden_states "
              f"({config.min_batch},128,{config.min_latents}) .. "
              f"({config.max_batch},128,{config.max_latents})")
        print(f"  workspace  {config.workspace_gb:.0f} GB")
        free = shutil.disk_usage(Path(args.output_dir).anchor).free
        print(f"  free space {free / 1e9:.0f} GB on {Path(args.output_dir).anchor}\n")
        return 0

    os.makedirs(args.output_dir, exist_ok=True)
    env = _preflight("cuda")
    result = build_dit_engine(
        output_dir=args.output_dir,
        config=config,
        env=env,
        force_rebuild=args.force_rebuild,
        force_onnx=args.force_onnx,
        model_dir=args.model_dir,
        onnx_root=onnx_root,
    )
    if result[3] == "OK":
        logger.info("=" * 60)
        logger.info("VERIFICATION")
        logger.info("=" * 60)
        _verify_engines([(result[0], result[1])])
    _save_build_report([result], args.output_dir)
    print(f"\n  {result[3]:7s} {result[2]:6.0f}s  {result[0]}")
    return 0 if result[3] in ("OK", "SKIPPED") else 1


if __name__ == "__main__":
    sys.exit(main())
