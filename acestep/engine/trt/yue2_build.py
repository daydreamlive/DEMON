"""Build the YuE2 TensorRT engines: the flexible acoustic NAR and the
37-frame VAE window decoder (consumed by :mod:`acestep.engine.yue2_trt`).

Ported from the spike that measured them (``scripts/spikes/
yue2_round2.py`` ``AcousticExport``, ``yue2_edit_profile.py``
``export_flexible``, ``yue2_nar_trt_build.py``, ``yue2_vae_trt.py``).
Run on an otherwise idle GPU (a build under contention can produce an
engine that segfaults on load)::

    DEMON_YUE2_ROOT=<weights root> DEMON_YUE2_YUE_SRC=<upstream src> \\
    DEMON_YUE2_EXTRA_PATH=<dir with tiktoken> \\
      python -m acestep.engine.trt.yue2_build nar --out <trt dir>
      python -m acestep.engine.trt.yue2_build vae --out <trt dir>

``nar`` writes ``<out>/flexible_song/{velocity.onnx,velocity.trt,
profile.json,build.json}``; ``vae`` writes ``<out>/vae_fp32_t37.{onnx,trt}``.
Point ``DEMON_YUE2_TRT_DIR`` at ``<out>``. The NAR export needs example
conditioning only for shapes, so it prefills a synthetic song; the
engine takes the real conditioning as inputs at run time.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from acestep.engine.yue2_trt import (
    FLEX_COND_TOKENS,
    FLEX_ENGINE_DIR,
    FLEX_FRAMES,
    FLEX_MAX_BATCH,
    VAE_CTX_FRAMES,
    VAE_ENGINE_FILE,
)

#: Example geometry for the export trace (the spike's prepared song).
EXAMPLE_FRAMES = 1579
EXAMPLE_PREFIX_TOKENS = 679


class AcousticExport(torch.nn.Module):
    """The NAR transformer only. Conditioning (KV, rotary, positions) is
    an input, never a constant, so one engine serves every song."""

    def __init__(self, model):
        super().__init__()
        self.vae2llm, self.llm2vae = model.vae2llm, model.llm2vae
        self.time_embedder, self.norm = model.time_embedder, model.model.norm
        self.layers = torch.nn.ModuleList([
            torch.nn.ModuleDict(dict(
                norm=layer.nar_input_layernorm, attention=layer.nar_self_attn,
                norm_mlp=layer.nar_pre_mlp_layernorm, mlp=layer.nar_mlp,
            ))
            for layer in model.model.layers
        ])
        self.shift = model.config.timestep_shift
        self.groups = model.config.num_attention_heads // model.config.num_key_value_heads

    def forward(self, state, raw_time, keys, values, cos, sin, position):
        batch, frames, _ = state.shape
        count = frames + 2
        shifted = raw_time.sigmoid()
        shifted = self.shift * shifted / (1 + (self.shift - 1) * shifted)
        x = self.vae2llm(F.pad(state, (0, 0, 1, 1)))
        times = shifted[:, None].expand(batch, count).reshape(-1)
        x = x + self.time_embedder(times).reshape(batch, count, -1) + position
        for index, layer in enumerate(self.layers):
            q, k, v = layer["attention"].project_qkv(layer["norm"](x), cos, sin)
            k = torch.cat((keys[index][None].expand(batch, -1, -1, -1), k), dim=1).transpose(1, 2)
            v = torch.cat((values[index][None].expand(batch, -1, -1, -1), v), dim=1).transpose(1, 2)
            # Explicit GQA repetition (the legacy ONNX exporter has no enable_gqa).
            k = k.repeat_interleave(self.groups, dim=1)
            v = v.repeat_interleave(self.groups, dim=1)
            h = F.scaled_dot_product_attention(q.transpose(1, 2), k, v)
            x = x + layer["attention"].o_proj(h.transpose(1, 2).reshape(batch, count, -1))
            x = x + layer["mlp"](layer["norm_mlp"](x))
        return self.llm2vae(self.norm(x))[:, 1:-1]


def _example_nar(model):
    """A prefilled CachedNAR for a synthetic song of the example shape."""
    from yue2.nar import CachedNAR, song_chunks

    generator = torch.Generator().manual_seed(0)
    prefix = torch.randint(0, 1000, (EXAMPLE_PREFIX_TOKENS,), generator=generator).tolist()
    codec = torch.randint(0, 1000, (EXAMPLE_FRAMES,), generator=generator).tolist()
    (chunk,) = song_chunks(prefix, codec, seed=0)
    return CachedNAR(model, chunk)


def _flexible_profile(inputs: dict) -> tuple:
    """(optimization shapes, ONNX dynamic axes) for the flexible profile."""
    bounds, dynamic = {}, {}
    for name, value in inputs.items():
        shape = list(value.shape)
        lo, opt, hi = shape.copy(), shape.copy(), shape.copy()
        if name == "state":
            lo[:2], opt[0], hi[:2] = [1, FLEX_FRAMES[0]], FLEX_MAX_BATCH, [FLEX_MAX_BATCH, FLEX_FRAMES[1]]
            dynamic[name] = {0: "batch", 1: "frames"}
        elif name == "raw_time":
            lo[0], opt[0], hi[0] = 1, FLEX_MAX_BATCH, FLEX_MAX_BATCH
            dynamic[name] = {0: "batch"}
        elif name in ("keys", "values"):
            lo[1], hi[1] = FLEX_COND_TOKENS
            dynamic[name] = {1: "conditioning"}
        else:
            lo[1], hi[1] = FLEX_FRAMES[0] + 2, FLEX_FRAMES[1] + 2
            dynamic[name] = {1: "padded_frames"}
        bounds[name] = [lo, opt, hi]
    dynamic["velocity"] = {0: "batch", 1: "frames"}
    return bounds, dynamic


def _build(onnx_path: Path, engine_path: Path, *, strongly_typed: bool, shapes=None) -> float:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED) if strongly_typed else 0
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path.resolve())):
        raise RuntimeError("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.clear_flag(trt.BuilderFlag.TF32)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    config.builder_optimization_level = 2
    config.avg_timing_iterations = 1
    config.max_aux_streams = 0
    if shapes:
        profile = builder.create_optimization_profile()
        for name, (lo, opt, hi) in shapes.items():
            profile.set_shape(name, lo, opt, hi)
        config.add_optimization_profile(profile)
    started = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"TensorRT build failed for {onnx_path}")
    engine_path.write_bytes(bytes(serialized))
    return time.perf_counter() - started


def build_nar(out: Path) -> dict:
    from acestep.engine.yue2_runtime import MODEL_DIR, load_verified, weights_root

    artifact = out / FLEX_ENGINE_DIR
    artifact.mkdir(parents=True, exist_ok=True)
    model, identity = load_verified(weights_root() / MODEL_DIR)
    nar = _example_nar(model)
    keys = torch.stack([k for k, _ in nar.cache])
    values = torch.stack([v for _, v in nar.cache])
    noise = nar.chunk.noise.to(device="cuda", dtype=torch.bfloat16)
    inputs = dict(
        state=noise[None], raw_time=torch.zeros(1, device="cuda", dtype=torch.bfloat16),
        keys=keys, values=values, cos=nar.cos, sin=nar.sin, position=nar.pos_emb,
    )
    module = AcousticExport(model).eval()
    with torch.inference_mode():
        exported = module(*inputs.values())[0].float()
        reference = nar.velocity(noise, 0.0).float()
    parity = float((exported - reference).norm() / reference.norm())
    bounds, dynamic = _flexible_profile(inputs)
    onnx_path = artifact / "velocity.onnx"
    torch.onnx.export(
        module, tuple(inputs.values()), str(onnx_path),
        input_names=list(inputs), output_names=["velocity"], dynamic_axes=dynamic,
        opset_version=18, dynamo=False, do_constant_folding=True,
    )
    meta = dict(
        inputs={k: list(v.shape) for k, v in inputs.items()}, optimization_shapes=bounds,
        precision="BF16; FP32 normalization and rotary inputs", model_identity=identity,
        export_wrapper_relative_l2=parity,
    )
    (artifact / "profile.json").write_text(json.dumps(meta, indent=2))
    nar.close()
    del model
    torch.cuda.empty_cache()
    seconds = _build(onnx_path, artifact / "velocity.trt", strongly_typed=True, shapes=bounds)
    report = dict(build_seconds=seconds, bytes=(artifact / "velocity.trt").stat().st_size, **meta)
    (artifact / "build.json").write_text(json.dumps(report, indent=2))
    return report


def build_vae(out: Path) -> dict:
    from torch.nn.utils import remove_weight_norm

    from acestep.engine.yue2_runtime import VAE_DIR, load_verified, weights_root

    out.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    vae, identity = load_verified(weights_root() / VAE_DIR, vae=True, device="cpu")
    for module in vae.decoder.modules():
        if hasattr(module, "weight_g"):
            remove_weight_norm(module)
    onnx_path = out / VAE_ENGINE_FILE.replace(".trt", ".onnx")
    with torch.inference_mode():
        torch.onnx.export(
            vae.decoder, (torch.zeros(1, 64, VAE_CTX_FRAMES),), str(onnx_path),
            input_names=["latent"], output_names=["audio"],
            opset_version=17, dynamo=False, do_constant_folding=True,
        )
    seconds = _build(onnx_path, out / VAE_ENGINE_FILE, strongly_typed=True)
    return dict(build_seconds=seconds, model_identity=identity,
                bytes=(out / VAE_ENGINE_FILE).stat().st_size)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("target", choices=("nar", "vae"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    from acestep.engine.yue2_runtime import ensure_import_paths, weights_root

    if weights_root() is None:
        parser.error("DEMON_YUE2_ROOT is not set")
    ensure_import_paths()
    report = build_nar(args.out) if args.target == "nar" else build_vae(args.out)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
