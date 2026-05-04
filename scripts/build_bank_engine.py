"""Build a TRT engine from the bank-aware decoder ONNX.

Minimal counterpart to ``acestep.engine.trt.export.build_trt_engine``,
extended for the 8-input / 3-output bank-aware graph. Static-shape
inputs (bank_k, bank_v, bank_bias) get a fixed profile; dynamic dims
(batch, enc) get a full min/opt/max range matching the existing
decoder engine.

Run:

    .venv/Scripts/python.exe scripts/build_bank_engine.py \\
        --onnx trt_engines/_onnx_bank/bank_decoder.onnx \\
        --engine trt_engines/bank_decoder_b8/bank_decoder_b8.engine
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loguru import logger


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", type=Path, required=True, help="Bank-aware ONNX path")
    p.add_argument("--engine", type=Path, required=True, help="Output .engine path")
    p.add_argument("--seq-len", type=int, default=1500, help="Static T (must match ONNX trace; 1500 = 60s)")
    p.add_argument("--enc-min", type=int, default=32)
    p.add_argument("--enc-opt", type=int, default=64)
    p.add_argument("--enc-max", type=int, default=512)
    p.add_argument("--batch-min", type=int, default=1)
    p.add_argument("--batch-opt", type=int, default=4)
    p.add_argument("--batch-max", type=int, default=8)
    p.add_argument("--num-steps", type=int, default=8)
    p.add_argument("--num-banked", type=int, default=12)
    p.add_argument("--kv-heads", type=int, default=8)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--workspace-gb", type=float, default=8.0)
    p.add_argument("--strongly-typed", action="store_true", default=True,
                   help="Strongly-typed network (bf16 baked into ONNX)")
    p.add_argument("--verbose", action="store_true", default=False,
                   help="trt.Logger.VERBOSE (massive output; pipe to a file)")
    p.add_argument("--opt-level", type=int, default=3,
                   help="builder_optimization_level 0..5")
    p.add_argument("--tactic-sources", default=None,
                   help="Comma-separated subset: CUBLAS,CUBLAS_LT,CUDNN,EDGE_MASK_CONVOLUTIONS,JIT_CONVOLUTIONS. "
                        "Default leaves builder default. Use 'NONE' to clear all sources.")
    args = p.parse_args()

    args.engine.parent.mkdir(parents=True, exist_ok=True)
    T_lat = args.seq_len // 2

    import tensorrt as trt

    log_level = trt.Logger.VERBOSE if args.verbose else trt.Logger.INFO
    trt_logger = trt.Logger(log_level)
    builder = trt.Builder(trt_logger)

    net_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    if args.strongly_typed and hasattr(trt.NetworkDefinitionCreationFlag, "STRONGLY_TYPED"):
        net_flags |= 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
        logger.info("Using STRONGLY_TYPED network")

    network = builder.create_network(net_flags)
    parser = trt.OnnxParser(network, trt_logger)

    if not args.strongly_typed:
        pass

    logger.info("Parsing ONNX from %s ...", args.onnx)
    if not parser.parse_from_file(str(args.onnx.resolve())):
        for i in range(parser.num_errors):
            logger.error("ONNX parse error: %s", parser.get_error(i))
        raise SystemExit(1)

    logger.info(
        "Network: %d inputs, %d outputs, %d layers",
        network.num_inputs, network.num_outputs, network.num_layers,
    )
    for i in range(network.num_inputs):
        t = network.get_input(i)
        logger.info("  IN  %d: %s shape=%s", i, t.name, list(t.shape))
    for i in range(network.num_outputs):
        t = network.get_output(i)
        logger.info("  OUT %d: %s shape=%s", i, t.name, list(t.shape))

    build_config = builder.create_builder_config()
    build_config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, int(args.workspace_gb * (1 << 30)),
    )
    if hasattr(build_config, "builder_optimization_level"):
        build_config.builder_optimization_level = args.opt_level

    if args.tactic_sources is not None:
        if args.tactic_sources.upper() == "NONE":
            build_config.set_tactic_sources(0)
            logger.info("Cleared all tactic sources")
        else:
            mask = 0
            for name in args.tactic_sources.split(","):
                name = name.strip().upper()
                if not hasattr(trt.TacticSource, name):
                    raise SystemExit(f"Unknown TacticSource {name!r}")
                mask |= 1 << int(getattr(trt.TacticSource, name))
            build_config.set_tactic_sources(mask)
            logger.info("Tactic sources set to: %s", args.tactic_sources)
    if not args.strongly_typed:
        if hasattr(trt.BuilderFlag, "BF16"):
            build_config.set_flag(trt.BuilderFlag.BF16)
        build_config.set_flag(trt.BuilderFlag.TF32)

    profile = builder.create_optimization_profile()

    Bmin, Bopt, Bmax = args.batch_min, args.batch_opt, args.batch_max
    T = args.seq_len
    Emin, Eopt, Emax = args.enc_min, args.enc_opt, args.enc_max
    NS = args.num_steps
    NB = args.num_banked
    KH = args.kv_heads
    HD = args.head_dim

    profile.set_shape(
        "hidden_states",
        min=(Bmin, T, 64), opt=(Bopt, T, 64), max=(Bmax, T, 64),
    )
    profile.set_shape(
        "timestep",
        min=(Bmin,), opt=(Bopt,), max=(Bmax,),
    )
    profile.set_shape(
        "encoder_hidden_states",
        min=(Bmin, Emin, 2048), opt=(Bopt, Eopt, 2048), max=(Bmax, Emax, 2048),
    )
    profile.set_shape(
        "context_latents",
        min=(Bmin, T, 128), opt=(Bopt, T, 128), max=(Bmax, T, 128),
    )
    profile.set_shape(
        "step_indices",
        min=(Bmin,), opt=(Bopt,), max=(Bmax,),
    )
    # Static-shape bank inputs, RANK-5 [N_banked, num_steps, kv_heads,
    # T_lat, head_dim]. Restored from the rank-4 collapse which was a
    # misdiagnosed workaround.
    bk5 = (NB, NS, KH, T_lat, HD)
    profile.set_shape("bank_k", min=bk5, opt=bk5, max=bk5)
    profile.set_shape("bank_v", min=bk5, opt=bk5, max=bk5)
    profile.set_shape(
        "bank_bias",
        min=(NS,), opt=(NS,), max=(NS,),
    )

    build_config.add_optimization_profile(profile)

    logger.info(
        "Building TRT engine (B=[%d,%d,%d] L_enc=[%d,%d,%d] T=%d T_lat=%d)",
        Bmin, Bopt, Bmax, Emin, Eopt, Emax, T, T_lat,
    )

    serialized = builder.build_serialized_network(network, build_config)
    if serialized is None:
        raise SystemExit("TRT engine build failed")

    with open(args.engine, "wb") as f:
        f.write(serialized)

    size_mb = args.engine.stat().st_size / (1 << 20)
    logger.info("Engine saved to %s (%.1f MB)", args.engine, size_mb)


if __name__ == "__main__":
    main()
