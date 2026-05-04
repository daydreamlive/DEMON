"""TEMPORARY: build the OLD 10s ONNX (rank-5, separate valid_steps + log_strength)
to test whether the existing bank_decoder_b8.engine artifact can still be reproduced
under the current TRT environment. If this builds and the current rank-5 path doesn't,
the difference is in the I/O surface (consolidated bank_bias vs separate
valid_steps+log_strength). If both crash, the environment has drifted since the
existing engine was built.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from loguru import logger


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", type=Path, required=True)
    p.add_argument("--engine", type=Path, required=True)
    p.add_argument("--seq-len", type=int, default=250)
    p.add_argument("--num-steps", type=int, default=8)
    p.add_argument("--num-banked", type=int, default=12)
    p.add_argument("--kv-heads", type=int, default=8)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--batch-min", type=int, default=1)
    p.add_argument("--batch-opt", type=int, default=4)
    p.add_argument("--batch-max", type=int, default=8)
    p.add_argument("--enc-min", type=int, default=32)
    p.add_argument("--enc-opt", type=int, default=64)
    p.add_argument("--enc-max", type=int, default=512)
    p.add_argument("--workspace-gb", type=float, default=8.0)
    p.add_argument("--opt-level", type=int, default=3)
    args = p.parse_args()

    args.engine.parent.mkdir(parents=True, exist_ok=True)
    T_lat = args.seq_len // 2
    import tensorrt as trt
    trt_logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(trt_logger)
    flags = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    flags |= (1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, trt_logger)
    if not parser.parse_from_file(str(args.onnx.resolve())):
        for i in range(parser.num_errors):
            logger.error("ONNX parse error: %s", parser.get_error(i))
        raise SystemExit(1)

    logger.info("Network: %d inputs, %d outputs", network.num_inputs, network.num_outputs)
    for i in range(network.num_inputs):
        t = network.get_input(i)
        logger.info("  IN  %d: %s shape=%s", i, t.name, list(t.shape))

    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(args.workspace_gb * (1 << 30)))
    cfg.builder_optimization_level = args.opt_level

    profile = builder.create_optimization_profile()
    Bmin, Bopt, Bmax = args.batch_min, args.batch_opt, args.batch_max
    Emin, Eopt, Emax = args.enc_min, args.enc_opt, args.enc_max
    NB, NS, KH, HD = args.num_banked, args.num_steps, args.kv_heads, args.head_dim
    T = args.seq_len

    profile.set_shape("hidden_states", min=(Bmin, T, 64), opt=(Bopt, T, 64), max=(Bmax, T, 64))
    profile.set_shape("timestep", min=(Bmin,), opt=(Bopt,), max=(Bmax,))
    profile.set_shape("encoder_hidden_states", min=(Bmin, Emin, 2048), opt=(Bopt, Eopt, 2048), max=(Bmax, Emax, 2048))
    profile.set_shape("context_latents", min=(Bmin, T, 128), opt=(Bopt, T, 128), max=(Bmax, T, 128))
    bk5 = (NB, NS, KH, T_lat, HD)
    profile.set_shape("bank_k", min=bk5, opt=bk5, max=bk5)
    profile.set_shape("bank_v", min=bk5, opt=bk5, max=bk5)
    profile.set_shape("valid_steps", min=(NS,), opt=(NS,), max=(NS,))
    profile.set_shape("log_strength", min=(1,), opt=(1,), max=(1,))
    profile.set_shape("step_indices", min=(Bmin,), opt=(Bopt,), max=(Bmax,))
    cfg.add_optimization_profile(profile)

    logger.info("Building engine...")
    serial = builder.build_serialized_network(network, cfg)
    if serial is None:
        raise SystemExit("build failed")
    args.engine.write_bytes(serial)
    logger.info("Engine saved %s (%d MB)", args.engine, len(serial) >> 20)


if __name__ == "__main__":
    main()
