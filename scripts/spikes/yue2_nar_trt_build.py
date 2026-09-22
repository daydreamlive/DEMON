"""Build one BF16 YuE2 acoustic engine with batch 1..4 and mutable prefix KV."""
import argparse
import json
from pathlib import Path
import time

import tensorrt as trt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, required=True)
    args = parser.parse_args()
    meta = json.loads((args.artifacts / 'profile.json').read_text())
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    reader = trt.OnnxParser(network, logger)
    if not reader.parse_from_file(str((args.artifacts / 'velocity.onnx').resolve())):
        raise RuntimeError('\n'.join(str(reader.get_error(i)) for i in range(reader.num_errors)))
    config = builder.create_builder_config()
    config.clear_flag(trt.BuilderFlag.TF32)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    config.builder_optimization_level = 2
    config.avg_timing_iterations = 1
    config.max_aux_streams = 0
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    profile = builder.create_optimization_profile()
    if 'optimization_shapes' in meta:
        for name, bounds in meta['optimization_shapes'].items():
            profile.set_shape(name, *bounds)
    else:
        for name in ['state', 'raw_time']:
            low = meta['inputs'][name]
            high = [4] + low[1:]
            profile.set_shape(name, low, high, high)
    config.add_optimization_profile(profile)
    start = time.perf_counter()
    print('BUILDING ACOUSTIC BF16 ENGINE: batch 1..4, no step reduction', flush=True)
    result = builder.build_serialized_network(network, config)
    if result is None:
        raise RuntimeError('TensorRT acoustic build failed')
    (args.artifacts / 'velocity.trt').write_bytes(bytes(result))
    report = dict(build_seconds=time.perf_counter()-start, bytes=(args.artifacts / 'velocity.trt').stat().st_size, tensorrt=trt.__version__,
                  input_shapes=meta['inputs'], profile_batch=[1, 4, 4], precision=meta['precision'])
    (args.artifacts / 'build.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
