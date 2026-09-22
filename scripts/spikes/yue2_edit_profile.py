"""Full-song stage costs and cached-acoustic editing costs; original 32 steps.

Run via yue2_worker, phases edit_profile_prepare and edit_profile_full.
Preparation generates a fresh complete score/semantic sequence for export.
Full profiling measures a resident-engine request from text to CPU waveform,
then repeats synthesis with its prepared conditioning. No acoustic crop is used.
"""
from __future__ import annotations

import gc
import json
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

from scripts.spikes.yue2_feasibility import GraphCall, bench, errors, graph_solve, timed, write_json
from scripts.spikes.yue2_round2 import AcousticExport, TensorRTVelocity, choose_case, emit, YuE2Ring, RingAdapter
from acestep.engine.stream import SlotRequest
from acestep.engine.diffusion import DiffusionConfig


def pipeline(worker):
    from yue2 import YuE2Pipeline
    identities = {'YuE2-3B': worker.model_identity, 'YuE2-Vae': worker.vae_identity}
    with patch('yue2.pipeline.model_identity', side_effect=lambda path, verify: identities[Path(path).name]):
        pipe = YuE2Pipeline(worker.root / 'YuE2-3B', worker.root / 'YuE2-Vae',
                           device='cuda', progress=False, memory_budget_gib=30)
    pipe._model = worker.model
    return pipe


def generate(pipe, prompt):
    from yue2.cuda_graph import GraphAR
    def graph(*args, **kwargs):
        return GraphAR(*args, **dict(kwargs, attention_backend='cudnn'))
    with patch('yue2.cuda_graph.GraphAR', side_effect=graph):
        plan, plan_ms = timed(lambda: pipe.plan(**prompt, abc_sampling={'min_tokens': 1, 'max_tokens': 4096}))
        semantic, semantic_ms = timed(lambda: pipe.generate_semantic(plan, sampling={'min_tokens': 1, 'max_tokens': 6000}))
    if plan.truncated or semantic.truncated:
        raise RuntimeError('A naturally complete score and semantic sequence are required')
    return dict(request=prompt, prefix=plan.prefix, abc=plan.abc, tokens=semantic.tokens,
                plan_truncated=plan.truncated, semantic_truncated=semantic.truncated,
                plan_ms=plan_ms, semantic_ms=semantic_ms,
                plan_timing=plan.timing, semantic_timing=semantic.timing)


class WindowDecoder:
    """Existing FP32 TRT decoder: 37-frame context, five-frame interior core."""
    def __init__(self, path):
        from polygraphy.backend.trt import engine_from_bytes
        self.engine = engine_from_bytes(path.read_bytes())
        self.context = self.engine.create_execution_context()
        self.input = torch.zeros(1, 64, 37, device='cuda', dtype=torch.float32)
        self.output = torch.empty(tuple(self.context.get_tensor_shape('audio')), device='cuda', dtype=torch.float32)
        self.context.set_tensor_address('latent', self.input.data_ptr())
        self.context.set_tensor_address('audio', self.output.data_ptr())
        self.graph = GraphCall(self._forward, self.input)

    def _forward(self, value):
        self.input.copy_(value)
        if not self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError('VAE TensorRT execution failed')
        return self.output

    def __call__(self, latent):
        start = latent.shape[0] // 2
        context = latent[start - 16:start + 21].T[None].float().contiguous()
        return self.graph(context)[..., 16 * 1920:21 * 1920]


def prepare(worker, request):
    folder = worker.output / request['case']
    previous = json.loads((folder / 'semantic.json').read_text())
    pipe = pipeline(worker)
    state = generate(pipe, previous['request'])
    emit(worker, 'fresh_preparation', case=request['case'], plan_ms=state['plan_ms'],
         semantic_ms=state['semantic_ms'], seconds=len(state['tokens']) / 25,
         prefix_matches_previous=state['prefix'] == previous['prefix'],
         semantics_match_previous=state['tokens'] == previous['tokens'],
         plan_timing=state['plan_timing'], semantic_timing=state['semantic_timing'])
    write_json(folder / 'semantic.json', state)
    (folder / 'score.abc').write_text(state['abc'], encoding='utf-8')
    pipe.close()


def flexible_inputs(engine, noise):
    keys, values = [torch.stack([pair[i] for pair in engine.cache]) for i in range(2)]
    return dict(state=noise[None], raw_time=torch.zeros(1, device='cuda', dtype=torch.bfloat16),
                keys=keys, values=values, cos=engine.cos, sin=engine.sin, position=engine.pos_emb)


def export_flexible(worker, request):
    engine, noise = choose_case(worker, request)
    inputs = flexible_inputs(engine, noise)
    module = AcousticExport(engine).eval()
    emit(worker, 'flexible_wrapper_parity', **errors(module(*inputs.values())[0], engine.velocity(noise, 0.)))
    artifact = worker.runtime / 'nar_trt' / 'flexible_song'
    artifact.mkdir(parents=True, exist_ok=True)
    shapes = {k:list(v.shape) for k,v in inputs.items()}
    bounds = {}
    dynamic = {}
    for name, shape in shapes.items():
        lo, opt, hi = shape.copy(), shape.copy(), shape.copy()
        if name == 'state':
            lo[:2], opt[0], hi[:2] = [1,1000], 4, [4,2500]
            dynamic[name] = {0:'batch',1:'frames'}
        elif name == 'raw_time':
            lo[0], opt[0], hi[0] = 1,4,4
            dynamic[name] = {0:'batch'}
        elif name in ('keys','values'):
            lo[1], hi[1] = 1001,4000
            dynamic[name] = {1:'conditioning'}
        else:
            lo[1], hi[1] = 1002,2502
            dynamic[name] = {1:'padded_frames'}
        bounds[name] = [lo,opt,hi]
    dynamic['velocity'] = {0:'batch',1:'frames'}
    write_json(artifact/'profile.json', dict(inputs=shapes, optimization_shapes=bounds,
               frames=len(noise), precision='BF16; FP32 normalization and rotary inputs', model_identity=worker.model_identity))
    torch.onnx.export(module, tuple(inputs.values()), str(artifact/'velocity.onnx'),
                      input_names=list(inputs), output_names=['velocity'], dynamic_axes=dynamic,
                      opset_version=18, dynamo=False, do_constant_folding=True)
    emit(worker, 'flexible_export_complete', artifact=str(artifact))
    engine.close()


class FlexibleVelocity:
    def __init__(self, artifact):
        from polygraphy.backend.trt import engine_from_bytes
        self.engine = engine_from_bytes((artifact/'velocity.trt').read_bytes())

    def bind(self, engine, noise, size=1):
        tensors = flexible_inputs(engine, noise)
        tensors['state'] = noise[None].repeat(size,1,1)
        tensors['raw_time'] = torch.zeros(size, device='cuda', dtype=torch.bfloat16)
        context = self.engine.create_execution_context()
        for name, value in tensors.items():
            if not context.set_input_shape(name, tuple(value.shape)):
                raise RuntimeError('Input shape outside flexible profile: '+name)
            context.set_tensor_address(name, value.data_ptr())
        output = torch.empty(tuple(context.get_tensor_shape('velocity')), device='cuda', dtype=torch.bfloat16)
        context.set_tensor_address('velocity', output.data_ptr())
        def forward(state, times):
            tensors['state'].copy_(state)
            tensors['raw_time'].copy_(times)
            if not context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
                raise RuntimeError('Flexible acoustic TensorRT execution failed')
            return output
        captured = GraphCall(forward, tensors['state'], tensors['raw_time'])
        # CUDA graph nodes retain device addresses, not Python buffer/context ownership.
        captured.owner = (context, tensors, output, forward)
        return captured


def full(worker, request):
    from yue2.nar import CachedNAR, song_chunks
    case = request['case']
    prepared = json.loads((worker.output / case / 'semantic.json').read_text())
    pipe = pipeline(worker)
    # Set up all fixed-profile acceleration before request timing starts.
    engine, noise = choose_case(worker, request)
    trt = FlexibleVelocity(worker.runtime / 'nar_trt' / 'flexible_song')
    graph, graph_ms = timed(lambda: trt.bind(engine, noise))
    decoder = WindowDecoder(worker.runtime / 'trt_engines' / 'vae_fp32_t37.trt')
    # Warm only the full decoder; avoid another acoustic solve just for warmup.
    worker.vae.decode(noise.T[None].float().contiguous())
    torch.cuda.synchronize()
    emit(worker, 'runtime', gpu=torch.cuda.get_device_name(), torch=torch.__version__,
         graph_capture_ms=graph_ms, frames=len(noise), model_identity=worker.model_identity,
         scope='Resident BF16 NAR TRT, CUDA-graph AR, full FP32 VAE, all 32 midpoint steps')
    engine.close()
    started = time.perf_counter()
    state = generate(pipe, prepared['request'])
    chunks, noise_ms = timed(lambda: song_chunks(state['prefix'], state['tokens'], state['request']['seed']))
    if len(chunks) != 1:
        raise RuntimeError('This profile requires one original acoustic context')
    new_engine, prefill_ms = timed(lambda: CachedNAR(worker.model, chunks[0]))
    new_noise, update_ms = timed(lambda: chunks[0].noise.cuda().to(torch.bfloat16)[None])
    # Dynamic request geometry is captured here and counted in end-to-end time.
    graph, request_graph_ms = timed(lambda: trt.bind(new_engine, new_noise[0]))
    latent, solve_ms = timed(lambda: graph_solve(graph, new_noise, 32))
    audio, decode_ms = timed(lambda: worker.vae.decode(latent[0].T[None].float().contiguous())[0].T.cpu().numpy())
    total_s = time.perf_counter() - started
    emit(worker, 'fresh_end_to_end', audio_seconds=len(audio)/48000, total_s=total_s,
         plan_ms=state['plan_ms'], semantic_ms=state['semantic_ms'], noise_ms=noise_ms,
         acoustic_prefill_ms=prefill_ms, update_inputs_ms=update_ms, request_graph_ms=request_graph_ms,
         acoustic_solve_ms=solve_ms,
         vae_and_cpu_audio_ms=decode_ms, audio_seconds_per_wall_second=(len(audio)/48000)/total_s,
         prefix_matches_prepared=state['prefix'] == prepared['prefix'],
         semantics_match_prepared=state['tokens'] == prepared['tokens'],
         plan_timing=state['plan_timing'], semantic_timing=state['semantic_timing'])
    sf.write(worker.output / (case + '_full_trt.wav'), np.clip(audio, -1, 1), 48000, subtype='PCM_24')
    write_json(worker.output / 'full_request_semantic.json', state)
    np.save(worker.output / (case + '_trt_latent.npy'), latent[0].float().cpu().numpy())
    reference, reference_ms = timed(new_engine.solve)
    np.save(worker.output / 'full_request_reference_latent.npy', reference.numpy())
    emit(worker, 'same_context_upstream_comparison', upstream_solve_ms=reference_ms,
         **errors(latent[0], reference.cuda()))
    emit(worker, 'cached_acoustic_solve', **bench(lambda: graph_solve(graph, new_noise, 32), repeats=3, warmups=0))
    emit(worker, 'full_vae_and_cpu_audio', **bench(lambda: worker.vae.decode(latent[0].T[None].float().contiguous())[0].T.cpu().numpy(), repeats=3, warmups=0))
    emitted_window = decoder(latent[0]).clone()
    position = len(new_noise[0])//2
    expected_window = torch.from_numpy(audio[position*1920:(position+5)*1920]).T[None].cuda()
    emit(worker, 'trt_window_parity', **errors(emitted_window, expected_window))
    emit(worker, 'trt_window_and_cpu_audio', core_seconds=.2,
         **bench(lambda: decoder(latent[0]).cpu().numpy(), repeats=5, warmups=1))
    def preview():
        updated = graph_solve(graph, new_noise, 32)
        return decoder(updated[0]).cpu().numpy()
    emit(worker, 'cached_solve_to_cpu_window', **bench(preview, repeats=3, warmups=0))
    # Re-prefill is needed when prefix/semantic contents change, but unchanged
    # conditioning uses the same resident keys/values for every acoustic solve.
    def prefill_again():
        value = CachedNAR(worker.model, chunks[0])
        value.close()
    emit(worker, 'conditioning_prefill', **bench(prefill_again, repeats=3, warmups=0))
    new_engine.close()
    pipe.close()
    del decoder, graph, trt, new_engine, engine
    gc.collect()
    torch.cuda.empty_cache()


def live(worker, request):
    """Numerical velocity-gain change with full-context denoising and PCM output.

    This measures first changed and fully settled completed output, not an
    audible perceptual threshold or a production playback/device latency.
    """
    engine, noise = choose_case(worker, request)
    keys, values = [torch.stack([pair[i] for pair in engine.cache]) for i in range(2)]
    trt = TensorRTVelocity(worker.runtime / 'nar_trt' / request['case'], keys, values)
    decoder = WindowDecoder(worker.runtime / 'trt_engines' / 'vae_fp32_t37.trt')
    depth = request.get('depth', 4)
    graphs = {size: GraphCall(trt.batch(size), noise[None].repeat(size, 1, 1),
                             torch.zeros(size, device='cuda', dtype=torch.bfloat16))
              for size in range(1, depth+1)}
    gain = 1.0
    def velocity(state, times):
        value = graphs[len(state)](state, times)
        return value if gain == 1.0 else value * gain
    baseline = graph_solve(velocity, noise[None].repeat(depth, 1, 1), 32)[0].clone()
    baseline_pcm = decoder(baseline).clone()
    gain = 1.05
    settled = graph_solve(velocity, noise[None].repeat(depth, 1, 1), 32)[0].clone()
    gain = 1.0
    pipeline = YuE2Ring(None, DiffusionConfig(infer_steps=32, dcw_enabled=False),
                       pipeline_depth=depth, adapter=RingAdapter(velocity, noise), queue_cap=1)
    warmup, measured = 40, 64
    times, records = [], []
    switched_at = None
    for tick in range(warmup + measured):
        if tick == warmup:
            torch.cuda.synchronize()
            switched_at = time.perf_counter()
            gain = 1.05
        if tick >= 32 or tick % max(1, 32 // depth) == 0:
            pipeline.submit(SlotRequest(seed=381, latent_frames=len(noise),
                                        aux_cond={'submitted_at': time.perf_counter()}))
        before = time.perf_counter()
        result = pipeline.tick()
        pcm = decoder(result[0]).cpu().numpy() if result is not None else None
        torch.cuda.synchronize()
        after = time.perf_counter()
        if tick >= warmup:
            times.append((after-before)*1000)
            if result is not None:
                completed = pipeline.last_finished_request
                # Keep comparisons outside measured GPU work; retain tiny latent
                # snapshots and PCM copies for a separate post-loop audit.
                records.append(dict(at_ms=(after-switched_at)*1000,
                                    slot_ms=(after-completed.aux_cond['born_at'])*1000,
                                    admitted_after_change=completed.aux_cond['born_at'] >= switched_at,
                                    latent=result[0], pcm=pcm))
    wall = time.perf_counter()-switched_at
    finished = []
    for row in records:
        z, pcm = row.pop('latent'), row.pop('pcm')
        row.update(baseline_latent_error=errors(z, baseline), settled_latent_error=errors(z, settled),
                   baseline_pcm_error=errors(torch.from_numpy(pcm).cuda(), baseline_pcm))
        finished.append(row)
    first_changed = next((r['at_ms'] for r in finished if r['baseline_pcm_error']['max_abs'] > 1e-5), None)
    first_settled = next((r['at_ms'] for r in finished if r['admitted_after_change'] and
                         r['settled_latent_error']['max_abs'] == 0), None)
    emit(worker, 'live_control_ring_to_pcm', depth=depth, frames=len(noise), steps=32,
         control='velocity gain 1.0 -> 1.05, applied to in-flight midpoint evaluations',
         first_numerically_changed_pcm_ms=first_changed, first_fully_settled_pcm_ms=first_settled,
         core_seconds=.2, includes_vae_and_cpu_copy=True,
         tick_p50_ms=float(np.median(times)), tick_p95_ms=float(np.percentile(times,95)),
         wall_s=wall, completions=len(finished), completions_per_wall_s=len(finished)/wall,
         slot_age_p50_ms=float(np.median([r['slot_ms'] for r in finished])),
         completion_intervals_ms=np.diff([r['at_ms'] for r in finished]).tolist(), outputs=finished,
         scope='Real DEMON ring with research midpoint/gain adaptation; PCM output, no playback device or hot prompt changes')
    pipeline.close()
    engine.close()
    del graphs, trt, decoder, engine, pipeline
    gc.collect()
    torch.cuda.empty_cache()


def run(worker, request):
    if worker.phase == 'edit_profile_prepare':
        prepare(worker, request)
    elif worker.phase == 'edit_profile_full':
        full(worker, request)
    elif worker.phase.startswith('edit_profile_live'):
        live(worker, request)
    elif worker.phase == 'edit_profile_export_flexible':
        export_flexible(worker, request)
    else:
        raise ValueError(worker.phase)
