"""YuE2 musical-duration, acoustic TensorRT, and actual DEMON ring experiments.

Invoked by a persistent local worker with verified resident weights. This keeps
checkpoint I/O and failures out of the critical path. All solves use 32 midpoint
steps; no raw-checkpoint step reduction is performed.
"""
from __future__ import annotations

import gc
import json
import math
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from scripts.spikes.yue2_feasibility import (
    BatchedVelocity, GraphCall, audio_stats, bench, errors, graph_solve,
    timed, write_json,
)
from acestep.engine.stream import StreamPipeline, SlotRequest
from acestep.engine.diffusion import DiffusionConfig


class AcousticExport(torch.nn.Module):
    """Only NAR weights; conditioning KV is mutable engine input, not constants."""
    def __init__(self, engine):
        super().__init__()
        model = engine.model
        self.vae2llm, self.llm2vae = model.vae2llm, model.llm2vae
        self.time_embedder, self.norm = model.time_embedder, model.model.norm
        self.layers = torch.nn.ModuleList([
            torch.nn.ModuleDict(dict(norm=layer.nar_input_layernorm, attention=layer.nar_self_attn,
                                     norm_mlp=layer.nar_pre_mlp_layernorm, mlp=layer.nar_mlp))
            for layer in model.model.layers
        ])
        self.shift = model.config.timestep_shift
        self.groups = model.config.num_attention_heads // model.config.num_key_value_heads
        self.register_buffer('cos', engine.cos)
        self.register_buffer('sin', engine.sin)
        self.register_buffer('position', engine.pos_emb)

    def forward(self, state, raw_time, keys, values, cos=None, sin=None, position=None):
        cos = self.cos if cos is None else cos
        sin = self.sin if sin is None else sin
        position = self.position if position is None else position
        batch, frames, _ = state.shape
        count = frames + 2
        shifted = raw_time.sigmoid()
        shifted = self.shift * shifted / (1 + (self.shift - 1) * shifted)
        x = self.vae2llm(F.pad(state, (0, 0, 1, 1)))
        times = shifted[:, None].expand(batch, count).reshape(-1)
        x = x + self.time_embedder(times).reshape(batch, count, -1) + position
        for index, layer in enumerate(self.layers):
            q, k, v = layer['attention'].project_qkv(layer['norm'](x), cos, sin)
            k = torch.cat((keys[index][None].expand(batch, -1, -1, -1), k), dim=1).transpose(1, 2)
            v = torch.cat((values[index][None].expand(batch, -1, -1, -1), v), dim=1).transpose(1, 2)
            # Explicit GQA repetition is supported by the legacy ONNX exporter.
            k = k.repeat_interleave(self.groups, dim=1)
            v = v.repeat_interleave(self.groups, dim=1)
            h = F.scaled_dot_product_attention(q.transpose(1, 2), k, v)
            x = x + layer['attention'].o_proj(h.transpose(1, 2).reshape(batch, count, -1))
            x = x + layer['mlp'](layer['norm_mlp'](x))
        return self.llm2vae(self.norm(x))[:, 1:-1]


class RingAdapter:
    name = 'yue2_research'
    latent_channels, latent_rate_hz, sample_rate = 64, 25., 48000

    def __init__(self, velocity, noise):
        self.velocity, self.noise = velocity, noise
        schedule = torch.arange(33, dtype=torch.float64) / 32
        self.schedule = 1 - schedule
        self.raw = torch.stack((self.schedule[:-1], self.schedule[:-1] - 1 / 64)).logit().clamp(-20, 20).to(noise)

    def request_frames(self, request):
        return request.latent_frames

    def request_device_dtype(self, request):
        return self.noise.device, self.noise.dtype

    def build_schedule(self, config, denoise, device, dtype):
        assert denoise == 1 and config.infer_steps == 32
        return self.schedule


class YuE2Ring(StreamPipeline):
    """Real DEMON queue/slot/tick lifecycle, with an explicit midpoint tick.

    The stock Euler tick and ACE/SA3 controls are not silently reused. This
    research subclass supports fixed conditioning and exact supplied noise.
    """
    def _make_noise(self, request):
        return self.adapter.noise.clone()[None]

    def _init_slot(self, request):
        slot = super()._init_slot(request)
        request.aux_cond['born_at'] = time.perf_counter()
        return slot

    def _tick_pt(self, slots, indices):
        state = torch.cat([slot.xt for slot in slots])
        steps = torch.tensor([slot.step_idx for slot in slots], device=state.device)
        first = self.adapter.velocity(state, self.adapter.raw[0, steps])
        midpoint = state - first * (1 / 64)
        second = self.adapter.velocity(midpoint, self.adapter.raw[1, steps])
        updated = state - second * (1 / 32)
        for index, slot in enumerate(slots):
            slot.xt = updated[index:index + 1]
            slot.step_idx += 1


def choose_case(worker, request):
    name = request['case']
    state = json.loads((worker.output / name / 'semantic.json').read_text())
    if state['semantic_truncated'] or state['plan_truncated']:
        raise ValueError('A complete symbolic plan and natural semantic end are required for this performance case')
    from yue2.nar import CachedNAR, song_chunks
    chunks = song_chunks(state['prefix'], state['tokens'], state['request']['seed'])
    if len(chunks) != 1:
        raise ValueError('This ring experiment requires one complete acoustic context')
    engine = CachedNAR(worker.model, chunks[0])
    noise = chunks[0].noise.to(device='cuda', dtype=torch.bfloat16)
    return engine, noise


def export_acoustic(worker, request):
    engine, noise = choose_case(worker, request)
    module = AcousticExport(engine).eval()
    keys, values = [torch.stack([pair[i] for pair in engine.cache]) for i in range(2)]
    artifact = worker.runtime / 'nar_trt' / request['case']
    artifact.mkdir(parents=True, exist_ok=True)
    raw = torch.zeros(1, device='cuda', dtype=torch.bfloat16)
    inputs = (noise[None], raw, keys, values)
    reference = engine.velocity(noise, 0.)
    emit(worker, 'export_wrapper_parity', frames=len(noise), **errors(module(*inputs)[0], reference))
    meta = dict(case=request['case'], frames=len(noise), inputs={
        name: list(value.shape) for name, value in zip(['state', 'raw_time', 'keys', 'values'], inputs)},
        precision='BF16 weights and activations; original FP32 normalization reductions',
        model_identity=worker.model_identity)
    write_json(artifact / 'profile.json', meta)
    start = time.perf_counter()
    torch.onnx.export(module, inputs, str(artifact / 'velocity.onnx'),
                      input_names=['state', 'raw_time', 'keys', 'values'], output_names=['velocity'],
                      dynamic_axes={'state': {0: 'batch'}, 'raw_time': {0: 'batch'}, 'velocity': {0: 'batch'}},
                      opset_version=18, dynamo=False, do_constant_folding=True)
    emit(worker, 'export_complete', seconds=time.perf_counter() - start, artifact=str(artifact))
    engine.close()


def ring_graph(worker, request):
    engine, noise = choose_case(worker, request)
    velocity = BatchedVelocity(engine).eval()
    reference, solve_ms = timed(engine.solve)
    emit(worker, 'reference', case=request['case'], frames=len(noise), solve_ms=solve_ms)
    graphs = {}
    for depth in request.get('depths', [1, 4]):
        for size in range(1, depth + 1):
            if size not in graphs:
                states = noise[None].expand(size, -1, -1).contiguous()
                times = torch.linspace(3, -3, size, device='cuda', dtype=torch.bfloat16)
                graphs[size] = GraphCall(velocity, states, times)
        def forward(state, times):
            return graphs[len(state)](state, times)
        backend_reference, elapsed = timed(lambda: graph_solve(graphs[depth], noise[None].repeat(depth, 1, 1), 32))
        emit(worker, 'batch_solve', backend='cuda_graph', depth=depth, solve_ms=elapsed,
             upstream_parity=errors(backend_reference[0], reference.cuda()))
        measure_ring(worker, request['case'], 'cuda_graph', depth, forward, noise, backend_reference[0])
    engine.close()


class TensorRTVelocity:
    def __init__(self, artifact, keys, values):
        import tensorrt as trt
        from polygraphy.backend.trt import engine_from_bytes
        self.engine = engine_from_bytes((artifact / 'velocity.trt').read_bytes())
        self.keys, self.values = keys, values
        self.frames = json.loads((artifact / 'profile.json').read_text())['frames']
        self.batches = {}
        inspector = self.engine.create_engine_inspector()
        (artifact / 'layers.json').write_text(inspector.get_engine_information(trt.LayerInformationFormat.JSON))

    def batch(self, size):
        if size in self.batches:
            return self.batches[size]
        context = self.engine.create_execution_context()
        state = torch.empty(size, self.frames, 64, device='cuda', dtype=torch.bfloat16)
        times = torch.empty(size, device='cuda', dtype=torch.bfloat16)
        context.set_input_shape('state', tuple(state.shape))
        context.set_input_shape('raw_time', tuple(times.shape))
        output = torch.empty(tuple(context.get_tensor_shape('velocity')), device='cuda', dtype=torch.bfloat16)
        for name, value in [('state', state), ('raw_time', times), ('keys', self.keys), ('values', self.values), ('velocity', output)]:
            if not context.set_tensor_address(name, value.data_ptr()):
                raise RuntimeError('Cannot bind ' + name)
        def forward(value, raw_time):
            state.copy_(value)
            times.copy_(raw_time)
            if not context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
                raise RuntimeError('TensorRT acoustic execution failed')
            return output
        self.batches[size] = forward
        return forward


def trt_ring(worker, request):
    engine, noise = choose_case(worker, request)
    reference = torch.from_numpy(np.load(worker.output / request['case'] / 'latent.npy')).cuda()
    keys, values = [torch.stack([pair[i] for pair in engine.cache]) for i in range(2)]
    artifact = worker.runtime / 'nar_trt' / request['case']
    trt_velocity = TensorRTVelocity(artifact, keys, values)
    exported = AcousticExport(engine).eval()
    graphs = {}
    for depth in request.get('depths', [1, 4]):
        for size in range(1, depth + 1):
            if size not in graphs:
                states = noise[None].expand(size, -1, -1).contiguous()
                times = torch.linspace(3, -3, size, device='cuda', dtype=torch.bfloat16)
                graphs[size] = GraphCall(trt_velocity.batch(size), states, times)
        states = noise[None].repeat(depth, 1, 1)
        times = torch.linspace(3, -3, depth, device='cuda', dtype=torch.bfloat16)
        actual = graphs[depth](states, times).clone()
        emit(worker, 'velocity_trt', depth=depth, frames=len(noise),
             export_wrapper_parity=errors(actual, exported(states, times, keys, values)),
             upstream_parity=errors(actual, torch.stack([engine.velocity(states[i], float(times[i])) for i in range(depth)])),
             **bench(lambda: graphs[depth](states, times)))
        backend_reference, elapsed = timed(lambda: graph_solve(graphs[depth], states, 32))
        emit(worker, 'batch_solve', backend='tensorrt_graph', depth=depth, solve_ms=elapsed,
             upstream_parity=errors(backend_reference[0], reference))
        if depth == 1:
            audio = worker.vae.decode(backend_reference[0].T[None].float().contiguous())[0].T.cpu().numpy()
            sf.write(worker.output / (request['case'] + '_trt_32steps.wav'), np.clip(audio, -1, 1), 48000, subtype='PCM_24')
        def forward(state, times):
            return graphs[len(state)](state, times)
        measure_ring(worker, request['case'], 'tensorrt_graph', depth, forward, noise, backend_reference[0])
    engine.close()
    del graphs, trt_velocity
    gc.collect()
    torch.cuda.empty_cache()


def measure_ring(worker, case, backend, depth, forward, noise, reference):
    adapter = RingAdapter(forward, noise)
    pipeline = YuE2Ring(None, DiffusionConfig(infer_steps=32, dcw_enabled=False), pipeline_depth=depth,
                       adapter=adapter, queue_cap=1)
    ticks, ages, queued_ages, finish_times, outputs, batch_steps = [], [], [], [], [], []
    warmup = worker.ring_options.get('warmup_ticks', 40)
    measured = worker.ring_options.get('measured_ticks', 64)
    for tick in range(warmup + measured):
        if tick == warmup:
            window_start = time.perf_counter()
        stagger = worker.ring_options.get('stagger_start', False)
        if not stagger or tick >= 32 or tick % max(1, 32 // depth) == 0:
            req = SlotRequest(seed=381, latent_frames=len(noise), aux_cond={'submitted_at': time.perf_counter()})
            pipeline.submit(req)
        before = time.perf_counter()
        result = pipeline.tick()
        torch.cuda.synchronize()
        now = time.perf_counter()
        if tick >= warmup:
            ticks.append((now - before) * 1000)
            if result is not None:
                completed = pipeline.last_finished_request
                ages.append((now - completed.aux_cond['born_at']) * 1000)
                queued_ages.append((now - completed.aux_cond['submitted_at']) * 1000)
                finish_times.append(now)
                outputs.append(result)
            batch_steps.append([slot.step_idx for slot in pipeline._slots if slot is not None])
    elapsed = now - window_start
    expected = reference.cuda()
    emit(worker, 'actual_demon_ring', case=case, backend=backend, depth=depth, steps=32,
         frames=len(noise), warmup_ticks=warmup, measured_ticks=measured,
         staggered_initial_admission=worker.ring_options.get('stagger_start', False),
         tick_p50_ms=float(np.median(ticks)), tick_p95_ms=float(np.percentile(ticks, 95)),
         measured_wall_s=elapsed, completions=len(ages), completions_per_wall_s=len(ages) / elapsed,
         slot_age_p50_ms=float(np.median(ages)), submit_age_p50_ms=float(np.median(queued_ages)),
         completion_intervals_ms=(np.diff(finish_times) * 1000).tolist() if len(finish_times) > 1 else [],
         output_parity=[errors(result[0], expected) for result in outputs], tick_samples_ms=ticks, active_slot_steps=batch_steps,
         scope='Real StreamPipeline.tick queue and lifecycle; experimental midpoint override, fixed conditioning, no playback runner')
    pipeline.close()


def emit(worker, label, **data):
    row = dict(label=label, **data)
    worker.rows.append(row)
    write_json(worker.output / (worker.phase + '.json'), worker.rows)
    print(json.dumps(row), flush=True)


def piano_score(bars):
    melody = ['C2 E2 G2 c2', 'B2 G2 E2 C2', 'D2 F2 A2 G2', 'E2 D2 C4'][:bars]
    return f'''X:1
T:Short piano phrase
M:4/4
L:1/8
Q:1/4=120
V: Vocal clef=treble name="Vocal Melody" snm="Vocal"
V: Ins clef=treble name="Ins Melody" snm="Inst."
K:C
% instrumental
V: Vocal
Z{bars}|]
V: Ins
{'|'.join(melody)}|]
'''


def duration(worker, request):
    from yue2 import YuE2Pipeline
    from yue2.cuda_graph import GraphAR
    from yue2.nar import CachedNAR, song_chunks
    identities = {'YuE2-3B': worker.model_identity, 'YuE2-Vae': worker.vae_identity}
    with patch('yue2.pipeline.model_identity', side_effect=lambda path, verify: identities[Path(path).name]):
        pipe = YuE2Pipeline(worker.root / 'YuE2-3B', worker.root / 'YuE2-Vae', device='cuda', progress=False)
    pipe._model = worker.model
    cases = []
    for bars in [1, 4]:
        cases.append((f'piano_{bars}bar', dict(
            style=f'Solo acoustic piano, instrumental, only piano, one complete {bars}-bar phrase at 120 BPM, clean dry recording, end after the written score',
            lyrics='[Verse]\n\n[Outro]\n', cot='full', abc=piano_score(bars), seed=381), 2000))
    for mode in ['full', 'off']:
        cases.append((f'jingle_{mode}', dict(
            style='English, short two-line jingle, female singer with acoustic piano, 120 BPM, finish after the last lyric',
            lyrics='[Verse]\nHear the morning light\nEverything is bright\n[Outro]\n', cot=mode, seed=381), 3000))
    for bars, seed in [(1, 381), (1, 382), (2, 381), (4, 381)]:
        tune = 'C8E8G8E4C4' if bars < 4 else 'C8E8G8E4C4|C12z20'
        lyric = 'Hear the morning light' if bars == 1 else 'Hear the morning light\nEverything is bright'
        tune = tune if bars == 1 else tune + '|' + tune
        abc = f'''X:1
T:
M:4/4
L:1/32
Q:1/4=120
V: Vocal clef=treble name="Vocal Melody" snm="Vocal"
V: Ins clef=treble name="Ins Melody" snm="Inst."
K:C
% verse
V: Vocal
{tune}|]
V: Ins
Z{bars}|]
'''
        cases.append((f'concise_{bars}bar_s{seed}', dict(
            style='English, female voice and acoustic piano, 120 BPM, short musical jingle, no intro, no outro, sing once and end',
            lyrics='[Verse]\n' + lyric, cot='full', abc=abc, seed=seed), 1500))
    vendor = worker.runtime / 'vendor/YuE-0edaf2f4053ef4731334b8329834b107977f9637'
    cases.append(('upstream_song', json.loads((vendor / 'examples/song.json').read_text()), 4000))
    if request.get('cases'):
        cases = [case for case in cases if case[0] in request['cases']]
    def graph(*args, **kwargs):
        return GraphAR(*args, **dict(kwargs, attention_backend='cudnn'))
    with patch('yue2.cuda_graph.GraphAR', side_effect=graph):
        for name, prompt, budget in cases:
            folder = worker.output / name
            folder.mkdir(exist_ok=True)
            state_path = folder / 'semantic.json'
            if state_path.exists():
                state = json.loads(state_path.read_text())
            else:
                emit(worker, 'begin_case', case=name, request=prompt, semantic_budget=budget)
                plan, plan_ms = timed(lambda: pipe.plan(**prompt, abc_sampling={'min_tokens': 1, 'max_tokens': 4096}))
                semantic, semantic_ms = timed(lambda: pipe.generate_semantic(plan, sampling={'min_tokens': 1, 'max_tokens': budget}))
                state = dict(request=prompt, prefix=plan.prefix, abc=plan.abc, tokens=semantic.tokens,
                             plan_truncated=plan.truncated, semantic_truncated=semantic.truncated,
                             plan_ms=plan_ms, semantic_ms=semantic_ms,
                             plan_timing=plan.timing, semantic_timing=semantic.timing)
                write_json(state_path, state)
                if plan.abc:
                    (folder / 'score.abc').write_text(plan.abc, encoding='utf-8')
            emit(worker, 'semantic', case=name, tokens=len(state['tokens']),
                 natural_end=not state['semantic_truncated'], plan_complete=not state['plan_truncated'],
                 plan_ms=state['plan_ms'], semantic_ms=state['semantic_ms'])
            if not state['tokens']:
                continue
            if (folder / 'latent.npy').exists():
                latent = torch.from_numpy(np.load(folder / 'latent.npy'))
                solve_ms = None
            else:
                latents, solve_ms = [], 0.0
                for chunk in song_chunks(state['prefix'], state['tokens'], prompt['seed']):
                    engine = CachedNAR(worker.model, chunk)
                    value, elapsed = timed(engine.solve)
                    latents.append(value)
                    solve_ms += elapsed
                    engine.close()
                latent = torch.cat(latents)
                np.save(folder / 'latent.npy', latent.numpy())
            audio = worker.vae.decode(latent.T[None].contiguous().cuda())[0].T.cpu().numpy()
            np.save(folder / 'audio_unclipped.npy', audio)
            sf.write(folder / 'audio.wav', np.clip(audio, -1, 1), 48000, subtype='PCM_24')
            emit(worker, 'audio', case=name, solve_32_midpoint_ms=solve_ms,
                 natural_end=not state['semantic_truncated'], plan_complete=not state['plan_truncated'],
                 audio=audio_stats(audio))
    pipe.close()
    gc.collect()
    torch.cuda.empty_cache()


def run(worker, request):
    worker.phase = request['phase']
    worker.rows = []
    worker.ring_options = request
    if worker.phase.startswith('duration'):
        duration(worker, request)
    elif worker.phase == 'export':
        export_acoustic(worker, request)
    elif worker.phase.startswith('ring_graph'):
        ring_graph(worker, request)
    elif worker.phase.startswith('trt_ring'):
        trt_ring(worker, request)
    elif worker.phase.startswith('edit_profile'):
        import importlib
        from scripts.spikes import yue2_edit_profile
        importlib.reload(yue2_edit_profile).run(worker, request)
    else:
        raise ValueError(worker.phase)
