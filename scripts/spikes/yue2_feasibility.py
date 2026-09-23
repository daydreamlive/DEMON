"""Bounded YuE2 GPU experiments; no DEMON session or model downloads.

Run with the pinned upstream package in its own environment. JSON is updated
after each experiment, and semantic outputs are cached to avoid repeating AR
generation. Short clips/instruments are measurements, never admission gates.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf

CODE_REVISION = "0edaf2f4053ef4731334b8329834b107977f9637"
MODEL_REVISION = "29b3558dd46954a0cd9021dc76d5c91864a0f1c7"
VAE_REVISION = "9a94e1d0ea9f8087e98f77fa88df4a4068104d2a"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    # Windows readers/antivirus can briefly hold a handle without delete-share.
    for attempt in range(20):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.05)


def sync():
    torch.cuda.synchronize()


def timed(fn):
    sync()
    start = time.perf_counter()
    result = fn()
    sync()
    return result, (time.perf_counter() - start) * 1000


def bench(fn, repeats=5, warmups=2):
    for _ in range(warmups):
        fn()
    sync()
    wall, gpu = [], []
    for _ in range(repeats):
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start = time.perf_counter()
        begin.record()
        fn()
        end.record()
        end.synchronize()
        wall.append((time.perf_counter() - start) * 1000)
        gpu.append(begin.elapsed_time(end))
    return {"repeats": repeats, "wall_p50_ms": float(np.median(wall)),
            "wall_p95_ms": float(np.percentile(wall, 95)),
            "gpu_p50_ms": float(np.median(gpu)), "wall_samples_ms": wall}


def errors(actual, expected):
    a, b = actual.detach().float(), expected.detach().float()
    if a.shape != b.shape:
        return {"shape_match": False, "actual": list(a.shape), "expected": list(b.shape)}
    diff = a - b
    return {"shape_match": True, "finite": bool(torch.isfinite(a).all()),
            "max_abs": float(diff.abs().max()), "rmse": float(diff.square().mean().sqrt()),
            "relative_l2": float(diff.norm() / b.norm().clamp_min(1e-12))}


def audio_stats(audio):
    return {"seconds": len(audio) / 48000, "finite": bool(np.isfinite(audio).all()),
            "peak_unclipped": float(np.abs(audio).max()),
            "rms": float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))),
            "clipped_fraction": float(np.mean(np.abs(audio) > 1))}


def load_verified_model(path, *, vae=False, device="cuda"):
    """Read once on a network drive, verify SHA256, then load via CPU RAM.

    Avoid per-tensor network mmap reads and a second full read for hashing.
    Meta construction avoids allocating randomly initialized checkpoint weights.
    """
    from safetensors.torch import load
    from yue2.modeling_vae import YuE2VAE, YuE2VAEConfig
    from yue2.modeling_yue2 import YuE2ForCausalLM, YuE2Config
    config_bytes = (path / "config.json").read_bytes()
    data = (path / "model.safetensors").read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    expected = json.loads((path / "weights_manifest.json").read_text())["files"]["model.safetensors"]
    if digest != expected["sha256"] or len(data) != expected["bytes"]:
        raise ValueError(f"Checkpoint integrity failure: {path}")
    identity = {"files": {"model.safetensors": {"sha256": digest, "bytes": len(data)}},
                "config_sha256": hashlib.sha256(config_bytes).hexdigest()}
    state = load(data)
    del data
    config = json.loads(config_bytes)
    with torch.device("meta"):
        model = YuE2VAE(YuE2VAEConfig(**config), decoder_only=True) if vae else YuE2ForCausalLM(YuE2Config(**config))
    if vae:
        state = {key: value for key, value in state.items() if key.startswith("decoder.")}
    model.load_state_dict(state, strict=True, assign=True)
    model.to(device=device, dtype=torch.float32 if vae else torch.bfloat16).eval().requires_grad_(False)
    return model, identity


class GraphCall:
    """A fixed-shape graph with mutable inputs and explicitly shared output."""

    def __init__(self, function, *inputs):
        self.inputs = [value.clone() for value in inputs]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                function(*self.inputs)
        torch.cuda.current_stream().wait_stream(stream)
        sync()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.output = function(*self.inputs)

    def __call__(self, *inputs):
        for destination, value in zip(self.inputs, inputs, strict=True):
            destination.copy_(value)
        self.graph.replay()
        return self.output


class BatchedVelocity(torch.nn.Module):
    """YuE2 cached velocity equations, batched with tensor-valued raw time.

    The cache belongs to one conditioning sequence and is shared across rows.
    This experiment does not yet batch different prompts or semantic sequences.
    """

    def __init__(self, engine):
        super().__init__()
        self.model = engine.model
        self.cache = engine.cache
        self.cos, self.sin, self.pos_emb = engine.cos, engine.sin, engine.pos_emb

    def forward(self, state, raw_time):
        model = self.model
        batch, frames, _ = state.shape
        count = frames + 2
        shifted = raw_time.to(state.dtype).sigmoid()
        shift = model.config.timestep_shift
        shifted = shift * shifted / (1 + (shift - 1) * shifted)
        x = model.vae2llm(F.pad(state, (0, 0, 1, 1)))
        times = shifted[:, None].expand(batch, count).reshape(-1)
        x = x + model.time_embedder(times).reshape(batch, count, -1)
        x = x + self.pos_emb
        for layer, (ar_k, ar_v) in zip(model.model.layers, self.cache, strict=True):
            q, k, v = layer.nar_self_attn.project_qkv(layer.nar_input_layernorm(x), self.cos, self.sin)
            k = torch.cat((ar_k[None].expand(batch, -1, -1, -1), k), dim=1)
            v = torch.cat((ar_v[None].expand(batch, -1, -1, -1), v), dim=1)
            h = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                              v.transpose(1, 2), enable_gqa=q.shape[2] != k.shape[2])
            x = x + layer.nar_self_attn.o_proj(h.transpose(1, 2).reshape(batch, count, -1))
            x = x + layer.nar_mlp(layer.nar_pre_mlp_layernorm(x))
        return model.llm2vae(model.model.norm(x))[:, 1:-1]


def graph_solve(graph, noise, steps=32):
    """Preserve upstream CPU-logit timing and midpoint BF16 arithmetic."""
    state = noise.clone()
    dt = 1 / steps
    for step in range(steps):
        t = 1 - step * dt
        raw = torch.logit(torch.tensor(t, dtype=torch.float64)).clamp(-20, 20).item()
        raw_mid = torch.logit(torch.tensor(t - dt / 2, dtype=torch.float64)).clamp(-20, 20).item()
        first = graph(state, torch.full((len(state),), raw, device=state.device, dtype=state.dtype))
        # Materialize midpoint before replay overwrites graph's output buffer.
        mid = state - first * (dt / 2)
        second = graph(mid, torch.full((len(state),), raw_mid, device=state.device, dtype=state.dtype))
        state = state - second * dt
    return state


class Recorder:
    def __init__(self, output, phase):
        self.path = output / f"{phase}.json"
        self.data = {"phase": phase, "status": "running", "code_revision": CODE_REVISION,
                     "model_revision": MODEL_REVISION, "vae_revision": VAE_REVISION,
                     "python": platform.python_version(), "torch": torch.__version__,
                     "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
                     "device_memory_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
                     "benchmark_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                     "measurements": []}
        self.started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        write_json(self.path, self.data)

    def add(self, label, **values):
        row = {"label": label, **values}
        self.data["measurements"].append(row)
        write_json(self.path, self.data)
        print(json.dumps(row, allow_nan=False), flush=True)

    def finish(self):
        self.data.update(status="complete", elapsed_s=time.perf_counter() - self.started,
                         peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                         peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
        write_json(self.path, self.data)


def window_decode(vae, latent, start, core, halo):
    frames = latent.shape[-1]
    end = min(frames, start + core)
    left, right = max(0, start - halo), min(frames, end + halo)
    tile = vae.decoder(latent[..., left:right].contiguous())
    offset = (start - left) * 1920
    samples = min(end * 1920, 1920 * frames - 64) - start * 1920
    return tile[..., offset:offset + samples]


def vae_phase(args):
    from yue2.modeling_vae import YuE2VAE
    record = Recorder(args.output, "vae")
    (vae, identity), load_ms = timed(lambda: load_verified_model(args.model_root / "YuE2-Vae", vae=True))
    record.add("load", wall_ms=load_ms)
    record.data["weight_identity"] = identity
    generator = torch.Generator(device="cpu").manual_seed(381)
    latent = torch.randn(1, 64, 800, generator=generator).cuda()
    real = args.output / "piano_score" / "latent.npy"
    if real.exists():
        latent = torch.from_numpy(np.load(real)).T[None].contiguous().cuda()
    record.add("input", frames=latent.shape[-1], source="generated_piano" if real.exists() else "seeded_gaussian")
    full = vae.decode(latent)
    record.add("full", frames=latent.shape[-1], **bench(lambda: vae.decode(latent), repeats=3))
    for core in [5, 10, 25, 50]:
        for halo in [16, 12]:
            worst = []
            for start in [0, min(40, max(0, latent.shape[-1] - core)), max(0, latent.shape[-1] - core)]:
                actual = window_decode(vae, latent, start, core, halo)
                expected = full[..., start * 1920:start * 1920 + actual.shape[-1]]
                worst.append(errors(actual, expected))
            record.add("window_parity", core=core, halo=halo, comparisons=worst)
        start = min(40, max(0, latent.shape[-1] - core))
        record.add("window_eager", core=core, halo=16, **bench(lambda: window_decode(vae, latent, start, core, 16)))
    # Fold inference-invariant weight normalization before fixed-shape capture.
    from torch.nn.utils import remove_weight_norm
    for module in vae.decoder.modules():
        if hasattr(module, "weight_g"):
            remove_weight_norm(module)
    record.add("fold_weight_norm", **errors(vae.decode(latent), full))
    for core in [5, 25]:
        size = core + 32
        z = latent[..., :size].contiguous()
        reference = vae.decoder(z).clone()
        graph, capture_ms = timed(lambda: GraphCall(vae.decoder, z))
        record.add("window_graph", core=core, halo=16, capture_ms=capture_ms,
                   parity=errors(graph(z), reference), **bench(lambda: graph(z)))
        del graph
        gc.collect()
    for frames in [1, 2, 5, 25]:
        audio = vae.decode(latent[..., :frames])
        record.add("minimum_shape", frames=frames, samples=audio.shape[-1], finite=bool(torch.isfinite(audio).all()))
    record.finish()
    del vae, full, latent
    gc.collect()
    torch.cuda.empty_cache()


PIANO_ABC = '''X:1
T:
M:4/4
L:1/8
Q:1/4=120
V: Vocal clef=treble name="Vocal Melody" snm="Vocal"
V: Ins clef=treble name="Ins Melody" snm="Inst."
K:C
% instrumental
V: Vocal
Z4|
V: Ins
"C"C2E2G2E2|"F"F2A2c2A2|"G"G2B2d2B2|"C"c2G2E2C2|
'''

CASES = {
    "piano_score": dict(style="Solo acoustic grand piano, instrumental, only piano, clean recording, gentle melody, 120 BPM",
                        lyrics="[Verse]\n\n[Outro]\n", cot="full", abc=PIANO_ABC, seed=381),
    "drums_off": dict(style="Solo acoustic drum kit, instrumental drum groove, only drums, dry close microphones, 120 BPM",
                      lyrics="[Verse]\n\n[Outro]\n", cot="off", seed=382, cfg_scale=1.0),
    "vocal_control": dict(style="English, warm female vocal, soulful pop, piano, bass, drums, 120 BPM",
                          lyrics="[Verse]\nMorning light across the floor\nHear the music through the door\n[Chorus]\nWe keep moving with the day\nLet the rhythm lead the way\n", cot="full", seed=383),
}


def model_phase(args):
    from yue2 import YuE2Pipeline
    from yue2.nar import CachedNAR, Chunk, song_chunks
    from yue2.modeling_vae import YuE2VAE
    record = Recorder(args.output, "model")
    record.data["ar_attention_backend"] = args.ar_attention
    (model, model_identity), load_ms = timed(lambda: load_verified_model(args.model_root / "YuE2-3B"))
    record.add("model_read_verify_load", wall_ms=load_ms)
    (vae, vae_identity), load_ms = timed(lambda: load_verified_model(args.model_root / "YuE2-Vae", vae=True))
    record.add("vae_load_resident_with_model", wall_ms=load_ms)
    identities = {"YuE2-3B": model_identity, "YuE2-Vae": vae_identity}
    # Supply identities already verified above; all generation remains upstream.
    with patch("yue2.pipeline.model_identity", side_effect=lambda path, verify: identities[Path(path).name]):
        pipe = YuE2Pipeline(args.model_root / "YuE2-3B", args.model_root / "YuE2-Vae", device="cuda", progress=False)
    pipe._model = model
    record.data["weight_identities"] = identities
    cached = {}
    for name in args.cases:
        request = CASES[name]
        folder = args.output / name
        folder.mkdir(parents=True, exist_ok=True)
        state_path = folder / "semantic.json"
        max_tokens = 800 if name == "vocal_control" else 400
        sampling = dict(min_tokens=25, max_tokens=max_tokens)
        identity = dict(request=request, sampling=sampling, code_revision=CODE_REVISION, model_revision=MODEL_REVISION,
                        torch=torch.__version__, ar_attention=args.ar_attention)
        if state_path.exists() and json.loads(state_path.read_text())['identity'] == identity:
            saved = json.loads(state_path.read_text())
            record.add("semantic_cache_hit", case=name, tokens=len(saved["tokens"]))
        else:
            plan, plan_ms = timed(lambda: pipe.plan(**request, abc_sampling=dict(min_tokens=32, max_tokens=512)))
            semantic, semantic_ms = timed(lambda: pipe.generate_semantic(plan, sampling=sampling))
            saved = dict(identity=identity, prefix=plan.prefix, tokens=semantic.tokens, abc=plan.abc,
                         plan_truncated=plan.truncated, semantic_truncated=semantic.truncated,
                         plan_ms=plan_ms, semantic_ms=semantic_ms, plan_timing=plan.timing, semantic_timing=semantic.timing)
            write_json(state_path, saved)
            record.add("semantic_generation", case=name, **{k: v for k, v in saved.items() if k not in {"identity", "prefix", "tokens", "abc"}}, tokens=len(saved["tokens"]))
        if not saved["tokens"]:
            record.add("empty_semantics", case=name)
            continue
        cached[name] = saved
        chunk = song_chunks(saved["prefix"], saved["tokens"], request["seed"])[0]
        engine, prefill_ms = timed(lambda: CachedNAR(model, chunk))
        latent, solve_ms = timed(engine.solve)
        z = latent.T[None].contiguous().cuda()
        decoded, decode_ms = timed(lambda: vae.decode(z))
        audio = decoded[0].T.float().cpu().numpy()
        np.save(folder / "latent.npy", latent.numpy())
        np.save(folder / "initial_noise.npy", chunk.noise.numpy())
        sf.write(folder / "audio.wav", np.clip(audio, -1, 1), 48000, subtype="PCM_24")
        record.add("baseline", case=name, frames=len(latent), prefill_ms=prefill_ms, solve_32_midpoint_ms=solve_ms,
                   decode_ms=decode_ms, audio=audio_stats(audio))
        engine.close()
    if "piano_score" not in cached:
        record.finish()
        return
    saved = cached["piano_score"]
    for frames in [1, 5, 25, 50, 100, 200]:
        if frames > len(saved["tokens"]):
            continue
        chunk = song_chunks(saved["prefix"], saved["tokens"][:frames], 381)[0]
        engine, prefill_ms = timed(lambda: CachedNAR(model, chunk))
        state = chunk.noise.to(device="cuda", dtype=torch.bfloat16)
        record.add("velocity_eager", frames=frames, prefill_ms=prefill_ms, **bench(lambda: engine.velocity(state, 0.0)))
        latent, solve_ms = timed(engine.solve)
        decoded = vae.decode(latent.T[None].contiguous().cuda())
        audio = decoded[0].T.float().cpu().numpy()
        sf.write(args.output / f"piano_excerpt_{frames:04d}f.wav", np.clip(audio, -1, 1), 48000, subtype="PCM_24")
        record.add("short_acoustic_context", frames=frames, solve_ms=solve_ms, audio=audio_stats(audio))
        if frames == 200:
            velocity = BatchedVelocity(engine).eval()
            raw = torch.zeros(1, device="cuda", dtype=torch.bfloat16)
            eager_reference = engine.velocity(state, 0.0)
            record.add("batched_wrapper_parity", frames=frames, **errors(velocity(state[None], raw)[0], eager_reference))
            graph, capture_ms = timed(lambda: GraphCall(velocity, state[None], raw))
            record.add("velocity_graph", frames=frames, depth=1, capture_ms=capture_ms,
                       parity=errors(graph(state[None], raw)[0], eager_reference), **bench(lambda: graph(state[None], raw)))
            solved, ms = timed(lambda: graph_solve(graph, state[None], 32))
            record.add("solver_graph", frames=frames, steps=32, solve_ms=ms, parity=errors(solved[0], latent.cuda()))
            # Step reduction on the undistilled checkpoint is not an approved
            # acceleration strategy. Preserve its 32 midpoint steps.
            del graph
            states = state[None].repeat(4, 1, 1)
            times = torch.tensor([20., 0., -1., -3.], device="cuda", dtype=torch.bfloat16)
            expected = torch.stack([engine.velocity(states[i], float(times[i])) for i in range(4)])
            graph, capture_ms = timed(lambda: GraphCall(velocity, states, times))
            record.add("velocity_graph", frames=frames, depth=4, capture_ms=capture_ms,
                       mixed_timestep_parity=errors(graph(states, times), expected), **bench(lambda: graph(states, times)))
            del graph, velocity
        engine.close()
        del engine
        gc.collect()
    # Preserve positions and hide semantic conditioning to probe codec dropout.
    frames = min(200, len(saved["tokens"]))
    chunk = song_chunks(saved["prefix"], saved["tokens"][:frames], 381)[0]
    text_chunk = Chunk(chunk.ar_tokens, chunk.noise, nar_cond_end=len(saved["prefix"]))
    engine = CachedNAR(model, text_chunk)
    latent, ms = timed(engine.solve)
    audio = vae.decode(latent.T[None].contiguous().cuda())[0].T.cpu().numpy()
    sf.write(args.output / "piano_text_conditioning_only.wav", np.clip(audio, -1, 1), 48000, subtype="PCM_24")
    record.add("codec_dropout_probe", frames=frames, solve_ms=ms, audio=audio_stats(audio),
               note="Semantic keys hidden; original sequence layout retained. Quality unjudged.")
    engine.close()
    record.finish()
    pipe.close()
    del model, vae, pipe, engine
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["vae", "model", "all"], required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "out/yue2/20260914")
    parser.add_argument("--cases", nargs="+", choices=list(CASES), default=list(CASES))
    parser.add_argument("--ar-attention", choices=["auto", "flash", "cudnn", "sdpa"],
                        default="cudnn" if sys.platform == "win32" else "auto",
                        help="Windows torch 2.9 exposes the native Flash schema but lacks its implementation.")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from yue2.cuda_graph import GraphAR
    def ar_graph(*positional, **keywords):
        keywords["attention_backend"] = args.ar_attention
        return GraphAR(*positional, **keywords)
    with torch.inference_mode(), patch("yue2.cuda_graph.GraphAR", side_effect=ar_graph):
        if args.phase in {"model", "all"}:
            model_phase(args)
        if args.phase in {"vae", "all"}:
            vae_phase(args)


if __name__ == "__main__":
    main()
