"""Empirical fidelity check: chunked TRT VAE encode vs full encode.

Encodes a 60 s fixture with the 60 s engine in one shot, then re-encodes
it in overlapping chunks (as the chunked-encode path would for sources
longer than the engine max) and compares the *moments* (mean + logvar)
frame-by-frame. Sampling noise is excluded by comparing moments, not
sampled latents.

The interesting number is how many frames from a chunk boundary the
means stay materially different — that bounds the receptive-field
margin the chunked path must trim.

Run: .venv/bin/python scripts/benchmarks/validate_chunked_vae_encode.py
"""
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

import soundfile as sf
import torch

SAMPLES_PER_FRAME = 1920
SR = 48_000


def encode_moments(audio_bct: torch.Tensor, entry, device) -> torch.Tensor:
    """Single-shot TRT encode returning moments [B, 128, T]."""
    from acestep.nodes.vae_nodes import _get_trt_stream

    ctx = entry["context"]
    stream = _get_trt_stream()
    dtypes = entry["tensor_dtypes"]
    inp = audio_bct.to(device=device, dtype=dtypes.get("audio", torch.float32)).contiguous()
    assert ctx.set_input_shape("audio", tuple(inp.shape))
    assert ctx.set_tensor_address("audio", inp.data_ptr())
    assert not ctx.infer_shapes()
    out_shape = tuple(ctx.get_tensor_shape("moments"))
    buf = torch.empty(out_shape, dtype=dtypes.get("moments", torch.float32), device=device)
    assert ctx.set_tensor_address("moments", buf.data_ptr())
    assert ctx.execute_async_v3(stream.ptr)
    stream.synchronize()
    return buf.clone().float()


def main():
    from acestep.nodes.vae_nodes import _get_trt_vae
    from acestep.paths import trt_engine_path

    device = torch.device("cuda")
    engine_path = str(trt_engine_path("vae_encode_fp16_60s"))
    entry = _get_trt_vae(engine_path, device)

    eng = entry["engine"]
    mn, opt, mx = eng.get_tensor_profile_shape("audio", 0)
    print(f"engine profile audio: min={mn} opt={opt} max={mx}")

    data, sr = sf.read(
        str(Path.home() / ".daydream-scope/models/demon/fixtures/prog_rock_loop_60s_enm/source.wav"),
        dtype="float32", always_2d=True,
    )
    assert sr == SR, f"fixture is {sr} Hz, expected {SR}"
    wav = torch.from_numpy(data.T)  # [C, N]
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    n = (wav.shape[-1] // SAMPLES_PER_FRAME) * SAMPLES_PER_FRAME
    wav = wav[:, :n].unsqueeze(0)  # [1, 2, N]
    total_frames = n // SAMPLES_PER_FRAME
    print(f"audio: {n} samples = {total_frames} frames = {n / SR:.1f}s")

    full = encode_moments(wav, entry, device)  # [1, 128, T]
    print(f"full moments: {tuple(full.shape)}")
    mean_full, logvar_full = full.chunk(2, dim=1)
    std_scale = torch.exp(0.5 * logvar_full).mean().item()
    mean_mag = mean_full.abs().mean().item()
    print(f"mean |mean|={mean_mag:.4f}  avg sampling std={std_scale:.4f}")

    # Re-run full encode to measure TRT nondeterminism noise floor.
    full2 = encode_moments(wav, entry, device)
    rerun_diff = (full - full2).abs().max().item()
    print(f"rerun max abs diff (noise floor): {rerun_diff:.6f}")

    # Chunked encode: core 20s, sweep margin.
    core_frames = 500  # 20 s
    for margin_frames in (0, 1, 2, 4, 8, 16, 32, 48, 96):
        chunks = []
        t = 0
        while t < total_frames:
            core_end = min(t + core_frames, total_frames)
            ctx_start = max(0, t - margin_frames)
            ctx_end = min(total_frames, core_end + margin_frames)
            seg = wav[:, :, ctx_start * SAMPLES_PER_FRAME : ctx_end * SAMPLES_PER_FRAME]
            m = encode_moments(seg, entry, device)
            chunks.append(m[:, :, t - ctx_start : t - ctx_start + (core_end - t)])
            t = core_end
        chunked = torch.cat(chunks, dim=2)
        assert chunked.shape == full.shape, (chunked.shape, full.shape)
        mean_c, _ = chunked.chunk(2, dim=1)
        d = (mean_c - mean_full).abs()
        # Per-frame max over channels
        per_frame = d.amax(dim=(0, 1))
        worst = per_frame.max().item()
        worst_frame = per_frame.argmax().item()
        print(
            f"margin={margin_frames:3d} frames ({margin_frames / 25:.2f}s): "
            f"max|Δmean|={worst:.5f} (frame {worst_frame}), "
            f"p99={per_frame.quantile(0.99).item():.5f}, "
            f"ratio to sampling std={worst / std_scale:.3f}"
        )


if __name__ == "__main__":
    main()
