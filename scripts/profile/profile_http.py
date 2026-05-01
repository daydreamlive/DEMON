#!/usr/bin/env python3
"""Profile HTTP overhead: measures round-trip vs server-side compute.

Requires the server to be running:
    uv run python -m acestep.engine.server --no-compile \
        --trt-decoder trt_engines/decoder_mixed_refit_b8_240s/decoder_mixed_refit_b8_240s.engine \
        --trt-vae-encode trt_engines/vae_encode_fp16_240s/vae_encode_fp16_240s.engine \
        --trt-vae-decode trt_engines/vae_decode_fp16_240s/vae_decode_fp16_240s.engine
"""

import io, json, os, sys, time, urllib.request

BASE = "http://127.0.0.1:8731"
AUDIO_FILE = os.path.join(
    os.path.dirname(__file__), "..", "..", "tests/fixtures", "new_order_confusion_60seconds.wav"
)


def post_json(path, data=None):
    body = json.dumps(data or {}).encode()
    req = urllib.request.Request(
        BASE + path, data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    resp = urllib.request.urlopen(req)
    rt_ms = (time.perf_counter() - t0) * 1000
    headers = dict(resp.headers)
    content = resp.read()
    return rt_ms, headers, content


def post_audio(path, filepath, name="default", max_duration=60.0):
    """Send audio file, trimmed to max_duration and pool-snapped."""
    import soundfile as sf_mod, numpy as np
    data, sr = sf_mod.read(filepath, dtype="float32")
    if data.ndim > 1:
        data = data.T  # [C, samples]
    else:
        data = data.reshape(1, -1)
    # Trim to max_duration
    max_samples = int(max_duration * sr)
    data = data[:2, :max_samples]
    # Pool snap (trim only)
    pool = 1920 * 5
    if sr != 48000:
        import torchaudio, torch as _torch
        wf = _torch.from_numpy(data)
        data = torchaudio.transforms.Resample(sr, 48000)(wf).numpy()
        sr = 48000
    rem = data.shape[-1] % pool
    if rem:
        data = data[:, :data.shape[-1] - rem]
    buf = io.BytesIO()
    sf_mod.write(buf, data.T, sr, format="WAV")
    body = buf.getvalue()

    req = urllib.request.Request(
        BASE + path, data=body,
        headers={
            "Content-Type": "audio/wav",
            "X-Source-Name": name,
        },
    )
    t0 = time.perf_counter()
    resp = urllib.request.urlopen(req)
    rt_ms = (time.perf_counter() - t0) * 1000
    return rt_ms, dict(resp.headers), resp.read()


def main():
    # Health check
    try:
        urllib.request.urlopen(BASE + "/health")
    except Exception:
        print(f"Server not reachable at {BASE}. Start it first.")
        sys.exit(1)

    # Prepare source
    print("=== prepare_source ===")
    rt, h, body = post_audio("/prepare_source", AUDIO_FILE)
    resp_data = json.loads(body)
    print(f"  round-trip: {rt:.0f}ms  frames: {resp_data.get('frames')}")

    # Encode text
    print("\n=== encode_text ===")
    rt, h, _ = post_json("/encode_text", {
        "tags": "deathstep",
        "bpm": 136, "duration": 60.0, "key": "G# minor",
    })
    print(f"  round-trip: {rt:.0f}ms")

    # Warmup generates
    print("\n=== warmup (3 runs) ===")
    for i in range(3):
        rt, h, _ = post_json("/generate", {
            "seed": 9000 + i, "denoise": 0.75, "method": "ode",
        })
        gen = h.get("X-Generate-Ms", "?")
        dec = h.get("X-Decode-Ms", "?")
        print(f"  run {i+1}: round-trip={rt:.0f}ms  server: gen={gen}ms dec={dec}ms")

    # Profile
    print("\n" + "=" * 60)
    print("  PROFILING: /generate  20 repeats, ODE denoise=0.75")
    print("=" * 60 + "\n")

    N = 20
    rts, gens, decs = [], [], []

    for i in range(N):
        rt, h, wav_bytes = post_json("/generate", {
            "seed": i, "denoise": 0.75, "method": "ode",
        })
        gen_ms = int(h.get("X-Generate-Ms", 0))
        dec_ms = int(h.get("X-Decode-Ms", 0))
        server_ms = gen_ms + dec_ms
        overhead = rt - server_ms
        rts.append(rt)
        gens.append(gen_ms)
        decs.append(dec_ms)
        if i < 5 or i == N - 1:
            print(f"  #{i+1:2d}: rt={rt:6.0f}ms  gen={gen_ms:4d}ms  dec={dec_ms:4d}ms  http_overhead={overhead:5.0f}ms  wav={len(wav_bytes)/1e6:.1f}MB")

    print(f"\n  --- Summary (N={N}) ---")

    def stat(vals, label):
        s = sorted(vals)
        print(f"  {label:20s}  mean={sum(s)/len(s):6.0f}ms  min={s[0]:6.0f}ms  p50={s[len(s)//2]:6.0f}ms")

    stat(rts, "Round-trip")
    stat(gens, "Server generate")
    stat(decs, "Server decode")
    stat([r - g - d for r, g, d in zip(rts, gens, decs)], "HTTP overhead")

    print(f"\n  WAV size: {len(wav_bytes)/1e6:.1f}MB (60s stereo 48kHz)")

    # SDE with curve
    print("\n" + "=" * 60)
    print("  PROFILING: /generate  10 repeats, SDE + sde_denoise_curve")
    print("=" * 60 + "\n")

    for i in range(10):
        rt, h, _ = post_json("/generate", {
            "seed": i, "denoise": 0.75, "method": "sde",
            "sde_denoise_curve": {"type": "ramp", "start": 0.3, "end": 1.0},
        })
        gen_ms = int(h.get("X-Generate-Ms", 0))
        dec_ms = int(h.get("X-Decode-Ms", 0))
        overhead = rt - gen_ms - dec_ms
        if i < 3 or i == 9:
            print(f"  #{i+1:2d}: rt={rt:6.0f}ms  gen={gen_ms:4d}ms  dec={dec_ms:4d}ms  http_overhead={overhead:5.0f}ms")


if __name__ == "__main__":
    main()
