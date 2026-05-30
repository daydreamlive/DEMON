"""Smoke test for realtime_motion_graph_web running behind cloudflared.

Connects to the public WSS URL, uploads the default 60s confusion.wav,
waits for the server's "ready" + initial buffer + first streaming slice.

Run locally (laptop):
    uv run python scripts/deploy/smoketest_demo.py
"""
import json
import struct
import sys
import time
import wave
from pathlib import Path

import numpy as np
import soundfile as sf
from websockets.sync.client import connect

from acestep.fixtures import audio_fixture

WS_URL = "ws://localhost:8765/"
# Same fixture the live demo opens by default; pulled (and cached) from
# the daydreamlive/demon-fixtures-v2 HF dataset inside main() so importing
# this module doesn't hit the network.
DEFAULT_FIXTURE = "inside_confusion_loop_60s_gsm.wav"

CFG = {
    "sde": False,
    "lora": True,
    "depth": 4,
    "vae_window": 6.0,
    "crop": 0.0,
    "steps": 8,
    "prompt": "instrumental electronic music, energetic",
    "fast_vae": False,
    "lora_paths": [
        "demos/realtime_motion_graph_web/loras/deathsteap_1.safetensors",
        "demos/realtime_motion_graph_web/loras/daftpunkstyle1200.safetensors",
    ],
}


def load_wav_48k_stereo(p: Path):
    audio, sr = sf.read(str(p), dtype="float32", always_2d=True)
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    if audio.shape[1] > 2:
        audio = audio[:, :2]
    if sr != 48000:
        # Linear resample to 48k.  Good enough for a smoke test —
        # the browser uses an OfflineAudioContext for the same job.
        ratio = 48000 / sr
        n_out = int(audio.shape[0] * ratio)
        x_old = np.arange(audio.shape[0]) / sr
        x_new = np.arange(n_out) / 48000.0
        audio = np.stack([np.interp(x_new, x_old, audio[:, c]) for c in range(audio.shape[1])], axis=1).astype(np.float32)
    pool = 1920 * 5
    n = audio.shape[0] - (audio.shape[0] % pool)
    audio = audio[:n]
    return audio.astype(np.float32), audio.shape[0], audio.shape[1]


def main():
    audio_path = audio_fixture(DEFAULT_FIXTURE)
    audio, n_samples, n_channels = load_wav_48k_stereo(audio_path)
    print(f"[smoke] loaded {audio_path.name}: {n_samples / 48000:.1f}s x {n_channels}ch")

    print(f"[smoke] connect {WS_URL}")
    t0 = time.time()
    with connect(WS_URL, max_size=200 * 1024 * 1024, open_timeout=120) as ws:
        print(f"[smoke] connected in {time.time()-t0:.2f}s")
        ws.send(json.dumps(CFG))
        hdr = struct.pack("<II", n_channels, n_samples)
        ws.send(hdr + audio.tobytes())
        print(f"[smoke] sent config + audio ({(len(hdr)+audio.nbytes)/1e6:.1f} MB)")

        print("[smoke] waiting for ready (cold-start: model load ~10s + TRT engine load ~30s)...")
        t1 = time.time()
        msg = ws.recv(timeout=300)
        if isinstance(msg, str):
            info = json.loads(msg)
            print(f"[smoke] ready after {time.time()-t1:.1f}s: {info}")
        else:
            raise SystemExit(f"first msg should be json ready, got {type(msg)} len {len(msg)}")
        init_buf = ws.recv(timeout=60)
        if isinstance(init_buf, (bytes, bytearray)):
            print(f"[smoke] initial buffer {len(init_buf)/1e6:.2f} MB")
        else:
            raise SystemExit(f"second msg should be binary initial buffer")

        raw = {"denoise": 0.5, "seed": 42}
        ws.send(json.dumps({"type": "params", "raw": raw, "playback_pos": 0.0}))

        print("[smoke] waiting for first streaming slice (pipeline build ~30s on first tick)...")
        t2 = time.time()
        slice_count = 0
        json_count = 0
        deadline = time.time() + 180
        while time.time() < deadline and slice_count < 2:
            try:
                m = ws.recv(timeout=120)
            except Exception as e:
                print(f"[smoke] recv err: {e}")
                break
            if isinstance(m, (bytes, bytearray)):
                slice_count += 1
                elapsed = time.time() - t2
                print(f"[smoke] slice #{slice_count}: {len(m)/1024:.1f} KB after {elapsed:.1f}s")
            else:
                json_count += 1
                try:
                    j = json.loads(m)
                    t = j.get("type", "?")
                    if t == "params_update":
                        p = j.get("params", {})
                        if json_count <= 3:
                            print(f"[smoke] params_update tick={p.get('tick_ms')}ms dec={p.get('dec_ms')}ms gens={p.get('num_gens')}")
                    else:
                        print(f"[smoke] json msg type={t}: {m[:120]}")
                except Exception:
                    print(f"[smoke] json msg (unparsed): {m[:120]}")

    if slice_count == 0:
        print("[smoke] FAILED: no slices received")
        sys.exit(1)
    print(f"[smoke] PASS: got {slice_count} slice(s)")


if __name__ == "__main__":
    main()
