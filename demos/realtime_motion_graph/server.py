"""
Remote GPU backend for the realtime motion-to-music demo.

Runs the SAME PipelineRunner as :mod:`full_demo`, with:
  - VirtualMidiKnobs fed by WebSocket params from the client
  - on_audio_ready callback that sends slices back over WebSocket

Usage:
    uv run python -u -m demos.realtime_motion_graph.server
    uv run python -u -m demos.realtime_motion_graph.server --host 0.0.0.0 --port 8765
"""

import json
import struct
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
import zstandard as zstd

torch.set_grad_enabled(False)
torch._dynamo.config.disable = True

from websockets.exceptions import ConnectionClosed
from websockets.sync.server import serve

from acestep.audio.key_detection import detect_key
from acestep.constants import TASK_INSTRUCTIONS
from acestep.engine.session import Session
from acestep.nodes.types import Audio
from acestep.paths import trt_engine_path, checkpoints_dir, select_trt_engines

from .client.audio_engine import AudioEngine
from .client.knobs import build_banks, CHANNEL_GROUPS, KEYSTONE_CHANNELS
from .client.protocol import (
    SAMPLE_RATE,
    SLICE_FLAG_DELTA,
    SLICE_HDR_FMT,
    SLICE_HDR_SIZE,
    T,
)
from .pipeline import PipelineRunner

# Default LoRA lives next to this module so it syncs with the repo /
# rsync / container layer instead of depending on an absolute user
# path.  Drop any .safetensors file in demos/realtime_motion_graph/
# assets/loras/ and point LORA_PATH at it to swap the default.
LORA_PATHS = [
    str(Path(__file__).parent / "assets" / "loras" / "deathsteap_1.safetensors"),
]


# ---------------------------------------------------------------------------
# Virtual MIDI knobs (same interface as MidiKnobs)
# ---------------------------------------------------------------------------

class VirtualMidiKnobs:
    """Drop-in replacement for MidiKnobs.  Values come from the WebSocket
    client instead of a physical MIDI controller."""

    def __init__(self, banks):
        self._banks = banks
        self._active_bank = 0
        self._values = {}
        self._all_knobs = {}
        for bank in banks:
            for name, k in bank.knobs.items():
                if name not in self._values:
                    self._values[name] = k.default
                self._all_knobs[name] = k
        self._lock = threading.Lock()

    def update(self, raw: dict):
        """Bulk-update values from a client raw dict."""
        with self._lock:
            self._values.update(raw)

    def get(self, name: str) -> float:
        with self._lock:
            return self._values.get(name, 0.0)

    def get_all(self) -> dict:
        with self._lock:
            bank = self._banks[self._active_bank]
            return {name: self._values[name] for name in bank.knobs}

    def get_all_values(self) -> dict:
        with self._lock:
            return dict(self._values)

    def get_param(self, name: str) -> float:
        with self._lock:
            return self._values.get(name, 0.0)

    def all_knob_defs(self) -> dict:
        return dict(self._all_knobs)

    @property
    def active_bank_index(self) -> int:
        return self._active_bank

    @property
    def active_bank(self):
        return self._banks[self._active_bank]

    def release(self):
        pass


# ---------------------------------------------------------------------------
# WebSocket handler
# ---------------------------------------------------------------------------

def handle_client(ws):
    print("[Server] Client connected")

    # ---- Phase 1: Init ----
    config = json.loads(ws.recv())
    print(f"[Server] Config: {config}")

    audio_bytes = ws.recv()
    channels, num_samples = struct.unpack("<II", audio_bytes[:8])
    audio_np = np.frombuffer(audio_bytes[8:], dtype=np.float32).reshape(
        num_samples, channels,
    )
    waveform = torch.from_numpy(audio_np.T.copy())
    waveform = waveform[:2, :int(60.0 * SAMPLE_RATE)]
    pool = 1920 * 5
    rem = waveform.shape[-1] % pool
    if rem:
        waveform = waveform[:, :waveform.shape[-1] - rem]
    print(f"[Server] Audio: {waveform.shape[1] / SAMPLE_RATE:.1f}s, {waveform.shape[0]}ch")

    use_sde = config.get("sde", False)
    use_lora = config.get("lora", False)
    vae_window = config.get("vae_window", 3.0)
    crop_seconds = config.get("crop", 0.0)
    depth = config.get("depth", 4)
    steps = config.get("steps", 8)
    prompt = config.get("prompt", "instrumental music")
    lora_paths = config.get("lora_paths") or (
        [config["lora_path"]] if config.get("lora_path") else None
    )
    fast_vae = config.get("fast_vae", False)

    # --- Session setup ---
    # Audio is clamped to 60s above, so the 60s engine set is what we need.
    # select_trt_engines() picks 60s by default; pass duration_s explicitly
    # to make the dependency on the upstream clamp visible.
    audio_duration_s = waveform.shape[1] / SAMPLE_RATE
    trt_engines = select_trt_engines(duration_s=audio_duration_s)
    if fast_vae:
        # fast_vae uses the dreamvae distilled decoder; profile must match.
        fast_name = "dreamvae_decode_fp16_60s" if audio_duration_s <= 60.0 else "dreamvae_decode_fp16_240s"
        if Path(str(trt_engine_path(fast_name))).exists():
            trt_engines["vae_decode"] = str(trt_engine_path(fast_name))
        else:
            print(f"[Server] WARNING: {fast_name} engine missing, falling back to {Path(trt_engines['vae_decode']).stem}")
            fast_vae = False

    print("[Server] Loading model...")
    t0 = time.time()
    session = Session(
        project_root=str(checkpoints_dir()),
        decoder_backend="tensorrt",
        vae_backend="tensorrt",
        trt_engines=trt_engines,
        vae_window=vae_window,
    )
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    lora_ids = []
    engine_obj = None
    if use_lora:
        if not lora_paths:
            lora_paths = list(LORA_PATHS)
        engine_obj = session.handler._diffusion_engine
        if engine_obj and engine_obj.trt_lora_available:
            for lp in lora_paths:
                if not Path(lp).exists():
                    print(f"[Server] WARNING: LoRA path missing: {lp}")
                    continue
                print(f"[Server] Applying LoRA: {Path(lp).name}")
                lid = engine_obj.apply_trt_lora(lp, strength=0.0)
                lora_ids.append(lid)
            if not lora_ids:
                print("[Server] WARNING: no valid LoRA files found")
                use_lora = False
        else:
            print("[Server] WARNING: LoRA engine unavailable on this decoder")
            use_lora = False

    audio_in = Audio(waveform=waveform, sample_rate=SAMPLE_RATE)

    print("[Server] Detecting BPM + key...")
    import librosa
    mono_np = waveform.mean(dim=0).numpy()
    detected_bpm, _ = librosa.beat.beat_track(y=mono_np, sr=SAMPLE_RATE)
    detected_bpm = int(round(float(np.asarray(detected_bpm).flat[0])))
    detected_key = detect_key(mono_np, SAMPLE_RATE)
    print(f"  BPM: {detected_bpm}  Key: {detected_key}")

    print("[Server] Preparing source...")
    source = session.prepare_source(audio_in)

    print("[Server] Text encode...")
    conditioning = session.encode_text(
        tags=prompt,
        instruction=TASK_INSTRUCTIONS["cover"],
        refer_latent=source.latent,
        bpm=detected_bpm, duration=60.0, key=detected_key,
    )

    print("[Server] Creating stream...")
    stream = session.stream(
        source=source,
        conditioning=conditioning,
        steps=steps,
        shift=3.0,
        pipeline_depth=depth,
    )
    print("[Server] Stream handle ready (pipeline built on first tick)")

    # Initial buffer
    src_np = waveform.numpy().T
    if crop_seconds > 0:
        src_np = src_np[:int(crop_seconds * SAMPLE_RATE)]
    n_channels = src_np.shape[1] if src_np.ndim > 1 else 1

    # Pre-blend the buffer's tail with its head so the very first loop
    # iteration is smooth before any model patches arrive. The client
    # worklet also crossfades at playback time, so this is belt-and-
    # suspenders for the initial source audio.
    _seam_fade_samples = int(0.05 * SAMPLE_RATE)
    _seam_fade_samples = min(_seam_fade_samples, len(src_np) // 4)
    if _seam_fade_samples > 0:
        if src_np.ndim == 1:
            _fade_out = np.linspace(1.0, 0.0, _seam_fade_samples).astype(src_np.dtype)
            _fade_in = np.linspace(0.0, 1.0, _seam_fade_samples).astype(src_np.dtype)
        else:
            _fade_out = np.linspace(1.0, 0.0, _seam_fade_samples).reshape(-1, 1).astype(src_np.dtype)
            _fade_in = np.linspace(0.0, 1.0, _seam_fade_samples).reshape(-1, 1).astype(src_np.dtype)
        _tail = src_np[-_seam_fade_samples:].copy()
        _head = src_np[:_seam_fade_samples].copy()
        src_np[-_seam_fade_samples:] = _tail * _fade_out + _head * _fade_in

    # AudioEngine on server (PipelineRunner reads audio_eng.position)
    audio_eng = AudioEngine(src_np, SAMPLE_RATE)
    # Don't start playback (no speakers on server), but we need position tracking.
    # We'll update position from the client's playback_pos.

    # Send ready + initial buffer
    ws.send(json.dumps({
        "type": "ready",
        "duration": len(src_np) / SAMPLE_RATE,
        "sample_rate": SAMPLE_RATE,
        "channels": n_channels,
        "lora_count": len(lora_ids),
    }))
    ws.send(src_np.astype(np.float16).tobytes())
    print(f"[Server] Sent initial buffer ({len(src_np) / SAMPLE_RATE:.1f}s)")

    # ---- Phase 2: Streaming ----

    running = [True]
    send_lock = threading.Lock()
    k1_name = "sde_amp" if use_sde else "denoise"
    banks = build_banks(use_sde, lora=len(lora_ids) if use_lora else 0)
    virtual_knobs = VirtualMidiKnobs(banks)
    params = {"num_gens": 0, "tick_ms": 0.0, "dec_ms": 0.0}
    prompt_text = [prompt]
    sde_curve_display = [None]
    motion_val = [0.0]
    motion_lock = threading.Lock()

    # Client mirror: tracks what audio the client currently has
    client_mirror = src_np.copy()
    zctx = zstd.ZstdCompressor(level=1)

    # --- on_audio_ready: delta-encode and send to client ---
    def on_audio_ready(wav_np, win_start=None, win_end=None):
        """Called by PipelineRunner when audio is decoded.
        Compute delta against client mirror, compress, send."""
        # Accumulate into server buffer (same as local swap)
        audio_eng.swap(wav_np)

        if win_start is not None:
            ss, se = win_start, min(win_end, len(wav_np))
        else:
            ss, se = 0, len(wav_np)

        if se <= ss:
            return

        # Delta = what server has now minus what client has
        region = wav_np[ss:se]
        mirror_region = client_mirror[ss:se]
        delta = (region - mirror_region).astype(np.float16)
        compressed = zctx.compress(delta.tobytes())

        # Update mirror to match what client will have after applying delta
        client_mirror[ss:se] = region

        hdr = struct.pack(
            SLICE_HDR_FMT,
            SLICE_FLAG_DELTA,
            ss, se - ss, n_channels,
            params.get("tick_ms", 0), params.get("dec_ms", 0),
            params.get("num_gens", 0),
        )
        try:
            with send_lock:
                ws.send(hdr + compressed)
                ws.send(json.dumps({"type": "params_update", "params": dict(params)}))
        except ConnectionClosed:
            running[0] = False

    # --- recv loop: drain client messages, update virtual knobs ---
    def recv_loop():
        while running[0]:
            latest_raw = None
            latest_pp = None
            try:
                while True:
                    msg = ws.recv(timeout=0.005)
                    if isinstance(msg, str):
                        data = json.loads(msg)
                        if data.get("type") == "params":
                            latest_raw = data.get("raw", {})
                            latest_pp = data.get("playback_pos", 0.0)
                        elif data.get("type") == "prompt":
                            # Re-encode on server
                            cond = session.encode_text(
                                tags=data["tags"],
                                instruction=TASK_INSTRUCTIONS["cover"],
                                refer_latent=source.latent,
                                bpm=detected_bpm, duration=60.0,
                                key=detected_key,
                            )
                            stream.conditioning = cond
                            prompt_text[0] = data["tags"]
                            try:
                                with send_lock:
                                    ws.send(json.dumps({
                                        "type": "prompt_applied",
                                        "tags": data["tags"],
                                    }))
                            except ConnectionClosed:
                                running[0] = False
                                break
            except TimeoutError:
                pass
            except ConnectionClosed:
                running[0] = False
                break
            except Exception as exc:
                print(f"[Server] Recv error: {exc}")
                running[0] = False
                break

            if latest_raw is not None:
                virtual_knobs.update(latest_raw)
            if latest_pp is not None:
                # Sync server's audio_eng position to client's playback
                audio_eng.position = int(latest_pp * SAMPLE_RATE) % max(1, len(audio_eng.current))

    recv_t = threading.Thread(target=recv_loop, daemon=True)
    recv_t.start()

    # --- PipelineRunner: the SAME code as local ---
    runner = PipelineRunner(
        session, stream, audio_eng,
        use_midi=True,  # always "MIDI" mode; VirtualMidiKnobs provides values
        use_sde=use_sde, use_lora=use_lora,
        midi_knobs=virtual_knobs, lora_ids=lora_ids,
        engine_obj=engine_obj,
        vae_window=vae_window, crop_seconds=crop_seconds,
        k1_name=k1_name, seed=1528, skip_threshold=1e-3,
        sde_curve_display=sde_curve_display, params=params,
        prompt_text=prompt_text, running=running,
        motion_val=motion_val, motion_lock=motion_lock,
        on_audio_ready=on_audio_ready,
    )

    try:
        print("[Server] Pipeline running...")
        runner.run()  # blocks until running[0] = False
    except Exception as exc:
        print(f"[Server] Pipeline error: {exc}")
        import traceback
        traceback.print_exc()
    finally:
        running[0] = False
        recv_t.join(timeout=2)
        print(f"[Server] Client disconnected ({params.get('num_gens', 0)} generations)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    host = "0.0.0.0"
    port = 8765
    args = sys.argv[1:]
    if "--host" in args:
        idx = args.index("--host")
        host = args[idx + 1]
    if "--port" in args:
        idx = args.index("--port")
        port = int(args[idx + 1])

    print(f"[Server] Starting on ws://{host}:{port}")
    srv = serve(
        handle_client,
        host,
        port,
        max_size=50 * 1024 * 1024,
    )
    srv_thread = threading.Thread(target=srv.serve_forever, daemon=True)
    srv_thread.start()
    print(f"[Server] Listening... (Ctrl+C to stop)")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[Server] Shutting down...")
        import os
        os._exit(0)


if __name__ == "__main__":
    main()
