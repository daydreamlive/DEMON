"""
Remote GPU backend for the realtime motion-to-music demo.

Runs the SAME PipelineRunner as :mod:`full_demo`, with:
  - VirtualMidiKnobs fed by WebSocket params from the client
  - on_audio_ready callback that sends slices back over WebSocket
  - Catalog-driven LoRA library (MODELS_DIR/loras): client toggles
    individual entries on/off via WebSocket messages instead of the
    server hardcoding which LoRAs to load.

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
from acestep.paths import (
    checkpoints_dir, loras_dir, select_trt_engines, trt_engine_path,
)

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

    def add_knob(self, name, knob_def):
        """Register a new knob after construction (used when the client
        enables a LoRA at runtime and we need a ``lora_str_<id>`` slot)."""
        with self._lock:
            if name not in self._values:
                self._values[name] = knob_def.default
            self._all_knobs[name] = knob_def

    def remove_knob(self, name):
        with self._lock:
            self._values.pop(name, None)
            self._all_knobs.pop(name, None)

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
    fast_vae = config.get("fast_vae", False)

    # LoRA selection.  ``enabled_loras`` is the new id-keyed protocol;
    # ``lora_paths`` / ``lora_path`` are interpreted as filesystem paths
    # for ad-hoc registration of LoRAs that aren't already in the
    # MODELS_DIR/loras catalog.  Both can be combined.
    #
    # ``lora_strengths`` is a dict {id: strength} — the value passed to
    # enable_trt_lora at init time.  Setting strength at enable time
    # (rather than enabling at 0 and waiting for the first per-tick
    # set_strength) is what keeps the first VAE-decode window from
    # sounding like the LoRA is missing.
    enabled_lora_ids = list(config.get("enabled_loras") or [])
    lora_strengths_init: dict[str, float] = {
        str(k): float(v) for k, v in (config.get("lora_strengths") or {}).items()
    }
    extra_lora_paths = list(
        config.get("lora_paths")
        or ([config["lora_path"]] if config.get("lora_path") else [])
    )

    # --- Session setup ---
    audio_duration_s = waveform.shape[1] / SAMPLE_RATE
    trt_engines = select_trt_engines(duration_s=audio_duration_s)
    if fast_vae:
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

    # --- LoRA library ---
    # The catalog was populated automatically by DiffusionEngine when it
    # scanned MODELS_DIR/loras at engine load.  Here we just decide which
    # subset to enable for this client and prewarm them in the background
    # so the eventual enable_lora is fast.
    engine_obj = session.handler._diffusion_engine
    lora_available = bool(engine_obj and engine_obj.trt_lora_available)
    if use_lora and not lora_available:
        print("[Server] WARNING: LoRA engine unavailable on this decoder")
        use_lora = False

    initial_enable_ids: list[str] = []
    if use_lora:
        # Resolve any explicit enable-by-id requests (these must already
        # be in the catalog from the auto-scan).
        catalog_ids = {d.id for d in engine_obj.list_trt_loras()}
        for lid in enabled_lora_ids:
            if lid in catalog_ids:
                initial_enable_ids.append(lid)
            else:
                print(f"[Server] WARNING: enabled_loras id not in catalog: {lid}")
        # Resolve ad-hoc paths: register if needed, then enable.
        for p in extra_lora_paths:
            pp = Path(p)
            if not pp.exists():
                print(f"[Server] WARNING: LoRA path missing: {p}")
                continue
            try:
                lid = engine_obj.register_trt_lora(str(pp))
                if lid not in initial_enable_ids:
                    initial_enable_ids.append(lid)
            except Exception as e:
                print(f"[Server] WARNING: failed to register {p}: {e}")
        # Kick off background materialization for everything we plan to
        # enable. Non-blocking; the eventual enable will block on the
        # future if the worker hasn't finished yet.
        for lid in initial_enable_ids:
            try:
                engine_obj.prewarm_trt_lora(lid)
            except Exception as e:
                print(f"[Server] Prewarm failed for {lid}: {e}")
        if not initial_enable_ids:
            print("[Server] No LoRAs enabled at startup (catalog-only)")

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

    audio_eng = AudioEngine(src_np, SAMPLE_RATE)

    def _catalog_payload():
        if not lora_available:
            return []
        return [
            {
                "id": d.id, "name": d.name, "path": d.path,
                "state": d.state, "strength": d.strength,
                "materialized_bytes": d.materialized_bytes,
            }
            for d in engine_obj.list_trt_loras()
        ]

    # Send ready + initial buffer
    ws.send(json.dumps({
        "type": "ready",
        "duration": len(src_np) / SAMPLE_RATE,
        "sample_rate": SAMPLE_RATE,
        "channels": n_channels,
        "lora_dir": str(loras_dir()),
        "lora_catalog": _catalog_payload(),
        "lora_pending_enable": list(initial_enable_ids),
    }))
    ws.send(src_np.astype(np.float16).tobytes())
    print(f"[Server] Sent initial buffer ({len(src_np) / SAMPLE_RATE:.1f}s)")

    # ---- Phase 2: Streaming ----

    running = [True]
    send_lock = threading.Lock()
    k1_name = "sde_amp" if use_sde else "denoise"
    initial_knob_ids = list(initial_enable_ids) if use_lora else []
    banks = build_banks(use_sde, loras=initial_knob_ids)
    virtual_knobs = VirtualMidiKnobs(banks)
    params = {"num_gens": 0, "tick_ms": 0.0, "dec_ms": 0.0}
    prompt_text = [prompt]
    sde_curve_display = [None]
    motion_val = [0.0]
    motion_lock = threading.Lock()

    client_mirror = src_np.copy()
    zctx = zstd.ZstdCompressor(level=1)

    # Cross-thread LoRA mutation rendezvous.  The recv thread enqueues
    # ids; the runner thread drains the queues in before_tick so the
    # refit (which mutates engine state) is serialized with inference.
    #
    # pending_enable items are (id, strength_or_None) tuples — strength
    # is the target the LoRA should be at when the refit fires, applied
    # in a single transition.  Enabling at 0 and ramping up via the
    # next per-tick set_strength causes the first decode window to
    # sound like the LoRA is missing (the streaming pipeline depth
    # spans several decoded seconds), so callers should always supply
    # the target strength when they have it.
    pending_enable: list[tuple[str, float | None]] = []
    pending_disable: list[str] = []
    pending_lock = threading.Lock()

    def _send_catalog_update():
        try:
            with send_lock:
                ws.send(json.dumps({
                    "type": "lora_catalog",
                    "catalog": _catalog_payload(),
                }))
        except ConnectionClosed:
            running[0] = False

    def apply_lora_pending():
        if not lora_available:
            return
        with pending_lock:
            local_disable = pending_disable[:]
            local_enable = pending_enable[:]
            pending_disable.clear()
            pending_enable.clear()
        if not local_disable and not local_enable:
            return
        for lid in local_disable:
            try:
                engine_obj.disable_trt_lora(lid)
                virtual_knobs.remove_knob(f"lora_str_{lid}")
            except Exception as e:
                print(f"[Server] disable_lora({lid}) failed: {e}")
        for lid, strength in local_enable:
            try:
                engine_obj.enable_trt_lora(lid, strength=strength)
                # Allocate a knob slot so set_lora_strength can be driven
                # by the client's params dict.  Default the slot to the
                # strength we just enabled at, so the runner's slider-
                # delta check (set_lora_strength only when the new value
                # differs by > 0.02) doesn't immediately fire a redundant
                # refit on tick 1.
                from .client.knobs import KnobDef
                virtual_knobs.add_knob(
                    f"lora_str_{lid}",
                    KnobDef(
                        cc=0,
                        default=float(strength) if strength is not None else 0.0,
                        sensitivity=2.0, max_val=2.0,
                    ),
                )
            except Exception as e:
                print(f"[Server] enable_lora({lid}) failed: {e}")
        _send_catalog_update()

    # --- on_audio_ready: delta-encode and send to client ---
    def on_audio_ready(wav_np, win_start=None, win_end=None):
        audio_eng.swap(wav_np)
        if win_start is not None:
            ss, se = win_start, min(win_end, len(wav_np))
        else:
            ss, se = 0, len(wav_np)
        if se <= ss:
            return
        region = wav_np[ss:se]
        mirror_region = client_mirror[ss:se]
        delta = (region - mirror_region).astype(np.float16)
        compressed = zctx.compress(delta.tobytes())
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

    # --- recv loop: drain client messages ---
    def recv_loop():
        while running[0]:
            latest_raw = None
            latest_pp = None
            try:
                while True:
                    msg = ws.recv(timeout=0.005)
                    if isinstance(msg, str):
                        data = json.loads(msg)
                        mtype = data.get("type")
                        if mtype == "params":
                            latest_raw = data.get("raw", {})
                            latest_pp = data.get("playback_pos", 0.0)
                        elif mtype == "prompt":
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
                        elif mtype == "enable_lora":
                            lid = data.get("id")
                            # Optional strength carries the target value
                            # the client wants the LoRA enabled at, so
                            # the engine refit lands at that strength in
                            # one shot instead of going through 0 first.
                            s = data.get("strength")
                            try:
                                strength = float(s) if s is not None else None
                            except (TypeError, ValueError):
                                strength = None
                            if lid:
                                with pending_lock:
                                    pending_enable.append((str(lid), strength))
                        elif mtype == "disable_lora":
                            lid = data.get("id")
                            if lid:
                                with pending_lock:
                                    pending_disable.append(str(lid))
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
                audio_eng.position = int(latest_pp * SAMPLE_RATE) % max(1, len(audio_eng.current))

    recv_t = threading.Thread(target=recv_loop, daemon=True)
    recv_t.start()

    # Stage the initial enable set so they get applied on the runner
    # thread before the first tick.  Each entry carries its target
    # strength (from config.lora_strengths) so the refit lands at the
    # right value in one shot — without this, the first decoded window
    # comes out as if the LoRA were missing, because the runner's
    # set_strength catch-up only kicks in after tick 1.  The prewarm
    # started at session setup is likely complete by now; any leftover
    # work is awaited synchronously inside enable_trt_lora.
    if use_lora and initial_enable_ids:
        with pending_lock:
            for lid in initial_enable_ids:
                pending_enable.append(
                    (lid, lora_strengths_init.get(lid)),
                )

    # --- PipelineRunner: the SAME code as local ---
    runner = PipelineRunner(
        session, stream, audio_eng,
        use_midi=True,  # always "MIDI" mode; VirtualMidiKnobs provides values
        use_sde=use_sde, use_lora=use_lora,
        midi_knobs=virtual_knobs,
        engine_obj=engine_obj,
        vae_window=vae_window, crop_seconds=crop_seconds,
        k1_name=k1_name, seed=1528, skip_threshold=1e-3,
        sde_curve_display=sde_curve_display, params=params,
        prompt_text=prompt_text, running=running,
        motion_val=motion_val, motion_lock=motion_lock,
        on_audio_ready=on_audio_ready,
        before_tick=apply_lora_pending,
    )

    try:
        print("[Server] Pipeline running...")
        runner.run()
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
