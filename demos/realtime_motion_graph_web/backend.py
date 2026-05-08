"""
GPU backend for the realtime motion-to-music demo.

Provides :func:`handle_client`, the per-WebSocket coroutine wired in by
:mod:`.server`. Drives a :class:`~acestep.engine.session.StreamHandle`
through :class:`.pipeline.PipelineRunner`, with:
  - VirtualMidiKnobs fed by WebSocket params from the client
  - on_audio_ready callback that sends slices back over WebSocket
  - Catalog-driven LoRA library (MODELS_DIR/loras): client toggles
    individual entries on/off via WebSocket messages instead of the
    server hardcoding which LoRAs to load.
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

from acestep.audio.key_detection import detect_key
from acestep.constants import TASK_INSTRUCTIONS
from acestep.engine.session import PreparedSource, Session
from acestep.fixtures import FixtureSidecar, fixture_sidecar
from acestep.nodes.types import Audio, Latent
from acestep.paths import (
    EngineNotBuiltError,
    available_dreamvae_decode_engine,
    available_trt_engines,
    checkpoints_dir,
    dreamvae_decode_engine_name,
    loras_dir,
    max_profile_duration_s,
    smallest_fitting_profile_duration_s,
    trt_engine_path,
)

from .audio_engine import AudioEngine
from .knobs import build_banks, CHANNEL_GROUPS, KEYSTONE_CHANNELS
from .protocol import (
    SAMPLE_RATE,
    SLICE_FLAG_DELTA,
    SLICE_HDR_FMT,
    SLICE_HDR_SIZE,
    T,
)
from .pipeline import PipelineRunner


# ---------------------------------------------------------------------------
# Source / conditioning resolver (sidecar-aware)
# ---------------------------------------------------------------------------

def _try_load_sidecar(
    fixture_name: str | None, *, checkpoint: str, samples: int,
) -> FixtureSidecar | None:
    """Look up a fixture sidecar; return None on miss / mismatch.

    Length check guards against runtime truncation that disagrees with
    what the sidecar was precomputed for (e.g. a smaller TRT profile
    cap kicking in). The caller falls back to live computation in that
    case so cached tensor shapes can't poison the streaming pipeline.
    """
    if not fixture_name:
        return None
    try:
        sc = fixture_sidecar(fixture_name, checkpoint=checkpoint)
    except Exception as e:
        print(f"[Server] sidecar lookup failed for {fixture_name!r}: {e}")
        return None
    if sc is None:
        return None
    if sc.samples != samples:
        print(
            f"[Server] sidecar length mismatch for {fixture_name}: "
            f"sidecar samples={sc.samples} vs runtime samples={samples}; "
            f"ignoring sidecar"
        )
        return None
    return sc


def _resolve_bpm_key_source(
    session: Session,
    *,
    audio_in: Audio,
    fixture_name: str | None,
    samples: int,
    checkpoint: str,
    key_override: str | None = None,
) -> tuple[PreparedSource, int, str]:
    """Resolve (source, bpm, key) for a (fixture, audio) pair.

    For known fixtures with a sidecar present (JSON+safetensors in the
    dataset or local staging dir, matching checkpoint and audio length),
    returns the cached source latent + context_latent and reads BPM /
    key from the sidecar JSON. Skips librosa beat tracking, CNN key
    detection, and ``Session.prepare_source`` — the prompt-independent
    half of the per-connect work.

    Conditioning is *not* cached (see fixtures.py). Callers run
    ``Session.encode_text`` against ``source.latent`` themselves; with
    the source latent already on GPU this is ~60ms warm.

    Falls through to live librosa + detect_key + prepare_source when:
      - ``fixture_name`` is None / unknown
      - sidecar files aren't in the dataset yet
      - checkpoint mismatch (tensors are tied to a specific build)
      - audio-length truncation mismatch (e.g. operator's TRT profile
        cap is smaller than the natural fixture length)

    ``key_override`` is the swap-source operator override; if supplied,
    it wins over both sidecar.key and detect_key.
    """
    sc = _try_load_sidecar(fixture_name, checkpoint=checkpoint, samples=samples)

    if sc is not None:
        device = session.handler.device
        dtype = session.handler.dtype
        source = PreparedSource(
            latent=Latent(tensor=sc.latent.to(device, dtype).contiguous()),
            context_latent=Latent(tensor=sc.context_latent.to(device, dtype).contiguous()),
        )
        bpm = sc.bpm
        key = key_override or sc.key
        print(f"[Server] sidecar hit ({fixture_name}): bpm={bpm} key={key!r}")
        return source, bpm, key

    # Live path: librosa BPM, CNN key detection, full prepare_source.
    import librosa
    print("[Server] Detecting BPM + key...")
    mono_np = audio_in.waveform.mean(dim=0).numpy()
    bpm_raw, _ = librosa.beat.beat_track(y=mono_np, sr=SAMPLE_RATE)
    bpm = int(round(float(np.asarray(bpm_raw).flat[0])))
    key = key_override or detect_key(mono_np, SAMPLE_RATE)
    print(f"  BPM: {bpm}  Key: {key}")

    print("[Server] Preparing source...")
    source = session.prepare_source(audio_in)
    return source, bpm, key


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

def handle_client(
    ws,
    *,
    decoder_backend: str = "tensorrt",
    vae_backend: str = "tensorrt",
    checkpoint: str = "acestep-v15-turbo",
):
    print(
        f"[Server] Client connected "
        f"(decoder={decoder_backend}, vae={vae_backend}, ckpt={checkpoint})"
    )

    # ---- Phase 1: Init ----
    config = json.loads(ws.recv())
    print(f"[Server] Config: {config}")

    audio_bytes = ws.recv()
    channels, num_samples = struct.unpack("<II", audio_bytes[:8])
    audio_np = np.frombuffer(audio_bytes[8:], dtype=np.float32).reshape(
        num_samples, channels,
    )
    waveform = torch.from_numpy(audio_np.T.copy())
    # Cap at the largest registered TRT engine profile rather than
    # hardcoding 60 s. Anything longer than the largest profile can't
    # be handled by any built engine, but we let the operator stretch
    # all the way up to that ceiling — picking the smallest-fitting
    # engine happens below in available_trt_engines().
    max_seconds = max_profile_duration_s()
    waveform = waveform[:2, :int(max_seconds * SAMPLE_RATE)]
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
    # Optional fixture-name hint enables sidecar lookup (precomputed BPM,
    # key, source latent, conditioning). Absent / unknown name -> fully
    # live path; same behavior as before sidecars existed.
    fixture_name = config.get("fixture_name")

    # LoRA selection.  ``enabled_loras`` is the new id-keyed protocol;
    # ``lora_paths`` / ``lora_path`` are interpreted as filesystem paths
    # for ad-hoc registration of LoRAs that aren't already in the
    # MODELS_DIR/loras catalog.  Both can be combined.
    #
    # ``lora_strengths`` is a dict {id: strength} — the value passed to
    # enable_lora at init time.  Setting strength at enable time
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
    use_trt = decoder_backend == "tensorrt" or vae_backend == "tensorrt"
    if use_trt:
        # Tell the picker exactly which engines this session will use, so a
        # mixed-backend setup (e.g. tensorrt decoder + eager VAE) doesn't
        # disqualify an otherwise-usable profile because of missing VAE
        # engines we wouldn't load anyway.
        needs: list[str] = []
        if decoder_backend == "tensorrt":
            needs.append("decoder")
        if vae_backend == "tensorrt":
            needs.extend(["vae_encode", "vae_decode"])
        try:
            trt_engines, picked_dur = available_trt_engines(
                duration_s=audio_duration_s, needs=tuple(needs),
            )
        except EngineNotBuiltError as exc:
            # Surface to the operator (server log) AND the client UI.
            # WebSocket close reason is capped at 123 bytes by the
            # protocol, so the build command goes in a JSON message
            # first and the close reason carries a short summary.
            print(f"[Server] {exc}")
            try:
                ws.send(json.dumps({
                    "type": "error",
                    "code": "engine_not_built",
                    "message": str(exc),
                    "build_command": exc.build_command,
                    "duration_s": exc.duration_s,
                }))
            except Exception:
                pass
            ws.close(1011, "TRT engine not built")
            return
        # Only warn when a *smaller* registered profile would have fit
        # but wasn't built (so we genuinely fell back). For a 119.8 s
        # source the 120 s engine is the smallest fitting profile, not
        # a fallback — the previous predicate fired on that case.
        ideal_dur = smallest_fitting_profile_duration_s(audio_duration_s)
        if picked_dur > ideal_dur:
            print(
                f"[Server] WARNING: using {picked_dur:.0f}s engine for "
                f"{audio_duration_s:.1f}s audio (fallback; {ideal_dur:.0f}s "
                f"profile not built — extra VRAM cost)"
            )
        # Prune unused keys for the same reason as before:
        # validate_backends() rejects engine entries whose backend isn't
        # tensorrt.
        if decoder_backend != "tensorrt":
            trt_engines.pop("decoder", None)
        if vae_backend != "tensorrt":
            trt_engines.pop("vae_encode", None)
            trt_engines.pop("vae_decode", None)
    else:
        trt_engines = None
    if fast_vae and vae_backend == "tensorrt":
        # fast_vae uses the dreamvae distilled decoder; profile must match
        # the same duration we picked above. dreamvae engines aren't in
        # _TRT_ENGINE_PROFILES (different decoder weights), so we look
        # them up via the dedicated helper which knows the naming
        # convention and falls back to a larger fitting profile when the
        # exact one isn't built (same logic as available_trt_engines).
        dv_path = available_dreamvae_decode_engine(picked_dur)
        if dv_path is not None:
            trt_engines["vae_decode"] = str(dv_path)
        else:
            wanted = dreamvae_decode_engine_name(int(picked_dur))
            print(f"[Server] WARNING: {wanted} engine missing, falling back to {Path(trt_engines['vae_decode']).stem}")
            fast_vae = False
    elif fast_vae:
        print(f"[Server] WARNING: fast_vae requires vae_backend=tensorrt; ignoring with vae_backend={vae_backend}")
        fast_vae = False

    print(f"[Server] Loading model... (decoder={decoder_backend}, vae={vae_backend}, ckpt={checkpoint})")
    t0 = time.time()
    session = Session(
        project_root=str(checkpoints_dir()),
        config_path=checkpoint,
        decoder_backend=decoder_backend,
        vae_backend=vae_backend,
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
    lora_available = bool(engine_obj and engine_obj.lora_available)
    if use_lora and not lora_available:
        print("[Server] WARNING: LoRA engine unavailable on this decoder")
        use_lora = False

    initial_enable_ids: list[str] = []
    if use_lora:
        # Resolve any explicit enable-by-id requests (these must already
        # be in the catalog from the auto-scan).
        catalog_ids = {d.id for d in engine_obj.list_loras()}
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
                lid = engine_obj.register_lora(str(pp))
                if lid not in initial_enable_ids:
                    initial_enable_ids.append(lid)
            except Exception as e:
                print(f"[Server] WARNING: failed to register {p}: {e}")
        # Kick off background materialization for everything we plan to
        # enable. Non-blocking; the eventual enable will block on the
        # future if the worker hasn't finished yet.
        for lid in initial_enable_ids:
            try:
                engine_obj.prewarm_lora(lid)
            except Exception as e:
                print(f"[Server] Prewarm failed for {lid}: {e}")
        if not initial_enable_ids:
            print("[Server] No LoRAs enabled at startup (catalog-only)")

    audio_in = Audio(waveform=waveform, sample_rate=SAMPLE_RATE)

    source, detected_bpm, detected_key = _resolve_bpm_key_source(
        session,
        audio_in=audio_in,
        fixture_name=fixture_name,
        samples=int(waveform.shape[1]),
        checkpoint=checkpoint,
    )

    print("[Server] Text encode...")
    conditioning = session.encode_text(
        tags=prompt,
        instruction=TASK_INSTRUCTIONS["cover"],
        refer_latent=source.latent,
        bpm=detected_bpm, duration=audio_duration_s, key=detected_key,
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
            for d in engine_obj.list_loras()
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
        "bpm": detected_bpm,
        "key": detected_key,
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

    # Mutable refs so the swap path can replace these in place from the
    # runner thread without invalidating closures captured by recv_loop /
    # on_audio_ready. Values are read via the [0] indirection everywhere
    # that needs the *current* (post-swap) version.
    source_ref = [source]
    bpm_ref = [detected_bpm]
    key_ref = [detected_key]
    duration_ref = [audio_duration_s]
    n_channels_ref = [n_channels]

    # Client mirror: tracks what audio the client currently has. Replaced
    # wholesale on swap so deltas continue to be computed against the
    # buffer the client just crossfaded into.
    client_mirror_ref = [src_np.copy()]
    zctx = zstd.ZstdCompressor(level=1)

    # Source-swap rendezvous between the recv thread (sets pending) and
    # the runner thread (consumes pending in before_tick). The recv loop
    # only stages audio bytes here; all GPU work happens on the runner
    # thread so we don't race the streaming pipeline.
    swap_pending: dict = {"bytes": None, "tags": None}
    swap_lock = threading.Lock()

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
                engine_obj.disable_lora(lid)
                virtual_knobs.remove_knob(f"lora_str_{lid}")
            except Exception as e:
                print(f"[Server] disable_lora({lid}) failed: {e}")
        for lid, strength in local_enable:
            try:
                engine_obj.enable_lora(lid, strength=strength)
                # Allocate a knob slot so set_lora_strength can be driven
                # by the client's params dict.  Default the slot to the
                # strength we just enabled at, so the runner's slider-
                # delta check (set_lora_strength only when the new value
                # differs by > 0.02) doesn't immediately fire a redundant
                # refit on tick 1.
                from .knobs import KnobDef
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

        client_mirror = client_mirror_ref[0]
        # If the runner emitted a slice that's longer than the freshly
        # swapped mirror (different source length), clip to the smaller
        # of the two; the client has no addressable space past mirror.
        se = min(se, len(client_mirror), len(wav_np))
        if se <= ss:
            return

        # Delta = what server has now minus what client has
        region = wav_np[ss:se]
        mirror_region = client_mirror[ss:se]
        delta = (region - mirror_region).astype(np.float16)
        compressed = zctx.compress(delta.tobytes())
        client_mirror[ss:se] = region
        hdr = struct.pack(
            SLICE_HDR_FMT,
            SLICE_FLAG_DELTA,
            ss, se - ss, n_channels_ref[0],
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
                            # Re-encode on server. Prefer the key sent by the
                            # client (operator override); fall back to the
                            # auto-detected key from the loaded source.
                            cond = session.encode_text(
                                tags=data["tags"],
                                instruction=TASK_INSTRUCTIONS["cover"],
                                refer_latent=source_ref[0].latent,
                                bpm=bpm_ref[0], duration=duration_ref[0],
                                key=data.get("key") or key_ref[0],
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
                        elif mtype == "swap_source":
                            # Followed by a binary audio frame in the same
                            # format as the init handshake. Block on that
                            # next message so we don't have to multiplex
                            # halfway through; the client only sends one
                            # swap_source at a time.
                            tags = data.get("tags") or prompt_text[0]
                            try:
                                audio_msg = ws.recv()
                            except ConnectionClosed:
                                running[0] = False
                                break
                            with swap_lock:
                                swap_pending["bytes"] = audio_msg
                                swap_pending["tags"] = tags
                                swap_pending["key"] = data.get("key")
                                swap_pending["fixture_name"] = data.get("fixture_name")
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
    # work is awaited synchronously inside enable_lora.
    if use_lora and initial_enable_ids:
        with pending_lock:
            for lid in initial_enable_ids:
                pending_enable.append(
                    (lid, lora_strengths_init.get(lid)),
                )

    # --- Source swap (runs on the runner thread via before_tick) ---
    # Forward decl so the closure can call runner._rebuild_silence_latent
    # after replacing stream.source. ``runner`` is assigned just below.
    runner_holder: list = [None]

    def apply_swap_if_pending():
        with swap_lock:
            audio_msg = swap_pending.get("bytes")
            tags = swap_pending.get("tags")
            requested_key = swap_pending.get("key")
            new_fixture_name = swap_pending.get("fixture_name")
            if audio_msg is None:
                return
            swap_pending["bytes"] = None
            swap_pending["tags"] = None
            swap_pending["key"] = None
            swap_pending["fixture_name"] = None
        try:
            new_channels, new_samples = struct.unpack("<II", audio_msg[:8])
            new_np = np.frombuffer(audio_msg[8:], dtype=np.float32).reshape(
                new_samples, new_channels,
            )
            new_wf = torch.from_numpy(new_np.T.copy())
            # Cap at the same ceiling the initial upload used so swaps
            # take advantage of every built engine profile, not a stale
            # 60 s default.
            new_wf = new_wf[:2, :int(max_seconds * SAMPLE_RATE)]
            rem = new_wf.shape[-1] % pool
            if rem:
                new_wf = new_wf[:, :new_wf.shape[-1] - rem]
            new_audio_duration_s = new_wf.shape[1] / SAMPLE_RATE
            print(
                f"[Server] Swapping source ({new_audio_duration_s:.1f}s, "
                f"{new_wf.shape[0]}ch)..."
            )

            new_audio_in = Audio(waveform=new_wf, sample_rate=SAMPLE_RATE)
            new_source, new_bpm, new_key = _resolve_bpm_key_source(
                session,
                audio_in=new_audio_in,
                fixture_name=new_fixture_name,
                samples=int(new_wf.shape[1]),
                checkpoint=checkpoint,
                key_override=requested_key,
            )
            new_cond = session.encode_text(
                tags=tags,
                instruction=TASK_INSTRUCTIONS["cover"],
                refer_latent=new_source.latent,
                bpm=new_bpm, duration=new_audio_duration_s, key=new_key,
            )

            stream.source = new_source
            stream.conditioning = new_cond
            stream.context_latent = new_source.context_latent
            source_ref[0] = new_source
            bpm_ref[0] = new_bpm
            key_ref[0] = new_key
            duration_ref[0] = new_audio_duration_s
            prompt_text[0] = tags
            r = runner_holder[0]
            if r is not None:
                # Source latent length may have changed; rebuild silence so
                # _update_hint_strength's blend operands match shapes.
                r._rebuild_silence_latent()

            new_src_np = new_wf.numpy().T
            new_n_channels = new_src_np.shape[1] if new_src_np.ndim > 1 else 1
            n_channels_ref[0] = new_n_channels
            client_mirror_ref[0] = new_src_np.copy()
            audio_eng.swap(new_src_np)
            audio_eng.position = 0

            # Notify the client. swap_ready is the streaming-phase analog
            # of "ready"; the binary follow-up is the new initial buffer
            # that the audio worklet swaps in with a 50ms crossfade.
            with send_lock:
                ws.send(json.dumps({
                    "type": "swap_ready",
                    "duration": len(new_src_np) / SAMPLE_RATE,
                    "sample_rate": SAMPLE_RATE,
                    "channels": new_n_channels,
                    "bpm": new_bpm,
                    "key": new_key,
                }))
                ws.send(new_src_np.astype(np.float16).tobytes())
            print(f"[Server] Source swap complete ({len(new_src_np) / SAMPLE_RATE:.1f}s)")
        except ConnectionClosed:
            running[0] = False
        except Exception as exc:
            print(f"[Server] Swap error: {exc}")
            import traceback
            traceback.print_exc()
            try:
                with send_lock:
                    ws.send(json.dumps({
                        "type": "swap_failed",
                        "error": str(exc),
                    }))
            except Exception:
                pass

    # Combined before_tick callback.  Both kinds of cross-thread
    # mutation (LoRA enable/disable refits and source swaps) are GPU-
    # bound and must run on the runner thread between ticks.  Drain
    # both queues each iteration so they share one rendezvous point.
    def apply_pending():
        apply_lora_pending()
        apply_swap_if_pending()

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
        before_tick=apply_pending,
    )
    runner_holder[0] = runner

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
