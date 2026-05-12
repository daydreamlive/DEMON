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
import socket
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
from acestep.constants import TASK_INSTRUCTIONS, VALID_TIME_SIGNATURES
from acestep.engine.session import PreparedSource, Session
from acestep.engine.trt.profile_manager import TRTProfileManager
from acestep.fixtures import (
    FixtureSidecar, KNOWN_FIXTURES, audio_fixture, fixture_sidecar,
)
from acestep.nodes.types import Audio, Latent
from acestep.paths import (
    EngineNotBuiltError,
    available_dreamvae_decode_engine,
    available_trt_engines,
    checkpoints_dir,
    dreamvae_decode_engine_name,
    lora_trigger,
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


def _decode_audio_msg(audio_msg: bytes) -> torch.Tensor:
    """Parse a binary audio frame into a [≤2, N] float32 tensor.

    Wire format (shared by the init handshake, ``swap_source``,
    ``set_timbre_source``, ``set_structure_source``): little-endian
    ``<II`` header carrying (channels, samples), followed by interleaved
    float32 PCM. Returns the waveform clipped to stereo (the model only
    consumes 2 channels).
    """
    ch, n = struct.unpack("<II", audio_msg[:8])
    arr = np.frombuffer(audio_msg[8:], dtype=np.float32).reshape(n, ch)
    return torch.from_numpy(arr.T.copy())[:2]


_VALID_TIME_SIG_STRS = frozenset(str(s) for s in VALID_TIME_SIGNATURES)


def _normalize_time_signature(value: object) -> str | None:
    """Coerce a wire-side time-signature value to one of
    ``VALID_TIME_SIGNATURES`` as a string. Returns ``None`` for
    unrecognized input (caller falls back to the sidecar / default
    instead of poisoning the encoder with junk)."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        s = str(int(value))
        return s if s in _VALID_TIME_SIG_STRS else None
    if isinstance(value, str):
        s = value.strip()
        return s if s in _VALID_TIME_SIG_STRS else None
    return None


def _resolve_bpm_key_source(
    session: Session,
    *,
    audio_in: Audio,
    fixture_name: str | None,
    samples: int,
    checkpoint: str,
    key_override: str | None = None,
    time_signature_override: str | None = None,
) -> tuple[PreparedSource, int, str, str]:
    """Resolve (source, bpm, key, time_signature) for a (fixture, audio) pair.

    For known fixtures with a sidecar present (JSON+safetensors in the
    dataset or local staging dir, matching checkpoint and audio length),
    returns the cached source latent + context_latent and reads BPM,
    key, and time_signature from the sidecar JSON. Skips librosa beat
    tracking, CNN key detection, and ``Session.prepare_source`` — the
    prompt-independent half of the per-connect work.

    Conditioning is *not* cached (see fixtures.py). Callers run
    ``Session.encode_text`` against ``source.latent`` themselves; with
    the source latent already on GPU this is ~60ms warm.

    Falls through to live librosa + detect_key + prepare_source when:
      - ``fixture_name`` is None / unknown
      - sidecar files aren't in the dataset yet
      - checkpoint mismatch (tensors are tied to a specific build)
      - audio-length truncation mismatch (e.g. operator's TRT profile
        cap is smaller than the natural fixture length)

    ``key_override`` and ``time_signature_override`` are the operator's
    manual choices coming from the swap_source path. They are **only**
    consulted on the live path: when a sidecar hits, the sidecar's
    recorded values are authoritative for the test fixture (a previous
    fixture's dropdown value or any other client-side staleness must
    not be allowed to mask the fixture's recorded ground truth). After
    the swap, post-hoc dropdown edits flow through ``mtype == "prompt"``
    instead, where overrides do apply.
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
        # Sidecar is the source of truth for known fixtures; do NOT
        # apply key_override here. (Earlier this read
        # `key = key_override or sc.key`, which let the previous
        # fixture's dropdown value, sent on swap_source, beat the new
        # fixture's recorded key — e.g. a swap from low_fi (G minor)
        # to prog_rock (E minor) printed
        # `sidecar hit (prog_rock_..._enm.wav) ... key='G minor'`.)
        key = sc.key
        if key_override and key_override != sc.key:
            print(
                f"[Server] sidecar hit ({fixture_name}): ignoring "
                f"key_override={key_override!r}; sidecar wins (key={sc.key!r})"
            )
        # Same precedence rule for time signature: sidecar.time_signature
        # beats any client-supplied override on a hit.
        time_signature = sc.time_signature
        if (
            time_signature_override
            and time_signature_override != sc.time_signature
        ):
            print(
                f"[Server] sidecar hit ({fixture_name}): ignoring "
                f"time_signature_override={time_signature_override!r}; "
                f"sidecar wins (time_signature={sc.time_signature!r})"
            )
        print(
            f"[Server] sidecar hit ({fixture_name}): bpm={bpm} "
            f"key={key!r} time_signature={time_signature!r}"
        )
        return source, bpm, key, time_signature

    # Live path: librosa BPM, CNN key detection, full prepare_source.
    # No automated time-signature detector today; the operator override
    # wins, otherwise we default to "4" (matches the model's most-
    # supported meter).
    import librosa
    print("[Server] Detecting BPM + key...")
    mono_np = audio_in.waveform.mean(dim=0).numpy()
    bpm_raw, _ = librosa.beat.beat_track(y=mono_np, sr=SAMPLE_RATE)
    bpm = int(round(float(np.asarray(bpm_raw).flat[0])))
    key = key_override or detect_key(mono_np, SAMPLE_RATE)
    time_signature = time_signature_override or "4"
    print(f"  BPM: {bpm}  Key: {key}  TimeSig: {time_signature}")

    print("[Server] Preparing source...")
    source = session.prepare_source(audio_in)
    return source, bpm, key, time_signature


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
    offload_text_encoder: bool = False,
):
    print(
        f"[Server] Client connected "
        f"(decoder={decoder_backend}, vae={vae_backend}, ckpt={checkpoint}, "
        f"text_encoder={'offload' if offload_text_encoder else 'resident'})"
    )

    # Disable Nagle on the connection socket. Param frames are tiny (<1 KB
    # of JSON each) and we send them at ~125 Hz; with Nagle on the kernel
    # may coalesce them into batches up to ~40 ms wide, which adds latency
    # on top of the recv-loop drain. The websockets sync server exposes
    # the underlying socket as ``ws.socket``; swallow AttributeError so
    # this stays a no-op if the library reorganizes.
    try:
        ws.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except (AttributeError, OSError):
        pass

    # ---- Phase 1: Init ----
    config = json.loads(ws.recv())
    print(f"[Server] Config: {config}")

    audio_bytes = ws.recv()
    waveform = _decode_audio_msg(audio_bytes)
    # Cap at the largest registered TRT engine profile rather than
    # hardcoding 60 s. Anything longer than the largest profile can't
    # be handled by any built engine, but we let the operator stretch
    # all the way up to that ceiling — picking the smallest-fitting
    # engine happens below in available_trt_engines().
    max_seconds = max_profile_duration_s()
    waveform = waveform[:, :int(max_seconds * SAMPLE_RATE)]
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
    # Walk-window mode: route long sources through the 60s DiT engine by
    # sliding a fixed-T window across the song each tick (avoids the
    # 240s engine's parameter-update latency). vae_encode still has to
    # fit the full song so the source can be pre-encoded once at load
    # time; only the decoder profile is pinned to walk_window_s.
    walk_window = bool(config.get("walk_window", False))
    walk_window_s = float(config.get("walk_window_s", 60.0))
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
    # Profile manager owns the engine slots. When use_trt is False, it
    # stays None and the swap path keeps the legacy engine-less behavior.
    profile_mgr: TRTProfileManager | None = None
    if use_trt:
        profile_mgr = TRTProfileManager(
            decoder_backend=decoder_backend,
            vae_backend=vae_backend,
        )
        try:
            trt_engines, picked_dur = profile_mgr.resolve(audio_duration_s)
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
        # Walk-window override: pin the decoder to the walk_window_s
        # profile (typically 60s) regardless of source duration, while
        # keeping vae_encode at a profile that fits the full song so the
        # source can be encoded once at load. The runner slides a
        # walk_window_s slice across the source each tick. We
        # intentionally bind the profile manager to the WALK duration
        # below; mid-session swap_source within walk mode keeps using
        # the 60s decoder, which is the desired invariant.
        if walk_window and use_trt and audio_duration_s > walk_window_s + 0.1:
            try:
                walk_engines, walk_dur = profile_mgr.resolve(walk_window_s)
            except EngineNotBuiltError as exc:
                print(f"[Server] walk_window: 60s engine not built: {exc}")
                try:
                    ws.send(json.dumps({
                        "type": "error",
                        "code": "engine_not_built",
                        "message": str(exc),
                        "build_command": exc.build_command,
                        "duration_s": float(walk_window_s),
                    }))
                except Exception:
                    pass
                ws.close(1011, "walk_window TRT engine not built")
                return
            print(
                f"[Server] walk_window={walk_window_s:.0f}s active: "
                f"decoder={Path(walk_engines['decoder']).stem}, "
                f"vae_encode={Path(trt_engines['vae_encode']).stem}"
            )
            trt_engines = {
                "decoder": walk_engines["decoder"],
                "vae_encode": trt_engines["vae_encode"],
                "vae_decode": walk_engines["vae_decode"],
            }
            picked_dur = walk_dur

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
        offload_text_encoder=offload_text_encoder,
        trt_engines=trt_engines,
        vae_window=vae_window,
    )
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    # Bind the manager to the live engines so future swaps can compare
    # against the loaded profile and skip the swap when the picked
    # profile would be the same.
    if profile_mgr is not None:
        profile_mgr.bind(
            session.handler._diffusion_engine, trt_engines, picked_dur,
        )

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

    source, detected_bpm, detected_key, detected_time_signature = (
        _resolve_bpm_key_source(
            session,
            audio_in=audio_in,
            fixture_name=fixture_name,
            samples=int(waveform.shape[1]),
            checkpoint=checkpoint,
        )
    )

    def _active_trigger_prefix() -> str:
        """Comma-separated activation tokens for every currently-ENABLED
        LoRA that ships with a `<stem>.trigger.txt` sidecar.

        Defined ABOVE `_encode_cond_pair` so the latter's free-variable
        binding to this name is populated before any early text-encode
        site runs (initial session start fires the first encode below,
        well before the recv loop). Python's free-variable cell only
        gets populated when the enclosing function executes the `def`
        statement that creates the local — define-after-use across the
        same scope NameErrors at call time.
        """
        if not lora_available:
            return ""
        parts: list[str] = []
        for d in engine_obj.list_loras():
            if d.state != "enabled":
                continue
            trig = lora_trigger(d.path)
            if trig:
                parts.append(trig)
        return ", ".join(parts)

    # Two-conditioning cache for the live timbre-strength slider.
    # cond_silence uses the model's silence latent (refer_latent=None);
    # cond_full uses whichever timbre reference is currently active —
    # the playback source's own latent by default, or an uploaded
    # timbre-track latent when timbre_latent_ref[0] is set. Live alpha-
    # blend between them via ConditioningBlend (encoder hidden-state
    # lerp) gives the operator a strength knob without paying an encoder
    # forward pass per slider tick. Same approximation already used for
    # prompt crossfades. Recomputed on prompt change, on swap_source,
    # and on set_timbre_source / clear_timbre_source.
    def _encode_cond_pair(tags, refer_latent, bpm, duration, key, time_signature):
        # Server-side LoRA trigger injection: every currently-ENABLED
        # LoRA whose `<stem>.trigger.txt` sidecar is present contributes
        # its activation word to a hidden prefix prepended to the user's
        # caption. The user-visible promptA / promptB strings stay
        # untouched in the UI — this prepend only happens here, just
        # before the text encoder forward, so all five encode sites
        # (session start, prompt change, timbre refresh, swap_source,
        # LoRA enable/disable re-encode) inherit the same prefix from
        # one place. Idempotent against re-encodes with the same LoRA
        # state (the prefix is deterministic) so a noop re-encode is a
        # noop. See `_active_trigger_prefix` for the lookup logic.
        prefix = _active_trigger_prefix()
        if prefix:
            full_tags = f"{prefix}, {tags}" if tags else prefix
        else:
            full_tags = tags
        cs = session.encode_text(
            tags=full_tags,
            instruction=TASK_INSTRUCTIONS["cover"],
            refer_latent=None,
            bpm=bpm, duration=duration, key=key,
            time_signature=time_signature,
        )
        cf = session.encode_text(
            tags=full_tags,
            instruction=TASK_INSTRUCTIONS["cover"],
            refer_latent=refer_latent,
            bpm=bpm, duration=duration, key=key,
            time_signature=time_signature,
        )
        return cs, cf

    def _blend_for_strength(cs, cf, strength):
        from acestep.nodes.cond_nodes import ConditioningBlend
        if strength >= 0.999:
            return cf
        if strength <= 0.001:
            return cs
        return ConditioningBlend().execute(
            conditioning_a=cs,
            conditioning_b=cf,
            alpha=float(strength),
        )["conditioning"]

    print("[Server] Text encode (silence + self)...")
    cond_silence, cond_full = _encode_cond_pair(
        prompt, source.latent, detected_bpm, audio_duration_s,
        detected_key, detected_time_signature,
    )
    conditioning = cond_full  # default strength=1.0 == cond_full

    # Negative conditioning for the RCFG path (Residual CFG). Empty-prompt
    # encode once; reused every tick by PipelineRunner when the operator
    # selects rcfg_mode "full" or "initialize". "self" mode ignores this
    # (virtual v_uncond = initial_noise). The expense is one extra text
    # encoder pass at session start (~60 ms warm).
    cond_negative = session.encode_text(
        tags="",
        instruction=TASK_INSTRUCTIONS["cover"],
        refer_latent=None,
        bpm=detected_bpm, duration=audio_duration_s, key=detected_key,
        time_signature=detected_time_signature,
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
                # `trigger` is the LoRA's activation word, read from a
                # sibling `<stem>.trigger.txt` sidecar at the same path
                # as the .safetensors. Empty string when the sidecar
                # doesn't exist (LoRAs trained without a documented
                # trigger). The engine reads the same sidecar at encode
                # time and prepends to the user caption when this LoRA
                # is in ENABLED state, so the UI doesn't need to act on
                # this field — it's surfaced for transparency / tooltips.
                "trigger": lora_trigger(d.path),
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
        # Echoed back so the client's "Detected: …" UI for time signature
        # mirrors the keyscale path. Sidecar-aware on a hit; defaults to
        # ``"4"`` on the live path. Operator can change it post-init via
        # the prompt re-encode message (carries ``time_signature``).
        "time_signature": detected_time_signature,
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
    # Tracks the time signature actively baked into the latest cond_pair.
    # Mirrors ``key_ref`` exactly: refreshed on prompt re-encode (operator
    # override), on swap_source (next track's sidecar / override), and
    # consulted by every encode_text call so timbre / structure refits
    # honour the current meter.
    time_sig_ref = [detected_time_signature]
    duration_ref = [audio_duration_s]
    n_channels_ref = [n_channels]
    # Live timbre strength: 1.0 == cond_full (full timbre reference);
    # 0.0 == cond_silence (model uses its silence baseline).
    # cond_pair_ref holds (cond_silence, cond_full) for the *current*
    # source + prompt + timbre-override; refreshed on prompt change,
    # swap_source, and set/clear_timbre_source.
    timbre_strength_ref = [1.0]
    cond_pair_ref = [(cond_silence, cond_full)]
    # Optional uploaded timbre-track latent. None == use the playback
    # source's own latent (self-timbre, current default).
    timbre_latent_ref: list = [None]
    # Display name for the active timbre track (sent back in acks so the
    # client can show it). None when no override is active.
    timbre_name_ref: list = [None]

    def _active_refer_latent():
        tl = timbre_latent_ref[0]
        return tl if tl is not None else source_ref[0].latent

    # Structure (semantic-hint) override. Holds the raw user waveform so
    # we can re-derive the override's context_latent against the current
    # playback source length on every swap_source — the runner's
    # _update_hint_strength does LatentBlend(silence, context_latent)
    # at sample time and silence is sized to the source's frame count,
    # so the override's context_latent must match exactly. We pad-with-
    # silence or trim to enforce parity.
    playback_samples_ref = [int(waveform.shape[-1])]
    struct_audio_ref: list = [None]    # torch.Tensor [C, N], raw user clip
    struct_context_ref: list = [None]  # computed override context_latent
    struct_name_ref: list = [None]

    def _apply_struct_override():
        """(Re)derive the override's context_latent against the current
        playback source length and replace stream.source with one that
        carries it. No-op when no override is active. Caller is
        responsible for catching exceptions."""
        if struct_audio_ref[0] is None:
            return
        target = playback_samples_ref[0]
        wf = struct_audio_ref[0]
        if wf.shape[-1] > target:
            wf = wf[:, :target]
        elif wf.shape[-1] < target:
            wf = torch.nn.functional.pad(wf, (0, target - wf.shape[-1]))
        # Sidecar fast path: when the structure ref is a known fixture
        # AND the post-pad/trim sample count matches what was precomputed,
        # the cached context_latent is exactly what prepare_source would
        # produce. Skips ~500ms of VAE+extract on the recv thread.
        sc = (
            _try_load_sidecar(
                struct_name_ref[0],
                checkpoint=checkpoint,
                samples=int(wf.shape[-1]),
            )
            if struct_name_ref[0] else None
        )
        if sc is not None:
            device = session.handler.device
            dtype = session.handler.dtype
            struct_context_ref[0] = Latent(
                tensor=sc.context_latent.to(device, dtype).contiguous(),
            )
            print(
                f"[Server] _apply_struct_override: sidecar hit "
                f"({struct_name_ref[0]})"
            )
        else:
            audio_in = Audio(waveform=wf, sample_rate=SAMPLE_RATE)
            prepared = session.prepare_source(audio_in)
            struct_context_ref[0] = prepared.context_latent
        # source_ref[0] keeps the unmodified playback PreparedSource so
        # clear can restore it as-is. stream.source carries the
        # overridden context_latent for the runner to read.
        stream.source = PreparedSource(
            latent=source_ref[0].latent,
            context_latent=struct_context_ref[0],
        )
        # Force the runner to re-blend on the next tick — the run loop
        # only fires _update_hint_strength on slider deltas, so without
        # this prod stream.context_latent stays the previously-blended
        # tensor and the diffusion keeps reading the old structure.
        r = runner_holder[0]
        if r is not None:
            r.mark_hint_dirty()

    def _clear_struct_override():
        struct_audio_ref[0] = None
        struct_context_ref[0] = None
        struct_name_ref[0] = None
        stream.source = source_ref[0]
        r = runner_holder[0]
        if r is not None:
            r.mark_hint_dirty()

    def _load_fixture_waveform(name: str) -> torch.Tensor:
        """Read a known fixture WAV from the local HF cache into a
        ``[≤2, N]`` float32 tensor. Used by the ``set_*_fixture`` fast
        path so a Library pick doesn't have to round-trip through the
        browser as decoded PCM (the file already lives on this pod's
        disk; ``audio_fixture`` resolves to the cache hit). Caller is
        responsible for any further truncation / pool alignment."""
        if name not in KNOWN_FIXTURES:
            raise ValueError(f"unknown fixture: {name}")
        # Lazy import: the byte-upload path doesn't pull soundfile, and
        # we don't want a hard import-time dep just for this fast path.
        import soundfile as sf

        path = audio_fixture(name)
        audio_data, sr = sf.read(str(path), always_2d=True)
        if sr != SAMPLE_RATE:
            raise ValueError(
                f"fixture {name!r} sample rate {sr}, expected {SAMPLE_RATE}",
            )
        return torch.from_numpy(audio_data.T.copy()).float()[:2]

    def _apply_timbre_waveform(t_wf: torch.Tensor, name: str) -> float:
        """Mutate timbre state for a new ref. Returns post-truncation
        duration (seconds). Rolls back to prior state and re-raises on
        any failure. Caller sends the ack.

        Shared by the byte-upload (``set_timbre_source``) and fixture
        fast (``set_timbre_fixture``) paths so cache lookup, encode
        fallback, cond-pair refresh, and rollback semantics stay in
        one place."""
        prev_timbre_latent = timbre_latent_ref[0]
        prev_timbre_name = timbre_name_ref[0]
        prev_cond_pair = cond_pair_ref[0]
        prev_stream_cond = stream.conditioning
        try:
            cap = int(duration_ref[0] * SAMPLE_RATE)
            t_wf = t_wf[:, :cap]
            rem = t_wf.shape[-1] % pool
            if rem:
                t_wf = t_wf[:, :t_wf.shape[-1] - rem]
            if t_wf.shape[-1] < pool:
                raise ValueError("timbre clip too short")
            clip_s = t_wf.shape[-1] / SAMPLE_RATE
            sc = _try_load_sidecar(
                name, checkpoint=checkpoint,
                samples=int(t_wf.shape[-1]),
            )
            if sc is not None:
                device = session.handler.device
                dtype = session.handler.dtype
                timbre_latent = Latent(
                    tensor=sc.latent.to(device, dtype).contiguous(),
                )
                print(f"[Server] timbre: sidecar hit ({name})")
            else:
                timbre_audio = Audio(
                    waveform=t_wf, sample_rate=SAMPLE_RATE,
                )
                print(
                    f"[Server] timbre: VAE encoding {clip_s:.1f}s "
                    f"({t_wf.shape[0]}ch)..."
                )
                timbre_latent = session.encode_audio(timbre_audio)
                print(
                    f"[Server] timbre: VAE done "
                    f"(latent {tuple(timbre_latent.tensor.shape)})"
                )
            timbre_latent_ref[0] = timbre_latent
            timbre_name_ref[0] = name
            new_pair = _encode_cond_pair(
                prompt_text[0], timbre_latent,
                bpm_ref[0], duration_ref[0], key_ref[0],
                time_sig_ref[0],
            )
            cond_pair_ref[0] = new_pair
            stream.conditioning = _blend_for_strength(
                new_pair[0], new_pair[1], timbre_strength_ref[0],
            )
            return clip_s
        except Exception:
            timbre_latent_ref[0] = prev_timbre_latent
            timbre_name_ref[0] = prev_timbre_name
            cond_pair_ref[0] = prev_cond_pair
            stream.conditioning = prev_stream_cond
            raise

    def _apply_structure_waveform(s_wf: torch.Tensor, name: str) -> tuple[float, float]:
        """Stash a structure-ref waveform and re-derive the override's
        context_latent against the current playback length. Returns
        ``(clip_s, target_s)`` for the ack. Fully clears any prior
        override (matching the existing failure semantics) and re-
        raises on any mid-flight failure. Caller sends the ack."""
        s_wf = s_wf[:2]
        try:
            struct_audio_ref[0] = s_wf
            struct_name_ref[0] = name
            clip_s = s_wf.shape[-1] / SAMPLE_RATE
            target_s = playback_samples_ref[0] / SAMPLE_RATE
            _apply_struct_override()
            return clip_s, target_s
        except Exception:
            struct_audio_ref[0] = None
            struct_context_ref[0] = None
            struct_name_ref[0] = None
            stream.source = source_ref[0]
            raise

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
        # If the set of active triggers changed (enable adds, disable
        # removes), re-encode the current prompt so the new prefix is
        # baked into the conditioning. Without this the trigger only
        # takes effect on the NEXT prompt change — toggling a LoRA in
        # the library would visibly change the LoRA matrices but the
        # audio wouldn't reflect the trigger's encoder-side bias until
        # the user touched a prompt field. Cheap: one text-encode
        # forward + a cond-blend. `_encode_cond_pair` queries the now-
        # fresh state via `_active_trigger_prefix`.
        try:
            new_pair = _encode_cond_pair(
                prompt_text[0],
                _active_refer_latent(),
                bpm_ref[0],
                duration_ref[0],
                key_ref[0],
                time_sig_ref[0],
            )
            cond_pair_ref[0] = new_pair
            stream.conditioning = _blend_for_strength(
                new_pair[0], new_pair[1], timbre_strength_ref[0],
            )
        except Exception as e:
            # Re-encode is best-effort — if it fails, the LoRA matrices
            # still applied (engine_obj.enable/disable above succeeded);
            # next prompt change will catch up. Log and move on.
            print(f"[Server] re-encode after LoRA change failed: {e}")

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
                    msg = ws.recv(timeout=0.001)
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
                            # Time signature: same precedence — explicit
                            # override wins, otherwise carry the current
                            # value forward. Unknown / invalid values are
                            # ignored (keep the active ref) rather than
                            # poisoning the encoder with junk.
                            ts_override = _normalize_time_signature(
                                data.get("time_signature")
                            )
                            if ts_override is not None:
                                time_sig_ref[0] = ts_override
                            new_pair = _encode_cond_pair(
                                data["tags"],
                                _active_refer_latent(),
                                bpm_ref[0],
                                duration_ref[0],
                                data.get("key") or key_ref[0],
                                time_sig_ref[0],
                            )
                            cond_pair_ref[0] = new_pair
                            stream.conditioning = _blend_for_strength(
                                new_pair[0], new_pair[1],
                                timbre_strength_ref[0],
                            )
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
                        elif mtype == "set_timbre_strength":
                            try:
                                v = float(data.get("value", 1.0))
                            except (TypeError, ValueError):
                                v = 1.0
                            v = max(0.0, min(1.0, v))
                            timbre_strength_ref[0] = v
                            cs, cf = cond_pair_ref[0]
                            stream.conditioning = _blend_for_strength(cs, cf, v)
                        elif mtype == "set_timbre_source":
                            # Followed by a binary audio frame (same wire
                            # format as init / swap_source). _apply_timbre_
                            # waveform handles cap / pool trim / sidecar
                            # lookup / encode fallback / cond-pair refresh
                            # / rollback. This block just shuttles bytes
                            # off the wire and acks.
                            name = data.get("name") or "timbre"
                            print(
                                f"[Server] set_timbre_source: receiving "
                                f"audio for {name!r}..."
                            )
                            try:
                                audio_msg = ws.recv()
                            except ConnectionClosed:
                                running[0] = False
                                break
                            print(
                                f"[Server] set_timbre_source: got "
                                f"{len(audio_msg)} bytes"
                            )
                            try:
                                t_wf = _decode_audio_msg(audio_msg)
                                clip_s = _apply_timbre_waveform(t_wf, name)
                                with send_lock:
                                    ws.send(json.dumps({
                                        "type": "timbre_set",
                                        "name": name,
                                        "duration": clip_s,
                                    }))
                                print(
                                    f"[Server] timbre set: {name} "
                                    f"({clip_s:.1f}s)"
                                )
                            except Exception as exc:
                                print(f"[Server] set_timbre_source failed: {exc}")
                                import traceback
                                traceback.print_exc()
                                try:
                                    with send_lock:
                                        ws.send(json.dumps({
                                            "type": "timbre_failed",
                                            "error": str(exc),
                                        }))
                                except Exception:
                                    pass
                        elif mtype == "set_timbre_fixture":
                            # Wire shortcut for Library picks: client
                            # sends just {name}; server resolves the WAV
                            # from the local HF cache via audio_fixture.
                            # Avoids the browser fetching the WAV, decoding
                            # it, and re-uploading the result as a ~10×
                            # larger float32 PCM payload (the fixture is
                            # already on this pod's disk).
                            name = data.get("name", "")
                            print(f"[Server] set_timbre_fixture: {name!r}")
                            try:
                                t_wf = _load_fixture_waveform(name)
                                clip_s = _apply_timbre_waveform(t_wf, name)
                                with send_lock:
                                    ws.send(json.dumps({
                                        "type": "timbre_set",
                                        "name": name,
                                        "duration": clip_s,
                                    }))
                                print(
                                    f"[Server] timbre set: {name} "
                                    f"({clip_s:.1f}s)"
                                )
                            except Exception as exc:
                                print(f"[Server] set_timbre_fixture failed: {exc}")
                                import traceback
                                traceback.print_exc()
                                try:
                                    with send_lock:
                                        ws.send(json.dumps({
                                            "type": "timbre_failed",
                                            "error": str(exc),
                                        }))
                                except Exception:
                                    pass
                        elif mtype == "clear_timbre_source":
                            timbre_latent_ref[0] = None
                            timbre_name_ref[0] = None
                            new_pair = _encode_cond_pair(
                                prompt_text[0],
                                source_ref[0].latent,
                                bpm_ref[0],
                                duration_ref[0],
                                key_ref[0],
                                time_sig_ref[0],
                            )
                            cond_pair_ref[0] = new_pair
                            stream.conditioning = _blend_for_strength(
                                new_pair[0], new_pair[1],
                                timbre_strength_ref[0],
                            )
                            try:
                                with send_lock:
                                    ws.send(json.dumps({
                                        "type": "timbre_cleared",
                                    }))
                            except Exception:
                                pass
                            print("[Server] timbre cleared")
                        elif mtype == "set_structure_source":
                            # Followed by a binary audio frame. _apply_
                            # structure_waveform handles pad/trim, sidecar
                            # lookup or live prepare_source, and rollback.
                            name = data.get("name") or "structure"
                            print(
                                f"[Server] set_structure_source: receiving "
                                f"audio for {name!r}..."
                            )
                            try:
                                audio_msg = ws.recv()
                            except ConnectionClosed:
                                running[0] = False
                                break
                            print(
                                f"[Server] set_structure_source: got "
                                f"{len(audio_msg)} bytes"
                            )
                            try:
                                s_wf = _decode_audio_msg(audio_msg)
                                clip_s, target_s = _apply_structure_waveform(
                                    s_wf, name,
                                )
                                with send_lock:
                                    ws.send(json.dumps({
                                        "type": "structure_set",
                                        "name": name,
                                        "duration": clip_s,
                                    }))
                                print(
                                    f"[Server] structure set: {name} "
                                    f"({clip_s:.1f}s, fitted to "
                                    f"{target_s:.1f}s)"
                                )
                            except Exception as exc:
                                print(
                                    f"[Server] set_structure_source "
                                    f"failed: {exc}"
                                )
                                import traceback
                                traceback.print_exc()
                                try:
                                    with send_lock:
                                        ws.send(json.dumps({
                                            "type": "structure_failed",
                                            "error": str(exc),
                                        }))
                                except Exception:
                                    pass
                        elif mtype == "set_structure_fixture":
                            # Wire shortcut for Library picks. Same
                            # rationale as set_timbre_fixture.
                            name = data.get("name", "")
                            print(f"[Server] set_structure_fixture: {name!r}")
                            try:
                                s_wf = _load_fixture_waveform(name)
                                clip_s, target_s = _apply_structure_waveform(
                                    s_wf, name,
                                )
                                with send_lock:
                                    ws.send(json.dumps({
                                        "type": "structure_set",
                                        "name": name,
                                        "duration": clip_s,
                                    }))
                                print(
                                    f"[Server] structure set: {name} "
                                    f"({clip_s:.1f}s, fitted to "
                                    f"{target_s:.1f}s)"
                                )
                            except Exception as exc:
                                print(f"[Server] set_structure_fixture failed: {exc}")
                                import traceback
                                traceback.print_exc()
                                try:
                                    with send_lock:
                                        ws.send(json.dumps({
                                            "type": "structure_failed",
                                            "error": str(exc),
                                        }))
                                except Exception:
                                    pass
                        elif mtype == "clear_structure_source":
                            _clear_struct_override()
                            try:
                                with send_lock:
                                    ws.send(json.dumps({
                                        "type": "structure_cleared",
                                    }))
                            except Exception:
                                pass
                            print("[Server] structure cleared")
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
                                swap_pending["time_signature"] = (
                                    _normalize_time_signature(
                                        data.get("time_signature")
                                    )
                                )
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

    # Forward decl so closures defined above (e.g. _apply_struct_override
    # via the recv thread) can resolve the cell without NameError before
    # the runner is constructed at the bottom of this function. The slot
    # is None until ``runner_holder[0] = runner`` lands; callers null-
    # check before invoking runner methods.
    runner_holder: list = [None]

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
    def apply_swap_if_pending():
        with swap_lock:
            audio_msg = swap_pending.get("bytes")
            tags = swap_pending.get("tags")
            requested_key = swap_pending.get("key")
            requested_time_sig = swap_pending.get("time_signature")
            new_fixture_name = swap_pending.get("fixture_name")
            if audio_msg is None:
                return
            swap_pending["bytes"] = None
            swap_pending["tags"] = None
            swap_pending["key"] = None
            swap_pending["time_signature"] = None
            swap_pending["fixture_name"] = None
        try:
            new_wf = _decode_audio_msg(audio_msg)
            # Cap at the same ceiling the initial upload used so swaps
            # take advantage of every built engine profile, not a stale
            # 60 s default.
            new_wf = new_wf[:, :int(max_seconds * SAMPLE_RATE)]
            rem = new_wf.shape[-1] % pool
            if rem:
                new_wf = new_wf[:, :new_wf.shape[-1] - rem]
            new_audio_duration_s = new_wf.shape[1] / SAMPLE_RATE
            print(
                f"[Server] Swapping source ({new_audio_duration_s:.1f}s, "
                f"{new_wf.shape[0]}ch)..."
            )

            # Profile swap (no-op when the new duration fits the same
            # profile that's currently loaded). Must run BEFORE
            # prepare_source: VAE-encode is the first GPU consumer and
            # needs the new vae_encode engine bound to its cache.
            if profile_mgr is not None:
                try:
                    profile_mgr.ensure_profile(new_audio_duration_s)
                except EngineNotBuiltError as exc:
                    print(f"[Server] Swap aborted: {exc}")
                    with send_lock:
                        ws.send(json.dumps({
                            "type": "swap_failed",
                            "error": str(exc),
                            "build_command": exc.build_command,
                        }))
                    return

            new_audio_in = Audio(waveform=new_wf, sample_rate=SAMPLE_RATE)
            new_source, new_bpm, new_key, new_time_sig = (
                _resolve_bpm_key_source(
                    session,
                    audio_in=new_audio_in,
                    fixture_name=new_fixture_name,
                    samples=int(new_wf.shape[1]),
                    checkpoint=checkpoint,
                    key_override=requested_key,
                    time_signature_override=requested_time_sig,
                )
            )
            # Use the active timbre reference if one is uploaded; otherwise
            # the new playback source's own latent. Override persists
            # across source swaps.
            stream.source = new_source
            source_ref[0] = new_source
            playback_samples_ref[0] = int(new_wf.shape[-1])
            tl = timbre_latent_ref[0]
            new_pair = _encode_cond_pair(
                tags,
                tl if tl is not None else new_source.latent,
                new_bpm, new_audio_duration_s, new_key, new_time_sig,
            )
            new_cond = _blend_for_strength(
                new_pair[0], new_pair[1], timbre_strength_ref[0],
            )

            stream.conditioning = new_cond
            stream.context_latent = new_source.context_latent
            cond_pair_ref[0] = new_pair
            # Re-derive structure override against the new source length.
            # On failure (e.g. VAE engine couldn't fit the new clip), drop
            # the override rather than block the swap — the user can re-
            # upload after the swap settles.
            if struct_audio_ref[0] is not None:
                try:
                    _apply_struct_override()
                except Exception as exc:
                    print(
                        f"[Server] swap: struct override re-apply failed: {exc}"
                    )
                    _clear_struct_override()
                    try:
                        with send_lock:
                            ws.send(json.dumps({
                                "type": "structure_failed",
                                "error": f"dropped after swap: {exc}",
                            }))
                    except Exception:
                        pass
            bpm_ref[0] = new_bpm
            key_ref[0] = new_key
            time_sig_ref[0] = new_time_sig
            duration_ref[0] = new_audio_duration_s
            prompt_text[0] = tags
            r = runner_holder[0]
            if r is not None:
                # Source latent length may have changed; rebuild silence so
                # _update_hint_strength's blend operands match shapes.
                r._rebuild_silence_latent()
                # Force a fresh hint blend on the next tick. Without
                # this, when hint_strength < 1.0, the runner keeps the
                # previously-blended stream.context_latent (sized to the
                # old source) until the operator nudges the slider —
                # diffusion would crash on the shape mismatch or read
                # stale structure.
                r.mark_hint_dirty()

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
                    "time_signature": new_time_sig,
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
        k1_name=k1_name, seed=1528, skip_threshold=5e-4,
        sde_curve_display=sde_curve_display, params=params,
        prompt_text=prompt_text, running=running,
        motion_val=motion_val, motion_lock=motion_lock,
        on_audio_ready=on_audio_ready,
        before_tick=apply_pending,
        walk_window=walk_window,
        walk_window_s=walk_window_s,
        neg_conditioning=cond_negative,
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

        # Tear down per-session GPU state. Order matters: stream.close()
        # drops the StreamPipeline's references into the engine before
        # session.close() actually destroys the engine + ModelContext.
        # session.close() ends with gc.collect() + cuda.empty_cache().
        try:
            stream.close()
        except Exception as exc:
            print(f"[Server] stream.close() raised: {exc}")
        try:
            session.close()
        except Exception as exc:
            print(f"[Server] session.close() raised: {exc}")
