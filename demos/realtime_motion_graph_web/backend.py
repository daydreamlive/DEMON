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

import contextlib
import json
import os
import queue
import socket
import struct
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
from acestep.engine.obs import logger, spawn_thread
from acestep.engine.session import PreparedSource, Session
from acestep.engine.trt.profile_manager import TRTProfileManager
from acestep.fixtures import (
    FixtureSidecar, KNOWN_FIXTURES, audio_fixture, fixture_sidecar,
)
from acestep.nodes.types import Audio, Latent
from acestep.lora_metadata import load_lora_metadata
from acestep.paths import (
    EngineNotBuiltError,
    available_dreamvae_decode_engine,
    checkpoint_scale,
    checkpoints_dir,
    dreamvae_decode_engine_name,
    loras_dir,
    max_profile_duration_s,
    smallest_fitting_profile_duration_s,
    trt_engine_path,
)

from .audio_engine import AudioEngine
from .knobs import build_banks, CHANNEL_GROUPS, KEYSTONE_CHANNELS
from .melband_reformer import (
    extract_upload_stems,
    normalize_stem_source_mode,
    resolve_upload_stem_source_mode,
)
from .protocol import (
    SAMPLE_RATE,
    SLICE_FLAG_DELTA,
    SLICE_HDR_FMT,
    SLICE_HDR_SIZE,
    T,
)
from .pipeline import PipelineRunner
from . import session_registry


# ---------------------------------------------------------------------------
# Source / conditioning resolver (sidecar-aware)
# ---------------------------------------------------------------------------

def _try_load_sidecar(
    fixture_name: str | None, *, samples: int,
) -> FixtureSidecar | None:
    """Look up a fixture sidecar; return None on miss / mismatch.

    Length check guards against runtime truncation that disagrees with
    what the sidecar was precomputed for (e.g. a smaller TRT profile
    cap kicking in). The caller falls back to live computation in that
    case so cached tensor shapes can't poison the streaming pipeline.

    Sidecars are not checkpoint-gated; the VAE and semantic
    tokenizer/detokenizer that produce the cached tensors are shared
    across the ACE-Step v1.5 family.
    """
    if not fixture_name:
        return None
    try:
        sc = fixture_sidecar(fixture_name)
    except Exception as e:
        logger.warning(
            "sidecar_lookup_failed fixture={} error={}", fixture_name, e,
        )
        return None
    if sc is None:
        return None
    if sc.samples != samples:
        logger.warning(
            "sidecar_length_mismatch fixture={} sidecar_samples={} "
            "runtime_samples={}",
            fixture_name, sc.samples, samples,
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


def _load_known_fixture_waveform(name: str) -> torch.Tensor:
    """Load a known fixture's audio from the pod's own fixture cache and
    return it in the exact shape ``_decode_audio_msg`` produces
    (``[≤2, N]`` float32 at ``SAMPLE_RATE``).

    The pod already serves this file at ``/fixtures/<name>``; for known
    fixtures the browser shouldn't have to download → decode → re-upload
    ~20 MB of PCM over the WebSocket (~11 s on the measured cold path).
    Same uniform-path output as the upload route, so every downstream
    consumer (sidecar resolve, echoed initial buffer, prepare_source
    fallback) is unchanged.
    """
    path = str(audio_fixture(name))  # resolves to the on-disk cache
    try:
        import soundfile as sf
        data, sr = sf.read(path, dtype="float32", always_2d=True)  # (n, ch)
    except Exception:
        import librosa
        mono, sr = librosa.load(path, sr=None, mono=True)
        data = np.stack([mono, mono], axis=1).astype(np.float32)
    if sr != SAMPLE_RATE:
        import librosa
        data = librosa.resample(
            data.T, orig_sr=sr, target_sr=SAMPLE_RATE
        ).T.astype(np.float32)
    if data.ndim == 1:
        data = data[:, None]
    return torch.from_numpy(np.ascontiguousarray(data.T, dtype=np.float32))[:2]


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


def _send_stem_payload(
    ws,
    *,
    fixture_name: str | None,
    source_mode: str | None,
    stems: dict[str, torch.Tensor],
) -> None:
    order = ["vocals", "instruments"]
    first = stems[order[0]]
    frames = int(first.shape[-1])
    channels = int(first.shape[0])
    ws.send(json.dumps({
        "type": "stem_assets",
        "fixture_name": fixture_name or "",
        "sample_rate": SAMPLE_RATE,
        "channels": channels,
        "frames": frames,
        "stems": order,
        "source_mode": source_mode or "full",
    }))
    for name in order:
        arr = stems[name].detach().cpu().numpy().T.astype(np.float16)
        ws.send(arr.tobytes())


def _extract_and_select_upload_stem(
    waveform: torch.Tensor,
    *,
    session: Session,
    source: PreparedSource,
    source_mode: str | None,
    log_context: str = "",
) -> tuple[dict[str, torch.Tensor] | None, str | None, PreparedSource, torch.Tensor]:
    if source_mode is None:
        return None, None, source, waveform

    logger.info(
        "stems_extract_start source_mode={} context={}",
        source_mode, log_context or None,
    )
    try:
        upload_stems = extract_upload_stems(
            waveform=waveform,
            device=session.handler.device,
            backend_sample_rate=SAMPLE_RATE,
        )
        if source_mode == "full":
            return upload_stems, None, source, waveform

        selected_wf = upload_stems[source_mode]
        selected_audio = Audio(waveform=selected_wf, sample_rate=SAMPLE_RATE)
        logger.info(
            "stem_prepare_source source_mode={} context={}",
            source_mode, log_context or None,
        )
        selected_source = session.prepare_source(selected_audio)
        return upload_stems, None, selected_source, selected_wf
    except Exception as exc:
        logger.exception(
            "stem_extract_failed context={} error={}",
            log_context or None, exc,
        )
        return None, str(exc), source, waveform


def _resolve_bpm_key_source(
    session: Session,
    *,
    audio_in: Audio,
    fixture_name: str | None,
    samples: int,
    key_override: str | None = None,
    time_signature_override: str | None = None,
) -> tuple[PreparedSource, int, str, str]:
    """Resolve (source, bpm, key, time_signature) for a (fixture, audio) pair.

    For known fixtures with a sidecar present (JSON+safetensors in the
    dataset or local staging dir, matching audio length), returns the
    cached source latent + context_latent and reads BPM, key, and
    time_signature from the sidecar JSON. Skips librosa beat tracking,
    CNN key detection, and ``Session.prepare_source`` — the
    prompt-independent half of the per-connect work.

    Conditioning is *not* cached (see fixtures.py). Callers run
    ``Session.encode_text`` against ``source.latent`` themselves; with
    the source latent already on GPU this is ~60ms warm.

    Falls through to live librosa + detect_key + prepare_source when:
      - ``fixture_name`` is None / unknown
      - sidecar files aren't in the dataset yet
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
    sc = _try_load_sidecar(fixture_name, samples=samples)

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
            logger.info(
                "sidecar_override_ignored fixture={} field=key "
                "override={} sidecar={}",
                fixture_name, key_override, sc.key,
            )
        # Same precedence rule for time signature: sidecar.time_signature
        # beats any client-supplied override on a hit.
        time_signature = sc.time_signature
        if (
            time_signature_override
            and time_signature_override != sc.time_signature
        ):
            logger.info(
                "sidecar_override_ignored fixture={} field=time_signature "
                "override={} sidecar={}",
                fixture_name, time_signature_override, sc.time_signature,
            )
        logger.info(
            "sidecar_hit fixture={} bpm={} key={} time_signature={}",
            fixture_name, bpm, key, time_signature,
        )
        return source, bpm, key, time_signature

    # Live path: librosa BPM, CNN key detection, full prepare_source.
    # No automated time-signature detector today; the operator override
    # wins, otherwise we default to "4" (matches the model's most-
    # supported meter).
    import librosa
    logger.info("bpm_key_detect_start")
    mono_np = audio_in.waveform.mean(dim=0).numpy()
    bpm_raw, _ = librosa.beat.beat_track(y=mono_np, sr=SAMPLE_RATE)
    bpm = int(round(float(np.asarray(bpm_raw).flat[0])))
    key = key_override or detect_key(mono_np, SAMPLE_RATE)
    time_signature = time_signature_override or "4"
    logger.info(
        "bpm_key_detected bpm={} key={} time_signature={}",
        bpm, key, time_signature,
    )

    logger.info("prepare_source_start")
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
# Pipeline depth bounds
# ---------------------------------------------------------------------------

# Hard floor for the StreamPipeline ring buffer. <1 makes the buffer empty
# and nothing ticks. The TRT cap is read from the loaded engine; the eager
# / compile cap is fixed.
MIN_PIPELINE_DEPTH = 1
EAGER_MAX_PIPELINE_DEPTH = 4

# Idle GPU pause threshold. After this many seconds with no incoming WS
# or control-bus message, the runner stops invoking the DiT each tick.
# The audio engine keeps serving from its existing buffer (which the
# walk_window LoRA designs to loop cleanly at walk_window_s), so audio
# continues uninterrupted while the GPU idles. Any incoming message
# resets the timer immediately; the next loop iteration resumes a normal
# tick. Set to 0 to disable the pause entirely (always tick).
IDLE_PAUSE_S = float(os.environ.get("DEMON_IDLE_PAUSE_S", "20"))


def _compute_max_pipeline_depth(diffusion_engine) -> int:
    """Largest ``pipeline_depth`` the loaded backend can serve.

    For TRT decoders this is the ``hidden_states`` batch dim's max bound
    on optimization profile 0 (the canonical / only profile we build).
    For eager / compile decoders the runtime has no fixed cap, so we
    return ``EAGER_MAX_PIPELINE_DEPTH`` to match the docs and the demo's
    knob ceiling.
    """
    trt_engine = getattr(diffusion_engine, "_trt_engine", None)
    if trt_engine is None:
        return EAGER_MAX_PIPELINE_DEPTH
    try:
        _, _, max_shape = trt_engine.get_tensor_profile_shape(
            "hidden_states", 0,
        )
        return max(MIN_PIPELINE_DEPTH, int(max_shape[0]))
    except Exception as exc:
        logger.warning(
            "trt_batch_cap_unreadable error={!r} fallback={}",
            exc, EAGER_MAX_PIPELINE_DEPTH,
        )
        return EAGER_MAX_PIPELINE_DEPTH


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
    """Connection entrypoint. The body lives in ``_handle_client_body``;
    this wrapper exists only to own a single ``ExitStack`` so the
    contextvar tokens bound for session / track / swap unwind in reverse
    order on every exit path (normal return, early return, or
    exception) — no early-return site has to remember an explicit
    ``__exit__`` call."""
    with contextlib.ExitStack() as ctx_stack:
        _handle_client_body(
            ws, ctx_stack,
            decoder_backend=decoder_backend,
            vae_backend=vae_backend,
            checkpoint=checkpoint,
            offload_text_encoder=offload_text_encoder,
        )


def _handle_client_body(
    ws,
    ctx_stack: contextlib.ExitStack,
    *,
    decoder_backend: str,
    vae_backend: str,
    checkpoint: str,
    offload_text_encoder: bool,
):
    logger.info(
        "client_connected decoder={} vae={} checkpoint={} text_encoder={}",
        decoder_backend, vae_backend, checkpoint,
        "offload" if offload_text_encoder else "resident",
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

    # Mint session_id immediately and bind it (plus the client's optional
    # client_id) into loguru's contextvars so every log record emitted on
    # this connection's lifetime carries the correlation IDs. The
    # ExitStack owned by handle_client unwinds these in reverse order
    # on any exit path — including the early returns below — so we
    # don't have to plumb explicit __exit__ calls through them.
    session_id = session_registry.new_session_id()
    _client_id = config.get("client_id") or None
    ctx_stack.enter_context(logger.contextualize(
        session_id=session_id,
        client_id=_client_id,
    ))
    logger.info(
        "session_init config_keys={} client_id={}",
        sorted(config.keys()), _client_id,
    )

    # Session-init timing instrumentation. t0 == config received; every
    # milestone prints wall-seconds since t0 so the per-connect latency
    # can be split into prepare / TRT-load / stream-build / first-gen
    # without guessing from interleaved loguru lines.
    _t0 = time.monotonic()
    _first_slice = [False]

    def _ms(stage: str) -> None:
        # Per-stage init timing. Stays at DEBUG so it's silent at INFO
        # in prod but available with DEMON_LOG_LEVEL=DEBUG when chasing
        # a slow per-connect latency complaint.
        logger.debug(
            "init_timing stage={} elapsed_s={:.3f}",
            stage, time.monotonic() - _t0,
        )

    # Server-side known-fixture load. When the client opts in via
    # ``use_server_fixture`` AND names a known fixture, skip the
    # download→decode→re-upload round-trip (~11 s of the measured cold
    # path) and read the waveform straight from the pod's fixture cache.
    # Old clients that don't send the flag take the unchanged upload
    # path below, so this is safe across the Vercel(UI)/bake(backend)
    # deploy skew.
    _fix_name = config.get("fixture_name")
    if config.get("use_server_fixture") and _fix_name in KNOWN_FIXTURES:
        try:
            waveform = _load_known_fixture_waveform(_fix_name)
            _ms("audio_serverside_loaded")
        except Exception as exc:
            logger.warning(
                "server_side_fixture_load_failed fixture={} error={} "
                "fallback=client_upload",
                _fix_name, exc,
            )
            audio_bytes = ws.recv()
            waveform = _decode_audio_msg(audio_bytes)
            _ms("audio_recv_decoded")
    else:
        audio_bytes = ws.recv()
        waveform = _decode_audio_msg(audio_bytes)
        _ms("audio_recv_decoded")
    use_trt = decoder_backend == "tensorrt" or vae_backend == "tensorrt"
    trt_profile_checkpoint = checkpoint if decoder_backend == "tensorrt" else "acestep-v15-turbo"

    # Cap at the largest registered TRT engine profile rather than
    # hardcoding 60 s. Anything longer than the largest profile can't
    # be handled by any built engine, but we let the operator stretch
    # all the way up to that ceiling — picking the smallest-fitting
    # engine happens below in available_trt_engines().
    if use_trt:
        try:
            max_seconds = max_profile_duration_s(checkpoint=trt_profile_checkpoint)
        except ValueError as exc:
            logger.error(
                "unsupported_trt_checkpoint error={}", exc,
            )
            try:
                ws.send(json.dumps({
                    "type": "error",
                    "code": "unsupported_trt_checkpoint",
                    "message": str(exc),
                }))
            except Exception:
                pass
            ws.close(1011, "unsupported TRT checkpoint")
            return
    else:
        max_seconds = max_profile_duration_s()
    waveform = waveform[:, :int(max_seconds * SAMPLE_RATE)]
    pool = 1920 * 5
    rem = waveform.shape[-1] % pool
    if rem:
        waveform = waveform[:, :waveform.shape[-1] - rem]
    logger.info(
        "audio_loaded duration_s={:.1f} channels={}",
        waveform.shape[1] / SAMPLE_RATE, waveform.shape[0],
    )

    use_sde = config.get("sde", False)
    use_lora = config.get("lora", False)
    vae_window = config.get("vae_window", 1.0)
    crop_seconds = config.get("crop", 0.0)
    depth = config.get("depth", 4)
    steps = config.get("steps", 8)
    prompt = config.get("prompt", "instrumental music")
    prompt_b = config.get("prompt_b", prompt)
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
    stem_source_mode = resolve_upload_stem_source_mode(
        fixture_name,
        normalize_stem_source_mode(config.get("stem_source_mode")),
        known_fixtures=KNOWN_FIXTURES,
    )

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

    # Bind the initial source as contextvars so any error that surfaces
    # downstream (VAE encode rejection, pipeline_error, etc.) carries the
    # fixture name + duration without each call site having to plumb
    # them through. Loguru's contextualize uses contextvars internally
    # via a token-stack: nested enters/exits work, but out-of-order
    # token resets corrupt the stack — so we don't try to *update* this
    # binding when the swap path runs. Instead the swap body opens its
    # own nested contextualize (see _swap_ctx in apply_swap_if_pending),
    # which means a swap-time error sees the NEW track but a
    # post-swap pipeline_error still sees the INITIAL one. The latter
    # is a small cosmetic limitation accepted for v1 — the swap is
    # where the high-signal failures live.
    ctx_stack.enter_context(logger.contextualize(
        fixture_name=fixture_name or None,
        audio_duration_s=round(audio_duration_s, 2),
    ))
    # Profile manager owns the engine slots. When use_trt is False, it
    # stays None and the swap path keeps the legacy engine-less behavior.
    profile_mgr: TRTProfileManager | None = None
    if use_trt:
        profile_mgr = TRTProfileManager(
            decoder_backend=decoder_backend,
            vae_backend=vae_backend,
            checkpoint=trt_profile_checkpoint,
        )
        try:
            trt_engines, picked_dur = profile_mgr.resolve(audio_duration_s)
        except EngineNotBuiltError as exc:
            # Surface to the operator (server log) AND the client UI.
            # WebSocket close reason is capped at 123 bytes by the
            # protocol, so the build command goes in a JSON message
            # first and the close reason carries a short summary.
            logger.error(
                "trt_engine_not_built duration_s={} error={}",
                exc.duration_s, exc,
            )
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
                logger.error(
                    "walk_window_engine_not_built window_s={} error={}",
                    walk_window_s, exc,
                )
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
            logger.info(
                "walk_window_active window_s={:.0f} decoder={} vae_encode={}",
                walk_window_s,
                Path(walk_engines["decoder"]).stem,
                Path(trt_engines["vae_encode"]).stem,
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
        ideal_dur = smallest_fitting_profile_duration_s(
            audio_duration_s,
            checkpoint=trt_profile_checkpoint,
        )
        if picked_dur > ideal_dur:
            logger.warning(
                "trt_profile_fallback picked_dur_s={:.0f} ideal_dur_s={:.0f} "
                "audio_duration_s={:.1f} reason=ideal_profile_not_built",
                picked_dur, ideal_dur, audio_duration_s,
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
            logger.warning(
                "dreamvae_engine_missing wanted={} fallback={}",
                wanted, Path(trt_engines["vae_decode"]).stem,
            )
            fast_vae = False
    elif fast_vae:
        logger.warning(
            "fast_vae_requires_tensorrt vae_backend={} ignoring=true",
            vae_backend,
        )
        fast_vae = False

    logger.info(
        "model_load_start decoder={} vae={} checkpoint={}",
        decoder_backend, vae_backend, checkpoint,
    )
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
    logger.info("model_loaded duration_s={:.1f}", time.time() - t0)

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
        logger.warning("lora_engine_unavailable decoder_backend={}", decoder_backend)
        use_lora = False

    max_pipeline_depth = _compute_max_pipeline_depth(engine_obj)
    depth = max(MIN_PIPELINE_DEPTH, min(int(depth), max_pipeline_depth))
    logger.info(
        "pipeline_depth_set depth={} max={} backend={}",
        depth, max_pipeline_depth,
        "trt" if engine_obj._trt_engine is not None else "eager",
    )

    initial_enable_ids: list[str] = []
    if use_lora:
        # Resolve any explicit enable-by-id requests (these must already
        # be in the catalog from the auto-scan).
        catalog_ids = {d.id for d in engine_obj.list_loras()}
        for lid in enabled_lora_ids:
            if lid in catalog_ids:
                initial_enable_ids.append(lid)
            else:
                logger.warning("lora_id_not_in_catalog id={}", lid)
        # Resolve ad-hoc paths: register if needed, then enable.
        for p in extra_lora_paths:
            pp = Path(p)
            if not pp.exists():
                logger.warning("lora_path_missing path={}", p)
                continue
            try:
                lid = engine_obj.register_lora(str(pp))
                if lid not in initial_enable_ids:
                    initial_enable_ids.append(lid)
            except Exception as e:
                logger.exception(
                    "lora_register_failed path={} error={}", p, e,
                )
        # Kick off background materialization for everything we plan to
        # enable. Non-blocking; the eventual enable will block on the
        # future if the worker hasn't finished yet.
        for lid in initial_enable_ids:
            try:
                engine_obj.prewarm_lora(lid)
            except Exception as e:
                logger.exception(
                    "lora_prewarm_failed id={} error={}", lid, e,
                )
        if not initial_enable_ids:
            logger.info("lora_startup_empty reason=catalog_only")

    audio_in = Audio(waveform=waveform, sample_rate=SAMPLE_RATE)

    _ms("resolve_source_start")
    source, detected_bpm, detected_key, detected_time_signature = (
        _resolve_bpm_key_source(
            session,
            audio_in=audio_in,
            fixture_name=fixture_name,
            samples=int(waveform.shape[1]),
        )
    )
    upload_stems: dict[str, torch.Tensor] | None = None
    stem_error: str | None = None
    upload_stems, stem_error, source, waveform = _extract_and_select_upload_stem(
        waveform,
        session=session,
        source=source,
        source_mode=stem_source_mode,
    )
    if stem_error is not None and stem_source_mode != "full":
        logger.error(
            "stem_extract_failed_fatal source_mode={} error={}",
            stem_source_mode, stem_error,
        )
        try:
            ws.send(json.dumps({
                "type": "error",
                "code": "stem_extract_failed",
                "message": f"Stem extraction failed: {stem_error}",
            }))
        except Exception:
            pass
        ws.close(1011, "stem extraction failed")
        return

    _ms("resolve_source_done")

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
        # WYSIWYG: the encoder sees exactly the text the UI sent. LoRA
        # trigger words land in `tags` via the client's visible-prepend
        # logic (see web/store/useLoraStore + the auto_prepend_lora_triggers
        # config flag) so the user can edit or remove them like any other
        # prompt token. The server intentionally does NOT inject anything
        # behind the user's back.
        cs = session.encode_text(
            tags=tags,
            lyrics="[Instrumental]",
            instruction=TASK_INSTRUCTIONS["cover"],
            refer_latent=None,
            bpm=bpm, duration=duration, key=key,
            time_signature=time_signature,
        )
        cf = session.encode_text(
            tags=tags,
            lyrics="[Instrumental]",
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

    logger.info("text_encode_start variant=silence_and_self")
    cond_silence, cond_full = _encode_cond_pair(
        prompt, source.latent, detected_bpm, audio_duration_s,
        detected_key, detected_time_signature,
    )
    # Encode prompt B at session start so the blend slider works
    # immediately without forcing the operator to click Send Tags.
    # When B matches A, reuse the pair (no second encoder pass).
    if prompt_b and prompt_b != prompt:
        cond_silence_b, cond_full_b = _encode_cond_pair(
            prompt_b, source.latent, detected_bpm, audio_duration_s,
            detected_key, detected_time_signature,
        )
    else:
        cond_silence_b, cond_full_b = cond_silence, cond_full
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

    logger.info("stream_create_start steps={} pipeline_depth={}", steps, depth)
    stream = session.stream(
        source=source,
        conditioning=conditioning,
        steps=steps,
        shift=3.0,
        pipeline_depth=depth,
    )
    logger.info("stream_handle_ready")
    _ms("stream_handle_ready")

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
        out = []
        for d in engine_obj.list_loras():
            # `metadata` is the full normalized record from the LoRA's
            # `<stem>.metadata.json` sidecar (falling back to a
            # synthesized record from `.trigger.txt`, or a sparse one
            # with id/name only when neither exists). The UI uses it for
            # the library tooltip, search, right-click "copy trigger",
            # and the visible-prepend logic. Cached by (path, mtime_ns)
            # so a catalog refresh is cheap.
            metadata = load_lora_metadata(d.path).to_wire()
            out.append({
                "id": d.id,
                "name": metadata.get("name") or d.name,
                "path": d.path,
                "state": d.state,
                "strength": d.strength,
                "materialized_bytes": d.materialized_bytes,
                "metadata": metadata,
            })
        return out

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
        # Active checkpoint identifier + its model-scale label ("2B" /
        # "5B" / null). The UI compares ``checkpoint_scale`` against
        # each LoRA's ``metadata.base_model_scale`` to hide LoRAs that
        # weren't trained for this checkpoint. ``null`` (unknown
        # checkpoint) disables filtering so undocumented checkpoints
        # don't accidentally hide every LoRA.
        "checkpoint": checkpoint,
        "checkpoint_scale": checkpoint_scale(checkpoint),
        # Active ring-buffer depth + the runtime-imposed ceiling
        # (TRT engine's hidden_states batch_max, or 4 for eager / compile).
        # The client clamps its depth control to [1, max_pipeline_depth]
        # and ships ``set_depth`` messages to retune live.
        "pipeline_depth": depth,
        "max_pipeline_depth": max_pipeline_depth,
        # Server-minted correlation id. The client should log this with
        # every local event (and pass it as a property on any analytics
        # events) so a pod-side log line and a browser-side trace for the
        # same complaint can be joined by session_id. Independent of any
        # client_id the client sent in its handshake.
        "session_id": session_id,
    }))
    ws.send(src_np.astype(np.float16).tobytes())
    if upload_stems is not None:
        _send_stem_payload(
            ws,
            fixture_name=fixture_name,
            source_mode=stem_source_mode,
            stems=upload_stems,
        )
    elif stem_error is not None:
        ws.send(json.dumps({
            "type": "stem_failed",
            "fixture_name": fixture_name or "",
            "error": stem_error,
        }))
    logger.info(
        "initial_buffer_sent duration_s={:.1f}",
        len(src_np) / SAMPLE_RATE,
    )
    _ms("initial_buffer_sent")

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
    # Prompt A/B blend. B is encoded at session start from config.prompt_b
    # (falls back to A when missing/equal, in which case the slider is a
    # no-op until the operator edits B and hits Send Tags). The pair is
    # encoded against the same source + timbre as A, and set_prompt_blend
    # lerps between them per tick.
    cond_pair_b_ref = [(cond_silence_b, cond_full_b)]
    prompt_text_b = [prompt_b]
    prompt_blend_ref = [0.0]
    # Optional uploaded timbre-track latent. None == use the playback
    # source's own latent (self-timbre, current default).
    timbre_latent_ref: list = [None]
    # Display name for the active timbre track (sent back in acks so the
    # client can show it). None when no override is active.
    timbre_name_ref: list = [None]

    def _active_refer_latent():
        tl = timbre_latent_ref[0]
        return tl if tl is not None else source_ref[0].latent

    def _refresh_conditioning():
        """Recompose ``stream.conditioning`` from the cached A/B pairs,
        current timbre strength, and current prompt blend. Two lerps
        when blend is in the open interval; one when it's at an extreme
        (``_blend_for_strength``'s own short-circuit handles that).
        Called from every site that changes any of those inputs."""
        cs_a, cf_a = cond_pair_ref[0]
        ca = _blend_for_strength(cs_a, cf_a, timbre_strength_ref[0])
        pb = prompt_blend_ref[0]
        if pb <= 0.001:
            stream.conditioning = ca
            return
        cs_b, cf_b = cond_pair_b_ref[0]
        cb = _blend_for_strength(cs_b, cf_b, timbre_strength_ref[0])
        if pb >= 0.999:
            stream.conditioning = cb
            return
        stream.conditioning = _blend_for_strength(ca, cb, pb)

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
            logger.debug(
                "structure_override_sidecar_hit name={}",
                struct_name_ref[0],
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
        prev_cond_pair_b = cond_pair_b_ref[0]
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
                name, samples=int(t_wf.shape[-1]),
            )
            if sc is not None:
                device = session.handler.device
                dtype = session.handler.dtype
                timbre_latent = Latent(
                    tensor=sc.latent.to(device, dtype).contiguous(),
                )
                logger.debug("timbre_sidecar_hit name={}", name)
            else:
                timbre_audio = Audio(
                    waveform=t_wf, sample_rate=SAMPLE_RATE,
                )
                logger.debug(
                    "timbre_vae_encode_start clip_s={:.1f} channels={}",
                    clip_s, t_wf.shape[0],
                )
                timbre_latent = session.encode_audio(timbre_audio)
                logger.debug(
                    "timbre_vae_encode_done latent_shape={}",
                    tuple(timbre_latent.tensor.shape),
                )
            timbre_latent_ref[0] = timbre_latent
            timbre_name_ref[0] = name
            cond_pair_ref[0] = _encode_cond_pair(
                prompt_text[0], timbre_latent,
                bpm_ref[0], duration_ref[0], key_ref[0],
                time_sig_ref[0],
            )
            # Re-encode B against the new timbre too — otherwise a non-
            # zero prompt blend would suddenly mix in B's *old-timbre*
            # conditioning the instant the user uploads a new ref.
            if prompt_text_b[0] != prompt_text[0]:
                cond_pair_b_ref[0] = _encode_cond_pair(
                    prompt_text_b[0], timbre_latent,
                    bpm_ref[0], duration_ref[0], key_ref[0],
                    time_sig_ref[0],
                )
            else:
                cond_pair_b_ref[0] = cond_pair_ref[0]
            _refresh_conditioning()
            return clip_s
        except Exception:
            timbre_latent_ref[0] = prev_timbre_latent
            timbre_name_ref[0] = prev_timbre_name
            cond_pair_ref[0] = prev_cond_pair
            cond_pair_b_ref[0] = prev_cond_pair_b
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

    # Live pipeline_depth retune. The StreamPipeline ring buffer can be
    # resized between ticks; doing it from the recv thread would race the
    # ongoing tick (slots may be mid-step), so we stash the target depth
    # here and the runner thread applies it inside before_tick. A single
    # slot is enough — a fresh value just replaces any unapplied one.
    pending_depth_ref: list[int | None] = [None]
    pending_depth_lock = threading.Lock()
    current_depth_ref: list[int] = [int(depth)]

    # Last meaningful-activity timestamp. Read by PipelineRunner to
    # decide whether to skip the DiT tick this iteration. The web client
    # resends a full ``params`` message every 8 ms via useParamSync
    # (mirrors DEMON's _sendTick — the engine samples params at the
    # start of each generation window, so the client floods them even
    # when nothing changed). Treating that flood as "activity" defeats
    # the pause entirely. Instead, ``params`` messages bump the timer
    # only when their ``raw`` dict differs from the previous one;
    # all other message types are discrete actions and always bump.
    # Plain list-wrapped float / dict: atomic enough under the GIL for
    # this single-slot rendezvous.
    last_activity_ts: list[float] = [time.monotonic()]
    _last_params_raw_ref: list = [None]

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
                logger.info("lora_disabled id={}", lid)
            except Exception as e:
                logger.exception("lora_disable_failed id={} error={}", lid, e)
        for lid, strength in local_enable:
            try:
                engine_obj.enable_lora(lid, strength=strength)
                logger.info(
                    "lora_enabled id={} strength={}",
                    lid, strength,
                )
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
                logger.exception("lora_enable_failed id={} error={}", lid, e)
        _send_catalog_update()
        # No automatic re-encode here. With WYSIWYG prompts, the trigger
        # word lives in the visible promptA/promptB text. The client's
        # visible-prepend logic (when `auto_prepend_lora_triggers` is on)
        # mutates the prompt on toggle and sends a prompt-update message,
        # which routes through the normal prompt-change path. If the flag
        # is off, the user explicitly opted into not auto-injecting the
        # trigger and we must not encode it behind their back.

    # --- on_audio_ready: delta-encode and send to client ---
    # Two call shapes:
    #   * Windowed (``win_start is not None``): ``wav_np`` is the patched
    #     window region — shape ``[win_end - win_start, channels]``. The
    #     runner has already written it into ``audio_eng`` via
    #     ``patch_window``, so we skip ``audio_eng.swap`` entirely
    #     (eliminates the full-buffer ``self.current.copy()`` that used to
    #     fire on every windowed decode — ~23 MB / call at 60 s buffer).
    #   * Full-buffer (``win_start is None``): legacy path; ``wav_np`` is
    #     the whole new buffer and we route it through
    #     ``audio_eng.swap`` for the global crossfade.
    def on_audio_ready(wav_np, win_start=None, win_end=None):
        client_mirror = client_mirror_ref[0]
        if win_start is not None:
            ss = int(win_start)
            se = min(int(win_end), ss + len(wav_np), len(client_mirror))
            if se <= ss:
                return
            region = wav_np[: se - ss]
        else:
            audio_eng.swap(wav_np)
            ss = 0
            se = min(len(wav_np), len(client_mirror))
            if se <= ss:
                return
            region = wav_np[ss:se]
        mirror_region = client_mirror[ss:se]

        if not _first_slice[0]:
            _first_slice[0] = True
            _ms("first_generated_slice")

        # Delta = what server has now minus what client has
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

    # --- Control bus ---
    # External commands (from the demo's onboard MCP server) land in this
    # queue and get dispatched through the same _dispatch_message handler
    # as live WebSocket frames. The MCP holds an HTTP control channel to
    # the server process, not a separate WebSocket, so the browser's WS
    # stays the single audio/video stream owner and the front-end can
    # mirror MCP-driven state via the same ack messages it already listens
    # to (plus a new ``params_echo`` for raw knob changes).
    control_queue: queue.Queue = queue.Queue()
    # session_id was minted at the top of handle_client so it can be bound
    # into loguru's contextvars before any session-scoped logging happens.

    def inject_control(data: dict, audio: bytes | None = None) -> None:
        control_queue.put((data, audio))

    def snapshot_session() -> dict:
        return {
            "id": session_id,
            "prompt": prompt_text[0],
            "prompt_b": prompt_text_b[0],
            "prompt_blend": prompt_blend_ref[0],
            "duration": duration_ref[0],
            "bpm": bpm_ref[0],
            "key": key_ref[0],
            "time_signature": time_sig_ref[0],
            "fixture_name": fixture_name,
            "timbre_name": timbre_name_ref[0],
            "timbre_strength": timbre_strength_ref[0],
            "structure_name": struct_name_ref[0],
            "lora_catalog": _catalog_payload(),
            "knob_values": virtual_knobs.get_all_values(),
            "channels": n_channels_ref[0],
            "sample_rate": SAMPLE_RATE,
        }

    def _apply_ref(
        kind: str,
        name: str,
        waveform_fn,
        origin: str,
    ) -> None:
        """Shared load → apply → ack flow for the four
        ``set_{timbre,structure}_{source,fixture}`` branches.

        ``waveform_fn`` is the only thing that varies across them: a
        thunk returning the decoded waveform tensor — either
        ``_decode_audio_msg`` on a binary frame or
        ``_load_fixture_waveform`` on a known fixture name. ``origin``
        is just the log label distinguishing "source" (audio-over-wire)
        vs "fixture" (server-side resolved) so the failure trace points
        at the right WS verb.
        """
        set_msg = f"{kind}_set"
        failed_msg = f"{kind}_failed"
        try:
            wf = waveform_fn()
            if kind == "timbre":
                clip_s = _apply_timbre_waveform(wf, name)
                extra = f"({clip_s:.1f}s)"
            else:
                clip_s, target_s = _apply_structure_waveform(wf, name)
                extra = f"({clip_s:.1f}s, fitted to {target_s:.1f}s)"
            with send_lock:
                ws.send(json.dumps({
                    "type": set_msg,
                    "name": name,
                    "duration": clip_s,
                }))
            logger.info(
                "ref_applied kind={} origin={} name={} detail={}",
                kind, origin, name, extra,
            )
        except Exception as exc:
            logger.opt(exception=True).error(
                "ref_apply_failed kind={} origin={} name={} error={}",
                kind, origin, name, exc,
            )
            try:
                with send_lock:
                    ws.send(json.dumps({
                        "type": failed_msg,
                        "error": str(exc),
                    }))
            except Exception:
                pass

    def _dispatch_message(
        data: dict,
        recv_audio,
        source: str,
    ) -> None:
        """Handle one parsed control message.

        ``recv_audio`` returns the next binary audio frame. For
        WebSocket-sourced messages it's ``ws.recv``; for control-bus
        messages it's a thunk that returns the pre-loaded bytes the MCP
        sent alongside the JSON.

        ``source`` is ``"ws"`` for the browser's own WebSocket and
        ``"control"`` for control-bus messages — used to gate
        ``params_echo`` so the browser only sees echoes of external
        changes (it owns its own params already).
        """
        mtype = data.get("type")
        # Activity gating for the idle-pause runner. ``params`` is a
        # client heartbeat (re-sent every 8 ms by useParamSync even
        # when nothing changed), so we only count it as activity when
        # ``raw`` actually differs from the previous one we saw —
        # otherwise the pause would never engage. Every other message
        # type represents a discrete action (LoRA toggle, source swap,
        # prompt change, etc.) and always counts as activity.
        # ``playback_pos`` on params messages advances every tick but
        # is intentionally excluded from the diff: it's a clock, not
        # user input.
        if mtype == "params":
            _new_raw = data.get("raw") or {}
            if _new_raw != _last_params_raw_ref[0]:
                last_activity_ts[0] = time.monotonic()
                _last_params_raw_ref[0] = dict(_new_raw)
                # DEBUG so 125 Hz heartbeats stay invisible at INFO; surfaces
                # the actual knob diffs when DEMON_LOG_LEVEL=DEBUG. Logs
                # post-diff so we don't fire on the no-op heartbeat.
                logger.debug(
                    "params_changed origin={} raw_keys={}",
                    source, sorted(_new_raw.keys()),
                )
        else:
            last_activity_ts[0] = time.monotonic()
        if mtype == "params":
            raw = data.get("raw") or {}
            if source == "control":
                # Don't apply server-side: the browser owns the Smooth
                # tween, and applying here would step virtual_knobs to
                # the target immediately and then watch useParamSync
                # send the tween back from the old value forward,
                # producing a jump-then-rewind on the engine. Echo
                # only; the browser tweens sliderValues toward this
                # target and ships the smoothed sequence back over WS
                # as a normal "params" message (source="ws") that
                # lands in virtual_knobs the usual way.
                try:
                    with send_lock:
                        ws.send(json.dumps({
                            "type": "params_echo",
                            "raw": dict(raw),
                        }))
                except ConnectionClosed:
                    running[0] = False
                except Exception:
                    pass
            else:
                try:
                    pp = float(data.get("playback_pos", 0.0))
                except (TypeError, ValueError):
                    pp = 0.0
                virtual_knobs.update(raw)
                try:
                    audio_eng.position = int(pp * SAMPLE_RATE) % max(
                        1, len(audio_eng.current)
                    )
                except Exception:
                    pass
        elif mtype == "loop_band":
            # Client armed / moved / cleared a loop band. The worklet
            # replays only [start_sec, end_sec] (wrapping end→start) while
            # the pipeline keeps generating; storing the band here lets the
            # runner wrap its predictive decode target inside the band so
            # the seam after ``start`` is regenerated before the playhead
            # loops back to it. Null / degenerate range clears the band
            # (linear chase resumes). Stored as a plain (start, end) tuple
            # of seconds; the runner clamps it to the live buffer length.
            try:
                s = data.get("start_sec")
                e = data.get("end_sec")
                if s is None or e is None or float(e) - float(s) <= 0.0:
                    audio_eng.loop_band = None
                else:
                    audio_eng.loop_band = (float(s), float(e))
            except (TypeError, ValueError):
                audio_eng.loop_band = None
        elif mtype == "prompt":
            ts_override = _normalize_time_signature(data.get("time_signature"))
            if ts_override is not None:
                time_sig_ref[0] = ts_override
            refer = _active_refer_latent()
            key_used = data.get("key") or key_ref[0]
            logger.info(
                "prompt_set origin={} tags={!r} tags_b={!r} key={} time_signature={}",
                source, data.get("tags"), data.get("tags_b"),
                key_used, time_sig_ref[0],
            )
            cond_pair_ref[0] = _encode_cond_pair(
                data["tags"], refer, bpm_ref[0], duration_ref[0],
                key_used, time_sig_ref[0],
            )
            prompt_text[0] = data["tags"]
            tags_b = data.get("tags_b")
            if tags_b and tags_b != data["tags"]:
                cond_pair_b_ref[0] = _encode_cond_pair(
                    tags_b, refer, bpm_ref[0], duration_ref[0],
                    key_used, time_sig_ref[0],
                )
                prompt_text_b[0] = tags_b
            else:
                cond_pair_b_ref[0] = cond_pair_ref[0]
                prompt_text_b[0] = data["tags"]
            _refresh_conditioning()
            try:
                with send_lock:
                    ws.send(json.dumps({
                        "type": "prompt_applied",
                        "tags": data["tags"],
                    }))
            except ConnectionClosed:
                running[0] = False
        elif mtype == "set_prompt_blend":
            try:
                v = float(data.get("value", 0.0))
            except (TypeError, ValueError):
                v = 0.0
            v = max(0.0, min(1.0, v))
            if source == "control":
                # Same shape as the params echo path: the browser owns
                # the smoothed prompt_blend slider, so just mirror the
                # target back and let usePromptBlendSync ship the
                # tweened sequence to the server.
                try:
                    with send_lock:
                        ws.send(json.dumps({
                            "type": "prompt_blend_echo",
                            "value": v,
                        }))
                except ConnectionClosed:
                    running[0] = False
                except Exception:
                    pass
            else:
                prompt_blend_ref[0] = v
                _refresh_conditioning()
                # DEBUG because the browser tween fires many updates per
                # slider drag — INFO would flood any active session.
                logger.debug(
                    "prompt_blend_set origin={} value={:.3f}", source, v,
                )
        elif mtype == "set_depth":
            try:
                v = int(data.get("value"))
            except (TypeError, ValueError):
                return
            v = max(MIN_PIPELINE_DEPTH, min(v, max_pipeline_depth))
            with pending_depth_lock:
                pending_depth_ref[0] = v
            logger.info(
                "set_depth_requested origin={} value={}", source, v,
            )
        elif mtype == "enable_lora":
            lid = data.get("id")
            s = data.get("strength")
            try:
                strength = float(s) if s is not None else None
            except (TypeError, ValueError):
                strength = None
            if lid:
                with pending_lock:
                    pending_enable.append((str(lid), strength))
                logger.info(
                    "enable_lora_requested origin={} id={} strength={}",
                    source, lid, strength,
                )
        elif mtype == "disable_lora":
            lid = data.get("id")
            if lid:
                with pending_lock:
                    pending_disable.append(str(lid))
                logger.info(
                    "disable_lora_requested origin={} id={}", source, lid,
                )
        elif mtype == "set_timbre_strength":
            try:
                v = float(data.get("value", 1.0))
            except (TypeError, ValueError):
                v = 1.0
            v = max(0.0, min(1.0, v))
            timbre_strength_ref[0] = v
            _refresh_conditioning()
            # DEBUG — slider drag fires many updates per gesture.
            logger.debug(
                "timbre_strength_set origin={} value={:.3f}", source, v,
            )
        elif mtype == "set_timbre_source":
            name = data.get("name") or "timbre"
            logger.info(
                "set_timbre_source_recv origin={} name={}", source, name,
            )
            try:
                audio_msg = recv_audio()
            except ConnectionClosed:
                running[0] = False
                return
            logger.debug(
                "set_timbre_source_bytes_received name={} bytes={}",
                name, len(audio_msg),
            )
            _apply_ref(
                "timbre", name,
                lambda: _decode_audio_msg(audio_msg),
                "source",
            )
        elif mtype == "set_timbre_fixture":
            name = data.get("name", "")
            logger.info(
                "set_timbre_fixture origin={} name={}", source, name,
            )
            _apply_ref(
                "timbre", name,
                lambda: _load_fixture_waveform(name),
                "fixture",
            )
        elif mtype == "clear_timbre_source":
            timbre_latent_ref[0] = None
            timbre_name_ref[0] = None
            refer = source_ref[0].latent
            cond_pair_ref[0] = _encode_cond_pair(
                prompt_text[0], refer,
                bpm_ref[0], duration_ref[0], key_ref[0],
                time_sig_ref[0],
            )
            if prompt_text_b[0] != prompt_text[0]:
                cond_pair_b_ref[0] = _encode_cond_pair(
                    prompt_text_b[0], refer,
                    bpm_ref[0], duration_ref[0], key_ref[0],
                    time_sig_ref[0],
                )
            else:
                cond_pair_b_ref[0] = cond_pair_ref[0]
            _refresh_conditioning()
            try:
                with send_lock:
                    ws.send(json.dumps({"type": "timbre_cleared"}))
            except Exception:
                pass
            logger.info("timbre_cleared origin={}", source)
        elif mtype == "set_structure_source":
            name = data.get("name") or "structure"
            logger.info(
                "set_structure_source_recv origin={} name={}",
                source, name,
            )
            try:
                audio_msg = recv_audio()
            except ConnectionClosed:
                running[0] = False
                return
            logger.debug(
                "set_structure_source_bytes_received name={} bytes={}",
                name, len(audio_msg),
            )
            _apply_ref(
                "structure", name,
                lambda: _decode_audio_msg(audio_msg),
                "source",
            )
        elif mtype == "set_structure_fixture":
            name = data.get("name", "")
            logger.info(
                "set_structure_fixture origin={} name={}", source, name,
            )
            _apply_ref(
                "structure", name,
                lambda: _load_fixture_waveform(name),
                "fixture",
            )
        elif mtype == "clear_structure_source":
            _clear_struct_override()
            try:
                with send_lock:
                    ws.send(json.dumps({"type": "structure_cleared"}))
            except Exception:
                pass
            logger.info("structure_cleared origin={}", source)
        elif mtype == "swap_source":
            tags = data.get("tags") or prompt_text[0]
            try:
                audio_msg = recv_audio()
            except ConnectionClosed:
                running[0] = False
                return
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
                swap_pending["stem_source_mode"] = (
                    normalize_stem_source_mode(
                        data.get("stem_source_mode")
                    )
                )
        else:
            # Unknown mtype — log but don't crash; lets future protocol
            # additions degrade gracefully on older servers.
            logger.warning(
                "unknown_message_type origin={} mtype={}", source, mtype,
            )

    # --- recv loop: drain WS + control bus into _dispatch_message ---
    def recv_loop():
        while running[0]:
            try:
                while True:
                    msg = ws.recv(timeout=0.001)
                    if isinstance(msg, str):
                        try:
                            data = json.loads(msg)
                        except Exception:
                            continue
                        try:
                            _dispatch_message(data, ws.recv, "ws")
                        except Exception as exc:
                            logger.exception(
                                "ws_dispatch_error error={}", exc,
                            )
                    if not running[0]:
                        break
            except TimeoutError:
                pass
            except ConnectionClosed:
                running[0] = False
                break
            except Exception as exc:
                logger.exception("recv_loop_error error={}", exc)
                running[0] = False
                break

            # Drain the MCP / external control bus. Audio bytes (if any)
            # were stashed at injection time, so recv_audio is a thunk
            # returning the pre-loaded buffer rather than a WS recv.
            while True:
                try:
                    cdata, caudio = control_queue.get_nowait()
                except queue.Empty:
                    break
                _audio_buf = caudio if caudio is not None else b""
                try:
                    _dispatch_message(
                        cdata,
                        lambda _b=_audio_buf: _b,
                        "control",
                    )
                except Exception as exc:
                    logger.exception(
                        "control_dispatch_error error={}", exc,
                    )

    # Forward decl so closures defined above (e.g. _apply_struct_override
    # via the recv thread) can resolve the cell without NameError before
    # the runner is constructed at the bottom of this function. The slot
    # is None until ``runner_holder[0] = runner`` lands; callers null-
    # check before invoking runner methods.
    runner_holder: list = [None]

    # spawn_thread copies the parent context (loguru contextvars), so
    # logs emitted from inside recv_loop still carry session_id and
    # friends — plain threading.Thread would drop the binding.
    recv_t = spawn_thread(recv_loop, name="recv_loop")

    # Register with the process-global session registry so the demo's
    # onboard MCP server can drive this session via the HTTP control bus.
    session_registry.register(session_registry.SessionHandle(
        id=session_id,
        started_at=time.time(),
        inject=inject_control,
        snapshot=snapshot_session,
    ))
    logger.info("session_registered")

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
            new_stem_source_mode = resolve_upload_stem_source_mode(
                new_fixture_name,
                swap_pending.get("stem_source_mode"),
                known_fixtures=KNOWN_FIXTURES,
            )
            if audio_msg is None:
                return
            swap_pending["bytes"] = None
            swap_pending["tags"] = None
            swap_pending["key"] = None
            swap_pending["time_signature"] = None
            swap_pending["fixture_name"] = None
            swap_pending["stem_source_mode"] = None
        # Initialized to None so the finally below can None-guard cleanly
        # in the (rare) case an exception fires between the start of the
        # try and the contextualize bind.
        _swap_ctx = None
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
            # Bind the *new* track on top of the session-scoped binding
            # so any error during the swap body (VAE encode, profile
            # mgmt, prepare_source) carries the track the user *tried*
            # to swap to — not the previous one. Scoped to the swap body
            # only: when this CM exits, loguru's contextvar stack
            # restores the initial-track binding from session setup, so
            # a post-swap pipeline_error still shows the *initial*
            # fixture (known v1 limitation — updating mid-stack would
            # require an out-of-order contextvar reset that corrupts
            # the stack). Manual __enter__ to avoid re-indenting the
            # ~190-line swap body; matching __exit__ runs in the
            # finally below.
            _swap_ctx = logger.contextualize(
                fixture_name=new_fixture_name or None,
                audio_duration_s=round(new_audio_duration_s, 2),
            )
            _swap_ctx.__enter__()
            logger.info(
                "source_swap_start duration_s={:.1f} channels={} "
                "fixture_name={} tags={!r}",
                new_audio_duration_s, new_wf.shape[0],
                new_fixture_name, tags,
            )

            # Profile swap (no-op when the new duration fits the same
            # profile that's currently loaded). Must run BEFORE
            # prepare_source: VAE-encode is the first GPU consumer and
            # needs the new vae_encode engine bound to its cache.
            #
            # Walk mode pins decoder + vae_decode at walk_window_s while
            # sizing vae_encode to the full new source (same mix that
            # the initial walk-mode wiring at session start uses). The
            # plain ensure_profile call would swap to the source's
            # natural profile and reload the larger decoder, which loses
            # the bf16 hybrid fix and stalls the producer during the
            # mid-playback engine swap.
            if profile_mgr is not None:
                try:
                    if walk_window:
                        profile_mgr.ensure_walk_profile(
                            walk_window_s=walk_window_s,
                            source_duration_s=new_audio_duration_s,
                        )
                    else:
                        profile_mgr.ensure_profile(new_audio_duration_s)
                except EngineNotBuiltError as exc:
                    logger.error(
                        "source_swap_aborted reason=engine_not_built error={}",
                        exc,
                    )
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
                    key_override=requested_key,
                    time_signature_override=requested_time_sig,
                )
            )
            new_upload_stems: dict[str, torch.Tensor] | None = None
            new_stem_error: str | None = None
            new_upload_stems, new_stem_error, new_source, new_wf = (
                _extract_and_select_upload_stem(
                    new_wf,
                    session=session,
                    source=new_source,
                    source_mode=new_stem_source_mode,
                    log_context="swap",
                )
            )
            if new_stem_error is not None and new_stem_source_mode != "full":
                with send_lock:
                    ws.send(json.dumps({
                        "type": "swap_failed",
                        "error": f"Stem extraction failed: {new_stem_error}",
                    }))
                return
            # Use the active timbre reference if one is uploaded; otherwise
            # the new playback source's own latent. Override persists
            # across source swaps.
            stream.source = new_source
            source_ref[0] = new_source
            playback_samples_ref[0] = int(new_wf.shape[-1])
            tl = timbre_latent_ref[0]
            refer = tl if tl is not None else new_source.latent
            cond_pair_ref[0] = _encode_cond_pair(
                tags,
                refer,
                new_bpm, new_audio_duration_s, new_key, new_time_sig,
            )
            # Carry promptB across the swap so the blend slider keeps
            # its meaning. If B was identical to A pre-swap, keep it
            # mirrored to skip a second encode pass.
            if prompt_text_b[0] != prompt_text[0]:
                cond_pair_b_ref[0] = _encode_cond_pair(
                    prompt_text_b[0],
                    refer,
                    new_bpm, new_audio_duration_s, new_key, new_time_sig,
                )
            else:
                cond_pair_b_ref[0] = cond_pair_ref[0]
                prompt_text_b[0] = tags
            stream.context_latent = new_source.context_latent
            # Re-derive structure override against the new source length.
            # On failure (e.g. VAE engine couldn't fit the new clip), drop
            # the override rather than block the swap — the user can re-
            # upload after the swap settles.
            if struct_audio_ref[0] is not None:
                try:
                    _apply_struct_override()
                except Exception as exc:
                    logger.exception(
                        "swap_struct_override_dropped error={}", exc,
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
            _refresh_conditioning()
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
            # A loop band from the previous song is meaningless against the
            # new buffer — drop it so the runner chases linearly until the
            # client re-arms a band on the new track.
            audio_eng.loop_band = None

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
                    # Echo the requested fixture / upload label so the
                    # client can mirror MCP-driven source swaps into the
                    # fixture dropdown (useMcpMirror). Falls back to the
                    # last-known fixture for swaps that didn't carry a
                    # name (legacy clients).
                    "fixture_name": new_fixture_name,
                }))
                ws.send(new_src_np.astype(np.float16).tobytes())
                if new_upload_stems is not None:
                    _send_stem_payload(
                        ws,
                        fixture_name=new_fixture_name,
                        source_mode=new_stem_source_mode,
                        stems=new_upload_stems,
                    )
                elif new_stem_error is not None:
                    ws.send(json.dumps({
                        "type": "stem_failed",
                        "fixture_name": new_fixture_name or "",
                        "error": new_stem_error,
                    }))
            logger.info(
                "source_swap_complete duration_s={:.1f}",
                len(new_src_np) / SAMPLE_RATE,
            )
        except ConnectionClosed:
            running[0] = False
        except Exception as exc:
            logger.opt(exception=True).error(
                "source_swap_error error={}", exc,
            )
            try:
                with send_lock:
                    ws.send(json.dumps({
                        "type": "swap_failed",
                        "error": str(exc),
                    }))
            except Exception:
                pass
        finally:
            # Always pop the swap-scoped contextualize, whether the swap
            # body completed, raised, or hit ConnectionClosed. None-guard
            # for the early-fail window before _swap_ctx was bound.
            if _swap_ctx is not None:
                try:
                    _swap_ctx.__exit__(None, None, None)
                except Exception:
                    pass

    def apply_depth_pending():
        with pending_depth_lock:
            target = pending_depth_ref[0]
            pending_depth_ref[0] = None
        if target is None or target == current_depth_ref[0]:
            return
        pipe = stream.pipeline
        if pipe is None:
            # First tick hasn't built the pipeline yet — re-queue and try
            # again next iteration. set_depth on a missing pipeline would
            # silently no-op.
            with pending_depth_lock:
                if pending_depth_ref[0] is None:
                    pending_depth_ref[0] = target
            return
        try:
            pipe.set_depth(target)
            current_depth_ref[0] = pipe.depth
            logger.info("pipeline_depth_applied depth={}", pipe.depth)
        except Exception as exc:
            logger.exception(
                "set_depth_failed target={} error={}", target, exc,
            )
            return
        try:
            with send_lock:
                ws.send(json.dumps({
                    "type": "depth_applied",
                    "value": current_depth_ref[0],
                }))
        except ConnectionClosed:
            running[0] = False
        except Exception:
            pass

    # Combined before_tick callback.  Both kinds of cross-thread
    # mutation (LoRA enable/disable refits and source swaps) are GPU-
    # bound and must run on the runner thread between ticks.  Drain
    # both queues each iteration so they share one rendezvous point.
    def apply_pending():
        apply_lora_pending()
        apply_swap_if_pending()
        apply_depth_pending()

    # --- PipelineRunner: the SAME code as local ---
    runner = PipelineRunner(
        session, stream, audio_eng,
        last_activity_ts=last_activity_ts,
        idle_threshold_s=IDLE_PAUSE_S,
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
        logger.info("pipeline_running")
        runner.run()
    except Exception as exc:
        logger.opt(exception=True).error("pipeline_error error={}", exc)
    finally:
        running[0] = False
        session_registry.unregister(session_id)
        recv_t.join(timeout=2)
        logger.info(
            "client_disconnected num_gens={}",
            params.get("num_gens", 0),
        )

        # Tear down per-session GPU state. Order matters: stream.close()
        # drops the StreamPipeline's references into the engine before
        # session.close() actually destroys the engine + ModelContext.
        # session.close() ends with gc.collect() + cuda.empty_cache().
        try:
            stream.close()
        except Exception as exc:
            logger.warning("stream_close_raised error={}", exc)
        try:
            session.close()
        except Exception as exc:
            logger.warning("session_close_raised error={}", exc)

        # The session-scoped and track-scoped contextvar bindings live on
        # the ExitStack owned by handle_client; it unwinds them in
        # reverse order on return, so this body doesn't need any
        # explicit __exit__ calls here.


# ---------------------------------------------------------------------------
# Startup self-warmup
#
# Measured 2026-05-18: a cold first session on a fresh engine takes ~40s to
# `ready` (TRT decoder-engine load ~3s + LoRA-refit manager ~7s + Session /
# ModelContext / conditioning ~10s + first-tick pipeline build), while the
# *second* session on the same warm engine is ~5-6s. ~30s of the cold path is
# one-time-after-engine-start state that persists in the process. Driving one
# synthetic default-fixture session through `handle_client` at boot — before
# the pod accepts real traffic — pays that once so every real "begin" gets the
# warm path. Behaviour-neutral for real clients; the warmup session is fully
# torn down (stream.close()+session.close()) by handle_client's own finally.
# ---------------------------------------------------------------------------

WARMUP_STATE: dict = {"done": False, "error": None, "seconds": None}

_WARMUP_FIXTURE = "low_fi_Gm_loop_60s_gnm.wav"  # PREFERRED_DEFAULT_FIXTURE
_WARMUP_PROMPT = "ambient electronic, warm pads"


class _WarmupWS:
    """In-process synthetic WebSocket that drives one default-fixture
    session through handle_client to warm one-time engine state.

    Scripts the Phase-1 handshake (config JSON, then the audio frame),
    lets Phase-2 spin long enough to build the pipeline + run the first
    generation tick, then raises ConnectionClosed so handle_client's
    teardown path frees all per-session GPU state.
    """

    def __init__(self, config_json: str, audio_frame: bytes, budget_s: float = 35.0):
        self._queue = [config_json, audio_frame]
        self._t0 = time.monotonic()
        self._budget_s = budget_s
        self._initial_seen_at = None
        self.closed = False

    def recv(self, timeout=None):
        if self._queue:
            return self._queue.pop(0)
        # Phase-2: spin until warm budget elapsed, then end the session.
        # Mimic "no client message" via TimeoutError when the caller
        # passed a timeout (the streaming loop polls that way); raise
        # ConnectionClosed once warmed so every recv site unwinds into
        # handle_client's finally (which closes the session).
        now = time.monotonic()
        warm_enough = (
            now - self._t0 > self._budget_s
            or (self._initial_seen_at is not None
                and now - self._initial_seen_at > 10.0)
        )
        if warm_enough:
            raise ConnectionClosed(None, None)
        if timeout is not None:
            time.sleep(min(timeout, 0.05))
            raise TimeoutError
        time.sleep(0.1)
        raise TimeoutError

    def send(self, msg):
        # The initial buffer is a large binary frame (the echoed source);
        # spotting it tells us Phase-1 finished so we can bound Phase-2.
        if isinstance(msg, (bytes, bytearray)) and len(msg) > 1_000_000:
            if self._initial_seen_at is None:
                self._initial_seen_at = time.monotonic()

    def close(self, *args, **kwargs):
        self.closed = True


def _load_warmup_audio_frame() -> bytes:
    """Build the wire frame (<II channels,samples> + interleaved f32)
    for the default fixture, matching the browser's upload format."""
    from acestep.fixtures import audio_fixture

    path = str(audio_fixture(_WARMUP_FIXTURE))
    try:
        import soundfile as sf
        data, _sr = sf.read(path, dtype="float32", always_2d=True)  # (n, ch)
    except Exception:
        import librosa
        mono, _sr = librosa.load(path, sr=None, mono=True)
        data = np.stack([mono, mono], axis=1).astype(np.float32)
    if data.ndim == 1:
        data = np.stack([data, data], axis=1)
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    ch = int(data.shape[1])
    samples = int(data.shape[0])
    interleaved = np.ascontiguousarray(data, dtype=np.float32).reshape(-1)
    return struct.pack("<II", ch, samples) + interleaved.tobytes()


def run_startup_warmup(
    *,
    decoder_backend: str,
    vae_backend: str,
    checkpoint: str,
    offload_text_encoder: bool,
) -> None:
    """Drive one synthetic default-fixture session at boot. Never raises
    — a failed warmup must not stop the server from serving."""
    t0 = time.monotonic()
    warm_log = logger.bind(component="warmup")
    warm_log.info("warmup_start fixture={}", _WARMUP_FIXTURE)
    try:
        cfg = {
            "fixture_name": _WARMUP_FIXTURE,
            "prompt": _WARMUP_PROMPT,
            "steps": 8,
            "depth": 4,
            "lora": False,
            "enabled_loras": [],
        }
        frame = _load_warmup_audio_frame()
        ws = _WarmupWS(json.dumps(cfg), frame)
        handle_client(
            ws,
            decoder_backend=decoder_backend,
            vae_backend=vae_backend,
            checkpoint=checkpoint,
            offload_text_encoder=offload_text_encoder,
        )
        WARMUP_STATE["done"] = True
        WARMUP_STATE["seconds"] = round(time.monotonic() - t0, 1)
        warm_log.info(
            "warmup_done elapsed_s={}", WARMUP_STATE["seconds"],
        )
    except Exception as exc:
        WARMUP_STATE["error"] = repr(exc)
        WARMUP_STATE["seconds"] = round(time.monotonic() - t0, 1)
        warm_log.opt(exception=True).error(
            "warmup_failed elapsed_s={} error={}",
            WARMUP_STATE["seconds"], exc,
        )
