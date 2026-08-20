"""WebSocket transport adapter for the realtime motion-to-music demo.

Provides :func:`handle_client`, the per-WebSocket coroutine wired in by
:mod:`.server`. Wraps a
:class:`~acestep.streaming.session.StreamingSession` behind the existing
WS wire protocol:

- Decodes incoming JSON frames into typed session method calls.
- Subscribes to the session's
  :class:`~acestep.streaming.events.EventBus` and serializes each typed
  event to the matching wire frame(s) under a per-connection
  ``send_lock`` so JSON + binary follow-ups stay atomic.
- Owns per-subscriber transport state: the :class:`SliceCodec` (zstd
  context + ``client_mirror`` delta basis), the control-bus inject
  queue, the init-timing latches.

Init handshake (the wire's ``ready`` JSON + binary buffer + optional
``stem_assets``/``stem_failed``) ships inline BEFORE the bus
subscription drains, since those frames are produced synchronously by
:meth:`StreamingSession.create` and have nothing to fan out yet.

Wire-format details live in :mod:`.audio_codec`; operations and
lifecycle live in :mod:`acestep.streaming.session`. :mod:`.server`
imports :func:`handle_client` directly from here.
"""

import contextlib
import json
import math
import os
import queue
import socket
import struct
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import torch
from websockets.exceptions import ConnectionClosed

torch.set_grad_enabled(False)
torch._dynamo.config.disable = True

import numpy as np

from acestep.audio.key_detection import detect_key
from acestep.engine.obs import logger, spawn_thread
from acestep.engine.session import Session
from acestep.fixtures import KNOWN_FIXTURES
from acestep.nodes.types import Audio
from acestep.paths import (
    EngineNotBuiltError,
    checkpoint_scale,
    checkpoints_dir,
    loras_dir,
    max_profile_duration_s,
)
from acestep.sidecars import truncate_to_pool

from acestep.streaming.commands import CommandOrigin
from acestep.streaming.config import SessionConfig
from acestep.streaming.events import (
    AudioReady,
    AudioWriteFailed,
    AudioWritten,
    CommandFailed,
    DepthApplied,
    LoraCatalogUpdate,
    ManualSlotCount,
    ParamsEcho,
    PromptApplied,
    PromptBlendEcho,
    SessionError,
    StemAssets,
    StemFailed,
    StructureCleared,
    StructureFailed,
    StructureSet,
    SubscriberDropped,
    SwapFailed,
    SwapReady,
    TimbreCleared,
    TimbreFailed,
    TimbreSet,
)
from acestep.streaming.session import (
    StemExtractFailedError,
    StreamingSession,
    UnsupportedTrtCheckpointError,
)
from acestep.streaming import registry as session_registry
from acestep.streaming.source import (
    _decode_audio_msg,
    _load_clip_waveform,
    _load_known_fixture_waveform,
    _normalize_time_signature,
)
from acestep.streaming.stems import (
    extract_upload_stems,
    finish_stems_pending,
    mark_stems_pending,
    stems_pending,
    wait_for_pending_stems,
)
from acestep.user_uploads import (
    UserUploadPacket,
    find_duplicate_upload,
    persist_user_upload_packet,
    persist_user_upload_stems,
    unique_user_upload_name,
)

from .audio_codec import SliceCodec, chunked_ws_send, send_stem_payload
from .protocol import COMMAND_NAMES, SAMPLE_RATE, coerce_command_payload


# ---------------------------------------------------------------------------
# Idle VRAM janitor
# ---------------------------------------------------------------------------

_JANITOR_STARTED = False
_JANITOR_LOCK = threading.Lock()


def start_idle_vram_janitor(
    *,
    interval_s: float = 5.0,
    min_trim_bytes: int = 512 * 2**20,
) -> None:
    """Trim the CUDA caching allocator while no session is live.

    Session teardown frees its last tensors asynchronously: the recv
    thread is joined with a timeout, and polygraphy/TRT finalizer
    chains can release buffers seconds after the connection handler
    returned — after every in-band ``empty_cache()`` call has already
    run. Those late frees land in PyTorch's caching pool and stay
    reserved against the driver for as long as the pod idles
    (measured: ~3 GB after a plain 60 s session, ~5 GB after one that
    swapped to the 240 s decoder profile). That reserved-but-unused
    pool is invisible to TensorRT, whose workspace comes from
    cudaMalloc, so it directly eats the headroom the next session's
    engine loads and stem extraction need.

    A point-in-time trim can't win that race, so this janitor owns it:
    every ``interval_s`` it checks that no session is registered and,
    when the pool holds more than ``min_trim_bytes`` of freed blocks,
    runs ``gc.collect()`` + ``torch.cuda.empty_cache()``. Idle-only by
    construction — it never competes with a live session's allocator.
    """
    global _JANITOR_STARTED
    with _JANITOR_LOCK:
        if _JANITOR_STARTED:
            return
        _JANITOR_STARTED = True

    def _loop() -> None:
        import gc

        while True:
            time.sleep(interval_s)
            try:
                if session_registry.list_handles():
                    continue
                if not torch.cuda.is_available():
                    continue
                reserved = torch.cuda.memory_reserved()
                allocated = torch.cuda.memory_allocated()
                if reserved - allocated < min_trim_bytes:
                    continue
                gc.collect()
                torch.cuda.empty_cache()
                freed = reserved - torch.cuda.memory_reserved()
                logger.info(
                    "idle_vram_trim freed_mib={:.0f} reserved_mib={:.0f} "
                    "allocated_mib={:.0f}",
                    freed / 2**20,
                    torch.cuda.memory_reserved() / 2**20,
                    torch.cuda.memory_allocated() / 2**20,
                )
            except Exception as exc:  # never kill the janitor
                logger.warning("idle_vram_trim_failed error={}", exc)

    threading.Thread(
        target=_loop, name="idle-vram-janitor", daemon=True,
    ).start()


# ---------------------------------------------------------------------------
# Canonical user-upload packet processing
# ---------------------------------------------------------------------------

# A single eager encoder Session per checkpoint, shared across all
# connections and held for the process lifetime. This is the lone shared
# mutable GPU object in the system (every StreamingSession otherwise owns
# its own Session). The tradeoff is deliberate: encoding uploads needs the
# VAE encoder, which the streaming TRT path doesn't expose, so we keep a
# second eager copy of the weights rather than rebuild one per upload.
# Two costs follow from the sharing:
#   - VRAM: the eager weights occupy GPU memory WHILE AN UPLOAD IS IN
#     FLIGHT. Between uploads they are parked in system RAM
#     (ModelContext.offload_eager_to_cpu in _handle_upload_track's
#     finally); ModelContext._load_model_context lazily restores exactly
#     the modules the next upload touches. Without the parking, the
#     first upload would permanently pin ~6 GB next to the live
#     streaming session.
#   - Concurrency: prepare_source / stem extraction are NOT thread-safe on
#     a shared Session, so _UPLOAD_INFER_LOCK serializes all GPU work on it.
_UPLOAD_ENCODERS: dict[str, Session] = {}
_UPLOAD_ENCODERS_LOCK = threading.Lock()
_UPLOAD_INFER_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Single-active-session policy
# ---------------------------------------------------------------------------
#
# The rtmg backend is one-session-per-pod: the TRT VAE cache, the LoRA
# library, and the GPU budget are all sized for exactly one streaming
# session. Two concurrent ``StreamingSession.create`` calls stack two
# full model stacks (OOM on a 24 GB card), and either session's
# teardown evicts shared TRT VAE cache entries out from under the
# other — the failure cascade is: dual create → one OOMs → its cleanup
# evicts the shared engines → the HEALTHY session crashes on its next
# decode. Doubled connections happen routinely (page reload while the
# old socket is still draining, dev StrictMode double-mount, a stale
# tab auto-reconnecting), so the policy is enforced here:
#
#   - ``_SESSION_LIFECYCLE_LOCK`` serializes preempt+create: at most one
#     session is ever being constructed, and construction never overlaps
#     another session's teardown.
#   - A new main-session connection PREEMPTS the active session: its
#     runner is stopped, its WebSocket is closed with
#     ``PREEMPTED_CLOSE_CODE`` (the client treats that close as final —
#     no reconnect war between two tabs), and the new connection WAITS
#     for ``StreamingSession.closed`` so the old stack's VRAM is
#     actually free before the new stack loads.
#
# The ``upload_track`` side-channel WS never touches this policy.

_SESSION_LIFECYCLE_LOCK = threading.Lock()
_ACTIVE_SLOT_LOCK = threading.Lock()
_ACTIVE_SESSION: list = [None]  # [_ActiveSession | None]

# 4000-range application close code: "this session was replaced by a
# newer connection". The web client (web/sdk/protocol.ts +
# web/hooks/useStartSession.ts) recognizes it and does NOT enter the
# reconnect loop — reconnecting would just preempt the newer session
# back and ping-pong the pod through full session rebuilds.
#
# DEPLOYMENT ORDERING: server and client must ship together. A client
# built before this constant existed treats 4001 as an unexpected close
# and reconnects, recreating the ping-pong. Stale already-open tabs are
# the residual hazard until refreshed (docs/VOCALSTEM.md § one session
# per pod).
PREEMPTED_CLOSE_CODE = 4001

# Hard cap on client-supplied render_anchor_queue_s entries: at ~0.36 s of
# audio per queued window, 1024 windows is ~6 min of prewarm — far beyond any
# real kit — while bounding what a hostile/buggy client can enqueue.
_RENDER_ANCHOR_QUEUE_CAP = 1024

# How long a preempting connection waits for the old session's teardown
# to release VRAM. Generous: the old runner may be mid stem-extraction
# (it only observes running=False between pipeline iterations).
_PREEMPT_TEARDOWN_TIMEOUT_S = 45.0


# ---------------------------------------------------------------------------
# Dead-client watchdog
# ---------------------------------------------------------------------------
#
# The `websockets` sync send path holds the connection's protocol lock
# across a *blocking* ``socket.sendall`` with no send timeout
# (websockets/sync/connection.py: send_context → send_data; close_deadline
# is None during normal operation). So when a client stops reading — a
# killed tab, a slept laptop, or a Cloudflare named-tunnel half-open where
# the pod↔edge TCP stays ESTABLISHED while the browser is gone — the next
# slice ``ws.send`` wedges on a full TCP window and pins the protocol lock
# forever. The library's own keepalive ping (ping_timeout=90 in server.py)
# then deadlocks acquiring that same lock, so it never tears the dead
# client down: ``state.running`` stays True, the runner holds the pod's
# one-session seat, and ``/sessions`` never empties (measured: a frozen
# client left ~2.6 MB stuck in the server's send queue with the session
# still registered 13+ minutes later). Our own ConnectionClosed detectors
# can't fire either — the recv loop sees no close frame (no clean close
# from a half-open peer) and the send paths are the threads *stuck* in
# sendall.
#
# This watchdog detects the dead client from OUTSIDE those wedged Python
# I/O threads, using GIL- and lock-independent syscalls on the socket FD:
#
#   - Send stall: ``SIOCOUTQ`` reports bytes still queued in the kernel
#     send buffer (unsent + unacked), and ``TCP_INFO.tcpi_bytes_acked`` is
#     the kernel's monotonic count of peer-acked bytes. A live client's
#     KERNEL ACKs and drains regardless of how busy/GIL-starved the client
#     app is, so a slow link, a GC pause, or a briefly-backgrounded tab
#     keeps bytes_acked advancing (and/or the queue draining) and is never
#     reaped — only a positive queue whose bytes_acked AND backlog both go
#     nowhere for the whole grace window (a client reading nothing) trips
#     it. Keying on bytes_acked (not the SIOCOUTQ delta alone) closes the
#     plateau gap where a backpressured queue sits pinned at a constant
#     positive level while data still flows. This is the key distinction
#     from an ack-progress timeout (DEMON PR #293), which is app-level and
#     false-reaps a GIL-starved-but-alive client.
#   - Peer close: ``TCP_INFO`` state leaves ESTABLISHED (FIN/RST) or the FD
#     errors — the "0 established sockets but session still registered"
#     fingerprint, which means the library's recv thread missed the EOF.
#
# On either signal it flips ``state.running`` False (the runner observes
# it between iterations and exits run() → the registry unregister in
# run()'s finally fires) and force-shuts the socket so the wedged sendall
# (holding the protocol lock) and the recv thread both unblock at once,
# rather than waiting on the library's graceful-close handshake — which
# needs that same pinned lock and would just re-block.
#
# SIOCOUTQ and the TCP_INFO byte layout are Linux-specific; off Linux the
# syscalls would fail and reap every session, so the watchdog no-ops there.
_SIOCOUTQ = 0x5411  # Linux ioctl: unsent+unacked bytes in the send queue
_TCP_ESTABLISHED = 1
# tcpi_bytes_acked: __u64 at this offset in struct tcp_info on 64-bit Linux
# (stable since kernel 4.6); None if the running kernel's struct is shorter.
_TCPI_BYTES_ACKED_OFF = 120
_WATCHDOG_SUPPORTED = sys.platform.startswith("linux") and hasattr(
    socket, "TCP_INFO",
)

# How long the kernel send queue must stay positive AND non-draining
# before we declare the client dead. Generous against any real stall (a
# slow uplink, a stop-the-world GC, a throttled background tab all keep
# draining *something*); a genuinely gone client drains exactly zero for
# as long as the half-open socket survives. Reset on ANY drain.
try:
    _DEAD_CLIENT_SEND_STALL_S = max(
        5.0,
        float(os.environ.get("DEMON_DEAD_CLIENT_STALL_S", "") or 30.0),
    )
except ValueError:
    _DEAD_CLIENT_SEND_STALL_S = 30.0
_DEAD_CLIENT_POLL_S = 2.0

# How long a JSON header may wait for the binary frame that must follow it
# (see _recv_binary_payload). This is a LIVENESS guard — it exists so an
# orphan header can't block the recv loop forever and wedge the session —
# so any finite value does its job, and it must be sized against the
# largest payload the paired verbs can legitimately carry, not against a
# typical one. The verbs are set_timbre_source / set_structure_source /
# swap_source / write_audio, whose frames are uncompressed interleaved
# float32 at 48 kHz: a 60 s stereo clip is 23 MB, which needs a sustained
# ~18 Mbit/s uplink to land inside 10 s (the old value, inherited from
# write_audio when it was the only guarded verb — see the recv-hardening
# half of 2bb7969). Below that the guard stopped reporting "peer is gone"
# and started reporting "peer's uplink is ordinary", failing refs and
# swaps on links that stream perfectly well. 120 s covers 23 MB down to
# ~2 Mbit/s. The cost of the larger window is bounded and self-healing:
# _recv_binary_payload runs on the recv-loop thread, so a genuinely
# orphaned header now stalls command dispatch for that long, which trips
# the IDLE_PAUSE_S GPU pause — by design that keeps serving audio from the
# existing buffer and resumes on the next inbound message. Floored at the
# old value so a bad env can only loosen it.
try:
    _BINARY_PAYLOAD_TIMEOUT_S = max(
        10.0,
        float(os.environ.get("DEMON_BINARY_PAYLOAD_TIMEOUT_S", "") or 120.0),
    )
except ValueError:
    _BINARY_PAYLOAD_TIMEOUT_S = 120.0


def _known_fixture_local_path(fixture_name) -> str | None:
    """Local path of a known fixture IF it is already cached — never
    downloads (this feeds the provenance input hash; provenance must not
    add network work to session registration). None for user uploads,
    unknown names, or a cache miss."""
    if not fixture_name or fixture_name not in KNOWN_FIXTURES:
        return None
    try:
        from acestep.paths import fixtures_dir
        from acestep.track_assets import source_audio_path
        local = source_audio_path(fixtures_dir(), fixture_name)
        return str(local) if local.is_file() else None
    except Exception:  # noqa: BLE001 — fail-open, hash is best-effort
        return None


def _socket_send_backlog(sock) -> int | None:
    """Bytes queued in the kernel send buffer (unsent + unacked) for
    ``sock``, or ``None`` if the FD can't be queried (closed / errored /
    unsupported platform).

    A plain ``ioctl`` on the FD — it touches neither the GIL-contended
    Python recv path nor the WS protocol lock, so it stays accurate even
    while another thread is blocked in ``sendall`` on the same socket
    (exactly the state we need to detect)."""
    try:
        import fcntl

        buf = fcntl.ioctl(sock.fileno(), _SIOCOUTQ, struct.pack("I", 0))
        return struct.unpack("I", buf)[0]
    except (OSError, ValueError, AttributeError):
        return None


def _tcp_state_and_acked(sock) -> tuple[int, int | None] | None:
    """``(tcp_state, bytes_acked)`` from ``TCP_INFO`` for ``sock``, or
    ``None`` if the FD can't be queried (closed / errored).

    ``bytes_acked`` is the kernel's monotonic count of peer-acked bytes —
    it advances only while the peer is alive — or ``None`` when the running
    kernel's ``struct tcp_info`` is too short to carry it (pre-4.6), in
    which case the caller falls back to the SIOCOUTQ drain signal alone.
    Like the backlog probe, a lock-/GIL-independent syscall on the FD."""
    try:
        buf = sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_INFO, 128)
    except OSError:
        return None
    if not buf:
        return None
    acked = None
    if len(buf) >= _TCPI_BYTES_ACKED_OFF + 8:
        acked = struct.unpack_from("<Q", buf, _TCPI_BYTES_ACKED_OFF)[0]
    return buf[0], acked


def _windowed_slice_drop_reason(
    *,
    acked: int | None,
    sent: int,
    window_bytes: int,
    age_s: float,
    max_age_s: float,
) -> tuple[str, float] | None:
    """Decide whether a windowed slice should be shed for backpressure.

    Pure helper for the slice serializer's two-layer flow control so the
    decision can be unit-tested without the WS subscriber machinery:

      1. In-flight window (load-bearing): once the client has acked at
         least once, drop while sent-minus-acked exceeds the window.
      2. Bus-queue age: drop a slice that waited too long between publish
         and serialization (TCP itself pushing back).

    The window is checked first so an unbounded backlog sheds before the
    age backstop ever trips. Returns ``(reason, detail)`` for the drop
    log, or ``None`` to send the slice. ``acked is None`` (no ack yet,
    e.g. an old client) disables only the window layer."""
    if acked is not None:
        in_flight = sent - acked
        if in_flight > window_bytes:
            return "window", float(in_flight)
    if age_s > max_age_s:
        return "age", age_s
    return None


def _coalesced_slice_lead(prev, new) -> float | None:
    """Worst (min) ``slice_lead_s`` across two coalesced params messages, or
    ``None`` if neither carries a numeric lead.

    ``slice_lead_s`` is the worst lead since the previous report (wire
    contract) and the feedback controller widens playback lead on that worst
    value, so fold the min forward — newest-wins coalescing would drop a
    transient negative spike hiding in a superseded report (e.g. leads
    ``-1.25``, omitted, ``0.40`` would surface ``0.40``). The numeric guard is
    crash-safety: this runs before the consumer's ``float()`` coercion, and
    ``min()`` over a malformed str+float would throw and tear down the recv
    loop."""
    leads = [v for v in (prev, new) if isinstance(v, (int, float))]
    return min(leads) if leads else None


# Sticky params fields folded forward by the recv loop's newest-wins
# coalescing. "Absent retains" (the wire contract for these fields) only
# helps once a message is APPLIED — a superseded snapshot that carried a
# stationary anchor (or the batch anchor queue) would otherwise silently
# vanish whenever the recv thread drains a backlog, which is exactly when
# GPU-heavy generation starves it. The VST re-sends the scalar anchor every
# tick so it survived by accident; the one-shot queue does not.
_STICKY_PARAMS_FIELDS = ("render_anchor_s", "render_anchor_queue_s")


def _fold_sticky_params(pending: dict, newer: dict) -> dict:
    """Carry sticky fields from a superseded params snapshot into the newer
    one (in place) unless the newer snapshot sets its own value — including
    an explicit ``null`` (a clear), which counts as "sets its own"."""
    for key in _STICKY_PARAMS_FIELDS:
        if key in pending and key not in newer:
            newer[key] = pending[key]
    return newer


class _ActiveSession:
    __slots__ = ("session_id", "streaming", "ws")

    def __init__(self, session_id: str, streaming, ws):
        self.session_id = session_id
        self.streaming = streaming
        self.ws = ws


def _preempt_active_session(new_session_id: str) -> None:
    """Stop and drain the currently-active session, if any.

    Caller must hold ``_SESSION_LIFECYCLE_LOCK``. Returns once the old
    session has released its GPU state (or after a bounded wait with a
    warning — create proceeds either way; the OOM-retry paths downstream
    are the backstop)."""
    with _ACTIVE_SLOT_LOCK:
        prev = _ACTIVE_SESSION[0]
    if prev is None:
        return
    logger.info(
        "session_preempt prev={} new={} reason=single_session_policy",
        prev.session_id, new_session_id,
    )
    # Stop the runner; it observes this between pipeline iterations and
    # exits run() into close().
    prev.streaming.state.running = False
    # Close the old socket so its handler unblocks from any recv/send
    # and the client sees a deliberate, final close (not a 1006 blip).
    try:
        prev.ws.close(PREEMPTED_CLOSE_CODE, "preempted by a newer session")
    except Exception:
        pass
    if not prev.streaming.closed.wait(timeout=_PREEMPT_TEARDOWN_TIMEOUT_S):
        logger.warning(
            "session_preempt_teardown_timeout prev={} waited_s={}",
            prev.session_id, _PREEMPT_TEARDOWN_TIMEOUT_S,
        )
    else:
        logger.info("session_preempt_complete prev={}", prev.session_id)
    with _ACTIVE_SLOT_LOCK:
        if _ACTIVE_SESSION[0] is prev:
            _ACTIVE_SESSION[0] = None


def _log_session_vram(stage: str) -> None:
    from acestep.gpu_config import get_vram_telemetry

    telemetry = get_vram_telemetry()
    if telemetry is not None:
        logger.info(
            "session_vram stage={} free_gb={:.2f} available_gb={:.2f} "
            "allocated_gb={:.2f} reserved_gb={:.2f}",
            stage,
            telemetry["free_gb"],
            telemetry["available_gb"],
            telemetry["allocated_gb"],
            telemetry["reserved_gb"],
        )


def _strip_upload_encoder_generation_stack(session: Session) -> int:
    """Drop the parts of the upload encoder that uploads never execute.

    The upload path runs exactly three model surfaces: VAE encode,
    semantic extract (``model.tokenizer`` / ``model.detokenizer``), and
    the conditioning encoder (``model.encoder``). The DiT *decoder* —
    the bulk of the checkpoint — is generation-only, and the eager
    DiffusionEngine + LoRA manager exist only to drive it. Dropping
    both shrinks the encoder's GPU working set from a full second model
    copy (~4.7 GB restored per upload on the 2B turbo) to just the
    conditioning stack (~1.5 GB), which is what keeps phase-1 encodes
    inside the headroom of a live session running a long (120 s+) TRT
    profile. Returns the number of decoder parameters dropped.
    """
    import gc

    import torch.nn as nn

    handler = session.handler
    engine = getattr(handler, "_diffusion_engine", None)
    if engine is not None:
        try:
            engine.close()
        except Exception as exc:
            logger.warning("upload_encoder_engine_close_failed error={}", exc)
        handler._diffusion_engine = None
    dropped = 0
    model = handler.model
    decoder = getattr(model, "decoder", None) if model is not None else None
    if decoder is not None:
        dropped = sum(p.numel() for p in decoder.parameters())
        model.decoder = nn.Module()
    # The weights live in system RAM at this point (the encoder is
    # built parked); collect so the decoder's CPU copy is returned too.
    gc.collect()
    return dropped


def _upload_encoder_session(checkpoint: str) -> Session:
    with _UPLOAD_ENCODERS_LOCK:
        session = _UPLOAD_ENCODERS.get(checkpoint)
        if session is None:
            logger.info("upload_encoder_load_start checkpoint={}", checkpoint)
            # Load the eager weights straight to system RAM (offload
            # flags on) so building the encoder never spikes VRAM next
            # to the live streaming session...
            session = Session(
                project_root=str(checkpoints_dir()),
                config_path=checkpoint,
                decoder_backend="eager",
                vae_backend="eager",
                offload_to_cpu=True,
                offload_dit_to_cpu=True,
            )
            # ...then flip the context to RESIDENT mode so placement is
            # governed by the persistent-park protocol instead of
            # per-op round trips: _load_model_context lazily restores
            # exactly the modules an upload touches (the conditioning
            # stack for semantic extract; the VAE only when no TRT
            # engine fits), and _offload_upload_encoder parks them
            # again afterwards.
            session.handler.offload_to_cpu = False
            session.handler.offload_dit_to_cpu = False
            dropped = _strip_upload_encoder_generation_stack(session)
            _UPLOAD_ENCODERS[checkpoint] = session
            logger.info(
                "upload_encoder_loaded checkpoint={} placement=parked "
                "decoder_params_dropped={}",
                checkpoint, dropped,
            )
        return session


def _truncate_upload_waveform(waveform: torch.Tensor) -> torch.Tensor:
    max_samples = int(max_profile_duration_s() * SAMPLE_RATE)
    return truncate_to_pool(waveform[:2, :max_samples])


# BPM/key are global track properties; a centered window this long
# estimates them as well as the full signal (measured identical on
# 120 s material) at a fraction of the beat-tracker cost, and it caps
# the analysis latency of 240 s uploads.
ANALYSIS_WINDOW_S = 60.0


def _analysis_window(mono: np.ndarray, sample_rate: int) -> np.ndarray:
    max_n = int(ANALYSIS_WINDOW_S * sample_rate)
    if mono.shape[-1] <= max_n:
        return mono
    start = (mono.shape[-1] - max_n) // 2
    return mono[start:start + max_n]


def _analyze_upload_waveform(waveform: torch.Tensor) -> tuple[int, str, str]:
    import librosa

    mono_np = waveform.mean(dim=0).detach().cpu().numpy()
    window = _analysis_window(mono_np, SAMPLE_RATE)
    bpm_raw, _ = librosa.beat.beat_track(y=window, sr=SAMPLE_RATE)
    bpm = int(round(float(np.asarray(bpm_raw).flat[0])))
    return bpm, detect_key(window, SAMPLE_RATE), "4"


# Single worker: uploads are serialized end-to-end anyway; the pool
# exists so the CPU-bound analysis overlaps the GPU-bound source encode
# inside one upload's phase 1.
_UPLOAD_ANALYSIS_POOL = ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="upload-analysis",
)


def _read_files_into_page_cache(paths) -> int:
    """Sequentially read ``paths`` and discard, pulling them into the OS
    page cache. Returns total bytes read."""
    total = 0
    for p in {str(p) for p in paths}:
        with open(p, "rb") as f:
            while True:
                chunk = f.read(1 << 24)
                if not chunk:
                    break
                total += len(chunk)
    return total


def _prewarm_trt_engines_for_duration(
    duration_s: float,
    *,
    checkpoint: str,
    decoder_backend: str,
    vae_backend: str,
):
    """Best-effort, in the background: page-cache the TRT engine files
    the post-upload swap will load for ``duration_s``. When the swap
    crosses profiles (e.g. 60 s session → 120 s upload) the engine load
    is dominated by a multi-GB disk read; warming it during phase 1
    makes the swap land seconds sooner. No-op for eager backends or
    unbuilt engines."""
    if "tensorrt" not in (decoder_backend, vae_backend):
        return None

    def _read() -> None:
        try:
            from acestep.engine.trt.profile_manager import TRTProfileManager
            from acestep.paths import _DEFAULT_TRT_CHECKPOINT

            mgr = TRTProfileManager(
                decoder_backend=decoder_backend,
                vae_backend=vae_backend,
                checkpoint=(
                    checkpoint if decoder_backend == "tensorrt"
                    else _DEFAULT_TRT_CHECKPOINT
                ),
            )
            paths, picked_dur = mgr.resolve(float(duration_s))
            total = _read_files_into_page_cache(paths.values())
            logger.info(
                "trt_engines_prewarmed duration_s={:.0f} profile_s={:.0f} "
                "bytes={}",
                duration_s, picked_dur, total,
            )
        except Exception as exc:
            # Purely opportunistic — a missing engine just means the
            # swap itself will surface the real, actionable error.
            logger.info("trt_engine_prewarm_skipped error={}", exc)

    return spawn_thread(_read, name="trt-prewarm")


def _send_upload_failure(ws, error: str) -> None:
    logger.warning("upload_track_failed error={}", error)
    try:
        ws.send(json.dumps({"type": "upload_failed", "error": error}))
    except Exception:
        pass


def _send_upload_ok(
    ws, packet: UserUploadPacket, *, stems_pending: bool = False,
) -> None:
    ws.send(json.dumps({
        "type": "upload_ok",
        "name": packet.name,
        "bpm": packet.bpm,
        "key": packet.key,
        "time_signature": packet.time_signature,
        "duration_s": packet.duration_s,
        "samples": packet.samples,
        "stems_pending": bool(stems_pending),
    }))


def _offload_upload_encoder(encoder: Session | None) -> None:
    """Park the shared eager encoder's weights in system RAM. The eager
    weights are only needed while an upload is in flight; between
    uploads they would pin ~6 GB of VRAM next to the live streaming
    session. ``ModelContext._load_model_context`` lazily restores
    exactly the modules the next upload touches."""
    if encoder is None:
        return
    try:
        parked = encoder.handler.offload_eager_to_cpu()
        if parked:
            logger.info("upload_encoder_offloaded modules={}", parked)
    except Exception as exc:
        logger.warning("upload_encoder_offload_failed error={}", exc)


def _publish_stems_to_active_session(
    name: str,
    stems: dict | None,
    error: str | None = None,
) -> bool:
    """Push a late ``stem_assets`` / ``stem_failed`` frame to the live
    session's client via its event bus (the adapter's subscriber owns
    the send_lock, so the JSON + binary follow-ups stay atomic with the
    slice stream). No-op when no session is active — the stems are on
    disk and the next swap serves them from cache."""
    with _ACTIVE_SLOT_LOCK:
        cur = _ACTIVE_SESSION[0]
    if cur is None:
        return False
    try:
        if stems is not None:
            first = next(iter(stems.values()))
            cur.streaming.bus.publish(StemAssets(
                fixture_name=name,
                # Empty = "don't touch the client's source-mode pick";
                # this push is overlay data, not a mode change.
                source_mode="",
                sample_rate=SAMPLE_RATE,
                channels=int(first.shape[0]),
                frames=int(first.shape[-1]),
                stems=stems,
            ))
        else:
            cur.streaming.bus.publish(StemFailed(
                fixture_name=name,
                error=error or "stem extraction failed",
            ))
        return True
    except Exception as exc:
        logger.warning(
            "stem_push_failed name={} error={}", name, exc,
        )
        return False


def _handle_upload_track(
    ws,
    header: dict,
    *,
    checkpoint: str,
    decoder_backend: str = "tensorrt",
    vae_backend: str = "tensorrt",
) -> None:
    requested_name = str(header.get("name") or "upload")
    key_override = header.get("key")
    key_override = key_override.strip() if isinstance(key_override, str) else None
    time_signature_override = _normalize_time_signature(header.get("time_signature"))
    try:
        audio_msg = ws.recv()
    except ConnectionClosed:
        return
    if isinstance(audio_msg, str):
        _send_upload_failure(ws, "expected binary PCM frame after upload header")
        return

    try:
        waveform = _truncate_upload_waveform(_decode_audio_msg(audio_msg))
        if waveform.shape[-1] <= 0:
            raise ValueError("audio too short after pool alignment")
    except Exception as exc:
        logger.exception("upload_track_decode_failed name={} error={}", requested_name, exc)
        _send_upload_failure(ws, str(exc))
        return

    # Content dedup: re-importing a "serialize inputs" config pushes audio
    # that's already on the pod back through this handler under its original
    # name. Reuse the on-disk track instead of minting a `(1)` duplicate and
    # re-encoding. Gated on no key/time-signature override (the import path
    # passes none) so an operator's explicit-override upload still re-encodes.
    if key_override is None and time_signature_override is None:
        dup = find_duplicate_upload(
            requested_name, waveform=waveform, sample_rate=SAMPLE_RATE,
        )
        if dup is not None:
            logger.info(
                "upload_track_dedup requested={} reused={}", requested_name, dup.name,
            )
            try:
                # A re-upload can land while the original's background
                # rip is still in flight — surface that so the client
                # keeps its "separating…" status.
                _send_upload_ok(ws, dup, stems_pending=stems_pending(dup.name))
            except ConnectionClosed:
                pass
            return

    # Two-phase upload. Phase 1 (synchronous): analyze + VAE-encode the
    # FULL source + persist it, then ack ``upload_ok`` — the client can
    # swap to the track (and hear audio) immediately. Phase 2
    # (background thread): Mel-Band RoFormer stem rip + per-stem
    # sidecars, with the ACE-Step encoder parked while the separator
    # runs; finished stems are pushed to the live session as a late
    # ``stem_assets`` frame. The swap path coordinates through the
    # pending-stems registry (see extract_and_select_upload_stem):
    # mode "full" swaps proceed without stems, stem-source swaps wait
    # for the rip instead of starting a duplicate separation.
    name = unique_user_upload_name(requested_name)
    encoder: Session | None = None
    try:
        # CPU analysis (librosa beat tracking + key detection) overlaps
        # the GPU source encode below; joined before persist.
        analysis_future = _UPLOAD_ANALYSIS_POOL.submit(
            _analyze_upload_waveform, waveform,
        )
        # Page-cache the engines the post-upload swap will need, while
        # phase 1 runs (background, best-effort).
        _prewarm_trt_engines_for_duration(
            waveform.shape[-1] / SAMPLE_RATE,
            checkpoint=checkpoint,
            decoder_backend=decoder_backend,
            vae_backend=vae_backend,
        )
        encoder = _upload_encoder_session(checkpoint)
        logger.info(
            "upload_track_process_start name={} samples={} duration_s={:.1f}",
            name, int(waveform.shape[-1]), waveform.shape[-1] / SAMPLE_RATE,
        )
        # Serialize all GPU work on the shared encoder: concurrent uploads
        # from multiple connections would otherwise drive prepare_source /
        # stem extraction on one Session at once and corrupt its state.
        # Known cost: a second upload's phase 1 queues here behind the
        # previous upload's background rip (~10 s+), so the sub-second
        # upload_ok holds for spaced uploads, not back-to-back ones.
        with _UPLOAD_INFER_LOCK:
            try:
                full_source = encoder.prepare_source(
                    Audio(waveform=waveform, sample_rate=SAMPLE_RATE),
                )
            except torch.cuda.OutOfMemoryError:
                # Tight headroom next to a long-profile live session can
                # leave torch's caching allocator fragmented; hand the
                # cached pages back and retry once before failing.
                logger.warning(
                    "upload_prepare_source_oom_retry name={} duration_s={:.1f}",
                    name, waveform.shape[-1] / SAMPLE_RATE,
                )
                torch.cuda.empty_cache()
                full_source = encoder.prepare_source(
                    Audio(waveform=waveform, sample_rate=SAMPLE_RATE),
                )
        bpm, detected_key, detected_time_signature = analysis_future.result(
            timeout=120.0,
        )
        key = key_override or detected_key
        time_signature = time_signature_override or detected_time_signature
        packet = persist_user_upload_packet(
            name,
            waveform=waveform,
            stems={},
            sources={"full": full_source},
            sample_rate=SAMPLE_RATE,
            checkpoint=checkpoint,
            bpm=bpm,
            key=key,
            time_signature=time_signature,
        )
        # Mark BEFORE the ack so the client's immediate swap always
        # observes the in-flight rip.
        mark_stems_pending(name)
    except Exception as exc:
        logger.exception("upload_track_process_failed name={} error={}", name, exc)
        _offload_upload_encoder(encoder)
        _send_upload_failure(ws, str(exc))
        return

    def _rip_stems_in_background() -> None:
        try:
            with _UPLOAD_INFER_LOCK:
                try:
                    stems = extract_upload_stems(
                        waveform=waveform,
                        device=encoder.handler.device,
                        backend_sample_rate=SAMPLE_RATE,
                        # Park the eager encoder (a full second copy of
                        # the ACE-Step weights) while the RoFormer runs.
                        # The live StreamingSession owns its own
                        # ModelContext and keeps streaming untouched.
                        model_context=encoder.handler,
                    )
                    stem_sources = {
                        mode: encoder.prepare_source(
                            Audio(waveform=stems[mode], sample_rate=SAMPLE_RATE),
                        )
                        for mode in ("vocals", "instruments")
                    }
                finally:
                    _offload_upload_encoder(encoder)
            wrote = persist_user_upload_stems(
                name,
                waveform=waveform,
                stems=stems,
                sources=stem_sources,
                sample_rate=SAMPLE_RATE,
                checkpoint=checkpoint,
                bpm=bpm,
                key=key,
                time_signature=time_signature,
            )
        except Exception as exc:
            logger.exception(
                "upload_stems_background_failed name={} error={}", name, exc,
            )
            finish_stems_pending(name)
            _publish_stems_to_active_session(name, None, error=str(exc))
            return
        # Files are on disk BEFORE waiters unblock, so a stem-source
        # swap that waited finds them in the cache.
        finish_stems_pending(name)
        if wrote:
            logger.info("upload_stems_ready name={}", name)
            _publish_stems_to_active_session(name, stems)
        else:
            # Track dir was wiped (session ended) mid-rip; nothing to
            # advertise and nothing left on disk.
            logger.info(
                "upload_stems_discarded name={} reason=track_wiped", name,
            )

    try:
        _send_upload_ok(ws, packet, stems_pending=True)
    except Exception as exc:
        # Client is gone (ConnectionClosed) or the ack failed some other
        # way — either way the track is persisted, so finish the rip
        # anyway. This except must stay BROAD: mark_stems_pending(name)
        # already ran, and only the rip thread below pops the registry
        # entry. An exception escaping here would leak the entry forever
        # (every stem-source swap of this track stalls on the 300 s
        # wait) and leave the upload encoder resident on the GPU.
        if not isinstance(exc, ConnectionClosed):
            logger.warning(
                "upload_ok_send_failed name={} error={}", name, exc,
            )
    logger.info(
        "upload_track_ready name={} bpm={} key={} duration_s={:.1f} stems=background",
        packet.name, packet.bpm, packet.key, packet.duration_s,
    )
    try:
        spawn_thread(_rip_stems_in_background, name=f"stem-rip-{name}")
    except BaseException:
        # Only the rip thread pops the registry entry; if the spawn
        # itself fails (thread limit, interpreter shutdown) the entry
        # would leak forever and stall every stem-source swap of this
        # track. Same discipline as the ack above.
        finish_stems_pending(name)
        _offload_upload_encoder(encoder)
        raise


# ---------------------------------------------------------------------------
# WebSocket handler
# ---------------------------------------------------------------------------

def handle_client(
    ws,
    *,
    decoder_backend: str = "tensorrt",
    vae_backend: str = "tensorrt",
    checkpoint: str = "acestep-v15-turbo",
    backend_family: str = "acestep",
    offload_text_encoder: bool = False,
):
    """Connection entrypoint. The body lives in ``_handle_client_body``;
    this wrapper exists only to own a single ``ExitStack`` so the
    contextvar tokens bound for session / track unwind in reverse
    order on every exit path."""
    try:
        with contextlib.ExitStack() as ctx_stack:
            _handle_client_body(
                ws, ctx_stack,
                decoder_backend=decoder_backend,
                vae_backend=vae_backend,
                checkpoint=checkpoint,
                backend_family=backend_family,
                offload_text_encoder=offload_text_encoder,
            )
    finally:
        # Final allocator trim, AFTER the body frame (the last holder of
        # the session, its state, codec, and recv-thread closures) is
        # gone. ``Session.close()`` runs its own gc + empty_cache, but at
        # that point this connection still references those objects, so
        # their pool blocks are live and the trim can't return them.
        # Once the body returns they are garbage — without this trim the
        # caching allocator keeps the session's transient peak reserved
        # (~3 GB after a 60s session, ~6 GB after a 240s-profile one,
        # measured driver-level) for as long as the pod idles, which is
        # exactly the headroom the next session's engine loads and stem
        # extraction need. Reserved-but-unallocated VRAM also can't be
        # used by TensorRT, whose workspace comes from cudaMalloc.
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _handle_client_body(
    ws,
    ctx_stack: contextlib.ExitStack,
    *,
    decoder_backend: str,
    vae_backend: str,
    checkpoint: str,
    backend_family: str,
    offload_text_encoder: bool,
):
    logger.info(
        "client_connected decoder={} vae={} checkpoint={} family={} text_encoder={}",
        decoder_backend, vae_backend, checkpoint, backend_family,
        "offload" if offload_text_encoder else "resident",
    )

    # Disable Nagle. Param frames are tiny (<1 KB of JSON each) and we
    # send them at ~125 Hz; Nagle would coalesce into ~40ms batches.
    try:
        ws.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except (AttributeError, OSError):
        pass

    # ---- Init handshake ----
    config_dict = json.loads(ws.recv())
    if isinstance(config_dict, dict) and config_dict.get("type") == "upload_track":
        try:
            _handle_upload_track(
                ws, config_dict,
                checkpoint=checkpoint,
                decoder_backend=decoder_backend,
                vae_backend=vae_backend,
            )
        finally:
            try:
                ws.close()
            except Exception:
                pass
        return

    # Mint session_id immediately and bind it (plus the client's
    # optional client_id) into loguru's contextvars so every log record
    # on this connection carries the correlation IDs.
    session_id = session_registry.new_session_id()
    _client_id = config_dict.get("client_id") or None
    ctx_stack.enter_context(logger.contextualize(
        session_id=session_id,
        client_id=_client_id,
    ))
    logger.info(
        "session_init config_keys={} client_id={}",
        sorted(config_dict.keys()), _client_id,
    )

    # When this main session ends, wipe MODELS_DIR/user_uploads/ so
    # the next renter on this pod can't list (or replay) what this
    # user uploaded. Registered AFTER the upload_track early-return
    # above on purpose: that branch is a separate, short-lived WS
    # used to PUSH a file into user_uploads/ — wiping on its exit
    # would delete the upload we just persisted. Only the main
    # session (this point onward) registers the wipe. Process-boot
    # also wipes once in server.main() so a crashed-process scenario
    # is still covered.
    # Gated by DEMON_WIPE_USER_UPLOADS so local installs preserve uploads.
    def _wipe_on_session_end() -> None:
        from acestep.user_uploads import (
            user_upload_wipe_enabled,
            wipe_user_uploads,
        )
        if not user_upload_wipe_enabled():
            logger.info("user_uploads_wipe_at_session_end skipped=env_disabled")
            return
        try:
            wiped = wipe_user_uploads()
            if wiped:
                logger.info("user_uploads_wiped_at_session_end entries={}", wiped)
        except Exception as exc:
            logger.warning("user_uploads_wipe_at_session_end_failed error={}", exc)

    ctx_stack.callback(_wipe_on_session_end)
    if config_dict.get("telemetry_version"):
        ws.send(json.dumps({
            "type": "init_ack",
            "session_id": session_id,
            "client_id": _client_id,
        }))

    _t0 = time.monotonic()
    _first_slice = [False]

    def _ms(stage: str) -> None:
        logger.debug(
            "init_timing stage={} elapsed_s={:.3f}",
            stage, time.monotonic() - _t0,
        )

    # Server-side known-fixture load. When the client opts in via
    # ``use_server_fixture`` AND names a known fixture, skip the
    # download→decode→re-upload round-trip and read the waveform
    # straight from the pod's fixture cache.
    fixture_name = config_dict.get("fixture_name")
    if config_dict.get("use_server_fixture") and fixture_name in KNOWN_FIXTURES:
        try:
            waveform = _load_known_fixture_waveform(fixture_name)
            _ms("audio_serverside_loaded")
        except Exception as exc:
            logger.warning(
                "server_side_fixture_load_failed fixture={} error={} "
                "fallback=client_upload",
                fixture_name, exc,
            )
            audio_bytes = ws.recv()
            waveform = _decode_audio_msg(audio_bytes)
            _ms("audio_recv_decoded")
    else:
        audio_bytes = ws.recv()
        waveform = _decode_audio_msg(audio_bytes)
        _ms("audio_recv_decoded")

    # Bind initial source contextvars BEFORE create() so errors during
    # setup carry the fixture + duration in logs. ``audio_duration_s``
    # uses the raw upload duration (pre TRT-cap); the session may trim
    # further but the bound value is within a few samples of the
    # post-trim value in the common case.
    _raw_duration_s = waveform.shape[1] / SAMPLE_RATE
    ctx_stack.enter_context(logger.contextualize(
        fixture_name=fixture_name or None,
        audio_duration_s=round(_raw_duration_s, 2),
    ))

    cfg = SessionConfig.from_dict(config_dict)
    # The server's resolved checkpoint family is the default; an
    # explicit client-config "backend" key still wins.
    if "backend" not in config_dict:
        cfg.backend = backend_family
    audio_in = Audio(waveform=waveform, sample_rate=SAMPLE_RATE)

    # A stem-source create (vocals/instruments) for a track whose
    # background rip is still in flight would otherwise wait for the
    # rip INSIDE StreamingSession.create — while holding
    # _SESSION_LIFECYCLE_LOCK, which preempting connections acquire
    # with no timeout. Drain the wait out here, before the lock, so a
    # slow rip stalls only this connection, never the pod's session
    # handoff. (The in-create wait still exists as a backstop for a rip
    # marked between here and create; it now maps a timeout to a fatal
    # stem error rather than a duplicate separation.)
    if (
        cfg.stem_source_mode in ("vocals", "instruments")
        and stems_pending(fixture_name)
    ):
        logger.info(
            "create_pending_stems_wait fixture_name={} source_mode={}",
            fixture_name, cfg.stem_source_mode,
        )
        finished = wait_for_pending_stems(fixture_name)
        _ms("pending_stems_wait_done")
        if not finished:
            logger.warning(
                "create_pending_stems_wait_timeout fixture_name={}",
                fixture_name,
            )

    _ms("resolve_source_start")
    try:
        # Single-active-session policy: serialize construction and
        # preempt whatever session currently owns the GPU. See the
        # policy comment block at module top.
        with _SESSION_LIFECYCLE_LOCK:
            _preempt_active_session(session_id)
            _log_session_vram("create_start")
            streaming = StreamingSession.create(
                audio=audio_in,
                config=cfg,
                checkpoint=checkpoint,
                decoder_backend=decoder_backend,
                vae_backend=vae_backend,
                offload_text_encoder=offload_text_encoder,
                session_id=session_id,
            )
            with _ACTIVE_SLOT_LOCK:
                _ACTIVE_SESSION[0] = _ActiveSession(session_id, streaming, ws)
            _log_session_vram("create_done")
    except UnsupportedTrtCheckpointError as exc:
        try:
            ws.send(json.dumps({
                "type": "error",
                "code": "unsupported_trt_checkpoint",
                "message": exc.message,
            }))
        except Exception:
            pass
        ws.close(1011, "unsupported TRT checkpoint")
        return
    except EngineNotBuiltError as exc:
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
    except StemExtractFailedError as exc:
        try:
            ws.send(json.dumps({
                "type": "error",
                "code": "stem_extract_failed",
                "message": exc.message,
            }))
        except Exception:
            pass
        ws.close(1011, "stem extraction failed")
        return
    _ms("resolve_source_done")

    streaming_entered_run = False
    session_registered = False

    def _release_active_slot() -> None:
        # Compare-and-swap: only clear the slot if it's still ours (a
        # preempting connection may have already replaced it).
        with _ACTIVE_SLOT_LOCK:
            cur = _ACTIVE_SESSION[0]
            if cur is not None and cur.streaming is streaming:
                _ACTIVE_SESSION[0] = None

    ctx_stack.callback(_release_active_slot)

    def _close_streaming_if_init_fails() -> None:
        if not streaming_entered_run:
            if session_registered:
                session_registry.unregister(session_id)
            streaming.close()

    ctx_stack.callback(_close_streaming_if_init_fails)

    state = streaming.state

    # ---- Per-subscriber transport state ----

    send_lock = threading.Lock()
    codec = SliceCodec(streaming.initial_buffer)

    # ---- Event subscriber: serializer to WS ----
    #
    # All server→client frames (after the init handshake) flow through
    # the bus. The subscriber walks event types via isinstance and
    # serializes each to its wire shape. send_lock is taken per event
    # so JSON + binary follow-ups for one event are atomic.

    def _send_json(payload: dict) -> None:
        try:
            with send_lock:
                ws.send(json.dumps(payload))
        except ConnectionClosed:
            state.running = False
        except Exception:
            pass

    # Backpressure shedding for the slice stream. Slices are produced at
    # ~20-50/s of heavily overlapping windows (~1-2.5 MB/s encoded); on a
    # link that can't drain that (SSH/IDE tunnel, congested uplink) the
    # backlog accumulates in queues and the client receives every slice
    # late — each patch lands behind the playhead and the listener hears
    # the raw source. Each windowed slice is superseded by the next write
    # over the same region (and the writer re-covers every region each
    # lap), so dropping is always safe; drops happen BEFORE encoding so
    # the delta mirror only advances for slices actually sent and the
    # delta chain stays consistent. Two independent layers:
    #
    #   1. In-flight window (the load-bearing one): the client acks
    #      cumulative received slice bytes via ``params.slice_bytes_rx``;
    #      while sent-minus-acked exceeds the window, emission stops.
    #      This is end-to-end — it sees the buffering a saturated SSH
    #      channel or kernel socket buffer hides from the server (where
    #      ws.send keeps succeeding while bytes crawl). Old clients send
    #      no ack -> no flow control (legacy behavior).
    #   2. Bus-queue age: a slice that waited > the age threshold between
    #      publish and serialization (send thread blocked in ws.send) is
    #      dropped. Catches the case where TCP itself pushes back.
    #
    # Healthy links keep in-flight at a few slices and queue age at
    # milliseconds — neither layer engages. Full-buffer renders and
    # unstamped events are never dropped.
    _SLICE_MAX_QUEUE_AGE_S = 2.0
    # 256 KiB default: covers bandwidth-delay products up to ~2.5 MB/s at
    # 100 ms RTT (full slice-stream rate on healthy remote links) while
    # keeping queue transit on a saturated link short — at 200 KB/s the
    # in-flight backlog is ~1.3 s, comfortably inside the runner's
    # transport-lead range. 512 KiB measured ~2.6 s transit at that rate,
    # forcing the lead controller to its cap and oscillating around it.
    try:
        _SLICE_WINDOW_BYTES = max(
            64 * 1024,
            int(os.environ.get("DEMON_SLICE_WINDOW_BYTES", "") or 256 * 1024),
        )
    except ValueError:
        _SLICE_WINDOW_BYTES = 256 * 1024
    # [bytes sent, bytes acked (None until first ack), drops since last
    # log, last log wall]. Shared between the WS subscriber thread
    # (writer of sent/drops) and the recv thread (writer of acked);
    # single-field updates under the GIL, no torn reads that matter.
    _slice_flow = {
        "sent": 0, "acked": None, "drops": 0, "log_wall": 0.0,
    }

    def _note_slice_drop(reason: str, detail: float) -> None:
        _slice_flow["drops"] += 1
        now = time.monotonic()
        if now - _slice_flow["log_wall"] > 5.0:
            logger.warning(
                "slice_backpressure_drop n={} reason={} detail={:.2f} "
                "sent={} acked={}",
                _slice_flow["drops"], reason, detail,
                _slice_flow["sent"], _slice_flow["acked"],
            )
            _slice_flow["drops"] = 0
            _slice_flow["log_wall"] = now

    def _serialize_audio_ready(event: AudioReady) -> None:
        is_windowed = (
            event.published_wall_s > 0.0
            and event.num_samples < len(codec.mirror)
        )
        if is_windowed:
            drop = _windowed_slice_drop_reason(
                acked=_slice_flow["acked"],
                sent=_slice_flow["sent"],
                window_bytes=_SLICE_WINDOW_BYTES,
                age_s=time.monotonic() - event.published_wall_s,
                max_age_s=_SLICE_MAX_QUEUE_AGE_S,
            )
            if drop is not None:
                _note_slice_drop(*drop)
                return
        frame = codec.encode(
            event.audio,
            start_sample=event.start_sample,
            channels=event.channels,
            tick_ms=event.tick_ms,
            dec_ms=event.dec_ms,
            num_gens=event.num_gens,
        )
        if frame is None:
            return
        if not _first_slice[0]:
            _first_slice[0] = True
            _ms("first_generated_slice")
        try:
            with send_lock:
                chunked_ws_send(ws, frame)
                ws.send(json.dumps({
                    "type": "params_update",
                    "params": dict(event.params),
                }))
            # Count only after the send call returned (frame handed to
            # the transport); the client acks the same byte total via
            # params.slice_bytes_rx. Counted for EVERY slice-path frame
            # (windowed and full-buffer renders alike) because the client
            # increments _sliceBytesRx for every binary slice it receives
            # — only swap/stem binaries are excluded, and those go through
            # separate serializers. Counting windowed-only here would let
            # acked outrun sent by each full-buffer render's bytes, walking
            # the in-flight window permanently negative so the load-bearing
            # flow-control layer stops engaging.
            _slice_flow["sent"] += len(frame)
            # Report the pod-side slice hash over the exact float16 bytes
            # just sent (spec 06 §2.3), for pod/client cross-checking.
            # Fully guarded + fail-open: provenance must never wedge the
            # downlink (spec 06 §7).
            slice_hash = codec.last_slice_hash
            if slice_hash is not None:
                try:
                    from acestep.provenance.session_log import (
                        record_pod_slice_hash,
                    )
                    record_pod_slice_hash(session_id, **slice_hash)
                except Exception:  # noqa: BLE001
                    pass
        except ConnectionClosed:
            state.running = False

    def _serialize_swap_ready(event: SwapReady) -> None:
        # Mirror the new buffer on this subscriber so subsequent
        # slices delta against the right basis.
        new_src_np = event.initial_buffer
        try:
            with send_lock:
                ws.send(json.dumps({
                    "type": "swap_ready",
                    "duration": event.duration,
                    "sample_rate": event.sample_rate,
                    "channels": event.channels,
                    "bpm": event.bpm,
                    "key": event.key,
                    "time_signature": event.time_signature,
                    "fixture_name": event.fixture_name,
                    "source_epoch": event.source_epoch,
                }))
                chunked_ws_send(ws, new_src_np.astype(np.float16).tobytes())
                codec.replace_mirror(new_src_np)
                if event.stems is not None:
                    send_stem_payload(
                        ws,
                        fixture_name=event.fixture_name,
                        source_mode=event.stem_source_mode,
                        stems=event.stems,
                    )
                elif event.stem_error is not None:
                    ws.send(json.dumps({
                        "type": "stem_failed",
                        "fixture_name": event.fixture_name or "",
                        "error": event.stem_error,
                    }))
        except ConnectionClosed:
            state.running = False

    def on_event(event) -> None:
        if isinstance(event, AudioReady):
            _serialize_audio_ready(event)
        elif isinstance(event, SwapReady):
            _serialize_swap_ready(event)
        elif isinstance(event, SwapFailed):
            payload = {"type": "swap_failed", "error": event.error}
            if event.build_command is not None:
                payload["build_command"] = event.build_command
            _send_json(payload)
        elif isinstance(event, ParamsEcho):
            _send_json({"type": "params_echo", "raw": event.raw})
        elif isinstance(event, PromptBlendEcho):
            _send_json({"type": "prompt_blend_echo", "value": event.value})
        elif isinstance(event, PromptApplied):
            _send_json({"type": "prompt_applied", "tags": event.tags})
        elif isinstance(event, LoraCatalogUpdate):
            _send_json({"type": "lora_catalog", "catalog": event.catalog})
        elif isinstance(event, DepthApplied):
            _send_json({"type": "depth_applied", "value": event.value})
        elif isinstance(event, ManualSlotCount):
            _send_json({"type": "manual_slot_count", "count": event.count})
        elif isinstance(event, CommandFailed):
            _send_json({
                "type": "command_failed",
                "command": event.command,
                "requires": event.requires,
                "error": event.error,
            })
        elif isinstance(event, SessionError):
            # Runtime failure after the ready handshake (e.g. the
            # pipeline runner died). Reuses the wire `error` event so
            # the client shows a reason instead of a silently frozen UI.
            _send_json({
                "type": "error",
                "code": event.code,
                "message": event.message,
            })
        elif isinstance(event, TimbreSet):
            _send_json({
                "type": "timbre_set", "name": event.name,
                "duration": event.duration,
            })
        elif isinstance(event, TimbreCleared):
            _send_json({"type": "timbre_cleared"})
        elif isinstance(event, TimbreFailed):
            _send_json({"type": "timbre_failed", "error": event.error})
        elif isinstance(event, StructureSet):
            _send_json({
                "type": "structure_set", "name": event.name,
                "duration": event.duration,
            })
        elif isinstance(event, StructureCleared):
            _send_json({"type": "structure_cleared"})
        elif isinstance(event, StructureFailed):
            _send_json({"type": "structure_failed", "error": event.error})
        elif isinstance(event, AudioWritten):
            _send_json({
                "type": "audio_written",
                "start_s": event.start_s, "end_s": event.end_s,
                "source_epoch": event.source_epoch,
            })
        elif isinstance(event, AudioWriteFailed):
            _send_json({"type": "audio_write_failed", "error": event.error})
        elif isinstance(event, StemAssets):
            # Late background-rip delivery (upload path): same wire
            # shape the init/swap paths send inline. send_lock keeps
            # the JSON header + per-stem binaries atomic vs slices.
            try:
                with send_lock:
                    send_stem_payload(
                        ws,
                        fixture_name=event.fixture_name,
                        source_mode=event.source_mode,
                        stems=event.stems,
                    )
            except ConnectionClosed:
                state.running = False
            except Exception as exc:
                logger.warning("stem_assets_send_failed error={}", exc)
        elif isinstance(event, StemFailed):
            _send_json({
                "type": "stem_failed",
                "fixture_name": event.fixture_name or "",
                "error": event.error,
            })
        elif isinstance(event, SubscriberDropped):
            # Terminal notice from the bus: our subscription overflowed
            # and was force-closed. Outbound delivery is dead; the
            # session keeps ticking otherwise, so flip the run flag to
            # tear the whole session down and let the client reconnect.
            logger.warning("ws_subscriber_dropped reason={}", event.reason)
            state.running = False

    streaming.bus.subscribe(on_event, name="ws")

    # ---- Init handshake: ready + binary initial buffer + optional stems ----
    #
    # These ship inline (not through the bus) because they're produced
    # synchronously by ``StreamingSession.create`` and have nothing to
    # fan out yet. After this block the bus subscriber takes over.
    src_np = streaming.initial_buffer
    ws.send(json.dumps({
        "type": "ready",
        "duration": len(src_np) / SAMPLE_RATE,
        "sample_rate": SAMPLE_RATE,
        "channels": state.n_channels,
        "lora_dir": str(loras_dir()),
        "lora_catalog": streaming.lora_catalog_payload(),
        "lora_pending_enable": list(streaming.initial_enable_ids),
        "bpm": state.bpm,
        "key": state.key,
        "time_signature": state.time_signature,
        "checkpoint": checkpoint,
        "checkpoint_scale": checkpoint_scale(checkpoint),
        "pipeline_depth": state.current_depth,
        "max_pipeline_depth": streaming.max_pipeline_depth,
        "session_id": session_id,
        "source_epoch": state.source_epoch,
        # Phase-2 contract surface, declared by the session's backend
        # (plan §3.1–3.3). The legacy flat duration/sample_rate/channels
        # fields above stay as-is for old clients; geometry is the
        # backend-declared truth new clients read instead of constants.
        "geometry": streaming.geometry_payload(),
        "capabilities": streaming.capabilities_payload(),
        "knob_manifest": streaming.knob_manifest_payload(),
        # Activation-steering surface (manual_slot_count /
        # manual_slot_cap / steering_available).
        **streaming.steering_payload(),
    }))
    chunked_ws_send(ws, src_np.astype(np.float16).tobytes())
    if streaming.initial_upload_stems is not None:
        send_stem_payload(
            ws,
            fixture_name=fixture_name,
            source_mode=streaming.initial_stem_source_mode,
            stems=streaming.initial_upload_stems,
        )
    elif streaming.initial_stem_error is not None:
        ws.send(json.dumps({
            "type": "stem_failed",
            "fixture_name": fixture_name or "",
            "error": streaming.initial_stem_error,
        }))
    logger.info(
        "initial_buffer_sent duration_s={:.1f}",
        len(src_np) / SAMPLE_RATE,
    )
    _ms("initial_buffer_sent")

    # ---- Streaming ----

    # --- Control bus ---
    # External commands (from the demo's onboard MCP server) land here
    # and get dispatched through the same router as live WS frames.
    # The single-dispatch-thread invariant (control + WS messages
    # serialize through one recv loop) is preserved by enqueueing
    # rather than calling session methods directly from the HTTP
    # handler thread.
    control_queue: queue.Queue = queue.Queue()

    def inject_control(data: dict, audio: bytes | None = None) -> None:
        control_queue.put((data, audio))

    def snapshot_session() -> dict:
        snap = streaming.snapshot()
        snap["fixture_name"] = fixture_name
        return snap

    # Throttle state for coercion warnings, keyed (source, mtype). The
    # params channel runs at ~125 Hz; a client that trips coercion on
    # every tick would otherwise emit a warning per message on the
    # dispatch thread (the same thread that feeds set_knobs).
    _coerce_warn_last: dict = {}
    _COERCE_WARN_INTERVAL_S = 5.0

    # --- Dispatcher router: WS / control bus JSON → session method ---
    def _dispatch_message(
        data: dict,
        recv_audio,
        source: str,
    ) -> None:
        """Route one parsed message into a typed session call.

        ``recv_audio`` returns the next binary audio frame. For
        WS-sourced messages it's ``ws.recv``; for control-bus
        messages it's a thunk that returns the pre-loaded bytes the
        MCP sent alongside the JSON.

        ``source`` is ``"ws"`` for the browser's own WebSocket and
        ``"control"`` for control-bus messages. Maps to
        ``CommandOrigin`` for the two origin-dependent verbs.

        Inbound envelopes are validated against the wire-contract registry
        before dispatch: unknown command names are rejected up front
        (registry-derived, replacing the old fall-through log), and
        declared fields are type-coerced by ``coerce_command_payload`` —
        the same enforcement point the MCP tools use, so per-field checks
        can't drift between transports. Hot-path semantics are preserved:
        coercion silently cleans (clamp-style, like the knob channel) and
        the arms' own defensive fallbacks still apply to dropped fields.
        """
        mtype = data.get("type")
        origin = (
            CommandOrigin.EXTERNAL if source == "control"
            else CommandOrigin.PRIMARY
        )
        if mtype not in COMMAND_NAMES:
            # Unknown mtype — log but don't crash; lets future protocol
            # additions degrade gracefully on older servers.
            logger.warning(
                "unknown_message_type origin={} mtype={}", source, mtype,
            )
            return
        data, coerce_errors = coerce_command_payload(mtype, data)
        if coerce_errors:
            _now = time.monotonic()
            _key = (source, mtype)
            if _now - _coerce_warn_last.get(_key, 0.0) >= _COERCE_WARN_INTERVAL_S:
                _coerce_warn_last[_key] = _now
                logger.warning(
                    "command_payload_coerced origin={} mtype={} errors={}",
                    source, mtype, coerce_errors,
                )

        def _recv_binary_payload(fail_type: str):
            """Read the binary frame that must follow ``mtype``.

            Bounded (``_BINARY_PAYLOAD_TIMEOUT_S``) and type-checked,
            so an orphan header can neither block the recv loop forever
            (wedging the whole session) nor consume the next JSON
            command as its payload. Both failure modes answer
            ``fail_type`` and keep the session alive. Returns ``None``
            on failure (the caller must bail out); flips
            ``state.running`` if the connection closed.

            The bound is a transfer deadline for multi-MB frames, not
            just an orphan-header guard — see the constant for why it is
            sized the way it is.
            """
            try:
                audio_msg = recv_audio(timeout=_BINARY_PAYLOAD_TIMEOUT_S)
            except TimeoutError:
                logger.error(
                    "{}_payload_timeout origin={} timeout_s={:.0f}",
                    mtype, origin, _BINARY_PAYLOAD_TIMEOUT_S,
                )
                _send_json({
                    "type": fail_type,
                    "error": (
                        "binary payload not received within "
                        f"{_BINARY_PAYLOAD_TIMEOUT_S:.0f}s"
                    ),
                })
                return None
            except ConnectionClosed:
                state.running = False
                return None
            if not isinstance(audio_msg, (bytes, bytearray)):
                # Log what got eaten: if it was the next JSON command,
                # the preview names it so the drop is traceable.
                logger.error(
                    "{}_payload_not_binary origin={} got={}",
                    mtype, origin, repr(audio_msg)[:120],
                )
                _send_json({
                    "type": fail_type,
                    "error": f"expected binary payload after {mtype}",
                })
                return None
            return audio_msg

        try:
            if mtype == "params":
                pp_raw = data.get("playback_pos")
                try:
                    pp = float(pp_raw) if pp_raw is not None else None
                except (TypeError, ValueError):
                    pp = None
                if pp is not None and not math.isfinite(pp):
                    pp = None
                anchor_kwargs = {}
                if "render_anchor_s" in data:
                    anchor_raw = data.get("render_anchor_s")
                    try:
                        anchor = float(anchor_raw) if anchor_raw is not None else None
                    except (TypeError, ValueError):
                        anchor = None
                    if anchor is not None and not math.isfinite(anchor):
                        anchor = None
                    anchor_kwargs["render_anchor_s"] = anchor
                if "render_anchor_queue_s" in data:
                    queue_raw = data.get("render_anchor_queue_s")
                    queue: list = []
                    if isinstance(queue_raw, (list, tuple)):
                        for item in queue_raw[:_RENDER_ANCHOR_QUEUE_CAP]:
                            try:
                                val = float(item)
                            except (TypeError, ValueError):
                                continue
                            if math.isfinite(val):
                                queue.append(val)
                    anchor_kwargs["render_anchor_queue_s"] = queue
                ct = data.get("client_time")
                try:
                    ct = float(ct) if ct is not None else None
                except (TypeError, ValueError):
                    ct = None
                sl = data.get("slice_lead_s")
                try:
                    sl = float(sl) if sl is not None else None
                except (TypeError, ValueError):
                    sl = None
                # Flow-control ack: monotone cumulative byte count. Only
                # ever ratchets forward — a reordered/stale report can't
                # reopen the window spuriously.
                ack = data.get("slice_bytes_rx")
                if ack is not None:
                    try:
                        ack = int(float(ack))
                    except (TypeError, ValueError):
                        ack = None
                    if ack is not None:
                        prev = _slice_flow["acked"]
                        if prev is None or ack > prev:
                            _slice_flow["acked"] = ack
                streaming.set_knobs(
                    data.get("raw") or {}, pp, origin=origin,
                    client_time=ct, slice_lead_s=sl,
                    **anchor_kwargs,
                )
            elif mtype == "loop_band":
                streaming.set_loop_band(
                    data.get("start_sec"), data.get("end_sec"),
                    origin=origin,
                )
            elif mtype == "prompt":
                streaming.set_prompt(
                    data["tags"],
                    tags_b=data.get("tags_b"),
                    bpm=data.get("bpm"),
                    key=data.get("key"),
                    time_signature=data.get("time_signature"),
                    origin=origin,
                )
            elif mtype == "set_prompt_blend":
                try:
                    v = float(data.get("value", 0.0))
                except (TypeError, ValueError):
                    v = 0.0
                streaming.set_prompt_blend(v, origin=origin)
            elif mtype == "set_interp_method":
                streaming.set_interp_method(
                    str(data.get("path", "")),
                    str(data.get("method", "")),
                    origin=origin,
                )
            elif mtype == "set_depth":
                try:
                    v = int(data.get("value"))
                except (TypeError, ValueError):
                    return
                streaming.set_depth(v, origin=origin)
            elif mtype == "enable_lora":
                lid = data.get("id")
                s = data.get("strength")
                try:
                    strength = float(s) if s is not None else None
                except (TypeError, ValueError):
                    strength = None
                if lid:
                    streaming.enable_lora(
                        str(lid), strength, origin=origin,
                    )
            elif mtype == "disable_lora":
                lid = data.get("id")
                if lid:
                    streaming.disable_lora(str(lid), origin=origin)
            elif mtype == "add_lora":
                lid = data.get("id")
                url = data.get("url")
                if lid and url:
                    # Validation (id charset, URL scheme + host allowlist)
                    # lives in the session, next to the code that writes
                    # the file — this dispatch only unpacks the message.
                    streaming.add_lora(str(url), str(lid), origin=origin)
            elif mtype == "manual_slot_add":
                streaming.manual_slot_add(origin=origin)
            elif mtype == "manual_slot_pop":
                streaming.manual_slot_pop(origin=origin)
            elif mtype == "set_timbre_strength":
                try:
                    v = float(data.get("value", 1.0))
                except (TypeError, ValueError):
                    v = 1.0
                streaming.set_timbre_strength(v, origin=origin)
            elif mtype == "set_timbre_source":
                name = data.get("name") or "timbre"
                audio_msg = _recv_binary_payload("timbre_failed")
                if audio_msg is None:
                    return
                logger.debug(
                    "set_timbre_source_bytes_received name={} bytes={}",
                    name, len(audio_msg),
                )
                # Decode is outside the session boundary, so a
                # malformed wire frame can't reach the typed-event
                # `*_failed` path. Ack the frontend explicitly so its
                # upload UI sees a deterministic terminal state instead
                # of waiting on a setTimeout fallback.
                try:
                    wf = _decode_audio_msg(audio_msg)
                except Exception as exc:
                    logger.opt(exception=True).error(
                        "set_timbre_source_decode_failed origin={} name={} error={}",
                        origin, name, exc,
                    )
                    _send_json({"type": "timbre_failed", "error": str(exc)})
                    return
                streaming.set_timbre_source(
                    Audio(waveform=wf, sample_rate=SAMPLE_RATE),
                    name, origin=origin,
                )
            elif mtype == "set_timbre_fixture":
                streaming.set_timbre_fixture(
                    data.get("name", ""), origin=origin,
                )
            elif mtype == "clear_timbre_source":
                streaming.clear_timbre_source(origin=origin)
            elif mtype == "set_structure_source":
                name = data.get("name") or "structure"
                audio_msg = _recv_binary_payload("structure_failed")
                if audio_msg is None:
                    return
                logger.debug(
                    "set_structure_source_bytes_received name={} bytes={}",
                    name, len(audio_msg),
                )
                try:
                    wf = _decode_audio_msg(audio_msg)
                except Exception as exc:
                    logger.opt(exception=True).error(
                        "set_structure_source_decode_failed origin={} name={} error={}",
                        origin, name, exc,
                    )
                    _send_json({"type": "structure_failed", "error": str(exc)})
                    return
                streaming.set_structure_source(
                    Audio(waveform=wf, sample_rate=SAMPLE_RATE),
                    name, origin=origin,
                )
            elif mtype == "set_structure_fixture":
                streaming.set_structure_fixture(
                    data.get("name", ""), origin=origin,
                )
            elif mtype == "clear_structure_source":
                streaming.clear_structure_source(origin=origin)
            elif mtype == "swap_source":
                # Server-side source load: when the user picks a track that
                # already lives on the pod (a built-in fixture or a
                # persisted upload), the client sends the name alone with
                # ``use_server_source`` set and NO binary PCM frame. We read
                # the waveform straight off disk by name, so the sidecar +
                # stem caches hit (no prepare_source, no Mel-Band RoFormer
                # re-rip) instead of treating a re-decoded, re-uploaded PCM
                # buffer as a brand-new source.
                if data.get("use_server_source"):
                    name = data.get("fixture_name")
                    try:
                        wf = _load_clip_waveform(str(name))
                    except Exception as exc:
                        logger.opt(exception=True).error(
                            "swap_source_server_load_failed origin={} "
                            "name={} error={}",
                            origin, name, exc,
                        )
                        _send_json({
                            "type": "swap_failed",
                            "error": f"server source load failed: {exc}",
                        })
                        return
                else:
                    audio_msg = _recv_binary_payload("swap_failed")
                    if audio_msg is None:
                        return
                    try:
                        wf = _decode_audio_msg(audio_msg)
                    except Exception as exc:
                        logger.opt(exception=True).error(
                            "swap_source_decode_failed origin={} error={}",
                            origin, exc,
                        )
                        _send_json({"type": "swap_failed", "error": str(exc)})
                        return
                streaming.swap_source(
                    Audio(waveform=wf, sample_rate=SAMPLE_RATE),
                    tags=data.get("tags"),
                    bpm=data.get("bpm"),
                    key=data.get("key"),
                    time_signature=data.get("time_signature"),
                    fixture_name=data.get("fixture_name"),
                    stem_source_mode=data.get("stem_source_mode"),
                    origin=origin,
                )
            elif mtype == "write_audio":
                # "Play into the model": a binary PCM frame (only the
                # audio being written) always follows. No song restart.
                audio_msg = _recv_binary_payload("audio_write_failed")
                if audio_msg is None:
                    return
                try:
                    wf = _decode_audio_msg(audio_msg)
                except Exception as exc:
                    logger.opt(exception=True).error(
                        "write_audio_decode_failed origin={} error={}",
                        origin, exc,
                    )
                    _send_json({
                        "type": "audio_write_failed", "error": str(exc),
                    })
                    return
                epoch = data.get("source_epoch")
                streaming.write_audio(
                    Audio(waveform=wf, sample_rate=SAMPLE_RATE),
                    at_s=float(data.get("at_s") or 0.0),
                    mix=str(data.get("mix") or "replace"),
                    repeat=str(data.get("repeat") or "none"),
                    source_epoch=int(epoch) if epoch is not None else None,
                    refresh_timbre=bool(data.get("refresh_timbre", False)),
                    origin=origin,
                )
        except ConnectionClosed:
            state.running = False

    def recv_loop():
        # Params coalescing: each ``params`` message is a full knob
        # snapshot plus a playhead report, sent at ~125 Hz, so when a
        # backlog forms (recv thread starved by GIL-heavy GPU work, or a
        # burst of delayed messages arriving at once after network
        # congestion) only the NEWEST queued snapshot matters. Applying
        # the whole backlog one-by-one re-anchors the runner's playhead
        # clock to progressively staler positions — slices then render
        # behind the live playhead and the client audibly falls back to
        # the raw source. Buffer consecutive ``params`` and dispatch only
        # the last one; any other message type flushes the pending params
        # first so cross-type ordering is preserved.
        def _dispatch_safe(data):
            try:
                _dispatch_message(data, ws.recv, "ws")
            except Exception as exc:
                logger.exception("ws_dispatch_error error={}", exc)

        while state.running:
            pending_params = None
            try:
                while True:
                    # While a params snapshot is pending, poll without
                    # blocking so the drain reaches the newest message
                    # before anything is applied; otherwise allow the
                    # normal 1 ms wait.
                    try:
                        msg = ws.recv(
                            timeout=0.0 if pending_params is not None else 0.001,
                        )
                    except TimeoutError:
                        break
                    if isinstance(msg, str):
                        try:
                            data = json.loads(msg)
                        except Exception:
                            continue
                        if (
                            isinstance(data, dict)
                            and data.get("type") == "params"
                        ):
                            # Newest snapshot wins for the playhead and all
                            # knobs, but slice_lead_s is worst-since-last-report
                            # (wire contract): fold the min forward so a
                            # transient negative spike in a superseded report
                            # isn't lost. See _coalesced_slice_lead.
                            if pending_params is not None:
                                carried = _coalesced_slice_lead(
                                    pending_params.get("slice_lead_s"),
                                    data.get("slice_lead_s"),
                                )
                                if carried is not None:
                                    data["slice_lead_s"] = carried
                                # Sticky placement must survive coalescing:
                                # see _fold_sticky_params.
                                _fold_sticky_params(pending_params, data)
                            pending_params = data  # newest wins
                        else:
                            if pending_params is not None:
                                _dispatch_safe(pending_params)
                                pending_params = None
                            _dispatch_safe(data)
                    if not state.running:
                        break
            except ConnectionClosed:
                state.running = False
            except Exception as exc:
                logger.exception("recv_loop_error error={}", exc)
                state.running = False
            if pending_params is not None and state.running:
                _dispatch_safe(pending_params)
            if not state.running:
                break

            # Drain the MCP / external control bus.
            while True:
                try:
                    cdata, caudio = control_queue.get_nowait()
                except queue.Empty:
                    break
                _audio_buf = caudio if caudio is not None else b""
                try:
                    # The thunk accepts ``timeout`` (and ignores it) so it
                    # matches the ``ws.recv`` signature _recv_binary_payload
                    # calls with.
                    _dispatch_message(
                        cdata,
                        lambda timeout=None, _b=_audio_buf: _b,
                        "control",
                    )
                except Exception as exc:
                    logger.exception(
                        "control_dispatch_error error={}", exc,
                    )

    def _reap_dead_client(
        reason: str,
        *,
        backlog: int | None,
        stall_s: float,
        tcp_state: int | None = None,
    ) -> None:
        logger.warning(
            "dead_client_reaped reason={} backlog_bytes={} stall_s={:.1f} "
            "tcp_state={}",
            reason, backlog, stall_s, tcp_state,
        )
        # Stop the runner: it observes this between pipeline iterations
        # and exits run() into close(), whose finally unregisters the
        # session — so /sessions empties and the pod returns to the pool.
        state.running = False
        # Force the socket down so the wedged sendall (holding the WS
        # protocol lock) and the library's recv thread both unblock now,
        # instead of waiting on the graceful-close handshake — which needs
        # that same pinned lock and would re-block.
        sock = getattr(ws, "socket", None)
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def dead_client_watchdog() -> None:
        """Tear the session down when its client is gone but the WS
        machinery hasn't noticed (see the module-top comment block).

        Polls GIL/lock-independent socket syscalls so it stays accurate
        even when the send path is wedged in ``sendall`` and the keepalive
        ping is deadlocked behind the protocol lock. No-ops off Linux,
        where the syscalls aren't available and would false-reap."""
        if not _WATCHDOG_SUPPORTED:
            return
        sock = getattr(ws, "socket", None)
        if sock is None:
            return
        last_drain_wall = time.monotonic()
        prev_backlog = 0
        prev_acked: int | None = None
        while state.running:
            time.sleep(_DEAD_CLIENT_POLL_S)
            if not state.running:
                return
            # Peer close / dead FD — independent of the recv thread, which
            # a wedged connection may never schedule to read the EOF.
            info = _tcp_state_and_acked(sock)
            if info is None:
                _reap_dead_client("socket_error", backlog=None, stall_s=0.0)
                return
            tcp_state, acked = info
            if tcp_state != _TCP_ESTABLISHED:
                _reap_dead_client(
                    "peer_closed", backlog=None, stall_s=0.0,
                    tcp_state=tcp_state,
                )
                return
            backlog = _socket_send_backlog(sock)
            if backlog is None:
                _reap_dead_client("socket_error", backlog=None, stall_s=0.0)
                return
            # The client is alive if the kernel is acking bytes
            # (bytes_acked advancing) or the send queue is draining (empty
            # or shrunk) — either resets the stall clock. bytes_acked is
            # the robust signal: it keeps a backpressured-but-alive client
            # whose queue plateaus at a constant positive level from being
            # reaped. Only a positive queue with no ack progress and no
            # drain for the whole grace window (a client reading nothing)
            # accrues toward the reap.
            acked_progress = (
                acked is not None
                and prev_acked is not None
                and acked > prev_acked
            )
            if backlog == 0 or backlog < prev_backlog or acked_progress:
                last_drain_wall = time.monotonic()
            prev_backlog = backlog
            prev_acked = acked
            stall_s = time.monotonic() - last_drain_wall
            if backlog > 0 and stall_s >= _DEAD_CLIENT_SEND_STALL_S:
                _reap_dead_client(
                    "send_stall", backlog=backlog, stall_s=stall_s,
                )
                return

    # spawn_thread copies the parent context (loguru contextvars), so
    # logs emitted from inside recv_loop still carry session_id and
    # friends.
    recv_t = spawn_thread(recv_loop, name="recv_loop")
    spawn_thread(dead_client_watchdog, name="dead_client_watchdog")

    # Register with the process-global session registry so the demo's
    # onboard MCP server can drive this session via the HTTP control
    # bus.
    session_registry.register(session_registry.SessionHandle(
        id=session_id,
        started_at=time.time(),
        inject=inject_control,
        snapshot=snapshot_session,
        # Carrying the session EventBus makes register() attach the local
        # provenance session-log tap (spec 06 §2.2); everything downstream
        # is fail-open and degrades to a no-op without the provenance
        # extra installed. provenance_meta supplies the adapter-known
        # identifiers the snapshot doesn't expose.
        bus=streaming.bus,
        provenance_meta={
            "checkpoint": checkpoint,
            "fixture_name": fixture_name,
            "decoder_backend": decoder_backend,
            # Local path of the initial source file so the tap can commit
            # its sha256 to the input chain (06 §2.2). Resolved without
            # network: by session start a served fixture is already in the
            # local cache; anything else hashes via its buffer fingerprint
            # instead (SessionReady/SwapReady).
            "fixture_path": _known_fixture_local_path(fixture_name),
        },
    ))
    session_registered = True
    logger.info("session_registered")

    # Stage the initial enable set so they get applied on the runner
    # thread before the first tick. Each entry carries its target
    # strength so the refit lands at the right value in one shot.
    if streaming.use_lora and streaming.initial_enable_ids:
        with state._lock:
            for lid in streaming.initial_enable_ids:
                state.pending_enable.append(
                    (lid, streaming.lora_strengths_init.get(lid)),
                )

    try:
        streaming_entered_run = True
        streaming.run()
    finally:
        session_registry.unregister(session_id)
        recv_t.join(timeout=2)
        logger.info(
            "client_disconnected num_gens={}",
            state.params.get("num_gens", 0),
        )
