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
import queue
import socket
import threading
import time

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
    DepthApplied,
    LoraCatalogUpdate,
    ParamsEcho,
    PromptApplied,
    PromptBlendEcho,
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
from acestep.streaming.stems import extract_upload_stems
from acestep.user_uploads import (
    UserUploadPacket,
    find_duplicate_upload,
    persist_user_upload_packet,
    unique_user_upload_name,
)

from .audio_codec import SliceCodec, send_stem_payload
from .protocol import SAMPLE_RATE


# ---------------------------------------------------------------------------
# Canonical user-upload packet processing
# ---------------------------------------------------------------------------

# A single eager encoder Session per checkpoint, shared across all
# connections and held for the process lifetime. This is the lone shared
# mutable GPU object in the system (every StreamingSession otherwise owns
# its own Session). The tradeoff is deliberate: encoding uploads needs the
# VAE encoder, which the streaming TRT path doesn't expose, so we keep a
# second resident eager copy of the weights rather than rebuild one per
# upload. Two costs follow from the sharing:
#   - VRAM: the first upload permanently adds a second model copy.
#   - Concurrency: prepare_source / stem extraction are NOT thread-safe on
#     a shared Session, so _UPLOAD_INFER_LOCK serializes all GPU work on it.
_UPLOAD_ENCODERS: dict[str, Session] = {}
_UPLOAD_ENCODERS_LOCK = threading.Lock()
_UPLOAD_INFER_LOCK = threading.Lock()


def _upload_encoder_session(checkpoint: str) -> Session:
    with _UPLOAD_ENCODERS_LOCK:
        session = _UPLOAD_ENCODERS.get(checkpoint)
        if session is None:
            logger.info("upload_encoder_load_start checkpoint={}", checkpoint)
            session = Session(
                project_root=str(checkpoints_dir()),
                config_path=checkpoint,
                decoder_backend="eager",
                vae_backend="eager",
            )
            _UPLOAD_ENCODERS[checkpoint] = session
            logger.info("upload_encoder_loaded checkpoint={}", checkpoint)
        return session


def _truncate_upload_waveform(waveform: torch.Tensor) -> torch.Tensor:
    max_samples = int(max_profile_duration_s() * SAMPLE_RATE)
    return truncate_to_pool(waveform[:2, :max_samples])


def _analyze_upload_waveform(waveform: torch.Tensor) -> tuple[int, str, str]:
    import librosa

    mono_np = waveform.mean(dim=0).detach().cpu().numpy()
    bpm_raw, _ = librosa.beat.beat_track(y=mono_np, sr=SAMPLE_RATE)
    bpm = int(round(float(np.asarray(bpm_raw).flat[0])))
    return bpm, detect_key(mono_np, SAMPLE_RATE), "4"


def _send_upload_failure(ws, error: str) -> None:
    logger.warning("upload_track_failed error={}", error)
    try:
        ws.send(json.dumps({"type": "upload_failed", "error": error}))
    except Exception:
        pass


def _send_upload_ok(ws, packet: UserUploadPacket) -> None:
    ws.send(json.dumps({
        "type": "upload_ok",
        "name": packet.name,
        "bpm": packet.bpm,
        "key": packet.key,
        "time_signature": packet.time_signature,
        "duration_s": packet.duration_s,
        "samples": packet.samples,
    }))


def _handle_upload_track(ws, header: dict, *, checkpoint: str) -> None:
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
                _send_upload_ok(ws, dup)
            except ConnectionClosed:
                pass
            return

    name = unique_user_upload_name(requested_name)
    try:
        bpm, detected_key, detected_time_signature = _analyze_upload_waveform(waveform)
        key = key_override or detected_key
        time_signature = time_signature_override or detected_time_signature
        encoder = _upload_encoder_session(checkpoint)
        logger.info(
            "upload_track_process_start name={} samples={} duration_s={:.1f}",
            name, int(waveform.shape[-1]), waveform.shape[-1] / SAMPLE_RATE,
        )
        # Serialize all GPU work on the shared encoder: concurrent uploads
        # from multiple connections would otherwise drive prepare_source /
        # stem extraction on one Session at once and corrupt its state.
        with _UPLOAD_INFER_LOCK:
            sources = {
                "full": encoder.prepare_source(
                    Audio(waveform=waveform, sample_rate=SAMPLE_RATE),
                ),
            }
            stems = extract_upload_stems(
                waveform=waveform,
                device=encoder.handler.device,
                backend_sample_rate=SAMPLE_RATE,
            )
            for mode in ("vocals", "instruments"):
                sources[mode] = encoder.prepare_source(
                    Audio(waveform=stems[mode], sample_rate=SAMPLE_RATE),
                )
        packet = persist_user_upload_packet(
            name,
            waveform=waveform,
            stems=stems,
            sources=sources,
            sample_rate=SAMPLE_RATE,
            checkpoint=checkpoint,
            bpm=bpm,
            key=key,
            time_signature=time_signature,
        )
    except Exception as exc:
        logger.exception("upload_track_process_failed name={} error={}", name, exc)
        _send_upload_failure(ws, str(exc))
        return

    try:
        _send_upload_ok(ws, packet)
    except ConnectionClosed:
        return
    logger.info(
        "upload_track_ready name={} bpm={} key={} duration_s={:.1f}",
        packet.name, packet.bpm, packet.key, packet.duration_s,
    )


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
    contextvar tokens bound for session / track unwind in reverse
    order on every exit path."""
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
            _handle_upload_track(ws, config_dict, checkpoint=checkpoint)
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
    audio_in = Audio(waveform=waveform, sample_rate=SAMPLE_RATE)

    _ms("resolve_source_start")
    try:
        streaming = StreamingSession.create(
            audio=audio_in,
            config=cfg,
            checkpoint=checkpoint,
            decoder_backend=decoder_backend,
            vae_backend=vae_backend,
            offload_text_encoder=offload_text_encoder,
            session_id=session_id,
        )
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

    def _serialize_audio_ready(event: AudioReady) -> None:
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
                ws.send(frame)
                ws.send(json.dumps({
                    "type": "params_update",
                    "params": dict(event.params),
                }))
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
                }))
                ws.send(new_src_np.astype(np.float16).tobytes())
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
    }))
    ws.send(src_np.astype(np.float16).tobytes())
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
        """
        mtype = data.get("type")
        origin = (
            CommandOrigin.EXTERNAL if source == "control"
            else CommandOrigin.PRIMARY
        )
        try:
            if mtype == "params":
                try:
                    pp = float(data.get("playback_pos", 0.0))
                except (TypeError, ValueError):
                    pp = 0.0
                streaming.set_knobs(
                    data.get("raw") or {}, pp, origin=origin,
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
            elif mtype == "set_timbre_strength":
                try:
                    v = float(data.get("value", 1.0))
                except (TypeError, ValueError):
                    v = 1.0
                streaming.set_timbre_strength(v, origin=origin)
            elif mtype == "set_timbre_source":
                name = data.get("name") or "timbre"
                try:
                    audio_msg = recv_audio()
                except ConnectionClosed:
                    state.running = False
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
                try:
                    audio_msg = recv_audio()
                except ConnectionClosed:
                    state.running = False
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
                    try:
                        audio_msg = recv_audio()
                    except ConnectionClosed:
                        state.running = False
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
                    key=data.get("key"),
                    time_signature=data.get("time_signature"),
                    fixture_name=data.get("fixture_name"),
                    stem_source_mode=data.get("stem_source_mode"),
                    origin=origin,
                )
            else:
                # Unknown mtype — log but don't crash; lets future
                # protocol additions degrade gracefully on older
                # servers.
                logger.warning(
                    "unknown_message_type origin={} mtype={}",
                    source, mtype,
                )
        except ConnectionClosed:
            state.running = False

    def recv_loop():
        while state.running:
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
                    if not state.running:
                        break
            except TimeoutError:
                pass
            except ConnectionClosed:
                state.running = False
                break
            except Exception as exc:
                logger.exception("recv_loop_error error={}", exc)
                state.running = False
                break

            # Drain the MCP / external control bus.
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

    # spawn_thread copies the parent context (loguru contextvars), so
    # logs emitted from inside recv_loop still carry session_id and
    # friends.
    recv_t = spawn_thread(recv_loop, name="recv_loop")

    # Register with the process-global session registry so the demo's
    # onboard MCP server can drive this session via the HTTP control
    # bus.
    session_registry.register(session_registry.SessionHandle(
        id=session_id,
        started_at=time.time(),
        inject=inject_control,
        snapshot=snapshot_session,
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
