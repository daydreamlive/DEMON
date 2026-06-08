"""MRT2Backend: Magenta RT 2 behind the GeneratorBackend seam.

The model is autoregressive (recurrentgemma over SpectroStream codec
tokens): it emits final 40 ms frames from a recurrent state and can
never revise them. That inverts most of the diffusion backend's
contract, and this class is the reference for how an append-only
family sits behind the seam:

* ``render_window`` IGNORES the runner's position hint and returns the
  next frontier chunk (the seam documents this for append-only
  backends). Committed audio is never re-rendered.
* Song shape is the rolling window (plan §3.6 v1): a synthetic
  ``WINDOW_S`` duration; the frontier writes advance modulo the window
  and the player loops it like any fixed song. The "song" is a tape
  loop continuously overwritten just behind the playhead.
* Each emitted chunk re-emits the previous chunk's final ``XFADE``
  samples at its head (overlap), so the runner's unconditional
  leading-edge crossfade blends new audio against identical samples —
  a no-op — instead of smearing every chunk start with last lap's
  stale audio.
* Generation runs out-of-process (JAX has no CUDA on native Windows):
  this class is a thin TCP client to ``scripts/mrt2_sidecar.py``
  (protocol: :mod:`acestep.streaming.mrt2.protocol`), pacing the
  sidecar with credit so the frontier stays ``mrt2_lead`` seconds
  ahead of the playhead. For an append-only model that buffered lead
  IS the knob-to-ear latency, so it is an operator knob, not a
  constant.

Conditioning: ``set_prompt`` / ``set_prompt_blend`` route here from
the session (backend control hooks); the sidecar embeds tags via
MusicCoCa and lerps A↔B embeddings for the blend. Sampling knobs
(temperature / top_k / three CFG scales) ride the ordinary params
channel as ``mrt2_``-prefixed bank knobs (group ``"mrt2"``) and are
forwarded to the sidecar when they move; they apply at 40 ms frame
granularity on the generation frontier.
"""

from __future__ import annotations

import os
import socket
import threading
import time
from collections import deque

import numpy as np

from acestep.engine.obs import logger
from acestep.streaming.generator_backend import (
    AudioChunk,
    AudioGeometry,
    Capabilities,
    LeadProfile,
    ProduceMode,
    TickContext,
)
from acestep.streaming.knobs import KnobSpec
from acestep.streaming.mrt2 import protocol as mp

# Rolling-window song shape (plan §3.6 v1): the synthetic duration the
# session declares and the player loops.
WINDOW_S = 60.0
WINDOW_SAMPLES = int(WINDOW_S * mp.SAMPLE_RATE)

# Overlap re-emitted at each chunk head so the runner's leading-edge
# crossfade blends against identical samples. Matches the runner's
# fade length (min(1200, len // 4) at 48 kHz = 25 ms).
XFADE = 1200

# Cap on one render's emission. Keeps a post-stall burst from writing
# a multi-second slab in one tick (and from out-running the wire).
MAX_EMIT_S = 1.5

# Heartbeat cadence / liveness deadline for the sidecar link.
PING_EVERY_S = 2.0
LOST_AFTER_S = 8.0

# Largest credit grant per tick. The sidecar holds at most ~this much
# un-asked-for work, so a stale grant can't run far past a knob change.
MAX_GRANT_FRAMES = 50


def mrt2_knob_specs() -> list:
    """The MRT2 family knob universe (also the homonym-guard manifest;
    see families.FAMILY_KNOB_UNIVERSES). All names carry the family
    prefix per plan §3.3 — none of these mean anything to another
    family. CFG bounds mirror the model's documented [-1, 7] range."""
    return [
        KnobSpec(
            "mrt2_temperature", default=1.3, min_val=0.0, max_val=4.0,
            group="mrt2",
            description="Sampling temperature for the token sampler.",
        ),
        KnobSpec(
            "mrt2_top_k", default=40, min_val=1.0, max_val=1024.0,
            type="int", group="mrt2",
            description="Top-k sampling threshold.",
        ),
        KnobSpec(
            "mrt2_cfg_musiccoca", default=3.0, min_val=-1.0, max_val=7.0,
            group="mrt2",
            description="MusicCoCa (style/prompt) CFG scale.",
        ),
        KnobSpec(
            "mrt2_cfg_notes", default=1.0, min_val=-1.0, max_val=7.0,
            group="mrt2",
            description="Notes-conditioning CFG scale.",
        ),
        KnobSpec(
            "mrt2_cfg_drums", default=1.0, min_val=-1.0, max_val=7.0,
            group="mrt2",
            description="Drums-conditioning CFG scale.",
        ),
        KnobSpec(
            "mrt2_lead", default=0.75, min_val=0.3, max_val=3.0,
            group="mrt2",
            description="Target generation lead over the playhead in "
                        "seconds. Append-only audio cannot be revised, "
                        "so this IS the knob-to-ear latency floor; "
                        "raise it for underrun safety, lower it for "
                        "responsiveness.",
        ),
    ]


# Knob name -> sidecar control field. (mrt2_lead is backend-local.)
_SIDECAR_KNOBS = {
    "mrt2_temperature": "temperature",
    "mrt2_top_k": "top_k",
    "mrt2_cfg_musiccoca": "cfg_musiccoca",
    "mrt2_cfg_notes": "cfg_notes",
    "mrt2_cfg_drums": "cfg_drums",
}


def sidecar_address() -> tuple:
    """Resolve the sidecar address from ``DEMON_MRT2_SIDECAR``
    (``host:port``), defaulting to the protocol module's localhost
    port. WSL2 forwards localhost, so the default reaches a sidecar
    inside WSL from the Windows-side server."""
    raw = os.environ.get("DEMON_MRT2_SIDECAR", "")
    if raw:
        host, _, port = raw.rpartition(":")
        return host or mp.DEFAULT_HOST, int(port)
    return mp.DEFAULT_HOST, mp.DEFAULT_PORT


class SidecarClient:
    """Blocking-socket client to the MRT2 sidecar with a reader thread.

    Audio chunks land in an internal queue the backend drains on the
    runner thread; control sends happen from whatever thread the
    session op runs on, serialized by a lock. All link failures
    degrade to ``lost = True`` (logged once) rather than raising into
    the runner loop — the session keeps serving its rolling buffer.
    """

    def __init__(self, host: str, port: int, connect_timeout_s: float = 5.0):
        self._addr = (host, port)
        self._sock = socket.create_connection((host, port), timeout=connect_timeout_s)
        self._send_lock = threading.Lock()
        self._audio: deque = deque()
        self._audio_lock = threading.Lock()
        self.frames_received = 0
        self.last_rx_ts = time.monotonic()
        self.lost = False
        self.meta: dict = {}

        # Handshake: hello -> meta, still on the connect timeout.
        self._sock.sendall(mp.pack_json({"type": "hello"}))
        kind, payload = mp.recv_msg(self._sock)
        if kind != mp.MSG_JSON:
            raise ConnectionError("mrt2 sidecar: expected meta, got audio")
        meta = mp.unpack_json(payload)
        if meta.get("type") != "meta":
            raise ConnectionError(f"mrt2 sidecar: expected meta, got {meta.get('type')!r}")
        if (
            meta.get("sample_rate") != mp.SAMPLE_RATE
            or meta.get("channels") != mp.CHANNELS
            or meta.get("frame_samples") != mp.FRAME_SAMPLES
        ):
            raise ConnectionError(f"mrt2 sidecar: geometry mismatch: {meta}")
        self.meta = meta

        self._sock.settimeout(None)
        self._reader = threading.Thread(
            target=self._read_loop, name="mrt2-sidecar-reader", daemon=True,
        )
        self._reader.start()

    # ---- reader thread ----------------------------------------------------

    def _read_loop(self) -> None:
        try:
            while True:
                kind, payload = mp.recv_msg(self._sock)
                self.last_rx_ts = time.monotonic()
                if kind == mp.MSG_AUDIO:
                    _idx, num_frames, pcm = mp.unpack_audio(payload)
                    arr = np.frombuffer(pcm, dtype=np.float32).reshape(
                        -1, mp.CHANNELS,
                    )
                    with self._audio_lock:
                        self._audio.append(arr)
                        self.frames_received += num_frames
                else:
                    msg = mp.unpack_json(payload)
                    if msg.get("type") == "err":
                        logger.error("mrt2_sidecar_err message={}", msg.get("message"))
                    # pong and anything else: last_rx_ts update is enough.
        except (ConnectionError, OSError) as exc:
            if not self.lost:
                self.lost = True
                logger.error("mrt2_sidecar_lost reader error={}", exc)

    # ---- senders (any thread) ----------------------------------------------

    def send_json(self, obj: dict) -> None:
        if self.lost:
            return
        try:
            with self._send_lock:
                self._sock.sendall(mp.pack_json(obj))
        except (ConnectionError, OSError) as exc:
            if not self.lost:
                self.lost = True
                logger.error("mrt2_sidecar_lost send error={}", exc)

    # ---- runner-thread drains ----------------------------------------------

    def pop_audio(self) -> list:
        with self._audio_lock:
            out = list(self._audio)
            self._audio.clear()
        return out

    def close(self) -> None:
        self.lost = True
        try:
            self._sock.close()
        except OSError:
            pass


class MRT2Backend:
    """Magenta RT 2 generation behind the GeneratorBackend seam.

    Append-only, sidecar-hosted, rolling-window song shape. See module
    docstring; the seam-contract notes live on each method.
    """

    name = "mrt2"

    def __init__(self, *, config, state, midi_knobs, client: SidecarClient | None = None):
        self.config = config
        self.state = state
        self.midi_knobs = midi_knobs

        # Runner slice-width bookkeeping (PipelineRunner reads this for
        # its stall/shortfall math and the windowed-mode switch). The
        # emission itself is frontier-driven and variable-length.
        self.vae_window = 1.0
        self.decode_span_s = 0.0
        self.last_tick_ms = 0.0
        self.last_dec_ms = 0.0

        owns_client = client is None
        if client is None:
            host, port = sidecar_address()
            try:
                client = SidecarClient(host, port)
            except (ConnectionError, OSError, ValueError) as exc:
                # Loud at session create (plan: config-time failure, the
                # client asked for a generator this deployment isn't
                # running).
                raise RuntimeError(
                    f"mrt2 sidecar unreachable at {host}:{port} "
                    f"(start scripts/mrt2_sidecar.py in the MRT2 venv, "
                    f"or set DEMON_MRT2_SIDECAR): {exc}"
                ) from exc
        self.client = client
        logger.info("mrt2_sidecar_connected meta={}", client.meta)

        # The TCP link is a real acquired resource: from here on, any
        # failure before __init__ returns (e.g. the initial prompt seed
        # below hitting a sidecar protocol error) must not strand it —
        # the caller can't close a backend it never received. Mirrors
        # ModelContext/Session's transactional constructors. Injected
        # clients stay caller-owned and are left alone.
        try:

            # ---- frontier / rolling-window state (runner thread only) ----
            # Absolute samples handed to the runner (exclusive); the wire
            # position is this modulo WINDOW_SAMPLES.
            self._abs_written = 0
            # Pending PCM drained from the client but not yet emitted.
            self._pending: deque = deque()
            self._pending_samples = 0
            # Last XFADE emitted samples, re-emitted at the next chunk head.
            self._tail: np.ndarray | None = None

            # Unwrapped playhead estimate (laps over the rolling window).
            self._playhead_wrapped_prev = 0.0
            self._playhead_laps = 0

            # Credit accounting (frames granted vs received).
            self._granted = 0

            # Last knob values forwarded to the sidecar.
            self._sent_knobs: dict = {}

            self._last_ping_ts = 0.0
            self._lost_logged = False

            # Stashed for the params echo.
            self._echo: dict = {}

            # Initial conditioning: the session seeds prompts into state
            # before the backend is constructed.
            self.handle_set_prompt(
                getattr(state, "prompt_text", "") or "",
                tags_b=getattr(state, "prompt_text_b", None),
            )
        except BaseException:
            if owns_client:
                try:
                    client.close()
                except Exception:
                    pass
            raise

    # ---- contract ----------------------------------------------------------

    def capabilities(self) -> Capabilities:
        # Everything defaults False: no refinement (append-only), no
        # swap/timbre/structure/stems (no positional source), no LoRA,
        # no depth, no curves. notes_conditioning is real in the model
        # but not wired yet (plan Phase 3.5).
        return Capabilities()

    def geometry(self) -> AudioGeometry:
        return AudioGeometry(
            sample_rate=mp.SAMPLE_RATE,
            channels=mp.CHANNELS,
            chunk_rate_hz=1.0 / mp.FRAME_SECONDS,
            duration_s=WINDOW_S,
        )

    def lead_profile(self) -> LeadProfile:
        # The runner's adaptive lead positions diffusion re-renders;
        # append-only emission ignores the position hint, so the
        # runner defaults are fine. The lead that matters here is the
        # mrt2_lead knob (credit pacing).
        return LeadProfile()

    def knob_specs(self, lora_ids=()) -> list:
        return mrt2_knob_specs()

    # ---- session control hooks ----------------------------------------------

    def handle_set_prompt(self, tags: str, *, tags_b: str | None = None) -> None:
        """Backend route for the universal ``set_prompt`` op: forward
        the raw tags; the sidecar owns MusicCoCa embedding (and caches
        per-tags embeddings, so blend moves don't re-embed)."""
        self.client.send_json({
            "type": "prompt", "tags": tags or "", "tags_b": tags_b,
        })

    def handle_set_prompt_blend(self, value: float) -> None:
        """Backend route for ``set_prompt_blend``: the sidecar lerps
        the cached A/B style embeddings."""
        self.client.send_json({"type": "blend", "value": float(value)})

    def close(self) -> None:
        self.client.close()

    # ---- hot loop -------------------------------------------------------------

    def sync_source(self, ctx: TickContext) -> None:
        # No positional source: nothing to reconcile. Unwrap the
        # playhead here (once per tick, before produce) so credit
        # pacing sees a monotonic clock across window laps.
        pos = ctx.playhead_s
        if pos < self._playhead_wrapped_prev - WINDOW_S * 0.5:
            self._playhead_laps += 1
        self._playhead_wrapped_prev = pos

    def read_knobs(self) -> dict:
        return self.midi_knobs.get_all_values()

    def has_pending_refit(self) -> bool:
        return False

    def rebuild_imminent(self, knobs: dict) -> bool:
        return False

    def has_renderable_state(self) -> bool:
        return self._abs_written > 0 or self._pending_samples > 0

    def playable_duration_s(self):
        return WINDOW_S

    def _playhead_abs_samples(self) -> int:
        return int(
            (self._playhead_laps * WINDOW_S + self._playhead_wrapped_prev)
            * mp.SAMPLE_RATE
        )

    def _forward_knobs(self, raw: dict) -> None:
        update = {}
        for knob, field in _SIDECAR_KNOBS.items():
            val = raw.get(knob)
            if val is None:
                continue
            val = int(val) if field == "top_k" else round(float(val), 4)
            if self._sent_knobs.get(field) != val:
                update[field] = val
        if update:
            self._sent_knobs.update(update)
            self.client.send_json({"type": "knobs", **update})

    def produce(self, knobs: dict, ctx: TickContext, mode: ProduceMode) -> bool:
        """One tick: forward control changes, pace the sidecar with
        credit, drain arrived audio. Modes are a no-op distinction
        here — there is no expensive local generate step to skip, and
        music must keep flowing through DiT-pause idle ("reuse"), so
        every mode runs the same pull path."""
        t0 = time.perf_counter()
        now = time.monotonic()

        client = self.client
        if not client.lost:
            # Heartbeat + liveness.
            if now - self._last_ping_ts >= PING_EVERY_S:
                self._last_ping_ts = now
                client.send_json({"type": "ping", "t": now})
            if now - client.last_rx_ts > LOST_AFTER_S:
                client.lost = True
                logger.error(
                    "mrt2_sidecar_lost reason=liveness deadline_s={}",
                    LOST_AFTER_S,
                )

            self._forward_knobs(knobs)

            # Credit pacing: keep (emitted + pending + outstanding)
            # ``mrt2_lead`` seconds ahead of the unwrapped playhead.
            lead_s = float(knobs.get("mrt2_lead", 0.75))
            outstanding = self._granted - client.frames_received
            covered = (
                self._abs_written
                + self._pending_samples
                + max(0, outstanding) * mp.FRAME_SAMPLES
            )
            target = self._playhead_abs_samples() + int(lead_s * mp.SAMPLE_RATE)
            deficit = target - covered
            if deficit > 0:
                grant = min(
                    MAX_GRANT_FRAMES,
                    -(-deficit // mp.FRAME_SAMPLES),  # ceil div
                )
                self._granted += grant
                client.send_json({"type": "credit", "frames": int(grant)})

        for arr in client.pop_audio():
            self._pending.append(arr)
            self._pending_samples += arr.shape[0]

        self._echo = {k: knobs.get(k) for k in _SIDECAR_KNOBS}
        self._echo["mrt2_lead"] = knobs.get("mrt2_lead")
        self.last_tick_ms = (time.perf_counter() - t0) * 1000

        if self._pending_samples == 0:
            # Nothing landed this tick. The ACE backend's GPU step paces
            # the loop; here the socket does, so nap briefly instead of
            # spinning the runner at CPU speed.
            time.sleep(0.01)
            return False
        return True

    def render_window(self, t_start_s: float):
        """Emit the next frontier chunk. The position hint is ignored
        (append-only: there is exactly one place new audio can go).
        Returns None when no new frames are pending — the runner's
        gap-fill tick then writes nothing, which is correct: committed
        audio is already in the buffer and never changes."""
        if self._pending_samples == 0:
            return None
        t0 = time.perf_counter()

        # Gather up to MAX_EMIT_S, clamped at the rolling-window edge
        # so a chunk never spans the wrap seam (the remainder stays
        # pending for the next tick, ~one loop iteration later).
        wrapped_start = self._abs_written % WINDOW_SAMPLES
        room = WINDOW_SAMPLES - wrapped_start
        budget = min(int(MAX_EMIT_S * mp.SAMPLE_RATE), room)

        parts = []
        taken = 0
        while self._pending and taken < budget:
            arr = self._pending.popleft()
            if taken + arr.shape[0] > budget:
                cut = budget - taken
                parts.append(arr[:cut])
                self._pending.appendleft(arr[cut:])
                taken = budget
            else:
                parts.append(arr)
                taken += arr.shape[0]
        self._pending_samples -= taken
        new_pcm = parts[0] if len(parts) == 1 else np.concatenate(parts)

        # Overlap head: re-emit the tail of the previous emission so
        # the runner's leading-edge crossfade blends identical samples.
        # Skipped when it would cross the wrap seam backwards (once per
        # lap) and on the very first chunk.
        head = None
        if self._tail is not None and 0 < XFADE <= wrapped_start:
            head = self._tail[-XFADE:]

        if head is not None:
            pcm = np.concatenate([head, new_pcm])
            start_sample = wrapped_start - head.shape[0]
        else:
            pcm = np.array(new_pcm, copy=True)  # runner mutates in place
            start_sample = wrapped_start

        # Update the tail from pristine data BEFORE handing the chunk
        # out (the runner crossfades win_np in place).
        if self._tail is None:
            self._tail = np.array(new_pcm[-XFADE:], copy=True)
        else:
            self._tail = np.concatenate([self._tail, new_pcm])[-XFADE:].copy()

        self._abs_written += taken
        self.last_dec_ms = (time.perf_counter() - t0) * 1000
        return AudioChunk(pcm=pcm, start_sample=int(start_sample))

    def render_full(self):
        # Legacy full-buffer mode (vae_window <= 0) never applies: this
        # backend always declares a positive window.
        return None

    # ---- bookkeeping ------------------------------------------------------

    def on_fresh_generation(self, knobs: dict) -> None:
        params = self.state.params
        params["num_gens"] = params.get("num_gens", 0) + 1
        params["tick_ms"] = self.last_tick_ms
        params["dec_ms"] = self.last_dec_ms
        for name, val in self._echo.items():
            if val is None:
                continue
            params[name] = round(float(val), 3)
        params["_prompt"] = self.state.prompt_text


# ---------------------------------------------------------------------------
# Session creation (families.SESSION_CREATORS["mrt2"])
# ---------------------------------------------------------------------------


def create_mrt2_session(cls, *, audio, config, checkpoint, session_id, **_unused):
    """Build a StreamingSession for the MRT2 family.

    The ACE create path (TRT profiles, model load, demucs, conditioning
    encode) doesn't apply: the model lives in the sidecar. What a
    session needs here is the rolling-window buffer, the family knob
    bank, and a SessionState whose source-derived metadata is honestly
    absent — geometry/capabilities in ``ready`` are the declared truth
    (plan §3.6: nullable metadata rides the capability mask).

    ``audio`` (the handshake upload/fixture) is intentionally ignored:
    there is no source to swap in, and starting from silence is honest
    — the frontier overwrites the window from t=0. ``checkpoint`` is
    recorded but unused; the sidecar picks the model variant at ITS
    launch (``--model``).
    """
    from contextlib import ExitStack

    from acestep.streaming.audio_engine import AudioEngine
    from acestep.streaming.knobs import KnobState
    from acestep.streaming.session import _cleanup_create_resource
    from acestep.streaming.state import SessionState

    if audio is not None and getattr(audio, "waveform", None) is not None:
        logger.info(
            "mrt2_create_ignoring_source duration_s={:.1f} reason=append_only_family",
            audio.waveform.shape[-1] / mp.SAMPLE_RATE,
        )

    buf = np.zeros((WINDOW_SAMPLES, mp.CHANNELS), dtype=np.float32)
    prompt = config.prompt or ""
    prompt_b = config.prompt_b if config.prompt_b is not None else prompt

    state = SessionState(
        source=None,
        bpm=0,
        key="",
        time_signature="4/4",
        duration=WINDOW_S,
        n_channels=mp.CHANNELS,
        playback_samples=WINDOW_SAMPLES,
        cond_pair=None,
        cond_pair_b=None,
        prompt_text=prompt,
        prompt_text_b=prompt_b,
        current_depth=1,
    )

    # Same transactional create shape as StreamingSession.create's ACE
    # body: resources acquired before ``cls(...)`` succeeds register on
    # the stack the moment they exist; ownership transfers via
    # ``pop_all()`` only on success. The sidecar TCP link itself is
    # acquired inside ``cls(...)`` (MRT2Backend.__init__, via
    # make_backend) — its no-leak guarantee lives in the transactional
    # constructors (MRT2Backend.__init__ closes a self-acquired client
    # on failure; StreamingSession.__init__ closes the backend if init
    # fails after make_backend).
    with ExitStack() as cleanup:
        audio_eng = AudioEngine(buf, mp.SAMPLE_RATE)
        cleanup.callback(
            _cleanup_create_resource,
            "audio_engine",
            audio_eng.stop,
        )
        streaming = cls(
            session_id=session_id,
            checkpoint=checkpoint,
            config=config,
            engine_session=None,
            stream=None,
            state=state,
            audio_eng=audio_eng,
            virtual_knobs=KnobState(mrt2_knob_specs()),
            engine_obj=None,
            profile_mgr=None,
            cond_negative=None,
            initial_buffer=buf,
            initial_upload_stems=None,
            initial_stem_error=None,
            initial_stem_source_mode=None,
            initial_enable_ids=[],
            lora_strengths_init={},
            lora_available=False,
            max_pipeline_depth=1,
            max_seconds=WINDOW_S,
            walk_window=False,
            walk_window_s=WINDOW_S,
            vae_window=1.0,
            crop_seconds=0.0,
            use_sde=False,
            use_lora=False,
            k1_name="mrt2_temperature",
        )
        cleanup.pop_all()
        return streaming
