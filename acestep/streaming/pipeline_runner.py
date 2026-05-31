"""PipelineRunner: the torch-heavy streaming loop (graph-driven).

Drives a :class:`~acestep.engine.session.StreamHandle` by calling
``handle.tick(**kwargs)`` each iteration, where the kwargs mirror the
knob state.
"""

import os
import time

import numpy as np
import torch

from acestep.engine.dcw import DCWAdvanced
from acestep.engine.obs import logger
from acestep.nodes.types import ChannelGuidanceEntry, Latent
from acestep.nodes.interpolation import INTERPOLATIONS
from acestep.nodes.vae_nodes import EmptyLatent, LatentBlend

from acestep.streaming.knobs import CHANNEL_GROUPS, KEYSTONE_CHANNELS

# Audio sample rate the ACE-Step v1.5 family is trained on. Duplicated
# from ``demos/realtime_motion_graph_web/protocol.py`` (and many other
# call sites — see tests/, scripts/) so this module stays free of demo
# imports. ``T`` is the latent frame count for a 60 s window at the
# tokenizer's 25 fps; both constants are model invariants the runner
# uses as plain magic numbers.
SAMPLE_RATE = 48000
T = 1500

# Hot-loop trace sampling. Cached at import time so the per-tick branch
# is a single int compare. Env-tunable since the loop runs at ~125 Hz —
# emitting every tick at TRACE level would dominate the log volume.
# 0 disables sampled tracing entirely regardless of log level.
try:
    _TRACE_SAMPLE_EVERY = max(0, int(os.environ.get("DEMON_TRACE_SAMPLE_EVERY", "50")))
except ValueError:
    _TRACE_SAMPLE_EVERY = 50

# Largest tap index the feedback delay can address. Matches the
# ``feedback_depth`` knob's ``max_val`` (knobs.py) and the SLIDER_META
# max in web/types/engine.ts; the three must stay in sync. depth=1
# reproduces the original behavior (blend with the most recent
# finished latent).
MAX_FEEDBACK_DEPTH = 8


def _fixed_windowed_vae_span_s() -> float:
    """Return the fixed VAE decode span in seconds, or 0 for dynamic engines."""
    try:
        from acestep.paths import WINDOWED_VAE_PROFILE_FRAMES
    except Exception:
        return 0.0
    pmin, popt, pmax = WINDOWED_VAE_PROFILE_FRAMES
    if pmin == popt == pmax and pmin > 0:
        return pmin / 25.0
    return 0.0


class _RemotePlayheadClock:
    """Monotonic estimate of the client's audible playhead.

    The browser sends periodic absolute playback positions over the params
    channel. Those messages are the authority, but they can arrive slower
    than the runner loop or be coalesced under load. This clock anchors on
    the most recent observed sample and advances by wall time between
    anchors, so VAE scheduling remains continuous even when controls and
    WebSocket heartbeats are quiet.
    """

    def __init__(self, audio_eng):
        self.audio_eng = audio_eng
        self._observed = int(audio_eng.position)
        self._anchor_sample = int(audio_eng.position)
        self._anchor_wall_s = time.monotonic()

    def sample(self) -> int:
        n = max(1, len(self.audio_eng.current))
        now = time.monotonic()
        observed = int(self.audio_eng.position) % n
        if observed != self._observed:
            self._observed = observed
            self._anchor_sample = observed
            self._anchor_wall_s = now
        elapsed = max(0.0, now - self._anchor_wall_s)
        return int(self._anchor_sample + elapsed * SAMPLE_RATE) % n

    def seconds(self) -> float:
        return self.sample() / SAMPLE_RATE


def _build_dcw_advanced(raw: dict) -> "DCWAdvanced | None":
    """Translate the client's three DCW fader values into a
    :class:`DCWAdvanced`, or return ``None`` when all three are zero.

    Returning ``None`` lets the corrector take its byte-identical fast
    path, so "all faders at the bottom" costs nothing over upstream DCW.
    """
    mult_blend = float(raw.get("dcw_mult_blend", 0.0))
    mag_phase = float(raw.get("dcw_mag_phase", 0.0))
    soft_thresh = float(raw.get("dcw_soft_thresh", 0.0))
    if mult_blend == 0.0 and mag_phase == 0.0 and soft_thresh == 0.0:
        return None
    return DCWAdvanced(
        mult_blend=mult_blend,
        mag_phase=mag_phase,
        soft_thresh=soft_thresh,
    )



def _curve_from_spec(spec, T):
    # Convert a client curve spec (constant/raw) into a (1, T, 1) tensor,
    # or return None if not supplied. Matches buildCurveSpec on the VST.
    import torch as _t
    if not isinstance(spec, dict):
        return None
    kind = spec.get("type", "constant")
    if kind == "constant":
        return _t.full((1, T, 1), float(spec.get("value", 1.0)), dtype=_t.float32)
    if kind == "raw":
        vals = spec.get("values", [])
        if not vals:
            return None
        t = _t.tensor(vals, dtype=_t.float32)
        if t.numel() != T:
            t = _t.nn.functional.interpolate(
                t.view(1, 1, -1), size=T, mode="linear", align_corners=True
            ).view(T)
        return t.view(1, T, 1)
    return None


class PipelineRunner:
    """Extracted pipeline loop.  Identical semantics to the pre-Phase-3
    closure, now wired through the node graph.

    One injection point: *on_audio_ready* receives decoded audio.
    ``on_audio_ready(wav_np)``                     -- full-buffer decode
    ``on_audio_ready(wav_np, win_start, win_end)`` -- windowed decode
    """

    def __init__(
        self, session, stream, audio_eng, *,
        state,
        use_midi, use_sde, use_lora,
        midi_knobs, engine_obj,
        vae_window, crop_seconds,
        k1_name, seed, skip_threshold,
        on_audio_ready=None,
        before_tick=None,
        walk_window=False,
        walk_window_s=60.0,
        neg_conditioning=None,
        idle_threshold_s=0.0,
    ):
        self.session = session
        self.stream = stream  # StreamHandle
        self.audio_eng = audio_eng
        # Single mutable session state object. The runner
        # reads ``state.running``, ``state.params``, ``state.prompt_text``,
        # ``state.last_activity_ts``, ``state.motion_val``, and
        # ``state.sde_curve_display``; it takes ``state._lock`` to read
        # motion atomically (per the pre-refactor motion_lock contract).
        self.state = state
        self.use_midi = use_midi
        self.use_sde = use_sde
        self.use_lora = use_lora
        self.midi_knobs = midi_knobs
        self.engine_obj = engine_obj
        # Use the effective Session value after engine-profile clamps.
        # The web config may still carry an old multi-second value, but
        # scheduling must match the slice length session.decode() emits.
        self.vae_window = float(getattr(session, "_vae_window", vae_window))
        self.crop_seconds = crop_seconds
        self.k1_name = k1_name
        self.SEED = seed
        self.skip_threshold = skip_threshold
        # Default callback: in full-buffer mode, hand off to
        # ``audio_eng.swap`` (legacy crossfade-on-swap path). In windowed
        # mode the runner has already written into the audio engine via
        # ``patch_window`` before invoking the callback, so the default
        # is a no-op there — callers that only want a side-effect (delta
        # send, monitoring) override this and skip the swap themselves.
        if on_audio_ready is None:
            def on_audio_ready(wav, win_start=None, win_end=None):
                if win_start is None:
                    audio_eng.swap(wav)
        self.on_audio_ready = on_audio_ready
        # before_tick: optional callable invoked at the top of every loop
        # iteration on the runner thread.  Used by the web server to
        # apply cross-thread mutations safely:
        #   - LoRA enable/disable (which triggers a refit; refit and
        #     inference are mutually exclusive)
        #   - source swap (prepare_source / encode_text / replace stream
        #     fields, which can't race the recv thread that holds the
        #     WebSocket)
        # The server's apply_pending() callback drains both queues each
        # iteration so they share one rendezvous point.
        self.before_tick = before_tick

        # Walk-window mode: drive the DiT with a fixed-T window sliced
        # from a longer pre-encoded source so the 60s TRT engine can
        # serve a multi-minute song. The source is split into
        # walk_window_s chunks (typically 60s == one engine slot worth
        # of latent); the runner picks the chunk that contains the
        # current playhead and feeds that SAME slice to the DiT for the
        # duration of the chunk. The slice only advances when the
        # playhead crosses a chunk boundary — not every tick — so the
        # ring buffer gets a steady source to denoise against and the
        # engine's parameter-update latency stays at the 60s engine's
        # smaller value.
        #
        # Requires ``stream.source.latent`` and
        # ``stream.source.context_latent`` to have been pre-encoded
        # against the FULL source (vae_encode profile must fit the
        # whole song even though the DiT/decoder run at walk_window_s).
        self.walk_window = bool(walk_window)
        self.walk_window_s = float(walk_window_s)
        self.walk_window_T = int(round(self.walk_window_s * 25.0))

        # Negative conditioning for the RCFG path. Encoded once at session
        # start (see StreamingSession.create) and reused across all ticks.
        # Required for ``rcfg_mode in {"full", "initialize"}``; ignored
        # by ``rcfg_mode == "self"`` (virtual uncond) and ``"off"``.
        # ``None`` is safe — modes that need it become quiet no-ops.
        self.neg_conditioning = neg_conditioning

        # Idle GPU pause. Two-stage shutdown:
        #   1) After ``idle_threshold_s`` with no inbound activity, skip
        #      ``self.stream.tick()`` (the dominant per-tick GPU cost)
        #      and reuse the most recent cached ``result_latent`` for
        #      the rest of the loop body. The VAE keeps windowed-
        #      decoding at the advancing playhead so audio continues
        #      uninterrupted, sending deltas to refresh the client's
        #      buffer from the stable cached latent.
        #   2) Once the playhead has wrapped through one full cycle
        #      since DiT paused, the client has the full denoised
        #      buffer; further decodes would produce identical audio.
        #      The VAE also stops (sleep+continue) until activity
        #      resumes. Any incoming WS message clears both stages.
        # Hot path is untouched when active or when disabled
        # (``idle_threshold_s <= 0``).
        self._idle_threshold_s = float(idle_threshold_s)
        self._last_result_latent = None
        self._dit_paused = False
        self._vae_paused = False
        self._dit_paused_at_wall_s = 0.0

        # Predictive decode: rolling EMA of (tick + decode) wall time.
        # Dynamic-window decoders use this as the target lead. Fixed 1 s
        # decoders also apply a floor from the decode span below so the
        # tiny kept wire slice does not collapse DiT lookahead.
        self._predicted_advance_s = 0.1
        # ``vae_window`` is now the kept wire slice for fixed-profile
        # windowed VAE decoders, not the full decode span. Keep the
        # patch cadence on ``vae_window`` for coverage, but lead the
        # decode target by the fixed engine's span so the DiT has time
        # to settle the region before the listener reaches it.
        fixed_vae_span_s = _fixed_windowed_vae_span_s()
        self._vae_advance_floor_s = (
            fixed_vae_span_s * 0.5
            if fixed_vae_span_s > self.vae_window > 0
            else 0.0
        )
        self._vae_advance_cap_s = max(self.vae_window, fixed_vae_span_s) * 0.5
        self._playhead_clock = _RemotePlayheadClock(self.audio_eng)

        # Cache silence once; used by the hint-strength blend node.
        self._rebuild_silence_latent()

        # Hint-strength gating: the run loop only re-runs the
        # silence/context blend when the slider value moves by > 0.02.
        # Outside callers that change ``stream.source.context_latent``
        # under the runner's feet (e.g. the structure-override upload
        # path on the recv thread) need a way to force the next tick
        # to re-blend even when the slider hasn't moved. ``mark_hint_dirty``
        # flips this flag and the run loop honors it on the next pass.
        self._hint_dirty = False

    def mark_hint_dirty(self) -> None:
        """Force ``_update_hint_strength`` to fire on the next tick.

        Use after replacing ``stream.source.context_latent`` (e.g. on
        structure-override apply / clear or after a source swap) so the
        runner re-blends silence ↔ context at the current
        ``hint_strength`` and writes a fresh ``stream.context_latent``
        for the diffusion step to read. Without this, the diffusion
        keeps reading the previously-blended tensor until the operator
        nudges the slider.
        """
        self._hint_dirty = True

    def _rebuild_silence_latent(self) -> None:
        """(Re)build the silence latent used by hint-strength blending.

        Picks the right T for the *current* hint-blend target: in walk
        mode that's the per-tick window slice (``walk_window_T``); in
        non-walk mode it's the full source latent. ``walk_window=True``
        with a source shorter than the window degrades to non-walk
        per-tick (``walk_active`` is computed in ``run()``) and the
        per-tick guard there will rebuild this if the size disagrees.
        """
        full_src_T = self.stream.source.latent.tensor.shape[1]
        walk_active = self.walk_window and full_src_T > self.walk_window_T
        T_frames = self.walk_window_T if walk_active else full_src_T
        self._silence_latent = EmptyLatent().execute(
            model=self.stream.model, duration=T_frames / 25.0,
        )["latent"]

    def _decode_advance_s(self) -> float:
        """Lead windowed VAE decode targets ahead of the live playhead."""
        if self.vae_window <= 0:
            return 0.0
        advance_s = max(self._predicted_advance_s, self._vae_advance_floor_s)
        if self._vae_advance_cap_s > 0:
            advance_s = min(advance_s, self._vae_advance_cap_s)
        return advance_s

    def _playhead_seconds_now(self) -> float:
        return self._playhead_clock.seconds()

    def _update_hint_strength(self, hint_str: float) -> None:
        """Blend source context with silence by ``hint_str`` into the handle.

        0.0 = no structural guidance, 1.0 = full hints. Takes effect on
        the next ``handle.tick`` call.
        """
        if hint_str >= 1.0:
            self.stream.context_latent = self.stream.source.context_latent
            return
        self.stream.context_latent = LatentBlend().execute(
            latent_a=self._silence_latent,
            latent_b=self.stream.source.context_latent,
            alpha=hint_str,
            method=self.state.interp_structure,
        )["latent"]

    def _sync_channel_guidance(self, raw: dict, last: list) -> list:
        """Push channel gains onto the handler when any knob moved.

        Reads live from ``handler._channel_guidance`` inside the
        ``StreamDenoise`` node every tick, so writing the list here is
        sufficient — no pipeline mutation needed.
        """
        ch_gains = (
            [raw.get(name, 1.0) for name, _, _ in CHANNEL_GROUPS]
            + [raw.get(name, 1.0) for name, _ in KEYSTONE_CHANNELS]
        )
        if ch_gains == last:
            return last

        configs = []
        for (name, ch_start, ch_end) in CHANNEL_GROUPS:
            scale = raw.get(name, 1.0)
            if abs(scale - 1.0) > 0.01:
                configs.append(ChannelGuidanceEntry(
                    channel_start=ch_start, channel_end=ch_end, scale=scale,
                ))
        for (name, ch) in KEYSTONE_CHANNELS:
            scale = raw.get(name, 1.0)
            if abs(scale - 1.0) > 0.01:
                configs.append(ChannelGuidanceEntry(
                    channel_start=ch, channel_end=ch, scale=scale,
                ))
        self.stream.model.handler._channel_guidance = configs
        return ch_gains[:]

    def run(self):
        last_latent = None
        # Ring of past finished latents for the feedback delay-tap.
        # latent_history[0] is the most recent generation (== last_latent
        # when set), latent_history[1] is one tick older, etc. Capped at
        # MAX_FEEDBACK_DEPTH because longer history would just sit unused
        # — the knob can't reach past that anyway.
        from collections import deque
        latent_history: deque = deque(maxlen=MAX_FEEDBACK_DEPTH)
        last_wav = None
        last_hint_str = 1.0
        last_channel_gains = [1.0] * (len(CHANNEL_GROUPS) + len(KEYSTONE_CHANNELS))
        current_shift = self.stream.base_kwargs["shift"]
        prev_src_T = self.stream.source.latent.tensor.shape[1]
        # Source-tensor identity tracking. Lets walk mode detect a source
        # swap when the new song happens to have the same latent length as
        # the old one (T-only check would miss it).
        prev_src_id = id(self.stream.source.latent.tensor)
        # Walk-mode chunk anchor (in latent frames). -1 forces the first
        # walk-active tick to "transition" into chunk 0 and reset caches.
        prev_walk_w0 = -1
        # Cached slice tensors for the active chunk. Invalidated whenever
        # ``prev_walk_w0`` is reset (source swap or chunk transition).
        cached_live_src_lat = None
        cached_live_ctx_raw_t = None
        logger.info(
            "stream decode: vae_window={:.3f}s decode_span={:.3f}s "
            "advance=[{:.3f},{:.3f}]s walk_window_s={:.3f}",
            self.vae_window,
            _fixed_windowed_vae_span_s(),
            self._vae_advance_floor_s,
            self._vae_advance_cap_s,
            self.walk_window_s,
        )

        while self.state.running:
            if self.before_tick is not None:
                # Hook for cross-thread mutations (LoRA enable/disable
                # AND source swap).  Runs on the runner thread so any
                # GPU/refit work the callback does is serialized with
                # the tick body.
                self.before_tick()

            # Idle GPU pause — stage detection. Updates the flags read
            # at the ``stream.tick()`` call site below, and at the end
            # of the iteration (where ``result_latent`` is cached for
            # the next DiT-paused iteration). Disabled when no shared
            # activity ref was supplied or the threshold is non-
            # positive (standalone callers fall through to the normal
            # path with zero overhead).
            idle_active = (
                self._idle_threshold_s > 0.0
                and (time.monotonic() - self.state.last_activity_ts) >= self._idle_threshold_s
            )
            if not idle_active:
                # Activity resumed — clear both stages.
                if self._dit_paused or self._vae_paused:
                    self._dit_paused = False
                    self._vae_paused = False
                    logger.info("pipeline_resumed stage=dit")
            else:
                # Idle. Enter DiT-pause on first hit; the VAE-pause
                # check fires once enough wall time has passed for the
                # advancing playhead to refresh every chunk of the
                # client's audio buffer with audio decoded from the
                # cached latent.
                if not self._dit_paused:
                    self._dit_paused = True
                    self._dit_paused_at_wall_s = time.monotonic()
                    logger.info(
                        "pipeline_paused stage=dit idle_threshold_s={:.0f}",
                        self._idle_threshold_s,
                    )

                if self._vae_paused:
                    # Both stages reached — full GPU idle. Nap and
                    # re-evaluate. 50ms wake granularity is well below
                    # human-perceptible resume latency.
                    time.sleep(0.05)
                    continue

                # Audio plays at real-time rate, so wall-clock since
                # DiT paused is equivalent to playhead advance. Once
                # we've passed one full buffer-duration's worth of
                # wall time, every chunk has been re-decoded from the
                # cached latent and shipped to the client; further
                # decodes would produce identical samples. Robust
                # against modulo wrap-around edge cases that a
                # playhead-position diff would miss when the runner
                # stalls or the buffer length changes.
                buf_dur_s = max(
                    1e-6, len(self.audio_eng.current) / SAMPLE_RATE,
                )
                wall_since_pause_s = time.monotonic() - self._dit_paused_at_wall_s
                if (
                    self._last_result_latent is not None
                    and wall_since_pause_s >= buf_dur_s
                ):
                    self._vae_paused = True
                    logger.info("pipeline_paused stage=vae reason=buffer_full")
                    time.sleep(0.05)
                    continue

                # If we don't have a cached latent yet (idle hit before
                # any successful tick), fall through to the normal
                # path — calling stream.tick() is the only way to
                # produce one. Rare in practice.

            # Walk-window mode selection. Only active when the source
            # actually has more frames than the window — short sources
            # fall back to whole-source submission so walk_window=True is
            # safe to leave on regardless of song length.
            full_src_T = self.stream.source.latent.tensor.shape[1]
            walk_active = self.walk_window and full_src_T > self.walk_window_T

            # Pick the static chunk that covers the current playhead.
            # The slice stays the same for the entire walk_window_s of
            # playback that sits inside this chunk — only when the
            # playhead crosses into the next chunk does ``w0`` advance.
            # Use ``playhead + decode_advance`` so the swap happens a
            # tick or two before the boundary, giving the new chunk's
            # ring-buffer warmup head room to land before the listener
            # actually crosses.
            walk_w0 = -1
            walk_w1 = -1
            walk_chunk_start_s = 0.0
            if walk_active:
                playhead_now_s = self._playhead_seconds_now()
                advance_s_for_chunk = min(
                    self._decode_advance_s(), self.walk_window_s * 0.5,
                )
                # Wrap target through the playable buffer length so the
                # song-end → song-start loop transitions cleanly back to
                # chunk 0 instead of jumping past the last chunk.
                buf_dur_s = max(
                    1e-6, len(self.audio_eng.current) / SAMPLE_RATE,
                )
                target_song_s = (
                    (playhead_now_s + advance_s_for_chunk) % buf_dur_s
                )
                chunk_idx = int(target_song_s // self.walk_window_s)
                walk_chunk_start_s = chunk_idx * self.walk_window_s
                walk_w0 = int(round(walk_chunk_start_s * 25.0))
                # Anchor the final chunk to the song end when the song
                # length isn't an exact multiple of walk_window_s — keeps
                # T == walk_window_T without padding tricks.
                walk_w0 = max(0, min(walk_w0, full_src_T - self.walk_window_T))
                walk_w1 = walk_w0 + self.walk_window_T
                walk_chunk_start_s = walk_w0 / 25.0

            # Reset cached per-tick state on either:
            #   - a source identity / length change (swap_source path), or
            #   - a walk-mode chunk transition (new slice = new init noise
            #     story for the ring buffer; the previous chunk's cached
            #     last_latent is in the wrong song-time region).
            cur_src_id = id(self.stream.source.latent.tensor)
            cur_src_T = self.walk_window_T if walk_active else full_src_T
            chunk_changed = walk_active and walk_w0 != prev_walk_w0
            if (
                cur_src_id != prev_src_id
                or cur_src_T != prev_src_T
                or chunk_changed
            ):
                last_latent = None
                latent_history.clear()
                last_wav = None
                prev_src_id = cur_src_id
                prev_src_T = cur_src_T
                cached_live_src_lat = None
                cached_live_ctx_raw_t = None
                # Invalidate the DiT-pause cache: in walk mode the latent
                # is chunk-specific, and on a source swap it's source-
                # specific. Without this, a chunk crossing (or swap) while
                # paused would have the VAE decode the previous chunk's
                # latent at the new chunk's playhead — audible glitch.
                # Cleared cache forces the DiT-pause branch below to fall
                # through to a normal tick on the next iteration.
                self._last_result_latent = None
                if walk_active:
                    prev_walk_w0 = walk_w0

            if self.use_midi:
                raw = self.midi_knobs.get_all_values()
            else:
                with self.state._lock:
                    m = self.state.motion_val
                raw = {
                    self.k1_name: m,
                    "seed": 0.0,
                    "feedback": 0.0,
                    "feedback_depth": 1.0,
                    "shift": 3.5,
                }
                if self.use_sde:
                    raw["periodicity"] = 0.0

            # Materialize the live source / context for this tick. In
            # walk mode this is the static chunk slice and is built once
            # per chunk transition (cached_live_* are reset above). In
            # non-walk mode the StreamHandle's source latent is used as-
            # is.
            if walk_active:
                if cached_live_src_lat is None:
                    full_src_t = self.stream.source.latent.tensor
                    full_ctx_t = self.stream.source.context_latent.tensor
                    cached_live_src_lat = Latent(
                        tensor=full_src_t[:, walk_w0:walk_w1, :].contiguous(),
                    )
                    cached_live_ctx_raw_t = (
                        full_ctx_t[:, walk_w0:walk_w1, :].contiguous()
                    )
                live_src_lat = cached_live_src_lat
                live_ctx_raw_t = cached_live_ctx_raw_t
                win_start_s = walk_chunk_start_s
            else:
                live_src_lat = self.stream.source.latent
                live_ctx_raw_t = None
                win_start_s = 0.0

            # Active source latent length seen by the DiT this tick. Curves
            # built below must match this T or broadcasting fails in
            # _init_slot / _step_sde. Walk mode pins this to the window.
            src_T = self.walk_window_T if walk_active else full_src_T

            k1 = raw[self.k1_name]
            # Seed flows in as a plain integer (UI store + MCP both ship
            # uint32). Legacy clients used to send a 0..1 float which the
            # pipeline then scaled by 1000 — that hidden multiplier
            # silently capped entropy at ~1000 values and was the source
            # of the "seed cell doesn't look like a seed" complaint. If a
            # stray sub-1 value still shows up (older client connected
            # against newer server), forward it as-is; the engine treats
            # 0 as a valid seed.
            seed = (
                int(raw["seed"]) if self.use_midi else self.SEED
            )
            feedback = raw["feedback"]
            shift_val = float(raw["shift"])
            if abs(shift_val - current_shift) > 0.05:
                current_shift = shift_val

            if self.use_lora and self.engine_obj is not None:
                # Iterate the catalog so the active set can change at
                # runtime (enable/disable from the client).  Strength
                # only flows to the engine for ENABLED LoRAs; sliders
                # for non-enabled rows are ignored, matching the UI
                # contract that strength sliders are only interactive
                # while the LoRA is on.
                for desc in self.engine_obj.list_loras():
                    if desc.state != "enabled":
                        continue
                    key = f"lora_str_{desc.id}"
                    lora_str = raw.get(key, desc.strength)
                    if abs(lora_str - self.state.params.get(key, -1)) > 0.02:
                        self.engine_obj.set_lora_strength(desc.id, lora_str)

            hint_str = self.midi_knobs.get_param("hint_strength") if self.use_midi else 1.0
            # Silence latent must match the T of the latent it's blended
            # against. walk_active can flip mid-session if a swap drops
            # the source below the window — rebuild on demand here so
            # the blend below sees consistent shapes either way. Cheap
            # (allocates one bf16 tensor) and only fires on the actual
            # transitions, not every tick.
            needed_silence_T = src_T
            if self._silence_latent.tensor.shape[1] != needed_silence_T:
                self._rebuild_silence_latent()
            if walk_active:
                # Walk mode does the silence/context blend per-tick on
                # the sliced context, since the slice changes every tick
                # and the cached stream.context_latent (full-song) is
                # the wrong T to feed the DiT. The result is passed via
                # tick kwargs below; stream.context_latent stays
                # untouched.
                if hint_str >= 1.0:
                    live_ctx_lat = Latent(tensor=live_ctx_raw_t)
                else:
                    live_ctx_lat = LatentBlend().execute(
                        latent_a=self._silence_latent,
                        latent_b=Latent(tensor=live_ctx_raw_t),
                        alpha=hint_str,
                        method=self.state.interp_structure,
                    )["latent"]
                last_hint_str = hint_str
                self._hint_dirty = False
            else:
                live_ctx_lat = None
                if self._hint_dirty or abs(hint_str - last_hint_str) > 0.02:
                    self._hint_dirty = False
                    last_hint_str = hint_str
                    self._update_hint_strength(hint_str)

            # Resolve the feedback tap. depth=1 is the legacy behavior
            # (most recent latent); depth>1 walks back through history.
            # If history is shorter than the requested depth (early
            # ticks, post-swap reset), fall back to the oldest available
            # tap rather than disabling feedback — the operator's intent
            # is "use feedback," and the oldest tap is still musically
            # in-bounds while history fills.
            try:
                fb_depth_raw = float(raw.get("feedback_depth", 1.0))
            except (TypeError, ValueError):
                fb_depth_raw = 1.0
            fb_depth = max(1, min(MAX_FEEDBACK_DEPTH, int(round(fb_depth_raw))))

            fb_latent = None
            if feedback > 0.0 and latent_history:
                tap_idx = min(fb_depth - 1, len(latent_history) - 1)
                fb_latent = latent_history[tap_idx]

            source_lat = None
            if fb_latent is not None:
                src_tensor = live_src_lat.tensor
                source_lat = INTERPOLATIONS[self.state.interp_feedback](
                    src_tensor, fb_latent, feedback,
                )

            sde_curve = None
            if self.use_sde:
                denoise = 1.0
                amplitude = k1
                client_sde = _curve_from_spec(raw.get("sde_denoise_curve"), src_T)
                if client_sde is not None:
                    sde_curve = client_sde
                else:
                    periodicity = raw.get("periodicity", 0.0)
                    if periodicity > 0.01:
                        cycles = periodicity * (src_T / 25.0)
                        t = torch.linspace(0, 1, src_T).unsqueeze(0).unsqueeze(-1)
                        sde_curve = amplitude * (0.5 + 0.5 * torch.sin(2 * 3.14159 * cycles * t))
                    else:
                        sde_curve = torch.full((1, src_T, 1), amplitude, dtype=torch.float32)
                self.state.sde_curve_display = sde_curve.squeeze().numpy()
            else:
                denoise = k1
                self.state.sde_curve_display = None

            # Source lock: x0_target_curve from client overrides the
            # scalar x0_target_strength knob. The latent is attached
            # unconditionally so that a strength bump via the shared
            # override can engage the blend on in-flight slots that
            # were submitted while strength was 0.
            x0_target_curve = _curve_from_spec(raw.get("x0_target_curve"), src_T)
            if x0_target_curve is not None:
                x0_str = 0.0
            else:
                x0_str = self.midi_knobs.get_param("x0_target") if self.use_midi else 0.0
            # Use the live (possibly sliced) source as the x0_target so
            # the per-frame curve / strength scalar lines up with the
            # latent the DiT actually denoises against.
            x0_tgt = live_src_lat

            velocity_curve = _curve_from_spec(raw.get("velocity_scale_curve"), src_T)
            initial_noise_curve = _curve_from_spec(raw.get("initial_noise_curve"), src_T)

            if self.use_midi:
                last_channel_gains = self._sync_channel_guidance(raw, last_channel_gains)

            # Route every curve-capable parameter through the shared
            # mutable curve system so knob changes take effect on ALL
            # in-flight slots on the next step, bypassing the ring
            # buffer drain (~depth ticks of latency on the per-slot
            # path). ``set_shared_curve(name, None)`` clears the
            # override; ``set_shared_curve(name, scalar)`` lifts to
            # ``[1, 1, 1]``; tensors flow through unchanged.
            #
            # ``self.stream.pipeline`` is None until the first tick
            # constructs it; on that warmup iteration the submitted
            # slot uses default per-slot fields, then the shared
            # overrides take over from tick 2 onward.
            pipe = self.stream.pipeline
            if pipe is not None:
                pipe.set_shared_curve("sde_denoise_curve", sde_curve)
                pipe.set_shared_curve("velocity_scale", velocity_curve)
                pipe.set_shared_curve("x0_target_strength", x0_str)

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            tick_kwargs = {}
            tick_kwargs["steps"] = int(raw.get("steps_override", 8))
            if walk_active:
                # In walk mode the StreamHandle's cached source/context
                # are the FULL song; the DiT must see the sliced versions
                # we computed above. Pass them as per-tick overrides so
                # StreamHandle.tick() merges them into the slot request.
                tick_kwargs["context_latent"] = live_ctx_lat

            # RCFG (Residual Classifier-Free Guidance). Engaged whenever
            # the operator picks a mode other than "off" from the EngineTile
            # dropdown. The guidance_scale slider feeds a uniform [1, T, 1]
            # curve; the engine lifts it through normalize_curve. "self"
            # mode skips the negative forward (virtual v_uncond), so we
            # only attach negative conditioning for "full" / "initialize".
            rcfg_mode = str(raw.get("rcfg_mode", "off"))
            if rcfg_mode != "off":
                guidance_scale = float(raw.get("guidance_scale", 1.0))
                guidance_curve = torch.full(
                    (1, src_T, 1), guidance_scale, dtype=torch.float32,
                )
                tick_kwargs["rcfg_mode"] = rcfg_mode
                tick_kwargs["guidance_curve"] = guidance_curve

                cfg_rescale = float(raw.get("cfg_rescale", 0.0))
                if cfg_rescale > 0.0:
                    tick_kwargs["cfg_rescale"] = torch.full(
                        (1, src_T, 1), cfg_rescale, dtype=torch.float32,
                    )

                if rcfg_mode in ("full", "initialize") and self.neg_conditioning is not None:
                    tick_kwargs["negative"] = self.neg_conditioning
            # DiT-pause: skip the expensive ``stream.tick()`` call and
            # reuse the cached latent from the most recent active tick.
            # Windowed VAE decode/on_audio_ready still runs below, using
            # the cached latent and the monotonic playhead estimate. We
            # only get here when a cached latent exists (see the stage-
            # detection block above; lack of cache falls through to the
            # normal tick).
            if self._dit_paused and self._last_result_latent is not None:
                result_latent = self._last_result_latent
                # In DiT-pause mode there is no expensive diffusion tick
                # to pace the loop. Keep VAE refresh comfortably faster
                # than the 0.36 s wire slice without spinning at CPU speed.
                time.sleep(0.02)
            else:
                result_latent = self.stream.tick(
                    denoise=denoise,
                    seed=seed,
                    source_latent=(
                        Latent(tensor=source_lat) if source_lat is not None
                        else live_src_lat
                    ),
                    x0_target=x0_tgt,
                    x0_target_curve=x0_target_curve,
                    shift=current_shift,
                    initial_noise_curve=initial_noise_curve,
                    **tick_kwargs,
                    # DCW (wavelet-domain post-step correction).
                    # Forwarded every tick so toggle / mode / wavelet
                    # changes from the client take effect on the next
                    # slot via pipe.set_dcw(). Default on — matches
                    # upstream v0.1.7.
                    dcw_enabled=bool(raw.get("dcw_enabled", True)),
                    dcw_mode=str(raw.get("dcw_mode", "double")),
                    dcw_scaler=float(raw.get("dcw_scaler", 0.05)),
                    dcw_high_scaler=float(raw.get("dcw_high_scaler", 0.02)),
                    dcw_wavelet=str(raw.get("dcw_wavelet", "haar")),
                    dcw_advanced=_build_dcw_advanced(raw),
                )
            # Cache the most recent successful latent so the DiT-pause
            # branch above has something to feed the VAE windowing.
            if result_latent is not None:
                self._last_result_latent = result_latent
            torch.cuda.synchronize()
            tick_ms = (time.perf_counter() - t0) * 1000

            dec_ms = 0.0
            if result_latent is not None:
                result = result_latent.tensor
                # Decode scheduling policy:
                #   * Windowed decode (vae_window > 0) is coverage-driven:
                #     it refreshes a fresh slice every tick so live control
                #     changes always reach the wire, and is NEVER skipped on
                #     a low latent-MSE — that throttling is what made the
                #     audio go stale between param moves.
                #   * Full-buffer decode (vae_window <= 0) keeps the legacy
                #     MSE skip: re-decoding the whole song each tick when the
                #     latent barely changed is pure waste.
                skip_full_decode = False
                if (
                    self.vae_window <= 0
                    and last_latent is not None
                    and last_wav is not None
                    and (result - last_latent).pow(2).mean().item() < self.skip_threshold
                ):
                    skip_full_decode = True

                last_latent = result.clone()
                # appendleft so latent_history[0] is the most recent;
                # tap_idx = depth-1 reads "N ticks back."
                latent_history.appendleft(last_latent)

                if not skip_full_decode:
                    t1 = time.perf_counter()
                    # eff_dur clamps the windowed-decode playhead so the
                    # window stays inside the latent. In walk mode the
                    # playable buffer length is the song length, not the
                    # 60s slice — the slice is just the DiT's view onto
                    # it. Read from the audio buffer to track crop and
                    # source swaps in both modes.
                    if walk_active:
                        eff_dur = len(self.audio_eng.current) / SAMPLE_RATE
                    else:
                        eff_dur = (
                            self.crop_seconds if self.crop_seconds > 0
                            else self.stream.source.latent.tensor.shape[1] / 25.0
                        )
                    if self.vae_window > 0:
                        playhead_now = self._playhead_seconds_now()
                        # Predictive decode start: target where the playhead
                        # WILL be by the time this window lands in the buffer,
                        # with fixed-profile decoders held ahead by their
                        # decode span rather than the tiny kept wire slice.
                        # Wrap modulo ``eff_dur`` since the decoder supports
                        # cyclic.
                        advance_s = self._decode_advance_s()
                        decode_start = playhead_now + advance_s
                        if eff_dur > 0:
                            decode_start = decode_start % eff_dur

                        # Loop-band awareness. When the client arms a band
                        # [A, B] the worklet replays only that region (hard-
                        # wrapping B→A) while we keep generating. Chasing the
                        # raw playhead would decode *past* B (audio that's
                        # never heard) and leave the region just after A
                        # holding pre-change audio until the playhead crawls
                        # back over it — the "snap back to the old buffer for
                        # one window" the operator hears at every loop
                        # restart. Wrapping the decode target inside the band
                        # instead pre-fills the seam after A while the
                        # playhead is still finishing the lap near B, so the
                        # restart plays freshly-generated audio. Only the
                        # standard windowed path is band-aware; walk mode
                        # (multi-minute sources) keeps its linear chase.
                        band_start_sample = None
                        band_end_sample = None
                        band_wrap_start_s = None
                        band = getattr(self.audio_eng, "loop_band", None)
                        if band is not None and eff_dur > 0 and not walk_active:
                            a_s = max(0.0, min(float(band[0]), eff_dur))
                            b_s = max(0.0, min(float(band[1]), eff_dur))
                            span = b_s - a_s
                            # Only pin the decode into the band while the
                            # playhead is actually INSIDE it. The operator can
                            # scrub the playhead out of an armed loop; when
                            # they do, pinning would keep regenerating audio
                            # inside the loop while the listener is somewhere
                            # else — exactly the "waveform changes in the loop
                            # instead of in front of the playhead" bug. Outside
                            # the band we fall through to the plain playhead
                            # chase (``decode_start`` set above).
                            playhead_in_band = a_s <= playhead_now <= b_s
                            if span > 1e-3 and playhead_in_band:
                                if span < self.vae_window:
                                    # Band shorter than one decode window:
                                    # pin the target at A so every decode
                                    # rewrites the whole band — no chase lag
                                    # on a region this small. The write clamp
                                    # below drops the spillover past B.
                                    decode_start = a_s
                                else:
                                    # Sawtooth that mirrors the worklet's
                                    # A→B→A playhead, led by ``advance_s``.
                                    decode_start = a_s + (
                                        (playhead_now + advance_s - a_s) % span
                                    )
                                band_start_sample = int(round(a_s * SAMPLE_RATE))
                                band_end_sample = int(round(b_s * SAMPLE_RATE))
                                band_wrap_start_s = a_s

                        if walk_active:
                            # The DiT output spans [win_start_s,
                            # win_start_s + walk_window_s] of the song.
                            # Decode at the offset *inside* the window
                            # corresponding to the song-time we want, then
                            # remap the decoder's start_sample (which is
                            # window-relative) to absolute song samples by
                            # adding the window's start sample. cyclic=
                            # False because the slice itself doesn't wrap.
                            local_t_start = decode_start - win_start_s
                            # Clamp inside the window. The window is
                            # centered around target_song_s (which equals
                            # decode_start under steady state), so the
                            # nominal local offset is walk_window_s/2,
                            # but a stale window from earlier in the loop
                            # can drift; clamp to keep VAE inside bounds.
                            local_t_start = max(
                                0.0,
                                min(local_t_start, self.walk_window_s - self.vae_window),
                            )
                            audio_out = self.session.decode(
                                result_latent, t_start=local_t_start, cyclic=False,
                            )
                            win_offset_samples = int(round(win_start_s * SAMPLE_RATE))
                        else:
                            audio_out = self.session.decode(result_latent, t_start=decode_start, cyclic=True)
                            win_offset_samples = 0
                        torch.cuda.synchronize()
                        dec_ms = (time.perf_counter() - t1) * 1000
                        win_wav = audio_out.waveform.detach().cpu().float().squeeze(0)
                        win_np = win_wav.numpy().T
                        win_start = audio_out.start_sample + win_offset_samples
                        win_end = win_start + win_np.shape[0]
                        # Read the boundary slices directly from the
                        # audio engine's live buffer. The previous
                        # ``buf = self.audio_eng.current.copy()`` cost
                        # ~23 MB of host RAM per windowed decode for a
                        # 60 s buffer; only ~4800 samples at each edge
                        # are actually needed for the crossfade, and
                        # ``self.current`` is single-writer (this
                        # thread) so a bare slice read is safe without
                        # the lock.
                        current = self.audio_eng.current
                        # 25 ms at 48 kHz — matches CROSSFADE_SECONDS.
                        # Cuts perceived "smear" of param transitions in
                        # half from the previous 50 ms.
                        xfade = min(1200, win_np.shape[0] // 4)
                        if win_start > 0 and xfade > 0:
                            t_in = np.linspace(0.0, 1.0, xfade).reshape(-1, 1)
                            win_np[:xfade] = (
                                current[win_start:win_start + xfade] * (1 - t_in)
                                + win_np[:xfade] * t_in
                            )
                        if win_end < current.shape[0] and xfade > 0:
                            t_out = np.linspace(1.0, 0.0, xfade).reshape(-1, 1)
                            tail = min(xfade, current.shape[0] - win_end + xfade)
                            s = win_np.shape[0] - tail
                            win_np[s:] = (
                                win_np[s:] * t_out[:tail]
                                + current[win_start + s:win_start + s + tail] * (1 - t_out[:tail])
                            )
                        clamp_end = min(win_end, current.shape[0])
                        # Loop band active: never write the first decode past
                        # B. If the window crossed B, a second decode anchored
                        # at A below patches the wrapped portion the worklet
                        # will play after the loop restart.
                        if band_end_sample is not None and band_end_sample > win_start:
                            clamp_end = min(clamp_end, band_end_sample)
                        patched = win_np[:clamp_end - win_start]
                        # Single in-place write under the audio
                        # engine's lock. Replaces the old "copy → write
                        # → swap" sequence (two full-buffer numpy
                        # memcpys) with one slice-assign.
                        self.audio_eng.patch_window(patched, win_start)
                        # Callback receives the patched window only.
                        # Backend handler uses it to delta-encode against
                        # its client mirror; standalone callers can
                        # ignore the args.
                        self.on_audio_ready(patched, win_start, win_end)
                        if (
                            band_start_sample is not None
                            and band_end_sample is not None
                            and band_wrap_start_s is not None
                            and band_start_sample < band_end_sample
                            and win_start < band_end_sample
                            and win_end > band_end_sample
                            and not (
                                win_start <= band_start_sample
                                and clamp_end >= band_end_sample
                            )
                        ):
                            wrap_len = min(
                                win_end - band_end_sample,
                                band_end_sample - band_start_sample,
                            )
                            if wrap_len > 0:
                                wrap_audio = self.session.decode(
                                    result_latent,
                                    t_start=band_wrap_start_s,
                                    cyclic=True,
                                )
                                wrap_wav = (
                                    wrap_audio.waveform.detach()
                                    .cpu()
                                    .float()
                                    .squeeze(0)
                                )
                                wrap_np = wrap_wav.numpy().T[:wrap_len].copy()
                                wrap_start = band_start_sample
                                wrap_end = wrap_start + wrap_np.shape[0]
                                xfade_wrap = min(1200, wrap_np.shape[0] // 4)
                                if wrap_start > 0 and xfade_wrap > 0:
                                    t_in = np.linspace(
                                        0.0, 1.0, xfade_wrap,
                                    ).reshape(-1, 1)
                                    wrap_np[:xfade_wrap] = (
                                        current[wrap_start:wrap_start + xfade_wrap]
                                        * (1 - t_in)
                                        + wrap_np[:xfade_wrap] * t_in
                                    )
                                if wrap_end < current.shape[0] and xfade_wrap > 0:
                                    t_out = np.linspace(
                                        1.0, 0.0, xfade_wrap,
                                    ).reshape(-1, 1)
                                    wrap_np[-xfade_wrap:] = (
                                        wrap_np[-xfade_wrap:] * t_out
                                        + current[wrap_end - xfade_wrap:wrap_end]
                                        * (1 - t_out)
                                )
                                self.audio_eng.patch_window(wrap_np, wrap_start)
                                self.on_audio_ready(wrap_np, wrap_start, wrap_end)
                                torch.cuda.synchronize()
                                dec_ms = (time.perf_counter() - t1) * 1000
                        # ``last_wav`` is only checked for non-None by
                        # the legacy full-buffer skip path. A view of
                        # the engine buffer is enough; we don't need to
                        # retain a snapshot.
                        last_wav = current
                    else:
                        audio_out = self.session.decode(result_latent)
                        torch.cuda.synchronize()
                        dec_ms = (time.perf_counter() - t1) * 1000
                        wav = audio_out.waveform.detach().cpu().float().squeeze(0)
                        wav_np = wav.numpy().T
                        if self.crop_seconds > 0:
                            wav_np = wav_np[:int(self.crop_seconds * SAMPLE_RATE)]
                        last_wav = wav_np
                        self.on_audio_ready(wav_np)

                self.state.params["num_gens"] = self.state.params.get("num_gens", 0) + 1
                self.state.params["tick_ms"] = tick_ms
                self.state.params["dec_ms"] = dec_ms

                # Sampled TRACE so DEMON_LOG_LEVEL=TRACE gives the operator
                # a tick-by-tick snapshot for perf chases without paying the
                # cost on every iteration. _TRACE_SAMPLE_EVERY=0 disables it
                # outright; loguru's level gate elides the call cheaply when
                # no sink is at TRACE.
                if _TRACE_SAMPLE_EVERY and (
                    self.state.params["num_gens"] % _TRACE_SAMPLE_EVERY == 0
                ):
                    logger.trace(
                        "tick num_gens={} tick_ms={:.1f} dec_ms={:.1f} "
                        "shift={:.2f} seed={} hint_str={:.2f}",
                        self.state.params["num_gens"], tick_ms, dec_ms,
                        shift_val, seed, hint_str,
                    )
                # Update predictive-decode EMA from this iteration's actual
                # wall time. alpha=0.3 reacts in a handful of ticks while
                # smoothing out one-off spikes (e.g. a CUDA sync stall).
                # Skipped ticks (no decode) leave dec_ms=0 and would
                # otherwise pull the EMA toward zero, so only update when
                # we actually decoded.
                if dec_ms > 0:
                    new_advance = (tick_ms + dec_ms) / 1000.0
                    self._predicted_advance_s = (
                        0.3 * new_advance + 0.7 * self._predicted_advance_s
                    )
                self.state.params[self.k1_name] = round(k1, 2)
                self.state.params["seed"] = seed
                self.state.params["feedback"] = round(feedback, 2)
                self.state.params["feedback_depth"] = fb_depth
                self.state.params["shift"] = round(shift_val, 2)
                if self.use_lora and self.engine_obj is not None:
                    for desc in self.engine_obj.list_loras():
                        if desc.state != "enabled":
                            continue
                        key = f"lora_str_{desc.id}"
                        self.state.params[key] = round(raw.get(key, desc.strength), 2)
                if self.use_sde:
                    self.state.params["periodicity"] = round(raw.get("periodicity", 0.0), 2)
                self.state.params["hint_strength"] = round(hint_str, 2)
                for name, _, _ in CHANNEL_GROUPS:
                    self.state.params[name] = round(raw.get(name, 1.0), 2)
                for name, _ in KEYSTONE_CHANNELS:
                    self.state.params[name] = round(raw.get(name, 1.0), 2)
                self.state.params["_prompt"] = self.state.prompt_text
