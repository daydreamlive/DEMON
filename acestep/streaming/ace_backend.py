"""ACEStepBackend: the ACE-Step diffusion stack behind the
:class:`~acestep.streaming.generator_backend.GeneratorBackend` seam.

Extracted verbatim from the pre-seam ``PipelineRunner.run()`` loop body
(Phase 1 of ``round_3_BACKEND_PLAN_FINAL.md``): everything that knows
about latents, the StreamHandle, the StreamPipeline, curves, walk
windows, LoRA refits, or the VAE lives here; the runner keeps pacing,
the adaptive lead, idle staging, loop-band targeting, crossfading, and
emission. Zero intended behavior change — the golden harness is the
acceptance gate.

Specializes :class:`~acestep.streaming.diffusion_backend.
DiffusionBackend` (the shared Tier-1 skeleton): the base owns the
produce-mode dispatch / renderable-state caching / timing; this class
owns the ACE knob translation, walk-window source choreography, LoRA
refit signaling, and windowed VAE rendering. Its Tier-2 adapter is the
default ``ACEAdapter`` constructed inside the StreamHandle's
``StreamPipeline``; its codec is the engine ``Session`` (windowed VAE
decode), passed to the base as ``codec=``.
"""

import os
import time
from collections import deque

import torch

from acestep.engine.dcw import DCWAdvanced
from acestep.engine.obs import logger
from acestep.nodes.types import ChannelGuidanceEntry, Latent
from acestep.nodes.interpolation import INTERPOLATIONS
from acestep.nodes.vae_nodes import EmptyLatent, LatentBlend

from acestep.streaming.diffusion_backend import DiffusionBackend
from acestep.streaming.generator_backend import (
    AudioChunk,
    AudioGeometry,
    Capabilities,
    TickContext,
)
from acestep.steering import SteeringController
from acestep.streaming.knobs import (
    CHANNEL_GROUPS,
    KEYSTONE_CHANNELS,
    knob_specs as registry_knob_specs,
    manual_slot_specs,
    steering_axis_spec,
)

# Audio sample rate the ACE-Step v1.5 family is trained on, and the
# latent frame count for a 60 s window at the tokenizer's 25 fps. Model
# invariants (see also pipeline_runner.py, which keeps its own copies
# for pacing math so it stays import-free of backend modules).
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


def fixed_windowed_vae_span_s() -> float:
    """Return the fixed VAE decode span in seconds, or 0 for dynamic engines."""
    try:
        from acestep.paths import WINDOWED_VAE_PROFILE_FRAMES
    except Exception:
        return 0.0
    pmin, popt, pmax = WINDOWED_VAE_PROFILE_FRAMES
    if pmin == popt == pmax and pmin > 0:
        return pmin / 25.0
    return 0.0


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


def steering_knob_specs(steering: "SteeringController") -> list:
    """Project a SteeringController's live surface into registry specs.

    Empty when no vector bundle is reachable (the knobs would be dead).
    The spec SHAPES come from the registry factories in
    ``acestep.streaming.knobs``; only the axis/catalog metadata is
    filled in here, where the steering policy lives.
    """
    if not steering.is_loaded:
        return []
    specs: list = []
    for ax in steering.auto_axes:
        inject_layer = max(
            0, min(steering.MANUAL_MAX_LAYER, ax.probe_layer + ax.layer_offset),
        )
        specs.append(steering_axis_spec(
            ax.name,
            axis=ax.axis,
            inject_layer=inject_layer,
            probe_step=ax.probe_step,
            probe_n=steering._probe_n,
            blurb=ax.blurb,
        ))
    src_max = max(0, len(steering.catalog) - 1)
    for slot in steering.active_slots():
        specs.extend(manual_slot_specs(
            slot,
            src_max=src_max,
            catalog_len=len(steering.catalog),
            layer_max=steering.MANUAL_MAX_LAYER,
            step_max=steering.MANUAL_MAX_STEP,
        ))
    return specs


class ACEStepBackend(DiffusionBackend):
    """ACE-Step v1.5 diffusion generation behind the GeneratorBackend seam.

    Construction mirrors the pre-seam PipelineRunner argument list; the
    session passes the same objects it always did, they just land here
    instead of on the runner.
    """

    name = "acestep"

    def __init__(
        self, session, stream, *,
        state,
        use_midi, use_sde, use_lora,
        midi_knobs, engine_obj,
        vae_window, crop_seconds,
        k1_name, seed, skip_threshold,
        walk_window=False,
        walk_window_s=60.0,
        neg_conditioning=None,
        steering: SteeringController | None = None,
    ):
        # The family codec is the engine Session: its windowed VAE
        # decode is what render_window()/render_full() drive. The
        # Tier-2 adapter is pipeline-owned (default ACEAdapter inside
        # the StreamHandle's StreamPipeline), so it stays None here.
        super().__init__(codec=session)
        self.session = session
        self.stream = stream  # StreamHandle
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

        # Walk-window mode: drive the DiT with a fixed-T window sliced
        # from a longer pre-encoded source so the 60s TRT engine can
        # serve a multi-minute song. The source is split into
        # walk_window_s chunks (typically 60s == one engine slot worth
        # of latent); the backend picks the chunk that contains the
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

        # VAE receptive-field span; the runner reads this for its
        # startup log line, the walk chunk-prewarm derives from it here.
        self.decode_span_s = fixed_windowed_vae_span_s()
        # Walk-mode chunk pre-warm lookahead. Independent of the playback
        # lead: it decides how early to swap to the next static source chunk
        # so its ring-buffer warmup lands before the playhead crosses the
        # boundary. Sized from the decode span, NOT the (now much smaller)
        # playback lead, so chunk swaps don't glitch.
        self._walk_chunk_prewarm_s = max(self.vae_window, self.decode_span_s) * 0.5

        # Rebuild-signature change detection (steps / LoRA enablement).
        # ``None`` on the first tick just seeds the baseline.
        self._last_rebuild_keys = None

        # Activation steering. The controller is the source of truth for
        # the slot count and vector catalog; the session mirrors its
        # slot ops into KnobState / the knob manifest. ``None`` (e.g. a
        # bare-construction test fixture) degrades to an unloaded
        # controller so every consumer can read it unconditionally.
        self.steering = (
            steering if steering is not None else SteeringController(None)
        )
        # (pipeline, snapshot) change-detection key for _sync_steering;
        # None forces a push on the first tick and after a
        # steps_override-driven pipeline rebuild.
        self._last_steering = None

        # ----- per-tick translation state (the old run() locals) -----
        self._last_latent = None
        # Previous fresh latent for the full-buffer MSE skip. Tracked
        # separately from ``_last_latent`` because produce() updates the
        # history BEFORE render_full() runs its compare (the historical
        # loop compared first, then updated, in one block).
        self._mse_prev = None
        # Ring of past finished latents for the feedback delay-tap.
        # latent_history[0] is the most recent generation (== _last_latent
        # when set), latent_history[1] is one tick older, etc. Capped at
        # MAX_FEEDBACK_DEPTH because longer history would just sit unused
        # — the knob can't reach past that anyway.
        self._latent_history: deque = deque(maxlen=MAX_FEEDBACK_DEPTH)
        self._last_wav = None
        self._last_hint_str = 1.0
        self._last_channel_gains = [1.0] * (len(CHANNEL_GROUPS) + len(KEYSTONE_CHANNELS))
        self._current_shift = self.stream.base_kwargs["shift"]
        self._prev_src_T = self.stream.source.latent.tensor.shape[1]
        # Source-tensor identity tracking. Lets walk mode detect a source
        # swap when the new song happens to have the same latent length as
        # the old one (T-only check would miss it).
        self._prev_src_id = id(self.stream.source.latent.tensor)
        # Walk-mode chunk anchor (in latent frames). -1 forces the first
        # walk-active tick to "transition" into chunk 0 and reset caches.
        self._prev_walk_w0 = -1
        # Cached slice tensors for the active chunk. Invalidated whenever
        # ``_prev_walk_w0`` is reset (source swap or chunk transition).
        self._cached_live_src_lat = None
        self._cached_live_ctx_raw_t = None

        # Per-tick state computed by sync_source() and consumed by
        # produce()/render_window() within the same iteration.
        self._full_src_T = self._prev_src_T
        self._walk_active = False
        self._walk_w0 = -1
        self._walk_w1 = -1
        self._walk_chunk_start_s = 0.0

        # Stashed per-produce values for the params echo + trace.
        # (last_tick_ms / last_dec_ms / the result caches live on the
        # DiffusionBackend base.)
        self._echo = {}
        self.last_denoise = 0.0

        # Cache silence once; used by the hint-strength blend node.
        self._rebuild_silence_latent()

        # Hint-strength gating: produce() only re-runs the
        # silence/context blend when the slider value moves by > 0.02.
        # Outside callers that change ``stream.source.context_latent``
        # under the backend's feet (e.g. the structure-override upload
        # path on the recv thread) need a way to force the next tick
        # to re-blend even when the slider hasn't moved. ``mark_hint_dirty``
        # flips this flag and produce() honors it on the next pass.
        self._hint_dirty = False

    # ---- contract ----------------------------------------------------------

    def capabilities(self) -> Capabilities:
        return Capabilities(
            refines_audio=True,
            swap=True,
            timbre=True,
            structure=True,
            lora=self.use_lora,
            stems=True,
            loop_band=True,
            depth=True,
            curves=True,
            notes_conditioning=False,
            steering=self.steering.is_loaded,
        )

    def geometry(self) -> AudioGeometry:
        dur = self.playable_duration_s()
        return AudioGeometry(
            sample_rate=SAMPLE_RATE,
            channels=2,
            chunk_rate_hz=25.0,
            duration_s=dur if dur is not None else (
                self.stream.source.latent.tensor.shape[1] / 25.0
            ),
        )

    def knob_specs(self, lora_ids=()) -> list:
        """The ACE-family manifest: the shared registry's spec list,
        parameterized by this session's SDE mode and the enabled-LoRA
        set the session passes in (see the protocol docstring), plus
        the activation-steering surface (auto axes + the live manual
        slots) when this session's checkpoint has a vector bundle."""
        return registry_knob_specs(
            self.use_sde,
            loras=list(lora_ids) if self.use_lora else [],
        ) + steering_knob_specs(self.steering)

    # ---- public hooks reachable from session ops ---------------------------

    def mark_hint_dirty(self) -> None:
        """Force the hint-strength re-blend on the next tick.

        Use after replacing ``stream.source.context_latent`` (e.g. on
        structure-override apply / clear or after a source swap) so the
        next produce() re-blends silence ↔ context at the current
        ``hint_strength`` and writes a fresh ``stream.context_latent``
        for the diffusion step to read. Without this, the diffusion
        keeps reading the previously-blended tensor until the operator
        nudges the slider.
        """
        self._hint_dirty = True

    # ---- internals ----------------------------------------------------------

    def _rebuild_silence_latent(self) -> None:
        """(Re)build the silence latent used by hint-strength blending.

        Picks the right T for the *current* hint-blend target: in walk
        mode that's the per-tick window slice (``walk_window_T``); in
        non-walk mode it's the full source latent. ``walk_window=True``
        with a source shorter than the window degrades to non-walk
        per-tick and the per-tick guard in produce() will rebuild this
        if the size disagrees.
        """
        full_src_T = self.stream.source.latent.tensor.shape[1]
        walk_active = self.walk_window and full_src_T > self.walk_window_T
        T_frames = self.walk_window_T if walk_active else full_src_T
        self._silence_latent = EmptyLatent().execute(
            model=self.stream.model, frames=T_frames,
        )["latent"]

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

    def _sync_steering(self, raw: dict, last):
        """Push activation-steering configs when the snapshot changes.

        ``last`` is ``(pipeline, snapshot_tuple)`` or ``None``. Pipeline
        identity is part of the key because ``steps_override`` rebuilds
        the StreamPipeline (fresh, empty steering state) without
        changing ``raw`` — without the identity check the new pipeline
        would never receive ``set_steering``.
        """
        if not self.steering.is_loaded:
            return last
        pipe = self.stream.pipeline
        if pipe is None:
            return last
        n = max(1, int(raw.get("steps_override", 8)))
        snapshot = self.steering.snapshot_key(raw, n)
        last_pipe, last_snapshot = last if last is not None else (None, None)
        if pipe is last_pipe and snapshot == last_snapshot:
            return last
        pipe.set_steering(self.steering.build_configs(raw, n))
        return (pipe, snapshot)

    # ---- GeneratorBackend hot loop -----------------------------------------

    def sync_source(self, ctx: TickContext) -> None:
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
        # Use ``playhead + prewarm`` so the swap happens a tick or two
        # before the boundary, giving the new chunk's ring-buffer
        # warmup head room to land before the listener actually
        # crosses.
        walk_w0 = -1
        walk_w1 = -1
        walk_chunk_start_s = 0.0
        if walk_active:
            playhead_now_s = ctx.playhead_s
            advance_s_for_chunk = min(
                self._walk_chunk_prewarm_s, self.walk_window_s * 0.5,
            )
            # Wrap target through the playable buffer length so the
            # song-end → song-start loop transitions cleanly back to
            # chunk 0 instead of jumping past the last chunk.
            buf_dur_s = max(1e-6, ctx.buffer_duration_s)
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
        chunk_changed = walk_active and walk_w0 != self._prev_walk_w0
        if (
            cur_src_id != self._prev_src_id
            or cur_src_T != self._prev_src_T
            or chunk_changed
        ):
            self._last_latent = None
            self._mse_prev = None
            self._latent_history.clear()
            self._last_wav = None
            self._prev_src_id = cur_src_id
            self._prev_src_T = cur_src_T
            self._cached_live_src_lat = None
            self._cached_live_ctx_raw_t = None
            # Invalidate the DiT-pause cache: in walk mode the latent
            # is chunk-specific, and on a source swap it's source-
            # specific. Without this, a chunk crossing (or swap) while
            # paused would have the VAE decode the previous chunk's
            # latent at the new chunk's playhead — audible glitch.
            # Cleared cache forces the runner's DiT-pause branch to
            # fall through to a normal tick on the next iteration.
            self._last_result_latent = None
            if walk_active:
                self._prev_walk_w0 = walk_w0

        self._full_src_T = full_src_T
        self._walk_active = walk_active
        self._walk_w0 = walk_w0
        self._walk_w1 = walk_w1
        self._walk_chunk_start_s = walk_chunk_start_s

    def read_knobs(self) -> dict:
        if self.use_midi:
            return self.midi_knobs.get_all_values()
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
        return raw

    def has_pending_refit(self) -> bool:
        """True when ``before_tick`` is about to apply LoRA commands.

        LoRA enable/disable is drained by the streaming session's
        ``before_tick`` hook and can synchronously materialize/refit weights.
        That stall happens before the rebuild-signature check, so the runner
        must look at the pending queue first if it wants to pre-cover the
        buffer with a gap-fill write.
        """
        if not self.use_lora:
            return False
        try:
            with self.state._lock:
                return bool(self.state.pending_enable or self.state.pending_disable)
        except AttributeError:
            return False

    def _rebuild_signature(self, raw: dict) -> tuple:
        """Params whose change forces a pipeline rebuild / multi-hundred-ms
        warmup stall. When this tuple changes between ticks the runner
        pre-covers the stall before it lands (the adaptive lead can't see it
        reactively, since the slow produce() and the new value arrive on the
        same iteration). Keep in sync with what actually triggers a
        rebuild/stall: step count and LoRA *enablement*.

        LoRA enablement is read from the engine's own state, NOT from
        ``raw``: the only ``lora_``-prefixed keys in ``raw`` are the live
        ``lora_str_{id}`` strength knobs, so a prefix scan would flip this
        signature whenever a strength is ridden through zero — firing a
        spurious rebuild bump for an operation that triggers no rebuild.
        Enable/disable is applied to the engine by ``before_tick`` (which runs
        at the top of the loop, before this is called), so ``list_loras()``
        already reflects the change here.
        """
        enabled_loras = ()
        if self.engine_obj is not None:
            enabled_loras = tuple(
                sorted(
                    desc.id for desc in self.engine_obj.list_loras()
                    if desc.state == "enabled"
                )
            )
        return (
            int(raw.get("steps_override", 8)),
            enabled_loras,
        )

    def rebuild_imminent(self, knobs: dict) -> bool:
        rebuild_keys = self._rebuild_signature(knobs)
        rebuild_changed = (
            self._last_rebuild_keys is not None
            and rebuild_keys != self._last_rebuild_keys
        )
        self._last_rebuild_keys = rebuild_keys
        return rebuild_changed

    def playable_duration_s(self):
        if self._walk_active:
            # The playable buffer length is the song length, not the
            # 60s slice — the slice is just the DiT's view onto it.
            # None lets the runner read it from the audio buffer, which
            # tracks crop and source swaps.
            return None
        return (
            self.crop_seconds if self.crop_seconds > 0
            else self.stream.source.latent.tensor.shape[1] / 25.0
        )

    def _prepare_tick(self, knobs: dict, ctx: TickContext) -> dict:
        """The ACE knob-translation half of the historical produce():
        runs on EVERY active tick (shared-curve writes land even on
        ticks that skip the engine). Returns the prepared state
        :meth:`_generate` consumes; the mode dispatch / caching /
        timing skeleton lives on :class:`DiffusionBackend`."""
        raw = knobs
        walk_active = self._walk_active
        full_src_T = self._full_src_T

        # Materialize the live source / context for this tick. In
        # walk mode this is the static chunk slice and is built once
        # per chunk transition (cached_live_* are reset in
        # sync_source). In non-walk mode the StreamHandle's source
        # latent is used as-is.
        if walk_active:
            if self._cached_live_src_lat is None:
                full_src_t = self.stream.source.latent.tensor
                full_ctx_t = self.stream.source.context_latent.tensor
                self._cached_live_src_lat = Latent(
                    tensor=full_src_t[:, self._walk_w0:self._walk_w1, :].contiguous(),
                )
                self._cached_live_ctx_raw_t = (
                    full_ctx_t[:, self._walk_w0:self._walk_w1, :].contiguous()
                )
            live_src_lat = self._cached_live_src_lat
            live_ctx_raw_t = self._cached_live_ctx_raw_t
        else:
            live_src_lat = self.stream.source.latent
            live_ctx_raw_t = None

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
        if abs(shift_val - self._current_shift) > 0.05:
            self._current_shift = shift_val

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
            self._last_hint_str = hint_str
            self._hint_dirty = False
        else:
            live_ctx_lat = None
            if self._hint_dirty or abs(hint_str - self._last_hint_str) > 0.02:
                self._hint_dirty = False
                self._last_hint_str = hint_str
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
        if feedback > 0.0 and self._latent_history:
            tap_idx = min(fb_depth - 1, len(self._latent_history) - 1)
            fb_latent = self._latent_history[tap_idx]

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
            self._last_channel_gains = self._sync_channel_guidance(
                raw, self._last_channel_gains,
            )
            self._last_steering = self._sync_steering(
                raw, self._last_steering,
            )

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

        return {
            "raw": raw,
            "denoise": denoise,
            "seed": seed,
            "live_src_lat": live_src_lat,
            "source_lat": source_lat,
            "x0_tgt": x0_tgt,
            "x0_target_curve": x0_target_curve,
            "initial_noise_curve": initial_noise_curve,
            "tick_kwargs": tick_kwargs,
            "echo": {
                "k1": k1, "seed": seed, "feedback": feedback,
                "fb_depth": fb_depth, "shift_val": shift_val,
                "hint_str": hint_str,
            },
        }

    def _generate(self, prep: dict):
        raw = prep["raw"]
        source_lat = prep["source_lat"]
        return self.stream.tick(
            denoise=prep["denoise"],
            seed=prep["seed"],
            source_latent=(
                Latent(tensor=source_lat) if source_lat is not None
                else prep["live_src_lat"]
            ),
            x0_target=prep["x0_tgt"],
            x0_target_curve=prep["x0_target_curve"],
            shift=self._current_shift,
            initial_noise_curve=prep["initial_noise_curve"],
            **prep["tick_kwargs"],
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

    def _after_produce(self, prep: dict, result_latent, is_fresh: bool) -> None:
        self.last_denoise = prep["denoise"]
        if is_fresh:
            result = result_latent.tensor
            # Previous fresh latent, for render_full()'s MSE skip
            # (the historical loop compared before updating).
            self._mse_prev = self._last_latent
            self._last_latent = result.clone()
            # appendleft so latent_history[0] is the most recent;
            # tap_idx = depth-1 reads "N ticks back."
            self._latent_history.appendleft(self._last_latent)
        # Stash the values the params echo + sampled trace need.
        self._echo = prep["echo"]

    def render_window(self, t_start_s: float):
        decode_src = (
            self._current_result if self._current_result is not None
            else self._last_result_latent
        )
        if decode_src is None:
            return None
        t1 = time.perf_counter()
        if self._walk_active:
            # The DiT output spans [win_start_s,
            # win_start_s + walk_window_s] of the song.
            # Decode at the offset *inside* the window
            # corresponding to the song-time we want, then
            # remap the decoder's start_sample (which is
            # window-relative) to absolute song samples by
            # adding the window's start sample. cyclic=
            # False because the slice itself doesn't wrap.
            win_start_s = self._walk_chunk_start_s
            local_t_start = t_start_s - win_start_s
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
            audio_out = self.codec.decode(
                decode_src, t_start=local_t_start, cyclic=False,
            )
            win_offset_samples = int(round(win_start_s * SAMPLE_RATE))
        else:
            audio_out = self.codec.decode(decode_src, t_start=t_start_s, cyclic=True)
            win_offset_samples = 0
        torch.cuda.synchronize()
        self.last_dec_ms += (time.perf_counter() - t1) * 1000
        win_wav = audio_out.waveform.detach().cpu().float().squeeze(0)
        win_np = win_wav.numpy().T
        win_start = audio_out.start_sample + win_offset_samples
        return AudioChunk(pcm=win_np, start_sample=win_start)

    def render_full(self):
        result_latent = self._current_result
        if result_latent is None:
            return None
        # Full-buffer decode keeps the legacy MSE skip: re-decoding the
        # whole song each tick when the latent barely changed is pure
        # waste. (Windowed decode is coverage-driven and never skips —
        # that throttling is what made the audio go stale between param
        # moves.)
        result = result_latent.tensor
        if (
            self._mse_prev is not None
            and self._last_wav is not None
            and (result - self._mse_prev).pow(2).mean().item() < self.skip_threshold
        ):
            return None
        t1 = time.perf_counter()
        audio_out = self.codec.decode(result_latent)
        torch.cuda.synchronize()
        self.last_dec_ms += (time.perf_counter() - t1) * 1000
        wav = audio_out.waveform.detach().cpu().float().squeeze(0)
        wav_np = wav.numpy().T
        if self.crop_seconds > 0:
            wav_np = wav_np[:int(self.crop_seconds * SAMPLE_RATE)]
        self._last_wav = wav_np
        return AudioChunk(pcm=wav_np, start_sample=0)

    def note_windowed_write(self, buffer_view) -> None:
        """Record that a windowed write landed (legacy ``last_wav``
        bookkeeping). Only the full-buffer MSE skip ever reads
        ``_last_wav``, so in windowed sessions this is unobservable; it
        is kept for exact parity with the historical loop, which set
        ``last_wav`` to the live buffer view after every windowed
        write."""
        self._last_wav = buffer_view

    def on_fresh_generation(self, knobs: dict) -> None:
        e = self._echo
        self.state.params["num_gens"] = self.state.params.get("num_gens", 0) + 1
        self.state.params["tick_ms"] = self.last_tick_ms
        self.state.params["dec_ms"] = self.last_dec_ms

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
                self.state.params["num_gens"], self.last_tick_ms, self.last_dec_ms,
                e["shift_val"], e["seed"], e["hint_str"],
            )
        self.state.params[self.k1_name] = round(e["k1"], 2)
        self.state.params["seed"] = e["seed"]
        self.state.params["feedback"] = round(e["feedback"], 2)
        self.state.params["feedback_depth"] = e["fb_depth"]
        self.state.params["shift"] = round(e["shift_val"], 2)
        if self.use_lora and self.engine_obj is not None:
            for desc in self.engine_obj.list_loras():
                if desc.state != "enabled":
                    continue
                key = f"lora_str_{desc.id}"
                self.state.params[key] = round(knobs.get(key, desc.strength), 2)
        if self.use_sde:
            self.state.params["periodicity"] = round(knobs.get("periodicity", 0.0), 2)
        self.state.params["hint_strength"] = round(e["hint_str"], 2)
        for name, _, _ in CHANNEL_GROUPS:
            self.state.params[name] = round(knobs.get(name, 1.0), 2)
        for name, _ in KEYSTONE_CHANNELS:
            self.state.params[name] = round(knobs.get(name, 1.0), 2)
        self.state.params["_prompt"] = self.state.prompt_text
