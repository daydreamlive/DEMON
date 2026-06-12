"""PipelineRunner: the backend-agnostic streaming loop.

Drives a :class:`~acestep.streaming.generator_backend.GeneratorBackend`
each iteration and owns everything position- and pacing-shaped:

* the monotonic playhead estimate,
* the adaptive playback lead (interval EMA + stall bump),
* idle GPU pause staging,
* loop-band-aware decode targeting,
* crossfading freshly rendered windows into the live buffer,
* emission (``patch_window`` + ``on_audio_ready``).

Everything model-shaped — knob supply/translation, generation,
decode/render, stall signaling, the per-generation params echo — lives
behind the backend seam (see ``generator_backend.py``; the ACE-Step
implementation is ``ace_backend.py``). Phase 1 of
``round_3_BACKEND_PLAN_FINAL.md``: this split is a pure refactor of the
old fused loop, gated by the golden harness staying bit-exact.
"""

import math
import os
import time

import numpy as np

from acestep.engine.obs import logger
from acestep.streaming.generator_backend import TickContext

# Audio sample rate the ACE-Step v1.5 family is trained on. Duplicated
# from ``demos/realtime_motion_graph_web/protocol.py`` (and many other
# call sites — see tests/, scripts/) so this module stays free of demo
# imports. ``T`` is the latent frame count for a 60 s window at the
# tokenizer's 25 fps; both constants are model invariants the runner
# uses as plain magic numbers. (Phase 2 of the backend-seam plan put
# the client-visible truth on the wire — ``ready.geometry``, declared
# by the backend; these copies remain the runner's internal pacing
# constants and switch to backend.geometry() only when a non-48k
# family actually lands behind this loop.)
SAMPLE_RATE = 48000
T = 1500

# Knob→ear latency tracing. When DEMON_LAT_TRACE is set, every windowed
# decode logs where the fresh slice lands relative to the live playhead
# (``lead_s``) so the knob→ear floor is directly observable in the log.
_LAT_TRACE = os.environ.get("DEMON_LAT_TRACE", "") not in ("", "0")


class ReportStalenessEstimator:
    """Estimates how stale each client playhead report is at arrival.

    The client stamps every params message with a monotonic send time
    (arbitrary origin). ``offset = arrival_wall - client_time`` is then
    ``(clock origin delta) + (transit time)``: the origin delta is
    unknowable but constant, so the *minimum* offset seen over a sliding
    window approximates ``delta + min_transit``, and any excess above
    that minimum is queueing delay — time the report spent buffered in
    the socket, a tunnel middlebox, or the server's own recv backlog.
    That excess is exactly how far the reported position lags the
    client's true playhead.

    Windowed (rather than all-time) minimum so client/server crystal
    drift (~tens of ppm) can't slowly poison the baseline over a long
    session. A congestion episode longer than the window degrades
    gracefully: the minimum rises, staleness reads low, and behavior
    falls back to today's uncompensated clock.
    """

    _WINDOW_S = 120.0
    _BUCKET_S = 10.0

    def __init__(self):
        # (bucket_index, min_offset_in_bucket), oldest first.
        self._buckets: list = []

    def staleness_s(self, client_time_s: float, now_s: float) -> float:
        offset = now_s - float(client_time_s)
        bucket = int(now_s / self._BUCKET_S)
        if self._buckets and self._buckets[-1][0] == bucket:
            if offset < self._buckets[-1][1]:
                self._buckets[-1] = (bucket, offset)
        else:
            self._buckets.append((bucket, offset))
            # Age-based eviction (not count-based): after a traffic gap
            # the bucket list is sparse, and counting buckets would let
            # a minimum far older than the window keep anchoring the
            # baseline.
            cutoff = bucket - int(self._WINDOW_S / self._BUCKET_S)
            while self._buckets and self._buckets[0][0] < cutoff:
                self._buckets.pop(0)
        floor = min(m for _, m in self._buckets)
        return max(0.0, offset - floor)


class _RemotePlayheadClock:
    """Monotonic estimate of the client's audible playhead.

    The browser sends periodic absolute playback positions over the params
    channel. Those messages are the authority, but they can arrive slower
    than the runner loop or be coalesced under load. This clock anchors on
    the most recent observed sample and advances by wall time between
    anchors, so VAE scheduling remains continuous even when controls and
    WebSocket heartbeats are quiet.

    Staleness compensation: a report that spent time queued (congested
    uplink, tunnel buffering, recv backlog) describes where the playhead
    was when it was SENT, not where it is now. The session estimates that
    queueing delay per report (see :class:`ReportStalenessEstimator`) and
    publishes it as ``audio_eng.position_staleness_s``; anchoring the
    report's wall time that far in the past projects the estimate forward
    onto the client's true playhead. Without this, a growing backlog
    walks the estimate steadily into the past and every rendered slice
    lands behind the listener — heard as the raw source playing instead
    of the processed audio.
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
            staleness = float(
                getattr(self.audio_eng, "position_staleness_s", 0.0) or 0.0
            )
            self._observed = observed
            self._anchor_sample = observed
            self._anchor_wall_s = now - staleness
        elapsed = max(0.0, now - self._anchor_wall_s)
        return int(self._anchor_sample + elapsed * SAMPLE_RATE) % n

    def seconds(self) -> float:
        return self.sample() / SAMPLE_RATE


class PipelineRunner:
    """The generic streaming loop over a :class:`GeneratorBackend`.

    One injection point: *on_audio_ready* receives rendered audio.
    ``on_audio_ready(wav_np)``                     -- full-buffer render
    ``on_audio_ready(wav_np, win_start, win_end)`` -- windowed render
    """

    def __init__(
        self, backend, audio_eng, *,
        state,
        vae_window,
        on_audio_ready=None,
        before_tick=None,
        idle_threshold_s=0.0,
        lead_floor_s=None,
        lead_ceiling_s=None,
        lead_release_tau_s=None,
    ):
        self.backend = backend
        self.audio_eng = audio_eng
        # Single mutable session state object. The runner reads
        # ``state.running``, ``state.params`` (trace only), and
        # ``state.last_activity_ts``; the backend reads the rest.
        self.state = state
        # Wire-slice width. Must match the slice length the backend's
        # windowed render emits (the session clamps both from the same
        # engine profile). <= 0 selects the legacy full-buffer mode.
        self.vae_window = float(vae_window)
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
        # before_tick: optional callable usually invoked at the top of each
        # loop iteration on the runner thread.  A pending LoRA command can
        # defer it by one iteration so gap-fill can pre-cover the refit stall.
        # Used by the web server to
        # apply cross-thread mutations safely:
        #   - LoRA enable/disable (which triggers a refit; refit and
        #     inference are mutually exclusive)
        #   - source swap (prepare_source / encode_text / replace stream
        #     fields, which can't race the recv thread that holds the
        #     WebSocket)
        # The server's apply_pending() callback drains both queues each
        # iteration so they share one rendezvous point.
        self.before_tick = before_tick

        # Idle GPU pause. Two-stage shutdown:
        #   1) After ``idle_threshold_s`` with no inbound activity, skip
        #      the backend's generate step (the dominant per-tick GPU
        #      cost) and reuse its most recent cached result for the
        #      rest of the loop body. The render keeps windowed-
        #      decoding at the advancing playhead so audio continues
        #      uninterrupted, sending deltas to refresh the client's
        #      buffer from the stable cached state.
        #   2) Once the playhead has wrapped through one full cycle
        #      since the generate step paused, the client has the full
        #      buffer; further renders would produce identical audio.
        #      Rendering also stops (sleep+continue) until activity
        #      resumes. Any incoming WS message clears both stages.
        # Hot path is untouched when active or when disabled
        # (``idle_threshold_s <= 0``).
        self._idle_threshold_s = float(idle_threshold_s)
        self._dit_paused = False
        self._vae_paused = False
        self._dit_paused_at_wall_s = 0.0

        # ----- Playback lead vs render span: two SEPARATE concerns -----
        # The runner used to fold these together, pinning the playhead lead
        # to half the VAE decode span (0.5s for the fixed 1s profile). That
        # made every param change land ~0.5s ahead of the playhead, so the
        # listener heard it ~0.5s late no matter how fast it was produced
        # (~0.47s of the felt knob→ear latency was this lead, not compute).
        # They are now decoupled:
        #
        #   * render span: the backend's receptive field around the kept
        #     slice. Lives entirely inside the backend's render and must
        #     NEVER feed the playhead lead.
        #
        #   * playback lead (``_decode_advance_s``): how far ahead of the live
        #     playhead a freshly rendered slice is written so it lands before
        #     the listener reaches it. Sized from the *observed production
        #     interval*, below.

        # ----- Lead sizing: gap-fill + adaptive interval EMA + stall bump ----
        # The playback lead must guarantee a freshly rendered slice lands AHEAD
        # of the live playhead and is replaced by the next write before the
        # playhead overruns it. The binding quantity is therefore the
        # *inter-write interval*, NOT the render span.
        #
        # The engine produces completed generations in BURSTS (a batch drains,
        # then it stalls ~steps ticks while the next batch generates): measured
        # ~0.25s stalls at steps=8 and ~0.65s at steps=16, at every depth. If
        # the lead chases that raw stall it slams up at a param change and
        # wobbles at the burst cadence — the audible bounce. The fix has two
        # parts that together make the lead small AND smooth across the whole
        # depth x steps grid:
        #
        #   1) GAP-FILL (in ``run()``): on an active tick where the backend
        #      produces no new generation, re-render its cached state at the
        #      advancing playhead so a slice still lands. This collapses the
        #      inter-write interval to ~1 tick everywhere, so the burst stalls
        #      never reach the lead at all.
        #   2) This adaptive lead: an EMA of the (now ~1-tick) inter-write
        #      interval, scaled by a small gain, plus a fixed transit margin.
        #      Smooth because gap-fill removed the bursts that used to jerk it.
        #      It still self-sizes: heavier per-step compute (RCFG full, LoRA
        #      stacks, steps=16) lengthens the tick, the EMA tracks it, the
        #      lead grows proportionally. No magic constant.
        #
        # The one interval gap-fill cannot remove is a genuine pipeline rebuild
        # / refit stall (steps or LoRA enable/disable), where a single loop
        # iteration blocks for ~1s and no loop turn runs to gap-fill it.
        # ``_stall_extra_s`` covers that: a predictive prewarm raises it before
        # the stall lands, and a reactive term raises it for any observed gap
        # whose SHORTFALL beyond the slice width would otherwise underrun. It
        # decays back over ``_stall_release_tau_s`` so it never becomes a
        # permanent latency.
        self._last_decode_wall_s = time.monotonic()
        # EMA of the inter-write interval. Seeded near a steady active tick.
        self._decode_interval_ema_s = 0.07
        # EMA smoothing per write (~15 writes/s under gap-fill -> ~0.7s tau).
        self._decode_interval_alpha = 0.1
        # Gaps above this are treated as stalls (rebuild / unexpected): they
        # feed ``_stall_extra_s`` via the shortfall term, NOT the steady EMA,
        # so one stall can't inflate the steady interval estimate.
        self._interval_ema_cap_s = 0.18
        # Lead = EMA * gain + margin + stall_extra. Gain gives headroom over a
        # single inter-write interval against tick jitter; margin covers
        # client scheduling slop and the measure->land transit.
        self._lead_interval_gain = 1.6
        self._lead_safety_margin_s = 0.05

        # Backend-declared lead bounds, overridden by per-session config
        # (SessionConfig.lead_*), falling back to the historical runner
        # defaults. For the ACE backend the profile declares no opinion,
        # so the resolved values are byte-identical to the pre-seam
        # behavior.
        profile = backend.lead_profile()
        # Floor on the steady lead. The original 0.05s self-sizes to an *idle*
        # GPU but leaves no slack for contention: ANY co-resident load (screen
        # capture, the WebGPU display, a second process) lengthens ticks, and a
        # tiny baseline lead then underruns on the first slow tick and has to
        # chase it reactively — the audible sawtooth that "explodes" under
        # contention. Set MIDWAY between that bare floor and the old fixed
        # ~0.5s lead: enough baseline slack to absorb moderate contention up
        # front without paying the full latency of the old fixed behavior.
        # Operator-overridable via config.json (engine.lead_floor_s); the
        # literal is the standalone-caller default when no override is passed.
        if lead_floor_s is None:
            lead_floor_s = profile.floor_s
        self._lead_floor_s = 0.25 if lead_floor_s is None else float(lead_floor_s)
        # Defensive ceiling: never park a slice more than this far ahead, so
        # the modulo-``eff_dur`` wrap below can't fold the write back onto the
        # playhead. Lowered from 1.6 (midway) so sustained contention can't
        # inflate the lead toward a multi-second latency; it still clears the
        # rebuild-prewarm bump (~1.1s) so a refit stall stays covered. Kept
        # BELOW ``_stall_release_tau_s`` so the decay rate (<= ceiling/tau <
        # 1.0/s) can never shrink the lead faster than the playhead advances —
        # i.e. ``decode_start`` stays monotonic during decay and we never
        # re-render an earlier position. Operator-overridable via config.json
        # (engine.lead_ceiling_s).
        if lead_ceiling_s is None:
            lead_ceiling_s = profile.ceiling_s
        self._decode_lead_ceiling_s = (
            1.35 if lead_ceiling_s is None else float(lead_ceiling_s)
        )
        # One-shot stall coverage (rebuild prewarm + reactive shortfall). Rises
        # immediately, decays over tau so it is never a permanent lead. tau is
        # kept >= the ceiling (the monotonic-decode invariant: 1.5 >= 1.35) and
        # shortened from 1.8s (midway) so a contention spike releases faster
        # instead of pinning the lead high. Operator-overridable via
        # config.json (engine.lead_release_tau_s); clamped up to the ceiling
        # just below so an operator misconfig can't break the invariant.
        self._stall_extra_s = 0.0
        if lead_release_tau_s is None:
            lead_release_tau_s = profile.release_tau_s
        self._stall_release_tau_s = (
            1.5 if lead_release_tau_s is None else float(lead_release_tau_s)
        )
        self._stall_release_tau_s = max(
            self._stall_release_tau_s, self._decode_lead_ceiling_s,
        )

        # ----- Pre-stall coverage on rebuild-triggering param changes -----
        # The single ~1s stall on the tick a rebuild-triggering param first
        # lands can't be covered reactively OR by merely raising the lead: the
        # rebuild happens INSIDE the backend's generate step, the loop is
        # blocked there so no gap-fill runs, and the next windowed write only
        # lands AFTER the stall. So when the backend signals such a change we
        # (1) raise ``_stall_extra_s`` to a learned worst-rebuild estimate, and
        # (2) SKIP the blocking operation for that one iteration (see
        # ``run()``), gap-filling a far-ahead slice from the backend's cached
        # state so the buffer is covered through the stall, which then lands
        # on the next tick. For LoRA enable/disable, the blocking work happens
        # inside ``before_tick`` rather than the generate step, so ``run()``
        # asks the backend about pending refits before calling that hook and
        # performs the same one-iteration gap-fill first. The estimate
        # self-calibrates toward the largest stall we actually observe, seeded
        # from the measured ~1.1s rebuild and capped so a one-off outlier (e.g.
        # the multi-second session-startup build) can't push it to the ceiling.
        self._rebuild_prewarm_s = 1.1
        self._rebuild_prewarm_cap_s = 1.3
        self._playhead_clock = _RemotePlayheadClock(self.audio_eng)

        # ----- Transport lead: close the loop on where slices LAND ---------
        # Everything above sizes the lead from server-side signals, but the
        # binding constraint is on the CLIENT: a slice must be applied to the
        # playback buffer before the playhead reaches it, and between our
        # write and that apply sit WS transit, decode, and the client's main-
        # thread scheduling (a background-throttled tab applies slices in
        # ~1 s bursts). The client reports the worst landing lead it observed
        # via the params channel (``slice_lead_s``, negative = the slice
        # landed in already-played audio and the listener heard raw source);
        # ``_transport_extra_s`` rises immediately to cover the deficit and
        # decays slowly while reports stay healthy. Asymmetric on purpose:
        # the deficit signal arrives late (one client report cycle + uplink),
        # so raise-fast/decay-slow is what keeps the loop stable.
        self._transport_extra_s = 0.0
        # Keep reported leads at or above this margin before easing off.
        self._slice_lead_margin_s = 0.15
        # Hard cap: beyond this the client isn't late, it's not consuming
        # (e.g. mid-drain of a long backlog) — writing further ahead just
        # queues more. Also bounds added knob→ear latency.
        self._transport_extra_cap_s = 3.0
        # Decay time constant. Max decay rate = cap/tau = 0.15 s/s, far
        # below the playhead rate, so decode_start stays monotonic during
        # release (same invariant the stall-bump tau preserves).
        self._transport_release_tau_s = 20.0
        # Wall stamp of the last report folded in (rate-limits the fold to
        # one per client report).
        self._last_slice_lead_wall_s = 0.0
        # Most recent reported lead value + when it was folded. The decay
        # below is CONDITIONAL on these: shrink the transport extra only
        # while reports show comfortable headroom (lead > margin +
        # hysteresis) or have gone stale. An unconditional fixed-rate
        # decay limit-cycles on a saturated link: extra decays, the next
        # report dips negative, extra re-raises — audible as alternating
        # clean/raw seconds. Hold-in-band turns that into a stable lead.
        self._last_slice_lead_value = None
        self._transport_decay_hysteresis_s = 0.2
        # Reports older than this allow decay regardless (client gone
        # quiet — drain the extra rather than pinning latency forever).
        self._transport_report_stale_s = 10.0

    # ---- delegates kept for the session's runner_holder contract ----------

    def mark_hint_dirty(self) -> None:
        """Delegate to the backend (see ACEStepBackend.mark_hint_dirty)."""
        fn = getattr(self.backend, "mark_hint_dirty", None)
        if fn is not None:
            fn()

    def _rebuild_silence_latent(self) -> None:
        """Delegate to the backend (see ACEStepBackend)."""
        fn = getattr(self.backend, "_rebuild_silence_latent", None)
        if fn is not None:
            fn()

    # ---- lead machinery -----------------------------------------------------

    def _decode_advance_s(self) -> float:
        """Playback lead: how far AHEAD of the live playhead to place a fresh
        slice. Adaptive, NOT a constant and NOT the render span:

            lead = interval_ema * gain + safety_margin + stall_extra

        ``interval_ema`` tracks the (gap-filled, ~1-tick) inter-write interval,
        so it self-sizes with per-step compute; ``stall_extra`` is the decaying
        one-shot bump that covers rebuild stalls. Clamped to ``[floor, ceiling]``.
        See the init block and ``_note_decode_gap``.
        """
        if self.vae_window <= 0:
            return 0.0
        lead = (
            self._decode_interval_ema_s * self._lead_interval_gain
            + self._lead_safety_margin_s
            + self._stall_extra_s
        )
        lead = min(max(lead, self._lead_floor_s), self._decode_lead_ceiling_s)
        # Transport extra rides OUTSIDE the clamp: the ceiling bounds the
        # server-side adaptive term, while the transport term covers the
        # client-side path the server can't observe locally. It has its
        # own cap (_transport_extra_cap_s).
        return lead + self._transport_extra_s

    def _note_decode_gap(self) -> float:
        """Fold this write's wall-clock gap since the previous write into the
        adaptive lead state, and return the gap (for the trace). Call once per
        successful windowed write — real generation OR gap-fill.

        Two updates:
          * The steady interval EMA tracks normal ~1-tick gaps (gaps are capped
            into the EMA so a one-off stall can't inflate the steady estimate).
          * ``_stall_extra_s`` decays toward 0 by the elapsed wall time, then
            is lifted by the SHORTFALL of this gap beyond the slice width — the
            only part a slice's own width does not already cover. Normal
            sub-slice gaps never move it; a genuine stall does, transiently.
        """
        now = time.monotonic()
        gap = now - self._last_decode_wall_s
        self._last_decode_wall_s = now
        if gap <= 0.0:
            return 0.0
        # Steady interval EMA, on the capped gap (stalls are excluded here and
        # handled by the stall bump below).
        capped = min(gap, self._interval_ema_cap_s)
        a = self._decode_interval_alpha
        self._decode_interval_ema_s = (
            (1.0 - a) * self._decode_interval_ema_s + a * capped
        )
        # Time-based decay of the one-shot stall bump.
        self._stall_extra_s *= math.exp(-gap / self._stall_release_tau_s)
        # Conditional decay of the transport lead — only while the client
        # reports comfortable headroom (or went quiet). See the hysteresis
        # comment in __init__; deficit reports re-raise it in
        # _fold_slice_lead_report.
        if self._transport_extra_s > 0.0:
            last_lead = self._last_slice_lead_value
            report_age = now - self._last_slice_lead_wall_s
            healthy = (
                last_lead is not None
                and last_lead > (
                    self._slice_lead_margin_s
                    + self._transport_decay_hysteresis_s
                )
            )
            stale = (
                last_lead is None
                or report_age > self._transport_report_stale_s
            )
            if healthy or stale:
                self._transport_extra_s *= math.exp(
                    -gap / self._transport_release_tau_s,
                )
        # Reactive shortfall: only the part of this gap beyond the slice width
        # (plus a small margin) can leave a hole; lift the bump to cover it.
        shortfall = gap - self.vae_window + self._lead_safety_margin_s
        if shortfall > self._stall_extra_s:
            self._stall_extra_s = min(shortfall, self._decode_lead_ceiling_s)
        # Self-calibrate the rebuild prewarm toward the worst real stall, but
        # cap it so a single outlier can't push the predictive lead toward
        # the ceiling.
        if gap > self._rebuild_prewarm_s:
            self._rebuild_prewarm_s = min(gap, self._rebuild_prewarm_cap_s)
        return gap

    def _fold_slice_lead_report(self) -> None:
        """Fold the client's latest landing-lead report into the transport
        lead. Called once per windowed tick, rate-limited to one fold per
        report by the report's arrival stamp.

        Skipped while a loop band is armed: the client computes leads
        linearly, but band playback wraps B→A, so a render correctly
        pre-filling the seam after A reads as a large NEGATIVE linear lead
        and would spuriously inflate the transport term.
        """
        if getattr(self.audio_eng, "loop_band", None) is not None:
            return
        lead = getattr(self.audio_eng, "observed_slice_lead_s", None)
        if lead is None:
            return
        wall = float(
            getattr(self.audio_eng, "observed_slice_lead_wall_s", 0.0) or 0.0
        )
        if wall <= self._last_slice_lead_wall_s:
            return
        self._last_slice_lead_wall_s = wall
        self._last_slice_lead_value = float(lead)
        deficit = self._slice_lead_margin_s - float(lead)
        if deficit > 0.0:
            # ADDITIVE raise: the reported lead was observed under the
            # transport extra already in effect, so a shortfall means the
            # current total is short by that amount. (A max() rule would
            # equilibrate with reports sitting BELOW the margin by the
            # extra's size.) Rate-limited to one fold per client report;
            # the cap bounds drain-phase overshoot and added latency.
            self._transport_extra_s = min(
                self._transport_extra_s + deficit,
                self._transport_extra_cap_s,
            )

    def _playhead_seconds_now(self) -> float:
        return self._playhead_clock.seconds()

    # ---- the loop -------------------------------------------------------------

    def run(self):
        backend = self.backend
        logger.info(
            "stream decode: vae_window={:.3f}s decode_span={:.3f}s "
            "lead_margin={:.3f}s lead~={:.3f}s backend={}",
            self.vae_window,
            float(getattr(backend, "decode_span_s", 0.0)),
            self._lead_safety_margin_s,
            self._decode_advance_s(),
            backend.name,
        )

        # Anchor the gap clock at loop entry so the first write doesn't fold
        # the (multi-second) model-load time into the envelope as a spurious
        # giant gap.
        self._last_decode_wall_s = time.monotonic()

        # Whether the previous iteration skipped its tick to pre-cover a
        # rebuild/refit stall. Used to forbid two skips in a row so a continuous
        # sweep of a rebuild-triggering knob can't starve real generation.
        deferred_rebuild_last = False

        while self.state.running:
            pre_defer_refit = (
                backend.has_pending_refit()
                and not deferred_rebuild_last
                and self.vae_window > 0
                and backend.has_renderable_state()
            )
            if pre_defer_refit:
                self._stall_extra_s = max(
                    self._stall_extra_s, self._rebuild_prewarm_s,
                )
            elif self.before_tick is not None:
                # Hook for cross-thread mutations (LoRA enable/disable
                # AND source swap).  Runs on the runner thread so any
                # GPU/refit work the callback does is serialized with
                # the tick body.
                self.before_tick()

            # Idle GPU pause — stage detection. Updates the flags read
            # at the produce-mode selection below. Disabled when the
            # threshold is non-positive (standalone callers fall
            # through to the normal path with zero overhead).
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
                # client's audio buffer with audio rendered from the
                # cached state.
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
                # the pause is equivalent to playhead advance. Once
                # we've passed one full buffer-duration's worth of
                # wall time, every chunk has been re-rendered from the
                # cached state and shipped to the client; further
                # renders would produce identical samples. Robust
                # against modulo wrap-around edge cases that a
                # playhead-position diff would miss when the runner
                # stalls or the buffer length changes.
                buf_dur_s = max(
                    1e-6, len(self.audio_eng.current) / SAMPLE_RATE,
                )
                wall_since_pause_s = time.monotonic() - self._dit_paused_at_wall_s
                if (
                    backend.has_renderable_state()
                    and wall_since_pause_s >= buf_dur_s
                ):
                    self._vae_paused = True
                    logger.info("pipeline_paused stage=vae reason=buffer_full")
                    time.sleep(0.05)
                    continue

                # If the backend has no cached state yet (idle hit
                # before any successful tick), fall through to the
                # normal path — generating is the only way to produce
                # one. Rare in practice.

            # Per-tick source reconciliation (source swap detection,
            # walk-window chunk selection) happens at this exact point:
            # after idle staging, before the knob read — the stall-
            # deferral checks above observe the backend's cache state
            # PRE-reset, the ones below observe it POST-reset, matching
            # the historical fused loop.
            ctx = TickContext(
                playhead_s=self._playhead_seconds_now(),
                buffer_duration_s=max(
                    1e-6, len(self.audio_eng.current) / SAMPLE_RATE,
                ),
            )
            backend.sync_source(ctx)

            raw = backend.read_knobs()

            # Pre-stall coverage: if a rebuild-triggering param changed since
            # the last tick, the rebuild will block ~1s INSIDE the backend's
            # generate step below. LoRA enable/disable is similar, except the
            # stall happens in ``before_tick``; that case is detected at the
            # top of the loop as ``pre_defer_refit`` and intentionally skips
            # ``before_tick`` once. In both cases we (1) raise the one-shot
            # stall bump, and (2) SKIP the blocking operation for this one
            # iteration (``defer_rebuild``), routing it through the gap-fill
            # path so a far-ahead slice is written from the cached state and
            # covers the buffer through the stall, which then lands on the
            # NEXT tick. The new params take effect one tick later (~tens of
            # ms). We never defer two ticks running, so a continuous sweep of
            # a rebuild-triggering knob can't starve real generation; the bump
            # decays back out via ``_note_decode_gap``.
            rebuild_changed = backend.rebuild_imminent(raw)
            defer_rebuild = (
                pre_defer_refit
                or (
                    rebuild_changed
                    and not deferred_rebuild_last
                    and self.vae_window > 0
                    and backend.has_renderable_state()
                )
            )
            if defer_rebuild and not pre_defer_refit:
                self._stall_extra_s = max(
                    self._stall_extra_s, self._rebuild_prewarm_s,
                )
            deferred_rebuild_last = defer_rebuild

            # Produce-mode selection. "reuse" (DiT-pause) re-adopts the
            # backend's cached state as a fresh result; "skip" (stall
            # deferral) produces nothing and routes through gap-fill;
            # "generate" is the normal engine step. The backend runs its
            # full prepare path in every mode so live control changes
            # keep landing on in-flight work.
            paused_reuse = self._dit_paused and backend.has_renderable_state()
            if paused_reuse:
                mode = "reuse"
            elif defer_rebuild:
                mode = "skip"
            else:
                mode = "generate"
            is_fresh = backend.produce(raw, ctx, mode)
            if paused_reuse:
                # No expensive generate step paced the loop. Keep the
                # render refresh comfortably faster than the wire slice
                # without spinning at CPU speed.
                time.sleep(0.02)

            # Gap-fill: on an active tick where the backend produced no new
            # generation, re-render its cached state at the ADVANCING playhead
            # so a fresh windowed slice still lands this tick. Without it the
            # inter-write gap balloons to the production stall (~0.25s at
            # steps=8, ~0.65s at steps=16) and the lead has to chase that
            # stall; with it the gap is ~1 tick everywhere, so the lead stays
            # small and smooth across the whole depth x steps grid. A gap-fill
            # tick does the windowed render+write ONLY — the backend's
            # per-generation bookkeeping is untouched (that belongs to real
            # generations and would corrupt the feedback delay-tap). Windowed
            # path only; the full-buffer path has no advancing slice to
            # refresh. The DiT-pause branch above reports fresh, so gap-fill
            # is purely additive to the active path.
            gap_fill = (
                not is_fresh
                and self.vae_window > 0
                and backend.has_renderable_state()
            )
            if is_fresh or gap_fill:
                if self.vae_window > 0:
                    # eff_dur clamps the windowed-render playhead so the
                    # window stays inside the song. A backend that
                    # declares no fixed duration (walk mode slides a
                    # window over a longer song) defers to the audio
                    # buffer length, which tracks crop and source swaps.
                    pd = backend.playable_duration_s()
                    position_chase_only = pd is None
                    eff_dur = (
                        len(self.audio_eng.current) / SAMPLE_RATE
                        if pd is None else pd
                    )

                    # Fold the client's latest landing-lead report into the
                    # transport lead before sizing this tick's advance.
                    self._fold_slice_lead_report()

                    playhead_now = self._playhead_seconds_now()
                    # Predictive render start: target where the playhead
                    # WILL be by the time this slice lands in the buffer.
                    # The lead is the adaptive interval EMA (+ stall bump),
                    # NOT the render span — see ``_decode_advance_s``. Wrap
                    # modulo ``eff_dur`` since the render supports cyclic.
                    advance_s = self._decode_advance_s()
                    decode_start = playhead_now + advance_s
                    if eff_dur > 0:
                        decode_start = decode_start % eff_dur

                    # Loop-band awareness. When the client arms a band
                    # [A, B] the worklet replays only that region (hard-
                    # wrapping B→A) while we keep generating. Chasing the
                    # raw playhead would render *past* B (audio that's
                    # never heard) and leave the region just after A
                    # holding pre-change audio until the playhead crawls
                    # back over it — the "snap back to the old buffer for
                    # one window" the operator hears at every loop
                    # restart. Wrapping the render target inside the band
                    # instead pre-fills the seam after A while the
                    # playhead is still finishing the lap near B, so the
                    # restart plays freshly-generated audio. Only the
                    # standard windowed path is band-aware; walk mode
                    # (multi-minute sources) keeps its linear chase.
                    band_start_sample = None
                    band_end_sample = None
                    band_wrap_start_s = None
                    band = getattr(self.audio_eng, "loop_band", None)
                    if band is not None and eff_dur > 0 and not position_chase_only:
                        a_s = max(0.0, min(float(band[0]), eff_dur))
                        b_s = max(0.0, min(float(band[1]), eff_dur))
                        span = b_s - a_s
                        # Only pin the render into the band while the
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
                                # Band shorter than one render window:
                                # pin the target at A so every render
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

                    chunk = backend.render_window(decode_start)
                    if chunk is not None:
                        win_np = chunk.pcm
                        win_start = chunk.start_sample
                        win_end = win_start + win_np.shape[0]
                        # Read the boundary slices directly from the
                        # audio engine's live buffer. The previous
                        # ``buf = self.audio_eng.current.copy()`` cost
                        # ~23 MB of host RAM per windowed render for a
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
                        # Loop band active: never write the first render past
                        # B. If the window crossed B, a second render anchored
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
                        # Fold this write's wall gap into the adaptive lead
                        # state. One call per successful write — real
                        # generation OR gap-fill; the band-wrap second render
                        # below is part of the same tick and must not count as
                        # its own interval.
                        decode_gap_s = self._note_decode_gap()
                        # Legacy ``last_wav`` mirror (only the full-buffer
                        # MSE skip reads it; kept for exact parity).
                        note = getattr(backend, "note_windowed_write", None)
                        if note is not None:
                            note(current)
                        if _LAT_TRACE:
                            # Wrap the lead modulo the playable duration and
                            # center it in [-dur/2, dur/2): a slice written at
                            # the buffer head while the playhead nears the end
                            # (the loop pre-write) is a small POSITIVE lead,
                            # not -duration. Without the fold, every wrap
                            # printed lead_s≈-57 and drowned real underruns.
                            lead_trace_s = win_start / SAMPLE_RATE - playhead_now
                            if eff_dur > 0:
                                lead_trace_s = (
                                    (lead_trace_s + eff_dur / 2) % eff_dur
                                    - eff_dur / 2
                                )
                            logger.info(
                                "lat_decode num_gens={} denoise={:.3f} "
                                "fresh={} playhead_s={:.3f} advance_s={:.3f} "
                                "gap_s={:.3f} ema_s={:.3f} stall_s={:.3f} "
                                "transport_s={:.3f} staleness_s={:.3f} "
                                "decode_start_s={:.3f} win_start_s={:.3f} "
                                "win_end_s={:.3f} lead_s={:.3f} "
                                "tick_ms={:.1f} dec_ms={:.1f}",
                                self.state.params.get("num_gens", 0) + 1,
                                float(getattr(backend, "last_denoise", 0.0)),
                                int(is_fresh), playhead_now, advance_s,
                                decode_gap_s, self._decode_interval_ema_s,
                                self._stall_extra_s,
                                self._transport_extra_s,
                                float(getattr(
                                    self.audio_eng, "position_staleness_s", 0.0,
                                ) or 0.0),
                                decode_start,
                                win_start / SAMPLE_RATE, win_end / SAMPLE_RATE,
                                lead_trace_s,
                                backend.last_tick_ms, backend.last_dec_ms,
                            )
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
                                wrap_chunk = backend.render_window(band_wrap_start_s)
                                if wrap_chunk is not None:
                                    wrap_np = wrap_chunk.pcm[:wrap_len].copy()
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
                else:
                    # Legacy full-buffer mode. Gap-fill never reaches
                    # here (it requires vae_window > 0), so only fresh
                    # generations render; the backend applies its own
                    # MSE skip and returns None when the latent barely
                    # moved.
                    chunk = backend.render_full()
                    if chunk is not None:
                        self.on_audio_ready(chunk.pcm)

                # Per-generation state mirror. A gap-fill tick produced no new
                # generation, so it must NOT bump ``num_gens`` or restamp the
                # param snapshot — only real generations advance this.
                if is_fresh:
                    backend.on_fresh_generation(raw)
