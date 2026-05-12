"""PipelineRunner: the torch-heavy streaming loop (graph-driven).

Drives a :class:`~acestep.engine.session.StreamHandle` by calling
``handle.tick(**kwargs)`` each iteration, where the kwargs mirror the
knob state.
"""

import time

import numpy as np
import torch

from acestep.engine.dcw import DCWAdvanced
from acestep.nodes.types import ChannelGuidanceEntry, Latent
from acestep.nodes.vae_nodes import EmptyLatent, LatentBlend

from .knobs import CHANNEL_GROUPS, KEYSTONE_CHANNELS
from .protocol import SAMPLE_RATE, T


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
        use_midi, use_sde, use_lora,
        midi_knobs, engine_obj,
        vae_window, crop_seconds,
        k1_name, seed, skip_threshold,
        sde_curve_display, params, prompt_text, running,
        motion_val, motion_lock,
        on_audio_ready=None,
        before_tick=None,
        walk_window=False,
        walk_window_s=60.0,
        neg_conditioning=None,
    ):
        self.session = session
        self.stream = stream  # StreamHandle
        self.audio_eng = audio_eng
        self.use_midi = use_midi
        self.use_sde = use_sde
        self.use_lora = use_lora
        self.midi_knobs = midi_knobs
        self.engine_obj = engine_obj
        self.vae_window = vae_window
        self.crop_seconds = crop_seconds
        self.k1_name = k1_name
        self.SEED = seed
        self.skip_threshold = skip_threshold
        self.sde_curve_display = sde_curve_display
        self.params = params
        self.prompt_text = prompt_text
        self.running = running
        self.motion_val = motion_val
        self.motion_lock = motion_lock
        if on_audio_ready is None:
            on_audio_ready = lambda wav, *_args: audio_eng.swap(wav)
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
        # start (see backend.py / fixtures.py) and reused across all ticks.
        # Required for ``rcfg_mode in {"full", "initialize"}``; ignored
        # by ``rcfg_mode == "self"`` (virtual uncond) and ``"off"``.
        # ``None`` is safe — modes that need it become quiet no-ops.
        self.neg_conditioning = neg_conditioning

        # Predictive decode: rolling EMA of (tick + decode) wall time. Each
        # decode targets ``playhead + _predicted_advance_s`` so that by the
        # time the freshly-decoded window is written into the buffer, its
        # leading edge (and the per-window crossfade ramp) lines up with
        # where the listener actually is. Without this, ``win_start`` is
        # set to the playhead at decode-START, which by write-time is
        # already ``dec_ms`` in the past — the listener has marched past
        # the crossfade region and hears the new params start abruptly
        # mid-window. Capped to half the VAE window so a transient stall
        # can't lock the prediction onto a value that puts new audio
        # arbitrarily far ahead of the actual playhead.
        self._predicted_advance_s = 0.1

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
        last_wav = None
        last_decode_pos = None
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

        while self.running[0]:
            if self.before_tick is not None:
                # Hook for cross-thread mutations (LoRA enable/disable
                # AND source swap).  Runs on the runner thread so any
                # GPU/refit work the callback does is serialized with
                # the tick body.
                self.before_tick()

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
            # Use ``playhead + predicted_advance`` so the swap happens a
            # tick or two before the boundary, giving the new chunk's
            # ring-buffer warmup head room to land before the listener
            # actually crosses.
            walk_w0 = -1
            walk_w1 = -1
            walk_chunk_start_s = 0.0
            if walk_active:
                playhead_now_s = self.audio_eng.position / SAMPLE_RATE
                advance_s_for_chunk = min(
                    self._predicted_advance_s, self.walk_window_s * 0.5,
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
                last_wav = None
                last_decode_pos = None
                prev_src_id = cur_src_id
                prev_src_T = cur_src_T
                cached_live_src_lat = None
                cached_live_ctx_raw_t = None
                if walk_active:
                    prev_walk_w0 = walk_w0

            if self.use_midi:
                raw = self.midi_knobs.get_all_values()
            else:
                with self.motion_lock:
                    m = self.motion_val[0]
                raw = {self.k1_name: m, "seed": 0.0, "feedback": 0.0, "shift": 0.5}
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
            seed = int(raw["seed"] * 1000) if self.use_midi else self.SEED
            feedback = raw["feedback"]
            shift_raw = raw["shift"]

            shift_val = 1.0 + shift_raw * 5.0
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
                    if abs(lora_str - self.params.get(key, -1)) > 0.02:
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
                    )["latent"]
                last_hint_str = hint_str
                self._hint_dirty = False
            else:
                live_ctx_lat = None
                if self._hint_dirty or abs(hint_str - last_hint_str) > 0.02:
                    self._hint_dirty = False
                    last_hint_str = hint_str
                    self._update_hint_strength(hint_str)

            source_lat = None
            if feedback > 0.0 and last_latent is not None:
                src_tensor = live_src_lat.tensor
                source_lat = (1.0 - feedback) * src_tensor + feedback * last_latent

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
                self.sde_curve_display[0] = sde_curve.squeeze().numpy()
            else:
                denoise = k1
                self.sde_curve_display[0] = None

            ode_curve = _curve_from_spec(raw.get("ode_noise_curve"), src_T)
            if ode_curve is None:
                ode_noise_val = self.midi_knobs.get_param("ode_noise") if self.use_midi else 0.0
                ode_curve = torch.full((1, src_T, 1), ode_noise_val) if ode_noise_val > 0.01 else None

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
                pipe.set_shared_curve("ode_noise_curve", ode_curve)
                pipe.set_shared_curve("velocity_scale", velocity_curve)
                pipe.set_shared_curve("x0_target_strength", x0_str)

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            tick_kwargs = {}
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
                # DCW (wavelet-domain post-step correction). Forwarded
                # every tick so toggle / mode / wavelet changes from the
                # client take effect on the next slot via pipe.set_dcw().
                # Default on — matches upstream v0.1.7.
                dcw_enabled=bool(raw.get("dcw_enabled", True)),
                dcw_mode=str(raw.get("dcw_mode", "double")),
                dcw_scaler=float(raw.get("dcw_scaler", 0.05)),
                dcw_high_scaler=float(raw.get("dcw_high_scaler", 0.02)),
                dcw_wavelet=str(raw.get("dcw_wavelet", "haar")),
                dcw_advanced=_build_dcw_advanced(raw),
            )
            torch.cuda.synchronize()
            tick_ms = (time.perf_counter() - t0) * 1000

            dec_ms = 0.0
            if result_latent is not None:
                result = result_latent.tensor
                skipped = False
                if last_latent is not None:
                    mse = (result - last_latent).pow(2).mean().item()
                    if mse < self.skip_threshold and last_wav is not None:
                        if self.vae_window > 0:
                            t_pos = self.audio_eng.position / SAMPLE_RATE
                            # Larger prefetch fraction → re-decode sooner
                            # before the playhead reaches the trailing
                            # edge of the previous window. Bumped from 0.2
                            # to 0.3 so fresh params land sooner at the
                            # cost of slightly more GPU work.
                            prefetch = min(1.0, self.vae_window * 0.3)
                            if last_decode_pos is not None and abs(t_pos - last_decode_pos) < self.vae_window - prefetch:
                                skipped = True
                        else:
                            skipped = True

                last_latent = result.clone()

                if not skipped:
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
                        playhead_now = self.audio_eng.position / SAMPLE_RATE
                        # Predictive decode start: target where the playhead
                        # WILL be by the time this window lands in the buffer
                        # (≈ tick + dec wall time from now). Cap at half the
                        # VAE window so a noisy spike can't push new audio
                        # arbitrarily far into the future. Wrap modulo
                        # ``eff_dur`` since the decoder supports cyclic.
                        advance_s = min(self._predicted_advance_s, self.vae_window * 0.5)
                        decode_start = playhead_now + advance_s
                        if eff_dur > 0:
                            decode_start = decode_start % eff_dur
                        # The skip-decode bookkeeping anchors on the
                        # *predicted* start so the next iteration's drift
                        # check measures distance from the start of the
                        # window we actually decoded, not from the playhead
                        # at decode-time.
                        last_decode_pos = decode_start
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
                        buf = self.audio_eng.current.copy()
                        # 25 ms at 48 kHz — matches CROSSFADE_SECONDS.
                        # Cuts perceived "smear" of param transitions in
                        # half from the previous 50 ms.
                        xfade = min(1200, win_np.shape[0] // 4)
                        if win_start > 0 and xfade > 0:
                            t_in = np.linspace(0.0, 1.0, xfade).reshape(-1, 1)
                            win_np[:xfade] = (
                                buf[win_start:win_start + xfade] * (1 - t_in)
                                + win_np[:xfade] * t_in
                            )
                        if win_end < buf.shape[0] and xfade > 0:
                            t_out = np.linspace(1.0, 0.0, xfade).reshape(-1, 1)
                            tail = min(xfade, buf.shape[0] - win_end + xfade)
                            s = win_np.shape[0] - tail
                            win_np[s:] = (
                                win_np[s:] * t_out[:tail]
                                + buf[win_start + s:win_start + s + tail] * (1 - t_out[:tail])
                            )
                        clamp_end = min(win_end, buf.shape[0])
                        buf[win_start:clamp_end] = win_np[:clamp_end - win_start]
                        self.on_audio_ready(buf, win_start, win_end)
                        last_wav = buf
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

                self.params["num_gens"] = self.params.get("num_gens", 0) + 1
                self.params["tick_ms"] = tick_ms
                self.params["dec_ms"] = dec_ms
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
                self.params[self.k1_name] = round(k1, 2)
                self.params["seed"] = seed
                self.params["feedback"] = round(feedback, 2)
                self.params["shift"] = round(shift_val, 2)
                if self.use_lora and self.engine_obj is not None:
                    for desc in self.engine_obj.list_loras():
                        if desc.state != "enabled":
                            continue
                        key = f"lora_str_{desc.id}"
                        self.params[key] = round(raw.get(key, desc.strength), 2)
                if self.use_sde:
                    self.params["periodicity"] = round(raw.get("periodicity", 0.0), 2)
                self.params["hint_strength"] = round(hint_str, 2)
                self.params["ode_noise"] = round(ode_noise_val, 2)
                for name, _, _ in CHANNEL_GROUPS:
                    self.params[name] = round(raw.get(name, 1.0), 2)
                for name, _ in KEYSTONE_CHANNELS:
                    self.params[name] = round(raw.get(name, 1.0), 2)
                self.params["_prompt"] = self.prompt_text[0]
