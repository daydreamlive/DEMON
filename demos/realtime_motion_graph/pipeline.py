"""PipelineRunner: the torch-heavy streaming loop (graph-driven).

Used by :mod:`full_demo` (local mode) and :mod:`server` (remote mode).
Not imported by the thin client under :mod:`client`.

Phase 3 migrated the per-tick path to the node graph: the runner now
drives a :class:`~acestep.engine.session.StreamHandle` by calling
``handle.tick(**kwargs)`` each iteration, where the kwargs mirror the
knob state. Every ``set_*`` mutator on the old ``SessionStream`` is
expressed as either a per-tick kwarg or a direct edit of handle fields
(``handle.conditioning``, ``handle.context_latent``).
"""

import time

import numpy as np
import torch

from acestep.nodes.types import ChannelGuidanceEntry, Latent
from acestep.nodes.vae_nodes import EmptyLatent, LatentBlend

from .client.knobs import CHANNEL_GROUPS, KEYSTONE_CHANNELS
from .client.protocol import SAMPLE_RATE, T



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

        # Cache silence once; used by the hint-strength blend node.
        self._rebuild_silence_latent()

    def _rebuild_silence_latent(self) -> None:
        """(Re)build the silence latent used by hint-strength blending.

        Tracks the current ``stream.source`` length, so call this after a
        source swap if the new source has a different latent length.
        """
        T_frames = self.stream.source.latent.tensor.shape[1]
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

        while self.running[0]:
            if self.before_tick is not None:
                # Hook for cross-thread mutations (LoRA enable/disable
                # AND source swap).  Runs on the runner thread so any
                # GPU/refit work the callback does is serialized with
                # the tick body.
                self.before_tick()

            # If a swap changed the source latent length, the cached
            # last_latent / last_wav / last_decode_pos are dimensioned
            # against the old source and would crash both the feedback
            # blend (source_lat = ... + feedback * last_latent) and the
            # skip-decode MSE check below. Reset them so the next tick
            # re-initializes against the new source cleanly.
            cur_src_T = self.stream.source.latent.tensor.shape[1]
            if cur_src_T != prev_src_T:
                last_latent = None
                last_wav = None
                last_decode_pos = None
                prev_src_T = cur_src_T
            if self.use_midi:
                raw = self.midi_knobs.get_all_values()
            else:
                with self.motion_lock:
                    m = self.motion_val[0]
                raw = {self.k1_name: m, "seed": 0.0, "feedback": 0.0, "shift": 0.5}
                if self.use_sde:
                    raw["periodicity"] = 0.0

            # Actual source latent length. Hardcoded T=1500 is a 60s default
            # but sources can be shorter (e.g. 25s → 645 frames). Curves must
            # match this T or broadcasting fails in _init_slot / _step_sde.
            src_T = self.stream.source.latent.tensor.shape[1]

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
                for desc in self.engine_obj.list_trt_loras():
                    if desc.state != "enabled":
                        continue
                    key = f"lora_str_{desc.id}"
                    lora_str = raw.get(key, desc.strength)
                    if abs(lora_str - self.params.get(key, -1)) > 0.02:
                        self.engine_obj.set_trt_lora_strength(desc.id, lora_str)

            hint_str = self.midi_knobs.get_param("hint_strength") if self.use_midi else 1.0
            if abs(hint_str - last_hint_str) > 0.02:
                last_hint_str = hint_str
                self._update_hint_strength(hint_str)

            noise_sharing = self.midi_knobs.get_param("noise_share") if self.use_midi else 0.0

            source_lat = None
            if feedback > 0.0 and last_latent is not None:
                src_tensor = self.stream.source.latent.tensor
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

            effective_seed = None if noise_sharing > 0.01 else seed

            ode_curve = _curve_from_spec(raw.get("ode_noise_curve"), src_T)
            if ode_curve is None:
                ode_noise_val = self.midi_knobs.get_param("ode_noise") if self.use_midi else 0.0
                ode_curve = torch.full((1, src_T, 1), ode_noise_val) if ode_noise_val > 0.01 else None

            # Source lock: x0_target_curve from client overrides the legacy
            # scalar x0_target knob. When the curve is present, we blend each
            # frame of the denoised x0 prediction toward the source latent by
            # curve[t] (0 = free, 1 = pure source). Server requires both
            # x0_target and x0_target_curve to be set for the advanced path.
            x0_target_curve = _curve_from_spec(raw.get("x0_target_curve"), src_T)
            if x0_target_curve is not None:
                x0_tgt = Latent(tensor=self.stream.source.latent.tensor)
                x0_str = 0.0
            else:
                x0_str = self.midi_knobs.get_param("x0_target") if self.use_midi else 0.0
                x0_tgt = Latent(tensor=self.stream.source.latent.tensor) if x0_str > 0.01 else None

            velocity_curve = _curve_from_spec(raw.get("velocity_scale_curve"), src_T)
            initial_noise_curve = _curve_from_spec(raw.get("initial_noise_curve"), src_T)

            if self.use_midi:
                last_channel_gains = self._sync_channel_guidance(raw, last_channel_gains)

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            result_latent = self.stream.tick(
                denoise=denoise,
                seed=effective_seed,
                source_latent=(
                    Latent(tensor=source_lat) if source_lat is not None
                    else self.stream.source.latent
                ),
                sde_denoise_curve=sde_curve,
                ode_noise_curve=ode_curve,
                x0_target=x0_tgt,
                x0_target_strength=x0_str,
                x0_target_curve=x0_target_curve,
                shift=current_shift,
                noise_sharing=noise_sharing,
                velocity_scale=velocity_curve,
                initial_noise_curve=initial_noise_curve,
                # DCW (wavelet-domain post-step correction). Forwarded
                # every tick so toggle / mode / wavelet changes from the
                # client take effect on the next slot via pipe.set_dcw().
                # Default on — matches upstream v0.1.7.
                dcw_enabled=bool(raw.get("dcw_enabled", True)),
                dcw_mode=str(raw.get("dcw_mode", "double")),
                dcw_scaler=float(raw.get("dcw_scaler", 0.05)),
                dcw_high_scaler=float(raw.get("dcw_high_scaler", 0.02)),
                dcw_wavelet=str(raw.get("dcw_wavelet", "haar")),
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
                            prefetch = min(1.0, self.vae_window * 0.2)
                            if last_decode_pos is not None and abs(t_pos - last_decode_pos) < self.vae_window - prefetch:
                                skipped = True
                        else:
                            skipped = True

                last_latent = result.clone()

                if not skipped:
                    t1 = time.perf_counter()
                    eff_dur = self.crop_seconds if self.crop_seconds > 0 else 60.0
                    if self.vae_window > 0:
                        t_pos = self.audio_eng.position / SAMPLE_RATE
                        max_t = max(0.0, eff_dur - self.vae_window)
                        t_pos = min(t_pos, max_t)
                        last_decode_pos = t_pos
                        audio_out = self.session.decode(result_latent, t_start=t_pos, cyclic=True)
                        torch.cuda.synchronize()
                        dec_ms = (time.perf_counter() - t1) * 1000
                        win_wav = audio_out.waveform.detach().cpu().float().squeeze(0)
                        win_np = win_wav.numpy().T
                        win_start = audio_out.start_sample
                        win_end = win_start + win_np.shape[0]
                        buf = self.audio_eng.current.copy()
                        xfade = min(2400, win_np.shape[0] // 4)
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
                self.params[self.k1_name] = round(k1, 2)
                self.params["seed"] = seed
                self.params["feedback"] = round(feedback, 2)
                self.params["shift"] = round(shift_val, 2)
                if self.use_lora and self.engine_obj is not None:
                    for desc in self.engine_obj.list_trt_loras():
                        if desc.state != "enabled":
                            continue
                        key = f"lora_str_{desc.id}"
                        self.params[key] = round(raw.get(key, desc.strength), 2)
                if self.use_sde:
                    self.params["periodicity"] = round(raw.get("periodicity", 0.0), 2)
                self.params["hint_strength"] = round(hint_str, 2)
                self.params["noise_share"] = round(raw.get("noise_share", 0.0), 2)
                self.params["ode_noise"] = round(ode_noise_val, 2)
                for name, _, _ in CHANNEL_GROUPS:
                    self.params[name] = round(raw.get(name, 1.0), 2)
                for name, _ in KEYSTONE_CHANNELS:
                    self.params[name] = round(raw.get(name, 1.0), 2)
                self.params["_prompt"] = self.prompt_text[0]
