"""Per-family session assembly for the sa3 backend family.

The serving-layer create path (canonical SA3 plan Phase 3; round_3 plan
§3.5: per-family construction code in its own module). Where
:meth:`~acestep.streaming.session.StreamingSession.create` builds the
ACE stack (TRT profiles, engine ``Session``, conditioning pairs, LoRA
catalog), :func:`create_sa3_session` builds the SA3 one:

* **SA3Context, process-cached.** One loaded model (DiT + SAME +
  conditioner) per ``model_id`` for the process lifetime, shared across
  sessions — the SA3 analog of the engine-state reuse the ACE startup
  warmup exists to exploit. First session pays the load; the rest get
  it warm.
* **Source anchor.** The uploaded 48 kHz source is SAME-encoded once as
  the audio-to-audio anchor (the spike-proven continuity mechanism:
  every emit is a partial-denoise cover of this latent). The 48 → 44.1
  resample rides the spike-validated ``prepare_audio`` path inside
  ``_encode_audio_input`` — the ``(48000, waveform)`` tuple is exactly
  how ``demos/test_stream_sa3_graph.py`` feeds it.
* **Conditioning, once.** ``prepare_cond(prompt, duration)`` per
  session create — twice when ``config.prompt_b`` differs (the A/B
  crossfade pair behind ``SA3Backend.handle_set_prompt_blend``);
  per-prompt re-captures afterwards go through
  ``SA3Backend.handle_set_prompt`` (the session dispatches there).
* **Duration.** ``config.sa3_duration_s`` when set, else the uploaded
  source length; capped at the small-music 120 s window.
* **ACE-only fields neutral.** ``StreamingSession`` is constructed with
  no engine session / stream / TRT profile manager / LoRA state; every
  operation that would touch them is already capability-gated off by
  the SA3 mask (loud ``command_failed``, plan §3.4).

The backend itself is NOT built here: ``StreamingSession.__init__``
asks the family registry (``families.make_backend``), whose sa3 factory
consumes the construction payload this module stashes via
``backend_init`` — so backend selection stays in exactly one place.
"""

from __future__ import annotations

import math
import threading

import numpy as np

from acestep.engine.obs import logger
from acestep.streaming.knobs import KnobState
from acestep.streaming.sa3_backend import (
    DELIVERY_SAMPLE_RATE,
    SA3_SAMPLE_RATE,
    sa3_knob_specs,
)
from acestep.streaming.source import SAMPLE_RATE
from acestep.streaming.state import SessionState

# small-music generates at most a 120 s window (sample_size 5292032 at
# 44.1 kHz); longer sources are anchored by their first 120 s.
SA3_MAX_DURATION_S = 120.0

# Ring-buffer depth ceiling. The eager SA3 pipeline has no TRT batch
# profile to read a cap from; depth 8 is what the spike stress demo
# (demos/test_stream_sa3_graph.py) ran end-to-end on the 5090.
SA3_MAX_PIPELINE_DEPTH = 8

# Wire-slice width for SA3 sessions, seconds. The reference branch's
# web demo ran SA3 at 3.0 s (``config.get("vae_window", 3.0)``);
# ``SessionConfig.vae_window``'s 0.36 default is the ACE windowed-VAE
# geometry and must not leak in here.
SA3_VAE_WINDOW_S = 3.0

# One loaded SA3 model per model_id per process (module docstring).
# Lock held across the load on purpose: a second concurrent first
# session should wait for the shared context, not load a duplicate.
_CONTEXTS: dict = {}
_CONTEXTS_LOCK = threading.Lock()


def get_sa3_context(model_id: str):
    """Load-or-reuse the process-cached SA3Context for ``model_id``."""
    from acestep.engine.sa3_context import SA3Context

    with _CONTEXTS_LOCK:
        context = _CONTEXTS.get(model_id)
        if context is None:
            context = SA3Context(model_id)
            _CONTEXTS[model_id] = context
        return context


def _delivered_samples(n_44k: int) -> int:
    """48 kHz sample count of the backend's delivery resample for an
    ``n_44k``-sample native decode — mirrors torchaudio's
    ``ceil(new * n / orig)`` (gcd-reduced) so the audio engine buffer
    and the rendered windows agree on geometry to the sample."""
    g = math.gcd(DELIVERY_SAMPLE_RATE, SA3_SAMPLE_RATE)
    new, orig = DELIVERY_SAMPLE_RATE // g, SA3_SAMPLE_RATE // g
    return -(-new * n_44k // orig)


def _resolve_accel(value: str, component: str) -> str:
    """Map the serving layer's accel value onto SA3's two execution
    modes. SA3 has no torch.compile path, so "compile" (the ACE server
    default for throughput) degrades to eager — loudly, not silently."""
    if value == "compile":
        logger.info("sa3_accel_compile_ignored component={} using=eager", component)
        return "eager"
    return value


def create_sa3_session(
    cls, *, audio, config, checkpoint, session_id,
    decoder_backend: str = "tensorrt", vae_backend: str = "tensorrt",
    **_unused,
):
    """Build a ready-to-run sa3 :class:`StreamingSession` (``cls``). See
    the module docstring for what differs from the ACE create path.
    Signature is the ``families.SESSION_CREATORS`` contract; the
    remaining ACE-shaped kwargs (offload) land in ``_unused``.
    ``decoder_backend`` / ``vae_backend`` are the same accel params the
    ACE path takes (server ``--accel``): "tensorrt" puts the DiT and
    the SAME-L window decode on their built engines (eager fallback
    when none covers the session), "compile" degrades to eager (no SA3
    torch.compile path), "eager" is fully eager. ``checkpoint`` is the
    family model id by this point (families.resolve_checkpoint mapped
    the server alias, e.g. "sa3-small" -> "small-music")."""
    from contextlib import ExitStack

    from acestep.streaming.audio_engine import AudioEngine
    from acestep.streaming.session import _cleanup_create_resource

    dit_backend = _resolve_accel(decoder_backend, "dit")
    codec_backend = _resolve_accel(vae_backend, "codec")
    model_id = checkpoint
    context = get_sa3_context(model_id)

    waveform = audio.waveform[:2].float()
    # SA3 decodes stereo and geometry() declares 2 channels, so the
    # source anchor / initial buffer must be stereo too — upmix a mono
    # upload rather than ship a 1-channel buffer against stereo patches.
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    source_duration_s = waveform.shape[-1] / SAMPLE_RATE
    duration_s = float(config.sa3_duration_s or 0.0) or source_duration_s
    duration_s = min(duration_s, SA3_MAX_DURATION_S)
    # Land on the TRT DiT fast path when engines are built (medium):
    # a duration whose padded latent window exceeds every engine
    # profile would silently fall back to the ~5x-slower eager DiT.
    duration_s = context.clamp_duration_for_trt(duration_s, backend=dit_backend)
    waveform = waveform[:, : int(duration_s * SAMPLE_RATE)]

    prompt = config.prompt
    prompt_b = config.prompt_b if config.prompt_b not in (None, "") else prompt
    steps = int(config.steps)
    depth = max(1, min(int(config.depth), SA3_MAX_PIPELINE_DEPTH))

    logger.info(
        "sa3_session_create model_id={} duration_s={:.1f} "
        "source_duration_s={:.1f} steps={} depth={} dit_backend={} "
        "codec_backend={}",
        model_id, duration_s, source_duration_s, steps, depth,
        dit_backend, codec_backend,
    )

    cond = context.prepare_cond(prompt=prompt, duration=duration_s, steps=steps)
    # Second capture for the A/B crossfade pair (SA3Backend
    # handle_set_prompt_blend); skipped when B is absent/identical —
    # the backend then blends A against A, a no-op.
    cond_b = (
        context.prepare_cond(prompt=prompt_b, duration=duration_s, steps=steps)
        if prompt_b != prompt else None
    )
    source_latent = context.encode_source(
        (SAMPLE_RATE, waveform), cond.audio_sample_size,
    )

    # Playable geometry comes from the conditioning capture (duration +
    # SA3's padding, what the DiT actually generates), not the request.
    playable_s = cond.audio_sample_size / context.sample_rate
    n_48k = _delivered_samples(int(cond.audio_sample_size))

    # Initial client buffer: the (truncated) source at the delivery
    # rate, zero-padded out to the rendered length so the buffer the
    # runner patches into matches the backend's render geometry.
    src_np = waveform.numpy().T.copy()  # [N, C] float32
    if src_np.shape[0] < n_48k:
        src_np = np.concatenate(
            [src_np, np.zeros((n_48k - src_np.shape[0], src_np.shape[1]),
                              dtype=src_np.dtype)],
        )
    else:
        src_np = src_np[:n_48k]
    n_channels = src_np.shape[1] if src_np.ndim > 1 else 1

    virtual_knobs = KnobState(sa3_knob_specs())

    state = SessionState(
        source=None,                 # no PreparedSource: swap/timbre/structure are capability-gated off
        bpm=None,                    # nullable metadata per the capability mask (plan §3.6)
        key=None,
        time_signature=None,
        duration=playable_s,
        n_channels=n_channels,
        playback_samples=int(src_np.shape[0]),
        cond_pair=None,              # ACE conditioning cache: absent for sa3
        cond_pair_b=None,            # (None = backend owns conditioning; see
                                     # _refresh_conditioning's guard)
        prompt_text=prompt,
        prompt_text_b=prompt_b,
        current_depth=depth,
    )

    # Same transactional create shape as StreamingSession.create's ACE
    # body: per-session resources acquired before ``cls(...)`` succeeds
    # register on the stack the moment they exist; ownership transfers
    # via ``pop_all()`` only on success. The process-cached SA3Context
    # deliberately does NOT register (it outlives the session by
    # design; closing it on a failed create would break the cache for
    # every later session). ``cond`` / ``source_latent`` are plain
    # tensors with no close() — GC reclaims them, same as the ACE
    # path's conditioning.
    with ExitStack() as cleanup:
        audio_eng = AudioEngine(src_np, SAMPLE_RATE)
        cleanup.callback(
            _cleanup_create_resource,
            "audio_engine",
            audio_eng.stop,
        )
        streaming = cls(
            session_id=session_id,
            checkpoint=model_id,
            config=config,
            engine_session=None,         # ACE-only fields, neutral from here down
            stream=None,
            state=state,
            audio_eng=audio_eng,
            canvas=None,                 # write_audio/swap gated off for sa3

            virtual_knobs=virtual_knobs,
            engine_obj=None,
            profile_mgr=None,
            cond_negative=None,
            initial_buffer=src_np,
            initial_upload_stems=None,
            initial_stem_error=None,
            initial_stem_source_mode=None,
            initial_enable_ids=[],
            lora_strengths_init={},
            lora_available=False,
            max_pipeline_depth=SA3_MAX_PIPELINE_DEPTH,
            max_seconds=playable_s,
            walk_window=False,
            walk_window_s=0.0,
            vae_window=SA3_VAE_WINDOW_S,
            crop_seconds=0.0,
            use_sde=False,
            use_lora=False,
            k1_name="sa3_denoise",
            # Construction payload for families._make_sa3 (the registry
            # factory assembles SA3Backend.from_context from this + the
            # session's own attributes).
            backend_init={
                "context": context,
                "cond": cond,
                "cond_b": cond_b,
                "source_latent_bct": source_latent,
                "duration_s": duration_s,
                "dit_backend": dit_backend,
                "codec_backend": codec_backend,
            },
        )
        cleanup.pop_all()
        return streaming
