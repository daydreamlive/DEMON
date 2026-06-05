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
  session create; per-prompt re-captures afterwards go through
  ``SA3Backend.set_prompt`` (the session dispatches there).
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


def create_sa3_session(*, audio, config, model_id: str, session_id: str):
    """Build a ready-to-run sa3 :class:`StreamingSession`. See the
    module docstring for what differs from the ACE create path."""
    from acestep.streaming.audio_engine import AudioEngine
    from acestep.streaming.session import StreamingSession

    waveform = audio.waveform[:2].float()
    source_duration_s = waveform.shape[-1] / SAMPLE_RATE
    duration_s = float(config.sa3_duration_s or 0.0) or source_duration_s
    duration_s = min(duration_s, SA3_MAX_DURATION_S)
    waveform = waveform[:, : int(duration_s * SAMPLE_RATE)]

    prompt = config.prompt
    if config.prompt_b not in (None, "", prompt):
        # v1 has no A/B conditioning cache (plan Phase 3a surface);
        # loud, not silent (§3.4 spirit — config field, not a command).
        logger.warning(
            "sa3_prompt_b_ignored tags_b={!r} reason=no_ab_blend_v1",
            config.prompt_b,
        )
    steps = int(config.steps)
    depth = max(1, min(int(config.depth), SA3_MAX_PIPELINE_DEPTH))

    logger.info(
        "sa3_session_create model_id={} duration_s={:.1f} "
        "source_duration_s={:.1f} steps={} depth={}",
        model_id, duration_s, source_duration_s, steps, depth,
    )

    context = get_sa3_context(model_id)
    cond = context.prepare_cond(prompt=prompt, duration=duration_s, steps=steps)
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
        cond_pair=(None, None),      # ACE conditioning cache: unused behind the sa3 mask
        cond_pair_b=(None, None),
        prompt_text=prompt,
        prompt_text_b=prompt,
        current_depth=depth,
    )

    audio_eng = AudioEngine(src_np, SAMPLE_RATE)

    return StreamingSession(
        session_id=session_id,
        checkpoint=model_id,
        config=config,
        engine_session=None,         # ACE-only fields, neutral from here down
        stream=None,
        state=state,
        audio_eng=audio_eng,
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
        vae_window=float(config.vae_window),
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
            "source_latent_bct": source_latent,
            "duration_s": duration_s,
        },
    )
