"""Per-family session create path for MiniMax-Music3.

Registered in :mod:`acestep.streaming.families` as
``SESSION_CREATORS["minimax"]``. Its job is everything that must happen
once per connect and cannot happen inside a tick: load (or reuse) the
process-cached model stack, pin the autoregressive stage to the device,
and open the AR session the stream will spend its life extending.

Three things differ from the SA3 path in ways worth stating.

**The uploaded audio is not a source.** MiniMax ships no converted audio
encoder, so there is no way to turn a user's file into a latent this
renderer understands, and the AR stage has no audio-prefix path in this
checkpoint either. The upload is ignored and the ring starts silent,
which is honest: the frontier overwrites it from t=0. Swap, write_audio,
timbre and structure are capability-gated off.

**The song shape is a rolling window, not a fixed song.** This is an
append-only family (see :mod:`acestep.streaming.minimax_backend`); the
declared duration is the tape length the frontier overwrites and the
player loops, and ``SessionConfig.minimax_duration_s`` sets it. The
piece's own length is the AR stage's business -- it ends when the LM
emits an end-of-audio token, at most 9000 frames (360 s) later.

**Create is fast and generation is slow.** There is no capture stage to
wait for: the AR session's prefill is well under a second, and the first
audio arrives once 200 AR frames plus one chunk render have happened
(~11 s at the measured 0.75x AR rate). The old create path ran the whole
composition through the 8.58B LM before returning, which meant a 30 s
request cost ~55 s of connect time.
"""

from __future__ import annotations

import os

import numpy as np

from acestep.engine.obs import logger
from acestep.streaming.knobs import KnobState
from acestep.streaming.minimax_backend import (
    AR_RESIDENT_VRAM_GB,
    DEFAULT_WINDOW_S,
    DELIVERY_SAMPLE_RATE,
    MiniMaxBackend,
    minimax_knob_specs,
)
from acestep.streaming.source import SAMPLE_RATE
from acestep.streaming.state import SessionState

# The family's step floor. ``SessionConfig.steps`` defaults to 8, which
# is ACE's number; on this model 8 unwarped steps is not a cheaper
# render but an audibly broken one (log-mel 0.24 from the reference
# against 0.03 at 16/2.0, with the leftover noise showing up as
# anti-correlated stereo). Take the floor, and let an operator who
# explicitly asks for more keep it.
from acestep.engine.minimax_render import DEFAULT_STEPS  # noqa: E402

# 25 Hz, the AR stage's ceiling. Not the DiT latent rate.
MINIMAX_MAX_AR_FRAMES = 9000


def _resolve_accel(name: str, what: str) -> str:
    """Normalize an accel request, degrading LOUDLY."""
    if name == "compile":
        logger.warning(
            "minimax_accel_degraded what={} requested=compile using=eager "
            "reason=no_compile_path", what,
        )
        return "eager"
    if name not in ("eager", "tensorrt"):
        logger.warning(
            "minimax_accel_unknown what={} requested={} using=eager", what, name,
        )
        return "eager"
    return name


def create_minimax_session(
    cls,
    *,
    audio=None,
    config=None,
    checkpoint=None,
    session_id=None,
    decoder_backend: str = "tensorrt",
    vae_backend: str = "tensorrt",
    **_unused,
):
    from contextlib import ExitStack

    from acestep.engine.minimax_context import get_minimax_context
    from acestep.streaming.audio_engine import AudioEngine
    from acestep.streaming.session import _cleanup_create_resource

    dit_backend = _resolve_accel(decoder_backend, "dit")
    codec_backend = _resolve_accel(vae_backend, "codec")

    if audio is not None and getattr(audio, "waveform", None) is not None:
        logger.info(
            "minimax_create_ignoring_source reason=append_only_family "
            "detail=no_audio_encoder_in_checkpoint",
        )

    window_s = float(getattr(config, "minimax_duration_s", 0.0) or 0.0)
    window_s = window_s or DEFAULT_WINDOW_S
    steps = max(int(getattr(config, "steps", 0) or 0), DEFAULT_STEPS)

    prompt = getattr(config, "prompt", "") or ""
    lyrics = getattr(config, "minimax_lyrics", None) or "[instrumental]"
    if getattr(config, "prompt_b", None):
        logger.warning(
            "minimax_prompt_b_ignored reason=no_ab_blend_on_ar_prefix",
        )

    # The AR stage runs continuously here, so it stays resident rather
    # than paging per capture. That is ~21 GB on top of the renderer;
    # say so at create, because the failure mode otherwise is an OOM
    # several seconds into a session.
    context = get_minimax_context(ar_policy="resident")

    # A saved capture replaces the language model entirely: the frames
    # are already written, so nothing steers the composition, but the
    # renderer, the chunk geometry and the whole frontier path run for
    # real without 21 GB resident. That is how the streaming gates run
    # on a machine that cannot hold the LM.
    capture = os.environ.get("DEMON_MINIMAX_CAPTURE") or None

    logger.info(
        "minimax_session_create window_s={:.1f} steps={} dit={} codec={} "
        "capture={} ar_vram_gb={:.0f}",
        window_s, steps, dit_backend, codec_backend,
        capture or "<live AR>", 0.0 if capture else AR_RESIDENT_VRAM_GB,
    )

    # The audio ring starts silent at the delivery geometry. The upload
    # cannot condition this model, so seeding the ring with it would be
    # a lie the first emission immediately contradicts.
    window_samples = int(round(window_s * DELIVERY_SAMPLE_RATE))
    src_np = np.zeros((window_samples, 2), dtype=np.float32)

    state = SessionState(
        source=None,        # no PreparedSource: swap/timbre/structure gated off
        bpm=None,
        key=None,
        time_signature=None,
        duration=window_s,
        n_channels=2,
        playback_samples=window_samples,
        cond_pair=None,     # ACE conditioning cache: absent, backend owns cond
        cond_pair_b=None,
        prompt_text=prompt,
        prompt_text_b=None,
        current_depth=1,    # no ring: there is nothing to pipeline
    )

    with ExitStack() as cleanup:
        audio_eng = AudioEngine(src_np, SAMPLE_RATE)
        cleanup.callback(
            _cleanup_create_resource, "audio_engine", audio_eng.stop,
        )
        streaming = cls(
            session_id=session_id,
            checkpoint=checkpoint or "MiniMaxAI/MiniMax-Music3",
            config=config,
            engine_session=None,      # ACE-only fields, neutral from here down
            stream=None,
            state=state,
            audio_eng=audio_eng,
            canvas=None,              # write_audio/swap gated off
            virtual_knobs=KnobState(minimax_knob_specs()),
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
            max_pipeline_depth=1,
            max_seconds=window_s,
            walk_window=False,
            walk_window_s=0.0,
            vae_window=1.0,
            crop_seconds=0.0,
            use_sde=False,
            use_lora=False,
            k1_name="minimax_temperature",
            backend_init={
                "context": context,
                "prompt": prompt,
                "lyrics": lyrics,
                "window_s": window_s,
                "steps": steps,
                "capture": capture,
                "dit_backend": dit_backend,
                "codec_backend": codec_backend,
            },
        )
        cleanup.pop_all()
        return streaming


def make_minimax_backend(ss) -> MiniMaxBackend:
    """``families.FAMILIES["minimax"]``: assemble the backend from the
    payload :func:`create_minimax_session` stashed on the session."""
    init = getattr(ss, "backend_init", None)
    if not init or "context" not in init:
        raise ValueError(
            "backend 'minimax' requires the per-family create path "
            "(acestep.streaming.minimax_session.create_minimax_session) "
            "to stash its construction payload"
        )
    context = init["context"]
    capture = init.get("capture")

    if capture:
        from acestep.engine.minimax_ar import load_replay_stream
        from acestep.engine.minimax_render import (
            CARRY_LATENT_FRAMES,
            CHUNK_AR_FRAMES,
            MiniMaxChunkRenderer,
        )

        renderer = MiniMaxChunkRenderer(
            context.make_dit(
                latent_frames=context.chunk_latent_frames,
                backend=init.get("dit_backend", "eager"),
            ),
            context.condition_encoder,
            device=context.device,
            dtype=context.dtype,
            chunk_ar_frames=CHUNK_AR_FRAMES,
            carry_latent_frames=CARRY_LATENT_FRAMES,
            latent_channels=context.latent_channels,
        )
        # Paced at the live stage's measured rate so the frontier
        # bookkeeping sees the same arrival pattern it will in
        # production. A replay that dumps every frame on the first tick
        # exercises none of it.
        stream = load_replay_stream(capture, rate_x_realtime=0.75)
        logger.info(
            "minimax_replay_capture path={} frames={}",
            capture, stream.max_frames,
        )
        return MiniMaxBackend(
            ar_stream=stream,
            renderer=renderer,
            codec=context.make_codec(backend=init.get("codec_backend", "eager")),
            knob_state=ss.virtual_knobs,
            state=ss.state,
            context=context,
            window_s=float(init["window_s"]),
            steps=int(init["steps"]),
        )

    return MiniMaxBackend.from_context(
        context,
        prompt=init["prompt"],
        lyrics=init.get("lyrics", ""),
        knob_state=ss.virtual_knobs,
        state=ss.state,
        window_s=float(init["window_s"]),
        steps=int(init["steps"]),
        max_ar_frames=MINIMAX_MAX_AR_FRAMES,
        dit_backend=init.get("dit_backend", "eager"),
        codec_backend=init.get("codec_backend", "eager"),
    )
