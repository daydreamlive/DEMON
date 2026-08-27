"""Per-family session create path for MiniMax-Music3.

Registered in :mod:`acestep.streaming.families` as ``SESSION_CREATORS
["minimax"]``. Its job is everything that must happen once per connect
and cannot happen inside a tick: load (or reuse) the process-cached
model stack, and capture the composition the stream will spend the rest
of its life covering.

Two things differ from the SA3 path in ways worth stating.

First, the uploaded audio is not a source. MiniMax ships no converted
audio encoder, so there is no way to turn a user's file into a latent
this renderer understands. The upload is used only to give the audio
ring something to play before the first generation lands, and the real
anchor is the stream's own first render — "continue from your own
generation", which is the audio-conditioning path that actually works
on this checkpoint. Swap and write_audio are capability-gated off.

Second, the composition can come off disk. ``prepare_cond`` runs an
8.58B LM when it has to, but a saved capture is a plain tensor load,
and ``DEMON_MINIMAX_CAPTURE`` selects one. That is what makes the
family usable without the autoregressive stage resident — and it is why
the session degrades to a warning rather than a failure when the AR
weights are absent.
"""

from __future__ import annotations

import os

import numpy as np

from acestep.engine.obs import logger
from acestep.streaming.knobs import KnobState
from acestep.streaming.minimax_backend import (
    MINIMAX_AR_FRAME_RATE_HZ,
    MINIMAX_CHUNK_AR_FRAMES,
    MINIMAX_DEFAULT_STEPS,
    MINIMAX_LATENT_RATE_HZ,
    MINIMAX_MIN_LATENT_FRAMES,
    minimax_delivery_samples,
    minimax_knob_specs,
)
from acestep.streaming.source import SAMPLE_RATE
from acestep.streaming.state import SessionState

# The DEFAULT REQUEST, not a limit. Upstream renders in 200-AR-frame
# windows, but that is its inference chunking contract -- the DiT config
# carries no length bound, its RoPE is built for whatever arrives, and
# nothing upstream states a trained span. Longer single-pass windows do
# work (measured: 1240 frames renders coherently at 9x realtime with no
# drift across the song).
#
# Duration is ultimately decided by the autoregressive stage, which
# emits an end-of-audio token when the piece is done; ``config
# .minimax_duration_s`` overrides this request, and the session adopts
# whatever length actually comes back.
MINIMAX_DURATION_S = MINIMAX_CHUNK_AR_FRAMES / MINIMAX_AR_FRAME_RATE_HZ  # 8.0

MINIMAX_MAX_PIPELINE_DEPTH = 8

# Window rendering is pure indexing into a cached full decode here (the
# decoder is deterministic and the song is short), so this is a chunk
# size rather than a decode cost, and it can sit near the ACE value.
MINIMAX_VAE_WINDOW_S = 0.36


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

    # ``SessionConfig`` names this ``depth``; reading ``pipeline_depth``
    # off it silently returned the literal default forever, so a client
    # asking for depth 8 got 4. SA3 reads the right field.
    depth = max(1, min(int(config.depth or 4), MINIMAX_MAX_PIPELINE_DEPTH))

    # ``SessionConfig.steps`` defaults to 8, which is ACE's number. On
    # this model 8 unwarped steps is not a cheaper render, it is an
    # audibly broken one: log-mel 0.24 from the reference against 0.03
    # at the family default, with the leftover noise showing up as
    # anti-correlated stereo. Take the family floor, and let an operator
    # who explicitly asks for more keep it.
    steps = max(int(getattr(config, "steps", 0) or 0), MINIMAX_DEFAULT_STEPS)

    capture = os.environ.get("DEMON_MINIMAX_CAPTURE") or None
    # Without the AR stage the family is still usable from a capture,
    # but not from a free-text prompt. Say so at create rather than
    # failing on the first set_prompt.
    ar_policy = "absent" if capture else "offload"

    context = get_minimax_context(ar_policy=ar_policy)

    prompt = getattr(config, "prompt", "") or ""
    prompt_b = getattr(config, "prompt_b", None) or None

    # Duration is REQUESTED here and DECIDED by the autoregressive stage.
    # The LM emits an end-of-audio token when the piece is done, so asking
    # for 30 s can legitimately return 14.4 s, and a saved capture is
    # whatever length it was captured at. The session therefore adopts the
    # conditioning's own length instead of asserting a constant -- the
    # previous code raised on any capture that was not exactly 689 frames,
    # which made every duration but one unreachable.
    requested_s = float(getattr(config, "minimax_duration_s", 0.0) or 0.0)
    requested_s = requested_s or MINIMAX_DURATION_S
    cond = context.prepare_cond(
        prompt=prompt, duration_s=requested_s, capture=capture,
    )
    latent_frames = int(cond["encoder_hidden_states"].shape[1])
    if latent_frames < MINIMAX_MIN_LATENT_FRAMES:
        raise ValueError(
            f"minimax conditioning is only {latent_frames} latent frames "
            f"({latent_frames / MINIMAX_LATENT_RATE_HZ:.2f}s); the decoder "
            f"needs at least {MINIMAX_MIN_LATENT_FRAMES} to fill one render "
            "window with its guard margin"
        )
    duration_s = latent_frames / MINIMAX_LATENT_RATE_HZ

    cond_b = (
        context.prepare_cond(prompt=prompt_b, duration_s=duration_s)
        if prompt_b and not capture
        else None
    )
    if cond_b is not None and cond_b["encoder_hidden_states"].shape[1] != latent_frames:
        # Both captures ride the same ring, so they must agree on T. The
        # AR stage decides length independently per prompt, so this is a
        # real possibility rather than a defensive check.
        logger.warning(
            "minimax_prompt_b_length_mismatch dropping b: {} != {}",
            cond_b["encoder_hidden_states"].shape[1], latent_frames,
        )
        cond_b = None
    logger.info(
        "minimax_session_cond frames={} seconds={:.3f} requested={:.1f} "
        "capture={} depth={} steps={}",
        latent_frames, duration_s, requested_s,
        capture or "<generated>", depth, steps,
    )

    # The audio ring is seeded with silence at the render geometry: the
    # upload cannot condition this model, so pretending it seeds the
    # song would be a lie the first generation immediately contradicts.
    # Derived from the same function the backend renders against. Sizing
    # this independently left the ring 34 samples longer than a decode
    # ever produces, so the song's last ~0.7 ms was never written.
    n_48k = minimax_delivery_samples(latent_frames)
    src_np = np.zeros((n_48k, 2), dtype=np.float32)

    virtual_knobs = KnobState(minimax_knob_specs())

    state = SessionState(
        source=None,        # no PreparedSource: swap/timbre/structure gated off
        bpm=None,
        key=None,
        time_signature=None,
        duration=duration_s,
        n_channels=2,
        playback_samples=int(src_np.shape[0]),
        cond_pair=None,     # ACE conditioning cache: absent, backend owns cond
        cond_pair_b=None,
        prompt_text=prompt,
        prompt_text_b=prompt_b,
        current_depth=depth,
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
            max_pipeline_depth=MINIMAX_MAX_PIPELINE_DEPTH,
            max_seconds=duration_s,
            walk_window=False,
            walk_window_s=0.0,
            vae_window=MINIMAX_VAE_WINDOW_S,
            crop_seconds=0.0,
            use_sde=False,
            use_lora=False,
            k1_name="minimax_denoise",
            backend_init={
                "context": context,
                "cond": cond,
                "cond_b": cond_b,
                "source_latent_bct": None,  # adopted from the first render
                "duration_s": duration_s,
                "steps": steps,
                "dit_backend": dit_backend,
                "codec_backend": codec_backend,
            },
        )
        cleanup.pop_all()
        return streaming
