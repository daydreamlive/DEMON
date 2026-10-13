"""Per-family session create path for ``live_input`` ("Audio input").

Registered in :mod:`acestep.streaming.families` as the ``LIVE_INPUT``
spec's ``create_session``. Nothing is loaded: no model, no TRT engine,
no conditioning, no GPU. A session is an audio canvas plus the event bus
the analysis taps listen on (see :mod:`acestep.streaming.live_input_backend`).

Two shapes, decided by the handshake:

* **from a clip** (an ordinary source upload): the upload is the canvas,
  at its own length and channel count (up to
  :data:`~acestep.streaming.live_input_backend.MAX_CAPTURE_S`); the client
  loops it and nothing rewrites it.
* **from nothing** (``text_only``, or an upload with no samples, which
  is what a client SDK that always sends a PCM frame sends): a silent
  mono canvas of
  ``live_input_capture_s`` seconds (default 240, clamped to 30..1800)
  that ``audio_in`` frames fill. The silent anchor the adapter synthesises
  for a text-only create is ignored; the capture length is this family's
  own config field.

The capture canvas is allocated whole at connect, so its length is paid
up front. Per minute of capture (mono, 48 kHz): 11.5 MB of float32 canvas
on the server, a 5.8 MB float16 source buffer sent to the client in the
connect-time ``ready`` (and again in the ``swap_ready`` of every new
take), and 11.5 MB of float32 once the browser decodes it. The 240 s
default is 46 MB server-side and 23 MB on the wire per take; the 1800 s
ceiling is 346 MB and 173 MB.
"""

from __future__ import annotations

import numpy as np

from acestep.engine.obs import logger
from acestep.streaming.live_input_backend import (
    MAX_CAPTURE_S,
    MODE_CAPTURE,
    MODE_CLIP,
    SAMPLE_RATE,
    LiveInputBackend,
    clamp_capture_s,
)

#: The family's model id (the value of its ``live-input`` checkpoint alias).
LIVE_INPUT_MODEL_ID = "live_input/v1"


def create_live_input_session(cls, *, audio=None, config=None, checkpoint=None,
                              session_id=None, **_unused):
    """Build a StreamingSession whose canvas is a clip or a capture."""
    from contextlib import ExitStack

    from acestep.streaming.audio_engine import AudioEngine
    from acestep.streaming.knobs import KnobState
    from acestep.streaming.session import _cleanup_create_resource
    from acestep.streaming.state import SessionState

    family_config = getattr(config, "family_config", None) or {}
    capture_s = clamp_capture_s(family_config.get("live_input_capture_s"))
    waveform = getattr(audio, "waveform", None) if audio is not None else None
    from_clip = (
        not bool(getattr(config, "text_only", False))
        and waveform is not None
        and int(waveform.shape[-1]) > 0
    )

    if from_clip:
        wf = waveform[:2, : int(MAX_CAPTURE_S * SAMPLE_RATE)]
        buf = np.ascontiguousarray(
            wf.detach().cpu().numpy().T if hasattr(wf, "detach") else np.asarray(wf).T,
            dtype=np.float32,
        )
        mode = MODE_CLIP
    else:
        # Mono: the analysers downmix anyway, and a mono capture halves the
        # canvas (and the ready buffer the client receives).
        buf = np.zeros((int(round(capture_s * SAMPLE_RATE)), 1), dtype=np.float32)
        mode = MODE_CAPTURE

    duration_s = buf.shape[0] / SAMPLE_RATE
    logger.info(
        "live_input_session_create mode={} duration_s={:.1f} channels={}",
        mode, duration_s, buf.shape[1],
    )
    state = SessionState(
        source=None,        # no PreparedSource: timbre/structure gated off
        bpm=None,
        key=None,
        time_signature=None,
        duration=duration_s,
        n_channels=int(buf.shape[1]),
        playback_samples=int(buf.shape[0]),
        cond_pair=None,
        cond_pair_b=None,
        prompt_text=getattr(config, "prompt", "") or "",
        prompt_text_b=None,
        current_depth=1,
    )

    with ExitStack() as cleanup:
        audio_eng = AudioEngine(buf, SAMPLE_RATE)
        cleanup.callback(_cleanup_create_resource, "audio_engine", audio_eng.stop)
        streaming = cls(
            session_id=session_id,
            checkpoint=checkpoint or LIVE_INPUT_MODEL_ID,
            config=config,
            engine_session=None,
            stream=None,
            state=state,
            audio_eng=audio_eng,
            canvas=None,            # write_audio gated off
            virtual_knobs=KnobState([]),
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
            max_seconds=max(duration_s, capture_s),
            walk_window=False,
            walk_window_s=0.0,
            vae_window=1.0,
            crop_seconds=0.0,
            use_sde=False,
            use_lora=False,
            k1_name="",
            backend_init={"mode": mode, "capture_s": capture_s},
        )
        cleanup.pop_all()
        return streaming


def make_live_input_backend(ss) -> LiveInputBackend:
    """The ``LIVE_INPUT`` spec's ``make_backend``: the backend reads the
    canvas shape :func:`create_live_input_session` stashed on the session."""
    init = getattr(ss, "backend_init", None)
    if not init or "mode" not in init:
        raise ValueError(
            "backend 'live_input' requires the per-family create path "
            "(acestep.streaming.live_input_session.create_live_input_session)"
        )
    return LiveInputBackend(
        audio_eng=ss.audio_eng,
        state=ss.state,
        bus=ss.bus,
        mode=init["mode"],
        capture_s=float(init["capture_s"]),
    )
