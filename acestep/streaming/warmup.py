"""Startup self-warmup that drives one synthetic session through the
streaming API at boot to pay one-time engine costs before real traffic
arrives.

Measured 2026-05-18: a cold first session on a fresh engine takes ~40s
to ``ready`` (TRT decoder-engine load ~3s + LoRA-refit manager ~7s +
Session / ModelContext / conditioning ~10s + first-tick pipeline build),
while the *second* session on the same warm engine is ~5-6s. ~30s of
the cold path is one-time-after-engine-start state that persists in the
process. Driving one synthetic default-fixture session at boot, before
the pod accepts real traffic, pays that once so every real "begin"
gets the warm path. Behaviour-neutral for real clients; the synthetic
session tears itself down via the session's own ``run`` finally.

Failure-must-not-stop-boot: any exception is caught, stamped into
``WARMUP_STATE["error"]``, and the server continues to start.
"""

from __future__ import annotations

import secrets
import time

import numpy as np
import torch

from acestep.engine.obs import logger
from acestep.fixtures import audio_fixture
from acestep.nodes.types import Audio

from acestep.streaming.config import SessionConfig
from acestep.streaming.session import StreamingSession


WARMUP_STATE: dict = {"done": False, "error": None, "seconds": None}

_WARMUP_FIXTURE = "low_fi_Gm_loop_60s_gnm.wav"  # PREFERRED_DEFAULT_FIXTURE
_WARMUP_PROMPT = "ambient electronic, warm pads"

# Maximum wall-clock budget for the warmup session. Picked at the
# upper end of measured cold-start times so the warmup completes
# under normal conditions but doesn't hang the boot if something
# wedges.
_WARMUP_BUDGET_S = 35.0

# Backend sample rate for the ACE-Step v1.5 family. Duplicated from
# ``acestep.streaming.source`` to avoid the heavier import here.
_SAMPLE_RATE = 48000


def _load_warmup_audio() -> Audio:
    """Load the warmup fixture WAV from the pod's cache and wrap it
    in an :class:`Audio` value object.

    Mirrors the upload path: stereo float32 at the backend sample
    rate, ``[channels, samples]`` layout. Falls back to a mono→stereo
    librosa load if soundfile can't read the file.
    """
    path = str(audio_fixture(_WARMUP_FIXTURE))
    try:
        import soundfile as sf
        data, _sr = sf.read(path, dtype="float32", always_2d=True)  # (n, ch)
    except Exception:
        import librosa
        mono, _sr = librosa.load(path, sr=None, mono=True)
        data = np.stack([mono, mono], axis=1).astype(np.float32)
    if data.ndim == 1:
        data = np.stack([data, data], axis=1)
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    waveform = torch.from_numpy(
        np.ascontiguousarray(data.T, dtype=np.float32),
    )[:2]
    return Audio(waveform=waveform, sample_rate=_SAMPLE_RATE)


def run_startup_warmup(
    *,
    decoder_backend: str,
    vae_backend: str,
    checkpoint: str,
    offload_text_encoder: bool,
) -> None:
    """Drive one synthetic default-fixture session at boot. Never
    raises — a failed warmup must not stop the server from serving.

    The session is created via the public :meth:`StreamingSession.create`
    surface (same path real clients take) and driven for at most
    :data:`_WARMUP_BUDGET_S` seconds via :meth:`StreamingSession.run_until`.
    """
    t0 = time.monotonic()
    warm_log = logger.bind(component="warmup")
    warm_log.info("warmup_start fixture={}", _WARMUP_FIXTURE)
    try:
        audio = _load_warmup_audio()
        cfg = SessionConfig.from_dict({
            "fixture_name": _WARMUP_FIXTURE,
            "prompt": _WARMUP_PROMPT,
            "steps": 8,
            "depth": 4,
            "lora": False,
            "enabled_loras": [],
        })
        # Mint a throwaway session_id so the warmup's log lines carry
        # a distinguishable correlation token in case the operator is
        # tailing logs during boot.
        session_id = "warmup-" + secrets.token_urlsafe(4)
        streaming = StreamingSession.create(
            audio=audio,
            config=cfg,
            checkpoint=checkpoint,
            decoder_backend=decoder_backend,
            vae_backend=vae_backend,
            offload_text_encoder=offload_text_encoder,
            session_id=session_id,
        )
        streaming.run_until(_WARMUP_BUDGET_S)
        WARMUP_STATE["done"] = True
        WARMUP_STATE["seconds"] = round(time.monotonic() - t0, 1)
        warm_log.info(
            "warmup_done elapsed_s={}", WARMUP_STATE["seconds"],
        )
    except Exception as exc:
        WARMUP_STATE["error"] = repr(exc)
        WARMUP_STATE["seconds"] = round(time.monotonic() - t0, 1)
        warm_log.opt(exception=True).error(
            "warmup_failed elapsed_s={} error={}",
            WARMUP_STATE["seconds"], exc,
        )
