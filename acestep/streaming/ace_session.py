"""Per-family session create path for ACE-Step.

Registered in :mod:`acestep.streaming.families` as
``ACESTEP.create_session``. This is the historical default body of
``StreamingSession.create``, moved here unchanged so the shared session
module holds no family-shaped setup: TRT profile selection, the model
load through the engine :class:`~acestep.engine.session.Session`, demucs
stem extraction, source encode, conditioning capture and the LoRA
catalog are all ACE decisions.

Contract (``FamilySpec.create_session``): ``creator(cls, *, audio,
config, checkpoint, session_id, **rest) -> StreamingSession``. ``cls``
is :class:`~acestep.streaming.session.StreamingSession`; the creator
assembles its constructor arguments and returns ``cls(...)``.
"""

from __future__ import annotations

import time
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from acestep.constants import TASK_INSTRUCTIONS
from acestep.engine.canvas import SourceCanvas
from acestep.engine.obs import logger
from acestep.engine.session import Session
from acestep.engine.trt.profile_manager import TRTProfileManager
from acestep.fixtures import KNOWN_FIXTURES
from acestep.lora_metadata import load_lora_metadata, lora_scale_compatible
from acestep.nodes.types import Audio
from acestep.paths import (
    available_dreamvae_decode_engine,
    checkpoint_scale,
    checkpoints_dir,
    dreamvae_decode_engine_name,
    max_profile_duration_s,
    smallest_fitting_profile_duration_s,
)
from acestep.streaming.audio_engine import AudioEngine
from acestep.streaming.encode import encode_cond_pair
from acestep.streaming.knobs import KnobState, knob_specs
# The session module is the platform half of this seam; it imports the
# family registry, which imports THIS module lazily (inside the creator
# wrapper), so the module-level import below does not cycle.
from acestep.streaming.session import (
    MIN_PIPELINE_DEPTH,
    StemExtractFailedError,
    UnsupportedTrtCheckpointError,
    _POOL,
    _cleanup_create_resource,
    _compute_max_pipeline_depth,
    _resolve_bpm_key_source,
    extract_and_select_upload_stem,
    normalize_stem_source_mode,
    resolve_lora_reference,
    resolve_upload_stem_source_mode,
)
from acestep.streaming.source import SAMPLE_RATE
from acestep.streaming.state import SessionState


def create_acestep_session(
    cls,
    *,
    audio,
    config,
    checkpoint: str,
    session_id: str,
    decoder_backend: str = "tensorrt",
    vae_backend: str = "tensorrt",
    offload_text_encoder: bool = False,
    **_unused,
):
    """Build a ready-to-run ACE-Step :class:`StreamingSession` (``cls``).

    Raises:
        UnsupportedTrtCheckpointError: ``checkpoint`` isn't in any
            registered TRT profile family.
        EngineNotBuiltError: a required TRT engine (decoder,
            vae_encode, vae_decode, or walk-window decoder) isn't
            built on disk for the picked source duration.
        StemExtractFailedError: stem extraction failed for an
            upload whose ``stem_source_mode`` selected a stem.
    """
    waveform = audio.waveform

    use_trt = decoder_backend == "tensorrt" or vae_backend == "tensorrt"
    trt_profile_checkpoint = (
        checkpoint if decoder_backend == "tensorrt"
        else "acestep-v15-turbo"
    )

    # Cap at the largest registered TRT engine profile. Operator
    # can stretch up to the ceiling; smallest-fitting selection
    # happens below in ``profile_mgr.resolve``.
    if use_trt:
        try:
            max_seconds = max_profile_duration_s(
                checkpoint=trt_profile_checkpoint,
            )
        except ValueError as exc:
            logger.error("unsupported_trt_checkpoint error={}", exc)
            raise UnsupportedTrtCheckpointError(str(exc))
    else:
        max_seconds = max_profile_duration_s()

    waveform = waveform[:, :int(max_seconds * SAMPLE_RATE)]
    rem = waveform.shape[-1] % _POOL
    if rem:
        waveform = waveform[:, :waveform.shape[-1] - rem]
    logger.info(
        "audio_loaded duration_s={:.1f} channels={}",
        waveform.shape[1] / SAMPLE_RATE, waveform.shape[0],
    )

    use_sde = config.sde
    use_lora = config.lora
    vae_window = config.vae_window
    crop_seconds = config.crop
    depth = config.depth
    steps = config.steps
    prompt = config.prompt
    prompt_b = config.prompt_b if config.prompt_b is not None else prompt
    fast_vae = config.fast_vae
    walk_window = config.walk_window
    walk_window_s = config.walk_window_s
    fixture_name = config.fixture_name
    stem_source_mode = resolve_upload_stem_source_mode(
        fixture_name,
        normalize_stem_source_mode(config.stem_source_mode),
        known_fixtures=KNOWN_FIXTURES,
    )

    enabled_lora_ids = list(config.enabled_loras)
    lora_strengths_init: dict[str, float] = dict(config.lora_strengths)
    extra_lora_paths = list(config.lora_paths)

    audio_duration_s = waveform.shape[1] / SAMPLE_RATE

    profile_mgr: TRTProfileManager | None = None
    trt_engines: dict | None = None
    picked_dur: float | None = None
    if use_trt:
        profile_mgr = TRTProfileManager(
            decoder_backend=decoder_backend,
            vae_backend=vae_backend,
            checkpoint=trt_profile_checkpoint,
        )
        # Walk-window override: pin decoder + vae_decode at
        # walk_window_s. vae_encode is duration-independent (the
        # encode path chunks long inputs and ``resolve`` always
        # pins the smallest built encode engine), so the walk
        # profile's own engine set serves any source length —
        # resolving the full-source profile here would wrongly
        # require a bigger decoder to exist on disk.
        if walk_window and use_trt and audio_duration_s > walk_window_s + 0.1:
            trt_engines, picked_dur = profile_mgr.resolve(walk_window_s)
            logger.info(
                "walk_window_active window_s={:.0f} decoder={} vae_encode={}",
                walk_window_s,
                Path(trt_engines["decoder"]).stem,
                Path(trt_engines["vae_encode"]).stem,
            )
        else:
            trt_engines, picked_dur = profile_mgr.resolve(audio_duration_s)

        ideal_dur = smallest_fitting_profile_duration_s(
            audio_duration_s,
            checkpoint=trt_profile_checkpoint,
        )
        if picked_dur > ideal_dur:
            logger.warning(
                "trt_profile_fallback picked_dur_s={:.0f} ideal_dur_s={:.0f} "
                "audio_duration_s={:.1f} reason=ideal_profile_not_built",
                picked_dur, ideal_dur, audio_duration_s,
            )
        if decoder_backend != "tensorrt":
            trt_engines.pop("decoder", None)
        if vae_backend != "tensorrt":
            trt_engines.pop("vae_encode", None)
            trt_engines.pop("vae_decode", None)

    if fast_vae and vae_backend == "tensorrt":
        dv_path = available_dreamvae_decode_engine(picked_dur)
        if dv_path is not None:
            trt_engines["vae_decode"] = str(dv_path)
        else:
            wanted = dreamvae_decode_engine_name(int(picked_dur))
            logger.warning(
                "dreamvae_engine_missing wanted={} fallback={}",
                wanted, Path(trt_engines["vae_decode"]).stem,
            )
            fast_vae = False
    elif fast_vae:
        logger.warning(
            "fast_vae_requires_tensorrt vae_backend={} ignoring=true",
            vae_backend,
        )
        fast_vae = False

    with ExitStack() as cleanup:
        logger.info(
            "model_load_start decoder={} vae={} checkpoint={}",
            decoder_backend, vae_backend, checkpoint,
        )
        t0 = time.time()
        engine_session = Session(
            project_root=str(checkpoints_dir()),
            config_path=checkpoint,
            decoder_backend=decoder_backend,
            vae_backend=vae_backend,
            offload_text_encoder=offload_text_encoder,
            trt_engines=trt_engines,
            vae_window=vae_window,
        )
        cleanup.callback(
            _cleanup_create_resource,
            "engine_session",
            engine_session.close,
        )
        logger.info("model_loaded duration_s={:.1f}", time.time() - t0)

        if profile_mgr is not None:
            profile_mgr.bind(
                engine_session.handler._diffusion_engine,
                trt_engines, picked_dur,
            )

        engine_obj = engine_session.handler._diffusion_engine
        lora_available = bool(engine_obj and engine_obj.lora_available)
        if use_lora and not lora_available:
            logger.warning(
                "lora_engine_unavailable decoder_backend={}",
                decoder_backend,
            )
            use_lora = False

        max_pipeline_depth = _compute_max_pipeline_depth(engine_obj)
        depth = max(MIN_PIPELINE_DEPTH, min(int(depth), max_pipeline_depth))
        logger.info(
            "pipeline_depth_set depth={} max={} backend={}",
            depth, max_pipeline_depth,
            "trt" if engine_obj._trt_engine is not None else "eager",
        )

        initial_enable_ids: list[str] = []
        if use_lora:
            # Same reference resolution as the runtime enable_lora
            # path, but the backend (and its lora_compatible
            # predicate) doesn't exist until __init__ — this IS the
            # ACE-family create path, so the scale axis is applied
            # directly, mirroring ACEStepBackend.lora_compatible.
            scale = checkpoint_scale(checkpoint)
            entries = []
            for d in engine_obj.list_loras():
                md = load_lora_metadata(d.path)
                entries.append((
                    d.id,
                    md.name or d.name,
                    lora_scale_compatible(md.base_model_scale, scale),
                ))
            for lid in enabled_lora_ids:
                resolved = resolve_lora_reference(lid, entries)
                if resolved is None:
                    logger.warning("lora_id_not_in_catalog id={}", lid)
                    continue
                if resolved != lid:
                    logger.info(
                        "lora_alias_resolved requested={} id={}",
                        lid, resolved,
                    )
                    # Re-key any client-supplied strength so the
                    # first-tick enable picks it up under the
                    # canonical id.
                    if lid in lora_strengths_init:
                        lora_strengths_init.setdefault(
                            resolved, lora_strengths_init[lid],
                        )
                if resolved not in initial_enable_ids:
                    initial_enable_ids.append(resolved)
            for p in extra_lora_paths:
                pp = Path(p)
                if not pp.exists():
                    logger.warning("lora_path_missing path={}", p)
                    continue
                try:
                    lid = engine_obj.register_lora(str(pp))
                    if lid not in initial_enable_ids:
                        initial_enable_ids.append(lid)
                except Exception as e:
                    logger.exception(
                        "lora_register_failed path={} error={}", p, e,
                    )
            for lid in initial_enable_ids:
                try:
                    engine_obj.prewarm_lora(lid)
                except Exception as e:
                    logger.exception(
                        "lora_prewarm_failed id={} error={}", lid, e,
                    )
            if not initial_enable_ids:
                logger.info("lora_startup_empty reason=catalog_only")

        audio_in = Audio(waveform=waveform, sample_rate=SAMPLE_RATE)

        source, detected_bpm, detected_key, detected_time_signature = (
            _resolve_bpm_key_source(
                engine_session,
                audio_in=audio_in,
                fixture_name=fixture_name,
                samples=int(waveform.shape[1]),
                bpm_override=config.bpm,
            )
        )

        upload_stems, stem_error, source, waveform = (
            extract_and_select_upload_stem(
                waveform,
                session=engine_session,
                source=source,
                source_mode=stem_source_mode,
                fixture_name=fixture_name,
            )
        )
        if stem_error is not None and stem_source_mode != "full":
            logger.error(
                "stem_extract_failed_fatal source_mode={} error={}",
                stem_source_mode, stem_error,
            )
            raise StemExtractFailedError(
                f"Stem extraction failed: {stem_error}",
            )

        # Two-conditioning cache for the live timbre-strength slider.
        logger.info("text_encode_start variant=silence_and_self")
        cond_silence, cond_full = encode_cond_pair(
            engine_session, prompt, source.latent,
            detected_bpm, audio_duration_s,
            detected_key, detected_time_signature,
        )
        # Encode prompt B at session start so the blend slider works
        # immediately.
        if prompt_b and prompt_b != prompt:
            cond_silence_b, cond_full_b = encode_cond_pair(
                engine_session, prompt_b, source.latent,
                detected_bpm, audio_duration_s,
                detected_key, detected_time_signature,
            )
        else:
            cond_silence_b, cond_full_b = cond_silence, cond_full
        conditioning = cond_full  # default strength=1.0 == cond_full

        # Negative conditioning for the RCFG path (Residual CFG).
        cond_negative = engine_session.encode_text(
            tags="",
            instruction=TASK_INSTRUCTIONS["cover"],
            refer_latent=None,
            bpm=detected_bpm, duration=audio_duration_s,
            key=detected_key,
            time_signature=detected_time_signature,
        )

        logger.info(
            "stream_create_start steps={} pipeline_depth={}", steps, depth,
        )
        stream = engine_session.stream(
            source=source,
            conditioning=conditioning,
            steps=steps,
            shift=3.0,
            pipeline_depth=depth,
        )
        cleanup.callback(
            _cleanup_create_resource,
            "stream",
            stream.close,
        )
        logger.info("stream_handle_ready")

        # Initial buffer
        src_np = waveform.numpy().T
        if crop_seconds > 0:
            src_np = src_np[:int(crop_seconds * SAMPLE_RATE)]
        n_channels = src_np.shape[1] if src_np.ndim > 1 else 1

        _seam_fade_samples = int(0.05 * SAMPLE_RATE)
        _seam_fade_samples = min(_seam_fade_samples, len(src_np) // 4)
        if _seam_fade_samples > 0:
            if src_np.ndim == 1:
                _fade_out = np.linspace(1.0, 0.0, _seam_fade_samples).astype(src_np.dtype)
                _fade_in = np.linspace(0.0, 1.0, _seam_fade_samples).astype(src_np.dtype)
            else:
                _fade_out = np.linspace(1.0, 0.0, _seam_fade_samples).reshape(-1, 1).astype(src_np.dtype)
                _fade_in = np.linspace(0.0, 1.0, _seam_fade_samples).reshape(-1, 1).astype(src_np.dtype)
            _tail = src_np[-_seam_fade_samples:].copy()
            _head = src_np[:_seam_fade_samples].copy()
            src_np[-_seam_fade_samples:] = _tail * _fade_out + _head * _fade_in

        audio_eng = AudioEngine(src_np, SAMPLE_RATE)
        cleanup.callback(
            _cleanup_create_resource,
            "audio_engine",
            audio_eng.stop,
        )

        # Audio mirror of the source latent (the write_audio
        # substrate), resident on the latent's device so encode
        # windows slice without per-write host transfers. ~23 MB
        # for a 60 s stereo source.
        canvas = SourceCanvas(
            waveform.to(source.latent.tensor.device),
        )

        k1_name = "sde_amp" if use_sde else "denoise"
        initial_knob_ids = list(initial_enable_ids) if use_lora else []
        # Seeded from the full registry (bank knobs and raw-param knobs
        # alike), so the session snapshot's knob_values is complete from
        # t=0 — before the first client param tick and for headless /
        # MCP-only sessions.
        virtual_knobs = KnobState(knob_specs(use_sde, loras=initial_knob_ids))

        state = SessionState(
            source=source,
            bpm=detected_bpm,
            key=detected_key,
            time_signature=detected_time_signature,
            duration=audio_duration_s,
            n_channels=n_channels,
            playback_samples=int(waveform.shape[-1]),
            cond_pair=(cond_silence, cond_full),
            cond_pair_b=(cond_silence_b, cond_full_b),
            prompt_text=prompt,
            prompt_text_b=prompt_b,
            current_depth=int(depth),
        )

        streaming = cls(
            session_id=session_id,
            checkpoint=checkpoint,
            config=config,
            engine_session=engine_session,
            stream=stream,
            state=state,
            audio_eng=audio_eng,
            canvas=canvas,
            virtual_knobs=virtual_knobs,
            engine_obj=engine_obj,
            profile_mgr=profile_mgr,
            cond_negative=cond_negative,
            initial_buffer=src_np,
            initial_upload_stems=upload_stems,
            initial_stem_error=stem_error,
            initial_stem_source_mode=stem_source_mode,
            initial_enable_ids=initial_enable_ids,
            lora_strengths_init=lora_strengths_init,
            lora_available=lora_available,
            max_pipeline_depth=max_pipeline_depth,
            max_seconds=max_seconds,
            walk_window=walk_window,
            walk_window_s=walk_window_s,
            vae_window=vae_window,
            crop_seconds=crop_seconds,
            use_sde=use_sde,
            use_lora=use_lora,
            k1_name=k1_name,
        )
        cleanup.pop_all()
        return streaming
