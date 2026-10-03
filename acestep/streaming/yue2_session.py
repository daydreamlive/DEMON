"""Per-family session assembly for the yue2 backend family.

Shaped like :mod:`acestep.streaming.sa3_session`. :func:`create_yue2_session`:

* **YuE2Context, process-cached** (:func:`get_yue2_context`): one model,
  VAE and engine set per weights root for the process lifetime.
* **Conditioning, synchronously**: compose (score plan + semantic AR),
  prefill the KV bundle, then one full solve for the song anchor; a
  distinct ``prompt_b`` composes a second song at the same length. This
  is seconds of AR before ``ready``; the client shows "composing".
* **Text only**: an uploaded or synthesised source is ignored; the song
  length is the semantic stage's, capped by ``yue2_duration_s`` (at most
  100 s, the flexible NAR engine's cover). Geometry is fixed for the
  session.
* **Initial buffer = the anchor**, decoded, so audio plays from ``ready``.
* **ACE-only fields neutral**; the backend is built by the registry
  factory (``families._make_yue2``) from ``backend_init``.
"""

from __future__ import annotations

import threading
import time

from acestep.engine.obs import logger
from acestep.streaming.knobs import KnobState
from acestep.streaming.source import SAMPLE_RATE
from acestep.streaming.state import SessionState
from acestep.streaming.yue2_backend import (
    LATENT_RATE_HZ,
    YUE2_MAX_SONG_S,
    playable_seconds,
    yue2_knob_specs,
)
from acestep.streaming.yue2_recompose import Song

#: Ring depth ceiling, a performance choice. The settled ring runs one
#: solve per change, so a second slot buys no throughput: on a 5090
#: (60 s songs, TRT) depth 2 ran a 102-124 ms tick against 70 ms at
#: depth 1 and every change-to-update time grew by 40-75 %. (Depth 2
#: once also re-emitted a slot frozen at settle; the backend now drops
#: in-flight slots when it settles.)
YUE2_MAX_PIPELINE_DEPTH = 1

#: Wire-slice width: two 200 ms VAE window cores.
YUE2_VAE_WINDOW_S = 0.4

#: Song budget when the client sets no ``yue2_duration_s``.
YUE2_DEFAULT_BUDGET_S = YUE2_MAX_SONG_S

_CONTEXTS: dict = {}
_CONTEXTS_LOCK = threading.Lock()


def get_yue2_context():
    """Load-or-reuse the YuE2Context for ``DEMON_YUE2_ROOT`` /
    ``DEMON_YUE2_TRT_DIR``. The lock is held across the load: a second
    concurrent first session waits instead of loading a duplicate."""
    from acestep.engine.yue2_context import YuE2Context
    from acestep.engine.yue2_runtime import trt_dir, weights_root

    root = weights_root()
    if root is None:
        raise RuntimeError("DEMON_YUE2_ROOT is not set")
    key = (str(root), str(trt_dir()) if trt_dir() else None)
    with _CONTEXTS_LOCK:
        context = _CONTEXTS.get(key)
        if context is None:
            context = YuE2Context(root, trt_dir=trt_dir())
            _CONTEXTS[key] = context
        return context


def evict_yue2_contexts() -> int:
    """Close every cached context (server shutdown). Returns how many."""
    with _CONTEXTS_LOCK:
        contexts = list(_CONTEXTS.values())
        _CONTEXTS.clear()
    for context in contexts:
        context.close()
    return len(contexts)


def song_budget_frames(config) -> int:
    """Semantic token budget (= latent frames) for this session."""
    requested = float(config.family_config.get("yue2_duration_s") or 0.0) or YUE2_DEFAULT_BUDGET_S
    seconds = max(1.0, min(requested, YUE2_MAX_SONG_S))
    return int(round(seconds * LATENT_RATE_HZ))


def nar_backend_for(bundle, has_trt: bool) -> str:
    """Which NAR path a song runs on: "trt" inside the flexible engine
    profile, else "eager" (the TRT velocity applies the same test)."""
    from acestep.engine.yue2_trt import flexible_profile_fits

    return "trt" if has_trt and flexible_profile_fits(bundle.frames, bundle.cond_tokens) else "eager"


def compose_song(context, cleanup, *, prompt: str, prompt_b: str, lyrics: str, seed: int,
                 max_frames: int) -> dict:
    """Run the create-time conditioning; returns the songs and a timing
    breakdown (ms). A distinct ``prompt_b`` is a second composition of
    the same lyrics at song A's length (style lives in the semantics).

    Every KV bundle is registered on ``cleanup`` (an ExitStack) for
    release, so a failure later in create frees its GPU memory; the
    caller pops the stack once the session owns the bundles. The anchor
    solves and the decode hold ``context.gpu_gate``, which excludes a
    CUDA graph capture on the conditioning worker (another session's
    re-compose), as the ring does."""
    times = {}
    t0 = time.perf_counter()
    composition = context.compose(style=prompt, lyrics=lyrics, seed=seed, max_frames=max_frames)
    times.update(composition.timings_ms)
    t1 = time.perf_counter()
    bundle = context.bundle(composition)
    cleanup.callback(context.release_bundle, bundle)
    t2 = time.perf_counter()
    with context.gpu_gate:
        anchor = context.solve(bundle)
    t3 = time.perf_counter()
    with context.gpu_gate:
        initial = context.decode_full(anchor).clamp(-1, 1).float().cpu().numpy().T.copy()
    t4 = time.perf_counter()
    song = Song(bundle=bundle, anchor=anchor, tags=prompt)
    song_b = None
    if prompt_b != prompt:
        composition_b = context.compose(style=prompt_b, lyrics=lyrics, seed=seed,
                                        max_frames=bundle.frames, exact_frames=bundle.frames)
        bundle_b = context.bundle(composition_b)
        cleanup.callback(context.release_bundle, bundle_b)
        with context.gpu_gate:
            anchor_b = context.solve(bundle_b)
        song_b = Song(bundle=bundle_b, anchor=anchor_b, tags=prompt_b)
        times["song_b_ms"] = (time.perf_counter() - t4) * 1000
    times.update(
        prefill_ms=(t2 - t1) * 1000, anchor_solve_ms=(t3 - t2) * 1000,
        anchor_decode_ms=(t4 - t3) * 1000, total_ms=(time.perf_counter() - t0) * 1000,
    )
    return dict(composition=composition, song=song, song_b=song_b, bundle=bundle,
                initial_buffer=initial, timings_ms=times)


def create_yue2_session(cls, *, audio, config, checkpoint, session_id, **_unused):
    """Build a ready-to-run yue2 :class:`StreamingSession` (``cls``).
    ``audio`` (the uploaded or text-only silent source) is ignored: YuE2
    composes its own song. Accel kwargs land in ``_unused``: the NAR runs
    on TRT whenever ``DEMON_YUE2_TRT_DIR`` holds the engines and the song
    fits their profile. A failure anywhere after the first KV prefill
    releases every bundle made so far and stops the audio engine."""
    from contextlib import ExitStack

    from acestep.engine.yue2_context import DEFAULT_LYRICS

    context = get_yue2_context()
    prompt = config.prompt
    prompt_b = config.prompt_b if config.prompt_b not in (None, "") else prompt
    lyrics = config.family_config.get("yue2_lyrics") or DEFAULT_LYRICS
    max_frames = song_budget_frames(config)
    depth = max(1, min(int(config.depth), YUE2_MAX_PIPELINE_DEPTH))
    virtual_knobs = KnobState(yue2_knob_specs())
    seed = int(virtual_knobs.get_all_values().get("seed", 0))

    with ExitStack() as cleanup:
        song = compose_song(context, cleanup, prompt=prompt, prompt_b=prompt_b, lyrics=lyrics,
                            seed=seed, max_frames=max_frames)
        streaming = _assemble_session(
            cls, context, song, cleanup, config=config, checkpoint=checkpoint,
            session_id=session_id, prompt=prompt, prompt_b=prompt_b, max_frames=max_frames,
            depth=depth, virtual_knobs=virtual_knobs,
        )
        cleanup.pop_all()  # the session's backend owns the bundles now
        return streaming


def _assemble_session(cls, context, song: dict, cleanup, *, config, checkpoint, session_id,
                      prompt, prompt_b, max_frames, depth, virtual_knobs):
    """The second half of create: session state, audio engine and the
    StreamingSession around the composed song(s)."""
    from acestep.streaming.audio_engine import AudioEngine
    from acestep.streaming.session import _cleanup_create_resource

    bundle = song["bundle"]
    playable_s = playable_seconds(bundle.frames)
    nar = nar_backend_for(bundle, context.has_trt_nar)
    if bundle.truncated:
        logger.warning(
            "yue2_song_truncated frames={} budget_frames={} (the score or semantic "
            "stage hit its token budget; the song may end abruptly)",
            bundle.frames, max_frames,
        )
    logger.info(
        "yue2_session_create frames={} playable_s={:.2f} cond_tokens={} nar={} depth={} "
        "truncated={} timings_ms={}",
        bundle.frames, playable_s, bundle.cond_tokens, nar, depth, bundle.truncated,
        {k: round(v, 1) for k, v in song["timings_ms"].items()},
    )

    initial = song["initial_buffer"]
    state = SessionState(
        source=None, bpm=None, key=None, time_signature=None,
        duration=playable_s, n_channels=int(initial.shape[1]),
        playback_samples=int(initial.shape[0]),
        cond_pair=None, cond_pair_b=None,
        prompt_text=prompt, prompt_text_b=prompt_b,
        current_depth=depth,
    )
    state.params["yue2_nar"] = nar
    state.params["yue2_truncated"] = bool(bundle.truncated)
    state.params["yue2_create_ms"] = round(song["timings_ms"]["total_ms"], 1)

    audio_eng = AudioEngine(initial, SAMPLE_RATE)
    cleanup.callback(_cleanup_create_resource, "audio_engine", audio_eng.stop)
    return cls(
        session_id=session_id,
        checkpoint=checkpoint,
        config=config,
        engine_session=None,
        stream=None,
        state=state,
        audio_eng=audio_eng,
        canvas=None,
        virtual_knobs=virtual_knobs,
        engine_obj=None,
        profile_mgr=None,
        cond_negative=None,
        initial_buffer=initial,
        initial_upload_stems=None,
        initial_stem_error=None,
        initial_stem_source_mode=None,
        initial_enable_ids=[],
        lora_strengths_init={},
        lora_available=False,
        max_pipeline_depth=YUE2_MAX_PIPELINE_DEPTH,
        max_seconds=playable_s,
        walk_window=False,
        walk_window_s=0.0,
        vae_window=YUE2_VAE_WINDOW_S,
        crop_seconds=0.0,
        use_sde=False,
        use_lora=False,
        k1_name="yue2_denoise",
        backend_init={
            "context": context,
            "composition": song["composition"],
            "song": song["song"],
            "song_b": song["song_b"],
        },
    )
