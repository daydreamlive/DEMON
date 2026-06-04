"""StreamingSession: the transport-agnostic streaming generative session.

The session class owns:

- **Per-connect setup** (:meth:`create`): TRT profile resolve, ``Session``
  build, LoRA catalog + prewarm, BPM/key/source resolve, optional stem
  extraction, two-conditioning cache, stream build, audio engine + knob
  state + ``SessionState`` construction.
- **Operations** (typed methods like :meth:`set_prompt`, :meth:`enable_lora`,
  :meth:`swap_source`, etc.). Each takes plain Python value objects (no
  bytes, no in-process tensors across the API boundary except where the
  underlying engine API requires it). Two operations are origin-aware:
  :meth:`set_knobs` and :meth:`set_prompt_blend` echo instead of applying
  when ``origin=CommandOrigin.EXTERNAL``, so the primary transport's UI
  layer keeps owning its smoothing tweens.
- **Runner lifecycle** (:meth:`run`): construct the PipelineRunner, drive
  it until ``state.running`` flips False, tear down GPU state, close
  the event bus.
- **Event bus** (``self.bus``): typed events the runner thread and the
  operation methods publish. Transport adapters subscribe and serialize.

The session knows nothing about JSON, WebSockets, ``send_lock``, zstd
contexts, or ``client_mirror`` deltas. Init failures raise typed
exceptions (:class:`UnsupportedTrtCheckpointError`,
:class:`StemExtractFailedError`, ``EngineNotBuiltError`` re-raised from
``acestep.paths``); adapters translate them to the wire's error frames.

Thread model: operations are typically called on a dispatcher thread
inside the adapter, while ``run()`` drives the runner thread. State
mutations that need atomicity take ``state._lock``; single-field
reads/writes rely on CPython GIL atomicity. GPU-bound ops that must
serialize with ticks (``enable_lora``/``disable_lora``/``set_depth``/
``swap_source``) post to ``state.pending_*`` and drain inside
``apply_pending`` which the runner calls from ``before_tick``.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import ExitStack
import os
import time
from pathlib import Path

import numpy as np
import torch

from acestep.constants import TASK_INSTRUCTIONS
from acestep.audio_clips import audio_clip_stems
from acestep.engine.obs import logger
from acestep.engine.session import PreparedSource, Session
from acestep.engine.trt.profile_manager import (
    TRTProfileLoadError,
    TRTProfileManager,
)
from acestep.fixtures import KNOWN_FIXTURES, audio_fixture
from acestep.lora_metadata import load_lora_metadata
from acestep.nodes.interpolation import INTERP_METHOD_NAMES
from acestep.nodes.types import Audio, Latent
from acestep.paths import (
    EngineNotBuiltError,
    available_dreamvae_decode_engine,
    checkpoints_dir,
    dreamvae_decode_engine_name,
    max_profile_duration_s,
    smallest_fitting_profile_duration_s,
)

from acestep.streaming.audio_engine import AudioEngine
from acestep.streaming.commands import CommandOrigin
from acestep.streaming.config import SessionConfig
from acestep.streaming.encode import blend_for_strength, encode_cond_pair
from acestep.streaming.events import (
    AudioReady,
    DepthApplied,
    EventBus,
    LoraCatalogUpdate,
    ParamsEcho,
    PromptApplied,
    PromptBlendEcho,
    StructureCleared,
    StructureFailed,
    StructureSet,
    SwapFailed,
    SwapReady,
    TimbreCleared,
    TimbreFailed,
    TimbreSet,
)
from acestep.streaming.knobs import KnobDef, KnobState, build_banks
from acestep.streaming.pipeline_runner import PipelineRunner
from acestep.streaming.source import (
    _normalize_time_signature,
    _resolve_bpm_key_source,
    _try_load_sidecar,
    SAMPLE_RATE,
)
from acestep.streaming.state import SessionState
from acestep.streaming.stems import (
    extract_upload_stems,
    normalize_stem_source_mode,
    resolve_upload_stem_source_mode,
)


# ---------------------------------------------------------------------------
# Pipeline depth bounds + idle pause threshold
# ---------------------------------------------------------------------------

# Hard floor for the StreamPipeline ring buffer. <1 makes the buffer
# empty and nothing ticks. The TRT cap is read from the loaded engine;
# the eager / compile cap is fixed.
MIN_PIPELINE_DEPTH = 1
EAGER_MAX_PIPELINE_DEPTH = 4

# Live-switchable interpolation. Each of the four blend paths can walk a
# straight average ("linear") or the per-frame geodesic ("slerp"); the
# dispatcher flips them via set_interp_method so an operator can A/B the
# two without restarting the session. ``prompt``/``timbre`` drive the
# cached conditioning recompute; ``structure``/``feedback`` are read live
# by the runner each tick. The method names are owned by
# acestep.nodes.interpolation (one source of truth shared with the
# ConditioningBlend / LatentBlend nodes).
INTERP_PATHS = ("prompt", "timbre", "structure", "feedback")
INTERP_METHODS = INTERP_METHOD_NAMES

# Idle GPU pause threshold. After this many seconds with no incoming
# WS or control-bus message, the runner stops invoking the DiT each
# tick. The audio engine keeps serving from its existing buffer (which
# the walk_window LoRA designs to loop cleanly at walk_window_s), so
# audio continues uninterrupted while the GPU idles. Any incoming
# message resets the timer immediately; the next loop iteration
# resumes a normal tick. Set to 0 to disable the pause entirely.
IDLE_PAUSE_S = float(os.environ.get("DEMON_IDLE_PAUSE_S", "20"))

# Sample-count alignment quantum for the source waveform. The vae_encode
# graph builds latents in 5-frame groups at 48 kHz / 25 fps; sources
# must be trimmed to a multiple of this length or the encoder rejects.
_POOL = 1920 * 5


def _compute_max_pipeline_depth(diffusion_engine) -> int:
    """Largest ``pipeline_depth`` the loaded backend can serve."""
    trt_engine = getattr(diffusion_engine, "_trt_engine", None)
    if trt_engine is None:
        return EAGER_MAX_PIPELINE_DEPTH
    try:
        _, _, max_shape = trt_engine.get_tensor_profile_shape(
            "hidden_states", 0,
        )
        return max(MIN_PIPELINE_DEPTH, int(max_shape[0]))
    except Exception as exc:
        logger.warning(
            "trt_batch_cap_unreadable error={!r} fallback={}",
            exc, EAGER_MAX_PIPELINE_DEPTH,
        )
        return EAGER_MAX_PIPELINE_DEPTH


def _cleanup_create_resource(name: str, close_fn: Callable[[], None]) -> None:
    """Run a startup cleanup callback without masking the original failure."""
    try:
        close_fn()
    except Exception as exc:
        logger.warning(
            "streaming_create_cleanup_raised resource={} error={}",
            name, exc,
        )


# ---------------------------------------------------------------------------
# Typed init-time errors
# ---------------------------------------------------------------------------


class UnsupportedTrtCheckpointError(Exception):
    """``checkpoint`` cannot be served by any registered TRT engine
    profile family."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class StemExtractFailedError(Exception):
    """Mel-Band RoFormer stem extraction failed for an upload whose
    ``stem_source_mode`` selected one of the stems as the source."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


# ---------------------------------------------------------------------------
# Upload stem extraction + selection
# ---------------------------------------------------------------------------


def extract_and_select_upload_stem(
    waveform: torch.Tensor,
    *,
    session: Session,
    source: PreparedSource,
    source_mode: str | None,
    fixture_name: str | None = None,
    log_context: str = "",
) -> tuple[dict[str, torch.Tensor] | None, str | None, PreparedSource, torch.Tensor]:
    """Run Mel-Band RoFormer and (when requested) substitute the chosen
    stem as the inference source. Returns ``(stems, error, source, wf)``.

    Reused by initial setup AND by the swap path inside the session, so
    cache lookup / encode fallback / rollback semantics stay in one
    place.
    """
    if source_mode is None:
        return None, None, source, waveform

    logger.info(
        "stems_extract_start source_mode={} context={}",
        source_mode, log_context or None,
    )
    try:
        upload_stems = (
            audio_clip_stems(
                fixture_name,
                waveform=waveform,
                sample_rate=SAMPLE_RATE,
            )
            if fixture_name else None
        )
        if upload_stems is not None:
            logger.info(
                "stems_cache_hit fixture_name={} source_mode={} context={}",
                fixture_name, source_mode, log_context or None,
            )
        else:
            upload_stems = extract_upload_stems(
                waveform=waveform,
                device=session.handler.device,
                backend_sample_rate=SAMPLE_RATE,
            )
        if source_mode == "full":
            return upload_stems, None, source, waveform

        selected_wf = upload_stems[source_mode]
        sc = _try_load_sidecar(
            fixture_name,
            samples=int(selected_wf.shape[-1]),
            source_mode=source_mode,
        )
        if sc is not None:
            device = session.handler.device
            dtype = session.handler.dtype
            selected_source = PreparedSource(
                latent=Latent(tensor=sc.latent.to(device, dtype).contiguous()),
                context_latent=Latent(
                    tensor=sc.context_latent.to(device, dtype).contiguous(),
                ),
            )
            logger.info(
                "stem_source_sidecar_hit fixture_name={} source_mode={} context={}",
                fixture_name, source_mode, log_context or None,
            )
        else:
            selected_audio = Audio(waveform=selected_wf, sample_rate=SAMPLE_RATE)
            logger.info(
                "stem_prepare_source source_mode={} context={}",
                source_mode, log_context or None,
            )
            selected_source = session.prepare_source(selected_audio)
        return upload_stems, None, selected_source, selected_wf
    except Exception as exc:
        logger.exception(
            "stem_extract_failed context={} error={}",
            log_context or None, exc,
        )
        return None, str(exc), source, waveform


# ---------------------------------------------------------------------------
# StreamingSession
# ---------------------------------------------------------------------------


class StreamingSession:
    """Transport-agnostic streaming generative session.

    Construct with :meth:`create`. Drive with :meth:`run`. Operate via
    typed methods (:meth:`set_prompt`, :meth:`enable_lora`, ...).

    Threading model (today):

    - LoRA enable/disable, depth retune, and source swap queue onto
      ``state.pending_*`` and drain inside ``apply_pending`` which the
      runner calls from ``before_tick``. Cheap from any thread.
    - Conditioning, timbre, structure, and prompt-blend mutations run
      ``encode_cond_pair`` / ``encode_audio`` on the caller thread and
      serialize against each other and against the source-swap commit
      via ``state._lock``. Two transports calling these concurrently
      will block one until the other returns. The swap's setup work
      (TRT profile resolve, ``prepare_source``, stem extract) runs
      unlocked; only the commit phase that writes the new track's
      fields holds the lock.
    - ``set_knobs`` and ``set_prompt_blend`` echo without mutating when
      ``origin=CommandOrigin.EXTERNAL``.

    A future restructure will route every mutating method through one
    typed-command lane so non-GPU work doesn't have to block on inline
    encodes; see the PR description for the deferred follow-up.
    """

    def __init__(
        self,
        *,
        session_id: str,
        checkpoint: str,
        config: SessionConfig,
        engine_session: Session,
        stream,
        state: SessionState,
        audio_eng: AudioEngine,
        virtual_knobs: KnobState,
        engine_obj,
        profile_mgr: TRTProfileManager | None,
        cond_negative,
        initial_buffer: np.ndarray,
        initial_upload_stems: dict[str, torch.Tensor] | None,
        initial_stem_error: str | None,
        initial_stem_source_mode: str | None,
        initial_enable_ids: list,
        lora_strengths_init: dict,
        lora_available: bool,
        max_pipeline_depth: int,
        max_seconds: float,
        walk_window: bool,
        walk_window_s: float,
        vae_window: float,
        crop_seconds: float,
        use_sde: bool,
        use_lora: bool,
        k1_name: str,
    ):
        self.session_id = session_id
        self.checkpoint = checkpoint
        self.config = config

        # The acestep engine handle. Named ``engine_session`` on the
        # constructor (because ``self`` is already a session) but kept
        # accessible as ``self.session`` for callers that previously
        # used ``session.session.handler.device``.
        self.session = engine_session

        self.stream = stream
        self.state = state
        self.audio_eng = audio_eng
        self.virtual_knobs = virtual_knobs
        self.engine_obj = engine_obj
        self.profile_mgr = profile_mgr
        self.cond_negative = cond_negative

        # Init artifacts the adapter consumes once for the wire ``ready``
        # frame + binary follow-up + optional stem payload.
        self.initial_buffer = initial_buffer
        self.initial_upload_stems = initial_upload_stems
        self.initial_stem_error = initial_stem_error
        self.initial_stem_source_mode = initial_stem_source_mode

        self.initial_enable_ids = initial_enable_ids
        self.lora_strengths_init = lora_strengths_init
        self.lora_available = lora_available

        self.max_pipeline_depth = max_pipeline_depth
        self.max_seconds = max_seconds
        self.walk_window = walk_window
        self.walk_window_s = walk_window_s
        self.vae_window = vae_window
        self.crop_seconds = crop_seconds
        self.use_sde = use_sde
        self.use_lora = use_lora
        self.k1_name = k1_name

        self.pool = _POOL

        # Runner cell. The runner is constructed inside ``run()`` and
        # stored here so the structure-override helpers can call
        # ``runner.mark_hint_dirty()`` / ``_rebuild_silence_latent()``
        # mid-stream. Preserves the ``runner_holder[0]`` pattern from
        # the pre-refactor code.
        self.runner_holder: list = [None]

        # Event bus: typed events the runner thread and operation
        # methods publish; transport adapters subscribe and serialize.
        self.bus = EventBus()

    # ---- Snapshot / catalog helpers -------------------------------------

    def lora_catalog_payload(self) -> list:
        """Wire-shaped LoRA catalog for the active engine. Empty list
        when LoRA isn't available on this backend."""
        if not self.lora_available:
            return []
        out = []
        for d in self.engine_obj.list_loras():
            # ``metadata`` is the full normalized record from the
            # LoRA's ``<stem>.metadata.json`` sidecar (falling back to
            # a synthesized record from ``.trigger.txt``, or a sparse
            # one with id/name only when neither exists).
            metadata = load_lora_metadata(d.path).to_wire()
            out.append({
                "id": d.id,
                "name": metadata.get("name") or d.name,
                "path": d.path,
                "state": d.state,
                "strength": d.strength,
                "materialized_bytes": d.materialized_bytes,
                "metadata": metadata,
            })
        return out

    def snapshot(self) -> dict:
        """JSON-serialisable snapshot of the session's current state.

        Shape matches the demo's previous ``snapshot_session`` closure
        so the HTTP control bus returns the same ``GET /sessions/<id>``
        body it did before the refactor.
        """
        state = self.state
        return {
            "id": self.session_id,
            "prompt": state.prompt_text,
            "prompt_b": state.prompt_text_b,
            "prompt_blend": state.prompt_blend,
            "duration": state.duration,
            "bpm": state.bpm,
            "key": state.key,
            "time_signature": state.time_signature,
            "fixture_name": self.config.fixture_name,
            "timbre_name": state.timbre_name,
            "timbre_strength": state.timbre_strength,
            "structure_name": state.struct_name,
            "interp_prompt": state.interp_prompt,
            "interp_timbre": state.interp_timbre,
            "interp_structure": state.interp_structure,
            "interp_feedback": state.interp_feedback,
            "lora_catalog": self.lora_catalog_payload(),
            "knob_values": self.virtual_knobs.get_all_values(),
            "channels": state.n_channels,
            "sample_rate": SAMPLE_RATE,
        }

    # ---- Runner lifecycle ----------------------------------------------

    def run_until(self, seconds: float) -> None:
        """Drive the runner for up to ``seconds`` wall-clock seconds,
        then tear down. Used by the startup warmup to pay one-time
        engine costs before real traffic arrives.

        A daemon watchdog thread flips ``state.running`` to False after
        the deadline; :meth:`run` then observes and exits via its
        normal teardown path. Failures inside the runner still raise
        through ``run`` and the watchdog will be reaped by process
        exit (daemon=True)."""
        import threading

        def _watchdog():
            time.sleep(float(seconds))
            self.state.running = False

        threading.Thread(
            target=_watchdog, name="streaming-run-until-watchdog",
            daemon=True,
        ).start()
        self.run()

    def run(self) -> None:
        """Construct the PipelineRunner, drive it until
        ``state.running`` flips False, and tear down GPU state + the
        event bus.
        """
        try:
            runner = PipelineRunner(
                self.session, self.stream, self.audio_eng,
                state=self.state,
                idle_threshold_s=IDLE_PAUSE_S,
                use_midi=True,  # always "MIDI" mode; KnobState provides values
                use_sde=self.use_sde, use_lora=self.use_lora,
                midi_knobs=self.virtual_knobs,
                engine_obj=self.engine_obj,
                vae_window=self.vae_window, crop_seconds=self.crop_seconds,
                k1_name=self.k1_name, seed=1528, skip_threshold=5e-4,
                on_audio_ready=self._on_audio_ready,
                before_tick=self.apply_pending,
                walk_window=self.walk_window,
                walk_window_s=self.walk_window_s,
                neg_conditioning=self.cond_negative,
                lead_floor_s=self.config.lead_floor_s,
                lead_ceiling_s=self.config.lead_ceiling_s,
                lead_release_tau_s=self.config.lead_release_tau_s,
            )
            self.runner_holder[0] = runner
            logger.info("pipeline_running")
            try:
                runner.run()
            except Exception as exc:
                logger.opt(exception=True).error("pipeline_error error={}", exc)
        finally:
            self.close()

    def close(self) -> None:
        """Tear down GPU state even when a session exits before ``run``."""
        # Order matters: stream.close() drops the StreamPipeline's
        # references into the engine before session.close()
        # actually destroys the engine + ModelContext.
        # session.close() ends with gc.collect() + cuda.empty_cache().
        self.state.running = False
        try:
            self.bus.close()
        except Exception as exc:
            logger.warning("bus_close_raised error={}", exc)
        try:
            self.audio_eng.stop()
        except Exception as exc:
            logger.warning("audio_engine_stop_raised error={}", exc)
        try:
            self.stream.close()
        except Exception as exc:
            logger.warning("stream_close_raised error={}", exc)
        try:
            self.session.close()
        except Exception as exc:
            logger.warning("session_close_raised error={}", exc)

    def _on_audio_ready(self, wav_np, win_start=None, win_end=None):
        """Runner callback. Mutates ``audio_eng`` for full-buffer
        decodes (mirroring the pre-refactor on_audio_ready), then
        publishes a single :class:`AudioReady` event.

        Two call shapes:
          * Windowed (``win_start is not None``): ``wav_np`` is the
            patched window region. The runner has already written it
            into ``audio_eng`` via ``patch_window``, so we skip
            ``audio_eng.swap`` (eliminates the full-buffer
            ``self.current.copy()`` that used to fire on every
            windowed decode — ~23 MB / call at 60 s buffer).
          * Full-buffer (``win_start is None``): ``wav_np`` is the
            whole new buffer and we route it through
            ``audio_eng.swap`` for the global crossfade.

        The audio array passed in the event is the same numpy array
        the runner produced; subscribers must treat it as immutable.
        """
        state = self.state
        if win_start is not None:
            ss = int(win_start)
            se = ss + len(wav_np)
        else:
            self.audio_eng.swap(wav_np)
            ss = 0
            se = len(wav_np)

        params_snapshot = dict(state.params)
        self.bus.publish(AudioReady(
            audio=wav_np,
            start_sample=ss,
            num_samples=int(se - ss),
            channels=state.n_channels,
            tick_ms=float(params_snapshot.get("tick_ms", 0) or 0),
            dec_ms=float(params_snapshot.get("dec_ms", 0) or 0),
            num_gens=int(params_snapshot.get("num_gens", 0) or 0),
            params=params_snapshot,
        ))

    # ---- Pending drain (runs inside before_tick) -----------------------

    def apply_pending(self) -> None:
        """Drain LoRA, swap, and depth pending queues. Called by the
        runner from ``before_tick`` so GPU mutations serialize with
        the streaming pipeline."""
        self._apply_lora_pending()
        self._apply_swap_if_pending()
        self._apply_depth_pending()

    def _apply_lora_pending(self) -> None:
        if not self.lora_available:
            return
        state = self.state
        with state._lock:
            local_disable = state.pending_disable[:]
            local_enable = state.pending_enable[:]
            state.pending_disable.clear()
            state.pending_enable.clear()
        if not local_disable and not local_enable:
            return
        for lid in local_disable:
            try:
                self.engine_obj.disable_lora(lid)
                self.virtual_knobs.remove_knob(f"lora_str_{lid}")
                logger.info("lora_disabled id={}", lid)
            except Exception as e:
                logger.exception("lora_disable_failed id={} error={}", lid, e)
        for lid, strength in local_enable:
            try:
                self.engine_obj.enable_lora(lid, strength=strength)
                logger.info(
                    "lora_enabled id={} strength={}",
                    lid, strength,
                )
                # Allocate a knob slot so set_lora_strength can be
                # driven by the client's params dict. Default the slot
                # to the strength we just enabled at so the runner's
                # slider-delta check (set_lora_strength only when the
                # new value differs by > 0.02) doesn't fire a
                # redundant refit on tick 1.
                self.virtual_knobs.add_knob(
                    f"lora_str_{lid}",
                    KnobDef(
                        default=float(strength) if strength is not None else 0.0,
                        sensitivity=2.0, max_val=2.0,
                    ),
                )
            except Exception as e:
                logger.exception("lora_enable_failed id={} error={}", lid, e)
        # Publish the refreshed catalog. No automatic re-encode here.
        # With WYSIWYG prompts the trigger word lives in the visible
        # promptA/promptB text; the client's visible-prepend logic
        # mutates the prompt on toggle and sends a normal
        # prompt-update message.
        self.bus.publish(LoraCatalogUpdate(catalog=self.lora_catalog_payload()))

    def _apply_depth_pending(self) -> None:
        state = self.state
        with state._lock:
            target = state.pending_depth
            state.pending_depth = None
        if target is None or target == state.current_depth:
            return
        pipe = self.stream.pipeline
        if pipe is None:
            # First tick hasn't built the pipeline yet — re-queue and
            # try again next iteration.
            with state._lock:
                if state.pending_depth is None:
                    state.pending_depth = target
            return
        try:
            pipe.set_depth(target)
            state.current_depth = pipe.depth
            logger.info("pipeline_depth_applied depth={}", pipe.depth)
        except Exception as exc:
            logger.exception(
                "set_depth_failed target={} error={}", target, exc,
            )
            return
        self.bus.publish(DepthApplied(value=state.current_depth))

    def _apply_swap_if_pending(self) -> None:
        state = self.state
        with state._lock:
            new_wf = state.swap_pending.get("waveform")
            tags = state.swap_pending.get("tags")
            requested_key = state.swap_pending.get("key")
            requested_time_sig = state.swap_pending.get("time_signature")
            new_fixture_name = state.swap_pending.get("fixture_name")
            new_stem_source_mode = resolve_upload_stem_source_mode(
                new_fixture_name,
                state.swap_pending.get("stem_source_mode"),
                known_fixtures=KNOWN_FIXTURES,
            )
            if new_wf is None:
                return
            state.swap_pending["waveform"] = None
            state.swap_pending["tags"] = None
            state.swap_pending["key"] = None
            state.swap_pending["time_signature"] = None
            state.swap_pending["fixture_name"] = None
            state.swap_pending["stem_source_mode"] = None

        # Initialized to None so the finally below can None-guard
        # cleanly in the (rare) case an exception fires between the
        # start of the try and the contextualize bind.
        _swap_ctx = None
        try:
            # Cap at the same ceiling the initial upload used so swaps
            # take advantage of every built engine profile, not a
            # stale 60 s default.
            new_wf = new_wf[:, :int(self.max_seconds * SAMPLE_RATE)]
            rem = new_wf.shape[-1] % self.pool
            if rem:
                new_wf = new_wf[:, :new_wf.shape[-1] - rem]
            new_audio_duration_s = new_wf.shape[1] / SAMPLE_RATE
            # Bind the *new* track on top of the session-scoped
            # binding so any error during the swap body (VAE encode,
            # profile mgmt, prepare_source) carries the track the user
            # *tried* to swap to. Scoped to the swap body only.
            _swap_ctx = logger.contextualize(
                fixture_name=new_fixture_name or None,
                audio_duration_s=round(new_audio_duration_s, 2),
            )
            _swap_ctx.__enter__()
            logger.info(
                "source_swap_start duration_s={:.1f} channels={} "
                "fixture_name={} tags={!r}",
                new_audio_duration_s, new_wf.shape[0],
                new_fixture_name, tags,
            )

            # Profile swap (no-op when the new duration fits the same
            # profile currently loaded). Must run BEFORE prepare_source:
            # VAE-encode is the first GPU consumer and needs the new
            # vae_encode engine bound to its cache.
            #
            # Walk mode pins decoder + vae_decode at walk_window_s
            # while sizing vae_encode to the full new source.
            if self.profile_mgr is not None:
                try:
                    if self.walk_window:
                        self.profile_mgr.ensure_walk_profile(
                            walk_window_s=self.walk_window_s,
                            source_duration_s=new_audio_duration_s,
                        )
                    else:
                        self.profile_mgr.ensure_profile(new_audio_duration_s)
                except EngineNotBuiltError as exc:
                    logger.error(
                        "source_swap_aborted reason=engine_not_built error={}",
                        exc,
                    )
                    self.bus.publish(SwapFailed(
                        error=str(exc), build_command=exc.build_command,
                    ))
                    return
                except TRTProfileLoadError as exc:
                    # Workspace-alloc OOM (or other load failure) during
                    # the profile swap. Profile manager has already
                    # cleared its loaded state; the prior engines were
                    # evicted before the load failed, so this session
                    # cannot continue on this pod. Publish SwapFailed so
                    # the client sees an actionable message instead of
                    # the pipeline crashing on the next tick with a
                    # bare ``'NoneType' object has no attribute 'decode'``.
                    logger.error(
                        "source_swap_aborted reason=trt_load_failed "
                        "component={} engine={} error={}",
                        exc.component, exc.engine_path, exc.cause,
                    )
                    self.bus.publish(SwapFailed(
                        error=(
                            f"GPU memory pressure prevented loading the "
                            f"{exc.component} engine for this source "
                            f"duration. Try a shorter source clip."
                        ),
                    ))
                    return

            new_audio_in = Audio(waveform=new_wf, sample_rate=SAMPLE_RATE)
            new_source, new_bpm, new_key, new_time_sig = (
                _resolve_bpm_key_source(
                    self.session,
                    audio_in=new_audio_in,
                    fixture_name=new_fixture_name,
                    samples=int(new_wf.shape[1]),
                    key_override=requested_key,
                    time_signature_override=requested_time_sig,
                )
            )
            new_upload_stems, new_stem_error, new_source, new_wf = (
                extract_and_select_upload_stem(
                    new_wf,
                    session=self.session,
                    source=new_source,
                    source_mode=new_stem_source_mode,
                    fixture_name=new_fixture_name,
                    log_context="swap",
                )
            )
            if new_stem_error is not None and new_stem_source_mode != "full":
                self.bus.publish(SwapFailed(
                    error=f"Stem extraction failed: {new_stem_error}",
                ))
                return

            # Commit phase: every state / stream / audio_eng mutation
            # for the swap lands under ``state._lock`` so a concurrent
            # ``set_prompt`` / timbre / structure call can't half-overwrite
            # the new track's fields. Setup work above runs unlocked so
            # the dispatcher isn't blocked on prepare_source's VAE encode.
            with state._lock:
                # Use the active timbre reference if one is uploaded;
                # otherwise the new playback source's own latent.
                # Override persists across source swaps.
                self.stream.source = new_source
                state.source = new_source
                state.playback_samples = int(new_wf.shape[-1])
                tl = state.timbre_latent
                refer = tl if tl is not None else new_source.latent
                state.cond_pair = encode_cond_pair(
                    self.session, tags, refer,
                    new_bpm, new_audio_duration_s, new_key, new_time_sig,
                )
                # Carry promptB across the swap so the blend slider keeps
                # its meaning. If B was identical to A pre-swap, keep it
                # mirrored to skip a second encode pass.
                if state.prompt_text_b != state.prompt_text:
                    state.cond_pair_b = encode_cond_pair(
                        self.session, state.prompt_text_b, refer,
                        new_bpm, new_audio_duration_s, new_key, new_time_sig,
                    )
                else:
                    state.cond_pair_b = state.cond_pair
                    state.prompt_text_b = tags
                self.stream.context_latent = new_source.context_latent
                # Re-derive structure override against the new source
                # length. On failure (e.g. VAE engine couldn't fit the
                # new clip), drop the override rather than block the swap.
                if state.struct_audio is not None:
                    try:
                        self._apply_struct_override()
                    except Exception as exc:
                        logger.exception(
                            "swap_struct_override_dropped error={}", exc,
                        )
                        self._clear_struct_override()
                        self.bus.publish(StructureFailed(
                            error=f"dropped after swap: {exc}",
                        ))
                state.bpm = new_bpm
                state.key = new_key
                state.time_signature = new_time_sig
                state.duration = new_audio_duration_s
                state.prompt_text = tags
                self._refresh_conditioning()
                r = self.runner_holder[0]
                if r is not None:
                    # Source latent length may have changed; rebuild
                    # silence so _update_hint_strength's blend operands
                    # match shapes.
                    r._rebuild_silence_latent()
                    # Force a fresh hint blend on the next tick.
                    r.mark_hint_dirty()

                new_src_np = new_wf.numpy().T
                new_n_channels = new_src_np.shape[1] if new_src_np.ndim > 1 else 1
                state.n_channels = new_n_channels
                self.audio_eng.swap(new_src_np)
                self.audio_eng.position = 0
                # A loop band from the previous song is meaningless
                # against the new buffer — drop it.
                self.audio_eng.loop_band = None

            self.bus.publish(SwapReady(
                duration=len(new_src_np) / SAMPLE_RATE,
                sample_rate=SAMPLE_RATE,
                channels=new_n_channels,
                bpm=new_bpm,
                key=new_key,
                time_signature=new_time_sig,
                fixture_name=new_fixture_name,
                initial_buffer=new_src_np,
                stems=new_upload_stems,
                stem_source_mode=new_stem_source_mode if new_upload_stems is not None else None,
                stem_error=new_stem_error if new_upload_stems is None else None,
            ))
            logger.info(
                "source_swap_complete duration_s={:.1f}",
                len(new_src_np) / SAMPLE_RATE,
            )
        except Exception as exc:
            logger.opt(exception=True).error(
                "source_swap_error error={}", exc,
            )
            self.bus.publish(SwapFailed(error=str(exc)))
        finally:
            # Always pop the swap-scoped contextualize.
            if _swap_ctx is not None:
                try:
                    _swap_ctx.__exit__(None, None, None)
                except Exception:
                    pass

    # ---- Internal helpers ----------------------------------------------

    def _active_refer_latent(self):
        tl = self.state.timbre_latent
        return tl if tl is not None else self.state.source.latent

    def _refresh_conditioning(self):
        """Recompose ``stream.conditioning`` from the cached A/B pairs,
        current timbre strength, and current prompt blend."""
        state = self.state
        cs_a, cf_a = state.cond_pair
        ca = blend_for_strength(
            cs_a, cf_a, state.timbre_strength, method=state.interp_timbre,
        )
        pb = state.prompt_blend
        if pb <= 0.001:
            self.stream.conditioning = ca
            return
        cs_b, cf_b = state.cond_pair_b
        cb = blend_for_strength(
            cs_b, cf_b, state.timbre_strength, method=state.interp_timbre,
        )
        if pb >= 0.999:
            self.stream.conditioning = cb
            return
        # The A↔B crossfade defaults to slerp: linear averaging collapses
        # the conditioning norm at the midpoint, so blend≈0.5 of two
        # unrelated prompts is under-conditioned and sounds washed out.
        # Both paths are operator-switchable live (set_interp_method).
        self.stream.conditioning = blend_for_strength(
            ca, cb, pb, method=state.interp_prompt,
        )

    def _load_fixture_waveform(self, name: str) -> torch.Tensor:
        """Read a known fixture WAV from the local HF cache into a
        ``[≤2, N]`` float32 tensor. Used by the ``set_*_fixture``
        fast path so a Library pick doesn't round-trip through the
        client."""
        if name not in KNOWN_FIXTURES:
            raise ValueError(f"unknown fixture: {name}")
        # Lazy import: the byte-upload path doesn't pull soundfile.
        import soundfile as sf

        path = audio_fixture(name)
        audio_data, sr = sf.read(str(path), always_2d=True)
        if sr != SAMPLE_RATE:
            raise ValueError(
                f"fixture {name!r} sample rate {sr}, expected {SAMPLE_RATE}",
            )
        return torch.from_numpy(audio_data.T.copy()).float()[:2]

    def _apply_struct_override(self):
        """(Re)derive the override's context_latent against the
        current playback source length and replace stream.source with
        one that carries it. No-op when no override is active."""
        state = self.state
        if state.struct_audio is None:
            return
        target = state.playback_samples
        wf = state.struct_audio
        if wf.shape[-1] > target:
            wf = wf[:, :target]
        elif wf.shape[-1] < target:
            wf = torch.nn.functional.pad(wf, (0, target - wf.shape[-1]))
        # Sidecar fast path: known fixture + matching sample count =>
        # cached context_latent is exactly what prepare_source would
        # produce. Skips ~500ms of VAE+extract.
        sc = (
            _try_load_sidecar(
                state.struct_name,
                samples=int(wf.shape[-1]),
            )
            if state.struct_name else None
        )
        if sc is not None:
            device = self.session.handler.device
            dtype = self.session.handler.dtype
            state.struct_context = Latent(
                tensor=sc.context_latent.to(device, dtype).contiguous(),
            )
            logger.debug(
                "structure_override_sidecar_hit name={}",
                state.struct_name,
            )
        else:
            audio_in = Audio(waveform=wf, sample_rate=SAMPLE_RATE)
            prepared = self.session.prepare_source(audio_in)
            state.struct_context = prepared.context_latent
        # state.source keeps the unmodified playback PreparedSource so
        # clear can restore it as-is. stream.source carries the
        # overridden context_latent for the runner to read.
        self.stream.source = PreparedSource(
            latent=state.source.latent,
            context_latent=state.struct_context,
        )
        r = self.runner_holder[0]
        if r is not None:
            r.mark_hint_dirty()

    def _clear_struct_override(self):
        state = self.state
        state.struct_audio = None
        state.struct_context = None
        state.struct_name = None
        self.stream.source = state.source
        r = self.runner_holder[0]
        if r is not None:
            r.mark_hint_dirty()

    def _apply_timbre_waveform(self, t_wf: torch.Tensor, name: str) -> float:
        """Mutate timbre state for a new ref. Returns post-truncation
        duration (seconds). Rolls back to prior state and re-raises on
        any failure."""
        state = self.state
        prev_timbre_latent = state.timbre_latent
        prev_timbre_name = state.timbre_name
        prev_cond_pair = state.cond_pair
        prev_cond_pair_b = state.cond_pair_b
        prev_stream_cond = self.stream.conditioning
        try:
            cap = int(state.duration * SAMPLE_RATE)
            t_wf = t_wf[:, :cap]
            rem = t_wf.shape[-1] % self.pool
            if rem:
                t_wf = t_wf[:, :t_wf.shape[-1] - rem]
            if t_wf.shape[-1] < self.pool:
                raise ValueError("timbre clip too short")
            clip_s = t_wf.shape[-1] / SAMPLE_RATE
            sc = _try_load_sidecar(
                name, samples=int(t_wf.shape[-1]),
            )
            if sc is not None:
                device = self.session.handler.device
                dtype = self.session.handler.dtype
                timbre_latent = Latent(
                    tensor=sc.latent.to(device, dtype).contiguous(),
                )
                logger.debug("timbre_sidecar_hit name={}", name)
            else:
                timbre_audio = Audio(
                    waveform=t_wf, sample_rate=SAMPLE_RATE,
                )
                logger.debug(
                    "timbre_vae_encode_start clip_s={:.1f} channels={}",
                    clip_s, t_wf.shape[0],
                )
                timbre_latent = self.session.encode_audio(timbre_audio)
                logger.debug(
                    "timbre_vae_encode_done latent_shape={}",
                    tuple(timbre_latent.tensor.shape),
                )
            state.timbre_latent = timbre_latent
            state.timbre_name = name
            state.cond_pair = encode_cond_pair(
                self.session, state.prompt_text, timbre_latent,
                state.bpm, state.duration, state.key,
                state.time_signature,
            )
            # Re-encode B against the new timbre too.
            if state.prompt_text_b != state.prompt_text:
                state.cond_pair_b = encode_cond_pair(
                    self.session, state.prompt_text_b, timbre_latent,
                    state.bpm, state.duration, state.key,
                    state.time_signature,
                )
            else:
                state.cond_pair_b = state.cond_pair
            self._refresh_conditioning()
            return clip_s
        except Exception:
            state.timbre_latent = prev_timbre_latent
            state.timbre_name = prev_timbre_name
            state.cond_pair = prev_cond_pair
            state.cond_pair_b = prev_cond_pair_b
            self.stream.conditioning = prev_stream_cond
            raise

    def _apply_structure_waveform(self, s_wf: torch.Tensor, name: str) -> tuple[float, float]:
        """Stash a structure-ref waveform and re-derive the override's
        context_latent against the current playback length. Returns
        ``(clip_s, target_s)``. Rolls back on failure."""
        state = self.state
        s_wf = s_wf[:2]
        try:
            state.struct_audio = s_wf
            state.struct_name = name
            clip_s = s_wf.shape[-1] / SAMPLE_RATE
            target_s = state.playback_samples / SAMPLE_RATE
            self._apply_struct_override()
            return clip_s, target_s
        except Exception:
            state.struct_audio = None
            state.struct_context = None
            state.struct_name = None
            self.stream.source = state.source
            raise

    def _apply_ref(
        self,
        kind: str,
        name: str,
        waveform_fn,
        origin_label: str,
    ) -> None:
        """Shared load → apply → publish flow for
        ``set_{timbre,structure}_{source,fixture}``.

        ``waveform_fn`` returns the decoded waveform tensor (varies
        across the four entry points). ``origin_label`` is the log
        label distinguishing source-uploaded vs server-side fixture.
        """
        try:
            wf = waveform_fn()
            if kind == "timbre":
                clip_s = self._apply_timbre_waveform(wf, name)
                self.bus.publish(TimbreSet(name=name, duration=clip_s))
                extra = f"({clip_s:.1f}s)"
            else:
                clip_s, target_s = self._apply_structure_waveform(wf, name)
                self.bus.publish(StructureSet(name=name, duration=clip_s))
                extra = f"({clip_s:.1f}s, fitted to {target_s:.1f}s)"
            logger.info(
                "ref_applied kind={} origin={} name={} detail={}",
                kind, origin_label, name, extra,
            )
        except Exception as exc:
            logger.opt(exception=True).error(
                "ref_apply_failed kind={} origin={} name={} error={}",
                kind, origin_label, name, exc,
            )
            if kind == "timbre":
                self.bus.publish(TimbreFailed(error=str(exc)))
            else:
                self.bus.publish(StructureFailed(error=str(exc)))

    # ---- Public typed operations --------------------------------------
    #
    # Each method is safe to call from any thread. Origin is metadata
    # for log traceability except on the two origin-dependent verbs
    # (:meth:`set_knobs` and :meth:`set_prompt_blend`) where
    # ``CommandOrigin.EXTERNAL`` echoes instead of applying.
    #
    # Activity-ts gating: every method bumps ``state.last_activity_ts``
    # so the idle pause re-arms on any operation. :meth:`set_knobs`
    # additionally gates on the raw-dict differing from the previous
    # one (the 125 Hz heartbeat would otherwise defeat the pause).

    def set_knobs(
        self,
        raw: dict,
        playback_pos: float = 0.0,
        *,
        origin: CommandOrigin = CommandOrigin.PRIMARY,
    ) -> None:
        """Apply or echo a knob update. ``raw`` is the unfiltered
        wire dict; values land in ``virtual_knobs`` only on PRIMARY.
        EXTERNAL emits :class:`ParamsEcho` so the primary transport's
        UI tween owns the smoothed sequence."""
        state = self.state
        raw = raw or {}
        # Activity gating: only bump on a real change. ``playback_pos``
        # advances every tick but is excluded from the diff because
        # it's a clock, not user input.
        if raw != state.last_params_raw:
            state.last_activity_ts = time.monotonic()
            # Knob→ear latency marker: log the denoise value + the playhead
            # at the instant a real knob change lands, so the lat_decode
            # line for the first slice carrying it can be correlated.
            if os.environ.get("DEMON_LAT_TRACE", "") not in ("", "0"):
                _prev = state.last_params_raw or {}
                _changed = sorted(
                    k for k in set(raw) | set(_prev)
                    if raw.get(k) != _prev.get(k)
                )
                logger.info(
                    "lat_knob origin={} playback_s={:.3f} changed={} "
                    "denoise={}",
                    origin.value, float(playback_pos), _changed,
                    raw.get("denoise"),
                )
            state.last_params_raw = dict(raw)
            logger.debug(
                "params_changed origin={} raw_keys={}",
                origin.value, sorted(raw.keys()),
            )
        if origin is CommandOrigin.EXTERNAL:
            self.bus.publish(ParamsEcho(raw=dict(raw)))
            return
        with state._lock:
            self.virtual_knobs.update(raw)
            try:
                self.audio_eng.position = int(playback_pos * SAMPLE_RATE) % max(
                    1, len(self.audio_eng.current),
                )
            except Exception:
                pass

    def set_loop_band(
        self,
        start_sec: float | None,
        end_sec: float | None,
        *,
        origin: CommandOrigin = CommandOrigin.PRIMARY,
    ) -> None:
        """Arm / move / clear the playback loop band. The worklet
        replays only ``[start_sec, end_sec]`` (wrapping end→start);
        the runner wraps its predictive decode target inside the band
        too. Null / degenerate range clears."""
        self.state.last_activity_ts = time.monotonic()
        try:
            if start_sec is None or end_sec is None or float(end_sec) - float(start_sec) <= 0.0:
                self.audio_eng.loop_band = None
            else:
                self.audio_eng.loop_band = (float(start_sec), float(end_sec))
        except (TypeError, ValueError):
            self.audio_eng.loop_band = None

    def set_prompt(
        self,
        tags: str,
        *,
        tags_b: str | None = None,
        key: str | None = None,
        time_signature: str | None = None,
        origin: CommandOrigin = CommandOrigin.PRIMARY,
    ) -> None:
        """Re-encode A (and optionally B) against the active timbre
        reference and refresh the live conditioning. Publishes
        :class:`PromptApplied`."""
        state = self.state
        state.last_activity_ts = time.monotonic()
        with state._lock:
            ts_override = _normalize_time_signature(time_signature)
            if ts_override is not None:
                state.time_signature = ts_override
            refer = self._active_refer_latent()
            key_used = key or state.key
            logger.info(
                "prompt_set origin={} tags={!r} tags_b={!r} key={} time_signature={}",
                origin.value, tags, tags_b, key_used, state.time_signature,
            )
            state.cond_pair = encode_cond_pair(
                self.session, tags, refer, state.bpm, state.duration,
                key_used, state.time_signature,
            )
            state.prompt_text = tags
            if tags_b and tags_b != tags:
                state.cond_pair_b = encode_cond_pair(
                    self.session, tags_b, refer, state.bpm, state.duration,
                    key_used, state.time_signature,
                )
                state.prompt_text_b = tags_b
            else:
                state.cond_pair_b = state.cond_pair
                state.prompt_text_b = tags
            self._refresh_conditioning()
        self.bus.publish(PromptApplied(tags=tags))

    def set_prompt_blend(
        self,
        value: float,
        *,
        origin: CommandOrigin = CommandOrigin.PRIMARY,
    ) -> None:
        """Crossfade between the cached A/B prompt cond pairs by
        ``value`` ∈ [0, 1]. EXTERNAL emits :class:`PromptBlendEcho`
        only (the primary transport's UI owns the smoothed tween)."""
        self.state.last_activity_ts = time.monotonic()
        v = max(0.0, min(1.0, float(value)))
        if origin is CommandOrigin.EXTERNAL:
            self.bus.publish(PromptBlendEcho(value=v))
            return
        with self.state._lock:
            self.state.prompt_blend = v
            self._refresh_conditioning()
        logger.debug("prompt_blend_set origin={} value={:.3f}", origin.value, v)

    def set_interp_method(
        self,
        path: str,
        method: str,
        *,
        origin: CommandOrigin = CommandOrigin.PRIMARY,
    ) -> None:
        """Switch the interpolation path for one of the four live blends
        (``prompt`` / ``timbre`` / ``structure`` / ``feedback``) between
        ``"slerp"`` and ``"linear"``. Applies for both origins (the
        setting is discrete, not a smoothed slider, so there's no UI
        tween to defer to). Unknown ``path``/``method`` are ignored."""
        path = str(path)
        method = str(method)
        if path not in INTERP_PATHS or method not in INTERP_METHODS:
            logger.warning(
                "set_interp_method_ignored path={} method={}", path, method,
            )
            return
        self.state.last_activity_ts = time.monotonic()
        with self.state._lock:
            setattr(self.state, f"interp_{path}", method)
            # prompt/timbre feed the cached conditioning; recompute now so
            # the change is audible without waiting on a slider move.
            if path in ("prompt", "timbre"):
                self._refresh_conditioning()
        if path == "structure":
            # The runner only rebuilds the hint blend when it's marked
            # dirty or the strength slider moves; flipping the method has
            # to force a rebuild or the change wouldn't land until the
            # next slider touch.
            r = self.runner_holder[0]
            if r is not None:
                r.mark_hint_dirty()
        logger.info(
            "set_interp_method origin={} path={} method={}",
            origin.value, path, method,
        )

    def set_depth(
        self,
        value: int,
        *,
        origin: CommandOrigin = CommandOrigin.PRIMARY,
    ) -> None:
        """Stage a pipeline-depth retune. The runner applies it inside
        ``before_tick`` and publishes :class:`DepthApplied` with the
        clamped value."""
        self.state.last_activity_ts = time.monotonic()
        v = max(MIN_PIPELINE_DEPTH, min(int(value), self.max_pipeline_depth))
        with self.state._lock:
            self.state.pending_depth = v
        logger.info(
            "set_depth_requested origin={} value={}", origin.value, v,
        )

    def enable_lora(
        self,
        lora_id: str,
        strength: float | None = None,
        *,
        origin: CommandOrigin = CommandOrigin.PRIMARY,
    ) -> None:
        """Stage a LoRA enable. The atomic-strength contract holds:
        the next refit lands at ``strength`` in one shot (no
        first-window-without-LoRA artifact)."""
        self.state.last_activity_ts = time.monotonic()
        with self.state._lock:
            self.state.pending_enable.append((str(lora_id), strength))
        logger.info(
            "enable_lora_requested origin={} id={} strength={}",
            origin.value, lora_id, strength,
        )

    def disable_lora(
        self,
        lora_id: str,
        *,
        origin: CommandOrigin = CommandOrigin.PRIMARY,
    ) -> None:
        self.state.last_activity_ts = time.monotonic()
        with self.state._lock:
            self.state.pending_disable.append(str(lora_id))
        logger.info(
            "disable_lora_requested origin={} id={}",
            origin.value, lora_id,
        )

    def set_timbre_strength(
        self,
        value: float,
        *,
        origin: CommandOrigin = CommandOrigin.PRIMARY,
    ) -> None:
        """Lerp the cached ``(cond_silence, cond_full)`` pair's encoder
        hidden states by ``value`` ∈ [0, 1]."""
        self.state.last_activity_ts = time.monotonic()
        v = max(0.0, min(1.0, float(value)))
        with self.state._lock:
            self.state.timbre_strength = v
            self._refresh_conditioning()
        logger.debug(
            "timbre_strength_set origin={} value={:.3f}",
            origin.value, v,
        )

    def set_timbre_source(
        self,
        audio: Audio,
        name: str,
        *,
        origin: CommandOrigin = CommandOrigin.PRIMARY,
    ) -> None:
        """Upload a clip as the active timbre reference. VAE-encodes
        (or hits the fixture sidecar) and replaces cond_full."""
        self.state.last_activity_ts = time.monotonic()
        logger.info("set_timbre_source_recv origin={} name={}", origin.value, name)
        with self.state._lock:
            self._apply_ref(
                "timbre", name,
                lambda: audio.waveform[:2],
                "source",
            )

    def set_timbre_fixture(
        self,
        name: str,
        *,
        origin: CommandOrigin = CommandOrigin.PRIMARY,
    ) -> None:
        """Pick a known fixture as the active timbre reference. Loads
        from the pod's local cache; same apply path as upload."""
        self.state.last_activity_ts = time.monotonic()
        logger.info("set_timbre_fixture origin={} name={}", origin.value, name)
        with self.state._lock:
            self._apply_ref(
                "timbre", name,
                lambda: self._load_fixture_waveform(name),
                "fixture",
            )

    def clear_timbre_source(
        self,
        *,
        origin: CommandOrigin = CommandOrigin.PRIMARY,
    ) -> None:
        """Drop the active timbre reference; fall back to self-timbre
        (encode against the playback source's own latent)."""
        state = self.state
        state.last_activity_ts = time.monotonic()
        with state._lock:
            state.timbre_latent = None
            state.timbre_name = None
            refer = state.source.latent
            state.cond_pair = encode_cond_pair(
                self.session, state.prompt_text, refer,
                state.bpm, state.duration, state.key,
                state.time_signature,
            )
            if state.prompt_text_b != state.prompt_text:
                state.cond_pair_b = encode_cond_pair(
                    self.session, state.prompt_text_b, refer,
                    state.bpm, state.duration, state.key,
                    state.time_signature,
                )
            else:
                state.cond_pair_b = state.cond_pair
            self._refresh_conditioning()
        self.bus.publish(TimbreCleared())
        logger.info("timbre_cleared origin={}", origin.value)

    def set_structure_source(
        self,
        audio: Audio,
        name: str,
        *,
        origin: CommandOrigin = CommandOrigin.PRIMARY,
    ) -> None:
        """Upload a clip as the active structure (semantic-hint)
        reference. Server pads/trims to match playback length and
        re-derives the override's context_latent."""
        self.state.last_activity_ts = time.monotonic()
        logger.info(
            "set_structure_source_recv origin={} name={}",
            origin.value, name,
        )
        with self.state._lock:
            self._apply_ref(
                "structure", name,
                lambda: audio.waveform,
                "source",
            )

    def set_structure_fixture(
        self,
        name: str,
        *,
        origin: CommandOrigin = CommandOrigin.PRIMARY,
    ) -> None:
        """Pick a known fixture as the active structure reference."""
        self.state.last_activity_ts = time.monotonic()
        logger.info(
            "set_structure_fixture origin={} name={}", origin.value, name,
        )
        with self.state._lock:
            self._apply_ref(
                "structure", name,
                lambda: self._load_fixture_waveform(name),
                "fixture",
            )

    def clear_structure_source(
        self,
        *,
        origin: CommandOrigin = CommandOrigin.PRIMARY,
    ) -> None:
        """Drop the active structure reference; restore the playback
        source's own context_latent."""
        self.state.last_activity_ts = time.monotonic()
        with self.state._lock:
            self._clear_struct_override()
        self.bus.publish(StructureCleared())
        logger.info("structure_cleared origin={}", origin.value)

    def swap_source(
        self,
        audio: Audio,
        *,
        tags: str | None = None,
        key: str | None = None,
        time_signature: str | None = None,
        fixture_name: str | None = None,
        stem_source_mode: str | None = None,
        origin: CommandOrigin = CommandOrigin.PRIMARY,
    ) -> None:
        """Stage a source swap. The runner applies it inside
        ``before_tick``; publishes :class:`SwapReady` or
        :class:`SwapFailed` when the swap completes."""
        state = self.state
        state.last_activity_ts = time.monotonic()
        effective_tags = tags or state.prompt_text
        with state._lock:
            state.swap_pending["waveform"] = audio.waveform
            state.swap_pending["tags"] = effective_tags
            state.swap_pending["key"] = key
            state.swap_pending["time_signature"] = _normalize_time_signature(
                time_signature,
            )
            state.swap_pending["fixture_name"] = fixture_name
            state.swap_pending["stem_source_mode"] = normalize_stem_source_mode(
                stem_source_mode,
            )

    # ---- Constructor ---------------------------------------------------

    @classmethod
    def create(
        cls,
        *,
        audio: Audio,
        config: SessionConfig,
        checkpoint: str,
        decoder_backend: str = "tensorrt",
        vae_backend: str = "tensorrt",
        offload_text_encoder: bool = False,
        session_id: str,
    ) -> "StreamingSession":
        """Build a ready-to-run session for one connection.

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
            trt_engines, picked_dur = profile_mgr.resolve(audio_duration_s)
            # Walk-window override: pin decoder + vae_decode at
            # walk_window_s while keeping vae_encode sized to the full
            # song so the source can be encoded once at load.
            if walk_window and use_trt and audio_duration_s > walk_window_s + 0.1:
                walk_engines, walk_dur = profile_mgr.resolve(walk_window_s)
                logger.info(
                    "walk_window_active window_s={:.0f} decoder={} vae_encode={}",
                    walk_window_s,
                    Path(walk_engines["decoder"]).stem,
                    Path(trt_engines["vae_encode"]).stem,
                )
                trt_engines = {
                    "decoder": walk_engines["decoder"],
                    "vae_encode": trt_engines["vae_encode"],
                    "vae_decode": walk_engines["vae_decode"],
                }
                picked_dur = walk_dur

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
                catalog_ids = {d.id for d in engine_obj.list_loras()}
                for lid in enabled_lora_ids:
                    if lid in catalog_ids:
                        initial_enable_ids.append(lid)
                    else:
                        logger.warning("lora_id_not_in_catalog id={}", lid)
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

            k1_name = "sde_amp" if use_sde else "denoise"
            initial_knob_ids = list(initial_enable_ids) if use_lora else []
            banks = build_banks(use_sde, loras=initial_knob_ids)
            virtual_knobs = KnobState(banks)

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
