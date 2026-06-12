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

import functools
import os
import threading
import time
from dataclasses import asdict
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

from acestep.engine.canvas import SourceCanvas
from acestep.streaming.audio_engine import AudioEngine
from acestep.streaming.commands import CommandOrigin
from acestep.streaming.config import SessionConfig
from acestep.streaming.encode import blend_for_strength, encode_cond_pair
from acestep.steering import CapacityError, EmptyError
from acestep.streaming.events import (
    AudioReady,
    AudioWriteFailed,
    AudioWritten,
    CommandFailed,
    DepthApplied,
    EventBus,
    LoraCatalogUpdate,
    ManualSlotCount,
    ParamsEcho,
    PromptApplied,
    PromptBlendEcho,
    SessionError,
    StructureCleared,
    StructureFailed,
    StructureSet,
    SwapFailed,
    SwapReady,
    TimbreCleared,
    TimbreFailed,
    TimbreSet,
)
from acestep.streaming.knobs import (
    KNOB_SCHEMA_VERSION,
    KnobState,
    catalog_from_specs,
    coerce_knob_values,
    knob_specs,
    lora_strength_spec,
)
from acestep.streaming.families import make_backend
from acestep.streaming.generator_backend import UnsupportedOperation
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
    stems_pending,
    wait_for_pending_stems,
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
_SAMPLES_PER_FRAME = 1920
_POOL = _SAMPLES_PER_FRAME * 5


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
    should_abort=None,
) -> tuple[dict[str, torch.Tensor] | None, str | None, PreparedSource, torch.Tensor]:
    """Run Mel-Band RoFormer and (when requested) substitute the chosen
    stem as the inference source. Returns ``(stems, error, source, wf)``.

    Reused by initial setup AND by the swap path inside the session, so
    cache lookup / encode fallback / rollback semantics stay in one
    place.

    ``should_abort`` (optional zero-arg callable) lets a stopping
    session cut the pending-stems wait short (see
    :func:`wait_for_pending_stems`); the abort maps to a normal
    stem-extraction error, which the swap path turns into SwapFailed.
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
        if upload_stems is None and stems_pending(fixture_name):
            # The upload path is ripping this track's stems on a
            # background thread (see ws_adapter._handle_upload_track).
            if source_mode == "full":
                # Stems are overlay-only here: proceed WITHOUT them so
                # the swap (and audio) lands immediately. The client's
                # overlays arrive via the pushed ``stem_assets`` frame
                # when the background rip completes.
                logger.info(
                    "stems_pending_deferred fixture_name={} context={}",
                    fixture_name, log_context or None,
                )
                return None, None, source, waveform
            # The selected stem IS the inference source: wait for the
            # in-flight rip instead of starting a duplicate separation.
            logger.info(
                "stems_pending_wait fixture_name={} source_mode={} context={}",
                fixture_name, source_mode, log_context or None,
            )
            finished = wait_for_pending_stems(
                fixture_name, should_abort=should_abort,
            )
            logger.info(
                "stems_pending_wait_done fixture_name={} finished={}",
                fixture_name, finished,
            )
            if not finished:
                # Whether the wait was aborted (session stopping) or
                # timed out (rip slow/hung, still holding _INFER_LOCK),
                # falling through would start a DUPLICATE inline
                # separation — first blocking on _INFER_LOCK behind the
                # very rip we just gave up waiting for. Fail the swap
                # instead; the client can re-swap once the rip lands.
                aborted = should_abort is not None and should_abort()
                return (
                    None,
                    (
                        "session stopping; aborted wait for background stems"
                        if aborted
                        else "timed out waiting for the in-flight background stem rip"
                    ),
                    source,
                    waveform,
                )
            upload_stems = audio_clip_stems(
                fixture_name,
                waveform=waveform,
                sample_rate=SAMPLE_RATE,
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
                # Park this session's eager modules while the RoFormer
                # runs (restored before the prepare_source below needs
                # them back). Safe here: at create the runner doesn't
                # exist yet, and at swap WE ARE the runner thread.
                model_context=session.handler,
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
# Capability gate
# ---------------------------------------------------------------------------


def requires_capability(capability: str, command: str):
    """Gate a session operation on the backend's capability mask.

    ``capability`` is the :class:`~acestep.streaming.generator_backend.
    Capabilities` field the operation needs; ``command`` is the wire
    command name carrying the matching ``requires`` tag in the
    wire-contract registry (``demos/realtime_motion_graph_web/
    protocol.py``). The drift guard in ``tests/unit/test_wire_contract``
    AST-parses these decorators and asserts the mapping matches the
    registry's tagged commands exactly, so a tagged command can't ship
    without defined failure behavior (and vice versa).

    On an unsupported backend the wrapped operation does NOT run; the
    session maps the :class:`UnsupportedOperation` to a typed
    :class:`CommandFailed` event instead (plan §3.4 — loud failure,
    never a silent no-op). ``UnsupportedOperation`` raised from inside
    the operation (a future backend's own control method) maps the
    same way.
    """
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            try:
                self._require_capability(capability, command)
                return fn(self, *args, **kwargs)
            except UnsupportedOperation as exc:
                logger.warning(
                    "command_unsupported command={} requires={} backend={}",
                    command, exc.capability, self.backend.name,
                )
                self.bus.publish(CommandFailed(
                    command=command,
                    requires=exc.capability,
                    error=str(exc),
                ))
                return None
        return wrapper
    return deco


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
        canvas: SourceCanvas,
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
        # Sample-exact audio mirror of the source latent, the substrate
        # for write_audio. Replaced wholesale (with a source_epoch bump)
        # by every swap; never mutated by playback.
        self.canvas = canvas
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

        # Set at the end of close(), after GPU state is released. A
        # preempting connection (ws_adapter's single-active-session
        # policy) waits on this before creating its own session so the
        # two model stacks never need VRAM simultaneously.
        self.closed = threading.Event()

        # The session's GeneratorBackend, selected by SessionConfig.backend
        # via the family registry. Constructed here (not in run()) so the
        # contract surface — capabilities()/geometry()/knob_specs() — is
        # readable for the wire ``ready`` frame and the session snapshot
        # before the runner starts, and so an unknown family fails at
        # session create (config-time, per families.make_backend).
        self.backend = make_backend(
            getattr(config, "backend", "acestep") or "acestep", self,
        )

        # Cached {name: KnobSpec} map for hot-path validation in set_knobs.
        # knob_specs() rebuilds 34 dataclasses, so we never call it per tick;
        # rebuilt only when the LoRA set changes (see _apply_lora_pending)
        # or a manual steering slot is added/popped.
        # Reassigned wholesale (atomic ref swap), so set_knobs can read it
        # without a lock from the dispatch thread.
        self._rebuild_knob_specs(self.initial_enable_ids)

        # The backend's manifest can extend the shared registry set the
        # KnobState was seeded from (today: the activation-steering
        # knobs, present only when the checkpoint has a vector bundle).
        # Seed any such knob so snapshot ``knob_values`` stays complete
        # from t=0 — add_knob is a no-op for already-seeded names.
        for spec in self._knob_specs_by_name.values():
            self.virtual_knobs.add_knob(spec)

    def _rebuild_knob_specs(self, lora_ids: list) -> None:
        # Backend-owned manifest: which specs the LoRA set expands to is
        # family knowledge behind the seam; the session only tracks the
        # enabled-id list.
        self._knob_specs_by_name = {
            s.name: s for s in self.backend.knob_specs(list(lora_ids or []))
        }

    def _require_capability(self, capability: str, command: str) -> None:
        """Raise :class:`UnsupportedOperation` when the backend's
        capability mask doesn't include ``capability``. The single
        check behind :func:`requires_capability`."""
        if not getattr(self.backend.capabilities(), capability):
            raise UnsupportedOperation(
                capability,
                f"backend {self.backend.name!r} does not support "
                f"{command!r} (requires capability {capability!r})",
            )

    def _enabled_lora_ids(self) -> list:
        if not (self.use_lora and self.engine_obj is not None):
            return []
        return [d.id for d in self.engine_obj.list_loras() if d.state == "enabled"]

    # ---- Snapshot / catalog helpers -------------------------------------

    def geometry_payload(self) -> dict:
        """Wire-shaped geometry block (plan §3.2), declared by the
        backend. Replaces deriving sample rate / latent cadence from
        client-side constants; ``duration_s`` is null for endless
        streams (reserved v2 ``append`` shape)."""
        g = self.backend.geometry()
        return {
            "sample_rate": g.sample_rate,
            "channels": g.channels,
            "chunk_rate_hz": g.chunk_rate_hz,
            "duration_s": g.duration_s,
        }

    def capabilities_payload(self) -> dict:
        """Wire-shaped capability mask (plan §3.1), declared by the
        backend. Consumed by client panel gating and the session's own
        capability gate (:func:`requires_capability`)."""
        return asdict(self.backend.capabilities())

    def knob_manifest_payload(self) -> dict:
        """The per-session knob manifest (plan §3.3): the same
        ``{version, knobs}`` envelope ``GET /api/knobs`` serves, but
        projected from this session's LIVE spec map — backend-owned,
        SDE-mode-resolved, including the enabled ``lora_str_<id>``
        knobs. ``/api/knobs`` stays for the replay server and static
        tooling; this is the gated per-session truth."""
        return {
            "version": KNOB_SCHEMA_VERSION,
            "knobs": catalog_from_specs(self._knob_specs_by_name.values()),
        }

    def steering_payload(self) -> dict:
        """Wire-shaped activation-steering block, shared by the ``ready``
        frame and the snapshot. ``steering_available`` mirrors the
        backend's ``steering`` capability bit; the count/cap fields drive
        the client's manual-slot row rendering and +/- enablement."""
        ctl = getattr(self.backend, "steering", None)
        if ctl is None:
            return {
                "manual_slot_count": 0,
                "manual_slot_cap": 0,
                "steering_available": False,
            }
        return {
            "manual_slot_count": ctl.slot_count,
            "manual_slot_cap": ctl.slot_cap,
            "steering_available": bool(ctl.is_loaded),
        }

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
            # One of the two inputs that determine this session's knob
            # universe (see knob_specs; the other is the enabled-LoRA set
            # in lora_catalog). Lets transport adapters (the MCP server's
            # schema validation) reproduce the knob set without sniffing
            # knob names out of knob_values.
            "sde": self.use_sde,
            # Phase-2 contract surface, mirroring the wire ``ready``
            # frame (plan §3.1–3.3). The manifest here tracks the LIVE
            # knob universe (runtime LoRA toggles included), unlike the
            # ready frame's session-start copy.
            "geometry": self.geometry_payload(),
            "capabilities": self.capabilities_payload(),
            "knob_manifest": self.knob_manifest_payload(),
            # Activation-steering surface (count drives manual-slot row
            # rendering; available=False hides the steering tiles).
            **self.steering_payload(),
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
            # Runner construction stays inside the try so a setup
            # failure still reaches the finally's close(), but it is
            # NOT inside the run() swallow-path below: constructor
            # errors propagate to the caller (warmup must see them).
            backend = self.backend
            runner = PipelineRunner(
                backend, self.audio_eng,
                state=self.state,
                # The backend clamps vae_window to the Session's effective
                # post-engine-profile value; the runner's slice scheduling
                # must match the slice length the backend actually renders.
                vae_window=backend.vae_window,
                idle_threshold_s=IDLE_PAUSE_S,
                on_audio_ready=self._on_audio_ready,
                before_tick=self.apply_pending,
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
                # Surface the failure to transports BEFORE close() tears
                # the bus down: without this the client sees a frozen UI
                # (no slices, no error) when generation dies post-ready.
                # Subscription drainers deliver already-queued events
                # after bus.close(), so publish-then-close is safe.
                try:
                    self.bus.publish(SessionError(
                        code="pipeline_error",
                        message=(
                            f"Generation stopped: {type(exc).__name__}: {exc}"
                        ),
                    ))
                except Exception:
                    pass
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
        # Last: signal waiters (preempting connections) that this
        # session's GPU state is gone.
        self.closed.set()

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
                self.virtual_knobs.remove_knob(lora_strength_spec(lid).name)
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
                # driven by the client's params dict. The knob's shape
                # comes from the registry (lora_strength_spec), never
                # re-declared here; only the default is overridden to
                # the strength we just enabled at (the spec is a fresh
                # instance, so mutating it is safe), so the runner's
                # slider-delta check (set_lora_strength only when the
                # new value differs by > 0.02) doesn't fire a
                # redundant refit on tick 1.
                spec = lora_strength_spec(lid)
                if strength is not None:
                    spec.default = float(strength)
                self.virtual_knobs.add_knob(spec)
            except Exception as e:
                logger.exception("lora_enable_failed id={} error={}", lid, e)
        # Publish the refreshed catalog. No automatic re-encode here.
        # With WYSIWYG prompts the trigger word lives in the visible
        # promptA/promptB text; the client's visible-prepend logic
        # mutates the prompt on toggle and sends a normal
        # prompt-update message.
        # The LoRA set changed, so the lora_str_<id> knob universe did too —
        # refresh the cached validation spec map (atomic ref swap; set_knobs
        # reads it lock-free from the dispatch thread).
        self._rebuild_knob_specs(self._enabled_lora_ids())
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
                    # We ARE the runner thread: a preempting connection
                    # flips running=False and waits only 45 s for
                    # teardown, so the pending-stems wait (300 s) must
                    # be interruptible or the preemptor builds a second
                    # stack next to this still-resident one.
                    should_abort=lambda: not self.state.running,
                )
            )
            if new_stem_error is not None and new_stem_source_mode != "full":
                self.bus.publish(SwapFailed(
                    error=f"Stem extraction failed: {new_stem_error}",
                ))
                return

            # New-source audio mirror, built unlocked like the rest of
            # the setup work; swapped in (with the epoch bump) under the
            # commit lock below.
            new_canvas = SourceCanvas(
                new_wf.to(new_source.latent.tensor.device),
            )

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
                # Retire the old canvas: a write_audio staged against it
                # epoch-checks at commit and discards itself.
                self.canvas = new_canvas
                state.source_epoch += 1
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
                source_epoch=state.source_epoch,
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
        # Enforce the knob contract on the load-bearing path: clamp numerics
        # to [min,max], coerce ints, drop invalid enum/bool values. Unknown
        # keys (curve specs, the playback clock) pass through untouched.
        # Silent clamp — never reject a 125 Hz tick over one bad value.
        clean, _errors = coerce_knob_values(raw, self._knob_specs_by_name)
        with state._lock:
            self.virtual_knobs.update(clean)
            try:
                self.audio_eng.position = int(playback_pos * SAMPLE_RATE) % max(
                    1, len(self.audio_eng.current),
                )
            except Exception:
                pass

    @requires_capability("loop_band", "loop_band")
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

    @requires_capability("depth", "set_depth")
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

    @requires_capability("lora", "enable_lora")
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

    @requires_capability("lora", "disable_lora")
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

    @requires_capability("steering", "manual_slot_add")
    def manual_slot_add(
        self,
        *,
        origin: CommandOrigin = CommandOrigin.PRIMARY,
    ) -> None:
        """Allocate the next manual steering slot (LIFO).

        The controller is the primary write; KnobState and the cached
        spec map mirror it so the four ``man_*_<N>`` knobs validate and
        snapshot from the moment the slot exists. No GPU work, so it
        applies inline (no pending queue). Publishes
        :class:`ManualSlotCount` on success AND refusal — the client's
        +/- UI resyncs from the echo either way.
        """
        self.state.last_activity_ts = time.monotonic()
        ctl = self.backend.steering
        try:
            new_slot = ctl.add_slot()
        except CapacityError:
            logger.info(
                "manual_slot_add_refused origin={} cap={}",
                origin.value, ctl.slot_cap,
            )
        else:
            self._rebuild_knob_specs(self._enabled_lora_ids())
            names = ctl.knob_names(new_slot)
            for name in (names.src, names.layer, names.step, names.alpha):
                spec = self._knob_specs_by_name.get(name)
                if spec is not None:
                    self.virtual_knobs.add_knob(spec)
            logger.info(
                "manual_slot_added origin={} slot={}", origin.value, new_slot,
            )
        self.bus.publish(ManualSlotCount(count=ctl.slot_count))

    @requires_capability("steering", "manual_slot_pop")
    def manual_slot_pop(
        self,
        *,
        origin: CommandOrigin = CommandOrigin.PRIMARY,
    ) -> None:
        """Remove the highest-numbered manual steering slot (LIFO;
        interior deletion is not supported). Refusal on an empty
        registry still publishes :class:`ManualSlotCount`."""
        self.state.last_activity_ts = time.monotonic()
        ctl = self.backend.steering
        try:
            popped = ctl.pop_slot()
        except EmptyError:
            logger.info("manual_slot_pop_refused origin={}", origin.value)
        else:
            names = ctl.knob_names(popped)
            for name in (names.src, names.layer, names.step, names.alpha):
                self.virtual_knobs.remove_knob(name)
            self._rebuild_knob_specs(self._enabled_lora_ids())
            logger.info(
                "manual_slot_popped origin={} slot={}", origin.value, popped,
            )
        self.bus.publish(ManualSlotCount(count=ctl.slot_count))

    @requires_capability("timbre", "set_timbre_strength")
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

    @requires_capability("timbre", "set_timbre_source")
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

    @requires_capability("timbre", "set_timbre_fixture")
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

    @requires_capability("timbre", "clear_timbre_source")
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

    @requires_capability("structure", "set_structure_source")
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

    @requires_capability("structure", "set_structure_fixture")
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

    @requires_capability("structure", "clear_structure_source")
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

    @requires_capability("swap", "swap_source")
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

    @requires_capability("write_audio", "write_audio")
    def write_audio(
        self,
        audio: Audio,
        *,
        at_s: float = 0.0,
        mix: str = "replace",
        repeat: str = "none",
        source_epoch: int | None = None,
        refresh_timbre: bool = False,
        origin: CommandOrigin = CommandOrigin.PRIMARY,
    ) -> None:
        """Write audio onto the live source — the "play into the model"
        path, distinct from :meth:`swap_source`.

        Unlike a swap, this does NOT restart the song, reset the
        playhead, run BPM/key detection, or re-run ``prepare_source``.
        The audio lands on the session's :class:`~acestep.engine.canvas.
        SourceCanvas` (the sample-exact mirror of the source audio) at
        ``at_s``; the dirty span is re-encoded with true context pulled
        from the mirror and committed into the live source latent in
        place, so the edit emerges within a few ticks. Clients ship ONLY
        the changed audio — no context margins, no frame math, no full-
        loop re-uploads; placement is plain seconds.

        * ``mix="replace"`` overwrites (declicked against the existing
          audio); ``mix="sum"`` overdubs on top.
        * ``repeat="fill"`` treats the buffer as one period of a loop
          and lays it across the whole canvas, phase-anchored at
          ``at_s`` — sample-exact in the audio domain, so any bar
          length works (the 25 fps latent grid never quantizes the
          period).
        * ``source_epoch`` (from ``ready`` / ``swap_ready``) pins the
          write to the canvas generation it was computed against; a
          mismatch is rejected instead of splicing into the wrong
          source. Omit to write against whatever is live.

        Locking: the GPU re-encode is STAGED outside ``state._lock``
        (the runner takes that lock every tick — an encode under it
        stalls the tick loop) and committed under it as one tensor copy,
        re-validating the epoch so a concurrent swap discards the stale
        block instead of racing it.

        The structure (semantic hint) is refreshed from the committed
        latent (~free); the conditioning's self-timbre reference is
        refreshed only when ``refresh_timbre`` is set (the expensive
        optional, ~+50 ms per prompt) and only in self-timbre mode (an
        explicit timbre override is left untouched).
        """
        state = self.state
        state.last_activity_ts = time.monotonic()
        logger.info(
            "write_audio_recv origin={} at_s={} mix={} repeat={} "
            "source_epoch={} refresh_timbre={} buffer_s={:.2f}",
            origin.value, at_s, mix, repeat, source_epoch, refresh_timbre,
            audio.waveform.shape[-1] / SAMPLE_RATE,
        )
        try:
            # Phase 1 (locked, cheap): validate the epoch and land the
            # audio on the mirror.
            with state._lock:
                epoch = state.source_epoch
                canvas = self.canvas
                if source_epoch is not None and source_epoch != epoch:
                    logger.warning(
                        "write_audio_stale_epoch sent={} current={}",
                        source_epoch, epoch,
                    )
                    self.bus.publish(AudioWriteFailed(error=(
                        f"stale source_epoch {source_epoch} (current "
                        f"{epoch}); re-read it from ready/swap_ready"
                    )))
                    return
                f0, f1 = canvas.write(
                    audio.waveform, at_s=at_s, mix=mix, repeat=repeat,
                )

            # Phase 2 (unlocked): GPU re-encode of the dirty span, with
            # context read from the mirror.
            staged, e0, e1 = self.stream.stage_source_frames(canvas, f0, f1)

            # Phase 3 (locked, one tensor copy): commit unless the
            # source swapped out from under the encode.
            with state._lock:
                if state.source_epoch != epoch:
                    logger.info("write_audio_discarded reason=source_swapped")
                    self.bus.publish(AudioWriteFailed(error=(
                        "source swapped during the write; re-send against "
                        "the new source_epoch"
                    )))
                    return
                self.stream.commit_source_frames(staged, e0, e1)

            # Structure hint derives from the source latent — refresh it
            # from the committed result (unlocked GPU read; reference
            # swapped under the lock). When no structure override is
            # active, ``state.source`` is the same PreparedSource the
            # runner reads via ``stream.source.context_latent``; when an
            # override IS active the runner keeps reading the override
            # and this only keeps ``state.source`` fresh for a later
            # clear.
            ctx = self.session.extract_hints(self.stream.source.latent)
            with state._lock:
                if state.source_epoch == epoch:
                    state.source.context_latent = ctx
                    r = self.runner_holder[0]
                    if r is not None:
                        r.mark_hint_dirty()

            # Self-timbre conditioning anchors on the source latent, so
            # it goes stale on a write. Refresh only on request and only
            # when there's no explicit timbre override; encode unlocked
            # (snapshot in, epoch-checked assign out).
            if refresh_timbre and state.timbre_latent is None:
                with state._lock:
                    prompt_a = state.prompt_text
                    prompt_b = state.prompt_text_b
                    bpm, dur = state.bpm, state.duration
                    key, tsig = state.key, state.time_signature
                refer = self.stream.source.latent
                cp = encode_cond_pair(
                    self.session, prompt_a, refer, bpm, dur, key, tsig,
                )
                cp_b = cp if prompt_b == prompt_a else encode_cond_pair(
                    self.session, prompt_b, refer, bpm, dur, key, tsig,
                )
                with state._lock:
                    if state.source_epoch == epoch:
                        state.cond_pair = cp
                        state.cond_pair_b = cp_b
                        self._refresh_conditioning()

            self.bus.publish(AudioWritten(
                start_s=e0 / 25.0, end_s=e1 / 25.0, source_epoch=epoch,
            ))
            logger.info(
                "write_audio_applied span_frames=({},{}) span_s=({:.2f},{:.2f})",
                e0, e1, e0 / 25.0, e1 / 25.0,
            )
        except Exception as exc:
            logger.opt(exception=True).error(
                "write_audio_failed origin={} error={}", origin.value, exc,
            )
            self.bus.publish(AudioWriteFailed(error=str(exc)))

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
