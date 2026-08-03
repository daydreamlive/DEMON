"""SA3Backend: Stable Audio 3 behind the GeneratorBackend seam.

The second :class:`~acestep.streaming.diffusion_backend.DiffusionBackend`
family, parameterized by (:class:`~acestep.engine.sa3_adapter.SA3Adapter`,
:class:`~acestep.engine.sa3_context.SA3SAMECodec`). It owns BOTH halves
of the parameterization (unlike ACE, whose adapter is pipeline-default
and whose codec is the engine Session): a shared
:class:`~acestep.engine.stream.StreamPipeline` is built here with
``engine=None`` and the SA3 adapter, and rendering decodes through the
SAME codec with the 44.1 → 48 kHz delivery resample applied at the
decode boundary (round_3 decision 2: AudioEngine / worklet / client
stay 48 kHz-untouched in v1; ``geometry().sample_rate`` declares the
DELIVERED rate, 48000 — native-44.1k delivery later is a geometry +
client change only).

Control surface (everything else off, capability-gated):

* ``prompt`` — per-prompt re-conditioning via
  :meth:`SA3Backend.handle_set_prompt`; an A/B pair when ``tags_b``
  differs, crossfaded by :meth:`SA3Backend.handle_set_prompt_blend`
  (per-token slerp of the T5Gemma cross-attn conditioning, the same
  geodesic ACE's ``blend_for_strength`` walks — a linear midpoint
  collapses conditioning norm and sounds washed out).
* ``sa3_denoise`` — SA3's ``init_noise_level``, the audio-to-audio
  blend against the source anchor. The name is load-bearing: ACE's
  ``denoise`` is a different control, and the homonym rule
  (``tests/unit/test_knob_homonyms.py``) forbids reusing the name with
  different semantics.
* ``sa3_shift`` — relative schedule warp on top of the checkpoint's
  dist_shift (``SA3Adapter.shift_alpha``); changes invalidate the
  pipeline's per-denoise schedule cache.
* ``x0_target`` / ``feedback`` / ``feedback_depth`` — taken FROM the
  shared registry (identical semantics to ACE, solver-level latent
  mechanics that are family-agnostic): the source-lock morph toward
  the anchor latent and the past-latent delay-tap blend. Both are only
  audible at ``sa3_denoise`` < 1, where slot init actually reads
  ``source_latents`` — at 1.0 every slot starts from pure noise.
* ``seed`` / ``steps_override`` — shared registry, as before.

Continuity comes the same way the spike demo proved
(``demos/test_stream_sa3_graph.py``): every emit is a partial-denoise
cover of the SAME source latent at the same seed, so advancing playback
windows reconstruct one evolving song.

Capabilities: ``refines_audio`` + ``loop_band`` + ``swap``. ``swap``
re-anchors the session in place: the new upload is SAME-encoded into a
replacement source latent (:meth:`SA3Backend.handle_swap_source`, the
session's backend-owned swap hook) while duration/conditioning stay
fixed — the geometry-invariant analog of ``handle_set_prompt``. No
timbre/structure/LoRA/stems/depth/curves until validated (canonical
plan Phase 5).
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Callable, Optional

import torch

from acestep.engine.obs import logger
from acestep.nodes.interpolation import INTERPOLATIONS, slerp
from acestep.streaming.diffusion_backend import DiffusionBackend
from acestep.streaming.generator_backend import (
    AudioChunk,
    AudioGeometry,
    Capabilities,
    TickContext,
)
from acestep.streaming.knobs import (
    KnobSpec,
    knob_specs as registry_knob_specs,
    lora_strength_spec,
)

# Delivery rate (v1): SA3's native 44.1 kHz is resampled at the decode
# boundary so everything downstream of the backend stays at the engine
# rate. See module docstring / round_3 decision 2.
DELIVERY_SAMPLE_RATE = 48000
SA3_SAMPLE_RATE = 44100
SA3_LATENT_RATE_HZ = 44100.0 / 4096.0

# Largest tap index the feedback delay can address. Derived from the
# shared registry spec (the same one the manifest serves) so the knob
# bound and the history ring can never drift apart.
MAX_FEEDBACK_DEPTH = int(
    next(s for s in registry_knob_specs(False) if s.name == "feedback_depth").max_val
)


def sa3_lora_compatible(metadata: dict, model_id: str) -> bool:
    """The SA3 compatibility predicate: weight-format family +
    training lineage.

    Family: a file whose sniffed ``lora_family`` names a different
    family (e.g. "ace") can never load on the SA3 parametrization
    engine — hard no. Unknown family stays permissive.

    Lineage: the file's ``base_model`` (sidecar wins, else the
    embedded trainer config) canonicalized to a runtime checkpoint id
    must match the loaded model. Unknown/unrecognized lineage stays
    permissive per the seam contract.

    Module-level (not a method) because the create path
    (:mod:`acestep.streaming.sa3_session`) needs the same verdict for
    startup alias resolution before the backend exists — mirroring how
    the ACE create path applies its scale axis directly.
    """
    from acestep.lora_metadata import canonical_sa3_lineage

    family = metadata.get("lora_family")
    if family and family != "sa3":
        return False
    lineage = canonical_sa3_lineage(metadata.get("base_model"))
    if lineage is None or not model_id:
        # Unknown on either side is compatible (the same permissive
        # stance as the scale axis).
        return True
    return lineage == model_id


def sa3_knob_specs(loras: tuple | list = ()) -> list:
    """The SA3 family knob manifest (backend-owned, plan §3.3).

    ``seed``, ``steps_override``, ``x0_target``, ``feedback`` and
    ``feedback_depth`` are genuinely neutral controls (solver-level
    latent mechanics for the latter three) and are taken FROM the
    shared registry by name, so their semantics can never fork from
    ACE's (the homonym test would catch it; this makes the fork
    impossible instead). ``sa3_denoise`` / ``sa3_shift`` are
    family-prefixed because ACE's ``denoise`` and ``shift`` mean
    different things.

    ``loras`` is the session's enabled-LoRA id list; each id expands to
    the shared ``lora_str_<id>`` strength spec — the same registry
    factory ACE uses, so the knob's shape (and therefore its wire
    semantics) cannot fork across families.
    """
    shared = {s.name: s for s in registry_knob_specs(False)}
    return [
        KnobSpec(
            "sa3_denoise", default=1.0, max_val=1.0, group="sa3",
            description=(
                "Measured SA3 audio-change amount: 1.0 generates from pure "
                "noise, while lower values stay progressively closer to the "
                "source. Mapped onto init_noise_level so useful change is "
                "spread across the dial. Distinct from ACE's 'denoise' "
                "(k1 strength), hence the prefix."
            ),
        ),
        KnobSpec(
            "sa3_shift", default=1.0, min_val=0.25, max_val=4.0, group="sa3",
            description=(
                "Relative timestep-schedule warp on top of the "
                "checkpoint's own dist_shift (Flux alpha map). 1.0 = "
                "stock schedule; >1 spends steps near noise (structure), "
                "<1 near data (refinement). Distinct from ACE's 'shift' "
                "(absolute flow-matching shift), hence the prefix."
            ),
        ),
        shared["x0_target"],
        shared["feedback"],
        shared["feedback_depth"],
        shared["seed"],
        shared["steps_override"],
    ] + [lora_strength_spec(lid) for lid in loras]


class SA3Backend(DiffusionBackend):
    """Stable Audio 3 streaming generation. See module docstring.

    Decoupled from :class:`~acestep.engine.sa3_context.SA3Context` for
    testability: takes the adapter, codec, conditioning, and a
    schedule-builder factory (``steps -> (denoise -> schedule)``)
    directly, so unit tests drive it with a mock DiT and codec. The
    production assembly (context → adapter/codec/cond → backend) is
    :meth:`from_context`.
    """

    name = "sa3"

    def __init__(
        self,
        *,
        adapter,
        codec,
        cond,
        schedule_builder_factory: Callable[[int], Callable],
        knob_state,
        state=None,
        source_latent_bct: Optional[torch.Tensor] = None,
        steps: int = 8,
        depth: int = 4,
        default_seed: int = 1528,
        vae_window_s: float = 3.0,
        # SA3 checkpoints are ``diffusion_objective: rf_denoiser`` —
        # upstream samples them with pingpong ONLY (euler isn't even
        # offered in their UI for this objective, and 8-step euler is
        # audibly degraded). Determinism — and therefore window-splice
        # continuity — is preserved via the seeded per-slot renoise
        # stream (SlotRequest.sde_noise_seeded), the spike pipeline's
        # per-slot generator semantics.
        sampler: str = "pingpong",
        prompt_rebuilder: Optional[Callable] = None,
        prompt_tags: Optional[str] = None,
        # Prompt-B conditioning capture for the A/B crossfade. None
        # (or identical to ``cond``) means no B prompt: the blend is a
        # no-op and handle_set_prompt_blend keeps serving bundle A.
        cond_b=None,
        # ``(waveform, sample_rate, audio_sample_size) -> [1, 256, T]``
        # — the source re-encode hook behind :meth:`handle_swap_source`.
        # Supplied by :meth:`from_context` (a closure over the
        # SA3Context); None on directly-constructed test backends, where
        # handle_swap_source fails loudly instead.
        source_encoder: Optional[Callable] = None,
        # SA3 LoRA (notes/SA3_LORA_PLAN.md Phase 1): the family's
        # parametrization manager (acestep.engine.sa3_lora), constructed
        # by the create path against the process-cached model. None =
        # no LoRA surface (directly-constructed test backends).
        lora_manager=None,
        # Session-level LoRA toggle (config.lora); gates the capability
        # bit, the per-tick strength reads, and has_pending_refit.
        use_lora: bool = False,
        # The always-resident eager DiT module, for the interim
        # TRT→eager swap (D6a): when a LoRA enables on a TRT-DiT
        # session the adapter's dit swaps to this; when the last
        # disables it swaps back. None/identical when the session is
        # already eager (swap becomes a no-op).
        eager_dit=None,
        # Runtime checkpoint id ("medium"/"small-music"/…) for the
        # lineage axis of lora_compatible.
        model_id: str = "",
        # Phase-2 TRT endgame: an SA3TRTRefitMirror over a refit-built,
        # exclusively-owned engine. When present, LoRA mutations refit
        # the engine in place instead of swapping to the eager DiT, and
        # knob-driven strength changes route through the pending stash
        # (D6b.5) so the refit stall is always announced.
        refit_mirror=None,
    ):
        super().__init__(adapter=adapter, codec=codec)
        self._cond = cond
        self._cond_b = cond_b if cond_b is not None else cond
        # Live A↔B crossfade value and the bundle the next submit
        # carries (A verbatim at 0, B verbatim at 1, slerp between).
        # Swapped GIL-atomically from the command thread, exactly like
        # handle_set_prompt's reference swaps.
        self._blend = 0.0
        self._active_bundle = cond.cond_bundle
        # Ring of past finished latents for the feedback delay-tap
        # (engine layout [1, T, C]; [0] is the most recent).
        self._latent_history: deque = deque(maxlen=MAX_FEEDBACK_DEPTH)
        self._schedule_builder_factory = schedule_builder_factory
        # ``(tags, steps) -> (cond, steps -> (denoise -> schedule))`` —
        # the per-prompt re-conditioning hook behind :meth:`handle_set_prompt`.
        # Supplied by :meth:`from_context` (a closure over the
        # SA3Context); None on directly-constructed test backends, where
        # handle_set_prompt fails loudly instead.
        self._prompt_rebuilder = prompt_rebuilder
        # Source re-encode hook for handle_swap_source (see ctor arg).
        self._source_encoder = source_encoder
        self.knob_state = knob_state
        self.state = state
        self._steps = int(steps)
        self._depth = int(depth)
        self._default_seed = int(default_seed)
        self.vae_window = float(vae_window_s)
        # "pingpong"/"sde" (rf_denoiser-native, deterministic via seeded
        # renoise) | "ode" (euler; off-objective for SA3, debug only)
        self._sampler = sampler

        # Source anchor for audio-to-audio: engine layout [1, T, 256].
        self._source_latent_btc = (
            source_latent_bct.movedim(1, 2).contiguous()
            if source_latent_bct is not None else None
        )

        # Emerged-generation observability. SA3 knob/prompt changes ride
        # the NEXT SlotRequest only (no shared-curve writes onto
        # in-flight slots), so the effect of a control change surfaces
        # one pipeline-flush later. To make that observable on the wire
        # (knob→ear measurement, params panel truth), we label each
        # fresh latent with the request it was generated FROM
        # (pipeline.last_finished_request) and stamp those values into
        # ``state.params`` as ``gen_*`` keys BEFORE the same tick's
        # windowed render publishes its params echo. Conditioning
        # bundles are tracked by identity: ``_cond_history`` holds
        # (bundle, epoch, tags) strong refs (bounded) so an in-flight
        # request's ``aux_cond`` can be mapped back to the prompt it
        # carried even after a handle_set_prompt swap.
        self._cond_epoch = 0
        self._cond_history: list = [(cond.cond_bundle, 0, prompt_tags)]
        self._emerged_request = None
        self._emerged_marker = None  # (denoise, epoch) of the last log

        # Guards the conditioning control state shared between the
        # command thread (handle_set_prompt / handle_set_prompt_blend,
        # called outside the session state lock) and the runner thread
        # (_prepare_tick / _generate / _cond_meta_for). A prompt swap
        # mutates several fields together — the schedule builder + cache,
        # _cond/_cond_b, _active_bundle, and the _cond_history list (a
        # non-atomic append+truncate the runner iterates) — so the bare
        # GIL-atomic reference-swap argument doesn't cover it. The runner
        # only ever holds this to snapshot, never across pipeline work.
        self._control_lock = threading.Lock()

        # ---- LoRA (plan D4/D5/D6a) ------------------------------------
        self._lora_mgr = lora_manager
        self._use_lora = bool(use_lora)
        self._model_id = str(model_id)
        # D5 serialization point: conditioner EXECUTION (prompt-swap
        # T5Gemma captures on the command thread) and conditioner
        # MUTATION (LoRA parametrization changes at the runner
        # rendezvous) are mutually exclusive under this lock. Mutating
        # parametrizations under a concurrently executing conditioner is
        # not safe; the create path's captures predate the backend, so
        # they need no lock.
        self._conditioner_lock = threading.Lock()
        # D6a: the accelerated (possibly TRT) DiT the session was built
        # with, and the always-resident eager module. _sync_dit_for_lora
        # flips adapter.dit between them by LoRA activity.
        self._dit_accel = adapter.dit if adapter is not None else None
        self._dit_eager = eager_dit if eager_dit is not None else self._dit_accel
        self._refit_mirror = refit_mirror
        # Knob-driven strength changes stashed by rebuild_imminent under
        # the refit mirror (a strength change is a full engine refit
        # there, and must be announced via rebuild_imminent /
        # has_pending_refit before it runs — never applied mid-tick
        # unannounced). Empty and unused in eager mode, where a
        # strength change is a ~5 ms buffer write applied inline.
        self._pending_lora_strengths: dict = {}

        # Rendered-audio cache: one full decode+resample per fresh
        # latent (SAME-S decodes the whole window in ~11 ms); window
        # renders slice it, so gap-fill re-renders are bit-stable.
        # Windowed codecs (SAME-L / medium: full decode ~80 ms) bypass
        # the cache and decode per render instead — see render_window.
        self._rendered_for = None     # latent tensor identity
        self._rendered_48k = None     # np.ndarray [N, C] float32
        # Seed pinning the SAME decoder's inference-time noise for the
        # latest emerged latent (stamped in _after_produce), so repeated
        # decodes — and re-runs of the same session inputs — produce
        # bit-identical audio. None until the first fresh latent.
        self._decode_seed = None
        self._windowed_codec = hasattr(codec, "decode_window")

        self.pipeline = self._build_pipeline(self._steps)

    # ---- assembly -----------------------------------------------------------

    @classmethod
    def from_context(
        cls,
        context,
        *,
        prompt: str,
        duration_s: float,
        knob_state,
        state=None,
        source_audio=None,
        cond=None,
        prompt_b: Optional[str] = None,
        cond_b=None,
        source_latent_bct=None,
        dit_backend: str = "eager",
        codec_backend: str = "eager",
        **kwargs,
    ) -> "SA3Backend":
        """Production assembly over a loaded
        :class:`~acestep.engine.sa3_context.SA3Context`.

        ``cond`` / ``cond_b`` / ``source_latent_bct`` accept precomputed
        values so the serving-layer create path
        (:mod:`acestep.streaming.sa3_session`), which runs
        ``prepare_cond`` + source encode itself before the session
        exists, doesn't pay them twice; absent, they're computed here
        (the in-process assembly the GPU smoke validated). ``prompt_b``
        seeds the A/B crossfade pair; a missing/empty/identical B means
        the blend is a no-op until a later ``set_prompt`` supplies one.

        ``dit_backend`` / ``codec_backend`` are the session's resolved
        acceleration values (the serving layer's decoder/vae accel
        params, compile already normalized to eager by the create
        path); the context maps them onto its components (``make_dit``
        / ``make_codec``): "tensorrt" selects the built engines when
        they cover the session, with eager fallback; small has no TRT
        flavors and runs the torch DiT + SAME-S full-decode codec
        either way."""
        from acestep.engine.sa3_adapter import SA3Adapter

        steps = int(kwargs.get("steps", 8))
        if cond is None:
            cond = context.prepare_cond(
                prompt=prompt, duration=duration_s, steps=steps,
            )
        if cond_b is None and prompt_b not in (None, "", prompt):
            cond_b = context.prepare_cond(
                prompt=prompt_b, duration=duration_s, steps=steps,
            )
        if cond_b is not None and int(cond_b.latent_frames) != int(cond.latent_frames):
            # Same fixed duration must mean the same latent geometry;
            # a mismatch would desync the blend against the ring
            # buffer and the source anchor (cf. handle_set_prompt).
            raise ValueError(
                f"sa3 prompt-B conditioning changed latent_frames "
                f"({cond.latent_frames} -> {cond_b.latent_frames})"
            )
        source_latent = (
            source_latent_bct if source_latent_bct is not None
            else context.encode_source(source_audio, cond.audio_sample_size)
            if source_audio is not None else None
        )
        # LoRA sessions prefer a refit-built engine (and avoid fp8,
        # whose refit story is unproven) — see find_dit_engine.
        want_lora = bool(kwargs.get("use_lora")) and (
            kwargs.get("lora_manager") is not None
        )
        adapter = SA3Adapter(
            context.make_dit(
                latent_frames=cond.latent_frames,
                seconds_total=duration_s,
                backend=dit_backend,
                prefer_refittable=want_lora,
            ),
            schedule_builder=context.make_schedule_builder(cond, steps),
            device=context.device,
            dtype=context.dtype,
        )

        # Phase-2 refit mirror: engaged only when the selected DiT is a
        # refit-built engine AND its validated manifest exists;
        # otherwise the interim eager swap (D6a) covers LoRA.
        refit_mirror = None
        if want_lora and getattr(adapter.dit, "refittable", False):
            from acestep.engine.sa3_trt_lora import (
                SA3TRTRefitMirror,
                find_refit_manifest,
            )

            manifest = find_refit_manifest(adapter.dit.engine_path)
            if manifest is None:
                logger.warning(
                    "sa3_refit_manifest_missing engine={} — LoRA will use "
                    "the eager-DiT swap; generate the manifest with "
                    "scripts/sa3/gen_sa3_refit_manifest.py",
                    adapter.dit.engine_path.parent.name,
                )
            else:
                try:
                    refit_mirror = SA3TRTRefitMirror(
                        adapter.dit.engine, context.sam.model.model, manifest,
                    )
                except Exception as exc:
                    logger.warning(
                        "sa3_refit_mirror_init_failed error={} — LoRA will "
                        "use the eager-DiT swap", exc,
                    )

        def _prompt_rebuilder(tags: str, steps_now: int):
            # Per-prompt re-conditioning (handle_set_prompt): same fixed
            # duration, fresh T5Gemma capture + a schedule-builder
            # factory closed over the NEW cond's sched_args.
            new_cond = context.prepare_cond(
                prompt=tags, duration=duration_s, steps=steps_now,
            )
            return new_cond, (
                lambda s, _c=new_cond: context.make_schedule_builder(_c, s)
            )

        def _source_encoder(waveform, sample_rate, sample_size):
            # Source re-anchor (handle_swap_source): the same
            # SAME-encode the create path ran for the initial upload —
            # the (sample_rate, waveform) tuple rides prepare_audio's
            # resample, and sample_size (the session's fixed
            # cond.audio_sample_size) pins the latent geometry.
            return context.encode_source(
                (int(sample_rate), waveform), int(sample_size),
            )

        return cls(
            adapter=adapter,
            codec=context.make_codec(backend=codec_backend),
            cond=cond,
            cond_b=cond_b,
            schedule_builder_factory=(
                lambda s: context.make_schedule_builder(cond, s)
            ),
            knob_state=knob_state,
            state=state,
            source_latent_bct=source_latent,
            prompt_rebuilder=_prompt_rebuilder,
            prompt_tags=prompt,
            source_encoder=_source_encoder,
            # D6a: the eager DiT module stays resident on the context
            # even when make_dit returned a TRT wrapper — the interim
            # LoRA fallback swaps to it.
            eager_dit=context.dit,
            model_id=str(context.model_id),
            refit_mirror=refit_mirror,
            **kwargs,
        )

    def _build_pipeline(self, steps: int):
        from acestep.engine.diffusion import DiffusionConfig
        from acestep.engine.stream import StreamPipeline

        self.adapter.schedule_builder = self._schedule_builder_factory(steps)
        config = DiffusionConfig(
            infer_steps=int(steps),
            infer_method="sde" if self._sampler in ("sde", "pingpong") else "ode",
            noise_on_cpu=True,
            dcw_enabled=False,  # ACE wavelet corrector semantics; off for SA3
        )
        return StreamPipeline(
            None, config, pipeline_depth=self._depth, adapter=self.adapter,
        )

    # ---- contract ------------------------------------------------------------

    def capabilities(self) -> Capabilities:
        # loop_band: arm the playback band [A, B] server-side so the windowed
        # renderer pre-fills the seam after A while the playhead finishes the
        # lap near B (pipeline_runner band-awareness). Without it the region
        # at the loop start holds pre-change audio for one window on every
        # restart — the audible "snap back to the old buffer" at the loop
        # point. SA3 uses the shared pipeline_runner, so the band path (and
        # its band-wrap second render) work unchanged via render_window.
        # swap: backend-owned in-place re-anchor (handle_swap_source) — the
        # session's _apply_swap_if_pending dispatches there instead of the
        # ACE prepare_source body, so duration/conditioning stay fixed.
        # render_anchor_queue: batch pad prewarm, drained by the shared
        # pipeline_runner exactly like the scalar stationary anchor.
        # lora: on when the session asked for it (config.lora) AND the
        # create path built the family manager — the same gate shape as
        # ACE's use_lora bit.
        return Capabilities(
            refines_audio=True, loop_band=True, swap=True,
            render_anchor_queue=True,
            lora=bool(self._use_lora and self._lora_mgr is not None),
        )

    def geometry(self) -> AudioGeometry:
        return AudioGeometry(
            sample_rate=DELIVERY_SAMPLE_RATE,
            channels=2,
            chunk_rate_hz=SA3_LATENT_RATE_HZ,
            duration_s=self.playable_duration_s(),
        )

    def knob_specs(self, lora_ids=()) -> list:
        return sa3_knob_specs(loras=list(lora_ids or []))

    def lora_compatible(self, metadata: dict) -> bool:
        return sa3_lora_compatible(metadata, self._model_id)

    def has_pending_refit(self) -> bool:
        """True when ``before_tick`` is about to apply LoRA commands —
        the same pending-queue read as ACE's implementation, so the
        runner pre-covers the enable stall (which can synchronously
        materialize + register 200+ parametrizations)."""
        if not self._use_lora:
            return False
        # Stashed TRT strength refits count too (D6b.5): they apply at
        # the next prepare and stall exactly like a queued enable.
        if self._pending_lora_strengths:
            return True
        if self.state is None:
            return False
        try:
            with self.state._lock:
                return bool(
                    self.state.pending_enable or self.state.pending_disable
                )
        except AttributeError:
            return False

    # ---- LoRA facade (D2 overrides; see notes/SA3_LORA_PLAN.md) ---------

    def _require_lora_manager(self):
        if self._lora_mgr is None:
            raise RuntimeError(
                "SA3 backend has no LoRA manager; the session's "
                "capability gate should have rejected this command"
            )
        return self._lora_mgr

    def lora_available(self) -> bool:
        return self._lora_mgr is not None

    def register_lora(self, path: str) -> str:
        return self._require_lora_manager().register_lora(path)

    def prewarm_lora(self, lora_id: str):
        return self._require_lora_manager().prewarm_lora(lora_id)

    def list_loras(self) -> list:
        return self._lora_mgr.list_loras() if self._lora_mgr else []

    def enable_lora(self, lora_id: str, strength=None) -> None:
        """Transactional manager enable + the backend-side effects:
        the D6a DiT swap and, for conditioner-targeting files, the D5
        cond-bundle rebuild. Runs on the runner thread inside the
        pending-drain rendezvous (session._apply_lora_pending)."""
        mgr = self._require_lora_manager()
        # D5: conditioner mutation is mutually exclusive with
        # conditioner execution (prompt-swap captures on the command
        # thread). DiT-only files pay this lock for microseconds.
        with self._conditioner_lock:
            mgr.enable_lora(lora_id, strength=strength)
        self._sync_dit_for_lora()
        if self._refit_mirror is not None:
            self._refit_mirror.sync(reason="enable_lora")
        if mgr.touches_conditioner(lora_id):
            self._rebuild_conditioning_after_lora("enable_lora", lora_id)

    def disable_lora(self, lora_id: str) -> None:
        mgr = self._require_lora_manager()
        # Read the conditioner flag BEFORE disable drops the staged
        # payload.
        touched = mgr.touches_conditioner(lora_id)
        with self._conditioner_lock:
            mgr.disable_lora(lora_id)
        self._sync_dit_for_lora()
        if self._refit_mirror is not None:
            # The mirror's dirty-set pushes this adapter's base weights
            # back — the engine must not keep the merged values.
            self._refit_mirror.sync(reason="disable_lora")
        self._pending_lora_strengths.pop(lora_id, None)
        if touched:
            self._rebuild_conditioning_after_lora("disable_lora", lora_id)

    def set_lora_strength(self, lora_id: str, strength: float) -> None:
        mgr = self._require_lora_manager()
        with self._conditioner_lock:
            mgr.set_lora_strength(lora_id, strength)
        if self._refit_mirror is not None:
            # Direct facade calls sync immediately; the knob-driven path
            # batches instead (_apply_pending_lora_strengths calls the
            # manager directly and syncs once for the whole stash).
            self._refit_mirror.sync(reason="set_lora_strength")
        # A strength change on a conditioner-targeting LoRA changes the
        # conditioner's output too — the cached cond bundle must follow.
        # (Per-tick slider sweeps on such files pay a T5Gemma capture
        # per applied step; the common DiT-only case skips this
        # entirely.)
        if mgr.touches_conditioner(lora_id):
            self._rebuild_conditioning_after_lora("set_lora_strength", lora_id)

    def _apply_pending_lora_strengths(self) -> None:
        """Drain the D6b.5 stash: apply each strength through the facade
        (manager buffer write + conditioner rebuild where the file
        targets it), then ONE mirror sync covers the whole batch."""
        if not self._pending_lora_strengths:
            return
        pending, self._pending_lora_strengths = (
            self._pending_lora_strengths, {},
        )
        mgr = self._lora_mgr
        for lora_id, strength in pending.items():
            try:
                with self._conditioner_lock:
                    mgr.set_lora_strength(lora_id, strength)
                if mgr.touches_conditioner(lora_id):
                    self._rebuild_conditioning_after_lora(
                        "set_lora_strength", lora_id,
                    )
            except Exception as exc:
                logger.exception(
                    "sa3_lora_pending_strength_failed id={} error={}",
                    lora_id, exc,
                )
        if self._refit_mirror is not None:
            self._refit_mirror.sync(reason="strength")

    def _sync_dit_for_lora(self) -> None:
        """D6a interim: LoRA weights live on the eager torch modules, so
        a TRT-DiT session swaps to the eager DiT while any LoRA is
        active and back when the last disables. Loud log both ways;
        no-op for sessions already on the eager DiT — and for refit-
        mirror sessions, where the engine itself carries the LoRA
        weights (the Phase-2 endgame) and must NOT be swapped away."""
        if self._refit_mirror is not None:
            return
        if self.adapter is None or self._dit_eager is None:
            return
        if self._dit_accel is self._dit_eager:
            return  # eager session; nothing to swap
        active = bool(self._lora_mgr is not None and self._lora_mgr.has_active_loras)
        if active and self.adapter.dit is not self._dit_eager:
            self.adapter.dit = self._dit_eager
            logger.warning(
                "sa3_dit_swap reason=lora_active dit=eager (TRT engine "
                "cannot serve LoRA weights until the refit path lands)"
            )
        elif not active and self.adapter.dit is not self._dit_accel:
            self.adapter.dit = self._dit_accel
            logger.info("sa3_dit_swap reason=lora_inactive dit=accelerated")

    def _rebuild_conditioning_after_lora(self, op: str, lora_id: str) -> None:
        """D5: re-run the conditioner (through the exact prompt-swap
        path, which takes the conditioner lock itself) after a
        conditioner-mutating LoRA change commits, so the cached
        cond_bundle reflects the mutated weights."""
        if self._prompt_rebuilder is None:
            logger.warning(
                "sa3_lora_cond_rebuild_skipped op={} id={} "
                "reason=no_prompt_rebuilder", op, lora_id,
            )
            return
        tags = (
            getattr(self.state, "prompt_text", None)
            if self.state is not None else None
        )
        tags_b = (
            getattr(self.state, "prompt_text_b", None)
            if self.state is not None else None
        )
        if not tags:
            # Directly-constructed backends without session state: fall
            # back to the newest tags the cond history recorded.
            with self._control_lock:
                tags = self._cond_history[-1][2]
        if not tags:
            logger.warning(
                "sa3_lora_cond_rebuild_skipped op={} id={} reason=no_tags",
                op, lora_id,
            )
            return
        t0 = time.perf_counter()
        self.handle_set_prompt(tags, tags_b=tags_b)
        logger.info(
            "sa3_lora_cond_rebuilt op={} id={} rebuild_ms={:.1f}",
            op, lora_id, (time.perf_counter() - t0) * 1000,
        )

    def playable_duration_s(self):
        return self._cond.audio_sample_size / SA3_SAMPLE_RATE

    def read_knobs(self) -> dict:
        return self.knob_state.get_all_values()

    def rebuild_imminent(self, knobs: dict) -> bool:
        steps_changed = (
            int(knobs.get("steps_override", self._steps)) != self._steps
        )
        return steps_changed or self._detect_pending_lora_strengths(knobs)

    def _detect_pending_lora_strengths(self, knobs: dict) -> bool:
        """D6b.5: under the refit mirror a strength change is a full
        engine refit, and ``rebuild_imminent`` is the sanctioned
        announcement point for knob-driven stalls (called once per tick,
        after the knob read, before produce). Deltas are stashed here
        and applied by the SAME tick's ``_prepare_tick`` — after the
        runner has pre-covered. Eager mode returns False: its strength
        writes are ~5 ms buffer updates applied inline."""
        if (
            self._refit_mirror is None
            or not self._use_lora
            or self._lora_mgr is None
        ):
            return False
        for desc in self._lora_mgr.list_loras():
            if desc.state != "enabled":
                continue
            try:
                v = float(knobs.get(f"lora_str_{desc.id}", desc.strength))
            except (TypeError, ValueError):
                continue
            if abs(v - desc.strength) > 0.02:
                self._pending_lora_strengths[desc.id] = v
        return bool(self._pending_lora_strengths)

    # ---- control (universal): per-prompt re-conditioning ------------------------

    def handle_set_prompt(self, tags: str, *, tags_b: Optional[str] = None) -> None:
        """Re-run ``prepare_cond`` for ``tags`` (and ``tags_b`` when it
        differs) and swap the conditioning captures (the session's
        backend control hook — plan §2: prompt is the universal
        control). Per-prompt, OUTSIDE the hot loop: T5Gemma captures on
        the dispatcher thread, then GIL-atomic reference swaps;
        in-flight slots finish on the old bundle, the next ``submit``
        carries the new one. An absent/empty/identical ``tags_b``
        resets B to A (the ACE ``set_prompt`` convention:
        ``cond_pair_b = cond_pair``), so a stale B can't linger behind
        the blend knob.
        """
        if self._prompt_rebuilder is None:
            raise RuntimeError(
                "SA3Backend was constructed without a prompt_rebuilder; "
                "handle_set_prompt requires the from_context assembly"
            )
        t0 = time.perf_counter()
        # Conditioner EXECUTION under the D5 lock: a concurrent LoRA
        # mutation of the conditioner's parametrizations (runner
        # rendezvous) must not interleave with this capture.
        with self._conditioner_lock:
            cond, sched_factory = self._prompt_rebuilder(tags, self._steps)
        rebuild_ms = (time.perf_counter() - t0) * 1000
        if int(cond.latent_frames) != int(self._cond.latent_frames):
            # Duration is fixed for the session lifetime, so the latent
            # geometry must hold: a mismatch would desync the ring
            # buffer, the source anchor, and the cond bundle.
            raise ValueError(
                f"sa3 prompt swap changed latent_frames "
                f"({self._cond.latent_frames} -> {cond.latent_frames}); "
                f"duration is fixed per session"
            )
        if tags_b and tags_b != tags:
            with self._conditioner_lock:
                cond_b, _ = self._prompt_rebuilder(tags_b, self._steps)
            if int(cond_b.latent_frames) != int(cond.latent_frames):
                raise ValueError(
                    f"sa3 prompt-B swap changed latent_frames "
                    f"({cond.latent_frames} -> {cond_b.latent_frames}); "
                    f"duration is fixed per session"
                )
        else:
            cond_b = cond
        # Publish the whole new conditioning state atomically w.r.t. the
        # runner (the GPU rebuild above ran lock-free). Without this the
        # runner can read a half-swapped state — e.g. iterate
        # _cond_history mid append+truncate, or submit the new bundle
        # against the stale schedule cache.
        with self._control_lock:
            self._schedule_builder_factory = sched_factory
            self.adapter.schedule_builder = sched_factory(self._steps)
            # The pipeline caches schedules per denoise value; the builder
            # swap changes what build_schedule returns for the same key.
            # (Same duration means the same effective_seq_len today, so
            # this is currently belt-and-braces — but the cache key carries
            # no prompt identity, so correctness must not depend on that.)
            self.pipeline.invalidate_schedule_cache()
            self._cond = cond
            self._cond_b = cond_b
            self._active_bundle = self._blend_bundles(self._blend)
            # Emerged-generation labeling (see __init__): the new bundle gets
            # the next cond epoch; keep a short identity history so latents
            # still in flight on the OLD bundle stay attributable.
            self._cond_epoch += 1
            self._cond_history.append((cond.cond_bundle, self._cond_epoch, tags))
            del self._cond_history[:-4]
        logger.info(
            "sa3_prompt_applied tags={!r} tags_b={!r} cond_epoch={} "
            "rebuild_ms={:.1f}",
            tags, tags_b, self._cond_epoch, rebuild_ms,
        )

    def handle_swap_source(self, waveform, sample_rate) -> None:
        """Re-anchor the session on a new source (the session's
        backend-owned ``swap_source`` hook, dispatched from
        ``_apply_swap_if_pending`` on the runner thread). SAME-encodes
        ``waveform`` at the session's FIXED latent geometry
        (``cond.audio_sample_size`` — prepare_audio pads/truncates, so
        any upload length lands on the same [1, T, 256] anchor shape),
        then swaps the anchor and drops the feedback latent ring (its
        taps are covers of the OLD source; blending them into the new
        anchor would smear the previous song across the swap).

        In-flight pipeline slots finish on the old anchor — the swap
        emerges within pipeline depth, exactly the handle_set_prompt
        convention. At ``sa3_denoise`` = 1.0 slot init is pure noise and
        the anchor only re-enters when denoise drops below 1 (or via
        x0_target), matching create-time semantics.
        """
        if self._source_encoder is None:
            raise RuntimeError(
                "SA3Backend was constructed without a source_encoder; "
                "handle_swap_source requires the from_context assembly"
            )
        with self._control_lock:
            sample_size = int(self._cond.audio_sample_size)
        t0 = time.perf_counter()
        latent_bct = self._source_encoder(waveform, sample_rate, sample_size)
        encode_ms = (time.perf_counter() - t0) * 1000
        latent_btc = latent_bct.movedim(1, 2).contiguous()
        # Publish atomically w.r.t. the command thread's conditioning
        # swaps (same lock discipline as handle_set_prompt); the runner
        # reads the anchor on its own thread, which is also the thread
        # calling this hook.
        with self._control_lock:
            self._source_latent_btc = latent_btc
            self._latent_history.clear()
        logger.info(
            "sa3_source_swapped samples={} sample_rate={} encode_ms={:.1f}",
            int(waveform.shape[-1]), int(sample_rate), encode_ms,
        )

    def handle_set_prompt_blend(self, value: float) -> None:
        """Crossfade the live conditioning between the A and B captures
        (the session's ``set_prompt_blend`` backend hook). Per-token
        slerp of the T5Gemma cross-attn conditioning — cheap tensor
        math on the command thread, then one GIL-atomic bundle swap;
        in-flight slots finish on their submitted bundle.
        """
        v = max(0.0, min(1.0, float(value)))
        # Blend under the lock: _blend_bundles reads _cond/_cond_b, which a
        # concurrent handle_set_prompt swaps under this same lock. Computing
        # it here (rather than before acquiring) keeps blend and prompt-swap
        # correct even if they ever run off different threads. _control_lock
        # is non-reentrant and _blend_bundles never re-acquires it.
        with self._control_lock:
            self._blend = v
            self._active_bundle = self._blend_bundles(v)

    def _blend_bundles(self, v: float) -> dict:
        """The active cond bundle for blend value ``v``: A verbatim at
        0, B verbatim at 1 (endpoint identity keeps the TRT wrapper's
        id()-keyed staging cache warm on the common path), otherwise a
        NEW dict with the cross-attn conditioning slerped and the token
        masks unioned. Slerp runs in float32 (norm math degrades in
        bf16) and degenerates to linear per token where either side is
        zero-padding, so A-only / B-only token positions survive at
        scaled strength under the unioned mask.

        Everything else (padding_mask, cfg/apg scalars, local_add_cond)
        comes from A: both captures share the session's fixed duration,
        which is what those encode.
        """
        a = self._cond.cond_bundle
        b = self._cond_b.cond_bundle
        if b is a or v <= 0.001:
            return a
        if v >= 0.999:
            return b
        ca, cb = a["cross_attn_cond"], b["cross_attn_cond"]
        if ca.shape != cb.shape:
            # Both captures are max-length-padded by the conditioner;
            # a shape mismatch means the captures disagree about more
            # than token content — refuse rather than mis-blend.
            raise ValueError(
                f"sa3 prompt blend shape mismatch: A {tuple(ca.shape)} "
                f"vs B {tuple(cb.shape)}"
            )
        blended = dict(a)
        blended["cross_attn_cond"] = slerp(
            ca.float(), cb.float(), v,
        ).to(ca.dtype)
        blended["cross_attn_mask"] = torch.maximum(
            a["cross_attn_mask"], b["cross_attn_mask"],
        )
        return blended

    # ---- produce hooks ---------------------------------------------------------

    def _prepare_tick(self, knobs: dict, ctx: TickContext) -> dict:
        # Schedule warp: hot-applied, but cache-coupled — the pipeline
        # caches schedules per denoise value, so a changed alpha must
        # invalidate or already-seen denoise values keep the old warp.
        shift = float(knobs.get("sa3_shift", 1.0))
        if abs(shift - float(self.adapter.shift_alpha)) > 1e-3:
            # Atomic w.r.t. a concurrent prompt swap, which also retargets
            # the schedule builder + invalidates this cache.
            with self._control_lock:
                self.adapter.shift_alpha = shift
                self.pipeline.invalidate_schedule_cache()

        # Per-LoRA live strength (the ACE per-tick convention): iterate
        # the catalog so the active set can change at runtime; strength
        # only flows for ENABLED entries, gated by the shared 0.02
        # slider-delta threshold. Eager mode applies inline (a buffer
        # write, no recompute). Refit-mirror mode instead applies the
        # deltas rebuild_imminent stashed THIS tick, after the runner
        # pre-covered — the D6b.5 route: never a refit unannounced.
        if self._use_lora and self._lora_mgr is not None:
            if self._refit_mirror is not None:
                self._apply_pending_lora_strengths()
            else:
                for desc in self._lora_mgr.list_loras():
                    if desc.state != "enabled":
                        continue
                    try:
                        lora_str = float(
                            knobs.get(f"lora_str_{desc.id}", desc.strength),
                        )
                    except (TypeError, ValueError):
                        continue
                    if abs(lora_str - desc.strength) > 0.02:
                        self.set_lora_strength(desc.id, lora_str)

        # Source-lock strength rides the shared override so a strength
        # bump engages the blend on in-flight slots submitted while it
        # was 0 — the ACE runner's exact per-tick convention.
        x0_str = float(knobs.get("x0_target", 0.0))
        if self._source_latent_btc is not None:
            self.pipeline.set_shared_curve("x0_target_strength", x0_str)

        try:
            fb_depth_raw = float(knobs.get("feedback_depth", 1.0))
        except (TypeError, ValueError):
            fb_depth_raw = 1.0
        return {
            "denoise": float(knobs.get("sa3_denoise", 1.0)),
            "seed": int(knobs.get("seed", self._default_seed)),
            "steps": int(knobs.get("steps_override", self._steps)),
            "shift": shift,
            "x0_target": x0_str,
            "feedback": float(knobs.get("feedback", 0.0)),
            "feedback_depth": max(
                1, min(MAX_FEEDBACK_DEPTH, int(round(fb_depth_raw))),
            ),
        }

    def _generate(self, prep: dict):
        from acestep.engine.stream import SlotRequest

        if prep["steps"] != self._steps:
            # Step-count change: schedules are (steps+1,)-shaped, so
            # the ring buffer rebuilds — the SA3 analog of ACE's
            # rebuild-signature stall, pre-covered via rebuild_imminent.
            self._steps = prep["steps"]
            self.pipeline = self._build_pipeline(self._steps)

        # Feedback delay-tap (the ACE mechanic, verbatim): blend the
        # tapped past latent into the source anchor at slot init. If
        # history is shorter than the requested depth (early ticks),
        # fall back to the oldest available tap rather than disabling
        # feedback. Audible only at sa3_denoise < 1 — at 1.0 slot init
        # is pure noise and source_latents never enters.
        source = self._source_latent_btc
        if (
            prep["feedback"] > 0.0
            and self._latent_history
            and source is not None
        ):
            tap_idx = min(
                prep["feedback_depth"] - 1, len(self._latent_history) - 1,
            )
            fb_latent = self._latent_history[tap_idx]
            method = (
                getattr(self.state, "interp_feedback", "slerp")
                if self.state is not None else "slerp"
            )
            source = INTERPOLATIONS[method](
                source, fb_latent, prep["feedback"],
            )

        # Snapshot the conditioning a prompt swap publishes atomically
        # (active bundle + its latent geometry) so this request can't pair
        # a new bundle with stale frames mid-swap. submit()/tick() below
        # run lock-free.
        with self._control_lock:
            aux_cond = self._active_bundle
            latent_frames = self._cond.latent_frames

        self.pipeline.submit(SlotRequest(
            seed=prep["seed"],
            denoise=prep["denoise"],
            source_latents=source,
            # The morph target stays the clean anchor (not the
            # feedback-blended source), matching ACE: x0_target is a
            # source LOCK, feedback is deliberately upstream of it.
            # Attached whenever an anchor exists so a strength bump via
            # the shared override engages on in-flight slots; the
            # request field carries the live value so a fresh pipeline
            # (steps rebuild) is correct before the next prepare
            # re-establishes the shared override.
            x0_target=self._source_latent_btc,
            x0_target_strength=prep["x0_target"],
            aux_cond=aux_cond,
            latent_frames=latent_frames,
            # Deterministic pingpong: identical requests must replay the
            # same trajectory or advancing windows splice different
            # realizations (incoherent audio). See SlotRequest.
            sde_noise_seeded=True,
        ))
        # With LoRAs active, wrap the tick in parametrize.cached():
        # without it, every weight ACCESS recomputes W + scaled delta,
        # and the measured overhead is +49% per step at 1 LoRA / +114%
        # at 3; under cached() each parametrized weight is computed once
        # per tick and the 3-LoRA overhead collapses to ~+15% (Phase 0.5
        # entry benchmark — this wrap is a requirement, not a tweak).
        # The cache is scoped to this context and strength changes only
        # land between ticks, so it can never serve a stale weight.
        if self._lora_mgr is not None and self._lora_mgr.has_active_loras:
            import torch.nn.utils.parametrize as parametrize

            with parametrize.cached():
                latent = self.pipeline.tick()
        else:
            latent = self.pipeline.tick()  # engine-layout [1, T, 256] | None
        if latent is not None:
            # The request this latent was generated from (valid only
            # right after a finishing tick — see StreamPipeline).
            self._emerged_request = getattr(
                self.pipeline, "last_finished_request", None,
            )
        return latent

    def _cond_meta_for(self, bundle) -> tuple:
        """(epoch, tags) for a request's aux_cond, by identity."""
        # Snapshot under the lock: a concurrent prompt swap does a
        # non-atomic append + truncate on this list.
        with self._control_lock:
            history = tuple(self._cond_history)
        for b, epoch, tags in history:
            if b is bundle:
                return epoch, tags
        return None, None

    def _after_produce(self, prep: dict, result_latent, is_fresh: bool) -> None:
        self.last_denoise = prep["denoise"]
        self._last_prep = prep
        if is_fresh:
            # appendleft so latent_history[0] is the most recent;
            # tap_idx = depth-1 reads "N ticks back" (ACE convention).
            self._latent_history.appendleft(result_latent.detach().clone())
            # Pin the decode RNG to the seed of the request that produced
            # this latent (the emerged request survives DiT-pause "reuse"
            # ticks, so the pairing holds there too). prep carries the
            # same int in production; it covers non-int request seeds.
            req_seed = getattr(self._emerged_request, "seed", None)
            self._decode_seed = (
                int(req_seed) if isinstance(req_seed, int)
                else int(prep["seed"])
            )
        if not is_fresh or self.state is None:
            return
        req = self._emerged_request
        if req is None:
            return
        # Stamp the EMERGED request's params (what this latent was
        # actually generated with) before the runner renders + publishes
        # this tick's slice, so the accompanying params echo describes
        # the audio it rides with. The plain knob keys stamped by
        # on_fresh_generation reflect prepare-time values instead (the
        # ACE-shaped convention) and stay as-is.
        epoch, tags = self._cond_meta_for(req.aux_cond)
        p = self.state.params
        p["gen_sa3_denoise"] = round(float(req.denoise), 4)
        if req.seed is not None and not isinstance(req.seed, list):
            p["gen_seed"] = int(req.seed)
        p["gen_cond_epoch"] = epoch
        p["gen_prompt"] = tags
        marker = (p["gen_sa3_denoise"], epoch)
        if marker != self._emerged_marker:
            self._emerged_marker = marker
            logger.info(
                "sa3_gen_emerged denoise={} cond_epoch={} tags={!r}",
                p["gen_sa3_denoise"], epoch, tags,
            )

    # ---- rendering -------------------------------------------------------------

    def _rendered_audio(self, latent_btc: torch.Tensor):
        """Full decode + delivery resample, cached per latent identity."""
        if self._rendered_for is latent_btc and self._rendered_48k is not None:
            return self._rendered_48k
        import torchaudio

        t0 = time.perf_counter()
        audio_ct = self.codec.decode_full(
            latent_btc.movedim(1, 2), decode_seed=self._decode_seed,
        )
        # The decode boundary (round_3 decision 2): one whole-window
        # resample per generation, so window slices share one filter
        # pass and seams can't come from per-slice resampling.
        audio_48 = torchaudio.functional.resample(
            audio_ct.float(), SA3_SAMPLE_RATE, DELIVERY_SAMPLE_RATE,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.last_dec_ms += (time.perf_counter() - t0) * 1000
        self._rendered_48k = audio_48.clamp(-1, 1).cpu().numpy().T  # [N, C]
        self._rendered_for = latent_btc
        return self._rendered_48k

    def render_window(self, t_start_s: float):
        decode_src = (
            self._current_result if self._current_result is not None
            else self._last_result_latent
        )
        if decode_src is None:
            return None
        if self._windowed_codec:
            return self._render_window_via_codec(decode_src, t_start_s)
        audio = self._rendered_audio(decode_src)
        n = int(round(self.vae_window * DELIVERY_SAMPLE_RATE))
        start = int(round(t_start_s * DELIVERY_SAMPLE_RATE))
        start = max(0, min(start, max(0, audio.shape[0] - n)))
        return AudioChunk(pcm=audio[start:start + n], start_sample=start)

    def _render_window_via_codec(self, latent_btc: torch.Tensor, t_start_s: float):
        """Windowed-codec render (SAME-L / medium): decode ONLY a small
        latent window around the target, then resample that window.

        44.1k↔48k bookkeeping uses the exact 147:160 ratio. The decode
        request carries a 588-sample (= 640 at 48 k) guard margin on
        each side so the resampler's filter edges land outside the kept
        slice; the runner's 25 ms crossfade against the live buffer
        covers the (deterministic) window seams, exactly as it does for
        ACE's windowed VAE decode.
        """
        import torchaudio

        n48 = int(round(self.vae_window * DELIVERY_SAMPLE_RATE))
        dur48 = int(round((self.playable_duration_s() or 0.0) * DELIVERY_SAMPLE_RATE))
        start48 = int(round(t_start_s * DELIVERY_SAMPLE_RATE))
        start48 = max(0, min(start48, max(0, dur48 - n48)))

        m44 = 588                                # guard margin; 588*160/147 == 640
        start44 = (start48 * SA3_SAMPLE_RATE) // DELIVERY_SAMPLE_RATE
        n44 = -(-n48 * 147 // 160)               # ceil to cover n48 after resample
        lo44 = max(0, start44 - m44)
        lead44 = start44 - lo44
        total44 = lead44 + n44 + m44

        t0 = time.perf_counter()
        audio_ct = self.codec.decode_window(
            latent_btc.movedim(1, 2), lo44, total44,
        )
        audio48 = torchaudio.functional.resample(
            audio_ct.float(), SA3_SAMPLE_RATE, DELIVERY_SAMPLE_RATE,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.last_dec_ms += (time.perf_counter() - t0) * 1000

        lead48 = (lead44 * DELIVERY_SAMPLE_RATE) // SA3_SAMPLE_RATE
        pcm48 = audio48[:, lead48:lead48 + n48]
        if pcm48.shape[-1] < n48:
            pcm48 = torch.nn.functional.pad(pcm48, (0, n48 - pcm48.shape[-1]))
        pcm = pcm48.clamp(-1, 1).cpu().numpy().T  # [N, C]
        return AudioChunk(pcm=pcm, start_sample=start48)

    def render_full(self):
        if self._current_result is None:
            return None
        return AudioChunk(
            pcm=self._rendered_audio(self._current_result), start_sample=0,
        )

    # ---- teardown ----------------------------------------------------------------

    def close(self) -> None:
        """Session teardown (called by StreamingSession.close). The SA3
        torch model is process-cached and shared with every later
        session, so the LoRA manager must strip every parametrization
        this session installed — the next session gets a pristine model
        (plan D4 teardown; bitwise-validated by Phase 0.5 check 4)."""
        if self._lora_mgr is not None:
            try:
                with self._conditioner_lock:
                    self._lora_mgr.close()
            except Exception as exc:
                logger.warning("sa3_lora_teardown_raised error={}", exc)
        # The refit mirror holds an IRefitter over the exclusively-owned
        # engine; dropping both with the session IS the base-weight
        # rollback (the mutated engine can never reach another session
        # because refittable engines are never process-cached).
        self._refit_mirror = None
        self._pending_lora_strengths.clear()

    # ---- bookkeeping -------------------------------------------------------------

    def on_fresh_generation(self, knobs: dict) -> None:
        if self.state is None:
            return
        p = self.state.params
        p["num_gens"] = p.get("num_gens", 0) + 1
        p["tick_ms"] = self.last_tick_ms
        p["dec_ms"] = self.last_dec_ms
        prep = getattr(self, "_last_prep", None)
        if prep:
            p["sa3_denoise"] = round(prep["denoise"], 2)
            p["seed"] = prep["seed"]
            p["steps_override"] = prep["steps"]
            p["sa3_shift"] = round(prep["shift"], 2)
            p["x0_target"] = round(prep["x0_target"], 2)
            p["feedback"] = round(prep["feedback"], 2)
            p["feedback_depth"] = prep["feedback_depth"]
        p["_prompt"] = getattr(self.state, "prompt_text", "")
