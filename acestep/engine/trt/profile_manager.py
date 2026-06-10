"""Profile-aware TRT engine swapping.

Holds at most one ``decoder`` and one ``vae_encode`` engine in VRAM at
a time, and swaps them when the requested duration profile changes.
Each registered profile (60s / 120s / 240s; see :mod:`acestep.paths`)
reserves several GB of workspace at TRT context creation, so holding
all profiles simultaneously is not viable — unloading the previous
engine before loading the next is mandatory.

Out of scope: ``vae_decode``. The realtime path runs with
``vae_window > 0``, which pins ``vae_decode`` to the windowed engine
(``vae_decode_fp16_1s_fixed``); that engine's fixed 1 s shape is
profile-independent. Optional dreamvae / fast_vae substitutions are
also applied at session build and survive every profile swap. The
manager intentionally leaves the vae_decode slot alone so it can't
clobber either of those.

Typical usage from a streaming session::

    mgr = TRTProfileManager(
        decoder_backend="tensorrt",
        vae_backend="tensorrt",
        checkpoint="acestep-v15-turbo",
        device="cuda",
    )

    initial_paths, picked_dur = mgr.resolve(initial_duration_s)
    session = Session(..., trt_engines=initial_paths)
    mgr.bind(session.handler._diffusion_engine, initial_paths, picked_dur)

    # On song swap (called on the runner thread before prepare_source):
    new_paths, picked_dur = mgr.ensure_profile(new_duration_s)
    # If new_paths != prior, decoder + vae_encode have already been
    # swapped in place; LoRAs that were ENABLED have been re-enabled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from loguru import logger
import torch

from acestep.paths import available_trt_engines


@dataclass
class _LoRASnapshot:
    """ENABLED LoRA captured before a decoder swap."""
    lora_id: str
    strength: float


class TRTProfileLoadError(RuntimeError):
    """Raised by :class:`TRTProfileManager` when a profile swap fails at
    engine-load time — typically because ``create_execution_context()``
    returns None under CUDA-OOM workspace pressure (the 240 s
    ``vae_encode`` profile reserves ~16 GiB), or because
    ``DiffusionEngine.load_trt_engine`` raises for the same reason.

    Distinct from :class:`acestep.paths.EngineNotBuiltError` (the engine
    file is missing) and from runtime decode/encode failures (which fire
    *after* a successful load).

    By the time this is raised the prior engines have already been
    evicted and the new ones failed to load — the profile manager's
    ``_loaded_paths`` / ``_loaded_dur`` are cleared to reflect "nothing
    loaded." Callers should treat the session as unrecoverable and close
    cleanly (sending a ``swap_failed`` to the client is the established
    pattern, mirroring how ``EngineNotBuiltError`` is handled in
    ``demos/realtime_motion_graph_web/backend.py``).
    """

    def __init__(
        self,
        *,
        component: str,
        engine_path: str,
        cause: BaseException,
    ) -> None:
        self.component = component
        self.engine_path = engine_path
        self.cause = cause
        super().__init__(
            f"TRT {component} engine load failed after profile swap eviction: "
            f"{engine_path} ({type(cause).__name__}: {cause}). "
            f"GPU may be under memory pressure; session is unrecoverable."
        )


class TRTProfileManager:
    """Owns the live decoder + vae_encode slots and swaps between profiles.

    The vae_decode slot is intentionally not managed here — see the
    module docstring.
    """

    def __init__(
        self,
        *,
        decoder_backend: str,
        vae_backend: str,
        checkpoint: str = "acestep-v15-turbo",
        device: str = "cuda",
    ):
        self._decoder_tensorrt = decoder_backend == "tensorrt"
        self._vae_tensorrt = vae_backend == "tensorrt"
        self._checkpoint = checkpoint
        self._device = torch.device(device)

        self._diffusion_engine = None
        self._loaded_dur: Optional[float] = None
        self._loaded_paths: dict[str, str] = {}

    @property
    def loaded_duration_s(self) -> Optional[float]:
        """Max-duration (seconds) of the currently-loaded profile, or
        ``None`` before the first :meth:`bind` call."""
        return self._loaded_dur

    @property
    def loaded_paths(self) -> dict[str, str]:
        """Engine paths currently active. Empty before :meth:`bind`."""
        return dict(self._loaded_paths)

    def _needs(self) -> tuple[str, ...]:
        """Engine keys this session actually consumes.

        Drives :func:`available_trt_engines` so a mixed-backend setup
        (e.g. tensorrt decoder + eager VAE) doesn't disqualify a profile
        for missing VAE engines that won't be loaded anyway.
        """
        keys: list[str] = []
        if self._decoder_tensorrt:
            keys.append("decoder")
        if self._vae_tensorrt:
            keys.extend(["vae_encode", "vae_decode"])
        return tuple(keys)

    def resolve(self, duration_s: float) -> tuple[dict[str, str], float]:
        """Pick the smallest built profile that fits ``duration_s``.

        Returns ``(paths, picked_dur)`` exactly as
        :func:`available_trt_engines`. Pure: does not swap engines or
        update internal state. Used by callers building the initial
        Session before :meth:`bind`.
        """
        return available_trt_engines(
            duration_s,
            needs=self._needs(),
            checkpoint=self._checkpoint,
        )

    def bind(
        self,
        diffusion_engine,
        paths: dict[str, str],
        picked_dur: float,
    ) -> None:
        """Record the profile that the live session was built with.

        Called once after Session construction so subsequent
        :meth:`ensure_profile` calls can compare against the loaded
        profile and skip the swap when the same profile would be
        picked. ``paths`` should be the dict returned from
        :meth:`resolve`; ``picked_dur`` the second tuple element.

        Also evicts any stale ``vae_encode`` engines left in the
        module-level cache by a previous session's profile manager.
        Without this, the very first VAE-encode call of this session
        could be served by a leaked prior-session engine (see the
        long comment in :meth:`_swap` for the full failure mode);
        the swap-time purge there only fires if the user actually
        triggers a swap.
        """
        self._diffusion_engine = diffusion_engine
        self._loaded_dur = float(picked_dur)
        self._loaded_paths = dict(paths)
        if self._vae_tensorrt:
            self._purge_stale_vae_encode_cache(paths.get("vae_encode"))

    @staticmethod
    def _purge_stale_vae_encode_cache(active_enc: Optional[str]) -> None:
        """Evict every cached ``vae_encode_*`` / ``dreamvae_encode_*``
        engine that isn't ``active_enc``. See :meth:`_swap` for why.

        Scoped to encode only — vae_decode engines have a different
        lifecycle (session.__init__ owns the windowed swap) and
        evicting one mid-session would break the live runner.
        """
        import os
        from acestep.nodes.vae_nodes import _evict_trt_vae, _trt_vae_cache

        encode_prefixes = ("vae_encode_", "dreamvae_encode_")
        active_abs = os.path.abspath(active_enc) if active_enc else None
        for cached_path in list(_trt_vae_cache.keys()):
            basename = os.path.basename(cached_path).lower()
            if not basename.startswith(encode_prefixes):
                continue
            if active_abs is not None and os.path.abspath(cached_path) == active_abs:
                continue
            _evict_trt_vae(cached_path)

    def ensure_walk_profile(
        self,
        *,
        walk_window_s: float,
        source_duration_s: float,
    ) -> tuple[dict[str, str], float]:
        """Ensure walk-mode engines: decoder + vae_decode pinned to
        ``walk_window_s``.

        Historically this also resized ``vae_encode`` to
        ``source_duration_s`` (the runner only sees walk_window_s of
        latent at a time, but VAE-encode still ingests the full song
        once at load). The encode path now chunks inputs longer than
        the engine's max profile shape (see ``acestep/nodes/vae_nodes``)
        and :func:`acestep.paths.available_trt_engines` pins vae_encode
        to the smallest built engine for every duration — so the walk
        profile's own engine set already serves any source length, and
        resolving a larger profile here would wrongly require a bigger
        *decoder* to exist on disk for long sources.

        ``source_duration_s`` is kept for API stability; it no longer
        affects engine choice.

        Returns ``(paths, picked_dur)`` where ``picked_dur`` reflects
        the decoder/vae_decode profile (i.e. ``walk_window_s``).
        """
        del source_duration_s  # encode is duration-independent now
        return self.ensure_profile(walk_window_s)

    def ensure_profile(
        self, duration_s: float,
    ) -> tuple[dict[str, str], float]:
        """Swap engines if the picked profile differs from the loaded one.

        Returns ``(paths, picked_dur)`` reflecting the engines now live
        in VRAM. When the picked profile matches the loaded profile,
        no GPU work happens and the returned paths equal the previously
        bound set.

        Must be called on the same thread that drives inference: the
        decoder swap touches ``_diffusion_engine`` state that is shared
        with the streaming pipeline, and the VAE cache eviction frees
        device buffers that the runtime is otherwise free to reuse.
        """
        if self._diffusion_engine is None and self._decoder_tensorrt:
            raise RuntimeError(
                "TRTProfileManager.ensure_profile called before bind(); "
                "build the Session first and call bind() with its "
                "DiffusionEngine."
            )

        target_paths, picked_dur = self.resolve(duration_s)

        if self._loaded_dur is not None and picked_dur == self._loaded_dur:
            # Same profile; no swap needed. Return the bound paths
            # rather than freshly resolved ones so callers see the same
            # dict identity across same-profile calls.
            return dict(self._loaded_paths), picked_dur

        prior_paths = dict(self._loaded_paths)
        prior_dur = self._loaded_dur
        logger.info(
            "TRT profile swap: {}s -> {}s",
            int(prior_dur) if prior_dur is not None else "?",
            int(picked_dur),
        )

        self._swap(prior_paths=prior_paths, target_paths=target_paths)

        self._loaded_dur = float(picked_dur)
        self._loaded_paths = dict(target_paths)
        return dict(target_paths), picked_dur

    def _swap(
        self,
        *,
        prior_paths: dict[str, str],
        target_paths: dict[str, str],
    ) -> None:
        """Tear down the old engines, load the new ones in place.

        Order matters for VRAM:
          1. Snapshot ENABLED LoRAs from the live decoder.
          2. Unload decoder (frees workspace + LoRA refit state).
          3. Evict old vae_encode if it changed.
          4. Load new decoder engine (rebuilds the LoRA manager).
          5. Preload new vae_encode so the first call is warm.
          6. Re-enable the snapshotted LoRAs against the new engine.
        """
        engine = self._diffusion_engine

        old_decoder = prior_paths.get("decoder")
        new_decoder = target_paths.get("decoder")
        decoder_changed = (
            self._decoder_tensorrt
            and engine is not None
            and old_decoder != new_decoder
        )

        snapshot: list[_LoRASnapshot] = []
        if decoder_changed and engine.lora_available:
            for d in engine.list_loras():
                if d.state == "enabled":
                    snapshot.append(
                        _LoRASnapshot(lora_id=d.id, strength=float(d.strength))
                    )

        if decoder_changed:
            engine.unload_trt_engine()

        if self._vae_tensorrt:
            new_enc = target_paths.get("vae_encode")
            # Purge every stale vae_encode_* / dreamvae_encode_* in the
            # module-level cache that isn't the engine we're about to
            # install. The old code only evicted `prior_paths.get(
            # "vae_encode")` — the entry THIS profile manager remembered
            # from bind(). Engines left in the cache by a *previous*
            # session's profile manager (or by runtime_init at session
            # start, before bind()) stay invisible to it and never get
            # evicted by name. That bites twice:
            #   (1) _find_best_vae_engine walks the cache in insertion
            #       order and returns the FIRST match — so a stale
            #       prior-session entry silently shadows the engine
            #       we actually want, surfacing as "TRT VAE encode
            #       rejected input shape" when the source duration
            #       doesn't fit the accidentally-picked engine;
            #   (2) those stale entries pin several GB of execution-
            #       context VRAM each, so long-uptime pods leak memory
            #       across session lifecycles and eventually OOM on
            #       swap.
            # The cache invariant we want is "at most one vae_encode-
            # class entry resident at a time" — restored on every swap.
            self._purge_stale_vae_encode_cache(new_enc)

        if decoder_changed:
            # Load failures here happen AFTER unload + vae_encode evict,
            # so any exception leaves the profile manager with nothing
            # loaded. Re-raise as a typed TRTProfileLoadError so the
            # caller (apply_swap_if_pending in backend.py) can surface
            # a clean swap_failed to the client instead of letting the
            # session crash on the next decode tick with a bare
            # NoneType / "engine not bound" attribute error.
            #
            # Same single-retry policy as the vae_encode load below: a
            # workspace-alloc OOM right after the eviction is often
            # just PyTorch's cache still holding the freed blocks, and
            # one empty_cache() + retry reliably recovers.
            try:
                engine.load_trt_engine(new_decoder)
            except BaseException as e_first:
                logger.warning(
                    "trt_profile_swap_decoder_load_retry engine={} "
                    "first_error={}",
                    new_decoder, e_first,
                )
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                try:
                    engine.load_trt_engine(new_decoder)
                except BaseException as e:
                    self._loaded_dur = None
                    self._loaded_paths = {}
                    logger.error(
                        "trt_profile_swap_decoder_load_failed engine={} error={}",
                        new_decoder, e,
                    )
                    raise TRTProfileLoadError(
                        component="decoder",
                        engine_path=new_decoder,
                        cause=e,
                    ) from e

        if self._vae_tensorrt:
            from acestep.nodes.vae_nodes import _get_trt_vae

            new_enc = target_paths.get("vae_encode")
            if new_enc:
                # Same atomicity concern as the decoder above, with one
                # additional move: a workspace-alloc OOM is often just
                # fragmentation from the eviction we just did, so a
                # single retry after ``torch.cuda.empty_cache()`` reliably
                # recovers (the cache hand-back of reserved-but-unused
                # blocks lets TRT's allocator find a contiguous span).
                # Only one retry — if that doesn't take, the profile
                # genuinely doesn't fit on the current card and no
                # amount of fiddling helps.
                try:
                    _get_trt_vae(new_enc, self._device)
                except BaseException as e_first:
                    logger.warning(
                        "trt_profile_swap_vae_encode_load_retry "
                        "engine={} first_error={}",
                        new_enc, e_first,
                    )
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        # empty_cache() can raise under extreme GPU
                        # state corruption; ignore so we don't mask the
                        # original load error with an unrelated one.
                        pass
                    try:
                        _get_trt_vae(new_enc, self._device)
                    except BaseException as e_second:
                        self._loaded_dur = None
                        self._loaded_paths = {}
                        logger.error(
                            "trt_profile_swap_vae_encode_load_failed "
                            "engine={} error={}",
                            new_enc, e_second,
                        )
                        raise TRTProfileLoadError(
                            component="vae_encode",
                            engine_path=new_enc,
                            cause=e_second,
                        ) from e_second

        if snapshot and engine is not None and engine.lora_available:
            for snap in snapshot:
                try:
                    engine.enable_lora(snap.lora_id, strength=snap.strength)
                except Exception as e:
                    logger.warning(
                        "Re-enable LoRA {} after profile swap failed: {}",
                        snap.lora_id, e,
                    )
        elif snapshot:
            logger.warning(
                "Could not restore {} LoRA(s) after profile swap "
                "(no LoRA backend on the new engine)",
                len(snapshot),
            )
