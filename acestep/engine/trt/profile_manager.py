"""Profile-aware TRT engine swapping.

Holds at most one ``decoder`` and one ``vae_encode`` engine in VRAM at
a time, and swaps them when the requested duration profile changes.
Each registered profile (60s / 120s / 240s; see :mod:`acestep.paths`)
reserves several GB of workspace at TRT context creation, so holding
all profiles simultaneously is not viable — unloading the previous
engine before loading the next is mandatory.

Out of scope: ``vae_decode``. The realtime path runs with
``vae_window > 0``, which pins ``vae_decode`` to the windowed engine
(``vae_decode_fp16_3to30s``); that engine's 3-30 s shape range is
profile-independent. Optional dreamvae / fast_vae substitutions are
also applied at session build and survive every profile swap. The
manager intentionally leaves the vae_decode slot alone so it can't
clobber either of those.

Typical usage from a streaming session::

    mgr = TRTProfileManager(
        decoder_backend="tensorrt",
        vae_backend="tensorrt",
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
        device: str = "cuda",
    ):
        self._decoder_tensorrt = decoder_backend == "tensorrt"
        self._vae_tensorrt = vae_backend == "tensorrt"
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
        return available_trt_engines(duration_s, needs=self._needs())

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
        """
        self._diffusion_engine = diffusion_engine
        self._loaded_dur = float(picked_dur)
        self._loaded_paths = dict(paths)

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

        snapshot: list[_LoRASnapshot] = []
        if engine is not None and engine.lora_available:
            for d in engine.list_loras():
                if d.state == "enabled":
                    snapshot.append(
                        _LoRASnapshot(lora_id=d.id, strength=float(d.strength))
                    )

        if self._decoder_tensorrt and engine is not None:
            engine.unload_trt_engine()

        if self._vae_tensorrt:
            from acestep.nodes.vae_nodes import _evict_trt_vae

            old_enc = prior_paths.get("vae_encode")
            new_enc = target_paths.get("vae_encode")
            if old_enc and old_enc != new_enc:
                _evict_trt_vae(old_enc)

        if self._decoder_tensorrt and engine is not None:
            new_decoder = target_paths["decoder"]
            engine.load_trt_engine(new_decoder)

        if self._vae_tensorrt:
            from acestep.nodes.vae_nodes import _get_trt_vae

            new_enc = target_paths.get("vae_encode")
            if new_enc:
                _get_trt_vae(new_enc, self._device)

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
