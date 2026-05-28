"""SteeringController: per-session entry point for activation steering.

Owns the auto-axis vector cache, the manual catalog, the LIFO slot
registry, and the ``raw`` knob dict → ``set_steering`` config-list
translation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from .catalog import enumerate_catalog, load_auto_vectors, load_vector
from .policy import (
    AUTO_AXES,
    MANUAL_MAX_LAYER,
    MANUAL_MAX_STEP,
    MANUAL_SLOT_CAP,
    MANUAL_SLOT_DEFAULT_COUNT,
    PROBE_N,
    fractional_inject_step,
    parse_probe_n,
)
from .types import AutoAxis, CapacityError, CatalogEntry, EmptyError, KnobNames


# Wire knob name prefixes — agreed on by pipeline raw dict, demo
# KnobDef registry, and the MCP catalog.
_MAN_SRC = "man_src_"
_MAN_LAYER = "man_layer_"
_MAN_STEP = "man_step_"
_MAN_ALPHA = "man_alpha_"


class SteeringController:
    """Engine-side steering state for one streaming session.

    Construction loads the catalog + auto-axis tensors. Degrades to
    ``is_loaded == False`` (empty catalog, ``build_configs`` returns
    ``[]``) when ``vector_dir`` is None or missing on disk.

    Slots are LIFO: ``add_slot`` allocates ``slot_count + 1`` so the
    active set is always ``{1..count}`` with no holes. Interior
    deletion is not supported, which keeps catalog indices stable
    across edits.
    """

    auto_axes: tuple[AutoAxis, ...]
    catalog: tuple[CatalogEntry, ...]
    slot_cap: int = MANUAL_SLOT_CAP
    MANUAL_MAX_LAYER: int = MANUAL_MAX_LAYER
    MANUAL_MAX_STEP: int = MANUAL_MAX_STEP

    def __init__(
        self,
        vector_dir: Path | None,
        *,
        default_slot_count: int = MANUAL_SLOT_DEFAULT_COUNT,
    ) -> None:
        self._vector_dir = vector_dir
        self.auto_axes = AUTO_AXES
        # Probe schedule N comes from the bundle dir name
        # (``shift3.5_n8_seed1528`` → 8); falls back to PROBE_N when the
        # token is absent or no dir was passed.
        self._probe_n = parse_probe_n(vector_dir.name) if vector_dir else PROBE_N
        if vector_dir is None:
            self.catalog = ()
            self._auto_vectors = {}
        else:
            self.catalog = enumerate_catalog(vector_dir)
            self._auto_vectors = load_auto_vectors(vector_dir, AUTO_AXES)
        # Lazy manual-vector cache keyed by catalog index. ``None`` is
        # the cached load-failure sentinel so a corrupted .pt logs once.
        self._manual_vectors: dict[int, dict | None] = {}
        if not self.is_loaded:
            self._slot_count = 0
        else:
            self._slot_count = max(0, min(int(default_slot_count), self.slot_cap))

    @property
    def is_loaded(self) -> bool:
        """True if at least one vector (auto or manual) is reachable."""
        return bool(self._auto_vectors) or bool(self.catalog)

    @property
    def slot_count(self) -> int:
        return self._slot_count

    def active_slots(self) -> tuple[int, ...]:
        """Sorted tuple of currently allocated slot ids (1..slot_count)."""
        return tuple(range(1, self._slot_count + 1))

    def add_slot(self) -> int:
        """Allocate the next LIFO slot id. Raises CapacityError at cap.

        Refused when no vectors are loaded: a slot with no catalog
        behind it is just a dead knob quadruple.
        """
        if not self.is_loaded:
            raise CapacityError("manual steering unavailable (no vectors loaded)")
        if self._slot_count >= self.slot_cap:
            raise CapacityError(
                f"manual steering at cap ({self.slot_cap})",
            )
        self._slot_count += 1
        return self._slot_count

    def pop_slot(self) -> int:
        """Pop the highest-numbered slot. Raises EmptyError when empty."""
        if self._slot_count <= 0:
            raise EmptyError("no manual steering slots to remove")
        popped = self._slot_count
        self._slot_count -= 1
        return popped

    @staticmethod
    def knob_names(slot_id: int) -> KnobNames:
        """The four wire-protocol knob names for slot ``slot_id``."""
        return KnobNames(
            src=f"{_MAN_SRC}{slot_id}",
            layer=f"{_MAN_LAYER}{slot_id}",
            step=f"{_MAN_STEP}{slot_id}",
            alpha=f"{_MAN_ALPHA}{slot_id}",
        )

    def snapshot_key(
        self,
        raw: Mapping[str, float],
        n: int,
    ) -> tuple:
        """Build the change-detection key for ``raw`` at schedule ``n``.

        Returns a tuple containing every input that ``build_configs``
        consults: the auto-axis alphas in declaration order, then per
        active slot the (src, layer, step, alpha) quadruple, then the
        schedule step count. Demo callers cache this and skip
        ``set_steering`` when it's unchanged.

        Length is dynamic in ``slot_count``: an add or pop changes the
        tuple length so equality fails and the next ``build_configs``
        actually fires.
        """
        out: list[float] = [float(raw.get(ax.name, 0.0)) for ax in self.auto_axes]
        for slot in self.active_slots():
            names = self.knob_names(slot)
            out.append(float(raw.get(names.src, 0.0)))
            out.append(float(raw.get(names.layer, 0.0)))
            out.append(float(raw.get(names.step, 0.0)))
            out.append(float(raw.get(names.alpha, 0.0)))
        out.append(float(max(1, int(n))))
        return tuple(out)

    def _get_manual_vector(self, src_idx: int) -> dict | None:
        """Lazy-load one catalog entry's ``{vector, magnitude}``.

        Caches successes and failures (failure as None) so a corrupted
        .pt logs once instead of every tick. Returns None for out-of-
        range indices or load failures.
        """
        if src_idx < 0 or src_idx >= len(self.catalog):
            return None
        cached = self._manual_vectors.get(src_idx, _MISSING)
        if cached is not _MISSING:
            return cached
        entry = self.catalog[src_idx]
        try:
            vec, mag = load_vector(self._vector_dir, entry)  # type: ignore[arg-type]
            blob: dict | None = {"vector": vec, "magnitude": mag}
        except Exception as exc:
            from loguru import logger
            logger.warning(
                "steering: failed to load manual vector {} (idx {}): {}",
                entry.filename, src_idx, exc,
            )
            blob = None
        self._manual_vectors[src_idx] = blob
        return blob

    def build_configs(
        self,
        raw: Mapping[str, float],
        n: int,
    ) -> list[dict]:
        """Translate live ``raw`` knob state into ``set_steering`` configs.

        Auto axes get the fractional step mapping, per-axis layer
        offset, and sign correction. Manual slots are verbatim:
        (layer, step) lands as picked, alpha is consumed sign-as-given.
        Zero-alpha entries are dropped.
        """
        if not self.is_loaded:
            return []
        n = max(1, int(n))
        configs: list[dict] = []
        for ax in self.auto_axes:
            alpha = float(raw.get(ax.name, 0.0))
            if alpha == 0.0:
                continue
            blob = self._auto_vectors.get(ax.name)
            if blob is None:
                continue
            inject_step = fractional_inject_step(
                ax.probe_step, n, probe_n=self._probe_n,
            )
            inject_layer = max(
                0, min(MANUAL_MAX_LAYER, ax.probe_layer + ax.layer_offset),
            )
            configs.append({
                "layer": inject_layer,
                "step": inject_step,
                "vector": blob["vector"],
                "magnitude": blob["magnitude"],
                "alpha": alpha * ax.sign,
            })
        for slot in self.active_slots():
            names = self.knob_names(slot)
            alpha = float(raw.get(names.alpha, 0.0))
            if alpha == 0.0:
                continue
            src_idx = int(round(float(raw.get(names.src, 0.0))))
            blob = self._get_manual_vector(src_idx)
            if blob is None:
                continue
            inject_layer = max(
                0, min(MANUAL_MAX_LAYER, int(round(float(raw.get(names.layer, 0.0))))),
            )
            inject_step = int(round(float(raw.get(names.step, 0.0))))
            configs.append({
                "layer": inject_layer,
                "step": inject_step,
                "vector": blob["vector"],
                "magnitude": blob["magnitude"],
                "alpha": alpha,
            })
        return configs


_MISSING = object()
