"""Optional in-flight timestep-schedule migration for ``StreamPipeline``.

Off by default. ``StreamPipeline`` is a StreamDiffusion-style ring buffer:
each slot is born with a timestep schedule derived from its request's
``denoise`` and keeps it for life, so a per-request ``denoise`` change only
reaches newly born slots and takes ``depth`` ticks to drain through the ring.

A :class:`ScheduleMigration` controller, when attached to a pipeline, bypasses
that drain: on the next ``tick`` every in-flight slot is moved onto a single
shared schedule (1-tick latency). It is the schedule analogue of the
pipeline's ``_shared_curves`` per-step overrides — set once, reach every
in-flight slot on the next tick. The difference is that a shared *curve* is a
per-step multiplier shadowed at point-of-use, whereas the *schedule* is
structural (the integrator reads ``slot.t_schedule[step_idx]``), so it is
migrated onto the slot rather than shadowed. This is well-defined because the
schedule builder always emits ``steps + 1`` entries regardless of denoise
(denoise shifts only where the ramp starts, not its length), so migration is a
value-swap at matching indices: a slot ``k`` steps in stays ``k`` steps in.

``None`` (no controller attached) is the normal heterogeneous per-request
path and costs a single identity check per ``tick``.

The empirical characterization — latency win, audio no-harm, and the
per-policy transient analysis — lives in ``experiments/shared_schedule/``
(see ``REFERENCE.md`` and ``REPROJECT_FIX.md``).

Migration policies (:meth:`ScheduleMigration.set_policy`):

- ``"reproject"`` (recommended): keep ``step_idx`` so the ring's step
  stagger is untouched, but on a DOWNWARD denoise change re-place ``xt`` onto
  the new schedule's manifold at this index, around the slot's cached clean
  prediction ``x0``, preserving the slot's own implied noise::

      eps = (xt - (1 - s_old) * x0) / s_old
      xt  = s_new * eps + (1 - s_new) * x0

  This is exactly one Euler step from the slot's TRUE sigma ``s_old`` to
  ``s_new``; it removes the residual-noise deficit a bare relabel strands and
  feeds this tick's forward a matching timestep, at +1-tick onset and full
  one-completion-per-tick cadence. Needs the slot's ``last_x0_pred`` (cached
  by the pipeline only while a re-anchoring policy is active).
- ``"renoise"``: the same re-anchor with fresh ``randn`` noise (an SDE-style
  re-noise); stochastic comparison arm for ``"reproject"``.
- ``"hold"``: keep ``step_idx`` and swap only the sigma values. Cheapest, and
  byte-identical / allocation-free, but introduces a deliberate sigma/``xt``
  mismatch that distorts large downward changes (residual noise frozen into
  the output). The original behavior.
- ``"remap"``: reposition ``step_idx`` to the new schedule's nearest-sigma
  index so the remaining trajectory keeps the right denoising budget. Slots
  noisier than the new schedule's ceiling defer until they descend into
  range. Budget-preserving, but collapses the ring stagger and degenerates to
  the drain on a large downward jump.
"""

from __future__ import annotations

from typing import Callable, List, Optional, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .stream import _Slot


# Policies that re-anchor ``xt`` (and so require the slot's cached x0).
_REANCHOR_POLICIES = ("reproject", "renoise")
VALID_POLICIES = ("hold", "remap", "reproject", "renoise")

# A slot at this sigma or below is effectively clean; re-anchoring divides by
# ``s_old`` so guard the degenerate final-step case.
_EPS_MIN = 1e-6


def _check_policy(policy: str) -> str:
    if policy not in VALID_POLICIES:
        raise ValueError(
            f"unknown migration policy: {policy!r} "
            f"(expected one of {VALID_POLICIES})"
        )
    return policy


class ScheduleMigration:
    """Controller for migrating in-flight slots onto a shared schedule.

    Constructed with ``schedule_fn`` — the pipeline's (cached)
    ``_get_schedule`` — so :meth:`set_schedule` reuses the same per-denoise
    cache objects the pipeline does; the identity check in :meth:`apply` then
    treats migrated and freshly-born slots as already on the schedule.

    ``policy`` defaults to ``"hold"`` (the original behavior) so direct API
    callers that only call :meth:`set_schedule` keep their prior results;
    callers wanting the production path select ``"reproject"`` explicitly via
    :meth:`set_policy`.
    """

    def __init__(
        self,
        schedule_fn: "Callable[[float], torch.Tensor]",
        policy: str = "hold",
    ) -> None:
        self._schedule_fn = schedule_fn
        self._policy = _check_policy(policy)
        self._schedule: Optional[torch.Tensor] = None
        self._denoise: Optional[float] = None  # label for traces only
        self._trace: Optional[list] = None

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    @property
    def policy(self) -> str:
        return self._policy

    def set_policy(self, policy: str) -> None:
        """Select the migration policy (see module docstring). Next tick."""
        self._policy = _check_policy(policy)

    @property
    def needs_x0_cache(self) -> bool:
        """True when the active policy re-anchors ``xt`` and needs ``x0``.

        The pipeline reads this to decide whether to cache each slot's raw
        model x0 prediction during the forward pass; ``hold``/``remap`` leave
        it ``False`` so they stay allocation-free.
        """
        return self._policy in _REANCHOR_POLICIES

    def set_schedule(self, denoise: float) -> None:
        """Build (cached) and store the shared schedule for ``denoise``."""
        self._schedule = self._schedule_fn(float(denoise))
        self._denoise = float(denoise)

    def clear(self) -> None:
        """Lift the override; new submits resume their per-request denoise.

        Already-migrated in-flight slots are NOT rewound — they own their
        adopted schedule for the rest of their trajectory (there is no
        coherent way to rewind mid-flight).
        """
        self._schedule = None
        self._denoise = None

    def active_schedule(self) -> Optional[torch.Tensor]:
        """Schedule a slot born now should adopt, or ``None`` if no override.

        Read by ``_init_slot`` so a slot born while the override is active
        starts on the shared schedule instead of being migrated a tick later.
        """
        return self._schedule

    def start_trace(self) -> list:
        """Attach a fresh migration trace and return it (diagnostics hook).

        After this, every :meth:`apply` call that migrates one or more slots
        appends one record. ``None`` (the default) keeps the path
        allocation-free; used by ``experiments/shared_schedule``.
        """
        self._trace = []
        return self._trace

    # ------------------------------------------------------------------
    # Hot path
    # ------------------------------------------------------------------
    def apply(self, slots: "List[Optional[_Slot]]", tick: int) -> None:
        """Migrate every in-flight slot onto the shared schedule.

        No-op when no schedule is set or a slot is already on it (object
        identity, since :meth:`set_schedule` reuses the per-denoise cache).
        Called at the top of ``tick()`` before anything reads schedule
        lengths.
        """
        sched = self._schedule
        if sched is None:
            return

        trace_on = self._trace is not None
        records: Optional[list] = [] if trace_on else None
        policy = self._policy
        remap = policy == "remap"
        reanchor = policy in _REANCHOR_POLICIES
        ceiling = float(sched[0]) if remap else 0.0

        for slot in slots:
            if slot is None or slot.t_schedule is sched:
                continue

            s_old = float(slot.t_schedule[slot.step_idx])
            if remap:
                # Noisier than the new schedule can represent: migrating now
                # would strand (s_old - ceiling) of noise with no budget left
                # to remove it. Defer; it migrates once it descends into range.
                if s_old > ceiling:
                    continue
                new_idx = int((sched - s_old).abs().argmin())
            else:
                # hold / reproject / renoise keep step_idx (stagger intact).
                new_idx = min(slot.step_idx, len(sched) - 1)
            s_new = float(sched[new_idx])

            reanchored = False
            if (
                reanchor
                and slot.last_x0_pred is not None
                and s_new < s_old        # downward only (hold self-heals up)
                and s_old > _EPS_MIN
            ):
                # Re-place xt onto the new schedule's manifold at s_new, around
                # the slot's cached clean prediction. After this, xt genuinely
                # sits at s_new, so the remaining (compressed) schedule has
                # exactly the budget to finish and this tick's forward is fed a
                # matching timestep.
                x0 = slot.last_x0_pred.to(
                    device=slot.xt.device, dtype=slot.xt.dtype,
                )
                if policy == "renoise":
                    eps = torch.randn_like(slot.xt)
                else:  # reproject: preserve the slot's own implied noise
                    eps = (slot.xt - (1.0 - s_old) * x0) / s_old
                slot.xt = s_new * eps + (1.0 - s_new) * x0
                reanchored = True

            if trace_on:
                records.append({
                    "from_denoise": slot.request.denoise,
                    "step_idx": slot.step_idx,
                    "new_step_idx": new_idx,
                    "s_old": s_old,
                    "s_new": s_new,
                    "reanchored": reanchored,
                })
            slot.step_idx = new_idx
            slot.t_schedule = sched

        if trace_on and records:
            self._trace.append({
                "tick": tick,
                "n_migrated": len(records),
                "to_denoise": self._denoise,
                "policy": policy,
                "slots": records,
            })
