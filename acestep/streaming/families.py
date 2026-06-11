"""Backend family registry.

Maps a ``SessionConfig.backend`` family name to a factory that builds
the session's :class:`~acestep.streaming.generator_backend.GeneratorBackend`.
Adding a model family is an entry here plus a backend module — the
session, runner, wire protocol, and UI do not change (see
``round_3_BACKEND_PLAN_FINAL.md``).

Factory contract: ``factory(streaming_session) -> GeneratorBackend``.
The factory pulls whatever it needs off the (fully constructed)
StreamingSession; that keeps this registry free of per-family argument
plumbing.

A registration dict is deliberate (vs entry points / import scanning):
with a handful of in-tree families, an explicit dict is greppable and
import-cheap. Revisit only if out-of-tree families become real.
"""

from acestep.engine.obs import logger


def _make_acestep(ss):
    from acestep.steering import SteeringController, ensure_steering_vectors
    from acestep.streaming.ace_backend import ACEStepBackend

    # SteeringController is the source of truth for slot_count and the
    # vector catalog; ensure_steering_vectors fetches/caches the
    # checkpoint's probe bundle (None for checkpoints without one — XL,
    # fetch failures — which degrades the controller to is_loaded=False
    # and drops the steering capability/knobs for the session).
    steering = SteeringController(ensure_steering_vectors(ss.checkpoint))

    return ACEStepBackend(
        ss.session, ss.stream,
        state=ss.state,
        use_midi=True,  # always "MIDI" mode; KnobState provides values
        use_sde=ss.use_sde, use_lora=ss.use_lora,
        midi_knobs=ss.virtual_knobs,
        engine_obj=ss.engine_obj,
        vae_window=ss.vae_window, crop_seconds=ss.crop_seconds,
        k1_name=ss.k1_name, seed=1528, skip_threshold=5e-4,
        walk_window=ss.walk_window,
        walk_window_s=ss.walk_window_s,
        neg_conditioning=ss.cond_negative,
        steering=steering,
    )


FAMILIES = {
    "acestep": _make_acestep,
}


def _acestep_knob_universe():
    from acestep.steering.policy import (
        AUTO_AXES,
        MANUAL_MAX_LAYER,
        MANUAL_MAX_STEP,
        PROBE_N,
    )
    from acestep.streaming.knobs import (
        knob_specs,
        manual_slot_specs,
        steering_axis_spec,
    )

    # Every spec the family can ever expose: both SDE-mode variants plus
    # a representative LoRA-strength knob (the per-id specs all come from
    # lora_strength_spec, so one placeholder id covers the pattern), plus
    # the steering surface — the four auto axes and one representative
    # manual slot (per-slot specs all come from manual_slot_specs).
    # Catalog geometry uses the canonical v15-turbo bundle's 144 cells;
    # no network fetch happens here (policy tables only).
    steering = [
        steering_axis_spec(
            ax.name,
            axis=ax.axis,
            inject_layer=max(
                0, min(MANUAL_MAX_LAYER, ax.probe_layer + ax.layer_offset),
            ),
            probe_step=ax.probe_step,
            probe_n=PROBE_N,
            blurb=ax.blurb,
        )
        for ax in AUTO_AXES
    ] + manual_slot_specs(
        1,
        src_max=143,
        catalog_len=144,
        layer_max=MANUAL_MAX_LAYER,
        step_max=MANUAL_MAX_STEP,
    )
    return (
        knob_specs(False, loras=["<lora_id>"])
        + knob_specs(True, loras=["<lora_id>"])
        + steering
    )


# Per-family knob universes for the cross-backend homonym rule (plan
# §3.3): the full set of KnobSpecs a family can ever expose, obtainable
# WITHOUT constructing the (GPU-heavy) backend. The homonym drift guard
# (tests/unit/test_knob_homonyms.py) runs over these manifests — a knob
# name shared across families must mean exactly the same thing, or it
# must be renamed (prefix / group), so the first lazily-reused name
# can't become a silent semantic fork. Keyed identically to FAMILIES;
# the guard enforces the keys stay in sync.
FAMILY_KNOB_UNIVERSES = {
    "acestep": _acestep_knob_universe,
}


def make_backend(name: str, streaming_session):
    """Build the GeneratorBackend for ``name``.

    Unknown families fail loudly at session create (config-time error,
    never a silent fallback): the client asked for a generator this
    server build does not ship.
    """
    try:
        factory = FAMILIES[name]
    except KeyError:
        known = ", ".join(sorted(FAMILIES))
        logger.error("unknown_backend_family name={} known={}", name, known)
        raise ValueError(
            f"unknown backend family {name!r} (registered: {known})"
        ) from None
    return factory(streaming_session)
