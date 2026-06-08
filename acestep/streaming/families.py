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
    from acestep.streaming.ace_backend import ACEStepBackend

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
    )


def _make_mrt2(ss):
    from acestep.streaming.mrt2.backend import MRT2Backend

    return MRT2Backend(
        config=ss.config,
        state=ss.state,
        midi_knobs=ss.virtual_knobs,
    )


FAMILIES = {
    "acestep": _make_acestep,
    "mrt2": _make_mrt2,
}


def _acestep_knob_universe():
    from acestep.streaming.knobs import knob_specs

    # Every spec the family can ever expose: both SDE-mode variants plus
    # a representative LoRA-strength knob (the per-id specs all come from
    # lora_strength_spec, so one placeholder id covers the pattern).
    return (
        knob_specs(False, loras=["<lora_id>"])
        + knob_specs(True, loras=["<lora_id>"])
    )


# Per-family knob universes for the cross-backend homonym rule (plan
# §3.3): the full set of KnobSpecs a family can ever expose, obtainable
# WITHOUT constructing the (GPU-heavy) backend. The homonym drift guard
# (tests/unit/test_knob_homonyms.py) runs over these manifests — a knob
# name shared across families must mean exactly the same thing, or it
# must be renamed (prefix / group), so the first lazily-reused name
# can't become a silent semantic fork. Keyed identically to FAMILIES;
# the guard enforces the keys stay in sync.
def _mrt2_knob_universe():
    from acestep.streaming.mrt2.backend import mrt2_knob_specs

    return mrt2_knob_specs()


FAMILY_KNOB_UNIVERSES = {
    "acestep": _acestep_knob_universe,
    "mrt2": _mrt2_knob_universe,
}


def _create_mrt2_session(cls, **kwargs):
    from acestep.streaming.mrt2.backend import create_mrt2_session

    return create_mrt2_session(cls, **kwargs)


# Families whose per-connect setup doesn't fit the ACE create path
# (TRT profiles, model load, demucs, conditioning encode). Keyed like
# FAMILIES; absent = the family rides StreamingSession.create's default
# body. Contract: ``creator(cls, *, audio, config, checkpoint,
# session_id, **rest) -> StreamingSession``.
SESSION_CREATORS = {
    "mrt2": _create_mrt2_session,
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
