"""Backend family registry.

One :class:`FamilySpec` per model family is the source of truth for
everything the platform needs to know about it: how to build its
:class:`~acestep.streaming.generator_backend.GeneratorBackend`, how to
create its session, which ``--checkpoint`` aliases select it, its knob
universe, its warmup policy, and whether it hosts model extensions.
The session, runner, wire protocol and UI read the spec (or one of the
views derived from it below) and never branch on a family name.

Adding a family is one spec plus a backend module; register the spec in
:data:`FAMILY_SPECS`. ``tests/unit/test_family_conformance.py`` runs
against every registered spec on CPU, and ``docs/FAMILIES.md`` is the
walkthrough.

Factory contract: ``make_backend(streaming_session) -> GeneratorBackend``.
The factory pulls whatever it needs off the (fully constructed)
StreamingSession; that keeps this registry free of per-family argument
plumbing.

An in-tree registry is deliberate (vs entry points / import scanning):
with a handful of in-tree families, an explicit tuple is greppable and
import-cheap. Out-of-tree families would ride the same
``acestep.plugins.discovery`` machinery the model extensions use;
revisit when one exists.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional

from acestep.engine.obs import logger
from acestep.streaming.config import DEFAULT_FAMILY, SessionConfig
from acestep.streaming.preflight import (
    PreflightRequest,
    PreflightResult,
    acestep_preflight,
    sa3_preflight,
)

# DEFAULT_FAMILY (re-exported from acestep.streaming.config): the family
# whose checkpoint names need no alias — an unrecognised ``--checkpoint``
# value is taken as one of its checkpoint directory names.

#: Startup-warmup policies a family may declare. "ace_trt" drives the
#: synthetic ACE warmup session (acestep.streaming.warmup: TRT decoder-
#: engine load, LoRA-refit manager, first-tick pipeline build — ~30s of
#: one-time engine-resident state). "none" skips it: a family whose
#: one-time cost is a process-cached model load pays it on the first
#: real session and the rest are warm.
WARMUP_POLICY_NAMES = ("ace_trt", "none")

#: Prompt-tooling policies a family may select (the enhancer, the
#: variations grid and the constraint checker each branch on one of
#: these). They are policy names, not family names: a new family picks
#: the policy whose prompt style its model was trained on.
PROMPT_POLICIES = ("acestep", "sa3")

_FAMILY_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")


#: Wire types a family config field may have. Every family field is
#: optional and nullable on the wire: absent or null means "the family's
#: own default", exactly like the platform's Optional fields.
CONFIG_FIELD_TYPES = ("float", "int", "bool", "str")


@dataclass(frozen=True)
class FamilyConfigField:
    """One session-config key a family adds to the handshake payload.

    Declared on :attr:`FamilySpec.config_fields`, parsed by
    ``SessionConfig.from_dict`` into ``SessionConfig.family_config`` and
    projected into the wire contract (``/api/protocol`` ``config``, the
    generated TS/C++ types) next to the platform fields. Names are unique
    across families and may not shadow a platform field.
    """

    name: str
    type: str = "float"
    description: str = ""

    def __post_init__(self):
        if not re.match(r"^[a-z][a-z0-9_]{0,47}$", self.name):
            raise ValueError(f"config field name {self.name!r} is not a wire key")
        if self.type not in CONFIG_FIELD_TYPES:
            raise ValueError(
                f"config field {self.name!r} type {self.type!r} not in "
                f"{CONFIG_FIELD_TYPES}"
            )

    def coerce(self, value):
        """Parse a raw wire value; ``None`` for absent, null or junk."""
        if value is None:
            return None
        try:
            if self.type == "bool":
                return bool(value)
            if self.type == "int":
                return int(value)
            if self.type == "float":
                return float(value)
            return str(value)
        except (TypeError, ValueError):
            return None


@dataclass(frozen=True)
class TextOnlySpec:
    """How a family serves a pure text-to-audio session (no source upload).

    The WS adapter synthesises a silent source anchor so the family still
    has geometry to hang the render off. ``duration_field`` names the
    config key that sets its length (``None`` = the client cannot choose;
    the default is used), clamped to ``[1, max_duration_s]``.
    """

    default_duration_s: float = 60.0
    max_duration_s: float = 60.0
    duration_field: Optional[str] = None

    def __post_init__(self):
        if not (1.0 <= self.default_duration_s <= self.max_duration_s):
            raise ValueError(
                f"text-only default {self.default_duration_s} must lie in "
                f"[1, max_duration_s={self.max_duration_s}]"
            )

    def duration_s(self, config_dict: Mapping[str, Any]) -> float:
        """The anchor length a config asks for, defaulted and clamped."""
        dur = 0.0
        if self.duration_field:
            try:
                dur = float(config_dict.get(self.duration_field) or 0.0)
            except (TypeError, ValueError):
                dur = 0.0
        if dur <= 0.0:
            dur = self.default_duration_s
        return max(1.0, min(dur, self.max_duration_s))


@dataclass(frozen=True)
class FamilySpec:
    """Everything one model family declares to the platform.

    ``name`` is the ``SessionConfig.backend`` value and the pool
    identity's family half. ``make_backend`` and ``knob_universe`` are
    callables so a spec imports nothing GPU-heavy at registry import;
    each does its own lazy import when called.

    ``checkpoint_aliases`` maps a ``--checkpoint`` alias to this
    family's model id; aliases are unique across families. A family
    owns its create path through ``create_session`` (contract
    ``creator(cls, *, audio, config, checkpoint, session_id, **rest) ->
    StreamingSession``); ``StreamingSession.create`` only dispatches.

    ``preflight`` is the family's boot check
    (:mod:`acestep.streaming.preflight`): pure, returns a verdict, and
    the server prints and exits on failure. ``prompt_policy`` selects
    which prompt-tooling policy the pod's ``/api/enhance`` infers.
    ``accepts_checkpoint_dir`` says whether the family can serve an
    operator-supplied checkpoint directory in place of its catalog
    location (``--sa3-base-checkpoint``).

    ``text_only`` describes how the family serves a session with no
    source upload (:class:`TextOnlySpec`), or ``None`` when it cannot;
    the adapter advertises ``supports_text_only`` from it.

    ``config_fields`` are the session-config keys the family adds to the
    handshake (:class:`FamilyConfigField`); they reach the family as
    ``config.family_config[name]``.

    ``shutdown`` releases process-wide state the family holds (a
    process-cached model, an installed extension) when the server exits.

    ``supports_extensions`` says whether ``--model-extension`` may
    target this family: the family's context must offer the install /
    decorate / controls hooks (see ``docs/PLUGINS.md``). Selection
    refuses a family that does not, because a selected extension that
    is never installed would generate with the stock model and sound
    entirely plausible.
    """

    name: str
    display_name: str
    make_backend: Callable[[Any], Any]
    knob_universe: Callable[[], list]
    checkpoint_aliases: Mapping[str, str] = field(default_factory=dict)
    create_session: Optional[Callable[..., Any]] = None
    warmup_policy: str = "none"
    preflight: Optional[Callable[[PreflightRequest], PreflightResult]] = None
    prompt_policy: str = "acestep"
    accepts_checkpoint_dir: bool = False
    text_only: Optional[TextOnlySpec] = None
    config_fields: tuple = ()
    shutdown: Optional[Callable[[], Any]] = None
    supports_extensions: bool = False

    def __post_init__(self):
        if self.prompt_policy not in PROMPT_POLICIES:
            raise ValueError(
                f"family {self.name!r} prompt_policy {self.prompt_policy!r} "
                f"not in {PROMPT_POLICIES}"
            )
        if not _FAMILY_NAME_RE.match(self.name):
            raise ValueError(
                f"family name {self.name!r} must match {_FAMILY_NAME_RE.pattern}"
            )
        if not self.display_name:
            raise ValueError(f"family {self.name!r} needs a display_name")
        if self.warmup_policy not in WARMUP_POLICY_NAMES:
            raise ValueError(
                f"family {self.name!r} warmup_policy {self.warmup_policy!r} "
                f"not in {WARMUP_POLICY_NAMES}"
            )
        for alias, model_id in self.checkpoint_aliases.items():
            if not isinstance(alias, str) or not alias or not isinstance(model_id, str) or not model_id:
                raise ValueError(
                    f"family {self.name!r} alias {alias!r} -> {model_id!r} "
                    "must be non-empty strings"
                )
        names = [f.name for f in self.config_fields]
        if len(set(names)) != len(names):
            raise ValueError(f"family {self.name!r} repeats a config field: {names}")
        for f in self.config_fields:
            if not isinstance(f, FamilyConfigField):
                raise ValueError(
                    f"family {self.name!r} config_fields must be FamilyConfigField"
                )


def _make_acestep(ss):
    from acestep.paths import checkpoint_scale
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
        # Scale label for the lora_compatible predicate; None for
        # checkpoints outside the scale map = "don't filter".
        checkpoint_scale=checkpoint_scale(ss.checkpoint),
    )


#: SA3Backend's longest render window (its SA3_MAX_DURATION_S). Mirrored
#: here so the registry does not import the backend module at import
#: time; the conformance test pins the two values together.
SA3_TEXT_ONLY_MAX_DURATION_S = 120.0


def _make_sa3(ss):
    # Assembles SA3Backend.from_context from the construction payload
    # the per-family create path (acestep.streaming.sa3_session) stashed
    # on the session: the process-cached SA3Context, the precomputed
    # conditioning capture, and the SAME-encoded source anchor. Plain
    # session attributes (knob state, steps/depth, vae_window) come off
    # the session itself, per the factory contract above.
    from acestep.streaming.sa3_backend import SA3Backend

    init = getattr(ss, "backend_init", None)
    if not init or "context" not in init:
        raise ValueError(
            "backend 'sa3' requires the per-family create path "
            "(acestep.streaming.sa3_session.create_sa3_session) to stash "
            "its construction payload; an ACE-shaped session cannot "
            "assemble an SA3 backend"
        )
    return SA3Backend.from_context(
        init["context"],
        prompt=ss.state.prompt_text,
        # The live B prompt seeds the backend's tag pair: a swap-resize
        # re-captures conditioning for BOTH prompts, and without this a
        # resize before any set_prompt would collapse the A/B blend to A.
        prompt_b=ss.state.prompt_text_b,
        duration_s=float(init["duration_s"]),
        knob_state=ss.virtual_knobs,
        state=ss.state,
        cond=init["cond"],
        # Prompt-B capture for the A/B crossfade; .get so an in-process
        # payload predating the blend surface stays blend-neutral.
        cond_b=init.get("cond_b"),
        source_latent_bct=init["source_latent_bct"],
        # The pre-encode audio behind that anchor, for the extension
        # conditioning hook. .get so an in-process payload predating it
        # still assembles, with the waveform half of the source view
        # simply absent.
        source_audio=init.get("source_audio"),
        # Resolved accel values (compile already normalized to eager by
        # the create path); .get so an in-process payload without them
        # stays on the eager default.
        dit_backend=init.get("dit_backend", "eager"),
        codec_backend=init.get("codec_backend", "eager"),
        # Startup-selected model extension, absent on stock sessions.
        model_extension=init.get("model_extension"),
        steps=int(ss.config.steps),
        depth=int(ss.state.current_depth),
        vae_window_s=float(ss.vae_window),
        # SA3 LoRA (plan Phase 1): the create path constructs the
        # family manager against the process-cached model and stashes
        # it here; .get so an in-process payload predating the LoRA
        # surface stays LoRA-less rather than failing assembly.
        lora_manager=init.get("lora_manager"),
        use_lora=bool(ss.use_lora),
    )


def resolve_checkpoint(name: str) -> tuple:
    """``--checkpoint`` name -> ``(backend_family, model_id)``.

    Aliases come from every registered spec's ``checkpoint_aliases``;
    an unaliased name is a :data:`DEFAULT_FAMILY` checkpoint directory.
    """
    return CHECKPOINT_ALIASES.get(name, (DEFAULT_FAMILY, name))


def warmup_policy(family: str) -> str:
    """Startup-warmup policy for ``family`` (one of :data:`WARMUP_POLICY_NAMES`)."""
    return WARMUP_POLICIES.get(family, "none")


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
def _sa3_knob_universe():
    from acestep.streaming.sa3_backend import sa3_knob_specs

    # One representative LoRA-strength knob (the per-id specs all come
    # from the shared lora_strength_spec factory, so one placeholder id
    # covers the pattern — same convention as the ACE universe). The
    # name is shared with ACE's universe deliberately: the homonym
    # guard proves the spec shapes are identical across families.
    return sa3_knob_specs(loras=["<lora_id>"])


def _shutdown_sa3() -> int:
    """Close every process-cached SA3 context (detaches an installed
    extension from the shared model). Returns how many were evicted."""
    from acestep.streaming.sa3_session import evict_sa3_contexts

    return evict_sa3_contexts()


def _create_acestep_session(cls, **kwargs):
    from acestep.streaming.ace_session import create_acestep_session

    return create_acestep_session(cls, **kwargs)


def _create_sa3_session(cls, **kwargs):
    from acestep.streaming.sa3_session import create_sa3_session

    return create_sa3_session(cls, **kwargs)


# ---------------------------------------------------------------------------
# The registered families. One spec each; everything below is derived.
# ---------------------------------------------------------------------------

ACESTEP = FamilySpec(
    name="acestep",
    display_name="ACE-Step 1.5",
    make_backend=_make_acestep,
    knob_universe=_acestep_knob_universe,
    checkpoint_aliases={"xl": "acestep-v15-xl-turbo"},
    create_session=_create_acestep_session,
    warmup_policy="ace_trt",
    preflight=acestep_preflight,
    prompt_policy="acestep",
    # A silent 60 s anchor: the length the client always got here, and the
    # 60 s TRT profile every pod builds. ACE has no per-session duration
    # field, so the client cannot choose another length.
    text_only=TextOnlySpec(default_duration_s=60.0, max_duration_s=60.0),
)

SA3 = FamilySpec(
    name="sa3",
    display_name="Stable Audio 3",
    make_backend=_make_sa3,
    knob_universe=_sa3_knob_universe,
    checkpoint_aliases={"sa3-small": "small-music", "sa3-medium": "medium"},
    # Per-connect setup doesn't fit the ACE create path (TRT profiles,
    # model load, demucs, conditioning encode), so SA3 owns its creator.
    create_session=_create_sa3_session,
    warmup_policy="none",
    preflight=sa3_preflight,
    prompt_policy="sa3",
    # --sa3-base-checkpoint: evaluate a non-catalog checkpoint directory.
    accepts_checkpoint_dir=True,
    config_fields=(
        FamilyConfigField(
            "sa3_duration_s", "float",
            "Fixed generation duration for sa3 sessions, seconds. Absent or "
            "null derives it from the uploaded source audio length (the "
            "audio-to-audio anchor); SA3 conditioning is captured per "
            "(prompt, duration), so this is fixed for the session lifetime.",
        ),
    ),
    # The anchor is synthesised at the REQUESTED render length so the
    # source and the render agree in sa3_session; capped at the family's
    # longest render window.
    text_only=TextOnlySpec(
        default_duration_s=60.0, max_duration_s=SA3_TEXT_ONLY_MAX_DURATION_S,
        duration_field="sa3_duration_s",
    ),
    shutdown=_shutdown_sa3,
    # SA3Context offers the model-extension veto/install/close hooks.
    supports_extensions=True,
)


def _register(*specs: FamilySpec) -> dict:
    from dataclasses import fields as _dc_fields

    platform_fields = {f.name for f in _dc_fields(SessionConfig)}
    out: dict = {}
    aliases: dict = {}
    config_keys: dict = {}
    for spec in specs:
        if spec.name in out:
            raise ValueError(f"family {spec.name!r} registered twice")
        for cf in spec.config_fields:
            if cf.name in platform_fields:
                raise ValueError(
                    f"family {spec.name!r} config field {cf.name!r} shadows a "
                    "SessionConfig platform field"
                )
            if cf.name in config_keys:
                raise ValueError(
                    f"config field {cf.name!r} declared by both "
                    f"{config_keys[cf.name]!r} and {spec.name!r}"
                )
            config_keys[cf.name] = spec.name
        for alias in spec.checkpoint_aliases:
            if alias in aliases:
                raise ValueError(
                    f"checkpoint alias {alias!r} claimed by both "
                    f"{aliases[alias]!r} and {spec.name!r}"
                )
            aliases[alias] = spec.name
        out[spec.name] = spec
    if DEFAULT_FAMILY not in out:
        raise ValueError(f"DEFAULT_FAMILY {DEFAULT_FAMILY!r} is not registered")
    return out


#: ``family name -> FamilySpec``. The source of truth.
FAMILY_SPECS: dict = _register(ACESTEP, SA3)


def family_config_fields() -> tuple:
    """Every family's config fields, in registration order. The wire
    contract and ``SessionConfig.from_dict`` both read this, so the
    payload a client can send and the keys the server parses cannot
    drift."""
    return tuple(cf for spec in FAMILY_SPECS.values() for cf in spec.config_fields)


def get_family(name: str) -> FamilySpec:
    """The spec for ``name``; unknown families fail loudly."""
    try:
        return FAMILY_SPECS[name]
    except KeyError:
        known = ", ".join(sorted(FAMILY_SPECS))
        raise ValueError(
            f"unknown backend family {name!r} (registered: {known})"
        ) from None


# ---------------------------------------------------------------------------
# Views derived from the specs, kept for existing importers. Never edit
# these by hand: change the spec, and the conformance test checks that
# each view still equals what the specs declare.
# ---------------------------------------------------------------------------

#: ``family -> make_backend`` factory.
FAMILIES = {n: s.make_backend for n, s in FAMILY_SPECS.items()}

#: ``--checkpoint`` alias -> ``(family, model_id)``.
CHECKPOINT_ALIASES = {
    alias: (n, model_id)
    for n, s in FAMILY_SPECS.items()
    for alias, model_id in s.checkpoint_aliases.items()
}

#: ``family -> warmup policy name``.
WARMUP_POLICIES = {n: s.warmup_policy for n, s in FAMILY_SPECS.items()}

#: ``family -> knob universe callable`` (the homonym guard's input).
FAMILY_KNOB_UNIVERSES = {n: s.knob_universe for n, s in FAMILY_SPECS.items()}

#: ``family -> session creator`` for families that own their create path.
SESSION_CREATORS = {
    n: s.create_session
    for n, s in FAMILY_SPECS.items()
    if s.create_session is not None
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
