"""Registry of analyzers: one :class:`AnalyzerSpec` per listening model.

The analysis counterpart of :class:`acestep.streaming.families.FamilySpec`:
declarative, torch-free at import, lazy callables for anything heavy.
The wire layer reads it to advertise ``ready.capabilities.analysis_taps``
and to validate ``analysis_tap`` commands; the host reads it to build
engines; the session-side tap reads ``window_s`` / ``context_s`` to lay
out its cell grid.

A second analyzer (beats, chords, key) is one spec here plus one engine
class, and, if it emits a new item kind, one deliberate edit to the
protocol's tagged union. No session or adapter surgery.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence

from acestep.analysis.engine import ITEM_KINDS, AnalysisEngine

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
PLACEMENTS = ("gpu", "cpu")


@dataclass(frozen=True)
class AnalyzerOption:
    """One option a client may pass in ``analysis_tap.options``.

    ``type`` is one of ``"float"``, ``"int"``, ``"str"``, ``"str_list"``;
    ``bounds`` is ``(lo, hi)`` for numbers, a tuple of allowed values for
    strings, or ``None``. ``validate`` may raise ``ValueError`` with a
    user-facing message for anything the type system cannot express
    (instrument names, say). Defaults apply when the option is absent.
    """

    name: str
    type: str
    default: Any = None
    bounds: Optional[tuple] = None
    help: str = ""
    validate: Optional[Callable[[Any], Any]] = None


@dataclass(frozen=True)
class AnalyzerSpec:
    """Everything one listening model declares to the platform.

    ``make_engine(size, device)`` loads the model and returns an
    :class:`AnalysisEngine`; it is the only callable that imports anything
    heavy. ``probe()`` answers "could this run here" without loading
    weights (dependency importable, not disabled by env, size allowed) and
    returns ``None`` when yes or a user-facing reason when no.

    ``sizes`` / ``default_size`` / ``size_env`` pick the model size: the
    env var wins, then the default. ``window_s`` and ``context_s`` are the
    engine's input needs per cell; ``stateful`` says whether it carries
    state across windows (the first engine does not; a stateful one is
    scheduled strictly in play order by the tap).

    ``placement`` is ``"gpu"`` or ``"cpu"``; CPU engines skip VRAM
    admission. ``licence`` is shown verbatim by clients.
    """

    name: str
    display_name: str
    item_kind: str
    make_engine: Callable[[str, Any], AnalysisEngine]
    probe: Callable[[], Optional[str]]
    licence: str = ""
    sizes: Sequence[str] = ("default",)
    default_size: str = "default"
    size_env: Optional[str] = None
    window_s: float = 4.0
    context_s: float = 0.5
    min_sr: int = 16000
    stateful: bool = False
    placement: str = "gpu"
    options: Sequence[AnalyzerOption] = ()
    #: Rough resident size in GiB per size, for VRAM admission
    #: (``{"small": 0.3}``); a size absent here is admitted optimistically.
    resident_gb: Mapping[str, float] = field(default_factory=dict)
    preload: Optional[Callable[..., Any]] = None
    shutdown: Optional[Callable[[], Any]] = None

    def __post_init__(self):
        if not _NAME_RE.match(self.name):
            raise ValueError(f"analyzer name {self.name!r} must match {_NAME_RE.pattern}")
        if not self.display_name:
            raise ValueError(f"analyzer {self.name!r} needs a display_name")
        if self.item_kind not in ITEM_KINDS:
            raise ValueError(
                f"analyzer {self.name!r} item_kind {self.item_kind!r} not in {ITEM_KINDS}"
            )
        if self.placement not in PLACEMENTS:
            raise ValueError(f"analyzer {self.name!r} placement {self.placement!r} not in {PLACEMENTS}")
        if self.default_size not in self.sizes:
            raise ValueError(
                f"analyzer {self.name!r} default_size {self.default_size!r} not in sizes {tuple(self.sizes)}"
            )
        if self.window_s <= 0 or self.context_s < 0:
            raise ValueError(f"analyzer {self.name!r} window_s must be > 0 and context_s >= 0")
        names = [o.name for o in self.options]
        if len(set(names)) != len(names):
            raise ValueError(f"analyzer {self.name!r} has duplicate option names {names}")

    def resolve_size(self, requested: Optional[str] = None) -> str:
        """The size to load: an explicit request, else the env var, else
        the default. Raises ``ValueError`` (user-facing) for an unknown size."""
        import os

        size = requested
        if size is None and self.size_env:
            size = os.environ.get(self.size_env, "").strip().lower() or None
        if size is None:
            size = self.default_size
        if size not in self.sizes:
            raise ValueError(
                f"{self.display_name}: size {size!r} not available (choose from {tuple(self.sizes)})"
            )
        return size

    def validate_options(self, options: Optional[Mapping[str, Any]]) -> dict:
        """Apply defaults, types, bounds and custom validators. Unknown keys
        raise ``ValueError`` so a typo never silently does nothing."""
        given = dict(options or {})
        out: dict = {}
        known = {o.name: o for o in self.options}
        for key in given:
            if key not in known:
                raise ValueError(f"{self.display_name}: unknown option {key!r}")
        for opt in self.options:
            if opt.name not in given:
                out[opt.name] = opt.default
                continue
            out[opt.name] = _coerce(self.display_name, opt, given[opt.name])
        return out


def _coerce(display: str, opt: AnalyzerOption, value: Any) -> Any:
    if value is None:
        return opt.default
    if opt.type in ("float", "int"):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{display}: option {opt.name!r} must be a number")
        value = float(value) if opt.type == "float" else int(value)
        if opt.bounds is not None:
            lo, hi = opt.bounds
            if value < lo or value > hi:
                raise ValueError(f"{display}: option {opt.name!r} must be within [{lo}, {hi}]")
    elif opt.type == "str":
        if not isinstance(value, str):
            raise ValueError(f"{display}: option {opt.name!r} must be a string")
        if opt.bounds is not None and value not in opt.bounds:
            raise ValueError(f"{display}: option {opt.name!r} must be one of {tuple(opt.bounds)}")
    elif opt.type == "str_list":
        if not isinstance(value, (list, tuple)) or not all(isinstance(v, str) for v in value):
            raise ValueError(f"{display}: option {opt.name!r} must be a list of strings")
        value = list(value)
    else:  # pragma: no cover - spec author error
        raise ValueError(f"{display}: option {opt.name!r} has unknown type {opt.type!r}")
    if opt.validate is not None:
        value = opt.validate(value)
    return value


_ANALYZERS: dict[str, AnalyzerSpec] = {}


def register(spec: AnalyzerSpec) -> AnalyzerSpec:
    """Register ``spec`` under its name. Re-registering the same name with
    a different object raises; modules that register at import stay
    idempotent across reloads."""
    existing = _ANALYZERS.get(spec.name)
    if existing is not None and existing is not spec:
        raise ValueError(f"analyzer {spec.name!r} already registered")
    _ANALYZERS[spec.name] = spec
    return spec


def get_analyzer(name: str) -> AnalyzerSpec:
    _ensure_builtin()
    try:
        return _ANALYZERS[name]
    except KeyError:
        raise KeyError(f"unknown analyzer {name!r}; known: {sorted(_ANALYZERS)}") from None


def list_analyzers() -> list[AnalyzerSpec]:
    _ensure_builtin()
    return [_ANALYZERS[k] for k in sorted(_ANALYZERS)]


def available_analyzers() -> dict[str, Optional[str]]:
    """``{name: None | reason}`` for every registered analyzer, from
    ``probe()`` (never loads weights; a probe that raises counts as
    unavailable with the exception text as the reason)."""
    out: dict[str, Optional[str]] = {}
    for spec in list_analyzers():
        try:
            out[spec.name] = spec.probe()
        except Exception as exc:  # noqa: BLE001 - probe failure is "no"
            out[spec.name] = f"{type(exc).__name__}: {exc}"
    return out


def _ensure_builtin() -> None:
    """Import the built-in analyzer modules so their ``register`` calls
    run. Each module is torch-free at import."""
    import importlib

    for mod in ("acestep.analysis.notes_engine",):
        try:
            importlib.import_module(mod)
        except ImportError:
            # The module itself is part of DEMON; a missing optional
            # dependency is handled inside its probe, not here.
            pass
