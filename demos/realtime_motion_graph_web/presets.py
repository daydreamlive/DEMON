"""Session presets: an operator's table of named session configurations.

``--presets <file>`` loads a JSON table at boot (a bad file fails the
boot). A client then opens a session by name::

    {"preset": "<name>", "values": {"seconds": 56.0}}

and the server builds the session config from the table. A preset
session's socket goes through :class:`PresetSocket`, the one place that
decides what a client sees and may send: the ``ready`` frame carries a
``plan`` (versions and settable params by id) instead of the session's
internals, a ``oneshot`` names a version (expanded here to the table's
knobs and seed), and every failure becomes ``{type, code, message,
ref}`` with a fixed sentence per code; the real text goes to the log
under ``ref``.

The table::

    {"sealed": true,
     "request_tools": ["<tool>"],
     "tool_routes": {"<tool>": ["POST /run", "GET /job/*", "DELETE /job/*"]},
     "presets": {"<name>": {
         "config":   {...a session config...},
         "values":   {"seconds": {"field": "<config key>", "min": 1, "max": 120}},
         "intake":   {"tool": "<tool>", "path": "/in"} | null,
         "refresh":  false,
         "versions": [{"id": "1", "knobs": {...}, "seed": 1}],
         "params":   {"amount": {"knob": "<knob>", "min": 0, "max": 1, "default": 0.5}},
         "tools":    ["<tool>"]}}}

``sealed``: a session config without a preset is refused, and every
plain HTTP route answers 404 (only the socket and ``/tools/`` remain).
``tool_routes`` lists, per tool, the ``METHOD /path`` globs a session may
call through the tool proxy; a route not listed does not exist.

Without ``--presets`` none of this runs. Torch-free.
"""

from __future__ import annotations

import fnmatch
import json
import re
import secrets
import struct
import threading
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

#: One fixed sentence per error code. Nothing else of a failure reaches a
#: client of a preset session.
SENTENCES = {
    "busy": "The server is busy. Try again in a moment.",
    "unavailable": "This is not available right now.",
    "bad_request": "The request was not accepted.",
    "too_long": "The request is longer than this server accepts.",
    "failed": "The request could not be completed.",
}

#: The session's own error codes, mapped to the codes a client sees.
_ERROR_CODES = {
    "engine_not_built": "too_long",
    "checkpoint_unavailable": "unavailable",
    "text_only_unsupported": "unavailable",
    "score_request_unsupported": "unavailable",
    "unsupported_trt_checkpoint": "unavailable",
    "fixture_unavailable": "unavailable",
}

_NAME_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")
_ROUTE_RE = re.compile(r"(GET|POST|PUT|DELETE) (/\S*)")
_METHODS = ("GET", "POST", "PUT", "DELETE")
#: How long a ``swap_source`` waits for its binary frame.
SWAP_FRAME_TIMEOUT_S = 30.0


class TableError(ValueError):
    """The preset file is not a valid table (fails the boot)."""


class Refused(Exception):
    """A client request refused with ``code``; ``detail`` is logged only."""

    def __init__(self, code: str, detail: str):
        super().__init__(detail)
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class Preset:
    name: str
    config: dict
    values: dict
    intake: dict | None
    refresh: bool
    versions: tuple
    params: dict
    tools: tuple


@dataclass(frozen=True)
class Table:
    sealed: bool
    request_tools: tuple
    tool_routes: dict
    presets: dict = field(default_factory=dict)


# ---- the table ----


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _keys(obj, allowed: Iterable[str], where: str) -> None:
    if not isinstance(obj, dict):
        raise TableError(f"{where} must be an object")
    extra = sorted(set(obj) - set(allowed))
    if extra:
        raise TableError(f"{where}: unknown keys {extra}")


def _range(spec: dict, where: str, *, default: bool) -> None:
    for k in ("min", "max") + (("default",) if default else ()):
        if not _num(spec.get(k)):
            raise TableError(f"{where}.{k} must be a number")
    if spec["min"] > spec["max"]:
        raise TableError(f"{where}: min is above max")
    if default and not spec["min"] <= spec["default"] <= spec["max"]:
        raise TableError(f"{where}: default is outside min..max")


def _tool_list(v, where: str, routes: Mapping) -> tuple:
    if not isinstance(v, list) or not all(isinstance(t, str) for t in v):
        raise TableError(f"{where} must be a list of tool names")
    missing = [t for t in v if t not in routes]
    if missing:
        raise TableError(f"{where}: tools without tool_routes {missing}")
    return tuple(v)


def _parse_preset(name: str, raw: Any, routes: Mapping) -> Preset:
    where = f"presets.{name}"
    _keys(raw, ("config", "values", "intake", "refresh", "versions",
                "params", "tools"), where)
    config = raw.get("config")
    if not isinstance(config, dict):
        raise TableError(f"{where}.config must be an object")
    if "preset" in config:
        raise TableError(f"{where}.config must not name a preset")
    # A preset session never reads an audio frame from its client.
    if not (config.get("text_only") is True
            or (config.get("use_server_fixture") and config.get("fixture_name"))):
        raise TableError(
            f"{where}.config needs text_only or use_server_fixture + fixture_name"
        )
    values = raw.get("values") or {}
    _keys(values, values.keys(), f"{where}.values")
    for vname, spec in values.items():
        vw = f"{where}.values.{vname}"
        if not _NAME_RE.fullmatch(vname):
            raise TableError(f"{vw}: bad name")
        _keys(spec, ("field", "min", "max", "default"), vw)
        if not isinstance(spec.get("field"), str) or not spec["field"]:
            raise TableError(f"{vw}.field must be a config key")
        _range(spec, vw, default=False)
        if "default" in spec and not (
            _num(spec["default"]) and spec["min"] <= spec["default"] <= spec["max"]
        ):
            raise TableError(f"{vw}.default is outside min..max")
    tools = _tool_list(raw.get("tools") or [], f"{where}.tools", routes)
    intake = raw.get("intake")
    if intake is not None:
        _keys(intake, ("tool", "path"), f"{where}.intake")
        if intake.get("tool") not in tools:
            raise TableError(f"{where}.intake.tool must be one of the preset's tools")
        path = intake.get("path")
        if not isinstance(path, str) or not path.startswith("/"):
            raise TableError(f"{where}.intake.path must start with /")
    refresh = raw.get("refresh", False)
    if not isinstance(refresh, bool):
        raise TableError(f"{where}.refresh must be true or false")
    versions = raw.get("versions")
    if not isinstance(versions, list) or not versions:
        raise TableError(f"{where}.versions must be a non-empty list")
    ids = set()
    for i, ver in enumerate(versions):
        vw = f"{where}.versions[{i}]"
        _keys(ver, ("id", "knobs", "seed"), vw)
        vid = ver.get("id")
        if not isinstance(vid, str) or not _NAME_RE.fullmatch(vid) or vid in ids:
            raise TableError(f"{vw}.id must be a unique short name")
        ids.add(vid)
        knobs = ver.get("knobs") or {}
        if not isinstance(knobs, dict) or not all(
            isinstance(k, str) and _num(v) for k, v in knobs.items()
        ):
            raise TableError(f"{vw}.knobs must map knob names to numbers")
        seed = ver.get("seed", 0)
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise TableError(f"{vw}.seed must be an integer >= 0")
    params = raw.get("params") or {}
    _keys(params, params.keys(), f"{where}.params")
    for pid, spec in params.items():
        pw = f"{where}.params.{pid}"
        if not _NAME_RE.fullmatch(pid):
            raise TableError(f"{pw}: bad name")
        _keys(spec, ("knob", "min", "max", "default"), pw)
        if not isinstance(spec.get("knob"), str) or not spec["knob"]:
            raise TableError(f"{pw}.knob must be a knob name")
        _range(spec, pw, default=True)
    return Preset(
        name=name, config=dict(config), values=dict(values), intake=intake,
        refresh=refresh,
        versions=tuple(
            {"id": v["id"], "knobs": dict(v.get("knobs") or {}),
             "seed": int(v.get("seed", 0))} for v in versions
        ),
        params=dict(params), tools=tools,
    )


def parse_table(raw: Any, proxies: Iterable[str] | None = None) -> Table:
    """A parsed and checked table. ``proxies``: the configured tool names
    (every tool the table names must be one). Raises :class:`TableError`."""
    _keys(raw, ("sealed", "request_tools", "tool_routes", "presets"), "table")
    sealed = raw.get("sealed", False)
    if not isinstance(sealed, bool):
        raise TableError("sealed must be true or false")
    routes_raw = raw.get("tool_routes") or {}
    _keys(routes_raw, routes_raw.keys(), "tool_routes")
    routes = {}
    for tool, globs in routes_raw.items():
        if not isinstance(globs, list) or not all(
            isinstance(g, str) and _ROUTE_RE.fullmatch(g) for g in globs
        ):
            raise TableError(
                f"tool_routes.{tool} must be a list of 'METHOD /path' "
                f"(METHOD one of {list(_METHODS)})"
            )
        routes[tool] = tuple(globs)
    if proxies is not None:
        unknown = sorted(set(routes) - set(proxies))
        if unknown:
            raise TableError(f"tool_routes names tools that are not configured: {unknown}")
    request_tools = _tool_list(raw.get("request_tools") or [], "request_tools", routes)
    presets_raw = raw.get("presets")
    if not isinstance(presets_raw, dict) or not presets_raw:
        raise TableError("presets must be a non-empty object")
    presets = {}
    for name, p in presets_raw.items():
        if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
            raise TableError(f"preset name {name!r} must be 1 to 64 letters, digits, - or _")
        presets[name] = _parse_preset(name, p, routes)
    return Table(sealed=sealed, request_tools=request_tools,
                 tool_routes=routes, presets=presets)


def load_table(path: str, proxies: Iterable[str] | None = None) -> Table:
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, ValueError) as exc:
        raise TableError(f"cannot read {path}: {exc}") from exc
    return parse_table(raw, proxies)


# ---- module state (set once at boot) ----

_table: Table | None = None
#: Session id -> its Preset, or REQUEST for a score_request socket.
REQUEST = "request"
_bound: dict[str, Any] = {}
_bound_lock = threading.Lock()


def configure(table: Table | None) -> None:
    global _table
    _table = table
    with _bound_lock:
        _bound.clear()


def current() -> Table | None:
    return _table


def sealed() -> bool:
    return _table is not None and _table.sealed


def bind(session_id: str, what) -> None:
    with _bound_lock:
        _bound[session_id] = what


def unbind(session_id: str) -> None:
    with _bound_lock:
        _bound.pop(session_id, None)


def tool_allowed(session_id: str, tool: str, method: str, rest: str) -> bool:
    """May this session call ``method rest`` on ``tool``? A preset session
    calls its preset's tools, a request socket ``request_tools``, each only
    on the routes ``tool_routes`` lists. Any other session: everything on an
    open table (as without one), nothing on a sealed one."""
    table = _table
    if table is None:
        return True
    with _bound_lock:
        what = _bound.get(session_id)
    if what is None:
        return not table.sealed
    tools = table.request_tools if what == REQUEST else what.tools
    if tool not in tools:
        return False
    target = f"{method} {rest}"
    return any(fnmatch.fnmatchcase(target, g) for g in table.tool_routes.get(tool, ()))


# ---- errors ----


def error_body(code: str, detail: str) -> dict:
    """``{code, message, ref}``; ``detail`` is logged under ``ref``."""
    from acestep.engine.obs import logger

    code = code if code in SENTENCES else "failed"
    ref = secrets.token_hex(4)
    logger.warning("preset_error ref={} code={} detail={}", ref, code, detail)
    return {"code": code, "message": SENTENCES[code], "ref": ref}


# ---- session config and one-shots ----


def expand(table: Table, msg: Any) -> tuple[Preset, dict]:
    """The session config a ``{"preset", "values"}`` frame stands for."""
    if not isinstance(msg, dict) or "preset" not in msg:
        raise Refused("bad_request", "session config without a preset")
    preset = table.presets.get(msg.get("preset")) if isinstance(msg.get("preset"), str) else None
    if preset is None:
        raise Refused("bad_request", f"unknown preset {msg.get('preset')!r}")
    given = msg.get("values") or {}
    if not isinstance(given, dict):
        raise Refused("bad_request", "values must be an object")
    unknown = sorted(set(given) - set(preset.values))
    if unknown:
        raise Refused("bad_request", f"unknown values {unknown}")
    config = dict(preset.config)
    # text_only is honoured only with telemetry_version (see ws_adapter).
    config.setdefault("telemetry_version", 1)
    if isinstance(msg.get("client_id"), str):
        config["client_id"] = msg["client_id"]
    prompt = config.get("prompt")
    for name, spec in preset.values.items():
        v = given.get(name, spec.get("default"))
        if not _num(v):
            raise Refused("bad_request", f"value {name!r} missing or not a number: {v!r}")
        if v > spec["max"]:
            raise Refused("too_long", f"value {name!r}={v} above {spec['max']}")
        if v < spec["min"]:
            raise Refused("bad_request", f"value {name!r}={v} below {spec['min']}")
        config[spec["field"]] = float(v)
        if isinstance(prompt, str):
            prompt = prompt.replace("{" + name + "}", str(int(round(v))))
    if isinstance(prompt, str):
        config["prompt"] = prompt
    return preset, config


def plan(preset: Preset) -> dict:
    intake = None
    if preset.intake is not None:
        intake = f"/tools/{preset.intake['tool']}{preset.intake['path']}"
    return {
        "intake": intake,
        "refresh": preset.refresh,
        "versions": [{"id": v["id"]} for v in preset.versions],
        "params": [
            {"id": pid, "min": s["min"], "max": s["max"], "default": s["default"]}
            for pid, s in preset.params.items()
        ],
    }


def oneshot(preset: Preset, data: dict) -> dict:
    """``{request_id, version, values?}`` -> the session's ``oneshot``."""
    if "knobs" in data or "seeds" in data:
        raise Refused("bad_request", "oneshot carries knobs or seeds")
    vid = data.get("version")
    ver = next((v for v in preset.versions if v["id"] == vid), None)
    if ver is None:
        raise Refused("bad_request", f"unknown version {vid!r}")
    given = data.get("values") or {}
    if not isinstance(given, dict):
        raise Refused("bad_request", "values must be an object")
    unknown = sorted(set(given) - set(preset.params))
    if unknown:
        raise Refused("bad_request", f"unknown params {unknown}")
    knobs = dict(ver["knobs"])
    for pid, spec in preset.params.items():
        v = given.get(pid, spec["default"])
        if not _num(v):
            raise Refused("bad_request", f"param {pid!r} not a number: {v!r}")
        knobs[spec["knob"]] = float(min(max(v, spec["min"]), spec["max"]))
    return {
        "type": "oneshot",
        "request_id": str(data.get("request_id") or ""),
        "seeds": [ver["seed"]],
        "knobs": knobs,
    }


# ---- the socket ----

#: Outbound: the fields a client keeps, per message type, and how many
#: binary frames follow it. Every other type is dropped with its frames.
_KEEP = {
    "queued": ("position",),
    "oneshot_ready": ("request_id", "ids", "streams", "sample_rate", "channels",
                      "frames", "encoding", "redo_of"),
    "oneshot_slice": ("request_id", "version", "index", "count", "start_frame",
                      "frames", "channels", "sample_rate", "encoding"),
    "oneshot_done": ("request_id", "version", "frames", "channels",
                     "sample_rate", "windows", "sha256"),
    "swap_ready": ("duration", "sample_rate", "channels", "source_epoch",
                   "window", "frames"),
}
_READY_KEEP = ("session_id", "sample_rate", "channels", "duration", "window", "frames")


class PresetSocket:
    """A preset session's socket (``preset``), or a score_request socket
    (``preset`` None): every frame either way passes here."""

    def __init__(self, ws, preset: Preset | None):
        self._ws = ws
        self._preset = preset
        self._lock = threading.Lock()
        self._binaries = 0  # binary frames allowed to follow the last JSON
        self._pending: list = []

    def __getattr__(self, name):
        return getattr(self._ws, name)

    def __iter__(self):
        return iter(self._ws)

    def close(self, code: int = 1000, reason: str = "") -> None:
        self._ws.close(code, "")

    # -- outbound --

    def _filter(self, msg: dict) -> tuple[dict | None, int]:
        mtype = msg.get("type")
        if mtype == "ready":
            if self._preset is None:
                return {"type": "ready", "session_id": msg.get("session_id")}, 0
            out = {"type": "ready"}
            out.update({k: msg[k] for k in _READY_KEEP if k in msg})
            out["plan"] = plan(self._preset)
            return out, 0 if "window" in msg else 1
        if mtype == "error" or (isinstance(mtype, str) and mtype.endswith("_failed")):
            code = "busy" if msg.get("reason") == "queue_full" else _ERROR_CODES.get(
                msg.get("code"), "failed")
            out = {"type": mtype}
            if "request_id" in msg:
                out["request_id"] = msg["request_id"]
            out.update(error_body(code, json.dumps(msg, default=str)))
            return out, 0
        keep = _KEEP.get(mtype)
        if keep is None or (mtype != "queued" and self._preset is None):
            return None, 0
        out = {"type": mtype}
        out.update({k: msg[k] for k in keep if k in msg})
        if mtype in ("oneshot_ready", "oneshot_slice"):
            n = 1
        elif mtype == "swap_ready":
            n = 0 if "window" in msg else 1
        else:
            n = 0
        return out, n

    def send(self, data) -> None:
        with self._lock:
            if isinstance(data, str):
                try:
                    msg = json.loads(data)
                except ValueError:
                    msg = None
                out, n = self._filter(msg) if isinstance(msg, dict) else (None, 0)
                self._binaries = n
                if out is not None:
                    self._ws.send(json.dumps(out))
                return
            if self._binaries > 0:
                self._binaries -= 1
                self._ws.send(data)

    def refuse(self, refused: Refused, mtype: str = "error", **extra) -> None:
        body = {"type": mtype, **extra}
        body.update(error_body(refused.code, refused.detail))
        with self._lock:
            self._binaries = 0
            self._ws.send(json.dumps(body))

    # -- inbound --

    def _recv(self, timeout):
        return self._ws.recv() if timeout is None else self._ws.recv(timeout=timeout)

    def recv(self, timeout=None):
        if self._pending:
            return self._pending.pop(0)
        while True:
            msg = self._recv(timeout)
            if self._preset is None or not isinstance(msg, str):
                if self._preset is None:
                    return msg
                continue  # a stray binary frame: nothing reads it
            try:
                data = json.loads(msg)
            except ValueError:
                data = None
            mtype = data.get("type") if isinstance(data, dict) else None
            if mtype == "oneshot":
                try:
                    return json.dumps(oneshot(self._preset, data))
                except Refused as r:
                    self.refuse(r, "oneshot_failed",
                                request_id=str(data.get("request_id") or ""))
                    continue
            if mtype == "swap_source":
                try:
                    frame = self._recv(SWAP_FRAME_TIMEOUT_S)
                except TimeoutError:
                    frame = None
                if (isinstance(frame, (bytes, bytearray)) and len(frame) == 8
                        and struct.unpack("<II", frame)[1] == 0):
                    self._pending.append(bytes(frame))
                    return json.dumps({"type": "swap_source"})
                self.refuse(Refused("bad_request", "swap_source without the empty frame"),
                            "swap_failed")
                continue
            self.refuse(Refused("bad_request", f"command {mtype!r} refused"))
