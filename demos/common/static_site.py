"""Static-site serving shared by the demo backends.

Standalone no-build demos are plain directories of HTML/JS/CSS served
from the same port as the WebSocket backend so the page and the WS share
one origin. This module owns that serving logic so the
realtime-motion-graph server (or any future backend host) mounts a demo
with one table entry instead of growing bespoke route code per demo.

A demo opts in by dropping a ``demon.demo.json`` manifest in its
directory; the older repo-local ``demo.static.json`` name is still
accepted::

    {"route": "/arp", "entry": "index.html"}

:func:`discover_static_demos` scans ``demos/*`` for manifests and
returns the mount table; :func:`serve_static_mounts` resolves a request
path against it. The shared demon-client browser bundle
(``packages/demon-client/dist``, see its ``build.mjs``) is mounted at
:data:`SDK_ROUTE` so every static demo loads ONE copy of the SDK /
slice-decoder worker / audio worklet instead of vendoring its own.

Depends only on the ``websockets`` library - no torch, no acestep - so
``--no-backend`` UI-only servers can import it for free.
"""

from __future__ import annotations

import json
import mimetypes
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from websockets.datastructures import Headers
from websockets.http11 import Response

# Windows' registry can map .js to a non-JS MIME type (e.g. text/plain),
# which makes browsers refuse to evaluate <script type="module">. Force
# the correct types so static-demo ES modules load everywhere.
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/javascript", ".mjs")

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Where the shared demon-client browser bundle is mounted. Static demos
# reference "/sdk/demon-client.js", "/sdk/sliceDecoder.worker.js" and
# "/sdk/audio-worklet.js" against the backend origin.
SDK_ROUTE = "/sdk"

# Route prefixes owned by the backend's API/asset endpoints. A static
# demo manifest may not claim them.
RESERVED_ROUTES = ("/api", "/fixtures", "/user_uploads", "/videos", SDK_ROUTE)

MANIFEST_NAMES = ("demon.demo.json", "demo.static.json")
DEFAULT_ENTRY = "index.html"

_NO_CACHE_HEADERS = [
    ("Cache-Control", "no-store, must-revalidate"),
    ("Pragma", "no-cache"),
    ("Expires", "0"),
]


@dataclass(frozen=True)
class StaticMount:
    """A validated static demo mount."""

    route: str
    root: Path
    entry: str = DEFAULT_ENTRY


def sdk_dist_dir() -> Path:
    """The committed demon-client browser bundle directory."""
    return _REPO_ROOT / "packages" / "demon-client" / "dist"


def _validate_route(manifest: Path, route: object) -> str:
    if not isinstance(route, str) or not route.startswith("/") or route == "/":
        raise ValueError(f"{manifest}: route must be a non-root '/...' string")
    route = route.rstrip("/")
    if not route:
        raise ValueError(f"{manifest}: route must be a non-root '/...' string")
    if any(route == r or route.startswith(r + "/") for r in RESERVED_ROUTES):
        raise ValueError(
            f"{manifest}: route {route!r} collides with reserved routes "
            f"{RESERVED_ROUTES}"
        )
    return route


def _validate_entry(manifest: Path, root: Path, entry: object) -> str:
    if entry is None:
        entry = DEFAULT_ENTRY
    if not isinstance(entry, str) or not entry:
        raise ValueError(f"{manifest}: entry must be a relative file path string")
    decoded = urllib.parse.unquote(entry).replace("\\", "/")
    if decoded.startswith("/") or decoded in (".", ".."):
        raise ValueError(f"{manifest}: entry must be relative to the demo directory")
    candidate = (root / decoded).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"{manifest}: entry {entry!r} escapes the demo directory")
    return decoded


def _manifest_for_directory(path: Path) -> Path:
    for name in MANIFEST_NAMES:
        manifest = path / name
        if manifest.is_file():
            return manifest
    raise ValueError(f"{path}: expected one of {MANIFEST_NAMES}")


def load_static_demo(path: str | Path) -> StaticMount:
    """Load one static demo from a manifest file or demo directory."""
    source = Path(path).expanduser()
    manifest = _manifest_for_directory(source) if source.is_dir() else source
    if manifest.name not in MANIFEST_NAMES:
        raise ValueError(f"{manifest}: expected manifest named one of {MANIFEST_NAMES}")
    if not manifest.is_file():
        raise ValueError(f"{manifest}: manifest file does not exist")

    root = manifest.parent.resolve()
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{manifest}: invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{manifest}: manifest must be a JSON object")

    route = _validate_route(manifest, data.get("route"))
    entry = _validate_entry(manifest, root, data.get("entry"))
    return StaticMount(route=route, root=root, entry=entry)


def _add_mount(
    mounts: dict[str, StaticMount],
    mount: StaticMount,
    source: Path,
) -> None:
    existing = mounts.get(mount.route)
    if existing is not None:
        raise ValueError(
            f"{source}: route {mount.route!r} already claimed by {existing.root}"
        )
    mounts[mount.route] = mount


def discover_static_demos(demos_root: Path | None = None) -> dict[str, StaticMount]:
    """Scan ``demos/*`` for static demo manifests.

    Malformed manifests and reserved/duplicate routes raise immediately:
    a typo'd mount should fail the server at boot, not 404 mysteriously.
    """
    if demos_root is None:
        demos_root = _REPO_ROOT / "demos"
    mounts: dict[str, StaticMount] = {}
    for demo_dir in sorted(p for p in demos_root.iterdir() if p.is_dir()):
        try:
            manifest = _manifest_for_directory(demo_dir)
        except ValueError:
            continue
        mount = load_static_demo(manifest)
        _add_mount(mounts, mount, manifest)
    return mounts


def build_static_mounts(
    extra_demos: list[str | Path] | tuple[str | Path, ...] = (),
) -> dict[str, StaticMount]:
    """Build the full static mount table, including /sdk and external demos."""
    mounts: dict[str, StaticMount] = {
        SDK_ROUTE: StaticMount(route=SDK_ROUTE, root=sdk_dist_dir()),
    }
    for mount in discover_static_demos().values():
        _add_mount(mounts, mount, mount.root)
    for demo in extra_demos:
        mount = load_static_demo(demo)
        _add_mount(mounts, mount, Path(demo))
    return mounts


def _file_response(target: Path) -> Response:
    try:
        body = target.read_bytes()
    except OSError as e:
        msg = f"500 {e}\n".encode()
        return Response(
            500, "Internal Server Error",
            Headers([
                ("Content-Type", "text/plain; charset=utf-8"),
                ("Content-Length", str(len(msg))),
                *_NO_CACHE_HEADERS,
            ]),
            msg,
        )
    # ES modules must be served with a JS MIME type or the browser
    # refuses to evaluate them; guess_type covers .js/.css/.html/.wav
    # (with the .js override forced at module import above).
    ctype, _ = mimetypes.guess_type(target.name)
    return Response(
        200, "OK",
        Headers([
            ("Content-Type", ctype or "application/octet-stream"),
            ("Content-Length", str(len(body))),
            *_NO_CACHE_HEADERS,
        ]),
        body,
    )


def serve_static_site(
    path_only: str,
    route: str,
    root: Path,
    entry: str = DEFAULT_ENTRY,
) -> Response | None:
    """Serve ``path_only`` from ``root`` mounted at ``route``.

    ``route`` is slash-prefixed with no trailing slash (``"/arp"``). Bare
    ``route`` 301-redirects to ``route + "/"`` (otherwise the page's
    relative asset URLs resolve against ``/`` and break); ``route + "/"``
    maps to the manifest entry; everything else is a path-escape-guarded
    file lookup inside ``root``. Returns ``None`` when the path doesn't
    match the mount or names no file, so the caller can fall through to
    its 404.
    """
    root = root.resolve()
    if path_only == route:
        return Response(
            301, "Moved Permanently",
            Headers([
                ("Location", route + "/"),
                ("Content-Length", "0"),
                *_NO_CACHE_HEADERS,
            ]),
            b"",
        )
    if path_only == route + "/":
        candidate = (root / entry).resolve()
        target = candidate if candidate.is_relative_to(root) else None
    elif path_only.startswith(route + "/"):
        rel = urllib.parse.unquote(path_only[len(route) + 1:])
        candidate = (root / rel).resolve()
        # Reject anything that escapes root (symlink/`..` traversal). A
        # string-prefix check would also admit sibling dirs sharing the
        # root's name as a prefix (root "/x/arp" matching "/x/arp_other").
        target = candidate if candidate.is_relative_to(root) else None
    else:
        return None
    if target is None or not target.is_file():
        return None
    return _file_response(target)


def serve_static_mounts(
    path_only: str,
    mounts: Mapping[str, StaticMount | Path],
) -> Response | None:
    """Resolve ``path_only`` against a static mount table."""
    for route, mount in mounts.items():
        if isinstance(mount, StaticMount):
            root = mount.root
            entry = mount.entry
        else:
            root = mount
            entry = DEFAULT_ENTRY
        resp = serve_static_site(path_only, route, root, entry)
        if resp is not None:
            return resp
    return None
