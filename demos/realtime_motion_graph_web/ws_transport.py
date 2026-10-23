"""WebSocket transport sizing for the demo server.

Two knobs, both read from the environment when the server starts:

``DEMON_WS_MAX_MESSAGE_BYTES`` (default 1 GiB): the largest inbound
message. A multichannel source upload is float32 PCM, so 300 s of 14
channels at 48 kHz is about 806 MB (60 s about 161 MB); the default
leaves headroom.

``DEMON_WS_RECV_BUFSIZE`` (default 4 MiB): bytes read per socket ``recv``
on each connection. websockets reads 64 KiB per call, and every call
needs the GIL; while generation threads hold it, a single upload of tens
of MB then needs thousands of GIL turns and can outlive the binary
payload deadline (``DEMON_BINARY_PAYLOAD_TIMEOUT_S``) although the bytes
already sit in the kernel buffer. Larger reads cut the turns per upload.
"""

from __future__ import annotations

import os

from websockets.sync.server import ServerConnection

DEFAULT_MAX_MESSAGE_BYTES = 1024 * 1024 * 1024
DEFAULT_RECV_BUFSIZE = 4 * 1024 * 1024
_MIN_RECV_BUFSIZE = 64 * 1024


def _env_int(name: str, default: int, minimum: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return max(minimum, int(raw))
    except ValueError:
        return default


def max_message_bytes() -> int:
    return _env_int("DEMON_WS_MAX_MESSAGE_BYTES", DEFAULT_MAX_MESSAGE_BYTES, 1024 * 1024)


def recv_bufsize() -> int:
    return _env_int("DEMON_WS_RECV_BUFSIZE", DEFAULT_RECV_BUFSIZE, _MIN_RECV_BUFSIZE)


def connection_class(bufsize: int | None = None) -> type[ServerConnection]:
    """A ``ServerConnection`` that reads ``bufsize`` bytes per ``recv``."""
    size = int(bufsize or recv_bufsize())

    class LargeRecvServerConnection(ServerConnection):
        recv_bufsize = size

    return LargeRecvServerConnection


def server_options() -> dict:
    """kwargs for ``websockets.sync.server.serve``."""
    return {"max_size": max_message_bytes(), "create_connection": connection_class()}


__all__ = ["DEFAULT_MAX_MESSAGE_BYTES", "DEFAULT_RECV_BUFSIZE", "max_message_bytes",
           "recv_bufsize", "connection_class", "server_options"]
