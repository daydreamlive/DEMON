"""Structured leveled logging for the DEMON server.

``configure()`` runs once at process start. Correlation IDs are bound
into log records via ``logger.contextualize`` at the call site (see
``handle_client``); ``spawn_thread`` propagates that binding into child
threads. Sinks are data-driven (list of names) so a future PostHog sink
can be added without touching call sites.
"""

from __future__ import annotations

import contextvars
import os
import sys
import threading
from typing import Any, Callable

from loguru import logger

from acestep.paths import load_local_config

_DEFAULT_LEVEL = "INFO"
_DEFAULT_SINKS = ("stderr_pretty",)
_DEFAULT_FILE_DIR = "logs/sessions"

_LEVEL_ENV = "DEMON_LOG_LEVEL"
_SINKS_ENV = "DEMON_LOG_SINKS"
_FILE_DIR_ENV = "DEMON_LOG_FILE_DIR"

_configured = False


def _resolved_config() -> dict:
    raw = load_local_config().get("logging", {})
    cfg = raw if isinstance(raw, dict) else {}

    level = os.environ.get(_LEVEL_ENV) or cfg.get("level") or _DEFAULT_LEVEL

    sinks_env = os.environ.get(_SINKS_ENV)
    if sinks_env is not None:
        sinks = tuple(s.strip() for s in sinks_env.split(",") if s.strip())
    else:
        cfg_sinks = cfg.get("sinks")
        sinks = tuple(cfg_sinks) if isinstance(cfg_sinks, list) and cfg_sinks else _DEFAULT_SINKS

    file_dir = os.environ.get(_FILE_DIR_ENV) or cfg.get("file_dir") or _DEFAULT_FILE_DIR

    return {
        "level": str(level).upper(),
        "sinks": sinks,
        "file_dir": file_dir,
    }


def _pretty_format(record: dict) -> str:
    # Inline session_id when present so a pod-side tail still threads per-client.
    sid = (record.get("extra") or {}).get("session_id")
    tag = f"[{sid}] " if sid else ""
    return (
        "<green>{time:HH:mm:ss.SSS}</green> "
        "<level>{level: <8}</level> "
        f"{tag}"
        "<cyan>{name}</cyan> - <level>{message}</level>\n"
    )


def _reconfigure_stream_utf8(stream) -> None:
    # Loguru's serialized records embed level icons (small emoji glyphs).
    # On Windows the default cp1252 codepage can't encode them and the
    # write raises UnicodeEncodeError, dropping every JSON line. Force
    # UTF-8 with ``errors="replace"`` so a stray un-encodable char in a
    # log message can't take down the sink. No-op when the stream
    # doesn't expose ``reconfigure`` (TextIOBase 3.7+; covers real stdio).
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is None:
        return
    try:
        reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def configure(*, force: bool = False) -> None:
    """Wire loguru sinks from resolved config. Idempotent unless ``force=True``."""
    global _configured
    if _configured and not force:
        return

    cfg = _resolved_config()
    level = cfg["level"]
    logger.remove()

    for sink in cfg["sinks"]:
        if sink == "stderr_pretty":
            # Local-dev default: human-readable, direct write, no enqueue
            # so the dev's terminal stays in lockstep with the process.
            _reconfigure_stream_utf8(sys.stderr)
            logger.add(sys.stderr, level=level, format=_pretty_format, enqueue=False)
        elif sink == "stdout_json":
            # ``enqueue=True`` moves serialization + write off the calling thread
            # so the audio-rate path never blocks on stdout buffer flushes.
            _reconfigure_stream_utf8(sys.stdout)
            logger.add(sys.stdout, level=level, serialize=True, enqueue=True)
        elif sink == "jsonl_file":
            # Daily rotation rather than per-session: the call-site
            # contextualize puts session_id in every record, so grepping
            # one file by id is enough.
            path = os.path.join(cfg["file_dir"], "demon-{time:YYYY-MM-DD}.jsonl")
            logger.add(
                path,
                level=level,
                serialize=True,
                enqueue=True,
                rotation="00:00",
                retention="7 days",
                encoding="utf-8",
            )
        else:
            logger.warning("unknown_log_sink name={}", sink)

    _configured = True


def spawn_thread(
    target: Callable[..., Any],
    *args: Any,
    name: str | None = None,
    daemon: bool = True,
    **kwargs: Any,
) -> threading.Thread:
    """``threading.Thread`` wrapper that inherits loguru's contextvars.

    Plain ``threading.Thread`` runs the target in a fresh context, so any
    ``logger.contextualize`` set in the spawning thread is lost. Copying
    the parent context here keeps session_id (and friends) on every
    record the child thread emits.
    """
    ctx = contextvars.copy_context()
    t = threading.Thread(
        target=lambda: ctx.run(target, *args, **kwargs),
        name=name,
        daemon=daemon,
    )
    t.start()
    return t


__all__ = ("configure", "spawn_thread", "logger")
