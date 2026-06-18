"""Minimal terminal styling for human-facing CLI output (demon-setup and
the TensorRT build matrix).

Color is added ONLY when the target stream is an interactive terminal.
On any pipe, file, or captured subprocess — i.e. every test and every
downstream consumer — the gate is off and :func:`style` returns its input
unchanged, so the emitted bytes are byte-identical to the uncolored
output. There is intentionally no ``FORCE_COLOR`` override (it would
inject ANSI into captured pipes and break that guarantee); ``NO_COLOR``
and ``TERM=dumb`` hard-disable per no-color.org.

SGR codes are pure ASCII, so they can never raise ``UnicodeEncodeError``
and are only ever written to a real terminal anyway. Stdlib only.
"""

from __future__ import annotations

import os
import sys

_SGR = {
    "ok": "\x1b[32m",      # green
    "green": "\x1b[32m",   # green (alias)
    "warn": "\x1b[33m",    # yellow
    "fail": "\x1b[1;31m",  # bold red
    "yellow": "\x1b[33m",  # yellow
    "blue": "\x1b[34m",    # blue
    "magenta": "\x1b[35m", # magenta / purple
    "orange": "\x1b[38;2;255;184;108m",      # light orange (truecolor #FFB86C)
    "orange_b": "\x1b[1;38;2;255;184;108m",  # bold light orange
    "header": "\x1b[1;35m",# bold magenta (phase banners + titles)
    "accent": "\x1b[1m",   # bold
    "rule": "\x1b[2m",     # dim
    "dim": "\x1b[2m",      # dim secondary text
}
_RESET = "\x1b[0m"


def should_color(stream=None) -> bool:
    """True only on an interactive TTY; ``NO_COLOR`` / ``TERM=dumb`` win."""
    stream = sys.stdout if stream is None else stream
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    return bool(getattr(stream, "isatty", lambda: False)())


def style(text: str, token: str, stream=None) -> str:
    """Wrap ``text`` in the SGR code for ``token`` on a TTY; return ``text``
    unchanged otherwise. Off-TTY this is a pure no-op, so output stays
    byte-identical to the uncolored form."""
    if not should_color(stream):
        return text
    return f"{_SGR.get(token, '')}{text}{_RESET}"
