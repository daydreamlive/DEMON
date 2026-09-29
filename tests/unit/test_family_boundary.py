"""The platform boundary: the shared core never names a model family.

``pipeline_runner``, ``session``, ``ws_adapter``, ``server``, ``protocol``,
``knobs`` and ``config`` are the platform. They change for platform
features and never for a family; a family lives in its own modules and
declares itself through ``FamilySpec`` (docs/FAMILIES.md). This test
walks the AST of each frozen file and fails on a family name used as a
string literal, an identifier or an import — outside a short allow-list
of what phase 1 has not moved yet. The allow-list must stay in use: an
entry the code no longer needs fails too, so the list only shrinks.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

FROZEN = (
    "acestep/streaming/pipeline_runner.py",
    "acestep/streaming/session.py",
    "acestep/streaming/generator_backend.py",
    "acestep/streaming/diffusion_backend.py",
    "acestep/streaming/knobs.py",
    "acestep/streaming/config.py",
    "demos/realtime_motion_graph_web/ws_adapter.py",
    "demos/realtime_motion_graph_web/server.py",
    "demos/realtime_motion_graph_web/protocol.py",
)

#: (file, kind, value) -> why it is still there. Remove an entry when the
#: code moves; the test fails if an entry is unused.
ALLOW = {
    ("acestep/streaming/config.py", "literal", "acestep"):
        "DEFAULT_FAMILY is spelled exactly once, here",
    ("demos/realtime_motion_graph_web/server.py", "literal", "ace"):
        "the warmup policy name 'ace_trt' the server compares against; "
        "becomes a callable hook when a family needs a different warmup",
}

# Family names come from the registry, so a third family is guarded the
# day it is registered. "ace" is the short form the ACE code uses.
from acestep.streaming.families import FAMILY_SPECS  # noqa: E402

_NAMES = sorted(set(FAMILY_SPECS) | {"ace"}, key=len, reverse=True)
_ALT = "|".join(re.escape(n) for n in _NAMES)
# A family name inside a string: "sa3", "sa3_duration_s", "acestep", but
# not a checkpoint directory like "acestep-v15-turbo", a CLI flag
# (--sa3-base-checkpoint) or a module path (acestep.streaming.warmup).
_LITERAL = re.compile(rf"(?<![a-z0-9_.-])({_ALT})(?![a-z0-9.-])", re.I)
# An identifier built on a family name, any case: sa3_duration_s,
# SA3_MAX_DURATION_S, evict_sa3_contexts, ACEStepBackend. The bare package
# name ``acestep`` is not a family reference and is excluded by the
# leading-underscore-or-start rule.
_IDENT = re.compile(rf"(^|_)({_ALT})(_|$|(?<=[a-z])[A-Z])", re.I)
# A family module imported into the core.
_MODULE = re.compile(rf"\.(({_ALT})_[a-z_]+|ace_backend|ace_session)(\.|$)", re.I)


def _docstring_nodes(tree: ast.AST) -> set:
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(
                getattr(body[0], "value", None), ast.Constant
            ) and isinstance(body[0].value.value, str):
                out.add(id(body[0].value))
    return out


def _prose_nodes(tree: ast.AST) -> set:
    """String constants that are documentation, not code: docstrings and
    the ``description=`` arguments of the wire registries. A family named
    in prose is not a branch."""
    out = _docstring_nodes(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "description":
            for sub in ast.walk(node.value):
                if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                    out.add(id(sub))
    return out


def _scan(rel: str) -> list:
    src = (REPO / rel).read_text(encoding="utf-8")
    tree = ast.parse(src)
    docstrings = _prose_nodes(tree)
    hits = []

    def hit(kind, value, node):
        hits.append((rel, kind, value, getattr(node, "lineno", 0)))

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) in docstrings:
                continue
            for m in _LITERAL.finditer(node.value):
                hit("literal", m.group(1), node)
        elif isinstance(node, ast.Name):
            if _IDENT.search(node.id):
                hit("name", node.id, node)
        elif isinstance(node, ast.Attribute):
            if _IDENT.search(node.attr):
                hit("name", node.attr, node)
        elif isinstance(node, ast.arg):
            if _IDENT.search(node.arg):
                hit("name", node.arg, node)
        elif isinstance(node, ast.keyword) and node.arg:
            if _IDENT.search(node.arg):
                hit("name", node.arg, node)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if _IDENT.search(node.name):
                hit("name", node.name, node)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if _MODULE.search("." + mod) or _MODULE.search("." + mod + "."):
                hit("import", mod, node)
            for alias in node.names:
                if _IDENT.search(alias.name):
                    hit("import", f"{mod}.{alias.name}", node)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if _MODULE.search("." + alias.name + "."):
                    hit("import", alias.name, node)
    return hits


@pytest.mark.parametrize("rel", FROZEN)
def test_core_file_names_no_family(rel: str):
    hits = _scan(rel)
    unexpected = [h for h in hits if (h[0], h[1], h[2]) not in ALLOW]
    assert not unexpected, (
        "family reference in a frozen core file; declare it on FamilySpec "
        "instead:\n" + "\n".join(f"  {f}:{ln} {k} {v!r}" for f, k, v, ln in unexpected)
    )


def test_allow_list_entries_are_still_needed():
    used = set()
    for rel in FROZEN:
        for f, k, v, _ln in _scan(rel):
            used.add((f, k, v))
    stale = sorted(set(ALLOW) - used)
    assert not stale, "remove from ALLOW, the code no longer needs it:\n" + "\n".join(map(str, stale))
