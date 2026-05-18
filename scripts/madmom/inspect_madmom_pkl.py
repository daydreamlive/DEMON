"""Tolerant unpickler for madmom's key_cnn.pkl — reads the layer structure
without requiring madmom to be installed.

For every unknown class encountered while unpickling we substitute a stub
that records the class name plus whatever state the pickle attaches to
it (via __dict__ / __setstate__). Walking the resulting tree gives the
architecture and the numpy weight arrays.
"""
from __future__ import annotations

import io
import pickle
import sys
from typing import Any

import numpy as np


class Stub:
    """Captures (module.classname, state) for any madmom class."""

    _cls: str = "Stub"

    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs
        self._state: dict = {}

    def __setstate__(self, state: Any) -> None:
        if isinstance(state, dict):
            self._state = state
        else:
            self._state = {"__raw_state__": state}

    def __repr__(self) -> str:
        return f"<Stub {self._cls}>"


_FACTORIES: dict[tuple[str, str], type] = {}


def _factory(module: str, name: str) -> type:
    key = (module, name)
    if key not in _FACTORIES:
        cls_name = f"{module}.{name}"
        # Make _cls a class attribute so it's set even when pickle skips
        # __init__ (uses __new__ + __setstate__).
        sub = type(name, (Stub,), {"_cls": cls_name})
        _FACTORIES[key] = sub
    return _FACTORIES[key]


class TolerantUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        # Re-route numpy classes to the real numpy so arrays reconstruct
        # correctly; everything else gets a stub.
        if module.startswith("numpy"):
            return super().find_class(module, name)
        if module in {"builtins", "__builtin__", "copy_reg", "_codecs"}:
            return super().find_class(module, name)
        return _factory(module, name)


def load(path: str):
    with open(path, "rb") as f:
        return TolerantUnpickler(f, encoding="latin1").load()


def _shape_of(x: Any) -> str:
    if isinstance(x, np.ndarray):
        return f"ndarray{tuple(x.shape)} {x.dtype}"
    return type(x).__name__


def walk(node: Any, depth: int = 0, name: str = "root") -> None:
    indent = "  " * depth
    if isinstance(node, Stub):
        print(f"{indent}{name}: {node._cls}")
        for k, v in node._state.items():
            walk(v, depth + 1, k)
    elif isinstance(node, (list, tuple)):
        print(f"{indent}{name}: {type(node).__name__}[{len(node)}]")
        for i, v in enumerate(node):
            walk(v, depth + 1, f"[{i}]")
    elif isinstance(node, dict):
        print(f"{indent}{name}: dict({len(node)})")
        for k, v in node.items():
            walk(v, depth + 1, str(k))
    elif isinstance(node, np.ndarray):
        print(f"{indent}{name}: ndarray{tuple(node.shape)} {node.dtype}")
    else:
        print(f"{indent}{name}: {type(node).__name__} = {node!r}"[:160])


if __name__ == "__main__":
    obj = load(sys.argv[1])
    walk(obj)
