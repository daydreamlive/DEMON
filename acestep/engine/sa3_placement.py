"""Where an SA3 PyTorch part lives while a TensorRT engine serves it.

A stock SA3 model has two PyTorch parts an engine can replace: the
transformer and the decoder half of the SAME autoencoder. When the
engine serves, the PyTorch copy on the card is a duplicate (0.88 GB
transformer and 0.10 GB decoder for small-sfx, 2.77 GB transformer and
0.81 GB decoder for medium). :class:`CardPart` keeps such a part in system memory unless
something holds it on the card: a fallback that runs it takes a hold,
and the part moves back to system memory when its last holder lets go.

Parts are parked lazily, never at load: the first time an engine is
chosen for the part (:meth:`CardPart.park_if_free`), or when the last
hold is released. A process that never finds an engine and never closes
a session therefore keeps every part on the card, as before.

A context with a model extension pins both parts (every method is a
no-op): the extension owns its model and reads it live. The exception is
an extension that asks for its transformer on demand: that part is then
an ordinary holder, parked from boot, and the modules the extension names
as its keep stay on the card when the rest of the part moves. The decoder
stays pinned.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager

import torch

from acestep.engine.obs import logger

#: Where a free part waits.
_PARKED = torch.device("cpu")


def _module_mib(module) -> float:
    total = 0
    for tensor in list(module.parameters()) + list(module.buffers()):
        total += tensor.numel() * tensor.element_size()
    return total / (1024 * 1024)


class CardPart:
    """One PyTorch module that waits in system memory unless something
    holds it on the card. Thread-safe: the hold count and the placement
    change together under one lock per part."""

    def __init__(self, name: str, module, device, *, pinned: bool = False):
        self.name = str(name)
        self.module = module
        self.device = torch.device(device) if device is not None else None
        self.pinned = bool(pinned)
        #: Modules under ``module`` that stay where they are when the part
        #: moves (their parameters and buffers are never touched).
        self.keep: tuple = ()
        self._count = 0
        self._parked = False
        self._lock = threading.Lock()

    @property
    def count(self) -> int:
        return self._count

    @property
    def parked(self) -> bool:
        return self._parked

    def _move(self, device) -> float:
        """Move the module; returns the MiB moved. Called under the lock.

        Without a keep this is ``module.to(device)``. With one, the
        module's own parameters and buffers move module by module the way
        ``Module.to`` moves them (parameter objects keep their identity),
        skipping every module inside a keep subtree and every tensor one
        of them owns."""
        if not self.keep:
            self.module.to(device)
            return _module_mib(self.module)
        keep_ids = {id(m) for m in self.keep}
        kept = set()
        for m in self.keep:
            for tensor in list(m.parameters()) + list(m.buffers()):
                kept.add(id(tensor))
        moved = 0
        seen = set()
        new_bufs: dict = {}

        def walk(mod):
            nonlocal moved
            if id(mod) in keep_ids:
                return
            for name, param in list(mod._parameters.items()):
                if param is None or id(param) in kept or id(param) in seen:
                    continue
                seen.add(id(param))
                new = param.data.to(device)
                moved += param.numel() * param.element_size()
                if torch._has_compatible_shallow_copy_type(param, new):
                    param.data = new
                else:
                    # As Module.to does when the tensor types differ.
                    mod._parameters[name] = torch.nn.Parameter(
                        new, requires_grad=param.requires_grad,
                    )
            for name, buf in list(mod._buffers.items()):
                if buf is None or id(buf) in kept:
                    continue
                if id(buf) not in new_bufs:
                    new_bufs[id(buf)] = buf.to(device)
                    moved += buf.numel() * buf.element_size()
                mod._buffers[name] = new_bufs[id(buf)]
            for child in mod.children():
                walk(child)

        walk(self.module)
        return moved / (1024 * 1024)

    def _park(self) -> None:
        mib = self._move(_PARKED)
        self._parked = True
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("sa3_torch_part_parked part={} mib={:.0f}", self.name, mib)

    def hold(self, reason: str) -> None:
        """Take a hold; the first one brings a parked part to the card."""
        if self.pinned:
            return
        with self._lock:
            self._count += 1
            if self._count == 1 and self._parked:
                t0 = time.perf_counter()
                mib = self._move(self.device)
                self._parked = False
                logger.warning(
                    "sa3_torch_part_loaded part={} reason={} ms={:.0f} mib={:.0f}",
                    self.name, reason, (time.perf_counter() - t0) * 1000.0, mib,
                )

    def release(self) -> None:
        """Drop a hold; the last one sends the part to system memory."""
        if self.pinned:
            return
        with self._lock:
            if self._count <= 0:
                logger.error("sa3_torch_part_release_unheld part={}", self.name)
                return
            self._count -= 1
            if self._count == 0 and not self._parked:
                self._park()

    def park_if_free(self) -> None:
        """Send the part to system memory if nothing holds it."""
        if self.pinned:
            return
        with self._lock:
            if self._count == 0 and not self._parked:
                self._park()

    @contextmanager
    def held(self, reason: str):
        """Hold the part on the card for the ``with`` body."""
        self.hold(reason)
        try:
            yield self.module
        finally:
            self.release()


def pinned_part(name: str, module=None) -> CardPart:
    """A part that never moves (an extension's model, or a stand-in)."""
    return CardPart(name, module, None, pinned=True)
