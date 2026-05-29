"""Knob bank definitions + transport-agnostic knob state.

Torch-free, acestep-free. Pure dataclasses + constants.

The ``cc`` MIDI-controller-number field is intentionally absent from
``KnobDef`` because hardware MIDI binding is a transport concern that
the streaming session does not need to know about. The browser-side
MIDI store (``web/store/useMidiStore.ts``) owns its own CC mapping for
the operator's hardware controller.
"""

import threading
from dataclasses import dataclass


# Channel groups / keystones used by the server-side pipeline for
# channel-guided generation. Shared here so the client can display them.
CHANNEL_GROUPS = [
    ("ch_g0", 0, 7),   ("ch_g1", 8, 15),  ("ch_g2", 16, 23),
    ("ch_g3", 24, 31),  ("ch_g4", 32, 39),  ("ch_g5", 40, 47),
    ("ch_g6", 48, 55),  ("ch_g7", 56, 63),
]
KEYSTONE_CHANNELS = [
    ("ch13", 13), ("ch14", 14), ("ch19", 19),
    ("ch23", 23), ("ch29", 29), ("ch56", 56),
]


@dataclass
class KnobDef:
    default: float = 0.0
    sensitivity: float = 2.0
    max_val: float = 1.0


@dataclass
class KnobBank:
    name: str
    knobs: dict


def build_banks(sde: bool, loras=None) -> list:
    """Build the knob banks driving the streaming pipeline.

    ``loras`` is an iterable of LoRA ids (filename stems). Each id gets
    a ``lora_str_<id>`` knob. Ids replace the old positional
    ``lora_str_1`` / ``lora_str_2`` naming so toggling catalog entries
    on and off doesn't shuffle knob identities.

    Backward-compat: an int ``loras`` is accepted and treated as
    ``[f"slot{i}" for i in range(1, n+1)]`` so callers that haven't
    migrated still get something usable, with the old-style names.
    """
    if isinstance(loras, int):
        lora_ids = [f"slot{i}" for i in range(1, loras + 1)]
    else:
        lora_ids = list(loras or [])

    core = {}
    if sde:
        core["sde_amp"] = KnobDef(sensitivity=2.0)
    else:
        core["denoise"] = KnobDef(sensitivity=2.0)
    core["seed"] = KnobDef(sensitivity=0.5)
    core["feedback"] = KnobDef(sensitivity=2.0)
    # Delay-tap depth for the feedback knob. 1 == blend with the most
    # recent finished latent (current behavior); N>1 reaches N ticks
    # back for an echo / ghost effect that the scalar feedback alone
    # can't reach without dominating the source. Integer-valued; capped
    # at 8 to match the StreamPipeline ring buffer ceiling and the
    # operator's mental tick budget.
    core["feedback_depth"] = KnobDef(
        default=1.0, sensitivity=4.0, max_val=8.0,
    )
    # Shift value flows verbatim into the diffusion solver. Useful
    # operator range is roughly [1, 6]; max_val caps the sweep at 6.
    core["shift"] = KnobDef(default=3.5, sensitivity=1.0, max_val=6.0)
    if sde:
        core["periodicity"] = KnobDef(sensitivity=2.0, max_val=12.5)
    for lid in lora_ids:
        core[f"lora_str_{lid}"] = KnobDef(
            default=0.0, sensitivity=2.0, max_val=2.0,
        )
    core["hint_strength"] = KnobDef(default=1.0, sensitivity=2.0)

    channels = {}
    for (name, _start, _end) in CHANNEL_GROUPS:
        channels[name] = KnobDef(default=1.0, sensitivity=1.5, max_val=3.0)
    keystones = {}
    for (name, _ch) in KEYSTONE_CHANNELS:
        keystones[name] = KnobDef(default=1.0, sensitivity=1.5, max_val=3.0)

    return [
        KnobBank(name="Core", knobs=core),
        KnobBank(name="Groups", knobs=channels),
        KnobBank(name="Keystones", knobs=keystones),
    ]


class KnobState:
    """Transport-agnostic knob state.

    Drop-in replacement for the demo's old ``VirtualMidiKnobs`` class.
    Values come from whatever source the transport adapter wires up
    (WebSocket client params, an MCP command, a VST plugin parameter
    update). The streaming runner reads via ``get_param`` /
    ``get_all_values`` and never knows or cares about the transport.
    """

    def __init__(self, banks):
        self._banks = banks
        self._active_bank = 0
        self._values = {}
        self._all_knobs = {}
        for bank in banks:
            for name, k in bank.knobs.items():
                if name not in self._values:
                    self._values[name] = k.default
                self._all_knobs[name] = k
        self._lock = threading.Lock()

    def update(self, raw: dict):
        """Bulk-update values from a client raw dict."""
        with self._lock:
            self._values.update(raw)

    def add_knob(self, name, knob_def):
        """Register a new knob after construction (used when the client
        enables a LoRA at runtime and we need a ``lora_str_<id>`` slot)."""
        with self._lock:
            if name not in self._values:
                self._values[name] = knob_def.default
            self._all_knobs[name] = knob_def

    def remove_knob(self, name):
        with self._lock:
            self._values.pop(name, None)
            self._all_knobs.pop(name, None)

    def get(self, name: str) -> float:
        with self._lock:
            return self._values.get(name, 0.0)

    def get_all(self) -> dict:
        with self._lock:
            bank = self._banks[self._active_bank]
            return {name: self._values[name] for name in bank.knobs}

    def get_all_values(self) -> dict:
        with self._lock:
            return dict(self._values)

    def get_param(self, name: str) -> float:
        with self._lock:
            return self._values.get(name, 0.0)

    def all_knob_defs(self) -> dict:
        return dict(self._all_knobs)

    @property
    def active_bank_index(self) -> int:
        return self._active_bank

    @property
    def active_bank(self):
        return self._banks[self._active_bank]

    def release(self):
        pass
