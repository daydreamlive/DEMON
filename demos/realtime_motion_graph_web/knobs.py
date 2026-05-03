"""MIDI knob bank definitions.

Torch-free, acestep-free. Pure dataclasses + constants that both the
full demo, the server, and the thin client import.
"""

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
    cc: int
    default: float = 0.0
    sensitivity: float = 2.0
    max_val: float = 1.0


@dataclass
class KnobBank:
    name: str
    knobs: dict


def build_banks(sde: bool, loras=None) -> list:
    """Build the knob banks driving the streaming pipeline.

    ``loras`` is an iterable of LoRA ids (filename stems).  Each id gets a
    ``lora_str_<id>`` knob with a freshly allocated CC slot.  Ids replace
    the old positional ``lora_str_1`` / ``lora_str_2`` naming so toggling
    catalog entries on and off doesn't shuffle knob identities.

    Backward-compat: an int ``loras`` is accepted and treated as
    ``[f"slot{i}" for i in range(1, n+1)]`` so callers that haven't
    migrated still get something usable, with the old-style names.
    """
    if isinstance(loras, int):
        lora_ids = [f"slot{i}" for i in range(1, loras + 1)]
    else:
        lora_ids = list(loras or [])

    core = {}
    cc = 70
    if sde:
        core["sde_amp"] = KnobDef(cc=cc, sensitivity=2.0); cc += 1
    else:
        core["denoise"] = KnobDef(cc=cc, sensitivity=2.0); cc += 1
    core["seed"] = KnobDef(cc=cc, sensitivity=0.5); cc += 1
    core["feedback"] = KnobDef(cc=cc, sensitivity=2.0); cc += 1
    core["shift"] = KnobDef(cc=cc, default=0.5, sensitivity=1.0); cc += 1
    if sde:
        core["periodicity"] = KnobDef(cc=cc, sensitivity=2.0, max_val=12.5); cc += 1
    for lid in lora_ids:
        core[f"lora_str_{lid}"] = KnobDef(
            cc=cc, default=0.0, sensitivity=2.0, max_val=2.0,
        )
        cc += 1
    core["hint_strength"] = KnobDef(cc=cc, default=1.0, sensitivity=2.0); cc += 1
    core["noise_share"] = KnobDef(cc=cc, default=0.0, sensitivity=2.0); cc += 1
    core["ode_noise"] = KnobDef(cc=cc, default=0.0, sensitivity=2.0, max_val=0.5); cc += 1

    channels = {}
    for i, (name, _start, _end) in enumerate(CHANNEL_GROUPS):
        channels[name] = KnobDef(cc=70 + i, default=1.0, sensitivity=1.5, max_val=3.0)
    keystones = {}
    for i, (name, _ch) in enumerate(KEYSTONE_CHANNELS):
        keystones[name] = KnobDef(cc=70 + i, default=1.0, sensitivity=1.5, max_val=3.0)

    return [
        KnobBank(name="Core", knobs=core),
        KnobBank(name="Groups", knobs=channels),
        KnobBank(name="Keystones", knobs=keystones),
    ]
