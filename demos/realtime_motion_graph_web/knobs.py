"""MIDI knob bank definitions.

Torch-free. Pure dataclasses + constants that both the full demo, the
server, and the thin client import. The activation-steering axis lists
and slot constants live in ``acestep.steering`` (the engine surface);
this module re-uses them through that import.
"""

from dataclasses import dataclass

from acestep.steering import (
    AUTO_AXES,
    MANUAL_MAX_LAYER,
    MANUAL_MAX_STEP,
)


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


def make_manual_slot_knobs(slot_id: int) -> dict:
    """Build the four KnobDef instances for one manual steering slot.

    CC=0 mirrors the LoRA dynamic-knob convention: software-side knob,
    operator can bind hardware via MIDI-learn.
    """
    return {
        f"man_src_{slot_id}": KnobDef(
            cc=0, default=0.0, sensitivity=0.5, max_val=143.0,
        ),
        f"man_layer_{slot_id}": KnobDef(
            cc=0, default=9.0, sensitivity=2.0, max_val=float(MANUAL_MAX_LAYER),
        ),
        f"man_step_{slot_id}": KnobDef(
            cc=0, default=0.0, sensitivity=2.0, max_val=float(MANUAL_MAX_STEP),
        ),
        f"man_alpha_{slot_id}": KnobDef(
            cc=0, default=0.0, sensitivity=2.0, max_val=30.0,
        ),
    }


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


def build_banks(sde: bool, loras=None, manual_slot_count: int = 0) -> list:
    """Build the knob banks driving the streaming pipeline.

    ``loras`` is an iterable of LoRA ids (filename stems).  Each id gets a
    ``lora_str_<id>`` knob with a freshly allocated CC slot.  Ids replace
    the old positional ``lora_str_1`` / ``lora_str_2`` naming so toggling
    catalog entries on and off doesn't shuffle knob identities.

    Backward-compat: an int ``loras`` is accepted and treated as
    ``[f"slot{i}" for i in range(1, n+1)]`` so callers that haven't
    migrated still get something usable, with the old-style names.

    ``manual_slot_count`` mirrors the SteeringController's slot count.
    Defaults to 0 so callers that don't pass it don't get phantom
    manual knobs.
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
    # Delay-tap depth for the feedback knob. 1 == blend with the most
    # recent finished latent (current behavior); N>1 reaches N ticks
    # back for an echo / ghost effect that the scalar feedback alone
    # can't reach without dominating the source. Integer-valued; capped
    # at 8 to match the StreamPipeline ring buffer ceiling and the
    # operator's mental tick budget.
    core["feedback_depth"] = KnobDef(
        cc=cc, default=1.0, sensitivity=4.0, max_val=8.0,
    ); cc += 1
    # Shift value flows verbatim into the diffusion solver. Useful operator
    # range is roughly [1, 6]; max_val caps the MIDI sweep at 6.
    core["shift"] = KnobDef(cc=cc, default=3.5, sensitivity=1.0, max_val=6.0); cc += 1
    if sde:
        core["periodicity"] = KnobDef(cc=cc, sensitivity=2.0, max_val=12.5); cc += 1
    for lid in lora_ids:
        core[f"lora_str_{lid}"] = KnobDef(
            cc=cc, default=0.0, sensitivity=2.0, max_val=2.0,
        )
        cc += 1
    core["hint_strength"] = KnobDef(cc=cc, default=1.0, sensitivity=2.0); cc += 1

    channels = {}
    for i, (name, _start, _end) in enumerate(CHANNEL_GROUPS):
        channels[name] = KnobDef(cc=70 + i, default=1.0, sensitivity=1.5, max_val=3.0)
    keystones = {}
    for i, (name, _ch) in enumerate(KEYSTONE_CHANNELS):
        keystones[name] = KnobDef(cc=70 + i, default=1.0, sensitivity=1.5, max_val=3.0)
    steering = {}
    for i, ax in enumerate(AUTO_AXES):
        # Wide testing range: default 0 = off. Per-step injection (the
        # research protocol) shifts the residual on only one of N denoise
        # steps per generation, so a given alpha produces a smaller
        # integrated effect than the prior all-step spike. The 30.0
        # ceiling lets the operator reach the research's alpha=3
        # equivalent and push past it. Useful range will be roughly
        # 5..15 by ear; breakage above that.
        steering[ax.name] = KnobDef(
            cc=70 + i, default=0.0, sensitivity=2.0, max_val=30.0,
        )

    # Manual steering. Runtime slot add/pop is mirrored into
    # VirtualMidiKnobs by the backend so the bank tracks the controller.
    manual = {}
    for slot in range(1, int(manual_slot_count) + 1):
        manual.update(make_manual_slot_knobs(slot))

    return [
        KnobBank(name="Core", knobs=core),
        KnobBank(name="Groups", knobs=channels),
        KnobBank(name="Keystones", knobs=keystones),
        KnobBank(name="Steering", knobs=steering),
        KnobBank(name="Manual", knobs=manual),
    ]
