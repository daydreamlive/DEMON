"""TADA: localised activation steering for audio diffusion models.

A port of the method of Staniszewski, Zaleska, Modrzejewski and Deja,
"TADA! Tuning Audio Diffusion Models through Activation Steering"
(arXiv 2602.11910; reference code github.com/luk-st/steer-audio, MIT),
written so that any DEMON model family can run it:

* :mod:`.target`: the capture and patching surface. A target exposes
  named hook points and a block count; recording and patching attach
  ordinary forward hooks, so nothing here knows a model.
* :mod:`.patching`: the activation-patching sweep that localises the
  "semantic bottleneck" (paper Sec. 3-4, Eq. 2, threshold tau = 0.10).
* :mod:`.caa`: contrastive activation addition on cross-attention
  outputs (paper Sec. 3, App. I.1.4, Eq. 6): per-step, per-layer,
  unit-norm difference of time-averaged means, applied as
  ``h + alpha * v`` on the conditional pass, optionally renormalised.
* :mod:`.packs`: CAA vectors as steering packs, so they load as
  ``steer_<concept>`` knobs through the family-agnostic steering slot.
* :mod:`.metrics`: the paper's evaluation (impact score, LPAPS
  preservation, sign-corrected alignment delta, alignment-preservation
  AUC, smoothness, quality sampling, strength grids).
* :mod:`.concepts`: the paper's concept sets, prompt templates,
  evaluation queries, benchmark prompts and calibrated ranges, loaded
  from ``data/`` (imported verbatim from the reference code by
  ``scripts/tada/import_upstream_data.py``).

Block numbering: the paper counts cross-attention layers from 1; this
package (like the reference code's ``tf<i>`` names) counts from 0. The
paper's ACE-Step bottleneck {7, 8} is blocks (6, 7) here.
"""

from .target import (
    HOOK_CROSS_ATTN_COND,
    HOOK_CROSS_ATTN_OUTPUT,
    ActivationRecorder,
    ActivationTarget,
    ConditioningPatcher,
    ModuleTarget,
    forward_counter,
)

__all__ = [
    "HOOK_CROSS_ATTN_COND",
    "HOOK_CROSS_ATTN_OUTPUT",
    "ActivationRecorder",
    "ActivationTarget",
    "ConditioningPatcher",
    "ModuleTarget",
    "forward_counter",
]
