"""Family-agnostic steering layout: what a model family exposes to steering.

A :class:`~acestep.engine.model_adapter.ModelAdapter` (ACE included)
declares one :class:`SteeringLayout` through its optional
``steering_layout()`` method. The shared
:class:`~acestep.engine.stream.StreamPipeline` owns the single steering
slot (a ``[B, num_blocks, hidden_size]`` additive tensor per forward,
filled from the active vectors times their knob values times the per-step
policy weight) and delivers it the way the layout says:

* ``engine_input=True``: the loaded engine carries a ``steering`` input of
  shape ``[B, num_blocks, hidden_size]`` that it adds to each block's
  output residual (the convention ``acestep/engine/trt/export.py``
  established for the ACE decoder, and that the SA3 DiT surgery in
  ``acestep/engine/trt/sa3_steering_onnx.py`` reproduces).
* otherwise the adapter's ``steering_blocks()`` returns the eager block
  modules, and the pipeline attaches forward hooks that add the same
  shift to each block's output.

Pure data; no torch import, so pack tooling and manifest builders can
read layouts without paying for torch.
"""

from __future__ import annotations

from dataclasses import dataclass

#: The residual stream right after a transformer block (block output),
#: broadcast over every token.
HOOK_POST_BLOCK_RESIDUAL = "post_block_residual"

#: The output of a block's cross-attention module, before it is added to
#: the residual stream, broadcast over every token. This is the
#: intervention site of TADA (Staniszewski et al., arXiv 2602.11910):
#: activation patching localises concepts to cross-attention layers and
#: steering vectors are added to their outputs.
HOOK_CROSS_ATTN_OUTPUT = "cross_attn_output"


@dataclass(frozen=True)
class SteeringLayout:
    """Where steering vectors can land on one loaded model."""

    num_blocks: int
    hidden_size: int
    hook: str = HOOK_POST_BLOCK_RESIDUAL
    # True when the loaded (accelerated) forward takes the steering tensor
    # as an engine input; False when steering rides eager block hooks.
    engine_input: bool = False
    # Further hook points this loaded model also serves, with the same
    # block count and hidden size (e.g. ``cross_attn_output`` next to the
    # primary ``post_block_residual``). Each is delivered the same way as
    # the primary one (engine input or eager hooks).
    extra_hooks: tuple = ()

    @property
    def hooks(self) -> tuple:
        """Every hook point this layout serves, primary first."""
        return (self.hook,) + tuple(h for h in self.extra_hooks if h != self.hook)

    def accepts(self, block: int, hidden_size: int, hook: str) -> bool:
        """Whether a vector built for ``(block, hidden_size, hook)`` fits."""
        return (
            hook in self.hooks
            and int(hidden_size) == int(self.hidden_size)
            and 0 <= int(block) < int(self.num_blocks)
        )
