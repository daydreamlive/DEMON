"""Wire types for the ACE-Step node system.

Each type is a dataclass representing data that flows between nodes.
Types carry a TYPE_NAME class variable used for port validation:
connecting an output to an input requires matching TYPE_NAMEs.

Type categories:
  - Handle types (MODEL, VAE, CLIP): opaque references to loaded objects.
    All point to the same AceStepHandler instance but are distinct types
    so the port system prevents mis-wiring.
  - Tensor payload types (AUDIO, LATENT, CONDITIONING, etc.): carry the
    actual data produced/consumed by nodes.
  - Config types (CONFIG): wrap engine configuration dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Optional, Tuple, List

import torch

from acestep.engine.diffusion import DiffusionConfig
from acestep.engine.masking import LatentNoiseMask

if TYPE_CHECKING:
    from acestep.engine.dcw import DCWAdvanced
    from acestep.engine.model_context import ModelContext


# -----------------------------------------------------------------------
# Type registry
# -----------------------------------------------------------------------

_TYPE_REGISTRY: dict[str, type] = {}


def _register(cls: type) -> type:
    """Register a wire type by its TYPE_NAME."""
    _TYPE_REGISTRY[cls.TYPE_NAME] = cls
    return cls


def get_type_class(type_name: str) -> type | None:
    """Look up a wire type class by name."""
    return _TYPE_REGISTRY.get(type_name)


def all_type_names() -> list[str]:
    """Return all registered type names."""
    return list(_TYPE_REGISTRY.keys())


def types_compatible(source_type: str, target_type: str) -> bool:
    """Check whether a source port type can connect to a target port type."""
    if source_type == target_type:
        return True
    # "ANY" accepts anything (for utility nodes like reroute)
    if target_type == "ANY" or source_type == "ANY":
        return True
    return False


# -----------------------------------------------------------------------
# Handle types (opaque references to loaded objects)
# -----------------------------------------------------------------------

@_register
@dataclass
class ModelHandle:
    """Reference to the loaded ACE-Step model (works with ModelContext or AceStepHandler)."""
    TYPE_NAME: ClassVar[str] = "MODEL"
    handler: Any  # ModelContext | AceStepHandler (duck-typed)


@_register
@dataclass
class VAEHandle:
    """Reference to the VAE (works with ModelContext or AceStepHandler)."""
    TYPE_NAME: ClassVar[str] = "VAE"
    handler: Any  # ModelContext | AceStepHandler (duck-typed)


@_register
@dataclass
class CLIPHandle:
    """Reference to the text encoder/tokenizer (works with ModelContext or AceStepHandler)."""
    TYPE_NAME: ClassVar[str] = "CLIP"
    handler: Any  # ModelContext | AceStepHandler (duck-typed)


# -----------------------------------------------------------------------
# Tensor payload types
# -----------------------------------------------------------------------

@_register
@dataclass
class Audio:
    """Waveform audio data."""
    TYPE_NAME: ClassVar[str] = "AUDIO"
    waveform: torch.Tensor  # [B, channels, samples]
    sample_rate: int = 48000
    start_sample: int = 0  # sample offset into the full signal (for windowed decode)


@_register
@dataclass
class Latent:
    """VAE-encoded audio latent, optionally carrying a noise mask."""
    TYPE_NAME: ClassVar[str] = "LATENT"
    tensor: torch.Tensor  # [B, T, D]
    mask: Optional[LatentNoiseMask] = None


@dataclass
class ConditioningEntry:
    """One condition within a combined set, with compositing metadata.

    Used internally by ConditioningCombine to attach temporal_weight
    and step_range to individual conditions within a Conditioning
    payload. Not a wire type (no TYPE_NAME).
    """
    encoder_hidden_states: torch.Tensor  # [B, L_enc, D]
    encoder_attention_mask: torch.Tensor  # [B, L_enc]
    temporal_weight: Optional[torch.Tensor] = None  # [T], [B,T], or [B,T,1]
    step_range: Optional[Tuple[float, float]] = None


@_register
@dataclass
class Conditioning:
    """Encoded cross-attention conditioning for the diffusion decoder.

    Contains encoder_hidden_states (packed text + lyrics + timbre) and
    the corresponding attention mask. Context latents (src_latents +
    chunk_mask) are built separately by Generate from explicit inputs.

    Can represent a single condition (from EncodeConditioning) or a
    combined set (from ConditioningCombine). When entries is None, the
    top-level tensors represent a single condition. When entries is
    populated, those are the authoritative conditions (top-level
    tensors are ignored).
    """
    TYPE_NAME: ClassVar[str] = "CONDITIONING"

    # Single condition tensors (populated by EncodeConditioning and similar)
    encoder_hidden_states: Optional[torch.Tensor] = None  # [B, L_enc, D]
    encoder_attention_mask: Optional[torch.Tensor] = None  # [B, L_enc]

    # Combined conditions (populated by ConditioningCombine)
    entries: Optional[List[ConditioningEntry]] = field(default=None, repr=False)

    @property
    def is_combined(self) -> bool:
        return self.entries is not None and len(self.entries) > 0

    def to_entries(self) -> List[ConditioningEntry]:
        """Return all conditions as a list of ConditioningEntry."""
        if self.entries is not None:
            return self.entries
        if self.encoder_hidden_states is None:
            return []
        return [
            ConditioningEntry(
                encoder_hidden_states=self.encoder_hidden_states,
                encoder_attention_mask=self.encoder_attention_mask,
            )
        ]


@_register
@dataclass
class Mask:
    """Per-frame spatial mask, values in [0, 1].

    Used for latent noise masking (which regions to preserve vs generate)
    and conditioning spatial blending.
    """
    TYPE_NAME: ClassVar[str] = "MASK"
    tensor: torch.Tensor  # [T] or [B, T]


@_register
@dataclass
class Curve:
    """Per-frame modulation signal, arbitrary range.

    Used for velocity scaling, SDE denoise curves, initial noise curves,
    x0 target blend curves, and any other per-frame parameter modulation.
    """
    TYPE_NAME: ClassVar[str] = "CURVE"
    tensor: torch.Tensor  # [T] or [B, T]


@_register
@dataclass
class Config:
    """Diffusion loop configuration."""
    TYPE_NAME: ClassVar[str] = "CONFIG"
    config: DiffusionConfig


@_register
@dataclass
class TextEmbed:
    """Tokenized + text-encoder-embedded prompt payload.

    Produced by ``EncodeText``; consumed by ``EncodeConditioning``. Holds
    the pre-encoder outputs of the text encoder for both the text prompt
    (tags + metadata) and the lyric prompt. Kept separate from
    ``Conditioning`` so that the text-encoding stage (tokenize, embed)
    is distinguishable from the conditioning-fusion stage (model.encoder
    with optional timbre reference).
    """
    TYPE_NAME: ClassVar[str] = "TEXT_EMBED"

    text_hidden_states: torch.Tensor   # [B, L_text, D]
    text_attention_mask: torch.Tensor  # [B, L_text]
    lyric_hidden_states: torch.Tensor   # [B, L_lyric, D]
    lyric_attention_mask: torch.Tensor  # [B, L_lyric]


@_register
@dataclass
class Solver:
    """Diffusion solver: which step function to use, plus a solver-specific curve.

    Produced by ``OdeSolver`` or ``SdeSolver`` nodes; consumed by
    ``StreamDenoise`` / ``Generate``. Moves the solver choice from a
    widget on the denoiser to an explicit graph edge, so the curve a
    solver uses (``ode_noise_curve`` for ODE, ``sde_denoise_curve`` for
    SDE) travels with the solver and can be type-checked structurally
    rather than silently ignored.

    Fields:
        method: "ode" or "sde".
        noise_curve: Solver-specific per-frame modulation tensor or
            ``None``. For ODE this is per-step noise injection
            (zeros-sentinel when absent); for SDE this is per-frame
            denoise blending (requires ``source_latent`` on the
            consumer to take effect).
    """
    TYPE_NAME: ClassVar[str] = "SOLVER"
    method: str
    noise_curve: Optional[torch.Tensor] = None


@_register
@dataclass
class DCW:
    """DCW (Differential Correction in Wavelet domain) settings bundle.

    Produced by ``DCWConfig``; consumed by ``StreamDenoise`` /
    ``Generate`` through an optional ``dcw`` input port. When unwired,
    the denoiser falls back to its hidden ``dcw_*`` NodeParam defaults
    (enabled=True, mode='double', scaler=0.05, high_scaler=0.02,
    wavelet='haar'), matching upstream v0.1.7.

    DCW is conceptually a peer of ``Solver``, sampler-side: it shapes
    the post-step ``xt`` correction rather than the velocity or the
    integration math.

    See ``acestep.engine.dcw`` for the math.
    """
    TYPE_NAME: ClassVar[str] = "DCW"
    enabled: bool = True
    mode: str = "double"
    scaler: float = 0.05
    high_scaler: float = 0.02
    wavelet: str = "haar"
    # Opt-in advanced surface. ``None`` keeps the upstream v0.1.7
    # behaviour (fast path). See :class:`DCWAdvanced` in
    # ``acestep.engine.dcw`` for the field list.
    advanced: Optional["DCWAdvanced"] = None


@_register
@dataclass
class Modulation:
    """Optional denoiser modulation bundle.

    Produced by the ``Modulation`` builder node; consumed by
    ``StreamDenoise`` / ``Generate`` through a single ``modulation``
    port. Keeping every optional denoiser input under one wire type
    makes the denoiser's surface minimal by default — the complexity
    only appears in the graph when a user opts in by adding a
    Modulation builder.

    All fields default to ``None``/``0.0`` so users can wire just the
    modulation they want and leave the rest alone. The scalar
    ``x0_target_strength`` and ``x0_target_gate`` live here (not as
    StreamDenoise widgets) because they're structurally coupled to
    ``x0_target`` — they're useless without the target latent.
    """
    TYPE_NAME: ClassVar[str] = "MODULATION"

    velocity_scale: Optional[torch.Tensor] = None
    initial_noise_curve: Optional[torch.Tensor] = None
    chunk_mask: Optional[torch.Tensor] = None
    x0_target: Optional["Latent"] = None
    x0_target_curve: Optional[torch.Tensor] = None
    x0_target_strength: float = 0.0
    x0_target_gate: float = 0.0
    guidance_curve: Optional[torch.Tensor] = None


# -----------------------------------------------------------------------
# Channel guidance (not a wire type; attached to handler, not a port)
# -----------------------------------------------------------------------

@dataclass
class ChannelGuidanceEntry:
    """One channel range to scale during denoising.

    Attached to the handler by the ChannelGuidance node.  The diffusion
    engine and stream pipeline read these to build a per-channel gain
    tensor applied to ``xt`` before each forward pass (input scaling).

    Fields:
        channel_start: First channel index (0-63 inclusive).
        channel_end: Last channel index (0-63 inclusive).
        scale: Multiplicative gain for these channels.  May be a scalar
            (uniform across the timeline) or a per-frame curve as a
            ``[T]`` / ``[1, T]`` / ``[1, T, 1]`` tensor.  Mixing scalar
            and curve entries in one config list is fine; entries live
            in disjoint channel ranges and the builder handles each.
    """
    channel_start: int
    channel_end: int
    scale: "float | torch.Tensor"
