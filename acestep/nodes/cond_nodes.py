"""Conditioning nodes: text encoding, zeroing, averaging, combining."""

from __future__ import annotations

import torch
from typing import Any, ClassVar, Optional

from .base import BaseNode, NodeDefinition, NodeParam, NodePort, NodeRegistry
from .interpolation import INTERP_METHOD_NAMES, INTERPOLATIONS
from .types import (
    CLIPHandle,
    Conditioning,
    ConditioningEntry,
    Latent,
    Mask,
    ModelHandle,
    TextEmbed,
)
from ..constants import (
    COMMON_KEYSCALES,
    TASK_INSTRUCTIONS,
    TASK_TYPES,
    VALID_LANGUAGES,
    VALID_TIME_SIGNATURES,
)


@NodeRegistry.register
class EncodeText(BaseNode):
    """Tokenize + text-encode the prompt (tags, metadata, lyrics).

    Produces a ``TextEmbed`` consumed by ``EncodeConditioning``. Does
    NOT handle the timbre reference or the final ``model.encoder``
    fusion — those live on ``EncodeConditioning`` so the text step and
    the fusion step can be reasoned about separately.

    Node parameters:
        tags: Genre/style tags string.
        lyrics: Song lyrics (empty string for instrumental).
        instruction: Instruction text for the model. Standard options:
            - "Fill the audio semantic mask based on the given conditions:" (text2music)
            - "Generate audio semantic tokens based on the given conditions:" (cover)
            - "Repaint the mask area based on the given conditions:" (repaint)
        bpm: Beats per minute.
        duration: Duration in seconds.
        key: Musical key (e.g. "G# minor").
        time_signature: Time signature (e.g. "4").
        language: Language code (e.g. "en").
    """

    node_type_id: ClassVar[str] = "acestep.EncodeText"

    @classmethod
    def get_definition(cls) -> NodeDefinition:
        return NodeDefinition(
            node_type_id=cls.node_type_id,
            display_name="Encode Text",
            category="conditioning",
            description="Tokenize and text-encoder-embed tags + lyrics + metadata.",
            inputs=(
                NodePort(name="clip", type="CLIP"),
            ),
            outputs=(
                NodePort(name="text_embed", type="TEXT_EMBED"),
            ),
            params=(
                NodeParam(
                    name="tags", type="string", default="",
                    description="Style tags",
                ),
                NodeParam(
                    name="lyrics", type="string", default="",
                    description="Lyrics (empty = instrumental)",
                ),
                NodeParam(
                    name="task_type", type="select", default="cover",
                    description="Task",
                    options=tuple(TASK_TYPES),
                ),
                NodeParam(
                    name="bpm", type="integer", default=120,
                    description="BPM",
                    min=30, max=300, step=1,
                ),
                NodeParam(
                    name="duration", type="number", default=60.0,
                    description="Duration (s)",
                    min=1, max=600, step=1,
                ),
                NodeParam(
                    name="key", type="select", default="C major",
                    description="Key",
                    options=tuple(COMMON_KEYSCALES),
                ),
                NodeParam(
                    name="time_signature", type="select", default="4",
                    description="Time signature",
                    options=tuple(str(s) for s in VALID_TIME_SIGNATURES),
                ),
                NodeParam(
                    name="language", type="select", default="en",
                    description="Language",
                    options=tuple(VALID_LANGUAGES),
                ),
                NodeParam(
                    name="instruction", type="string", default="",
                    description="Advanced: raw instruction override (empty = derive from task_type)",
                    hidden=True,
                ),
            ),
        )

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        clip: CLIPHandle = kwargs["clip"]
        handler = clip.handler
        device = handler.device

        tags = kwargs.get("tags", "")
        lyrics = kwargs.get("lyrics", "")
        # instruction overrides task_type when explicitly set; otherwise
        # the task_type widget drives the instruction via TASK_INSTRUCTIONS.
        instruction = kwargs.get("instruction") or ""
        if not instruction:
            task_type = kwargs.get("task_type", "text2music")
            instruction = TASK_INSTRUCTIONS.get(
                task_type, TASK_INSTRUCTIONS["text2music"]
            )
        bpm = kwargs.get("bpm", 120)
        duration = kwargs.get("duration", 60.0)
        key = kwargs.get("key", "C major")
        time_signature = kwargs.get("time_signature", "4")
        language = kwargs.get("language", "en")

        # --- Build text prompt ---
        meta_cap = (
            f"- bpm: {bpm}\n"
            f"- timesignature: {time_signature}\n"
            f"- keyscale: {key}\n"
            f"- duration: {duration}\n"
        )
        text_prompt = (
            f"# Instruction\n{instruction}\n\n"
            f"# Caption\n{tags}\n\n"
            f"# Metas\n{meta_cap}"
            f"<|endoftext|>\n"
        )

        # --- Build lyrics prompt ---
        if lyrics:
            lyrics_prompt = f"# Languages\n{language}\n\n# Lyric\n{lyrics}<|endoftext|><|endoftext|>"
        else:
            lyrics_prompt = f"# Languages\n{language}\n\n# Lyric\n<|endoftext|><|endoftext|>"

        # --- Tokenize and embed ---
        with handler._load_model_context("text_encoder"):
            tokens = handler.text_tokenizer(
                text_prompt, return_tensors="pt", add_special_tokens=False
            )
            text_hidden = handler.infer_text_embeddings(
                tokens["input_ids"].to(device)
            )
            text_mask = tokens["attention_mask"].to(device).bool()

            lyric_tokens = handler.text_tokenizer(
                lyrics_prompt, return_tensors="pt", add_special_tokens=False
            )
            lyric_hidden = handler.infer_lyric_embeddings(
                lyric_tokens["input_ids"].to(device)
            )
            lyric_mask = torch.ones(
                lyric_hidden.shape[:2], device=device, dtype=torch.bool
            )

        return {
            "text_embed": TextEmbed(
                text_hidden_states=text_hidden,
                text_attention_mask=text_mask,
                lyric_hidden_states=lyric_hidden,
                lyric_attention_mask=lyric_mask,
            ),
        }


@NodeRegistry.register
class EncodeConditioning(BaseNode):
    """Fuse text embedding + optional timbre reference into Conditioning.

    Consumes a ``TextEmbed`` (from ``EncodeText``) and optionally a
    ``timbre_ref: LATENT``, calling ``model.encoder`` to produce the
    final cross-attention ``Conditioning`` the diffusion decoder sees.

    When ``timbre_ref`` is absent, the encoder uses the handler's
    silence latent as the reference. Callers who want a partial timbre
    blend should do that upstream with ``LatentBlend`` between silence
    and their source latent — it's not a knob on this node.
    """

    node_type_id: ClassVar[str] = "acestep.EncodeConditioning"

    @classmethod
    def get_definition(cls) -> NodeDefinition:
        return NodeDefinition(
            node_type_id=cls.node_type_id,
            display_name="Encode Conditioning",
            category="conditioning",
            description="Fuse text embedding + optional timbre reference into Conditioning.",
            inputs=(
                NodePort(name="model", type="MODEL"),
                NodePort(name="text_embed", type="TEXT_EMBED"),
                NodePort(
                    name="timbre_ref",
                    type="LATENT",
                    required=False,
                    description="Timbre reference latent. Defaults to silence.",
                ),
            ),
            outputs=(
                NodePort(name="conditioning", type="CONDITIONING"),
            ),
        )

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        model: ModelHandle = kwargs["model"]
        text_embed: TextEmbed = kwargs["text_embed"]
        timbre_ref: Optional[Latent] = kwargs.get("timbre_ref")

        handler = model.handler
        device = handler.device
        dtype = handler.dtype

        if timbre_ref is not None:
            refer_packed = timbre_ref.tensor.to(device=device, dtype=dtype)
        else:
            handler._ensure_silence_latent_on_device()
            refer_packed = handler.silence_latent[:, :750, :].to(
                device=device, dtype=dtype,
            )
        refer_order_mask = torch.zeros(1, device=device, dtype=torch.long)

        with handler._load_model_context("model"):
            enc_hidden, enc_mask = handler.model.encoder(
                text_hidden_states=text_embed.text_hidden_states.to(dtype),
                text_attention_mask=text_embed.text_attention_mask,
                lyric_hidden_states=text_embed.lyric_hidden_states.to(dtype),
                lyric_attention_mask=text_embed.lyric_attention_mask,
                refer_audio_acoustic_hidden_states_packed=refer_packed,
                refer_audio_order_mask=refer_order_mask,
            )

        return {
            "conditioning": Conditioning(
                encoder_hidden_states=enc_hidden,
                encoder_attention_mask=enc_mask,
            )
        }


@NodeRegistry.register
class ConditioningZeroOut(BaseNode):
    """Zero out the encoder hidden states to produce an unconditional embedding.

    Used as the negative/uncond input for CFG when guidance_scale > 1.0.
    With the turbo model (guidance_scale=1.0), this is effectively ignored.
    """

    node_type_id: ClassVar[str] = "acestep.ConditioningZeroOut"

    @classmethod
    def get_definition(cls) -> NodeDefinition:
        return NodeDefinition(
            node_type_id=cls.node_type_id,
            display_name="Conditioning Zero Out",
            category="conditioning",
            description="Zero out encoder hidden states for unconditional embedding.",
            inputs=(
                NodePort(name="conditioning", type="CONDITIONING"),
            ),
            outputs=(
                NodePort(name="conditioning", type="CONDITIONING"),
            ),
        )

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        cond: Conditioning = kwargs["conditioning"]
        entries = cond.to_entries()
        if not entries:
            return {"conditioning": cond}

        entry = entries[0]
        return {
            "conditioning": Conditioning(
                encoder_hidden_states=torch.zeros_like(entry.encoder_hidden_states),
                encoder_attention_mask=entry.encoder_attention_mask,
            )
        }


@NodeRegistry.register
class ConditioningAverage(BaseNode):
    """Blend two conditionings by weighted average.

    Interpolates the encoder hidden states. Attention mask
    is taken from conditioning_a.

    Node parameters:
        weight: Blend weight. 0.0 = all A, 1.0 = all B.
    """

    node_type_id: ClassVar[str] = "acestep.ConditioningAverage"

    @classmethod
    def get_definition(cls) -> NodeDefinition:
        return NodeDefinition(
            node_type_id=cls.node_type_id,
            display_name="Conditioning Average",
            category="conditioning",
            description="Weighted average of two conditionings.",
            inputs=(
                NodePort(name="conditioning_a", type="CONDITIONING"),
                NodePort(name="conditioning_b", type="CONDITIONING"),
            ),
            outputs=(
                NodePort(name="conditioning", type="CONDITIONING"),
            ),
            params=(
                NodeParam(
                    name="weight", type="number", default=0.5,
                    description="Blend weight (0 = all A, 1 = all B)",
                    min=0.0, max=1.0, step=0.01,
                ),
            ),
        )

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        cond_a: Conditioning = kwargs["conditioning_a"]
        cond_b: Conditioning = kwargs["conditioning_b"]
        weight = kwargs.get("weight", 0.5)

        entries_a = cond_a.to_entries()
        entries_b = cond_b.to_entries()
        if not entries_a or not entries_b:
            return {"conditioning": cond_a}

        a = entries_a[0]
        b = entries_b[0]

        w = float(weight)

        # Match encoder_hidden_states lengths
        enc_a = a.encoder_hidden_states
        enc_b = b.encoder_hidden_states
        len_a = enc_a.shape[1]
        len_b = enc_b.shape[1]
        if len_b > len_a:
            enc_b = enc_b[:, :len_a]
        elif len_b < len_a:
            enc_b = torch.nn.functional.pad(enc_b, (0, 0, 0, len_a - len_b))

        blended_enc = (1.0 - w) * enc_a + w * enc_b

        return {
            "conditioning": Conditioning(
                encoder_hidden_states=blended_enc,
                encoder_attention_mask=a.encoder_attention_mask,
            )
        }


@NodeRegistry.register
class ConditioningBlend(BaseNode):
    """Interpolate two conditionings by a scalar alpha, padding to max length.

    Unlike ``ConditioningAverage`` (which truncates to the shorter
    sequence and keeps A's mask), this node zero-pads both conditionings
    to their common max length and emits a mask that is the elementwise
    ``max`` of the two inputs. Intended for live conditioning crossfades
    in streaming graphs — the caller updates ``alpha`` between ticks to
    glide from A to B.

    Node parameters:
        alpha: Blend factor (0.0 = all A, 1.0 = all B).
    """

    node_type_id: ClassVar[str] = "acestep.ConditioningBlend"

    @classmethod
    def get_definition(cls) -> NodeDefinition:
        return NodeDefinition(
            node_type_id=cls.node_type_id,
            display_name="Conditioning Blend",
            category="conditioning",
            description="Pad-and-blend two conditionings (streaming-friendly).",
            inputs=(
                NodePort(name="conditioning_a", type="CONDITIONING"),
                NodePort(name="conditioning_b", type="CONDITIONING"),
            ),
            outputs=(
                NodePort(name="conditioning", type="CONDITIONING"),
            ),
            params=(
                NodeParam(
                    name="alpha", type="number", default=0.5,
                    description="Blend factor (0 = all A, 1 = all B)",
                    min=0.0, max=1.0, step=0.01,
                ),
                NodeParam(
                    name="method", type="select", default="linear",
                    description=(
                        "Interpolation method (see acestep.nodes.interpolation). "
                        "'linear' (default) averages the two conditionings, which "
                        "dips the norm at the midpoint so the blend sounds washed "
                        "out halfway. 'slerp' walks the per-token geodesic instead, "
                        "holding each token's norm constant across the sweep."
                    ),
                    options=INTERP_METHOD_NAMES,
                ),
            ),
        )

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        cond_a: Conditioning = kwargs["conditioning_a"]
        cond_b: Conditioning = kwargs["conditioning_b"]
        alpha = float(kwargs.get("alpha", 0.5))
        method = str(kwargs.get("method", "linear"))

        ea = cond_a.to_entries()[0]
        eb = cond_b.to_entries()[0]
        sa, sb = ea.encoder_hidden_states, eb.encoder_hidden_states
        ma, mb = ea.encoder_attention_mask, eb.encoder_attention_mask

        La, Lb = sa.shape[1], sb.shape[1]
        if La < Lb:
            sa = torch.nn.functional.pad(sa, (0, 0, 0, Lb - La))
            ma = torch.nn.functional.pad(ma, (0, Lb - La), value=0)
        elif Lb < La:
            sb = torch.nn.functional.pad(sb, (0, 0, 0, La - Lb))
            mb = torch.nn.functional.pad(mb, (0, La - Lb), value=0)

        blended = INTERPOLATIONS[method](sa, sb, alpha)
        mask = torch.max(ma, mb)

        return {
            "conditioning": Conditioning(
                encoder_hidden_states=blended,
                encoder_attention_mask=mask,
            )
        }


@NodeRegistry.register
class ConditioningCombine(BaseNode):
    """Combine two conditionings into a multi-condition set.

    Unlike ConditioningAverage (which fuses into one), this preserves
    both conditions as separate entries with optional compositing
    metadata. The Generate node will run separate decoder calls
    and blend velocities per-frame.

    Node parameters:
        step_range_start_b: Diffusion step fraction where B activates.
        step_range_end_b: Diffusion step fraction where B deactivates.
    """

    node_type_id: ClassVar[str] = "acestep.ConditioningCombine"

    @classmethod
    def get_definition(cls) -> NodeDefinition:
        return NodeDefinition(
            node_type_id=cls.node_type_id,
            display_name="Conditioning Combine",
            category="conditioning",
            description="Combine two conditionings for multi-condition generation.",
            inputs=(
                NodePort(name="conditioning_a", type="CONDITIONING"),
                NodePort(name="conditioning_b", type="CONDITIONING"),
                NodePort(
                    name="temporal_weight_b",
                    type="MASK",
                    required=False,
                    description="Per-frame blend weight for condition B.",
                ),
            ),
            outputs=(
                NodePort(name="conditioning", type="CONDITIONING"),
            ),
            params=(
                NodeParam(
                    name="step_range_start_b", type="number", default=0.0,
                    description="Diffusion step fraction where B activates (0-1)",
                    min=0.0, max=1.0, step=0.01,
                ),
                NodeParam(
                    name="step_range_end_b", type="number", default=1.0,
                    description="Diffusion step fraction where B deactivates (0-1)",
                    min=0.0, max=1.0, step=0.01,
                ),
            ),
        )

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        cond_a: Conditioning = kwargs["conditioning_a"]
        cond_b: Conditioning = kwargs["conditioning_b"]
        temporal_mask: Optional[Mask] = kwargs.get("temporal_weight_b")

        step_start = kwargs.get("step_range_start_b")
        step_end = kwargs.get("step_range_end_b")

        entries_a = cond_a.to_entries()
        entries_b = cond_b.to_entries()

        # B entries get compositing metadata
        step_range = None
        if step_start is not None and step_end is not None:
            step_range = (float(step_start), float(step_end))

        temporal_weight_b = None
        temporal_weight_a = None
        if temporal_mask is not None:
            ref = entries_b[0].encoder_hidden_states if entries_b else (
                entries_a[0].encoder_hidden_states if entries_a else None
            )
            if ref is not None:
                temporal_weight_b = temporal_mask.tensor.to(device=ref.device, dtype=ref.dtype)
            else:
                temporal_weight_b = temporal_mask.tensor
            temporal_weight_a = 1.0 - temporal_weight_b

        combined = []
        for entry in entries_a:
            combined.append(
                ConditioningEntry(
                    encoder_hidden_states=entry.encoder_hidden_states,
                    encoder_attention_mask=entry.encoder_attention_mask,
                    temporal_weight=temporal_weight_a,
                    step_range=entry.step_range,
                )
            )

        for entry in entries_b:
            combined.append(
                ConditioningEntry(
                    encoder_hidden_states=entry.encoder_hidden_states,
                    encoder_attention_mask=entry.encoder_attention_mask,
                    temporal_weight=temporal_weight_b,
                    step_range=step_range,
                )
            )

        return {
            "conditioning": Conditioning(entries=combined)
        }
