"""Conditioning encode + timbre-blend helpers.

Pure, transport-agnostic helpers for building the (silence, full) cond
pair against a given timbre reference and lerping between them on the
live timbre-strength slider.
"""

from acestep.constants import TASK_INSTRUCTIONS
from acestep.engine.session import Session


def encode_cond_pair(
    session: Session,
    tags,
    refer_latent,
    bpm,
    duration,
    key,
    time_signature,
):
    # WYSIWYG: the encoder sees exactly the text the UI sent. LoRA
    # trigger words land in `tags` via the client's visible-prepend
    # logic (see web/store/useLoraStore + the auto_prepend_lora_triggers
    # config flag) so the user can edit or remove them like any other
    # prompt token. The server intentionally does NOT inject anything
    # behind the user's back.
    cs = session.encode_text(
        tags=tags,
        lyrics="[Instrumental]",
        instruction=TASK_INSTRUCTIONS["cover"],
        refer_latent=None,
        bpm=bpm, duration=duration, key=key,
        time_signature=time_signature,
    )
    cf = session.encode_text(
        tags=tags,
        lyrics="[Instrumental]",
        instruction=TASK_INSTRUCTIONS["cover"],
        refer_latent=refer_latent,
        bpm=bpm, duration=duration, key=key,
        time_signature=time_signature,
    )
    return cs, cf


def blend_for_strength(cs, cf, strength, method="slerp"):
    """Lerp between two conditionings by ``strength`` ∈ [0, 1].

    ``method="slerp"`` (default) walks the per-token geodesic instead of a
    straight average, which holds conditioning norm constant across the
    sweep. This matters most for the prompt A↔B crossfade (two unrelated
    prompts, where a linear midpoint collapses in norm and sounds washed
    out). For near-parallel endpoints (e.g. the timbre silence↔full blend,
    which shares a prompt) slerp degrades gracefully to linear per token,
    so the default is safe everywhere.
    """
    from acestep.nodes.cond_nodes import ConditioningBlend
    if strength >= 0.999:
        return cf
    if strength <= 0.001:
        return cs
    return ConditioningBlend().execute(
        conditioning_a=cs,
        conditioning_b=cf,
        alpha=float(strength),
        method=method,
    )["conditioning"]
