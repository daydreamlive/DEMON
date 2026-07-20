"""Backend-neutral live audio-edit state and mask/composite helpers.

Edits are controls on the ordinary streaming pipeline: each generation
request snapshots one immutable :class:`LiveAudioEdit`, carries it through the
ring buffer, and uses that same snapshot when its latent is window-decoded.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np


SOURCE_MODES = ("waveform", "structure")
EDIT_CROSSFADE_S = 0.025
_EPS = 1e-6


class AudioEditError(ValueError):
    """The requested live edit cannot be represented exactly."""


@dataclass(frozen=True)
class EditRegion:
    start_s: float
    end_s: float

    def to_wire(self) -> dict:
        return {"start_s": self.start_s, "end_s": self.end_s}


@dataclass(frozen=True)
class LiveAudioEdit:
    enabled: bool = False
    regions: tuple[EditRegion, ...] = ()
    source_mode: str = "waveform"
    strength: float = 1.0

    def to_wire(self) -> dict:
        return {
            "enabled": self.enabled,
            "regions": [r.to_wire() for r in self.regions],
            "source_mode": self.source_mode,
            "strength": self.strength,
        }


DISABLED_AUDIO_EDIT = LiveAudioEdit()


def constrain_audio_edit(
    current: LiveAudioEdit | None,
    emerged: LiveAudioEdit | None,
) -> LiveAudioEdit:
    """Limit one decoded request to the currently armed waveform mask.

    A request can finish after the user has changed or cleared the selected
    regions.  Only spans present in both snapshots are allowed to reach the
    playback buffer.  In particular, an enabled current edit with no regions
    is an all-preserve mask even when the emerged request predates Edit mode.
    """
    if current is None or not current.enabled or current.source_mode != "waveform":
        return emerged or DISABLED_AUDIO_EDIT
    if emerged is None or not emerged.enabled or emerged.source_mode != "waveform":
        return LiveAudioEdit(True, (), "waveform", current.strength)

    intersections: list[EditRegion] = []
    i = j = 0
    current_regions = current.regions
    emerged_regions = emerged.regions
    while i < len(current_regions) and j < len(emerged_regions):
        left = current_regions[i]
        right = emerged_regions[j]
        start = max(left.start_s, right.start_s)
        end = min(left.end_s, right.end_s)
        if end > start + _EPS:
            intersections.append(EditRegion(start, end))
        if left.end_s < right.end_s:
            i += 1
        else:
            j += 1
    return LiveAudioEdit(
        True,
        tuple(intersections),
        "waveform",
        min(current.strength, emerged.strength),
    )


def parse_live_audio_edit(
    regions: Iterable[dict] | None,
    *,
    enabled: bool,
    source_mode: str,
    strength: float,
    canvas_duration_s: float,
    source_duration_s: float,
    left_extension_s: float = 0.0,
    right_extension_s: float = 0.0,
) -> LiveAudioEdit:
    """Validate the wire vocabulary and return an immutable tick snapshot."""
    if not enabled:
        return DISABLED_AUDIO_EDIT
    if source_mode not in SOURCE_MODES:
        raise AudioEditError(
            f"source_mode must be one of {SOURCE_MODES}, got {source_mode!r}"
        )
    if not math.isfinite(strength) or not 0.0 <= strength <= 1.0:
        raise AudioEditError(f"strength must be finite and in [0, 1], got {strength!r}")
    if not math.isfinite(canvas_duration_s) or canvas_duration_s <= 0:
        raise AudioEditError("the session has no finite editable canvas")
    if (
        not math.isfinite(left_extension_s)
        or not math.isfinite(right_extension_s)
        or left_extension_s < 0.0
        or right_extension_s < 0.0
        or left_extension_s + right_extension_s > canvas_duration_s + _EPS
    ):
        raise AudioEditError("explicit extension spans are invalid for this canvas")

    parsed: list[EditRegion] = []
    for i, raw in enumerate(regions or ()):
        if not isinstance(raw, dict):
            raise AudioEditError(f"regions[{i}] must be an object")
        try:
            start = float(raw["start_s"])
            end = float(raw["end_s"])
        except (KeyError, TypeError, ValueError) as exc:
            raise AudioEditError(
                f"regions[{i}] requires numeric start_s and end_s"
            ) from exc
        if not math.isfinite(start) or not math.isfinite(end):
            raise AudioEditError(f"regions[{i}] has non-finite bounds")
        if start < -_EPS or end <= start + _EPS:
            raise AudioEditError(f"regions[{i}] is negative, empty, or inverted")
        if end > canvas_duration_s + _EPS:
            raise AudioEditError(
                f"regions[{i}].end_s={end} exceeds canvas {canvas_duration_s}s"
            )
        if parsed and start < parsed[-1].end_s - _EPS:
            raise AudioEditError("regions must be ordered and non-overlapping")
        parsed.append(EditRegion(max(0.0, start), min(canvas_duration_s, end)))

    if source_mode == "structure":
        if len(parsed) != 1 or parsed[0].start_s > _EPS or parsed[0].end_s < canvas_duration_s - _EPS:
            raise AudioEditError("structure mode requires one region covering the full canvas")
    else:
        # The model canvas is routinely a little (and for SA3 sometimes much)
        # longer than the uploaded content because of latent/profile padding.
        # That is preserved silence, not a user-requested extension.  Require
        # coverage only for explicit extension spans supplied by the serving
        # layer; ordinary canvas/source geometry differences remain editable
        # with any valid interior region.
        def _covered(required_start: float, required_end: float) -> bool:
            if required_end <= required_start + _EPS:
                return True
            covered_until = required_start
            for region in parsed:
                if region.end_s <= covered_until + _EPS:
                    continue
                if region.start_s > covered_until + _EPS:
                    return False
                covered_until = max(covered_until, region.end_s)
                if covered_until >= required_end - _EPS:
                    return True
            return False

        # Anchor the right span to the end of uploaded content, not the
        # backend canvas end: latent/model padding may continue beyond the
        # exact requested duration and is not part of the extension.
        right_extension_end_s = min(
            canvas_duration_s, source_duration_s + right_extension_s,
        )
        if not _covered(0.0, left_extension_s) or not _covered(
            source_duration_s, right_extension_end_s,
        ):
            raise AudioEditError(
                "regenerate regions must cover every explicit waveform extension tail"
            )
    return LiveAudioEdit(True, tuple(parsed), source_mode, float(strength))


def regenerate_mask(
    edit: LiveAudioEdit,
    *,
    total_frames: int,
    rate_hz: float,
    offset_s: float = 0.0,
    device=None,
    dtype=None,
):
    """ACE-polarity mask: 1 regenerates, 0 preserves."""
    import torch

    mask = torch.zeros((1, total_frames, 1), device=device, dtype=dtype)
    if not edit.enabled or edit.source_mode != "waveform":
        return mask
    for region in edit.regions:
        start = max(0, int(math.floor((region.start_s - offset_s) * rate_hz + _EPS)))
        end = min(total_frames, int(math.ceil((region.end_s - offset_s) * rate_hz - _EPS)))
        if end > start:
            mask[:, start:end, :] = edit.strength
    return mask


def sa3_inpaint_bundle(base: dict, source_btc, edit: LiveAudioEdit) -> dict:
    """Return an SA3 conditioning bundle with its live binary inpaint input."""
    import torch

    if not edit.enabled or edit.source_mode != "waveform":
        return base
    frames = source_btc.shape[1]
    regenerate = regenerate_mask(
        edit,
        total_frames=frames,
        rate_hz=44100.0 / 4096.0,
        device=source_btc.device,
        dtype=source_btc.dtype,
    ).movedim(1, 2)
    preserve = 1.0 - regenerate.clamp(0, 1)
    local_add = torch.cat(
        [preserve, source_btc.movedim(1, 2) * preserve], dim=1,
    )
    old = base.get("local_add_cond")
    if old is not None:
        local_add = local_add.to(device=old.device, dtype=old.dtype)
    out = dict(base)
    out["local_add_cond"] = local_add
    return out


def composite_window(
    generated: np.ndarray,
    *,
    start_sample: int,
    source,
    edit: LiveAudioEdit | None,
    sample_rate: int = 48000,
    crossfade_s: float = EDIT_CROSSFADE_S,
) -> np.ndarray:
    """Restore preserved source samples in one absolute decoded window."""
    if edit is None or not edit.enabled or edit.source_mode != "waveform":
        return generated
    pcm = np.asarray(generated, dtype=np.float32)
    if pcm.ndim != 2 or pcm.shape[0] == 0:
        return pcm
    if hasattr(source, "detach"):
        src = source.detach().cpu().float().numpy()
    else:
        src = np.asarray(source, dtype=np.float32)
    if src.ndim != 2:
        return pcm
    # Source providers use [C,N]; playback windows use [N,C].
    if src.shape[0] <= 8 and src.shape[1] > src.shape[0]:
        src = src.T
    if src.shape[1] == 1 and pcm.shape[1] > 1:
        src = np.repeat(src, pcm.shape[1], axis=1)
    elif src.shape[1] != pcm.shape[1]:
        src = src[:, :pcm.shape[1]]

    absolute = start_sample + np.arange(pcm.shape[0])
    generated_weight = np.zeros(pcm.shape[0], dtype=np.float32)
    fade = max(0, int(round(crossfade_s * sample_rate)))
    for region in edit.regions:
        r0 = int(math.floor(region.start_s * sample_rate + 0.5))
        r1 = int(math.floor(region.end_s * sample_rate + 0.5))
        inside = (absolute >= r0) & (absolute < r1)
        generated_weight[inside] = 1.0
        width = min(fade, max((r1 - r0) // 2, 0))
        if width:
            left = (absolute >= r0) & (absolute < r0 + width)
            generated_weight[left] = np.minimum(
                generated_weight[left], (absolute[left] - r0) / max(1, width - 1),
            )
            right = (absolute >= r1 - width) & (absolute < r1)
            generated_weight[right] = np.minimum(
                generated_weight[right], (r1 - 1 - absolute[right]) / max(1, width - 1),
            )

    valid = (absolute >= 0) & (absolute < src.shape[0])
    out = pcm.copy()
    if np.any(valid):
        w = generated_weight[valid, None]
        out[valid] = pcm[valid] * w + src[absolute[valid]] * (1.0 - w)
    return out
