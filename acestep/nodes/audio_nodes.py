"""Audio analysis nodes."""

from __future__ import annotations

import torch
import torchaudio
from typing import Any, ClassVar

from ..audio.key_detection import detect_key
from .base import BaseNode, NodeDefinition, NodePort, NodeRegistry
from .types import Audio


def _detect_key(waveform: torch.Tensor, sr: int) -> str:
    """Detect musical key. Thin tensor->numpy bridge over the CNN module."""
    if waveform.dim() == 2:
        mono = waveform.mean(dim=0)
    else:
        mono = waveform
    return detect_key(mono.detach().cpu().float().numpy(), sr)


def _detect_bpm(waveform: torch.Tensor, sr: int) -> int:
    """Detect BPM using onset envelope autocorrelation."""
    # Mono, resample to 22050 for speed
    if waveform.dim() == 2:
        mono = waveform.mean(dim=0)
    else:
        mono = waveform
    mono = mono.float()

    if sr != 22050:
        mono = torchaudio.functional.resample(mono, sr, 22050)
        sr = 22050

    # Compute onset envelope via spectral flux
    n_fft = 2048
    hop = 512
    spec = torch.stft(
        mono, n_fft=n_fft, hop_length=hop, return_complex=True,
        window=torch.hann_window(n_fft, device=mono.device),
    )
    mag = spec.abs()
    # Half-wave rectified spectral flux
    flux = torch.clamp(mag[:, 1:] - mag[:, :-1], min=0).sum(dim=0)

    # Autocorrelation
    # BPM range: 60-200 -> lag range in onset frames
    min_bpm, max_bpm = 60, 200
    fps = sr / hop
    min_lag = int(fps * 60.0 / max_bpm)
    max_lag = int(fps * 60.0 / min_bpm)
    max_lag = min(max_lag, len(flux) // 2)

    if max_lag <= min_lag:
        return 120

    flux = flux - flux.mean()
    autocorr = torch.zeros(max_lag - min_lag)
    for i, lag in enumerate(range(min_lag, max_lag)):
        autocorr[i] = torch.dot(flux[:len(flux) - lag], flux[lag:]).item()

    if autocorr.max() <= 0:
        return 120

    best_lag = autocorr.argmax().item() + min_lag
    bpm = 60.0 * fps / best_lag
    return int(round(bpm))


@NodeRegistry.register
class AudioInfo(BaseNode):
    """Detect BPM, key, and duration from audio.

    Uses onset autocorrelation for BPM and a small CNN
    (acestep.audio.key_detection) for key.  Duration is computed from
    sample count and rate.
    """

    node_type_id: ClassVar[str] = "acestep.AudioInfo"

    @classmethod
    def get_definition(cls) -> NodeDefinition:
        return NodeDefinition(
            node_type_id=cls.node_type_id,
            display_name="Audio Info",
            category="audio",
            description="Detect BPM, key, and duration from audio.",
            inputs=(
                NodePort(name="audio", type="AUDIO"),
            ),
            outputs=(),
        )

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        audio: Audio = kwargs["audio"]
        waveform = audio.waveform
        sr = audio.sample_rate

        # Handle [B, C, samples] or [C, samples]
        if waveform.dim() == 3:
            waveform = waveform[0]

        samples = waveform.shape[-1]
        duration = samples / sr

        bpm = _detect_bpm(waveform, sr)
        key = _detect_key(waveform, sr)

        return {
            "bpm": bpm,
            "key": key,
            "duration": round(duration, 2),
        }
