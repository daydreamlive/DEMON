"""Musical key detection via a CNN ported from madmom.

Architecture and pretrained weights are the 2018 model from madmom's
``key_cnn.pkl`` (Korzeniowski & Widmer, "End-to-End Musical Key
Estimation Using a Convolutional Neural Network", ICASSP 2018). madmom
itself doesn't install on numpy>=2 / py>=3.11, so the conv layers were
exported, the BatchNorm layers were folded into the preceding conv,
and everything was re-saved as a torch state dict in
``assets/key_cnn.pt``. See ``scripts/convert_madmom_key_cnn.py``.

License of the original model and architecture: BSD 3-Clause (madmom).

Public API:
    detect_key(mono_np, sr) -> str   e.g. "F# minor"
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


_WEIGHTS_PATH = Path(__file__).parent / "assets" / "key_cnn.pt"


class _KeyCNN(torch.nn.Module):
    """Fully-convolutional key recognition network (24-class output).

    Layout: nine conv layers in five blocks, ELU after each, max-pool
    after blocks 1/2/3. Block 5 is a 1x1 conv classifier head followed
    by global average pooling. The original network has BatchNorm
    after every conv with an ELU on the BN; we fold BN into the conv
    weights at export time so the runtime is conv->ELU->maybe-pool.

    Input:  (B, 1, T, F=105) log-magnitude filterbank spectrogram
    Output: (B, 24) class scores
    """

    def __init__(self, blocks: list[tuple[int, int]]):
        super().__init__()
        in_ch = 1
        out_chs = (24, 24, 48, 48, 96, 96, 192, 192, 24)
        self.convs = torch.nn.ModuleList()
        for (k, p), out_ch in zip(blocks, out_chs):
            self.convs.append(
                torch.nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p)
            )
            in_ch = out_ch
        # Indices after which to apply 2x2 max-pool (after blocks 1/3/5).
        self._pool_after = {1, 3, 5}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = F.elu(conv(x))
            if i in self._pool_after:
                x = F.max_pool2d(x, kernel_size=2, stride=2)
        # Global average pool over (T, F) -> (B, 24).
        return x.mean(dim=(2, 3))


_CACHE: dict[str, Any] = {}


def _load() -> dict[str, Any]:
    """Lazily load the model + filterbank onto the appropriate device."""
    if _CACHE:
        return _CACHE
    if not _WEIGHTS_PATH.is_file():
        raise FileNotFoundError(
            f"key_cnn.pt missing at {_WEIGHTS_PATH}. "
            f"Run scripts/convert_madmom_key_cnn.py."
        )
    blob = torch.load(_WEIGHTS_PATH, map_location="cpu", weights_only=False)
    state = blob["state_dict"]
    filterbank = state.pop("filterbank")
    model = _KeyCNN(blob["blocks"])
    # Re-key to nn.ModuleList naming.
    remapped = {}
    for k, v in state.items():
        idx = int(k.split(".")[0].removeprefix("conv"))
        suffix = k.split(".", 1)[1]
        remapped[f"convs.{idx}.{suffix}"] = v
    model.load_state_dict(remapped, strict=True)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    filterbank = filterbank.to(device)
    window = torch.hann_window(blob["n_fft"], periodic=False, device=device)

    _CACHE.update(
        model=model,
        filterbank=filterbank,
        window=window,
        labels=blob["key_labels"],
        sr=int(blob["sample_rate"]),
        n_fft=int(blob["n_fft"]),
        hop=int(blob["hop"]),
        device=device,
    )
    return _CACHE


def _resample_to_44100(mono: np.ndarray, sr: int) -> np.ndarray:
    if sr == 44100:
        return mono.astype(np.float32, copy=False)
    # Use soxr (already a project dep) for high-quality resampling.
    import soxr
    return soxr.resample(mono.astype(np.float32, copy=False), sr, 44100, "HQ")


@torch.inference_mode()
def detect_key(mono_np: np.ndarray, sr: int) -> str:
    """Return the predicted key as a string like ``"F# minor"``.

    Pipeline: resample to 44100 mono -> centered STFT (n_fft=8192,
    hop=8820 = 5 fps, Hann) -> magnitude -> log-spaced triangular
    filterbank (105 bands, 65-2100 Hz) -> log10(x + 1) -> CNN ->
    argmax of 24 class scores.
    """
    if mono_np.size == 0:
        return "C major"

    bundle = _load()
    sr_target = bundle["sr"]
    n_fft = bundle["n_fft"]
    hop = bundle["hop"]
    device = bundle["device"]

    audio = _resample_to_44100(np.ascontiguousarray(mono_np), sr)
    if audio.size < n_fft:
        # Need at least one frame's worth of samples.
        pad = n_fft - audio.size
        audio = np.pad(audio, (0, pad))

    audio_t = torch.from_numpy(audio).to(device)
    spec = torch.stft(
        audio_t,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=bundle["window"],
        center=True,
        pad_mode="reflect",
        return_complex=True,
    )
    mag = spec.abs().T  # (T, F_fft)
    filt = mag @ bundle["filterbank"]  # (T, 105)
    log_spec = torch.log10(filt + 1.0)
    x = log_spec.unsqueeze(0).unsqueeze(0)  # (1, 1, T, 105)

    logits = bundle["model"](x)
    idx = int(logits.argmax(dim=1).item())
    return bundle["labels"][idx]
