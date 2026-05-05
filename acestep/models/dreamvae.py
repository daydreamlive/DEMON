"""FastOobleckDecoder — the model backing DreamVAE.

DreamVAE is a distilled student of the ACE-Step 1.5 Oobleck VAE
decoder, released at ``daydreamlive/DreamVAE``. Same I/O contract as
the teacher: ``[B, 64, T]`` latents at 25 fps -> ``[B, 2, 1920*T]``
audio at 48 kHz, so the resulting weights / ONNX / TRT engine are a
drop-in replacement wherever the teacher decoder runs.

This file is the canonical source-of-truth for the architecture inside
this repo. The HF release ships an identical ``modeling.py`` so users
loading DreamVAE via ``transformers``' ``trust_remote_code`` get the
same class definition.

Default config matches the released checkpoint:
    channels=128, input_channels=64, audio_channels=2,
    upsampling_ratios=[10, 6, 4, 4, 2],
    channel_multiples=[1, 2, 4, 8, 8].
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Snake1d(nn.Module):
    """Snake activation (DAC, NeurIPS 2023)."""

    def __init__(self, hidden_dim: int, logscale: bool = True) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1, hidden_dim, 1))
        self.beta = nn.Parameter(torch.zeros(1, hidden_dim, 1))
        self.logscale = logscale

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shape = hidden_states.shape
        alpha = torch.exp(self.alpha) if self.logscale else self.alpha
        beta = torch.exp(self.beta) if self.logscale else self.beta
        hidden_states = hidden_states.reshape(shape[0], shape[1], -1)
        hidden_states = hidden_states + (beta + 1e-9).reciprocal() * torch.sin(alpha * hidden_states).pow(2)
        return hidden_states.reshape(shape)


class FastResidualUnit(nn.Module):
    def __init__(self, dim: int, dilation: int = 1) -> None:
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.snake1 = Snake1d(dim)
        self.conv1 = weight_norm(nn.Conv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad))
        self.snake2 = Snake1d(dim)
        self.conv2 = weight_norm(nn.Conv1d(dim, dim, kernel_size=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.snake1(x))
        h = self.conv2(self.snake2(h))
        pad = (x.shape[-1] - h.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + h


class FastDecoderBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, stride: int = 1) -> None:
        super().__init__()
        self.snake1 = Snake1d(in_dim)
        self.conv_t = weight_norm(
            nn.ConvTranspose1d(
                in_dim,
                out_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            )
        )
        self.res1 = FastResidualUnit(out_dim, dilation=1)
        self.res2 = FastResidualUnit(out_dim, dilation=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.snake1(x)
        x = self.conv_t(x)
        x = self.res1(x)
        x = self.res2(x)
        return x


class FastOobleckDecoder(nn.Module):
    """Distilled ACE-Step 1.5 VAE decoder (51.7M params).

    Drop-in replacement for diffusers' OobleckDecoder with identical
    input/output shapes. Accepts ``[B, 64, T]`` latents at 25 fps and
    produces ``[B, 2, 1920*T]`` audio at 48 kHz.
    """

    def __init__(
        self,
        channels: int = 128,
        input_channels: int = 64,
        audio_channels: int = 2,
        upsampling_ratios: Optional[List[int]] = None,
        channel_multiples: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        if upsampling_ratios is None:
            upsampling_ratios = [10, 6, 4, 4, 2]
        if channel_multiples is None:
            channel_multiples = [1, 2, 4, 8, 8]

        strides = upsampling_ratios
        cm = [1] + channel_multiples

        self.conv1 = weight_norm(
            nn.Conv1d(input_channels, channels * cm[-1], kernel_size=7, padding=3)
        )

        blocks = []
        for i, stride in enumerate(strides):
            in_dim = channels * cm[len(strides) - i]
            out_dim = channels * cm[len(strides) - i - 1]
            blocks.append(FastDecoderBlock(in_dim, out_dim, stride=stride))
        self.blocks = nn.ModuleList(blocks)

        self.final_snake = Snake1d(channels)
        self.conv2 = weight_norm(
            nn.Conv1d(channels, audio_channels, kernel_size=7, padding=3, bias=False)
        )

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        x = self.conv1(latents)
        for block in self.blocks:
            x = block(x)
        x = self.final_snake(x)
        x = self.conv2(x)
        return x


# ------------------------------------------------------------------
# Loaders
# ------------------------------------------------------------------

def load_dreamvae_from_hf(
    repo_id: str = "daydreamlive/DreamVAE",
    *,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float32,
) -> FastOobleckDecoder:
    """Load a FastOobleckDecoder from the public DreamVAE HF release.

    Pulls ``config.json`` + ``model.safetensors`` from ``repo_id``,
    instantiates the model with the exact released config, and loads
    weights. Returns the model in ``eval()`` mode on ``device``.
    """
    import json
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    cfg_path = hf_hub_download(repo_id, "config.json")
    weights_path = hf_hub_download(repo_id, "model.safetensors")
    with open(cfg_path) as f:
        cfg = json.load(f)
    model = FastOobleckDecoder(
        channels=cfg["channels"],
        input_channels=cfg["input_channels"],
        audio_channels=cfg["audio_channels"],
        upsampling_ratios=cfg["upsampling_ratios"],
        channel_multiples=cfg["channel_multiples"],
    )
    state = load_file(weights_path)
    model.load_state_dict(state)
    return model.to(device=device, dtype=dtype).eval()
