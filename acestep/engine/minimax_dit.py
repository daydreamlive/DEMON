"""Dependency-free PyTorch renderer stack for MiniMax-Music3.

Three inference-time modules, reimplemented straight against the
diffusers-layout safetensors: :class:`MiniMaxDiT` (the 2.43B
flow-matching renderer), :class:`MiniMaxDAV` (the DAC-style waveform
decoder shipped under ``vocoder/``) and :class:`MiniMaxConditionEncoder`
(the AR-frame-to-latent-frame projector under ``condition_encoder/``).

Why reimplement rather than import: upstream's reference lives in
``diffusers >= 0.40``. DEMON pins ``diffusers==0.37.1`` because ACE-Step
needs it, and bumping the pin repo-wide is not on the table. The second
reason is TensorRT — these modules are written to be ONNX-exportable by
construction: no Python branching on tensor shapes, no ``.item()``, no
data-dependent control flow, positional tensor args only, a bare tensor
back. RoPE tables are built from the running sequence length rather than
baked at a fixed one.

The parity gate against the reference implementation is
``scripts/minimax/minimax_dit_parity.py``.

Supported dtypes are fp32 and bf16. The acoustic stage does not survive
fp16 — there is deliberately no fp16 path here.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

# Latent frames per second: 44100 / 512, the DAV decoder's total upsample.
MINIMAX_LATENT_RATE = 44100 / 512


def _load_config(model_dir: Path, subdir: str, keys: Tuple[str, ...]) -> Dict:
    """Read ``config.json`` from either the repo root's ``subdir`` or from
    ``model_dir`` itself, keeping only the constructor arguments."""
    directory = Path(model_dir)
    if (directory / subdir / "config.json").is_file():
        directory = directory / subdir
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    return {key: config[key] for key in keys if key in config}


def _load_state_dict(model_dir: Path, subdir: str) -> Dict[str, torch.Tensor]:
    """Load a (possibly sharded) diffusers ``diffusion_pytorch_model`` on CPU."""
    directory = Path(model_dir)
    if (directory / subdir / "config.json").is_file():
        directory = directory / subdir
    index = directory / "diffusion_pytorch_model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        shards = sorted(set(weight_map.values()))
    else:
        shards = ["diffusion_pytorch_model.safetensors"]
    state: Dict[str, torch.Tensor] = {}
    for shard in shards:
        state.update(load_file(str(directory / shard)))
    return state


def _build(cls, config: Dict, state: Dict[str, torch.Tensor], dtype, device):
    """Meta-init, assign the checkpoint tensors, then move/cast once. Avoids
    materialising a second full-size copy of a 2.43B parameter model."""
    with torch.device("meta"):
        model = cls(**config)
    model.load_state_dict(state, strict=True, assign=True)
    model.to(device=device, dtype=dtype)
    model.eval()
    model.requires_grad_(False)
    return model


def _remap(state: Dict[str, torch.Tensor], key_map: Dict[str, str]) -> Dict[str, torch.Tensor]:
    """Apply a checkpoint-key -> parameter-name map. Keys absent from the map
    are used as-is."""
    return {key_map.get(key, key): value for key, value in state.items()}


# ---------------------------------------------------------------------------
# DiT
# ---------------------------------------------------------------------------


class MiniMaxFourierEmbedding(nn.Module):
    """Random Fourier features over the flow-matching time. The projection is a
    trained checkpoint weight, not a fixed frequency ladder."""

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(embedding_dim // 2, 1))

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        # Scale first, then project: upstream's precedence is (2*pi*t) @ W.T.
        angles = (2.0 * math.pi * timestep.unsqueeze(-1)) @ self.weight.transpose(0, 1)
        return torch.cat((angles.cos(), angles.sin()), dim=-1)


class MiniMaxTimestepEmbedding(nn.Module):
    """The two-layer SiLU MLP diffusers calls ``TimestepEmbedding``."""

    def __init__(self, in_channels: int, time_embed_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        return self.linear_2(F.silu(self.linear_1(sample)))


def _rope_tables(
    seq_len: int, rotary_dim: int, theta: float, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Partial-RoPE cos/sin tables of shape ``(seq_len, rotary_dim)``. Built per
    call from the running length so export never bakes a constant."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, device=device).float() / rotary_dim))
    steps = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(steps, inv_freq)
    freqs = torch.cat((freqs, freqs), dim=-1)
    return freqs.cos(), freqs.sin()


def _apply_partial_rope(
    hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rotary_dim: int
) -> torch.Tensor:
    """``hidden_states`` is ``(B, S, H, D)``; only the leading ``rotary_dim`` of
    each head's ``D`` rotates, the tail passes through untouched."""
    cos = cos[:, None, :].to(hidden_states.dtype)
    sin = sin[:, None, :].to(hidden_states.dtype)
    rotated = hidden_states[..., :rotary_dim]
    half_first, half_second = rotated.chunk(2, dim=-1)
    rotate_half = torch.cat((-half_second, half_first), dim=-1)
    rotated = rotated * cos + rotate_half * sin
    return torch.cat((rotated, hidden_states[..., rotary_dim:]), dim=-1)


class MiniMaxAttention(nn.Module):
    """Self-attention only — no cross-attention, no KV cache, no q/k norm, no
    biases anywhere, and no attention mask."""

    def __init__(self, dim: int, heads: int, head_dim: int, rotary_dim: int):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        inner_dim = heads * head_dim
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        query = self.to_q(hidden_states).view(batch_size, seq_len, self.heads, self.head_dim)
        key = self.to_k(hidden_states).view(batch_size, seq_len, self.heads, self.head_dim)
        value = self.to_v(hidden_states).view(batch_size, seq_len, self.heads, self.head_dim)

        query = _apply_partial_rope(query, cos, sin, self.rotary_dim)
        key = _apply_partial_rope(key, cos, sin, self.rotary_dim)

        # SDPA wants (B, H, S, D); upstream's dispatcher does the same permute.
        out = F.scaled_dot_product_attention(
            query.permute(0, 2, 1, 3),
            key.permute(0, 2, 1, 3),
            value.permute(0, 2, 1, 3),
        )
        out = out.permute(0, 2, 1, 3).flatten(2, 3).to(query.dtype)
        return self.to_out(out)


class MiniMaxTransformerBlock(nn.Module):
    """Pre-LayerNorm block with a SwiGLU feed-forward. Note the gating order:
    ``ff_in`` splits into ``(gate_states, gate)`` and the SECOND half is the one
    that goes through SiLU."""

    def __init__(self, dim: int, heads: int, head_dim: int, ff_inner_dim: int, rotary_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MiniMaxAttention(dim, heads, head_dim, rotary_dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff_in = nn.Linear(dim, ff_inner_dim * 2)
        self.ff_out = nn.Linear(ff_inner_dim, dim)

    def forward(self, hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), cos, sin)
        gate_states, gate = self.ff_in(self.norm2(hidden_states)).chunk(2, dim=-1)
        return hidden_states + self.ff_out(gate_states * F.silu(gate))


class MiniMaxDiT(nn.Module):
    """The 2.43B flow-matching renderer.

    Denoises 128-channel Flow-VAE latents ``(B, in_channels, L)`` against
    frame-aligned conditioning ``(B, L, condition_dim)``. ``timestep`` is
    ``(B,)`` — one flow-matching time per batch row, and rows may carry
    different times. Returns the predicted velocity, same shape as the input
    latent.

    Time runs 0 (noise) to 1 (data), i.e. ``x_t = (1 - t) * noise + t * data``.
    That is the opposite of DEMON's rectified-flow convention; the conversion
    lives in the adapter, not here.
    """

    _SUBDIR = "transformer"
    _CONFIG_KEYS = (
        "in_channels",
        "condition_dim",
        "num_layers",
        "num_attention_heads",
        "attention_head_dim",
        "ff_inner_dim",
        "rotary_dim",
        "fourier_embedding_dim",
    )

    def __init__(
        self,
        in_channels: int = 128,
        condition_dim: int = 2048,
        num_layers: int = 36,
        num_attention_heads: int = 32,
        attention_head_dim: int = 64,
        ff_inner_dim: int = 8192,
        rotary_dim: int = 32,
        fourier_embedding_dim: int = 256,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.condition_dim = condition_dim
        self.rotary_dim = rotary_dim
        self.rope_theta = rope_theta
        inner_dim = num_attention_heads * attention_head_dim
        # The input is [latent, zeros(in_channels), condition] along channels:
        # the middle block is a hard-zeroed spare slot, not a second latent.
        concat_channels = 2 * in_channels + condition_dim

        self.time_proj = MiniMaxFourierEmbedding(fourier_embedding_dim)
        self.time_embed = MiniMaxTimestepEmbedding(fourier_embedding_dim, inner_dim)

        self.preprocess_conv = nn.Conv1d(concat_channels, concat_channels, 1, bias=False)
        self.proj_in = nn.Linear(concat_channels, inner_dim, bias=False)
        self.transformer_blocks = nn.ModuleList(
            [
                MiniMaxTransformerBlock(inner_dim, num_attention_heads, attention_head_dim, ff_inner_dim, rotary_dim)
                for _ in range(num_layers)
            ]
        )
        self.proj_out = nn.Linear(inner_dim, in_channels, bias=False)
        self.postprocess_conv = nn.Conv1d(in_channels, in_channels, 1, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        zeros = torch.zeros_like(hidden_states)
        hidden_states = torch.cat((hidden_states, zeros, encoder_hidden_states.transpose(1, 2)), dim=1)
        # Additive residual around the 1x1 mixer, not a plain projection.
        hidden_states = self.preprocess_conv(hidden_states) + hidden_states
        hidden_states = hidden_states.transpose(1, 2)

        temb = self.time_embed(self.time_proj(timestep))

        hidden_states = self.proj_in(hidden_states)
        # Time enters as one prepended token, stripped again after the blocks.
        hidden_states = torch.cat((temb.unsqueeze(1), hidden_states), dim=1)

        cos, sin = _rope_tables(hidden_states.shape[1], self.rotary_dim, self.rope_theta, hidden_states.device)
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, cos, sin)

        hidden_states = self.proj_out(hidden_states[:, 1:])
        hidden_states = hidden_states.transpose(1, 2)
        return self.postprocess_conv(hidden_states) + hidden_states

    @staticmethod
    def checkpoint_key_map(num_layers: int) -> Dict[str, str]:
        """Checkpoint key -> parameter name, for the keys whose names differ.
        Upstream wraps the attention output projection in a ``ModuleList``
        alongside a dropout; we keep a plain ``Linear``."""
        return {
            f"transformer_blocks.{i}.attn.to_out.0.weight": f"transformer_blocks.{i}.attn.to_out.weight"
            for i in range(num_layers)
        }

    @classmethod
    def from_pretrained(cls, model_dir, dtype: torch.dtype = torch.float32, device="cpu") -> "MiniMaxDiT":
        config = _load_config(model_dir, cls._SUBDIR, cls._CONFIG_KEYS)
        state = _load_state_dict(model_dir, cls._SUBDIR)
        state = _remap(state, cls.checkpoint_key_map(config.get("num_layers", 36)))
        return _build(cls, config, state, dtype, device)


# ---------------------------------------------------------------------------
# DAV decoder
# ---------------------------------------------------------------------------


class MiniMaxSnake1d(nn.Module):
    """Snake activation with a learned per-channel period."""

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.empty(1, channels, 1))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + (self.alpha + 1e-9).reciprocal() * torch.sin(self.alpha * hidden_states).pow(2)


class MiniMaxDAVResidualUnit(nn.Module):
    def __init__(self, dim: int, dilation: int):
        super().__init__()
        self.snake1 = MiniMaxSnake1d(dim)
        self.conv1 = nn.Conv1d(dim, dim, kernel_size=7, dilation=dilation, padding=(7 - 1) * dilation // 2)
        self.snake2 = MiniMaxSnake1d(dim)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + self.conv2(self.snake2(self.conv1(self.snake1(hidden_states))))


class MiniMaxDAVBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, stride: int):
        super().__init__()
        self.snake1 = MiniMaxSnake1d(input_dim)
        self.conv_t1 = nn.ConvTranspose1d(
            input_dim, output_dim, kernel_size=2 * stride, stride=stride, padding=math.ceil(stride / 2)
        )
        self.res_units = nn.ModuleList(
            [MiniMaxDAVResidualUnit(output_dim, dilation=d) for d in (1, 3, 9)]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv_t1(self.snake1(hidden_states))
        for unit in self.res_units:
            hidden_states = unit(hidden_states)
        return hidden_states


class MiniMaxDAV(nn.Module):
    """The Flow-VAE waveform decoder (upstream ships it as ``vocoder/``).

    DAC-style and fully deterministic: there is no sampling anywhere in this
    path. Stereo is folded through the channel axis — a ``(B, 128, L)`` latent
    is decoded as ``2B`` mono streams of 64 channels and unfolded again, so the
    two sides never see each other.
    """

    _SUBDIR = "vocoder"
    _CONFIG_KEYS = (
        "latent_channels",
        "decoder_input_dim",
        "decoder_hidden_dim",
        "upsampling_ratios",
        "sampling_rate",
    )

    def __init__(
        self,
        latent_channels: int = 128,
        decoder_input_dim: int = 1024,
        decoder_hidden_dim: int = 1536,
        upsampling_ratios=(8, 8, 4, 2),
        sampling_rate: int = 44100,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.sampling_rate = sampling_rate
        self.upsampling_ratios = tuple(upsampling_ratios)
        self.upsample = math.prod(self.upsampling_ratios)

        self.dec_in_proj = nn.Conv1d(latent_channels // 2, decoder_input_dim, kernel_size=1)
        self.conv_in = nn.Conv1d(decoder_input_dim, decoder_hidden_dim, kernel_size=7, padding=3)
        blocks = []
        output_dim = decoder_hidden_dim
        for index, stride in enumerate(self.upsampling_ratios):
            input_dim = decoder_hidden_dim // (2**index)
            output_dim = decoder_hidden_dim // (2 ** (index + 1))
            blocks.append(MiniMaxDAVBlock(input_dim, output_dim, stride))
        self.blocks = nn.ModuleList(blocks)
        self.snake_out = MiniMaxSnake1d(output_dim)
        self.conv_out = nn.Conv1d(output_dim, 1, kernel_size=7, padding=3)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """``(B, latent_channels, L)`` -> ``(B, 2, L * upsample)`` in ``[-1, 1]``."""
        hidden_states = latents.reshape(-1, self.latent_channels // 2, latents.shape[-1])
        hidden_states = self.conv_in(self.dec_in_proj(hidden_states))
        for block in self.blocks:
            hidden_states = block(hidden_states)
        waveform = torch.tanh(self.conv_out(self.snake_out(hidden_states)))
        return waveform.reshape(-1, 2, waveform.shape[-1])

    @staticmethod
    def checkpoint_key_map(num_blocks: int) -> Dict[str, str]:
        """Checkpoint key -> parameter name. Upstream names the three residual
        units ``res_unit1/2/3``; we hold them in a ``ModuleList`` so the forward
        is a plain loop."""
        key_map: Dict[str, str] = {}
        for block in range(num_blocks):
            for unit in (1, 2, 3):
                for leaf in (
                    "snake1.alpha",
                    "snake2.alpha",
                    "conv1.bias",
                    "conv1.weight_g",
                    "conv1.weight_v",
                    "conv2.bias",
                    "conv2.weight_g",
                    "conv2.weight_v",
                ):
                    key_map[f"blocks.{block}.res_unit{unit}.{leaf}"] = f"blocks.{block}.res_units.{unit - 1}.{leaf}"
        return key_map

    @staticmethod
    def fold_weight_norm(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Upstream wraps most convs in the legacy ``torch.nn.utils.weight_norm``,
        which stores ``weight_g``/``weight_v`` and recomputes the product in a
        pre-forward hook. Inference only ever needs the product, so fold it once
        at load time and keep the forward a plain convolution — which is also
        what ONNX export needs."""
        folded: Dict[str, torch.Tensor] = {}
        for key, value in state.items():
            if key.endswith(".weight_v"):
                continue
            if key.endswith(".weight_g"):
                v = state[key[: -len("_g")] + "_v"]
                norm = v.pow(2).sum(dim=tuple(range(1, v.dim())), keepdim=True).sqrt()
                folded[key[: -len(".weight_g")] + ".weight"] = v * (value / norm)
                continue
            folded[key] = value
        return folded

    @classmethod
    def from_pretrained(cls, model_dir, dtype: torch.dtype = torch.float32, device="cpu") -> "MiniMaxDAV":
        config = _load_config(model_dir, cls._SUBDIR, cls._CONFIG_KEYS)
        state = _load_state_dict(model_dir, cls._SUBDIR)
        state = _remap(state, cls.checkpoint_key_map(len(config.get("upsampling_ratios", (8, 8, 4, 2)))))
        # Fold after remapping so the g/v pairs are already under their final path.
        state = cls.fold_weight_norm(state)
        return _build(cls, config, state, dtype, device)


# ---------------------------------------------------------------------------
# Condition encoder
# ---------------------------------------------------------------------------


class MiniMaxConditionEncoder(nn.Module):
    """Projects the AR stage's per-frame hidden states onto the latent timeline.

    Each AR frame carries ``num_condition_layers`` hidden states of size
    ``condition_hidden_dim`` (one from the language model, one per residual
    codebook step). They are mixed with learned softmax weights, scaled,
    projected, and resampled from the 25 Hz LM frame rate to the 86.133 Hz
    latent rate by nearest-neighbour interpolation.
    """

    _SUBDIR = "condition_encoder"
    _CONFIG_KEYS = (
        "condition_hidden_dim",
        "num_condition_layers",
        "out_dim",
        "input_sampling_rate",
        "input_hop_length",
        "output_sampling_rate",
        "output_hop_length",
    )

    def __init__(
        self,
        condition_hidden_dim: int = 4096,
        num_condition_layers: int = 8,
        out_dim: int = 2048,
        input_sampling_rate: int = 24000,
        input_hop_length: int = 960,
        output_sampling_rate: int = 44100,
        output_hop_length: int = 512,
    ):
        super().__init__()
        self.condition_hidden_dim = condition_hidden_dim
        self.num_condition_layers = num_condition_layers
        # 44100/24000 * 960/512 == 441/128 latent frames per LM frame. Kept as
        # an exact integer ratio; upstream evaluates the same product in float64
        # and truncates, which is exact for every frame count below 9344 (its own
        # chunker never exceeds 200) and one frame short at a handful of very
        # long takes above that. Measured, not assumed -- see the length-policy
        # probe in scripts/minimax/minimax_dit_parity.py.
        self.latent_frames_num = output_sampling_rate * input_hop_length
        self.latent_frames_den = input_sampling_rate * output_hop_length

        self.layer_weight_logits = nn.Parameter(torch.empty(num_condition_layers))
        self.layer_scale = nn.Parameter(torch.empty(1))
        self.proj = nn.Conv1d(condition_hidden_dim, out_dim, kernel_size=3, padding=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """``(B, frames, num_condition_layers * condition_hidden_dim)`` ->
        ``(B, latent_length, out_dim)``."""
        num_frames = hidden_states.shape[1]
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = hidden_states.reshape(
            -1, self.num_condition_layers, self.condition_hidden_dim, num_frames
        )
        layer_weights = torch.softmax(self.layer_weight_logits, dim=0).to(hidden_states.dtype)
        hidden_states = (hidden_states * layer_weights.view(1, -1, 1, 1)).sum(dim=1)
        hidden_states = self.layer_scale.to(hidden_states.dtype) * hidden_states
        hidden_states = self.proj(hidden_states)
        latent_length = num_frames * self.latent_frames_num // self.latent_frames_den
        hidden_states = F.interpolate(hidden_states, size=latent_length, mode="nearest")
        return hidden_states.transpose(1, 2)

    @staticmethod
    def checkpoint_key_map() -> Dict[str, str]:
        """No renames: every checkpoint key already matches a parameter name."""
        return {}

    @classmethod
    def from_pretrained(cls, model_dir, dtype: torch.dtype = torch.float32, device="cpu") -> "MiniMaxConditionEncoder":
        config = _load_config(model_dir, cls._SUBDIR, cls._CONFIG_KEYS)
        state = _remap(_load_state_dict(model_dir, cls._SUBDIR), cls.checkpoint_key_map())
        return _build(cls, config, state, dtype, device)
