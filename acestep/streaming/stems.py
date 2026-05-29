"""Mel-Band RoFormer helpers for the realtime motion graph backend."""

from __future__ import annotations

import gc
import importlib
import math
import os
import threading
import time
from collections.abc import Collection
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.functional as TAF
from einops import pack, rearrange, reduce, repeat, unpack
from torch import nn
from torch.nn import Module, ModuleList

from acestep.model_downloader import resolve_melband_roformer_model_path

try:
    _ExternalRotaryEmbedding = getattr(
        importlib.import_module("rotary_embedding_torch"),
        "RotaryEmbedding",
    )
except ImportError:
    _ExternalRotaryEmbedding = None

MELBAND_SAMPLE_RATE = 44_100


def exists(val):
    return val is not None


def default(v, d):
    return v if exists(v) else d


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


def hz_to_mel(frequencies, *, htk=False):
    frequencies = np.asanyarray(frequencies)

    if htk:
        mels: np.ndarray = 2595.0 * np.log10(1.0 + frequencies / 700.0)
        return mels

    f_min = 0.0
    f_sp = 200.0 / 3
    mels = (frequencies - f_min) / f_sp

    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0

    if frequencies.ndim:
        log_t = frequencies >= min_log_hz
        mels[log_t] = min_log_mel + np.log(frequencies[log_t] / min_log_hz) / logstep
    elif frequencies >= min_log_hz:
        mels = min_log_mel + np.log(frequencies / min_log_hz) / logstep

    return mels


def mel_to_hz(mels, *, htk=False):
    mels = np.asanyarray(mels)

    if htk:
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels

    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0

    if mels.ndim:
        log_t = mels >= min_log_mel
        freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))
    elif mels >= min_log_mel:
        freqs = min_log_hz * np.exp(logstep * (mels - min_log_mel))

    return freqs


def mel_frequencies(n_mels=128, *, fmin=0.0, fmax=11025.0, htk=False):
    min_mel = hz_to_mel(fmin, htk=htk)
    max_mel = hz_to_mel(fmax, htk=htk)
    mels = np.linspace(min_mel, max_mel, n_mels)
    hz: np.ndarray = mel_to_hz(mels, htk=htk)
    return hz


def fft_frequencies(*, sr: float = 22050, n_fft: int = 2048) -> np.ndarray:
    return np.fft.rfftfreq(n=n_fft, d=1.0 / sr)


def librosa_mel_fn(
    *,
    sr: float,
    n_fft: int,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax=None,
    htk=False,
    norm="slaney",
    dtype=np.float32,
) -> np.ndarray:
    if fmax is None:
        fmax = float(sr) / 2

    n_mels = int(n_mels)
    weights = np.zeros((n_mels, int(1 + n_fft // 2)), dtype=dtype)
    fftfreqs = fft_frequencies(sr=sr, n_fft=n_fft)
    mel_f = mel_frequencies(n_mels + 2, fmin=fmin, fmax=fmax, htk=htk)

    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)

    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0, np.minimum(lower, upper))

    if norm == "slaney":
        enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])
        weights *= enorm[:, np.newaxis]

    return weights


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    return rearrange(torch.stack((-x2, x1), dim=-1), "... d r -> ... (d r)")


class RotaryEmbedding(Module):
    """Small compatibility fallback for rotary_embedding_torch.RotaryEmbedding."""

    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("freqs", freqs)

    def rotate_queries_or_keys(self, t: torch.Tensor) -> torch.Tensor:
        seq_len = t.shape[-2]
        positions = torch.arange(seq_len, device=t.device, dtype=self.freqs.dtype)
        freqs = torch.einsum("n,d->n d", positions, self.freqs.to(t.device))
        freqs = repeat(freqs, "... d -> ... (d r)", r=2).to(dtype=t.dtype)
        while freqs.ndim < t.ndim:
            freqs = freqs.unsqueeze(0)
        return (t * freqs.cos()) + (_rotate_half(t) * freqs.sin())


if _ExternalRotaryEmbedding is not None:
    RotaryEmbedding = _ExternalRotaryEmbedding


class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.gamma


class FeedForward(Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        dim_inner = int(dim * mult)
        self.net = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, dim_inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_inner, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(Module):
    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        dropout=0.0,
        rotary_embed=None,
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head**-0.5
        dim_inner = heads * dim_head

        self.rotary_embed = rotary_embed
        self.attend = F.scaled_dot_product_attention
        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias=False)
        self.to_gates = nn.Linear(dim, heads)
        self.to_out = nn.Sequential(
            nn.Linear(dim_inner, dim, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = self.norm(x)
        q, k, v = rearrange(self.to_qkv(x), "b n (qkv h d) -> qkv b h n d", qkv=3, h=self.heads)

        if exists(self.rotary_embed):
            q = self.rotary_embed.rotate_queries_or_keys(q)
            k = self.rotary_embed.rotate_queries_or_keys(k)

        out = self.attend(q, k, v)
        gates = self.to_gates(x)
        out = out * rearrange(gates, "b n h -> b h n 1").sigmoid()
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head=64,
        heads=8,
        attn_dropout=0.0,
        ff_dropout=0.0,
        ff_mult=4,
        norm_output=True,
        rotary_embed=None,
        flash_attn=True,
    ):
        super().__init__()
        self.layers = ModuleList([])

        for _ in range(depth):
            self.layers.append(
                ModuleList(
                    [
                        Attention(
                            dim=dim,
                            dim_head=dim_head,
                            heads=heads,
                            dropout=attn_dropout,
                            rotary_embed=rotary_embed,
                        ),
                        FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout),
                    ]
                )
            )

        self.norm = RMSNorm(dim) if norm_output else nn.Identity()

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


class BandSplit(Module):
    def __init__(self, dim, dim_inputs):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_features = ModuleList([])

        for dim_in in dim_inputs:
            net = nn.Sequential(
                RMSNorm(dim_in),
                nn.Linear(dim_in, dim),
            )
            self.to_features.append(net)

    def forward(self, x):
        x = x.split(self.dim_inputs, dim=-1)

        outs = []
        for split_input, to_feature in zip(x, self.to_features):
            split_output = to_feature(split_input)
            outs.append(split_output)

        return torch.stack(outs, dim=-2)


def MLP(dim_in, dim_out, dim_hidden=None, depth=1, activation=nn.Tanh):
    dim_hidden = default(dim_hidden, dim_in)

    net = []
    dims = (dim_in, *((dim_hidden,) * depth), dim_out)

    for ind, (layer_dim_in, layer_dim_out) in enumerate(zip(dims[:-1], dims[1:])):
        is_last = ind == (len(dims) - 2)
        net.append(nn.Linear(layer_dim_in, layer_dim_out))

        if is_last:
            continue

        net.append(activation())

    return nn.Sequential(*net)


class MaskEstimator(Module):
    def __init__(self, dim, dim_inputs, depth, mlp_expansion_factor=4):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_freqs = ModuleList([])
        dim_hidden = dim * mlp_expansion_factor

        for dim_in in dim_inputs:
            mlp = nn.Sequential(
                MLP(dim, dim_in * 2, dim_hidden=dim_hidden, depth=depth),
                nn.GLU(dim=-1),
            )
            self.to_freqs.append(mlp)

    def forward(self, x):
        x = x.unbind(dim=-2)
        outs = []

        for band_features, mlp in zip(x, self.to_freqs):
            freq_out = mlp(band_features)
            outs.append(freq_out)

        return torch.cat(outs, dim=-1)


class MelBandRoformer(Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        stereo=False,
        num_stems=1,
        time_transformer_depth=2,
        freq_transformer_depth=2,
        num_bands=60,
        dim_head=64,
        heads=8,
        attn_dropout=0.1,
        ff_dropout=0.1,
        flash_attn=True,
        dim_freqs_in=1025,
        sample_rate=44100,
        stft_n_fft=2048,
        stft_hop_length=512,
        stft_win_length=2048,
        stft_normalized=False,
        stft_window_fn=None,
        mask_estimator_depth=1,
        multi_stft_resolution_loss_weight=1.0,
        multi_stft_resolutions_window_sizes=(4096, 2048, 1024, 512, 256),
        multi_stft_hop_size=147,
        multi_stft_normalized=False,
        multi_stft_window_fn=torch.hann_window,
        match_input_audio_length=False,
    ):
        super().__init__()

        self.stereo = stereo
        self.audio_channels = 2 if stereo else 1
        self.num_stems = num_stems
        self.layers = ModuleList([])

        transformer_kwargs = dict(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            flash_attn=flash_attn,
        )

        time_rotary_embed = RotaryEmbedding(dim=dim_head)
        freq_rotary_embed = RotaryEmbedding(dim=dim_head)

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Transformer(
                            depth=time_transformer_depth,
                            rotary_embed=time_rotary_embed,
                            **transformer_kwargs,
                        ),
                        Transformer(
                            depth=freq_transformer_depth,
                            rotary_embed=freq_rotary_embed,
                            **transformer_kwargs,
                        ),
                    ]
                )
            )

        self.stft_window_fn = partial(default(stft_window_fn, torch.hann_window), stft_win_length)
        self.stft_kwargs = dict(
            n_fft=stft_n_fft,
            hop_length=stft_hop_length,
            win_length=stft_win_length,
            normalized=stft_normalized,
        )

        freqs = torch.stft(torch.randn(1, 4096), **self.stft_kwargs, return_complex=True).shape[1]
        if freqs != dim_freqs_in:
            raise ValueError(f"Expected {dim_freqs_in} STFT bins, got {freqs}")

        mel_filter_bank_numpy = librosa_mel_fn(sr=sample_rate, n_fft=stft_n_fft, n_mels=num_bands)
        mel_filter_bank = torch.from_numpy(mel_filter_bank_numpy)

        mel_filter_bank[0][0] = 1.0
        mel_filter_bank[-1, -1] = 1.0

        freqs_per_band = mel_filter_bank > 0
        assert freqs_per_band.any(dim=0).all(), "all frequencies need to be covered by all bands for now"

        repeated_freq_indices = repeat(torch.arange(freqs), "f -> b f", b=num_bands)
        freq_indices = repeated_freq_indices[freqs_per_band]

        if stereo:
            freq_indices = repeat(freq_indices, "f -> f s", s=2)
            freq_indices = freq_indices * 2 + torch.arange(2)
            freq_indices = rearrange(freq_indices, "f s -> (f s)")

        self.register_buffer("freq_indices", freq_indices, persistent=False)
        self.register_buffer("freqs_per_band", freqs_per_band, persistent=False)

        num_freqs_per_band = reduce(freqs_per_band, "b f -> b", "sum")
        num_bands_per_freq = reduce(freqs_per_band, "b f -> f", "sum")

        self.register_buffer("num_freqs_per_band", num_freqs_per_band, persistent=False)
        self.register_buffer("num_bands_per_freq", num_bands_per_freq, persistent=False)

        freqs_per_bands_with_complex = tuple(2 * f * self.audio_channels for f in num_freqs_per_band.tolist())
        self.band_split = BandSplit(dim=dim, dim_inputs=freqs_per_bands_with_complex)
        self.mask_estimators = nn.ModuleList([])

        for _ in range(num_stems):
            mask_estimator = MaskEstimator(
                dim=dim,
                dim_inputs=freqs_per_bands_with_complex,
                depth=mask_estimator_depth,
            )
            self.mask_estimators.append(mask_estimator)

        self.multi_stft_resolution_loss_weight = multi_stft_resolution_loss_weight
        self.multi_stft_resolutions_window_sizes = multi_stft_resolutions_window_sizes
        self.multi_stft_n_fft = stft_n_fft
        self.multi_stft_window_fn = multi_stft_window_fn
        self.multi_stft_kwargs = dict(
            hop_length=multi_stft_hop_size,
            normalized=multi_stft_normalized,
        )
        self.match_input_audio_length = match_input_audio_length

    def forward(self, raw_audio, target=None, return_loss_breakdown=False):
        device = raw_audio.device

        if raw_audio.ndim == 2:
            raw_audio = rearrange(raw_audio, "b t -> b 1 t")

        batch, channels, raw_audio_length = raw_audio.shape
        istft_length = raw_audio_length if self.match_input_audio_length else None

        assert (not self.stereo and channels == 1) or (
            self.stereo and channels == 2
        ), "stereo needs to be set to True if passing in audio signal that is stereo"

        raw_audio, batch_audio_channel_packed_shape = pack_one(raw_audio, "* t")

        stft_window = self.stft_window_fn(device=device)
        stft_repr = torch.stft(raw_audio, **self.stft_kwargs, window=stft_window, return_complex=True)
        stft_repr = torch.view_as_real(stft_repr)

        stft_repr = unpack_one(stft_repr, batch_audio_channel_packed_shape, "* f t c")
        stft_repr = rearrange(stft_repr, "b s f t c -> b (f s) t c")

        batch_arange = torch.arange(batch, device=device)[..., None]
        x = stft_repr[batch_arange, self.freq_indices]
        x = rearrange(x, "b f t c -> b t (f c)")
        x = self.band_split(x)

        for time_transformer, freq_transformer in self.layers:
            x = rearrange(x, "b t f d -> b f t d")
            x, ps = pack([x], "* t d")
            x = time_transformer(x)

            (x,) = unpack(x, ps, "* t d")
            x = rearrange(x, "b f t d -> b t f d")
            x, ps = pack([x], "* f d")
            x = freq_transformer(x)

            (x,) = unpack(x, ps, "* f d")

        num_stems = len(self.mask_estimators)
        masks = torch.stack([fn(x) for fn in self.mask_estimators], dim=1)
        masks = rearrange(masks, "b n t (f c) -> b n f t c", c=2)

        stft_repr = rearrange(stft_repr, "b f t c -> b 1 f t c")
        stft_repr = torch.view_as_complex(stft_repr)
        masks = torch.view_as_complex(masks)
        masks = masks.type(stft_repr.dtype)

        scatter_indices = repeat(
            self.freq_indices,
            "f -> b n f t",
            b=batch,
            n=num_stems,
            t=stft_repr.shape[-1],
        )
        stft_repr_expanded_stems = repeat(stft_repr, "b 1 ... -> b n ...", n=num_stems)
        masks_summed = torch.zeros_like(stft_repr_expanded_stems).scatter_add_(2, scatter_indices, masks)

        denom = repeat(self.num_bands_per_freq, "f -> (f r) 1", r=channels)
        masks_averaged = masks_summed / denom.clamp(min=1e-8)

        stft_repr = stft_repr * masks_averaged
        stft_repr = rearrange(stft_repr, "b n (f s) t -> (b n s) f t", s=self.audio_channels)

        recon_audio = torch.istft(
            stft_repr,
            **self.stft_kwargs,
            window=stft_window,
            return_complex=False,
            length=istft_length,
        )
        recon_audio = rearrange(recon_audio, "(b n s) t -> b n s t", b=batch, s=self.audio_channels, n=num_stems)

        if num_stems == 1:
            recon_audio = rearrange(recon_audio, "b 1 s t -> b s t")

        if not exists(target):
            return recon_audio

        if self.num_stems > 1:
            assert target.ndim == 4 and target.shape[1] == self.num_stems

        if target.ndim == 2:
            target = rearrange(target, "... t -> ... 1 t")

        target = target[..., : recon_audio.shape[-1]]
        loss = F.l1_loss(recon_audio, target)
        multi_stft_resolution_loss = 0.0

        for window_size in self.multi_stft_resolutions_window_sizes:
            res_stft_kwargs = dict(
                n_fft=max(window_size, self.multi_stft_n_fft),
                win_length=window_size,
                return_complex=True,
                window=self.multi_stft_window_fn(window_size, device=device),
                **self.multi_stft_kwargs,
            )

            recon_y = torch.stft(rearrange(recon_audio, "... s t -> (... s) t"), **res_stft_kwargs)
            target_y = torch.stft(rearrange(target, "... s t -> (... s) t"), **res_stft_kwargs)
            multi_stft_resolution_loss = multi_stft_resolution_loss + F.l1_loss(recon_y, target_y)

        weighted_multi_resolution_loss = multi_stft_resolution_loss * self.multi_stft_resolution_loss_weight
        total_loss = loss + weighted_multi_resolution_loss

        if not return_loss_breakdown:
            return total_loss

        return total_loss, (loss, multi_stft_resolution_loss)


def get_windowing_array(window_size, fade_size, device):
    fadein = torch.linspace(0, 1, fade_size)
    fadeout = torch.linspace(1, 0, fade_size)
    window = torch.ones(window_size)
    window[-fade_size:] *= fadeout
    window[:fade_size] *= fadein
    return window.to(device)


def melband_model_config() -> dict:
    return {
        "dim": 384,
        "depth": 6,
        "stereo": True,
        "num_stems": 1,
        "time_transformer_depth": 1,
        "freq_transformer_depth": 1,
        "num_bands": 60,
        "dim_head": 64,
        "heads": 8,
        "attn_dropout": 0,
        "ff_dropout": 0,
        "flash_attn": True,
        "dim_freqs_in": 1025,
        "sample_rate": MELBAND_SAMPLE_RATE,
        "stft_n_fft": 2048,
        "stft_hop_length": 441,
        "stft_win_length": 2048,
        "stft_normalized": False,
        "mask_estimator_depth": 2,
        "multi_stft_resolution_loss_weight": 1.0,
        "multi_stft_resolutions_window_sizes": (4096, 2048, 1024, 512, 256),
        "multi_stft_hop_size": 147,
        "multi_stft_normalized": False,
    }


def load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix.lower() == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise RuntimeError(
                "safetensors is required to load .safetensors checkpoints. "
                "Install it or pass a PyTorch .pt/.pth checkpoint."
            ) from exc
        return load_file(str(path), device="cpu")

    checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint format: {path}")
    return checkpoint


def load_model(model_path: Path, device: torch.device) -> MelBandRoformer:
    print(f"loading Mel-Band RoFormer checkpoint: {model_path}")
    model = MelBandRoformer(**melband_model_config()).eval()
    model.load_state_dict(load_state_dict(model_path), strict=True)
    model.to(device)
    return model


def fit_to_length(waveform: torch.Tensor, frames: int) -> torch.Tensor:
    if waveform.shape[-1] > frames:
        return waveform[..., :frames]
    if waveform.shape[-1] < frames:
        return F.pad(waveform, (0, frames - waveform.shape[-1]))
    return waveform


@torch.inference_mode()
def separate_stems(
    model: MelBandRoformer,
    audio_input: torch.Tensor,
    sample_rate: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    _, audio_channels, _ = audio_input.shape
    sr = MELBAND_SAMPLE_RATE

    if audio_channels == 1:
        audio_input = audio_input.repeat(1, 2, 1)
        audio_channels = 2
        print("Converted mono input to stereo.")

    if sample_rate != sr:
        print(f"Resampling input {sample_rate} to {sr}")
        audio_input = TAF.resample(audio_input, orig_freq=sample_rate, new_freq=sr)

    audio_input = original_audio = audio_input[0]
    audio_length = audio_input.shape[-1]

    chunk_size = 352800
    overlaps = 2
    step = chunk_size // overlaps
    fade_size = chunk_size // 10
    border = chunk_size - step

    if audio_length > 2 * border and border > 0:
        audio_input = F.pad(audio_input, (border, border), mode="reflect")

    windowing_array = get_windowing_array(chunk_size, fade_size, device)

    audio_input = audio_input.to(device)
    vocals = torch.zeros_like(audio_input, dtype=torch.float32, device=device)
    counter = torch.zeros_like(audio_input, dtype=torch.float32, device=device)

    total_length = audio_input.shape[1]
    num_chunks = math.ceil(total_length / step)
    print(f"processing {num_chunks} chunks...")

    try:
        from tqdm import tqdm

        chunk_iter = tqdm(range(0, total_length, step), desc="Processing chunks")
    except ImportError:
        chunk_iter = range(0, total_length, step)

    for i in chunk_iter:
        part = audio_input[:, i : i + chunk_size]
        length = part.shape[-1]
        if length < chunk_size:
            if length > chunk_size // 2 + 1:
                part = F.pad(input=part, pad=(0, chunk_size - length), mode="reflect")
            else:
                part = F.pad(input=part, pad=(0, chunk_size - length, 0, 0), mode="constant", value=0)

        x = model(part.unsqueeze(0))[0].float()

        window = windowing_array.clone()
        if i == 0:
            window[:fade_size] = 1
        elif i + chunk_size >= total_length:
            window[-fade_size:] = 1

        vocals[..., i : i + length] += x[..., :length] * window[..., :length]
        counter[..., i : i + length] += window[..., :length]

    estimated_sources = vocals / counter.clamp_min(1e-8)

    if audio_length > 2 * border and border > 0:
        estimated_sources = estimated_sources[..., border:-border]

    original_audio = original_audio.to(device)
    estimated_sources = fit_to_length(estimated_sources, original_audio.shape[-1])
    instruments = original_audio - estimated_sources

    return estimated_sources.cpu(), instruments.cpu()

STEM_SOURCE_MODES = frozenset({"full", "vocals", "instruments"})

_INFER_LOCK = threading.Lock()


def normalize_stem_source_mode(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    mode = value.strip().lower()
    return mode if mode in STEM_SOURCE_MODES else None


def resolve_upload_stem_source_mode(
    fixture_name: object,
    requested_mode: str | None,
    *,
    known_fixtures: Collection[str],
) -> str | None:
    """Auto-stem user uploads while keeping built-in fixtures cheap by default."""
    if requested_mode is not None:
        return requested_mode
    if isinstance(fixture_name, str) and fixture_name in known_fixtures:
        return None
    return "full"


def extract_upload_stems(
    *,
    waveform: torch.Tensor,
    device: torch.device | str,
    backend_sample_rate: int,
) -> dict[str, torch.Tensor]:
    """Use Mel-Band RoFormer for vocal and instrumental separation.

    The realtime backend runs sources at 48 kHz, while the RoFormer checkpoint
    is trained for 44.1 kHz. The separator handles the downsample internally;
    we resample its returned stems back to the backend sample rate before
    sending overlays or preparing a selected stem as the inference source.
    """
    torch_device = _coerce_device(device)
    t0 = time.time()
    with _INFER_LOCK:
        model: MelBandRoformer | None = None
        try:
            model_path = _resolve_model_path()
            print(f"[Server] Loading Mel-Band RoFormer model on {torch_device}...")
            load_t0 = time.time()
            model = load_model(model_path, torch_device)
            print(f"[Server] Mel-Band RoFormer loaded in {time.time() - load_t0:.1f}s")
            vocals_44k, instruments_44k = separate_stems(
                model,
                waveform.detach().cpu().float().unsqueeze(0),
                backend_sample_rate,
                torch_device,
            )
            if torch_device.type == "cuda":
                torch.cuda.synchronize(torch_device)
        finally:
            if model is not None:
                print(f"[Server] Releasing Mel-Band RoFormer model from {torch_device}...")
                del model
                _collect_device_cache(torch_device)
    print(f"[Server] Mel-Band RoFormer stems complete in {time.time() - t0:.1f}s")

    vocals = _fit_stem_waveform(
        _resample_stem_to_backend_rate(vocals_44k, backend_sample_rate),
        waveform,
    )
    instruments = _fit_stem_waveform(
        _resample_stem_to_backend_rate(instruments_44k, backend_sample_rate),
        waveform,
    )
    return {
        "vocals": vocals.contiguous(),
        "instruments": instruments.contiguous(),
    }


def _coerce_device(device: torch.device | str) -> torch.device:
    return device if isinstance(device, torch.device) else torch.device(device)


def _fit_stem_waveform(wf: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Coerce decoded model output to the uploaded waveform's [C, N] shape."""
    if wf.ndim == 3:
        wf = wf[0]
    if wf.ndim == 1:
        wf = wf.unsqueeze(0)
    wf = wf.detach().to(dtype=torch.float32, device=target.device)
    if wf.shape[0] == 1 and target.shape[0] == 2:
        wf = wf.repeat(2, 1)
    elif wf.shape[0] > target.shape[0]:
        wf = wf[:target.shape[0]]
    elif wf.shape[0] < target.shape[0]:
        wf = torch.cat(
            [wf, wf[-1:].repeat(target.shape[0] - wf.shape[0], 1)],
            dim=0,
        )
    if wf.shape[-1] > target.shape[-1]:
        wf = wf[:, :target.shape[-1]]
    elif wf.shape[-1] < target.shape[-1]:
        wf = torch.nn.functional.pad(wf, (0, target.shape[-1] - wf.shape[-1]))
    return torch.nan_to_num(wf)


def _resolve_model_path() -> Path:
    explicit_path = os.environ.get("MELBAND_ROFORMER_MODEL_PATH")
    if explicit_path:
        return Path(explicit_path).expanduser()

    return resolve_melband_roformer_model_path()


def _collect_device_cache(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _resample_stem_to_backend_rate(
    stem: torch.Tensor,
    backend_sample_rate: int,
) -> torch.Tensor:
    stem = stem.detach().cpu().float()
    if MELBAND_SAMPLE_RATE == backend_sample_rate:
        return stem
    return TAF.resample(
        stem,
        orig_freq=MELBAND_SAMPLE_RATE,
        new_freq=backend_sample_rate,
    )
